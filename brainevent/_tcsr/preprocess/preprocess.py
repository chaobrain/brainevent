# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Host-only preparation; tile arrays remain dynamic matrix pytree leaves."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core

from brainevent._compatible_import import Tracer
from brainevent._tcsr.binary.combined.planner import (
    INT32_MAX,
    TARGET_BLOCKS,
    tile_schedule,
)

__all__ = [
    'TileGeometry', 'build_tile_geometries', 'geometry_widths',
    'ensure_preparation_context', 'sort_csr', 'validate_preparation',
]


@dataclass(frozen=True, eq=False)
class TileGeometry:
    """Immutable tile arrays shared by all matching batches and value dtypes.

    Parameters
    ----------
    shape : tuple of int
        Canonical CSR dimensions in both traversal directions.
    nnz : int
        Stored edge count, including duplicates.
    transpose : bool
        True for the primary traversal, False for the public CSR mirror.
    tile_width : int
        Actual accumulation tile width represented by these arrays.
    offset_dtype : numpy.dtype
        Absolute-offset dtype matching the selected traversal's pointers.
    local_targets, tile_offsets : jax.Array
        Dynamic uint16 targets and flattened absolute int32/int64 boundaries.

    Notes
    -----
    The owning structure cache establishes structural identity. This object
    stores no mirror permutation, weight array or call-specific scratch.
    """

    shape: tuple[int, int]
    nnz: int
    transpose: bool
    tile_width: int
    offset_dtype: np.dtype
    local_targets: Any
    tile_offsets: Any

    @property
    def key(self) -> tuple[bool, int, np.dtype]:
        """Return the immutable direction, width and offset-dtype key.

        Returns
        -------
        tuple
            Direction, actual tile width and NumPy absolute-offset dtype.
        """
        return self.transpose, self.tile_width, self.offset_dtype

    @property
    def metadata(self) -> tuple:
        """Return static pytree metadata without any dynamic arrays.

        Returns
        -------
        tuple
            Canonical shape, edge count, direction, width and offset dtype.
        """
        return self.shape, self.nnz, self.transpose, self.tile_width, self.offset_dtype

    @classmethod
    def from_metadata(cls, metadata: tuple, local_targets: Any, tile_offsets: Any) -> TileGeometry:
        """Restore a geometry without inspecting or converting its leaves.

        Parameters
        ----------
        metadata : tuple
            Immutable description returned by :attr:`metadata`.
        local_targets, tile_offsets : object
            Exact leaves supplied by JAX, including tracers or placeholders.

        Returns
        -------
        TileGeometry
            Restored metadata with the supplied dynamic leaves.
        """
        return cls(metadata[0], metadata[1], metadata[2], metadata[3],
                   metadata[4], local_targets, tile_offsets)


@lru_cache(maxsize=256, typed=True)
def geometry_widths(target: int) -> tuple[int, ...]:
    """Enumerate widths covering every supported batch and accumulator type.

    Parameters
    ----------
    target : int
        Output dimension of the selected traversal, from zero to INT32_MAX.

    Returns
    -------
    tuple of int
        Sorted unique adaptive widths covering batch 0..65535 with 4-byte
        and 8-byte accumulators.

    Raises
    ------
    ValueError
        The output dimension is not a supported static integer.

    Notes
    -----
    Positive batches affect width only through ``ceil(TARGET_BLOCKS / batch)``.
    This equals one above TARGET_BLOCKS, so the finite 0..TARGET_BLOCKS range
    includes every possible width, including the special zero-batch widths.
    """
    if isinstance(target, (bool, np.bool_)):
        raise ValueError('target must be an integer between 0 and INT32_MAX')
    try:
        target = operator.index(target)
    except TypeError as exc:
        raise ValueError('target must be an integer between 0 and INT32_MAX') from exc
    if not 0 <= target <= INT32_MAX:
        raise ValueError('target must be an integer between 0 and INT32_MAX')
    return tuple(sorted({tile_schedule(target, batch, size).tile_width
                         for size in (4, 8) for batch in range(TARGET_BLOCKS + 1)}))


def _geometry_placement(*arrays: Any) -> Any:
    """Keep whole-structure metadata on the structure's known device or mesh."""
    for array in arrays:
        sharding = getattr(array, 'sharding', None)
        if sharding is not None:
            if sharding.is_fully_replicated:
                return sharding
            if isinstance(sharding, jax.sharding.NamedSharding):
                return jax.sharding.NamedSharding(sharding.mesh, jax.sharding.PartitionSpec())
    return None


def _check_sorted_rows(idx: np.ndarray, ptr: np.ndarray) -> None:
    """Reject a column decrease inside any row, in one vectorized pass."""
    if idx.size < 2:
        return
    decreasing = np.flatnonzero(idx[1:] < idx[:-1]) + 1
    # A decrease is legal only where a new row starts at that slot.
    if decreasing.size and not np.isin(decreasing, ptr[1:-1]).all():
        raise ValueError('traversal indices must be sorted within each row')


_OFFSET_BLOCK = 1 << 22


def _tile_offsets(idx: np.ndarray, ptr: np.ndarray, target: int, width: int) -> np.ndarray:
    """Absolute row/tile boundaries, ``(source, tile_count + 1)``, in O(nnz + source*tiles).

    Rows are processed in blocks so the int64 temporaries stay bounded by
    about ``_OFFSET_BLOCK`` tile counters plus the block's edges.
    """
    source = ptr.size - 1
    tile_count = (target + width - 1) // width
    offsets = np.empty((source, tile_count + 1), dtype=ptr.dtype)
    offsets[:, 0] = ptr[:-1]
    if not source or not tile_count:
        return offsets
    block = max(1, _OFFSET_BLOCK // tile_count)
    for first in range(0, source, block):
        last = min(source, first + block)
        begin, end = int(ptr[first]), int(ptr[last])
        lengths = np.diff(ptr[first:last + 1]).astype(np.int64)
        rows = np.repeat(np.arange(last - first, dtype=np.int64), lengths)
        counts = np.bincount(rows * tile_count + idx[begin:end] // width,
                             minlength=(last - first) * tile_count)
        ends = np.cumsum(counts.reshape(last - first, tile_count), axis=1)
        offsets[first:last, 1:] = ends + ptr[first:last, None]
    return offsets


def _checked_widths(widths: Any) -> list[int]:
    """Deduplicate static tile widths that fit uint16 local targets."""
    result = set()
    for width in widths:
        if isinstance(width, (bool, np.bool_)):
            raise ValueError('widths must be positive integers no larger than 65536')
        try:
            width = operator.index(width)
        except TypeError as exc:
            raise ValueError('widths must be positive integers no larger than 65536') from exc
        if not 0 < width <= 65536:
            raise ValueError('widths must be positive integers no larger than 65536')
        result.add(width)
    return sorted(result)


def build_tile_geometries(
    indices: Any, indptr: Any, *, shape: tuple[int, int], transpose: bool,
    widths: Any = None,
) -> dict[tuple[bool, int, np.dtype], TileGeometry]:
    """Prepare one traversal's geometry for the requested tile widths.

    Parameters
    ----------
    indices, indptr : array_like
        Concrete sorted traversal indices (int32) and pointers (int32/int64).
        Duplicates remain independent edges. The mirror must already be in
        its public CSR slot order; no second permutation is constructed.
    shape : tuple of int
        Canonical CSR shape, regardless of traversal direction.
    transpose : bool
        True selects primary traversal; False selects the public CSR mirror.
    widths : iterable of int or None, optional
        Tile widths to prepare, normally the ``tile_width`` that
        :func:`~brainevent._tcsr.binary.combined.planner.tile_schedule`
        selects for one batch and accumulator size. Duplicates are ignored.
        ``None`` prepares every width in :func:`geometry_widths`: about two
        bytes per edge for each of roughly 25 widths, so prefer the widths
        that will actually run.

    Returns
    -------
    dict
        Geometry keyed by direction, actual width and offset dtype. Arrays
        for one width serve every batch and value dtype that selects it. The
        caller publishes this table only after success.

    Raises
    ------
    ValueError
        Called while staging outside compile-time evaluation, or shape,
        pointer values, widths or sortedness are invalid. Explicit int64
        offsets must survive JAX placement unchanged.
    TypeError
        Indices or pointers have unsupported rank or dtype.

    Notes
    -----
    Each width stores ``nnz`` uint16 local targets and
    ``source * (ceil(target / width) + 1)`` absolute offsets of the pointer
    dtype. Host work is vectorized: O(nnz + source * tile_count) per width.
    """
    # The public compile-time context enables eager constant folding without
    # replacing the active trace in JAX >= 0.9. Concrete closures can prepare
    # their mirror once there; dynamic matrix arguments must never escape.
    if (any(isinstance(a, Tracer) for a in (indices, indptr))
            or not (core.trace_ctx.is_top_level() or jax.config.eager_constant_folding)):
        raise ValueError('TCSR geometry preparation must run outside JIT or in explicit compile-time evaluation.')
    if not isinstance(transpose, (bool, np.bool_)):
        raise ValueError('transpose must be True or False')
    if len(shape) != 2 or any(isinstance(s, (bool, np.bool_)) for s in shape):
        raise ValueError('shape must contain two nonnegative int32 dimensions')
    try:
        shape = operator.index(shape[0]), operator.index(shape[1])
    except TypeError as exc:
        raise ValueError('shape must contain two nonnegative int32 dimensions') from exc
    if any(s < 0 or s > INT32_MAX for s in shape):
        raise ValueError('shape must contain two nonnegative int32 dimensions')
    source, target = shape if transpose else shape[::-1]
    widths = _checked_widths(geometry_widths(target) if widths is None else widths)
    idx, ptr = (np.asarray(jax.device_get(a)) for a in (indices, indptr))
    if idx.ndim != 1 or idx.dtype != np.dtype('int32'):
        raise TypeError('TCSR traversal indices must be rank-one int32')
    if ptr.ndim != 1 or ptr.dtype not in (np.dtype('int32'), np.dtype('int64')):
        raise TypeError('TCSR traversal indptr must be rank-one int32/int64')
    if ptr.shape != (source + 1,):
        raise ValueError('traversal indptr length does not match the canonical shape')
    if ptr[0] != 0 or ptr[-1] != idx.size or np.any(ptr[1:] < ptr[:-1]):
        raise ValueError('invalid CSR pointers')
    if ptr.dtype == np.dtype('int32') and idx.size > INT32_MAX:
        raise ValueError('int32 indptr cannot address this many edges')
    if np.any(idx < 0) or np.any(idx >= target):
        raise ValueError('traversal index out of range')
    _check_sorted_rows(idx, ptr)
    placement = _geometry_placement(indices, indptr)
    result = {}
    for width in widths:
        local = (idx % width).astype(np.uint16)
        flat_offsets = _tile_offsets(idx, ptr, target, width).ravel()
        local.flags.writeable = flat_offsets.flags.writeable = False
        local_array = jax.device_put(local, placement)
        offset_array = jax.device_put(flat_offsets, placement)
        if np.dtype(offset_array.dtype) != ptr.dtype:
            raise ValueError('JAX must preserve int64 tile offsets; enable x64 before preparation.')
        geometry = TileGeometry(shape, idx.size, bool(transpose), width, ptr.dtype,
                                local_array, offset_array)
        result[geometry.key] = geometry
    return result


def ensure_preparation_context(*arrays: Any) -> None:
    """Require concrete structure and an execution context outside JAX transforms.

    Parameters
    ----------
    *arrays : array_like
        Structure arrays about to be read on the host.

    Raises
    ------
    ValueError
        If called while tracing or supplied a traced structure array.
    """
    if not core.trace_ctx.is_top_level() or any(isinstance(a, Tracer) for a in arrays):
        raise ValueError('TCSR sorting and tile preparation must run outside JIT with concrete structure.')


def validate_preparation(batch: Any, value_dtype: Any) -> int:
    """Validate static preparation arguments before constructing a mirror.

    Parameters
    ----------
    batch : int
        Static event batch size, from zero to 65535 inclusive.
    value_dtype : dtype
        Weight/output dtype, one of float16, bfloat16, float32 or float64.

    Returns
    -------
    int
        Validated batch size as a Python integer.

    Raises
    ------
    ValueError
        Called while tracing, or batch/dtype is unsupported.
    """
    ensure_preparation_context()
    if isinstance(batch, (bool, np.bool_)):
        raise ValueError('batch must be a static integer between 0 and 65535')
    try:
        batch = operator.index(batch)
    except TypeError as exc:
        raise ValueError('batch must be a static integer between 0 and 65535') from exc
    if not 0 <= batch <= 65535:
        raise ValueError('invalid dimensions: batch must be between 0 and 65535')
    if np.dtype(value_dtype) not in tuple(np.dtype(t) for t in (
        jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64,
    )):
        raise ValueError('values must be f16, bf16, f32 or f64')
    return batch


def sort_csr(csr: Any) -> tuple[Any, jax.Array, np.ndarray | None]:
    """Stably sort columns within each CSR row and reorder heterogeneous weights.

    Parameters
    ----------
    csr : CSR
        Concrete source matrix; its arrays and caches are never changed.

    Returns
    -------
    data : jax.Array or brainunit.Quantity
        Sorted edge weights. Scalars and one-element shared weights are
        returned unchanged, without expansion.
    indices : jax.Array
        Sorted column indices. Duplicate edges retain their relative order.
    permutation : numpy.ndarray or None
        Read-only host array with ``indices == csr.indices[permutation]``:
        sorted slot ``k`` holds source slot ``permutation[k]``. Apply it to
        any per-edge array kept in the source order (for example delays or
        edge state). ``None`` when every row was already sorted; ``data``
        and ``indices`` are then the source arrays themselves.
    """
    ensure_preparation_context(csr.indices, csr.indptr)
    indices, indptr = (np.asarray(jax.device_get(a)) for a in (csr.indices, csr.indptr))
    try:
        _check_sorted_rows(indices, indptr)
    except ValueError:
        pass
    else:
        return csr.data, csr.indices, None
    rows = np.repeat(np.arange(indptr.size - 1, dtype=np.int64), np.diff(indptr).astype(np.int64))
    # lexsort is stable: equal (row, column) pairs keep their source order.
    permutation = np.lexsort((indices, rows))
    permutation.flags.writeable = False
    # The permutation only addresses existing slots; no duplicate coalescing.
    order = jnp.asarray(permutation, dtype=jnp.int64 if jax.config.jax_enable_x64 else jnp.int32)
    data = csr.data if u.get_mantissa(csr.data).size == 1 else csr.data[order]
    return data, csr.indices[order], permutation
