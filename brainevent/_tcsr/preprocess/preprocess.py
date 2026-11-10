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
from brainevent._tcsr.binary.combined.col_major.jax_ffi import (
    prepare as _prepare_sorted,
)

__all__ = [
    'TileGeometry', 'build_tile_geometries', 'geometry_widths',
    'TilePlan', 'build_binary_mm_plan', 'ensure_preparation_context',
    'sort_csr', 'validate_preparation',
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


def build_tile_geometries(
    indices: Any, indptr: Any, *, shape: tuple[int, int], transpose: bool,
) -> dict[tuple[bool, int, np.dtype], TileGeometry]:
    """Prepare one traversal's complete geometry table from sorted structure.

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

    Returns
    -------
    dict
        Complete geometry table keyed by direction, actual width and offset
        dtype. Arrays for each width cover every batch and value dtype that
        selects it. The caller publishes this table only after success.

    Raises
    ------
    ValueError
        Called while staging outside compile-time evaluation, or shape,
        pointer values or sortedness are invalid. Explicit int64 offsets
        must survive JAX placement unchanged.
    TypeError
        Indices or pointers have unsupported rank or dtype.
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
    for row in range(source):
        begin, end = int(ptr[row]), int(ptr[row + 1])
        if end - begin > 1 and np.any(idx[begin + 1:end] < idx[begin:end - 1]):
            raise ValueError('traversal indices must be sorted within each row')
    placement = _geometry_placement(indices, indptr)
    result = {}
    for width in geometry_widths(target):
        tile_count = (target + width - 1) // width
        bounds: Any = np.arange(tile_count + 1, dtype=np.int64) * width
        offsets = np.empty((source, tile_count + 1), dtype=ptr.dtype)
        for row in range(source):
            begin, end = int(ptr[row]), int(ptr[row + 1])
            offsets[row] = begin + np.searchsorted(idx[begin:end], bounds)
        local = (idx % width).astype(np.uint16)
        flat_offsets = offsets.ravel()
        local.flags.writeable = flat_offsets.flags.writeable = False
        local_array = jax.device_put(local, placement)
        offset_array = jax.device_put(flat_offsets, placement)
        if np.dtype(offset_array.dtype) != ptr.dtype:
            raise ValueError('JAX must preserve int64 tile offsets; enable x64 before preparation.')
        geometry = TileGeometry(shape, idx.size, bool(transpose), width, ptr.dtype,
                                local_array, offset_array)
        result[geometry.key] = geometry
    return result


@dataclass(frozen=True, eq=False)
class TilePlan:
    """Immutable tile description belonging to one matrix structure cache.

    Parameters
    ----------
    shape : tuple of int
        Canonical CSR shape, independent of the selected traversal.
    nnz : int
        Number of stored edges, including duplicate coordinates.
    transpose : bool
        True for primary traversal; False for the common mirror traversal.
    batch : int
        Prepared event batch size.
    value_dtype : numpy.dtype
        Prepared weight and output dtype.
    tile_width : int
        Actual adaptive accumulation tile width.
    offset_dtype : numpy.dtype
        Absolute offset dtype, matching the selected traversal's pointers.
    local_targets, tile_offsets : jax.Array
        Dynamic uint16 local targets and flattened absolute edge boundaries.

    Notes
    -----
    Structure ownership is supplied by the containing cache. Matching shape
    and edge count alone never permits a plan to be moved to another cache.
    No mirror permutation, weight arrays or native scratch are stored here.
    """

    shape: tuple[int, int]
    nnz: int
    transpose: bool
    batch: int
    value_dtype: np.dtype
    tile_width: int
    offset_dtype: np.dtype
    local_targets: Any
    tile_offsets: Any

    @property
    def key(self) -> tuple[bool, int, np.dtype]:
        """Return the immutable direction, batch and value-dtype cache key.

        Returns
        -------
        tuple
            Direction, batch size and NumPy value dtype.
        """
        return self.transpose, self.batch, self.value_dtype

    @property
    def metadata(self) -> tuple:
        """Return an immutable pytree description, excluding dynamic arrays.

        Returns
        -------
        tuple
            Canonical shape, edge count, direction, batch size, value dtype,
            tile width and offset dtype.
        """
        return (self.shape, self.nnz, self.transpose, self.batch,
                self.value_dtype, self.tile_width, self.offset_dtype)

    @classmethod
    def from_metadata(cls, metadata: tuple, local_targets: Any, tile_offsets: Any) -> TilePlan:
        """Restore a plan without inspecting or converting its dynamic leaves.

        Parameters
        ----------
        metadata : tuple
            Static description returned by :attr:`metadata`.
        local_targets, tile_offsets : object
            Leaves supplied by JAX, including tracers or placeholders.

        Returns
        -------
        TilePlan
            Restored description with the exact supplied leaves.
        """
        return cls(metadata[0], metadata[1], metadata[2], metadata[3],
                   metadata[4], metadata[5], metadata[6], local_targets, tile_offsets)


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


def sort_csr(csr: Any) -> tuple[Any, jax.Array]:
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
    """
    ensure_preparation_context(csr.indices, csr.indptr)
    indices, indptr = (np.asarray(jax.device_get(a)) for a in (csr.indices, csr.indptr))
    permutation = np.arange(indices.size, dtype=np.int64)
    for row in range(indptr.size - 1):
        begin, end = int(indptr[row]), int(indptr[row + 1])
        permutation[begin:end] = begin + np.argsort(indices[begin:end], kind='stable')
    # The permutation only addresses existing slots; no duplicate coalescing.
    order = jnp.asarray(permutation, dtype=jnp.int64 if jax.config.jax_enable_x64 else jnp.int32)
    data = csr.data if u.get_mantissa(csr.data).size == 1 else csr.data[order]
    return data, csr.indices[order]


def build_binary_mm_plan(
    indices: Any, indptr: Any, *, shape: tuple[int, int], transpose: bool,
    batch: int, value_dtype: Any,
) -> TilePlan:
    """Prepare adaptive tiles from a sorted primary or common mirror traversal.

    Parameters
    ----------
    indices, indptr : array_like
        Sorted traversal indices and int32/int64 row pointers. These arrays
        belong to the primary CSR when transposed, or its common mirror.
    shape : tuple of int
        Canonical CSR shape in both directions.
    transpose : bool
        Select primary (True) or mirror (False) traversal.
    batch : int
        Static batch size between zero and 65535, inclusive.
    value_dtype : dtype
        Supported weight/output dtype: float16, bfloat16, float32 or float64.

    Returns
    -------
    TilePlan
        Plan with dynamic JAX arrays, requiring no host work during MM.

    Raises
    ------
    ValueError
        If called while tracing, the CSR structure is invalid or unsorted,
        dimensions are invalid, or JAX would narrow int64 offsets.
    TypeError
        If traversal indices are not int32.
    """
    ensure_preparation_context(indices, indptr)
    batch = validate_preparation(batch, value_dtype)
    if not isinstance(transpose, (bool, np.bool_)):
        raise ValueError('transpose must be True or False')
    if len(shape) != 2 or any(isinstance(s, (bool, np.bool_)) for s in shape):
        raise ValueError('shape must contain two nonnegative integer dimensions')
    try:
        shape = (operator.index(shape[0]), operator.index(shape[1]))
    except TypeError as exc:
        raise ValueError('shape must contain two nonnegative integer dimensions') from exc
    idx, ptr = (np.asarray(jax.device_get(a)) for a in (indices, indptr))
    if idx.dtype != np.dtype('int32'):
        raise TypeError('TCSR traversal indices must be int32')
    source, target = shape if transpose else shape[::-1]
    if ptr.shape != (source + 1,):
        raise ValueError('traversal indptr length does not match the canonical shape')
    prepared = _prepare_sorted(ptr, idx, target, batch, dtype=value_dtype)
    local_targets = jnp.asarray(prepared.local_targets)
    tile_offsets = jnp.asarray(prepared.tile_offsets)
    if np.dtype(tile_offsets.dtype) != prepared.tile_offsets.dtype:
        raise ValueError('JAX must preserve int64 tile offsets; enable x64 before preparation.')
    return TilePlan(shape, idx.size, bool(transpose), batch, np.dtype(value_dtype),
                    prepared.tile_width, np.dtype(tile_offsets.dtype), local_targets, tile_offsets)
