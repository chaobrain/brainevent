# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Binary/weighted column-major tile SpMM with runtime weight permutations.

For W(M, N), events are (B, N) and output is (B, M). Preparation preserves
sorted mirror edge order; only the CUDA weight reads use perm.
"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from brainevent._op import load_cuda_inline
from .. import jax_ffi as _base
from ..jax_ffi import workspace_words

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "combined_spmm_col_major_binary.cu"


@dataclass(frozen=True, eq=False)
class Plan:
    """Store immutable tile metadata in the supplied mirror's edge order.

    Parameters
    ----------
    rows, cols, batch, nnz : int
        Input columns N, output rows M, batch size B and edge count.
    value_dtype : dtype
        Bound weight/output dtype.
    tile_width : int
        Output rows per accumulation tile.
    local_targets : numpy.ndarray
        Read-only uint16 targets within each tile, shape (nnz,).
    tile_offsets : numpy.ndarray
        Read-only int32/int64 absolute edge boundaries, flattened from
        (N, ceil(M / tile_width) + 1). Dtype matches the mirror pointers.
    """

    rows: int
    cols: int
    batch: int
    nnz: int
    value_dtype: Any
    tile_width: int
    local_targets: npt.NDArray[np.uint16]
    tile_offsets: npt.NDArray[Any]


def prepare(indptr: npt.NDArray[Any], indices: npt.NDArray[Any], output_rows: int,
            batch: int, *, dtype: Any = jnp.float32) -> Plan:
    """Prepare tile boundaries from a sorted host column mirror.

    Parameters
    ----------
    indptr, indices : numpy.ndarray
        Int32/int64 mirror pointers and output row indices. Indices must be
        nondecreasing within every column; duplicate edges are preserved.
    output_rows, batch : int
        Original matrix row count M and batch size B.
    dtype : dtype, optional
        Weight/output dtype: float16, bfloat16, float32 or float64. Tile width
        also depends on output rows and batch; changed batch needs new metadata.

    Returns
    -------
    Plan
        Tile metadata preserving the mirror's order. Its existing perm remains
        a runtime input; preparation never reads or reorders weights.

    Raises
    ------
    ValueError
        If dimensions, structure, sortedness or dtype are invalid.
    """
    ptr, idx, width = _base._csr(indptr, indices, output_rows, batch, dtype)
    rows, nnz = ptr.size - 1, idx.size
    tiles = (output_rows + width - 1) // width
    bounds = np.arange(tiles + 1, dtype=np.int64) * width
    offsets = np.empty((rows, tiles + 1), dtype=ptr.dtype)
    for col in range(rows):
        begin, end = int(ptr[col]), int(ptr[col + 1])
        targets = idx[begin:end]
        if np.any(targets[1:] < targets[:-1]):
            raise ValueError("mirror indices must be sorted within each column; sort perm together")
        offsets[col] = begin + np.searchsorted(targets, bounds)
    local = (idx % width).astype(np.uint16)
    offsets = offsets.ravel()
    _base._freeze(local, offsets)
    return Plan(rows, output_rows, batch, nnz, jnp.dtype(dtype), width, local, offsets)


@lru_cache(maxsize=2)
def _load(weighted: bool = False) -> str:
    backend = ROOT.parent / "combined_spmm_binary.cu"
    entry = ROOT / "combined_spmm_col_major_weighted.cu" if weighted else SOURCE
    sources = [entry, ROOT / "combined_col_major.cuh", ROOT / "combined_col_major_ffi.cuh", backend,
               ROOT.parent / "combined_impl.cuh",
               ROOT.parent / "combined_kernels.cuh",
               ROOT.parent / "planner.h", ROOT.parent / "planner.py",
               _base.COMMON / "cuda_traits.cuh", _base.COMMON / "op_traits.cuh"]
    if weighted:
        sources.append(ROOT.parent / "combined_spmm_weighted.cu")
    digest = hashlib.sha256(b"".join(path.read_bytes() for path in sources)).hexdigest()[:12]
    family = "weighted" if weighted else "binary"
    name = f"spmm_combined_col_major_{family}_{digest}"
    # Like density_split: JIT one translation unit, exposing only this family's
    # FFI annotations. Native callers compile and link the two .cu files.
    source = f'#define BE_SPMM_STANDALONE\n#include "{backend}"\n'
    if weighted:
        source += f'#include "{ROOT.parent / "combined_spmm_weighted.cu"}"\n'
    source += '#undef BE_SPMM_STANDALONE\n' + entry.read_text()
    load_cuda_inline(name, source, extra_include_paths=[str(ROOT), str(_base.COMMON)],
                     allow_cuda_graph=False)
    return name


def _validate(values: jax.Array, perm: jax.Array, events: jax.Array, plan: Plan,
              weighted: bool = False) -> None:
    if (jnp.dtype(values.dtype) != plan.value_dtype or
            plan.tile_width != _base._schedule(plan.cols, plan.batch, plan.value_dtype).tile_width):
        raise ValueError("values dtype or tile width disagrees with the plan")
    if values.shape != (plan.nnz,):
        raise ValueError("hetero weights must have shape (nnz,)")
    if events.shape != (plan.batch, plan.rows):
        raise ValueError("events must have shape (batch, input_columns)")
    if weighted:
        if events.dtype != values.dtype:
            raise ValueError("weighted SpMM requires events and values to share the plan's floating dtype")
    elif events.dtype not in _base.EVENT_SUFFIX:
        raise ValueError("events must be bool/float32")
    if perm.dtype != plan.tile_offsets.dtype:
        raise ValueError("perm dtype must match the mirror indptr/tile_offsets")
    if perm.ndim != 1 or perm.shape != (plan.nnz,):
        raise ValueError("perm must be rank one with length nnz")


def spmm(values: jax.Array, perm: jax.Array, events: jax.Array, plan: Plan) -> jax.Array:
    """Compute binary e(events) @ W.T using original-order weights.

    Parameters
    ----------
    values : jax.Array
        Original-order weights, shape (nnz,), in the plan's dtype. For a TCSC
        mirror these are canonical TCSR values, not mirror-ordered values.
    perm : jax.Array
        Mirror edge to value slot mapping, shape (nnz,), same integer dtype as
        the prepared mirror pointers. Caller guarantees a valid permutation.
    events : jax.Array
        Bool/float32 (B, N) array; nonzero values, including NaN, are active.
    plan : Plan
        Prepared metadata for the same mirror, batch and weight dtype.

    Returns
    -------
    jax.Array
        Overwritten (B, M) result in the weight dtype.

    Raises
    ------
    ValueError
        If input types/shapes disagree with the plan or JAX narrows offsets.
    """
    return _spmm(values, perm, events, plan, weighted=False)


def spmm_weighted(values: jax.Array, perm: jax.Array, events: jax.Array, plan: Plan) -> jax.Array:
    """Compute event-scaled events @ W.T with canonical weights and no gather.

    Parameters
    ----------
    values : jax.Array
        Original-order (nnz,) weights, in the plan's floating dtype.
    perm : jax.Array
        Mirror edge to canonical value slot mapping, shape (nnz,), same dtype
        as prepared tile_offsets. Caller guarantees a valid permutation.
    events : jax.Array
        Shape (B, N), same dtype as values. Nonzero events (including negative
        and NaN values) scale their edges; zero events skip all their weights.
    plan : Plan
        Sorted mirror metadata from :func:`prepare`; shared with binary.

    Returns
    -------
    jax.Array
        Shape (B, M), overwritten in the weights dtype. Products and sums use
        float32 for f16/bf16/f32, float64 for f64.

    Raises
    ------
    ValueError
        If input types/shapes disagree with the plan or offsets are narrowed.
    """
    return _spmm(values, perm, events, plan, weighted=True)


def _spmm(values: jax.Array, perm: jax.Array, events: jax.Array, plan: Plan,
          *, weighted: bool) -> jax.Array:
    _validate(values, perm, events, plan, weighted)
    offsets = jnp.asarray(plan.tile_offsets)
    if offsets.dtype != plan.tile_offsets.dtype:
        raise ValueError("JAX must preserve int64 tile_offsets; enable x64 or allow explicit int64 dtypes")
    outputs = (jax.ShapeDtypeStruct((plan.batch, plan.cols), values.dtype),
               jax.ShapeDtypeStruct((max(1, plan.batch * plan.rows),), jnp.int32),
               jax.ShapeDtypeStruct((workspace_words(plan.rows, plan.batch),), jnp.uint32))
    family = "weighted" if weighted else "binary"
    suffix = _base.VALUE_SUFFIX[values.dtype]
    if not weighted:
        suffix += f"_{_base.EVENT_SUFFIX[events.dtype]}"
    target = f"{_load(weighted)}.combined_col_major_{family}_hetero_{suffix}"
    output, _, _ = jax.ffi.ffi_call(target, outputs)(
        values, jnp.asarray(plan.local_targets), offsets, perm, events,
        tile_width=plan.tile_width, value_type=_base.VALUE_CODE[plan.value_dtype])
    return output
