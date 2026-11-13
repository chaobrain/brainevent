# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Typed JAX bindings for ordered tile SpMM; preparation runs once on the host."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from brainevent._op import load_cuda_file, load_cuda_inline
from brainevent._tcsr.binary.combined.planner import tile_schedule

ROOT = Path(__file__).resolve().parent
COMMON = ROOT.parents[2] / "include" / "brainevent"
VALUE_SUFFIX = {jnp.dtype(t): name for t, name in ((jnp.float16, "f16"), (jnp.bfloat16, "bf16"),
                                                  (jnp.float32, "f32"), (jnp.float64, "f64"))}
VALUE_CODE = {dtype: i for i, dtype in enumerate(VALUE_SUFFIX)}
EVENT_SUFFIX = {jnp.dtype(jnp.bool_): "bool", jnp.dtype(jnp.float32): "float"}
INT32_MAX = np.iinfo(np.int32).max


@dataclass(frozen=True, eq=False)
class Plan:
    """Immutable metadata tied to CSR structure, shape, batch and output dtype."""
    rows: int
    cols: int
    batch: int
    nnz: int
    value_dtype: Any
    tile_width: int
    permutation: np.ndarray
    local_targets: np.ndarray
    tile_offsets: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray


def tile_width(dtype: Any) -> int:
    """Return the maximum columns per 96 KiB tile for a supported dtype."""
    dtype = jnp.dtype(dtype)
    if dtype not in VALUE_SUFFIX:
        raise ValueError("values must be f16, bf16, f32 or f64")
    return 12288 if dtype == jnp.dtype(jnp.float64) else 24576


def _schedule(cols: int, batch: int, dtype: Any):
    accumulator_bytes = 8 if tile_width(dtype) == 12288 else 4
    return tile_schedule(cols, batch, accumulator_bytes)


def _csr(indptr, indices, cols, batch, dtype):
    ptr, idx = np.asarray(indptr), np.asarray(indices)
    if ptr.ndim != 1 or ptr.size < 1 or ptr.dtype not in (np.int32, np.int64):
        raise ValueError("expected rank-one int32/int64 CSR pointers")
    if idx.ndim != 1 or idx.dtype not in (np.int32, np.int64):
        raise ValueError("expected rank-one int32/int64 columns")
    rows = ptr.size - 1
    if not (0 <= cols <= INT32_MAX and 0 <= batch <= 65535 and rows <= INT32_MAX):
        raise ValueError("invalid dimensions")
    if ptr[0] != 0 or ptr[-1] != idx.size or np.any(ptr[1:] < ptr[:-1]):
        raise ValueError("invalid CSR pointers")
    if ptr.dtype == np.int32 and idx.size > INT32_MAX:
        raise ValueError("int32 indptr cannot address this many edges")
    if np.any(idx < 0) or np.any(idx >= cols):
        raise ValueError("column out of range")
    return ptr.copy(), idx.copy(), _schedule(cols, batch, dtype).tile_width


def _freeze(*arrays):
    for array in arrays:
        array.flags.writeable = False


def prepare(indptr, indices, cols: int, batch: int, *, dtype: Any = jnp.float32) -> Plan:
    """Prepare sorted tile metadata from host CSR structure.

    Parameters
    ----------
    indptr, indices : numpy.ndarray
        int32/int64 row pointers and column indices; duplicates are allowed.
    cols, batch : int
        Output columns and static batch size.
    dtype : dtype, optional
        Weight/output dtype. Tile width also depends on columns and batch;
        changing batch requires matching prepared metadata.

    Returns
    -------
    Plan
        Reusable metadata. Values remain runtime inputs in their original CSR order.
    """
    ptr, idx, width = _csr(indptr, indices, cols, batch, dtype)
    rows, nnz = ptr.size - 1, idx.size
    tiles = (cols + width - 1) // width
    permutation = np.arange(nnz, dtype=np.int64)
    offsets = np.empty((rows, tiles + 1), dtype=ptr.dtype)
    bounds = np.arange(tiles + 1, dtype=np.int64) * width
    for row in range(rows):
        begin, end = int(ptr[row]), int(ptr[row + 1])
        order = np.argsort(idx[begin:end], kind="stable")
        permutation[begin:end] = begin + order
        offsets[row] = begin + np.searchsorted(idx[begin:end][order], bounds)
    local = (idx[permutation] % width).astype(np.uint16)
    offsets = offsets.ravel()
    _freeze(ptr, idx, permutation, local, offsets)
    return Plan(rows, cols, batch, nnz, jnp.dtype(dtype), width, permutation, local, offsets, ptr, idx)


@lru_cache(maxsize=2)
def _load(weighted: bool = False) -> str:
    sources = [ROOT / "combined_spmm_binary.cu", ROOT / "combined_kernels.cuh",
               ROOT / "combined_impl.cuh", ROOT / "combined_ffi.cuh",
               ROOT / "planner.h", ROOT / "planner.py",
               COMMON / "cuda_traits.cuh", COMMON / "op_traits.cuh"]
    if weighted:
        sources.append(ROOT / "combined_spmm_weighted.cu")
    family = "weighted_" if weighted else ""
    name = f"spmm_combined_{family}" + hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()[:12]
    if weighted:
        source = (f'#define BE_SPMM_STANDALONE\n#include "{sources[0]}"\n'
                  '#undef BE_SPMM_STANDALONE\n' + sources[-1].read_text())
        load_cuda_inline(name, source, extra_include_paths=[str(ROOT), str(COMMON)], allow_cuda_graph=False)
    else:
        load_cuda_file(sources[0], name=name, extra_include_paths=[str(ROOT), str(COMMON)], allow_cuda_graph=False)
    return name


def _validate(values, events, plan, weighted=False):
    if (jnp.dtype(values.dtype) != plan.value_dtype or
            plan.tile_width != _schedule(plan.cols, plan.batch, plan.value_dtype).tile_width):
        raise ValueError("values dtype or tile width disagrees with the plan")
    if values.shape not in ((), (1,), (plan.nnz,)) or events.shape != (plan.batch, plan.rows):
        raise ValueError("input shape disagrees with the plan")
    if weighted:
        if events.dtype != values.dtype:
            raise ValueError("weighted SpMM requires events and values to share the plan's floating dtype")
    elif jnp.dtype(events.dtype) not in EVENT_SUFFIX:
        raise ValueError("events must be bool or float32")


def workspace_words(rows: int, batch: int) -> int:
    """Number of uint32 scratch words for the native combined launcher."""
    chunks = (rows + 4095) // 4096
    flags_offset = ((batch * 3 + 1) // 2 * 2) * 4
    scratch_bytes = flags_offset + batch * chunks * 8 if rows and batch else 0
    return max(1, (scratch_bytes + 3) // 4)


def spmm(values: jax.Array, events: jax.Array, plan: Plan) -> jax.Array:
    """Compute binary events @ CSR weights, overwriting a BN output.

    Parameters
    ----------
    values : jax.Array
        Scalar/shape (1,) shared weight, or shape (nnz,) weights in original
        CSR order, in the plan's dtype. Both modes reuse the same plan.
    events : jax.Array
        bool/float32 array of shape (batch, rows); nonzero values are active.
    plan : Plan
        Metadata from :func:`prepare`.

    Returns
    -------
    jax.Array
        Array of shape (batch, cols), in the weights dtype.
    """
    return _spmm(values, events, plan, weighted=False)


def spmm_weighted(values: jax.Array, events: jax.Array, plan: Plan) -> jax.Array:
    """Compute event-scaled SpMM, skipping rows whose event equals zero.

    Parameters
    ----------
    values : jax.Array
        Scalar/shape (1,) shared weight or (nnz,) original CSR weights.
    events : jax.Array
        Shape (batch, rows), same f16/bf16/f32/f64 dtype as values. Negative
        and NaN events are active; zero events skip even NaN/Inf weights.
    plan : Plan
        Metadata from :func:`prepare`, also reusable for binary execution.

    Returns
    -------
    jax.Array
        Shape (batch, cols), overwritten in the weights dtype. Products and
        sums use float32 for f16/bf16/f32, float64 for f64.

    Raises
    ------
    ValueError
        If input shapes/dtypes disagree with the plan or offsets are narrowed.
    """
    return _spmm(values, events, plan, weighted=True)


def _spmm(values: jax.Array, events: jax.Array, plan: Plan, *, weighted: bool) -> jax.Array:
    _validate(values, events, plan, weighted)
    if values.ndim == 0:
        values = values.reshape((1,))
    homo = values.shape == (1,)
    outputs = (jax.ShapeDtypeStruct((plan.batch, plan.cols), values.dtype),
               jax.ShapeDtypeStruct((max(1, plan.batch * plan.rows),), jnp.int32),
               jax.ShapeDtypeStruct((workspace_words(plan.rows, plan.batch),), jnp.uint32))
    weight_mode = "_homo" if homo else ""
    family = "combined_weighted" if weighted else "combined"
    suffix = VALUE_SUFFIX[values.dtype]
    if not weighted:
        suffix += f"_{EVENT_SUFFIX[events.dtype]}"
    offsets = jnp.asarray(plan.tile_offsets)
    if offsets.dtype != plan.tile_offsets.dtype:
        raise ValueError("JAX must preserve int64 tile_offsets; enable x64 or allow explicit int64 dtypes")
    name = f"{_load(weighted)}.{family}{weight_mode}_{suffix}"
    prepared_values = values if homo else values[jnp.asarray(plan.permutation)]
    out, _, _ = jax.ffi.ffi_call(name, outputs)(prepared_values,
        jnp.asarray(plan.local_targets), offsets, events,
        tile_width=plan.tile_width, value_type=VALUE_CODE[plan.value_dtype])
    return out
