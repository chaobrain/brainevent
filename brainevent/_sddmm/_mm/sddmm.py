# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Matrix sampled products with portable JAX and transposed CSR CUDA backends.

GPU auto selects CUDA for transposed CSR events within the native dimensions;
other directions and larger batches use JAX. CUDA uses batch-first operands,
zero-initialized weight cotangents and private scratch. Its schedule is sized
from shapes, and its packed row mask is computed from the runtime CSR pointers,
so a compiled call supports changing sparsity patterns without host transfers.
Native batch contributions use atomic additions; rounding order can vary.
Weighted CUDA tests event activity after casting events to the cotangent dtype.
Explicit ``cusparse`` uses a shared cuSPARSE SDDMM FFI for CSR products in both
directions and all modes. Half types compute in float32; nonfinite event
cotangents use exact selection, with activity tested after casting.
JAX masks each slot's cotangents where its event factor is zero, without a
branch on global finiteness, and accumulates half-precision batches in float32.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import sparse

from brainevent._compatible_import import match_varying_axes, varying_axes_of
from brainevent._misc import _csr_to_coo
from brainevent._op import load_cuda_file
from brainevent._op.util import event_factor
from brainevent._sddmm import _common
from brainevent._sddmm._common import native_event_target


_DIRECTORY = Path(__file__).parent
NAME, LABEL = 'SDDMM', 'MM'
_TASK_NNZ = 512  # batched_v3::kTask, also 16 * the fixed lane group of 32.
_BATCH_CHUNK = 64
_MAX_BATCH = 65535
_INT32_MAX = 2**31 - 1


@lru_cache(maxsize=None)
def _load_hybrid(weighted: bool) -> str:
    kind = 'weighted' if weighted else 'binary'
    name = f'csr_sddmm_hybrid_{kind}'
    load_cuda_file(
        _DIRECTORY / f'hybrid_sddmm_{kind}.cu', name=name,
        extra_include_paths=[str(_DIRECTORY)],
        allow_cuda_graph=True,
    )
    return name


def _cuda_plan(rows: int, batch: int, nnz: int, dtype: Any, weighted: bool):
    """Bound scratch without reading CSR values while tracing.

    The native launcher only tests short_count/long_count for zero. Use them
    as schedule-presence flags, enabling both whenever either is possible.
    For one row the shape determines its exact classification. Runtime mask
    bits and device counters determine the actual rows/tasks for every call.
    """
    chunk = min(batch, _BATCH_CHUNK)
    long_count = int(nnz > _TASK_NNZ)
    short_count = int(rows > 1 or not long_count)
    # For h long rows, sum ceil(length/512) <= floor((nnz + 511*h)/512),
    # and h <= min(rows, nnz//513). All events may be active in every batch.
    heavy_rows = min(rows, nnz // (_TASK_NNZ + 1))
    capacity = ((nnz + (_TASK_NNZ - 1) * heavy_rows) // _TASK_NNZ) * chunk if long_count else 0
    groups = (batch + chunk - 1) // chunk
    value_bytes = jnp.dtype(dtype).itemsize
    queue_words = (capacity * value_bytes + 3) // 4 if weighted else 0
    words = 2 * batch + 2 * groups + 3 * capacity + queue_words
    if value_bytes == 2:
        words += nnz  # Native half/bfloat16 accumulation is float32.
    selected = rows * (chunk if long_count else batch)
    if max(words * 4, selected * 4) > 2**63 - 1:
        raise ValueError('matrix SDDMM CUDA scratch exceeds int64 range')
    attrs = dict(group=32, chunk=chunk, short_count=short_count,
                 long_count=long_count, capacity=capacity)
    infos = (
        jax.ShapeDtypeStruct((nnz,), dtype),
        jax.ShapeDtypeStruct((words,), jnp.uint32),
        jax.ShapeDtypeStruct((selected,), jnp.int32),
    )
    return attrs, infos


def _long_mask(indptr: jax.Array) -> jax.Array:
    """Pack the current row classification into the native uint32 bit mask."""
    rows = indptr.size - 1
    bits = (jnp.diff(indptr) > _TASK_NNZ).astype(jnp.uint32)
    bits = jnp.pad(bits, (0, -rows % 32)).reshape((-1, 32))
    return jnp.sum(bits << jnp.arange(32, dtype=jnp.uint32), axis=1, dtype=jnp.uint32)


def _jax_kernel(events: jax.Array, ct: jax.Array, indices: jax.Array,
                indptr: jax.Array | None, *, transpose: bool, weighted: bool,
                **kwargs: Any) -> tuple[jax.Array]:
    """Compute event slot values for CSR lowering or direct CSR/COO calls.

    Branch-free: each slot and batch entry masks its own cotangent before the
    batch reduction, instead of testing the whole cotangent for finiteness.
    Weighted activity is tested after the cast to ``ct.dtype``, as in the
    native kernels. Half types accumulate the batch in float32.
    """
    if indptr is not None:
        rows, cols = _csr_to_coo(indices, indptr)
    else:
        rows, cols = indices[:, 0], indices[:, 1]
    unit, out = (rows, cols) if transpose else (cols, rows)
    factors = event_factor(events, dtype=ct.dtype, weighted=weighted)
    # CSR coordinates are non-negative: skip the per-slot negative-index wrap.
    wrap = indptr is None
    factor = factors.at[unit, :].get(mode='fill', fill_value=0, wrap_negative_indices=wrap)
    cotangent = ct.at[out, :].get(mode='fill', fill_value=0, wrap_negative_indices=wrap)
    cotangent = jnp.where((factor != 0) | jnp.isfinite(cotangent), cotangent, jnp.zeros_like(cotangent))
    accumulate = jnp.float32 if ct.dtype.itemsize < 4 else None
    result = jax.lax.dot_general(
        factor, cotangent, dimension_numbers=(((1,), (1,)), ((0,), (0,))),
        precision=jax.lax.Precision.HIGHEST, preferred_element_type=accumulate,
    )
    return (result.astype(ct.dtype),)


def _cuda_kernel(*, transpose: bool, weighted: bool, ct_info: jax.ShapeDtypeStruct,
                 indices_info: jax.ShapeDtypeStruct, indptr_info: jax.ShapeDtypeStruct,
                 **kwargs: Any):
    if not transpose:
        raise ValueError('cuda_raw SDDMM requires transpose=True; use jax_raw or auto')
    rows, nnz, batch = indptr_info.size - 1, indices_info.size, ct_info.shape[1]
    if not batch or not nnz:
        return lambda events, ct, indices, indptr: (jnp.zeros(indices.shape, ct.dtype),)
    if not native_dims(ct_info.shape):
        raise ValueError('matrix SDDMM CUDA requires batch <= 65535 and cols <= INT32_MAX')
    dtype = jnp.dtype(ct_info.dtype)
    attrs, infos = _cuda_plan(rows, batch, nnz, dtype, weighted)
    module = _load_hybrid(weighted)

    def kernel(events, ct, indices, indptr):
        target, vector = native_event_target(
            'batched_v3_sddmm_weighted' if weighted else 'batched_v3_sddmm', dtype, events, weighted=weighted)
        mask = _long_mask(indptr) if attrs['long_count'] else jnp.zeros(((rows + 31) // 32,), jnp.uint32)
        out = jax.ffi.ffi_call(
            f'{module}.{target}', infos, input_output_aliases={5: 0}, vmap_method='sequential',
        )(vector.T, ct.T, indices, indptr, mask, jnp.zeros((nnz,), dtype), **attrs)[0]
        return (match_varying_axes(out, events, ct, indices, indptr),)
    return kernel


def native_dims(ct_shape: tuple[int, ...]) -> bool:
    """Whether a ``(cols, batch)`` cotangent fits the native CUDA kernel."""
    return ct_shape[1] <= _MAX_BATCH and ct_shape[0] <= _INT32_MAX


def bind_events(events: jax.Array, ct: jax.Array, indices: jax.Array, indptr: jax.Array, *,
                config_spec: Any = None, **kwargs: Any) -> jax.Array:
    """Bind matrix CSR events; the native plan is sized from shapes, not ``config_spec``."""
    if kwargs['backend'] == 'cuda_raw' and indices.size and not native_dims(ct.shape):
        raise ValueError('matrix SDDMM CUDA requires batch <= 65535 and cols <= INT32_MAX')
    return _common.bind_events(csr_sddmm_p, events, ct, indices, indptr, **kwargs)


def float_product(A: jax.Array, B: jax.Array, indices: jax.Array) -> jax.Array:
    """Sample ``A (rows, batch) @ B (batch, cols)`` at COO coordinates."""
    if varying_axes_of(A, B, indices):
        left = A.at[indices[:, 0], :].get(mode='fill', fill_value=0)
        right = B.at[:, indices[:, 1]].get(mode='fill', fill_value=0)
        return jax.lax.dot_general(
            left, right, dimension_numbers=(((1,), (0,)), ((0,), (1,))),
            precision=jax.lax.Precision.HIGHEST,
        )
    return sparse.bcoo_dot_general_sampled(
        A, B, indices, dimension_numbers=(((1,), (0,)), ((), ())),
    )


csr_sddmm_p = _common.event_primitive('csr_sddmm', 'sddmm', _cuda_kernel, _jax_kernel, native_dims)
