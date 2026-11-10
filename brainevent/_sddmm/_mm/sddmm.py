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
"""

from functools import lru_cache, partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import sparse

from brainevent._compatible_import import match_varying_axes, varying_axes_of
from brainevent._misc import _csr_to_coo
from brainevent._op import XLACustomKernel, load_cuda_file
from brainevent._op.util import event_active, select_active
from .._common import (
    BINARY_TARGET_SUFFIXES, VALUE_SUFFIXES, Mode, _CSRBackend, check_jax_backend,
    _cusparse_event_kernel, _cusparse_sampled,
)


_DIRECTORY = Path(__file__).parent
_TASK_NNZ = 512  # batched_v3::kTask, also 16 * the fixed lane group of 32.
_BATCH_CHUNK = 64


@lru_cache(maxsize=None)
def _load_hybrid(weighted: bool) -> str:
    kind = 'weighted' if weighted else 'binary'
    name = f'csr_sddmm_hybrid_{kind}'
    load_cuda_file(
        _DIRECTORY / f'hybrid_sddmm_{kind}.cu', name=name,
        extra_include_paths=[str(_DIRECTORY), str(_DIRECTORY.parent / '_common')],
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
    """Compute event slot values for CSR lowering or direct CSR/COO calls."""
    if indptr is not None:
        rows, cols = _csr_to_coo(indices, indptr)
        indices = jnp.stack([rows, cols], axis=1)
    else:
        rows, cols = indices[:, 0], indices[:, 1]
    unit, out = (rows, cols) if transpose else (cols, rows)
    factors = events.astype(ct.dtype) if weighted else event_active(events).astype(ct.dtype)
    left, right = (factors, ct.T) if transpose else (ct, factors.T)
    if varying_axes_of(left, right, indices):
        sampled_dot = partial(
            jax.lax.dot_general,
            left.at[rows, :].get(mode='fill', fill_value=0),
            right.at[:, cols].get(mode='fill', fill_value=0),
            dimension_numbers=(((1,), (0,)), ((0,), (1,))),
            precision=jax.lax.Precision.HIGHEST,
        )
    else:
        sampled_dot = partial(
            sparse.bcoo_dot_general_sampled, left, right, indices,
            dimension_numbers=(((1,), (0,)), ((), ())),
        )
    # Inactive events must suppress NaN/Inf before the batch reduction.
    return (jax.lax.cond(
        jnp.all(jnp.isfinite(ct)), sampled_dot,
        lambda: jnp.sum(select_active(
            event_active(events.at[unit, :].get(mode='fill', fill_value=0)),
            (events.at[unit, :].get(mode='fill', fill_value=0).astype(ct.dtype)
             * ct.at[out, :].get(mode='fill', fill_value=0))
            if weighted else ct.at[out, :].get(mode='fill', fill_value=0),
        ), axis=1),
    ),)


def _cuda_kernel(*, transpose: bool, weighted: bool, ct_info: jax.ShapeDtypeStruct,
                 indices_info: jax.ShapeDtypeStruct, indptr_info: jax.ShapeDtypeStruct,
                 **kwargs: Any):
    if not transpose:
        raise ValueError('cuda_raw SDDMM requires transpose=True; use jax_raw or auto')
    rows, nnz, batch = indptr_info.size - 1, indices_info.size, ct_info.shape[1]
    if not batch or not nnz:
        return lambda events, ct, indices, indptr: (jnp.zeros(indices.shape, ct.dtype),)
    if batch > 65535 or ct_info.shape[0] > 2**31 - 1:
        raise ValueError('matrix SDDMM CUDA requires batch <= 65535 and cols <= INT32_MAX')
    dtype = jnp.dtype(ct_info.dtype)
    attrs, infos = _cuda_plan(rows, batch, nnz, dtype, weighted)
    module = _load_hybrid(weighted)

    def kernel(events, ct, indices, indptr):
        if weighted:
            vector = events.astype(dtype)
            target = f'{module}.batched_v3_sddmm_weighted_{VALUE_SUFFIXES[dtype]}'
        else:
            vector = events
            if (dtype, jnp.dtype(vector.dtype)) not in BINARY_TARGET_SUFFIXES:
                vector = event_active(vector)
            target = f'{module}.batched_v3_sddmm_{BINARY_TARGET_SUFFIXES[(dtype, jnp.dtype(vector.dtype))]}'
        mask = _long_mask(indptr) if attrs['long_count'] else jnp.zeros(((rows + 31) // 32,), jnp.uint32)
        out = jax.ffi.ffi_call(
            target, infos, input_output_aliases={5: 0}, vmap_method='sequential',
        )(vector.T, ct.T, indices, indptr, mask, jnp.zeros((nnz,), dtype), **attrs)[0]
        return (match_varying_axes(out, events, ct, indices, indptr),)
    return kernel


def _auto_gpu_kernel(*, transpose: bool, ct_info: jax.ShapeDtypeStruct, csr_primitive=None, **kwargs: Any):
    backend = None if csr_primitive is None else csr_primitive._resolve_backend('gpu')
    if backend == 'cusparse':
        return _cusparse_event_kernel(transpose=transpose, ct_info=ct_info, **kwargs)
    native = transpose and ct_info.shape[1] <= 65535 and ct_info.shape[0] <= 2**31 - 1
    native = native and backend in (None, 'cuda_raw')
    if native:
        return _cuda_kernel(transpose=transpose, ct_info=ct_info, **kwargs)
    return partial(_jax_kernel, transpose=transpose, **kwargs)


def sampled(A: jax.Array, B: jax.Array, indices: jax.Array, indptr: jax.Array | None,
            *, mode: Mode, transpose: bool, backend: str | None | _CSRBackend) -> jax.Array:
    """Execute a normalized matrix sampled product.

    Parameters
    ----------
    A, B : jax.Array
        Dense operands of shape ``(rows, batch)`` and ``(batch, cols)``.
    indices, indptr : jax.Array or None
        COO coordinates, or CSR columns and row offsets.
    mode : {'float', 'binary', 'weighted'}
        Ordinary multiplication or event selection semantics.
    transpose : bool
        Events are on the left when true and on the right otherwise.
    backend : str or None
        GPU auto uses CUDA for supported transposed CSR events; other cases
        use JAX. ``jax_raw`` forces JAX in both directions.
        ``cusparse`` supports CSR products on GPU in both directions and all
        modes, with float32 computation for half types. COO and patterns with
        more slots than rows times columns are unsupported. Nonfinite event
        cotangents use exact JAX selection. Explicit native backends have no AD
        rules; use ``jax_raw`` for differentiation.

    Returns
    -------
    jax.Array
        One result per physical sparse slot.
    """
    check_jax_backend(backend)
    if backend == 'cusparse':
        if indptr is None:
            raise NotImplementedError('cusparse sampled products require CSR indices and indptr')
        if mode == 'float':
            return _cusparse_sampled(A, B, indices, indptr, mode=mode, transpose=transpose)
    # Explicit JAX calls stay differentiable without binding the event primitive.
    if mode != 'float' and indptr is not None and backend != 'jax_raw':
        if backend == 'cuda_raw' and not transpose:
            raise ValueError('cuda_raw SDDMM requires transpose=True; use jax_raw or auto')
        events, ct = (A, B.T) if transpose else (B.T, A)
        if backend == 'cuda_raw' and indices.size and (ct.shape[1] > 65535 or ct.shape[0] > 2**31 - 1):
            raise ValueError('matrix SDDMM CUDA requires batch <= 65535 and cols <= INT32_MAX')
        source = backend.primitive if isinstance(backend, _CSRBackend) else None
        return csr_sddmm_p(
            events, ct, indices, indptr,
            outs=[jax.ShapeDtypeStruct(indices.shape, ct.dtype)],
            transpose=transpose, weighted=mode == 'weighted', backend='auto' if source is not None else backend,
            csr_primitive=source,
            ct_info=jax.ShapeDtypeStruct(ct.shape, ct.dtype),
            indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
            indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        )[0]
    if backend == 'cuda_raw':
        raise NotImplementedError('MM CUDA supports transposed CSR event products; use backend="jax_raw"')
    if mode != 'float':
        events, ct = (A, B.T) if transpose else (B.T, A)
        return _jax_kernel(events, ct, indices, indptr,
                           transpose=transpose, weighted=mode == 'weighted')[0]
    if indptr is not None:
        rows, cols = _csr_to_coo(indices, indptr)
        indices = jnp.stack([rows, cols], axis=1)
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


csr_sddmm_p = XLACustomKernel('csr_sddmm')
csr_sddmm_p.def_cuda_raw_kernel(_cuda_kernel)
csr_sddmm_p.def_kernel('cusparse', 'gpu', _cusparse_event_kernel)
csr_sddmm_p.def_kernel('auto', 'gpu', _auto_gpu_kernel, asdefault=True)
for _platform in ('cpu', 'gpu', 'tpu'):
    csr_sddmm_p.def_kernel('jax_raw', _platform, lambda **kwargs: partial(_jax_kernel, **kwargs))
for _platform in ('cpu', 'tpu'):
    csr_sddmm_p.def_kernel('auto', _platform, lambda **kwargs: partial(_jax_kernel, **kwargs))
csr_sddmm_p.def_tags('csr', 'sddmm')
