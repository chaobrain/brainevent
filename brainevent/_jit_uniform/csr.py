# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-

"""
Materialize a uniform-weight just-in-time connectivity (JITC) matrix directly
into Compressed Sparse Row (CSR) format using the light-RNG chunk kernels.

The mv (32-lane) and mm (4-thread AW-T4) kernels draw *different* matrices, so
``matrix_mode`` (``'mv'`` or ``'mm'``) is **required**: ``jitu_to_csr(..., matrix_mode='mv')``
reproduces exactly the matrix used by ``jitumv`` / ``jitu(matrix_mode='mv')`` (and the mv
event kernels), while ``matrix_mode='mm'`` reproduces the ``jitumm`` / ``jitu(matrix_mode='mm')``
matrix.  ``corder`` keeps its usual meaning (it selects the notrans/trans generation, exactly
as in ``jitumv``); the matrix class flips it on transpose.

Generation is split into two passes (``nnz`` is data dependent, XLA needs static shapes):

1. count : per-(row, chunk) counts (notrans) or per-row counts (trans)
2. fill  : column indices + values, given the offsets derived from the counts

Both CUDA and ``numba`` backends draw the same matrix; the two-pass flow is
eager either way (``nnz`` is read back between the passes).
"""

from pathlib import Path
from typing import Optional

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainevent._compatible_import import Tracer
from brainevent._data import _initialize_conn_length, _initialize_seed
from brainevent._misc import (
    _resolve_indptr_dtype, _require_jax_x64_for_int64, _as_int32_cuda_offsets,
)
from brainevent._numba_random import get_numba_light_rng_funcs
from brainevent._op import XLACustomKernel, load_cuda_file, numba_kernel
from brainevent._typing import MatrixShape
from .float import MatrixMode, _normalize_chunk_size, _normalize_matrix_mode, _MV_STRIDE, _MM_STRIDE

__all__ = [
    'jitu_to_csr',
    'jitu_csr_count_p',
    'jitu_csr_count_p_call',
    'jitu_csr_fill_p',
    'jitu_csr_fill_p_call',
]

_dtype_sfx = {
    np.dtype('float16'): '_f16',
    np.dtype('float32'): '_f32',
    np.dtype('float64'): '_f64',
    np.dtype('bfloat16'): '_bf16',
}


def _is_static_zero(value) -> bool:
    if isinstance(value, Tracer):
        return False
    try:
        return float(np.asarray(value)) == 0.0
    except (TypeError, ValueError):
        return False


def _n_chunks(n_cols: int, chunk_size: int) -> int:
    return 0 if n_cols <= 0 else (int(n_cols) + int(chunk_size) - 1) // int(chunk_size)


def _mode_infix(matrix_mode: MatrixMode) -> str:
    """CSR kernel infix: '' for mv (plain), '_mm_aw_t4' for mm."""
    return '' if _normalize_matrix_mode(matrix_mode) == 'mv' else '_mm_aw_t4'


# ──────────────────────────────────────────────────────────────────────
#  Count pass
# ──────────────────────────────────────────────────────────────────────

def _jitu_csr_count_cuda_kernel(
    shape: MatrixShape,
    corder: bool,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    matrix_mode: MatrixMode = 'mv',
    **kwargs,
):
    load_cuda_file(Path(__file__).parent.joinpath('csr.cu'), name='jit_uniform_csr')
    sfx = _dtype_sfx.get(np.dtype(kwargs['w_low_info'].dtype), '_f32')
    infix = _mode_infix(matrix_mode)
    n_rows, n_cols = int(shape[0]), int(shape[1])
    chunk_size_value = _normalize_chunk_size(n_cols, chunk_size, target_chunks)

    if corder:
        # notrans: chunk the mat columns; per-(row, chunk) counts.
        kernel_name = f'jit_uniform_csr.count_chunks_notrans{infix}{sfx}'

        def kernel(w_low, w_high, clen, seed):
            return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
                w_low, w_high, clen, seed,
                n_cols=np.int32(n_cols), chunk_size=np.int32(chunk_size_value),
            )
    else:
        # trans: seed over mat columns, walk mat rows, atomic per-row counts.
        kernel_name = f'jit_uniform_csr.count_chunks_trans{infix}{sfx}'

        def kernel(w_low, w_high, clen, seed):
            return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
                w_low, w_high, clen, seed,
                n_rows=np.int32(n_cols), n_cols=np.int32(n_rows),
                chunk_size=np.int32(chunk_size_value),
            )

    return kernel


def _jitu_csr_count_numba_kernel(
    shape: MatrixShape,
    corder: bool,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    matrix_mode: MatrixMode = 'mv',
    **kwargs,
):
    """Numba CPU count pass mirroring ``csr.cu``."""
    import numba
    _rng = get_numba_light_rng_funcs()
    _rng_init = _rng['init']
    _rng_next = _rng['next']
    _rng_bounded = _rng['bounded']
    _rng_initial_q = _rng['initial_q']

    stride = _MV_STRIDE if _normalize_matrix_mode(matrix_mode) == 'mv' else _MM_STRIDE
    n_cols = int(shape[1])
    cs_val = _normalize_chunk_size(n_cols, chunk_size, target_chunks)

    if corder:
        k = n_cols

        @numba.njit(fastmath=True)
        def kernel_impl(w_low, w_high, clen, seed, chunk_counts):
            m = chunk_counts.shape[0]
            n_chunks = chunk_counts.shape[1]
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            for row in range(m):
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * cs_val
                    if chunk_start >= k:
                        chunk_counts[row, chunk_id] = 0
                        continue
                    chunk_end = chunk_start + cs_val
                    if chunk_end > k:
                        chunk_end = k
                    chunk_width = chunk_end - chunk_start
                    cnt = 0
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            cnt += 1
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)
                    chunk_counts[row, chunk_id] = cnt
    else:
        m_walk = n_cols

        @numba.njit(fastmath=True)
        def kernel_impl(w_low, w_high, clen, seed, row_counts):
            row_counts[:] = 0
            k = row_counts.shape[0]
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            n_chunks = (k + cs_val - 1) // cs_val
            for row in range(m_walk):
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * cs_val
                    if chunk_start >= k:
                        break
                    chunk_end = chunk_start + cs_val
                    if chunk_end > k:
                        chunk_end = k
                    chunk_width = chunk_end - chunk_start
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            row_counts[chunk_start + local_j] += 1
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)

    def kernel(w_low, w_high, clen, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(w_low, w_high, clen, seed)

    return kernel


def jitu_csr_count_p_call(
    w_low,
    w_high,
    clen,
    seed,
    *,
    shape: MatrixShape,
    corder: bool,
    matrix_mode: MatrixMode = 'mv',
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    backend: Optional[str] = None,
):
    """Count pass. Returns ``chunk_counts`` (n_rows, n_chunks) for ``corder=True``,
    or ``row_counts`` (n_rows,) for ``corder=False``."""
    n_rows, n_cols = int(shape[0]), int(shape[1])
    matrix_mode = _normalize_matrix_mode(matrix_mode)
    chunk_size_value = _normalize_chunk_size(n_cols, chunk_size, target_chunks)
    w_low = jnp.atleast_1d(w_low)
    w_high = jnp.atleast_1d(w_high)
    clen = jnp.atleast_1d(clen)
    seed = jnp.atleast_1d(seed)

    if corder:
        n_chunks = _n_chunks(n_cols, chunk_size_value)
        if n_rows == 0 or n_chunks == 0:
            return (jnp.zeros((n_rows, n_chunks), dtype=jnp.int32),)
        out_info = jax.ShapeDtypeStruct((n_rows, n_chunks), jnp.int32)
    else:
        if n_rows == 0 or n_cols == 0:
            return (jnp.zeros((n_rows,), dtype=jnp.int32),)
        out_info = jax.ShapeDtypeStruct((n_rows,), jnp.int32)

    return jitu_csr_count_p(
        w_low, w_high, clen, seed,
        outs=[out_info],
        shape=(n_rows, n_cols),
        corder=corder,
        matrix_mode=matrix_mode,
        chunk_size=chunk_size_value,
        target_chunks=target_chunks,
        backend=backend,
        w_low_info=jax.ShapeDtypeStruct(w_low.shape, w_low.dtype),
        w_high_info=jax.ShapeDtypeStruct(w_high.shape, w_high.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
    )


jitu_csr_count_p = XLACustomKernel(
    'jitu_csr_count',
    doc="Low-level CUDA primitive counting non-zeros for light uniform CSR materialization.",
)
jitu_csr_count_p.def_cuda_raw_kernel(_jitu_csr_count_cuda_kernel, asdefault=True)
jitu_csr_count_p.def_numba_kernel(_jitu_csr_count_numba_kernel)
jitu_csr_count_p.def_call(jitu_csr_count_p_call)
jitu_csr_count_p.def_tags('jit_uniform', 'csr', 'light_rng')


# ──────────────────────────────────────────────────────────────────────
#  Fill pass
# ──────────────────────────────────────────────────────────────────────

def _jitu_csr_fill_cuda_kernel(
    shape: MatrixShape,
    corder: bool,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    matrix_mode: MatrixMode = 'mv',
    **kwargs,
):
    load_cuda_file(Path(__file__).parent.joinpath('csr.cu'), name='jit_uniform_csr')
    sfx = _dtype_sfx.get(np.dtype(kwargs['w_low_info'].dtype), '_f32')
    infix = _mode_infix(matrix_mode)
    n_rows, n_cols = int(shape[0]), int(shape[1])
    chunk_size_value = _normalize_chunk_size(n_cols, chunk_size, target_chunks)

    if corder:
        kernel_name = f'jit_uniform_csr.fill_notrans{infix}{sfx}'

        def kernel(w_low, w_high, clen, seed, offsets):
            return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
                w_low, w_high, clen, seed, offsets,
                n_cols=np.int32(n_cols), chunk_size=np.int32(chunk_size_value),
            )
    else:
        kernel_name = f'jit_uniform_csr.fill_trans{infix}{sfx}'

        def kernel(w_low, w_high, clen, seed, offsets):
            return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
                w_low, w_high, clen, seed, offsets,
                n_rows=np.int32(n_cols), n_cols=np.int32(n_rows),
                chunk_size=np.int32(chunk_size_value),
            )

    return kernel


def _jitu_csr_fill_numba_kernel(
    shape: MatrixShape,
    corder: bool,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    matrix_mode: MatrixMode = 'mv',
    **kwargs,
):
    """Numba CPU fill pass mirroring ``csr.cu``."""
    import numba
    _rng = get_numba_light_rng_funcs()
    _rng_init = _rng['init']
    _rng_next = _rng['next']
    _rng_bounded = _rng['bounded']
    _rng_initial_q = _rng['initial_q']
    _rng_uniform01 = _rng['uniform01']

    stride = _MV_STRIDE if _normalize_matrix_mode(matrix_mode) == 'mv' else _MM_STRIDE
    n_cols = int(shape[1])
    cs_val = _normalize_chunk_size(n_cols, chunk_size, target_chunks)

    if corder:
        k = n_cols

        @numba.njit(fastmath=True)
        def kernel_impl(w_low, w_high, clen, seed, offsets, indices, data):
            m = offsets.shape[0]
            n_chunks = offsets.shape[1]
            w_low0 = w_low[0]
            w_high0 = w_high[0]
            span = w_high0 - w_low0
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            for row in range(m):
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * cs_val
                    if chunk_start >= k:
                        continue
                    chunk_end = chunk_start + cs_val
                    if chunk_end > k:
                        chunk_end = k
                    chunk_width = chunk_end - chunk_start
                    pos = offsets[row, chunk_id]
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            col = chunk_start + local_j
                            u01 = _rng_uniform01(seed0, row, col)
                            indices[pos] = col
                            data[pos] = w_low0 + u01 * span
                            pos += 1
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)
    else:
        m_walk = n_cols

        @numba.njit(fastmath=True)
        def kernel_impl(w_low, w_high, clen, seed, offsets, indices, data, cursor):
            cursor[:] = 0
            k = cursor.shape[0]
            w_low0 = w_low[0]
            w_high0 = w_high[0]
            span = w_high0 - w_low0
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            n_chunks = (k + cs_val - 1) // cs_val
            for row in range(m_walk):
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * cs_val
                    if chunk_start >= k:
                        break
                    chunk_end = chunk_start + cs_val
                    if chunk_end > k:
                        chunk_end = k
                    chunk_width = chunk_end - chunk_start
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            j = chunk_start + local_j
                            pos = offsets[j] + cursor[j]
                            cursor[j] += 1
                            u01 = _rng_uniform01(seed0, row, j)
                            indices[pos] = row
                            data[pos] = w_low0 + u01 * span
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)

    def kernel(w_low, w_high, clen, seed, offsets):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(w_low, w_high, clen, seed, offsets)

    return kernel


def jitu_csr_fill_p_call(
    w_low,
    w_high,
    clen,
    seed,
    offsets,
    nnz: int,
    *,
    shape: MatrixShape,
    corder: bool,
    matrix_mode: MatrixMode = 'mv',
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    backend: Optional[str] = None,
):
    """Fill pass. ``offsets`` are the per-(row, chunk) exclusive offsets
    (``corder=True``, shape ``(n_rows, n_chunks)``) or the ``indptr``
    (``corder=False``, shape ``(n_rows + 1,)``). Returns ``(indices, data)``."""
    n_rows, n_cols = int(shape[0]), int(shape[1])
    matrix_mode = _normalize_matrix_mode(matrix_mode)
    chunk_size_value = _normalize_chunk_size(n_cols, chunk_size, target_chunks)
    w_low = jnp.atleast_1d(w_low)
    w_high = jnp.atleast_1d(w_high)
    clen = jnp.atleast_1d(clen)
    seed = jnp.atleast_1d(seed)
    offsets = jnp.asarray(offsets, dtype=jnp.int32)
    nnz = int(nnz)
    if nnz == 0:
        return (jnp.zeros((0,), dtype=jnp.int32), jnp.zeros((0,), dtype=w_low.dtype))

    outs = [
        jax.ShapeDtypeStruct((nnz,), jnp.int32),
        jax.ShapeDtypeStruct((nnz,), w_low.dtype),
    ]
    if not corder:
        # trans fill also needs an int32 write cursor (one per output row).
        outs.append(jax.ShapeDtypeStruct((n_rows,), jnp.int32))

    res = jitu_csr_fill_p(
        w_low, w_high, clen, seed, offsets,
        outs=outs,
        shape=(n_rows, n_cols),
        corder=corder,
        matrix_mode=matrix_mode,
        chunk_size=chunk_size_value,
        target_chunks=target_chunks,
        backend=backend,
        w_low_info=jax.ShapeDtypeStruct(w_low.shape, w_low.dtype),
        w_high_info=jax.ShapeDtypeStruct(w_high.shape, w_high.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        offsets_info=jax.ShapeDtypeStruct(offsets.shape, offsets.dtype),
    )
    return res[0], res[1]


jitu_csr_fill_p = XLACustomKernel(
    'jitu_csr_fill',
    doc="Low-level CUDA primitive filling CSR indices/data for light uniform CSR.",
)
jitu_csr_fill_p.def_cuda_raw_kernel(_jitu_csr_fill_cuda_kernel, asdefault=True)
jitu_csr_fill_p.def_numba_kernel(_jitu_csr_fill_numba_kernel)
jitu_csr_fill_p.def_call(jitu_csr_fill_p_call)
jitu_csr_fill_p.def_tags('jit_uniform', 'csr', 'light_rng')


# ──────────────────────────────────────────────────────────────────────
#  High-level orchestration
# ──────────────────────────────────────────────────────────────────────

def jitu_to_csr(
    w_low,
    w_high,
    prob,
    seed,
    *,
    shape: MatrixShape,
    corder: bool,
    matrix_mode: MatrixMode,
    backend: Optional[str] = None,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
):
    """Materialize the light-RNG JIT uniform matrix as a :class:`~brainevent.CSR`.

    ``matrix_mode`` (``'mv'``/``'mm'``) is required: the mv and mm light kernels
    draw different matrices, so ``jitu_to_csr`` reproduces exactly the matrix used
    by ``jitu(matrix_mode=..., corder=...)`` (and thus by ``jitumv``/``jitumm``).
    """
    from brainevent._csr import CSR

    n_rows, n_cols = int(shape[0]), int(shape[1])
    matrix_mode = _normalize_matrix_mode(matrix_mode)
    chunk_size_value = _normalize_chunk_size(n_cols, chunk_size, target_chunks)

    u.fail_for_dimension_mismatch(w_low, w_high, "w_low and w_high must have the same dimension.")
    w_low, unitd = u.split_mantissa_unit(w_low)
    w_high = u.Quantity(w_high).to(unitd).mantissa
    dtype = jnp.result_type(w_low, w_high)
    w_low = jnp.atleast_1d(jnp.asarray(w_low, dtype=dtype))
    w_high = jnp.atleast_1d(jnp.asarray(w_high, dtype=dtype))
    seed = _initialize_seed(seed)

    if n_rows == 0 or n_cols == 0 or _is_static_zero(prob):
        indptr = jnp.zeros((n_rows + 1,), dtype=jnp.int32)
        indices = jnp.zeros((0,), dtype=jnp.int32)
        data = u.maybe_decimal(jnp.zeros((0,), dtype=w_low.dtype) * unitd)
        return CSR((data, indices, indptr), shape=(n_rows, n_cols))

    clen = _initialize_conn_length(prob)

    chunk_counts = jitu_csr_count_p_call(
        w_low, w_high, clen, seed,
        shape=(n_rows, n_cols), corder=corder, matrix_mode=matrix_mode,
        chunk_size=chunk_size_value, target_chunks=target_chunks, backend=backend,
    )[0]
    # notrans returns 2D per-(row, chunk) counts; trans returns 1D per-row counts.
    row_counts = chunk_counts.sum(axis=1, dtype=jnp.int32) if corder else chunk_counts
    row_counts_np = np.asarray(jax.device_get(row_counts), dtype=np.int64)
    nnz = int(row_counts_np.sum())

    # ``indptr`` offsets range up to ``nnz``; auto-promote to int64 when that
    # exceeds the int32 range (gated on ``jax_enable_x64``). ``indices`` (columns)
    # stay int32.
    offset_dtype = _resolve_indptr_dtype(nnz, "auto")
    _require_jax_x64_for_int64(offset_dtype, "jitu_to_csr indptr")
    indptr = jnp.concatenate(
        [jnp.zeros((1,), dtype=offset_dtype), jnp.cumsum(row_counts, dtype=offset_dtype)]
    )
    if nnz == 0:
        indices = jnp.zeros((0,), dtype=jnp.int32)
        data = u.maybe_decimal(jnp.zeros((0,), dtype=w_low.dtype) * unitd)
        return CSR((data, indices, indptr), shape=(n_rows, n_cols))

    if corder:
        # per-(row, chunk) exclusive offsets within each row's CSR slice.
        cc = chunk_counts.astype(offset_dtype)
        offsets = indptr[:-1, None] + jnp.cumsum(cc, axis=1, dtype=offset_dtype) - cc
    else:
        offsets = indptr
    # The light fill kernel has an int32-only offset ABI; refuse (rather than
    # silently truncate) when the nnz forced int64 offsets.
    offsets = _as_int32_cuda_offsets(offsets, "jitu_to_csr fill offsets")

    indices, data = jitu_csr_fill_p_call(
        w_low, w_high, clen, seed, offsets, nnz,
        shape=(n_rows, n_cols), corder=corder, matrix_mode=matrix_mode,
        chunk_size=chunk_size_value, target_chunks=target_chunks, backend=backend,
    )
    # The light fill emits per-row entries in lane order (notrans) or atomic order
    # (trans, non-deterministic); reorder to canonical column-sorted CSR so the
    # output is deterministic and matches the conventional CSR layout. Rows are
    # already contiguous, so a stable sort by (row, col) only reorders within rows.
    if nnz > 1:
        seg = np.repeat(np.arange(n_rows, dtype=np.int64), row_counts_np)
        order = jnp.asarray(np.lexsort((np.asarray(jax.device_get(indices)), seg)))
        indices = indices[order]
        data = data[order]
    data = u.maybe_decimal(data * unitd)
    return CSR((data, indices, indptr), shape=(n_rows, n_cols))
