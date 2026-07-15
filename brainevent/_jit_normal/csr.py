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

"""CUDA-only CSR materialization for the light RNG chunk-WPR JIT normal matrix.

This module materializes the same row-major logical matrix used by
``light_rng-chunk-wpr.cu``.  It is intentionally independent from
``csr.py``: the latter mirrors the exact dense ``jitn`` random walk, while this
module mirrors the light row/chunk/lane generator used by the fastest event
backend.
"""

from pathlib import Path
from typing import Literal, Optional
import warnings

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainevent._compatible_import import Tracer
from brainevent._data import _initialize_conn_length, _initialize_seed
from brainevent._op import XLACustomKernel, load_cuda_file
from brainevent._typing import MatrixShape

MatrixMode = Literal['mv', 'mm']

__all__ = [
    'jitn_to_csr',
    'jitn_csr_count_p',
    'jitn_csr_count_p_call',
    'jitn_csr_fill_p',
    'jitn_csr_fill_p_call',
]


def _is_static_zero(value) -> bool:
    if isinstance(value, Tracer):
        return False
    try:
        return float(np.asarray(value)) == 0.0
    except (TypeError, ValueError):
        return False


def _normalize_shape(shape: MatrixShape) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError(f"shape must be a pair of integers, got {shape!r}.")
    n_rows = int(shape[0])
    n_cols = int(shape[1])
    if n_rows < 0 or n_cols < 0:
        raise ValueError(f"shape dimensions must be non-negative, got {shape!r}.")
    return n_rows, n_cols


def _normalize_chunk_size(n_cols: int, chunk_size: Optional[int], target_chunks: int) -> int:
    if chunk_size is None:
        target_chunks = int(target_chunks)
        if target_chunks <= 0:
            raise ValueError("target_chunks must be positive")
        chunk_size = max(1, (int(n_cols) + target_chunks - 1) // target_chunks)
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return chunk_size


def _n_chunks(n_cols: int, chunk_size: int) -> int:
    return 0 if n_cols <= 0 else (n_cols + chunk_size - 1) // chunk_size


def _normalize_matrix_mode(matrix_mode: MatrixMode) -> MatrixMode:
    if matrix_mode not in ('mv', 'mm'):
        raise ValueError(f"matrix_mode must be 'mv' or 'mm', got {matrix_mode!r}.")
    return matrix_mode


def _warn_corder_ignored(corder: bool) -> None:
    warnings.warn(
        "corder is ignored by the light JIT normal implementation.",
        UserWarning,
        stacklevel=3,
    )


def _as_f32_weights(w_loc, w_scale):
    w_loc, unitd = u.split_mantissa_unit(w_loc)
    w_scale = u.Quantity(w_scale).to(unitd).mantissa
    common_dtype = jnp.result_type(w_loc, w_scale)
    if np.dtype(common_dtype) != np.dtype('float32'):
        raise NotImplementedError("light CSR currently supports float32 weights only")
    w_loc = jnp.atleast_1d(jnp.asarray(w_loc, dtype=jnp.float32))
    w_scale = jnp.atleast_1d(jnp.asarray(w_scale, dtype=jnp.float32))
    if w_loc.shape != (1,) or w_scale.shape != (1,):
        raise ValueError("w_loc and w_scale must be scalar values.")
    return w_loc, w_scale, unitd


def _jitn_csr_count_cuda_kernel(
    corder: bool,
    shape: MatrixShape,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    matrix_mode: MatrixMode = 'mv',
    **kwargs,
):
    del corder
    w0_dtype = np.dtype(kwargs['w0_info'].dtype)
    if w0_dtype != np.dtype('float32'):
        raise NotImplementedError("light CSR currently supports float32 weights only")

    n_rows, n_cols = _normalize_shape(shape)
    del n_rows
    matrix_mode = _normalize_matrix_mode(matrix_mode)
    chunk_size_value = _normalize_chunk_size(n_cols, chunk_size, target_chunks)
    load_cuda_file(
        Path(__file__).parent.joinpath('csr.cu'),
        name='jit_normal_csr',
    )
    n_cols_attr = np.int32(n_cols)
    chunk_size_attr = np.int32(chunk_size_value)
    kernel_name = (
        'jit_normal_csr.count_chunks_f32'
        if matrix_mode == 'mv'
        else 'jit_normal_csr.count_chunks_mm_aw_t4_f32'
    )

    def kernel(w0, w1, clen, seed):
        return jax.ffi.ffi_call(
            kernel_name,
            kwargs['outs'],
        )(
            w0,
            w1,
            clen,
            seed,
            n_cols=n_cols_attr,
            chunk_size=chunk_size_attr,
        )

    return kernel


def jitn_csr_count_p_call(
    w0,
    w1,
    clen,
    seed,
    *,
    shape: MatrixShape,
    corder: bool,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    matrix_mode: MatrixMode = 'mv',
    backend: Optional[str] = None,
):
    """Return per-``(row, chunk)`` non-zero counts for the light CSR matrix."""
    n_rows, n_cols = _normalize_shape(shape)
    matrix_mode = _normalize_matrix_mode(matrix_mode)
    _warn_corder_ignored(corder)
    chunk_size_value = _normalize_chunk_size(n_cols, chunk_size, target_chunks)
    n_chunks = _n_chunks(n_cols, chunk_size_value)

    w0 = jnp.atleast_1d(w0)
    w1 = jnp.atleast_1d(w1)
    clen = jnp.atleast_1d(clen)
    seed = jnp.atleast_1d(seed)
    if np.dtype(w0.dtype) != np.dtype('float32') or np.dtype(w1.dtype) != np.dtype('float32'):
        raise NotImplementedError("light CSR currently supports float32 weights only")
    if n_rows == 0 or n_chunks == 0:
        return (jnp.zeros((n_rows, n_chunks), dtype=jnp.int32),)

    return jitn_csr_count_p(
        w0,
        w1,
        clen,
        seed,
        outs=[jax.ShapeDtypeStruct((n_rows, n_chunks), jnp.int32)],
        shape=(n_rows, n_cols),
        chunk_size=chunk_size_value,
        target_chunks=target_chunks,
        matrix_mode=matrix_mode,
        corder=corder,
        backend=backend,
        w0_info=jax.ShapeDtypeStruct(w0.shape, w0.dtype),
        w1_info=jax.ShapeDtypeStruct(w1.shape, w1.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
    )


jitn_csr_count_p = XLACustomKernel(
    'jitn_csr_count',
    doc="""
Low-level CUDA primitive counting per-row/per-chunk non-zeros for light CSR.

The returned array has shape ``(n_rows, n_chunks)`` and follows exactly the
row/chunk/lane generator used by ``light_rng-chunk-wpr.cu``.
"""
)
jitn_csr_count_p.def_cuda_raw_kernel(_jitn_csr_count_cuda_kernel, asdefault=True)
jitn_csr_count_p.def_call(jitn_csr_count_p_call)
jitn_csr_count_p.def_tags('jit_normal', 'csr', 'light_rng')


def _jitn_csr_fill_cuda_kernel(
    corder: bool,
    shape: MatrixShape,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    matrix_mode: MatrixMode = 'mv',
    **kwargs,
):
    del corder
    w0_dtype = np.dtype(kwargs['w0_info'].dtype)
    if w0_dtype != np.dtype('float32'):
        raise NotImplementedError("light CSR currently supports float32 weights only")

    n_rows, n_cols = _normalize_shape(shape)
    del n_rows
    matrix_mode = _normalize_matrix_mode(matrix_mode)
    chunk_size_value = _normalize_chunk_size(n_cols, chunk_size, target_chunks)
    load_cuda_file(
        Path(__file__).parent.joinpath('csr.cu'),
        name='jit_normal_csr',
    )
    n_cols_attr = np.int32(n_cols)
    chunk_size_attr = np.int32(chunk_size_value)
    kernel_name = (
        'jit_normal_csr.fill_f32'
        if matrix_mode == 'mv'
        else 'jit_normal_csr.fill_mm_aw_t4_f32'
    )

    def kernel(w0, w1, clen, seed, chunk_offsets):
        return jax.ffi.ffi_call(
            kernel_name,
            kwargs['outs'],
        )(
            w0,
            w1,
            clen,
            seed,
            chunk_offsets,
            n_cols=n_cols_attr,
            chunk_size=chunk_size_attr,
        )

    return kernel


def jitn_csr_fill_p_call(
    w0,
    w1,
    clen,
    seed,
    chunk_offsets,
    nnz: int,
    *,
    shape: MatrixShape,
    corder: bool,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    matrix_mode: MatrixMode = 'mv',
    backend: Optional[str] = None,
):
    """Fill light CSR ``indices`` and ``data`` using precomputed chunk offsets."""
    n_rows, n_cols = _normalize_shape(shape)
    matrix_mode = _normalize_matrix_mode(matrix_mode)
    _warn_corder_ignored(corder)
    chunk_size_value = _normalize_chunk_size(n_cols, chunk_size, target_chunks)
    n_chunks = _n_chunks(n_cols, chunk_size_value)

    w0 = jnp.atleast_1d(w0)
    w1 = jnp.atleast_1d(w1)
    clen = jnp.atleast_1d(clen)
    seed = jnp.atleast_1d(seed)
    chunk_offsets = jnp.asarray(chunk_offsets, dtype=jnp.int32)
    nnz = int(nnz)
    if np.dtype(w0.dtype) != np.dtype('float32') or np.dtype(w1.dtype) != np.dtype('float32'):
        raise NotImplementedError("light CSR currently supports float32 weights only")
    if chunk_offsets.shape != (n_rows, n_chunks):
        raise ValueError(
            f"chunk_offsets must have shape {(n_rows, n_chunks)}, got {chunk_offsets.shape}."
        )
    if nnz < 0:
        raise ValueError("nnz must be non-negative")
    if nnz == 0:
        return (
            jnp.zeros((0,), dtype=jnp.int32),
            jnp.zeros((0,), dtype=w0.dtype),
        )

    return jitn_csr_fill_p(
        w0,
        w1,
        clen,
        seed,
        chunk_offsets,
        outs=[
            jax.ShapeDtypeStruct((nnz,), jnp.int32),
            jax.ShapeDtypeStruct((nnz,), w0.dtype),
        ],
        shape=(n_rows, n_cols),
        chunk_size=chunk_size_value,
        target_chunks=target_chunks,
        matrix_mode=matrix_mode,
        corder=corder,
        backend=backend,
        w0_info=jax.ShapeDtypeStruct(w0.shape, w0.dtype),
        w1_info=jax.ShapeDtypeStruct(w1.shape, w1.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        chunk_offsets_info=jax.ShapeDtypeStruct(chunk_offsets.shape, chunk_offsets.dtype),
    )


jitn_csr_fill_p = XLACustomKernel(
    'jitn_csr_fill',
    doc="""
Low-level CUDA primitive filling CSR ``indices``/``data`` for light CSR.

``chunk_offsets`` must be the exclusive per-row/per-chunk prefix derived from
``jitn_csr_count_p``.
"""
)
jitn_csr_fill_p.def_cuda_raw_kernel(_jitn_csr_fill_cuda_kernel, asdefault=True)
jitn_csr_fill_p.def_call(jitn_csr_fill_p_call)
jitn_csr_fill_p.def_tags('jit_normal', 'csr', 'light_rng')


def jitn_to_csr(
    w_loc,
    w_scale,
    prob,
    seed,
    *,
    shape: MatrixShape,
    corder: bool,
    backend: Optional[str] = None,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    matrix_mode: MatrixMode = 'mv',
):
    """Materialize the light RNG chunk-WPR JIT normal matrix as ``CSR``.

    The returned matrix is the row-major logical matrix ``A`` used by
    ``light_rng-chunk-wpr`` (``matrix_mode='mv'``) or its AW-T4 matrix-matrix
    variant (``matrix_mode='mm'``).  It is eager-only because the data-dependent
    ``nnz`` is read back between the count and fill passes.
    """
    from brainevent._csr import CSR

    n_rows, n_cols = _normalize_shape(shape)
    matrix_mode = _normalize_matrix_mode(matrix_mode)
    _warn_corder_ignored(corder)
    chunk_size_value = _normalize_chunk_size(n_cols, chunk_size, target_chunks)
    n_chunks = _n_chunks(n_cols, chunk_size_value)
    w_loc, w_scale, unitd = _as_f32_weights(w_loc, w_scale)
    seed = _initialize_seed(seed)

    if n_rows == 0 or n_cols == 0 or _is_static_zero(prob):
        indptr = jnp.zeros((n_rows + 1,), dtype=jnp.int32)
        indices = jnp.zeros((0,), dtype=jnp.int32)
        data = u.maybe_decimal(jnp.zeros((0,), dtype=w_loc.dtype) * unitd)
        return CSR((data, indices, indptr), shape=(n_rows, n_cols))

    clen = _initialize_conn_length(prob)
    chunk_counts = jitn_csr_count_p_call(
        w_loc,
        w_scale,
        clen,
        seed,
        shape=(n_rows, n_cols),
        corder=corder,
        chunk_size=chunk_size_value,
        target_chunks=target_chunks,
        matrix_mode=matrix_mode,
        backend=backend,
    )[0]
    row_counts = chunk_counts.sum(axis=1, dtype=jnp.int32)
    indptr = jnp.concatenate(
        [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(row_counts, dtype=jnp.int32)]
    )
    nnz = int(indptr[-1])
    if nnz == 0:
        indices = jnp.zeros((0,), dtype=jnp.int32)
        data = u.maybe_decimal(jnp.zeros((0,), dtype=w_loc.dtype) * unitd)
        return CSR((data, indices, indptr), shape=(n_rows, n_cols))

    chunk_offsets = (
        indptr[:-1, None]
        + jnp.cumsum(chunk_counts, axis=1, dtype=jnp.int32)
        - chunk_counts
    )
    indices, data = jitn_csr_fill_p_call(
        w_loc,
        w_scale,
        clen,
        seed,
        chunk_offsets,
        nnz,
        shape=(n_rows, n_cols),
        corder=corder,
        chunk_size=chunk_size_value,
        target_chunks=target_chunks,
        matrix_mode=matrix_mode,
        backend=backend,
    )
    data = u.maybe_decimal(data * unitd)
    return CSR((data, indices, indptr), shape=(n_rows, n_cols))
