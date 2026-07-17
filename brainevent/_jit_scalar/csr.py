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

"""CUDA-only CSR materialization for the light RNG chunk-WPR JIT scalar matrix."""

from pathlib import Path
from typing import Literal, Optional
import warnings

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainevent._compatible_import import Tracer
from brainevent._data import _initialize_conn_length, _initialize_seed
from brainevent._misc import _as_int32_cuda_offsets, _require_jax_x64_for_int64, _resolve_indptr_dtype
from brainevent._op import XLACustomKernel, load_cuda_file
from brainevent._typing import MatrixShape

MatrixMode = Literal['mv', 'mm']

__all__ = [
    'jits_to_csr',
    'jits_csr_count_p',
    'jits_csr_count_p_call',
    'jits_csr_fill_p',
    'jits_csr_fill_p_call',
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


def _warn_corder_deprecated(corder: Optional[bool]) -> None:
    if corder is None:
        return
    warnings.warn(
        "corder is deprecated and ignored by the light JIT scalar implementation.",
        FutureWarning,
        stacklevel=3,
    )


_warn_corder_ignored = _warn_corder_deprecated


def _as_f32_weight(weight):
    weight, unitd = u.split_mantissa_unit(weight)
    common_dtype = jnp.result_type(weight)
    if np.dtype(common_dtype) != np.dtype('float32'):
        raise NotImplementedError("light CSR currently supports float32 weights only")
    weight = jnp.atleast_1d(jnp.asarray(weight, dtype=jnp.float32))
    if weight.shape != (1,):
        raise ValueError("weight must be a scalar value.")
    return weight, unitd


def _jits_csr_count_cuda_kernel(
    shape: MatrixShape,
    transpose: bool = False,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    matrix_mode: MatrixMode = 'mv',
    **kwargs,
):
    weight_dtype = np.dtype(kwargs['weight_info'].dtype)
    if weight_dtype != np.dtype('float32'):
        raise NotImplementedError("light CSR currently supports float32 weights only")

    n_rows, n_cols = _normalize_shape(shape)
    matrix_mode = _normalize_matrix_mode(matrix_mode)
    chunk_size_value = _normalize_chunk_size(n_cols, chunk_size, target_chunks)
    load_cuda_file(
        Path(__file__).parent.joinpath('csr.cu'),
        name='jit_scalar_csr',
    )
    n_cols_attr = np.int32(n_cols)
    chunk_size_attr = np.int32(chunk_size_value)
    if transpose:
        kernel_name = (
            'jit_scalar_csr.count_chunks_trans_f32'
            if matrix_mode == 'mv'
            else 'jit_scalar_csr.count_chunks_trans_mm_aw_t4_f32'
        )

        def kernel(weight, clen, seed):
            return jax.ffi.ffi_call(
                kernel_name,
                kwargs['outs'],
            )(
                weight,
                clen,
                seed,
                n_rows=np.int32(n_rows),
                n_cols=n_cols_attr,
                chunk_size=chunk_size_attr,
            )
    else:
        kernel_name = (
            'jit_scalar_csr.count_chunks_notrans_f32'
            if matrix_mode == 'mv'
            else 'jit_scalar_csr.count_chunks_notrans_mm_aw_t4_f32'
        )

        def kernel(weight, clen, seed):
            return jax.ffi.ffi_call(
                kernel_name,
                kwargs['outs'],
            )(
                weight,
                clen,
                seed,
                n_cols=n_cols_attr,
                chunk_size=chunk_size_attr,
            )

    return kernel


def jits_csr_count_p_call(
    weight,
    clen,
    seed,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    matrix_mode: MatrixMode = 'mv',
    backend: Optional[str] = None,
):
    """Return per-output-row non-zero counts for the light CSR matrix."""
    n_rows, n_cols = _normalize_shape(shape)
    matrix_mode = _normalize_matrix_mode(matrix_mode)
    chunk_size_value = _normalize_chunk_size(n_cols, chunk_size, target_chunks)
    n_chunks = _n_chunks(n_cols, chunk_size_value)

    weight = jnp.atleast_1d(weight)
    clen = jnp.atleast_1d(clen)
    seed = jnp.atleast_1d(seed)
    if np.dtype(weight.dtype) != np.dtype('float32'):
        raise NotImplementedError("light CSR currently supports float32 weights only")
    if transpose:
        if n_rows == 0 or n_cols == 0:
            return (jnp.zeros((n_cols,), dtype=jnp.int32),)
        out_info = jax.ShapeDtypeStruct((n_cols,), jnp.int32)
    else:
        if n_rows == 0 or n_chunks == 0:
            return (jnp.zeros((n_rows, n_chunks), dtype=jnp.int32),)
        out_info = jax.ShapeDtypeStruct((n_rows, n_chunks), jnp.int32)

    return jits_csr_count_p(
        weight,
        clen,
        seed,
        outs=[out_info],
        shape=(n_rows, n_cols),
        transpose=transpose,
        chunk_size=chunk_size_value,
        target_chunks=target_chunks,
        matrix_mode=matrix_mode,
        backend=backend,
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
    )


jits_csr_count_p = XLACustomKernel(
    'jits_csr_count',
    doc="""
Low-level CUDA primitive counting non-zeros for light scalar CSR materialization.
"""
)
jits_csr_count_p.def_cuda_raw_kernel(_jits_csr_count_cuda_kernel, asdefault=True)
jits_csr_count_p.def_call(jits_csr_count_p_call)
jits_csr_count_p.def_tags('jit_scalar', 'csr', 'light_rng')


def _jits_csr_fill_cuda_kernel(
    shape: MatrixShape,
    transpose: bool = False,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    matrix_mode: MatrixMode = 'mv',
    **kwargs,
):
    weight_dtype = np.dtype(kwargs['weight_info'].dtype)
    if weight_dtype != np.dtype('float32'):
        raise NotImplementedError("light CSR currently supports float32 weights only")

    n_rows, n_cols = _normalize_shape(shape)
    matrix_mode = _normalize_matrix_mode(matrix_mode)
    chunk_size_value = _normalize_chunk_size(n_cols, chunk_size, target_chunks)
    load_cuda_file(
        Path(__file__).parent.joinpath('csr.cu'),
        name='jit_scalar_csr',
    )
    n_cols_attr = np.int32(n_cols)
    chunk_size_attr = np.int32(chunk_size_value)
    if transpose:
        kernel_name = (
            'jit_scalar_csr.fill_trans_f32'
            if matrix_mode == 'mv'
            else 'jit_scalar_csr.fill_trans_mm_aw_t4_f32'
        )

        def kernel(weight, clen, seed, chunk_offsets):
            return jax.ffi.ffi_call(
                kernel_name,
                kwargs['outs'],
            )(
                weight,
                clen,
                seed,
                chunk_offsets,
                n_rows=np.int32(n_rows),
                n_cols=n_cols_attr,
                chunk_size=chunk_size_attr,
            )
    else:
        kernel_name = (
            'jit_scalar_csr.fill_notrans_f32'
            if matrix_mode == 'mv'
            else 'jit_scalar_csr.fill_notrans_mm_aw_t4_f32'
        )

        def kernel(weight, clen, seed, chunk_offsets):
            return jax.ffi.ffi_call(
                kernel_name,
                kwargs['outs'],
            )(
                weight,
                clen,
                seed,
                chunk_offsets,
                n_cols=n_cols_attr,
                chunk_size=chunk_size_attr,
            )

    return kernel


def jits_csr_fill_p_call(
    weight,
    clen,
    seed,
    chunk_offsets,
    nnz: int,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    matrix_mode: MatrixMode = 'mv',
    backend: Optional[str] = None,
):
    """Fill light scalar CSR ``indices`` and ``data`` using precomputed offsets."""
    n_rows, n_cols = _normalize_shape(shape)
    matrix_mode = _normalize_matrix_mode(matrix_mode)
    chunk_size_value = _normalize_chunk_size(n_cols, chunk_size, target_chunks)
    n_chunks = _n_chunks(n_cols, chunk_size_value)

    weight = jnp.atleast_1d(weight)
    clen = jnp.atleast_1d(clen)
    seed = jnp.atleast_1d(seed)
    chunk_offsets = _as_int32_cuda_offsets(chunk_offsets, "jits_csr_fill_p_call chunk_offsets")
    nnz = int(nnz)
    if np.dtype(weight.dtype) != np.dtype('float32'):
        raise NotImplementedError("light CSR currently supports float32 weights only")
    expected_offsets_shape = (n_cols + 1,) if transpose else (n_rows, n_chunks)
    if chunk_offsets.shape != expected_offsets_shape:
        raise ValueError(
            f"chunk_offsets must have shape {expected_offsets_shape}, got {chunk_offsets.shape}."
        )
    if nnz < 0:
        raise ValueError("nnz must be non-negative")
    if nnz == 0:
        return (
            jnp.zeros((0,), dtype=jnp.int32),
            jnp.zeros((0,), dtype=weight.dtype),
        )

    outs = [
        jax.ShapeDtypeStruct((nnz,), jnp.int32),
        jax.ShapeDtypeStruct((nnz,), weight.dtype),
    ]
    if transpose:
        outs.append(jax.ShapeDtypeStruct((n_cols,), jnp.int32))

    res = jits_csr_fill_p(
        weight,
        clen,
        seed,
        chunk_offsets,
        outs=outs,
        shape=(n_rows, n_cols),
        transpose=transpose,
        chunk_size=chunk_size_value,
        target_chunks=target_chunks,
        matrix_mode=matrix_mode,
        backend=backend,
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        chunk_offsets_info=jax.ShapeDtypeStruct(chunk_offsets.shape, chunk_offsets.dtype),
    )
    return res[:2]


jits_csr_fill_p = XLACustomKernel(
    'jits_csr_fill',
    doc="""
Low-level CUDA primitive filling CSR ``indices``/``data`` for light scalar CSR.
"""
)
jits_csr_fill_p.def_cuda_raw_kernel(_jits_csr_fill_cuda_kernel, asdefault=True)
jits_csr_fill_p.def_call(jits_csr_fill_p_call)
jits_csr_fill_p.def_tags('jit_scalar', 'csr', 'light_rng')


def jits_to_csr(
    weight,
    prob,
    seed,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: Optional[bool] = None,
    backend: Optional[str] = None,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    matrix_mode: MatrixMode = 'mv',
):
    """Materialize the light RNG chunk-WPR JIT scalar matrix as ``CSR``."""
    from brainevent._csr import CSR

    n_rows, n_cols = _normalize_shape(shape)
    matrix_mode = _normalize_matrix_mode(matrix_mode)
    _warn_corder_deprecated(corder)
    chunk_size_value = _normalize_chunk_size(n_cols, chunk_size, target_chunks)
    n_chunks = _n_chunks(n_cols, chunk_size_value)
    weight, unitd = _as_f32_weight(weight)
    seed = _initialize_seed(seed)

    out_rows, out_cols = (n_cols, n_rows) if transpose else (n_rows, n_cols)
    if n_rows == 0 or n_cols == 0 or _is_static_zero(prob):
        indptr = jnp.zeros((out_rows + 1,), dtype=jnp.int32)
        indices = jnp.zeros((0,), dtype=jnp.int32)
        data = u.maybe_decimal(jnp.zeros((0,), dtype=weight.dtype) * unitd)
        return CSR((data, indices, indptr), shape=(out_rows, out_cols))

    clen = _initialize_conn_length(prob)
    chunk_counts = jits_csr_count_p_call(
        weight,
        clen,
        seed,
        shape=(n_rows, n_cols),
        transpose=transpose,
        chunk_size=chunk_size_value,
        target_chunks=target_chunks,
        matrix_mode=matrix_mode,
        backend=backend,
    )[0]
    row_counts = chunk_counts if transpose else chunk_counts.sum(axis=1, dtype=jnp.int32)
    nnz = int(np.asarray(jax.device_get(row_counts), dtype=np.int64).sum())
    offset_dtype = _resolve_indptr_dtype(nnz, requested="auto")
    _require_jax_x64_for_int64(offset_dtype, "jits_to_csr indptr")
    offset_jdtype = jnp.dtype(offset_dtype)
    row_counts = row_counts.astype(offset_jdtype)
    indptr = jnp.concatenate(
        [jnp.zeros((1,), dtype=offset_jdtype), jnp.cumsum(row_counts, dtype=offset_jdtype)]
    )
    if nnz == 0:
        indices = jnp.zeros((0,), dtype=jnp.int32)
        data = u.maybe_decimal(jnp.zeros((0,), dtype=weight.dtype) * unitd)
        return CSR((data, indices, indptr), shape=(out_rows, out_cols))

    if transpose:
        chunk_offsets = indptr
    else:
        chunk_counts_offsets = chunk_counts.astype(offset_jdtype)
        chunk_offsets = (
            indptr[:-1, None]
            + jnp.cumsum(chunk_counts_offsets, axis=1, dtype=offset_jdtype)
            - chunk_counts_offsets
        )
    indices, data = jits_csr_fill_p_call(
        weight,
        clen,
        seed,
        chunk_offsets,
        nnz,
        shape=(n_rows, n_cols),
        transpose=transpose,
        chunk_size=chunk_size_value,
        target_chunks=target_chunks,
        matrix_mode=matrix_mode,
        backend=backend,
    )
    data = u.maybe_decimal(data * unitd)
    return CSR((data, indices, indptr), shape=(out_rows, out_cols))
