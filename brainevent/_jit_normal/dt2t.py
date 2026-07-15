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
Direct per-synapse ``y * w`` generation for normal-weight just-in-time
connectivity (JITC) matrices.

The public :func:`jitnmv_dt2t` wrapper mirrors the CSR ``dt2t`` contract:
it returns one value per generated structural non-zero, in the same flat CSR
data order as ``jitn_to_csr(..., matrix_mode="mv")``. Unlike a wrapper around
``tocsr().dt2t(...)``, the fill pass draws each normal weight and multiplies by
``y[row]`` or ``y[col]`` directly.

Because the number of structural non-zeros is data dependent, generation is
eager-only and split into:

1. the light JIT-normal CSR count pass, which determines row/chunk offsets; and
2. a dedicated CUDA ``dt2t`` fill pass, which writes ``sampled_weight * y[...]``.
"""

from pathlib import Path
from typing import Optional

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainevent._data import _initialize_conn_length, _initialize_seed
from brainevent._jit_normal.csr import (
    _is_static_zero,
    _n_chunks,
    _normalize_chunk_size,
    _normalize_shape,
    _warn_corder_ignored,
    jitn_csr_count_p_call,
)
from brainevent._op import XLACustomKernel, load_cuda_file
from brainevent._typing import MatrixShape

__all__ = [
    'jitnmv_dt2t',
    'jitnmv_dt2t_p',
    'jitnmv_dt2t_p_call',
]

def jitnmv_dt2t(
    w_loc,
    w_scale,
    prob,
    y,
    seed,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
    backend: Optional[str] = None,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
):
    """Generate per-synapse ``y * w`` values for a normal JITC matrix.

    The result is a flat vector of length ``nnz`` in the same order as
    ``jitn_to_csr(..., matrix_mode="mv").data``. The output equals
    ``csr.dt2t(y, csr.data)`` when ``transpose=False`` and
    ``csr.dt2t_transposed(y, csr.data)`` when ``transpose=True``, without first
    materialising the CSR weight data.
    """
    n_rows, n_cols = _normalize_shape(shape)
    chunk_size_value = _normalize_chunk_size(n_cols, chunk_size, target_chunks)
    n_chunks = _n_chunks(n_cols, chunk_size_value)

    w_loc, unitd = u.split_mantissa_unit(w_loc)
    w_scale = u.Quantity(w_scale).to(unitd).mantissa
    y, unity = u.split_mantissa_unit(y)

    common_dtype = jnp.result_type(w_loc, w_scale, y)
    if np.dtype(common_dtype) != np.dtype('float32'):
        raise NotImplementedError("light dt2t currently supports float32 values only")
    w_loc = jnp.atleast_1d(jnp.asarray(w_loc, dtype=common_dtype))
    w_scale = jnp.atleast_1d(jnp.asarray(w_scale, dtype=common_dtype))
    y = jnp.asarray(y, dtype=common_dtype)

    if y.ndim != 1:
        raise AssertionError("y must be 1D.")
    if transpose:
        assert n_cols == y.shape[0], "Shape mismatch for transpose operation."
    else:
        assert n_rows == y.shape[0], "Shape mismatch for non-transpose operation."

    if n_rows == 0 or n_cols == 0 or _is_static_zero(prob):
        data = jnp.zeros(0, dtype=common_dtype)
        return u.maybe_decimal(data * unitd * unity)

    clen = _initialize_conn_length(prob)
    seed = _initialize_seed(seed)
    chunk_counts = jitn_csr_count_p_call(
        w_loc,
        w_scale,
        clen,
        seed,
        shape=(n_rows, n_cols),
        corder=corder,
        chunk_size=chunk_size_value,
        target_chunks=target_chunks,
        matrix_mode='mv',
        backend=backend,
    )[0]
    row_counts = chunk_counts.sum(axis=1, dtype=jnp.int32)
    indptr = jnp.concatenate(
        [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(row_counts, dtype=jnp.int32)]
    )
    nnz = int(indptr[-1])
    if nnz == 0:
        data = jnp.zeros(0, dtype=common_dtype)
        return u.maybe_decimal(data * unitd * unity)

    chunk_offsets = (
        indptr[:-1, None]
        + jnp.cumsum(chunk_counts, axis=1, dtype=jnp.int32)
        - chunk_counts
    )

    data = jitnmv_dt2t_p_call(
        w_loc,
        w_scale,
        clen,
        y,
        seed,
        chunk_offsets,
        nnz,
        shape=(n_rows, n_cols),
        transpose=transpose,
        corder=corder,
        chunk_size=chunk_size_value,
        target_chunks=target_chunks,
        backend=backend,
    )[0]
    return u.maybe_decimal(data * unitd * unity)


# ---------------------------------------------------------------------- #
#  Count pass - per-row/per-chunk non-zero counts
# ---------------------------------------------------------------------- #
#
# The dt2t path deliberately reuses the light JIT-normal CSR count pass instead
# of duplicating it here. This keeps the data-dependent ``nnz``/chunk-offset
# logic aligned with ``jitn_to_csr(..., matrix_mode="mv")``; the dedicated dt2t
# work starts at the fill pass.


# ---------------------------------------------------------------------- #
#  Fill pass - per-synapse y * w values
# ---------------------------------------------------------------------- #

def _jitnmv_dt2t_fill_cuda_kernel(
    corder: bool,
    shape: MatrixShape,
    transpose: bool,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    **kwargs,
):
    """Build the CUDA kernel callable for the normal JITC ``dt2t`` fill pass."""
    del corder
    w0_dtype = np.dtype(kwargs['w0_info'].dtype)
    if w0_dtype != np.dtype('float32'):
        raise NotImplementedError("light dt2t currently supports float32 values only")

    _, n_cols = _normalize_shape(shape)
    chunk_size_value = _normalize_chunk_size(n_cols, chunk_size, target_chunks)
    load_cuda_file(
        Path(__file__).parent.joinpath('dt2t.cu'),
        name='jit_normal_dt2t',
    )
    n_cols = np.int32(shape[1])
    chunk_size_attr = np.int32(chunk_size_value)
    kernel_name = (
        'jit_normal_dt2t.fill_transpose_f32'
        if transpose
        else 'jit_normal_dt2t.fill_f32'
    )

    def kernel(w0, w1, clen, y, seed, chunk_offsets):
        return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
            w0,
            w1,
            clen,
            y,
            seed,
            chunk_offsets,
            n_cols=n_cols,
            chunk_size=chunk_size_attr,
        )

    return kernel


def jitnmv_dt2t_p_call(
    w0,
    w1,
    clen,
    y,
    seed,
    chunk_offsets,
    nnz: int,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    backend: Optional[str] = None,
):
    """Invoke the normal JITC ``dt2t`` fill primitive."""
    n_rows, n_cols = _normalize_shape(shape)
    chunk_size_value = _normalize_chunk_size(n_cols, chunk_size, target_chunks)
    n_chunks = _n_chunks(n_cols, chunk_size_value)
    _warn_corder_ignored(corder)

    w0 = jnp.atleast_1d(w0)
    w1 = jnp.atleast_1d(w1)
    clen = jnp.atleast_1d(clen)
    y = jnp.asarray(y)
    seed = jnp.atleast_1d(seed)
    chunk_offsets = jnp.asarray(chunk_offsets, dtype=jnp.int32)
    assert w0.ndim == w1.ndim == clen.ndim == seed.ndim == 1
    assert w0.size == w1.size == clen.size == seed.size == 1
    assert y.ndim == 1, "y must be 1D."
    assert chunk_offsets.ndim == 2, "chunk_offsets must be 2D."
    assert chunk_offsets.shape == (n_rows, n_chunks), (
        f"chunk_offsets shape mismatch, expected {(n_rows, n_chunks)}, "
        f"got {chunk_offsets.shape}."
    )
    assert jnp.issubdtype(chunk_offsets.dtype, jnp.integer), "chunk_offsets must be an integer type."
    assert jnp.issubdtype(w0.dtype, jnp.floating), "w0 must be a floating-point type."
    assert jnp.issubdtype(w1.dtype, jnp.floating), "w1 must be a floating-point type."
    assert jnp.issubdtype(y.dtype, jnp.floating), "y must be a floating-point type."
    assert w0.dtype == w1.dtype == y.dtype, (
        f"w0, w1 and y must have the same dtype, got {w0.dtype}, {w1.dtype}, {y.dtype}."
    )
    if np.dtype(w0.dtype) != np.dtype('float32'):
        raise NotImplementedError("light dt2t currently supports float32 values only")
    if transpose:
        assert n_cols == y.shape[0], "Shape mismatch for transpose operation."
    else:
        assert n_rows == y.shape[0], "Shape mismatch for non-transpose operation."

    nnz = int(nnz)
    if nnz < 0:
        raise ValueError("nnz must be non-negative")
    if nnz == 0:
        return (jnp.zeros((0,), dtype=y.dtype),)

    return jitnmv_dt2t_p(
        w0,
        w1,
        clen,
        y,
        seed,
        chunk_offsets,
        outs=[jax.ShapeDtypeStruct((nnz,), y.dtype)],
        shape=(n_rows, n_cols),
        transpose=transpose,
        corder=corder,
        chunk_size=chunk_size_value,
        target_chunks=target_chunks,
        backend=backend,
        w0_info=jax.ShapeDtypeStruct(w0.shape, w0.dtype),
        w1_info=jax.ShapeDtypeStruct(w1.shape, w1.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        y_info=jax.ShapeDtypeStruct(y.shape, y.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        chunk_offsets_info=jax.ShapeDtypeStruct(chunk_offsets.shape, chunk_offsets.dtype),
    )


jitnmv_dt2t_p = XLACustomKernel(
    'jitnmv_dt2t_fill',
    doc="""
Low-level XLA custom-kernel primitive filling per-synapse ``y * w`` values for a
normal JITC matrix.

Given the ``chunk_offsets`` produced from the light JIT-normal CSR count pass,
this primitive walks the same deterministic random stream as
``jitn_to_csr(..., matrix_mode="mv")``. For every generated connection it writes
``weight * y[row]`` (``transpose=False``) or ``weight * y[col]``
(``transpose=True``), preserving the same flat CSR data order.
"""
)
jitnmv_dt2t_p.def_cuda_raw_kernel(_jitnmv_dt2t_fill_cuda_kernel)
jitnmv_dt2t_p.def_call(jitnmv_dt2t_p_call)
jitnmv_dt2t_p.def_tags('jit_normal', 'dt2t')
