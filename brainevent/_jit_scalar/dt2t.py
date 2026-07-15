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
Direct per-synapse ``y * w`` generation for scalar-weight just-in-time
connectivity (JITC) matrices.

The public :func:`jitsmv_dt2t` wrapper mirrors the CSR ``dt2t`` contract:
it returns one value per generated structural non-zero, in the same flat CSR
data order as ``jits_to_csr(..., matrix_mode="mv")``. Unlike a wrapper around
``tocsr().dt2t(...)``, the fill pass draws each scalar weight and multiplies by
``y[row]`` or ``y[col]`` directly.

Because the number of structural non-zeros is data dependent, generation is
eager-only and split into:

1. the light JIT-scalar CSR count pass, which determines row/chunk offsets; and
2. a dedicated CUDA ``dt2t`` fill pass, which writes ``sampled_weight * y[...]``.
"""

from pathlib import Path
from typing import Optional

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainevent._data import _initialize_conn_length, _initialize_seed
from brainevent._jit_scalar.csr import (
    _is_static_zero,
    _n_chunks,
    _normalize_chunk_size,
    _normalize_shape,
    _warn_corder_ignored,
    jits_csr_count_p_call,
)
from brainevent._op import XLACustomKernel, load_cuda_file
from brainevent._typing import MatrixShape

__all__ = [
    'jitsmv_dt2t',
    'jitsmv_dt2t_p',
    'jitsmv_dt2t_p_call',
]

def jitsmv_dt2t(
    weight,
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
    """Generate per-synapse ``y * w`` values for a scalar JITC matrix.

    The result is a flat vector of length ``nnz`` in the same order as
    ``jits_to_csr(..., matrix_mode="mv").data``. The output equals
    ``csr.dt2t(y, csr.data)`` when ``transpose=False`` and
    ``csr.dt2t_transposed(y, csr.data)`` when ``transpose=True``, without first
    materialising the CSR weight data.
    """
    n_rows, n_cols = _normalize_shape(shape)
    chunk_size_value = _normalize_chunk_size(n_cols, chunk_size, target_chunks)
    n_chunks = _n_chunks(n_cols, chunk_size_value)

    weight, unitd = u.split_mantissa_unit(weight)
    y, unity = u.split_mantissa_unit(y)

    common_dtype = jnp.result_type(weight, y)
    if np.dtype(common_dtype) != np.dtype('float32'):
        raise NotImplementedError("light dt2t currently supports float32 values only")
    weight = jnp.atleast_1d(jnp.asarray(weight, dtype=common_dtype))
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
    chunk_counts = jits_csr_count_p_call(
        weight,
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

    data = jitsmv_dt2t_p_call(
        weight,
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
# The dt2t path deliberately reuses the light JIT-scalar CSR count pass instead
# of duplicating it here. This keeps the data-dependent ``nnz``/chunk-offset
# logic aligned with ``jits_to_csr(..., matrix_mode="mv")``; the dedicated dt2t
# work starts at the fill pass.


# ---------------------------------------------------------------------- #
#  Fill pass - per-synapse y * w values
# ---------------------------------------------------------------------- #

def _jitsmv_dt2t_fill_cuda_kernel(
    corder: bool,
    shape: MatrixShape,
    transpose: bool,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    **kwargs,
):
    """Build the CUDA kernel callable for the scalar JITC ``dt2t`` fill pass."""
    del corder
    weight_dtype = np.dtype(kwargs['weight_info'].dtype)
    if weight_dtype != np.dtype('float32'):
        raise NotImplementedError("light dt2t currently supports float32 values only")

    _, n_cols = _normalize_shape(shape)
    chunk_size_value = _normalize_chunk_size(n_cols, chunk_size, target_chunks)
    load_cuda_file(
        Path(__file__).parent.joinpath('dt2t.cu'),
        name='jit_scalar_dt2t',
    )
    n_cols = np.int32(shape[1])
    chunk_size_attr = np.int32(chunk_size_value)
    kernel_name = (
        'jit_scalar_dt2t.fill_transpose_f32'
        if transpose
        else 'jit_scalar_dt2t.fill_f32'
    )

    def kernel(weight, clen, y, seed, chunk_offsets):
        return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
            weight,
            clen,
            y,
            seed,
            chunk_offsets,
            n_cols=n_cols,
            chunk_size=chunk_size_attr,
        )

    return kernel


def jitsmv_dt2t_p_call(
    weight,
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
    """Invoke the scalar JITC ``dt2t`` fill primitive."""
    n_rows, n_cols = _normalize_shape(shape)
    chunk_size_value = _normalize_chunk_size(n_cols, chunk_size, target_chunks)
    n_chunks = _n_chunks(n_cols, chunk_size_value)
    _warn_corder_ignored(corder)

    weight = jnp.atleast_1d(weight)
    clen = jnp.atleast_1d(clen)
    y = jnp.asarray(y)
    seed = jnp.atleast_1d(seed)
    chunk_offsets = jnp.asarray(chunk_offsets, dtype=jnp.int32)
    assert weight.ndim == clen.ndim == seed.ndim == 1
    assert weight.size == clen.size == seed.size == 1
    assert y.ndim == 1, "y must be 1D."
    assert chunk_offsets.ndim == 2, "chunk_offsets must be 2D."
    assert chunk_offsets.shape == (n_rows, n_chunks), (
        f"chunk_offsets shape mismatch, expected {(n_rows, n_chunks)}, "
        f"got {chunk_offsets.shape}."
    )
    assert jnp.issubdtype(chunk_offsets.dtype, jnp.integer), "chunk_offsets must be an integer type."
    assert jnp.issubdtype(weight.dtype, jnp.floating), "weight must be a floating-point type."
    assert jnp.issubdtype(y.dtype, jnp.floating), "y must be a floating-point type."
    assert weight.dtype == y.dtype, (
        f"weight and y must have the same dtype, got {weight.dtype}, {y.dtype}."
    )
    if np.dtype(weight.dtype) != np.dtype('float32'):
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

    return jitsmv_dt2t_p(
        weight,
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
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        y_info=jax.ShapeDtypeStruct(y.shape, y.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        chunk_offsets_info=jax.ShapeDtypeStruct(chunk_offsets.shape, chunk_offsets.dtype),
    )


jitsmv_dt2t_p = XLACustomKernel(
    'jitsmv_dt2t_fill',
    doc="""
Low-level XLA custom-kernel primitive filling per-synapse ``y * w`` values for a
scalar JITC matrix.

Given the ``chunk_offsets`` produced from the light JIT-scalar CSR count pass,
this primitive walks the same deterministic random stream as
``jits_to_csr(..., matrix_mode="mv")``. For every generated connection it writes
``weight * y[row]`` (``transpose=False``) or ``weight * y[col]``
(``transpose=True``), preserving the same flat CSR data order.
"""
)
jitsmv_dt2t_p.def_cuda_raw_kernel(_jitsmv_dt2t_fill_cuda_kernel)
jitsmv_dt2t_p.def_call(jitsmv_dt2t_p_call)
jitsmv_dt2t_p.def_tags('jit_scalar', 'dt2t')
