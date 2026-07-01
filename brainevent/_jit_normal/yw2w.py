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

The public :func:`jitn_yw2w` wrapper returns one value per generated structural
non-zero in canonical CSR flat order. It does not return ``indices`` or
``indptr``; callers that need structure should materialize CSR explicitly.
"""

from pathlib import Path
from typing import Optional

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainevent._compatible_import import Tracer
from brainevent._data import _initialize_conn_length, _initialize_seed
from brainevent._jit_normal.csr import jitn_csr_count_p_call
from brainevent._numba_random import (
    get_numba_lfsr_seed,
    get_numba_lfsr_random_integers,
    get_numba_lfsr_normal,
)
from brainevent._op import XLACustomKernel, load_cuda_file, numba_kernel
from brainevent._typing import MatrixShape

__all__ = [
    'jitn_yw2w',
    'jitn_yw2w_fill_p',
    'jitn_yw2w_fill_p_call',
]

_dtype_sfx = {
    np.dtype('float16'): '_f16',
    np.dtype('float32'): '_f32',
    np.dtype('float64'): '_f64',
    np.dtype('bfloat16'): '_bf16',
}


def jitn_yw2w(
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
):
    """Generate per-synapse ``y * w`` values for a normal JITC matrix."""
    shape = (int(shape[0]), int(shape[1]))

    w_loc, unitd = u.split_mantissa_unit(w_loc)
    w_scale = u.Quantity(w_scale).to(unitd).mantissa
    y, unity = u.split_mantissa_unit(y)

    common_dtype = jnp.result_type(w_loc, w_scale, y)
    w_loc = jnp.atleast_1d(jnp.asarray(w_loc, dtype=common_dtype))
    w_scale = jnp.atleast_1d(jnp.asarray(w_scale, dtype=common_dtype))
    y = jnp.asarray(y, dtype=common_dtype)
    seed = _initialize_seed(seed)

    if y.ndim != 1:
        raise AssertionError("y must be 1D.")
    if transpose:
        assert shape[1] == y.shape[0], "Shape mismatch for transpose operation."
    else:
        assert shape[0] == y.shape[0], "Shape mismatch for non-transpose operation."

    if not isinstance(prob, Tracer) and float(np.asarray(prob)) == 0.0:
        data = jnp.zeros(0, dtype=common_dtype)
        return u.maybe_decimal(data * unitd * unity)

    clen = _initialize_conn_length(prob)
    row_counts = jitn_csr_count_p_call(
        w_loc, w_scale, clen, seed, shape=shape, corder=corder, backend=backend,
    )[0]
    indptr = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(row_counts, dtype=jnp.int32)]
    )
    nnz = int(indptr[-1])

    data = jitn_yw2w_fill_p_call(
        w_loc,
        w_scale,
        clen,
        y,
        seed,
        indptr,
        nnz,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=backend,
    )[0]
    return u.maybe_decimal(data * unitd * unity)


# ---------------------------------------------------------------------- #
#  Fill pass - per-synapse y * w values
# ---------------------------------------------------------------------- #

def _jitn_yw2w_fill_numba_kernel_generator(
    corder: bool,
    shape: MatrixShape,
    transpose: bool,
    **kwargs,
):
    """Build the Numba CPU kernel for the normal JITC ``yw2w`` fill pass."""
    import numba  # pylint: disable=import-outside-toplevel

    _lfsr_seed = get_numba_lfsr_seed()
    _lfsr_random_integers = get_numba_lfsr_random_integers()
    _draw = get_numba_lfsr_normal()
    n_rows, n_cols = int(shape[0]), int(shape[1])

    if corder:
        @numba.njit(fastmath=True)
        def kernel_impl(w0, w1, clen, y, seed, indptr, out):
            loc = w0[0]
            scale = w1[0]
            cl = clen[0]
            s = seed[0]
            for r in range(n_rows):
                state = _lfsr_seed(s + r * n_cols)
                c = _lfsr_random_integers(state, 0, cl - 1)
                pos = indptr[r]
                while c < n_cols:
                    y_value = y[c] if transpose else y[r]
                    out[pos] = _draw(state, loc, scale) * y_value
                    pos += 1
                    c += _lfsr_random_integers(state, 1, cl - 1)
    else:
        @numba.njit(fastmath=True)
        def kernel_impl(w0, w1, clen, y, seed, indptr, out):
            loc = w0[0]
            scale = w1[0]
            cl = clen[0]
            s = seed[0]
            wptr = indptr[:n_rows].copy()
            for c in range(n_cols):
                state = _lfsr_seed(s + c * n_rows)
                rr = _lfsr_random_integers(state, 0, cl - 1)
                while rr < n_rows:
                    pos = wptr[rr]
                    y_value = y[c] if transpose else y[rr]
                    out[pos] = _draw(state, loc, scale) * y_value
                    wptr[rr] += 1
                    rr += _lfsr_random_integers(state, 1, cl - 1)

    def kernel(w0, w1, clen, y, seed, indptr):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(w0, w1, clen, y, seed, indptr)

    return kernel


def _jitn_yw2w_fill_cuda_kernel(
    corder: bool,
    shape: MatrixShape,
    transpose: bool,
    **kwargs,
):
    """Build the CUDA kernel callable for the normal JITC ``yw2w`` fill pass."""
    load_cuda_file(
        Path(__file__).parent.joinpath('yw2w.cu'),
        name='jit_normal_yw2w',
    )
    sfx = _dtype_sfx.get(np.dtype(kwargs['w0_info'].dtype), '_f32')
    order = 'corder_true' if corder else 'corder_false'
    direction = 't' if transpose else 'nt'
    kernel_name = f'jit_normal_yw2w.fill_{order}_{direction}{sfx}'
    n_cols = np.int32(shape[1])

    def kernel(w0, w1, clen, y, seed, indptr):
        return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
            w0, w1, clen, y, seed, indptr, n_cols=n_cols,
        )

    return kernel


def jitn_yw2w_fill_p_call(
    w0,
    w1,
    clen,
    y,
    seed,
    indptr,
    nnz: int,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool,
    backend: Optional[str] = None,
):
    """Invoke the normal JITC ``yw2w`` fill primitive."""
    w0 = jnp.atleast_1d(w0)
    w1 = jnp.atleast_1d(w1)
    clen = jnp.atleast_1d(clen)
    y = jnp.asarray(y)
    seed = jnp.atleast_1d(seed)
    indptr = jnp.asarray(indptr, dtype=jnp.int32)
    shape = tuple(shape)

    assert len(shape) == 2, f"shape must be two-dimensional, but got {shape}."
    assert w0.ndim == w1.ndim == clen.ndim == seed.ndim == 1
    assert w0.size == w1.size == clen.size == seed.size == 1
    assert y.ndim == 1, "y must be 1D."
    assert indptr.ndim == 1, "indptr must be 1D."
    assert indptr.shape[0] == shape[0] + 1, (
        f"indptr shape mismatch, expected {(shape[0] + 1,)}, got {indptr.shape}."
    )
    assert jnp.issubdtype(indptr.dtype, jnp.integer), "indptr must be an integer type."
    assert jnp.issubdtype(w0.dtype, jnp.floating), "w0 must be a floating-point type."
    assert jnp.issubdtype(w1.dtype, jnp.floating), "w1 must be a floating-point type."
    assert jnp.issubdtype(y.dtype, jnp.floating), "y must be a floating-point type."
    assert w0.dtype == w1.dtype == y.dtype, (
        f"w0, w1 and y must have the same dtype, got {w0.dtype}, {w1.dtype}, {y.dtype}."
    )
    if transpose:
        assert shape[1] == y.shape[0], "Shape mismatch for transpose operation."
    else:
        assert shape[0] == y.shape[0], "Shape mismatch for non-transpose operation."

    return jitn_yw2w_fill_p(
        w0,
        w1,
        clen,
        y,
        seed,
        indptr,
        outs=[jax.ShapeDtypeStruct((nnz,), y.dtype)],
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=backend,
        w0_info=jax.ShapeDtypeStruct(w0.shape, w0.dtype),
        w1_info=jax.ShapeDtypeStruct(w1.shape, w1.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        y_info=jax.ShapeDtypeStruct(y.shape, y.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
    )


jitn_yw2w_fill_p = XLACustomKernel(
    'jitn_yw2w_fill',
    doc="""
Low-level XLA custom-kernel primitive filling per-synapse ``y * w`` values for a
normal JITC matrix.

Given the ``indptr`` produced by the JIT-normal CSR count pass, this primitive
walks the same deterministic random connectivity and normal weight stream as
``jitn_to_csr`` and writes either ``weight * y[row]`` or ``weight * y[col]``
into flat CSR order.
"""
)
jitn_yw2w_fill_p.def_numba_kernel(_jitn_yw2w_fill_numba_kernel_generator)
jitn_yw2w_fill_p.def_cuda_raw_kernel(_jitn_yw2w_fill_cuda_kernel)
jitn_yw2w_fill_p.def_call(jitn_yw2w_fill_p_call)
jitn_yw2w_fill_p.def_tags('jit_normal', 'yw2w')
