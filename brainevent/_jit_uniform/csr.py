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
Dedicated CPU/CUDA operators that materialize a uniform-weight just-in-time
connectivity (JITC) matrix directly into Compressed Sparse Row (CSR) format.

Unlike :meth:`~brainevent.CSR.fromdense`, these operators never allocate the
full dense matrix. They reproduce the *exact* random walk used by the dense
``jitu`` kernels (and thus by ``.todense()``), emitting the non-zero structure
``(data, indices, indptr)`` instead of a dense array. Each stored entry draws
one variate from ``U(low, high)``.

Because the number of stored elements (``nnz``) is data dependent and XLA
requires static output shapes, generation is split into two passes:

1. a *count* pass returning the per-row non-zero counts (``row_counts``), and
2. a *fill* pass that, given the resulting ``indptr``, writes ``indices`` and
   ``data``.

Both passes walk the same pseudo-random stream; the count pass mirrors the
per-connection weight draw of the fill pass so that the two passes agree on the
connection positions. ``nnz`` is read back between the passes, so
:func:`jitu_to_csr` is an eager-only conversion (it cannot be traced under
``jax.jit``), mirroring the ``nse`` requirement of ``CSR.fromdense``.
"""

from pathlib import Path
from typing import Optional

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainevent._compatible_import import Tracer
from brainevent._data import _initialize_conn_length
from brainevent._numba_random import (
    get_numba_lfsr_seed,
    get_numba_lfsr_random_integers,
    get_numba_lfsr_uniform,
)
from brainevent._op import XLACustomKernel, numba_kernel, load_cuda_file
from brainevent._typing import MatrixShape

__all__ = [
    'jitu_to_csr',
    'jitu_csr_count_p',
    'jitu_csr_fill_p',
]

_dtype_sfx = {
    np.dtype('float16'): '_f16',
    np.dtype('float32'): '_f32',
    np.dtype('float64'): '_f64',
    np.dtype('bfloat16'): '_bf16',
}


# ──────────────────────────────────────────────────────────────────────
#  Count pass — per-row non-zero counts
# ──────────────────────────────────────────────────────────────────────

def _jitu_csr_count_numba_kernel_generator(corder: bool, shape: MatrixShape, **kwargs):
    """Build the Numba CPU kernel for the uniform CSR count pass.

    Parameters
    ----------
    corder : bool
        If True, walk rows in the outer loop (each row samples its columns).
        If False, walk columns in the outer loop (each column samples its rows).
    shape : tuple of int
        Logical matrix shape ``(n_rows, n_cols)``.

    Returns
    -------
    callable
        A function ``kernel(w0, w1, clen, seed)`` returning ``row_counts``. The
        per-connection uniform draw is replicated and discarded to keep the PRNG
        stream in sync with the fill pass.
    """
    import numba

    _lfsr_seed = get_numba_lfsr_seed()
    _lfsr_random_integers = get_numba_lfsr_random_integers()
    _draw = get_numba_lfsr_uniform()
    n_rows, n_cols = int(shape[0]), int(shape[1])

    if corder:
        @numba.njit(fastmath=True)
        def kernel_impl(w0, w1, clen, seed, row_counts):
            row_counts[:] = 0
            a = w0[0]
            b = w1[0]
            cl = clen[0]
            s = seed[0]
            for r in range(n_rows):
                state = _lfsr_seed(s + r * n_cols)
                c = _lfsr_random_integers(state, 0, cl - 1)
                cnt = 0
                while c < n_cols:
                    _draw(state, a, b)  # discard; keep PRNG aligned with fill pass
                    cnt += 1
                    c += _lfsr_random_integers(state, 1, cl - 1)
                row_counts[r] = cnt
    else:
        @numba.njit(fastmath=True)
        def kernel_impl(w0, w1, clen, seed, row_counts):
            row_counts[:] = 0
            a = w0[0]
            b = w1[0]
            cl = clen[0]
            s = seed[0]
            for c in range(n_cols):
                state = _lfsr_seed(s + c * n_rows)
                rr = _lfsr_random_integers(state, 0, cl - 1)
                while rr < n_rows:
                    _draw(state, a, b)  # discard; keep PRNG aligned with fill pass
                    row_counts[rr] += 1
                    rr += _lfsr_random_integers(state, 1, cl - 1)

    def kernel(w0, w1, clen, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(w0, w1, clen, seed)

    return kernel


def _jitu_csr_count_cuda_kernel(corder: bool, shape: MatrixShape, **kwargs):
    """Build the CUDA kernel callable for the uniform CSR count pass.

    The matrix column count ``n_cols`` is forwarded as an XLA FFI scalar
    attribute (``attr.n_cols:int32``); ``n_rows`` is recovered on the device
    from ``row_counts.size(0)``.
    """
    load_cuda_file(
        Path(__file__).parent.joinpath('csr.cu'),
        name='jit_uniform_csr',
    )
    sfx = _dtype_sfx.get(np.dtype(kwargs['w0_info'].dtype), '_f32')
    variant = 'corder_true' if corder else 'corder_false'
    kernel_name = f'jit_uniform_csr.count_{variant}{sfx}'
    n_cols = np.int32(shape[1])

    def kernel(w0, w1, clen, seed):
        return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(w0, w1, clen, seed, n_cols=n_cols)

    return kernel


def jitu_csr_count_p_call(w0, w1, clen, seed, *, shape, corder: bool, backend=None):
    """Invoke the uniform CSR count primitive, returning per-row non-zero counts."""
    w0 = jnp.atleast_1d(w0)
    w1 = jnp.atleast_1d(w1)
    clen = jnp.atleast_1d(clen)
    seed = jnp.atleast_1d(seed)
    n_rows = int(shape[0])
    return jitu_csr_count_p(
        w0, w1, clen, seed,
        outs=[jax.ShapeDtypeStruct((n_rows,), jnp.int32)],
        shape=tuple(shape),
        corder=corder,
        backend=backend,
        w0_info=jax.ShapeDtypeStruct(w0.shape, w0.dtype),
        w1_info=jax.ShapeDtypeStruct(w1.shape, w1.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
    )


jitu_csr_count_p = XLACustomKernel(
    'jitu_csr_count',
    doc="""
Low-level XLA custom-kernel primitive counting per-row non-zeros of a uniform JITC matrix.

Walks the same deterministic random connectivity stream as the dense ``jitu``
kernels and returns, for each row, the number of non-zero entries. Used as the
first pass of :func:`jitu_to_csr` to build the CSR ``indptr`` array before the
fill pass writes ``indices`` and ``data``.

See Also
--------
jitu_csr_fill_p : Companion fill primitive.
jitu_to_csr : High-level CSR conversion wrapper.
"""
)
jitu_csr_count_p.def_numba_kernel(_jitu_csr_count_numba_kernel_generator)
jitu_csr_count_p.def_cuda_raw_kernel(_jitu_csr_count_cuda_kernel, asdefault=True)
jitu_csr_count_p.def_call(jitu_csr_count_p_call)
jitu_csr_count_p.def_tags('jit_uniform', 'csr')


# ──────────────────────────────────────────────────────────────────────
#  Fill pass — column indices and values
# ──────────────────────────────────────────────────────────────────────

def _jitu_csr_fill_numba_kernel_generator(corder: bool, shape: MatrixShape, **kwargs):
    """Build the Numba CPU kernel for the uniform CSR fill pass.

    Returns a function ``kernel(w0, w1, clen, seed, indptr)`` writing the
    ``indices`` and ``data`` output arrays. For ``corder=True`` each row writes
    its slice ``[indptr[r], indptr[r+1])`` sequentially. For ``corder=False`` a
    per-row write cursor is advanced as columns are visited in increasing order,
    which yields column-sorted CSR rows. Each entry draws one variate from
    ``U(w0, w1)``.
    """
    import numba

    _lfsr_seed = get_numba_lfsr_seed()
    _lfsr_random_integers = get_numba_lfsr_random_integers()
    _draw = get_numba_lfsr_uniform()
    n_rows, n_cols = int(shape[0]), int(shape[1])

    if corder:
        @numba.njit(fastmath=True)
        def kernel_impl(w0, w1, clen, seed, indptr, indices, data):
            a = w0[0]
            b = w1[0]
            cl = clen[0]
            s = seed[0]
            for r in range(n_rows):
                state = _lfsr_seed(s + r * n_cols)
                c = _lfsr_random_integers(state, 0, cl - 1)
                pos = indptr[r]
                while c < n_cols:
                    indices[pos] = c
                    data[pos] = _draw(state, a, b)
                    pos += 1
                    c += _lfsr_random_integers(state, 1, cl - 1)
    else:
        @numba.njit(fastmath=True)
        def kernel_impl(w0, w1, clen, seed, indptr, indices, data):
            a = w0[0]
            b = w1[0]
            cl = clen[0]
            s = seed[0]
            wptr = indptr[:n_rows].copy()
            for c in range(n_cols):
                state = _lfsr_seed(s + c * n_rows)
                rr = _lfsr_random_integers(state, 0, cl - 1)
                while rr < n_rows:
                    pos = wptr[rr]
                    indices[pos] = c
                    data[pos] = _draw(state, a, b)
                    wptr[rr] += 1
                    rr += _lfsr_random_integers(state, 1, cl - 1)

    def kernel(w0, w1, clen, seed, indptr):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(w0, w1, clen, seed, indptr)

    return kernel


def _jitu_csr_fill_cuda_kernel(corder: bool, shape: MatrixShape, **kwargs):
    """Build the CUDA kernel callable for the uniform CSR fill pass.

    The matrix column count ``n_cols`` is forwarded as an XLA FFI scalar
    attribute (``attr.n_cols:int32``); ``n_rows`` is recovered on the device
    from ``indptr.size(0) - 1``.
    """
    load_cuda_file(
        Path(__file__).parent.joinpath('csr.cu'),
        name='jit_uniform_csr',
    )
    sfx = _dtype_sfx.get(np.dtype(kwargs['w0_info'].dtype), '_f32')
    variant = 'corder_true' if corder else 'corder_false'
    kernel_name = f'jit_uniform_csr.fill_{variant}{sfx}'
    n_cols = np.int32(shape[1])

    def kernel(w0, w1, clen, seed, indptr):
        return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(w0, w1, clen, seed, indptr, n_cols=n_cols)

    return kernel


def jitu_csr_fill_p_call(w0, w1, clen, seed, indptr, nnz: int, *, shape, corder: bool, backend=None):
    """Invoke the uniform CSR fill primitive, returning ``(indices, data)`` of length ``nnz``."""
    w0 = jnp.atleast_1d(w0)
    w1 = jnp.atleast_1d(w1)
    clen = jnp.atleast_1d(clen)
    seed = jnp.atleast_1d(seed)
    indptr = jnp.asarray(indptr, dtype=jnp.int32)
    return jitu_csr_fill_p(
        w0, w1, clen, seed, indptr,
        outs=[
            jax.ShapeDtypeStruct((nnz,), jnp.int32),
            jax.ShapeDtypeStruct((nnz,), w0.dtype),
        ],
        shape=tuple(shape),
        corder=corder,
        backend=backend,
        w0_info=jax.ShapeDtypeStruct(w0.shape, w0.dtype),
        w1_info=jax.ShapeDtypeStruct(w1.shape, w1.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
    )


jitu_csr_fill_p = XLACustomKernel(
    'jitu_csr_fill',
    doc="""
Low-level XLA custom-kernel primitive filling CSR ``indices``/``data`` of a uniform JITC matrix.

Given the ``indptr`` produced from :func:`jitu_csr_count_p`, walks the same
deterministic random connectivity stream as the dense ``jitu`` kernels and
writes the column indices and the uniform weight values for every non-zero
entry. The second pass of :func:`jitu_to_csr`.

See Also
--------
jitu_csr_count_p : Companion count primitive.
jitu_to_csr : High-level CSR conversion wrapper.
"""
)
jitu_csr_fill_p.def_numba_kernel(_jitu_csr_fill_numba_kernel_generator)
jitu_csr_fill_p.def_cuda_raw_kernel(_jitu_csr_fill_cuda_kernel, asdefault=True)
jitu_csr_fill_p.def_call(jitu_csr_fill_p_call)
jitu_csr_fill_p.def_tags('jit_uniform', 'csr')


# ──────────────────────────────────────────────────────────────────────
#  High-level orchestration
# ──────────────────────────────────────────────────────────────────────

def jitu_to_csr(w_low, w_high, prob, seed, *, shape: MatrixShape, corder: bool, backend: Optional[str] = None):
    """Materialize a uniform-weight JITC matrix directly into a :class:`~brainevent.CSR`.

    Parameters
    ----------
    w_low, w_high : array_like or brainunit.Quantity
        Lower and upper bounds of the per-connection uniform weight draw
        ``U(low, high)``.
    prob : float
        Connection probability in ``[0, 1]``.
    seed : int or array_like
        Random seed controlling the connectivity (and weights).
    shape : tuple of int
        Matrix shape ``(n_rows, n_cols)``.
    corder : bool
        Memory layout order flag, matching the JITC matrix.
    backend : str or None, optional
        Compute backend (``'numba'`` or CUDA). Default ``None`` (auto-select).

    Returns
    -------
    CSR
        A :class:`~brainevent.CSR` matrix reproducing the same dense matrix as
        ``.todense()`` for the active backend.

    Notes
    -----
    This is an eager-only conversion: the number of stored elements is read back
    between the count and fill passes, so it cannot be traced under ``jax.jit``.
    """
    from brainevent._csr import CSR

    n_rows = int(shape[0])
    out_shape = (int(shape[0]), int(shape[1]))

    # Unit handling mirrors the dense ``jitu`` wrapper.
    w_low, unitd = u.split_mantissa_unit(w_low)
    w_high = u.Quantity(w_high).to(unitd).mantissa
    # Promote both weight parameters to a common dtype so the kernels (which read
    # ``w0``/``w1`` through a single ``WEIGHT_T`` pointer) see identical types.
    common_dtype = jnp.result_type(w_low, w_high)
    w_low = jnp.atleast_1d(jnp.asarray(w_low, dtype=common_dtype))
    w_high = jnp.atleast_1d(jnp.asarray(w_high, dtype=common_dtype))

    # Statically-zero probability => the empty matrix (avoids clen = 2/0).
    if not isinstance(prob, Tracer) and float(np.asarray(prob)) == 0.0:
        indptr = jnp.zeros(n_rows + 1, dtype=jnp.int32)
        indices = jnp.zeros(0, dtype=jnp.int32)
        data = u.maybe_decimal(jnp.zeros(0, dtype=w_low.dtype) * unitd)
        return CSR((data, indices, indptr), shape=out_shape)

    clen = _initialize_conn_length(prob)

    row_counts = jitu_csr_count_p_call(
        w_low, w_high, clen, seed, shape=shape, corder=corder, backend=backend,
    )[0]
    indptr = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(row_counts, dtype=jnp.int32)]
    )
    nnz = int(indptr[-1])

    indices, data = jitu_csr_fill_p_call(
        w_low, w_high, clen, seed, indptr, nnz, shape=shape, corder=corder, backend=backend,
    )
    data = u.maybe_decimal(data * unitd)
    return CSR((data, indices, indptr), shape=out_shape)
