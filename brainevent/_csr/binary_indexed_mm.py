# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Fused event-driven CSR matrix-matrix product with a weight gather index.

``binary_csrmm_indexed`` performs the same event-driven scatter as
:func:`brainevent._csr.binary.binary_csrmm`, except heterogeneous weights are
read through a permutation ``perm``: structural slot ``j`` uses
``weights[perm[j]]``.  This lets an *unfavorable* direction (``CSR @ M`` or
``M @ CSC``) reuse the event-driven scatter over a transposed structure while
keeping ``weights`` canonical -- the gather is fused into the kernel so only
active rows/columns are touched.
"""

from pathlib import Path
from typing import Optional

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._misc import _csr_to_coo, namescope
from brainevent._op import numba_kernel, XLACustomKernel, general_batching_rule, load_cuda_file
from brainevent._sddmm import sddmm_coo_indices
from brainevent._typing import Data, Indptr, Index, MatrixShape
from brainevent.config import get_numba_parallel
from .binary import binary_csrmm
from .float import csrmm

__all__ = [
    'binary_csrmm_indexed',
    'binary_csrmm_indexed_p',
]


@namescope(static_argnames=("shape", "transpose"))
def binary_csrmm_indexed(
    data: Data,
    indices: Index,
    indptr: Indptr,
    perm: Index,
    B: Data,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    backend: Optional[str] = None,
) -> Data:
    """Event-driven CSR matrix-matrix product with a fused weight gather index.

    Equivalent to ``binary_csrmm(data[perm], indices, indptr, B, ...)`` for
    heterogeneous weights, but the gather ``data[perm]`` is fused into the
    scatter kernel so only weights of *active* rows/columns are read.  For
    homogeneous weights (``data.size == 1``) ``perm`` is ignored.

    Parameters
    ----------
    data : jax.Array, numpy.ndarray, or brainunit.Quantity
        Non-zero weights in canonical order.  Shape ``(nse,)`` (heterogeneous)
        or ``(1,)`` (homogeneous).
    indices : jax.Array or numpy.ndarray
        Secondary-axis indices of the traversed (transposed) structure,
        shape ``(nse,)``, integer dtype.
    indptr : jax.Array or numpy.ndarray
        Pointer array of the traversed structure, shape ``(shape[0] + 1,)``.
    perm : jax.Array or numpy.ndarray
        Permutation mapping structural slot ``j`` to canonical weight index
        ``perm[j]``, shape ``(nse,)``, same integer dtype as ``indices``.
    B : jax.Array, numpy.ndarray, or brainunit.Quantity
        Dense event matrix.  Shape ``(shape[0], n)`` when ``transpose=True`` or
        ``(shape[1], n)`` when ``transpose=False``.  Boolean or floating dtype.
    shape : tuple of int
        Logical ``(m, k)`` of the traversed structure.
    transpose : bool, optional
        If ``True`` compute the transposed (scatter) product.  Default ``False``.
    backend : str or None, optional
        Compute backend.  Default ``None`` (auto-select).

    Returns
    -------
    C : jax.Array or brainunit.Quantity
        Result matrix, preserving units of ``data`` and ``B``.  Shape
        ``(shape[1], n)`` when ``transpose=True`` or ``(shape[0], n)``.

    See Also
    --------
    brainevent._csr.binary.binary_csrmm : The non-indexed event-driven product.
    brainevent._csr.binary_indexed.binary_csrmv_indexed : The matvec analog.
    """
    data, unitd = u.split_mantissa_unit(data)
    B, unitb = u.split_mantissa_unit(B)
    res = binary_csrmm_indexed_p_call(
        data, indices, indptr, perm, B,
        shape=shape, transpose=transpose, backend=backend,
    )[0]
    return u.maybe_decimal(res * (unitd * unitb))


# --------------------------------------------------------------------------- #
# numba CPU kernel
# --------------------------------------------------------------------------- #
def _csrmm_indexed_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba

    if weight_info.size == 1:
        # Homogeneous weights ignore ``perm``.
        if transpose:
            if vector_info.dtype == jnp.bool_:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mm(weights, indices, indptr, perm, B, posts):
                    w = weights[0]
                    posts[:] = 0.
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            if B[i, k]:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += w
            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mm(weights, indices, indptr, perm, B, posts):
                    w = weights[0]
                    posts[:] = 0.
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            if B[i, k] > 0.:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += w
        else:
            if vector_info.dtype == jnp.bool_:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mm(weights, indices, indptr, perm, B, posts):
                    w = weights[0]
                    n_cols = B.shape[1]
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = np.zeros(n_cols, dtype=posts.dtype)
                        for j in range(indptr[i], indptr[i + 1]):
                            col_idx = indices[j]
                            for k in range(n_cols):
                                if B[col_idx, k]:
                                    r[k] += w
                        posts[i] = r
            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mm(weights, indices, indptr, perm, B, posts):
                    w = weights[0]
                    n_cols = B.shape[1]
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = np.zeros(n_cols, dtype=posts.dtype)
                        for j in range(indptr[i], indptr[i + 1]):
                            col_idx = indices[j]
                            for k in range(n_cols):
                                if B[col_idx, k] > 0.:
                                    r[k] += w
                        posts[i] = r
    else:
        # Heterogeneous weights read through ``perm``.
        if transpose:
            if vector_info.dtype == jnp.bool_:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mm(weights, indices, indptr, perm, B, posts):
                    posts[:] = 0.
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            if B[i, k]:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += weights[perm[j]]
            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mm(weights, indices, indptr, perm, B, posts):
                    posts[:] = 0.
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            if B[i, k] > 0.:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += weights[perm[j]]
        else:
            if vector_info.dtype == jnp.bool_:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mm(weights, indices, indptr, perm, B, posts):
                    n_cols = B.shape[1]
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = np.zeros(n_cols, dtype=posts.dtype)
                        for j in range(indptr[i], indptr[i + 1]):
                            col_idx = indices[j]
                            w = weights[perm[j]]
                            for k in range(n_cols):
                                if B[col_idx, k]:
                                    r[k] += w
                        posts[i] = r
            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mm(weights, indices, indptr, perm, B, posts):
                    n_cols = B.shape[1]
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = np.zeros(n_cols, dtype=posts.dtype)
                        for j in range(indptr[i], indptr[i + 1]):
                            col_idx = indices[j]
                            w = weights[perm[j]]
                            for k in range(n_cols):
                                if B[col_idx, k] > 0.:
                                    r[k] += w
                        posts[i] = r

    def kernel(weights, indices, indptr, perm, B):
        return numba_kernel(mm, outs=kwargs['outs'])(weights, indices, indptr, perm, B)

    return kernel


# --------------------------------------------------------------------------- #
# pure-JAX kernel (CPU/GPU/TPU fallback)
# --------------------------------------------------------------------------- #
def _binary_csrmm_indexed_jax_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs,
):
    """Pure-JAX kernel for fused indexed event-driven CSR matrix-matrix product."""
    m, k = shape
    n = vector_info.shape[1]
    is_homo = (weight_info.size == 1)
    is_bool = (vector_info.dtype == jnp.bool_)
    nse = kwargs['indices_info'].size
    out_dtype = kwargs['outs'][0].dtype

    if transpose:
        def kernel(weights, indices, indptr, perm, B):
            row_ids = jnp.repeat(
                jnp.arange(m, dtype=indptr.dtype),
                jnp.diff(indptr),
                total_repeat_length=nse,
            )
            B_rows = B[row_ids]  # [nse, n]
            events = B_rows.astype(out_dtype) if is_bool else (B_rows > 0.).astype(out_dtype)
            w = weights[0] if is_homo else weights[perm][:, None]
            return (jnp.zeros((k, n), dtype=out_dtype).at[indices].add(w * events),)
    else:
        def kernel(weights, indices, indptr, perm, B):
            row_ids = jnp.repeat(
                jnp.arange(m, dtype=indptr.dtype),
                jnp.diff(indptr),
                total_repeat_length=nse,
            )
            B_rows = B[indices]  # [nse, n]
            events = B_rows.astype(out_dtype) if is_bool else (B_rows > 0.).astype(out_dtype)
            w = weights[0] if is_homo else weights[perm][:, None]
            return (jnp.zeros((m, n), dtype=out_dtype).at[row_ids].add(w * events),)

    return kernel


# --------------------------------------------------------------------------- #
# CUDA warp kernel (source in binary_csrmm.cu)
# --------------------------------------------------------------------------- #
def _binary_csrmm_indexed_cuda_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs,
):
    """Build the CUDA FFI callable for the fused indexed event-driven matmat.

    Mirrors :func:`brainevent._csr.binary._binary_csrmm_cuda_kernel`, but for
    the heterogeneous case selects the ``..._perm_hetero`` kernels that read
    ``weights[perm[j]]`` and pass ``perm`` as an extra tensor.  Homogeneous
    weights reuse the plain homogeneous kernels (called without ``perm``).
    """
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_csrmm.cu'),
        name='csr_binary_csrmm',
    )

    out_info = kwargs['outs']
    is_homo = (weight_info.size == 1)
    spk_suffix = '_bool' if vector_info.dtype == jnp.bool_ else '_float'
    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16',
    }
    wt_sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')

    if is_homo:
        base = 'binary_csrmm_t_warp_homo' if transpose else 'binary_csrmm_nt_auto_homo'
        kernel_name = f'csr_binary_csrmm.{base}{wt_sfx}{spk_suffix}'

        def kernel(weights, indices, indptr, perm, B):
            return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, indptr, B)
    else:
        base = 'binary_csrmm_t_warp_perm_hetero' if transpose else 'binary_csrmm_nt_auto_perm_hetero'
        kernel_name = f'csr_binary_csrmm.{base}{wt_sfx}{spk_suffix}'

        def kernel(weights, indices, indptr, perm, B):
            return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, indptr, perm, B)

    return kernel


def binary_csrmm_indexed_p_call(
    weights,
    indices,
    indptr,
    perm,
    B,
    *,
    shape: MatrixShape,
    transpose: bool,
    backend: Optional[str] = None,
):
    """Low-level primitive call for fused indexed event-driven CSR matmat.

    Validates inputs and dispatches ``binary_csrmm_indexed_p``.  See
    :func:`binary_csrmm_indexed` for the math.
    """
    assert indices.dtype in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64], "Indices must be int32 or int64."
    assert indptr.dtype in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64], "Indptr must be int32 or int64."
    assert perm.dtype in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64], "perm must be int32 or int64."
    assert indptr.ndim == 1, "Indptr must be 1D."
    assert indices.ndim == 1, "Indices must be 1D."
    assert perm.ndim == 1, "perm must be 1D."
    assert perm.shape == indices.shape, "perm must have the same shape as indices."
    assert indptr.dtype == indices.dtype, "Indices and indptr must have the same dtype."
    assert B.ndim == 2, "B must be 2D."
    if transpose:
        assert shape[0] == B.shape[0], "Shape mismatch for transpose operation."
    else:
        assert shape[1] == B.shape[0], "Shape mismatch for non-transpose operation."
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'

    if jnp.ndim(weights) == 0:
        weights = jnp.asarray([weights])

    out_info = (
        jax.ShapeDtypeStruct([shape[1], B.shape[1]], weights.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0], B.shape[1]], weights.dtype)
    )
    return binary_csrmm_indexed_p(
        weights, indices, indptr, perm, B,
        outs=[out_info],
        shape=shape,
        transpose=transpose,
        backend=backend,
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        perm_info=jax.ShapeDtypeStruct(perm.shape, perm.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
    )


# --------------------------------------------------------------------------- #
# autodiff rules
# --------------------------------------------------------------------------- #
def _csrmm_idx_jvp_data(data_dot, data, indices, indptr, perm, B, *, shape, transpose, **kw):
    # Linear in weights (events fixed): re-enter the indexed primitive.
    return binary_csrmm_indexed_p_call(
        data_dot, indices, indptr, perm, B, shape=shape, transpose=transpose, backend=kw['backend']
    )


def _csrmm_idx_jvp_B(B_dot, data, indices, indptr, perm, B, *, shape, transpose, **kw):
    # Event indicator is a step; surrogate tangent routes through float csrmm.
    w = data if data.shape[0] == 1 else data[perm]
    return [csrmm(w, indices, indptr, B_dot, shape=shape, transpose=transpose, backend=kw['backend'])]


def _csrmm_idx_transpose_rule(ct, data, indices, indptr, perm, B, *, shape, transpose, **kw):
    if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr) or ad.is_undefined_primal(perm):
        raise ValueError("Cannot transpose with respect to sparse structure / perm.")
    ct = ct[0]
    if ad.is_undefined_primal(B):
        if type(ct) is ad.Zero:
            ct_B = ad.Zero(B)
        else:
            w = data if data.shape[0] == 1 else data[perm]
            ct_B = csrmm(w, indices, indptr, ct, shape=shape, transpose=not transpose, backend=kw['backend'])
        return data, indices, indptr, perm, ct_B
    else:
        if type(ct) is ad.Zero:
            ct_w = ad.Zero(data)
        elif data.aval.shape[0] == 1:
            base = binary_csrmm_indexed_p_call(
                jnp.ones(1, dtype=data.aval.dtype), indices, indptr, perm, B,
                shape=shape, transpose=transpose, backend=kw['backend'],
            )[0]
            ct_w = jnp.sum(base * ct).reshape(*data.aval.shape)
        else:
            # weight at slot k is weights[perm[k]]; scatter per-slot sddmm back via perm.
            row, col = _csr_to_coo(indices, indptr)
            if transpose:
                d_data = sddmm_coo_indices(B, ct.T, row, col).data
            else:
                d_data = sddmm_coo_indices(B, ct.T, col, row).data
            ct_w = jnp.zeros(data.aval.shape, data.aval.dtype).at[perm].add(d_data)
        return ct_w, indices, indptr, perm, B


def _csrmm_idx_batching(args, axes, **kwargs):
    return general_batching_rule(binary_csrmm_indexed_p, args, axes, **kwargs)


binary_csrmm_indexed_p = XLACustomKernel(
    'binary_csrmm_indexed',
    doc="""
Low-level XLA custom-kernel primitive for ``binary_csrmm_indexed``.

Event-driven CSR sparse matrix-matrix multiplication where heterogeneous
weights are read through a permutation ``perm`` (structural slot ``j`` uses
``weights[perm[j]]``).  Homogeneous weights ignore ``perm``.
""",
)
binary_csrmm_indexed_p.def_numba_kernel(_csrmm_indexed_numba_kernel)
binary_csrmm_indexed_p.def_kernel('jax_raw', 'cpu', _binary_csrmm_indexed_jax_kernel)
binary_csrmm_indexed_p.def_kernel('jax_raw', 'gpu', _binary_csrmm_indexed_jax_kernel)
binary_csrmm_indexed_p.def_kernel('jax_raw', 'tpu', _binary_csrmm_indexed_jax_kernel)
binary_csrmm_indexed_p.def_cuda_raw_kernel(_binary_csrmm_indexed_cuda_kernel, asdefault=True)
binary_csrmm_indexed_p.def_jvp_rule2(_csrmm_idx_jvp_data, None, None, None, _csrmm_idx_jvp_B)
binary_csrmm_indexed_p.def_transpose_rule(_csrmm_idx_transpose_rule)
binary_csrmm_indexed_p.def_batching_rule(_csrmm_idx_batching)
binary_csrmm_indexed_p.def_call(binary_csrmm_indexed_p_call)
binary_csrmm_indexed_p.def_tags('csr', 'binary', 'indexed')
