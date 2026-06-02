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

"""Fused event-driven CSR matrix-vector / matrix-matrix products with a weight gather index.

``binary_csrmv_indexed`` (mat-vec) and ``binary_csrmm_indexed`` (mat-mat)
perform the same event-driven scatter as
:func:`brainevent._csr.binary.binary_csrmv` /
:func:`brainevent._csr.binary.binary_csrmm`, except that heterogeneous weights
are read through a permutation ``perm``: structural slot ``j`` uses
``weights[perm[j]]``.  This lets an *unfavorable* direction (``CSR @ event`` or
``event @ CSC``) reuse the event-driven column/row scatter over a transposed
structure while keeping ``weights`` in its canonical order -- the gather is
*fused* into the kernel so only active rows/columns are touched (no full
``weights[perm]`` materialization on the forward path).
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
from .binary import binary_csrmv, binary_csrmm
from .float import csrmv, csrmm

__all__ = [
    'binary_csrmv_indexed',
    'binary_csrmv_indexed_p',
    'binary_csrmm_indexed',
    'binary_csrmm_indexed_p',
]


@namescope(static_argnames=("shape", "transpose"))
def binary_csrmv_indexed(
    data: Data,
    indices: Index,
    indptr: Indptr,
    perm: Index,
    v: Data,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    backend: Optional[str] = None,
) -> Data:
    """Event-driven CSR matrix-vector product with a fused weight gather index.

    Equivalent to ``binary_csrmv(data[perm], indices, indptr, v, ...)`` for
    heterogeneous weights, but the gather ``data[perm]`` is fused into the
    scatter kernel so that only weights of *active* rows/columns are read.
    For homogeneous weights (``data.size == 1``) ``perm`` is ignored.

    Parameters
    ----------
    data : jax.Array, numpy.ndarray, or brainunit.Quantity
        Non-zero weight values in their *canonical* order.  Shape ``(nse,)``
        for heterogeneous weights or ``(1,)`` for a homogeneous weight.
    indices : jax.Array or numpy.ndarray
        Secondary-axis indices of the (transposed) structure being traversed.
        Shape ``(nse,)`` with integer dtype.
    indptr : jax.Array or numpy.ndarray
        Pointer array of the traversed structure.  Shape ``(shape[0] + 1,)``.
    perm : jax.Array or numpy.ndarray
        Permutation mapping structural slot ``j`` to the canonical weight
        index ``perm[j]``.  Shape ``(nse,)`` with the same integer dtype as
        ``indices``.
    v : jax.Array, numpy.ndarray, or brainunit.Quantity
        Dense event vector.  Shape ``(shape[0],)`` when ``transpose=True`` or
        ``(shape[1],)`` when ``transpose=False``.  Boolean or floating dtype.
    shape : tuple of int
        Logical shape ``(m, k)`` of the traversed structure.
    transpose : bool, optional
        If ``True``, compute the transposed (scatter) product.  Default
        ``False``.
    backend : str or None, optional
        Compute backend.  Default ``None`` (auto-select).

    Returns
    -------
    y : jax.Array or brainunit.Quantity
        Result vector, preserving units of ``data`` and ``v``.

    See Also
    --------
    brainevent._csr.binary.binary_csrmv : The non-indexed event-driven product.
    """
    data, unitd = u.split_mantissa_unit(data)
    v, unitv = u.split_mantissa_unit(v)
    res = binary_csrmv_indexed_p_call(
        data,
        indices,
        indptr,
        perm,
        v,
        shape=shape,
        transpose=transpose,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * (unitd * unitv))


# --------------------------------------------------------------------------- #
# numba CPU kernel
# --------------------------------------------------------------------------- #
def _csrmv_indexed_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba
    if weight_info.size == 1:
        # Homogeneous weights ignore ``perm`` entirely.
        if transpose:
            if vector_info.dtype == jnp.bool_:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, perm, v, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(v.shape[0]):
                        if v[i]:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += w
            else:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, perm, v, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(v.shape[0]):
                        if v[i] > 0.:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += w
        else:
            if vector_info.dtype == jnp.bool_:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mv(weights, indices, indptr, perm, v, posts):
                    w = weights[0]
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = 0.0
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]]:
                                r += w
                        posts[i] = r
            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mv(weights, indices, indptr, perm, v, posts):
                    w = weights[0]
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = 0.0
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]] > 0.:
                                r += w
                        posts[i] = r
    else:
        # Heterogeneous weights read through ``perm``.
        if transpose:
            if vector_info.dtype == jnp.bool_:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, perm, v, posts):
                    posts[:] = 0.
                    for i in range(v.shape[0]):
                        if v[i]:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += weights[perm[j]]
            else:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, perm, v, posts):
                    posts[:] = 0.
                    for i in range(v.shape[0]):
                        if v[i] > 0.:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += weights[perm[j]]
        else:
            if vector_info.dtype == jnp.bool_:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mv(weights, indices, indptr, perm, v, posts):
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = 0.0
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]]:
                                r += weights[perm[j]]
                        posts[i] = r
            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mv(weights, indices, indptr, perm, v, posts):
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = 0.0
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]] > 0.:
                                r += weights[perm[j]]
                        posts[i] = r

    def kernel(weights, indices, indptr, perm, vector):
        return numba_kernel(mv, outs=kwargs['outs'])(weights, indices, indptr, perm, vector)

    return kernel


# --------------------------------------------------------------------------- #
# pure-JAX kernel (CPU/GPU/TPU fallback)
# --------------------------------------------------------------------------- #
def _binary_csrmv_indexed_jax_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs,
):
    """Pure-JAX kernel for fused indexed event-driven CSR matrix-vector product."""
    m, k = shape
    is_homo = (weight_info.size == 1)
    is_bool = (vector_info.dtype == jnp.bool_)
    nse = kwargs['indices_info'].size
    out_dtype = kwargs['outs'][0].dtype

    if transpose:
        def kernel(weights, indices, indptr, perm, vector):
            row_ids = jnp.repeat(
                jnp.arange(m, dtype=indptr.dtype),
                jnp.diff(indptr),
                total_repeat_length=nse,
            )
            v_row = vector[row_ids]
            events = v_row.astype(out_dtype) if is_bool else (v_row > 0.).astype(out_dtype)
            w = weights[0] if is_homo else weights[perm]
            return (jnp.zeros(k, dtype=out_dtype).at[indices].add(w * events),)
    else:
        def kernel(weights, indices, indptr, perm, vector):
            row_ids = jnp.repeat(
                jnp.arange(m, dtype=indptr.dtype),
                jnp.diff(indptr),
                total_repeat_length=nse,
            )
            v_col = vector[indices]
            events = v_col.astype(out_dtype) if is_bool else (v_col > 0.).astype(out_dtype)
            w = weights[0] if is_homo else weights[perm]
            return (jnp.zeros(m, dtype=out_dtype).at[row_ids].add(w * events),)

    return kernel


# --------------------------------------------------------------------------- #
# CUDA warp kernel (homo source in binary_csrmv.cu, indexed source in
# binary_indexed_csrmv.cu)
# --------------------------------------------------------------------------- #
def _binary_csrmv_indexed_cuda_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs,
):
    """Build the CUDA FFI callable for the fused indexed event-driven mat-vec.

    Mirrors :func:`brainevent._csr.binary._binary_csrmv_cuda_kernel`, but for the
    heterogeneous case selects the ``..._perm_hetero`` kernels that read
    ``weights[perm[j]]`` and pass ``perm`` as an extra tensor.  Homogeneous
    weights ignore ``perm`` entirely, so they reuse the plain homogeneous kernels
    (called without ``perm``).
    """
    out_info = kwargs['outs']
    is_homo = (weight_info.size == 1)

    # Spike type suffix
    spk_suffix = '_bool' if vector_info.dtype == jnp.bool_ else '_float'

    # Weight dtype suffix
    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16',
    }
    wt_sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')

    if is_homo:
        # Homogeneous weights ignore ``perm``; reuse the plain homo kernels in
        # ``binary_csrmv.cu`` and call them without the ``perm`` operand.
        load_cuda_file(
            Path(__file__).parent.joinpath('binary_csrmv.cu'),
            name='csr_binary_csrmv',
        )
        base = 'binary_csrmv_t_warp_homo' if transpose else 'binary_csrmv_nt_auto_homo'
        kernel_name = f'csr_binary_csrmv.{base}{wt_sfx}{spk_suffix}'

        def kernel(weights, indices, indptr, perm, vector):
            return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, indptr, vector)
    else:
        # Heterogeneous weights read through ``perm`` via the dedicated indexed
        # kernels in ``binary_indexed_csrmv.cu``.
        load_cuda_file(
            Path(__file__).parent.joinpath('binary_indexed_csrmv.cu'),
            name='csr_binary_indexed_csrmv',
        )
        base = 'binary_csrmv_t_warp_perm_hetero' if transpose else 'binary_csrmv_nt_auto_perm_hetero'
        kernel_name = f'csr_binary_indexed_csrmv.{base}{wt_sfx}{spk_suffix}'

        def kernel(weights, indices, indptr, perm, vector):
            return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, indptr, perm, vector)

    return kernel


def binary_csrmv_indexed_p_call(
    weights,
    indices,
    indptr,
    perm,
    vector,
    *,
    shape: MatrixShape,
    transpose: bool,
    backend: Optional[str] = None,
):
    """Low-level primitive call for fused indexed event-driven CSR mat-vec.

    Validates inputs and dispatches the ``binary_csrmv_indexed_p`` custom
    kernel.  See :func:`binary_csrmv_indexed` for the math.

    Parameters
    ----------
    weights, indices, indptr, perm, vector : jax.Array
        See :func:`binary_csrmv_indexed`.  ``perm`` shares ``indices``'
        shape and integer dtype.
    shape : tuple of int
        Logical shape ``(m, k)`` of the traversed structure.
    transpose : bool
        Transpose flag of the scatter.
    backend : str or None, optional
        Compute backend.

    Returns
    -------
    list of jax.Array
        Single-element list with the result vector.
    """
    assert indices.dtype in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64], "Indices must be int32 or int64."
    assert indptr.dtype in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64], "Indptr must be int32 or int64."
    assert perm.dtype in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64], "perm must be int32 or int64."
    assert indptr.ndim == 1, "Indptr must be 1D."
    assert indices.ndim == 1, "Indices must be 1D."
    assert perm.ndim == 1, "perm must be 1D."
    assert perm.shape == indices.shape, "perm must have the same shape as indices."
    assert indptr.dtype == indices.dtype, "Indices and indptr must have the same dtype."
    if transpose:
        assert shape[0] == vector.shape[0], "Shape mismatch for transpose operation."
    else:
        assert shape[1] == vector.shape[0], "Shape mismatch for non-transpose operation."
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'

    if jnp.ndim(weights) == 0:
        weights = jnp.asarray([weights])

    out_info = (
        jax.ShapeDtypeStruct([shape[1]], weights.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0]], weights.dtype)
    )
    return binary_csrmv_indexed_p(
        weights,
        indices,
        indptr,
        perm,
        vector,
        outs=[out_info],
        shape=shape,
        transpose=transpose,
        backend=backend,
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        perm_info=jax.ShapeDtypeStruct(perm.shape, perm.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(vector.shape, vector.dtype),
    )


# --------------------------------------------------------------------------- #
# autodiff rules (registered in Phase 3 / below)
# --------------------------------------------------------------------------- #
def _idx_jvp_v(v_dot, weights, indices, indptr, perm, v, *, shape, transpose, **kw):
    w = weights if weights.shape[0] == 1 else weights[perm]
    return [csrmv(w, indices, indptr, v_dot, shape=shape, transpose=transpose, backend=kw['backend'])]


def _idx_jvp_weights(w_dot, weights, indices, indptr, perm, v, *, shape, transpose, **kw):
    return binary_csrmv_indexed_p_call(
        w_dot, indices, indptr, perm, v, shape=shape, transpose=transpose, backend=kw['backend']
    )


def _idx_transpose_rule(ct, weights, indices, indptr, perm, events, *, shape, transpose, **kw):
    if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr) or ad.is_undefined_primal(perm):
        raise ValueError("Cannot transpose with respect to sparse structure / perm.")
    ct = ct[0]
    if ad.is_undefined_primal(events):
        if type(ct) is ad.Zero:
            ct_events = ad.Zero(events)
        else:
            w = weights if weights.shape[0] == 1 else weights[perm]
            ct_events = csrmv(w, indices, indptr, ct, shape=shape, transpose=not transpose, backend=kw['backend'])
        return weights, indices, indptr, perm, ct_events
    else:
        if type(ct) is ad.Zero:
            ct_w = ad.Zero(weights)
        elif weights.aval.shape[0] == 1:
            base = binary_csrmv_indexed_p_call(
                jnp.ones(1, dtype=weights.aval.dtype), indices, indptr, perm, events,
                shape=shape, transpose=transpose, backend=kw['backend'],
            )[0]
            ct_w = jnp.inner(ct, base).reshape(*weights.aval.shape)
        else:
            # (row, col) of the traversed structure at slot k; weight at slot k
            # is weights[perm[k]], so scatter the per-slot cotangent back via perm.
            row, col = _csr_to_coo(indices, indptr)
            per_slot = events[row] * ct[col] if transpose else events[col] * ct[row]
            ct_w = jnp.zeros(weights.aval.shape, weights.aval.dtype).at[perm].add(per_slot)
        return ct_w, indices, indptr, perm, events


def _idx_batching(args, axes, **kwargs):
    return general_batching_rule(binary_csrmv_indexed_p, args, axes, **kwargs)


binary_csrmv_indexed_p = XLACustomKernel(
    'binary_csrmv_indexed',
    doc="""
Low-level XLA custom-kernel primitive for ``binary_csrmv_indexed``.

Event-driven CSR sparse matrix-vector multiplication where heterogeneous
weights are read through a permutation ``perm`` (structural slot ``j`` uses
``weights[perm[j]]``).  Homogeneous weights ignore ``perm``.  Registered
backends: ``numba`` (CPU), ``jax_raw`` (CPU/GPU/TPU), and ``cuda_raw`` (CUDA
warp kernels in ``binary_csrmv.cu`` / ``binary_indexed_csrmv.cu``).
""",
)
binary_csrmv_indexed_p.def_numba_kernel(_csrmv_indexed_numba_kernel)
binary_csrmv_indexed_p.def_kernel('jax_raw', 'cpu', _binary_csrmv_indexed_jax_kernel)
binary_csrmv_indexed_p.def_kernel('jax_raw', 'gpu', _binary_csrmv_indexed_jax_kernel)
binary_csrmv_indexed_p.def_kernel('jax_raw', 'tpu', _binary_csrmv_indexed_jax_kernel)
binary_csrmv_indexed_p.def_cuda_raw_kernel(_binary_csrmv_indexed_cuda_kernel, asdefault=True)
binary_csrmv_indexed_p.def_jvp_rule2(_idx_jvp_weights, None, None, None, _idx_jvp_v)
binary_csrmv_indexed_p.def_transpose_rule(_idx_transpose_rule)
binary_csrmv_indexed_p.def_batching_rule(_idx_batching)
binary_csrmv_indexed_p.def_call(binary_csrmv_indexed_p_call)
binary_csrmv_indexed_p.def_tags('csr', 'binary', 'indexed')


# =========================================================================== #
# Matrix-matrix product: binary_csrmm_indexed
# =========================================================================== #


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
# CUDA warp kernel (homo source in binary_csrmm.cu, indexed source in
# binary_indexed_csrmm.cu)
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
        # Homogeneous weights ignore ``perm``; reuse the plain homo kernels in
        # ``binary_csrmm.cu`` and call them without the ``perm`` operand.
        load_cuda_file(
            Path(__file__).parent.joinpath('binary_csrmm.cu'),
            name='csr_binary_csrmm',
        )
        base = 'binary_csrmm_t_warp_homo' if transpose else 'binary_csrmm_nt_auto_homo'
        kernel_name = f'csr_binary_csrmm.{base}{wt_sfx}{spk_suffix}'

        def kernel(weights, indices, indptr, perm, B):
            return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, indptr, B)
    else:
        # Heterogeneous weights read through ``perm`` via the dedicated indexed
        # kernels in ``binary_indexed_csrmm.cu``.
        load_cuda_file(
            Path(__file__).parent.joinpath('binary_indexed_csrmm.cu'),
            name='csr_binary_indexed_csrmm',
        )
        base = 'binary_csrmm_t_warp_perm_hetero' if transpose else 'binary_csrmm_nt_auto_perm_hetero'
        kernel_name = f'csr_binary_indexed_csrmm.{base}{wt_sfx}{spk_suffix}'

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
