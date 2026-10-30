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
:func:`brainevent._csr.binary.binary.binary_csrmv` /
:func:`brainevent._csr.binary.binary.binary_csrmm`, except that heterogeneous weights
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

from brainevent._misc import (
    _check_csr_cuda_structure_dtypes,
    _check_csr_structure_dtypes,
    _csr_to_coo,
    namescope,
)
from brainevent._op import numba_kernel, XLACustomKernel, general_batching_rule, load_cuda_file
from .hybrid_config import (
    get_hybrid_config,
    compile_flags_for_config,
    module_suffix_for_config,
)
from brainevent._typing import Data, Indptr, Index, MatrixShape
from brainevent.config import get_numba_parallel
from .binary import (
    _EVENT_FASTMATH,
    _csrmm_entry_weight_ct,
    _entry_weight_ct,
    _event_contrib,
    _numba_event_mask,
    _numba_weighted_events,
    _hybrid_scratch,
    _task_capacity,
    _warn_workspace_deprecated,
    _check_max_row_nnz,
)
from ..float.float import csrmv, csrmm
from brainevent._op.util import cuda_event_abi, dtype_suffix, event_active

__all__ = [
    'binary_csrmv_indexed',
    'binary_csrmv_indexed_p',
    'binary_csrmm_indexed',
    'binary_csrmm_indexed_p',
]


@namescope(name="brainevent.binary_csrmv_indexed", static_argnames=("shape", "transpose", "weighted", "max_row_nnz"))
def _binary_csrmv_indexed_jit(data, indices, indptr, perm, v, *, shape, transpose, backend, weighted, max_row_nnz):
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
        weighted=weighted,
        max_row_nnz=max_row_nnz,
    )[0]
    return u.maybe_decimal(res * (unitd * unitv))


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
    weighted: bool = False,
    max_row_nnz: Optional[int] = None,
    workspace=None,
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
        index ``perm[j]``. Shape ``(nse,)`` with the same dtype as
        ``indptr`` (int32 or int64).
    v : jax.Array, numpy.ndarray, or brainunit.Quantity
        Dense event vector.  Shape ``(shape[0],)`` when ``transpose=True`` or
        ``(shape[1],)`` when ``transpose=False``.  Boolean, integer or
        floating dtype; an entry is active iff it is ``True`` or non-zero (negative values and NaN are
        active), and weights are never scaled by its value.
    shape : tuple of int
        Logical shape ``(m, k)`` of the traversed structure.
    transpose : bool, optional
        If ``True``, compute the transposed (scatter) product.  Default
        ``False``.
    backend : str or None, optional
        Compute backend.  Default ``None`` (auto-select).
    weighted : bool, optional
        If ``True``, ``v`` holds value-weighted events
        (:class:`~brainevent.WeightedArray`): active entries multiply the
        weights by their value instead of by ``1``.  Default ``False``.
    max_row_nnz : int or None, optional
        Upper bound on the longest row of the traversed ``indptr``, used to
        size the ``cuda_raw`` hybrid task queue (see
        :func:`brainevent.binary_csrmv`); validated when ``indptr`` is
        concrete.
    workspace : optional
        Deprecated and ignored; passing it emits a ``DeprecationWarning``.

    Returns
    -------
    y : jax.Array or brainunit.Quantity
        Result vector, preserving units of ``data`` and ``v``.

    See Also
    --------
    brainevent._csr.binary.binary.binary_csrmv : The non-indexed event-driven product.
    """
    _warn_workspace_deprecated(workspace)
    _check_max_row_nnz(indptr, max_row_nnz)
    return _binary_csrmv_indexed_jit(
        data, indices, indptr, perm, v,
        shape=shape, transpose=transpose, backend=backend, weighted=weighted, max_row_nnz=max_row_nnz,
    )


# --------------------------------------------------------------------------- #
# numba CPU kernel
# --------------------------------------------------------------------------- #
def _csrmv_indexed_numba_weighted(homo: bool, transpose: bool):
    """Numba indexed mat-vec kernels for value-weighted events (``w * x``, ``x != 0``)."""
    import numba

    if transpose:
        if homo:
            @numba.njit(fastmath=_EVENT_FASTMATH)
            def mv(weights, indices, indptr, perm, v, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(v.shape[0]):
                    x = v[i]
                    if x != 0:
                        wx = w * x
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j]] += wx
        else:
            @numba.njit(fastmath=_EVENT_FASTMATH)
            def mv(weights, indices, indptr, perm, v, posts):
                posts[:] = 0.
                for i in range(v.shape[0]):
                    x = v[i]
                    if x != 0:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j]] += weights[perm[j]] * x
    else:
        if homo:
            @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
            def mv(weights, indices, indptr, perm, v, posts):
                w = weights[0]
                for i in numba.prange(indptr.shape[0] - 1):
                    r = 0.0
                    for j in range(indptr[i], indptr[i + 1]):
                        x = v[indices[j]]
                        if x != 0:
                            r += w * x
                    posts[i] = r
        else:
            @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
            def mv(weights, indices, indptr, perm, v, posts):
                for i in numba.prange(indptr.shape[0] - 1):
                    r = 0.0
                    for j in range(indptr[i], indptr[i + 1]):
                        x = v[indices[j]]
                        if x != 0:
                            r += weights[perm[j]] * x
                    posts[i] = r
    return mv


def _csrmv_indexed_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba
    if kwargs.get('weighted', False):
        mv_w = _csrmv_indexed_numba_weighted(weight_info.size == 1, transpose)

        def weighted_kernel(weights, indices, indptr, perm, vector):
            output = numba_kernel(mv_w, outs=kwargs['outs'][0])(
                weights, indices, indptr, perm, _numba_weighted_events(vector))
            math_out = output[0] if isinstance(output, (tuple, list)) else output
            return (math_out,)

        return weighted_kernel

    # Numba cannot read float16/bfloat16; such events use the boolean kernel on their mask.
    mask_events = _numba_event_mask(vector_info.dtype)
    bool_events = vector_info.dtype == jnp.bool_ or mask_events
    if weight_info.size == 1:
        # Homogeneous weights ignore ``perm`` entirely.
        if transpose:
            if bool_events:
                @numba.njit(fastmath=_EVENT_FASTMATH)
                def mv(weights, indices, indptr, perm, v, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(v.shape[0]):
                        if v[i]:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += w
            else:
                @numba.njit(fastmath=_EVENT_FASTMATH)
                def mv(weights, indices, indptr, perm, v, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(v.shape[0]):
                        if v[i] != 0.:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += w
        else:
            if bool_events:
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def mv(weights, indices, indptr, perm, v, posts):
                    w = weights[0]
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = 0.0
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]]:
                                r += w
                        posts[i] = r
            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def mv(weights, indices, indptr, perm, v, posts):
                    w = weights[0]
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = 0.0
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]] != 0.:
                                r += w
                        posts[i] = r
    else:
        # Heterogeneous weights read through ``perm``.
        if transpose:
            if bool_events:
                @numba.njit(fastmath=_EVENT_FASTMATH)
                def mv(weights, indices, indptr, perm, v, posts):
                    posts[:] = 0.
                    for i in range(v.shape[0]):
                        if v[i]:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += weights[perm[j]]
            else:
                @numba.njit(fastmath=_EVENT_FASTMATH)
                def mv(weights, indices, indptr, perm, v, posts):
                    posts[:] = 0.
                    for i in range(v.shape[0]):
                        if v[i] != 0.:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += weights[perm[j]]
        else:
            if bool_events:
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def mv(weights, indices, indptr, perm, v, posts):
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = 0.0
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]]:
                                r += weights[perm[j]]
                        posts[i] = r
            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def mv(weights, indices, indptr, perm, v, posts):
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = 0.0
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]] != 0.:
                                r += weights[perm[j]]
                        posts[i] = r

    def kernel(weights, indices, indptr, perm, vector):
        if mask_events:
            vector = event_active(vector)
        output = numba_kernel(mv, outs=kwargs['outs'][0])(weights, indices, indptr, perm, vector)
        math_out = output[0] if isinstance(output, (tuple, list)) else output
        return (math_out,)

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
    nse = kwargs['indices_info'].size
    out_dtype = kwargs['outs'][0].dtype
    weighted = kwargs.get('weighted', False)

    if transpose:
        def kernel(weights, indices, indptr, perm, vector):
            row_ids = jnp.repeat(
                jnp.arange(m, dtype=indptr.dtype),
                jnp.diff(indptr),
                total_repeat_length=nse,
            )
            w = weights[0] if is_homo else weights[perm]
            contrib = _event_contrib(w, vector[row_ids], dtype=out_dtype, weighted=weighted)
            math_out = jnp.zeros(k, dtype=out_dtype).at[indices].add(contrib)
            return (math_out,)
    else:
        def kernel(weights, indices, indptr, perm, vector):
            row_ids = jnp.repeat(
                jnp.arange(m, dtype=indptr.dtype),
                jnp.diff(indptr),
                total_repeat_length=nse,
            )
            w = weights[0] if is_homo else weights[perm]
            contrib = _event_contrib(w, vector[indices], dtype=out_dtype, weighted=weighted)
            math_out = jnp.zeros(m, dtype=out_dtype).at[row_ids].add(contrib)
            return (math_out,)

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

    Mirrors :func:`brainevent._csr.binary.binary._binary_csrmv_cuda_kernel`, but for the
    heterogeneous case selects the ``..._perm_hetero`` kernels that read
    ``weights[perm[j]]`` and pass ``perm`` as an extra tensor.  Homogeneous
    weights ignore ``perm`` entirely, so they reuse the plain homogeneous kernels
    (called without ``perm``).
    """
    _check_csr_cuda_structure_dtypes(kwargs['indices_info'], kwargs['indptr_info'])
    out_info = kwargs['outs'][0]
    is_homo = (weight_info.size == 1)

    event_abi = cuda_event_abi(
        vector_info.dtype, weighted=kwargs.get('weighted', False), value_dtype=weight_info.dtype,
    )
    spk_suffix = event_abi.suffix

    # Weight dtype suffix
    wt_sfx = dtype_suffix(weight_info.dtype)

    if transpose and is_homo:
        _cfg = get_hybrid_config()
        _mod = 'csr_binary_csrmv_hybrid' + module_suffix_for_config(_cfg)
        load_cuda_file(
            Path(__file__).parent.joinpath('binary_csrmv_hybrid.cu'),
            name=_mod,
            extra_cuda_cflags=compile_flags_for_config(_cfg),
            allow_cuda_graph=False,
        )
        kernel_name = f'{_mod}.binary_csrmv_wat_hybrid_homo{wt_sfx}{spk_suffix}'

        def kernel(weights, indices, indptr, perm, vector):
            vector = event_abi.prepare(vector)
            scratch, scratch_info = _hybrid_scratch(kwargs['task_capacity'], indptr.dtype)
            math_out = jax.ffi.ffi_call(
                kernel_name,
                (out_info, *scratch_info),
                input_output_aliases={4: 1, 5: 2, 6: 3},
            )(
                weights,
                indices,
                indptr,
                vector,
                *scratch,
                task_capacity=kwargs['task_capacity'],
            )[0]
            return (math_out,)

    elif transpose:
        _cfg = get_hybrid_config()
        _mod = 'csr_binary_indexed_csrmv_hybrid' + module_suffix_for_config(_cfg)
        load_cuda_file(
            Path(__file__).parent.joinpath('binary_indexed_csrmv_hybrid.cu'),
            name=_mod,
            extra_cuda_cflags=compile_flags_for_config(_cfg),
            allow_cuda_graph=False,
        )
        kernel_name = f'{_mod}.binary_indexed_csrmv_wat_hybrid_hetero{wt_sfx}{spk_suffix}'

        def kernel(weights, indices, indptr, perm, vector):
            vector = event_abi.prepare(vector)
            scratch, scratch_info = _hybrid_scratch(kwargs['task_capacity'], indptr.dtype)
            math_out = jax.ffi.ffi_call(
                kernel_name,
                (out_info, *scratch_info),
                input_output_aliases={5: 1, 6: 2, 7: 3},
            )(
                weights,
                indices,
                indptr,
                perm,
                vector,
                *scratch,
                task_capacity=kwargs['task_capacity'],
            )[0]
            return (math_out,)

    elif is_homo:
        load_cuda_file(
            Path(__file__).parent.joinpath('binary_csrmv.cu'),
            name='csr_binary_csrmv',
        )
        kernel_name = f'csr_binary_csrmv.binary_csrmv_nt_auto_homo{wt_sfx}{spk_suffix}'

        def kernel(weights, indices, indptr, perm, vector):
            vector = event_abi.prepare(vector)
            math_out = jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, indptr, vector)
            return (math_out,)

    else:
        load_cuda_file(
            Path(__file__).parent.joinpath('binary_indexed_csrmv.cu'),
            name='csr_binary_indexed_csrmv',
        )
        kernel_name = f'csr_binary_indexed_csrmv.binary_csrmv_nt_auto_perm_hetero{wt_sfx}{spk_suffix}'

        def kernel(weights, indices, indptr, perm, vector):
            vector = event_abi.prepare(vector)
            math_out = jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, indptr, perm, vector)
            return (math_out,)

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
    weighted: bool = False,
    max_row_nnz: Optional[int] = None,
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
    weighted : bool, optional
        Multiply the weights by the event value instead of the activity
        mask.  Default ``False``.
    max_row_nnz : int or None, optional
        Upper bound on the longest row of ``indptr`` (hybrid task queue size).

    Returns
    -------
    tuple of jax.Array
        A one-element tuple holding the result vector.
    """
    assert perm.dtype in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64], "perm must be int32 or int64."
    assert indptr.ndim == 1, "Indptr must be 1D."
    assert indices.ndim == 1, "Indices must be 1D."
    assert perm.ndim == 1, "perm must be 1D."
    assert perm.shape == indices.shape, "perm must have the same shape as indices."
    _check_csr_structure_dtypes(indices, indptr)
    if transpose:
        assert shape[0] == vector.shape[0], "Shape mismatch for transpose operation."
    else:
        assert shape[1] == vector.shape[0], "Shape mismatch for non-transpose operation."
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'

    if jnp.ndim(weights) == 0:
        weights = jnp.asarray([weights])
    if jnp.dtype(perm.dtype) != jnp.dtype(indptr.dtype):
        perm = perm.astype(indptr.dtype)

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
        weighted=bool(weighted),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        perm_info=jax.ShapeDtypeStruct(perm.shape, perm.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(vector.shape, vector.dtype),
        max_row_nnz=max_row_nnz,
        task_capacity=_task_capacity(indices, indptr, max_row_nnz),
    )


# --------------------------------------------------------------------------- #
# autodiff rules (registered in Phase 3 / below)
# --------------------------------------------------------------------------- #
def _idx_jvp_v(v_dot, weights, indices, indptr, perm, v, *, shape, transpose, **kw):
    w = weights if weights.shape[0] == 1 else weights[perm]
    # v_dot is a dense/float tangent, so this intentionally uses float csrmv.
    return [csrmv(w, indices, indptr, v_dot.astype(w.dtype), shape=shape, transpose=transpose, backend=kw['backend'])]


def _idx_jvp_weights(w_dot, weights, indices, indptr, perm, v, *, shape, transpose, **kw):
    return binary_csrmv_indexed_p_call(
        w_dot,
        indices,
        indptr,
        perm,
        v,
        shape=shape,
        transpose=transpose,
        backend=kw['backend'],
        weighted=kw.get('weighted', False),
        max_row_nnz=kw.get('max_row_nnz'),
    )


def _idx_transpose_rule(ct, weights, indices, indptr, perm, events, *, shape, transpose, **kw):
    if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr) or ad.is_undefined_primal(perm):
        raise ValueError("Cannot transpose with respect to sparse structure / perm.")
    ct = ct[0]
    if ad.is_undefined_primal(events):
        if type(ct) is ad.Zero:
            ct_events = ad.Zero(events.aval)
        else:
            w = weights if weights.shape[0] == 1 else weights[perm]
            # ct_events is dense/float cotangent data, not binary events.
            ct_events = csrmv(w, indices, indptr, ct, shape=shape, transpose=not transpose, backend=kw['backend'])
        return weights, indices, indptr, perm, ct_events
    else:
        if type(ct) is ad.Zero:
            ct_w = ad.Zero(weights.aval)
        else:
            weighted = kw.get('weighted', False)
            # (row, col) of the traversed structure at slot k; weight at slot k is
            # weights[perm[k]]. Inactive slots get exactly 0 (select), even for a
            # NaN/Inf cotangent.
            row, col = _csr_to_coo(indices, indptr)
            if weights.aval.shape[0] == 1:

                def unit_product(ct):
                    base = binary_csrmv_indexed_p_call(
                        jnp.ones(1, dtype=weights.aval.dtype), indices, indptr, perm, events,
                        shape=shape, transpose=transpose, backend=kw['backend'],
                        weighted=weighted, max_row_nnz=kw.get('max_row_nnz'),
                    )[0]
                    return jnp.inner(ct, base)

                def select_sum(ct):
                    return jnp.sum(_entry_weight_ct(events, ct, row, col, transpose=transpose, weighted=weighted))

                # ``ct * 0`` is exact for a finite cotangent only.
                ct_w = jax.lax.cond(
                    jnp.all(jnp.isfinite(ct)), unit_product, select_sum, ct,
                ).astype(weights.aval.dtype).reshape(*weights.aval.shape)
            else:
                per_slot = _entry_weight_ct(events, ct, row, col, transpose=transpose, weighted=weighted)
                ct_w = jnp.zeros(weights.aval.shape, weights.aval.dtype).at[perm].add(per_slot)
        return ct_w, indices, indptr, perm, events


def _idx_batching(args, axes, **kwargs):
    axes = tuple(axes)
    if axes in ((None, None, None, None, 0), (None, None, None, None, 1)):
        assert args[4].ndim == 2, f'Batching axis {axes[4]} requires 2D input.'
        r = binary_csrmm_indexed_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4].T if axes[4] == 0 else args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend'],
            weighted=kwargs.get('weighted', False),
            max_row_nnz=kwargs.get('max_row_nnz'),
        )
        return r, [1]

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


@namescope(name="brainevent.binary_csrmm_indexed", static_argnames=("shape", "transpose", "weighted", "max_row_nnz"))
def _binary_csrmm_indexed_jit(data, indices, indptr, perm, B, *, shape, transpose, backend, weighted, max_row_nnz):
    data, unitd = u.split_mantissa_unit(data)
    B, unitb = u.split_mantissa_unit(B)
    res = binary_csrmm_indexed_p_call(
        data, indices, indptr, perm, B,
        shape=shape, transpose=transpose, backend=backend, weighted=weighted,
        max_row_nnz=max_row_nnz,
    )[0]
    return u.maybe_decimal(res * (unitd * unitb))


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
    weighted: bool = False,
    max_row_nnz: Optional[int] = None,
    workspace=None,
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
        ``perm[j]``, shape ``(nse,)``, same dtype as ``indptr`` (int32 or
        int64).
    B : jax.Array, numpy.ndarray, or brainunit.Quantity
        Dense event matrix.  Shape ``(shape[0], n)`` when ``transpose=True`` or
        ``(shape[1], n)`` when ``transpose=False``.  Boolean, integer or
        floating dtype; an entry is active iff it is ``True`` or non-zero (negative values and NaN are
        active), and weights are never scaled by its value.
    shape : tuple of int
        Logical ``(m, k)`` of the traversed structure.
    transpose : bool, optional
        If ``True`` compute the transposed (scatter) product.  Default ``False``.
    backend : str or None, optional
        Compute backend.  Default ``None`` (auto-select).
    weighted : bool, optional
        If ``True``, ``B`` holds value-weighted events
        (:class:`~brainevent.WeightedArray`): active entries multiply the
        weights by their value instead of by ``1``.  Default ``False``.
    max_row_nnz : int or None, optional
        Upper bound on the longest row of the traversed ``indptr``, used to
        size the ``cuda_raw`` hybrid task queue (see
        :func:`brainevent.binary_csrmv`); validated when ``indptr`` is
        concrete.
    workspace : optional
        Deprecated and ignored; passing it emits a ``DeprecationWarning``.

    Returns
    -------
    C : jax.Array or brainunit.Quantity
        Result matrix, preserving units of ``data`` and ``B``.  Shape
        ``(shape[1], n)`` when ``transpose=True`` or ``(shape[0], n)``.

    See Also
    --------
    brainevent._csr.binary.binary.binary_csrmm : The non-indexed event-driven product.
    brainevent._csr.binary.binary_indexed.binary_csrmv_indexed : The matvec analog.
    """
    _warn_workspace_deprecated(workspace)
    _check_max_row_nnz(indptr, max_row_nnz)
    return _binary_csrmm_indexed_jit(
        data, indices, indptr, perm, B,
        shape=shape, transpose=transpose, backend=backend, weighted=weighted, max_row_nnz=max_row_nnz,
    )


# --------------------------------------------------------------------------- #
# numba CPU kernel
# --------------------------------------------------------------------------- #
def _csrmm_indexed_numba_weighted(homo: bool, transpose: bool):
    """Numba indexed mat-mat kernels for value-weighted events (``w * x``, ``x != 0``)."""
    import numba

    if transpose:
        if homo:
            @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
            def mm(weights, indices, indptr, perm, B, posts):
                w = weights[0]
                posts[:] = 0.
                for k in numba.prange(B.shape[1]):
                    for i in range(B.shape[0]):
                        x = B[i, k]
                        if x != 0:
                            wx = w * x
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j], k] += wx
        else:
            @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
            def mm(weights, indices, indptr, perm, B, posts):
                posts[:] = 0.
                for k in numba.prange(B.shape[1]):
                    for i in range(B.shape[0]):
                        x = B[i, k]
                        if x != 0:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j], k] += weights[perm[j]] * x
    else:
        if homo:
            @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
            def mm(weights, indices, indptr, perm, B, posts):
                w = weights[0]
                n_cols = B.shape[1]
                for i in numba.prange(indptr.shape[0] - 1):
                    r = np.zeros(n_cols, dtype=posts.dtype)
                    for j in range(indptr[i], indptr[i + 1]):
                        col_idx = indices[j]
                        for k in range(n_cols):
                            x = B[col_idx, k]
                            if x != 0:
                                r[k] += w * x
                    posts[i] = r
        else:
            @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
            def mm(weights, indices, indptr, perm, B, posts):
                n_cols = B.shape[1]
                for i in numba.prange(indptr.shape[0] - 1):
                    r = np.zeros(n_cols, dtype=posts.dtype)
                    for j in range(indptr[i], indptr[i + 1]):
                        col_idx = indices[j]
                        w = weights[perm[j]]
                        for k in range(n_cols):
                            x = B[col_idx, k]
                            if x != 0:
                                r[k] += w * x
                    posts[i] = r
    return mm


def _csrmm_indexed_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba
    if kwargs.get('weighted', False):
        mm_w = _csrmm_indexed_numba_weighted(weight_info.size == 1, transpose)

        def weighted_kernel(weights, indices, indptr, perm, B):
            output = numba_kernel(mm_w, outs=kwargs['outs'][0])(
                weights, indices, indptr, perm, _numba_weighted_events(B))
            math_out = output[0] if isinstance(output, (tuple, list)) else output
            return (math_out,)

        return weighted_kernel

    # Numba cannot read float16/bfloat16; such events use the boolean kernel on their mask.
    mask_events = _numba_event_mask(vector_info.dtype)
    bool_events = vector_info.dtype == jnp.bool_ or mask_events

    if weight_info.size == 1:
        # Homogeneous weights ignore ``perm``.
        if transpose:
            if bool_events:
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def mm(weights, indices, indptr, perm, B, posts):
                    w = weights[0]
                    posts[:] = 0.
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            if B[i, k]:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += w
            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def mm(weights, indices, indptr, perm, B, posts):
                    w = weights[0]
                    posts[:] = 0.
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            if B[i, k] != 0.:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += w
        else:
            if bool_events:
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
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
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def mm(weights, indices, indptr, perm, B, posts):
                    w = weights[0]
                    n_cols = B.shape[1]
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = np.zeros(n_cols, dtype=posts.dtype)
                        for j in range(indptr[i], indptr[i + 1]):
                            col_idx = indices[j]
                            for k in range(n_cols):
                                if B[col_idx, k] != 0.:
                                    r[k] += w
                        posts[i] = r
    else:
        # Heterogeneous weights read through ``perm``.
        if transpose:
            if bool_events:
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def mm(weights, indices, indptr, perm, B, posts):
                    posts[:] = 0.
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            if B[i, k]:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += weights[perm[j]]
            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def mm(weights, indices, indptr, perm, B, posts):
                    posts[:] = 0.
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            if B[i, k] != 0.:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += weights[perm[j]]
        else:
            if bool_events:
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
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
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def mm(weights, indices, indptr, perm, B, posts):
                    n_cols = B.shape[1]
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = np.zeros(n_cols, dtype=posts.dtype)
                        for j in range(indptr[i], indptr[i + 1]):
                            col_idx = indices[j]
                            w = weights[perm[j]]
                            for k in range(n_cols):
                                if B[col_idx, k] != 0.:
                                    r[k] += w
                        posts[i] = r

    def kernel(weights, indices, indptr, perm, B):
        if mask_events:
            B = event_active(B)
        output = numba_kernel(mm, outs=kwargs['outs'][0])(weights, indices, indptr, perm, B)
        math_out = output[0] if isinstance(output, (tuple, list)) else output
        return (math_out,)

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
    nse = kwargs['indices_info'].size
    out_dtype = kwargs['outs'][0].dtype
    weighted = kwargs.get('weighted', False)

    if transpose:
        def kernel(weights, indices, indptr, perm, B):
            row_ids = jnp.repeat(
                jnp.arange(m, dtype=indptr.dtype),
                jnp.diff(indptr),
                total_repeat_length=nse,
            )
            w = weights[0] if is_homo else weights[perm][:, None]
            contrib = _event_contrib(w, B[row_ids], dtype=out_dtype, weighted=weighted)  # [nse, n]
            math_out = jnp.zeros((k, n), dtype=out_dtype).at[indices].add(contrib)
            return (math_out,)
    else:
        def kernel(weights, indices, indptr, perm, B):
            row_ids = jnp.repeat(
                jnp.arange(m, dtype=indptr.dtype),
                jnp.diff(indptr),
                total_repeat_length=nse,
            )
            w = weights[0] if is_homo else weights[perm][:, None]
            contrib = _event_contrib(w, B[indices], dtype=out_dtype, weighted=weighted)  # [nse, n]
            math_out = jnp.zeros((m, n), dtype=out_dtype).at[row_ids].add(contrib)
            return (math_out,)

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

    Mirrors :func:`brainevent._csr.binary.binary._binary_csrmm_cuda_kernel`, but for
    the heterogeneous case selects the ``..._perm_hetero`` kernels that read
    ``weights[perm[j]]`` and pass ``perm`` as an extra tensor.  Homogeneous
    weights reuse the plain homogeneous kernels (called without ``perm``).
    """
    _check_csr_cuda_structure_dtypes(kwargs['indices_info'], kwargs['indptr_info'])
    out_info = kwargs['outs'][0]
    is_homo = (weight_info.size == 1)
    event_abi = cuda_event_abi(
        vector_info.dtype, weighted=kwargs.get('weighted', False), value_dtype=weight_info.dtype,
    )
    spk_suffix = event_abi.suffix
    wt_sfx = dtype_suffix(weight_info.dtype)

    if transpose and is_homo:
        _cfg = get_hybrid_config()
        _mod = 'csr_binary_csrmm_hybrid' + module_suffix_for_config(_cfg)
        load_cuda_file(
            Path(__file__).parent.joinpath('binary_csrmm_hybrid.cu'),
            name=_mod,
            extra_cuda_cflags=compile_flags_for_config(_cfg),
            allow_cuda_graph=False,
        )
        logical_out_info = kwargs['outs'][0]
        physical_out_info = jax.ShapeDtypeStruct(
            (logical_out_info.shape[1], logical_out_info.shape[0]),
            logical_out_info.dtype,
        )
        kernel_name = f'{_mod}.binary_csrmm_sraw_hybrid_homo{wt_sfx}{spk_suffix}'

        def kernel(weights, indices, indptr, perm, B):
            B = event_abi.prepare(B)
            scratch, scratch_info = _hybrid_scratch(kwargs['task_capacity'], indptr.dtype)
            output = jax.ffi.ffi_call(
                kernel_name,
                (physical_out_info, *scratch_info),
                input_output_aliases={4: 1, 5: 2, 6: 3},
            )(
                weights,
                indices,
                indptr,
                B,
                *scratch,
                task_capacity=kwargs['task_capacity'],
            )[0]
            return (jnp.transpose(output),)

    elif transpose:
        _cfg = get_hybrid_config()
        _mod = 'csr_binary_indexed_csrmm_hybrid' + module_suffix_for_config(_cfg)
        load_cuda_file(
            Path(__file__).parent.joinpath('binary_indexed_csrmm_hybrid.cu'),
            name=_mod,
            extra_cuda_cflags=compile_flags_for_config(_cfg),
            allow_cuda_graph=False,
        )
        logical_out_info = kwargs['outs'][0]
        physical_out_info = jax.ShapeDtypeStruct(
            (logical_out_info.shape[1], logical_out_info.shape[0]),
            logical_out_info.dtype,
        )
        kernel_name = f'{_mod}.binary_indexed_csrmm_sraw_hybrid_hetero{wt_sfx}{spk_suffix}'

        def kernel(weights, indices, indptr, perm, B):
            B = event_abi.prepare(B)
            scratch, scratch_info = _hybrid_scratch(kwargs['task_capacity'], indptr.dtype)
            output = jax.ffi.ffi_call(
                kernel_name,
                (physical_out_info, *scratch_info),
                input_output_aliases={5: 1, 6: 2, 7: 3},
            )(
                weights,
                indices,
                indptr,
                perm,
                B,
                *scratch,
                task_capacity=kwargs['task_capacity'],
            )[0]
            return (jnp.transpose(output),)

    elif is_homo:
        load_cuda_file(
            Path(__file__).parent.joinpath('binary_csrmm.cu'),
            name='csr_binary_csrmm',
        )
        kernel_name = f'csr_binary_csrmm.binary_csrmm_nt_auto_homo{wt_sfx}{spk_suffix}'

        def kernel(weights, indices, indptr, perm, B):
            B = event_abi.prepare(B)
            math_out = jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, indptr, B)
            return (math_out,)

    else:
        load_cuda_file(
            Path(__file__).parent.joinpath('binary_indexed_csrmm.cu'),
            name='csr_binary_indexed_csrmm',
        )
        kernel_name = f'csr_binary_indexed_csrmm.binary_csrmm_nt_auto_perm_hetero{wt_sfx}{spk_suffix}'

        def kernel(weights, indices, indptr, perm, B):
            B = event_abi.prepare(B)
            math_out = jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, indptr, perm, B)
            return (math_out,)

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
    weighted: bool = False,
    max_row_nnz: Optional[int] = None,
):
    """Low-level primitive call for fused indexed event-driven CSR matmat.

    Validates inputs and dispatches ``binary_csrmm_indexed_p``.  See
    :func:`binary_csrmm_indexed` for the math.
    """
    assert perm.dtype in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64], "perm must be int32 or int64."
    assert indptr.ndim == 1, "Indptr must be 1D."
    assert indices.ndim == 1, "Indices must be 1D."
    assert perm.ndim == 1, "perm must be 1D."
    assert perm.shape == indices.shape, "perm must have the same shape as indices."
    _check_csr_structure_dtypes(indices, indptr)
    assert B.ndim == 2, "B must be 2D."
    if transpose:
        assert shape[0] == B.shape[0], "Shape mismatch for transpose operation."
    else:
        assert shape[1] == B.shape[0], "Shape mismatch for non-transpose operation."
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'

    if jnp.ndim(weights) == 0:
        weights = jnp.asarray([weights])
    if jnp.dtype(perm.dtype) != jnp.dtype(indptr.dtype):
        perm = perm.astype(indptr.dtype)

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
        weighted=bool(weighted),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        perm_info=jax.ShapeDtypeStruct(perm.shape, perm.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
        max_row_nnz=max_row_nnz,
        task_capacity=_task_capacity(indices, indptr, max_row_nnz),
    )


# --------------------------------------------------------------------------- #
# autodiff rules
# --------------------------------------------------------------------------- #
def _csrmm_idx_jvp_data(data_dot, data, indices, indptr, perm, B, *, shape, transpose, **kw):
    # Linear in weights (events fixed): re-enter the indexed primitive.
    return binary_csrmm_indexed_p_call(
        data_dot, indices, indptr, perm, B, shape=shape, transpose=transpose, backend=kw['backend'],
        weighted=kw.get('weighted', False), max_row_nnz=kw.get('max_row_nnz'),
    )


def _csrmm_idx_jvp_B(B_dot, data, indices, indptr, perm, B, *, shape, transpose, **kw):
    # Event indicator is a step; surrogate tangent routes through float csrmm.
    w = data if data.shape[0] == 1 else data[perm]
    return [csrmm(w, indices, indptr, B_dot.astype(w.dtype), shape=shape, transpose=transpose, backend=kw['backend'])]


def _csrmm_idx_transpose_rule(ct, data, indices, indptr, perm, B, *, shape, transpose, **kw):
    if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr) or ad.is_undefined_primal(perm):
        raise ValueError("Cannot transpose with respect to sparse structure / perm.")
    ct = ct[0]
    if ad.is_undefined_primal(B):
        if type(ct) is ad.Zero:
            ct_B = ad.Zero(B.aval)
        else:
            w = data if data.shape[0] == 1 else data[perm]
            ct_B = csrmm(w, indices, indptr, ct, shape=shape, transpose=not transpose, backend=kw['backend'])
        return data, indices, indptr, perm, ct_B
    else:
        if type(ct) is ad.Zero:
            ct_w = ad.Zero(data.aval)
        else:
            # weight at slot k is weights[perm[k]]; scatter the per-slot cotangent
            # back via perm. Inactive (slot, column) pairs contribute exactly 0.
            row, col = _csr_to_coo(indices, indptr)
            d_data = _csrmm_entry_weight_ct(
                B, ct, row, col, transpose=transpose, weighted=kw.get('weighted', False),
            )
            if data.aval.shape[0] == 1:
                ct_w = jnp.sum(d_data).astype(data.aval.dtype).reshape(*data.aval.shape)
            else:
                ct_w = jnp.zeros(data.aval.shape, data.aval.dtype).at[perm].add(d_data)
        return ct_w, indices, indptr, perm, B


def _csrmm_idx_batching(args, axes, **kwargs):
    axes = tuple(axes)
    if axes[:4] == (None, None, None, None) and axes[4] in (0, 1, 2):
        assert args[4].ndim == 3, f'Batching axis {axes[4]} requires 3D input.'
        if axes[4] == 0:
            batch_size, m, n = args[4].shape
            B = jnp.transpose(args[4], (1, 0, 2)).reshape(m, batch_size * n)
        elif axes[4] == 1:
            m, batch_size, n = args[4].shape
            B = args[4].reshape(m, batch_size * n)
        else:
            m, n, batch_size = args[4].shape
            B = args[4].reshape(m, batch_size * n)
        r = binary_csrmm_indexed_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            B,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend'],
            weighted=kwargs.get('weighted', False),
            max_row_nnz=kwargs.get('max_row_nnz'),
        )[0]
        if axes[4] == 2:
            return [jnp.reshape(r, [r.shape[0], n, batch_size])], [2]
        return [jnp.reshape(r, [r.shape[0], batch_size, n])], [1]

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
