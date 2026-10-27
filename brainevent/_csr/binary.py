# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

import warnings
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
from brainevent._op import load_cuda_file
from brainevent._op import numba_kernel, XLACustomKernel, general_batching_rule
from .hybrid_config import (
    get_hybrid_config,
    compile_flags_for_config,
    module_suffix_for_config,
    hybrid_capacity_bound,
    max_row_nnz as _longest_row,
)
from brainevent._op.benchmark import BenchmarkConfig
from brainevent._sddmm import sddmm_coo_indices
from brainevent._typing import Data, Indptr, Index, MatrixShape
from brainevent.config import get_numba_parallel
from .float import csrmv, csrmm, csrmv_p, csrmm_p
from brainevent._op.util import (
    cuda_event_abi, dtype_suffix, event_active, event_factor, NUMBA_EVENT_FASTMATH, numba_event_mask, select_active,
)

__all__ = [
    'binary_csrmv',
    'binary_csrmv_p',
    'binary_csrmm',
    'binary_csrmm_p',
]


def _task_capacity(indices, indptr, max_row_nnz) -> int:
    """Hybrid task-queue capacity bound from static shapes and a row-length hint."""
    return hybrid_capacity_bound(indices.shape[0], indptr.shape[0] - 1, max_row_nnz)


def _hybrid_scratch(task_capacity: int, indptr_dtype):
    """Scratch operands and result types of the hybrid CUDA kernels.

    The kernels zero ``status`` and write ``task_begin``/``task_end`` before
    reading them, so the scratch is created inside the lowering for every call
    and never lives on a matrix.
    """
    infos = (
        jax.ShapeDtypeStruct((task_capacity,), indptr_dtype),
        jax.ShapeDtypeStruct((task_capacity,), indptr_dtype),
        jax.ShapeDtypeStruct((2,), jnp.int32),
    )
    return tuple(jnp.empty(info.shape, info.dtype) for info in infos), infos


def _check_max_row_nnz(indptr, max_row_nnz) -> None:
    """Reject a ``max_row_nnz`` hint below the longest row of a concrete ``indptr``.

    An understated hint makes the ``cuda_raw`` task queue too small, which
    silently drops work; under a trace the hint is trusted.
    """
    if max_row_nnz is None or isinstance(indptr, jax.core.Tracer):
        return
    longest = _longest_row(indptr)
    if int(max_row_nnz) < longest:
        raise ValueError(
            f"max_row_nnz={max_row_nnz} is below the longest row of indptr ({longest})."
        )


def _warn_workspace_deprecated(workspace) -> None:
    if workspace is not None:
        warnings.warn(
            "The `workspace` argument is deprecated and ignored: the hybrid task "
            "scratch is now created inside the kernel. It will be removed in a "
            "future release.",
            DeprecationWarning,
            stacklevel=3,
        )


@namescope(name="brainevent.binary_csrmv", static_argnames=("shape", "transpose", "weighted", "max_row_nnz"))
def _binary_csrmv_jit(data, indices, indptr, v, *, shape, transpose, backend, weighted, max_row_nnz):
    data, unitd = u.split_mantissa_unit(data)
    v, unitv = u.split_mantissa_unit(v)
    res = binary_csrmv_p_call(
        data,
        indices,
        indptr,
        v,
        shape=shape,
        transpose=transpose,
        backend=backend,
        weighted=weighted,
        max_row_nnz=max_row_nnz,
    )[0]
    return u.maybe_decimal(res * (unitd * unitv))


def binary_csrmv(
    data: Data,
    indices: Index,
    indptr: Indptr,
    v: Data,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    backend: Optional[str] = None,
    weighted: bool = False,
    max_row_nnz: Optional[int] = None,
    workspace=None,
) -> Data:
    """
    Product of a CSR sparse matrix and a dense vector using event-driven
    (binary) computation.

    Computes ``y = A @ v`` (or ``y = A.T @ v`` when ``transpose=True``)
    where ``A`` is stored in Compressed Sparse Row format and ``v`` is
    treated as a binary event vector.  Elements of ``v`` that are ``True``
    (boolean) or non-zero (numeric, including negative values and NaN) are
    considered *active events*; only
    those contribute to the result, enabling efficient event-driven
    sparse--dense products commonly used in spiking neural networks.

    The function supports physical units via :mod:`brainunit`.  If ``data``
    or ``v`` carry units, the result is returned in the corresponding
    product unit.

    Parameters
    ----------
    data : jax.Array, numpy.ndarray, or brainunit.Quantity
        Non-zero weight values of the CSR matrix.  Shape ``(nse,)`` for
        heterogeneous weights or ``(1,)`` for a single homogeneous weight
        shared across all connections.
    indices : jax.Array or numpy.ndarray
        Column indices of the non-zero elements.  Shape ``(nse,)`` with
        integer dtype (``int32``, ``int64``, ``uint32``, or ``uint64``).
    indptr : jax.Array or numpy.ndarray
        Row index pointer array.  Shape ``(shape[0] + 1,)`` and same dtype
        as ``indices``.  ``indptr[i]`` and ``indptr[i+1]`` delimit the
        non-zero entries of row ``i``.
    v : jax.Array, numpy.ndarray, or brainunit.Quantity
        Dense event vector.  Shape ``(shape[0],)`` when ``transpose=True``
        or ``(shape[1],)`` when ``transpose=False``.  Dtype may be boolean
        (events indicated by ``True``) or numeric (events indicated
        by non-zero values, including negative values and NaN).
    shape : tuple of int
        Two-element tuple ``(m, k)`` giving the logical shape of the
        sparse matrix ``A``.
    transpose : bool, optional
        If ``True``, the sparse matrix is transposed before multiplication,
        i.e. compute ``A.T @ v``.  Default is ``False``.
    backend : str or None, optional
        Compute backend to use.  One of ``'numba'``,
        ``'pallas'``, or ``None`` (auto-select).  Default is ``None``.
    weighted : bool, optional
        If ``True``, ``v`` holds value-weighted events
        (:class:`~brainevent.WeightedArray`): active entries (``v != 0``)
        multiply the weights by their value, ``e(v[j]) = v[j]``, so the
        result is the exact product ``A @ v``.  Default is ``False``
        (binary events).
    max_row_nnz : int or None, optional
        Upper bound on the number of stored entries in any row of ``indptr``.
        Lets the ``cuda_raw`` scatter kernel size its task queue exactly;
        ``None`` (default) uses a bound from the array sizes alone. A value
        below the true longest row raises ``ValueError`` when ``indptr`` is
        concrete and gives wrong results on ``cuda_raw`` under a trace.
    workspace : optional
        Deprecated and ignored; passing it emits a ``DeprecationWarning``.

    Returns
    -------
    y : jax.Array or brainunit.Quantity
        Result vector.  Shape ``(shape[1],)`` when ``transpose=True`` or
        ``(shape[0],)`` when ``transpose=False``.

    See Also
    --------
    binary_csrmm : Binary CSR matrix--matrix multiplication.
    csrmv : Standard (non-event-driven) CSR matrix--vector multiplication.

    Notes
    -----
    This operation is *event-driven*: instead of performing a full
    sparse--dense product, it skips columns (or rows, when transposed)
    for which the corresponding entry in ``v`` is inactive (``False`` or
    ``0``; negative values and NaN are active).  This yields significant speed-ups when the event vector is
    sparse.  Inactive events contribute exactly ``0`` on every backend, even
    when the weights they would multiply are NaN or +-Inf, and the weight
    cotangent of such entries is exactly ``0``.

    Mathematically, the non-transposed operation computes:

    ``y[i] = sum_{j in nz(i)} A[i, j] * e(v[j])``

    where ``nz(i)`` denotes the set of column indices with non-zero
    entries in row ``i``, and ``e(v[j])`` is the event indicator:

    ``e(v[j]) = 1  if v[j] is True (bool) or v[j] != 0 (numeric)``
    ``e(v[j]) = 0  otherwise``

    When ``transpose=True``, the transposed operation computes:

    ``y[j] = sum_{i in nz_col(j)} A[i, j] * e(v[i])``

    where ``nz_col(j)`` denotes the set of row indices with non-zero
    entries in column ``j``.

    For homogeneous weights (``data`` of shape ``(1,)``), ``A[i, j]``
    is the constant ``data[0]`` for all non-zero positions.

    The operation is differentiable with respect to both ``data`` and
    ``v`` via custom JVP and transpose rules.

    References
    ----------
    .. [1] R. Brette, "Simulation of networks of spiking neurons:
       A review of tools and strategies," *Journal of Computational
       Neuroscience*, vol. 23, pp. 349--398, 2007.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent import binary_csrmv
        >>> data = jnp.array([0.5])           # homogeneous weight
        >>> indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        >>> v = jnp.array([True, False, True])  # binary event vector
        >>> binary_csrmv(data, indices, indptr, v, shape=(2, 3))
        Array([1. , 0.5], dtype=float32)
    """
    _warn_workspace_deprecated(workspace)
    _check_max_row_nnz(indptr, max_row_nnz)
    return _binary_csrmv_jit(
        data, indices, indptr, v,
        shape=shape, transpose=transpose, backend=backend, weighted=weighted, max_row_nnz=max_row_nnz,
    )


@namescope(name="brainevent.binary_csrmm", static_argnames=("shape", "transpose", "weighted", "max_row_nnz"))
def _binary_csrmm_jit(data, indices, indptr, B, *, shape, transpose, backend, weighted, max_row_nnz):
    data, unitd = u.split_mantissa_unit(data)
    B, unitb = u.split_mantissa_unit(B)
    res = binary_csrmm_p_call(
        data,
        indices,
        indptr,
        B,
        shape=shape,
        transpose=transpose,
        backend=backend,
        weighted=weighted,
        max_row_nnz=max_row_nnz,
    )[0]
    return u.maybe_decimal(res * (unitd * unitb))


def binary_csrmm(
    data: Data,
    indices: Index,
    indptr: Indptr,
    B: Data,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    backend: Optional[str] = None,
    weighted: bool = False,
    max_row_nnz: Optional[int] = None,
    workspace=None,
) -> Data:
    """
    Product of a CSR sparse matrix and a dense matrix using event-driven
    (binary) computation.

    Computes ``C = A @ B`` (or ``C = A.T @ B`` when ``transpose=True``)
    where ``A`` is stored in Compressed Sparse Row format and ``B`` is a
    dense matrix whose entries are treated as binary events.  Entries of
    ``B`` that are ``True`` (boolean) or positive (float) are the only
    ones that contribute to the result.

    The function supports physical units via :mod:`brainunit`.

    Parameters
    ----------
    data : jax.Array, numpy.ndarray, or brainunit.Quantity
        Non-zero weight values of the CSR matrix.  Shape ``(nse,)`` for
        heterogeneous weights or ``(1,)`` for a single homogeneous weight.
    indices : jax.Array or numpy.ndarray
        Column indices of the non-zero elements.  Shape ``(nse,)`` with
        integer dtype.
    indptr : jax.Array or numpy.ndarray
        Row index pointer array.  Shape ``(shape[0] + 1,)`` and same dtype
        as ``indices``.
    B : jax.Array, numpy.ndarray, or brainunit.Quantity
        Dense event matrix.  Shape
        ``(shape[0], cols)`` when ``transpose=True`` or
        ``(shape[1], cols)`` when ``transpose=False``.
        Dtype may be boolean, integer or floating-point; an entry is active
        iff it is ``True`` or non-zero (negative values and NaN are active).
    shape : tuple of int
        Two-element tuple ``(m, k)`` giving the logical shape of the
        sparse matrix ``A``.
    transpose : bool, optional
        If ``True``, transpose ``A`` before multiplication.  Default is
        ``False``.
    backend : str or None, optional
        Compute backend.  One of ``'numba'``, ``'pallas'``, or
        ``None`` (auto-select).  Default is ``None``.
    weighted : bool, optional
        If ``True``, ``B`` holds value-weighted events
        (:class:`~brainevent.WeightedArray`): active entries multiply the
        weights by their value, ``e(B[j, l]) = B[j, l]``, so the result is
        the exact product ``A @ B``.  Default is ``False`` (binary events).
    max_row_nnz : int or None, optional
        Upper bound on the number of stored entries in any row of ``indptr``.
        Lets the ``cuda_raw`` scatter kernel size its task queue exactly;
        ``None`` (default) uses a bound from the array sizes alone. A value
        below the true longest row raises ``ValueError`` when ``indptr`` is
        concrete and gives wrong results on ``cuda_raw`` under a trace.
    workspace : optional
        Deprecated and ignored; passing it emits a ``DeprecationWarning``.

    Returns
    -------
    C : jax.Array or brainunit.Quantity
        Result matrix.  Shape ``(shape[1], cols)`` when ``transpose=True``
        or ``(shape[0], cols)`` when ``transpose=False``.

    See Also
    --------
    binary_csrmv : Binary CSR matrix--vector multiplication.
    csrmm : Standard (non-event-driven) CSR matrix--matrix multiplication.

    Notes
    -----
    The operation is *event-driven*: entries of ``B`` that are inactive
    (``False`` or ``0``; negative values and NaN are active) are skipped, so
    they contribute exactly ``0`` on every backend even against NaN/+-Inf
    weights, and the weight cotangent of such entries is exactly ``0``.
    Custom JVP and transpose rules
    are provided for automatic differentiation.

    Mathematically, the non-transposed operation computes:

    ``C[i, l] = sum_{j in nz(i)} A[i, j] * e(B[j, l])``

    where ``nz(i)`` denotes the set of column indices with non-zero
    entries in row ``i`` of the CSR matrix, and ``e(B[j, l])`` is the
    event indicator:

    ``e(B[j, l]) = 1  if B[j, l] is True (bool) or B[j, l] != 0 (numeric)``
    ``e(B[j, l]) = 0  otherwise``

    When ``transpose=True``, the transposed operation computes:

    ``C[j, l] = sum_{i in nz_col(j)} A[i, j] * e(B[i, l])``

    where ``nz_col(j)`` denotes the set of row indices with non-zero
    entries in column ``j``.

    For homogeneous weights (``data`` of shape ``(1,)``), ``A[i, j]``
    is the constant ``data[0]`` for all non-zero positions.

    References
    ----------
    .. [1] R. Brette, "Simulation of networks of spiking neurons:
       A review of tools and strategies," *Journal of Computational
       Neuroscience*, vol. 23, pp. 349--398, 2007.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent import binary_csrmm
        >>> data = jnp.array([1.0, 2.0, 3.0, 4.0])
        >>> indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        >>> B = jnp.array([[True, False],
        ...                [False, True],
        ...                [True, True]])
        >>> binary_csrmm(data, indices, indptr, B, shape=(2, 3))
        Array([[3., 2.],
               [4., 7.]], dtype=float32)
    """
    _warn_workspace_deprecated(workspace)
    _check_max_row_nnz(indptr, max_row_nnz)
    return _binary_csrmm_jit(
        data, indices, indptr, B,
        shape=shape, transpose=transpose, backend=backend, weighted=weighted, max_row_nnz=max_row_nnz,
    )


_EVENT_FASTMATH = NUMBA_EVENT_FASTMATH
_numba_event_mask = numba_event_mask


def _numba_weighted_events(x):
    """Cast float16/bfloat16 weighted events to float32 for numba (exact)."""
    return x.astype(jnp.float32) if _numba_event_mask(x.dtype) else x


def _event_contrib(w, x, *, dtype, weighted: bool) -> jax.Array:
    """Per-entry contribution of an event to a weighted sum, by selection.

    Returns ``where(x != 0, w * f(x), 0)`` with ``f(x) = 1`` (binary) or
    ``f(x) = x`` (weighted). Inactive events contribute exactly ``0`` even
    when ``w`` is NaN or +-Inf: the weight is selected away, never multiplied
    by a 0/1 mask (``NaN * 0`` is NaN).

    Parameters
    ----------
    w : jax.Array
        Weights, broadcastable to ``x``.
    x : jax.Array
        Event values (any boolean, integer or floating dtype).
    dtype : dtype_like
        Output dtype.
    weighted : bool
        Multiply by the event value (``WeightedArray``) instead of its mask.

    Returns
    -------
    jax.Array
        The contributions, with the broadcast shape of ``w`` and ``x``.
    """
    w = jnp.asarray(w).astype(dtype)
    value = w * x.astype(dtype) if weighted else w
    return select_active(event_active(x), value)


def _event_ct(events, ct, *, weighted: bool) -> jax.Array:
    """Weight cotangent ``where(active, f(x) * ct, 0)`` of a per-entry product.

    ``events`` and ``ct`` are already gathered per stored entry. Inactive
    entries get exactly ``0`` even if ``ct`` is NaN or +-Inf.

    Parameters
    ----------
    events : jax.Array
        Event value of every stored entry.
    ct : jax.Array
        Output cotangent of every stored entry.
    weighted : bool
        Value-weighted events.

    Returns
    -------
    jax.Array
        The weight cotangent of every stored entry, in ``ct.dtype``.
    """
    value = events.astype(ct.dtype) * ct if weighted else ct
    return select_active(event_active(events), value)


def _entry_weight_ct(events, ct, row, col, *, transpose: bool, weighted: bool) -> jax.Array:
    """Exact per-entry weight cotangent of a CSR event product, by selection.

    Entry ``e`` reads the event of unit ``row(e)`` (``transpose``) or
    ``col(e)`` and writes output ``col(e)`` or ``row(e)``. The result is
    ``sum_b where(active[unit, b], f(x[unit, b]) * ct[out, b], 0)``: an event
    that is inactive in a batch column contributes exactly ``0`` there, even
    where ``ct`` is NaN/Inf.

    Parameters
    ----------
    events : jax.Array
        ``(n_ev,)`` or ``(n_ev, n)`` events.
    ct : jax.Array
        ``(n_out,)`` or ``(n_out, n)`` output cotangent.
    row, col : jax.Array
        COO coordinates of the stored entries.
    transpose : bool
        Product direction.
    weighted : bool
        Value-weighted events.

    Returns
    -------
    jax.Array
        ``(nse,)`` weight cotangent in ``ct.dtype``.
    """
    unit, out = (row, col) if transpose else (col, row)
    values = _event_ct(events[unit], ct[out], weighted=weighted)
    return values if values.ndim == 1 else jnp.sum(values, axis=1)


def _homo_count_output(count, w, dtype):
    """Scale exact active-event counts by a homogeneous weight, by selection.

    Outputs that received no active event (``count == 0``) stay exactly ``0``
    even for a NaN/Inf weight. Only valid for binary events, whose unit-weight
    product is an exact count (weighted event sums may cancel to ``0``).
    """
    return select_active(count != 0, count * w.astype(dtype))


def _csrmv_numba_weighted(homo: bool, transpose: bool):
    """Numba CSR mat-vec kernels for value-weighted events (``w * x``, ``x != 0``)."""
    import numba

    if transpose:
        if homo:
            @numba.njit(fastmath=_EVENT_FASTMATH)
            def mv(weights, indices, indptr, v, posts):
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
            def mv(weights, indices, indptr, v, posts):
                posts[:] = 0.
                for i in range(v.shape[0]):
                    x = v[i]
                    if x != 0:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j]] += weights[j] * x
    else:
        if homo:
            @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
            def mv(weights, indices, indptr, v, posts):
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
            def mv(weights, indices, indptr, v, posts):
                for i in numba.prange(indptr.shape[0] - 1):
                    r = 0.0
                    for j in range(indptr[i], indptr[i + 1]):
                        x = v[indices[j]]
                        if x != 0:
                            r += weights[j] * x
                    posts[i] = r
    return mv


def _csrmm_numba_weighted(homo: bool, transpose: bool):
    """Numba CSR mat-mat kernels for value-weighted events (``w * x``, ``x != 0``)."""
    import numba

    if transpose:
        if homo:
            @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
            def mm(weights, indices, indptr, B, posts):
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
            def mm(weights, indices, indptr, B, posts):
                posts[:] = 0.
                for k in numba.prange(B.shape[1]):
                    for i in range(B.shape[0]):
                        x = B[i, k]
                        if x != 0:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j], k] += weights[j] * x
    else:
        if homo:
            @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
            def mm(weights, indices, indptr, B, posts):
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
            def mm(weights, indices, indptr, B, posts):
                n_cols = B.shape[1]
                for i in numba.prange(indptr.shape[0] - 1):
                    r = np.zeros(n_cols, dtype=posts.dtype)
                    for j in range(indptr[i], indptr[i + 1]):
                        col_idx = indices[j]
                        w = weights[j]
                        for k in range(n_cols):
                            x = B[col_idx, k]
                            if x != 0:
                                r[k] += w * x
                    posts[i] = r
    return mm


def _csrmv_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba
    if kwargs.get('weighted', False):
        mv_w = _csrmv_numba_weighted(weight_info.size == 1, transpose)

        def weighted_kernel(weights, indices, indptr, vector):
            output = numba_kernel(mv_w, outs=kwargs['outs'][0])(
                weights, indices, indptr, _numba_weighted_events(vector))
            math_out = output[0] if isinstance(output, (tuple, list)) else output
            return (math_out,)

        return weighted_kernel

    # Numba cannot read float16/bfloat16; such events use the boolean kernel on their mask.
    mask_events = _numba_event_mask(vector_info.dtype)
    bool_events = vector_info.dtype == jnp.bool_ or mask_events
    if weight_info.size == 1:
        if transpose:
            if bool_events:
                # Cannot parallelize due to race condition on posts[indices[j]]
                @numba.njit(fastmath=_EVENT_FASTMATH)
                def mv(weights, indices, indptr, v, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(v.shape[0]):
                        if v[i]:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += w

            else:
                @numba.njit(fastmath=_EVENT_FASTMATH)
                def mv(weights, indices, indptr, v, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(v.shape[0]):
                        if v[i] != 0.:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += w

        else:
            if bool_events:
                # Can parallelize by row
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def mv(weights, indices, indptr, v, posts):
                    w = weights[0]
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = 0.0
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]]:
                                r += w
                        posts[i] = r

            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def mv(weights, indices, indptr, v, posts):
                    w = weights[0]
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = 0.0
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]] != 0.:
                                r += w
                        posts[i] = r

    else:
        if transpose:
            if bool_events:
                # Cannot parallelize due to race condition
                @numba.njit(fastmath=_EVENT_FASTMATH)
                def mv(weights, indices, indptr, v, posts):
                    posts[:] = 0.
                    for i in range(v.shape[0]):
                        if v[i]:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += weights[j]

            else:
                @numba.njit(fastmath=_EVENT_FASTMATH)
                def mv(weights, indices, indptr, v, posts):
                    posts[:] = 0.
                    for i in range(v.shape[0]):
                        if v[i] != 0.:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += weights[j]

        else:
            if bool_events:
                # Can parallelize by row
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def mv(weights, indices, indptr, v, posts):
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = 0.0
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]]:
                                r += weights[j]
                        posts[i] = r

            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def mv(weights, indices, indptr, v, posts):
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = 0.0
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]] != 0.:
                                r += weights[j]
                        posts[i] = r

    def kernel(weights, indices, indptr, vector):
        if mask_events:
            vector = event_active(vector)
        output = numba_kernel(mv, outs=kwargs['outs'][0])(weights, indices, indptr, vector)
        math_out = output[0] if isinstance(output, (tuple, list)) else output
        return (math_out,)

    return kernel


def _binary_csrmv_jax_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs,
):
    """Pure-JAX kernel for binary (event-driven) CSR matrix-vector multiplication."""
    m, k = shape
    is_homo = (weight_info.size == 1)
    nse = kwargs['indices_info'].size
    out_dtype = kwargs['outs'][0].dtype
    weighted = kwargs.get('weighted', False)

    if transpose:
        def kernel(weights, indices, indptr, vector):
            row_ids = jnp.repeat(
                jnp.arange(m, dtype=indptr.dtype),
                jnp.diff(indptr),
                total_repeat_length=nse,
            )
            w = weights[0] if is_homo else weights
            contrib = _event_contrib(w, vector[row_ids], dtype=out_dtype, weighted=weighted)
            math_out = jnp.zeros(k, dtype=out_dtype).at[indices].add(contrib)
            return (math_out,)
    else:
        def kernel(weights, indices, indptr, vector):
            row_ids = jnp.repeat(
                jnp.arange(m, dtype=indptr.dtype),
                jnp.diff(indptr),
                total_repeat_length=nse,
            )
            w = weights[0] if is_homo else weights
            contrib = _event_contrib(w, vector[indices], dtype=out_dtype, weighted=weighted)
            math_out = jnp.zeros(m, dtype=out_dtype).at[row_ids].add(contrib)
            return (math_out,)

    return kernel


def _binary_csrmv_cusparse_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs,
):
    """cuSPARSE-backed binary CSR SpMV kernel via ``jax.experimental.sparse`` (GPU only).

    cuSPARSE multiplies every stored weight by its event factor, which leaks a
    NaN/Inf weight through an inactive event (``NaN * 0``). A homogeneous
    weight with binary events scales the exact active-event count by
    selection. Otherwise cuSPARSE runs unchanged when every weight is finite
    (``0 * w`` is then exact) and any NaN/Inf weight takes the select-based
    ``jax_raw`` product via ``lax.cond``.
    """
    import jax.experimental.sparse as jsparse
    m, k = shape
    is_homo = (weight_info.size == 1)
    nse = kwargs['indices_info'].size
    out_dtype = kwargs['outs'][0].dtype
    weighted = kwargs.get('weighted', False)
    # Exact select-based fallback for non-finite weights.
    select_kernel = _binary_csrmv_jax_kernel(weight_info, vector_info, shape, transpose, **kwargs)

    def kernel(weights, indices, indptr, vector):
        events = event_factor(vector, dtype=out_dtype, weighted=weighted)
        cast_indices = indices.astype(indptr.dtype) if indices.dtype != indptr.dtype else indices
        if is_homo and not weighted:
            data = jnp.ones(nse, dtype=out_dtype)
            mat = jsparse.CSR((data, cast_indices, indptr), shape=(m, k))
            count = jsparse.csr_matvec(mat, events, transpose=transpose)
            # ``count`` is an exact active-event count: select instead of
            # ``0 * w`` so a NaN/Inf weight leaves idle outputs exactly 0.
            return (_homo_count_output(count, weights[0], out_dtype),)

        def library_product(weights):
            data = jnp.broadcast_to(weights, (nse,)).astype(out_dtype)
            mat = jsparse.CSR((data, cast_indices, indptr), shape=(m, k))
            return jsparse.csr_matvec(mat, events, transpose=transpose)

        def select_product(weights):
            return select_kernel(weights, indices, indptr, vector)[0]

        # Masking the operand per call costs a full O(nse) row-id expansion
        # (several times the SpMV); the finiteness check is one reduction.
        math_out = jax.lax.cond(
            jnp.all(jnp.isfinite(weights)), library_product, select_product, weights,
        )
        return (math_out,)

    return kernel


def _binary_csrmv_cuda_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs,
):
    _check_csr_cuda_structure_dtypes(kwargs['indices_info'], kwargs['indptr_info'])

    # Determine if weights are homogeneous or heterogeneous
    is_homo = (weight_info.size == 1)
    homo_suffix = '_homo' if is_homo else '_hetero'

    event_abi = cuda_event_abi(
        vector_info.dtype, weighted=kwargs.get('weighted', False), value_dtype=weight_info.dtype,
    )
    spk_suffix = event_abi.suffix

    # Weight dtype suffix
    wt_sfx = dtype_suffix(weight_info.dtype)

    if transpose:
        _cfg = get_hybrid_config()
        _mod = 'csr_binary_csrmv_hybrid' + module_suffix_for_config(_cfg)
        load_cuda_file(
            Path(__file__).parent.joinpath('binary_csrmv_hybrid.cu'),
            name=_mod,
            extra_cuda_cflags=compile_flags_for_config(_cfg),
            allow_cuda_graph=False,
        )
        kernel_name = f'{_mod}.binary_csrmv_wat_hybrid{homo_suffix}{wt_sfx}{spk_suffix}'

        def kernel(weights, indices, indptr, vector):
            vector = event_abi.prepare(vector)
            scratch, scratch_info = _hybrid_scratch(kwargs['task_capacity'], indptr.dtype)
            math_out = jax.ffi.ffi_call(
                kernel_name,
                (kwargs['outs'][0], *scratch_info),
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

    else:
        load_cuda_file(
            Path(__file__).parent.joinpath('binary_csrmv.cu'),
            name='csr_binary_csrmv',
        )
        out_info = kwargs['outs'][0]
        kernel_name = f'csr_binary_csrmv.binary_csrmv_nt_auto{homo_suffix}{wt_sfx}{spk_suffix}'

        def kernel(weights, indices, indptr, vector):
            vector = event_abi.prepare(vector)
            math_out = jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, indptr, vector)
            return (math_out,)

    return kernel


def _grad_backend(backend, primitive):
    """Backend to use when an autodiff rule defers to a *float* CSR primitive.

    The binary primitives expose GPU-only backends (e.g. ``cusparse``) that the
    plain float :func:`csrmv` / :func:`csrmm` used to form tangents and cotangents
    do not register.  Forwarding such a name would raise
    :class:`~brainevent.KernelFallbackExhaustedError` during the backward pass, so
    fall back to automatic backend selection (``None``) whenever the float
    ``primitive`` cannot service the requested ``backend``.  Backends the float
    primitive *does* register (``cuda_raw``, ``jax_raw``, ``numba``) are forwarded
    unchanged, preserving the existing behaviour.

    Parameters
    ----------
    backend : str or None
        The backend requested for the forward (binary) primitive.
    primitive : XLACustomKernel
        The float primitive the autodiff rule dispatches to.

    Returns
    -------
    str or None
        ``backend`` if it is registered for *primitive* on any platform, else
        ``None`` for automatic selection.
    """
    if backend is None:
        return None
    if any(backend in kernels for kernels in primitive._kernels.values()):
        return backend
    return None


def _csrmv_jvp_v(v_dot, data, indices, indptr, v, *, shape, transpose, **kwargs):
    # The float primitive needs a weight-dtype tangent (numba has no float16).
    return [
        csrmv(data, indices, indptr, v_dot.astype(data.dtype), shape=shape, transpose=transpose,
              backend=_grad_backend(kwargs['backend'], csrmv_p))
    ]


def _csrmv_jvp_weights(data_dot, data, indices, indptr, v, *, shape, transpose, **kwargs):
    return binary_csrmv_p_call(
        data_dot,
        indices,
        indptr,
        v,
        shape=shape,
        transpose=transpose,
        backend=kwargs['backend'],
        weighted=kwargs.get('weighted', False),
        max_row_nnz=kwargs.get('max_row_nnz'),
    )


def _csrmv_transpose_rule(ct, data, indices, indptr, events, *, shape, transpose, **kwargs):
    if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
        raise ValueError("Cannot transpose with respect to sparse indices.")

    ct = ct[0]
    if ad.is_undefined_primal(events):
        if type(ct) is ad.Zero:
            ct_events = ad.Zero(events.aval)
        else:
            ct_events = csrmv(data, indices, indptr, ct,
                              shape=shape, transpose=not transpose,
                              backend=_grad_backend(kwargs['backend'], csrmv_p))
        return data, indices, indptr, ct_events
    else:
        if type(ct) is ad.Zero:
            ct_values = ad.Zero(data.aval)
        else:
            weighted = kwargs.get('weighted', False)
            row, col = _csr_to_coo(indices, indptr)
            if data.aval.shape[0] == 1:  # scalar

                def unit_product(ct):
                    r = binary_csrmv_p_call(
                        jnp.ones(1, dtype=data.aval.dtype),
                        indices,
                        indptr,
                        events,
                        shape=shape,
                        transpose=transpose,
                        backend=kwargs['backend'],
                        weighted=weighted,
                        max_row_nnz=kwargs.get('max_row_nnz'),
                    )[0]
                    return jnp.inner(ct, r)

                def select_sum(ct):
                    return jnp.sum(_entry_weight_ct(events, ct, row, col, transpose=transpose, weighted=weighted))

                # ``ct * 0`` is exact for a finite cotangent; a NaN/Inf one
                # takes the per-entry select so it cannot leak through an
                # output that received only inactive events.
                ct_values = jax.lax.cond(
                    jnp.all(jnp.isfinite(ct)), unit_product, select_sum, ct,
                ).astype(data.aval.dtype).reshape(*data.aval.shape)
            else:  # heterogeneous values
                # Binary: the forward selects the weight of active events. Weighted:
                # it multiplies by the value. Inactive entries get exactly 0, by
                # selection, even for a NaN/Inf output cotangent.
                ct_values = _entry_weight_ct(events, ct, row, col, transpose=transpose, weighted=weighted)
        return ct_values, indices, indptr, events


def _csrmv_batching(args, axes, **kwargs):
    axes = tuple(axes)
    if axes in ((None, None, None, 0), (None, None, None, 1)):
        assert args[3].ndim == 2, 'Batching requires 2D input.'
        r = binary_csrmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3].T if axes[3] == 0 else args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend'],
            weighted=kwargs.get('weighted', False),
            max_row_nnz=kwargs.get('max_row_nnz'),
        )
        return r, [1]

    return general_batching_rule(binary_csrmv_p, args, axes, **kwargs)


def _binary_csrmv_benchmark_data(*, platform):
    rng = np.random.default_rng(0)
    dtype = jnp.float32

    # ── Sweep A: size × sparsity × variant sweep (square matrices) ───────────
    for n in (1000, 5000, 10000):
        for conn_prob in (0.01, 0.1):
            n_conn = max(1, int(n * conn_prob))
            nnz = n * n_conn
            p_pct = int(round(conn_prob * 100))
            for transpose in (False, True):
                for homo in (True, False):
                    for event_prob in [0.001, 0.01, 0.1]:
                        for event_type in ('float', 'bool'):
                            indptr = np.arange(n + 1, dtype=np.int32) * n_conn
                            indices = rng.integers(0, n, size=nnz, dtype=np.int32)
                            weights = jnp.ones(1, dtype=dtype) if homo else jnp.ones(nnz, dtype=dtype)
                            v_size = n  # square: n_pre == n_post == n
                            data = rng.random(v_size) > event_prob
                            event_dtype = jnp.float32 if event_type == 'float' else jnp.bool_
                            vector = jnp.asarray(data, dtype=event_dtype)
                            indptr = jnp.asarray(indptr)
                            name = (f"{n}x{n},p={p_pct}%,"
                                    f"{'T' if transpose else 'NT'},"
                                    f"{'homo' if homo else 'hetero'},"
                                    f"{event_type}")
                            yield BenchmarkConfig(
                                name,
                                jax.block_until_ready(
                                    (weights, jnp.asarray(indices), indptr, vector)
                                ),
                                {'shape': (n, n), 'transpose': transpose},
                                {'n_pre': n, 'n_post': n, 'nnz': nnz, 'csr_sparsity': conn_prob,
                                 'event_sparsity': event_prob, 'event_dtype': event_type},
                            )

    # ── Sweep C: rectangular (n_pre ≠ n_post), p=0.1 ─────────────────────────
    for n_pre, n_post in ((1000, 5000), (5000, 1000), (1000, 10000), (10000, 1000)):
        conn_prob = 0.1
        n_conn = max(1, int(n_post * conn_prob))
        nnz = n_pre * n_conn
        for transpose in (False, True):
            for homo in (True, False):
                for event_prob in [0.001, 0.01, 0.1]:
                    for event_type in ('float', 'bool'):
                        indptr = np.arange(n_pre + 1, dtype=np.int32) * n_conn
                        indices = rng.integers(0, n_post, size=nnz, dtype=np.int32)
                        weights = jnp.ones(1, dtype=dtype) if homo else jnp.ones(nnz, dtype=dtype)
                        v_size = n_pre if transpose else n_post
                        data = rng.random(v_size) > event_prob
                        event_dtype = jnp.float32 if event_type == 'float' else jnp.bool_
                        vector = jnp.asarray(data, dtype=event_dtype)
                        indptr = jnp.asarray(indptr)
                        name = (f"{n_pre}x{n_post},p=10%,"
                                f"{'T' if transpose else 'NT'},"
                                f"{'homo' if homo else 'hetero'},"
                                f"{event_type}")
                        yield BenchmarkConfig(
                            name,
                            jax.block_until_ready(
                                (weights, jnp.asarray(indices), indptr, vector)
                            ),
                            {'shape': (n_pre, n_post), 'transpose': transpose},
                            {'n_pre': n_pre, 'n_post': n_post, 'nnz': nnz, 'csr_sparsity': conn_prob,
                             'event_sparsity': event_prob, 'event_dtype': event_type},
                        )


def binary_csrmv_p_call(
    weights,
    indices,
    indptr,
    vector,
    *,
    shape: MatrixShape,
    transpose: bool,
    backend: Optional[str] = None,
    weighted: bool = False,
    max_row_nnz: Optional[int] = None,
):
    """
    Low-level primitive call for event-driven CSR matrix--vector
    multiplication.

    Prepares inputs, validates shapes and dtypes, and dispatches the
    ``binary_csrmv_p`` XLA custom kernel to perform the computation
    ``y = A @ v`` (or ``y = A.T @ v``), where ``A`` is a CSR matrix and
    ``v`` is a binary event vector.

    Parameters
    ----------
    weights : jax.Array
        Non-zero weight values of the CSR matrix.  Shape ``(nse,)`` for
        heterogeneous weights, ``(1,)`` for a homogeneous weight, or a
        scalar (automatically promoted to shape ``(1,)``).
    indices : jax.Array
        Column indices of non-zero elements.  Shape ``(nse,)`` with dtype
        ``int32``, ``int64``, ``uint32``, or ``uint64``.
    indptr : jax.Array
        Row index pointer array.  Shape ``(shape[0] + 1,)`` and same dtype
        as ``indices``.
    vector : jax.Array
        Dense event vector.  Shape ``(shape[0],)`` when
        ``transpose=True`` or ``(shape[1],)`` when ``transpose=False``.
        Dtype may be boolean or floating-point.
    shape : tuple of int
        Two-element tuple ``(m, k)`` giving the logical shape of the
        sparse matrix.
    transpose : bool
        If ``True``, transpose the sparse matrix before multiplication.
    backend : str or None, optional
        Compute backend to use.  Default is ``None`` (auto-select).
    weighted : bool, optional
        Multiply the weights by the event value instead of the activity
        mask (:class:`~brainevent.WeightedArray`).  Default is ``False``.
    max_row_nnz : int or None, optional
        Upper bound on the longest row of ``indptr``, used to size the
        ``cuda_raw`` hybrid task queue (see :func:`binary_csrmv`).

    Returns
    -------
    tuple of jax.Array
        A one-element tuple holding the result vector, of shape
        ``(shape[1],)`` when ``transpose=True`` or ``(shape[0],)`` when
        ``transpose=False``.

    Raises
    ------
    AssertionError
        If ``indices`` or ``indptr`` have a dtype other than ``int32``,
        ``int64``, ``uint32``, or ``uint64``.
    AssertionError
        If ``indices`` and ``indptr`` do not share the same dtype.
    AssertionError
        If ``indptr`` or ``indices`` is not 1-D.
    AssertionError
        If ``weights`` does not have a floating-point dtype.
    AssertionError
        If there is a shape mismatch between ``vector`` and the sparse
        matrix ``shape`` (considering the ``transpose`` flag).

    See Also
    --------
    binary_csrmv : High-level wrapper with unit support.

    Notes
    -----
    Scalar ``weights`` (0-d arrays) are automatically promoted to
    shape ``(1,)`` to indicate a homogeneous weight across all
    connections.

    The computation performed is:

    ``y[i] = sum_{j in nz(i)} w[j] * e(v[j])``  (non-transposed)

    ``y[j] = sum_{i in nz_col(j)} w[i] * e(v[i])``  (transposed)

    where ``e(x)`` is ``1`` when ``x`` is ``True`` (boolean) or
    ``x != 0`` (numeric), and ``0`` otherwise.  ``w[j]`` is either
    ``weights[j]`` (heterogeneous) or ``weights[0]`` (homogeneous).

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._csr.binary import binary_csrmv_p_call
        >>> weights = jnp.array([0.5])
        >>> indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        >>> vector = jnp.array([True, False, True])
        >>> binary_csrmv_p_call(weights, indices, indptr, vector, shape=(2, 3), transpose=False)[0]
        Array([1. , 0.5], dtype=float32)
    """
    assert indptr.ndim == 1, "Indptr must be 1D."
    assert indices.ndim == 1, "Indices must be 1D."
    _check_csr_structure_dtypes(indices, indptr)
    if transpose:
        assert shape[0] == vector.shape[0], "Shape mismatch for transpose operation."
    else:
        assert shape[1] == vector.shape[0], "Shape mismatch for non-transpose operation."
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'

    # Check if weights is a scalar. If so, convert it to a one-dimensional array.
    if jnp.ndim(weights) == 0:
        weights = jnp.asarray([weights])

    # Determine the output shape and data type based on whether the sparse matrix is transposed.
    out_info = (
        # If transpose is True, the output shape is (shape[1],).
        jax.ShapeDtypeStruct([shape[1]], weights.dtype)
        if transpose else
        # If transpose is False, the output shape is (shape[0],).
        jax.ShapeDtypeStruct([shape[0]], weights.dtype)
    )

    # Call the binary_csrmv_p custom operation to perform the matrix-vector multiplication.
    return binary_csrmv_p(
        weights,
        indices,
        indptr,
        vector,
        outs=[out_info],
        shape=shape,
        transpose=transpose,
        backend=backend,
        weighted=bool(weighted),
        # Provide shape and data type information for indices.
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        # Provide shape and data type information for indptr.
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        # Provide shape and data type information for weights.
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        # Provide shape and data type information for v.
        vector_info=jax.ShapeDtypeStruct(vector.shape, vector.dtype),
        max_row_nnz=max_row_nnz,
        task_capacity=_task_capacity(indices, indptr, max_row_nnz),
    )


binary_csrmv_p = XLACustomKernel(
    'binary_csrmv',
    doc="""
Low-level XLA custom-kernel primitive for ``binary_csrmv``.

This ``XLACustomKernel`` instance dispatches the binary (event-driven) CSR sparse matrix-vector multiplication
operation to registered backends (``numba``, ``pallas``, ``cuda_raw``),
using runtime shape/dtype metadata provided by the high-level wrapper.

Only entries of ``v`` that are ``True`` (boolean) or non-zero (numeric) are considered active events
and contribute to the output, enabling efficient event-driven sparse-dense products commonly used
in spiking neural networks.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``binary_csrmv_p.available_backends(platform)``,
and the default backend can be configured with ``binary_csrmv_p.set_default(platform, backend)``.

See Also
--------
binary_csrmv : High-level user-facing function wrapper.
"""
)
binary_csrmv_p.def_numba_kernel(_csrmv_numba_kernel)
binary_csrmv_p.def_cuda_raw_kernel(_binary_csrmv_cuda_kernel, asdefault=True)
binary_csrmv_p.def_kernel('jax_raw', 'cpu', _binary_csrmv_jax_kernel)
binary_csrmv_p.def_kernel('jax_raw', 'gpu', _binary_csrmv_jax_kernel)
binary_csrmv_p.def_kernel('jax_raw', 'tpu', _binary_csrmv_jax_kernel)
binary_csrmv_p.def_kernel('cusparse', 'gpu', _binary_csrmv_cusparse_kernel)
binary_csrmv_p.def_jvp_rule2(_csrmv_jvp_weights, None, None, _csrmv_jvp_v)
binary_csrmv_p.def_transpose_rule(_csrmv_transpose_rule)
binary_csrmv_p.def_batching_rule(_csrmv_batching)
binary_csrmv_p.def_call(binary_csrmv_p_call)
binary_csrmv_p.def_tags('csr', 'binary')
binary_csrmv_p.def_benchmark_data(_binary_csrmv_benchmark_data)


def _csrmm_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba
    if kwargs.get('weighted', False):
        mm_w = _csrmm_numba_weighted(weight_info.size == 1, transpose)

        def weighted_kernel(weights, indices, indptr, B):
            output = numba_kernel(mm_w, outs=kwargs['outs'][0])(
                weights, indices, indptr, _numba_weighted_events(B))
            math_out = output[0] if isinstance(output, (tuple, list)) else output
            return (math_out,)

        return weighted_kernel

    # Numba cannot read float16/bfloat16; such events use the boolean kernel on their mask.
    mask_events = _numba_event_mask(vector_info.dtype)
    bool_events = vector_info.dtype == jnp.bool_ or mask_events

    if weight_info.size == 1:
        if transpose:
            #
            # csr.T @ B
            #
            # [k, m] @ [k, n]
            #
            if bool_events:
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def mm(weights, indices, indptr, B, posts):
                    w = weights[0]
                    posts[:] = 0.
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            if B[i, k]:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += w

            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def mm(weights, indices, indptr, B, posts):
                    B = B != 0.
                    w = weights[0]
                    posts[:] = 0.
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            if B[i, k]:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += w

        else:
            # csr @ B
            if bool_events:
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def mm(weights, indices, indptr, B, posts):
                    w = weights[0]
                    posts[:] = 0.
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = np.zeros(B.shape[1], dtype=weights.dtype)
                        for j in range(indptr[i], indptr[i + 1]):
                            index = indices[j]
                            for k in range(B.shape[1]):
                                if B[index, k]:
                                    r[k] += w
                        posts[i] = r

            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def mm(weights, indices, indptr, B, posts):
                    w = weights[0]
                    B = B != 0.
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = np.zeros(B.shape[1], dtype=weights.dtype)
                        for j in range(indptr[i], indptr[i + 1]):
                            index = indices[j]
                            for k in range(B.shape[1]):
                                if B[index, k]:
                                    r[k] += w
                        posts[i] = r

    else:
        if transpose:
            # csr.T @ B

            if bool_events:
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def mm(weights, indices, indptr, B, posts):
                    posts[:] = 0.
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            if B[i, k]:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += weights[j]

            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def mm(weights, indices, indptr, B, posts):
                    B = B != 0.
                    posts[:] = 0.
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            if B[i, k]:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += weights[j]

        else:
            # csr @ B
            # Fixed: Changed range to prange for parallelization

            if bool_events:
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def mm(weights, indices, indptr, B, posts):
                    n_cols = B.shape[1]
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = np.zeros(n_cols, dtype=posts.dtype)
                        for j in range(indptr[i], indptr[i + 1]):
                            col_idx = indices[j]
                            w = weights[j]
                            B_row = B[col_idx]  # Load entire row once (cache-friendly)
                            for k in range(n_cols):
                                if B_row[k]:
                                    r[k] += w
                        posts[i] = r

            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def mm(weights, indices, indptr, B, posts):
                    n_cols = B.shape[1]
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = np.zeros(n_cols, dtype=posts.dtype)
                        for j in range(indptr[i], indptr[i + 1]):
                            col_idx = indices[j]
                            w = weights[j]
                            for k in range(n_cols):
                                if B[col_idx, k] != 0.:
                                    r[k] += w
                        posts[i] = r

    def kernel(weights, indices, indptr, B):
        if mask_events:
            B = event_active(B)
        output = numba_kernel(mm, outs=kwargs['outs'][0])(weights, indices, indptr, B)
        math_out = output[0] if isinstance(output, (tuple, list)) else output
        return (math_out,)

    return kernel


def _binary_csrmm_jax_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs,
):
    """Pure-JAX kernel for binary (event-driven) CSR matrix-matrix multiplication."""
    m, k = shape
    n = vector_info.shape[1]
    is_homo = (weight_info.size == 1)
    nse = kwargs['indices_info'].size
    out_dtype = kwargs['outs'][0].dtype
    weighted = kwargs.get('weighted', False)

    if transpose:
        def kernel(weights, indices, indptr, B):
            row_ids = jnp.repeat(
                jnp.arange(m, dtype=indptr.dtype),
                jnp.diff(indptr),
                total_repeat_length=nse,
            )
            w = weights[0] if is_homo else weights[:, None]
            contrib = _event_contrib(w, B[row_ids], dtype=out_dtype, weighted=weighted)  # [nse, n]
            math_out = jnp.zeros((k, n), dtype=out_dtype).at[indices].add(contrib)
            return (math_out,)
    else:
        def kernel(weights, indices, indptr, B):
            row_ids = jnp.repeat(
                jnp.arange(m, dtype=indptr.dtype),
                jnp.diff(indptr),
                total_repeat_length=nse,
            )
            w = weights[0] if is_homo else weights[:, None]
            contrib = _event_contrib(w, B[indices], dtype=out_dtype, weighted=weighted)  # [nse, n]
            math_out = jnp.zeros((m, n), dtype=out_dtype).at[row_ids].add(contrib)
            return (math_out,)

    return kernel


def _binary_csrmm_cusparse_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs,
):
    """cuSPARSE-backed binary CSR SpMM kernel via ``jax.experimental.sparse`` (GPU only).

    Activity varies per batch column, so one masked operand cannot serve
    every column. A homogeneous weight with binary events scales the exact
    active-event counts by selection. Otherwise cuSPARSE runs unchanged when
    every weight is finite (``0 * w`` is then exact) and any NaN/Inf weight
    takes the select-based ``jax_raw`` product via ``lax.cond``.
    """
    import jax.experimental.sparse as jsparse
    m, k = shape
    is_homo = (weight_info.size == 1)
    nse = kwargs['indices_info'].size
    out_dtype = kwargs['outs'][0].dtype
    weighted = kwargs.get('weighted', False)
    # Exact select-based fallback for non-finite weights.
    select_kernel = _binary_csrmm_jax_kernel(weight_info, vector_info, shape, transpose, **kwargs)

    def kernel(weights, indices, indptr, B):
        events = event_factor(B, dtype=out_dtype, weighted=weighted)
        cast_indices = indices.astype(indptr.dtype) if indices.dtype != indptr.dtype else indices
        if is_homo and not weighted:
            data = jnp.ones(nse, dtype=out_dtype)
            mat = jsparse.CSR((data, cast_indices, indptr), shape=(m, k))
            count = jsparse.csr_matmat(mat, events, transpose=transpose)
            return (_homo_count_output(count, weights[0], out_dtype),)

        def library_product(weights):
            data = jnp.broadcast_to(weights, (nse,)).astype(out_dtype)
            mat = jsparse.CSR((data, cast_indices, indptr), shape=(m, k))
            return jsparse.csr_matmat(mat, events, transpose=transpose)

        def select_product(weights):
            return select_kernel(weights, indices, indptr, B)[0]

        math_out = jax.lax.cond(
            jnp.all(jnp.isfinite(weights)), library_product, select_product, weights,
        )
        return (math_out,)

    return kernel


def _binary_csrmm_cuda_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs,
):
    _check_csr_cuda_structure_dtypes(kwargs['indices_info'], kwargs['indptr_info'])

    event_abi = cuda_event_abi(
        vector_info.dtype, weighted=kwargs.get('weighted', False), value_dtype=weight_info.dtype,
    )
    spk_suffix = event_abi.suffix

    # Weight dtype suffix
    wt_sfx = dtype_suffix(weight_info.dtype)

    # Homogeneous vs heterogeneous suffix
    is_homo = (weight_info.size == 1)
    homo_suffix = '_homo' if is_homo else '_hetero'

    if transpose:
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
        kernel_name = f'{_mod}.binary_csrmm_sraw_hybrid{homo_suffix}{wt_sfx}{spk_suffix}'

        def kernel(weights, indices, indptr, B):
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

    else:
        load_cuda_file(
            Path(__file__).parent.joinpath('binary_csrmm.cu'),
            name='csr_binary_csrmm',
        )
        out_info = kwargs['outs'][0]
        kernel_name = f'csr_binary_csrmm.binary_csrmm_nt_auto{homo_suffix}{wt_sfx}{spk_suffix}'

        def kernel(weights, indices, indptr, B):
            B = event_abi.prepare(B)
            math_out = jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, indptr, B)
            return (math_out,)

    return kernel


def _csrmm_jvp_data(data_dot, data, indices, indptr, B, *, shape, transpose, **kwargs):
    return binary_csrmm_p_call(
        data_dot,
        indices,
        indptr,
        B,
        shape=shape,
        transpose=transpose,
        backend=kwargs['backend'],
        weighted=kwargs.get('weighted', False),
        max_row_nnz=kwargs.get('max_row_nnz'),
    )


def _csrmm_jvp_B(B_dot, data, indices, indptr, B, *, shape, transpose, **kwargs):
    return [
        csrmm(data, indices, indptr, B_dot.astype(data.dtype), shape=shape, transpose=transpose,
              backend=_grad_backend(kwargs['backend'], csrmm_p))
    ]


def _csrmm_transpose_rule(ct, data, indices, indptr, B, *, shape, transpose, **kwargs):
    assert not ad.is_undefined_primal(indices)
    assert not ad.is_undefined_primal(indptr)

    ct = ct[0]
    if type(ct) is ad.Zero:
        if ad.is_undefined_primal(B):
            return data, indices, indptr, ad.Zero(B.aval)
        return ad.Zero(data.aval), indices, indptr, B
    if ad.is_undefined_primal(B):
        dB = csrmm(data, indices, indptr, ct, shape=shape, transpose=not transpose,
                   backend=_grad_backend(kwargs['backend'], csrmm_p))
        return data, indices, indptr, dB
    else:
        B = jnp.asarray(B)
        weighted = kwargs.get('weighted', False)
        row, col = _csr_to_coo(indices, indptr)
        d_data = _csrmm_entry_weight_ct(B, ct, row, col, transpose=transpose, weighted=weighted)
        if data.aval.shape[0] == 1:  # scalar
            d_data = jnp.sum(d_data).reshape(*data.aval.shape)
        d_data = d_data.astype(data.aval.dtype)
        return d_data, indices, indptr, B


def _csrmm_entry_weight_ct(B, ct, row, col, *, transpose: bool, weighted: bool) -> jax.Array:
    """Per-entry weight cotangent of a CSR event mat-mat product.

    A finite cotangent uses the fused SDDMM ``sum_b f(B[unit, b]) * ct[out, b]``
    (``0 * ct`` is exact there); a NaN/Inf cotangent takes the per-column
    select of :func:`_entry_weight_ct`, so it cannot leak through an event that
    is inactive in that batch column.

    Parameters
    ----------
    B : jax.Array
        ``(n_ev, n)`` events.
    ct : jax.Array
        ``(n_out, n)`` output cotangent.
    row, col : jax.Array
        COO coordinates of the stored entries.
    transpose : bool
        Product direction.
    weighted : bool
        Value-weighted events.

    Returns
    -------
    jax.Array
        ``(nse,)`` weight cotangent in ``ct.dtype``.
    """
    unit, out = (row, col) if transpose else (col, row)

    def sddmm(ct):
        B_factor = event_factor(B, dtype=ct.dtype, weighted=weighted)
        return sddmm_coo_indices(B_factor, ct.T, unit, out).data

    def select(ct):
        return _entry_weight_ct(B, ct, row, col, transpose=transpose, weighted=weighted)

    return jax.lax.cond(jnp.all(jnp.isfinite(ct)), sddmm, select, ct)


def _csrmm_batching(args, axes, **kwargs):
    axes = tuple(axes)
    if axes[:3] == (None, None, None) and axes[3] in (0, 1, 2):
        assert args[3].ndim == 3, 'Batching requires 3D input.'
        # Fold the batch into the event columns: ``B`` becomes ``(m, batch * n)``.
        if axes[3] == 0:
            batch_size, m, n = args[3].shape
            B = jnp.transpose(args[3], (1, 0, 2)).reshape(m, batch_size * n)
        else:
            m = args[3].shape[0]
            B = args[3].reshape(m, -1)
        r = binary_csrmm_p_call(
            args[0],
            args[1],
            args[2],
            B,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend'],
            weighted=kwargs.get('weighted', False),
            max_row_nnz=kwargs.get('max_row_nnz'),
        )[0]
        if axes[3] == 2:
            n, batch_size = args[3].shape[1:]
            return [r.reshape(r.shape[0], n, batch_size)], [2]
        batch_size = args[3].shape[axes[3]]
        return [r.reshape(r.shape[0], batch_size, -1)], [1]

    return general_batching_rule(binary_csrmm_p, args, axes, **kwargs)


def _binary_csrmm_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for homo in (True, False):
            for bool_event in (True, False):
                n_conn = max(1, int(n_post * prob))
                indptr = np.arange(n_pre + 1, dtype=np.int32) * n_conn
                indices = np.random.randint(0, n_post, (n_pre * n_conn,), dtype=np.int32)
                weights = jnp.ones(1, dtype=dtype) if homo else jnp.ones(n_pre * n_conn, dtype=dtype)
                b_rows = n_post if not transpose else n_pre
                if bool_event:
                    B = jnp.asarray(np.random.rand(b_rows, 10) > 0.5, dtype=jnp.bool_)
                else:
                    B = jnp.asarray(np.random.rand(b_rows, 10), dtype=dtype)
                indptr = jnp.asarray(indptr)
                name = (f"{'T' if transpose else 'NT'},"
                        f"{'homo' if homo else 'hetero'},"
                        f"{'bool' if bool_event else 'float'}")
                configs.append(
                    BenchmarkConfig(
                        name,
                        (weights, indices, indptr, B),
                        {'shape': (n_pre, n_post), 'transpose': transpose}
                    )
                )
    return configs


def binary_csrmm_p_call(
    weights,
    indices,
    indptr,
    B,
    *,
    shape: MatrixShape,
    transpose: bool,
    backend: Optional[str] = None,
    weighted: bool = False,
    max_row_nnz: Optional[int] = None,
):
    """
    Low-level primitive call for event-driven CSR matrix--matrix
    multiplication.

    Prepares inputs, validates shapes and dtypes, and dispatches the
    ``binary_csrmm_p`` XLA custom kernel to compute ``C = A @ B`` (or
    ``C = A.T @ B``), where ``A`` is a CSR matrix and ``B`` is a dense
    event matrix.

    Parameters
    ----------
    weights : jax.Array
        Non-zero weight values of the CSR matrix.  Shape ``(nse,)`` for
        heterogeneous weights, ``(1,)`` for a homogeneous weight, or a
        scalar (automatically promoted to shape ``(1,)``).
    indices : jax.Array
        Column indices of non-zero elements.  Shape ``(nse,)`` with dtype
        ``int32``, ``int64``, ``uint32``, or ``uint64``.
    indptr : jax.Array
        Row index pointer array.  Shape ``(shape[0] + 1,)`` and same dtype
        as ``indices``.
    B : jax.Array
        Dense event matrix.  Shape ``(shape[0], cols)`` when
        ``transpose=True`` or ``(shape[1], cols)`` when
        ``transpose=False``.  Dtype may be boolean or floating-point.
    shape : tuple of int
        Two-element tuple ``(m, k)`` giving the logical shape of the
        sparse matrix.
    transpose : bool
        If ``True``, transpose the sparse matrix before multiplication.
    backend : str or None, optional
        Compute backend to use.  Default is ``None`` (auto-select).
    weighted : bool, optional
        Multiply the weights by the event value instead of the activity
        mask (:class:`~brainevent.WeightedArray`).  Default is ``False``.
    max_row_nnz : int or None, optional
        Upper bound on the longest row of ``indptr``, used to size the
        ``cuda_raw`` hybrid task queue (see :func:`binary_csrmm`).

    Returns
    -------
    tuple of jax.Array
        A one-element tuple holding the result matrix, of shape
        ``(shape[1], cols)`` when ``transpose=True`` or ``(shape[0], cols)``
        when ``transpose=False``.

    Raises
    ------
    AssertionError
        If ``indices`` or ``indptr`` have a dtype other than ``int32``,
        ``int64``, ``uint32``, or ``uint64``.
    AssertionError
        If ``indices`` and ``indptr`` do not share the same dtype.
    AssertionError
        If ``indptr`` or ``indices`` is not 1-D.
    AssertionError
        If ``weights`` does not have a floating-point dtype.
    AssertionError
        If there is a shape mismatch between ``B`` and the sparse
        matrix ``shape`` (considering the ``transpose`` flag).

    See Also
    --------
    binary_csrmm : High-level wrapper with unit support.

    Notes
    -----
    Scalar ``weights`` (0-d arrays) are automatically promoted to
    shape ``(1,)`` to indicate a homogeneous weight across all
    connections.

    The computation performed is:

    ``C[i, l] = sum_{j in nz(i)} w[j] * e(B[j, l])``  (non-transposed)

    ``C[j, l] = sum_{i in nz_col(j)} w[i] * e(B[i, l])``  (transposed)

    where ``e(x)`` is ``1`` when ``x`` is ``True`` (boolean) or
    ``x != 0`` (numeric), and ``0`` otherwise.  ``w[j]`` is either
    ``weights[j]`` (heterogeneous) or ``weights[0]`` (homogeneous).

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._csr.binary import binary_csrmm_p_call
        >>> weights = jnp.array([1.0, 2.0, 3.0, 4.0])
        >>> indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        >>> B = jnp.array([[True, False],
        ...                [False, True],
        ...                [True, True]])
        >>> binary_csrmm_p_call(weights, indices, indptr, B, shape=(2, 3), transpose=False)[0]
        Array([[3., 2.],
               [4., 7.]], dtype=float32)
    """
    assert indptr.ndim == 1, "Indptr must be 1D."
    assert indices.ndim == 1, "Indices must be 1D."
    _check_csr_structure_dtypes(indices, indptr)
    if transpose:
        assert shape[0] == B.shape[0], "Shape mismatch for transpose operation."
    else:
        assert shape[1] == B.shape[0], f"Shape mismatch for non-transpose operation. {shape[1]} != {B.shape[0]}"
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'

    # Check if weights is a scalar. If so, convert it to a one-dimensional array.
    if jnp.ndim(weights) == 0:
        weights = jnp.asarray([weights])

    # Determine the output shape and data type based on whether the sparse matrix is transposed.
    out_info = (
        # If transpose is True, the output shape is (shape[1], B.shape[1]).
        jax.ShapeDtypeStruct([shape[1], B.shape[1]], weights.dtype)
        if transpose else
        # If transpose is False, the output shape is (shape[0], B.shape[1]).
        jax.ShapeDtypeStruct([shape[0], B.shape[1]], weights.dtype)
    )
    # Call the binary_csrmm_p custom operation to perform the matrix-matrix multiplication.
    return binary_csrmm_p(
        weights,
        indices,
        indptr,
        B,
        outs=[out_info],
        shape=shape,
        transpose=transpose,
        backend=backend,
        weighted=bool(weighted),
        # Provide shape and data type information for indices.
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        # Provide shape and data type information for indptr.
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        # Provide shape and data type information for weights.
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        # Provide shape and data type information for B.
        vector_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
        max_row_nnz=max_row_nnz,
        task_capacity=_task_capacity(indices, indptr, max_row_nnz),
    )


binary_csrmm_p = XLACustomKernel(
    'binary_csrmm',
    doc="""
Low-level XLA custom-kernel primitive for ``binary_csrmm``.

This ``XLACustomKernel`` instance dispatches the binary (event-driven) CSR sparse matrix-matrix multiplication
operation to registered backends (``numba``, ``pallas``),
using runtime shape/dtype metadata provided by the high-level wrapper.

Only entries of ``B`` that are ``True`` (boolean) or non-zero (numeric) are considered active events
and contribute to the output, enabling efficient event-driven sparse-dense products commonly used
in spiking neural networks.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``binary_csrmm_p.available_backends(platform)``,
and the default backend can be configured with ``binary_csrmm_p.set_default(platform, backend)``.

See Also
--------
binary_csrmm : High-level user-facing function wrapper.
"""
)
binary_csrmm_p.def_numba_kernel(_csrmm_numba_kernel)
binary_csrmm_p.def_cuda_raw_kernel(_binary_csrmm_cuda_kernel, asdefault=True)
binary_csrmm_p.def_kernel('jax_raw', 'cpu', _binary_csrmm_jax_kernel)
binary_csrmm_p.def_kernel('jax_raw', 'gpu', _binary_csrmm_jax_kernel)
binary_csrmm_p.def_kernel('jax_raw', 'tpu', _binary_csrmm_jax_kernel)
binary_csrmm_p.def_kernel('cusparse', 'gpu', _binary_csrmm_cusparse_kernel)
binary_csrmm_p.def_jvp_rule2(_csrmm_jvp_data, None, None, _csrmm_jvp_B)
binary_csrmm_p.def_transpose_rule(_csrmm_transpose_rule)
binary_csrmm_p.def_batching_rule(_csrmm_batching)
binary_csrmm_p.def_call(binary_csrmm_p_call)
binary_csrmm_p.def_tags('csr', 'binary')
binary_csrmm_p.def_benchmark_data(_binary_csrmm_benchmark_data)
