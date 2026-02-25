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

from pathlib import Path
from typing import Optional

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._misc import _csr_to_coo, generate_block_dim, namescope
from brainevent._op import numba_kernel, XLACustomKernel, general_batching_rule, \
    jaxinfo_to_warpinfo
from brainevent._op.benchmark import BenchmarkConfig
from brainevent._sddmm import sddmm_coo_indices
from brainevent._typing import Data, Indptr, Index, MatrixShape
from brainevent.config import get_numba_parallel
from brainevent.kernix import load_cuda_file
from .float import csrmv, csrmm

__all__ = [
    'binary_csrmv',
    'binary_csrmv_p',
    'binary_csrmm',
    'binary_csrmm_p',
]


@namescope(static_argnames=("shape", "transpose"))
def binary_csrmv(
    data: Data,
    indices: Index,
    indptr: Indptr,
    v: Data,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    backend: Optional[str] = None,
) -> Data:
    """
    Product of a CSR sparse matrix and a dense vector using event-driven
    (binary) computation.

    Computes ``y = A @ v`` (or ``y = A.T @ v`` when ``transpose=True``)
    where ``A`` is stored in Compressed Sparse Row format and ``v`` is
    treated as a binary event vector.  Elements of ``v`` that are ``True``
    (boolean) or positive (float) are considered *active events*; only
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
        (events indicated by ``True``) or floating-point (events indicated
        by values ``> 0``).
    shape : tuple of int
        Two-element tuple ``(m, k)`` giving the logical shape of the
        sparse matrix ``A``.
    transpose : bool, optional
        If ``True``, the sparse matrix is transposed before multiplication,
        i.e. compute ``A.T @ v``.  Default is ``False``.
    backend : str or None, optional
        Compute backend to use.  One of ``'numba'``,
        ``'pallas'``, or ``None`` (auto-select).  Default is ``None``.

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
    ``<= 0``).  This yields significant speed-ups when the event vector is
    sparse.

    Mathematically, the non-transposed operation computes:

    ``y[i] = sum_{j in nz(i)} A[i, j] * e(v[j])``

    where ``nz(i)`` denotes the set of column indices with non-zero
    entries in row ``i``, and ``e(v[j])`` is the event indicator:

    ``e(v[j]) = 1  if v[j] is True (bool) or v[j] > 0 (float)``
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
        >>> from brainevent._csr.binary import binary_csrmv
        >>> data = jnp.array([0.5])           # homogeneous weight
        >>> indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        >>> v = jnp.array([True, False, True])  # binary event vector
        >>> binary_csrmv(data, indices, indptr, v, shape=(2, 3))
    """
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
    )[0]
    return u.maybe_decimal(res * (unitd * unitv))


@namescope(static_argnames=("shape", "transpose"))
def binary_csrmm(
    data: Data,
    indices: Index,
    indptr: Indptr,
    B: Data,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    backend: Optional[str] = None,
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
        Dtype may be boolean or floating-point.
    shape : tuple of int
        Two-element tuple ``(m, k)`` giving the logical shape of the
        sparse matrix ``A``.
    transpose : bool, optional
        If ``True``, transpose ``A`` before multiplication.  Default is
        ``False``.
    backend : str or None, optional
        Compute backend.  One of ``'numba'``, ``'pallas'``, or
        ``None`` (auto-select).  Default is ``None``.

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
    (``False`` or ``<= 0``) are skipped.  Custom JVP and transpose rules
    are provided for automatic differentiation.

    Mathematically, the non-transposed operation computes:

    ``C[i, l] = sum_{j in nz(i)} A[i, j] * e(B[j, l])``

    where ``nz(i)`` denotes the set of column indices with non-zero
    entries in row ``i`` of the CSR matrix, and ``e(B[j, l])`` is the
    event indicator:

    ``e(B[j, l]) = 1  if B[j, l] is True (bool) or B[j, l] > 0 (float)``
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
        >>> from brainevent._csr.binary import binary_csrmm
        >>> data = jnp.array([1.0, 2.0, 3.0, 4.0])
        >>> indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        >>> B = jnp.array([[True, False],
        ...                [False, True],
        ...                [True, True]])
        >>> binary_csrmm(data, indices, indptr, B, shape=(2, 3))
    """
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
    )[0]
    return u.maybe_decimal(res * (unitd * unitb))


def _csrmv_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba
    if weight_info.size == 1:
        if transpose:
            if vector_info.dtype == jnp.bool_:
                # Cannot parallelize due to race condition on posts[indices[j]]
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, v, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(v.shape[0]):
                        if v[i]:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += w

            else:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, v, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(v.shape[0]):
                        if v[i] > 0.:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += w

        else:
            if vector_info.dtype == jnp.bool_:
                # Can parallelize by row
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mv(weights, indices, indptr, v, posts):
                    w = weights[0]
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = 0.0
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]]:
                                r += w
                        posts[i] = r

            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mv(weights, indices, indptr, v, posts):
                    w = weights[0]
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = 0.0
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]] > 0.:
                                r += w
                        posts[i] = r

    else:
        if transpose:
            if vector_info.dtype == jnp.bool_:
                # Cannot parallelize due to race condition
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, v, posts):
                    posts[:] = 0.
                    for i in range(v.shape[0]):
                        if v[i]:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += weights[j]

            else:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, v, posts):
                    posts[:] = 0.
                    for i in range(v.shape[0]):
                        if v[i] > 0.:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += weights[j]

        else:
            if vector_info.dtype == jnp.bool_:
                # Can parallelize by row
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mv(weights, indices, indptr, v, posts):
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = 0.0
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]]:
                                r += weights[j]
                        posts[i] = r

            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mv(weights, indices, indptr, v, posts):
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = 0.0
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]] > 0.:
                                r += weights[j]
                        posts[i] = r

    def kernel(weights, indices, indptr, vector):
        return numba_kernel(mv, outs=kwargs['outs'])(weights, indices, indptr, vector)

    return kernel


def _csrmv_warp_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    indptr_info: jax.ShapeDtypeStruct,
    transpose: bool,
    shape: MatrixShape,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    indices_warp_info = jaxinfo_to_warpinfo(indices_info)
    indptr_warp_info = jaxinfo_to_warpinfo(indptr_info)
    spike_warp_info = jaxinfo_to_warpinfo(vector_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if transpose:
        if weight_info.size == 1:
            if vector_info.dtype == jnp.bool_:
                @warp.kernel
                def mv(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    v: spike_warp_info,
                    posts: out_warp_info,
                ):
                    i = warp.tid()
                    w = weights[0]
                    if v[i]:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j]] += w

            else:
                @warp.kernel
                def mv(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    v: spike_warp_info,
                    posts: out_warp_info,
                ):
                    i = warp.tid()
                    w = weights[0]
                    if v[i] > 0.:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j]] += w


        else:
            if vector_info.dtype == jnp.bool_:
                @warp.kernel
                def mv(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    v: spike_warp_info,
                    posts: out_warp_info,
                ):
                    i = warp.tid()
                    if v[i]:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j]] += weights[j]

            else:
                @warp.kernel
                def mv(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    v: spike_warp_info,
                    posts: out_warp_info,
                ):
                    i = warp.tid()
                    if v[i] > 0.:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j]] += weights[j]

        def kernel(weights, indices, indptr, v):
            out_info = (
                jax.ShapeDtypeStruct([shape[1]], weights.dtype)
                if transpose else
                jax.ShapeDtypeStruct([shape[0]], weights.dtype)
            )
            dim = vector_info.shape[0] if transpose else indptr_info.shape[0] - 1
            fn = jax_kernel(mv, launch_dims=[dim], num_outputs=1, in_out_argnames=['posts'])
            return fn(weights, indices, indptr, v, jnp.zeros(out_info.shape, out_info.dtype))


    else:
        if weight_info.size == 1:
            if vector_info.dtype == jnp.bool_:
                @warp.kernel
                def mv(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    v: spike_warp_info,
                    posts: out_warp_info,
                ):
                    i = warp.tid()
                    w = weights[0]
                    r = weights.dtype(0.)
                    for j in range(indptr[i], indptr[i + 1]):
                        if v[indices[j]]:
                            r += w
                    posts[i] = r

            else:
                @warp.kernel
                def mv(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    v: spike_warp_info,
                    posts: out_warp_info,
                ):
                    i = warp.tid()
                    w = weights[0]
                    r = weights.dtype(0.)
                    for j in range(indptr[i], indptr[i + 1]):
                        if v[indices[j]] > 0.:
                            r += w
                    posts[i] = r

        else:
            if vector_info.dtype == jnp.bool_:
                @warp.kernel
                def mv(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    v: spike_warp_info,
                    posts: out_warp_info,
                ):
                    i = warp.tid()
                    r = weights.dtype(0.)
                    for j in range(indptr[i], indptr[i + 1]):
                        if v[indices[j]]:
                            r += weights[j]
                    posts[i] = r

            else:
                @warp.kernel
                def mv(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    v: spike_warp_info,
                    posts: out_warp_info,
                ):
                    i = warp.tid()
                    r = weights.dtype(0.)
                    for j in range(indptr[i], indptr[i + 1]):
                        if v[indices[j]] > 0.:
                            r += weights[j]
                    posts[i] = r

        def kernel(weights, indices, indptr, v):
            out_info = (
                jax.ShapeDtypeStruct([shape[1]], weights.dtype)
                if transpose else
                jax.ShapeDtypeStruct([shape[0]], weights.dtype)
            )
            dim = vector_info.shape[0] if transpose else indptr_info.shape[0] - 1
            fn = jax_kernel(mv, launch_dims=[dim], num_outputs=1, output_dims={'posts': out_info.shape})
            return fn(weights, indices, indptr, v)

    return kernel


def _csrmv_pallas_kernel(
    weight_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import load

    m, k = shape
    block_dim = generate_block_dim(pl.cdiv(indices_info.size, shape[1] if transpose else shape[0]))
    block_dim = block_dim // 2
    block_dim = 32 if block_dim < 32 else block_dim

    if weight_info.size == 1:
        # csr @ B (homogeneous weights)
        #
        # csr: [m, k]
        # B: [k]
        # result: [m]
        #
        def mm(
            data_ref,  # [1]
            indices_ref,  # [nse]
            indptr_ref,  # [m + 1]
            vector_ref,  # [k]
            posts_ref,  # [m]
        ):
            i_row = pl.program_id(0)
            row_start = indptr_ref[i_row]
            row_end = indptr_ref[i_row + 1]
            row_nnz = row_end - row_start
            num_blocks = (row_nnz + block_dim - 1) // block_dim
            val_A = data_ref[0]

            def loop_fn(index, sum_):
                offset = row_start + index * block_dim
                mask = offset + jnp.arange(block_dim) < row_end
                cols = load(indices_ref.at[pl.ds(offset, block_dim)], mask=mask, other=0)
                events = vector_ref[cols]
                if vector_ref.dtype == jnp.bool_:
                    events = jnp.asarray(events & mask, dtype=posts_ref.dtype)
                else:
                    events = jnp.where(
                        (events > 0.) & mask,
                        jnp.ones(events.shape, dtype=posts_ref.dtype),
                        0.
                    )
                sum_ += val_A * jnp.sum(events)
                return sum_

            i_row_sum = jax.lax.fori_loop(
                0,
                num_blocks,
                loop_fn,
                jnp.asarray(0., dtype=posts_ref.dtype)
            )
            posts_ref[i_row] = i_row_sum

    else:
        # csr @ B (heterogeneous weights)
        #
        # csr: [m, k]
        # B: [k]
        # result: [m]
        #
        def mm(
            data_ref,  # [nse]
            indices_ref,  # [nse]
            indptr_ref,  # [m + 1]
            vector_ref,  # [k]
            posts_ref,  # [m]
        ):
            i_row = pl.program_id(0)
            row_start = indptr_ref[i_row]
            row_end = indptr_ref[i_row + 1]
            row_nnz = row_end - row_start
            num_blocks = (row_nnz + block_dim - 1) // block_dim

            def loop_fn(index, sum_):
                offset = row_start + index * block_dim
                mask = offset + jnp.arange(block_dim) < row_end
                cols = load(indices_ref.at[pl.ds(offset, block_dim)], mask=mask, other=0)
                val_A = load(data_ref.at[pl.ds(offset, block_dim)], mask=mask, other=0.0)
                events = vector_ref[cols]
                if vector_ref.dtype == jnp.bool_:
                    events = jnp.asarray(events & mask, dtype=posts_ref.dtype)
                else:
                    events = jnp.where((events > 0.) & mask, jnp.ones(events.shape, dtype=posts_ref.dtype), 0.)
                sum_ += jnp.sum(val_A * events)
                return sum_

            i_row_sum = jax.lax.fori_loop(
                0,
                num_blocks,
                loop_fn,
                jnp.asarray(0., dtype=posts_ref.dtype)
            )
            posts_ref[i_row] = i_row_sum

    def kernel(data, indices, indptr, vector):
        fn = pl.pallas_call(mm, grid=(m,), out_shape=kwargs['outs'], backend='triton')
        return fn(data, indices, indptr, vector)

    return kernel


def _csrmv_pallas_gpu_kernel(
    weight_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add

    m, k = shape
    block_dim = generate_block_dim(pl.cdiv(indices_info.size, shape[1] if transpose else shape[0]))
    block_dim = block_dim // 2
    block_dim = 32 if block_dim < 32 else block_dim

    if transpose:
        if weight_info.size == 1:
            # csr.T @ B (homogeneous weights)
            #
            # csr: [m, k]
            # B: [m]
            # result: [k]
            #
            def mm(
                data_ref,  # [1]
                indices_ref,  # [nse]
                indptr_ref,  # [m + 1]
                vector_ref,  # [m]
                _,  # [k]
                posts_ref,  # [k]
            ):
                i_row = pl.program_id(0)
                row_start = indptr_ref[i_row]
                row_end = indptr_ref[i_row + 1]
                row_nnz = row_end - row_start
                num_blocks = (row_nnz + block_dim - 1) // block_dim
                event = vector_ref[i_row]
                data = jnp.ones((block_dim,), dtype=posts_ref.dtype) * data_ref[0]

                @pl.when(event if vector_ref.dtype == jnp.bool_ else event > 0.)
                def event_processing():
                    def loop_fn(index, _):
                        offset = row_start + index * block_dim
                        mask = offset + jnp.arange(block_dim) < row_end
                        cols = indices_ref[pl.ds(offset, block_dim)]
                        atomic_add(posts_ref, cols, data, mask=mask)

                    jax.lax.fori_loop(0, num_blocks, loop_fn, None)

        else:
            # csr.T @ B (heterogeneous weights)
            #
            # csr: [m, k]
            # B: [m]
            # result: [k]
            #
            def mm(
                data_ref,  # [nse]
                indices_ref,  # [nse]
                indptr_ref,  # [m + 1]
                vector_ref,  # [m]
                _,  # [k]
                posts_ref,  # [k]
            ):
                i_row = pl.program_id(0)
                row_start = indptr_ref[i_row]
                row_end = indptr_ref[i_row + 1]
                row_nnz = row_end - row_start
                num_blocks = (row_nnz + block_dim - 1) // block_dim
                event = vector_ref[i_row]

                @pl.when(event if vector_ref.dtype == jnp.bool_ else event > 0.)
                def event_processing():
                    def loop_fn(index, _):
                        offset = row_start + index * block_dim
                        mask = offset + jnp.arange(block_dim) < row_end
                        cols = indices_ref[pl.ds(offset, block_dim)]
                        weights = data_ref[pl.ds(offset, block_dim)]
                        data = jnp.asarray(weights, dtype=posts_ref.dtype)
                        atomic_add(posts_ref, cols, data, mask=mask)

                    jax.lax.fori_loop(0, num_blocks, loop_fn, None)

        def kernel(data, indices, indptr, vector):
            fn = pl.pallas_call(mm, grid=(m,), input_output_aliases={4: 0}, out_shape=kwargs['outs'], backend='triton')
            out = kwargs['outs'][0]
            return fn(data, indices, indptr, vector, jnp.zeros(out.shape, dtype=out.dtype))

        return kernel
    else:
        return _csrmv_pallas_kernel(
            weight_info=weight_info, indices_info=indices_info, shape=shape, transpose=transpose, **kwargs
        )


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
    is_bool = (vector_info.dtype == jnp.bool_)
    nse = kwargs['indices_info'].size
    out_dtype = kwargs['outs'][0].dtype

    if transpose:
        def kernel(weights, indices, indptr, vector):
            row_ids = jnp.repeat(
                jnp.arange(m, dtype=indptr.dtype),
                jnp.diff(indptr),
                total_repeat_length=nse,
            )
            v_row = vector[row_ids]
            events = v_row.astype(out_dtype) if is_bool else (v_row > 0.).astype(out_dtype)
            w = weights[0] if is_homo else weights
            return (jnp.zeros(k, dtype=out_dtype).at[indices].add(w * events),)
    else:
        def kernel(weights, indices, indptr, vector):
            row_ids = jnp.repeat(
                jnp.arange(m, dtype=indptr.dtype),
                jnp.diff(indptr),
                total_repeat_length=nse,
            )
            v_col = vector[indices]
            events = v_col.astype(out_dtype) if is_bool else (v_col > 0.).astype(out_dtype)
            w = weights[0] if is_homo else weights
            return (jnp.zeros(m, dtype=out_dtype).at[row_ids].add(w * events),)

    return kernel


def _binary_csrmv_cusparse_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs,
):
    """cuSPARSE-backed kernel for binary CSR SpMV via jax.experimental.sparse (GPU only)."""
    import jax.experimental.sparse as jsparse
    m, k = shape
    is_homo = (weight_info.size == 1)
    is_bool = (vector_info.dtype == jnp.bool_)
    nse = kwargs['indices_info'].size
    out_dtype = kwargs['outs'][0].dtype

    if transpose:
        if is_homo:
            def kernel(weights, indices, indptr, vector):
                events = vector.astype(out_dtype) if is_bool else (vector > 0.).astype(out_dtype)
                ones = jnp.ones(nse, dtype=out_dtype)
                row, col = _csr_to_coo(indices, indptr)
                mat = jsparse.BCOO((ones, jnp.stack([row, col], axis=1)), shape=(m, k))
                return ((mat.T @ events) * weights[0].astype(out_dtype),)
        else:
            def kernel(weights, indices, indptr, vector):
                events = vector.astype(out_dtype) if is_bool else (vector > 0.).astype(out_dtype)
                row, col = _csr_to_coo(indices, indptr)
                mat = jsparse.BCOO((weights.astype(out_dtype), jnp.stack([row, col], axis=1)), shape=(m, k))
                return (mat.T @ events,)
    else:
        if is_homo:
            def kernel(weights, indices, indptr, vector):
                events = vector.astype(out_dtype) if is_bool else (vector > 0.).astype(out_dtype)
                ones = jnp.ones(nse, dtype=out_dtype)
                mat = jsparse.BCSR((ones, indices, indptr), shape=(m, k))
                return ((mat @ events) * weights[0].astype(out_dtype),)
        else:
            def kernel(weights, indices, indptr, vector):
                events = vector.astype(out_dtype) if is_bool else (vector > 0.).astype(out_dtype)
                mat = jsparse.BCSR((weights.astype(out_dtype), indices, indptr), shape=(m, k))
                return (mat @ events,)
    return kernel


def _binary_csrmv_cuda_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs,
):
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_csrmv.cu'),
        name='csr_binary_csrmv',
    )

    out_info = kwargs['outs']

    # Determine if weights are homogeneous or heterogeneous
    is_homo = (weight_info.size == 1)
    homo_suffix = '_homo' if is_homo else '_hetero'

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

    if transpose:
        kernel_name = f'csr_binary_csrmv.binary_csrmv_t_warp{homo_suffix}{wt_sfx}{spk_suffix}'
    else:
        kernel_name = f'csr_binary_csrmv.binary_csrmv_nt_auto{homo_suffix}{wt_sfx}{spk_suffix}'

    def kernel(weights, indices, indptr, vector):
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, indptr, vector)

    return kernel


def _csrmv_jvp_v(v_dot, data, indices, indptr, v, *, shape, transpose, **kwargs):
    return [csrmv(data, indices, indptr, v_dot, shape=shape, transpose=transpose, backend=kwargs['backend'])]


def _csrmv_jvp_weights(data_dot, data, indices, indptr, v, *, shape, transpose, **kwargs):
    backend = kwargs['backend']
    return binary_csrmv_p_call(data_dot, indices, indptr, v, shape=shape, transpose=transpose, backend=backend)


def _csrmv_transpose_rule(ct, data, indices, indptr, events, *, shape, transpose, **kwargs):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")

    ct = ct[0]

    if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
        raise ValueError("Cannot transpose with respect to sparse indices.")
    if ad.is_undefined_primal(events):
        if type(ct) is ad.Zero:
            ct_events = ad.Zero(events)
        else:
            ct_events = csrmv(data, indices, indptr, ct,
                              shape=shape, transpose=not transpose, backend=kwargs['backend'])
        return data, indices, indptr, ct_events
    else:
        if type(ct) is ad.Zero:
            ct_values = ad.Zero(data)
        else:
            if data.aval.shape[0] == 1:  # scalar
                ct_values = binary_csrmv_p_call(
                    jnp.ones(1, dtype=data.aval.dtype),
                    indices,
                    indptr,
                    events,
                    shape=shape,
                    transpose=transpose,
                    backend=kwargs['backend'],
                )[0]
                ct_values = jnp.inner(ct, ct_values).reshape(*data.aval.shape)
            else:  # heterogeneous values
                row, col = _csr_to_coo(indices, indptr)
                ct_values = events[row] * ct[col] if transpose else events[col] * ct[row]
        return ct_values, indices, indptr, events


def _csrmv_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = binary_csrmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3].T,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend'],
        )
        return r, [1]

    elif tuple(axes) == (None, None, None, 1):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = binary_csrmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend'],
        )
        return r, [1]

    else:
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
                            name = (f"{n}x{n},p={p_pct}%,"
                                    f"{'T' if transpose else 'NT'},"
                                    f"{'homo' if homo else 'hetero'},"
                                    f"{event_type}")
                            yield BenchmarkConfig(
                                name,
                                jax.block_until_ready(
                                    (weights, jnp.asarray(indices), jnp.asarray(indptr), vector)
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
                        name = (f"{n_pre}x{n_post},p=10%,"
                                f"{'T' if transpose else 'NT'},"
                                f"{'homo' if homo else 'hetero'},"
                                f"{event_type}")
                        yield BenchmarkConfig(
                            name,
                            jax.block_until_ready(
                                (weights, jnp.asarray(indices), jnp.asarray(indptr), vector)
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

    Returns
    -------
    list of jax.Array
        A single-element list containing the result vector.  Shape
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
    ``x > 0`` (float), and ``0`` otherwise.  ``w[j]`` is either
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
        >>> result = binary_csrmv_p_call(
        ...     weights, indices, indptr, vector,
        ...     shape=(2, 3), transpose=False)
    """
    assert indices.dtype in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64], "Indices must be int32 or int64."
    assert indptr.dtype in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64], "Indptr must be int32 or int64."
    assert indptr.ndim == 1, "Indptr must be 1D."
    assert indices.ndim == 1, "Indices must be 1D."
    assert indptr.dtype == indices.dtype, "Indices and indptr must have the same dtype."
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
        # Initialize a zero vector with the output shape and data type.
        outs=[out_info],
        shape=shape,
        transpose=transpose,
        backend=backend,
        # Provide shape and data type information for indices.
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        # Provide shape and data type information for indptr.
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        # Provide shape and data type information for weights.
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        # Provide shape and data type information for v.
        vector_info=jax.ShapeDtypeStruct(vector.shape, vector.dtype),
    )


binary_csrmv_p = XLACustomKernel(
    'binary_csrmv',
    doc="""
Low-level XLA custom-kernel primitive for ``binary_csrmv``.

This ``XLACustomKernel`` instance dispatches the binary (event-driven) CSR sparse matrix-vector multiplication
operation to registered backends (``numba``, ``pallas``, ``tvmffi``),
using runtime shape/dtype metadata provided by the high-level wrapper.

Only entries of ``v`` that are ``True`` (boolean) or positive (float) are considered active events
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
binary_csrmv_p.def_warp_kernel(_csrmv_warp_kernel)
binary_csrmv_p.def_pallas_kernel('gpu', _csrmv_pallas_gpu_kernel)
binary_csrmv_p.def_cuda_kernel(_binary_csrmv_cuda_kernel)
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

    if weight_info.size == 1:
        if transpose:
            #
            # csr.T @ B
            #
            # [k, m] @ [k, n]
            #
            if vector_info.dtype == jnp.bool_:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mm(weights, indices, indptr, B, posts):
                    w = weights[0]
                    posts[:] = 0.
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            if B[i, k]:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += w

            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mm(weights, indices, indptr, B, posts):
                    B = B > 0.
                    w = weights[0]
                    posts[:] = 0.
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            if B[i, k]:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += w

        else:
            # csr @ B
            if vector_info.dtype == jnp.bool_:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
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
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mm(weights, indices, indptr, B, posts):
                    w = weights[0]
                    B = B > 0.
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

            if vector_info.dtype == jnp.bool_:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mm(weights, indices, indptr, B, posts):
                    posts[:] = 0.
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            if B[i, k]:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += weights[j]

            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mm(weights, indices, indptr, B, posts):
                    B = B > 0.
                    posts[:] = 0.
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            if B[i, k]:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += weights[j]

        else:
            # csr @ B
            # Fixed: Changed range to prange for parallelization

            if vector_info.dtype == jnp.bool_:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
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
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mm(weights, indices, indptr, B, posts):
                    n_cols = B.shape[1]
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = np.zeros(n_cols, dtype=posts.dtype)
                        for j in range(indptr[i], indptr[i + 1]):
                            col_idx = indices[j]
                            w = weights[j]
                            for k in range(n_cols):
                                if B[col_idx, k] > 0.:
                                    r[k] += w
                        posts[i] = r

    def kernel(weights, indices, indptr, B):
        return numba_kernel(mm, kwargs['outs'])(weights, indices, indptr, B)

    return kernel


def _csrmm_warp_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    indptr_info: jax.ShapeDtypeStruct,
    transpose: bool,
    shape: MatrixShape,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    indices_warp_info = jaxinfo_to_warpinfo(indices_info)
    indptr_warp_info = jaxinfo_to_warpinfo(indptr_info)
    spike_warp_info = jaxinfo_to_warpinfo(vector_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if transpose:
        # csr.T @ B
        if weight_info.size == 1:
            if vector_info.dtype == jnp.bool_:
                @warp.kernel
                def mm(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    B: spike_warp_info,
                    posts: out_warp_info,
                ):
                    k, i = warp.tid()
                    w = weights[0]
                    if B[i, k]:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j], k] += w

            else:
                @warp.kernel
                def mm(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    B: spike_warp_info,
                    posts: out_warp_info,
                ):
                    k, i = warp.tid()
                    w = weights[0]
                    if B[i, k] > 0.:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j], k] += w

        else:
            if vector_info.dtype == jnp.bool_:
                @warp.kernel
                def mm(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    B: spike_warp_info,
                    posts: out_warp_info,
                ):
                    k, i = warp.tid()
                    if B[i, k]:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j], k] += weights[j]

            else:
                @warp.kernel
                def mm(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    B: spike_warp_info,
                    posts: out_warp_info,
                ):
                    k, i = warp.tid()
                    if B[i, k] > 0.:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j], k] += weights[j]

        def kernel(weights, indices, indptr, B):
            n = vector_info.shape[1]
            out_info = jax.ShapeDtypeStruct([shape[1], n], weights.dtype)
            dim = tuple(reversed(vector_info.shape))  # (n, m)
            fn = jax_kernel(mm, launch_dims=dim, num_outputs=1, in_out_argnames=['posts'])
            return fn(weights, indices, indptr, B, jnp.zeros(out_info.shape, out_info.dtype))

    else:
        # csr @ B
        if weight_info.size == 1:
            if vector_info.dtype == jnp.bool_:
                @warp.kernel
                def mm(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    B: spike_warp_info,
                    posts: out_warp_info,
                ):
                    k, i = warp.tid()
                    w = weights[0]
                    r = weights.dtype(0.)
                    for j in range(indptr[i], indptr[i + 1]):
                        index = indices[j]
                        if B[index, k]:
                            r += w
                    posts[i, k] = r

            else:
                @warp.kernel
                def mm(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    B: spike_warp_info,
                    posts: out_warp_info,
                ):
                    k, i = warp.tid()
                    w = weights[0]
                    r = weights.dtype(0.)
                    for j in range(indptr[i], indptr[i + 1]):
                        index = indices[j]
                        if B[index, k] > 0.:
                            r += w
                    posts[i, k] = r

        else:
            # csr @ B

            if vector_info.dtype == jnp.bool_:
                @warp.kernel
                def mm(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    B: spike_warp_info,
                    posts: out_warp_info,
                ):
                    k, i = warp.tid()
                    r = weights.dtype(0.)
                    for j in range(indptr[i], indptr[i + 1]):
                        index = indices[j]
                        if B[index, k]:
                            r += weights[j]
                    posts[i, k] = r

            else:
                @warp.kernel
                def mm(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    B: spike_warp_info,
                    posts: out_warp_info,
                ):
                    k, i = warp.tid()
                    r = weights.dtype(0.)
                    for j in range(indptr[i], indptr[i + 1]):
                        index = indices[j]
                        if B[index, k] > 0.:
                            r += weights[j]
                    posts[i, k] = r

        def kernel(weights, indices, indptr, B):
            n = vector_info.shape[1]
            out_info = jax.ShapeDtypeStruct([shape[0], n], weights.dtype)
            dim = (vector_info.shape[1], indptr_info.shape[0] - 1)
            fn = jax_kernel(mm, launch_dims=dim, num_outputs=1, output_dims={'posts': out_info.shape})
            return fn(weights, indices, indptr, B)

    return kernel


def _csrmm_pallas_kernel(
    weight_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import load, store

    m, k = shape
    n = vector_info.shape[1]

    # Block dimension for output columns
    block_dim_n = generate_block_dim(n, 512)

    # Block dimension for non-zeros (csrmv pattern)
    block_dim = generate_block_dim(pl.cdiv(indices_info.size, m))
    block_dim = block_dim // 2
    block_dim = 32 if block_dim < 32 else block_dim

    if weight_info.size == 1:
        #
        # Gustavson algorithm: Sparse matrix–matrix multiplication is performed in a row-wise fashion.
        #
        # Each nonzero value in a row is multiplied by the nonzero values corresponding to the column index.
        # These values are summed and stored in a temporary row buffer based on their column indices.

        # csr @ B (homogeneous weights)
        #
        # csr: [m, k]
        # B: [k, n]
        # result: [m, n]
        #
        def mm(
            data_ref,  # [1]
            indices_ref,  # [nse]
            indptr_ref,  # [m + 1]
            B_ref,  # [k, n]
            posts_ref,  # [m, n]
        ):
            # 1. Grid Info
            i_row = pl.program_id(0)
            i_n = pl.program_id(1)

            # Grid Guard
            num_rows = indptr_ref.shape[0] - 1

            def _body():
                # 2. Column Blocking Setup
                i_col_start = i_n * block_dim_n
                col_mask = (i_col_start + jnp.arange(block_dim_n)) < B_ref.shape[1]

                # 3. Row Metadata
                row_start = indptr_ref[i_row]
                row_end = indptr_ref[i_row + 1]
                row_nnz = row_end - row_start
                num_blocks = (row_nnz + block_dim - 1) // block_dim
                val_A = data_ref[0]

                limit_k = B_ref.shape[0] - 1

                def loop_fn(index, sum_):
                    offset = row_start + index * block_dim
                    nnz_mask = offset + jnp.arange(block_dim) < row_end

                    cols = load(indices_ref.at[pl.ds(offset, block_dim)], mask=nnz_mask, other=0)

                    safe_cols = jnp.minimum(cols, limit_k)

                    events = B_ref[safe_cols, pl.ds(i_col_start, block_dim_n)]
                    events = jnp.asarray(events, dtype=posts_ref.dtype)

                    contribution = jnp.sum(jnp.where(nnz_mask[:, None], events, 0.), axis=0)
                    sum_ += val_A * contribution
                    return sum_

                i_row_sum = jax.lax.fori_loop(
                    0, num_blocks, loop_fn,
                    jnp.zeros([block_dim_n], dtype=posts_ref.dtype)
                )

                store(
                    posts_ref.at[i_row, pl.ds(i_col_start, block_dim_n)],
                    i_row_sum,
                    mask=col_mask
                )

            # guard
            jax.lax.cond(i_row < num_rows, _body, lambda: None)

    else:
        #
        # Gustavson algorithm: Sparse matrix–matrix multiplication is performed in a row-wise fashion.
        #
        # Each nonzero value in a row is multiplied by the nonzero values corresponding to the column index.
        # These values are summed and stored in a temporary row buffer based on their column indices.

        # csr @ B (heterogeneous weights)
        #
        # csr: [m, k]
        # B: [k, n]
        # result: [m, n]
        #
        def mm(
            data_ref,  # [nse]
            indices_ref,  # [nse]
            indptr_ref,  # [m + 1]
            B_ref,  # [k, n]
            posts_ref,  # [m, n]
        ):
            i_row = pl.program_id(0)
            num_rows = indptr_ref.shape[0] - 1

            def _body():
                i_n = pl.program_id(1)
                i_col_start = i_n * block_dim_n

                col_mask = (i_col_start + jnp.arange(block_dim_n)) < B_ref.shape[1]

                row_start = indptr_ref[i_row]
                row_end = indptr_ref[i_row + 1]
                row_nnz = row_end - row_start
                num_blocks = (row_nnz + block_dim - 1) // block_dim

                num_B_rows = B_ref.shape[0]

                def loop_fn(index, sum_):
                    offset = row_start + index * block_dim

                    nnz_mask = offset + jnp.arange(block_dim) < row_end

                    cols = load(indices_ref.at[pl.ds(offset, block_dim)], mask=nnz_mask, other=0)
                    val_A = load(data_ref.at[pl.ds(offset, block_dim)], mask=nnz_mask, other=0.0)

                    valid_cols = cols < num_B_rows
                    safe_cols = jnp.minimum(cols, num_B_rows - 1)

                    mask_B_dim0 = nnz_mask & valid_cols
                    mask_B = mask_B_dim0[:, None] & col_mask[None, :]

                    events = load(B_ref.at[safe_cols, pl.ds(i_col_start, block_dim_n)], mask=mask_B, other=0.0)

                    weighted = val_A[:, None] * events

                    contribution = jnp.sum(weighted, axis=0)

                    sum_ += contribution
                    return sum_

                i_row_sum = jax.lax.fori_loop(
                    0, num_blocks, loop_fn,
                    jnp.zeros([block_dim_n], dtype=posts_ref.dtype)
                )

                store(posts_ref.at[i_row, pl.ds(i_col_start, block_dim_n)], i_row_sum, mask=col_mask)

            jax.lax.cond(i_row < num_rows, _body, lambda: None)

    def kernel(data, indices, indptr, B):
        fn = pl.pallas_call(
            mm,
            grid=(m, pl.cdiv(n, block_dim_n)),
            out_shape=kwargs['outs'],
            backend='triton',
        )
        return fn(data, indices, indptr, B)

    return kernel


def _csrmm_pallas_gpu_kernel(
    weight_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import load, atomic_add

    m, k = shape
    n = vector_info.shape[1]

    # Block dimension for output columns
    block_dim_n = generate_block_dim(n, 512)

    # Block dimension for non-zeros (csrmv pattern)
    block_dim = generate_block_dim(pl.cdiv(indices_info.size, m))
    block_dim = block_dim // 2
    block_dim = 32 if block_dim < 32 else block_dim

    if transpose:
        if weight_info.size == 1:
            # csr.T @ B (homogeneous weights)
            #
            # csr: [m, k]
            # B: [m, n]
            # result: [k, n]
            #
            def mm(
                data_ref,  # [1]
                indices_ref,  # [nse]
                indptr_ref,  # [k + 1]
                B_ref,  # [k, n]
                _,  # [m, n]
                posts_ref,  # [m, n]
            ):
                i_k = pl.program_id(0)
                i_n = pl.program_id(1)
                i_col_start = i_n * block_dim_n
                col_start = indptr_ref[i_k]
                col_end = indptr_ref[i_k + 1]
                mask = (i_col_start + jnp.arange(block_dim_n)) < B_ref.shape[1]
                events = load(B_ref.at[i_k, pl.ds(i_col_start, block_dim_n)], mask=mask)
                if B_ref.dtype == jnp.bool_:
                    mask = mask & events
                    val = jnp.where(events, data_ref[0], 0.)
                else:
                    mask = mask & (events > 0.)
                    val = events * data_ref[0]

                def loop_fn(index, _):
                    i_row = indices_ref[index]
                    atomic_add(posts_ref, (i_row, pl.ds(i_col_start, block_dim_n)), val, mask=mask)

                jax.lax.fori_loop(col_start, col_end, loop_fn, None, )

        else:
            # csr.T @ B (heterogeneous weights)
            #
            # csr: [m, k]
            # B: [m, n]
            # result: [k, n]
            #
            def mm(
                data_ref,  # [nse]
                indices_ref,  # [nse]
                indptr_ref,  # [k + 1]
                B_ref,  # [k, n]
                _,  # [m, n]
                posts_ref,  # [m, n]
            ):
                i_k = pl.program_id(0)
                i_n = pl.program_id(1)
                i_col_start = i_n * block_dim_n
                mask = (i_col_start + jnp.arange(block_dim_n)) < B_ref.shape[1]
                col_start = indptr_ref[i_k]
                col_end = indptr_ref[i_k + 1]
                events = load(B_ref.at[i_k, pl.ds(i_col_start, block_dim_n)], mask=mask)
                if B_ref.dtype == jnp.bool_:
                    mask = mask & events
                else:
                    mask = mask & (events > 0.)

                def loop_fn(index, _):
                    i_row = indices_ref[index]
                    val_A = data_ref[index]
                    if B_ref.dtype == jnp.bool_:
                        val = jnp.where(events, val_A, 0.)
                    else:
                        val = events * val_A
                    atomic_add(posts_ref, (i_row, pl.ds(i_col_start, block_dim_n)), val, mask=mask)

                jax.lax.fori_loop(col_start, col_end, loop_fn, None, )

        def kernel(data, indices, indptr, B):
            out_info = kwargs['outs'][0]
            fn = pl.pallas_call(
                mm,
                grid=(m, pl.cdiv(n, block_dim_n)),
                input_output_aliases={4: 0},
                out_shape=kwargs['outs'],
                backend='triton',
            )
            posts = jnp.zeros(out_info.shape, dtype=out_info.dtype)
            return fn(data, indices, indptr, B, posts)

        return kernel
    else:
        return _csrmm_pallas_kernel(
            weight_info=weight_info, indices_info=indices_info, vector_info=vector_info, shape=shape, **kwargs
        )


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
    is_bool = (vector_info.dtype == jnp.bool_)
    nse = kwargs['indices_info'].size
    out_dtype = kwargs['outs'][0].dtype

    if transpose:
        def kernel(weights, indices, indptr, B):
            row_ids = jnp.repeat(
                jnp.arange(m, dtype=indptr.dtype),
                jnp.diff(indptr),
                total_repeat_length=nse,
            )
            B_rows = B[row_ids]  # [nse, n]
            events = B_rows.astype(out_dtype) if is_bool else (B_rows > 0.).astype(out_dtype)
            w = weights[0] if is_homo else weights[:, None]
            return (jnp.zeros((k, n), dtype=out_dtype).at[indices].add(w * events),)
    else:
        def kernel(weights, indices, indptr, B):
            row_ids = jnp.repeat(
                jnp.arange(m, dtype=indptr.dtype),
                jnp.diff(indptr),
                total_repeat_length=nse,
            )
            B_rows = B[indices]  # [nse, n]
            events = B_rows.astype(out_dtype) if is_bool else (B_rows > 0.).astype(out_dtype)
            w = weights[0] if is_homo else weights[:, None]
            return (jnp.zeros((m, n), dtype=out_dtype).at[row_ids].add(w * events),)

    return kernel


def _binary_csrmm_cusparse_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs,
):
    """cuSPARSE-backed kernel for binary CSR SpMM via jax.experimental.sparse (GPU only)."""
    import jax.experimental.sparse as jsparse
    m, k = shape
    is_homo = (weight_info.size == 1)
    is_bool = (vector_info.dtype == jnp.bool_)
    nse = kwargs['indices_info'].size
    out_dtype = kwargs['outs'][0].dtype

    if transpose:
        if is_homo:
            def kernel(weights, indices, indptr, B):
                events = B.astype(out_dtype) if is_bool else (B > 0.).astype(out_dtype)
                ones = jnp.ones(nse, dtype=out_dtype)
                row, col = _csr_to_coo(indices, indptr)
                mat = jsparse.BCOO((ones, jnp.stack([row, col], axis=1)), shape=(m, k))
                return ((mat.T @ events) * weights[0].astype(out_dtype),)
        else:
            def kernel(weights, indices, indptr, B):
                events = B.astype(out_dtype) if is_bool else (B > 0.).astype(out_dtype)
                row, col = _csr_to_coo(indices, indptr)
                mat = jsparse.BCOO((weights.astype(out_dtype), jnp.stack([row, col], axis=1)), shape=(m, k))
                return (mat.T @ events,)
    else:
        if is_homo:
            def kernel(weights, indices, indptr, B):
                events = B.astype(out_dtype) if is_bool else (B > 0.).astype(out_dtype)
                ones = jnp.ones(nse, dtype=out_dtype)
                mat = jsparse.BCSR((ones, indices, indptr), shape=(m, k))
                return ((mat @ events) * weights[0].astype(out_dtype),)
        else:
            def kernel(weights, indices, indptr, B):
                events = B.astype(out_dtype) if is_bool else (B > 0.).astype(out_dtype)
                mat = jsparse.BCSR((weights.astype(out_dtype), indices, indptr), shape=(m, k))
                return (mat @ events,)
    return kernel


def _binary_csrmm_cuda_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs,
):
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_csrmm.cu'),
        name='csr_binary_csrmm',
    )

    out_info = kwargs['outs']

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

    # Homogeneous vs heterogeneous suffix
    is_homo = (weight_info.size == 1)
    homo_suffix = '_homo' if is_homo else '_hetero'

    if transpose:
        kernel_name = f'csr_binary_csrmm.binary_csrmm_t_warp{homo_suffix}{wt_sfx}{spk_suffix}'
    else:
        kernel_name = f'csr_binary_csrmm.binary_csrmm_nt_auto{homo_suffix}{wt_sfx}{spk_suffix}'

    def kernel(weights, indices, indptr, B):
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, indptr, B)

    return kernel


def _csrmm_jvp_data(data_dot, data, indices, indptr, B, *, shape, transpose, **kwargs):
    return [csrmm(data_dot, indices, indptr, B, shape=shape, transpose=transpose, backend=kwargs['backend'])]


def _csrmm_jvp_B(B_dot, data, indices, indptr, B, *, shape, transpose, **kwargs):
    return [csrmm(data, indices, indptr, B_dot, shape=shape, transpose=transpose, backend=kwargs['backend'])]


def _csrmm_transpose_rule(ct, data, indices, indptr, B, *, shape, transpose, **kwargs):
    assert not ad.is_undefined_primal(indices)
    assert not ad.is_undefined_primal(indptr)

    if ad.is_undefined_primal(B):
        dB = csrmm(data, indices, indptr, ct, shape=shape, transpose=not transpose, backend=kwargs['backend'])
        return data, indices, indptr, dB
    else:
        B = jnp.asarray(B)
        if data.aval.shape[0] == 1:  # scalar
            r = binary_csrmm_p_call(
                jnp.ones(1, dtype=data.aval.dtype),
                indices,
                indptr,
                B,
                shape=shape,
                transpose=transpose,
                backend=kwargs['backend'],
            )[0]
            return jnp.expand_dims(jnp.sum(r * ct), axis=0), indices, indptr, B
        else:
            row, col = _csr_to_coo(indices, indptr)
            if transpose:
                d_data = sddmm_coo_indices(B, ct.T, row, col).data
            else:
                d_data = sddmm_coo_indices(B, ct.T, col, row).data
            return d_data, indices, indptr, B


def _csrmm_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[3].shape
        B = jnp.transpose(args[3], (1, 0, 2)).reshape(m, batch_size * n)
        r = binary_csrmm_p_call(
            args[0],
            args[1],
            args[2],
            B,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend'],
        )[0]
        r = jnp.reshape(r, [r.shape[0], batch_size, n])
        return [r], [1]

    elif tuple(axes) == (None, None, None, 1):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        m, batch_size, n = args[3].shape
        B = args[3].reshape(m, batch_size * n)
        r = binary_csrmm_p_call(
            args[0],
            args[1],
            args[2],
            B,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend'],
        )[0]
        r = jnp.reshape(r, [r.shape[0], batch_size, n])
        return [r], [1]

    elif tuple(axes) == (None, None, None, 2):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        m, n, batch_size = args[3].shape
        B = args[3].reshape(m, batch_size * n)
        r = binary_csrmm_p_call(
            args[0],
            args[1],
            args[2],
            B,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend'],
        )[0]
        r = jnp.reshape(r, [r.shape[0], n, batch_size])
        return [r], [2]

    else:
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
                name = (f"{'T' if transpose else 'NT'},"
                        f"{'homo' if homo else 'hetero'},"
                        f"{'bool' if bool_event else 'float'}")
                configs.append(
                    BenchmarkConfig(
                        name,
                        (weights, indices, jnp.asarray(indptr), B),
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

    Returns
    -------
    list of jax.Array
        A single-element list containing the result matrix.  Shape
        ``(shape[1], cols)`` when ``transpose=True`` or
        ``(shape[0], cols)`` when ``transpose=False``.

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
    ``x > 0`` (float), and ``0`` otherwise.  ``w[j]`` is either
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
        >>> result = binary_csrmm_p_call(
        ...     weights, indices, indptr, B,
        ...     shape=(2, 3), transpose=False)
    """
    assert indices.dtype in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64], "Indices must be int32 or int64."
    assert indptr.dtype in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64], "Indptr must be int32 or int64."
    assert indptr.ndim == 1, "Indptr must be 1D."
    assert indices.ndim == 1, "Indices must be 1D."
    assert indptr.dtype == indices.dtype, "Indices and indptr must have the same dtype."
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
        outs=(out_info,),
        shape=shape,
        transpose=transpose,
        backend=backend,
        # Provide shape and data type information for indices.
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        # Provide shape and data type information for indptr.
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        # Provide shape and data type information for weights.
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        # Provide shape and data type information for B.
        vector_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
    )


binary_csrmm_p = XLACustomKernel(
    'binary_csrmm',
    doc="""
Low-level XLA custom-kernel primitive for ``binary_csrmm``.

This ``XLACustomKernel`` instance dispatches the binary (event-driven) CSR sparse matrix-matrix multiplication
operation to registered backends (``numba``, ``pallas``),
using runtime shape/dtype metadata provided by the high-level wrapper.

Only entries of ``B`` that are ``True`` (boolean) or positive (float) are considered active events
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
binary_csrmm_p.def_warp_kernel(_csrmm_warp_kernel)
binary_csrmm_p.def_pallas_kernel('gpu', _csrmm_pallas_gpu_kernel)
binary_csrmm_p.def_cuda_kernel(_binary_csrmm_cuda_kernel)
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
