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

from typing import Optional, Sequence

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._misc import _csr_to_coo, generate_block_dim, namescope
from brainevent._op import numba_kernel, jaxinfo_to_warpinfo, XLACustomKernel, general_batching_rule
from brainevent._op.benchmark import BenchmarkConfig
from brainevent._sddmm import sddmm_coo_indices
from brainevent._typing import Data, Indptr, Index, MatrixShape
from brainevent.config import get_numba_parallel

__all__ = [
    'csrmv',
    'csrmv_p',
    'csrmm',
    'csrmm_p',
]


@namescope(static_argnames=("shape", "transpose"))
def csrmv(
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
    Product of a CSR sparse matrix and a dense vector.

    Computes ``y = A @ v`` (or ``y = A.T @ v`` when ``transpose=True``)
    where ``A`` is stored in Compressed Sparse Row format and ``v`` is a
    dense vector.  Unlike the binary (event-driven) variant, every element
    of ``v`` contributes to the result regardless of sign or magnitude.

    The function supports physical units via :mod:`brainunit`.  If ``data``
    or ``v`` carry units, the result is returned in the corresponding
    product unit.

    Parameters
    ----------
    data : jax.Array, numpy.ndarray, or brainunit.Quantity
        Non-zero values of the CSR matrix.  Shape ``(nse,)`` for
        heterogeneous weights or ``(1,)`` for a single homogeneous weight
        shared across all connections.
    indices : jax.Array or numpy.ndarray
        Column indices of the non-zero elements.  Shape ``(nse,)`` with
        integer dtype (``int32``, ``int64``, ``uint32``, or ``uint64``).
    indptr : jax.Array or numpy.ndarray
        Row index pointer array.  Shape ``(shape[0] + 1,)`` and same dtype
        as ``indices``.
    v : jax.Array, numpy.ndarray, or brainunit.Quantity
        Dense vector.  Shape ``(shape[0],)`` when ``transpose=True`` or
        ``(shape[1],)`` when ``transpose=False``.
    shape : tuple of int
        Two-element tuple ``(m, k)`` giving the logical shape of the
        sparse matrix ``A``.
    transpose : bool, optional
        If ``True``, the sparse matrix is transposed before multiplication,
        i.e. compute ``A.T @ v``.  Default is ``False``.
    backend : str or None, optional
        Compute backend to use.  One of ``'numba'``, ``'warp'``,
        ``'pallas'``, or ``None`` (auto-select).  Default is ``None``.

    Returns
    -------
    y : jax.Array or brainunit.Quantity
        Result vector.  Shape ``(shape[1],)`` when ``transpose=True`` or
        ``(shape[0],)`` when ``transpose=False``.

    See Also
    --------
    csrmm : CSR matrix--matrix multiplication.
    binary_csrmv : Event-driven (binary) CSR matrix--vector multiplication.

    Notes
    -----
    This operation is differentiable with respect to both ``data`` and
    ``v`` via custom JVP and transpose rules.

    Mathematically, the non-transposed operation computes:

    ``y[i] = sum_{j in nz(i)} A[i, j] * v[j]``

    where ``nz(i)`` denotes the set of column indices with non-zero
    entries in row ``i`` of the CSR matrix.

    When ``transpose=True``, the transposed operation computes:

    ``y[j] = sum_{i in nz_col(j)} A[i, j] * v[i]``

    where ``nz_col(j)`` denotes the set of row indices with non-zero
    entries in column ``j``.

    For homogeneous weights (``data`` of shape ``(1,)``), ``A[i, j]``
    equals the constant ``data[0]`` for all structural non-zero
    positions.

    References
    ----------
    .. [1] Y. Saad, *Iterative Methods for Sparse Linear Systems*,
       2nd ed., SIAM, 2003, ch. 3.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._csr.float import csrmv
        >>> data = jnp.array([1.0, 2.0, 3.0, 4.0])
        >>> indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        >>> v = jnp.array([1.0, 2.0, 3.0])
        >>> csrmv(data, indices, indptr, v, shape=(2, 3))
    """
    data, unitd = u.split_mantissa_unit(data)
    v, unitv = u.split_mantissa_unit(v)
    res = csrmv_p_call(data, indices, indptr, v, shape=shape, transpose=transpose, backend=backend)[0]
    return u.maybe_decimal(res * unitd * unitv)


def _csrmv_numba_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba  # pylint: disable=import-outside-toplevel

    if weight_info.size == 1:
        if transpose:
            # [m, k].T @ [m] - cannot parallelize due to race condition
            @numba.njit(fastmath=True)
            def mv(weights, indices, indptr, vector, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(vector.shape[0]):
                    wsp = w * vector[i]
                    for j in range(indptr[i], indptr[i + 1]):
                        posts[indices[j]] += wsp

        else:
            # [m, k] @ [k] - can parallelize by row
            @numba.njit(parallel=get_numba_parallel(), fastmath=True)
            def mv(weights, indices, indptr, vector, posts):
                w = weights[0]
                for i_m in numba.prange(indptr.shape[0] - 1):
                    r = 0.0
                    for j in range(indptr[i_m], indptr[i_m + 1]):
                        r += vector[indices[j]]
                    posts[i_m] = w * r

    else:
        if transpose:
            # [m, k].T @ [m] - cannot parallelize due to race condition
            @numba.njit(fastmath=True)
            def mv(weights, indices, indptr, vector, posts):
                posts[:] = 0.
                for i in range(vector.shape[0]):
                    sp = vector[i]
                    for j in range(indptr[i], indptr[i + 1]):
                        posts[indices[j]] += weights[j] * sp

        else:
            # [m, k] @ [k] - can parallelize by row
            @numba.njit(parallel=get_numba_parallel(), fastmath=True)
            def mv(weights, indices, indptr, vector, posts):
                for i in numba.prange(indptr.shape[0] - 1):
                    r = 0.0
                    for j in range(indptr[i], indptr[i + 1]):
                        r += weights[j] * vector[indices[j]]
                    posts[i] = r

    def kernel(weights, indices, indptr, vector):
        return numba_kernel(mv, outs=kwargs['outs'])(weights, indices, indptr, vector)

    return kernel


def _csrmv_warp_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    indptr_info: jax.ShapeDtypeStruct,
    transpose: bool,
    shape: MatrixShape,
    **kwargs
):
    import warp  # pylint: disable=import-outside-toplevel
    from warp.jax_experimental import jax_kernel

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    indices_warp_info = jaxinfo_to_warpinfo(indices_info)
    indptr_warp_info = jaxinfo_to_warpinfo(indptr_info)
    vector_warp_info = jaxinfo_to_warpinfo(vector_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if transpose:
        if weight_info.size == 1:
            # [m, k].T @ [m]
            @warp.kernel
            def mv(
                weights: weight_warp_info,
                indices: indices_warp_info,
                indptr: indptr_warp_info,
                vector: vector_warp_info,
                posts: out_warp_info,
            ):
                i = warp.tid()
                w = weights[0]
                wsp = w * vector[i]
                for j in range(indptr[i], indptr[i + 1]):
                    posts[indices[j]] += wsp

        else:
            # [m, k].T @ [m]
            @warp.kernel
            def mv(
                weights: weight_warp_info,
                indices: indices_warp_info,
                indptr: indptr_warp_info,
                v: vector_warp_info,
                posts: out_warp_info,
            ):
                i = warp.tid()
                sp = v[i]
                for j in range(indptr[i], indptr[i + 1]):
                    posts[indices[j]] += weights[j] * sp

        def kernel(weights, indices, indptr, vector):
            out_info = jax.ShapeDtypeStruct([shape[1]], weights.dtype)
            dim = vector_info.shape[0]
            fn = jax_kernel(mv, launch_dims=[dim], num_outputs=1, in_out_argnames=['posts'])
            return fn(weights, indices, indptr, vector, jnp.zeros(out_info.shape, out_info.dtype))

    else:
        if weight_info.size == 1:
            # [m, k] @ [k]
            @warp.kernel
            def mv(
                weights: weight_warp_info,
                indices: indices_warp_info,
                indptr: indptr_warp_info,
                v: vector_warp_info,
                posts: out_warp_info,
            ):
                i_m = warp.tid()
                w = weights[0]
                r = weights.dtype(0.)
                for j in range(indptr[i_m], indptr[i_m + 1]):
                    r += w * v[indices[j]]
                posts[i_m] = r

        else:
            # [m, k] @ [k]
            @warp.kernel
            def mv(
                weights: weight_warp_info,
                indices: indices_warp_info,
                indptr: indptr_warp_info,
                v: vector_warp_info,
                posts: out_warp_info,
            ):
                i_row = warp.tid()
                r = weights.dtype(0.)
                for index in range(indptr[i_row], indptr[i_row + 1]):
                    i_k = indices[index]
                    c = v[i_k]
                    w = weights[index]
                    r += w * c
                posts[i_row] = r

        def kernel(weights, indices, indptr, vector):
            out_info = jax.ShapeDtypeStruct([shape[0]], weights.dtype)
            dim = indptr_info.shape[0] - 1
            fn = jax_kernel(mv, launch_dims=[dim], num_outputs=1, output_dims={'posts': out_info.shape})
            return fn(weights, indices, indptr, vector)

    return kernel


def _csrmv_pallas_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add, load

    m, k = shape
    block_dim = generate_block_dim(pl.cdiv(indices_info.size, shape[1] if transpose else shape[0]))
    block_dim = max(1, block_dim // 2)
    block_dim = 32 if block_dim < 32 else block_dim

    if transpose:

        if weight_info.size == 1:
            # csr.T @ B (Scalar Weight)

            def mv(
                data_ref,  # [1]
                indices_ref,  # [nse]
                indptr_ref,  # [rows + 1]
                vector_ref,  # [rows]
                _,  # [?]
                posts_ref,  # [cols]
            ):
                i_row = pl.program_id(0)  # Iterating over rows of the input CSR matrix
                num_rows = indptr_ref.shape[0] - 1

                def _body():
                    col_start = indptr_ref[i_row]
                    col_end = indptr_ref[i_row + 1]
                    col_nnz = col_end - col_start
                    num_blocks = (col_nnz + block_dim - 1) // block_dim

                    val_vector = vector_ref[i_row]
                    data = data_ref[0] * val_vector

                    data_block = jnp.ones((block_dim,), dtype=posts_ref.dtype) * data

                    # Output dimension limit for validity check
                    out_dim = posts_ref.shape[0]

                    def loop_fn(index, _):
                        offset = col_start + index * block_dim
                        mask = offset + jnp.arange(block_dim) < col_end

                        rows = load(indices_ref.at[pl.ds(offset, block_dim)], mask=mask, other=0)

                        # GUARD 2: Indirect Access Boundary Check (Scatter)
                        valid_idx_mask = rows < out_dim
                        final_mask = mask & valid_idx_mask
                        atomic_add(posts_ref, rows, data_block, mask=final_mask)

                    jax.lax.fori_loop(0, num_blocks, loop_fn, None)

                # GUARD 1: Grid / Indptr Boundary Check using jax.lax.cond
                jax.lax.cond(i_row < num_rows, _body, lambda: None)

        else:
            # csr.T @ B (Vector Weight)
            def mv(
                data_ref,  # [nse]
                indices_ref,  # [nse]
                indptr_ref,  # [rows + 1]
                vector_ref,  # [rows]
                _,  # [?]
                posts_ref,  # [cols]
            ):
                i_row = pl.program_id(0)
                num_rows = indptr_ref.shape[0] - 1

                def _body():
                    col_start = indptr_ref[i_row]
                    col_end = indptr_ref[i_row + 1]
                    col_nnz = col_end - col_start
                    num_blocks = (col_nnz + block_dim - 1) // block_dim
                    val_vector = vector_ref[i_row]

                    out_dim = posts_ref.shape[0]

                    def loop_fn(index, _):
                        offset = col_start + index * block_dim
                        mask = offset + jnp.arange(block_dim) < col_end
                        rows = load(indices_ref.at[pl.ds(offset, block_dim)], mask=mask, other=0)
                        val_A = load(data_ref.at[pl.ds(offset, block_dim)], mask=mask, other=0.0)

                        # GUARD 2: Indirect Access Boundary Check
                        valid_idx_mask = rows < out_dim
                        final_mask = mask & valid_idx_mask

                        contrib = jnp.where(final_mask, val_A * val_vector, 0.0)
                        atomic_add(posts_ref, rows, contrib, mask=final_mask)

                    jax.lax.fori_loop(0, num_blocks, loop_fn, None)

                # GUARD 1: Grid / Indptr Boundary Check using jax.lax.cond
                jax.lax.cond(i_row < num_rows, _body, lambda: None)

        def kernel(data, indices, indptr, vector):
            out_info = kwargs['outs'][0]
            placeholder = jnp.zeros(out_info.shape, out_info.dtype)

            # has_batch = vector.ndim > 1
            launch_rows = shape[0]  # CSR rows

            grid = (launch_rows,)
            fn = pl.pallas_call(
                mv,
                grid=grid,
                input_output_aliases={4: 0},
                out_shape=kwargs['outs'],
                backend='triton'
            )

            return fn(data, indices, indptr, vector, placeholder)

    else:
        # Non-Transpose: csr @ B -> A @ x

        if weight_info.size == 1:
            def mv(
                data_ref,  # [1]
                indices_ref,  # [nse]
                indptr_ref,  # [m + 1]
                vector_ref,  # [k]
                posts_ref,  # [m]
            ):
                i_row = pl.program_id(0)
                num_rows = indptr_ref.shape[0] - 1

                def _body():
                    row_start = indptr_ref[i_row]
                    row_end = indptr_ref[i_row + 1]
                    row_nnz = row_end - row_start
                    num_blocks = (row_nnz + block_dim - 1) // block_dim

                    val_A = data_ref[0]

                    vec_len = vector_ref.shape[0]

                    def loop_fn(index, sum_):
                        offset = row_start + index * block_dim
                        mask = offset + jnp.arange(block_dim) < row_end

                        cols = load(indices_ref.at[pl.ds(offset, block_dim)], mask=mask, other=0)

                        # GUARD 2: Indirect Access Boundary Check (Gather)
                        safe_cols = jnp.minimum(cols, vec_len - 1)
                        valid_col_mask = cols < vec_len

                        val_B = load(vector_ref.at[safe_cols])  # ??!!
                        calc_mask = mask & valid_col_mask

                        sum_ += val_A * jnp.sum(jnp.where(calc_mask, val_B, 0.0))
                        return sum_

                    i_row_sum = jax.lax.fori_loop(
                        0,
                        num_blocks,
                        loop_fn,
                        jnp.asarray(0., dtype=posts_ref.dtype)
                    )
                    posts_ref[i_row] = i_row_sum

                # GUARD 1: Grid / Indptr Boundary Check using jax.lax.cond
                jax.lax.cond(i_row < num_rows, _body, lambda: None)

        else:
            def mv(
                data_ref,  # [nse]
                indices_ref,  # [nse]
                indptr_ref,  # [m + 1]
                vector_ref,  # [k]
                posts_ref,  # [m]
            ):
                i_row = pl.program_id(0)
                num_rows = indptr_ref.shape[0] - 1

                def _body():
                    row_start = indptr_ref[i_row]
                    row_end = indptr_ref[i_row + 1]
                    row_nnz = row_end - row_start
                    num_blocks = (row_nnz + block_dim - 1) // block_dim

                    vec_len = vector_ref.shape[0]

                    def loop_fn(index, sum_):
                        offset = row_start + index * block_dim
                        mask = offset + jnp.arange(block_dim) < row_end

                        cols = load(indices_ref.at[pl.ds(offset, block_dim)], mask=mask, other=0)
                        val_A = load(data_ref.at[pl.ds(offset, block_dim)], mask=mask, other=0.0)

                        # GUARD 2: Indirect Access Boundary Check (Gather)
                        safe_cols = jnp.minimum(cols, vec_len - 1)
                        valid_col_mask = cols < vec_len
                        val_B = load(vector_ref.at[safe_cols])

                        calc_mask = mask & valid_col_mask
                        sum_ += jnp.sum(jnp.where(calc_mask, val_A * val_B, 0.0))
                        return sum_

                    i_row_sum = jax.lax.fori_loop(
                        0,
                        num_blocks,
                        loop_fn,
                        jnp.asarray(0., dtype=posts_ref.dtype)
                    )
                    posts_ref[i_row] = i_row_sum

                # GUARD 1: Grid / Indptr Boundary Check using jax.lax.cond
                jax.lax.cond(i_row < num_rows, _body, lambda: None)

        def kernel(data, indices, indptr, vector):
            launch_rows = shape[0]
            fn = pl.pallas_call(mv, grid=(launch_rows,), out_shape=kwargs['outs'], backend='triton')
            return fn(data, indices, indptr, vector)

    return kernel


def _csrmv_jvp_v(v_dot, data, indices, indptr, v, *, shape, transpose, **kwargs):
    return [csrmv(data, indices, indptr, v_dot, shape=shape, transpose=transpose, backend=kwargs['backend'])]


def _csrmv_jvp_weights(data_dot, data, indices, indptr, v, *, shape, transpose, **kwargs):
    return csrmv_p_call(data_dot, indices, indptr, v, shape=shape, transpose=transpose, backend=kwargs['backend'])


def _csrmv_transpose_rule(ct, data, indices, indptr, vector, *, shape, transpose, **kwargs):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")

    ct = ct[0]

    if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
        raise ValueError("Cannot transpose with respect to sparse indices.")
    if ad.is_undefined_primal(vector):
        if type(ct) is ad.Zero:
            ct_events = ad.Zero(vector)
        else:
            ct_events = csrmv(
                data,
                indices,
                indptr,
                ct,
                shape=shape,
                transpose=not transpose,
                backend=kwargs['backend'],
            )
        return data, indices, indptr, ct_events
    else:
        if type(ct) is ad.Zero:
            ct_values = ad.Zero(data)
        else:
            if data.aval.shape[0] == 1:  # scalar
                ct_values = csrmv_p_call(
                    jnp.ones(1, dtype=data.aval.dtype),
                    indices,
                    indptr,
                    vector,
                    shape=shape,
                    transpose=transpose,
                    backend=kwargs['backend'],
                )[0]
                ct_values = jnp.inner(ct, ct_values).reshape(*data.aval.shape)
            else:  # heterogeneous values
                row, col = _csr_to_coo(indices, indptr)
                ct_values = vector[row] * ct[col] if transpose else vector[col] * ct[row]
        return ct_values, indices, indptr, vector


def _csrmv_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = csrmm_p_call(
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
        r = csrmm_p_call(
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
        return general_batching_rule(csrmv_p, args, axes, **kwargs)


def _csrmv_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for homo in (True, False):
            n_conn = max(1, int(n_post * prob))
            indptr = np.arange(n_pre + 1, dtype=np.int32) * n_conn
            indices = np.random.randint(0, n_post, (n_pre * n_conn,), dtype=np.int32)
            weights = jnp.ones(1, dtype=dtype) if homo else jnp.ones(n_pre * n_conn, dtype=dtype)
            v_size = n_post if not transpose else n_pre
            vector = jnp.asarray(np.random.randn(v_size), dtype=dtype)
            name = f"{'T' if transpose else 'NT'},{'homo' if homo else 'hetero'}"
            configs.append(
                BenchmarkConfig(
                    name,
                    (weights, indices, jnp.asarray(indptr), vector),
                    {'shape': (n_pre, n_post), 'transpose': transpose}
                )
            )
    return configs


def csrmv_p_call(
    weights,
    indices,
    indptr,
    vector,
    *,
    shape: Sequence[int],
    transpose: bool,
    backend: Optional[str] = None,
):
    """
    Low-level primitive call for CSR matrix--vector multiplication.

    Prepares inputs, validates shapes and dtypes, and dispatches the
    ``csrmv_p`` XLA custom kernel to compute ``y = A @ v`` (or
    ``y = A.T @ v``), where ``A`` is a CSR matrix and ``v`` is a dense
    vector.

    Parameters
    ----------
    weights : jax.Array
        Non-zero values of the CSR matrix.  Shape ``(nse,)`` for
        heterogeneous weights, ``(1,)`` for a homogeneous weight, or a
        scalar (automatically promoted to shape ``(1,)``).
    indices : jax.Array
        Column indices of non-zero elements.  Shape ``(nse,)`` with dtype
        ``int32``, ``int64``, ``uint32``, or ``uint64``.
    indptr : jax.Array
        Row index pointer array.  Shape ``(shape[0] + 1,)`` and same dtype
        as ``indices``.
    vector : jax.Array
        Dense vector.  Shape ``(shape[0],)`` when ``transpose=True`` or
        ``(shape[1],)`` when ``transpose=False``.
    shape : sequence of int
        Two-element sequence ``(m, k)`` giving the logical shape of the
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
    csrmv : High-level wrapper with unit support.

    Notes
    -----
    Scalar ``weights`` (0-d arrays) are automatically promoted to
    shape ``(1,)`` to indicate a homogeneous weight across all
    connections.

    The computation performed is:

    ``y[i] = sum_{j in nz(i)} w[j] * v[j]``  (non-transposed)

    ``y[j] = sum_{i in nz_col(j)} w[i] * v[i]``  (transposed)

    where ``w[j]`` is either ``weights[j]`` (heterogeneous) or
    ``weights[0]`` (homogeneous), and ``nz(i)`` is the set of column
    indices with structural non-zeros in row ``i``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._csr.float import csrmv_p_call
        >>> weights = jnp.array([1.0, 2.0, 3.0, 4.0])
        >>> indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        >>> vector = jnp.array([1.0, 2.0, 3.0])
        >>> result = csrmv_p_call(
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

    if jnp.ndim(weights) == 0:
        weights = jnp.asarray([weights])

    out_info = (
        jax.ShapeDtypeStruct([shape[1]], weights.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0]], weights.dtype)
    )
    return csrmv_p(
        weights,
        indices,
        indptr,
        vector,
        outs=[out_info],
        shape=shape,
        transpose=transpose,
        backend=backend,
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(vector.shape, vector.dtype),
    )


csrmv_p = XLACustomKernel(
    'csrmv',
    doc="""
Low-level XLA custom-kernel primitive for ``csrmv``.

This ``XLACustomKernel`` instance dispatches the CSR sparse matrix-vector multiplication with floating-point weights
operation to registered backends (``numba``, ``warp``, ``pallas``),
using runtime shape/dtype metadata provided by the high-level wrapper.

All elements of the input vector contribute to the result, regardless of sign or magnitude,
performing standard sparse matrix-vector multiplication with explicit floating-point weights.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``csrmv_p.available_backends(platform)``,
and the default backend can be configured with ``csrmv_p.set_default(platform, backend)``.

See Also
--------
csrmv : High-level user-facing function wrapper.
"""
)
csrmv_p.def_numba_kernel(_csrmv_numba_kernel_generator)
csrmv_p.def_warp_kernel(_csrmv_warp_kernel_generator)
csrmv_p.def_pallas_kernel('gpu', _csrmv_pallas_kernel_generator)
csrmv_p.def_jvp_rule2(_csrmv_jvp_weights, None, None, _csrmv_jvp_v)
csrmv_p.def_transpose_rule(_csrmv_transpose_rule)
csrmv_p.def_batching_rule(_csrmv_batching)
csrmv_p.def_call(csrmv_p_call)
csrmv_p.def_tags('csr', 'float')
csrmv_p.def_benchmark_data(_csrmv_benchmark_data)


@namescope(static_argnames=("shape", "transpose"))
def csrmm(
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
    Product of a CSR sparse matrix and a dense matrix.

    Computes ``C = A @ B`` (or ``C = A.T @ B`` when ``transpose=True``)
    where ``A`` is stored in Compressed Sparse Row format and ``B`` is a
    dense matrix.

    The function supports physical units via :mod:`brainunit`.

    Parameters
    ----------
    data : jax.Array, numpy.ndarray, or brainunit.Quantity
        Non-zero values of the CSR matrix.  Shape ``(nse,)`` for
        heterogeneous weights or ``(1,)`` for a single homogeneous weight.
    indices : jax.Array or numpy.ndarray
        Column indices of the non-zero elements.  Shape ``(nse,)`` with
        integer dtype.
    indptr : jax.Array or numpy.ndarray
        Row index pointer array.  Shape ``(shape[0] + 1,)`` and same dtype
        as ``indices``.
    B : jax.Array, numpy.ndarray, or brainunit.Quantity
        Dense matrix.  Shape ``(shape[0], cols)`` when ``transpose=True``
        or ``(shape[1], cols)`` when ``transpose=False``.
    shape : tuple of int
        Two-element tuple ``(m, k)`` giving the logical shape of the
        sparse matrix ``A``.
    transpose : bool, optional
        If ``True``, transpose ``A`` before multiplication.  Default is
        ``False``.
    backend : str or None, optional
        Compute backend.  One of ``'numba'``, ``'warp'``, ``'pallas'``, or
        ``None`` (auto-select).  Default is ``None``.

    Returns
    -------
    C : jax.Array or brainunit.Quantity
        Result matrix.  Shape ``(shape[1], cols)`` when ``transpose=True``
        or ``(shape[0], cols)`` when ``transpose=False``.

    See Also
    --------
    csrmv : CSR matrix--vector multiplication.
    binary_csrmm : Event-driven (binary) CSR matrix--matrix multiplication.

    Notes
    -----
    Custom JVP and transpose rules are provided for automatic
    differentiation with respect to ``data`` and ``B``.

    Mathematically, the non-transposed operation computes:

    ``C[i, l] = sum_{j in nz(i)} A[i, j] * B[j, l]``

    where ``nz(i)`` denotes the set of column indices with non-zero
    entries in row ``i`` of the CSR matrix.

    When ``transpose=True``, the transposed operation computes:

    ``C[j, l] = sum_{i in nz_col(j)} A[i, j] * B[i, l]``

    where ``nz_col(j)`` denotes the set of row indices with non-zero
    entries in column ``j``.

    For homogeneous weights (``data`` of shape ``(1,)``), ``A[i, j]``
    equals the constant ``data[0]`` for all structural non-zero
    positions.

    References
    ----------
    .. [1] F. G. Gustavson, "Two Fast Algorithms for Sparse Matrices:
       Multiplication and Permuted Transposition," *ACM Transactions on
       Mathematical Software*, vol. 4, no. 3, pp. 250--269, 1978.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._csr.float import csrmm
        >>> data = jnp.array([1.0, 2.0, 3.0, 4.0])
        >>> indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        >>> B = jnp.array([[1.0, 0.5],
        ...                [2.0, 1.5],
        ...                [3.0, 2.5]])
        >>> csrmm(data, indices, indptr, B, shape=(2, 3))
    """
    data, unitd = u.split_mantissa_unit(data)
    B, unitb = u.split_mantissa_unit(B)
    res = csrmm_p_call(
        data,
        indices,
        indptr,
        B,
        shape=shape,
        transpose=transpose,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * (unitd * unitb))


def _csrmm_numba_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba  # pylint: disable=import-outside-toplevel

    if weight_info.size == 1:
        if transpose:
            # csr.T @ B - cannot parallelize due to race condition
            #
            # CSR: [k, m]
            # B: [k, n]
            #
            @numba.njit(fastmath=True)
            def mm(weights, indices, indptr, B, posts):
                posts[:] = 0.
                w = weights[0]
                for i_k in range(B.shape[0]):
                    wsp = w * B[i_k]
                    for index in range(indptr[i_k], indptr[i_k + 1]):
                        i_row = indices[index]
                        posts[i_row] += wsp

        else:
            # csr @ B - can parallelize by row
            #
            # CSR: [m, k]
            # B: [k, n]
            #
            @numba.njit(parallel=get_numba_parallel(), fastmath=True)
            def mm(weights, indices, indptr, B, posts):
                w = weights[0]
                for i_m in numba.prange(indptr.shape[0] - 1):
                    r = np.zeros(B.shape[1], dtype=posts.dtype)
                    for index in range(indptr[i_m], indptr[i_m + 1]):
                        i_k = indices[index]
                        r += B[i_k]
                    posts[i_m] = w * r

    else:
        if transpose:
            # csr.T @ B - cannot parallelize due to race condition
            #
            # CSR: [k, m]
            # B: [k, n]
            #
            @numba.njit(fastmath=True)
            def mm(weights, indices, indptr, B, posts):
                posts[:] = 0.
                for i_k in range(B.shape[0]):
                    B_row = B[i_k]
                    for index in range(indptr[i_k], indptr[i_k + 1]):
                        i_row = indices[index]
                        posts[i_row] += weights[index] * B_row

        else:
            # csr @ B - can parallelize by row
            #
            # CSR: [m, k]
            # B: [k, n]
            #
            @numba.njit(parallel=get_numba_parallel(), fastmath=True)
            def mm(weights, indices, indptr, B, posts):
                for i_m in numba.prange(indptr.shape[0] - 1):
                    r = np.zeros(B.shape[1], dtype=posts.dtype)
                    for index in range(indptr[i_m], indptr[i_m + 1]):
                        i_k = indices[index]
                        r += weights[index] * B[i_k]
                    posts[i_m] = r

    def kernel(weights, indices, indptr, B):
        return numba_kernel(mm, outs=kwargs['outs'])(weights, indices, indptr, B)

    return kernel


def _csrmm_warp_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    indptr_info: jax.ShapeDtypeStruct,
    transpose: bool,
    shape: MatrixShape,
    **kwargs
):
    import warp  # pylint: disable=import-outside-toplevel
    from warp.jax_experimental import jax_kernel

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    indices_warp_info = jaxinfo_to_warpinfo(indices_info)
    indptr_warp_info = jaxinfo_to_warpinfo(indptr_info)
    B_warp_info = jaxinfo_to_warpinfo(vector_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    n = vector_info.shape[1]

    if transpose:
        if weight_info.size == 1:
            # csr.T @ B — scatter pattern, needs atomic_add
            @warp.kernel
            def mm(
                weights: weight_warp_info,
                indices: indices_warp_info,
                indptr: indptr_warp_info,
                B: B_warp_info,
                posts: out_warp_info,
            ):
                i_k = warp.tid()
                w = weights[0]
                col_start = indptr[i_k]
                col_end = indptr[i_k + 1]
                for index in range(col_start, col_end):
                    i_row = indices[index]
                    for j in range(B.shape[1]):
                        warp.atomic_add(posts, i_row, j, w * B[i_k, j])

        else:
            # csr.T @ B — scatter pattern, needs atomic_add
            @warp.kernel
            def mm(
                weights: weight_warp_info,
                indices: indices_warp_info,
                indptr: indptr_warp_info,
                B: B_warp_info,
                posts: out_warp_info,
            ):
                i_k = warp.tid()
                col_start = indptr[i_k]
                col_end = indptr[i_k + 1]
                for index in range(col_start, col_end):
                    i_row = indices[index]
                    weight = weights[index]
                    for j in range(B.shape[1]):
                        warp.atomic_add(posts, i_row, j, weight * B[i_k, j])

        def kernel(weights, indices, indptr, B):
            out_info = jax.ShapeDtypeStruct([shape[1], n], weights.dtype)
            dim = vector_info.shape[0]
            fn = jax_kernel(mm, launch_dims=[dim], num_outputs=1, in_out_argnames=['posts'])
            return fn(weights, indices, indptr, B, jnp.zeros(out_info.shape, out_info.dtype))

    else:
        if weight_info.size == 1:
            # csr @ B — gather pattern, each thread owns its output row
            @warp.kernel
            def mm(
                weights: weight_warp_info,
                indices: indices_warp_info,
                indptr: indptr_warp_info,
                B: B_warp_info,
                posts: out_warp_info,
            ):
                i_m = warp.tid()
                weight = weights[0]
                for index in range(indptr[i_m], indptr[i_m + 1]):
                    i_k = indices[index]
                    for j in range(B.shape[1]):
                        posts[i_m, j] += weight * B[i_k, j]

        else:
            # csr @ B — gather pattern, each thread owns its output row
            @warp.kernel
            def mm(
                weights: weight_warp_info,
                indices: indices_warp_info,
                indptr: indptr_warp_info,
                B: B_warp_info,
                posts: out_warp_info,
            ):
                i_m = warp.tid()
                for index in range(indptr[i_m], indptr[i_m + 1]):
                    i_k = indices[index]
                    weight = weights[index]
                    for j in range(B.shape[1]):
                        posts[i_m, j] += weight * B[i_k, j]

        def kernel(weights, indices, indptr, B):
            out_info = jax.ShapeDtypeStruct([shape[0], n], weights.dtype)
            dim = indptr_info.shape[0] - 1
            fn = jax_kernel(mm, launch_dims=[dim], num_outputs=1, in_out_argnames=['posts'])
            return fn(weights, indices, indptr, B, jnp.zeros(out_info.shape, out_info.dtype))

    return kernel


def _csrmm_pallas_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    shape,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import store, atomic_add

    m, k = shape
    n = vector_info.shape[1]
    block_dim_n = generate_block_dim(n, 512)

    if transpose:
        if weight_info.size == 1:
            # csr.T @ B
            #
            # csr: [k, m]
            # B: [k, n]
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
                num_k = indptr_ref.shape[0] - 1

                def _body():
                    i_n_block = pl.program_id(1)
                    i_n_start = i_n_block * block_dim_n
                    mask = (i_n_start + jnp.arange(block_dim_n)) < B_ref.shape[1]
                    B_row = B_ref[i_k, pl.ds(i_n_start, block_dim_n)]
                    B_row = jnp.where(mask, B_row, 0.0)
                    val = B_row * data_ref[0]

                    out_rows = posts_ref.shape[0]

                    def loop_fn(index, _):
                        i_row = indices_ref[index]

                        # Indirect Write Guard (Scatter)
                        row_valid = i_row < out_rows
                        final_mask = mask & row_valid

                        atomic_add(posts_ref, (i_row, pl.ds(i_n_start, block_dim_n)), val, mask=final_mask)

                    jax.lax.fori_loop(indptr_ref[i_k], indptr_ref[i_k + 1], loop_fn, None)

                # GUARD 1: Grid / Indptr Boundary Check using jax.lax.cond
                jax.lax.cond(i_k < num_k, _body, lambda: None)

        else:
            # csr.T @ B
            #
            # csr: [k, m]
            # B: [k, n]
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
                num_k = indptr_ref.shape[0] - 1

                def _body():
                    i_n_block = pl.program_id(1)
                    i_n_start = i_n_block * block_dim_n
                    mask = (i_n_start + jnp.arange(block_dim_n)) < B_ref.shape[1]
                    B_row = B_ref[i_k, pl.ds(i_n_start, block_dim_n)]
                    B_row = jnp.where(mask, B_row, 0.0)

                    out_rows = posts_ref.shape[0]

                    def loop_fn(index, _):
                        i_row = indices_ref[index]
                        A_val = data_ref[index]
                        val = A_val * B_row

                        # Indirect Write Guard (Scatter)
                        row_valid = i_row < out_rows
                        final_mask = mask & row_valid

                        atomic_add(posts_ref, (i_row, pl.ds(i_n_start, block_dim_n)), val, mask=final_mask)

                    jax.lax.fori_loop(indptr_ref[i_k], indptr_ref[i_k + 1], loop_fn, None)

                # GUARD 1: Grid / Indptr Boundary Check using jax.lax.cond
                jax.lax.cond(i_k < num_k, _body, lambda: None)

        def kernel(data, indices, indptr, B):
            # Transpose: input rows = shape[0] if it was (m,k) ?
            # csrmm_p_call ensures shape matches B. If transpose, B has shape[0] rows.
            # So we iterate shape[0].
            launch_rows = shape[0]

            fn = pl.pallas_call(
                mm,
                grid=(launch_rows, pl.cdiv(n, block_dim_n)),
                input_output_aliases={4: 0},
                out_shape=kwargs['outs'],
                backend='triton'
            )
            out_info = kwargs['outs'][0]
            placeholder = jnp.zeros(out_info.shape, out_info.dtype)
            return fn(data, indices, indptr, B, placeholder)

    else:
        #
        # Gustavson algorithm: Sparse matrix–matrix multiplication is performed in a row-wise fashion.
        #
        # Each nonzero value in a row is multiplied by the nonzero values corresponding to the column index.
        # These values are summed and stored in a temporary row buffer based on their column indices.
        if weight_info.size == 1:

            # csr @ B
            #
            # csr: [m, k]
            # B: [k, n]
            #
            def mm(
                data_ref,  # [1]
                indices_ref,  # [nse]
                indptr_ref,  # [m + 1]
                B_ref,  # [k, n]
                posts_ref,  # [m, n]
            ):
                i_m = pl.program_id(0)
                num_m = indptr_ref.shape[0] - 1

                def _body():
                    i_n_block = pl.program_id(1)
                    i_n_start = i_n_block * block_dim_n
                    mask = (i_n_start + jnp.arange(block_dim_n)) < B_ref.shape[1]
                    weight = data_ref[0]

                    b_rows = B_ref.shape[0]

                    def loop_fn(index, out):
                        i_k = indices_ref[index]

                        # Indirect Read Guard (Gather)
                        safe_k = jnp.minimum(i_k, b_rows - 1)
                        k_valid = i_k < b_rows
                        final_mask = mask & k_valid

                        B_row = B_ref[safe_k, pl.ds(i_n_start, block_dim_n)]
                        B_row = jnp.where(final_mask, B_row, 0.0)
                        out += weight * B_row
                        return out

                    i_row_out = jax.lax.fori_loop(
                        indptr_ref[i_m],
                        indptr_ref[i_m + 1],
                        loop_fn,
                        jnp.zeros([block_dim_n], dtype=posts_ref.dtype)
                    )
                    store(
                        posts_ref.at[i_m, pl.ds(i_n_start, block_dim_n)],
                        i_row_out,
                        mask=mask
                    )

                # GUARD 1: Grid / Indptr Boundary Check using jax.lax.cond
                jax.lax.cond(i_m < num_m, _body, lambda: None)

        else:
            # csr @ B
            #
            # csr: [m, k]
            # B: [k, n]
            #
            def mm(
                data_ref,  # [nse]
                indices_ref,  # [nse]
                indptr_ref,  # [m + 1]
                B_ref,  # [k, n]
                posts_ref,  # [m, n]
            ):
                i_m = pl.program_id(0)
                num_m = indptr_ref.shape[0] - 1

                def _body():
                    i_n_block = pl.program_id(1)
                    i_n_start = i_n_block * block_dim_n
                    mask = (i_n_start + jnp.arange(block_dim_n)) < B_ref.shape[1]

                    b_rows = B_ref.shape[0]

                    def loop_fn(index, out):
                        i_col = indices_ref[index]
                        val_A = data_ref[index]

                        # Indirect Read Guard (Gather)
                        safe_col = jnp.minimum(i_col, b_rows - 1)
                        col_valid = i_col < b_rows
                        final_mask = mask & col_valid

                        val_B = B_ref[safe_col, pl.ds(i_n_start, block_dim_n)]
                        val_B = jnp.where(final_mask, val_B, 0.0)
                        out += val_A * val_B
                        return out

                    i_row_out = jax.lax.fori_loop(
                        indptr_ref[i_m],
                        indptr_ref[i_m + 1],
                        loop_fn,
                        jnp.zeros([block_dim_n], dtype=posts_ref.dtype)
                    )
                    store(
                        posts_ref.at[i_m, pl.ds(i_n_start, block_dim_n)],
                        i_row_out,
                        mask=mask
                    )

                # GUARD 1: Grid / Indptr Boundary Check using jax.lax.cond
                jax.lax.cond(i_m < num_m, _body, lambda: None)

        def kernel(data, indices, indptr, B):
            # Not transpose, B shape matches shape[1]. Indptr matches shape[0].
            launch_rows = shape[0]
            fn = pl.pallas_call(mm, grid=(launch_rows, pl.cdiv(n, block_dim_n)), out_shape=kwargs['outs'],
                                backend='triton')
            return fn(data, indices, indptr, B)

    return kernel


def _csrmm_jvp_data(data_dot, data, indices, indptr, B, *, shape, transpose, **kwargs):
    return [csrmm(data_dot, indices, indptr, B, shape=shape, transpose=transpose, backend=kwargs['backend'])]


def _csrmm_jvp_B(B_dot, data, indices, indptr, B, *, shape, transpose, **kwargs):
    return [csrmm(data, indices, indptr, B_dot, shape=shape, transpose=transpose, backend=kwargs['backend'])]


def _csrmm_transpose_rule(ct, data, indices, indptr, B, *, shape, transpose, **kwargs):
    assert not ad.is_undefined_primal(indices)
    assert not ad.is_undefined_primal(indptr)
    ct = ct[0]

    if ad.is_undefined_primal(B):
        dB = csrmm(data, indices, indptr, ct, shape=shape, transpose=not transpose, backend=kwargs['backend'])
        return data, indices, indptr, dB
    else:
        B = jnp.asarray(B)
        if data.aval.shape[0] == 1:  # scalar
            r = csrmm_p_call(
                jnp.ones(1, dtype=data.aval.dtype),
                indices,
                indptr,
                B,
                shape=shape,
                transpose=transpose,
                backend=kwargs['backend']
            )[0]
            return jnp.expand_dims(jnp.sum(r * ct), axis=0), indices, indptr, B
        else:
            # TODO
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
        r = csrmm_p_call(
            args[0],
            args[1],
            args[2],
            B,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend']
        )[0]
        r = jnp.reshape(r, [r.shape[0], batch_size, n])
        return [r], [1]

    elif tuple(axes) == (None, None, None, 1):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        m, batch_size, n = args[3].shape
        B = args[3].reshape(m, batch_size * n)
        r = csrmm_p_call(
            args[0],
            args[1],
            args[2],
            B,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend']
        )[0]
        r = jnp.reshape(r, [r.shape[0], batch_size, n])
        return [r], [1]

    elif tuple(axes) == (None, None, None, 2):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        m, n, batch_size = args[3].shape
        B = args[3].reshape(m, batch_size * n)
        r = csrmm_p_call(
            args[0],
            args[1],
            args[2],
            B,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend']
        )[0]
        r = jnp.reshape(r, [r.shape[0], n, batch_size])
        return [r], [2]

    else:
        return general_batching_rule(csrmm_p, args, axes, **kwargs)


def _csrmm_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for homo in (True, False):
            n_conn = max(1, int(n_post * prob))
            indptr = np.arange(n_pre + 1, dtype=np.int32) * n_conn
            indices = np.random.randint(0, n_post, (n_pre * n_conn,), dtype=np.int32)
            weights = jnp.ones(1, dtype=dtype) if homo else jnp.ones(n_pre * n_conn, dtype=dtype)
            b_rows = n_post if not transpose else n_pre
            B = jnp.asarray(np.random.randn(b_rows, 10), dtype=dtype)
            name = f"{'T' if transpose else 'NT'},{'homo' if homo else 'hetero'}"
            configs.append(
                BenchmarkConfig(
                    name,
                    (weights, indices, jnp.asarray(indptr), B),
                    {'shape': (n_pre, n_post), 'transpose': transpose}
                )
            )
    return configs


def csrmm_p_call(
    weights,
    indices,
    indptr,
    B,
    *,
    shape: Sequence[int],
    transpose: bool,
    backend: Optional[str] = None,
):
    """
    Low-level primitive call for CSR matrix--matrix multiplication.

    Prepares inputs, validates shapes and dtypes, and dispatches the
    ``csrmm_p`` XLA custom kernel to compute ``C = A @ B`` (or
    ``C = A.T @ B``), where ``A`` is a CSR matrix and ``B`` is a dense
    matrix.

    Parameters
    ----------
    weights : jax.Array
        Non-zero values of the CSR matrix.  Shape ``(nse,)`` for
        heterogeneous weights, ``(1,)`` for a homogeneous weight, or a
        scalar (automatically promoted to shape ``(1,)``).
    indices : jax.Array
        Column indices of non-zero elements.  Shape ``(nse,)`` with dtype
        ``int32``, ``int64``, ``uint32``, or ``uint64``.
    indptr : jax.Array
        Row index pointer array.  Shape ``(shape[0] + 1,)`` and same dtype
        as ``indices``.
    B : jax.Array
        Dense matrix.  Shape ``(shape[0], cols)`` when
        ``transpose=True`` or ``(shape[1], cols)`` when
        ``transpose=False``.
    shape : sequence of int
        Two-element sequence ``(m, k)`` giving the logical shape of the
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
    csrmm : High-level wrapper with unit support.

    Notes
    -----
    Scalar ``weights`` (0-d arrays) are automatically promoted to
    shape ``(1,)`` to indicate a homogeneous weight across all
    connections.

    The computation performed is:

    ``C[i, l] = sum_{j in nz(i)} w[j] * B[j, l]``  (non-transposed)

    ``C[j, l] = sum_{i in nz_col(j)} w[i] * B[i, l]``  (transposed)

    where ``w[j]`` is either ``weights[j]`` (heterogeneous) or
    ``weights[0]`` (homogeneous), and ``nz(i)`` is the set of column
    indices with structural non-zeros in row ``i``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._csr.float import csrmm_p_call
        >>> weights = jnp.array([1.0, 2.0, 3.0, 4.0])
        >>> indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        >>> B = jnp.array([[1.0, 0.5],
        ...                [2.0, 1.5],
        ...                [3.0, 2.5]])
        >>> result = csrmm_p_call(
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
        assert shape[1] == B.shape[0], "Shape mismatch for non-transpose operation."
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'

    if jnp.ndim(weights) == 0:
        weights = jnp.asarray([weights])

    out_info = (
        jax.ShapeDtypeStruct([shape[1], B.shape[1]], weights.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0], B.shape[1]], weights.dtype)
    )
    return csrmm_p(
        weights,
        indices,
        indptr,
        B,
        outs=[out_info],
        shape=shape,
        transpose=transpose,
        backend=backend,
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
    )


csrmm_p = XLACustomKernel(
    'csrmm',
    doc="""
Low-level XLA custom-kernel primitive for ``csrmm``.

This ``XLACustomKernel`` instance dispatches the CSR sparse matrix-matrix multiplication with floating-point weights
operation to registered backends (``numba``, ``warp``, ``pallas``),
using runtime shape/dtype metadata provided by the high-level wrapper.

All elements of the input matrix contribute to the result, performing standard
sparse matrix-matrix multiplication with explicit floating-point weights.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``csrmm_p.available_backends(platform)``,
and the default backend can be configured with ``csrmm_p.set_default(platform, backend)``.

See Also
--------
csrmm : High-level user-facing function wrapper.
"""
)
csrmm_p.def_numba_kernel(_csrmm_numba_kernel_generator)
csrmm_p.def_warp_kernel(_csrmm_warp_kernel_generator)
csrmm_p.def_pallas_kernel('gpu', _csrmm_pallas_kernel_generator)
csrmm_p.def_jvp_rule2(_csrmm_jvp_data, None, None, _csrmm_jvp_B)
csrmm_p.def_transpose_rule(_csrmm_transpose_rule)
csrmm_p.def_batching_rule(_csrmm_batching)
csrmm_p.def_call(csrmm_p_call)
csrmm_p.def_tags('csr', 'float')
csrmm_p.def_benchmark_data(_csrmm_benchmark_data)
