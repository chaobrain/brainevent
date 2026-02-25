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
from brainevent._op import numba_kernel, XLACustomKernel, general_batching_rule
from brainevent._op.benchmark import BenchmarkConfig
from brainevent._sddmm import sddmm_coo_indices
from brainevent._typing import Data, Indptr, Index, MatrixShape
from brainevent.config import get_numba_parallel
from brainevent.kernix import load_cuda_file
from .float import csrmv, csrmm

__all__ = [
    'spfloat_csrmv',
    'spfloat_csrmv_p',
    'spfloat_csrmm',
    'spfloat_csrmm_p',
]


@namescope(static_argnames=("shape", "transpose"))
def spfloat_csrmv(
    data: Data,
    indices: Index,
    indptr: Indptr,
    v: Data,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    backend: Optional[str] = None,
) -> Data:
    """Compute the product of a CSR sparse matrix with sparse float data and a dense vector.

    Performs the matrix-vector product ``y = A @ v`` (or ``y = A.T @ v`` when
    ``transpose=True``), where ``A`` is a sparse matrix in Compressed Sparse Row
    (CSR) format with explicit float-valued non-zero entries. Unlike binary event
    CSR operations, this function uses the actual floating-point values stored in
    ``data`` during multiplication.

    Parameters
    ----------
    data : jax.Array or Quantity
        Non-zero element values of the CSR sparse matrix, with shape ``(nse,)``
        where ``nse`` is the number of stored elements. A scalar (shape ``(1,)``)
        indicates a homogeneous weight shared by all connections. May carry
        physical units via ``brainunit.Quantity``.
    indices : jax.Array
        Column indices of the non-zero elements, with shape ``(nse,)`` and
        integer dtype.
    indptr : jax.Array
        Row pointer array of the CSR format, with shape ``(shape[0] + 1,)`` and
        the same integer dtype as ``indices``.
    v : jax.Array or Quantity
        Dense vector to multiply, with shape
        ``(shape[0],)`` if ``transpose=True``, or ``(shape[1],)`` otherwise. May
        carry physical units.
    shape : tuple of int
        Shape of the sparse matrix as ``(m, k)``.
    transpose : bool, optional
        If ``True``, compute ``A.T @ v`` instead of ``A @ v``. Default is
        ``False``.
    backend : str or None, optional
        Compute backend to use. One of ``'numba'``, ``'pallas'``, or ``None`` for
        automatic selection.

    Returns
    -------
    jax.Array or Quantity
        Result vector with shape ``(shape[1],)`` if ``transpose=True``, or
        ``(shape[0],)`` otherwise. Carries the product of the units of ``data``
        and ``v``.

    Raises
    ------
    AssertionError
        If ``indices`` or ``indptr`` have unsupported dtypes (must be ``int32``,
        ``int64``, ``uint32``, or ``uint64``), if they are not 1-D, if their
        dtypes do not match, or if the vector shape does not match ``shape``
        given the ``transpose`` flag.  These checks are performed by the
        underlying :func:`sparse_float_csrmv_p_call`.

    See Also
    --------
    spfloat_csrmm : Sparse-float CSR matrix--dense matrix multiplication.
    spfloat_csrmv_p : Low-level XLA custom kernel primitive for this operation.

    Notes
    -----
    Given a CSR matrix ``A`` of shape ``(m, k)`` and a dense vector ``v``, the
    non-transpose operation computes for each row ``i``:

    ``y[i] = sum_{j in nz(i)} data[j] * v[indices[j]]``

    where ``nz(i)`` denotes the stored non-zero positions in row ``i``, i.e.,
    ``range(indptr[i], indptr[i+1])``.

    When ``transpose=True`` the operation computes:

    ``y[j] = sum_{i : j in nz(i)} data[pos(i,j)] * v[i]``

    This operation is differentiable with respect to both ``data`` and ``v``. The
    JVP and transpose rules fall through to :func:`csrmv` for gradient
    computation.

    When ``data`` has shape ``(1,)``, a homogeneous (scalar) weight is broadcast
    to all stored elements, which enables optimized kernel paths.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._csr.sparse_float import spfloat_csrmv
        >>> data = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        >>> indices = jnp.array([0, 2, 1], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 2, 3], dtype=jnp.int32)
        >>> v = jnp.array([1.0, 0.5, 0.0], dtype=jnp.float32)
        >>> y = spfloat_csrmv(data, indices, indptr, v, shape=(2, 3))
    """
    data, unitd = u.split_mantissa_unit(data)
    v, unitv = u.split_mantissa_unit(v)
    res = sparse_float_csrmv_p_call(
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
def spfloat_csrmm(
    data: Data,
    indices: Index,
    indptr: Indptr,
    B: Data,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    backend: Optional[str] = None,
) -> Data:
    """Compute the product of a CSR sparse matrix with sparse float data and a dense matrix.

    Performs the matrix-matrix product ``C = A @ B`` (or ``C = A.T @ B`` when
    ``transpose=True``), where ``A`` is a sparse matrix in Compressed Sparse Row
    (CSR) format with explicit float-valued non-zero entries.

    Parameters
    ----------
    data : jax.Array or Quantity
        Non-zero element values of the CSR sparse matrix, with shape ``(nse,)``
        where ``nse`` is the number of stored elements. A scalar (shape ``(1,)``)
        indicates a homogeneous weight shared by all connections. May carry
        physical units via ``brainunit.Quantity``.
    indices : jax.Array
        Column indices of the non-zero elements, with shape ``(nse,)`` and
        integer dtype.
    indptr : jax.Array
        Row pointer array of the CSR format, with shape ``(shape[0] + 1,)`` and
        the same integer dtype as ``indices``.
    B : jax.Array or Quantity
        Dense matrix to multiply, with shape
        ``(shape[0], n)`` if ``transpose=True``, or ``(shape[1], n)`` otherwise,
        where ``n`` is the number of columns. May carry physical units.
    shape : tuple of int
        Shape of the sparse matrix as ``(m, k)``.
    transpose : bool, optional
        If ``True``, compute ``A.T @ B`` instead of ``A @ B``. Default is
        ``False``.
    backend : str or None, optional
        Compute backend to use. One of ``'numba'``, ``'pallas'``, or ``None`` for
        automatic selection.

    Returns
    -------
    jax.Array or Quantity
        Result matrix with shape ``(shape[1], n)`` if ``transpose=True``, or
        ``(shape[0], n)`` otherwise. Carries the product of the units of ``data``
        and ``B``.

    Raises
    ------
    AssertionError
        If ``indices`` or ``indptr`` have unsupported dtypes (must be ``int32``,
        ``int64``, ``uint32``, or ``uint64``), if they are not 1-D, if their
        dtypes do not match, or if the first dimension of ``B`` does not match
        ``shape`` given the ``transpose`` flag.  These checks are performed by
        the underlying :func:`sparse_float_csrmm_p_call`.

    See Also
    --------
    spfloat_csrmv : Sparse-float CSR matrix--dense vector multiplication.
    spfloat_csrmm_p : Low-level XLA custom kernel primitive for this operation.

    Notes
    -----
    Given a CSR matrix ``A`` of shape ``(m, k)`` and a dense matrix ``B`` of
    shape ``(k, n)``, the non-transpose operation computes for each output
    element:

    ``C[i, l] = sum_{j in nz(i)} data[j] * B[indices[j], l]``

    where ``nz(i) = range(indptr[i], indptr[i+1])`` are the stored positions in
    row ``i``.

    When ``transpose=True`` and ``B`` has shape ``(m, n)``, the operation
    computes:

    ``C[j, l] = sum_{i : j in nz(i)} data[pos(i,j)] * B[i, l]``

    This operation is differentiable with respect to both ``data`` and ``B``. The
    JVP and transpose rules fall through to :func:`csrmm` for gradient
    computation.

    Batching over the last dimension(s) of ``B`` is supported via custom vmap
    rules that reshape into a single matrix multiplication call.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._csr.sparse_float import spfloat_csrmm
        >>> data = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        >>> indices = jnp.array([0, 2, 1], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 2, 3], dtype=jnp.int32)
        >>> B = jnp.ones((3, 4), dtype=jnp.float32)
        >>> C = spfloat_csrmm(data, indices, indptr, B, shape=(2, 3))
    """
    data, unitd = u.split_mantissa_unit(data)
    B, unitb = u.split_mantissa_unit(B)
    res = sparse_float_csrmm_p_call(
        data,
        indices,
        indptr,
        B,
        shape=shape,
        transpose=transpose,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * (unitd * unitb))


def _sparse_float_csrmv_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba

    if weight_info.size == 1:
        if transpose:
            @numba.njit(fastmath=True)
            def mv(weights, indices, indptr, v, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(v.shape[0]):
                    sp = v[i]
                    if sp != 0.:
                        wsp = w * sp
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j]] += wsp

        else:
            @numba.njit(parallel=get_numba_parallel(), fastmath=True)
            def mv(weights, indices, indptr, v, posts):
                w = weights[0]
                for i in numba.prange(indptr.shape[0] - 1):
                    r = 0.
                    for j in range(indptr[i], indptr[i + 1]):
                        c = v[indices[j]]
                        if c != 0.:
                            r += w * c
                    posts[i] = r

    else:
        if transpose:
            @numba.njit(fastmath=True)
            def mv(weights, indices, indptr, v, posts):
                posts[:] = 0.
                for i in range(v.shape[0]):
                    sp = v[i]
                    if sp != 0.:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j]] += weights[j] * sp

        else:
            @numba.njit(parallel=get_numba_parallel(), fastmath=True)
            def mv(weights, indices, indptr, v, posts):
                for i in numba.prange(indptr.shape[0] - 1):
                    r = 0.
                    for j in range(indptr[i], indptr[i + 1]):
                        c = v[indices[j]]
                        if c != 0.:
                            r += weights[j] * c
                    posts[i] = r

    def kernel(weights, indices, indptr, vector):
        return numba_kernel(mv, outs=kwargs['outs'])(weights, indices, indptr, vector)

    return kernel


def _sparse_float_csrmv_pallas_kernel(
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

            def mv_transpose_scalar_single(
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
                    val_vector = vector_ref[i_row]

                    @pl.when(val_vector != 0.)
                    def event_process():
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

            def mv_transpose_scalar_batch(
                data_ref, indices_ref, indptr_ref, vector_ref, _, posts_ref
            ):
                idx_batch = pl.program_id(0)
                i_row = pl.program_id(1)
                num_rows = indptr_ref.shape[0] - 1

                def _body():
                    col_start = indptr_ref[i_row]
                    col_end = indptr_ref[i_row + 1]
                    col_nnz = col_end - col_start
                    num_blocks = (col_nnz + block_dim - 1) // block_dim

                    val_vector = vector_ref[idx_batch, i_row]
                    data = data_ref[0] * val_vector
                    data_block = jnp.ones((block_dim,), dtype=posts_ref.dtype) * data

                    target_posts_ref = posts_ref[idx_batch]
                    out_dim = target_posts_ref.shape[0]

                    def loop_fn(index, _):
                        offset = col_start + index * block_dim
                        mask = offset + jnp.arange(block_dim) < col_end
                        rows = load(indices_ref.at[pl.ds(offset, block_dim)], mask=mask, other=0)

                        # GUARD 2: Indirect Access Boundary Check
                        valid_idx_mask = rows < out_dim
                        final_mask = mask & valid_idx_mask

                        atomic_add(target_posts_ref, rows, data_block, mask=final_mask)

                    jax.lax.fori_loop(0, num_blocks, loop_fn, None)

                # GUARD 1: Grid / Indptr Boundary Check using jax.lax.cond
                jax.lax.cond(i_row < num_rows, _body, lambda: None)

            mm_single_impl = mv_transpose_scalar_single
            mm_batch_impl = mv_transpose_scalar_batch

        else:
            # csr.T @ B (Vector Weight)
            def mv_transpose_vector_single(
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
                    val_vector = vector_ref[i_row]

                    @pl.when(val_vector != 0.)
                    def event_process():
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

            def mv_transpose_vector_batch(
                data_ref, indices_ref, indptr_ref, vector_ref, _, posts_ref
            ):
                idx_batch = pl.program_id(0)
                i_row = pl.program_id(1)
                num_rows = indptr_ref.shape[0] - 1

                def _body():
                    col_start = indptr_ref[i_row]
                    col_end = indptr_ref[i_row + 1]
                    col_nnz = col_end - col_start
                    num_blocks = (col_nnz + block_dim - 1) // block_dim

                    val_vector = vector_ref[idx_batch, i_row]
                    target_posts_ref = posts_ref[idx_batch]
                    out_dim = target_posts_ref.shape[0]

                    def loop_fn(index, _):
                        offset = col_start + index * block_dim
                        mask = offset + jnp.arange(block_dim) < col_end
                        rows = load(indices_ref, (pl.ds(offset, block_dim),), mask=mask, other=0)
                        val_A = load(data_ref, (pl.ds(offset, block_dim),), mask=mask, other=0.0)

                        # GUARD 2: Indirect Access Boundary Check
                        valid_idx_mask = rows < out_dim
                        final_mask = mask & valid_idx_mask

                        contrib = jnp.where(final_mask, val_A * val_vector, 0.0)
                        atomic_add(target_posts_ref, rows, contrib, mask=final_mask)

                    jax.lax.fori_loop(0, num_blocks, loop_fn, None)

                # GUARD 1: Grid / Indptr Boundary Check using jax.lax.cond
                jax.lax.cond(i_row < num_rows, _body, lambda: None)

            mm_single_impl = mv_transpose_vector_single
            mm_batch_impl = mv_transpose_vector_batch

        def kernel(data, indices, indptr, vector):
            out_info = kwargs['outs'][0]
            placeholder = jnp.zeros(out_info.shape, out_info.dtype)

            has_batch = vector.ndim > 1
            launch_rows = shape[0]  # CSR rows

            if has_batch:
                batch_size = vector.shape[0]
                grid = (batch_size, launch_rows)
                fn = pl.pallas_call(
                    mm_batch_impl,
                    grid=grid,
                    input_output_aliases={4: 0},
                    out_shape=kwargs['outs'],
                    backend='triton'
                )
            else:
                grid = (launch_rows,)
                fn = pl.pallas_call(
                    mm_single_impl,
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

                        val_B = load(vector_ref.at[safe_cols])
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


def _spfloat_csrmv_jax_kernel(
    weight_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs,
):
    """Pure-JAX kernel for sparse-float CSR matrix-vector multiplication.

    Zeros in the input vector naturally contribute 0 to the scatter sum,
    so no explicit zero-skipping is needed in the JAX implementation.
    """
    m, k = shape
    is_homo = (weight_info.size == 1)
    nse = kwargs['indices_info'].size
    out_dtype = kwargs['outs'][0].dtype

    if transpose:
        def kernel(weights, indices, indptr, vector):
            row_ids = jnp.repeat(
                jnp.arange(m, dtype=indptr.dtype),
                jnp.diff(indptr),
                total_repeat_length=nse,
            )
            v_vals = vector[row_ids].astype(out_dtype)
            w = weights[0] if is_homo else weights
            return (jnp.zeros(k, dtype=out_dtype).at[indices].add(w * v_vals),)
    else:
        def kernel(weights, indices, indptr, vector):
            row_ids = jnp.repeat(
                jnp.arange(m, dtype=indptr.dtype),
                jnp.diff(indptr),
                total_repeat_length=nse,
            )
            v_vals = vector[indices].astype(out_dtype)
            w = weights[0] if is_homo else weights
            return (jnp.zeros(m, dtype=out_dtype).at[row_ids].add(w * v_vals),)

    return kernel


def _spfloat_csrmv_cusparse_kernel(
    weight_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs,
):
    """cuSPARSE-backed kernel for sparse-float CSR SpMV via jax.experimental.sparse (GPU only).

    Zero entries in the input vector contribute 0 naturally; no special handling required.
    """
    import jax.experimental.sparse as jsparse
    m, k = shape
    is_homo = (weight_info.size == 1)
    nse = kwargs['indices_info'].size
    out_dtype = kwargs['outs'][0].dtype

    if transpose:
        if is_homo:
            def kernel(weights, indices, indptr, vector):
                ones = jnp.ones(nse, dtype=out_dtype)
                row, col = _csr_to_coo(indices, indptr)
                mat = jsparse.BCOO((ones, jnp.stack([row, col], axis=1)), shape=(m, k))
                return ((mat.T @ vector.astype(out_dtype)) * weights[0].astype(out_dtype),)
        else:
            def kernel(weights, indices, indptr, vector):
                row, col = _csr_to_coo(indices, indptr)
                mat = jsparse.BCOO((weights.astype(out_dtype), jnp.stack([row, col], axis=1)), shape=(m, k))
                return (mat.T @ vector.astype(out_dtype),)
    else:
        if is_homo:
            def kernel(weights, indices, indptr, vector):
                ones = jnp.ones(nse, dtype=out_dtype)
                mat = jsparse.BCSR((ones, indices, indptr), shape=(m, k))
                return ((mat @ vector.astype(out_dtype)) * weights[0].astype(out_dtype),)
        else:
            def kernel(weights, indices, indptr, vector):
                mat = jsparse.BCSR((weights.astype(out_dtype), indices, indptr), shape=(m, k))
                return (mat @ vector.astype(out_dtype),)
    return kernel


def _spfloat_csrmv_cuda_kernel(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs,
):
    indices_info = kwargs.get('indices_info')
    if indices_info is not None and indices_info.dtype != jnp.int32:
        raise ValueError(
            f"CUDA spfloat_csrmv backend requires int32 indices; "
            f"got {indices_info.dtype}.  Pass indices.astype(jnp.int32) "
            f"or select a different backend."
        )

    load_cuda_file(
        Path(__file__).parent.joinpath('sparse_float_csrmv.cu'),
        name='csr_sparse_float_csrmv',
    )

    out_info = kwargs['outs']
    homo = weight_info.shape[0] == 1
    mode_sfx = '_homo' if homo else '_hetero'

    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16',
    }
    wt_sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')

    if transpose:
        kernel_name = f'csr_sparse_float_csrmv.spfloat_csrmv_t{mode_sfx}_warp{wt_sfx}'
    else:
        kernel_name = f'csr_sparse_float_csrmv.spfloat_csrmv_nt{mode_sfx}_auto{wt_sfx}'

    def kernel(weights, indices, indptr, vector):
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, indptr, vector)

    return kernel


def _sparse_float_csrmv_jvp_v(v_dot, data, indices, indptr, v, *, shape, transpose, **kwargs):
    return [csrmv(data, indices, indptr, v_dot, shape=shape, transpose=transpose, backend=kwargs['backend'])]


def _sparse_float_csrmv_jvp_weights(data_dot, data, indices, indptr, v, *, shape, transpose, **kwargs):
    return sparse_float_csrmv_p_call(data_dot, indices, indptr, v,
                                     shape=shape, transpose=transpose, backend=kwargs['backend'])


def _sparse_float_csrmv_transpose_rule(ct, data, indices, indptr, events, *, shape, transpose, **kwargs):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")

    ct = ct[0]

    if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
        raise ValueError("Cannot transpose with respect to sparse indices.")
    if ad.is_undefined_primal(events):
        if type(ct) is ad.Zero:
            ct_events = ad.Zero(events)
        else:
            ct_events = csrmv(
                data,
                indices,
                indptr,
                ct,
                shape=shape,
                transpose=not transpose,
                backend=kwargs['backend']
            )
        return data, indices, indptr, ct_events
    else:
        if type(ct) is ad.Zero:
            ct_values = ad.Zero(data)
        else:
            if data.aval.shape[0] == 1:  # scalar
                ct_values = sparse_float_csrmv_p_call(
                    jnp.ones(1, dtype=data.aval.dtype),
                    indices,
                    indptr,
                    events,
                    shape=shape,
                    transpose=transpose,
                    backend=kwargs['backend']
                )[0]
                ct_values = jnp.inner(ct, ct_values).reshape(*data.aval.shape)
            else:  # heterogeneous values
                row, col = _csr_to_coo(indices, indptr)
                ct_values = events[row] * ct[col] if transpose else events[col] * ct[row]
        return ct_values, indices, indptr, events


def _sparse_float_csrmv_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = sparse_float_csrmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3].T,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend']
        )
        return r, [1]

    elif tuple(axes) == (None, None, None, 1):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = sparse_float_csrmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend']
        )
        return r, [1]

    else:
        return general_batching_rule(spfloat_csrmv_p, args, axes, **kwargs)


def _spfloat_csrmv_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for homo in (True, False):
            n_conn = max(1, int(n_post * prob))
            indptr = np.arange(n_pre + 1, dtype=np.int32) * n_conn
            indices = np.random.randint(0, n_post, (n_pre * n_conn,), dtype=np.int32)
            weights = jnp.ones(1, dtype=dtype) if homo else jnp.ones(n_pre * n_conn, dtype=dtype)
            v_size = n_post if not transpose else n_pre
            vector_data = jnp.asarray(np.random.randn(v_size), dtype=dtype)
            name = f"{'T' if transpose else 'NT'},{'homo' if homo else 'hetero'}"
            configs.append(
                BenchmarkConfig(
                    name,
                    (weights, indices, jnp.asarray(indptr), vector_data),
                    {'shape': (n_pre, n_post), 'transpose': transpose}
                )
            )
    return configs


def sparse_float_csrmv_p_call(
    weights,
    indices,
    indptr,
    vector,
    *,
    shape: MatrixShape,
    transpose: bool,
    backend: Optional[str] = None,
):
    """Invoke the low-level XLA custom kernel for sparse-float CSR matrix-vector multiplication.

    Validates inputs, normalizes scalar weights to a 1-D array, determines the
    output shape, and dispatches to :data:`spfloat_csrmv_p`. Most users should
    prefer the higher-level :func:`spfloat_csrmv` which additionally handles
    physical units.

    Parameters
    ----------
    weights : jax.Array
        Non-zero element values of the CSR sparse matrix, with shape ``(nse,)``
        or scalar. Scalars are promoted to shape ``(1,)``.
    indices : jax.Array
        Column indices of the non-zero elements, with shape ``(nse,)`` and
        integer dtype (``int32``, ``int64``, ``uint32``, or ``uint64``).
    indptr : jax.Array
        Row pointer array, with shape ``(shape[0] + 1,)`` and the same dtype as
        ``indices``.
    vector : jax.Array
        Dense vector, with shape ``(shape[0],)`` if ``transpose=True``, or
        ``(shape[1],)`` otherwise.
    shape : tuple of int
        Shape of the sparse matrix as ``(m, k)``.
    transpose : bool
        If ``True``, compute ``A.T @ v`` instead of ``A @ v``.
    backend : str or None, optional
        Compute backend to use. One of ``'numba'``, ``'pallas'``, or ``None`` for
        automatic selection.

    Returns
    -------
    tuple of jax.Array
        A single-element tuple containing the result vector with shape
        ``(shape[1],)`` if ``transpose=True``, or ``(shape[0],)`` otherwise.

    Raises
    ------
    AssertionError
        If ``indices`` or ``indptr`` have unsupported dtypes, if they are not
        1-D, if their dtypes do not match, or if the vector shape does not match
        ``shape`` given the ``transpose`` flag.

    See Also
    --------
    spfloat_csrmv : High-level wrapper with unit handling.

    Notes
    -----
    Scalar weights (0-D arrays) are promoted to shape ``(1,)`` before dispatch.
    This allows the backend kernels to distinguish between homogeneous
    (``weights.size == 1``) and heterogeneous (``weights.size == nse``) weight
    modes and select optimized code paths accordingly.

    The output shape is determined by the ``transpose`` flag:

    * ``transpose=False``: output shape ``(m,)``
    * ``transpose=True``:  output shape ``(k,)``

    where ``(m, k) = shape``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._csr.sparse_float import sparse_float_csrmv_p_call
        >>> data = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        >>> indices = jnp.array([0, 2, 1], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 2, 3], dtype=jnp.int32)
        >>> v = jnp.array([1.0, 0.5, 0.0], dtype=jnp.float32)
        >>> (y,) = sparse_float_csrmv_p_call(
        ...     data, indices, indptr, v, shape=(2, 3), transpose=False,
        ... )
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
    # Call the spfloat_csrmv_p custom operation to perform the matrix-vector multiplication.
    return spfloat_csrmv_p(
        weights,
        indices,
        indptr,
        vector,
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


spfloat_csrmv_p = XLACustomKernel(
    'sparse_float_csrmv',
    doc="""
Low-level XLA custom-kernel primitive for ``spfloat_csrmv``.

This ``XLACustomKernel`` instance dispatches the CSR sparse matrix-vector multiplication with sparse-float inputs
operation to registered backends (``numba``, ``pallas``),
using runtime shape/dtype metadata provided by the high-level wrapper.

Performs standard sparse matrix-vector multiplication with explicit floating-point weights,
skipping zero-valued entries in the input vector for efficiency.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``spfloat_csrmv_p.available_backends(platform)``,
and the default backend can be configured with ``spfloat_csrmv_p.set_default(platform, backend)``.

See Also
--------
spfloat_csrmv : High-level user-facing function wrapper.
"""
)
spfloat_csrmv_p.def_numba_kernel(_sparse_float_csrmv_numba_kernel)
spfloat_csrmv_p.def_pallas_kernel('gpu', _sparse_float_csrmv_pallas_kernel)
spfloat_csrmv_p.def_pallas_kernel('tpu', _sparse_float_csrmv_pallas_kernel)
spfloat_csrmv_p.def_cuda_raw_kernel(_spfloat_csrmv_cuda_kernel)
spfloat_csrmv_p.def_kernel('jax_raw', 'cpu', _spfloat_csrmv_jax_kernel)
spfloat_csrmv_p.def_kernel('jax_raw', 'gpu', _spfloat_csrmv_jax_kernel)
spfloat_csrmv_p.def_kernel('jax_raw', 'tpu', _spfloat_csrmv_jax_kernel)
spfloat_csrmv_p.def_kernel('cusparse', 'gpu', _spfloat_csrmv_cusparse_kernel)
spfloat_csrmv_p.def_jvp_rule2(_sparse_float_csrmv_jvp_weights, None, None, _sparse_float_csrmv_jvp_v)
spfloat_csrmv_p.def_transpose_rule(_sparse_float_csrmv_transpose_rule)
spfloat_csrmv_p.def_batching_rule(_sparse_float_csrmv_batching)
spfloat_csrmv_p.def_call(sparse_float_csrmv_p_call)
spfloat_csrmv_p.def_tags('csr', 'sparse_float')
spfloat_csrmv_p.def_benchmark_data(_spfloat_csrmv_benchmark_data)


def _sparse_float_csrmm_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
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
            @numba.njit(parallel=get_numba_parallel(), fastmath=True)
            def mm(weights, indices, indptr, B, posts):
                posts[:] = 0.
                w = weights[0]
                for k in numba.prange(B.shape[1]):
                    for i in range(B.shape[0]):
                        sp = B[i, k]
                        if sp != 0.:
                            wsp = w * sp
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j], k] += wsp

        else:
            # csr @ B
            @numba.njit(parallel=get_numba_parallel(), fastmath=True)
            def mm(weights, indices, indptr, B, posts):
                w = weights[0]
                for i in numba.prange(indptr.shape[0] - 1):
                    for k in range(B.shape[1]):
                        r = 0.
                        for j in range(indptr[i], indptr[i + 1]):
                            c = B[indices[j], k]
                            if c != 0.:
                                r += w * c
                        posts[i, k] = r

    else:
        if transpose:
            # csr.T @ B
            @numba.njit(parallel=get_numba_parallel(), fastmath=True)
            def mm(weights, indices, indptr, B, posts):
                posts[:] = 0.
                for k in numba.prange(B.shape[1]):
                    for i in range(B.shape[0]):
                        sp = B[i, k]
                        if sp != 0.:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j], k] += weights[j] * sp

        else:
            # csr @ B
            @numba.njit(parallel=get_numba_parallel(), fastmath=True)
            def mm(weights, indices, indptr, B, posts):
                for i in numba.prange(indptr.shape[0] - 1):
                    for k in range(B.shape[1]):
                        r = 0.
                        for j in range(indptr[i], indptr[i + 1]):
                            c = B[indices[j], k]
                            if c != 0.:
                                r += weights[j] * c
                        posts[i, k] = r

    def kernel(weights, indices, indptr, B):
        return numba_kernel(mm, outs=kwargs['outs'])(weights, indices, indptr, B)

    return kernel


def _sparse_float_csrmm_pallas_kernel(
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

                indptr_start = indptr_ref[i_k]
                indptr_end = indptr_ref[i_k + 1]

                def _body():
                    i_n_block = pl.program_id(1)
                    i_n_start = i_n_block * block_dim_n

                    mask = (i_n_start + jnp.arange(block_dim_n)) < B_ref.shape[1]

                    B_row = B_ref[i_k, pl.ds(i_n_start, block_dim_n)]
                    # B_row = pl.load(B_ref, (pl.ds(i_n_start, block_dim_n),), mask=mask, other=0.0)

                    val = B_row * data_ref[0]

                    mask = mask & (B_row != 0.)

                    out_rows = posts_ref.shape[0]

                    def loop_fn(index, _):
                        i_row = indices_ref[index]

                        # Indirect Write Guard (Scatter)
                        row_valid = i_row < out_rows
                        final_mask = mask & row_valid

                        atomic_add(posts_ref, (i_row, pl.ds(i_n_start, block_dim_n)), val, mask=final_mask)

                    jax.lax.fori_loop(indptr_start, indptr_end, loop_fn, None)

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
                i_n_block = pl.program_id(1)

                def _body():
                    i_n_start = i_n_block * block_dim_n

                    mask = (i_n_start + jnp.arange(block_dim_n)) < B_ref.shape[1]
                    B_row = B_ref[i_k, pl.ds(i_n_start, block_dim_n)]
                    # B_row = pl.load(B_ref, (pl.ds(i_n_start, block_dim_n),), mask=mask, other=0.0)

                    out_rows = posts_ref.shape[0]

                    mask = mask & (B_row != 0.)

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
        # Gustavson algorithm: Sparse matrix-matrix multiplication is performed in a row-wise fashion.
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

                    row_start, row_end = indptr_ref[i_m], indptr_ref[i_m + 1]
                    weight = data_ref[0]

                    b_rows = B_ref.shape[0]

                    def loop_fn(index, out):
                        i_k = indices_ref[index]

                        # Indirect Read Guard (Gather)
                        safe_k = jnp.minimum(i_k, b_rows - 1)
                        k_valid = i_k < b_rows
                        final_mask = mask & k_valid

                        # B_row = pl.load(B_ref, (safe_k, pl.ds(i_n_start, block_dim_n),), mask=final_mask, other=0.0)
                        B_row = B_ref[safe_k, pl.ds(i_n_start, block_dim_n)]
                        B_row = jnp.where(final_mask, B_row, 0.0)
                        out += weight * B_row
                        return out

                    i_row_out = jax.lax.fori_loop(
                        row_start,
                        row_end,
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
                i_n_block = pl.program_id(1)

                def _body():
                    i_n_start = i_n_block * block_dim_n
                    mask = (i_n_start + jnp.arange(block_dim_n)) < B_ref.shape[1]

                    b_rows = B_ref.shape[0]

                    row_start, row_end = indptr_ref[i_m], indptr_ref[i_m + 1]

                    def loop_fn(index, out):
                        i_col = indices_ref[index]
                        val_A = data_ref[index]

                        # Indirect Read Guard (Gather)
                        safe_col = jnp.minimum(i_col, b_rows - 1)
                        col_valid = i_col < b_rows
                        final_mask = mask & col_valid

                        # val_B = pl.load(B_ref, (safe_col, pl.ds(i_n_start, block_dim_n),), mask=final_mask, other=0.0)

                        val_B = B_ref[safe_col, pl.ds(i_n_start, block_dim_n)]
                        val_B = jnp.where(final_mask, val_B, 0.0)
                        out += val_A * val_B
                        return out

                    i_row_out = jax.lax.fori_loop(
                        row_start,
                        row_end,
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


def _spfloat_csrmm_cuda_kernel(
    transpose: bool,
    weight_info: jax.ShapeDtypeStruct,
    **kwargs
):
    indices_info = kwargs.get('indices_info')
    if indices_info is not None and indices_info.dtype != jnp.int32:
        raise ValueError(
            f"CUDA spfloat_csrmm backend requires int32 indices; "
            f"got {indices_info.dtype}.  Pass indices.astype(jnp.int32) "
            f"or select a different backend."
        )

    load_cuda_file(
        Path(__file__).parent.joinpath('sparse_float_csrmm.cu'),
        name='csr_sparse_float_csrmm',
    )

    out_info = kwargs['outs']
    homo = weight_info.shape[0] == 1
    mode_sfx = '_homo' if homo else '_hetero'

    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16',
    }
    wt_sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')

    if transpose:
        kernel_name = f'csr_sparse_float_csrmm.spfloat_csrmm_t{mode_sfx}_warp{wt_sfx}'
    else:
        kernel_name = f'csr_sparse_float_csrmm.spfloat_csrmm_nt{mode_sfx}_auto{wt_sfx}'

    def kernel(weights, indices, indptr, B):
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, indptr, B)

    return kernel


def _spfloat_csrmm_jax_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs,
):
    """Pure-JAX kernel for sparse-float CSR matrix-matrix multiplication.

    Zeros in B naturally contribute 0 to the scatter sum.
    """
    m, k = shape
    n = vector_info.shape[1]
    is_homo = (weight_info.size == 1)
    nse = kwargs['indices_info'].size
    out_dtype = kwargs['outs'][0].dtype

    if transpose:
        def kernel(weights, indices, indptr, B):
            row_ids = jnp.repeat(
                jnp.arange(m, dtype=indptr.dtype),
                jnp.diff(indptr),
                total_repeat_length=nse,
            )
            B_rows = B[row_ids].astype(out_dtype)  # [nse, n]
            w = weights[0] if is_homo else weights[:, None]
            return (jnp.zeros((k, n), dtype=out_dtype).at[indices].add(w * B_rows),)
    else:
        def kernel(weights, indices, indptr, B):
            row_ids = jnp.repeat(
                jnp.arange(m, dtype=indptr.dtype),
                jnp.diff(indptr),
                total_repeat_length=nse,
            )
            B_rows = B[indices].astype(out_dtype)  # [nse, n]
            w = weights[0] if is_homo else weights[:, None]
            return (jnp.zeros((m, n), dtype=out_dtype).at[row_ids].add(w * B_rows),)

    return kernel


def _spfloat_csrmm_cusparse_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs,
):
    """cuSPARSE-backed kernel for sparse-float CSR SpMM via jax.experimental.sparse (GPU only).

    Zero entries in B contribute 0 naturally; no special handling required.
    """
    import jax.experimental.sparse as jsparse
    m, k = shape
    is_homo = (weight_info.size == 1)
    nse = kwargs['indices_info'].size
    out_dtype = kwargs['outs'][0].dtype

    if transpose:
        if is_homo:
            def kernel(weights, indices, indptr, B):
                ones = jnp.ones(nse, dtype=out_dtype)
                row, col = _csr_to_coo(indices, indptr)
                mat = jsparse.BCOO((ones, jnp.stack([row, col], axis=1)), shape=(m, k))
                return ((mat.T @ B.astype(out_dtype)) * weights[0].astype(out_dtype),)
        else:
            def kernel(weights, indices, indptr, B):
                row, col = _csr_to_coo(indices, indptr)
                mat = jsparse.BCOO((weights.astype(out_dtype), jnp.stack([row, col], axis=1)), shape=(m, k))
                return (mat.T @ B.astype(out_dtype),)
    else:
        if is_homo:
            def kernel(weights, indices, indptr, B):
                ones = jnp.ones(nse, dtype=out_dtype)
                mat = jsparse.BCSR((ones, indices, indptr), shape=(m, k))
                return ((mat @ B.astype(out_dtype)) * weights[0].astype(out_dtype),)
        else:
            def kernel(weights, indices, indptr, B):
                mat = jsparse.BCSR((weights.astype(out_dtype), indices, indptr), shape=(m, k))
                return (mat @ B.astype(out_dtype),)
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
            r = sparse_float_csrmm_p_call(
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
            row, col = _csr_to_coo(indices, indptr)
            if transpose:
                d_data = sddmm_coo_indices(B, ct.T, row, col).data
            else:
                d_data = sddmm_coo_indices(B, ct.T, col, row).data
            return d_data, indices, indptr, B


def _sparse_float_csrmm_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[3].shape
        B = jnp.transpose(args[3], (1, 0, 2)).reshape(m, batch_size * n)
        r = sparse_float_csrmm_p_call(
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
        r = sparse_float_csrmm_p_call(
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
        r = sparse_float_csrmm_p_call(
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
        return general_batching_rule(spfloat_csrmm_p, args, axes, **kwargs)


def _spfloat_csrmm_benchmark_data(*, platform):
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


def sparse_float_csrmm_p_call(
    weights,
    indices,
    indptr,
    B,
    *,
    shape: MatrixShape,
    transpose: bool,
    backend: Optional[str] = None,
):
    """Invoke the low-level XLA custom kernel for sparse-float CSR matrix-matrix multiplication.

    Validates inputs, normalizes scalar weights to a 1-D array, determines the
    output shape, and dispatches to :data:`spfloat_csrmm_p`. Most users should
    prefer the higher-level :func:`spfloat_csrmm` which additionally handles
    physical units.

    Parameters
    ----------
    weights : jax.Array
        Non-zero element values of the CSR sparse matrix, with shape ``(nse,)``
        or scalar. Scalars are promoted to shape ``(1,)``.
    indices : jax.Array
        Column indices of the non-zero elements, with shape ``(nse,)`` and
        integer dtype (``int32``, ``int64``, ``uint32``, or ``uint64``).
    indptr : jax.Array
        Row pointer array, with shape ``(shape[0] + 1,)`` and the same dtype as
        ``indices``.
    B : jax.Array
        Dense matrix to multiply, with shape ``(shape[0], n)`` if
        ``transpose=True``, or ``(shape[1], n)`` otherwise.
    shape : tuple of int
        Shape of the sparse matrix as ``(m, k)``.
    transpose : bool
        If ``True``, compute ``A.T @ B`` instead of ``A @ B``.
    backend : str or None, optional
        Compute backend to use. One of ``'numba'``, ``'pallas'``, or ``None`` for
        automatic selection.

    Returns
    -------
    tuple of jax.Array
        A single-element tuple containing the result matrix with shape
        ``(shape[1], n)`` if ``transpose=True``, or ``(shape[0], n)`` otherwise.

    Raises
    ------
    AssertionError
        If ``indices`` or ``indptr`` have unsupported dtypes, if they are not
        1-D, if their dtypes do not match, or if the first dimension of ``B``
        does not match ``shape`` given the ``transpose`` flag.

    See Also
    --------
    spfloat_csrmm : High-level wrapper with unit handling.

    Notes
    -----
    Scalar weights (0-D arrays) are promoted to shape ``(1,)`` before dispatch.
    This allows the backend kernels to distinguish between homogeneous
    (``weights.size == 1``) and heterogeneous (``weights.size == nse``) weight
    modes and select optimized code paths accordingly.

    The output shape is determined by the ``transpose`` flag:

    * ``transpose=False``: output shape ``(m, n)``
    * ``transpose=True``:  output shape ``(k, n)``

    where ``(m, k) = shape`` and ``n = B.shape[1]``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._csr.sparse_float import sparse_float_csrmm_p_call
        >>> data = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        >>> indices = jnp.array([0, 2, 1], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 2, 3], dtype=jnp.int32)
        >>> B = jnp.ones((3, 4), dtype=jnp.float32)
        >>> (C,) = sparse_float_csrmm_p_call(
        ...     data, indices, indptr, B, shape=(2, 3), transpose=False,
        ... )
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
    # Call the spfloat_csrmm_p custom operation to perform the matrix-matrix multiplication.
    return spfloat_csrmm_p(
        weights,
        indices,
        indptr,
        B,
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
        # Provide shape and data type information for B.
        vector_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
    )


spfloat_csrmm_p = XLACustomKernel(
    'sparse_float_csrmm',
    doc="""
Low-level XLA custom-kernel primitive for ``spfloat_csrmm``.

This ``XLACustomKernel`` instance dispatches the CSR sparse matrix-matrix multiplication with sparse-float inputs
operation to registered backends (``numba``, ``pallas``),
using runtime shape/dtype metadata provided by the high-level wrapper.

Performs standard sparse matrix-matrix multiplication with explicit floating-point weights,
skipping zero-valued entries in the input matrix for efficiency.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``spfloat_csrmm_p.available_backends(platform)``,
and the default backend can be configured with ``spfloat_csrmm_p.set_default(platform, backend)``.

See Also
--------
spfloat_csrmm : High-level user-facing function wrapper.
"""
)
spfloat_csrmm_p.def_numba_kernel(_sparse_float_csrmm_numba_kernel)
spfloat_csrmm_p.def_pallas_kernel('gpu', _sparse_float_csrmm_pallas_kernel)
spfloat_csrmm_p.def_pallas_kernel('tpu', _sparse_float_csrmm_pallas_kernel)
spfloat_csrmm_p.def_cuda_raw_kernel(_spfloat_csrmm_cuda_kernel)
spfloat_csrmm_p.def_kernel('jax_raw', 'cpu', _spfloat_csrmm_jax_kernel)
spfloat_csrmm_p.def_kernel('jax_raw', 'gpu', _spfloat_csrmm_jax_kernel)
spfloat_csrmm_p.def_kernel('jax_raw', 'tpu', _spfloat_csrmm_jax_kernel)
spfloat_csrmm_p.def_kernel('cusparse', 'gpu', _spfloat_csrmm_cusparse_kernel)
spfloat_csrmm_p.def_jvp_rule2(_csrmm_jvp_data, None, None, _csrmm_jvp_B)
spfloat_csrmm_p.def_transpose_rule(_csrmm_transpose_rule)
spfloat_csrmm_p.def_batching_rule(_sparse_float_csrmm_batching)
spfloat_csrmm_p.def_call(sparse_float_csrmm_p_call)
spfloat_csrmm_p.def_tags('csr', 'sparse_float')
spfloat_csrmm_p.def_benchmark_data(_spfloat_csrmm_benchmark_data)
