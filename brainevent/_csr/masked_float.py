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


from typing import Sequence

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._compatible_import import pallas as pl
from brainevent._misc import _csr_to_coo, generate_block_dim, namescoped_jit
from brainevent._typing import Data, Indptr, Index, MatrixShape
from brainevent._op.main import XLACustomKernel
from brainevent._op.op_numba import numba_kernel
from brainevent._op.op_pallas import pallas_kernel
from brainevent._op.util import general_batching_rule
from brainevent._op.op_warp import jaxtype_to_warptype, warp_kernel
from .float import csr_matvec, csr_matmat


@namescoped_jit(static_argnames=("shape", "transpose"))
def masked_float_csr_matvec(
    data: Data,
    indices: Index,
    indptr: Indptr,
    v: Data,
    *,
    shape: MatrixShape,
    transpose: bool = False,
) -> Data:
    """
    Product of CSR sparse matrix and a dense vector.

    Args:
      data : array of shape ``(nse,)``.
      indices : array of shape ``(nse,)``
      indptr : array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``
      v : array of shape ``(shape[0] if transpose else shape[1],)``
        and dtype ``data.dtype``
      shape : length-2 tuple representing the matrix shape
      transpose : boolean specifying whether to transpose the sparse matrix
        before computing.

    Returns:
      y : array of shape ``(shape[1] if transpose else shape[0],)`` representing
        the matrix vector product.
    """
    data, unitd = u.split_mantissa_unit(data)
    v, unitv = u.split_mantissa_unit(v)
    res = masked_float_csrmv_p_call(
        data,
        indices,
        indptr,
        v,
        shape=shape,
        transpose=transpose,
    )[0]
    return u.maybe_decimal(res * (unitd * unitv))


@namescoped_jit(static_argnames=("shape", "transpose"))
def masked_float_csr_matmat(
    data: Data,
    indices: Index,
    indptr: Indptr,
    B: Data,
    *,
    shape: MatrixShape,
    transpose: bool = False,
) -> Data:
    """
    Product of CSR sparse matrix and a dense matrix.

    Args:
      data : array of shape ``(nse,)``.
      indices : array of shape ``(nse,)``
      indptr : array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``
      B : array of shape ``(shape[0] if transpose else shape[1], cols)`` and
        dtype ``data.dtype``
      shape : length-2 tuple representing the matrix shape
      transpose : boolean specifying whether to transpose the sparse matrix
        before computing.

    Returns:
      C : array of shape ``(shape[1] if transpose else shape[0], cols)``
        representing the matrix-matrix product.
    """
    data, unitd = u.split_mantissa_unit(data)
    B, unitb = u.split_mantissa_unit(B)
    res = masked_float_csrmm_p_call(
        data,
        indices,
        indptr,
        B,
        shape=shape,
        transpose=transpose,
    )[0]
    return u.maybe_decimal(res * (unitd * unitb))


def _masked_float_csrmv_numba_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    if weight_info.size == 1:
        if transpose:
            @numba_kernel(parallel=False, input_output_aliases={4: 0})
            def mv(weights, indices, indptr, v, _, posts):
                w = weights[0]
                for i in range(v.shape[0]):
                    sp = v[i]
                    if sp != 0.:
                        wsp = w * sp
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j]] += wsp

        else:

            @numba_kernel(parallel=False, input_output_aliases={4: 0})
            def mv(weights, indices, indptr, v, _, posts):
                w = weights[0]
                for i in range(indptr.shape[0] - 1):
                    r = np.asarray(0., dtype=posts.dtype)
                    for j in range(indptr[i], indptr[i + 1]):
                        c = v[indices[j]]
                        if c != 0.:
                            r += w * c
                    posts[i] = r

    else:
        if transpose:
            @numba_kernel(parallel=False, input_output_aliases={4: 0})
            def mv(weights, indices, indptr, v, _, posts):
                for i in range(v.shape[0]):
                    sp = v[i]
                    if sp != 0.:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j]] += weights[j] * sp

        else:
            @numba_kernel(parallel=False, input_output_aliases={4: 0})
            def mv(weights, indices, indptr, v, _, posts):
                for i in range(indptr.shape[0] - 1):
                    r = np.asarray(0., dtype=posts.dtype)
                    for j in range(indptr[i], indptr[i + 1]):
                        c = v[indices[j]]
                        if c != 0.:
                            r += weights[j] * c
                    posts[i] = r

    return mv


def _masked_float_csrmv_warp_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    indptr_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import warp  # pylint: disable=import-outside-toplevel

    weight_dtype = jaxtype_to_warptype(weight_info.dtype)
    indices_dtype = jaxtype_to_warptype(indices_info.dtype)
    indptr_dtype = jaxtype_to_warptype(indptr_info.dtype)
    spike_dtype = jaxtype_to_warptype(vector_info.dtype)

    if weight_info.size == 1:
        if transpose:
            def mv(
                weights: warp.array1d(dtype=weight_dtype),
                indices: warp.array1d(dtype=indices_dtype),
                indptr: warp.array1d(dtype=indptr_dtype),
                v: warp.array1d(dtype=spike_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype),
            ):
                i = warp.tid()
                w = weights[0]
                sp = v[i]
                if sp != 0.:
                    wsp = w * sp
                    for j in range(indptr[i], indptr[i + 1]):
                        posts[indices[j]] += wsp

        else:
            def mv(
                weights: warp.array1d(dtype=weight_dtype),
                indices: warp.array1d(dtype=indices_dtype),
                indptr: warp.array1d(dtype=indptr_dtype),
                v: warp.array1d(dtype=spike_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype),
            ):
                i = warp.tid()
                w = weights[0]
                r = weights.dtype(0.)
                for j in range(indptr[i], indptr[i + 1]):
                    c = v[indices[j]]
                    if c != 0.:
                        r += w * c
                posts[i] = r

    else:
        if transpose:
            def mv(
                weights: warp.array1d(dtype=weight_dtype),
                indices: warp.array1d(dtype=indices_dtype),
                indptr: warp.array1d(dtype=indptr_dtype),
                v: warp.array1d(dtype=spike_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype),
            ):
                i = warp.tid()
                sp = v[i]
                if sp != 0.:
                    for j in range(indptr[i], indptr[i + 1]):
                        posts[indices[j]] += weights[j] * sp

        else:
            def mv(
                weights: warp.array1d(dtype=weight_dtype),
                indices: warp.array1d(dtype=indices_dtype),
                indptr: warp.array1d(dtype=indptr_dtype),
                v: warp.array1d(dtype=spike_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype),
            ):
                i = warp.tid()
                r = weights.dtype(0.)
                for j in range(indptr[i], indptr[i + 1]):
                    c = v[indices[j]]
                    if c != 0.:
                        r += weights[j] * c
                posts[i] = r
    dim = (
        vector_info.shape[0] if transpose else indptr_info.shape[0] - 1
    )
    return warp_kernel(mv, dim=dim, input_output_aliases={4: 0})


def _masked_float_csrmv_pallas_tiled_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs
):
    m, k = shape
    block_dim = generate_block_dim(pl.cdiv(indices_info.size, shape[1] if transpose else shape[0]))
    block_dim = block_dim // 2
    block_dim = 32 if block_dim < 32 else block_dim

    if weight_info.size == 1:
        if transpose:
            # csr.T @ B
            #
            # csr: [k, m]
            # B: [k]
            #
            def mm(
                data_ref,  # [1]
                indices_ref,  # [nse]
                indptr_ref,  # [k + 1]
                vector_ref,  # [k]
                _,  # [m]
                posts_ref,  # [m]
            ):
                i_col = pl.program_id(0)
                col_start = indptr_ref[i_col]
                col_end = indptr_ref[i_col + 1]
                col_nnz = col_end - col_start
                num_blocks = (col_nnz + block_dim - 1) // block_dim
                event = vector_ref[i_col]
                weight = data_ref[0]
                data = jnp.ones((block_dim,), dtype=posts_ref.dtype) * (weight * event)

                @pl.when(event if vector_ref.dtype == jnp.bool_ else event != 0.)
                def event_processing():
                    def loop_fn(index, _):
                        offset = col_start + index * block_dim
                        mask = offset + jnp.arange(block_dim) < col_end
                        rows = pl.load(indices_ref, pl.dslice(offset, block_dim), mask=mask)
                        pl.atomic_add(posts_ref, rows, data, mask=mask)

                    jax.lax.fori_loop(0, num_blocks, loop_fn, None)

        else:
            # csr @ B
            #
            # csr: [m, k]
            # B: [k]
            #
            def mm(
                data_ref,  # [1]
                indices_ref,  # [nse]
                indptr_ref,  # [m + 1]
                vector_ref,  # [k, n]
                _,  # [m, ]
                posts_ref,  # [m, ]
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

                    cols = pl.load(indices_ref, pl.dslice(offset, block_dim), mask=mask)
                    events = pl.load(vector_ref, cols, mask=mask)
                    events = jnp.asarray(events, dtype=posts_ref.dtype)
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
        if transpose:
            # csr.T @ B
            #
            # csr: [k, m]
            # B: [k, ]
            #
            def mm(
                data_ref,  # [nse]
                indices_ref,  # [nse]
                indptr_ref,  # [k + 1]
                vector_ref,  # [k]
                _,  # [m]
                posts_ref,  # [m]
            ):
                i_col = pl.program_id(0)
                col_start = indptr_ref[i_col]
                col_end = indptr_ref[i_col + 1]
                col_nnz = col_end - col_start
                num_blocks = (col_nnz + block_dim - 1) // block_dim
                event = vector_ref[i_col]

                @pl.when(event if vector_ref.dtype == jnp.bool_ else event != 0.)
                def event_processing():
                    def loop_fn(index, _):
                        offset = col_start + index * block_dim
                        mask = offset + jnp.arange(block_dim) < col_end
                        rows = pl.load(indices_ref, pl.dslice(offset, block_dim), mask=mask)
                        val_A = pl.load(data_ref, pl.dslice(offset, block_dim), mask=mask, other=0.0)
                        contrib = val_A * event
                        pl.atomic_add(posts_ref, rows, contrib, mask=mask)

                    jax.lax.fori_loop(0, num_blocks, loop_fn, None)

        else:
            # csr @ B
            #
            # csr: [m, k]
            # B: [k]
            #
            def mm(
                data_ref,  # [nse]
                indices_ref,  # [nse]
                indptr_ref,  # [m + 1]
                vector_ref,  # [k, n]
                _,  # [m, ]
                posts_ref,  # [m, ]
            ):
                i_row = pl.program_id(0)
                row_start = indptr_ref[i_row]
                row_end = indptr_ref[i_row + 1]
                row_nnz = row_end - row_start
                num_blocks = (row_nnz + block_dim - 1) // block_dim

                def loop_fn(index, sum_):
                    offset = row_start + index * block_dim
                    mask = offset + jnp.arange(block_dim) < row_end

                    cols = pl.load(indices_ref, pl.dslice(offset, block_dim), mask=mask)
                    val_A = pl.load(data_ref, pl.dslice(offset, block_dim), mask=mask, other=0.0)
                    events = pl.load(vector_ref, cols, mask=mask)
                    events = jnp.asarray(events, dtype=posts_ref.dtype)
                    sum_ += jnp.sum(val_A * events)
                    return sum_

                i_row_sum = jax.lax.fori_loop(
                    0,
                    num_blocks,
                    loop_fn,
                    jnp.asarray(0., dtype=posts_ref.dtype)
                )
                posts_ref[i_row] = i_row_sum

    return pallas_kernel(
        mm,
        outs=kwargs['outs'],
        tile=(k if transpose else m,),
        input_output_aliases={4: 0},
    )


def _masked_float_csrmv_jvp_v(
    v_dot,
    data,
    indices,
    indptr,
    v,
    _,
    *,
    shape,
    transpose,
    **kwargs
):
    return [
        csr_matvec(
            data,
            indices,
            indptr,
            v_dot,
            shape=shape,
            transpose=transpose
        )
    ]


def _masked_float_csrmv_jvp_weights(
    data_dot,
    data,
    indices,
    indptr,
    v,
    _,
    *,
    shape,
    transpose,
    **kwargs
):
    return masked_float_csrmv_p_call(
        data_dot,
        indices,
        indptr,
        v,
        shape=shape,
        transpose=transpose,
    )


def _masked_float_csrmv_transpose_rule(
    ct,
    data,
    indices,
    indptr,
    events,
    _,
    *,
    shape,
    transpose,
    **kwargs
):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")

    ct = ct[0]

    if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
        raise ValueError("Cannot transpose with respect to sparse indices.")
    if ad.is_undefined_primal(events):
        if type(ct) is ad.Zero:
            ct_events = ad.Zero(events)
        else:
            ct_events = csr_matvec(
                data,
                indices,
                indptr,
                ct,
                shape=shape,
                transpose=not transpose
            )
        return data, indices, indptr, ct_events, _
    else:
        if type(ct) is ad.Zero:
            ct_values = ad.Zero(data)
        else:
            if data.aval.shape[0] == 1:  # scalar
                ct_values = masked_float_csrmv_p_call(
                    jnp.ones(1, dtype=data.aval.dtype),
                    indices,
                    indptr,
                    events,
                    shape=shape,
                    transpose=transpose,
                )[0]
                ct_values = jnp.inner(ct, ct_values).reshape(*data.aval.shape)
            else:  # heterogeneous values
                row, col = _csr_to_coo(indices, indptr)
                ct_values = events[row] * ct[col] if transpose else events[col] * ct[row]
        return ct_values, indices, indptr, events, _


def _masked_float_csrmv_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = masked_float_csrmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3].T,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
        )
        return r, [1]

    elif tuple(axes) == (None, None, None, 1, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = masked_float_csrmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
        )
        return r, [1]

    else:
        return general_batching_rule(masked_float_csrmv_p, args, axes, **kwargs)


def masked_float_csrmv_p_call(
    weights,
    indices,
    indptr,
    vector,
    *,
    shape: MatrixShape,
    transpose: bool,
):
    """
    Perform a call to the event CSR matrix-vector multiplication custom operation.

    This function prepares the inputs and calls the masked_float_csrmv_p custom operation
    to perform matrix-vector multiplication using a CSR (Compressed Sparse Row) format.

    Args:
        weights (jax.Array): Non-zero elements of the CSR sparse matrix.
        indices (jax.Array): Column indices of non-zero elements in the CSR sparse matrix.
        indptr (jax.Array): Index pointers of the CSR sparse matrix, indicating the start of each row.
        vector (jax.Array): The dense vector to be multiplied with the sparse matrix.
        shape (Sequence[int]): A sequence of length 2, representing the shape of the sparse matrix.
        transpose (bool): Whether to transpose the sparse matrix before multiplication.

    Returns:
        jax.Array: The result of the matrix-vector multiplication.
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
    # Call the masked_float_csrmv_p custom operation to perform the matrix-vector multiplication.
    return masked_float_csrmv_p(
        weights,
        indices,
        indptr,
        vector,
        # Initialize a zero vector with the output shape and data type.
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        shape=shape,
        transpose=transpose,
        # Provide shape and data type information for indices.
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        # Provide shape and data type information for indptr.
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        # Provide shape and data type information for weights.
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        # Provide shape and data type information for v.
        vector_info=jax.ShapeDtypeStruct(vector.shape, vector.dtype),
    )


masked_float_csrmv_p = XLACustomKernel('masked_float_csrmv')
masked_float_csrmv_p.def_cpu_kernel(_masked_float_csrmv_numba_kernel_generator)
masked_float_csrmv_p.def_gpu_kernel(
    default='pallas',
    warp=_masked_float_csrmv_warp_kernel_generator,
    pallas=_masked_float_csrmv_pallas_tiled_kernel_generator,
)
masked_float_csrmv_p.def_tpu_kernel(_masked_float_csrmv_pallas_tiled_kernel_generator)
masked_float_csrmv_p.def_jvp_rule2(_masked_float_csrmv_jvp_weights, None, None, _masked_float_csrmv_jvp_v)
masked_float_csrmv_p.def_transpose_rule(_masked_float_csrmv_transpose_rule)
masked_float_csrmv_p.def_batching_rule(_masked_float_csrmv_batching)


def _masked_float_csrmm_numba_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba  # pylint: disable=import-outside-toplevel

    if weight_info.size == 1:
        if transpose:
            #
            # csr.T @ B
            #
            # [k, m] @ [k, n]
            #
            @numba_kernel(parallel=True, input_output_aliases={4: 0})
            def mv(weights, indices, indptr, B, _, posts):
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
            @numba_kernel(parallel=True, input_output_aliases={4: 0})
            def mv(weights, indices, indptr, B, _, posts):
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
            @numba_kernel(parallel=True, input_output_aliases={4: 0})
            def mv(weights, indices, indptr, B, _, posts):
                for k in numba.prange(B.shape[1]):
                    for i in range(B.shape[0]):
                        sp = B[i, k]
                        if sp != 0.:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j], k] += weights[j] * sp

        else:
            # csr @ B
            @numba_kernel(parallel=False, input_output_aliases={4: 0})
            def mv(weights, indices, indptr, B, _, posts):
                for i in range(indptr.shape[0] - 1):
                    for k in range(B.shape[1]):
                        r = 0.
                        for j in range(indptr[i], indptr[i + 1]):
                            c = B[indices[j], k]
                            if c != 0.:
                                r += weights[j] * c
                        posts[i, k] = r

    return mv


def _masked_float_csrmm_warp_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    indptr_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import warp  # pylint: disable=import-outside-toplevel

    weight_dtype = jaxtype_to_warptype(weight_info.dtype)
    spike_dtype = jaxtype_to_warptype(vector_info.dtype)
    indices_dtype = jaxtype_to_warptype(indices_info.dtype)
    indptr_dtype = jaxtype_to_warptype(indptr_info.dtype)

    if weight_info.size == 1:
        if transpose:
            # csr.T @ B

            def mm(
                weights: warp.array1d(dtype=weight_dtype),
                indices: warp.array1d(dtype=indices_dtype),
                indptr: warp.array1d(dtype=indptr_dtype),
                B: warp.array2d(dtype=spike_dtype),
                _: warp.array2d(dtype=weight_dtype),
                posts: warp.array2d(dtype=weight_dtype),
            ):
                k, i = warp.tid()
                w = weights[0]
                sp = B[i, k]
                if sp != 0.:
                    wsp = w * sp
                    for j in range(indptr[i], indptr[i + 1]):
                        posts[indices[j], k] += wsp

        else:
            # csr @ B
            def mm(
                weights: warp.array1d(dtype=weight_dtype),
                indices: warp.array1d(dtype=indices_dtype),
                indptr: warp.array1d(dtype=indptr_dtype),
                B: warp.array2d(dtype=spike_dtype),
                _: warp.array2d(dtype=weight_dtype),
                posts: warp.array2d(dtype=weight_dtype),
            ):
                k, i = warp.tid()
                w = weights[0]
                r = weights.dtype(0.)
                for j in range(indptr[i], indptr[i + 1]):
                    index = indices[j]
                    c = B[index, k]
                    if c != 0.:
                        r += w * c
                posts[i, k] = r

    else:
        if transpose:
            # csr.T @ B

            def mm(
                weights: warp.array1d(dtype=weight_dtype),
                indices: warp.array1d(dtype=indices_dtype),
                indptr: warp.array1d(dtype=indptr_dtype),
                B: warp.array2d(dtype=spike_dtype),
                _: warp.array2d(dtype=weight_dtype),
                posts: warp.array2d(dtype=weight_dtype),
            ):
                k, i = warp.tid()
                sp = B[i, k]
                if sp != 0.:
                    for j in range(indptr[i], indptr[i + 1]):
                        posts[indices[j], k] += weights[j] * sp

        else:
            # csr @ B

            def mm(
                weights: warp.array1d(dtype=weight_dtype),
                indices: warp.array1d(dtype=indices_dtype),
                indptr: warp.array1d(dtype=indptr_dtype),
                B: warp.array2d(dtype=spike_dtype),
                _: warp.array2d(dtype=weight_dtype),
                posts: warp.array2d(dtype=weight_dtype),
            ):
                k, i = warp.tid()
                r = weights.dtype(0.)
                for j in range(indptr[i], indptr[i + 1]):
                    index = indices[j]
                    c = B[index, k]
                    if c != 0.:
                        r += weights[j] * c
                posts[i, k] = r

    dim = (
        tuple(reversed(vector_info.shape)) if transpose else
        [vector_info.shape[1], indptr_info.shape[0] - 1]
    ),

    return warp_kernel(mm, input_output_aliases={4: 0}, dim=dim)


def _masked_float_csrmm_pallas_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    shape: MatrixShape,
    **kwargs
):
    m, k = shape
    n = vector_info.shape[1]

    block_dim_n = generate_block_dim(n, 512)

    if weight_info.size == 1:
        if transpose:
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
                i_n = pl.program_id(1)
                i_col_start = i_n * block_dim_n
                col_start = indptr_ref[i_k]
                col_end = indptr_ref[i_k + 1]
                mask = (i_col_start + jnp.arange(block_dim_n)) < B_ref.shape[1]
                events = pl.load(B_ref, (i_k, pl.dslice(i_col_start, block_dim_n)), mask=mask)
                val = events * data_ref[0]
                mask = mask & (events != 0.)

                def loop_fn(index, _):
                    i_row = indices_ref[index]
                    pl.atomic_add(posts_ref, (i_row, pl.dslice(i_col_start, block_dim_n)), val, mask=mask)

                jax.lax.fori_loop(col_start, col_end, loop_fn, None, )

        else:
            #
            # Gustavson algorithm: Sparse matrix–matrix multiplication is performed in a row-wise fashion.
            #
            # Each nonzero value in a row is multiplied by the nonzero values corresponding to the column index.
            # These values are summed and stored in a temporary row buffer based on their column indices.

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
                _,  # [m, n]
                posts_ref,  # [m, n]
            ):
                i_m = pl.program_id(0)
                i_n = pl.program_id(1)
                i_col_start = i_n * block_dim_n
                mask = (i_col_start + jnp.arange(block_dim_n)) < B_ref.shape[1]
                row_start = indptr_ref[i_m]
                row_end = indptr_ref[i_m + 1]
                weight = data_ref[0]

                def loop_fn(i_k, sum_):
                    index = indices_ref[i_k]
                    events = pl.load(B_ref, (index, pl.dslice(i_col_start, block_dim_n)), mask=mask)
                    sum_ += weight * events
                    return sum_

                i_row_sum = jax.lax.fori_loop(
                    row_start,
                    row_end,
                    loop_fn,
                    jnp.zeros([block_dim_n], dtype=posts_ref.dtype)
                )
                pl.store(
                    posts_ref,
                    (i_m, pl.dslice(i_col_start, block_dim_n)),
                    i_row_sum,
                    mask=mask
                )


    else:
        if transpose:
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
                i_n = pl.program_id(1)
                i_col_start = i_n * block_dim_n
                mask = (i_col_start + jnp.arange(block_dim_n)) < B_ref.shape[1]
                col_start = indptr_ref[i_k]
                col_end = indptr_ref[i_k + 1]
                events = pl.load(B_ref, (i_k, pl.dslice(i_col_start, block_dim_n)), mask=mask)
                mask = mask & (events != 0.)

                def loop_fn(index, _):
                    i_row = indices_ref[index]
                    val_A = data_ref[index]
                    val = events * val_A
                    pl.atomic_add(posts_ref, (i_row, pl.dslice(i_col_start, block_dim_n)), val, mask=mask)

                jax.lax.fori_loop(col_start, col_end, loop_fn, None, )


        else:
            #
            # Gustavson algorithm: Sparse matrix–matrix multiplication is performed in a row-wise fashion.
            #
            # Each nonzero value in a row is multiplied by the nonzero values corresponding to the column index.
            # These values are summed and stored in a temporary row buffer based on their column indices.

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
                _,  # [m, n]
                posts_ref,  # [m, n]
            ):
                i_m = pl.program_id(0)
                i_n = pl.program_id(1)
                i_col_start = i_n * block_dim_n
                mask = (i_col_start + jnp.arange(block_dim_n)) < B_ref.shape[1]
                row_start = indptr_ref[i_m]
                row_end = indptr_ref[i_m + 1]

                def loop_fn(index, sum_):
                    i_col = indices_ref[index]
                    val_A = data_ref[index]
                    events = pl.load(B_ref, (i_col, pl.dslice(i_col_start, block_dim_n)), mask=mask)
                    sum_ += val_A * events
                    return sum_

                i_row_sum = jax.lax.fori_loop(
                    row_start,
                    row_end,
                    loop_fn,
                    jnp.zeros([block_dim_n], dtype=posts_ref.dtype)
                )
                pl.store(
                    posts_ref,
                    (i_m, pl.dslice(i_col_start, block_dim_n)),
                    i_row_sum,
                    mask=mask
                )

    return pallas_kernel(
        mm,
        outs=kwargs['outs'],
        tile=(k if transpose else m, pl.cdiv(n, block_dim_n)),
        input_output_aliases={4: 0},
    )


def _csrmm_jvp_data(
    data_dot,
    data,
    indices,
    indptr,
    B,
    _,
    *,
    shape,
    transpose,
    **kwargs
):
    return [
        csr_matmat(
            data_dot,
            indices,
            indptr,
            B,
            shape=shape,
            transpose=transpose
        )
    ]


def _csrmm_jvp_B(
    B_dot,
    data,
    indices,
    indptr,
    B,
    _,
    *,
    shape,
    transpose,
    **kwargs
):
    return [
        csr_matmat(
            data,
            indices,
            indptr,
            B_dot,
            shape=shape,
            transpose=transpose
        )
    ]


def _csrmm_transpose_rule(
    ct,
    data,
    indices,
    indptr,
    B,
    _,
    *,
    shape,
    transpose,
    **kwargs
):
    assert not ad.is_undefined_primal(indices)
    assert not ad.is_undefined_primal(indptr)

    if ad.is_undefined_primal(B):
        dB = csr_matmat(data, indices, indptr, ct, shape=shape, transpose=not transpose)
        return data, indices, indptr, dB, _
    else:
        B = jnp.asarray(B)
        if data.aval.shape[0] == 1:  # scalar
            r = masked_float_csrmm_p_call(
                jnp.ones(1, dtype=data.aval.dtype),
                indices,
                indptr,
                B,
                shape=shape,
                transpose=transpose,
            )[0]
            return jnp.expand_dims(jnp.sum(r * ct), axis=0), indices, indptr, B, _
        else:
            row, col = _csr_to_coo(indices, indptr)
            if transpose:
                dCSR = B @ ct.T
                d_data = dCSR[row, col]
            else:
                dCSR = B @ ct.T
                d_data = dCSR[col, row]
            return d_data, indices, indptr, B, _


def _masked_float_csrmm_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[3].shape
        B = jnp.transpose(args[3], (1, 0, 2)).reshape(m, batch_size * n)
        r = masked_float_csrmm_p_call(
            args[0],
            args[1],
            args[2],
            B,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
        )[0]
        r = jnp.reshape(r, [r.shape[0], batch_size, n])
        return [r], [1]

    elif tuple(axes) == (None, None, None, 1, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        m, batch_size, n = args[3].shape
        B = args[3].reshape(m, batch_size * n)
        r = masked_float_csrmm_p_call(
            args[0],
            args[1],
            args[2],
            B,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
        )[0]
        r = jnp.reshape(r, [r.shape[0], batch_size, n])
        return [r], [1]

    elif tuple(axes) == (None, None, None, 2, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        m, n, batch_size = args[3].shape
        B = args[3].reshape(m, batch_size * n)
        r = masked_float_csrmm_p_call(
            args[0],
            args[1],
            args[2],
            B,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
        )[0]
        r = jnp.reshape(r, [r.shape[0], n, batch_size])
        return [r], [2]

    else:
        return general_batching_rule(masked_float_csrmm_p, args, axes, **kwargs)


def masked_float_csrmm_p_call(
    weights,
    indices,
    indptr,
    B,
    *,
    shape: MatrixShape,
    transpose: bool,
):
    """
    Perform a call to the event CSR matrix-matrix multiplication custom operation.

    Args:
        weights (jax.Array): Non-zero elements of the CSR sparse matrix.
        indices (jax.Array): Column indices of non-zero elements in the CSR sparse matrix.
        indptr (jax.Array): Index pointers of the CSR sparse matrix, indicating the start of each row.
        B (jax.Array): A dense matrix.
        shape (Sequence[int]): A sequence of length 2, representing the shape of the sparse matrix.
        transpose (bool): A boolean indicating whether to transpose the sparse matrix before multiplication.

    Returns:
        jax.Array: The result of the matrix-matrix multiplication.
    """
    assert indices.dtype in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64], "Indices must be int32 or int64."
    assert indptr.dtype in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64], "Indptr must be int32 or int64."
    assert indptr.ndim == 1, "Indptr must be 1D."
    assert indices.ndim == 1, "Indices must be 1D."
    assert indptr.dtype == indices.dtype, "Indices and indptr must have the same dtype."
    if transpose:
        assert shape[0] == B.shape[0], "Shape mismatch for non-transpose operation."
    else:
        assert shape[1] == B.shape[0], "Shape mismatch for transpose operation."

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
    # Call the masked_float_csrmm_p custom operation to perform the matrix-matrix multiplication.
    return masked_float_csrmm_p(
        weights,
        indices,
        indptr,
        B,
        # Initialize a zero matrix with the output shape and data type.
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        shape=shape,
        transpose=transpose,
        # Provide shape and data type information for indices.
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        # Provide shape and data type information for indptr.
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        # Provide shape and data type information for weights.
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        # Provide shape and data type information for B.
        vector_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
    )


masked_float_csrmm_p = XLACustomKernel('masked_float_csrmm')
masked_float_csrmm_p.def_cpu_kernel(_masked_float_csrmm_numba_kernel_generator)
masked_float_csrmm_p.def_gpu_kernel(
    default='pallas',
    warp=_masked_float_csrmm_warp_kernel_generator,
    pallas=_masked_float_csrmm_pallas_kernel_generator,
)
masked_float_csrmm_p.def_tpu_kernel(_masked_float_csrmm_pallas_kernel_generator)
masked_float_csrmm_p.def_jvp_rule2(_csrmm_jvp_data, None, None, _csrmm_jvp_B)
masked_float_csrmm_p.def_transpose_rule(_csrmm_transpose_rule)
masked_float_csrmm_p.def_batching_rule(_masked_float_csrmm_batching)
