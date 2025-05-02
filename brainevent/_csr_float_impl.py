# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

from ._compatible_import import pallas as pl
from ._config import numba_environ
from ._misc import _csr_to_coo, generate_block_dim
from ._typing import Kernel, Data, Indptr, Index, MatrixShape
from ._xla_custom_op import XLACustomKernel, GPUKernelChoice
from ._xla_custom_op_numba import NumbaKernelGenerator
from ._xla_custom_op_pallas import PallasKernelGenerator
from ._xla_custom_op_util import general_batching_rule
from ._xla_custom_op_warp import dtype_to_warp_type, WarpKernelGenerator


def csr_matvec(
    data: Data,
    indices: Index,
    indptr: Indptr,
    v: Data,
    *,
    shape: MatrixShape,
    transpose: bool = False
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
    res = csrmv_p_call(data, indices, indptr, v, shape=shape, transpose=transpose)[0]
    return u.maybe_decimal(res * unitd * unitv)


def csr_matmat(
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
    res = csrmm_p_call(
        data,
        indices,
        indptr,
        B,
        shape=shape,
        transpose=transpose,
    )[0]
    return u.maybe_decimal(res * (unitd * unitb))


def _csrmv_numba_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
) -> Kernel:
    import numba  # pylint: disable=import-outside-toplevel

    if weight_info.size == 1:
        if transpose:
            def mv(weights, indices, indptr, v, _, posts):
                w = weights[0]
                for i in range(v.shape[0]):
                    wsp = w * v[i]
                    for j in range(indptr[i], indptr[i + 1]):
                        posts[indices[j]] += wsp

        else:
            def mv(weights, indices, indptr, v, _, posts):
                w = weights[0]
                for i in numba.prange(indptr.shape[0] - 1):
                    r = np.asarray(0., dtype=posts.dtype)
                    for j in range(indptr[i], indptr[i + 1]):
                        r += w * v[indices[j]]
                    posts[i] = r

    else:
        if transpose:
            def mv(weights, indices, indptr, v, _, posts):
                for i in range(v.shape[0]):
                    sp = v[i]
                    for j in range(indptr[i], indptr[i + 1]):
                        posts[indices[j]] += weights[j] * sp

        else:
            def mv(weights, indices, indptr, v, _, posts):
                for i in range(indptr.shape[0] - 1):
                    r = np.asarray(0., dtype=posts.dtype)
                    for j in numba.prange(indptr[i], indptr[i + 1]):
                        r += weights[j] * v[indices[j]]
                    posts[i] = r

    return numba.njit(**numba_environ.setting)(mv)


def _csrmv_warp_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    indptr_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import warp  # pylint: disable=import-outside-toplevel

    weight_dtype = dtype_to_warp_type(weight_info.dtype)
    indices_dtype = dtype_to_warp_type(indices_info.dtype)
    indptr_dtype = dtype_to_warp_type(indptr_info.dtype)
    vector_dtype = dtype_to_warp_type(vector_info.dtype)

    if weight_info.size == 1:
        if transpose:
            def mv(
                weights: warp.array1d(dtype=weight_dtype),
                indices: warp.array1d(dtype=indices_dtype),
                indptr: warp.array1d(dtype=indptr_dtype),
                v: warp.array1d(dtype=vector_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype),
            ):
                i = warp.tid()
                w = weights[0]
                wsp = w * v[i]
                for j in range(indptr[i], indptr[i + 1]):
                    posts[indices[j]] += wsp

        else:
            def mv(
                weights: warp.array1d(dtype=weight_dtype),
                indices: warp.array1d(dtype=indices_dtype),
                indptr: warp.array1d(dtype=indptr_dtype),
                v: warp.array1d(dtype=vector_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype),
            ):
                i = warp.tid()
                w = weights[0]
                r = weights.dtype(0.)
                for j in range(indptr[i], indptr[i + 1]):
                    r += w * v[indices[j]]
                posts[i] = r

    else:
        if transpose:
            def mv(
                weights: warp.array1d(dtype=weight_dtype),
                indices: warp.array1d(dtype=indices_dtype),
                indptr: warp.array1d(dtype=indptr_dtype),
                v: warp.array1d(dtype=vector_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype),
            ):
                i = warp.tid()
                sp = v[i]
                for j in range(indptr[i], indptr[i + 1]):
                    posts[indices[j]] += weights[j] * sp

        else:
            def mv(
                weights: warp.array1d(dtype=weight_dtype),
                indices: warp.array1d(dtype=indices_dtype),
                indptr: warp.array1d(dtype=indptr_dtype),
                v: warp.array1d(dtype=vector_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype),
            ):
                i_row = warp.tid()
                r = weights.dtype(0.)
                for index in range(indptr[i_row], indptr[i_row + 1]):
                    i_k = indices[index]
                    c = v[i_k]
                    w = weights[index]
                    r += w * c
                posts[i_row] = r

    return warp.kernel(mv)


def _csrmv_pallas_tiled_kernel_generator(
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
                val_vector = vector_ref[i_col]
                data = data_ref[0] * val_vector
                data = jnp.ones((block_dim,), dtype=posts_ref.dtype) * data

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
                    val_B = pl.load(vector_ref, cols, mask=mask, other=0.0)
                    sum_ += val_A * jnp.sum(val_B)
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
                val_vector = vector_ref[i_col]

                def loop_fn(index, _):
                    offset = col_start + index * block_dim
                    mask = offset + jnp.arange(block_dim) < col_end
                    rows = pl.load(indices_ref, pl.dslice(offset, block_dim), mask=mask)
                    val_A = pl.load(data_ref, pl.dslice(offset, block_dim), mask=mask, other=0.0)
                    contrib = val_A * val_vector
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
                    val_B = pl.load(vector_ref, cols, mask=mask, other=0.0)
                    sum_ += jnp.sum(val_A * val_B)
                    return sum_

                i_row_sum = jax.lax.fori_loop(
                    0,
                    num_blocks,
                    loop_fn,
                    jnp.asarray(0., dtype=posts_ref.dtype)
                )
                posts_ref[i_row] = i_row_sum

    def kernel(data, indices, indptr, B, placeholder):
        fn = pl.pallas_call(
            mm,
            out_shape=jax.ShapeDtypeStruct([k if transpose else m], weight_info.dtype),
            grid=(k if transpose else m,),
            input_output_aliases={4: 0},
        )
        return [fn(data, indices, indptr, B, placeholder)]

    return kernel


def _csrmv_jvp_v(
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


def _csrmv_jvp_weights(
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
    return csrmv_p_call(
        data_dot,
        indices,
        indptr,
        v,
        shape=shape,
        transpose=transpose,
    )


def _csrmv_transpose_rule(
    ct,
    data,
    indices,
    indptr,
    vector,
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
    if ad.is_undefined_primal(vector):
        if type(ct) is ad.Zero:
            ct_events = ad.Zero(vector)
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
                ct_values = csrmv_p_call(
                    jnp.ones(1, dtype=data.aval.dtype),
                    indices,
                    indptr,
                    vector,
                    shape=shape,
                    transpose=transpose,
                )[0]
                ct_values = jnp.inner(ct, ct_values).reshape(*data.aval.shape)
            else:  # heterogeneous values
                row, col = _csr_to_coo(indices, indptr)
                ct_values = vector[row] * ct[col] if transpose else vector[col] * ct[row]
        return ct_values, indices, indptr, vector, _


def _csrmv_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = csrmm_p_call(
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
        r = csrmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
        )
        return r, [1]

    else:
        return general_batching_rule(csrmv_p, args, axes, **kwargs)


def csrmv_p_call(
    weights,
    indices,
    indptr,
    vector,
    *,
    shape: Sequence[int],
    transpose: bool,
):
    assert indices.dtype in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64], "Indices must be int32 or int64."
    assert indptr.dtype in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64], "Indptr must be int32 or int64."
    assert indptr.ndim == 1, "Indptr must be 1D."
    assert indices.ndim == 1, "Indices must be 1D."
    assert indptr.dtype == indices.dtype, "Indices and indptr must have the same dtype."
    if transpose:
        assert shape[0] == vector.shape[0], "Shape mismatch for transpose operation."
    else:
        assert shape[1] == vector.shape[0], "Shape mismatch for non-transpose operation."

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
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        shape=shape,
        transpose=transpose,
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(vector.shape, vector.dtype),
    )


csrmv_p = XLACustomKernel('csrmv')
csrmv_p.def_cpu_kernel(
    NumbaKernelGenerator(_csrmv_numba_kernel_generator, input_output_aliases={4: 0})
)
csrmv_p.def_gpu_kernel(
    GPUKernelChoice(
        default='pallas',
        warp_kernel=WarpKernelGenerator(
            _csrmv_warp_kernel_generator,
            dim=lambda indptr_info, vector_info, transpose, **kwargs: (
                vector_info.shape[0] if transpose else indptr_info.shape[0] - 1
            ),
            input_output_aliases={4: 0}
        ),
        pallas_kernel=PallasKernelGenerator(
            _csrmv_pallas_tiled_kernel_generator,
            input_output_aliases={4: 0}
        )
    )
)
csrmv_p.def_tpu_kernel(
    PallasKernelGenerator(_csrmv_pallas_tiled_kernel_generator,
                          input_output_aliases={4: 0})
)
csrmv_p.defjvp(_csrmv_jvp_weights, None, None, _csrmv_jvp_v)
csrmv_p.def_transpose_rule(_csrmv_transpose_rule)
csrmv_p.def_batching_rule(_csrmv_batching)


def _csrmm_numba_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
) -> Kernel:
    import numba  # pylint: disable=import-outside-toplevel

    if weight_info.size == 1:
        if transpose:
            # csr.T @ B
            #
            # CSR: [k, m]
            # B: [k, n]
            #
            @numba_environ.jit_fn
            def mm(weights, indices, indptr, B, _, posts):
                w = weights[0]
                for i in range(B.shape[0]):
                    wsp = w * B[i]
                    for j in range(indptr[i], indptr[i + 1]):
                        posts[indices[j]] += wsp

        else:
            # csr @ B
            #
            # CSR: [m, k]
            # B: [k, n]
            #
            @numba_environ.pjit_fn
            def mm(weights, indices, indptr, B, _, posts):
                w = weights[0]
                for i in numba.prange(indptr.shape[0] - 1):
                    r = np.zeros(B.shape[1], dtype=posts.dtype)
                    for j in range(indptr[i], indptr[i + 1]):
                        r += w * B[indices[j]]
                    posts[i] = r

    else:
        if transpose:
            # csr.T @ B
            #
            # CSR: [k, m]
            # B: [k, n]
            #
            @numba_environ.jit_fn
            def mm(weights, indices, indptr, B, _, posts):
                for i in range(B.shape[0]):
                    for j in range(indptr[i], indptr[i + 1]):
                        posts[indices[j]] += weights[j] * B[i]

        else:
            # csr @ B
            #
            # CSR: [m, k]
            # B: [k, n]
            #
            @numba_environ.pjit_fn
            def mm(weights, indices, indptr, B, _, posts):
                for i in numba.prange(indptr.shape[0] - 1):
                    r = np.zeros(B.shape[1], dtype=posts.dtype)
                    for j in range(indptr[i], indptr[i + 1]):
                        r += weights[j] * B[indices[j]]
                    posts[i] = r

    return mm


def _csrmm_warp_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    indptr_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
) -> Kernel:
    import warp  # pylint: disable=import-outside-toplevel

    weight_dtype = dtype_to_warp_type(weight_info.dtype)
    spike_dtype = dtype_to_warp_type(vector_info.dtype)
    indices_dtype = dtype_to_warp_type(indices_info.dtype)
    indptr_dtype = dtype_to_warp_type(indptr_info.dtype)

    M, K = kwargs['shape']
    TILE_SIZE_N = vector_info.shape[1]
    TILE_SIZE_M = 8

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
                i_k = warp.tid()
                B_row = warp.tile_load(B[i_k], TILE_SIZE_N) * weights[0]
                col_start = indptr[i_k]
                col_end = indptr[i_k + 1]
                for index in range(col_start, col_end):
                    i_row = indices[index]
                    warp.tile_atomic_add(posts[i_row], B_row)

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
                i_row = warp.tid()
                weight = weights[0]
                r = warp.tile_zeros(TILE_SIZE_N, dtype=weight_dtype)
                for index in range(indptr[i_row], indptr[i_row + 1]):
                    i_k = indices[index]
                    r += weight * warp.tile_load(B[i_k], TILE_SIZE_N)
                warp.tile_store(posts[i_row], r)

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
                i_k = warp.tid()
                B_row = warp.tile_load(B[i_k], TILE_SIZE_N)
                col_start = indptr[i_k]
                col_end = indptr[i_k + 1]
                for index in range(col_start, col_end):
                    i_row = indices[index]
                    weight = weights[index]
                    warp.tile_atomic_add(posts[i_row], weight * B_row)

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
                i_row = warp.tid()
                r = warp.tile_zeros(TILE_SIZE_N, dtype=weight_dtype)
                for index in range(indptr[i_row], indptr[i_row + 1]):
                    i_k = indices[index]
                    weight = weights[index]
                    r += weight * warp.tile_load(B[i_k], TILE_SIZE_N)
                warp.tile_store(posts[i_row], r)

                # i_row = warp.tid()
                # i_row_start = i_row * TILE_SIZE_M
                # row_length = warp.min(TILE_SIZE_M, M - i_row_start)
                # r = warp.tile_zeros(shape=(TILE_SIZE_M, TILE_SIZE_N), dtype=weight_dtype)
                # for i in range(row_length):
                #     for index in range(indptr[i + i_row_start], indptr[i + i_row_start + 1]):
                #         i_k = indices[index]
                #         weight = weights[index]
                #         r[i] += weight * warp.tile_load(B[i_k], TILE_SIZE_N)
                # warp.tile_store(posts, r, (i_row_start, 0))

    return warp.kernel(mm)


def _csrmm_pallas_kernel1(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    shape,
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
                val_B = pl.load(B_ref, (i_k, pl.dslice(i_col_start, block_dim_n)), mask=mask)
                val = val_B * data_ref[0]

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
                    val_B = pl.load(B_ref, (index, pl.dslice(i_col_start, block_dim_n)), mask=mask)
                    sum_ += weight * val_B
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

                val_B = pl.load(B_ref, (i_k, pl.dslice(i_col_start, block_dim_n)), mask=mask, other=0.0)

                def loop_fn(index, _):
                    i_row = indices_ref[index]
                    val_A = data_ref[index]
                    val = val_A * val_B
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
                    val_B = pl.load(B_ref, (i_col, pl.dslice(i_col_start, block_dim_n)), mask=mask, other=0.0)
                    sum_ += val_A * val_B
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

    def kernel(data, indices, indptr, B, placeholder):
        fn = pl.pallas_call(
            mm,
            out_shape=jax.ShapeDtypeStruct([k if transpose else m, n], weight_info.dtype),
            grid=(k if transpose else m, pl.cdiv(n, block_dim_n)),
            input_output_aliases={4: 0},
        )
        return [fn(data, indices, indptr, B, placeholder)]

    return kernel


def _csrmm_pallas_kernel2(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    shape,
    **kwargs
):
    m, k = shape
    n = vector_info.shape[1]

    block_dim_n = generate_block_dim(n, 256)
    block_dim_m = 32

    if weight_info.size == 1:
        if transpose:
            raise NotImplementedError

        else:
            # csr @ B
            #
            # csr: [m, k]
            # B: [k, n]
            #
            def mm(
                data_ref,
                indices_ref,
                indptr_ref,
                B_ref,
                _,
                posts_ref,
            ):
                i_m = pl.program_id(0)
                i_n = pl.program_id(1)
                weight = data_ref[0]
                i_row_start = i_m * block_dim_m
                i_col_start = i_n * block_dim_n
                col_mask = (i_col_start + jnp.arange(block_dim_n)) < n

                def outer_loop_fn(i_row, _):
                    def inner_loop_fn(i_k, inner_acc):
                        index = indices_ref[i_k]
                        val_B = pl.load(
                            B_ref,
                            (index, pl.dslice(i_n * block_dim_n, block_dim_n)),
                            mask=col_mask
                        )
                        inner_acc += weight * val_B
                        return inner_acc

                    row_start = indptr_ref[i_row]
                    row_end = indptr_ref[i_row + 1]
                    outer_acc = jax.lax.fori_loop(
                        row_start,
                        row_end,
                        inner_loop_fn,
                        jnp.zeros([block_dim_n], dtype=posts_ref.dtype)
                    )
                    pl.store(
                        posts_ref,
                        (i_row, pl.dslice(i_col_start, block_dim_n)),
                        outer_acc,
                        mask=col_mask
                    )

                jax.lax.fori_loop(
                    i_row_start,
                    jnp.minimum(i_row_start + block_dim_m, m),
                    outer_loop_fn,
                    None
                )


    else:
        if transpose:
            raise NotImplementedError

        else:
            #
            # csr @ B
            #
            # csr: [m, k]
            # B: [k, n]
            #
            def mm(
                data_ref,
                indices_ref,
                indptr_ref,
                B_ref,
                _,
                posts_ref,
            ):
                i_m = pl.program_id(0)
                i_n = pl.program_id(1)
                i_row_start = i_m * block_dim_m
                i_col_start = i_n * block_dim_n
                col_mask = (i_col_start + jnp.arange(block_dim_n)) < n

                def outer_loop_fn(i_row, _):
                    def inner_loop_fn(i_k, inner_acc):
                        index = indices_ref[i_k]
                        val_A = data_ref[i_k]
                        val_B = pl.load(
                            B_ref,
                            (index, pl.dslice(i_n * block_dim_n, block_dim_n)),
                            mask=col_mask
                        )
                        inner_acc += val_A * val_B
                        return inner_acc

                    row_start = indptr_ref[i_row]
                    row_end = indptr_ref[i_row + 1]
                    outer_acc = jax.lax.fori_loop(
                        row_start,
                        row_end,
                        inner_loop_fn,
                        jnp.zeros([block_dim_n], dtype=posts_ref.dtype)
                    )
                    pl.store(
                        posts_ref,
                        (i_row, pl.dslice(i_col_start, block_dim_n)),
                        outer_acc,
                        mask=col_mask
                    )

                jax.lax.fori_loop(
                    i_row_start,
                    jnp.minimum(i_row_start + block_dim_m, m),
                    outer_loop_fn,
                    None
                )

    def kernel(data, indices, indptr, B, placeholder):
        fn = pl.pallas_call(
            mm,
            out_shape=jax.ShapeDtypeStruct([m, n], weight_info.dtype),
            grid=(pl.cdiv(m, block_dim_m), pl.cdiv(n, block_dim_n)),
            input_output_aliases={4: 0},
        )
        return [fn(data, indices, indptr, B, placeholder)]

    return kernel


def _csrmm_pallas_kernel_generator(**kwargs) -> Kernel:
    version = 1

    if version == 1:
        return _csrmm_pallas_kernel1(**kwargs)

    elif version == 2:
        return _csrmm_pallas_kernel2(**kwargs)

    else:
        raise ValueError(f'Unknown version: {version}')


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
    ct = ct[0]

    if ad.is_undefined_primal(B):
        dB = csr_matmat(data, indices, indptr, ct, shape=shape, transpose=not transpose)
        return data, indices, indptr, dB, _
    else:
        B = jnp.asarray(B)
        if data.aval.shape[0] == 1:  # scalar
            r = csrmm_p_call(
                jnp.ones(1, dtype=data.aval.dtype),
                indices,
                indptr,
                B,
                shape=shape,
                transpose=transpose
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


def _csrmm_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0, None):
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
        )[0]
        r = jnp.reshape(r, [r.shape[0], batch_size, n])
        return [r], [1]

    elif tuple(axes) == (None, None, None, 1, None):
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
        )[0]
        r = jnp.reshape(r, [r.shape[0], batch_size, n])
        return [r], [1]

    elif tuple(axes) == (None, None, None, 2, None):
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
        )[0]
        r = jnp.reshape(r, [r.shape[0], n, batch_size])
        return [r], [2]

    else:
        return general_batching_rule(csrmm_p, args, axes, **kwargs)


def csrmm_p_call(
    weights,
    indices,
    indptr,
    B,
    *,
    shape: Sequence[int],
    transpose: bool,
):
    assert indices.dtype in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64], "Indices must be int32 or int64."
    assert indptr.dtype in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64], "Indptr must be int32 or int64."
    assert indptr.ndim == 1, "Indptr must be 1D."
    assert indices.ndim == 1, "Indices must be 1D."
    assert indptr.dtype == indices.dtype, "Indices and indptr must have the same dtype."
    if transpose:
        assert shape[0] == B.shape[0], "Shape mismatch for non-transpose operation."
    else:
        assert shape[1] == B.shape[0], "Shape mismatch for transpose operation."

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
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        shape=shape,
        transpose=transpose,
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
    )


csrmm_p = XLACustomKernel('csrmm')
csrmm_p.def_cpu_kernel(
    NumbaKernelGenerator(_csrmm_numba_kernel_generator, input_output_aliases={4: 0})
)
csrmm_p.def_gpu_kernel(
    GPUKernelChoice(
        default='pallas',
        warp_kernel=WarpKernelGenerator(
            _csrmm_warp_kernel_generator,
            tile=lambda shape, transpose, **kwargs: shape[1] if transpose else shape[0],
            block_dim=lambda vector_info, **kwargs: (
                generate_block_dim(vector_info.shape[1], 1024)
            ),
            input_output_aliases={4: 0}
        ),
        pallas_kernel=PallasKernelGenerator(
            _csrmm_pallas_kernel_generator,
            input_output_aliases={4: 0}
        )
    )
)
csrmm_p.def_tpu_kernel(
    PallasKernelGenerator(_csrmm_pallas_kernel_generator, input_output_aliases={4: 0})
)
csrmm_p.defjvp(_csrmm_jvp_data, None, None, _csrmm_jvp_B)
csrmm_p.def_transpose_rule(_csrmm_transpose_rule)
csrmm_p.def_batching_rule(_csrmm_batching)
