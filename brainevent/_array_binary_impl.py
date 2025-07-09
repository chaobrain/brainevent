# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

from typing import Sequence

import brainunit as u
import jax
import jax.numpy as jnp
from jax.interpreters import ad

from ._coo_impl_float import coo_matvec, coo_matmat
from ._typing import Data, Row, Col, MatrixShape
from ._xla_custom_op import XLACustomKernel
from ._xla_custom_op_numba import numba_kernel
from ._xla_custom_op_util import general_batching_rule
from ._xla_custom_op_warp import jaxtype_to_warptype, warp_kernel

from ._misc import cdiv, generate_block_dim


"""
This module defines a custom XLA-optimized operation for performing a sparse-dense vector-matrix multiplication.
The main function `indices_dot_dense_mat` computes the sum of selected rows from the `weights` matrix
based on the provided `indices`.

Functions:
----------
indices_dot_dense_mat(weights, indices, count_arr):
    Computes a dense vector by selecting and summing rows from `weights` as specified by `indices`.
    
    Parameters:
        weights (array-like): A 2D array representing the weight matrix (shape: [num_rows, num_cols]).
        indices (array-like): A 1D array of indices indicating which rows of `weights` to select and sum (shape: [count_arr[0]]).
        count_arr (array-like): An auxiliary 1D array whose length determines the number of index accesses (shape: [1]).
    
    Returns:
        array-like: A 1D array resulting from the sum of selected rows in `weights`, scaled by units extracted
                    from `weights` and `indices`.
"""

def indices_dot_dense_mat(weights, indices, count_data_struct):
    with jax.ensure_compile_time_eval():
        weights = u.math.asarray(weights)
        indices = u.math.asarray(indices)
    weight_val, wunit = u.split_mantissa_unit(weights)
    indices_val, indicesunit = u.split_mantissa_unit(indices)
    r = indices_dot_dense_mat_p_call(weight_val, indices_val, count_data_struct)
    return u.maybe_decimal(r[0] * wunit * indicesunit)


def _indices_dot_dense_mat_numba_kernel_generator(
    indices_info: jax.ShapeDtypeStruct,
    count_info: jax.ShapeDtypeStruct,
    **kwargs
):
    indices_length = count_info.shape[0]
    def _kernel(indices, weights, out):
        out[:] = 0.
        for i in range(indices_length):
            out += weights[indices[i]]
    return numba_kernel(_kernel)



def _indices_dot_dense_mat_warp_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    count_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp  # pylint: disable=import-outside-toplevel
    indices_length = count_info.shape[0]
    block_dim = generate_block_dim(weight_info.shape[1], maximum=128)
    weight_dtype = jaxtype_to_warptype(weight_info.dtype)
    def kernel(
        indices: warp.array1d(dtype=int),
        weights: warp.array2d(dtype=weight_dtype),
        out: warp.array1d(dtype=weight_dtype),
    ):
        i_col_block = warp.tid()
        temp = warp.tile_zeros(shape=(block_dim,), dtype=weight_dtype)
        for j in range(indices_length):
            temp += warp.tile_load(weights[indices[j]], shape=(block_dim,), offset=(i_col_block * block_dim,))
        warp.tile_store(out, temp, offset=(i_col_block * block_dim,))
    return warp_kernel(
        kernel,
        tile=(cdiv(weight_info.shape[1], block_dim),),
        block_dim=block_dim,
    )

def indices_dot_dense_mat_p_call(indices, weights, count_data_struct):
    out = jax.ShapeDtypeStruct([weights.shape[1]], weights.dtype)

    return indices_dot_dense_mat_p(
        indices,
        weights,
        outs=[out],
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        count_info =jax.ShapeDtypeStruct([count_data_struct.shape[0]], count_data_struct.dtype)
    )


indices_dot_dense_mat_p = XLACustomKernel('indices_dot_dense_matrix')
indices_dot_dense_mat_p.def_cpu_kernel(_indices_dot_dense_mat_numba_kernel_generator)
indices_dot_dense_mat_p.def_gpu_kernel(warp=_indices_dot_dense_mat_warp_kernel_generator,
                                          default='warp')


"""
This module defines a custom XLA-optimized operation for extracting the indices of non-zero (or True) values in a binary or boolean vector.
The main function `binary_vec_get_indices` returns the indices of active elements along with the count of such elements.

Functions:
----------
binary_vec_get_indices(spikes):
    Extracts the indices of non-zero/True elements from the input vector.

    Parameters:
        spikes (array-like): A 1D array representing a binary or boolean vector. Can be of type bool or numeric (e.g., float, int).

    Returns:
        tuple:
            - array-like: A 1D array containing the indices of non-zero/True elements.
            - array-like: A scalar array indicating the number of non-zero/True elements.
"""

def binary_vec_get_indices(spikes, count_array):
    r = binary_vec_get_indices_p_call(spikes, count_array)
    return r[0]



def _binary_vec_get_indices_numba_kernel_generator(
    spike_info: jax.ShapeDtypeStruct,
    **kwargs
):
    if spike_info.dtype == jnp.bool_:
        def _kernel(spikes,cnt, indices):
            idx = 0
            for i in range(spikes.shape[0]):
                if spikes[i]:
                    indices[idx] = i
                    idx += 1
            cnt[0] = idx
    else:
        def _kernel(spikes, cnt, indices):
            idx = 0
            for i in range(spikes.shape[0]):
                if spikes[i] != 0.:
                    indices[idx] = i
                    idx += 1
            cnt[0] = idx

    return numba_kernel(_kernel)


def _binary_vec_get_indices_warp_kernel_generator(
    spike_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp  # pylint: disable=import-outside-toplevel
    indices_dtype = jaxtype_to_warptype(indices_info.dtype)
    indices_length = indices_info.shape[0]
    if spike_info.dtype == jnp.bool_:
        def kernel(
            spikes: warp.array(dtype=float),
            cnt: warp.array(dtype=int),
            indices: warp.array(dtype=int),
            
        ):
            i_col_block = warp.tid()

            if (spikes[i_col_block]):
                idx = warp.atomic_add(cnt, 0, 1)
                indices[idx] = i_col_block
    else:
        def kernel(
            spikes: warp.array(dtype=float),
            cnt: warp.array(dtype=int),
            indices: warp.array(dtype=int),
        ):
            i_col_block = warp.tid()
            temp = warp.tile_zeros(shape=(1,), dtype=int)
            if (spikes[i_col_block] != 0.):
                idx = warp.atomic_add(cnt, 0, 1)
                # print(idx)
                indices[idx] = i_col_block        


    return warp_kernel(
        kernel,
        dim = indices_length,
    )


def binary_vec_get_indices_p_call(spikes, count_array):
    out = jax.ShapeDtypeStruct([spikes.shape[0]], jnp.int32)
    #cnt = jax.ShapeDtypeStruct([1], jnp.int32)

    return binary_vec_get_indices_p(
        spikes,
        count_array,
        outs=[out],
        spike_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        indices_info=jax.ShapeDtypeStruct(out.shape, out.dtype),
    )


binary_vec_get_indices_p = XLACustomKernel('binary_vec_get_indices')
binary_vec_get_indices_p.def_cpu_kernel(_binary_vec_get_indices_numba_kernel_generator)
binary_vec_get_indices_p.def_gpu_kernel(warp=_binary_vec_get_indices_warp_kernel_generator,
                                          default='warp')

