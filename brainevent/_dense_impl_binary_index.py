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

import brainunit as u
import jax

from ._misc import cdiv, generate_block_dim
from ._xla_custom_op import XLACustomKernel
from ._xla_custom_op_numba import numba_kernel
from ._xla_custom_op_warp import warp_kernel, jaxinfo_to_warpinfo, jaxtype_to_warptype


def binary_vec_dot_dense_mat(binary_arr, weights):
    weight_val, wunit = u.split_mantissa_unit(weights)
    spikes = binary_arr.value
    indices = binary_arr.spike_indices
    count = binary_arr.spike_count
    r = binary_vec_dot_dense_mat_p_call(spikes, indices, count, weight_val)
    return u.maybe_decimal(r[0] * wunit)


def _binary_vec_dot_dense_mat_numba_kernel_generator(
    **kwargs
):
    def _kernel(spikes, indices, count, weights, out):
        out[:] = 0.
        for i in range(count[0]):
            out += weights[indices[i]]

    return numba_kernel(_kernel)


def _binary_vec_dot_dense_mat_warp_kernel_generator(
    spikes_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    count_info: jax.ShapeDtypeStruct,
    weights_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp  # pylint: disable=import-outside-toplevel
    block_dim = generate_block_dim(weights_info.shape[1], maximum=128)
    weight_dtype = jaxtype_to_warptype(weights_info.dtype)

    def kernel(
        spikes: jaxinfo_to_warpinfo(spikes_info),
        indices: jaxinfo_to_warpinfo(indices_info),
        count: jaxinfo_to_warpinfo(count_info),
        weights: jaxinfo_to_warpinfo(weights_info),
        out: warp.array1d(dtype=weight_dtype),
    ):
        i_col_block = warp.tid()
        temp = warp.tile_zeros(shape=(block_dim,), dtype=weight_dtype)
        for j in range(count[0]):
            temp += warp.tile_load(weights[indices[j]], shape=(block_dim,), offset=(i_col_block * block_dim,))
        warp.tile_store(out, temp, offset=(i_col_block * block_dim,))

    return warp_kernel(
        kernel,
        tile=(cdiv(weights_info.shape[1], block_dim),),
        block_dim=block_dim,
    )


def binary_vec_dot_dense_mat_p_call(spikes, indices, count, weights):
    return binary_vec_dot_dense_mat_p(
        spikes,
        indices,
        count,
        weights,
        outs=[
            jax.ShapeDtypeStruct([weights.shape[1]], weights.dtype)
        ],
        spikes_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        count_info=jax.ShapeDtypeStruct(count.shape, count.dtype),
        weights_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
    )


binary_vec_dot_dense_mat_p = XLACustomKernel('binary_vec_dot_dense_matrix')
binary_vec_dot_dense_mat_p.def_cpu_kernel(_binary_vec_dot_dense_mat_numba_kernel_generator)
binary_vec_dot_dense_mat_p.def_gpu_kernel(
    warp=_binary_vec_dot_dense_mat_warp_kernel_generator,
    default='warp'
)


def binary_mat_dot_dense_mat(binary_arr, weights):
    weights, wunit = u.split_mantissa_unit(weights)
    spikes = binary_arr.value
    indices = binary_arr.spike_indices
    count = binary_arr.spike_count
    raise ValueError


def dense_mat_dot_binary_vec(weights, binary_arr):
    weight_val, wunit = u.split_mantissa_unit(weights)
    spikes = binary_arr.value
    indices = binary_arr.spike_indices
    count = binary_arr.spike_count
    raise ValueError


def dense_mat_dot_binary_mat(weights, binary_arr):
    weight_val, wunit = u.split_mantissa_unit(weights)
    spikes = binary_arr.value
    indices = binary_arr.spike_indices
    count = binary_arr.spike_count
    raise ValueError
