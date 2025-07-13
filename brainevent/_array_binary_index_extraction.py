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

import jax
import jax.numpy as jnp

from ._xla_custom_op import XLACustomKernel
from ._xla_custom_op_numba import numba_kernel
from ._xla_custom_op_warp import warp_kernel


def binary_array_index(spikes):
    if spikes.ndim == 1:
        indices, count = binary_1d_array_index_p_call(spikes)
    elif spikes.ndim == 2:
        indices, count = binary_2d_array_index_p_call(spikes)
    else:
        raise ValueError("Only 1D and 2D binary arrays are supported for index extraction.")
    return indices, count


def _binary_1d_array_index_numba_kernel_generator(
    spikes_info: jax.ShapeDtypeStruct,
    **kwargs
):
    if spikes_info.dtype == jnp.bool_:
        def _kernel(spikes, _, indices, count):
            idx = 0
            for i in range(spikes.shape[0]):
                if spikes[i]:
                    indices[idx] = i
                    idx += 1
            count[0] = idx
    else:
        def _kernel(spikes, _, indices, count):
            idx = 0
            for i in range(spikes.shape[0]):
                if spikes[i] != 0.:
                    indices[idx] = i
                    idx += 1
            count[0] = idx

    return numba_kernel(_kernel, input_output_aliases={1: 1})


def _binary_1d_array_index_warp_kernel_generator(
    spikes_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp  # pylint: disable=import-outside-toplevel

    if spikes_info.dtype == jnp.bool_:
        def kernel(
            spikes: warp.array(dtype=float),
            _: warp.array(dtype=int),
            indices: warp.array(dtype=int),
            count: warp.array(dtype=int),
        ):
            i_col_block = warp.tid()
            if spikes[i_col_block]:
                idx = warp.atomic_add(count, 0, 1)
                indices[idx] = i_col_block

    else:
        def kernel(
            spikes: warp.array(dtype=float),
            _: warp.array(dtype=int),
            indices: warp.array(dtype=int),
            count: warp.array(dtype=int),
        ):
            i_col_block = warp.tid()
            if spikes[i_col_block] != 0.:
                idx = warp.atomic_add(count, 0, 1)
                indices[idx] = i_col_block

    return warp_kernel(kernel, dim=spikes_info.shape[0], input_output_aliases={1: 1})


def binary_1d_array_index_p_call(spikes):
    indices_info = jax.ShapeDtypeStruct([spikes.shape[0]], jnp.int32)
    count_info = jax.ShapeDtypeStruct([1], jnp.int32)
    return binary_1d_array_index_p(
        spikes,
        jnp.zeros([1], dtype=jnp.int32),
        outs=[indices_info, count_info],
        spikes_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        indices_info=indices_info,
    )


binary_1d_array_index_p = XLACustomKernel('binary_1d_array_index')
binary_1d_array_index_p.def_cpu_kernel(_binary_1d_array_index_numba_kernel_generator)
binary_1d_array_index_p.def_gpu_kernel(
    warp=_binary_1d_array_index_warp_kernel_generator,
    default='warp'
)


def binary_2d_array_index_p_call(spikes):
    out = jax.ShapeDtypeStruct([spikes.shape[0]], jnp.int32)
    raise NotImplementedError("2D binary array index extraction is not implemented yet.")
