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

# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp

from brainevent._compatible_import import pallas as pl
from brainevent._op.main import XLACustomKernel
from brainevent._op.op_numba import numba_kernel
from brainevent._op.op_pallas import pallas_kernel
from brainevent._op.op_warp import warp_kernel, jaxinfo_to_warpinfo


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
    count_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp  # pylint: disable=import-outside-toplevel

    if spikes_info.dtype == jnp.bool_:
        def kernel(
            spikes: jaxinfo_to_warpinfo(spikes_info),
            _: jaxinfo_to_warpinfo(count_info),
            indices: jaxinfo_to_warpinfo(indices_info),
            count: jaxinfo_to_warpinfo(count_info),
        ):
            i_col_block = warp.tid()
            if spikes[i_col_block]:
                idx = warp.atomic_add(count, 0, 1)
                indices[idx] = i_col_block

    else:
        def kernel(
            spikes: jaxinfo_to_warpinfo(spikes_info),
            _: jaxinfo_to_warpinfo(count_info),
            indices: jaxinfo_to_warpinfo(indices_info),
            count: jaxinfo_to_warpinfo(count_info),
        ):
            i_col_block = warp.tid()
            if spikes[i_col_block] != 0.:
                idx = warp.atomic_add(count, 0, 1)
                indices[idx] = i_col_block

    return warp_kernel(kernel, dim=spikes_info.shape[0], input_output_aliases={1: 1})


def _binary_1d_array_index_pallas_kernel_generator(
    spikes_info: jax.ShapeDtypeStruct,
    **kwargs
):
    BLOCK_SIZE = 64

    def _raw_kernel(
        spikes,
        _,
        indices,
        count,
    ):
        pid = pl.program_id(0)
        start = pid * BLOCK_SIZE
        idxs = start + jnp.arange(0, BLOCK_SIZE)

        # Check valid indices
        valid_mask = idxs < spikes.shape[0]

        # Load values safely
        if spikes_info.dtype == jnp.bool_:
            default_val = False
        else:
            default_val = 0.0
        x_vals = pl.load(spikes, (idxs,), mask=valid_mask, other=default_val)

        # Create value mask
        if spikes_info.dtype == jnp.bool_:
            value_mask = x_vals
        else:
            value_mask = x_vals != 0.0
        combined_mask = valid_mask & value_mask

        # Count non-zero elements in this block
        total_in_block = jnp.sum(combined_mask.astype(jnp.int32))

        # Atomically reserve space in global count
        base_pos = pl.atomic_add(count, (0,), total_in_block)
        prefix_offsets = jnp.cumsum(combined_mask) - combined_mask

        # Calculate write positions
        write_positions = base_pos + prefix_offsets

        # Store indices
        pl.store(indices, (write_positions,), idxs, mask=combined_mask)

    return pallas_kernel(
        _raw_kernel,
        outs=kwargs['outs'],
        tile=(pl.cdiv(spikes_info.shape[0], BLOCK_SIZE),),
        input_output_aliases={1: 1}
    )


def binary_1d_array_index_p_call(spikes):
    indices_info = jax.ShapeDtypeStruct([spikes.shape[0]], jnp.int32)
    count_info = jax.ShapeDtypeStruct([1], jnp.int32)
    return binary_1d_array_index_p(
        spikes,
        jnp.zeros([1], dtype=jnp.int32),
        outs=[indices_info, count_info],
        spikes_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        indices_info=indices_info,
        count_info=count_info,
    )


binary_1d_array_index_p = XLACustomKernel('binary_1d_array_index')
binary_1d_array_index_p.def_cpu_kernel(_binary_1d_array_index_numba_kernel_generator)
binary_1d_array_index_p.def_gpu_kernel(
    warp=_binary_1d_array_index_warp_kernel_generator,
    pallas=_binary_1d_array_index_pallas_kernel_generator,
    default='pallas'
)
binary_1d_array_index_p.def_tpu_kernel(_binary_1d_array_index_pallas_kernel_generator)


def binary_2d_array_index_p_call(spikes):
    out = jax.ShapeDtypeStruct([spikes.shape[0]], jnp.int32)
    raise NotImplementedError("2D binary array index extraction is not implemented yet.")
