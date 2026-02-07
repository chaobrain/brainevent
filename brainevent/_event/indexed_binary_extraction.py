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

import jax
import jax.numpy as jnp
from jax.interpreters import ad

from brainevent._misc import namescope
from brainevent._op import XLACustomKernel, numba_kernel, jaxinfo_to_warpinfo, general_batching_rule
from brainevent._op.benchmark import BenchmarkConfig


@namescope
def binary_array_index(spikes):
    if spikes.ndim == 1:
        indices, count = binary_1d_array_index_p_call(spikes)
    elif spikes.ndim == 2:
        indices, count = binary_2d_array_index_p_call(spikes)
    else:
        raise ValueError("Only 1D and 2D binary arrays are supported for index extraction.")
    return indices, count


def _binary_1d_array_index_numba_kernel(
    spikes_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import numba

    if spikes_info.dtype == jnp.bool_:
        @numba.njit(fastmath=True)
        def mv(spikes, indices, count):
            idx = 0
            for i in range(spikes.shape[0]):
                if spikes[i]:
                    indices[idx] = i
                    idx += 1
            count[0] = idx
    else:
        @numba.njit(fastmath=True)
        def mv(spikes, indices, count):
            idx = 0
            for i in range(spikes.shape[0]):
                if spikes[i] != 0.:
                    indices[idx] = i
                    idx += 1
            count[0] = idx

    def kernel(spikes):
        return numba_kernel(mv, outs=kwargs['outs'])(spikes)

    return kernel


def _binary_1d_array_index_warp_kernel(
    spikes_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    count_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    spikes_warp_info = jaxinfo_to_warpinfo(spikes_info)
    indices_warp_info = jaxinfo_to_warpinfo(indices_info)
    count_warp_info = jaxinfo_to_warpinfo(count_info)

    if spikes_info.dtype == jnp.bool_:
        @warp.kernel
        def mv(
            spikes: spikes_warp_info,
            indices: indices_warp_info,
            count: count_warp_info,
        ):
            i_col_block = warp.tid()
            if spikes[i_col_block]:
                idx = warp.atomic_add(count, 0, 1)
                indices[idx] = i_col_block

    else:
        @warp.kernel
        def mv(
            spikes: spikes_warp_info,
            indices: indices_warp_info,
            count: count_warp_info,
        ):
            i_col_block = warp.tid()
            if spikes[i_col_block] != 0.:
                idx = warp.atomic_add(count, 0, 1)
                indices[idx] = i_col_block

    def kernel(spikes, indices, count):
        dim = spikes_info.shape[0]
        fn = jax_kernel(mv, launch_dims=[dim], num_outputs=1, in_out_argnames=['count'])
        return fn(spikes, indices, count, jnp.zeros(count_info.shape, count_info.dtype))

    return kernel


def _binary_1d_array_index_pallas_kernel(
    spikes_info: jax.ShapeDtypeStruct,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add

    BLOCK_SIZE = 64

    def _raw_kernel(
        spikes_ref,
        indices_ref,
        count_ref,
    ):
        pid = pl.program_id(0)
        start = pid * BLOCK_SIZE
        idxs = start + jnp.arange(0, BLOCK_SIZE)

        # Check valid indices
        valid_mask = idxs < spikes_ref.shape[0]

        # Load values using direct indexing
        if spikes_info.dtype == jnp.bool_:
            x_vals = spikes_ref[idxs]
            value_mask = x_vals
        else:
            x_vals = spikes_ref[idxs]
            value_mask = x_vals != 0.0

        # Apply valid mask
        combined_mask = valid_mask & value_mask

        # Count non-zero elements in this block
        total_in_block = jnp.sum(combined_mask.astype(jnp.int32))

        # Atomically reserve space in global count
        base_pos = atomic_add(count_ref, (0,), total_in_block, mask=combined_mask[0:1])
        prefix_offsets = jnp.cumsum(combined_mask) - combined_mask

        # Calculate write positions
        write_positions = base_pos + prefix_offsets

        # Store indices using direct assignment
        indices_ref[write_positions] = jnp.where(combined_mask, idxs, 0)

    def kernel(spikes, indices, count):
        num_blocks = pl.cdiv(spikes_info.shape[0], BLOCK_SIZE)
        fn = pl.pallas_call(_raw_kernel, grid=(num_blocks,), out_shape=kwargs['outs'])
        return fn(spikes, indices, count)

    return kernel


def binary_1d_array_index_p_call(spikes):
    indices_info = jax.ShapeDtypeStruct([spikes.shape[0]], jnp.int32)
    count_info = jax.ShapeDtypeStruct([1], jnp.int32)
    return binary_1d_array_index_p(
        spikes,
        outs=[indices_info, count_info],
        spikes_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        indices_info=indices_info,
        count_info=count_info,
    )


def _binary_1d_array_index_jvp_spikes(spikes_dot, spikes, **kwargs):
    return binary_1d_array_index_p_call(spikes_dot)


def _binary_1d_array_index_transpose_rule(ct, spikes, indices, count, **kwargs):
    ct_indices, ct_count = ct
    if ad.is_undefined_primal(spikes):
        if type(ct_indices) is ad.Zero and type(ct_count) is ad.Zero:
            ct_spikes = ad.Zero(spikes)
        else:
            # Gradient: sum of gradients at indexed positions
            ct_spikes = jnp.zeros_like(spikes)
            if type(ct_indices) is not ad.Zero:
                valid_count = count if type(count) is not ad.Zero else 0
                if type(ct_indices) is not ad.Zero:
                    ct_spikes = ct_spikes.at[ct_indices].add(1.0)
        return ct_spikes, indices, count
    else:
        return spikes, indices, count


def _binary_1d_array_index_batching(args, axes, **kwargs):
    return general_batching_rule(binary_1d_array_index_p, args, axes, **kwargs)


binary_1d_array_index_p = XLACustomKernel('binary_1d_array_index')
binary_1d_array_index_p.def_numba_kernel(_binary_1d_array_index_numba_kernel)
binary_1d_array_index_p.def_warp_kernel(_binary_1d_array_index_warp_kernel)
binary_1d_array_index_p.def_pallas_kernel('gpu', _binary_1d_array_index_pallas_kernel)
binary_1d_array_index_p.def_pallas_kernel('tpu', _binary_1d_array_index_pallas_kernel)
binary_1d_array_index_p.def_jvp_rule2(_binary_1d_array_index_jvp_spikes)
binary_1d_array_index_p.def_transpose_rule(_binary_1d_array_index_transpose_rule)
binary_1d_array_index_p.def_batching_rule(_binary_1d_array_index_batching)
binary_1d_array_index_p.def_call(binary_1d_array_index_p_call)
binary_1d_array_index_p.def_tags('event', 'binary')


def _binary_1d_array_index_benchmark_data(*, platform):
    import numpy as _np
    n = 1000
    configs = []
    for bool_event in (True, False):
        if bool_event:
            spikes = jnp.asarray(_np.random.rand(n) > 0.9, dtype=jnp.bool_)
        else:
            spikes = jnp.asarray(
                _np.where(_np.random.rand(n) > 0.9, _np.random.rand(n), 0.0),
                dtype=jnp.float32,
            )
        name = "bool" if bool_event else "float"
        configs.append(BenchmarkConfig(name, (spikes,)))
    return configs


binary_1d_array_index_p.def_benchmark_data(_binary_1d_array_index_benchmark_data)


def binary_2d_array_index_p_call(spikes):
    out = jax.ShapeDtypeStruct([spikes.shape[0]], jnp.int32)
    raise NotImplementedError("2D binary array index extraction is not implemented yet.")
