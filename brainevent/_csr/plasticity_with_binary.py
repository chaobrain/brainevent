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

from typing import Union, Optional

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainevent._misc import generate_block_dim
from brainevent._op import XLACustomKernel, numba_kernel, jaxinfo_to_warpinfo
from brainevent._typing import MatrixShape

__all__ = [
    'binary_csr_plast',
    'binary_csr_plast_p',
    'csr2csc_on_post',
    'csr2csc_on_post_p',
]


def binary_csr_plast(
    weight: Union[u.Quantity, jax.Array],
    indices: Union[np.ndarray, jax.Array],
    indptr: Union[np.ndarray, jax.Array],
    pre_spike: jax.Array,
    post_trace: Union[u.Quantity, jax.Array],
    w_min: Optional[Union[u.Quantity, jax.Array]] = None,
    w_max: Optional[Union[u.Quantity, jax.Array]] = None,
    *,
    shape: MatrixShape,
):
    """Updates synaptic weights in CSR format based on presynaptic spike events and postsynaptic traces.

    This function implements a plasticity rule for sparse connectivity matrices where
    presynaptic spikes trigger weight updates modulated by postsynaptic trace values.
    The weight matrix is stored in Compressed Sparse Row (CSR) format.

    Specifically, for each presynaptic neuron, if it spikes ``pre_spike[i]`` is True,
    the weights of all synapses originating from that neuron are updated by adding the
    corresponding postsynaptic trace values ``post_trace[indices[index: index_end]]`` to
    the weights ``weight[index: index_end]``.

    Args:
        weight: Sparse synaptic weight array in CSR format, shape (n_nonzero,).
        indices: Column indices array of the CSR format, shape (n_nonzero,).
        indptr: Row pointers array of the CSR format, shape (n_rows + 1,).
        pre_spike: Binary/boolean array indicating presynaptic spike events, shape (n_pre,).
        post_trace: Postsynaptic trace values, shape (n_post,).
        w_min: Optional lower bound for weight clipping. Must have same units as weight.
        w_max: Optional upper bound for weight clipping. Must have same units as weight.
        shape: Tuple specifying the full matrix shape as (n_pre, n_post).

    Returns:
        Updated weight array with the same shape and units as the input weight.

    Note:
        The function handles unit conversion internally, ensuring that the post_trace
        is converted to the same unit as the weight before computation.
    """
    weight, wunit = u.split_mantissa_unit(weight)
    post_trace = u.Quantity(post_trace).to(wunit).mantissa
    weight = u.maybe_decimal(
        _csr_on_pre_prim_call(
            weight, indices, indptr, pre_spike, post_trace, shape=shape
        )[0] * wunit
    )
    weight = u.math.clip(weight, w_min, w_max)
    return weight


def _csr_on_pre_numba_kernel_generator(
    spike_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import numba

    if spike_info.dtype == jnp.bool_:
        @numba.njit(fastmath=True, cache=True)
        def kernel(weight, indices, indptr, pre_spike, post_trace, out_w):
            for i in range(pre_spike.shape[0]):
                if pre_spike[i]:
                    i_start = indptr[i]
                    i_end = indptr[i + 1]
                    out_w[i_start: i_end] += post_trace[indices[i_start: i_end]]
    else:
        @numba.njit(fastmath=True, cache=True)
        def kernel(weight, indices, indptr, pre_spike, post_trace, out_w):
            for i in range(pre_spike.shape[0]):
                if pre_spike[i] != 0.:
                    i_start = indptr[i]
                    i_end = indptr[i + 1]
                    out_w[i_start: i_end] += post_trace[indices[i_start: i_end]]

    def fn(weight, indices, indptr, pre_spike, post_trace):
        return numba_kernel(kernel, outs=kwargs['outs'], input_output_aliases={0: 0})(
            weight, indices, indptr, pre_spike, post_trace
        )

    return fn


def _csr_on_pre_warp_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    indptr_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    trace_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    indices_warp_info = jaxinfo_to_warpinfo(indices_info)
    indptr_warp_info = jaxinfo_to_warpinfo(indptr_info)
    spike_warp_info = jaxinfo_to_warpinfo(spike_info)
    trace_warp_info = jaxinfo_to_warpinfo(trace_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if spike_info.dtype == jnp.bool_:
        @warp.kernel
        def plasticity_kernel(
            weight: weight_warp_info,
            indices: indices_warp_info,
            indptr: indptr_warp_info,
            pre_spike: spike_warp_info,
            post_trace: trace_warp_info,
            out_w: out_warp_info,
        ):
            i = warp.tid()
            if pre_spike[i]:
                for j in range(indptr[i], indptr[i + 1]):
                    out_w[j] += post_trace[indices[j]]
    else:
        @warp.kernel
        def plasticity_kernel(
            weight: weight_warp_info,
            indices: indices_warp_info,
            indptr: indptr_warp_info,
            pre_spike: spike_warp_info,
            post_trace: trace_warp_info,
            out_w: out_warp_info,
        ):
            i = warp.tid()
            if pre_spike[i] != 0.:
                for j in range(indptr[i], indptr[i + 1]):
                    out_w[j] += post_trace[indices[j]]

    def kernel(weight, indices, indptr, pre_spike, post_trace):
        fn = jax_kernel(
            plasticity_kernel,
            launch_dims=shape[0],
            num_outputs=0,
            in_out_argnames=['out_w']
        )
        return fn(weight, indices, indptr, pre_spike, post_trace, weight.copy())

    return kernel


def _csr_on_pre_pallas_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    **kwargs
):
    from jax.experimental import pallas as pl

    block_dim = generate_block_dim(weight_info.shape[0], 512)

    if spike_info.dtype == jnp.bool_:
        def kernel_fn(weight_ref, indices_ref, indptr_ref, spike_ref, trace_ref, _, out_w_ref):
            i_row = pl.program_id(0)
            i_col_start = indptr_ref[i_row]
            i_col_end = indptr_ref[i_row + 1]
            num_blocks = (i_col_end - i_col_start + block_dim - 1) // block_dim

            @pl.when(spike_ref[i_row])
            def run():
                def loop_fn(i_block, _):
                    offset = i_col_start + i_block * block_dim
                    mask = (offset + jnp.arange(block_dim)) < i_col_end
                    post_ids = indices_ref[pl.dslice(offset, block_dim)]
                    current_w = out_w_ref[pl.dslice(offset, block_dim)]
                    post_trace = trace_ref[post_ids]
                    updated_w = jnp.where(mask, current_w + post_trace, current_w)
                    out_w_ref[pl.dslice(offset, block_dim)] = updated_w

                jax.lax.fori_loop(0, num_blocks, loop_fn, None)
    else:
        def kernel_fn(weight_ref, indices_ref, indptr_ref, spike_ref, trace_ref, _, out_w_ref):
            i_row = pl.program_id(0)
            i_col_start = indptr_ref[i_row]
            i_col_end = indptr_ref[i_row + 1]
            num_blocks = (i_col_end - i_col_start + block_dim - 1) // block_dim

            @pl.when(spike_ref[i_row] != 0.)
            def run():
                def loop_fn(i_block, _):
                    offset = i_col_start + i_block * block_dim
                    mask = (offset + jnp.arange(block_dim)) < i_col_end
                    post_ids = indices_ref[pl.dslice(offset, block_dim)]
                    current_w = out_w_ref[pl.dslice(offset, block_dim)]
                    post_trace = trace_ref[post_ids]
                    updated_w = jnp.where(mask, current_w + post_trace, current_w)
                    out_w_ref[pl.dslice(offset, block_dim)] = updated_w

                jax.lax.fori_loop(0, num_blocks, loop_fn, None)

    def kernel(weight, indices, indptr, pre_spike, post_trace):
        fn = pl.pallas_call(
            kernel_fn,
            grid=(shape[0],),
            input_output_aliases={5: 0},
            out_shape=kwargs['outs']
        )
        return fn(weight, indices, indptr, pre_spike, post_trace, weight)

    return kernel


def _csr_on_pre_prim_call(weight, indices, indptr, pre_spike, post_trace, *, shape):
    assert weight.ndim == 1, 'dense_one_pre only support 1D weight.'
    assert pre_spike.ndim == 1, 'pre_spike should be 1D.'
    assert post_trace.ndim == 1, 'post_trace should be 1D.'
    assert shape[0] == pre_spike.shape[0], f'pre_spike shape {pre_spike.shape} does not match with shape {shape}.'
    assert shape[1] == post_trace.shape[0], f'post_trace shape {post_trace.shape} does not match with shape {shape}.'
    assert weight.shape[0] == indices.shape[0], (
        f'weight shape {weight.shape}, indices shape {indices.shape}, indptr shape {indptr.shape} do not match.'
    )
    return binary_csr_plast_p(
        weight, indices, indptr, pre_spike, post_trace,
        outs=[jax.ShapeDtypeStruct(weight.shape, weight.dtype)],
        shape=shape,
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        spike_info=jax.ShapeDtypeStruct(pre_spike.shape, pre_spike.dtype),
        trace_info=jax.ShapeDtypeStruct(post_trace.shape, post_trace.dtype),
    )


binary_csr_plast_p = XLACustomKernel('binary_csr_plast')
binary_csr_plast_p.def_numba_kernel(_csr_on_pre_numba_kernel_generator)
binary_csr_plast_p.def_warp_kernel(_csr_on_pre_warp_kernel_generator)
binary_csr_plast_p.def_pallas_kernel('gpu', _csr_on_pre_pallas_kernel_generator)
binary_csr_plast_p.def_pallas_kernel('tpu', _csr_on_pre_pallas_kernel_generator)


def csr2csc_on_post(
    weight: Union[u.Quantity, jax.Array],
    indices: Union[np.ndarray, jax.Array],
    indptr: Union[np.ndarray, jax.Array],
    weight_indices: Union[np.ndarray, jax.Array],
    pre_trace: Union[u.Quantity, jax.Array],
    post_spike: jax.Array,
    w_min: Optional[Union[u.Quantity, jax.Array]] = None,
    w_max: Optional[Union[u.Quantity, jax.Array]] = None,
    *,
    shape: MatrixShape,
):
    """Updates synaptic weights in CSC format based on postsynaptic spike events and presynaptic traces.

    This function implements a plasticity rule for sparse connectivity matrices where
    postsynaptic spikes trigger weight updates modulated by presynaptic trace values.
    The weight matrix is stored in Compressed Sparse Column (CSC) format.

    Specifically, for each postsynaptic neuron, if it spikes ``post_spike[i]`` is True,
    the weights of all synapses targeting that neuron are updated by adding the
    corresponding presynaptic trace values ``pre_trace[indices[index: index_end]]`` to
    the weights ``weight[index: index_end]``.

    Args:
        weight: Sparse synaptic weight array in CSC format, shape (n_nonzero,).
        indices: Row indices array of the CSC format, shape (n_nonzero,).
        indptr: Column pointers array of the CSC format, shape (n_cols + 1,).
        weight_indices: Array of weight indices corresponding to the synapses, shape (n_nonzero,).
        pre_trace: Presynaptic trace values, shape (n_pre,).
        post_spike: Binary/boolean array indicating postsynaptic spike events, shape (n_post,).
        w_min: Optional lower bound for weight clipping. Must have same units as weight.
        w_max: Optional upper bound for weight clipping. Must have same units as weight.
        shape: Tuple specifying the full matrix shape as (n_pre, n_post).

    Returns:
        Updated weight array with the same shape and units as the input weight.

    Note:
        The function handles unit conversion internally, ensuring that the pre_trace
        is converted to the same unit as the weight before computation.
    """
    weight, wunit = u.split_mantissa_unit(weight)
    pre_trace = u.Quantity(pre_trace).to(wunit).mantissa
    weight = u.maybe_decimal(
        _csr2csc_on_post_prim_call(
            weight, indices, indptr, weight_indices, pre_trace, post_spike, shape=shape
        )[0] * wunit
    )
    weight = u.math.clip(weight, w_min, w_max)
    return weight


def _csr2csc_on_post_numba_kernel_generator(
    spike_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import numba

    # Note: Cannot parallelize due to potential race conditions when updating out_w[weight_ids]
    if spike_info.dtype == jnp.bool_:
        @numba.njit(fastmath=True, cache=True)
        def kernel(weight, indices, indptr, weight_indices, pre_trace, post_spike, out_w):
            for i in range(post_spike.shape[0]):
                if post_spike[i]:
                    index = indptr[i]
                    index_end = indptr[i + 1]
                    weight_ids = weight_indices[index: index_end]
                    pre_ids = indices[index: index_end]
                    out_w[weight_ids] += pre_trace[pre_ids]
    else:
        @numba.njit(fastmath=True, cache=True)
        def kernel(weight, indices, indptr, weight_indices, pre_trace, post_spike, out_w):
            for i in range(post_spike.shape[0]):
                if post_spike[i] != 0.:
                    index = indptr[i]
                    index_end = indptr[i + 1]
                    weight_ids = weight_indices[index: index_end]
                    pre_ids = indices[index: index_end]
                    out_w[weight_ids] += pre_trace[pre_ids]

    def fn(weight, indices, indptr, weight_indices, pre_trace, post_spike):
        return numba_kernel(kernel, outs=kwargs['outs'], input_output_aliases={0: 0})(
            weight, indices, indptr, weight_indices, pre_trace, post_spike
        )

    return fn


def _csr2csc_on_post_warp_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    indptr_info: jax.ShapeDtypeStruct,
    weight_indices_info: jax.ShapeDtypeStruct,
    trace_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    indices_warp_info = jaxinfo_to_warpinfo(indices_info)
    indptr_warp_info = jaxinfo_to_warpinfo(indptr_info)
    weight_indices_warp_info = jaxinfo_to_warpinfo(weight_indices_info)
    trace_warp_info = jaxinfo_to_warpinfo(trace_info)
    spike_warp_info = jaxinfo_to_warpinfo(spike_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if spike_info.dtype == jnp.bool_:
        @warp.kernel
        def plasticity_kernel(
            weight: weight_warp_info,
            indices: indices_warp_info,
            indptr: indptr_warp_info,
            weight_indices: weight_indices_warp_info,
            pre_trace: trace_warp_info,
            post_spike: spike_warp_info,
            out_w: out_warp_info,
        ):
            i = warp.tid()
            if post_spike[i]:
                for j in range(indptr[i], indptr[i + 1]):
                    weight_id = weight_indices[j]
                    pre_id = indices[j]
                    out_w[weight_id] += pre_trace[pre_id]
    else:
        @warp.kernel
        def plasticity_kernel(
            weight: weight_warp_info,
            indices: indices_warp_info,
            indptr: indptr_warp_info,
            weight_indices: weight_indices_warp_info,
            pre_trace: trace_warp_info,
            post_spike: spike_warp_info,
            out_w: out_warp_info,
        ):
            i = warp.tid()
            if post_spike[i] != 0.:
                for j in range(indptr[i], indptr[i + 1]):
                    weight_id = weight_indices[j]
                    pre_id = indices[j]
                    out_w[weight_id] += pre_trace[pre_id]

    def kernel(weight, indices, indptr, weight_indices, pre_trace, post_spike):
        fn = jax_kernel(
            plasticity_kernel,
            launch_dims=shape[1],
            num_outputs=0,
            in_out_argnames=['out_w']
        )
        return fn(weight, indices, indptr, weight_indices, pre_trace, post_spike, weight.copy())

    return kernel


def _csr2csc_on_post_pallas_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    **kwargs
):
    from jax.experimental import pallas as pl

    block_dim = generate_block_dim(weight_info.shape[0], 512)

    if spike_info.dtype == jnp.bool_:
        def kernel_fn(weight_ref, indices_ref, indptr_ref, weight_indices_ref, trace_ref, spike_ref, _, out_w_ref):
            i_col = pl.program_id(0)
            i_row_start = indptr_ref[i_col]
            i_row_end = indptr_ref[i_col + 1]
            num_blocks = (i_row_end - i_row_start + block_dim - 1) // block_dim

            @pl.when(spike_ref[i_col])
            def run():
                def loop_fn(i_block, _):
                    offset = i_row_start + i_block * block_dim
                    mask = (offset + jnp.arange(block_dim)) < i_row_end
                    pre_ids = indices_ref[pl.dslice(offset, block_dim)]
                    weight_ids = weight_indices_ref[pl.dslice(offset, block_dim)]
                    pre_trace_vals = trace_ref[pre_ids]
                    current_w = out_w_ref[weight_ids]
                    updated_w = jnp.where(mask, current_w + pre_trace_vals, current_w)

                    # Scatter update: for each position, update out_w_ref at weight_ids
                    # Using a loop to handle potential non-contiguous writes
                    def scatter_fn(j, _):
                        w_id = weight_ids[j]
                        out_w_ref[w_id] = jnp.where(mask[j], out_w_ref[w_id] + pre_trace_vals[j], out_w_ref[w_id])

                    jax.lax.fori_loop(0, block_dim, scatter_fn, None)

                jax.lax.fori_loop(0, num_blocks, loop_fn, None)
    else:
        def kernel_fn(weight_ref, indices_ref, indptr_ref, weight_indices_ref, trace_ref, spike_ref, _, out_w_ref):
            i_col = pl.program_id(0)
            i_row_start = indptr_ref[i_col]
            i_row_end = indptr_ref[i_col + 1]
            num_blocks = (i_row_end - i_row_start + block_dim - 1) // block_dim

            @pl.when(spike_ref[i_col] != 0.)
            def run():
                def loop_fn(i_block, _):
                    offset = i_row_start + i_block * block_dim
                    mask = (offset + jnp.arange(block_dim)) < i_row_end
                    pre_ids = indices_ref[pl.dslice(offset, block_dim)]
                    weight_ids = weight_indices_ref[pl.dslice(offset, block_dim)]
                    pre_trace_vals = trace_ref[pre_ids]

                    # Scatter update: for each position, update out_w_ref at weight_ids
                    def scatter_fn(j, _):
                        w_id = weight_ids[j]
                        out_w_ref[w_id] = jnp.where(mask[j], out_w_ref[w_id] + pre_trace_vals[j], out_w_ref[w_id])

                    jax.lax.fori_loop(0, block_dim, scatter_fn, None)

                jax.lax.fori_loop(0, num_blocks, loop_fn, None)

    def kernel(weight, indices, indptr, weight_indices, pre_trace, post_spike):
        fn = pl.pallas_call(
            kernel_fn,
            grid=(shape[1],),
            input_output_aliases={6: 0},
            out_shape=kwargs['outs']
        )
        return fn(weight, indices, indptr, weight_indices, pre_trace, post_spike, weight)

    return kernel


def _csr2csc_on_post_prim_call(weight, indices, indptr, weight_indices, pre_trace, post_spike, *, shape):
    assert weight.ndim == 1, 'dense_one_post only support 1D weight.'
    assert post_spike.ndim == 1, 'post_spike should be 1D.'
    assert pre_trace.ndim == 1, 'pre_trace should be 1D.'
    assert shape[1] == post_spike.shape[0], f'post_spike shape {post_spike.shape} does not match with shape {shape}.'
    assert shape[0] == pre_trace.shape[0], f'pre_trace shape {pre_trace.shape} does not match with shape {shape}.'
    assert weight.shape == weight_indices.shape == indices.shape, (
        f'weight shape {weight.shape}, weight_indices shape {weight_indices.shape}, '
        f'indices shape {indices.shape}, indptr shape {indptr.shape} do not match.'
    )
    return csr2csc_on_post_p(
        weight, indices, indptr, weight_indices, pre_trace, post_spike,
        outs=[jax.ShapeDtypeStruct(weight.shape, weight.dtype)],
        shape=shape,
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        weight_indices_info=jax.ShapeDtypeStruct(weight_indices.shape, weight_indices.dtype),
        trace_info=jax.ShapeDtypeStruct(pre_trace.shape, pre_trace.dtype),
        spike_info=jax.ShapeDtypeStruct(post_spike.shape, post_spike.dtype),
    )


csr2csc_on_post_p = XLACustomKernel('csr2csc_on_post')
csr2csc_on_post_p.def_numba_kernel(_csr2csc_on_post_numba_kernel_generator)
csr2csc_on_post_p.def_warp_kernel(_csr2csc_on_post_warp_kernel_generator)
csr2csc_on_post_p.def_pallas_kernel('gpu', _csr2csc_on_post_pallas_kernel_generator)
csr2csc_on_post_p.def_pallas_kernel('tpu', _csr2csc_on_post_pallas_kernel_generator)

csr_on_pre = binary_csr_plast
