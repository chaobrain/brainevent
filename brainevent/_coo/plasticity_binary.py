# Copyright 2024- BrainX Ecosystem Limited. All Rights Reserved.
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

from typing import Union, Optional

import brainunit as u
import jax
import jax.numpy as jnp

from brainevent._misc import generate_block_dim
from brainevent._op import XLACustomKernel, numba_kernel, jaxinfo_to_warpinfo

__all__ = [
    'plast_coo_on_binary_pre',
    'plast_coo_on_binary_post',
    'plast_coo_on_binary_pre_p',
    'plast_coo_on_binary_post_p',
]


def plast_coo_on_binary_pre(
    weight: Union[u.Quantity, jax.Array],
    pre_ids: jax.Array,
    post_ids: jax.Array,
    pre_spike: jax.Array,
    post_trace: Union[u.Quantity, jax.Array],
    w_min: Optional[Union[u.Quantity, jax.Array]] = None,
    w_max: Optional[Union[u.Quantity, jax.Array]] = None,
):
    """Updates synaptic weights in COO format based on presynaptic spike events and postsynaptic traces.

    This function implements a plasticity rule for sparse connectivity matrices where
    presynaptic spikes trigger weight updates modulated by postsynaptic trace values.

    Specifically, for each synapse, if the presynaptic neuron spikes ``pre_spike[i]`` is True,
    the weight of the synapse is updated by adding the corresponding postsynaptic trace value
    ``post_trace[post_ids[i]]`` to the weight ``weight[i]``.

    Args:
        weight: Sparse synaptic weight array in COO format, shape (n_synapses,).
        pre_ids: Array of presynaptic neuron indices for each synapse, shape (n_synapses,).
        post_ids: Array of postsynaptic neuron indices for each synapse, shape (n_synapses,).
        pre_spike: Binary/boolean array indicating presynaptic spike events, shape (n_pre,).
        post_trace: Postsynaptic trace values, shape (n_post,).
        w_min: Optional lower bound for weight clipping. Must have same units as weight.
        w_max: Optional upper bound for weight clipping. Must have same units as weight.

    Returns:
        Updated weight array with the same shape and units as the input weight.

    Note:
        The function handles unit conversion internally, ensuring that the post_trace
        is converted to the same unit as the weight before computation.
    """
    weight, wunit = u.split_mantissa_unit(weight)
    post_trace = u.Quantity(post_trace).to(wunit).mantissa
    weight = u.maybe_decimal(_coo_on_pre_prim_call(weight, pre_ids, post_ids, pre_spike, post_trace)[0] * wunit)
    weight = u.math.clip(weight, w_min, w_max)
    return weight


def _coo_on_pre_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import numba

    @numba.njit(fastmath=True)
    def kernel(weight, pre_ids, post_ids, pre_spike, post_trace, out_w):
        out_w[:] = weight[:]
        for i in range(weight.shape[0]):
            if pre_spike[pre_ids[i]]:
                out_w[i] += post_trace[post_ids[i]]

    def run(weight, pre_ids, post_ids, pre_spike, post_trace):
        return numba_kernel(kernel, outs=kwargs['outs'])(weight, pre_ids, post_ids, pre_spike, post_trace)

    return run


def _coo_on_pre_warp_kernel(
    weight_info: jax.ShapeDtypeStruct,
    pre_ids_info: jax.ShapeDtypeStruct,
    post_ids_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    trace_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    pre_ids_warp_info = jaxinfo_to_warpinfo(pre_ids_info)
    post_ids_warp_info = jaxinfo_to_warpinfo(post_ids_info)
    spike_warp_info = jaxinfo_to_warpinfo(spike_info)
    trace_warp_info = jaxinfo_to_warpinfo(trace_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if spike_info.dtype == jnp.bool_:
        @warp.kernel
        def plast_kernel(
            weight: weight_warp_info,
            pre_ids: pre_ids_warp_info,
            post_ids: post_ids_warp_info,
            pre_spike: spike_warp_info,
            post_trace: trace_warp_info,
            out_w: out_warp_info,
        ):
            i = warp.tid()
            out_w[i] = weight[i]
            if pre_spike[pre_ids[i]]:
                out_w[i] += post_trace[post_ids[i]]
    else:
        @warp.kernel
        def plast_kernel(
            weight: weight_warp_info,
            pre_ids: pre_ids_warp_info,
            post_ids: post_ids_warp_info,
            pre_spike: spike_warp_info,
            post_trace: trace_warp_info,
            out_w: out_warp_info,
        ):
            i = warp.tid()
            out_w[i] = weight[i]
            if pre_spike[pre_ids[i]] != 0.:
                out_w[i] += post_trace[post_ids[i]]

    n_syn = weight_info.shape[0]
    out_info = kwargs['outs'][0]

    def run(weight, pre_ids, post_ids, pre_spike, post_trace):
        fn = jax_kernel(plast_kernel, launch_dims=n_syn, num_outputs=1, in_out_argnames=['out_w'])
        return fn(weight, pre_ids, post_ids, pre_spike, post_trace, jnp.zeros(out_info.shape, out_info.dtype))

    return run


def _coo_on_pre_pallas_gpu_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    **kwargs
):
    from jax.experimental import pallas as pl

    n_syn = weight_info.shape[0]
    block_dim = generate_block_dim(n_syn)
    block_dim = 32 if block_dim < 32 else block_dim

    if spike_info.dtype == jnp.bool_:
        def kernel(weight_ref, pre_ids_ref, post_ids_ref, spike_ref, trace_ref, _, out_w_ref):
            i = pl.program_id(0)
            i_start = i * block_dim
            mask = i_start + jnp.arange(block_dim) < n_syn

            # Read indices
            pre_ids = pre_ids_ref[pl.dslice(i_start, block_dim)]
            post_ids = post_ids_ref[pl.dslice(i_start, block_dim)]

            # Read spikes and compute spike mask
            spikes = spike_ref[pre_ids]
            all_mask = mask & spikes

            # Read traces and weights
            traces = trace_ref[post_ids]
            old_w = weight_ref[pl.dslice(i_start, block_dim)]

            # Compute update (only where spike occurred)
            new_w = jnp.where(all_mask, old_w + traces, old_w)

            # Write result
            out_w_ref[pl.dslice(i_start, block_dim)] = jnp.where(mask, new_w, out_w_ref[pl.dslice(i_start, block_dim)])
    else:
        def kernel(weight_ref, pre_ids_ref, post_ids_ref, spike_ref, trace_ref, _, out_w_ref):
            i = pl.program_id(0)
            i_start = i * block_dim
            mask = i_start + jnp.arange(block_dim) < n_syn

            # Read indices
            pre_ids = pre_ids_ref[pl.dslice(i_start, block_dim)]
            post_ids = post_ids_ref[pl.dslice(i_start, block_dim)]

            # Read spikes and compute spike mask
            spikes = spike_ref[pre_ids]
            all_mask = mask & (spikes != 0.)

            # Read traces and weights
            traces = trace_ref[post_ids]
            old_w = weight_ref[pl.dslice(i_start, block_dim)]

            # Compute update (only where spike occurred)
            new_w = jnp.where(all_mask, old_w + traces, old_w)

            # Write result
            out_w_ref[pl.dslice(i_start, block_dim)] = jnp.where(mask, new_w, out_w_ref[pl.dslice(i_start, block_dim)])

    def run(weight, pre_ids, post_ids, pre_spike, post_trace):
        fn = pl.pallas_call(
            kernel,
            grid=(pl.cdiv(n_syn, block_dim),),
            input_output_aliases={5: 0},
            out_shape=kwargs['outs']
        )
        out = jnp.zeros(kwargs['outs'][0].shape, dtype=kwargs['outs'][0].dtype)
        return fn(weight, pre_ids, post_ids, pre_spike, post_trace, out)

    return run


def _coo_on_pre_prim_call(weight, pre_ids, post_ids, pre_spike, post_trace):
    assert weight.ndim == 1, 'coo_on_pre only supports 1D weight.'
    assert weight.shape == pre_ids.shape == post_ids.shape, (
        f'weight shape ({weight.shape}), '
        f'pre_ids shape ({pre_ids.shape}), '
        f'and post_ids shape ({post_ids.shape}) '
        'should all match.'
    )
    assert pre_spike.ndim == 1, 'pre_spike should be 1D.'
    assert post_trace.ndim == 1, 'post_trace should be 1D.'
    return plast_coo_on_binary_pre_p(
        weight, pre_ids, post_ids, pre_spike, post_trace,
        outs=[jax.ShapeDtypeStruct(weight.shape, weight.dtype)],
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        pre_ids_info=jax.ShapeDtypeStruct(pre_ids.shape, pre_ids.dtype),
        post_ids_info=jax.ShapeDtypeStruct(post_ids.shape, post_ids.dtype),
        spike_info=jax.ShapeDtypeStruct(pre_spike.shape, pre_spike.dtype),
        trace_info=jax.ShapeDtypeStruct(post_trace.shape, post_trace.dtype),
    )


plast_coo_on_binary_pre_p = XLACustomKernel('coo_on_pre')
plast_coo_on_binary_pre_p.def_numba_kernel(_coo_on_pre_numba_kernel)
plast_coo_on_binary_pre_p.def_warp_kernel(_coo_on_pre_warp_kernel)
plast_coo_on_binary_pre_p.def_pallas_kernel('gpu', _coo_on_pre_pallas_gpu_kernel)
plast_coo_on_binary_pre_p.def_pallas_kernel('tpu', _coo_on_pre_pallas_gpu_kernel)
plast_coo_on_binary_pre_p.def_call(_coo_on_pre_prim_call)
plast_coo_on_binary_pre_p.def_tags('coo', 'plasticity')


def _plast_coo_pre_benchmark_data(*, platform):
    import numpy as _np
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for bool_event in (True, False):
        nnz = max(1, int(n_pre * n_post * prob))
        pre_ids = _np.random.randint(0, n_pre, nnz, dtype=_np.int32)
        post_ids = _np.random.randint(0, n_post, nnz, dtype=_np.int32)
        weight = jnp.ones(nnz, dtype=dtype)
        if bool_event:
            pre_spike = jnp.asarray(_np.random.rand(n_pre) > 0.5, dtype=jnp.bool_)
        else:
            pre_spike = jnp.asarray(_np.random.rand(n_pre), dtype=dtype)
        post_trace = jnp.asarray(_np.random.randn(n_post), dtype=dtype)
        name = f"{'bool' if bool_event else 'float'}"
        configs.append((name, (weight, jnp.asarray(pre_ids), jnp.asarray(post_ids), pre_spike, post_trace), {}))
    return configs


plast_coo_on_binary_pre_p.def_benchmark_data(_plast_coo_pre_benchmark_data)


# =============================================================================
# COO On-Post Plasticity
# =============================================================================


def plast_coo_on_binary_post(
    weight: Union[u.Quantity, jax.Array],
    pre_ids: jax.Array,
    post_ids: jax.Array,
    pre_trace: Union[u.Quantity, jax.Array],
    post_spike: jax.Array,
    w_min: Optional[Union[u.Quantity, jax.Array]] = None,
    w_max: Optional[Union[u.Quantity, jax.Array]] = None,
):
    """Updates synaptic weights in COO format based on postsynaptic spike events and presynaptic traces.

    This function implements a plasticity rule for sparse connectivity matrices where
    postsynaptic spikes trigger weight updates modulated by presynaptic trace values.

    Specifically, for each synapse, if the postsynaptic neuron spikes ``post_spike[post_ids[i]]`` is True,
    the weight of the synapse is updated by adding the corresponding presynaptic trace value
    ``pre_trace[pre_ids[i]]`` to the weight ``weight[i]``.

    Args:
        weight: Sparse synaptic weight array in COO format, shape (n_synapses,).
        pre_ids: Array of presynaptic neuron indices for each synapse, shape (n_synapses,).
        post_ids: Array of postsynaptic neuron indices for each synapse, shape (n_synapses,).
        pre_trace: Presynaptic trace values, shape (n_pre,).
        post_spike: Binary/boolean array indicating postsynaptic spike events, shape (n_post,).
        w_min: Optional lower bound for weight clipping. Must have same units as weight.
        w_max: Optional upper bound for weight clipping. Must have same units as weight.

    Returns:
        Updated weight array with the same shape and units as the input weight.

    Note:
        The function handles unit conversion internally, ensuring that the pre_trace
        is converted to the same unit as the weight before computation.
    """
    weight, wunit = u.split_mantissa_unit(weight)
    pre_trace = u.Quantity(pre_trace).to(wunit).mantissa
    weight = u.maybe_decimal(_coo_on_post_prim_call(weight, pre_ids, post_ids, pre_trace, post_spike)[0] * wunit)
    weight = u.math.clip(weight, w_min, w_max)
    return weight


def _coo_on_post_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import numba

    @numba.njit(fastmath=True)
    def kernel(weight, pre_ids, post_ids, pre_trace, post_spike, out_w):
        out_w[:] = weight[:]
        for i in range(weight.shape[0]):
            if post_spike[post_ids[i]]:
                out_w[i] += pre_trace[pre_ids[i]]

    def run(weight, pre_ids, post_ids, pre_trace, post_spike):
        return numba_kernel(kernel, outs=kwargs['outs'])(weight, pre_ids, post_ids, pre_trace, post_spike)

    return run


def _coo_on_post_warp_kernel(
    weight_info: jax.ShapeDtypeStruct,
    pre_ids_info: jax.ShapeDtypeStruct,
    post_ids_info: jax.ShapeDtypeStruct,
    trace_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    pre_ids_warp_info = jaxinfo_to_warpinfo(pre_ids_info)
    post_ids_warp_info = jaxinfo_to_warpinfo(post_ids_info)
    trace_warp_info = jaxinfo_to_warpinfo(trace_info)
    spike_warp_info = jaxinfo_to_warpinfo(spike_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if spike_info.dtype == jnp.bool_:
        @warp.kernel
        def plast_kernel(
            weight: weight_warp_info,
            pre_ids: pre_ids_warp_info,
            post_ids: post_ids_warp_info,
            pre_trace: trace_warp_info,
            post_spike: spike_warp_info,
            out_w: out_warp_info,
        ):
            i = warp.tid()
            out_w[i] = weight[i]
            if post_spike[post_ids[i]]:
                out_w[i] += pre_trace[pre_ids[i]]
    else:
        @warp.kernel
        def plast_kernel(
            weight: weight_warp_info,
            pre_ids: pre_ids_warp_info,
            post_ids: post_ids_warp_info,
            pre_trace: trace_warp_info,
            post_spike: spike_warp_info,
            out_w: out_warp_info,
        ):
            i = warp.tid()
            out_w[i] = weight[i]
            if post_spike[post_ids[i]] != 0.:
                out_w[i] += pre_trace[pre_ids[i]]

    n_syn = weight_info.shape[0]
    out_info = kwargs['outs'][0]

    def run(weight, pre_ids, post_ids, pre_trace, post_spike):
        fn = jax_kernel(plast_kernel, launch_dims=n_syn, num_outputs=1, in_out_argnames=['out_w'])
        return fn(weight, pre_ids, post_ids, pre_trace, post_spike, jnp.zeros(out_info.shape, out_info.dtype))

    return run


def _coo_on_post_pallas_gpu_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    **kwargs
):
    from jax.experimental import pallas as pl

    n_syn = weight_info.shape[0]
    block_dim = generate_block_dim(n_syn)
    block_dim = 32 if block_dim < 32 else block_dim

    if spike_info.dtype == jnp.bool_:
        def kernel(weight_ref, pre_ids_ref, post_ids_ref, trace_ref, spike_ref, _, out_w_ref):
            i = pl.program_id(0)
            i_start = i * block_dim
            mask = i_start + jnp.arange(block_dim) < n_syn

            # Read indices
            pre_ids = pre_ids_ref[pl.dslice(i_start, block_dim)]
            post_ids = post_ids_ref[pl.dslice(i_start, block_dim)]

            # Read spikes and compute spike mask
            spikes = spike_ref[post_ids]
            all_mask = mask & spikes

            # Read traces and weights
            traces = trace_ref[pre_ids]
            old_w = weight_ref[pl.dslice(i_start, block_dim)]

            # Compute update (only where spike occurred)
            new_w = jnp.where(all_mask, old_w + traces, old_w)

            # Write result
            out_w_ref[pl.dslice(i_start, block_dim)] = jnp.where(mask, new_w, out_w_ref[pl.dslice(i_start, block_dim)])
    else:
        def kernel(weight_ref, pre_ids_ref, post_ids_ref, trace_ref, spike_ref, _, out_w_ref):
            i = pl.program_id(0)
            i_start = i * block_dim
            mask = i_start + jnp.arange(block_dim) < n_syn

            # Read indices
            pre_ids = pre_ids_ref[pl.dslice(i_start, block_dim)]
            post_ids = post_ids_ref[pl.dslice(i_start, block_dim)]

            # Read spikes and compute spike mask
            spikes = spike_ref[post_ids]
            all_mask = mask & (spikes != 0.)

            # Read traces and weights
            traces = trace_ref[pre_ids]
            old_w = weight_ref[pl.dslice(i_start, block_dim)]

            # Compute update (only where spike occurred)
            new_w = jnp.where(all_mask, old_w + traces, old_w)

            # Write result
            out_w_ref[pl.dslice(i_start, block_dim)] = jnp.where(mask, new_w, out_w_ref[pl.dslice(i_start, block_dim)])

    def run(weight, pre_ids, post_ids, pre_trace, post_spike):
        fn = pl.pallas_call(
            kernel,
            grid=(pl.cdiv(n_syn, block_dim),),
            input_output_aliases={5: 0},
            out_shape=kwargs['outs']
        )
        out = jnp.zeros(kwargs['outs'][0].shape, dtype=kwargs['outs'][0].dtype)
        return fn(weight, pre_ids, post_ids, pre_trace, post_spike, out)

    return run


def _coo_on_post_prim_call(weight, pre_ids, post_ids, pre_trace, post_spike):
    assert weight.ndim == 1, 'coo_on_post only supports 1D weight.'
    assert weight.shape == pre_ids.shape == post_ids.shape, (
        f'weight shape ({weight.shape}), '
        f'pre_ids shape ({pre_ids.shape}), '
        f'and post_ids shape ({post_ids.shape}) '
        'should all match.'
    )
    assert pre_trace.ndim == 1, 'pre_trace should be 1D.'
    assert post_spike.ndim == 1, 'post_spike should be 1D.'
    return plast_coo_on_binary_post_p(
        weight, pre_ids, post_ids, pre_trace, post_spike,
        outs=[jax.ShapeDtypeStruct(weight.shape, weight.dtype)],
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        pre_ids_info=jax.ShapeDtypeStruct(pre_ids.shape, pre_ids.dtype),
        post_ids_info=jax.ShapeDtypeStruct(post_ids.shape, post_ids.dtype),
        trace_info=jax.ShapeDtypeStruct(pre_trace.shape, pre_trace.dtype),
        spike_info=jax.ShapeDtypeStruct(post_spike.shape, post_spike.dtype),
    )


plast_coo_on_binary_post_p = XLACustomKernel('coo_on_post')
plast_coo_on_binary_post_p.def_numba_kernel(_coo_on_post_numba_kernel)
plast_coo_on_binary_post_p.def_warp_kernel(_coo_on_post_warp_kernel)
plast_coo_on_binary_post_p.def_pallas_kernel('gpu', _coo_on_post_pallas_gpu_kernel)
plast_coo_on_binary_post_p.def_pallas_kernel('tpu', _coo_on_post_pallas_gpu_kernel)
plast_coo_on_binary_post_p.def_call(_coo_on_post_prim_call)
plast_coo_on_binary_post_p.def_tags('coo', 'plasticity')


def _plast_coo_post_benchmark_data(*, platform):
    import numpy as _np
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for bool_event in (True, False):
        nnz = max(1, int(n_pre * n_post * prob))
        pre_ids = _np.random.randint(0, n_pre, nnz, dtype=_np.int32)
        post_ids = _np.random.randint(0, n_post, nnz, dtype=_np.int32)
        weight = jnp.ones(nnz, dtype=dtype)
        pre_trace = jnp.asarray(_np.random.randn(n_pre), dtype=dtype)
        if bool_event:
            post_spike = jnp.asarray(_np.random.rand(n_post) > 0.5, dtype=jnp.bool_)
        else:
            post_spike = jnp.asarray(_np.random.rand(n_post), dtype=dtype)
        name = f"{'bool' if bool_event else 'float'}"
        configs.append((name, (weight, jnp.asarray(pre_ids), jnp.asarray(post_ids), pre_trace, post_spike), {}))
    return configs


plast_coo_on_binary_post_p.def_benchmark_data(_plast_coo_post_benchmark_data)
