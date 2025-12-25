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
import numpy as np

from brainevent._compatible_import import pallas as pl
from brainevent._misc import generate_block_dim
from brainevent._op.main import XLACustomKernel
from brainevent._op.op_numba import numba_kernel
from brainevent._op.op_pallas import pallas_kernel


def coo_on_pre(
    weight: Union[u.Quantity, jax.Array],
    pre_ids: Union[np.ndarray, jax.Array],
    post_ids: Union[np.ndarray, jax.Array],
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


def _coo_on_pre_numba_kernel_generator(**kwargs):
    def kernel(weight, pre_ids, post_ids, pre_spike, post_trace, out_w):
        for i in range(out_w.shape[0]):
            i_pre = pre_ids[i]
            if pre_spike[i_pre]:
                out_w[i] += post_trace[post_ids[i]]

    return numba_kernel(kernel, input_output_aliases={0: 0})


def _coo_on_pre_pallas_kernel_generator(weight_info, **kwargs):
    block_dim = generate_block_dim(weight_info.shape[0], 1024)

    def kernel(weight_ref, pre_ids_ref, post_ids_ref, spike_ref, trace_ref, out_w_ref):
        i_block = pl.program_id(0)
        i_post_start = i_block * block_dim
        mask = jax.numpy.arange(block_dim) + i_post_start < weight_info.shape[0]
        pre_ids = pl.load(pre_ids_ref, pl.dslice(i_post_start, block_dim), mask=mask)
        spikes = pl.load(spike_ref, pre_ids, mask=mask)
        all_mask = spikes & mask
        post_ids = pl.load(post_ids_ref, pl.dslice(i_post_start, block_dim), mask=all_mask)
        post_trace = pl.load(trace_ref, post_ids, mask=all_mask)
        old_weight = pl.load(out_w_ref, pl.dslice(i_post_start, block_dim), mask=all_mask)
        pl.store(out_w_ref, pl.dslice(i_post_start, block_dim), old_weight + post_trace, mask=all_mask)

    return pallas_kernel(kernel, input_output_aliases={0: 0}, tile=(pl.cdiv(weight_info.shape[0], block_dim),))


def _coo_on_pre_prim_call(weight, pre_ids, post_ids, pre_spike, post_trace):
    assert weight.ndim == 1, 'doo_one_pre only support 1D weight.'
    assert weight.shape == pre_ids.shape == post_ids.shape, (
        f'weight shape ({weight.shape}), '
        f'pre_ids shape ({pre_ids.shape}), '
        f'and post_ids shape ({post_ids.shape}) '
        'should all match.'
    )
    assert pre_spike.ndim == 1, 'pre_spike should be 1D.'
    assert post_trace.ndim == 1, 'post_trace should be 1D.'
    return _coo_on_pre_prim(
        weight, pre_ids, post_ids, pre_spike, post_trace,
        outs=[jax.ShapeDtypeStruct(weight.shape, weight.dtype)],
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
    )


_coo_on_pre_prim = XLACustomKernel('coo_on_pre')
_coo_on_pre_prim.def_cpu_kernel(_coo_on_pre_numba_kernel_generator)
_coo_on_pre_prim.def_gpu_kernel(pallas=_coo_on_pre_pallas_kernel_generator)
_coo_on_pre_prim.def_tpu_kernel(_coo_on_pre_pallas_kernel_generator)


def coo_on_post(
    weight: Union[u.Quantity, jax.Array],
    pre_ids: Union[np.ndarray, jax.Array],
    post_ids: Union[np.ndarray, jax.Array],
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


def _coo_on_post_numba_kernel_generator(**kwargs):
    def kernel(weight, pre_ids, post_ids, pre_trace, post_spike, out_w):
        for i in range(out_w.shape[0]):
            i_post = post_ids[i]
            if post_spike[i_post]:
                out_w[i] += pre_trace[pre_ids[i]]

    return numba_kernel(kernel, input_output_aliases={0: 0})


def _coo_on_post_pallas_kernel_generator(weight_info, **kwargs):
    block_dim = generate_block_dim(weight_info.shape[0], 1024)

    def kernel(weight_ref, pre_ids_ref, post_ids_ref, trace_ref, spike_ref, out_w_ref):
        i_block = pl.program_id(0)
        i_post_start = i_block * block_dim
        mask = jax.numpy.arange(block_dim) + i_post_start < weight_info.shape[0]
        post_ids = pl.load(post_ids_ref, pl.dslice(i_post_start, block_dim), mask=mask)
        spikes = pl.load(spike_ref, post_ids, mask=mask)
        all_mask = spikes & mask
        pre_ids = pl.load(pre_ids_ref, pl.dslice(i_post_start, block_dim), mask=all_mask)
        pre_trace = pl.load(trace_ref, pre_ids, mask=all_mask)
        old_weight = pl.load(out_w_ref, pl.dslice(i_post_start, block_dim), mask=all_mask)
        pl.store(out_w_ref, pl.dslice(i_post_start, block_dim), old_weight + pre_trace, mask=all_mask)

    return pallas_kernel(kernel, input_output_aliases={0: 0}, tile=(pl.cdiv(weight_info.shape[0], block_dim),))


def _coo_on_post_prim_call(weight, pre_ids, post_ids, pre_trace, post_spike):
    assert weight.ndim == 1, 'coo_on_post only support 1D weight.'
    assert weight.shape == pre_ids.shape == post_ids.shape, (
        f'weight shape ({weight.shape}), '
        f'pre_ids shape ({pre_ids.shape}), '
        f'and post_ids shape ({post_ids.shape}) '
        'should all match.'
    )
    assert pre_trace.ndim == 1, 'pre_trace should be 1D.'
    assert post_spike.ndim == 1, 'post_spike should be 1D.'
    return _coo_on_post_prim(
        weight, pre_ids, post_ids, pre_trace, post_spike,
        outs=[jax.ShapeDtypeStruct(weight.shape, weight.dtype)],
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
    )


_coo_on_post_prim = XLACustomKernel('coo_on_post')
_coo_on_post_prim.def_cpu_kernel(_coo_on_post_numba_kernel_generator)
_coo_on_post_prim.def_gpu_kernel(pallas=_coo_on_post_pallas_kernel_generator)
_coo_on_post_prim.def_tpu_kernel(_coo_on_post_pallas_kernel_generator)
