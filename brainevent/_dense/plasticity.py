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

from brainevent._compatible_import import pallas as pl
from brainevent._misc import generate_block_dim
from brainevent._op.main import XLACustomKernel
from brainevent._op.op_numba import numba_kernel
from brainevent._op.op_pallas import pallas_kernel


def dense_on_pre(
    weight: Union[u.Quantity, jax.Array],
    pre_spike: jax.Array,
    post_trace: Union[u.Quantity, jax.Array],
    w_min: Optional[Union[u.Quantity, jax.Array]] = None,
    w_max: Optional[Union[u.Quantity, jax.Array]] = None,
):
    """Updates synaptic weights based on presynaptic spike events and postsynaptic traces.

    This function implements a plasticity rule where presynaptic spikes trigger weight updates
    modulated by postsynaptic trace values. The weight update is performed element-wise.

    Args:
        weight: Synaptic weight matrix of shape (n_pre, n_post).
        pre_spike: Binary/boolean array indicating presynaptic spike events, shape (n_pre,).
        post_trace: Postsynaptic trace values, shape (n_post,).
        w_min: Optional lower bound for weight clipping. Must have same units as weight.
        w_max: Optional upper bound for weight clipping. Must have same units as weight.

    Returns:
        Updated weight matrix with the same shape and units as the input weight.

    Note:
        The function handles unit conversion internally, ensuring that the pre_trace
        is converted to the same unit as the weight before computation.
    """
    weight, wunit = u.split_mantissa_unit(weight)
    post_trace = u.Quantity(post_trace).to(wunit).mantissa
    weight = u.maybe_decimal(_dense_on_pre_prim_call(weight, pre_spike, post_trace)[0] * wunit)
    weight = u.math.clip(weight, w_min, w_max)
    return weight


def _dense_on_pre_numba_kernel_generator(**kwargs):
    def kernel(weight, spike, trace, out_w):
        for i in range(spike.shape[0]):
            if spike[i]:
                out_w[i] += trace

    return numba_kernel(kernel, parallel=True, input_output_aliases={0: 0})


def _dense_on_pre_pallas_kernel_generator(weight_info, **kwargs):
    block_dim = generate_block_dim(weight_info.shape[1], 512)

    def kernel(weight_ref, spike_ref, trace_ref, out_w_ref):
        i_block = pl.program_id(0)
        i_post_start = i_block * block_dim
        mask = jax.numpy.arange(block_dim) + i_post_start < weight_info.shape[1]
        post_trace = pl.load(trace_ref, pl.dslice(i_post_start, block_dim), mask=mask)

        def loop_fn(i, _):
            @pl.when(spike_ref[i] if spike_ref.dtype == jnp.bool_ else spike_ref[i] != 0)
            def run():
                pl.store(
                    out_w_ref,
                    pl.dslice(i_post_start, block_dim),
                    pl.load(out_w_ref, (i, pl.dslice(i, i_post_start, block_dim)), mask=mask) + post_trace,
                    mask=mask,
                )

        jax.lax.fori_loop(0, spike_ref.shape[0], loop_fn, None)

    return pallas_kernel(kernel, input_output_aliases={0: 0}, tile=(pl.cdiv(weight_info.shape[1], block_dim),))


def _dense_on_pre_prim_call(weight, pre_spike, post_trace):
    assert weight.ndim == 2, f'dense_one_pre only support 2D weight. But got shape: {weight.shape}.'
    assert pre_spike.ndim == 1, f'pre_spike should be 1D, But got shape: {pre_spike.shape}.'
    assert post_trace.ndim == 1, f'post_trace should be 1D. But got shape: {post_trace.shape}.'
    assert weight.shape[0] == pre_spike.shape[0], (
        f'weight shape[0] ({weight.shape[0]}) should '
        f'match pre_spike shape[0] ({pre_spike.shape[0]}).'
    )
    assert weight.shape[1] == post_trace.shape[0], (
        f'weight shape[1] ({weight.shape[1]}) should '
        f'match post_trace shape[0] ({post_trace.shape[0]}).'
    )
    return _dense_on_pre_prim(
        weight, pre_spike, post_trace,
        outs=[jax.ShapeDtypeStruct(weight.shape, weight.dtype)],
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
    )


_dense_on_pre_prim = XLACustomKernel('dense_on_pre')
_dense_on_pre_prim.def_cpu_kernel(_dense_on_pre_numba_kernel_generator)
_dense_on_pre_prim.def_gpu_kernel(pallas=_dense_on_pre_pallas_kernel_generator)
_dense_on_pre_prim.def_tpu_kernel(_dense_on_pre_pallas_kernel_generator)


def dense_on_post(
    weight: Union[u.Quantity, jax.Array],
    pre_trace: Union[u.Quantity, jax.Array],
    post_spike: jax.Array,
    w_min: Optional[Union[u.Quantity, jax.Array]] = None,
    w_max: Optional[Union[u.Quantity, jax.Array]] = None,
):
    """Updates synaptic weights based on postsynaptic spike events and presynaptic traces.

    This function implements a plasticity rule where postsynaptic spikes trigger weight updates
    modulated by presynaptic trace values. The weight update is performed element-wise.

    Args:
        weight: Synaptic weight matrix of shape (n_pre, n_post).
        pre_trace: Presynaptic trace values, shape (n_pre,).
        post_spike: Binary/boolean array indicating postsynaptic spike events, shape (n_post,).
        w_min: Optional lower bound for weight clipping. Must have same units as weight.
        w_max: Optional upper bound for weight clipping. Must have same units as weight.

    Returns:
        Updated weight matrix with the same shape and units as the input weight.

    Note:
        The function handles unit conversion internally, ensuring that the pre_trace
        is converted to the same unit as the weight before computation.
    """
    weight, wunit = u.split_mantissa_unit(weight)
    pre_trace = u.Quantity(pre_trace).to(wunit).mantissa
    weight = u.maybe_decimal(_dense_one_post_prim_call(weight, pre_trace, post_spike)[0] * wunit)
    weight = u.math.clip(weight, w_min, w_max)
    return weight


def _dense_on_post_numba_kernel_generator(**kwargs):
    def kernel(weight, trace, spike, out_w):
        for i in range(spike.shape[0]):
            if spike[i]:
                out_w[:, i] += trace

    return numba_kernel(kernel, parallel=True, input_output_aliases={0: 0})


def _dense_on_post_pallas_kernel_generator(weight_info, **kwargs):
    block_dim = generate_block_dim(weight_info.shape[0], 512)

    def kernel(weight_ref, trace_ref, spike_ref, out_w_ref):
        i_block = pl.program_id(0)
        i_post_start = i_block * block_dim
        mask = jax.numpy.arange(block_dim) + i_post_start < weight_info.shape[1]
        post_trace = pl.load(trace_ref, pl.dslice(i_post_start, block_dim), mask=mask)

        def loop_fn(i, _):
            @pl.when(spike_ref[i] if spike_ref.dtype == jnp.bool_ else spike_ref[i] != 0)
            def run():
                pl.store(
                    out_w_ref,
                    (pl.dslice(i_post_start, block_dim), i),
                    pl.load(out_w_ref, (pl.dslice(i, i_post_start, block_dim), i), mask=mask) + post_trace,
                    mask=mask,
                )

        jax.lax.fori_loop(0, spike_ref.shape[0], loop_fn, None)

    return pallas_kernel(kernel, input_output_aliases={0: 0}, tile=(pl.cdiv(weight_info.shape[0], block_dim),))


def _dense_one_post_prim_call(weight, pre_trace, post_spike):
    assert weight.ndim == 2, f'dense_one_pre only support 2D weight. But got shape: {weight.shape}.'
    assert pre_trace.ndim == 1, f'pre_trace should be 1D. But got shape: {pre_trace.shape}.'
    assert post_spike.ndim == 1, f'post_spike should be 1D. But got shape: {post_spike.shape}.'
    assert weight.shape[0] == pre_trace.shape[0], (f'weight shape[0] ({weight.shape[0]}) should '
                                                   f'match pre_trace shape[0] ({pre_trace.shape[0]}).')
    assert weight.shape[1] == post_spike.shape[0], (f'weight shape[1] ({weight.shape[1]}) should '
                                                    f'match post_spike shape[0] ({post_spike.shape[0]}).')
    return _dense_on_post_prim(
        weight, pre_trace, post_spike,
        outs=[jax.ShapeDtypeStruct(weight.shape, weight.dtype)],
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
    )


_dense_on_post_prim = XLACustomKernel('dense_on_post')
_dense_on_post_prim.def_cpu_kernel(_dense_on_post_numba_kernel_generator)
_dense_on_post_prim.def_gpu_kernel(pallas=_dense_on_post_pallas_kernel_generator)
_dense_on_post_prim.def_tpu_kernel(_dense_on_post_pallas_kernel_generator)
