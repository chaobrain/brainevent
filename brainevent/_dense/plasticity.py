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

from typing import Optional, Union

import brainunit as u
import jax
import jax.numpy as jnp
from jax.interpreters import ad

from brainevent._misc import generate_block_dim
from brainevent._op import XLACustomKernel, numba_kernel, general_batching_rule

__all__ = [
    'plast_dense_on_binary_pre',
    'plast_dense_on_binary_pre_p',
    'plast_dense_on_binary_post',
    'plast_dense_on_binary_post_p'
]


def plast_dense_on_binary_pre(
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


def _dense_on_pre_numba_kernel(spike_info: jax.ShapeDtypeStruct, **kwargs):
    import numba

    if spike_info.dtype == jnp.bool_:
        @numba.njit(parallel=True, fastmath=True, nogil=True, cache=True)
        def kernel(weight, spike, trace, out_w):
            for i in numba.prange(spike.shape[0]):
                if spike[i]:
                    out_w[i] += trace

    else:
        @numba.njit(parallel=True, fastmath=True, nogil=True, cache=True)
        def kernel(weight, spike, trace, out_w):
            for i in numba.prange(spike.shape[0]):
                if spike[i] != 0.:
                    out_w[i] += trace

    def run(weight, spike, trace):
        return numba_kernel(kernel, outs=kwargs['outs'], input_output_aliases={0: 0})(weight, spike, trace)

    return run


def _dense_on_pre_pallas_kernel(weight_info, spike_info: jax.ShapeDtypeStruct, **kwargs):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    block_dim = generate_block_dim(weight_info.shape[1], 512)

    def kernel(weight_ref, spike_ref, trace_ref, out_w_ref):
        i_block = pl.program_id(0)
        col_start = i_block * block_dim
        cols = col_start + jnp.arange(block_dim)
        mask = cols < weight_info.shape[1]
        safe_cols = jnp.where(mask, cols, 0)
        trace_block = plt.load(trace_ref[safe_cols])
        trace_block = jnp.where(mask, trace_block, 0.0)

        def loop_fn(i, _):
            spike = spike_ref[i]

            @pl.when(spike if spike_info.dtype == jnp.bool_ else spike != 0.)
            def run():
                row_ref = out_w_ref[i, safe_cols]
                row_val = plt.load(row_ref)
                plt.store(row_ref, row_val + trace_block, mask=mask)

        jax.lax.fori_loop(0, spike_ref.shape[0], loop_fn, None)

    def run(weight, spike, trace):
        fn = pl.pallas_call(
            kernel,
            grid=(pl.cdiv(weight_info.shape[1], block_dim),),
            input_output_aliases={0: 0},
            out_shape=kwargs['outs'],
        )
        return fn(weight, spike, trace)

    return run


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
    return plast_dense_on_binary_pre_p(
        weight, pre_spike, post_trace,
        outs=[jax.ShapeDtypeStruct(weight.shape, weight.dtype)],
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        spike_info=jax.ShapeDtypeStruct(pre_spike.shape, pre_spike.dtype),
        trace_info=jax.ShapeDtypeStruct(post_trace.shape, post_trace.dtype),
    )


def _dense_on_pre_jvp_weight(w_dot, weight, pre_spike, post_trace, **kwargs):
    return [w_dot]


def _dense_on_pre_transpose_rule(ct, weight, pre_spike, post_trace, **kwargs):
    if ad.is_undefined_primal(pre_spike) or ad.is_undefined_primal(post_trace):
        raise ValueError("Cannot transpose with respect to pre_spike or post_trace.")
    ct = ct[0]
    if ad.is_undefined_primal(weight):
        return (ad.Zero(weight) if type(ct) is ad.Zero else ct), pre_spike, post_trace
    return weight, pre_spike, post_trace


def _dense_on_pre_batching(args, axes, **kwargs):
    if axes == (0, None, None):
        weight, pre_spike, post_trace = args
        if pre_spike.dtype == jnp.bool_:
            mask = pre_spike.astype(weight.dtype)
        else:
            mask = (pre_spike != 0.).astype(weight.dtype)
        update = mask[:, None] * post_trace[None, :]
        return [weight + update[None, :, :]], [0]
    return general_batching_rule(plast_dense_on_binary_pre_p, args, axes, **kwargs)


plast_dense_on_binary_pre_p = XLACustomKernel('dense_on_pre')
plast_dense_on_binary_pre_p.def_numba_kernel(_dense_on_pre_numba_kernel)
plast_dense_on_binary_pre_p.def_pallas_kernel('gpu', _dense_on_pre_pallas_kernel)
plast_dense_on_binary_pre_p.def_pallas_kernel('tpu', _dense_on_pre_pallas_kernel)
plast_dense_on_binary_pre_p.def_jvp_rule2(_dense_on_pre_jvp_weight, None, None)
plast_dense_on_binary_pre_p.def_transpose_rule(_dense_on_pre_transpose_rule)
plast_dense_on_binary_pre_p.def_batching_rule(_dense_on_pre_batching)
plast_dense_on_binary_pre_p.def_call(_dense_on_pre_prim_call)


def plast_dense_on_binary_post(
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


def _dense_on_post_numba_kernel(spike_info: jax.ShapeDtypeStruct, **kwargs):
    import numba

    if spike_info.dtype == jnp.bool_:
        @numba.njit(parallel=True, fastmath=True, nogil=True, cache=True)
        def kernel(weight, trace, spike, out_w):
            for i in numba.prange(spike.shape[0]):
                if spike[i]:
                    out_w[:, i] += trace

    else:
        @numba.njit(parallel=True, fastmath=True, nogil=True, cache=True)
        def kernel(weight, trace, spike, out_w):
            for i in numba.prange(spike.shape[0]):
                if spike[i] != 0.:
                    out_w[:, i] += trace

    def run(weight, trace, spike):
        return numba_kernel(kernel, outs=kwargs['outs'], input_output_aliases={0: 0})(weight, trace, spike)

    return run


def _dense_on_post_pallas_kernel(weight_info, spike_info: jax.ShapeDtypeStruct, **kwargs):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    block_dim = generate_block_dim(weight_info.shape[0], 512)

    def kernel(weight_ref, trace_ref, spike_ref, out_w_ref):
        i_block = pl.program_id(0)
        row_start = i_block * block_dim
        rows = row_start + jnp.arange(block_dim)
        mask = rows < weight_info.shape[0]
        safe_rows = jnp.where(mask, rows, 0)
        trace_block = plt.load(trace_ref[safe_rows])
        trace_block = jnp.where(mask, trace_block, 0.0)

        def loop_fn(i, _):
            spike = spike_ref[i]

            @pl.when(spike if spike_info.dtype == jnp.bool_ else spike != 0.)
            def run():
                col_ref = out_w_ref[safe_rows, i]
                col_val = plt.load(col_ref)
                plt.store(col_ref, col_val + trace_block, mask=mask)

        jax.lax.fori_loop(0, spike_ref.shape[0], loop_fn, None)

    def run(weight, trace, spike):
        fn = pl.pallas_call(
            kernel,
            grid=(pl.cdiv(weight_info.shape[0], block_dim),),
            input_output_aliases={0: 0},
            out_shape=kwargs['outs'],
        )
        return fn(weight, trace, spike)

    return run


def _dense_one_post_prim_call(weight, pre_trace, post_spike):
    assert weight.ndim == 2, f'dense_one_pre only support 2D weight. But got shape: {weight.shape}.'
    assert pre_trace.ndim == 1, f'pre_trace should be 1D. But got shape: {pre_trace.shape}.'
    assert post_spike.ndim == 1, f'post_spike should be 1D. But got shape: {post_spike.shape}.'
    assert weight.shape[0] == pre_trace.shape[0], (f'weight shape[0] ({weight.shape[0]}) should '
                                                   f'match pre_trace shape[0] ({pre_trace.shape[0]}).')
    assert weight.shape[1] == post_spike.shape[0], (f'weight shape[1] ({weight.shape[1]}) should '
                                                    f'match post_spike shape[0] ({post_spike.shape[0]}).')
    return plast_dense_on_binary_post_p(
        weight, pre_trace, post_spike,
        outs=[jax.ShapeDtypeStruct(weight.shape, weight.dtype)],
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        spike_info=jax.ShapeDtypeStruct(post_spike.shape, post_spike.dtype),
        trace_info=jax.ShapeDtypeStruct(pre_trace.shape, pre_trace.dtype),
    )


def _dense_on_post_jvp_weight(w_dot, weight, pre_trace, post_spike, **kwargs):
    return [w_dot]


def _dense_on_post_transpose_rule(ct, weight, pre_trace, post_spike, **kwargs):
    if ad.is_undefined_primal(pre_trace) or ad.is_undefined_primal(post_spike):
        raise ValueError("Cannot transpose with respect to pre_trace or post_spike.")
    ct = ct[0]
    if ad.is_undefined_primal(weight):
        return (ad.Zero(weight) if type(ct) is ad.Zero else ct), pre_trace, post_spike
    return weight, pre_trace, post_spike


def _dense_on_post_batching(args, axes, **kwargs):
    if axes == (0, None, None):
        weight, pre_trace, post_spike = args
        if post_spike.dtype == jnp.bool_:
            mask = post_spike.astype(weight.dtype)
        else:
            mask = (post_spike != 0.).astype(weight.dtype)
        update = pre_trace[:, None] * mask[None, :]
        return [weight + update[None, :, :]], [0]
    return general_batching_rule(plast_dense_on_binary_post_p, args, axes, **kwargs)


plast_dense_on_binary_post_p = XLACustomKernel('dense_on_post')
plast_dense_on_binary_post_p.def_numba_kernel(_dense_on_post_numba_kernel)
plast_dense_on_binary_post_p.def_pallas_kernel('gpu', _dense_on_post_pallas_kernel)
plast_dense_on_binary_post_p.def_pallas_kernel('tpu', _dense_on_post_pallas_kernel)
plast_dense_on_binary_post_p.def_jvp_rule2(_dense_on_post_jvp_weight, None, None)
plast_dense_on_binary_post_p.def_transpose_rule(_dense_on_post_transpose_rule)
plast_dense_on_binary_post_p.def_batching_rule(_dense_on_post_batching)
plast_dense_on_binary_post_p.def_call(_dense_one_post_prim_call)
