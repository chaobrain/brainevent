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

# -*- coding: utf-8 -*-

from typing import Callable

import brainunit as u
import jax
import jax.numpy as jnp
from jax.interpreters import ad

from ._xla_custom_op import XLACustomKernel
from ._xla_custom_op_numba import numba_environ, NumbaKernelGenerator
from ._xla_custom_op_warp import WarpKernelGenerator, dtype_to_warp_type

Kernel = Callable


def event_mv(
    weights,
    spikes,
    *,
    float_as_event: bool = True,
    transpose: bool = False,
):
    r"""
    Multiply event by matrix, $\mathrm{matrix} @ \mathrm{spikes}$.

    Args:
        weights: The matrix.
        spikes: The event.
        float_as_event: If True, treat the event as a float.
        transpose: If True, transpose the matrix.
    """
    weight_val, weight_unit = u.split_mantissa_unit(weights)
    spk_val, spk_unit = u.split_mantissa_unit(spikes)
    r = event_liner_p_call(weight_val, spk_val, float_as_event=float_as_event, transpose=transpose)
    return u.maybe_decimal(r[0] * weight_unit * spk_unit)


def mv_cpu_kernel_generator(
    float_as_event: bool,
    transpose: bool,
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
) -> Kernel:
    import numba  # pylint: disable=import-outside-toplevel

    if transpose:
        if spk_info.dtype == jnp.bool_:

            @numba.njit(**numba_environ.setting)
            def _kernel(weights, spikes, _, posts):
                for i in range(spikes.shape[0]):
                    if spikes[i]:
                        posts += weights[i]

        elif float_as_event:
            @numba.njit(**numba_environ.setting)
            def _kernel(weights, spikes, _, posts):
                for i in range(spikes.shape[0]):
                    if spikes[i] != 0.:
                        posts += weights[i]

        else:
            @numba.njit(**numba_environ.setting)
            def _kernel(weights, spikes, _, posts):
                for i in range(spikes.shape[0]):
                    sp = spikes[i]
                    if sp != 0.:
                        posts += weights[i] * sp

    else:
        if spk_info.dtype == jnp.bool_:

            @numba.njit(**numba_environ.setting)
            def _kernel(weights, spikes, _, posts):
                for i in range(spikes.shape[0]):
                    if spikes[i]:
                        posts += weights[:, i]

        elif float_as_event:
            @numba.njit(**numba_environ.setting)
            def _kernel(weights, spikes, _, posts):
                for i in range(spikes.shape[0]):
                    if spikes[i] != 0.:
                        posts += weights[:, i]

        else:
            @numba.njit(**numba_environ.setting)
            def _kernel(weights, spikes, _, posts):
                for i in range(spikes.shape[0]):
                    sp = spikes[i]
                    if sp != 0.:
                        posts += weights[i] * sp

    return _kernel


def mv_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
) -> Kernel:
    import warp  # pylint: disable=import-outside-toplevel

    spike_dtype = dtype_to_warp_type(spk_info.dtype)
    weight_dtype = dtype_to_warp_type(weight_info.dtype)

    if transpose:
        if spk_info.dtype == jnp.bool_:
            @warp.kernel
            def kernel(
                weights: warp.array1d(dtype=weight_dtype),
                spikes: warp.array1d(dtype=spike_dtype),
                _: warp.array1d(dtype=weight_dtype),
                out: warp.array1d(dtype=weight_dtype),
            ):
                i = warp.tid()
                if spikes[i]:
                    out[i] += weights[i]

    return kernel()


def jvp_weights(w_dot, weights, spikes, *, float_as_event, transpose, **kwargs):
    return event_liner_p_call(
        w_dot,
        spikes,
        transpose=transpose,
        float_as_event=float_as_event,
    )


def jvp_spikes(spk_dot, weights, spikes, *, transpose, **kwargs):
    if transpose:
        return [spk_dot @ weights]
    else:
        return [weights @ spk_dot]


def transpose_rule(
    ct,
    weights,
    spikes,
    *,
    float_as_event,
    transpose,
    **kwargs
):
    if ad.is_undefined_primal(spikes):
        if transpose:
            ct_events = jnp.matmul(weights, ct[0])
        else:
            ct_events = jnp.matmul(ct[0], weights)
        return weights, (ad.Zero(spikes) if type(ct[0]) is ad.Zero else ct_events)

    else:
        def map_fn(sp):
            if spikes.dtype == jnp.bool_:
                d_gmax = jnp.where(sp, ct[0], jnp.zeros_like(ct[0]))
            else:
                if float_as_event:
                    d_gmax = jnp.where(sp == 0., jnp.zeros_like(ct[0]), ct[0])
                else:
                    # d_gmax = jnp.where(sp == 0., jnp.zeros_like(ct[0]), ct[0] * sp)
                    d_gmax = jax.lax.cond(sp == 0., lambda: jnp.zeros_like(ct[0]), lambda: ct[0] * sp)
            return d_gmax

        if transpose:
            # ct_weights = jax.vmap(map_fn)(spikes)
            ct_weights = jnp.outer(spikes, ct[0])
        else:
            ct_weights = jnp.outer(ct[0], spikes)

        return (ad.Zero(weights) if type(ct[0]) is ad.Zero else ct_weights), spikes


def event_liner_p_call(
    weights,
    spikes,
    *,
    transpose: bool,
    float_as_event: bool,
):
    if transpose:
        out = jax.ShapeDtypeStruct([weights.shape[1]], weights.dtype)
    else:
        out = jax.ShapeDtypeStruct([weights.shape[0]], weights.dtype)

    return event_linear_p(
        weights,
        spikes,
        jnp.zeros(out.shape, out.dtype),
        outs=[out],
        float_as_event=float_as_event,
        transpose=transpose,
        spk_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
    )


event_linear_p = XLACustomKernel(
    'event_linear',
    cpu_kernel=NumbaKernelGenerator(mv_cpu_kernel_generator, input_output_aliases={2: 0}),
    gpu_kernel=WarpKernelGenerator(
        mv_gpu_kernel_generator,
        dim=lambda transpose, weight_info, **kwargs: weight_info.shape[1] if transpose else weight_info.shape[0],
        input_output_aliases={2: 0}
    ),
)
event_linear_p.defjvp(jvp_weights, jvp_spikes)
event_linear_p.def_transpose_rule(transpose_rule)
