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

import brainunit as u
import jax
import jax.numpy as jnp
from jax.interpreters import ad

from ._compatible_import import pallas as pl
from ._misc import cdiv
from ._typing import Kernel
from ._xla_custom_op import XLACustomKernel
from ._xla_custom_op_numba import numba_environ, NumbaKernelGenerator
from ._xla_custom_op_warp import WarpKernelGenerator, dtype_to_warp_type


def event_dense_mv(
    weights,
    spikes,
    *,
    float_as_event: bool = True,
    transpose: bool = False,
):
    r"""
    Multiply event vector by matrix, computing either weights @ spikes or weights.T @ spikes.

    This function is optimized for sparse event vectors, where most elements are zero or false.
    The implementation uses custom kernels for efficient computation on CPU/GPU hardware.

    For matrix-vector multiplication: y = W @ x (or y = W.T @ x when transpose=True)

    Mathematical formulations:
    - Standard mode (transpose=False): output[i] = sum_j(weights[i,j] * spikes[j])
    - Transpose mode (transpose=True): output[j] = sum_i(weights[i,j] * spikes[i])

    The function handles three types of event vectors:
    1. Boolean events (spikes is a boolean array): Only indices where spikes[i]=True contribute
    2. Float events treated as events (float_as_event=True): Only non-zero values contribute
    3. Float events treated as values (float_as_event=False): Non-zero values are scaled by their value

    Args:
        weights: A matrix of shape (M, N) representing synaptic weights or connection strengths
        spikes: An event vector of shape (N,) for standard mode or (M,) for transpose mode
        float_as_event: If True, treat non-zero float values as binary events.
                        If False, use the actual float values for scaling. Default: True.
        transpose: If True, compute weights.T @ spikes instead of weights @ spikes. Default: False.

    Returns:
        An array representing the result of the matrix-vector product, with shape (M,) for
        standard mode or (N,) for transpose mode. Preserves units from the input arrays.

    Note:
        The function leverages custom XLA kernels for efficient sparse computation and
        handles unit management through the brainunit library.
    """
    with jax.ensure_compile_time_eval():
        weights = u.math.asarray(weights)
        spikes = u.math.asarray(spikes)
    weight_val, wunit = u.split_mantissa_unit(weights)
    spk_val, spkunit = u.split_mantissa_unit(spikes)
    r = event_dense_mv_p_call(
        weight_val,
        spk_val,
        float_as_event=float_as_event,
        transpose=transpose
    )
    return u.maybe_decimal(r[0] * wunit * spkunit)


def _mv_cpu_kernel_generator(
    float_as_event: bool,
    transpose: bool,
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
) -> Kernel:
    import numba  # pylint: disable=import-outside-toplevel

    if transpose:
        if spk_info.dtype == jnp.bool_:
            def _kernel(weights, spikes, posts):
                posts[:] = 0.
                for i in range(spikes.shape[0]):
                    if spikes[i]:
                        posts += weights[i]

        elif float_as_event:
            def _kernel(weights, spikes, posts):
                posts[:] = 0.
                for i in range(spikes.shape[0]):
                    if spikes[i] != 0.:
                        posts += weights[i]

        else:
            def _kernel(weights, spikes, posts):
                posts[:] = 0.
                for i in range(spikes.shape[0]):
                    sp = spikes[i]
                    if sp != 0.:
                        posts += weights[i] * sp

    else:
        if spk_info.dtype == jnp.bool_:
            def _kernel(weights, spikes, posts):
                posts[:] = 0.
                for i in range(spikes.shape[0]):
                    if spikes[i]:
                        posts += weights[:, i]

        elif float_as_event:
            def _kernel(weights, spikes, posts):
                posts[:] = 0.
                for i in range(spikes.shape[0]):
                    if spikes[i] != 0.:
                        posts += weights[:, i]

        else:
            def _kernel(weights, spikes, posts):
                posts[:] = 0.
                for i in range(spikes.shape[0]):
                    sp = spikes[i]
                    if sp != 0.:
                        posts += weights[:, i] * sp

    return numba.njit(**numba_environ.setting)(_kernel)


def _mv_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    transpose: bool,
    float_as_event: bool,
    TILE_SIZE: int,
    block_dim: int,
    **kwargs
) -> Kernel:
    import warp  # pylint: disable=import-outside-toplevel

    spike_dtype = dtype_to_warp_type(spk_info.dtype)
    weight_dtype = dtype_to_warp_type(weight_info.dtype)

    if transpose:
        if spk_info.dtype == jnp.bool_:
            def kernel(
                weight_ref: warp.array2d(dtype=weight_dtype),
                spike_ref: warp.array1d(dtype=spike_dtype),
                out_ref: warp.array1d(dtype=weight_dtype),
            ):
                i_col = warp.tid()
                spikes = warp.tile_load(spike_ref, shape=(TILE_SIZE,))
                temp = warp.tile_zeros(shape=(block_dim,), dtype=weight_dtype)
                for j in range(TILE_SIZE):
                    if spikes[j]:
                        temp += warp.tile_load(weight_ref[j], shape=(block_dim,), offset=(i_col * block_dim,))
                warp.tile_store(out_ref, temp, offset=(i_col * block_dim,))

        elif float_as_event:
            def kernel(
                weight_ref: warp.array2d(dtype=weight_dtype),
                spike_ref: warp.array1d(dtype=spike_dtype),
                out_ref: warp.array1d(dtype=weight_dtype),
            ):
                i_col = warp.tid()
                spikes = warp.tile_load(spike_ref, shape=(TILE_SIZE,))
                temp = warp.tile_zeros(shape=(block_dim,), dtype=weight_dtype)
                for j in range(TILE_SIZE):
                    if spikes[j] != 0.:
                        temp += warp.tile_load(weight_ref[j], shape=(block_dim,), offset=(i_col * block_dim,))
                warp.tile_store(out_ref, temp, offset=(i_col * block_dim,))

        else:
            def kernel(
                weight_ref: warp.array2d(dtype=weight_dtype),
                spike_ref: warp.array1d(dtype=spike_dtype),
                out_ref: warp.array1d(dtype=weight_dtype),
            ):
                i_col = warp.tid()
                spikes = warp.tile_load(spike_ref, shape=(TILE_SIZE,))
                temp = warp.tile_zeros(shape=(block_dim,), dtype=weight_dtype)
                for j in range(TILE_SIZE):
                    s = spikes[j]
                    if s != 0.:
                        temp += warp.tile_load(weight_ref[j], shape=(block_dim,), offset=(i_col * block_dim,)) * s
                warp.tile_store(out_ref, temp, offset=(i_col * block_dim,))

    else:
        if spk_info.dtype == jnp.bool_:
            def kernel(
                weight_ref: warp.array2d(dtype=weight_dtype),
                spike_ref: warp.array1d(dtype=spike_dtype),
                out_ref: warp.array1d(dtype=weight_dtype),
            ):
                i_col = warp.tid()
                spikes = warp.tile_load(spike_ref, shape=(TILE_SIZE,))
                temp = warp.tile_zeros(shape=(block_dim,), dtype=weight_dtype)
                for j in range(TILE_SIZE):
                    if spikes[j]:
                        data = warp.tile_load(weight_ref, shape=(block_dim, 1), offset=(i_col * block_dim, j))
                        temp += data[:, 0]  # TODO
                warp.tile_store(out_ref, temp, offset=(i_col * block_dim,))

        elif float_as_event:
            def kernel(
                weight_ref: warp.array2d(dtype=weight_dtype),
                spike_ref: warp.array1d(dtype=spike_dtype),
                out_ref: warp.array1d(dtype=weight_dtype),
            ):
                i_col = warp.tid()
                spikes = warp.tile_load(spike_ref, shape=(TILE_SIZE,))
                temp = warp.tile_zeros(shape=(block_dim,), dtype=weight_dtype)
                for j in range(TILE_SIZE):
                    if spikes[j] != 0.:
                        data = warp.tile_load(weight_ref, shape=(block_dim, 1), offset=(i_col * block_dim, j))
                        temp += data[:, 0]  # TODO
                warp.tile_store(out_ref, temp, offset=(i_col * block_dim,))

        else:
            def kernel(
                weight_ref: warp.array2d(dtype=weight_dtype),
                spike_ref: warp.array1d(dtype=spike_dtype),
                out_ref: warp.array1d(dtype=weight_dtype),
            ):
                i_col = warp.tid()
                spikes = warp.tile_load(spike_ref, shape=(TILE_SIZE,))
                temp = warp.tile_zeros(shape=(block_dim,), dtype=weight_dtype)
                for j in range(TILE_SIZE):
                    s = spikes[j]
                    if s != 0.:
                        data = warp.tile_load(weight_ref, shape=(block_dim, 1), offset=(i_col * block_dim, j))
                        temp += data[:, 0] * s
                warp.tile_store(out_ref, temp, offset=(i_col * block_dim,))

    return warp.kernel(kernel)


def _mv_tpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    transpose: bool,
    float_as_event: bool,
    block_dim: int,
    **kwargs
) -> Kernel:
    def _add_result_from_temp(cond, temp, res, reset=True):
        if reset:
            return jax.lax.cond(
                cond,
                lambda temp_, res_: (temp_.at[:].set(0.), res_.at[:].add(temp_.sum(axis=0))),
                lambda temp_, res_: (temp_, res_),
                temp,
                res
            )
        else:
            return jax.lax.cond(
                cond,
                lambda temp_, res_: res_.at[:].add(temp_.sum(axis=0)),
                lambda temp_, res_: res_,
            )

    if transpose:
        if spk_info.dtype == jnp.bool_:
            def kernel(weights_ref, spikes_ref, out_ref):
                i_pid = pl.program_id(0)
                i_load = 0
                res = jnp.zeros(block_dim, dtype=weight_info.dtype)
                temp = jnp.zeros((block_dim, block_dim), dtype=weight_info.dtype)
                mask = jnp.arange(block_dim) + i_pid * block_dim < weights_ref.shape[1]

                def _f_true(res_, temp_, i_load_, i_spk):
                    temp_[i_load_].set(
                        pl.load(weights_ref, (i_spk, pl.dslice(i_pid * block_dim, block_dim)), mask=mask))
                    i_load_ += 1
                    res_, temp_ = _add_result_from_temp(i_load == block_dim, temp_, res_, reset=True)
                    return res_, temp_, i_load_

                def _f_false(res_, temp_, i_load_):
                    return res_, temp_, i_load_

                def body_fn(i_spk, _):
                    return jax.lax.cond(spikes_ref[i_spk], _f_true, _f_false, res, temp, i_load)

                res, temp, i_load = jax.lax.fori_loop(0, spk_info.shape[0], body_fn, None)
                res = _add_result_from_temp(i_load > 0, temp, res, reset=False)
                pl.store(out_ref, (pl.dslice(i_pid * block_dim, block_dim),), res, mask=mask)

    return kernel


def _mv_jvp_weights(w_dot, weights, spikes, *, float_as_event, transpose, **kwargs):
    return event_dense_mv_p_call(
        w_dot,
        spikes,
        transpose=transpose,
        float_as_event=float_as_event,
    )


def _mv_jvp_spikes(spk_dot, weights, spikes, *, transpose, **kwargs):
    if transpose:
        return [spk_dot @ weights]
    else:
        return [weights @ spk_dot]


def _mv_transpose_rule(
    ct,
    weights,
    spikes,
    *,
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
        if transpose:
            ct_weights = jnp.outer(spikes, ct[0])
        else:
            ct_weights = jnp.outer(ct[0], spikes)

        return (ad.Zero(weights) if type(ct[0]) is ad.Zero else ct_weights), spikes


def event_dense_mv_p_call(
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

    return event_dense_mv_p(
        weights,
        spikes,
        outs=[out],
        float_as_event=float_as_event,
        transpose=transpose,
        spk_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        TILE_SIZE=spikes.shape[0],
    )



N_THREAD = 256


event_dense_mv_p = XLACustomKernel(
    'event_dense_mv',
    cpu_kernel=NumbaKernelGenerator(_mv_cpu_kernel_generator),
    gpu_kernel=WarpKernelGenerator(
        _mv_gpu_kernel_generator,
        tile=lambda weight_info, transpose, **kwargs: (
            cdiv(weight_info.shape[1], N_THREAD)
            if transpose else
            cdiv(weight_info.shape[0], N_THREAD)
        ),
        block_dim=N_THREAD,
    ),
)
event_dense_mv_p.defjvp(_mv_jvp_weights, _mv_jvp_spikes)
event_dense_mv_p.def_transpose_rule(_mv_transpose_rule)
