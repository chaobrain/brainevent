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

import brainunit as u
import jax
import jax.numpy as jnp
from jax.interpreters import ad

from ._event_matrix_impl import matrix_event_mm, event_matrix_mm
from ._misc import cdiv
from ._typing import Kernel
from ._xla_custom_op import XLACustomKernel
from ._xla_custom_op_numba import numba_environ, NumbaKernelGenerator
from ._xla_custom_op_util import general_batching_rule
from ._xla_custom_op_warp import WarpKernelGenerator, dtype_to_warp_type

TILE_THREAD = 256

__all__ = [
    'matrix_event_mv',
    'event_matrix_mv',
]


def matrix_event_mv(
    weights,
    spikes,
    *,
    float_as_event: bool = True,
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
    r = matrix_event_mv_p_call(
        weight_val,
        spk_val,
        float_as_event=float_as_event,
    )
    return u.maybe_decimal(r[0] * wunit * spkunit)


def _matrix_event_mv_cpu_kernel_generator(
    float_as_event: bool,
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
) -> Kernel:
    import numba  # pylint: disable=import-outside-toplevel

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


def _matrix_event_mv_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    float_as_event: bool,
    TILE_SIZE: int,
    block_dim: int,
    **kwargs
) -> Kernel:
    import warp  # pylint: disable=import-outside-toplevel

    spike_dtype = dtype_to_warp_type(spk_info.dtype)
    weight_dtype = dtype_to_warp_type(weight_info.dtype)

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


def _matrix_event_mv_jvp_weights(w_dot, weights, spikes, *, float_as_event, **kwargs):
    return matrix_event_mv_p_call(w_dot, spikes, float_as_event=float_as_event)


def _matrix_event_mv_jvp_spikes(spk_dot, weights, spikes, **kwargs):
    return [weights @ spk_dot]


def _matrix_event_mv_transpose_rule(ct, weights, spikes, **kwargs):
    if ad.is_undefined_primal(spikes):
        ct_events = jnp.matmul(ct[0], weights)
        return weights, (ad.Zero(spikes) if type(ct[0]) is ad.Zero else ct_events)
    else:
        ct_weights = jnp.outer(ct[0], spikes)
        return (ad.Zero(weights) if type(ct[0]) is ad.Zero else ct_weights), spikes


def _matrix_event_batching(args, axes, **kwargs):
    if axes == (None, 0):
        r = matrix_event_mm(args[0], args[1].T, float_as_event=kwargs['float_as_event'])
        return [r], [1]
    if axes == (None, 1):
        r = matrix_event_mm(args[0], args[1], float_as_event=kwargs['float_as_event'])
        return [r], [1]
    else:
        return general_batching_rule(matrix_event_mv_p, args, axes, **kwargs)


def matrix_event_mv_p_call(weights, spikes, *, float_as_event: bool):
    out = jax.ShapeDtypeStruct([weights.shape[0]], weights.dtype)
    return matrix_event_mv_p(
        weights,
        spikes,
        outs=[out],
        float_as_event=float_as_event,
        spk_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        TILE_SIZE=spikes.shape[0],
    )


matrix_event_mv_p = XLACustomKernel(
    'matrix_event_mv_op',
    cpu_kernel=NumbaKernelGenerator(_matrix_event_mv_cpu_kernel_generator),
    gpu_kernel=WarpKernelGenerator(
        _matrix_event_mv_gpu_kernel_generator,
        tile=lambda weight_info, **kwargs: cdiv(weight_info.shape[0], TILE_THREAD),
        block_dim=TILE_THREAD,
    ),
)
matrix_event_mv_p.defjvp(_matrix_event_mv_jvp_weights, _matrix_event_mv_jvp_spikes)
matrix_event_mv_p.def_transpose_rule(_matrix_event_mv_transpose_rule)
matrix_event_mv_p.def_batching_rule(_matrix_event_batching)


def event_matrix_mv(
    spikes,
    weights,
    *,
    float_as_event: bool = True,
):
    with jax.ensure_compile_time_eval():
        weights = u.math.asarray(weights)
        spikes = u.math.asarray(spikes)
    weight_val, wunit = u.split_mantissa_unit(weights)
    spk_val, spkunit = u.split_mantissa_unit(spikes)
    r = event_matrix_mv_p_call(
        spk_val,
        weight_val,
        float_as_event=float_as_event,
    )
    return u.maybe_decimal(r[0] * wunit * spkunit)


def _event_matrix_mv_cpu_kernel_generator(
    float_as_event: bool,
    transpose: bool,
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
) -> Kernel:
    import numba  # pylint: disable=import-outside-toplevel

    if spk_info.dtype == jnp.bool_:
        def _kernel(spikes, weights, posts):
            posts[:] = 0.
            for i in range(spikes.shape[0]):
                if spikes[i]:
                    posts += weights[i]

    elif float_as_event:
        def _kernel(spikes, weights, posts):
            posts[:] = 0.
            for i in range(spikes.shape[0]):
                if spikes[i] != 0.:
                    posts += weights[i]

    else:
        def _kernel(spikes, weights, posts):
            posts[:] = 0.
            for i in range(spikes.shape[0]):
                sp = spikes[i]
                if sp != 0.:
                    posts += weights[i] * sp

    return numba.njit(**numba_environ.setting)(_kernel)


def _event_matrix_mv_gpu_kernel_generator(
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

    if spk_info.dtype == jnp.bool_:
        def kernel(
            spike_ref: warp.array1d(dtype=spike_dtype),
            weight_ref: warp.array2d(dtype=weight_dtype),
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
            spike_ref: warp.array1d(dtype=spike_dtype),
            weight_ref: warp.array2d(dtype=weight_dtype),
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
            spike_ref: warp.array1d(dtype=spike_dtype),
            weight_ref: warp.array2d(dtype=weight_dtype),
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

    return warp.kernel(kernel)


def _event_matrix_mv_jvp_weights(w_dot, spikes, weights, *, float_as_event, **kwargs):
    return event_matrix_mv_p_call(spikes, w_dot, float_as_event=float_as_event)


def _event_matrix_mv_jvp_spikes(spk_dot, spikes, weights, **kwargs):
    return [spk_dot @ weights]


def _event_matrix_mv_transpose_rule(ct, weights, spikes, **kwargs):
    if ad.is_undefined_primal(spikes):
        ct_events = jnp.matmul(weights, ct[0])
        return weights, (ad.Zero(spikes) if type(ct[0]) is ad.Zero else ct_events)

    else:
        ct_weights = jnp.outer(spikes, ct[0])
        return (ad.Zero(weights) if type(ct[0]) is ad.Zero else ct_weights), spikes


def _event_matrix_batching(args, axes, **kwargs):
    if axes == (0, None):
        r = event_matrix_mm(args[0], args[1], float_as_event=kwargs['float_as_event'])
        return [r], [0]
    if axes == (1, None):
        r = event_matrix_mm(args[0].T, args[1], float_as_event=kwargs['float_as_event'])
        return [r], [0]
    else:
        return general_batching_rule(event_matrix_mv_p, args, axes, **kwargs)


def event_matrix_mv_p_call(spikes, weights, *, float_as_event: bool):
    out = jax.ShapeDtypeStruct([weights.shape[1]], weights.dtype)
    return event_matrix_mv_p(
        spikes,
        weights,
        outs=[out],
        float_as_event=float_as_event,
        spk_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        TILE_SIZE=spikes.shape[0],
    )


event_matrix_mv_p = XLACustomKernel(
    'event_matrix_mv_op',
    cpu_kernel=NumbaKernelGenerator(_event_matrix_mv_cpu_kernel_generator),
    gpu_kernel=WarpKernelGenerator(
        _event_matrix_mv_gpu_kernel_generator,
        tile=lambda weight_info, **kwargs: cdiv(weight_info.shape[1], TILE_THREAD),
        block_dim=TILE_THREAD,
    ),
)
event_matrix_mv_p.defjvp(_event_matrix_mv_jvp_spikes, _event_matrix_mv_jvp_weights, )
event_matrix_mv_p.def_transpose_rule(_event_matrix_mv_transpose_rule)
event_matrix_mv_p.def_batching_rule(_event_matrix_batching)
