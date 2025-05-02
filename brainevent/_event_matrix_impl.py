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
import numpy as np
from jax.interpreters import ad

from ._config import numba_environ, config
from ._misc import cdiv
from ._typing import Kernel
from ._xla_custom_op import XLACustomKernel, KernelChoice
from ._xla_custom_op_numba import NumbaKernelGenerator
from ._xla_custom_op_util import general_batching_rule
from ._xla_custom_op_warp import WarpKernelGenerator, dtype_to_warp_type

TILE_THREAD = 256


def matrix_event_mm(
    weights,
    spikes,
    *,
    float_as_event: bool = True,
):
    """Performs event-driven matrix multiplication: `weights @ spikes`.

    This function computes the matrix product of a weight matrix and a spike
    matrix, where the spike matrix often represents events (e.g., neural spikes).
    It handles potential units associated with the input arrays using the
    `brainunit` library. The computation is dispatched to specialized
    CPU/GPU kernels via `matrix_event_mm_p_call`.

    The exact multiplication behavior depends on the data type of `spikes` and
    the `float_as_event` flag.

    Parameters
    ----------
    weights : array_like
        The weight matrix, typically with shape (M, K). Can be a `brainunit`
        quantity.
    spikes : array_like
        The spike matrix, typically with shape (K, N). Can be boolean or float.
        If boolean, True indicates an event. If float, non-zero values
        indicate an event, and the value itself might be used depending on
        `float_as_event`. Can be a `brainunit` quantity.
    float_as_event : bool, optional
        Controls how float `spikes` are interpreted.
        - If True (default): Non-zero spike values indicate an event, but the
          weight is added without scaling by the spike value (effectively
          treating float spikes like boolean events).
        - If False: Non-zero spike values indicate an event, and the
          corresponding weight is scaled by the spike value before accumulation.

    Returns
    -------
    array_like
        The result of the event-driven matrix multiplication, with shape (M, N).
        If inputs had units, the output will also have appropriate units
        (product of weights unit and spikes unit).

    Notes
    -----
    The core computation performed is equivalent to:

    `output[m, n] = sum_{k} weights[m, k] * f(spikes[k, n])`

    where the function `f(s)` is defined as:
    - If `spikes` is boolean: `f(s) = 1` if `s` is True, `0` otherwise.
    - If `spikes` is float and `float_as_event` is True: `f(s) = 1` if `s != 0`, `0` otherwise.
    - If `spikes` is float and `float_as_event` is False: `f(s) = s` if `s != 0`, `0` otherwise.

    The function ensures inputs are JAX arrays and handles unit consistency
    using `brainunit`. The actual computation is delegated to a JAX primitive
    `matrix_event_mm_p` for potential hardware acceleration.
    """
    with jax.ensure_compile_time_eval():
        weights = u.math.asarray(weights)
        spikes = u.math.asarray(spikes)
    weight_val, wunit = u.split_mantissa_unit(weights)
    spk_val, spkunit = u.split_mantissa_unit(spikes)
    # Call the underlying primitive with unitless values
    r = matrix_event_mm_p_call(
        weight_val,
        spk_val,
        float_as_event=float_as_event,
    )
    # Re-attach units to the result
    return u.maybe_decimal(r[0] * wunit * spkunit)


def _matrix_event_mm_cpu_kernel_generator(
    float_as_event: bool,
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
) -> Kernel:
    # weights: [m, k]
    # spikes: [k, n]

    # if spk_info.dtype == jnp.bool_:
    #     def _kernel(weights, spikes, posts):
    #         posts[:] = 0.
    #         for i_k in range(spikes.shape[0]):
    #             col = weights[:, i_k]
    #             for i_n in range(spikes.shape[1]):
    #                 if spikes[i_k, i_n]:
    #                     posts[:, i_n] += col
    #
    # elif float_as_event:
    #     def _kernel(weights, spikes, posts):
    #         posts[:] = 0.
    #         for i_k in range(spikes.shape[0]):
    #             col = weights[:, i_k]
    #             for i_n in range(spikes.shape[1]):
    #                 if spikes[i_k, i_n] != 0.:
    #                     posts[:, i_n] += col
    #
    # else:
    #     def _kernel(weights, spikes, posts):
    #         posts[:] = 0.
    #         for i_k in range(spikes.shape[0]):
    #             col = weights[:, i_k]
    #             for i_n in range(spikes.shape[1]):
    #                 sp = spikes[i_k, i_n]
    #                 if sp != 0.:
    #                     posts[:, i_n] += col * sp

    if spk_info.dtype == jnp.bool_:
        def _kernel(weights, spikes, posts):
            posts[:] = 0.
            i_indices, j_indices = np.where(spikes)
            for i, j in zip(i_indices, j_indices):
                posts[:, j] += weights[:, i]

    elif float_as_event:
        def _kernel(weights, spikes, posts):
            posts[:] = 0.
            i_indices, j_indices = np.where(spikes != 0.)
            for i, j in zip(i_indices, j_indices):
                posts[:, j] += weights[:, i]

    else:
        def _kernel(weights, spikes, posts):
            posts[:] = 0.
            for i_k in range(spikes.shape[0]):
                col = weights[:, i_k]
                for i_n in range(spikes.shape[1]):
                    sp = spikes[i_k, i_n]
                    if sp != 0.:
                        posts[:, i_n] += col * sp

    return numba_environ.jit_fn(_kernel)


def _matrix_event_mm_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    float_as_event: bool,
    TILE_N: int,
    TILE_K: int,
    TILE_M: int,
    block_dim: int,
    **kwargs
) -> Kernel:
    import warp  # pylint: disable=import-outside-toplevel

    spike_dtype = dtype_to_warp_type(spk_info.dtype)
    weight_dtype = dtype_to_warp_type(weight_info.dtype)

    if spk_info.dtype == jnp.bool_:
        def kernel(
            weight_ref: warp.array2d(dtype=weight_dtype),
            spike_ref: warp.array2d(dtype=spike_dtype),
            out_ref: warp.array2d(dtype=weight_dtype)
        ):
            # output tile index
            i, j = warp.tid()
            sum = warp.tile_zeros(shape=(TILE_M, TILE_N), dtype=weight_dtype)
            M = weight_ref.shape[0]
            N = spike_ref.shape[1]
            K = weight_ref.shape[1]
            count = int(K / TILE_K)
            for k in range(0, count):
                a = warp.tile_load(weight_ref, shape=(TILE_M, TILE_K), offset=(i * TILE_M, k * TILE_K))
                b = warp.tile_load(spike_ref, shape=(TILE_K, TILE_N), offset=(k * TILE_K, j * TILE_N))
                # sum += a*b
                warp.tile_matmul(a, b, sum)
            warp.tile_store(out_ref, sum, offset=(i * TILE_M, j * TILE_N))

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


def _matrix_event_mm_jvp_weights(w_dot, weights, spikes, *, float_as_event, **kwargs):
    return matrix_event_mm_p_call(w_dot, spikes, float_as_event=float_as_event)


def _matrix_event_mm_jvp_spikes(spk_dot, weights, spikes, **kwargs):
    return [weights @ spk_dot]


def _matrix_event_mm_transpose_rule(ct, weights, spikes, **kwargs):
    if ad.is_undefined_primal(spikes):
        ct_events = weights.T @ ct[0]
        return weights, (ad.Zero(spikes) if type(ct[0]) is ad.Zero else ct_events)
    else:
        ct_weights = ct[0] @ spikes.T
        return (ad.Zero(weights) if type(ct[0]) is ad.Zero else ct_weights), spikes


def _matrix_event_mm_batching_axis1(args, axes, **kwargs):
    assert args[1].ndim == 3, 'Batching axis 0 requires 3D input.'
    m, batch_size, n = args[1].shape
    events = args[1].reshape(m, batch_size * n)
    r = matrix_event_mm_p_call(args[0], events, float_as_event=kwargs['float_as_event'])
    r = jnp.reshape(r[0], [r[0].shape[0], batch_size, n])
    return [r], [1]


def _matrix_event_mm_batching_axis2(args, axes, **kwargs):
    assert args[1].ndim == 3, 'Batching axis 0 requires 3D input.'
    m, n, batch_size = args[1].shape
    events = args[1].reshape(m, batch_size * n)
    r = matrix_event_mm_p_call(args[0], events, float_as_event=kwargs['float_as_event'])
    r = jnp.reshape(r[0], [r[0].shape[0], n, batch_size])
    return [r], [2]


def _matrix_event_mm_batching(args, axes, **kwargs):
    if axes == (None, 0):
        args = list(args)
        args[1] = jnp.transpose(args[1], (1, 0, 2))
        return _matrix_event_mm_batching_axis1(args, axes, **kwargs)
    elif axes == (None, 1):
        return _matrix_event_mm_batching_axis1(args, axes, **kwargs)
    elif axes == (None, 2):
        return _matrix_event_mm_batching_axis2(args, axes, **kwargs)
    else:
        return general_batching_rule(matrix_event_mm_p, args, axes, **kwargs)


def matrix_event_mm_p_call(weights, spikes, *, float_as_event: bool):
    assert weights.shape[1] == spikes.shape[0], (
        f"weights.shape[1] ({weights.shape[1]}) != spikes.shape[0] ({spikes.shape[0]})"
        f", weights: {weights.shape}, spikes: {spikes.shape} in matrix_event_mm_p_call"
    )
    out = jax.ShapeDtypeStruct([weights.shape[0], spikes.shape[1]], weights.dtype)
    return matrix_event_mm_p(
        weights,
        spikes,
        outs=[out],
        float_as_event=float_as_event,
        spk_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        TILE_SIZE=spikes.shape[0],
    )


def _matrix_event_mm_gpu_kernel():
    if config.gpu_kernel_use_warp:
        return WarpKernelGenerator(
            _matrix_event_mm_gpu_kernel_generator,
            tile=lambda weight_info, transpose, **kwargs: cdiv(weight_info.shape[0], TILE_THREAD),
            block_dim=TILE_THREAD,
        )
    else:
        return WarpKernelGenerator(
            _matrix_event_mm_gpu_kernel_generator,
            tile=lambda weight_info, transpose, **kwargs: cdiv(weight_info.shape[0], TILE_THREAD),
            block_dim=TILE_THREAD,
        )


matrix_event_mm_p = XLACustomKernel('matrix_event_mm')
matrix_event_mm_p.def_cpu_kernel(NumbaKernelGenerator(_matrix_event_mm_cpu_kernel_generator))
matrix_event_mm_p.def_gpu_kernel(KernelChoice(_matrix_event_mm_gpu_kernel))
matrix_event_mm_p.defjvp(_matrix_event_mm_jvp_weights, _matrix_event_mm_jvp_spikes)
matrix_event_mm_p.def_transpose_rule(_matrix_event_mm_transpose_rule)
matrix_event_mm_p.def_batching_rule(_matrix_event_mm_batching)


def event_matrix_mm(
    spikes,
    weights,
    *,
    float_as_event: bool = True,
):
    """Performs event-driven matrix multiplication: `spikes @ weights`.

    This function computes the matrix product of a spike matrix and a weight
    matrix, where the spike matrix often represents events (e.g., neural spikes).
    It handles potential units associated with the input arrays using the
    `brainunit` library. The computation is dispatched to specialized
    CPU/GPU kernels via `event_matrix_mm_p_call`.

    The exact multiplication behavior depends on the data type of `spikes` and
    the `float_as_event` flag.

    Parameters
    ----------
    spikes : array_like
        The spike matrix, typically with shape (M, K). Can be boolean or float.
        If boolean, True indicates an event. If float, non-zero values
        indicate an event, and the value itself might be used depending on
        `float_as_event`. Can be a `brainunit` quantity.
    weights : array_like
        The weight matrix, typically with shape (K, N). Can be a `brainunit`
        quantity.
    float_as_event : bool, optional
        Controls how float `spikes` are interpreted.
        - If True (default): Non-zero spike values indicate an event, but the
          weight is added without scaling by the spike value (effectively
          treating float spikes like boolean events).
        - If False: Non-zero spike values indicate an event, and the
          corresponding weight is scaled by the spike value before accumulation.

    Returns
    -------
    array_like
        The result of the event-driven matrix multiplication, with shape (M, N).
        If inputs had units, the output will also have appropriate units
        (product of spikes unit and weights unit).

    Notes
    -----
    The core computation performed is equivalent to:

    `output[m, n] = sum_{k} f(spikes[m, k]) * weights[k, n]`

    where the function `f(s)` is defined as:
    - If `spikes` is boolean: `f(s) = 1` if `s` is True, `0` otherwise.
    - If `spikes` is float and `float_as_event` is True: `f(s) = 1` if `s != 0`, `0` otherwise.
    - If `spikes` is float and `float_as_event` is False: `f(s) = s` if `s != 0`, `0` otherwise.

    The function ensures inputs are JAX arrays and handles unit consistency
    using `brainunit`. The actual computation is delegated to a JAX primitive
    `event_matrix_mm_p` for potential hardware acceleration. This function
    differs from `matrix_event_mm` in the order of matrix multiplication.
    """
    with jax.ensure_compile_time_eval():
        # Ensure inputs are JAX arrays, potentially handling brainunit quantities
        weights = u.math.asarray(weights)
        spikes = u.math.asarray(spikes)
    # Separate numerical values and units
    weight_val, wunit = u.split_mantissa_unit(weights)
    spk_val, spkunit = u.split_mantissa_unit(spikes)
    # Call the underlying primitive with unitless values
    r = event_matrix_mm_p_call(
        spk_val,
        weight_val,
        float_as_event=float_as_event,
    )
    # Re-attach units to the result, handling potential Decimal types
    return u.maybe_decimal(r[0] * spkunit * wunit)


def _event_matrix_mm_cpu_kernel_generator(
    float_as_event: bool,
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
) -> Kernel:
    # spikes: [m, k]
    # weights: [k, n]

    if spk_info.dtype == jnp.bool_:
        def _kernel(spikes, weights, posts):
            posts[:] = 0.
            for i_k in range(weights.shape[0]):
                row = weights[i_k]
                for i_m in range(spikes.shape[0]):
                    if spikes[i_m, i_k]:
                        posts[i_m] += row

    elif float_as_event:
        def _kernel(spikes, weights, posts):
            posts[:] = 0.
            for i_k in range(weights.shape[0]):
                row = weights[i_k]
                for i_m in range(spikes.shape[0]):
                    if spikes[i_m, i_k] != 0.:
                        posts[i_m] += row

    else:
        def _kernel(spikes, weights, posts):
            posts[:] = 0.
            for i_k in range(weights.shape[0]):
                row = weights[i_k]
                for i_m in range(spikes.shape[0]):
                    s = spikes[i_m, i_k]
                    if s != 0.:
                        posts[i_m] += row * s

    return numba_environ.jit_fn(_kernel)


def _event_matrix_mm_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    float_as_event: bool,
    TILE_N: int,
    TILE_K: int,
    TILE_M: int,
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

    return warp.kernel(kernel)


def _event_matrix_mm_jvp_weights(w_dot, spikes, weights, *, float_as_event, **kwargs):
    return event_matrix_mm_p_call(spikes, w_dot, float_as_event=float_as_event)


def _event_matrix_mm_jvp_spikes(spk_dot, spikes, weights, **kwargs):
    return [spk_dot @ weights]


def _event_matrix_mm_transpose_rule(ct, spikes, weights, **kwargs):
    if ad.is_undefined_primal(spikes):
        ct_events = ct[0] @ weights.T
        return (ad.Zero(spikes) if type(ct[0]) is ad.Zero else ct_events), weights

    else:
        ct_weights = spikes.T @ ct[0]
        return spikes, (ad.Zero(weights) if type(ct[0]) is ad.Zero else ct_weights)


def _event_matrix_mm_batching_axis0(args, axes, **kwargs):
    assert args[0].ndim == 3, 'Batching axis 0 requires 3D input.'
    batch_size, m, n = args[0].shape
    events = args[0].reshape(batch_size * m, n)
    r = event_matrix_mm_p_call(events, args[1], float_as_event=kwargs['float_as_event'])
    r = jnp.reshape(r[0], [batch_size, m, r[0].shape[1]])
    return [r], [0]


def _event_matrix_mm_batching(args, axes, **kwargs):
    if axes == (0, None):
        return _event_matrix_mm_batching_axis0(args, axes, **kwargs)
    elif axes == (1, None):
        args = list(args)
        args[0] = jnp.transpose(args[0], (1, 0, 2))
        return _event_matrix_mm_batching_axis0(args, axes, **kwargs)
    elif axes == (2, None):
        args = list(args)
        args[0] = jnp.transpose(args[0], (2, 0, 1))
        return _event_matrix_mm_batching_axis0(args, axes, **kwargs)
    else:
        return general_batching_rule(event_matrix_mm_p, args, axes, **kwargs)


def event_matrix_mm_p_call(spikes, weights, *, float_as_event: bool):
    assert spikes.shape[1] == weights.shape[0], (
        f"spikes shape {spikes.shape} and weights shape {weights.shape} do not match"
        f"for event matrix multiplication"
    )
    out = jax.ShapeDtypeStruct([spikes.shape[0], weights.shape[1]], weights.dtype)
    return event_matrix_mm_p(
        spikes,
        weights,
        outs=[out],
        float_as_event=float_as_event,
        spk_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        TILE_SIZE=spikes.shape[0],
    )


def _event_matrix_mm_gpu_kernel():
    if config.gpu_kernel_use_warp:
        return WarpKernelGenerator(
            _event_matrix_mm_gpu_kernel_generator,
            tile=lambda weight_info, transpose, **kwargs: cdiv(weight_info.shape[1], TILE_THREAD),
            block_dim=TILE_THREAD,
        )
    else:
        return WarpKernelGenerator(
            _event_matrix_mm_gpu_kernel_generator,
            tile=lambda weight_info, transpose, **kwargs: cdiv(weight_info.shape[1], TILE_THREAD),
            block_dim=TILE_THREAD,
        )


event_matrix_mm_p = XLACustomKernel('event_matrix_mm', )
event_matrix_mm_p.def_cpu_kernel(NumbaKernelGenerator(_event_matrix_mm_cpu_kernel_generator))
event_matrix_mm_p.def_gpu_kernel(KernelChoice(_event_matrix_mm_gpu_kernel))
event_matrix_mm_p.defjvp(_event_matrix_mm_jvp_spikes, _event_matrix_mm_jvp_weights, )
event_matrix_mm_p.def_transpose_rule(_event_matrix_mm_transpose_rule)
event_matrix_mm_p.def_batching_rule(_event_matrix_mm_batching)
