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

from ._misc import cdiv
from ._typing import Kernel
from ._xla_custom_op import XLACustomKernel
from ._xla_custom_op_numba import numba_environ, NumbaKernelGenerator
from ._xla_custom_op_warp import WarpKernelGenerator, dtype_to_warp_type

N_THREAD = 256


def matrix_event_mm(
    weights,
    spikes,
    *,
    float_as_event: bool = True,
):
    with jax.ensure_compile_time_eval():
        weights = u.math.asarray(weights)
        spikes = u.math.asarray(spikes)
    weight_val, wunit = u.split_mantissa_unit(weights)
    spk_val, spkunit = u.split_mantissa_unit(spikes)
    r = matrix_event_mm_p_call(
        weight_val,
        spk_val,
        float_as_event=float_as_event,
    )
    return u.maybe_decimal(r[0] * wunit * spkunit)


def _matrix_event_mm_cpu_kernel_generator(
    float_as_event: bool,
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
) -> Kernel:
    import numba  # pylint: disable=import-outside-toplevel

    # weights: [m, k]
    # spikes: [k, n]

    if spk_info.dtype == jnp.bool_:
        def _kernel(weights, spikes, posts):
            posts[:] = 0.
            for i_k in range(spikes.shape[0]):
                col = weights[:, i_k]
                for i_n in range(spikes.shape[1]):
                    if spikes[i_k, i_n]:
                        posts[:, i_n] += col

    elif float_as_event:
        def _kernel(weights, spikes, posts):
            posts[:] = 0.
            for i_k in range(spikes.shape[0]):
                col = weights[:, i_k]
                for i_n in range(spikes.shape[1]):
                    if spikes[i_k, i_n] != 0.:
                        posts[:, i_n] += col

    else:
        def _kernel(weights, spikes, posts):
            posts[:] = 0.
            for i_k in range(spikes.shape[0]):
                col = weights[:, i_k]
                for i_n in range(spikes.shape[1]):
                    sp = spikes[i_k, i_n]
                    if sp != 0.:
                        posts[:, i_n] += col * sp

    return numba.njit(**numba_environ.setting)(_kernel)


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
    return [spk_dot @ weights]


def _matrix_event_mm_transpose_rule(ct, weights, spikes, **kwargs):
    if ad.is_undefined_primal(spikes):
        ct_events = jnp.matmul(ct[0], weights)
        return weights, (ad.Zero(spikes) if type(ct[0]) is ad.Zero else ct_events)
    else:
        ct_weights = jnp.outer(ct[0], spikes)
        return (ad.Zero(weights) if type(ct[0]) is ad.Zero else ct_weights), spikes


def matrix_event_mm_p_call(weights, spikes, *, float_as_event: bool):
    out = jax.ShapeDtypeStruct([weights.shape[0]], weights.dtype)
    return matrix_event_mm_p(
        weights,
        spikes,
        outs=[out],
        float_as_event=float_as_event,
        spk_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        TILE_SIZE=spikes.shape[0],
    )


matrix_event_mm_p = XLACustomKernel(
    'matrix_event_mm',
    cpu_kernel=NumbaKernelGenerator(_matrix_event_mm_cpu_kernel_generator),
    gpu_kernel=WarpKernelGenerator(
        _matrix_event_mm_gpu_kernel_generator,
        tile=lambda weight_info, transpose, **kwargs: cdiv(weight_info.shape[0], N_THREAD),
        block_dim=N_THREAD,
    ),
)
matrix_event_mm_p.defjvp(_matrix_event_mm_jvp_weights, _matrix_event_mm_jvp_spikes)
matrix_event_mm_p.def_transpose_rule(_matrix_event_mm_transpose_rule)


def event_matrix_mm(
    weights,
    spikes,
    *,
    float_as_event: bool = True,
    transpose: bool = False,
):
    with jax.ensure_compile_time_eval():
        weights = u.math.asarray(weights)
        spikes = u.math.asarray(spikes)
    weight_val, wunit = u.split_mantissa_unit(weights)
    spk_val, spkunit = u.split_mantissa_unit(spikes)
    r = event_matrix_mm_p_call(
        weight_val,
        spk_val,
        float_as_event=float_as_event,
        transpose=transpose
    )
    return u.maybe_decimal(r[0] * wunit * spkunit)


def _event_matrix_mm_cpu_kernel_generator(
    float_as_event: bool,
    transpose: bool,
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
) -> Kernel:
    import numba  # pylint: disable=import-outside-toplevel

    if transpose:
        # weights: [k, m]
        # spikes: [k, n]

        if spk_info.dtype == jnp.bool_:
            def _kernel(weights, spikes, posts):
                posts[:] = 0.
                for i_k in range(spikes.shape[0]):
                    col = weights[i_k]
                    for i_n in range(spikes.shape[1]):
                        if spikes[i_k, i_n]:
                            posts[i_n] += col

        elif float_as_event:
            def _kernel(weights, spikes, posts):
                posts[:] = 0.
                for i_k in range(spikes.shape[0]):
                    col = weights[i_k]
                    for i_n in range(spikes.shape[1]):
                        if spikes[i_k, i_n] != 0.:
                            posts[i_n] += col

        else:
            def _kernel(weights, spikes, posts):
                posts[:] = 0.
                for i_k in range(spikes.shape[0]):
                    col = weights[i_k]
                    for i_n in range(spikes.shape[1]):
                        sp = spikes[i_k, i_n]
                        if sp != 0.:
                            posts[i_n] += col * sp

    else:
        # weights: [m, k]
        # spikes: [k, n]

        if spk_info.dtype == jnp.bool_:
            def _kernel(weights, spikes, posts):
                posts[:] = 0.
                for i_k in range(spikes.shape[0]):
                    col = weights[:, i_k]
                    for i_n in range(spikes.shape[1]):
                        if spikes[i_k, i_n]:
                            posts[:, i_n] += col

        elif float_as_event:
            def _kernel(weights, spikes, posts):
                posts[:] = 0.
                for i_k in range(spikes.shape[0]):
                    col = weights[:, i_k]
                    for i_n in range(spikes.shape[1]):
                        if spikes[i_k, i_n] != 0.:
                            posts[:, i_n] += col

        else:
            def _kernel(weights, spikes, posts):
                posts[:] = 0.
                for i_k in range(spikes.shape[0]):
                    col = weights[:, i_k]
                    for i_n in range(spikes.shape[1]):
                        sp = spikes[i_k, i_n]
                        if sp != 0.:
                            posts[:, i_n] += col * sp

    return numba.njit(**numba_environ.setting)(_kernel)


def _event_matrix_mm_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    transpose: bool,
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


def _event_matrix_mm_jvp_weights(w_dot, weights, spikes, *, float_as_event, transpose, **kwargs):
    return event_matrix_mm_p_call(w_dot, spikes, transpose=transpose, float_as_event=float_as_event)


def _event_matrix_mm_jvp_spikes(spk_dot, weights, spikes, *, transpose, **kwargs):
    if transpose:
        return [spk_dot @ weights]
    else:
        return [weights @ spk_dot]


def _event_matrix_mm_transpose_rule(ct, weights, spikes, *, transpose, **kwargs):
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


def event_matrix_mm_p_call(weights, spikes, *, transpose: bool, float_as_event: bool):
    if transpose:
        out = jax.ShapeDtypeStruct([weights.shape[1]], weights.dtype)
    else:
        out = jax.ShapeDtypeStruct([weights.shape[0]], weights.dtype)

    return event_matrix_mm_p(
        weights,
        spikes,
        outs=[out],
        float_as_event=float_as_event,
        transpose=transpose,
        spk_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        TILE_SIZE=spikes.shape[0],
    )


event_matrix_mm_p = XLACustomKernel(
    'event_matrix_mm',
    cpu_kernel=NumbaKernelGenerator(_event_matrix_mm_cpu_kernel_generator),
    gpu_kernel=WarpKernelGenerator(
        _event_matrix_mm_gpu_kernel_generator,
        tile=lambda weight_info, transpose, **kwargs: (
            cdiv(weight_info.shape[1], N_THREAD)
            if transpose else
            cdiv(weight_info.shape[0], N_THREAD)
        ),
        block_dim=N_THREAD,
    ),
)
event_matrix_mm_p.defjvp(_event_matrix_mm_jvp_weights, _event_matrix_mm_jvp_spikes)
event_matrix_mm_p.def_transpose_rule(_event_matrix_mm_transpose_rule)
