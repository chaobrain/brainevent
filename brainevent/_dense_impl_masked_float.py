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

from ._compatible_import import pallas as pl
from ._misc import cdiv
from ._xla_custom_op import XLACustomKernel
from ._xla_custom_op_numba import numba_kernel
from ._xla_custom_op_util import general_batching_rule
from ._xla_custom_op_warp import jaxtype_to_warptype, warp_kernel

TILE_THREAD = 256


def dense_mat_dot_masked_float_vec(
    weights,
    spikes,
):
    with jax.ensure_compile_time_eval():
        weights = u.math.asarray(weights)
        spikes = u.math.asarray(spikes)
    weight_val, wunit = u.split_mantissa_unit(weights)
    spk_val, spkunit = u.split_mantissa_unit(spikes)
    r = dense_mat_dot_masked_float_vec_p_call(
        weight_val,
        spk_val,
    )
    return u.maybe_decimal(r[0] * wunit * spkunit)


def _dense_mat_dot_masked_float_vec_numba_cpu_kernel_generator(
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
):
    def _kernel(weights, spikes, posts):
        posts[:] = 0.
        for i in range(spikes.shape[0]):
            sp = spikes[i]
            if sp != 0.:
                posts += weights[:, i] * sp

    return numba_kernel(_kernel)


def _dense_mat_dot_masked_float_vec_warp_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
):
    TILE_SIZE = spk_info.shape[0]
    block_dim = TILE_THREAD

    import warp  # pylint: disable=import-outside-toplevel
    assert warp.__version__ >= '1.8.0', "warp version >= 1.8.0 is required"

    spike_dtype = jaxtype_to_warptype(spk_info.dtype)
    weight_dtype = jaxtype_to_warptype(weight_info.dtype)

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
                temp += warp.tile_squeeze(data) * s  # need warp>=1.8.0
        warp.tile_store(out_ref, temp, offset=(i_col * block_dim,))

    tile = cdiv(weight_info.shape[0], TILE_THREAD)
    return warp_kernel(kernel, tile=tile, block_dim=TILE_THREAD)


def _dense_mat_dot_masked_float_vec_jvp_weights(w_dot, weights, spikes, **kwargs):
    return dense_mat_dot_masked_float_vec_p_call(w_dot, spikes)


def _dense_mat_dot_masked_float_vec_jvp_spikes(spk_dot, weights, spikes, **kwargs):
    return [weights @ spk_dot]


def _dense_mat_dot_masked_float_vec_transpose_rule(ct, weights, spikes, **kwargs):
    if ad.is_undefined_primal(spikes):
        ct_events = jnp.matmul(ct[0], weights)
        return weights, (ad.Zero(spikes) if type(ct[0]) is ad.Zero else ct_events)
    else:
        ct_weights = jnp.outer(ct[0], spikes)
        return (ad.Zero(weights) if type(ct[0]) is ad.Zero else ct_weights), spikes


def _dense_mat_dot_masked_float_vec_batching(args, axes, **kwargs):
    if axes == (None, 0):
        r = dense_mat_dot_masked_float_mat(args[0], args[1].T)
        return [r], [1]
    if axes == (None, 1):
        r = dense_mat_dot_masked_float_mat(args[0], args[1])
        return [r], [1]
    else:
        return general_batching_rule(dense_mat_dot_masked_float_vec_p, args, axes, **kwargs)


def dense_mat_dot_masked_float_vec_p_call(weights, spikes):
    assert spikes.shape[0] == weights.shape[1], (
        f"spikes shape {spikes.shape} and weights shape {weights.shape} are not compatible"
    )
    out = jax.ShapeDtypeStruct([weights.shape[0]], weights.dtype)
    return dense_mat_dot_masked_float_vec_p(
        weights,
        spikes,
        outs=[out],
        spk_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
    )


dense_mat_dot_masked_float_vec_p = XLACustomKernel('dense_mat_dot_masked_float_vector')
dense_mat_dot_masked_float_vec_p.def_cpu_kernel(_dense_mat_dot_masked_float_vec_numba_cpu_kernel_generator)
dense_mat_dot_masked_float_vec_p.def_gpu_kernel(warp=_dense_mat_dot_masked_float_vec_warp_kernel_generator)
dense_mat_dot_masked_float_vec_p.def_jvp_rule2(_dense_mat_dot_masked_float_vec_jvp_weights,
                                               _dense_mat_dot_masked_float_vec_jvp_spikes)
dense_mat_dot_masked_float_vec_p.def_transpose_rule(_dense_mat_dot_masked_float_vec_transpose_rule)
dense_mat_dot_masked_float_vec_p.def_batching_rule(_dense_mat_dot_masked_float_vec_batching)


def masked_float_vec_dot_dense_mat(
    spikes,
    weights,
):
    with jax.ensure_compile_time_eval():
        weights = u.math.asarray(weights)
        spikes = u.math.asarray(spikes)
    weight_val, wunit = u.split_mantissa_unit(weights)
    spk_val, spkunit = u.split_mantissa_unit(spikes)
    r = masked_float_vec_dot_dense_mat_p_call(
        spk_val,
        weight_val,
    )
    return u.maybe_decimal(r[0] * wunit * spkunit)


def _masked_float_vec_dot_dense_mat_numba_kernel_generator(
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
):
    def _kernel(spikes, weights, posts):
        posts[:] = 0.
        for i in range(spikes.shape[0]):
            sp = spikes[i]
            if sp != 0.:
                posts += weights[i] * sp

    return numba_kernel(_kernel)


def _masked_float_vec_dot_dense_mat_warp_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp  # pylint: disable=import-outside-toplevel

    TILE_SIZE = spk_info.shape[0]
    block_dim = TILE_THREAD

    spike_dtype = jaxtype_to_warptype(spk_info.dtype)
    weight_dtype = jaxtype_to_warptype(weight_info.dtype)

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

    tile = cdiv(weight_info.shape[1], TILE_THREAD)

    return warp_kernel(kernel, tile=tile, block_dim=TILE_THREAD)


def _masked_float_vec_dot_dense_mat_pallas_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    TILE_SIZE: int,
    block_dim: int,
    **kwargs
):
    n_pre = spk_info.shape[0]
    n_post = weight_info.shape[1]

    def _raw_kernel(
        spike_ref,  # [n_pre]
        weight_ref,  # [n_pre, n_post]
        _,
        out_ref,  # [n_post]
    ):
        i_row = pl.program_id(0)
        spike = spike_ref[i_row]

        def true_fn():
            def loop_fn(i, _):
                i = i * block_dim
                mask = i + jnp.arange(block_dim) < n_post
                weight = pl.load(weight_ref, (i_row, pl.dslice(i, block_dim)), mask=mask)
                weight2 = weight * spike
                pl.atomic_add(out_ref, ind, weight2, mask=mask)

            jax.lax.fori_loop(0, pl.cdiv(n_conn, block_dim), loop_fn, None)

        jax.lax.cond(spike if spike_ref.dtype == jnp.bool_ else (spike != 0.), true_fn, lambda: None)

    # homogenous weights
    def kernel(weight, indices, spikes, out):
        fn = pl.pallas_call(
            _raw_kernel,
            out_shape=jax.ShapeDtypeStruct(out.shape, out.dtype),
            grid=(n_pre,),
            input_output_aliases={2: 0},
        )
        return [fn(spikes, indices, out) * weight]

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


def _masked_float_vec_dot_dense_mat_jvp_weights(w_dot, spikes, weights, **kwargs):
    return masked_float_vec_dot_dense_mat_p_call(spikes, w_dot)


def _masked_float_vec_dot_dense_mat_jvp_spikes(spk_dot, spikes, weights, **kwargs):
    return [spk_dot @ weights]


def _masked_float_vec_dot_dense_mat_transpose_rule(ct, spikes, weights, **kwargs):
    if ad.is_undefined_primal(spikes):
        ct_events = jnp.matmul(weights, ct[0])
        return (ad.Zero(spikes) if type(ct[0]) is ad.Zero else ct_events), weights

    else:
        ct_weights = jnp.outer(spikes, ct[0])
        return spikes, (ad.Zero(weights) if type(ct[0]) is ad.Zero else ct_weights)


def _event_matrix_batching(args, axes, **kwargs):
    if axes == (0, None):
        r = masked_float_mat_dot_dense_mat(args[0], args[1])
        return [r], [0]
    if axes == (1, None):
        r = masked_float_mat_dot_dense_mat(args[0].T, args[1])
        return [r], [0]
    else:
        return general_batching_rule(masked_float_vec_dot_dense_mat_p, args, axes, **kwargs)


def masked_float_vec_dot_dense_mat_p_call(spikes, weights):
    assert spikes.shape[0] == weights.shape[0], (
        f"shapes {spikes.shape} and {weights.shape} not aligned: "
        f"{spikes.shape[0]} (dim 0) != {weights.shape[0]} (dim 0)"
    )
    out = jax.ShapeDtypeStruct([weights.shape[1]], weights.dtype)
    return masked_float_vec_dot_dense_mat_p(
        spikes,
        weights,
        outs=[out],
        spk_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
    )


masked_float_vec_dot_dense_mat_p = XLACustomKernel('masked_float_vector_dot_dense_matrix')
masked_float_vec_dot_dense_mat_p.def_cpu_kernel(_masked_float_vec_dot_dense_mat_numba_kernel_generator)
masked_float_vec_dot_dense_mat_p.def_gpu_kernel(warp=_masked_float_vec_dot_dense_mat_warp_kernel_generator)
masked_float_vec_dot_dense_mat_p.def_jvp_rule2(_masked_float_vec_dot_dense_mat_jvp_spikes,
                                               _masked_float_vec_dot_dense_mat_jvp_weights)
masked_float_vec_dot_dense_mat_p.def_transpose_rule(_masked_float_vec_dot_dense_mat_transpose_rule)
masked_float_vec_dot_dense_mat_p.def_batching_rule(_event_matrix_batching)


def dense_mat_dot_masked_float_mat(
    weights,
    spikes,
):
    """Performs event-driven matrix multiplication: `weights @ spikes`.

    This function computes the matrix product of a weight matrix and a spike
    matrix, where the spike matrix often represents events (e.g., neural spikes).
    It handles potential units associated with the input arrays using the
    `brainunit` library. The computation is dispatched to specialized
    CPU/GPU kernels via `dense_mat_dot_masked_float_mat_p_call`.

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
    `dense_mat_dot_masked_float_mat_p` for potential hardware acceleration.
    """
    with jax.ensure_compile_time_eval():
        weights = u.math.asarray(weights)
        spikes = u.math.asarray(spikes)
    weight_val, wunit = u.split_mantissa_unit(weights)
    spk_val, spkunit = u.split_mantissa_unit(spikes)
    # Call the underlying primitive with unitless values
    r = dense_mat_dot_masked_float_mat_p_call(
        weight_val,
        spk_val,
    )
    # Re-attach units to the result
    return u.maybe_decimal(r[0] * wunit * spkunit)


def _dense_mat_dot_masked_float_mat_cpu_kernel_generator(
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
):
    # weights: [m, k]
    # spikes: [k, n]

    def _kernel(weights, spikes, posts):
        posts[:] = 0.
        for i_k in range(spikes.shape[0]):
            col = weights[:, i_k]
            for i_n in range(spikes.shape[1]):
                sp = spikes[i_k, i_n]
                if sp != 0.:
                    posts[:, i_n] += col * sp

    return numba_kernel(_kernel)


def _dense_mat_dot_masked_float_mat_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    TILE_N: int,
    TILE_K: int,
    TILE_M: int,
    TILE_SIZE: int,
    block_dim: int,
    **kwargs
):
    block_dim = TILE_THREAD

    import warp  # pylint: disable=import-outside-toplevel

    spike_dtype = jaxtype_to_warptype(spk_info.dtype)
    weight_dtype = jaxtype_to_warptype(weight_info.dtype)

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

    tile = cdiv(weight_info.shape[0], TILE_THREAD)
    return warp_kernel(kernel, tile=tile, block_dim=TILE_THREAD)


def _dense_mat_dot_masked_float_mat_jvp_weights(w_dot, weights, spikes, **kwargs):
    return dense_mat_dot_masked_float_mat_p_call(w_dot, spikes)


def _dense_mat_dot_masked_float_mat_jvp_spikes(spk_dot, weights, spikes, **kwargs):
    return [weights @ spk_dot]


def _dense_mat_dot_masked_float_mat_transpose_rule(ct, weights, spikes, **kwargs):
    if ad.is_undefined_primal(spikes):
        ct_events = weights.T @ ct[0]
        return weights, (ad.Zero(spikes) if type(ct[0]) is ad.Zero else ct_events)
    else:
        ct_weights = ct[0] @ spikes.T
        return (ad.Zero(weights) if type(ct[0]) is ad.Zero else ct_weights), spikes


def _dense_mat_dot_masked_float_mat_batching_axis1(args, axes, **kwargs):
    assert args[1].ndim == 3, 'Batching axis 0 requires 3D input.'
    m, batch_size, n = args[1].shape
    events = args[1].reshape(m, batch_size * n)
    r = dense_mat_dot_masked_float_mat_p_call(args[0], events)
    r = jnp.reshape(r[0], [r[0].shape[0], batch_size, n])
    return [r], [1]


def _dense_mat_dot_masked_float_mat_batching_axis2(args, axes, **kwargs):
    assert args[1].ndim == 3, 'Batching axis 0 requires 3D input.'
    m, n, batch_size = args[1].shape
    events = args[1].reshape(m, batch_size * n)
    r = dense_mat_dot_masked_float_mat_p_call(args[0], events)
    r = jnp.reshape(r[0], [r[0].shape[0], n, batch_size])
    return [r], [2]


def _dense_mat_dot_masked_float_mat_batching(args, axes, **kwargs):
    if axes == (None, 0):
        args = list(args)
        args[1] = jnp.transpose(args[1], (1, 0, 2))
        return _dense_mat_dot_masked_float_mat_batching_axis1(args, axes, **kwargs)
    elif axes == (None, 1):
        return _dense_mat_dot_masked_float_mat_batching_axis1(args, axes, **kwargs)
    elif axes == (None, 2):
        return _dense_mat_dot_masked_float_mat_batching_axis2(args, axes, **kwargs)
    else:
        return general_batching_rule(dense_mat_dot_masked_float_mat_p, args, axes, **kwargs)


def dense_mat_dot_masked_float_mat_p_call(weights, spikes):
    assert weights.shape[1] == spikes.shape[0], (
        f"weights.shape[1] ({weights.shape[1]}) != spikes.shape[0] ({spikes.shape[0]})"
        f", weights: {weights.shape}, spikes: {spikes.shape} in dense_mat_dot_masked_float_mat_p_call"
    )
    out = jax.ShapeDtypeStruct([weights.shape[0], spikes.shape[1]], weights.dtype)
    return dense_mat_dot_masked_float_mat_p(
        weights,
        spikes,
        outs=[out],
        spk_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        TILE_SIZE=spikes.shape[0],
    )


dense_mat_dot_masked_float_mat_p = XLACustomKernel('dense_matrix_dot_masked_float_matrix')
dense_mat_dot_masked_float_mat_p.def_cpu_kernel(_dense_mat_dot_masked_float_mat_cpu_kernel_generator)
dense_mat_dot_masked_float_mat_p.def_gpu_kernel(warp=_dense_mat_dot_masked_float_mat_gpu_kernel_generator)
dense_mat_dot_masked_float_mat_p.def_jvp_rule2(_dense_mat_dot_masked_float_mat_jvp_weights,
                                               _dense_mat_dot_masked_float_mat_jvp_spikes)
dense_mat_dot_masked_float_mat_p.def_transpose_rule(_dense_mat_dot_masked_float_mat_transpose_rule)
dense_mat_dot_masked_float_mat_p.def_batching_rule(_dense_mat_dot_masked_float_mat_batching)


def masked_float_mat_dot_dense_mat(
    spikes,
    weights,
):
    """Performs event-driven matrix multiplication: `spikes @ weights`.

    This function computes the matrix product of a spike matrix and a weight
    matrix, where the spike matrix often represents events (e.g., neural spikes).
    It handles potential units associated with the input arrays using the
    `brainunit` library. The computation is dispatched to specialized
    CPU/GPU kernels via `masked_float_mat_dot_dense_mat_p_call`.

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
    `masked_float_mat_dot_dense_mat_p` for potential hardware acceleration. This function
    differs from `dense_mat_dot_masked_float_mat` in the order of matrix multiplication.
    """
    with jax.ensure_compile_time_eval():
        # Ensure inputs are JAX arrays, potentially handling brainunit quantities
        weights = u.math.asarray(weights)
        spikes = u.math.asarray(spikes)
    # Separate numerical values and units
    weight_val, wunit = u.split_mantissa_unit(weights)
    spk_val, spkunit = u.split_mantissa_unit(spikes)
    # Call the underlying primitive with unitless values
    r = masked_float_mat_dot_dense_mat_p_call(
        spk_val,
        weight_val,
    )
    # Re-attach units to the result, handling potential Decimal types
    return u.maybe_decimal(r[0] * spkunit * wunit)


def _masked_float_mat_dot_dense_mat_cpu_kernel_generator(
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
):
    # spikes: [m, k]
    # weights: [k, n]

    def _kernel(spikes, weights, posts):
        posts[:] = 0.
        for i_k in range(weights.shape[0]):
            row = weights[i_k]
            for i_m in range(spikes.shape[0]):
                s = spikes[i_m, i_k]
                if s != 0.:
                    posts[i_m] += row * s

    return numba_kernel(_kernel)


def _masked_float_mat_dot_dense_mat_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    TILE_N: int,
    TILE_K: int,
    TILE_M: int,
    TILE_SIZE: int,
    block_dim: int,
    **kwargs
):
    import warp  # pylint: disable=import-outside-toplevel

    spike_dtype = jaxtype_to_warptype(spk_info.dtype)
    weight_dtype = jaxtype_to_warptype(weight_info.dtype)

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

    tile = cdiv(weight_info.shape[1], TILE_THREAD)
    return warp_kernel(kernel, tile=tile, block_dim=TILE_THREAD)


def _masked_float_mat_dot_dense_mat_jvp_weights(w_dot, spikes, weights, **kwargs):
    return masked_float_mat_dot_dense_mat_p_call(spikes, w_dot)


def _masked_float_mat_dot_dense_mat_jvp_spikes(spk_dot, spikes, weights, **kwargs):
    return [spk_dot @ weights]


def _masked_float_mat_dot_dense_mat_transpose_rule(ct, spikes, weights, **kwargs):
    if ad.is_undefined_primal(spikes):
        ct_events = ct[0] @ weights.T
        return (ad.Zero(spikes) if type(ct[0]) is ad.Zero else ct_events), weights

    else:
        ct_weights = spikes.T @ ct[0]
        return spikes, (ad.Zero(weights) if type(ct[0]) is ad.Zero else ct_weights)


def _masked_float_mat_dot_dense_mat_batching_axis0(args, axes, **kwargs):
    assert args[0].ndim == 3, 'Batching axis 0 requires 3D input.'
    batch_size, m, n = args[0].shape
    events = args[0].reshape(batch_size * m, n)
    r = masked_float_mat_dot_dense_mat_p_call(events, args[1])
    r = jnp.reshape(r[0], [batch_size, m, r[0].shape[1]])
    return [r], [0]


def _masked_float_mat_dot_dense_mat_batching(args, axes, **kwargs):
    if axes == (0, None):
        return _masked_float_mat_dot_dense_mat_batching_axis0(args, axes, **kwargs)
    elif axes == (1, None):
        args = list(args)
        args[0] = jnp.transpose(args[0], (1, 0, 2))
        return _masked_float_mat_dot_dense_mat_batching_axis0(args, axes, **kwargs)
    elif axes == (2, None):
        args = list(args)
        args[0] = jnp.transpose(args[0], (2, 0, 1))
        return _masked_float_mat_dot_dense_mat_batching_axis0(args, axes, **kwargs)
    else:
        return general_batching_rule(masked_float_mat_dot_dense_mat_p, args, axes, **kwargs)


def masked_float_mat_dot_dense_mat_p_call(spikes, weights):
    assert spikes.shape[1] == weights.shape[0], (
        f"spikes shape {spikes.shape} and weights shape {weights.shape} do not match"
        f"for event matrix multiplication"
    )
    out = jax.ShapeDtypeStruct([spikes.shape[0], weights.shape[1]], weights.dtype)
    return masked_float_mat_dot_dense_mat_p(
        spikes,
        weights,
        outs=[out],
        spk_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        TILE_SIZE=spikes.shape[0],
    )


masked_float_mat_dot_dense_mat_p = XLACustomKernel('masked_float_matrix_dot_dense_matrix')
masked_float_mat_dot_dense_mat_p.def_cpu_kernel(_masked_float_mat_dot_dense_mat_cpu_kernel_generator)
masked_float_mat_dot_dense_mat_p.def_gpu_kernel(warp=_masked_float_mat_dot_dense_mat_gpu_kernel_generator)
masked_float_mat_dot_dense_mat_p.def_jvp_rule2(_masked_float_mat_dot_dense_mat_jvp_spikes,
                                               _masked_float_mat_dot_dense_mat_jvp_weights)
masked_float_mat_dot_dense_mat_p.def_transpose_rule(_masked_float_mat_dot_dense_mat_transpose_rule)
masked_float_mat_dot_dense_mat_p.def_batching_rule(_masked_float_mat_dot_dense_mat_batching)
