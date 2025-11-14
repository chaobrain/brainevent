# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

from brainevent._compatible_import import pallas as pl
from brainevent._misc import cdiv, generate_block_dim, namescoped_jit
from brainevent._op.main import XLACustomKernel
from brainevent._op.op_numba import numba_kernel
from brainevent._op.op_pallas import pallas_kernel
from brainevent._op.util import general_batching_rule
from brainevent._op.op_warp import jaxtype_to_warptype, warp_kernel


@namescoped_jit()
def dense_mat_dot_masked_float_vec(weights, spikes):
    """
    Performs event-driven matrix-vector multiplication: `weights @ spikes`.

    This function computes the product of a dense weight matrix and a binary
    vector, where the binary vector often represents events (e.g., neural spikes).
    It handles potential units associated with the input arrays using the
    `brainunit` library. The actual computation is dispatched to specialized
    CPU/GPU kernels via `dense_mat_dot_masked_float_vec_p_call`.

    Parameters
    ----------
    weights : array_like
        The weight matrix, typically with shape (M, K). Can be a `brainunit`
        quantity.
    spikes : array_like
        The binary vector, typically with shape (K,). Can be boolean or float.
        If boolean, True indicates an event. If float, non-zero values
        indicate an event. Can be a `brainunit` quantity.

    Returns
    -------
    array_like
        The result of the event-driven matrix-vector multiplication, with shape (M,).
        If inputs had units, the output will also have appropriate units
        (product of weights unit and spikes unit).

    Notes
    -----
    The core computation performed is equivalent to:

    `output[m] = sum_{k} weights[m, k] * f(spikes[k])`

    where the function `f(s)` is defined as:
    - If `spikes` is boolean: `f(s) = 1` if `s` is True, `0` otherwise.
    - If `spikes` is float: `f(s) = 1` if `s != 0`, `0` otherwise.

    The function ensures inputs are JAX arrays and handles unit consistency
    using `brainunit`. The computation is delegated to a JAX primitive
    `dense_mat_dot_masked_float_vec_p` for potential hardware acceleration.
    """
    with jax.ensure_compile_time_eval():
        weights = u.math.asarray(weights)
        spikes = u.math.asarray(spikes)
    weight_val, wunit = u.split_mantissa_unit(weights)
    spk_val, spkunit = u.split_mantissa_unit(spikes)
    r = dense_mat_dot_masked_float_vec_p_call(weight_val, spk_val)
    return u.maybe_decimal(r[0] * wunit * spkunit)


def _dense_mat_dot_masked_float_vec_numba_cpu_kernel_generator(**kwargs):
    def _kernel(weights, spikes, posts):
        posts[:] = 0.
        for i in range(spikes.shape[0]):
            spk = spikes[i]
            if spk != 0.:
                posts += weights[:, i] * spk

    return numba_kernel(_kernel)


def _dense_mat_dot_masked_float_vec_warp_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
):
    spike_length = spk_info.shape[0]
    block_dim = generate_block_dim(weight_info.shape[0], maximum=512)

    import warp  # pylint: disable=import-outside-toplevel
    assert warp.__version__ >= '1.8.0', "warp version >= 1.8.0 is required"

    spike_dtype = jaxtype_to_warptype(spk_info.dtype)
    weight_dtype = jaxtype_to_warptype(weight_info.dtype)

    def kernel(
        weight_ref: warp.array2d(dtype=weight_dtype),
        spike_ref: warp.array1d(dtype=spike_dtype),
        out_ref: warp.array1d(dtype=weight_dtype),
    ):
        i_row_block = warp.tid()
        spikes = warp.tile_load(spike_ref, shape=(spike_length,))
        temp = warp.tile_zeros(shape=(block_dim,), dtype=weight_dtype)
        for j in range(spike_length):
            spk = spikes[j]
            if spk != 0.:
                data = warp.tile_load(weight_ref, shape=(block_dim, 1), offset=(i_row_block * block_dim, j))
                temp += warp.tile_squeeze(data) * spk  # need warp>=1.8.0
        warp.tile_store(out_ref, temp, offset=(i_row_block * block_dim,))

    return warp_kernel(
        kernel,
        tile=cdiv(weight_info.shape[0], block_dim),
        block_dim=block_dim,
    )


def _dense_mat_dot_masked_float_vec_pallas_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    **kwargs
):
    mat_block_dim = generate_block_dim(weight_info.shape[0], maximum=1024)

    def kernel(weight_ref, spike_ref, out_ref):
        i_row_block = pl.program_id(0)
        i_row_start = i_row_block * mat_block_dim
        i_row_mask = i_row_start + jnp.arange(mat_block_dim) < weight_ref.shape[0]

        def loop_fn(i_spike, temp):
            spike = spike_ref[i_spike]
            return jax.lax.cond(
                spike != 0.,
                lambda out: out + pl.load(
                    weight_ref,
                    (pl.dslice(i_row_start, mat_block_dim), i_spike),
                    mask=i_row_mask,
                    other=0.0
                ) * spike,
                lambda out: out,
                temp
            )

        i_row_out = jax.lax.fori_loop(
            0,
            spike_ref.shape[0],
            loop_fn,
            jnp.zeros((mat_block_dim,), dtype=weight_ref.dtype)
        )
        pl.store(out_ref, pl.dslice(i_row_start, mat_block_dim), i_row_out, mask=i_row_mask)

    return pallas_kernel(
        kernel,
        tile=[cdiv(weight_info.shape[0], mat_block_dim)],
        outs=kwargs['outs'],
    )


def _dense_mat_dot_masked_float_vec_jvp_weights(w_dot, weights, spikes, **kwargs):
    return dense_mat_dot_masked_float_vec_p_call(w_dot, spikes)


def _dense_mat_dot_masked_float_vec_jvp_spikes(spk_dot, weights, spikes, **kwargs):
    return [weights @ spk_dot]


def _dense_mat_dot_masked_float_vec_transpose_rule(ct, weights, spikes, **kwargs):
    ct = ct[0]
    if ad.is_undefined_primal(spikes):
        ct_events = jnp.matmul(ct, weights)
        return weights, (ad.Zero(spikes) if type(ct[0]) is ad.Zero else ct_events)
    else:
        ct_weights = jnp.outer(ct, spikes)
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
dense_mat_dot_masked_float_vec_p.def_gpu_kernel(warp=_dense_mat_dot_masked_float_vec_warp_kernel_generator,
                                                pallas=_dense_mat_dot_masked_float_vec_pallas_kernel_generator,
                                                default='warp')
dense_mat_dot_masked_float_vec_p.def_tpu_kernel(_dense_mat_dot_masked_float_vec_pallas_kernel_generator)
dense_mat_dot_masked_float_vec_p.def_jvp_rule2(_dense_mat_dot_masked_float_vec_jvp_weights,
                                               _dense_mat_dot_masked_float_vec_jvp_spikes)
dense_mat_dot_masked_float_vec_p.def_transpose_rule(_dense_mat_dot_masked_float_vec_transpose_rule)
dense_mat_dot_masked_float_vec_p.def_batching_rule(_dense_mat_dot_masked_float_vec_batching)


def masked_float_vec_dot_dense_mat(spikes, weights):
    """Performs event-driven vector-matrix multiplication: `spikes @ weights`.

    This function computes the vector-matrix product of a spike vector and a
    weight matrix, where the spike vector typically represents binary events
    (e.g., neural spikes). It handles units attached to input arrays using the
    `brainunit` library and dispatches computation to specialized kernels via
    `masked_float_vec_dot_dense_mat_p_call`.

    Parameters
    ----------
    spikes : array_like
        The spike vector with shape (K,). Can be boolean or float.
        If boolean, True indicates an event. If float, non-zero values indicate events.
        Can be a `brainunit` quantity.
    weights : array_like
        The weight matrix, typically with shape (K, N). Can be a `brainunit` quantity.

    Returns
    -------
    array_like
        The result of the event-driven vector-matrix multiplication, with shape (N,).
        If inputs had units, the output will have appropriate units
        (product of spikes unit and weights unit).

    Notes
    -----
    The computation is optimized for sparse activations in the spike vector.
    For boolean spikes, only the rows corresponding to True values contribute
    to the output. For float spikes, only non-zero values contribute.
    """
    with jax.ensure_compile_time_eval():
        weights = u.math.asarray(weights)
        spikes = u.math.asarray(spikes)
    weight_val, wunit = u.split_mantissa_unit(weights)
    spk_val, spkunit = u.split_mantissa_unit(spikes)
    r = masked_float_vec_dot_dense_mat_p_call(spk_val, weight_val)
    return u.maybe_decimal(r[0] * wunit * spkunit)


def _masked_float_vec_dot_dense_mat_numba_kernel_generator(**kwargs):
    def _kernel(spikes, weights, posts):
        posts[:] = 0.
        for i in range(spikes.shape[0]):
            spk = spikes[i]
            if spk != 0.:
                posts += weights[i] * spk

    return numba_kernel(_kernel)


def _masked_float_vec_dot_dense_mat_warp_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp  # pylint: disable=import-outside-toplevel

    SPIKE_LEGNTH = spk_info.shape[0]
    block_dim = generate_block_dim(weight_info.shape[1], maximum=512)

    spike_dtype = jaxtype_to_warptype(spk_info.dtype)
    weight_dtype = jaxtype_to_warptype(weight_info.dtype)

    def kernel(
        spike_ref: warp.array1d(dtype=spike_dtype),
        weight_ref: warp.array2d(dtype=weight_dtype),
        out_ref: warp.array1d(dtype=weight_dtype),
    ):
        i_col_block = warp.tid()
        spikes = warp.tile_load(spike_ref, shape=(SPIKE_LEGNTH,))
        temp = warp.tile_zeros(shape=(block_dim,), dtype=weight_dtype)
        for j in range(SPIKE_LEGNTH):
            spk = spikes[j]
            if spk != 0.:
                temp += warp.tile_load(weight_ref[j], shape=(block_dim,), offset=(i_col_block * block_dim,)) * spk
        warp.tile_store(out_ref, temp, offset=(i_col_block * block_dim,))

    return warp_kernel(
        kernel,
        tile=cdiv(weight_info.shape[1], block_dim),
        block_dim=block_dim,
    )


def _masked_float_vec_dot_dense_mat_pallas_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    **kwargs
):
    block_dim = generate_block_dim(weight_info.shape[1], maximum=1024)

    def kernel(spike_ref, weight_ref, out_ref):
        i_col_block = pl.program_id(0)
        i_col_start = i_col_block * block_dim
        i_col_mask = i_col_start + jnp.arange(block_dim) < weight_ref.shape[1]

        def loop_fn(i_spike, temp):
            spike = spike_ref[i_spike]
            return jax.lax.cond(
                spike != 0.,
                lambda out: out + pl.load(
                    weight_ref,
                    (i_spike, pl.dslice(i_col_start, block_dim)),
                    mask=i_col_mask,
                    other=0.0
                ) * spike,
                lambda out: out,
                temp,
            )

        i_col_out = jax.lax.fori_loop(
            0,
            spike_ref.shape[0],
            loop_fn,
            jnp.zeros((block_dim,), dtype=weight_ref.dtype)
        )
        pl.store(out_ref, pl.dslice(i_col_start, block_dim), i_col_out, mask=i_col_mask)

    return pallas_kernel(
        kernel,
        tile=[cdiv(weight_info.shape[1], block_dim)],
        outs=kwargs['outs'],
    )


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
masked_float_vec_dot_dense_mat_p.def_gpu_kernel(warp=_masked_float_vec_dot_dense_mat_warp_kernel_generator,
                                                pallas=_masked_float_vec_dot_dense_mat_pallas_kernel_generator,
                                                default='warp')
masked_float_vec_dot_dense_mat_p.def_tpu_kernel(_masked_float_vec_dot_dense_mat_pallas_kernel_generator)
masked_float_vec_dot_dense_mat_p.def_jvp_rule2(_masked_float_vec_dot_dense_mat_jvp_spikes,
                                               _masked_float_vec_dot_dense_mat_jvp_weights)
masked_float_vec_dot_dense_mat_p.def_transpose_rule(_masked_float_vec_dot_dense_mat_transpose_rule)
masked_float_vec_dot_dense_mat_p.def_batching_rule(_event_matrix_batching)


def dense_mat_dot_masked_float_mat(weights, spikes):
    """
    Performs event-driven matrix-matrix multiplication: `weights @ spikes`.

    This function computes the product of a dense weight matrix and a binary
    matrix, where the binary matrix typically represents events (e.g., neural spikes).
    It handles potential units associated with the input arrays using the
    `brainunit` library. The actual computation is dispatched to specialized
    CPU/GPU kernels via `dense_mat_dot_masked_float_mat_p_call`.

    Parameters
    ----------
    weights : array_like
        The weight matrix, typically with shape (M, K). Can be a `brainunit`
        quantity.
    spikes : array_like
        The binary matrix, typically with shape (K, N). Can be boolean or float.
        If boolean, True indicates an event. If float, non-zero values
        indicate an event. Can be a `brainunit` quantity.

    Returns
    -------
    array_like
        The result of the event-driven matrix-matrix multiplication, with shape (M, N).
        If inputs had units, the output will also have appropriate units
        (product of weights unit and spikes unit).

    Notes
    -----
    The core computation performed is equivalent to:

    `output[m, n] = sum_{k} weights[m, k] * f(spikes[k, n])`

    where the function `f(s)` is defined as:
    - If `spikes` is boolean: `f(s) = 1` if `s` is True, `0` otherwise.
    - If `spikes` is float: `f(s) = 1` if `s != 0`, `0` otherwise.

    The function ensures inputs are JAX arrays and handles unit consistency
    using `brainunit`. The computation is delegated to a JAX primitive
    `dense_mat_dot_masked_float_mat_p` for potential hardware acceleration.
    """
    with jax.ensure_compile_time_eval():
        weights = u.math.asarray(weights)
        spikes = u.math.asarray(spikes)
    weight_val, wunit = u.split_mantissa_unit(weights)
    spk_val, spkunit = u.split_mantissa_unit(spikes)
    # Call the underlying primitive with unitless values
    r = dense_mat_dot_masked_float_mat_p_call(weight_val, spk_val)
    # Re-attach units to the result
    return u.maybe_decimal(r[0] * wunit * spkunit)


def _dense_mat_dot_masked_float_mat_numba_kernel_generator(**kwargs):
    # weights: [m, k]
    # spikes: [k, n]

    import numba

    def _kernel(weights, spikes, posts):
        for i_n in numba.prange(spikes.shape[1]):
            out = np.zeros(weights.shape[0], dtype=weights.dtype)
            for i_k in range(spikes.shape[0]):
                spk = spikes[i_k, i_n]
                if spk != 0.:
                    out += weights[:, i_k] * spk
            posts[:, i_n] = out

    return numba_kernel(_kernel, parallel=True)


def _dense_mat_dot_masked_float_mat_warp_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
):
    # weights: [m, k]
    # spikes: [k, n]

    import warp  # pylint: disable=import-outside-toplevel

    k = spk_info.shape[0]
    n = spk_info.shape[1]
    m = weight_info.shape[0]
    out_tile_size = m
    block_dim = generate_block_dim(m, maximum=1024)

    spike_dtype = jaxtype_to_warptype(spk_info.dtype)
    weight_dtype = jaxtype_to_warptype(weight_info.dtype)

    def kernel(
        weight_ref: warp.array2d(dtype=weight_dtype),
        spike_ref: warp.array2d(dtype=spike_dtype),
        out_ref: warp.array2d(dtype=weight_dtype)
    ):
        i_n = warp.tid()
        out = warp.tile_zeros(shape=(out_tile_size, 1), dtype=weight_dtype)
        spike = warp.tile_load(spike_ref, shape=(k, 1), offset=(0, i_n))
        spike = warp.tile_squeeze(spike)
        for i_k in range(k):
            spk = spike[i_k]
            if spk != 0.:
                out += warp.tile_load(weight_ref, shape=(out_tile_size, 1), offset=(0, i_k)) * spk
        warp.tile_store(out_ref, out, offset=(0, i_n))

    return warp_kernel(kernel, tile=n, block_dim=block_dim)


def _dense_mat_dot_masked_float_mat_pallas_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
):
    # weights: [m, k]
    # spikes: [k, n]

    k = spk_info.shape[0]
    n = spk_info.shape[1]
    m = weight_info.shape[0]
    block_dim = generate_block_dim(m, maximum=1024)

    def kernel(
        weight_ref,  # [m, k]
        spike_ref,  # [k, n]
        out_ref,  # [m, n]
    ):
        i_n = pl.program_id(0)
        i_m_block = pl.program_id(1)
        i_m_start = i_m_block * block_dim
        i_m_mask = i_m_start + jnp.arange(block_dim) < m

        def loop_fn(i_k, temp):
            spike = spike_ref[i_k, i_n]
            return jax.lax.cond(
                spike != 0.,
                lambda out: out + pl.load(
                    weight_ref,
                    (pl.dslice(i_m_start, block_dim), i_k),
                    mask=i_m_mask,
                    other=0.0
                ) * spike,
                lambda out: out,
                temp,
            )

        final_out = jax.lax.fori_loop(0, k, loop_fn, jnp.zeros(block_dim, dtype=weight_ref.dtype))
        pl.store(out_ref, (pl.dslice(i_m_start, block_dim), i_n), final_out, mask=i_m_mask)

    return pallas_kernel(kernel, tile=(n, cdiv(m, block_dim)), outs=kwargs['outs'])


def _dense_mat_dot_masked_float_mat_jvp_weights(w_dot, weights, spikes, **kwargs):
    return dense_mat_dot_masked_float_mat_p_call(w_dot, spikes)


def _dense_mat_dot_masked_float_mat_jvp_spikes(spk_dot, weights, spikes, **kwargs):
    return [weights @ spk_dot]


def _dense_mat_dot_masked_float_mat_transpose_rule(ct, weights, spikes, **kwargs):
    if ad.is_undefined_primal(spikes):
        ct_events = weights.T @ ct[0]
        return weights, (ad.Zero(spikes) if type(ct[0]) is ad.Zero else ct_events)
    else:
        ct_weights = dense_mat_dot_masked_float_mat(ct[0], spikes.T)
        return (ad.Zero(weights) if type(ct[0]) is ad.Zero else ct_weights), spikes


def _dense_mat_dot_masked_float_mat_batching_events_fn(args, axis=1, **kwargs):
    assert args[0].ndim == 2, 'requires 2D input for weights'
    assert args[1].ndim == 3, 'requires 3D input for events'
    assert axis > 0, 'axis must be greater than 0'
    k, maybe_batch1, maybe_batch2 = args[1].shape
    events = args[1].reshape(k, maybe_batch1 * maybe_batch2)
    r = dense_mat_dot_masked_float_mat_p_call(args[0], events)
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _dense_mat_dot_masked_float_mat_batching_weight_fn(args, axis=0, **kwargs):
    assert args[0].ndim == 3, 'requires 3D input for weights'
    assert args[1].ndim == 2, 'requires 2D input for events'
    assert axis < 2, 'axis must be less than 2'
    maybe_batch1, maybe_batch2, k = args[1].shape
    weights = args[0].reshape(maybe_batch1 * maybe_batch2, k)
    r = dense_mat_dot_masked_float_mat_p_call(weights, args[1])
    r = jnp.reshape(r[0], [maybe_batch1, maybe_batch2, r[0].shape[-1]])
    return [r], [axis]


def _dense_mat_dot_masked_float_mat_batching(args, axes, **kwargs):
    if axes == (None, 0):
        args = list(args)
        args[1] = jnp.transpose(args[1], (1, 0, 2))
        return _dense_mat_dot_masked_float_mat_batching_events_fn(args, axis=1, **kwargs)
    elif axes == (None, 1):
        return _dense_mat_dot_masked_float_mat_batching_events_fn(args, axis=1, **kwargs)
    elif axes == (None, 2):
        return _dense_mat_dot_masked_float_mat_batching_events_fn(args, axs=2, **kwargs)

    elif axes == (0, None):
        return _dense_mat_dot_masked_float_mat_batching_weight_fn(args, axis=0, **kwargs)
    elif axes == (1, None):
        return _dense_mat_dot_masked_float_mat_batching_weight_fn(args, axis=1, **kwargs)
    elif axes == (2, None):
        args = list(args)
        args[0] = jnp.transpose(args[0], (0, 2, 1))
        return _dense_mat_dot_masked_float_mat_batching_weight_fn(args, axis=1, **kwargs)

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
    )


dense_mat_dot_masked_float_mat_p = XLACustomKernel('dense_matrix_dot_masked_float_matrix')
dense_mat_dot_masked_float_mat_p.def_cpu_kernel(_dense_mat_dot_masked_float_mat_numba_kernel_generator)
dense_mat_dot_masked_float_mat_p.def_gpu_kernel(warp=_dense_mat_dot_masked_float_mat_warp_kernel_generator,
                                                pallas=_dense_mat_dot_masked_float_mat_pallas_kernel_generator,
                                                default='warp')
dense_mat_dot_masked_float_mat_p.def_tpu_kernel(_dense_mat_dot_masked_float_mat_pallas_kernel_generator)
dense_mat_dot_masked_float_mat_p.def_jvp_rule2(_dense_mat_dot_masked_float_mat_jvp_weights,
                                               _dense_mat_dot_masked_float_mat_jvp_spikes)
dense_mat_dot_masked_float_mat_p.def_transpose_rule(_dense_mat_dot_masked_float_mat_transpose_rule)
dense_mat_dot_masked_float_mat_p.def_batching_rule(_dense_mat_dot_masked_float_mat_batching)


def masked_float_mat_dot_dense_mat(spikes, weights):
    """
    Performs event-driven binary matrix - dense matrix multiplication: `spikes @ weights`.

    This function computes the product of a binary matrix and a dense matrix,
    where the binary matrix typically represents events (e.g., neural spikes).
    It handles potential units associated with the input arrays using the
    `brainunit` library. The actual computation is dispatched to specialized
    CPU/GPU kernels via `masked_float_mat_dot_dense_mat_p_call`.

    Parameters
    ----------
    spikes : array_like
        The binary matrix, typically with shape (M, K). Can be boolean or float.
        If boolean, True indicates an event. If float, non-zero values
        indicate an event. Can be a `brainunit` quantity.
    weights : array_like
        The dense weight matrix, typically with shape (K, N). Can be a `brainunit` quantity.

    Returns
    -------
    array_like
        The result of the event-driven matrix-matrix multiplication, with shape (M, N).
        If inputs had units, the output will also have appropriate units
        (product of spikes unit and weights unit).
    """
    with jax.ensure_compile_time_eval():
        # Ensure inputs are JAX arrays, potentially handling brainunit quantities
        # Convert the input weights to a JAX array, which may include handling units from brainunit
        weights = u.math.asarray(weights)
        # Convert the input spikes to a JAX array, which may include handling units from brainunit
        spikes = u.math.asarray(spikes)
    # Separate numerical values and units
    # Split the weights into its numerical value and unit components
    weight_val, wunit = u.split_mantissa_unit(weights)
    # Split the spikes into its numerical value and unit components
    spk_val, spkunit = u.split_mantissa_unit(spikes)
    # Call the underlying primitive with unitless values
    # Perform the actual matrix multiplication using the unitless values
    r = masked_float_mat_dot_dense_mat_p_call(spk_val, weight_val)
    # Re-attach units to the result, handling potential Decimal types
    # Multiply the result by the units of spikes and weights, and handle Decimal types if necessary
    return u.maybe_decimal(r[0] * spkunit * wunit)


def _masked_float_mat_dot_dense_mat_numba_kernel_generator(
    **kwargs
):
    # spikes: [m, k]
    # weights: [k, n]

    import numba

    def _kernel(spikes, weights, posts):
        for i_m in numba.prange(spikes.shape[0]):
            out = np.zeros(weights.shape[1], dtype=posts.dtype)
            for i_k in range(spikes.shape[0]):
                spk = spikes[i_m, i_k]
                if spk != 0.:
                    out += weights[i_k] * spk
            posts[i_m] = out

    return numba_kernel(_kernel, parallel=True)


def _masked_float_mat_dot_dense_mat_warp_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp  # pylint: disable=import-outside-toplevel

    m, k = spk_info.shape
    n = weight_info.shape[1]
    out_tile_size = n
    block_dim = generate_block_dim(n, maximum=1024)

    spike_dtype = jaxtype_to_warptype(spk_info.dtype)
    weight_dtype = jaxtype_to_warptype(weight_info.dtype)

    def kernel(
        spike_ref: warp.array2d(dtype=spike_dtype),
        weight_ref: warp.array2d(dtype=weight_dtype),
        out_ref: warp.array2d(dtype=weight_dtype),
    ):
        i_m = warp.tid()
        out = warp.tile_zeros(shape=(out_tile_size,), dtype=weight_dtype)
        spike = warp.tile_load(spike_ref[i_m], shape=(k,))
        for i_k in range(k):
            spk = spike[i_k]
            if spk != 0.:
                out += warp.tile_load(weight_ref[i_k], shape=(out_tile_size,)) * spk
        warp.tile_store(out_ref[i_m], out)

    return warp_kernel(kernel, tile=m, block_dim=block_dim)


def _masked_float_mat_dot_dense_mat_pallas_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
):
    # spikes: [m, k]
    # weights: [k, n]

    m = spk_info.shape[0]
    k, n = weight_info.shape
    block_dim = generate_block_dim(n, maximum=1024)

    def kernel(
        spike_ref,  # [m, k]
        weight_ref,  # [k, n]
        out_ref,  # [m, n]
    ):
        i_m = pl.program_id(0)
        i_n_block = pl.program_id(1)
        i_n_start = i_n_block * block_dim
        i_n_mask = i_n_start + jnp.arange(block_dim) < n

        def loop_fn(i_k, temp):
            spike = spike_ref[i_m, i_k]
            return jax.lax.cond(
                spike != 0.,
                lambda out: out + pl.load(
                    weight_ref,
                    (i_k, pl.dslice(i_n_start, block_dim)),
                    mask=i_n_mask,
                    other=0.0
                ) * spike,
                lambda out: out,
                temp,
            )

        final_out = jax.lax.fori_loop(0, k, loop_fn, jnp.zeros(block_dim, dtype=weight_ref.dtype))
        pl.store(out_ref, (i_m, pl.dslice(i_n_start, block_dim)), final_out, mask=i_n_mask)

    return pallas_kernel(kernel, tile=(m, cdiv(n, block_dim)), outs=kwargs['outs'])


def _masked_float_mat_dot_dense_mat_jvp_weights(w_dot, spikes, weights, **kwargs):
    return masked_float_mat_dot_dense_mat_p_call(spikes, w_dot)


def _masked_float_mat_dot_dense_mat_jvp_spikes(spk_dot, spikes, weights, **kwargs):
    return [spk_dot @ weights]


def _masked_float_mat_dot_dense_mat_transpose_rule(ct, spikes, weights, **kwargs):
    if ad.is_undefined_primal(spikes):
        ct_events = ct[0] @ weights.T
        return (ad.Zero(spikes) if type(ct[0]) is ad.Zero else ct_events), weights

    else:
        ct_weights = masked_float_mat_dot_dense_mat(spikes.T, ct[0])
        return spikes, (ad.Zero(weights) if type(ct[0]) is ad.Zero else ct_weights)


def _masked_float_mat_dot_dense_mat_batching_spk_base_fn(args, axis=0, **kwargs):
    assert args[0].ndim == 3, 'requires 3D events.'
    assert args[1].ndim == 2, 'requires 3D weights.'
    maybe_batch1, maybe_batch2, n = args[0].shape
    events = args[0].reshape(maybe_batch1 * maybe_batch2, n)
    r = masked_float_mat_dot_dense_mat_p_call(events, args[1])
    r = jnp.reshape(r[0], [maybe_batch1, maybe_batch2, r[0].shape[1]])
    return [r], [axis]


def _masked_float_mat_dot_dense_mat_batching_weight_base_fn(args, axis=0, **kwargs):
    assert args[0].ndim == 0, 'requires 2D events.'
    assert args[1].ndim == 3, 'requires 3D weights.'
    k, maybe_batch1, maybe_batch2 = args[1].shape
    events = args[0]
    weights = args[1].reshape(k, maybe_batch1 * maybe_batch2)
    r = masked_float_mat_dot_dense_mat_p_call(events, weights)
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _masked_float_mat_dot_dense_mat_batching(args, axes, **kwargs):
    if axes == (0, None):
        return _masked_float_mat_dot_dense_mat_batching_spk_base_fn(args, axis=0, **kwargs)
    elif axes == (1, None):
        return _masked_float_mat_dot_dense_mat_batching_spk_base_fn(args, axis=1, **kwargs)
    elif axes == (2, None):
        args = list(args)
        args[0] = jnp.transpose(args[0], (0, 2, 1))
        return _masked_float_mat_dot_dense_mat_batching_spk_base_fn(args, axis=1, **kwargs)

    elif axes == (None, 0):
        args = list(args)
        args[1] = jnp.transpose(args[0], (1, 0, 2))
        return _masked_float_mat_dot_dense_mat_batching_weight_base_fn(args, axis=1, **kwargs)
    elif axes == (None, 1):
        return _masked_float_mat_dot_dense_mat_batching_weight_base_fn(args, axis=1, **kwargs)
    elif axes == (None, 2):
        return _masked_float_mat_dot_dense_mat_batching_weight_base_fn(args, axis=2, **kwargs)

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
    )


masked_float_mat_dot_dense_mat_p = XLACustomKernel('masked_float_matrix_dot_dense_matrix')
masked_float_mat_dot_dense_mat_p.def_cpu_kernel(_masked_float_mat_dot_dense_mat_numba_kernel_generator)
masked_float_mat_dot_dense_mat_p.def_gpu_kernel(warp=_masked_float_mat_dot_dense_mat_warp_kernel_generator,
                                                pallas=_masked_float_mat_dot_dense_mat_pallas_kernel_generator,
                                                default='warp')
masked_float_mat_dot_dense_mat_p.def_tpu_kernel(_masked_float_mat_dot_dense_mat_pallas_kernel_generator)
masked_float_mat_dot_dense_mat_p.def_jvp_rule2(_masked_float_mat_dot_dense_mat_jvp_spikes,
                                               _masked_float_mat_dot_dense_mat_jvp_weights)
masked_float_mat_dot_dense_mat_p.def_transpose_rule(_masked_float_mat_dot_dense_mat_transpose_rule)
masked_float_mat_dot_dense_mat_p.def_batching_rule(_masked_float_mat_dot_dense_mat_batching)
