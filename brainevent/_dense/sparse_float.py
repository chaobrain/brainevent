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

from brainevent._misc import cdiv, generate_block_dim, namescoped_jit
from brainevent._op import XLACustomKernel, numba_kernel, jaxinfo_to_warpinfo, general_batching_rule

__all__ = [
    'dm_sfv',
    'dm_sfv_p',
    'sfv_dm',
    'sfv_dm_p',
    'dm_sfm',
    'dm_sfm_p',
]


@namescoped_jit()
def dm_sfv(weights, spikes):
    """
    Performs event-driven matrix-vector multiplication: `weights @ spikes`.

    This function computes the product of a dense weight matrix and a binary
    vector, where the binary vector often represents events (e.g., neural spikes).
    It handles potential units associated with the input arrays using the
    `brainunit` library. The actual computation is dispatched to specialized
    CPU/GPU kernels via `dmsfv_p_call`.

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
    `dm_sfv_p` for potential hardware acceleration.
    """
    with jax.ensure_compile_time_eval():
        weights = u.math.asarray(weights)
        spikes = u.math.asarray(spikes)
    weight_val, wunit = u.split_mantissa_unit(weights)
    spk_val, spkunit = u.split_mantissa_unit(spikes)
    r = dmsfv_p_call(weight_val, spk_val)
    return u.maybe_decimal(r[0] * wunit * spkunit)


def _dmsfv_numba_kernel(**kwargs):
    import numba

    @numba.njit(fastmath=True, cache=True)
    def kernel(weights, spikes, posts):
        posts[:] = 0.
        for i in range(spikes.shape[0]):
            spk = spikes[i]
            if spk != 0.:
                posts += weights[:, i] * spk

    def run(weights, spikes):
        return numba_kernel(kernel, outs=kwargs['outs'])(weights, spikes)

    return run


def _dmsfv_warp_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    spike_length = spk_info.shape[0]
    m = weight_info.shape[0]
    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    spike_warp_info = jaxinfo_to_warpinfo(spk_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    @warp.kernel
    def kernel(
        weight_ref: weight_warp_info,
        spike_ref: spike_warp_info,
        out_ref: out_warp_info,
    ):
        i_row = warp.tid()
        r = weight_ref.dtype(0.)
        for j in range(spike_length):
            spk = spike_ref[j]
            if spk != 0.:
                r += weight_ref[i_row, j] * spk
        out_ref[i_row] = r

    def run(weights, spikes):
        out_info = kwargs['outs'][0]
        fn = jax_kernel(kernel, launch_dims=[m], num_outputs=1, output_dims={'out_ref': out_info.shape})
        return fn(weights, spikes)

    return run


def _dmsfv_pallas_kernel(
    weight_info: jax.ShapeDtypeStruct,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    mat_block_dim = generate_block_dim(weight_info.shape[0], maximum=1024)

    def kernel(weight_ref, spike_ref, out_ref):
        i_row_block = pl.program_id(0)
        row_start = i_row_block * mat_block_dim
        rows = row_start + jnp.arange(mat_block_dim)
        mask = rows < weight_ref.shape[0]
        safe_rows = jnp.where(mask, rows, 0)

        def loop_fn(i_spike, temp):
            spike = spike_ref[i_spike]
            return jax.lax.cond(
                spike != 0.,
                lambda out: out + jnp.where(
                    mask,
                    plt.load(weight_ref[safe_rows, i_spike]) * spike,
                    0.0,
                ),
                lambda out: out,
                temp,
            )

        i_row_out = jax.lax.fori_loop(
            0,
            spike_ref.shape[0],
            loop_fn,
            jnp.zeros((mat_block_dim,), dtype=weight_ref.dtype)
        )
        plt.store(out_ref[safe_rows], i_row_out, mask=mask)

    def run(weights, spikes):
        fn = pl.pallas_call(
            kernel,
            grid=(cdiv(weight_info.shape[0], mat_block_dim),),
            out_shape=kwargs['outs'],
        )
        return fn(weights, spikes)

    return run


def _dmsfv_jvp_weights(w_dot, weights, spikes, **kwargs):
    return dmsfv_p_call(w_dot, spikes)


def _dmsfv_jvp_spikes(spk_dot, weights, spikes, **kwargs):
    return [weights @ spk_dot]


def _dmsfv_transpose_rule(ct, weights, spikes, **kwargs):
    ct = ct[0]
    if ad.is_undefined_primal(spikes):
        ct_events = jnp.matmul(ct, weights)
        return weights, (ad.Zero(spikes) if type(ct[0]) is ad.Zero else ct_events)
    else:
        ct_weights = jnp.outer(ct, spikes)
        return (ad.Zero(weights) if type(ct[0]) is ad.Zero else ct_weights), spikes


def _dmsfv_batching(args, axes, **kwargs):
    if axes == (None, 0):
        weights, spikes = args
        r = spikes @ weights.T
        return [r], [0]
    if axes == (None, 1):
        weights, spikes = args
        r = weights @ spikes
        return [r], [1]
    else:
        return general_batching_rule(dm_sfv_p, args, axes, **kwargs)


def dmsfv_p_call(weights, spikes):
    assert spikes.shape[0] == weights.shape[1], (
        f"spikes shape {spikes.shape} and weights shape {weights.shape} are not compatible"
    )
    out = jax.ShapeDtypeStruct([weights.shape[0]], weights.dtype)
    return dm_sfv_p(
        weights,
        spikes,
        outs=[out],
        spk_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
    )


dm_sfv_p = XLACustomKernel('dmsfv')
dm_sfv_p.def_numba_kernel(_dmsfv_numba_kernel)
dm_sfv_p.def_warp_kernel(_dmsfv_warp_kernel)
dm_sfv_p.def_pallas_kernel('gpu', _dmsfv_pallas_kernel)
dm_sfv_p.def_pallas_kernel('tpu', _dmsfv_pallas_kernel)
dm_sfv_p.def_jvp_rule2(_dmsfv_jvp_weights, _dmsfv_jvp_spikes)
dm_sfv_p.def_transpose_rule(_dmsfv_transpose_rule)
dm_sfv_p.def_batching_rule(_dmsfv_batching)
dm_sfv_p.def_call(dmsfv_p_call)


def sfv_dm(spikes, weights):
    """Performs event-driven vector-matrix multiplication: `spikes @ weights`.

    This function computes the vector-matrix product of a spike vector and a
    weight matrix, where the spike vector typically represents binary events
    (e.g., neural spikes). It handles units attached to input arrays using the
    `brainunit` library and dispatches computation to specialized kernels via
    `sfvdm_p_call`.

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
    r = sfvdm_p_call(spk_val, weight_val)
    return u.maybe_decimal(r[0] * wunit * spkunit)


def _sfvdm_numba_kernel(**kwargs):
    import numba

    @numba.njit(fastmath=True, cache=True)
    def kernel(spikes, weights, posts):
        posts[:] = 0.
        for i in range(spikes.shape[0]):
            spk = spikes[i]
            if spk != 0.:
                posts += weights[i] * spk

    def run(spikes, weights):
        return numba_kernel(kernel, outs=kwargs['outs'])(spikes, weights)

    return run


def _sfvdm_warp_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    spike_length = spk_info.shape[0]
    n = weight_info.shape[1]
    spike_warp_info = jaxinfo_to_warpinfo(spk_info)
    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    @warp.kernel
    def kernel(
        spike_ref: spike_warp_info,
        weight_ref: weight_warp_info,
        out_ref: out_warp_info,
    ):
        j = warp.tid()
        r = weight_ref.dtype(0.)
        for i in range(spike_length):
            spk = spike_ref[i]
            if spk != 0.:
                r += weight_ref[i, j] * spk
        out_ref[j] = r

    def run(spikes, weights):
        out_info = kwargs['outs'][0]
        fn = jax_kernel(kernel, launch_dims=[n], num_outputs=1, output_dims={'out_ref': out_info.shape})
        return fn(spikes, weights)

    return run


def _sfvdm_pallas_kernel(
    weight_info: jax.ShapeDtypeStruct,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    block_dim = generate_block_dim(weight_info.shape[1], maximum=1024)

    def kernel(spike_ref, weight_ref, out_ref):
        i_col_block = pl.program_id(0)
        col_start = i_col_block * block_dim
        cols = col_start + jnp.arange(block_dim)
        mask = cols < weight_ref.shape[1]
        safe_cols = jnp.where(mask, cols, 0)

        def loop_fn(i_spike, temp):
            spike = spike_ref[i_spike]
            return jax.lax.cond(
                spike != 0.,
                lambda out: out + jnp.where(
                    mask,
                    plt.load(weight_ref[i_spike, safe_cols]) * spike,
                    0.0,
                ),
                lambda out: out,
                temp,
            )

        i_col_out = jax.lax.fori_loop(
            0,
            spike_ref.shape[0],
            loop_fn,
            jnp.zeros((block_dim,), dtype=weight_ref.dtype)
        )
        plt.store(out_ref[safe_cols], i_col_out, mask=mask)

    def run(spikes, weights):
        fn = pl.pallas_call(
            kernel,
            grid=(cdiv(weight_info.shape[1], block_dim),),
            out_shape=kwargs['outs'],
        )
        return fn(spikes, weights)

    return run


def _sfvdm_jvp_weights(w_dot, spikes, weights, **kwargs):
    return sfvdm_p_call(spikes, w_dot)


def _sfvdm_jvp_spikes(spk_dot, spikes, weights, **kwargs):
    return [spk_dot @ weights]


def _sfvdm_transpose_rule(ct, spikes, weights, **kwargs):
    if ad.is_undefined_primal(spikes):
        ct_events = jnp.matmul(weights, ct[0])
        return (ad.Zero(spikes) if type(ct[0]) is ad.Zero else ct_events), weights

    else:
        ct_weights = jnp.outer(spikes, ct[0])
        return spikes, (ad.Zero(weights) if type(ct[0]) is ad.Zero else ct_weights)


def _event_matrix_batching(args, axes, **kwargs):
    if axes == (0, None):
        r = sparse_float_mat_dot_dense_mat(args[0], args[1])
        return [r], [0]
    if axes == (1, None):
        r = sparse_float_mat_dot_dense_mat(args[0].T, args[1])
        return [r], [0]
    else:
        return general_batching_rule(sfv_dm_p, args, axes, **kwargs)


def sfvdm_p_call(spikes, weights):
    assert spikes.shape[0] == weights.shape[0], (
        f"shapes {spikes.shape} and {weights.shape} not aligned: "
        f"{spikes.shape[0]} (dim 0) != {weights.shape[0]} (dim 0)"
    )
    out = jax.ShapeDtypeStruct([weights.shape[1]], weights.dtype)
    return sfv_dm_p(
        spikes,
        weights,
        outs=[out],
        spk_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
    )


sfv_dm_p = XLACustomKernel('sparse_float_vector_dot_dense_matrix')
sfv_dm_p.def_numba_kernel(_sfvdm_numba_kernel)
sfv_dm_p.def_warp_kernel(_sfvdm_warp_kernel)
sfv_dm_p.def_pallas_kernel('gpu', _sfvdm_pallas_kernel)
sfv_dm_p.def_pallas_kernel('tpu', _sfvdm_pallas_kernel)
sfv_dm_p.def_jvp_rule2(_sfvdm_jvp_spikes,
                       _sfvdm_jvp_weights)
sfv_dm_p.def_transpose_rule(_sfvdm_transpose_rule)
sfv_dm_p.def_batching_rule(_event_matrix_batching)
sfv_dm_p.def_call(sfvdm_p_call)


def dm_sfm(weights, spikes):
    """
    Performs event-driven matrix-matrix multiplication: `weights @ spikes`.

    This function computes the product of a dense weight matrix and a binary
    matrix, where the binary matrix typically represents events (e.g., neural spikes).
    It handles potential units associated with the input arrays using the
    `brainunit` library. The actual computation is dispatched to specialized
    CPU/GPU kernels via `dmsfm_p_call`.

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
    `dm_sfm_p` for potential hardware acceleration.
    """
    with jax.ensure_compile_time_eval():
        weights = u.math.asarray(weights)
        spikes = u.math.asarray(spikes)
    weight_val, wunit = u.split_mantissa_unit(weights)
    spk_val, spkunit = u.split_mantissa_unit(spikes)
    # Call the underlying primitive with unitless values
    r = dmsfm_p_call(weight_val, spk_val)
    # Re-attach units to the result
    return u.maybe_decimal(r[0] * wunit * spkunit)


def _dmsfm_numba_kernel(**kwargs):
    # weights: [m, k]
    # spikes: [k, n]

    import numba

    @numba.njit(parallel=True, fastmath=True, nogil=True, cache=True)
    def kernel(weights, spikes, posts):
        for i_n in numba.prange(spikes.shape[1]):
            out = np.zeros(weights.shape[0], dtype=weights.dtype)
            for i_k in range(spikes.shape[0]):
                spk = spikes[i_k, i_n]
                if spk != 0.:
                    out += weights[:, i_k] * spk
            posts[:, i_n] = out

    def run(weights, spikes):
        return numba_kernel(kernel, outs=kwargs['outs'])(weights, spikes)

    return run


def _dmsfm_warp_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
):
    # weights: [m, k]
    # spikes: [k, n]

    import warp
    from warp.jax_experimental import jax_kernel

    k = spk_info.shape[0]
    n = spk_info.shape[1]
    m = weight_info.shape[0]

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    spike_warp_info = jaxinfo_to_warpinfo(spk_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    @warp.kernel
    def kernel(
        weight_ref: weight_warp_info,
        spike_ref: spike_warp_info,
        out_ref: out_warp_info,
    ):
        i_n = warp.tid()
        for i_m in range(m):
            r = weight_ref.dtype(0.)
            for i_k in range(k):
                spk = spike_ref[i_k, i_n]
                if spk != 0.:
                    r += weight_ref[i_m, i_k] * spk
            out_ref[i_m, i_n] = r

    def run(weights, spikes):
        out_info = kwargs['outs'][0]
        fn = jax_kernel(kernel, launch_dims=[n], num_outputs=1, output_dims={'out_ref': out_info.shape})
        return fn(weights, spikes)

    return run


def _dmsfm_pallas_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
):
    # weights: [m, k]
    # spikes: [k, n]

    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

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
        row_start = i_m_block * block_dim
        rows = row_start + jnp.arange(block_dim)
        mask = rows < m
        safe_rows = jnp.where(mask, rows, 0)

        def loop_fn(i_k, temp):
            spike = spike_ref[i_k, i_n]
            return jax.lax.cond(
                spike != 0.,
                lambda out: out + jnp.where(
                    mask,
                    plt.load(weight_ref[safe_rows, i_k]) * spike,
                    0.0,
                ),
                lambda out: out,
                temp,
            )

        final_out = jax.lax.fori_loop(0, k, loop_fn, jnp.zeros(block_dim, dtype=weight_ref.dtype))
        plt.store(out_ref[safe_rows, i_n], final_out, mask=mask)

    def run(weights, spikes):
        fn = pl.pallas_call(kernel, grid=(n, cdiv(m, block_dim)), out_shape=kwargs['outs'])
        return fn(weights, spikes)

    return run


def _dmsfm_jvp_weights(w_dot, weights, spikes, **kwargs):
    return dmsfm_p_call(w_dot, spikes)


def _dmsfm_jvp_spikes(spk_dot, weights, spikes, **kwargs):
    return [weights @ spk_dot]


def _dmsfm_transpose_rule(ct, weights, spikes, **kwargs):
    if ad.is_undefined_primal(spikes):
        ct_events = weights.T @ ct[0]
        return weights, (ad.Zero(spikes) if type(ct[0]) is ad.Zero else ct_events)
    else:
        ct_weights = dm_sfm(ct[0], spikes.T)
        return (ad.Zero(weights) if type(ct[0]) is ad.Zero else ct_weights), spikes


def _dmsfm_batching_events_fn(args, axis=1, **kwargs):
    assert args[0].ndim == 2, 'requires 2D input for weights'
    assert args[1].ndim == 3, 'requires 3D input for events'
    assert axis > 0, 'axis must be greater than 0'
    k, maybe_batch1, maybe_batch2 = args[1].shape
    events = args[1].reshape(k, maybe_batch1 * maybe_batch2)
    r = dmsfm_p_call(args[0], events)
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _dmsfm_batching_weight_fn(args, axis=0, **kwargs):
    assert args[0].ndim == 3, 'requires 3D input for weights'
    assert args[1].ndim == 2, 'requires 2D input for events'
    assert axis < 2, 'axis must be less than 2'
    maybe_batch1, maybe_batch2, k = args[1].shape
    weights = args[0].reshape(maybe_batch1 * maybe_batch2, k)
    r = dmsfm_p_call(weights, args[1])
    r = jnp.reshape(r[0], [maybe_batch1, maybe_batch2, r[0].shape[-1]])
    return [r], [axis]


def _dmsfm_batching(args, axes, **kwargs):
    if axes == (None, 0):
        args = list(args)
        args[1] = jnp.transpose(args[1], (1, 0, 2))
        return _dmsfm_batching_events_fn(args, axis=1, **kwargs)
    elif axes == (None, 1):
        return _dmsfm_batching_events_fn(args, axis=1, **kwargs)
    elif axes == (None, 2):
        return _dmsfm_batching_events_fn(args, axs=2, **kwargs)

    elif axes == (0, None):
        return _dmsfm_batching_weight_fn(args, axis=0, **kwargs)
    elif axes == (1, None):
        return _dmsfm_batching_weight_fn(args, axis=1, **kwargs)
    elif axes == (2, None):
        args = list(args)
        args[0] = jnp.transpose(args[0], (0, 2, 1))
        return _dmsfm_batching_weight_fn(args, axis=1, **kwargs)

    else:
        return general_batching_rule(dm_sfm_p, args, axes, **kwargs)


def dmsfm_p_call(weights, spikes):
    assert weights.shape[1] == spikes.shape[0], (
        f"weights.shape[1] ({weights.shape[1]}) != spikes.shape[0] ({spikes.shape[0]})"
        f", weights: {weights.shape}, spikes: {spikes.shape} in dmsfm_p_call"
    )
    out = jax.ShapeDtypeStruct([weights.shape[0], spikes.shape[1]], weights.dtype)
    return dm_sfm_p(
        weights,
        spikes,
        outs=[out],
        spk_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
    )


dm_sfm_p = XLACustomKernel('dense_matrix_dot_sparse_float_matrix')
dm_sfm_p.def_numba_kernel(_dmsfm_numba_kernel)
dm_sfm_p.def_warp_kernel(_dmsfm_warp_kernel)
dm_sfm_p.def_pallas_kernel('gpu', _dmsfm_pallas_kernel)
dm_sfm_p.def_pallas_kernel('tpu', _dmsfm_pallas_kernel)
dm_sfm_p.def_jvp_rule2(_dmsfm_jvp_weights,
                       _dmsfm_jvp_spikes)
dm_sfm_p.def_transpose_rule(_dmsfm_transpose_rule)
dm_sfm_p.def_batching_rule(_dmsfm_batching)
dm_sfm_p.def_call(dmsfm_p_call)


def sparse_float_mat_dot_dense_mat(spikes, weights):
    """
    Performs event-driven binary matrix - dense matrix multiplication: `spikes @ weights`.

    This function computes the product of a binary matrix and a dense matrix,
    where the binary matrix typically represents events (e.g., neural spikes).
    It handles potential units associated with the input arrays using the
    `brainunit` library. The actual computation is dispatched to specialized
    CPU/GPU kernels via `sparse_float_mat_dot_dense_mat_p_call`.

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
    r = sparse_float_mat_dot_dense_mat_p_call(spk_val, weight_val)
    # Re-attach units to the result, handling potential Decimal types
    # Multiply the result by the units of spikes and weights, and handle Decimal types if necessary
    return u.maybe_decimal(r[0] * spkunit * wunit)


def _sparse_float_mat_dot_dense_mat_numba_kernel(**kwargs):
    # spikes: [m, k]
    # weights: [k, n]

    import numba

    @numba.njit(parallel=True, fastmath=True, nogil=True, cache=True)
    def kernel(spikes, weights, posts):
        for i_m in numba.prange(spikes.shape[0]):
            out = np.zeros(weights.shape[1], dtype=posts.dtype)
            for i_n in range(weights.shape[1]):
                r = 0.0
                for i_k in range(spikes.shape[1]):
                    spk = spikes[i_m, i_k]
                    if spk != 0.:
                        r += weights[i_k, i_n] * spk
                out[i_n] = r
            posts[i_m] = out

    def run(spikes, weights):
        return numba_kernel(kernel, outs=kwargs['outs'])(spikes, weights)

    return run


def _sparse_float_mat_dot_dense_mat_warp_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    m, k = spk_info.shape
    n = weight_info.shape[1]

    spike_warp_info = jaxinfo_to_warpinfo(spk_info)
    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    @warp.kernel
    def kernel(
        spike_ref: spike_warp_info,
        weight_ref: weight_warp_info,
        out_ref: out_warp_info,
    ):
        i_m = warp.tid()
        for i_n in range(n):
            r = weight_ref.dtype(0.)
            for i_k in range(k):
                spk = spike_ref[i_m, i_k]
                if spk != 0.:
                    r += weight_ref[i_k, i_n] * spk
            out_ref[i_m, i_n] = r

    def run(spikes, weights):
        out_info = kwargs['outs'][0]
        fn = jax_kernel(kernel, launch_dims=[m], num_outputs=1, output_dims={'out_ref': out_info.shape})
        return fn(spikes, weights)

    return run


def _sparse_float_mat_dot_dense_mat_pallas_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
):
    # spikes: [m, k]
    # weights: [k, n]

    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

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
        col_start = i_n_block * block_dim
        cols = col_start + jnp.arange(block_dim)
        mask = cols < n
        safe_cols = jnp.where(mask, cols, 0)

        def loop_fn(i_k, temp):
            spike = spike_ref[i_m, i_k]
            return jax.lax.cond(
                spike != 0.,
                lambda out: out + jnp.where(
                    mask,
                    plt.load(weight_ref[i_k, safe_cols]) * spike,
                    0.0,
                ),
                lambda out: out,
                temp,
            )

        final_out = jax.lax.fori_loop(0, k, loop_fn, jnp.zeros(block_dim, dtype=weight_ref.dtype))
        plt.store(out_ref[i_m, safe_cols], final_out, mask=mask)

    def run(spikes, weights):
        fn = pl.pallas_call(kernel, grid=(m, cdiv(n, block_dim)), out_shape=kwargs['outs'])
        return fn(spikes, weights)

    return run


def _sparse_float_mat_dot_dense_mat_jvp_weights(w_dot, spikes, weights, **kwargs):
    return sparse_float_mat_dot_dense_mat_p_call(spikes, w_dot)


def _sparse_float_mat_dot_dense_mat_jvp_spikes(spk_dot, spikes, weights, **kwargs):
    return [spk_dot @ weights]


def _sparse_float_mat_dot_dense_mat_transpose_rule(ct, spikes, weights, **kwargs):
    if ad.is_undefined_primal(spikes):
        ct_events = ct[0] @ weights.T
        return (ad.Zero(spikes) if type(ct[0]) is ad.Zero else ct_events), weights

    else:
        ct_weights = sparse_float_mat_dot_dense_mat(spikes.T, ct[0])
        return spikes, (ad.Zero(weights) if type(ct[0]) is ad.Zero else ct_weights)


def _sparse_float_mat_dot_dense_mat_batching_spk_base_fn(args, axis=0, **kwargs):
    assert args[0].ndim == 3, 'requires 3D events.'
    assert args[1].ndim == 2, 'requires 3D weights.'
    maybe_batch1, maybe_batch2, n = args[0].shape
    events = args[0].reshape(maybe_batch1 * maybe_batch2, n)
    r = sparse_float_mat_dot_dense_mat_p_call(events, args[1])
    r = jnp.reshape(r[0], [maybe_batch1, maybe_batch2, r[0].shape[1]])
    return [r], [axis]


def _sparse_float_mat_dot_dense_mat_batching_weight_base_fn(args, axis=0, **kwargs):
    assert args[0].ndim == 0, 'requires 2D events.'
    assert args[1].ndim == 3, 'requires 3D weights.'
    k, maybe_batch1, maybe_batch2 = args[1].shape
    events = args[0]
    weights = args[1].reshape(k, maybe_batch1 * maybe_batch2)
    r = sparse_float_mat_dot_dense_mat_p_call(events, weights)
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _sparse_float_mat_dot_dense_mat_batching(args, axes, **kwargs):
    if axes == (0, None):
        return _sparse_float_mat_dot_dense_mat_batching_spk_base_fn(args, axis=0, **kwargs)
    elif axes == (1, None):
        return _sparse_float_mat_dot_dense_mat_batching_spk_base_fn(args, axis=1, **kwargs)
    elif axes == (2, None):
        args = list(args)
        args[0] = jnp.transpose(args[0], (0, 2, 1))
        return _sparse_float_mat_dot_dense_mat_batching_spk_base_fn(args, axis=1, **kwargs)

    elif axes == (None, 0):
        args = list(args)
        args[1] = jnp.transpose(args[0], (1, 0, 2))
        return _sparse_float_mat_dot_dense_mat_batching_weight_base_fn(args, axis=1, **kwargs)
    elif axes == (None, 1):
        return _sparse_float_mat_dot_dense_mat_batching_weight_base_fn(args, axis=1, **kwargs)
    elif axes == (None, 2):
        return _sparse_float_mat_dot_dense_mat_batching_weight_base_fn(args, axis=2, **kwargs)

    else:
        return general_batching_rule(sparse_float_mat_dot_dense_mat_p, args, axes, **kwargs)


def sparse_float_mat_dot_dense_mat_p_call(spikes, weights):
    assert spikes.shape[1] == weights.shape[0], (
        f"spikes shape {spikes.shape} and weights shape {weights.shape} do not match"
        f"for event matrix multiplication"
    )
    out = jax.ShapeDtypeStruct([spikes.shape[0], weights.shape[1]], weights.dtype)
    return sparse_float_mat_dot_dense_mat_p(
        spikes,
        weights,
        outs=[out],
        spk_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
    )


sparse_float_mat_dot_dense_mat_p = XLACustomKernel('sparse_float_matrix_dot_dense_matrix')
sparse_float_mat_dot_dense_mat_p.def_numba_kernel(_sparse_float_mat_dot_dense_mat_numba_kernel)
sparse_float_mat_dot_dense_mat_p.def_warp_kernel(_sparse_float_mat_dot_dense_mat_warp_kernel)
sparse_float_mat_dot_dense_mat_p.def_pallas_kernel('gpu', _sparse_float_mat_dot_dense_mat_pallas_kernel)
sparse_float_mat_dot_dense_mat_p.def_pallas_kernel('tpu', _sparse_float_mat_dot_dense_mat_pallas_kernel)
sparse_float_mat_dot_dense_mat_p.def_jvp_rule2(_sparse_float_mat_dot_dense_mat_jvp_spikes,
                                               _sparse_float_mat_dot_dense_mat_jvp_weights)
sparse_float_mat_dot_dense_mat_p.def_transpose_rule(_sparse_float_mat_dot_dense_mat_transpose_rule)
sparse_float_mat_dot_dense_mat_p.def_batching_rule(_sparse_float_mat_dot_dense_mat_batching)
sparse_float_mat_dot_dense_mat_p.def_call(sparse_float_mat_dot_dense_mat_p_call)
