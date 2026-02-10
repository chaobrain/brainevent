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

from typing import Optional

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._config import get_numba_parallel
from brainevent._misc import cdiv, generate_block_dim, namescope
from brainevent._op import jaxinfo_to_warpinfo, numba_kernel, XLACustomKernel, general_batching_rule
from brainevent._op.benchmark import BenchmarkConfig

__all__ = [
    'binary_densemv', 'binary_densemv_p', 'binary_densemv_p_call',
    'dbmm', 'dbmm_p',
    'bdmm', 'bdmm_p',
]


# ==============================================================================
# Unified binary dense matrix-vector product (binary_densemv)
# ==============================================================================
#
# transpose=False: weights[m,k] @ spikes[k] -> out[m]  (old dbmv)
# transpose=True:  spikes[k] @ weights[k,n] -> out[n]  (old bdvm)
#
# Argument order is always (weights, spikes).


@namescope(static_argnames=['transpose'])
def binary_densemv(weights, spikes, *, transpose, backend=None):
    """
    Performs event-driven dense matrix-vector multiplication.

    When ``transpose=False``, computes ``weights[m,k] @ spikes[k] -> out[m]``
    (dense matrix times binary vector).

    When ``transpose=True``, computes ``spikes[k] @ weights[k,n] -> out[n]``
    (binary vector times dense matrix).

    Parameters
    ----------
    weights : array_like
        The weight matrix. Shape ``(m, k)`` when ``transpose=False``,
        or ``(k, n)`` when ``transpose=True``. Can be a ``brainunit`` quantity.
    spikes : array_like
        The binary vector with shape ``(k,)``. Can be boolean or float.
        Can be a ``brainunit`` quantity.
    transpose : bool
        If False, compute ``weights @ spikes``. If True, compute ``spikes @ weights``.
    backend : str, optional
        Backend to use for the computation.

    Returns
    -------
    array_like
        Result vector. Shape ``(m,)`` when ``transpose=False``,
        or ``(n,)`` when ``transpose=True``.
    """
    with jax.ensure_compile_time_eval():
        weights = u.math.asarray(weights)
        spikes = u.math.asarray(spikes)
    weight_val, wunit = u.split_mantissa_unit(weights)
    spk_val, spkunit = u.split_mantissa_unit(spikes)
    r = binary_densemv_p_call(weight_val, spk_val, transpose=transpose, backend=backend)
    return u.maybe_decimal(r[0] * wunit * spkunit)


def _binary_densemv_numba_kernel(
    spk_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba

    if transpose:
        # weights[k,n], spikes[k] -> out[n]
        if spk_info.dtype == jnp.bool_:
            @numba.njit(fastmath=True)
            def kernel(weights, spikes, posts):
                posts[:] = 0.
                for i in range(spikes.shape[0]):
                    if spikes[i]:
                        posts += weights[i]
        else:
            @numba.njit(fastmath=True)
            def kernel(weights, spikes, posts):
                posts[:] = 0.
                for i in range(spikes.shape[0]):
                    if spikes[i] > 0.:
                        posts += weights[i]
    else:
        # weights[m,k], spikes[k] -> out[m]
        if spk_info.dtype == jnp.bool_:
            @numba.njit(fastmath=True)
            def kernel(weights, spikes, posts):
                posts[:] = 0.
                for i in range(spikes.shape[0]):
                    if spikes[i]:
                        posts += weights[:, i]
        else:
            @numba.njit(fastmath=True)
            def kernel(weights, spikes, posts):
                posts[:] = 0.
                for i in range(spikes.shape[0]):
                    if spikes[i] > 0.:
                        posts += weights[:, i]

    def run(weights, spikes):
        return numba_kernel(kernel, outs=kwargs['outs'])(weights, spikes)

    return run


def _binary_densemv_warp_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    spike_length = spk_info.shape[0]
    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    spike_warp_info = jaxinfo_to_warpinfo(spk_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if transpose:
        # weights[k,n], spikes[k] -> out[n]
        n = weight_info.shape[1]

        if spk_info.dtype == jnp.bool_:
            @warp.kernel
            def kernel(
                weights: weight_warp_info,
                spikes: spike_warp_info,
                out: out_warp_info,
            ):
                j = warp.tid()
                r = weights.dtype(0.)
                for i in range(spike_length):
                    if spikes[i]:
                        r += weights[i, j]
                out[j] = r
        else:
            @warp.kernel
            def kernel(
                weights: weight_warp_info,
                spikes: spike_warp_info,
                out: out_warp_info,
            ):
                j = warp.tid()
                r = weights.dtype(0.)
                for i in range(spike_length):
                    if spikes[i] > 0.:
                        r += weights[i, j]
                out[j] = r

        def run(weights, spikes):
            out_info = kwargs['outs'][0]
            fn = jax_kernel(kernel, launch_dims=[n], num_outputs=1, output_dims={'out': out_info.shape})
            return fn(weights, spikes)

    else:
        # weights[m,k], spikes[k] -> out[m]
        m = weight_info.shape[0]

        if spk_info.dtype == jnp.bool_:
            @warp.kernel
            def kernel(
                weights: weight_warp_info,
                spikes: spike_warp_info,
                out: out_warp_info,
            ):
                i = warp.tid()
                r = weights.dtype(0.)
                for j in range(spike_length):
                    if spikes[j]:
                        r += weights[i, j]
                out[i] = r
        else:
            @warp.kernel
            def kernel(
                weights: weight_warp_info,
                spikes: spike_warp_info,
                out: out_warp_info,
            ):
                i = warp.tid()
                r = weights.dtype(0.)
                for j in range(spike_length):
                    if spikes[j] > 0.:
                        r += weights[i, j]
                out[i] = r

        def run(weights, spikes):
            out_info = kwargs['outs'][0]
            fn = jax_kernel(kernel, launch_dims=[m], num_outputs=1, output_dims={'out': out_info.shape})
            return fn(weights, spikes)

    return run


def _binary_densemv_pallas_kernel(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    from jax.experimental import pallas as pl

    if transpose:
        # weights[k,n], spikes[k] -> out[n]
        n = weight_info.shape[1]
        block_dim = generate_block_dim(n, maximum=1024)

        def kernel(weight_ref, spike_ref, out_ref):
            i_col_block = pl.program_id(0)
            i_col_start = i_col_block * block_dim
            i_col_mask = i_col_start + jnp.arange(block_dim) < weight_ref.shape[1]

            def loop_fn(i_spike, temp):
                spike = spike_ref[i_spike]
                weight_row = weight_ref[i_spike, pl.ds(i_col_start, block_dim)]
                weight_row = jnp.where(i_col_mask, weight_row, 0.0)
                return jax.lax.cond(
                    spike if (spike_ref.dtype == jnp.bool_) else spike > 0.,
                    lambda out: out + weight_row,
                    lambda out: out,
                    temp,
                )

            i_col_out = jax.lax.fori_loop(
                0,
                spike_ref.shape[0],
                loop_fn,
                jnp.zeros((block_dim,), dtype=weight_ref.dtype)
            )
            out_ref[pl.ds(i_col_start, block_dim)] = jnp.where(i_col_mask, i_col_out, 0.0)

        def run(weights, spikes):
            fn = pl.pallas_call(kernel, grid=(cdiv(n, block_dim),), out_shape=kwargs['outs'])
            return fn(weights, spikes)

    else:
        # weights[m,k], spikes[k] -> out[m]
        m = weight_info.shape[0]
        mat_block_dim = generate_block_dim(m, maximum=1024)

        def kernel(weight_ref, spike_ref, out_ref):
            i_row_block = pl.program_id(0)
            i_row_start = i_row_block * mat_block_dim
            i_row_mask = i_row_start + jnp.arange(mat_block_dim) < weight_ref.shape[0]

            def loop_fn(i_spike, temp):
                spike = spike_ref[i_spike]
                weight_col = weight_ref[pl.ds(i_row_start, mat_block_dim), i_spike]
                weight_col = jnp.where(i_row_mask, weight_col, 0.0)
                return jax.lax.cond(
                    spike if spike_ref.dtype == jnp.bool_ else spike > 0.,
                    lambda out: out + weight_col,
                    lambda out: out,
                    temp
                )

            i_row_out = jax.lax.fori_loop(
                0,
                spike_ref.shape[0],
                loop_fn,
                jnp.zeros((mat_block_dim,), dtype=weight_ref.dtype)
            )
            out_ref[pl.ds(i_row_start, mat_block_dim)] = jnp.where(i_row_mask, i_row_out, 0.0)

        def run(weights, spikes):
            fn = pl.pallas_call(kernel, grid=(cdiv(m, mat_block_dim),), out_shape=kwargs['outs'])
            return fn(weights, spikes)

    return run


def _binary_densemv_jvp_weights(w_dot, weights, spikes, *, transpose, **kwargs):
    return binary_densemv_p_call(w_dot, spikes, transpose=transpose)


def _binary_densemv_jvp_spikes(spk_dot, weights, spikes, *, transpose, **kwargs):
    if transpose:
        return [spk_dot @ weights]
    else:
        return [weights @ spk_dot]


def _binary_densemv_transpose_rule(ct, weights, spikes, *, transpose, **kwargs):
    ct = ct[0]
    if ad.is_undefined_primal(spikes):
        if transpose:
            ct_spikes = jnp.matmul(weights, ct)
        else:
            ct_spikes = jnp.matmul(ct, weights)
        return weights, (ad.Zero(spikes) if type(ct) is ad.Zero else ct_spikes)
    else:
        if transpose:
            ct_weights = jnp.outer(spikes, ct)
        else:
            ct_weights = jnp.outer(ct, spikes)
        return (ad.Zero(weights) if type(ct) is ad.Zero else ct_weights), spikes


def _binary_densemv_batching(args, axes, *, transpose, **kwargs):
    if transpose:
        # weights[k,n], spikes[k] -> out[n]
        if axes == (None, 0):
            r = bdmm(args[1], args[0])
            return [r], [0]
        if axes == (None, 1):
            r = bdmm(args[1].T, args[0])
            return [r], [0]
    else:
        # weights[m,k], spikes[k] -> out[m]
        if axes == (None, 0):
            r = dbmm(args[0], args[1].T)
            return [r], [1]
        if axes == (None, 1):
            r = dbmm(args[0], args[1])
            return [r], [1]
    return general_batching_rule(binary_densemv_p, args, axes, transpose=transpose, **kwargs)


def _binary_densemv_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for bool_event in (True, False):
        # transpose=False benchmark (dbmv style)
        weights = jnp.asarray(np.random.randn(n_pre, n_post), dtype=dtype)
        if bool_event:
            spikes = jnp.asarray(np.random.rand(n_post) > (1 - prob), dtype=jnp.bool_)
        else:
            spikes = jnp.asarray(np.random.rand(n_post), dtype=dtype)
        name = f"{'bool' if bool_event else 'float'}"
        configs.append(BenchmarkConfig(name, (weights, spikes)))
    return configs


def binary_densemv_p_call(weights, spikes, *, transpose, backend=None):
    if transpose:
        assert spikes.shape[0] == weights.shape[0], (
            f"shapes {spikes.shape} and {weights.shape} not aligned: "
            f"{spikes.shape[0]} (dim 0) != {weights.shape[0]} (dim 0)"
        )
        out = jax.ShapeDtypeStruct([weights.shape[1]], weights.dtype)
    else:
        assert spikes.shape[0] == weights.shape[1], (
            f"spikes shape {spikes.shape} and weights shape {weights.shape} are not compatible"
        )
        out = jax.ShapeDtypeStruct([weights.shape[0]], weights.dtype)
    return binary_densemv_p(
        weights,
        spikes,
        outs=[out],
        transpose=transpose,
        spk_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        backend=backend,
    )


binary_densemv_p = XLACustomKernel('binary_densemv')
binary_densemv_p.def_numba_kernel(_binary_densemv_numba_kernel)
binary_densemv_p.def_warp_kernel(_binary_densemv_warp_kernel)
binary_densemv_p.def_pallas_kernel('gpu', _binary_densemv_pallas_kernel)
binary_densemv_p.def_pallas_kernel('tpu', _binary_densemv_pallas_kernel)
binary_densemv_p.def_jvp_rule2(_binary_densemv_jvp_weights, _binary_densemv_jvp_spikes)
binary_densemv_p.def_transpose_rule(_binary_densemv_transpose_rule)
binary_densemv_p.def_batching_rule(_binary_densemv_batching)
binary_densemv_p.def_call(binary_densemv_p_call)
binary_densemv_p.def_tags('dense', 'binary')
binary_densemv_p.def_benchmark_data(_binary_densemv_benchmark_data)

# ==============================================================================
# Dense matrix @ binary matrix (dbmm)
# ==============================================================================


@namescope
def dbmm(weights, spikes, *, backend=None):
    """
    Performs event-driven matrix-matrix multiplication: `weights @ spikes`.

    This function computes the product of a dense weight matrix and a binary
    matrix, where the binary matrix typically represents events (e.g., neural spikes).
    It handles potential units associated with the input arrays using the
    `brainunit` library. The actual computation is dispatched to specialized
    CPU/GPU kernels via `dense_mat_dot_binary_mat_p_call`.

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
    - If `spikes` is float: `f(s) = 1` if `s > 0`, `0` otherwise.

    The function ensures inputs are JAX arrays and handles unit consistency
    using `brainunit`. The computation is delegated to a JAX primitive
    `dbmm_p` for potential hardware acceleration.
    """
    with jax.ensure_compile_time_eval():
        weights = u.math.asarray(weights)
        spikes = u.math.asarray(spikes)
    weight_val, wunit = u.split_mantissa_unit(weights)
    spk_val, spkunit = u.split_mantissa_unit(spikes)
    # Call the underlying primitive with unitless values
    r = dbmm_p_call(weight_val, spk_val)
    # Re-attach units to the result
    return u.maybe_decimal(r[0] * wunit * spkunit)


def _dbmm_numba_kernel(
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
):
    # weights: [m, k]
    # spikes: [k, n]

    import numba

    if spk_info.dtype == jnp.bool_:
        @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
        def kernel(weights, spikes, posts):
            for i_n in numba.prange(spikes.shape[1]):
                out = np.zeros(weights.shape[0], dtype=weights.dtype)
                for i_k in range(spikes.shape[0]):
                    if spikes[i_k, i_n]:
                        out += weights[:, i_k]
                posts[:, i_n] = out

    else:
        @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
        def kernel(weights, spikes, posts):
            for i_n in numba.prange(spikes.shape[1]):
                out = np.zeros(weights.shape[0], dtype=weights.dtype)
                for i_k in range(spikes.shape[0]):
                    if spikes[i_k, i_n] > 0.:
                        out += weights[:, i_k]
                posts[:, i_n] = out

    def run(weights, spikes):
        return numba_kernel(kernel, outs=kwargs['outs'])(weights, spikes)

    return run


def _dbmm_warp_kernel(
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
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if spk_info.dtype == jnp.bool_:
        # Cast bool spikes to float32 to avoid 2D boolean array indexing bug in warp
        spk_float_info = jax.ShapeDtypeStruct(spk_info.shape, jnp.float32)
        spike_warp_info = jaxinfo_to_warpinfo(spk_float_info)

        @warp.kernel
        def kernel(
            weights: weight_warp_info,
            spikes: spike_warp_info,
            out: out_warp_info
        ):
            i_m, i_n = warp.tid()
            r = weights.dtype(0.)
            for i_k in range(k):
                if spikes[i_k, i_n] > 0.:
                    r += weights[i_m, i_k]
            out[i_m, i_n] = r

        def run(weights, spikes):
            spikes = spikes.astype(jnp.float32)
            out_info = kwargs['outs'][0]
            fn = jax_kernel(kernel, launch_dims=(m, n), num_outputs=1, output_dims={'out': out_info.shape})
            return fn(weights, spikes)

    else:
        spike_warp_info = jaxinfo_to_warpinfo(spk_info)

        @warp.kernel
        def kernel(
            weights: weight_warp_info,
            spikes: spike_warp_info,
            out: out_warp_info
        ):
            i_m, i_n = warp.tid()
            r = weights.dtype(0.)
            for i_k in range(k):
                if spikes[i_k, i_n] > 0.:
                    r += weights[i_m, i_k]
            out[i_m, i_n] = r

        def run(weights, spikes):
            out_info = kwargs['outs'][0]
            fn = jax_kernel(kernel, launch_dims=(m, n), num_outputs=1, output_dims={'out': out_info.shape})
            return fn(weights, spikes)

    return run


def _dbmm_pallas_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
):
    # weights: [m, k]
    # spikes: [k, n]
    from jax.experimental import pallas as pl

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
            weight_col = weight_ref[pl.ds(i_m_start, block_dim), i_k]
            weight_col = jnp.where(i_m_mask, weight_col, 0.0)
            return jax.lax.cond(
                spike if spike_ref.dtype == jnp.bool_ else spike > 0.,
                lambda out: out + weight_col,
                lambda out: out,
                temp,
            )

        final_out = jax.lax.fori_loop(0, k, loop_fn, jnp.zeros(block_dim, dtype=weight_ref.dtype))
        out_ref[pl.ds(i_m_start, block_dim), i_n] = jnp.where(i_m_mask, final_out, 0.0)

    def run(weights, spikes):
        fn = pl.pallas_call(kernel, grid=(n, cdiv(m, block_dim)), out_shape=kwargs['outs'])
        return fn(weights, spikes)

    return run


def _dbmm_jvp_weights(w_dot, weights, spikes, **kwargs):
    return dbmm_p_call(w_dot, spikes)


def _dbmm_jvp_spikes(spk_dot, weights, spikes, **kwargs):
    return [weights @ spk_dot]


def _dbmm_transpose_rule(ct, weights, spikes, **kwargs):
    ct = ct[0]
    if ad.is_undefined_primal(spikes):
        ct_events = weights.T @ ct
        return weights, (ad.Zero(spikes) if type(ct) is ad.Zero else ct_events)
    else:
        ct_weights = dbmm(ct, spikes.T)
        return (ad.Zero(weights) if type(ct) is ad.Zero else ct_weights), spikes


def _dbmm_batching_events_fn(args, axis=1, **kwargs):
    assert args[0].ndim == 2, 'requires 2D input for weights'
    assert args[1].ndim == 3, 'requires 3D input for events'
    assert axis > 0, 'axis must be greater than 0'
    k, maybe_batch1, maybe_batch2 = args[1].shape
    events = args[1].reshape(k, maybe_batch1 * maybe_batch2)
    r = dbmm_p_call(args[0], events)
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _dbmm_batching_weight_fn(args, axis=0, **kwargs):
    assert args[0].ndim == 3, 'requires 3D input for weights'
    assert args[1].ndim == 2, 'requires 2D input for events'
    assert axis < 2, 'axis must be less than 2'
    maybe_batch1, maybe_batch2, k = args[0].shape
    weights = args[0].reshape(maybe_batch1 * maybe_batch2, k)
    r = dbmm_p_call(weights, args[1])
    r = jnp.reshape(r[0], [maybe_batch1, maybe_batch2, r[0].shape[-1]])
    return [r], [axis]


def _dbmm_batching(args, axes, **kwargs):
    if axes == (None, 0):
        args = list(args)
        args[1] = jnp.transpose(args[1], (1, 0, 2))
        return _dbmm_batching_events_fn(args, axis=1, **kwargs)
    elif axes == (None, 1):
        return _dbmm_batching_events_fn(args, axis=1, **kwargs)
    elif axes == (None, 2):
        return _dbmm_batching_events_fn(args, axis=2, **kwargs)

    elif axes == (0, None):
        return _dbmm_batching_weight_fn(args, axis=0, **kwargs)
    elif axes == (1, None):
        return _dbmm_batching_weight_fn(args, axis=1, **kwargs)
    elif axes == (2, None):
        args = list(args)
        args[0] = jnp.transpose(args[0], (0, 2, 1))
        return _dbmm_batching_weight_fn(args, axis=1, **kwargs)

    else:
        return general_batching_rule(dbmm_p, args, axes, **kwargs)


def _dbmm_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for bool_event in (True, False):
        weights = jnp.asarray(np.random.randn(n_pre, n_post), dtype=dtype)
        if bool_event:
            spikes = jnp.asarray(np.random.rand(n_post, 10) > (1 - prob), dtype=jnp.bool_)
        else:
            spikes = jnp.asarray(np.random.rand(n_post, 10), dtype=dtype)
        name = f"{'bool' if bool_event else 'float'}"
        configs.append(BenchmarkConfig(name, (weights, spikes)))
    return configs


def dbmm_p_call(weights, spikes, *, backend: Optional[str] = None):
    assert weights.shape[1] == spikes.shape[0], (
        f"weights.shape[1] ({weights.shape[1]}) > spikes.shape[0] ({spikes.shape[0]})"
        f", weights: {weights.shape}, spikes: {spikes.shape} in dense_mat_dot_binary_mat_p_call"
    )
    out = jax.ShapeDtypeStruct([weights.shape[0], spikes.shape[1]], weights.dtype)
    return dbmm_p(
        weights,
        spikes,
        outs=[out],
        spk_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        backend=backend,
    )


dbmm_p = XLACustomKernel('dense_matrix_dot_binary_matrix')
dbmm_p.def_numba_kernel(_dbmm_numba_kernel)
dbmm_p.def_warp_kernel(_dbmm_warp_kernel)
dbmm_p.def_pallas_kernel('gpu', _dbmm_pallas_kernel)
dbmm_p.def_pallas_kernel('tpu', _dbmm_pallas_kernel)
dbmm_p.def_jvp_rule2(_dbmm_jvp_weights, _dbmm_jvp_spikes)
dbmm_p.def_transpose_rule(_dbmm_transpose_rule)
dbmm_p.def_batching_rule(_dbmm_batching)
dbmm_p.def_call(dbmm_p_call)
dbmm_p.def_tags('dense', 'binary')
dbmm_p.def_benchmark_data(_dbmm_benchmark_data)


# ==============================================================================
# Binary matrix @ dense matrix (bdmm)
# ==============================================================================


@namescope
def bdmm(spikes, weights, *, backend=None):
    """
    Performs event-driven binary matrix - dense matrix multiplication: `spikes @ weights`.

    This function computes the product of a binary matrix and a dense matrix,
    where the binary matrix typically represents events (e.g., neural spikes).
    It handles potential units associated with the input arrays using the
    `brainunit` library. The actual computation is dispatched to specialized
    CPU/GPU kernels via `binary_mat_dot_dense_mat_p_call`.

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
    r = bdmm_p_call(spk_val, weight_val)
    # Re-attach units to the result, handling potential Decimal types
    # Multiply the result by the units of spikes and weights, and handle Decimal types if necessary
    return u.maybe_decimal(r[0] * spkunit * wunit)


def _bdmm_numba_kernel(
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
):
    # spikes: [m, k]
    # weights: [k, n]

    import numba

    if spk_info.dtype == jnp.bool_:
        @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
        def kernel(spikes, weights, posts):
            for i_m in numba.prange(spikes.shape[0]):
                out = np.zeros(weights.shape[1], dtype=posts.dtype)
                for i_k in range(spikes.shape[1]):
                    if spikes[i_m, i_k]:
                        out += weights[i_k]
                posts[i_m] = out

    else:
        @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
        def kernel(spikes, weights, posts):
            for i_m in numba.prange(spikes.shape[0]):
                out = np.zeros(weights.shape[1], dtype=posts.dtype)
                for i_k in range(spikes.shape[1]):
                    if spikes[i_m, i_k] > 0.:
                        out += weights[i_k]
                posts[i_m] = out

    def run(spikes, weights):
        return numba_kernel(kernel, outs=kwargs['outs'])(spikes, weights)

    return run


def _bdmm_warp_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    m, k = spk_info.shape
    n = weight_info.shape[1]

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if spk_info.dtype == jnp.bool_:
        # Cast bool spikes to float32 to avoid 2D boolean array indexing bug in warp
        spk_float_info = jax.ShapeDtypeStruct(spk_info.shape, jnp.float32)
        spike_warp_info = jaxinfo_to_warpinfo(spk_float_info)

        @warp.kernel
        def kernel(
            spikes: spike_warp_info,
            weights: weight_warp_info,
            out: out_warp_info,
        ):
            i_m, i_n = warp.tid()
            r = weights.dtype(0.)
            for i_k in range(k):
                if spikes[i_m, i_k] > 0.:
                    r += weights[i_k, i_n]
            out[i_m, i_n] = r

        def run(spikes, weights):
            spikes = spikes.astype(jnp.float32)
            out_info = kwargs['outs'][0]
            fn = jax_kernel(kernel, launch_dims=(m, n), num_outputs=1, output_dims={'out': out_info.shape})
            return fn(spikes, weights)

    else:
        spike_warp_info = jaxinfo_to_warpinfo(spk_info)

        @warp.kernel
        def kernel(
            spikes: spike_warp_info,
            weights: weight_warp_info,
            out: out_warp_info,
        ):
            i_m, i_n = warp.tid()
            r = weights.dtype(0.)
            for i_k in range(k):
                if spikes[i_m, i_k] > 0.:
                    r += weights[i_k, i_n]
            out[i_m, i_n] = r

        def run(spikes, weights):
            out_info = kwargs['outs'][0]
            fn = jax_kernel(kernel, launch_dims=(m, n), num_outputs=1, output_dims={'out': out_info.shape})
            return fn(spikes, weights)

    return run


def _bdmm_pallas_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
):
    # spikes: [m, k]
    # weights: [k, n]
    from jax.experimental import pallas as pl

    m = spk_info.shape[0]
    k, n = weight_info.shape
    block_dim = generate_block_dim(n, maximum=1024)

    def kernel(
        spike_ref,  # [m, k]
        weight_t_ref,  # [n, k] (transposed)
        out_ref,  # [m, n]
    ):
        i_m = pl.program_id(0)
        i_n_block = pl.program_id(1)
        i_n_start = i_n_block * block_dim
        i_n_mask = i_n_start + jnp.arange(block_dim) < n

        def loop_fn(i_k, temp):
            spike = spike_ref[i_m, i_k]
            # Use block slice on first dim (working pattern on GPU triton)
            weight_row = weight_t_ref[pl.ds(i_n_start, block_dim), i_k]
            weight_row = jnp.where(i_n_mask, weight_row, 0.0)
            return jax.lax.cond(
                spike if spike_ref.dtype == jnp.bool_ else spike > 0.,
                lambda out: out + weight_row,
                lambda out: out,
                temp,
            )

        final_out = jax.lax.fori_loop(0, k, loop_fn, jnp.zeros(block_dim, dtype=weight_t_ref.dtype))
        out_ref[i_m, pl.ds(i_n_start, block_dim)] = jnp.where(i_n_mask, final_out, 0.0)

    def run(spikes, weights):
        weights_t = weights.T  # [k, n] -> [n, k]
        fn = pl.pallas_call(kernel, grid=(m, cdiv(n, block_dim)), out_shape=kwargs['outs'])
        return fn(spikes, weights_t)

    return run


def _bdmm_jvp_weights(w_dot, spikes, weights, **kwargs):
    return bdmm_p_call(spikes, w_dot)


def _bdmm_jvp_spikes(spk_dot, spikes, weights, **kwargs):
    return [spk_dot @ weights]


def _bdmm_transpose_rule(ct, spikes, weights, **kwargs):
    ct = ct[0]
    if ad.is_undefined_primal(spikes):
        ct_events = ct @ weights.T
        return (ad.Zero(spikes) if type(ct) is ad.Zero else ct_events), weights

    else:
        ct_weights = bdmm(spikes.T, ct)
        return spikes, (ad.Zero(weights) if type(ct) is ad.Zero else ct_weights)


def _bdmm_batching_spk_base_fn(args, axis=0, **kwargs):
    assert args[0].ndim == 3, 'requires 3D events.'
    assert args[1].ndim == 2, 'requires 3D weights.'
    maybe_batch1, maybe_batch2, n = args[0].shape
    events = args[0].reshape(maybe_batch1 * maybe_batch2, n)
    r = bdmm_p_call(events, args[1])
    r = jnp.reshape(r[0], [maybe_batch1, maybe_batch2, r[0].shape[1]])
    return [r], [axis]


def _bdmm_batching_weight_base_fn(args, axis=0, **kwargs):
    assert args[0].ndim == 2, 'requires 2D events.'
    assert args[1].ndim == 3, 'requires 3D weights.'
    k, maybe_batch1, maybe_batch2 = args[1].shape
    events = args[0]
    weights = args[1].reshape(k, maybe_batch1 * maybe_batch2)
    r = bdmm_p_call(events, weights)
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _bdmm_batching(args, axes, **kwargs):
    if axes == (0, None):
        return _bdmm_batching_spk_base_fn(args, axis=0, **kwargs)
    elif axes == (1, None):
        return _bdmm_batching_spk_base_fn(args, axis=1, **kwargs)
    elif axes == (2, None):
        args = list(args)
        args[0] = jnp.transpose(args[0], (0, 2, 1))
        return _bdmm_batching_spk_base_fn(args, axis=1, **kwargs)

    elif axes == (None, 0):
        args = list(args)
        args[1] = jnp.transpose(args[1], (1, 0, 2))
        return _bdmm_batching_weight_base_fn(args, axis=1, **kwargs)
    elif axes == (None, 1):
        return _bdmm_batching_weight_base_fn(args, axis=1, **kwargs)
    elif axes == (None, 2):
        return _bdmm_batching_weight_base_fn(args, axis=2, **kwargs)

    else:
        return general_batching_rule(bdmm_p, args, axes, **kwargs)


def _bdmm_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for bool_event in (True, False):
        if bool_event:
            spikes = jnp.asarray(np.random.rand(10, n_post) > (1 - prob), dtype=jnp.bool_)
        else:
            spikes = jnp.asarray(np.random.rand(10, n_post), dtype=dtype)
        weights = jnp.asarray(np.random.randn(n_post, n_post), dtype=dtype)
        name = f"{'bool' if bool_event else 'float'}"
        configs.append(BenchmarkConfig(name, (spikes, weights)))
    return configs


def bdmm_p_call(spikes, weights, *, backend: Optional[str] = None):
    assert spikes.shape[1] == weights.shape[0], (
        f"spikes shape {spikes.shape} and weights shape {weights.shape} do not match"
        f"for event matrix multiplication"
    )
    out = jax.ShapeDtypeStruct([spikes.shape[0], weights.shape[1]], weights.dtype)
    return bdmm_p(
        spikes,
        weights,
        outs=[out],
        spk_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        backend=backend,
    )


bdmm_p = XLACustomKernel('binary_matrix_dot_dense_matrix')
bdmm_p.def_numba_kernel(_bdmm_numba_kernel)
bdmm_p.def_warp_kernel(_bdmm_warp_kernel)
bdmm_p.def_pallas_kernel('gpu', _bdmm_pallas_kernel)
bdmm_p.def_pallas_kernel('tpu', _bdmm_pallas_kernel)
bdmm_p.def_jvp_rule2(_bdmm_jvp_spikes, _bdmm_jvp_weights)
bdmm_p.def_transpose_rule(_bdmm_transpose_rule)
bdmm_p.def_batching_rule(_bdmm_batching)
bdmm_p.def_call(bdmm_p_call)
bdmm_p.def_tags('dense', 'binary')
bdmm_p.def_benchmark_data(_bdmm_benchmark_data)
