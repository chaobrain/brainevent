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
    'binary_densemm', 'binary_densemm_p', 'binary_densemm_p_call',
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
        # mm transpose=True: weights[k,m].T @ spikes[k,n] -> out[m,n]
        # result shape from mm: [weights.shape[1], spikes.shape[1]]
        if axes == (None, 0):
            # spikes batched on axis 0: [batch, k] -> .T gives [k, batch]
            # mm(weights[k,n], [k,batch], transpose=True) -> [n, batch]
            r = binary_densemm(args[0], args[1].T, transpose=True)
            return [r], [1]
        if axes == (None, 1):
            # spikes batched on axis 1: [k, batch]
            # mm(weights[k,n], [k,batch], transpose=True) -> [n, batch]
            r = binary_densemm(args[0], args[1], transpose=True)
            return [r], [1]
    else:
        # weights[m,k], spikes[k] -> out[m]
        if axes == (None, 0):
            r = binary_densemm(args[0], args[1].T, transpose=False)
            return [r], [1]
        if axes == (None, 1):
            r = binary_densemm(args[0], args[1], transpose=False)
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
# Unified binary dense matrix-matrix product (binary_densemm)
# ==============================================================================
#
# transpose=False: weights[m,k] @ spikes[k,n] -> out[m,n]  (old dbmm)
# transpose=True:  weights[k,m].T @ spikes[k,n] -> out[m,n]  (old bdmm)
#
# Argument order is always (weights, spikes).


@namescope(static_argnames=['transpose'])
def binary_densemm(weights, spikes, *, transpose, backend=None):
    """
    Performs event-driven dense matrix-matrix multiplication.

    When ``transpose=False``, computes ``weights[m,k] @ spikes[k,n] -> out[m,n]``
    (dense matrix times binary matrix).

    When ``transpose=True``, computes ``weights[k,m].T @ spikes[k,n] -> out[m,n]``
    (transposed dense matrix times binary matrix). Both weights and spikes share
    their first dimension ``k``.

    Parameters
    ----------
    weights : array_like
        The weight matrix. Shape ``(m, k)`` when ``transpose=False``,
        or ``(k, m)`` when ``transpose=True``. Can be a ``brainunit`` quantity.
    spikes : array_like
        The binary matrix. Shape ``(k, n)`` in both modes. Can be boolean or float.
        Can be a ``brainunit`` quantity.
    transpose : bool
        If False, compute ``weights @ spikes``. If True, compute ``weights.T @ spikes``.
    backend : str, optional
        Backend to use for the computation.

    Returns
    -------
    array_like
        Result matrix with shape ``(m, n)``.
    """
    with jax.ensure_compile_time_eval():
        weights = u.math.asarray(weights)
        spikes = u.math.asarray(spikes)
    weight_val, wunit = u.split_mantissa_unit(weights)
    spk_val, spkunit = u.split_mantissa_unit(spikes)
    r = binary_densemm_p_call(weight_val, spk_val, transpose=transpose, backend=backend)
    return u.maybe_decimal(r[0] * wunit * spkunit)


def _binary_densemm_numba_kernel(
    spk_info: jax.ShapeDtypeStruct,
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba

    if transpose:
        # weights[k,m].T @ spikes[k,n] -> out[m,n]
        # primitive args: (weights, spikes)
        if spk_info.dtype == jnp.bool_:
            @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
            def kernel(weights, spikes, posts):
                for i_n in numba.prange(spikes.shape[1]):
                    out = np.zeros(weights.shape[1], dtype=posts.dtype)
                    for i_k in range(spikes.shape[0]):
                        if spikes[i_k, i_n]:
                            out += weights[i_k]
                    posts[:, i_n] = out
        else:
            @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
            def kernel(weights, spikes, posts):
                for i_n in numba.prange(spikes.shape[1]):
                    out = np.zeros(weights.shape[1], dtype=posts.dtype)
                    for i_k in range(spikes.shape[0]):
                        if spikes[i_k, i_n] > 0.:
                            out += weights[i_k]
                    posts[:, i_n] = out
    else:
        # weights[m,k] @ spikes[k,n] -> out[m,n]
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


def _binary_densemm_warp_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if transpose:
        # weights[k,m].T @ spikes[k,n] -> out[m,n]
        k = weight_info.shape[0]
        m = weight_info.shape[1]
        n = spk_info.shape[1]

        if spk_info.dtype == jnp.bool_:
            spk_float_info = jax.ShapeDtypeStruct(spk_info.shape, jnp.float32)
            spike_warp_info = jaxinfo_to_warpinfo(spk_float_info)

            @warp.kernel
            def kernel(
                weights: weight_warp_info,
                spikes: spike_warp_info,
                out: out_warp_info,
            ):
                i_m, i_n = warp.tid()
                r = weights.dtype(0.)
                for i_k in range(k):
                    if spikes[i_k, i_n] > 0.:
                        r += weights[i_k, i_m]
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
                out: out_warp_info,
            ):
                i_m, i_n = warp.tid()
                r = weights.dtype(0.)
                for i_k in range(k):
                    if spikes[i_k, i_n] > 0.:
                        r += weights[i_k, i_m]
                out[i_m, i_n] = r

            def run(weights, spikes):
                out_info = kwargs['outs'][0]
                fn = jax_kernel(kernel, launch_dims=(m, n), num_outputs=1, output_dims={'out': out_info.shape})
                return fn(weights, spikes)

    else:
        # weights[m,k] @ spikes[k,n] -> out[m,n]
        k = spk_info.shape[0]
        n = spk_info.shape[1]
        m = weight_info.shape[0]

        if spk_info.dtype == jnp.bool_:
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


def _binary_densemm_pallas_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    from jax.experimental import pallas as pl

    if transpose:
        # weights[k,m].T @ spikes[k,n] -> out[m,n]
        k = weight_info.shape[0]
        m = weight_info.shape[1]
        n = spk_info.shape[1]
        block_dim = generate_block_dim(m, maximum=1024)

        def kernel(
            weight_ref,  # [k, m]
            spike_ref,  # [k, n]
            out_ref,  # [m, n]
        ):
            i_n = pl.program_id(0)
            i_m_block = pl.program_id(1)
            i_m_start = i_m_block * block_dim
            i_m_mask = i_m_start + jnp.arange(block_dim) < m

            def loop_fn(i_k, temp):
                spike = spike_ref[i_k, i_n]
                weight_row = weight_ref[i_k, pl.ds(i_m_start, block_dim)]
                weight_row = jnp.where(i_m_mask, weight_row, 0.0)
                return jax.lax.cond(
                    spike if spk_info.dtype == jnp.bool_ else spike > 0.,
                    lambda out: out + weight_row,
                    lambda out: out,
                    temp,
                )

            final_out = jax.lax.fori_loop(0, k, loop_fn, jnp.zeros(block_dim, dtype=weight_ref.dtype))
            out_ref[pl.ds(i_m_start, block_dim), i_n] = jnp.where(i_m_mask, final_out, 0.0)

        def run(weights, spikes):
            fn = pl.pallas_call(kernel, grid=(n, cdiv(m, block_dim)), out_shape=kwargs['outs'])
            return fn(weights, spikes)

    else:
        # weights[m,k] @ spikes[k,n] -> out[m,n]
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
                    spike if spk_info.dtype == jnp.bool_ else spike > 0.,
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


def _binary_densemm_jvp_weights(w_dot, weights, spikes, *, transpose, **kwargs):
    return binary_densemm_p_call(w_dot, spikes, transpose=transpose)


def _binary_densemm_jvp_spikes(spk_dot, weights, spikes, *, transpose, **kwargs):
    if transpose:
        return [weights.T @ spk_dot]
    else:
        return [weights @ spk_dot]


def _binary_densemm_transpose_rule(ct, weights, spikes, *, transpose, **kwargs):
    ct = ct[0]
    if transpose:
        # weights[k,m].T @ spikes[k,n] -> out[m,n]
        if ad.is_undefined_primal(spikes):
            # ct_spikes: weights[k,m] @ ct[m,n] -> [k,n]
            ct_events = weights @ ct
            return weights, (ad.Zero(spikes) if type(ct) is ad.Zero else ct_events)
        else:
            # ct_weights: spikes[k,n] @ ct.T[n,m] -> [k,m]
            ct_weights = spikes @ ct.T
            return (ad.Zero(weights) if type(ct) is ad.Zero else ct_weights), spikes
    else:
        # weights[m,k] @ spikes[k,n] -> out[m,n]
        if ad.is_undefined_primal(spikes):
            ct_events = weights.T @ ct
            return weights, (ad.Zero(spikes) if type(ct) is ad.Zero else ct_events)
        else:
            # ct[m,n], spikes[k,n] -> ct_weights[m,k]
            # ct[m,n] @ spikes.T[n,k] -> [m,k]
            ct_weights = binary_densemm(ct, spikes.T, transpose=False)
            return (ad.Zero(weights) if type(ct) is ad.Zero else ct_weights), spikes


def _binary_densemm_batching_spikes_fn(args, axes, *, transpose, **kwargs):
    weights, spikes = args
    if transpose:
        # weights[k,m].T @ spikes[k,n] -> out[m,n]
        # out shape: [weights.shape[1], spikes.shape[1]]
        # spikes batched: [batch,k,n] or [k,batch,n] or [k,n,batch]
        assert spikes.ndim == 3, 'requires 3D events.'
        assert weights.ndim == 2, 'requires 2D weights.'
        spk_axis = axes[1]
        if spk_axis == 0:
            # [batch,k,n] -> transpose(1,0,2) -> [k,batch,n] -> reshape [k, batch*n]
            spikes = jnp.transpose(spikes, (1, 0, 2))
            k, batch, n_val = spikes.shape
            events = spikes.reshape(k, batch * n_val)
            r = binary_densemm_p_call(weights, events, transpose=True)
            # result: [m, batch*n] -> reshape [m, batch, n]
            r = jnp.reshape(r[0], [r[0].shape[0], batch, n_val])
            return [r], [1]
        elif spk_axis == 1:
            # [k,batch,n] -> reshape [k, batch*n]
            k, batch, n_val = spikes.shape
            events = spikes.reshape(k, batch * n_val)
            r = binary_densemm_p_call(weights, events, transpose=True)
            # result: [m, batch*n] -> reshape [m, batch, n]
            r = jnp.reshape(r[0], [r[0].shape[0], batch, n_val])
            return [r], [1]
        elif spk_axis == 2:
            # [k,n,batch] -> reshape [k, n*batch]
            k, n_val, batch = spikes.shape
            events = spikes.reshape(k, n_val * batch)
            r = binary_densemm_p_call(weights, events, transpose=True)
            # result: [m, n*batch] -> reshape [m, n, batch]
            r = jnp.reshape(r[0], [r[0].shape[0], n_val, batch])
            return [r], [2]
    else:
        # weights[m,k] @ spikes[k,n] -> out[m,n]
        # spikes batched
        assert spikes.ndim == 3, 'requires 3D events.'
        assert weights.ndim == 2, 'requires 2D weights.'
        spk_axis = axes[1]
        if spk_axis == 0:
            spikes = jnp.transpose(spikes, (1, 0, 2))
            k, maybe_batch1, maybe_batch2 = spikes.shape
            events = spikes.reshape(k, maybe_batch1 * maybe_batch2)
            r = binary_densemm_p_call(weights, events, transpose=False)
            r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
            return [r], [1]
        elif spk_axis == 1:
            k, maybe_batch1, maybe_batch2 = spikes.shape
            events = spikes.reshape(k, maybe_batch1 * maybe_batch2)
            r = binary_densemm_p_call(weights, events, transpose=False)
            r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
            return [r], [1]
        elif spk_axis == 2:
            k, maybe_batch1, maybe_batch2 = spikes.shape
            events = spikes.reshape(k, maybe_batch1 * maybe_batch2)
            r = binary_densemm_p_call(weights, events, transpose=False)
            r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
            return [r], [2]
    return general_batching_rule(binary_densemm_p, args, axes, transpose=transpose, **kwargs)


def _binary_densemm_batching_weights_fn(args, axes, *, transpose, **kwargs):
    weights, spikes = args
    if transpose:
        # weights[k,m].T @ spikes[k,n] -> out[m,n]
        # out shape: [weights.shape[1], spikes.shape[1]]
        # weights batched: [batch,k,m] or [k,batch,m] or [k,m,batch]
        assert weights.ndim == 3, 'requires 3D weights.'
        assert spikes.ndim == 2, 'requires 2D events.'
        w_axis = axes[0]
        if w_axis == 0:
            # [batch,k,m] -> transpose(1,0,2) -> [k,batch,m] -> reshape [k, batch*m]
            weights = jnp.transpose(weights, (1, 0, 2))
            k, batch, m_val = weights.shape
            w = weights.reshape(k, batch * m_val)
            r = binary_densemm_p_call(w, spikes, transpose=True)
            # result: [batch*m, n] -> reshape [batch, m, n]
            r = jnp.reshape(r[0], [batch, m_val, r[0].shape[1]])
            return [r], [0]
        elif w_axis == 1:
            # [k,batch,m] -> reshape [k, batch*m]
            k, batch, m_val = weights.shape
            w = weights.reshape(k, batch * m_val)
            r = binary_densemm_p_call(w, spikes, transpose=True)
            # result: [batch*m, n] -> reshape [batch, m, n]
            r = jnp.reshape(r[0], [batch, m_val, r[0].shape[1]])
            return [r], [0]
        elif w_axis == 2:
            # [k,m,batch] -> reshape [k, m*batch]
            k, m_val, batch = weights.shape
            w = weights.reshape(k, m_val * batch)
            r = binary_densemm_p_call(w, spikes, transpose=True)
            # result: [m*batch, n] -> reshape [m, batch, n]
            r = jnp.reshape(r[0], [m_val, batch, r[0].shape[1]])
            return [r], [1]
    else:
        # weights[m,k] @ spikes[k,n] -> out[m,n]
        # weights batched
        assert weights.ndim == 3, 'requires 3D weights.'
        assert spikes.ndim == 2, 'requires 2D events.'
        w_axis = axes[0]
        if w_axis == 0:
            maybe_batch1, maybe_batch2, k = weights.shape
            w = weights.reshape(maybe_batch1 * maybe_batch2, k)
            r = binary_densemm_p_call(w, spikes, transpose=False)
            r = jnp.reshape(r[0], [maybe_batch1, maybe_batch2, r[0].shape[-1]])
            return [r], [0]
        elif w_axis == 1:
            maybe_batch1, maybe_batch2, k = weights.shape
            w = weights.reshape(maybe_batch1 * maybe_batch2, k)
            r = binary_densemm_p_call(w, spikes, transpose=False)
            r = jnp.reshape(r[0], [maybe_batch1, maybe_batch2, r[0].shape[-1]])
            return [r], [1]
        elif w_axis == 2:
            weights = jnp.transpose(weights, (0, 2, 1))
            maybe_batch1, maybe_batch2, k = weights.shape
            w = weights.reshape(maybe_batch1 * maybe_batch2, k)
            r = binary_densemm_p_call(w, spikes, transpose=False)
            r = jnp.reshape(r[0], [maybe_batch1, maybe_batch2, r[0].shape[-1]])
            return [r], [1]
    return general_batching_rule(binary_densemm_p, args, axes, transpose=transpose, **kwargs)


def _binary_densemm_batching(args, axes, *, transpose, **kwargs):
    w_axis, spk_axis = axes
    if w_axis is None:
        return _binary_densemm_batching_spikes_fn(args, axes, transpose=transpose, **kwargs)
    elif spk_axis is None:
        return _binary_densemm_batching_weights_fn(args, axes, transpose=transpose, **kwargs)
    else:
        return general_batching_rule(binary_densemm_p, args, axes, transpose=transpose, **kwargs)


def _binary_densemm_benchmark_data(*, platform):
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


def binary_densemm_p_call(weights, spikes, *, transpose, backend: Optional[str] = None):
    if transpose:
        # weights[k,m].T @ spikes[k,n] -> out[m,n]
        assert weights.shape[0] == spikes.shape[0], (
            f"weights shape {weights.shape} and spikes shape {spikes.shape} do not match"
            f" for event matrix multiplication: weights dim 0 ({weights.shape[0]}) != spikes dim 0 ({spikes.shape[0]})"
        )
        out = jax.ShapeDtypeStruct([weights.shape[1], spikes.shape[1]], weights.dtype)
    else:
        # weights[m,k] @ spikes[k,n] -> out[m,n]
        assert weights.shape[1] == spikes.shape[0], (
            f"weights.shape[1] ({weights.shape[1]}) != spikes.shape[0] ({spikes.shape[0]})"
            f", weights: {weights.shape}, spikes: {spikes.shape}"
        )
        out = jax.ShapeDtypeStruct([weights.shape[0], spikes.shape[1]], weights.dtype)
    return binary_densemm_p(
        weights,
        spikes,
        outs=[out],
        transpose=transpose,
        spk_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        backend=backend,
    )


binary_densemm_p = XLACustomKernel('binary_densemm')
binary_densemm_p.def_numba_kernel(_binary_densemm_numba_kernel)
binary_densemm_p.def_warp_kernel(_binary_densemm_warp_kernel)
binary_densemm_p.def_pallas_kernel('gpu', _binary_densemm_pallas_kernel)
binary_densemm_p.def_pallas_kernel('tpu', _binary_densemm_pallas_kernel)
binary_densemm_p.def_jvp_rule2(_binary_densemm_jvp_weights, _binary_densemm_jvp_spikes)
binary_densemm_p.def_transpose_rule(_binary_densemm_transpose_rule)
binary_densemm_p.def_batching_rule(_binary_densemm_batching)
binary_densemm_p.def_call(binary_densemm_p_call)
binary_densemm_p.def_tags('dense', 'binary')
binary_densemm_p.def_benchmark_data(_binary_densemm_benchmark_data)
