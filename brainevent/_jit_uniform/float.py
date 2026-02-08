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
import numpy as np
from jax import numpy as jnp
from jax.interpreters import ad

from brainevent._jitc_matrix import _initialize_seed, _initialize_conn_length
from brainevent._misc import generate_block_dim, namescope
from brainevent._op import XLACustomKernel, numba_kernel, jaxinfo_to_warpinfo, general_batching_rule
from brainevent._op.benchmark import BenchmarkConfig
from brainevent._pallas_random import PallasLFSR88RNG
from brainevent._typing import Data, MatrixShape

__all__ = [
    "jitu",
    "jitu_p",
    "jitumv",
    "jitumv_p",
    "jitumm",
    "jitumm_p",
]


@namescope(static_argnames=("shape", "transpose", "corder"))
def jitu(
    w_low: Data,
    w_high: Data,
    prob: float,
    seed: int,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
    backend: Optional[str] = None,
) -> Data:
    u.fail_for_dimension_mismatch(w_low, w_high, "w_low and w_high must have the same dimension.")
    w_low, unitd = u.split_mantissa_unit(w_low)
    w_high = u.Quantity(w_high).to(unitd).mantissa
    clen = _initialize_conn_length(prob)
    res = jitu_p_call(
        w_low,
        w_high,
        clen,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd)


@namescope(static_argnames=("shape", "transpose", "corder"))
def jitumv(
    w_low: Data,
    w_high: Data,
    prob: float,
    vector: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
    backend: Optional[str] = None,
) -> Data:
    u.fail_for_dimension_mismatch(w_low, w_high, "w_low and w_high must have the same dimension.")
    seed = _initialize_seed(seed)
    w_low, unitd = u.split_mantissa_unit(w_low)
    w_high = u.Quantity(w_high).to(unitd).mantissa
    vector, unitv = u.split_mantissa_unit(vector)
    clen = _initialize_conn_length(prob)
    res = jitumv_p_call(
        w_low,
        w_high,
        clen,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd * unitv)


@namescope(static_argnames=("shape", "transpose", "corder"))
def jitumm(
    w_low: Data,
    w_high: Data,
    prob: float,
    B: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
    backend: Optional[str] = None,
) -> Data:
    u.fail_for_dimension_mismatch(w_low, w_high, "w_low and w_high must have the same dimension.")
    seed = _initialize_seed(seed)
    w_low, unitd = u.split_mantissa_unit(w_low)
    w_high = u.Quantity(w_high).to(unitd).mantissa
    B, unitB = u.split_mantissa_unit(B)
    clen = _initialize_conn_length(prob)
    res = jitumm_p_call(
        w_low,
        w_high,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd * unitB)


def _jitu_numba_kernel_generator(
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    import numba

    if corder:
        if transpose:
            # JIT matrix.T
            # - JIT matrix shape = [m, n]
            @numba.njit(fastmath=True)
            def kernel_impl(w_low, w_high, clen, seed, posts):
                posts[:] = 0.
                m = posts.shape[1]
                n = posts.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                clen0 = clen[0]
                seed0 = seed[0]
                np.random.seed(seed0)
                for i_row in range(n):
                    i_col = np.random.randint(0, clen0)
                    while i_col < m:
                        posts[i_row, i_col] = np.random.uniform(low=w_low0, high=w_high0)
                        i_col += np.random.randint(1, clen0)

        else:
            # JIT matrix
            # - JIT matrix shape = [m, n]
            @numba.njit(fastmath=True)
            def kernel_impl(w_low, w_high, clen, seed, posts):
                posts[:] = 0.
                m = posts.shape[0]
                n = posts.shape[1]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                seed0 = seed[0]
                clen0 = clen[0]
                np.random.seed(seed0)
                for i_row in range(m):
                    i_col = np.random.randint(0, clen0)
                    while i_col < n:
                        posts[i_row, i_col] = np.random.uniform(low=w_low0, high=w_high0)
                        i_col += np.random.randint(1, clen0)

    else:
        if transpose:
            # JIT matrix.T
            # - JIT matrix shape = [m, n]
            @numba.njit(fastmath=True)
            def kernel_impl(w_low, w_high, clen, seed, posts):
                posts[:] = 0.
                m = posts.shape[1]
                n = posts.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                clen0 = clen[0]
                seed0 = seed[0]
                np.random.seed(seed0)
                for i_col in range(m):
                    i_row = np.random.randint(0, clen0)
                    while i_row < n:
                        posts[i_row, i_col] = np.random.uniform(low=w_low0, high=w_high0)
                        i_row += np.random.randint(1, clen0)

        else:
            # JIT matrix
            # - JIT matrix shape = [m, n]
            @numba.njit(fastmath=True)
            def kernel_impl(w_low, w_high, clen, seed, posts):
                posts[:] = 0.
                m = posts.shape[0]
                n = posts.shape[1]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                clen0 = clen[0]
                seed0 = seed[0]
                np.random.seed(seed0)
                for i_col in range(n):
                    i_row = np.random.randint(0, clen0)
                    while i_row < m:
                        posts[i_row, i_col] = np.random.uniform(low=w_low0, high=w_high0)
                        i_row += np.random.randint(1, clen0)

    def kernel(w_low, w_high, clen, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(w_low, w_high, clen, seed)

    return kernel


def _jitu_warp_kernel_generator(
    w_low_info: jax.ShapeDtypeStruct,
    w_high_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    w_low_warp = jaxinfo_to_warpinfo(w_low_info)
    w_high_warp = jaxinfo_to_warpinfo(w_high_info)
    clen_warp = jaxinfo_to_warpinfo(clen_info)
    seed_warp = jaxinfo_to_warpinfo(seed_info)
    out_warp = jaxinfo_to_warpinfo(out_info)

    if corder:
        if transpose:
            # JIT matrix.T
            # - JIT matrix shape = [m, n]
            @warp.kernel
            def kernel_impl(
                w_low: w_low_warp,
                w_high: w_high_warp,
                clen: clen_warp,
                seed: seed_warp,
                posts: out_warp,
            ):
                m = posts.shape[1]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                w_diff = w_high0 - w_low0
                clen0 = clen[0]
                seed0 = seed[0]
                i_row = warp.tid()
                state = warp.rand_init(seed0 + i_row)
                i_col = warp.randi(state, 0, clen0)
                while i_col < m:
                    posts[i_row, i_col] = warp.randf(state) * w_diff + w_low0
                    i_col += warp.randi(state, 1, clen0)

        else:
            # JIT matrix
            # - JIT matrix shape = [m, n]
            @warp.kernel
            def kernel_impl(
                w_low: w_low_warp,
                w_high: w_high_warp,
                clen: clen_warp,
                seed: seed_warp,
                posts: out_warp,
            ):
                n = posts.shape[1]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                w_diff = w_high0 - w_low0
                clen0 = clen[0]
                seed0 = seed[0]
                i_row = warp.tid()
                state = warp.rand_init(seed0 + i_row)
                i_col = warp.randi(state, 0, clen0)
                while i_col < n:
                    posts[i_row, i_col] = warp.randf(state) * w_diff + w_low0
                    i_col += warp.randi(state, 1, clen0)

    else:
        if transpose:
            # JIT matrix.T
            # - JIT matrix shape = [m, n]
            @warp.kernel
            def kernel_impl(
                w_low: w_low_warp,
                w_high: w_high_warp,
                clen: clen_warp,
                seed: seed_warp,
                posts: out_warp,
            ):
                n = posts.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                w_diff = w_high0 - w_low0
                clen0 = clen[0]
                seed0 = seed[0]
                i_col = warp.tid()
                state = warp.rand_init(seed0 + i_col)
                i_row = warp.randi(state, 0, clen0)
                while i_row < n:
                    posts[i_row, i_col] = warp.randf(state) * w_diff + w_low0
                    i_row += warp.randi(state, 1, clen0)

        else:
            # JIT matrix
            # - JIT matrix shape = [m, n]
            @warp.kernel
            def kernel_impl(
                w_low: w_low_warp,
                w_high: w_high_warp,
                clen: clen_warp,
                seed: seed_warp,
                posts: out_warp,
            ):
                m = posts.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                w_diff = w_high0 - w_low0
                clen0 = clen[0]
                seed0 = seed[0]
                i_col = warp.tid()
                state = warp.rand_init(seed0 + i_col)
                i_row = warp.randi(state, 0, clen0)
                while i_row < m:
                    posts[i_row, i_col] = warp.randf(state) * w_diff + w_low0
                    i_row += warp.randi(state, 1, clen0)

    def kernel(w_low, w_high, clen, seed):
        dim = out_info.shape[0] if corder else out_info.shape[1]
        fn = jax_kernel(kernel_impl, launch_dims=[dim], num_outputs=1, output_dims={'posts': out_info.shape})
        return fn(w_low, w_high, clen, seed)

    return kernel


def _jitu_pallas_kernel_generator(
    out_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    from jax.experimental import pallas as pl

    dim = out_info.shape[0] if corder else out_info.shape[1]
    block_size = generate_block_dim(dim, maximum=128)

    if corder:
        def kernel(w_low_ref, w_high_ref, clen_ref, seed_ref, _, post_ref):
            m = post_ref.shape[1]
            w_low = w_low_ref[0]
            w_high = w_high_ref[0]
            clen0 = clen_ref[0]
            seed0 = seed_ref[0]
            i_row_block = pl.program_id(0)
            i_rows = i_row_block * block_size + jnp.arange(block_size)
            i_row_mask = i_rows < dim

            def body(data):
                i_cols, i_col_mask, rng = data
                val = rng.uniform(w_low, w_high)
                post_ref[i_rows, i_cols] = jnp.where(i_row_mask & i_col_mask, val, post_ref[i_rows, i_cols])
                i_cols += rng.random_integers(1, clen0)
                return i_cols, i_cols < m, rng

            rng = PallasLFSR88RNG(seed0 + i_rows)
            i_cols = rng.random_integers(0, clen0)
            i_col_mask = i_cols < m
            jax.lax.while_loop(
                lambda data: jnp.sum(data[1]) > 0,
                body,
                (i_cols, i_col_mask, rng)
            )

    else:
        def kernel(w_low_ref, w_high_ref, clen_ref, seed_ref, _, post_ref):
            n = post_ref.shape[0]
            w_low = w_low_ref[0]
            w_high = w_high_ref[0]
            clen0 = clen_ref[0]
            seed0 = seed_ref[0]
            i_col_block = pl.program_id(0)
            i_cols = i_col_block * block_size + jnp.arange(block_size)
            i_col_mask = i_cols < dim

            def body(data):
                i_rows, i_row_mask, rng = data
                val = rng.uniform(w_low, w_high)
                post_ref[i_rows, i_cols] = jnp.where(i_row_mask & i_col_mask, val, post_ref[i_rows, i_cols])
                i_rows = i_rows + rng.random_integers(1, clen0)
                return i_rows, i_rows < n, rng

            rng = PallasLFSR88RNG(seed0 + i_cols)
            i_rows = rng.random_integers(0, clen0)
            i_row_mask = i_rows < n
            jax.lax.while_loop(
                lambda data: jnp.sum(data[1]) > 0,
                body,
                (i_rows, i_row_mask, rng)
            )

    def run(w_low, w_high, clen, seed):
        fn = pl.pallas_call(
            kernel,
            grid=(pl.cdiv(dim, block_size),),
            input_output_aliases={4: 0},
            out_shape=kwargs['outs']
        )
        placeholder = jnp.zeros(kwargs['outs'][0].shape, kwargs['outs'][0].dtype)
        return fn(w_low, w_high, clen, seed, placeholder)

    return run


def _jitu_jvp_wlow(w_low_dot, w_low, w_high, clen, seed, *, shape, transpose: bool, corder: bool, **kwargs):
    res = jitu_p_call(0., w_low_dot, clen, seed, shape=shape, transpose=transpose, corder=corder)[0]
    return [w_low_dot - res]


def _jitu_jvp_whigh(w_high_dot, w_low, w_high, clen, seed, *, shape, transpose: bool, corder: bool, **kwargs):
    res = jitu_p_call(0., w_high_dot, clen, seed, shape=shape, transpose=transpose, corder=corder)
    return res


def _wlow_tranpose(ct, seed, clen, **kwargs):
    # JITC * (high - low) + low
    forward = jitu_p_call(0., 1., clen, seed, **kwargs)[0]
    return jnp.expand_dims((ct * (-forward + 1.)).sum(), axis=0)


def _whigh_tranpose(ct, seed, clen, **kwargs):
    # JITC * (high - low) + low
    forward = jitu_p_call(0., 1., clen, seed, **kwargs)[0]
    return jnp.expand_dims((ct * forward).sum(), axis=0)


def _jitu_transpose(ct, w_low, w_high, clen, seed, *, shape, transpose: bool, corder: bool, **kwargs):
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)
    ct = ct[0]
    if ad.is_undefined_primal(w_low):
        dwlow = _wlow_tranpose(
            ct,
            seed,
            clen,
            shape=shape,
            transpose=transpose,
            corder=corder,
        )
        return (dwlow, w_high, clen, seed)
    elif ad.is_undefined_primal(w_high):
        dwhigh = _whigh_tranpose(
            ct,
            seed,
            clen,
            shape=shape,
            transpose=transpose,
            corder=corder,
        )
        return (w_low, dwhigh, clen, seed)
    else:
        raise NotImplementedError(
            'JITC matrix transpose is only implemented for the w_low and w_high arguments.'
        )


def _jitu_batching(args, axes, **kwargs):
    return general_batching_rule(jitu_p, args, axes, **kwargs)


def _jitu_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for corder in (True, False):
            w_low = jnp.zeros(1, dtype=dtype)
            w_high = jnp.ones(1, dtype=dtype)
            clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
            seed = jnp.asarray(42, dtype=jnp.uint32)
            name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'}"
            configs.append(BenchmarkConfig(name, (w_low, w_high, clen, seed), {
                'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder
            }))
    return configs


def jitu_p_call(
    w_low,
    w_high,
    clen,
    seed,
    *,
    shape,
    transpose: bool,
    corder: bool,
    backend: Optional[str] = None,
):
    w_low = jnp.atleast_1d(w_low)
    w_high = jnp.atleast_1d(w_high)
    clen = jnp.atleast_1d(clen)
    seed = jnp.atleast_1d(seed)
    assert jnp.issubdtype(w_low.dtype, jnp.floating), 'Weights must be a floating-point type.'
    assert w_low.dtype == w_high.dtype, "w_low and w_high must have the same dtype."

    out_info = (
        jax.ShapeDtypeStruct(shape[::-1], dtype=w_low.dtype)
        if transpose else
        jax.ShapeDtypeStruct(shape, dtype=w_low.dtype)
    )

    return jitu_p(
        w_low,
        w_high,
        clen,
        seed,
        outs=[out_info],
        w_low_info=jax.ShapeDtypeStruct(w_low.shape, w_low.dtype),
        w_high_info=jax.ShapeDtypeStruct(w_high.shape, w_high.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=backend,
    )


jitu_p = XLACustomKernel('float_jitu')
jitu_p.def_numba_kernel(_jitu_numba_kernel_generator)
jitu_p.def_warp_kernel(_jitu_warp_kernel_generator)
jitu_p.def_pallas_kernel('gpu', _jitu_pallas_kernel_generator)
jitu_p.def_pallas_kernel('tpu', _jitu_pallas_kernel_generator)
jitu_p.def_jvp_rule2(_jitu_jvp_wlow, _jitu_jvp_whigh, None, None)
jitu_p.def_transpose_rule(_jitu_transpose)
jitu_p.def_batching_rule(_jitu_batching)
jitu_p.def_call(jitu_p_call)
jitu_p.def_tags('jit_uniform', 'float')
jitu_p.def_benchmark_data(_jitu_benchmark_data)


# Kernel generators for JIT connection SPMV

def _jitumv_numba_kernel_generator(
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    import numba

    if corder:
        if transpose:
            @numba.njit(fastmath=True)
            def kernel_impl(w_low, w_high, clen, vector, seed, posts):
                n_col = posts.shape[0]
                n_row = vector.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                clen0 = clen[0]
                seed0 = seed[0]
                np.random.seed(seed0)
                for i_col in range(n_col):
                    i_row = np.random.randint(0, clen0)
                    out = np.asarray(0., dtype=vector.dtype)
                    while i_row < n_row:
                        out += vector[i_row] * np.random.uniform(low=w_low0, high=w_high0)
                        i_row += np.random.randint(1, clen0)
                    posts[i_col] = out

        else:
            @numba.njit(fastmath=True)
            def kernel_impl(w_low, w_high, clen, vector, seed, posts):
                num_row = posts.shape[0]
                num_col = vector.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                seed0 = seed[0]
                clen0 = clen[0]
                np.random.seed(seed0)
                for i_row in range(num_row):
                    i_col = np.random.randint(0, clen0)
                    out = np.asarray(0., dtype=vector.dtype)
                    while i_col < num_col:
                        out += vector[i_col] * np.random.uniform(low=w_low0, high=w_high0)
                        i_col += np.random.randint(1, clen0)
                    posts[i_row] = out

    else:
        if transpose:
            @numba.njit(fastmath=True)
            def kernel_impl(w_low, w_high, clen, vector, seed, posts):
                posts[:] = 0.
                num_col = posts.shape[0]
                num_row = vector.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                clen0 = clen[0]
                seed0 = seed[0]
                np.random.seed(seed0)
                for i_row in range(num_row):
                    v = vector[i_row]
                    i_col = np.random.randint(0, clen0)
                    while i_col < num_col:
                        posts[i_col] += v * np.random.uniform(low=w_low0, high=w_high0)
                        i_col += np.random.randint(1, clen0)

        else:
            @numba.njit(fastmath=True)
            def kernel_impl(w_low, w_high, clen, vector, seed, posts):
                posts[:] = 0.
                num_row = posts.shape[0]
                num_col = vector.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                clen0 = clen[0]
                seed0 = seed[0]
                np.random.seed(seed0)
                for i_col in range(num_col):
                    v = vector[i_col]
                    i_row = np.random.randint(0, clen0)
                    while i_row < num_row:
                        posts[i_row] += v * np.random.uniform(low=w_low0, high=w_high0)
                        i_row += np.random.randint(1, clen0)

    def kernel(w_low, w_high, clen, vector, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(w_low, w_high, clen, vector, seed)

    return kernel


def _jitumv_warp_kernel_generator(
    w_low_info: jax.ShapeDtypeStruct,
    w_high_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    w_low_warp = jaxinfo_to_warpinfo(w_low_info)
    w_high_warp = jaxinfo_to_warpinfo(w_high_info)
    clen_warp = jaxinfo_to_warpinfo(clen_info)
    v_warp = jaxinfo_to_warpinfo(vector_info)
    seed_warp = jaxinfo_to_warpinfo(seed_info)
    out_warp = jaxinfo_to_warpinfo(out_info)

    if corder:
        if transpose:
            @warp.kernel
            def kernel_impl(
                w_low: w_low_warp,
                w_high: w_high_warp,
                clen: clen_warp,
                vector: v_warp,
                seed: seed_warp,
                posts: out_warp,
            ):
                num_row = vector.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                w_diff = w_high0 - w_low0
                clen0 = clen[0]
                seed0 = seed[0]
                i_col = warp.tid()
                r = float(0.0)
                state = warp.rand_init(seed0 + i_col)
                i_row = warp.randi(state, 0, clen0)
                while i_row < num_row:
                    r += vector[i_row] * (warp.randf(state) * w_diff + w_low0)
                    i_row += warp.randi(state, 1, clen0)
                posts[i_col] = r

        else:
            @warp.kernel
            def kernel_impl(
                w_low: w_low_warp,
                w_high: w_high_warp,
                clen: clen_warp,
                vector: v_warp,
                seed: seed_warp,
                posts: out_warp,
            ):
                num_col = vector.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                w_diff = w_high0 - w_low0
                clen0 = clen[0]
                seed0 = seed[0]
                i_row = warp.tid()
                r = float(0.0)
                state = warp.rand_init(seed0 + i_row)
                i_col = warp.randi(state, 0, clen0)
                while i_col < num_col:
                    r += vector[i_col] * (warp.randf(state) * w_diff + w_low0)
                    i_col += warp.randi(state, 1, clen0)
                posts[i_row] = r
    else:
        if transpose:
            @warp.kernel
            def kernel_impl(
                w_low: w_low_warp,
                w_high: w_high_warp,
                clen: clen_warp,
                vector: v_warp,
                seed: seed_warp,
                posts: out_warp,
            ):
                num_col = posts.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                w_diff = w_high0 - w_low0
                clen0 = clen[0]
                seed0 = seed[0]
                i_row = warp.tid()
                v = vector[i_row]
                state = warp.rand_init(seed0 + i_row)
                i_col = warp.randi(state, 0, clen0)
                while i_col < num_col:
                    posts[i_col] += v * (warp.randf(state) * w_diff + w_low0)
                    i_col += warp.randi(state, 1, clen0)

        else:
            @warp.kernel
            def kernel_impl(
                w_low: w_low_warp,
                w_high: w_high_warp,
                clen: clen_warp,
                vector: v_warp,
                seed: seed_warp,
                posts: out_warp,
            ):
                num_row = posts.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                w_diff = w_high0 - w_low0
                clen0 = clen[0]
                seed0 = seed[0]
                i_col = warp.tid()
                v = vector[i_col]
                state = warp.rand_init(seed0 + i_col)
                i_row = warp.randi(state, 0, clen0)
                while i_row < num_row:
                    posts[i_row] += v * (warp.randf(state) * w_diff + w_low0)
                    i_row += warp.randi(state, 1, clen0)

    def kernel(w_low, w_high, clen, vector, seed):
        dim = out_info.shape[0] if corder else vector_info.shape[0]
        fn = jax_kernel(kernel_impl, launch_dims=[dim], num_outputs=1, output_dims={'posts': out_info.shape})
        return fn(w_low, w_high, clen, vector, seed)

    return kernel


def _jitumv_pallas_kernel_generator(
    vector_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add

    dim = (out_info.shape[0] if corder else vector_info.shape[0])
    block_size = generate_block_dim(dim, maximum=128)

    if corder:
        def kernel(w_low_ref, w_high_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
            num_row = vector_ref.shape[0]
            w_low = w_low_ref[0]
            w_high = w_high_ref[0]
            clen = clen_ref[0]
            seed = seed_ref[0]
            i_col_block = pl.program_id(0)
            i_cols = i_col_block * block_size + jnp.arange(block_size)
            i_col_mask = i_cols < dim

            def body(data):
                i_rows, i_row_mask, rng, out = data
                v = jnp.where(i_row_mask, vector_ref[i_rows], 0.)
                out += v * rng.uniform(w_low, w_high)
                i_rows += rng.random_integers(1, clen)
                return i_rows, i_rows < num_row, rng, out

            rng = PallasLFSR88RNG(seed + i_cols)
            i_rows = rng.random_integers(0, clen)
            i_row_mask = i_rows < num_row
            out = jnp.zeros(block_size, dtype=post_ref.dtype)
            out = jax.lax.while_loop(
                lambda data: jnp.sum(data[1]) > 0,
                body,
                (i_rows, i_row_mask, rng, out)
            )[-1]
            post_ref[i_cols] = jnp.where(i_col_mask, out, post_ref[i_cols])

    else:
        def kernel(w_low_ref, w_high_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
            num_col = post_ref.shape[0]
            w_low = w_low_ref[0]
            w_high = w_high_ref[0]
            clen = clen_ref[0]
            seed = seed_ref[0]
            i_row_block = pl.program_id(0)
            i_rows = i_row_block * block_size + jnp.arange(block_size)
            i_row_mask = i_rows < dim
            vector = jnp.where(i_row_mask, vector_ref[i_rows], 0.)

            def body(data):
                i_cols, i_col_mask, rng = data
                atomic_add(post_ref, (i_cols,), vector * rng.uniform(w_low, w_high), mask=i_row_mask & i_col_mask)
                i_cols += rng.random_integers(1, clen)
                return i_cols, i_cols < num_col, rng

            rng = PallasLFSR88RNG(seed + i_rows)
            i_cols = rng.random_integers(0, clen)
            i_col_mask = i_cols < num_col
            jax.lax.while_loop(
                lambda data: jnp.sum(data[1]) > 0,
                body,
                (i_cols, i_col_mask, rng)
            )

    def run(w_low, w_high, clen, vector, seed):
        fn = pl.pallas_call(
            kernel,
            grid=(pl.cdiv(dim, block_size),),
            input_output_aliases={5: 0},
            out_shape=kwargs['outs']
        )
        placeholder = jnp.zeros(kwargs['outs'][0].shape, kwargs['outs'][0].dtype)
        return fn(w_low, w_high, clen, vector, seed, placeholder)

    return run


def _jitumv_jvp_v(v_dot, w_low, w_high, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    return jitumv_p_call(w_low, w_high, clen, v_dot, seed, shape=shape, transpose=transpose, corder=corder)


def _jitumv_jvp_wlow(w_dot, w_low, w_high, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    return jitumv_p_call(w_dot, w_high, clen, vector, seed, shape=shape, transpose=transpose, corder=corder)


def _jitumv_jvp_whigh(w_dot, w_low, w_high, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    return jitumv_p_call(w_low, w_dot, clen, vector, seed, shape=shape, transpose=transpose, corder=corder)


def _jitumv_transpose_rules(ct, w_low, w_high, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)

    ct = ct[0]
    if ad.is_undefined_primal(vector):
        r = jitumv_p_call(
            w_low,
            w_high,
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder
        )[0]
        return w_low, w_high, clen, r, seed
    elif ad.is_undefined_primal(w_low):
        # Fix the sampled connectivity and RNG stream (same `clen/seed/shape/transpose/corder`).
        # For each active entry:
        #   w_ij = w_low + (w_high - w_low) * u_ij,  u_ij in [0, 1).
        # The linear map output is therefore affine in (w_low, w_high):
        #   y = w_low * C(v) + (w_high - w_low) * U(v),
        # where
        #   U(v) = y(0, 1)  and  C(v) = y(1, 1).
        # Given cotangent ct, with inner product <a, b> = sum(a * b):
        #   dL/dw_high = <ct, U(v)>
        #   dL/dw_low  = <ct, C(v) - U(v)>.
        ones = jnp.ones((1,), dtype=ct.dtype)
        zeros = jnp.zeros((1,), dtype=ct.dtype)
        u_basis = jitumv_p_call(
            zeros,
            ones,
            clen,
            vector,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder
        )[0]
        c_basis = jitumv_p_call(
            ones,
            ones,
            clen,
            vector,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder
        )[0]
        dw_low = jnp.expand_dims(jnp.sum(ct * (c_basis - u_basis)), axis=0)
        return dw_low, w_high, clen, vector, seed
    elif ad.is_undefined_primal(w_high):
        zeros = jnp.zeros((1,), dtype=ct.dtype)
        ones = jnp.ones((1,), dtype=ct.dtype)
        u_basis = jitumv_p_call(
            zeros,
            ones,
            clen,
            vector,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder
        )[0]
        dw_high = jnp.expand_dims(jnp.sum(ct * u_basis), axis=0)
        return w_low, dw_high, clen, vector, seed
    else:
        raise NotImplementedError(
            f"Transpose rule for {ct} not implemented "
            f"for event-driven COO matrix-vector product."
        )


def _jitumv_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = jitumm_p_call(
            args[0],
            args[1],
            args[2],
            args[3].T,
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
        )
        return r, [1]
    elif tuple(axes) == (None, None, None, 1, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = jitumm_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
        )
        return r, [1]
    else:
        return general_batching_rule(jitumv_p, args, axes, **kwargs)


def _jitumv_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for corder in (True, False):
            w_low = jnp.zeros(1, dtype=dtype)
            w_high = jnp.ones(1, dtype=dtype)
            clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
            v_size = n_post if not transpose else n_pre
            vector = jnp.asarray(np.random.randn(v_size), dtype=dtype)
            seed = jnp.asarray(42, dtype=jnp.uint32)
            name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'}"
            configs.append(
                BenchmarkConfig(
                    name,
                    (w_low, w_high, clen, vector, seed),
                    {'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder}
                )
            )
    return configs


def jitumv_p_call(
    w_low,
    w_high,
    clen,
    vector,
    seed,
    *,
    shape,
    transpose: bool,
    corder: bool,
    backend: Optional[str] = None,
):
    w_low = jnp.atleast_1d(w_low)
    w_high = jnp.atleast_1d(w_high)
    clen = jnp.atleast_1d(clen)

    assert len(shape) == 2, "The matrix shape should be a tuple of two integers."
    assert w_low.shape == (1,), f"The weight shape should be (1,), but got {w_low.shape}."
    assert w_high.shape == (1,), f"The weight shape should be (1,), but got {w_high.shape}."
    assert clen.shape == (1,), f"The clen shape should be (1,), but got {clen.shape}."
    assert vector.ndim == 1, f"The vector should be a 1D array, but got {vector.ndim}D."
    assert seed.shape == (1,), f"The seed shape should be (1,), but got {seed.shape}."
    assert jnp.issubdtype(w_low.dtype, jnp.floating), 'Weights must be a floating-point type.'
    assert w_low.dtype == w_high.dtype, "w_low and w_high must have the same dtype."

    if transpose:
        assert shape[0] == len(vector), f"The matrix shape and vector length do not match. {vector.shape} @ {shape}"
    else:
        assert shape[1] == len(vector), f"The matrix shape and vector length do not match. {shape} @ {vector.shape}"

    out_info = (
        jax.ShapeDtypeStruct([shape[1]], w_low.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0]], w_low.dtype)
    )

    return jitumv_p(
        w_low,
        w_high,
        clen,
        vector,
        seed,
        outs=[out_info],
        w_low_info=jax.ShapeDtypeStruct(w_low.shape, w_low.dtype),
        w_high_info=jax.ShapeDtypeStruct(w_high.shape, w_high.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        vector_info=jax.ShapeDtypeStruct(vector.shape, vector.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=backend,
    )


jitumv_p = XLACustomKernel('float_jitumv')
jitumv_p.def_numba_kernel(_jitumv_numba_kernel_generator)
jitumv_p.def_warp_kernel(_jitumv_warp_kernel_generator)
jitumv_p.def_pallas_kernel('gpu', _jitumv_pallas_kernel_generator)
jitumv_p.def_pallas_kernel('tpu', _jitumv_pallas_kernel_generator)
jitumv_p.def_jvp_rule2(_jitumv_jvp_wlow, _jitumv_jvp_whigh, None, _jitumv_jvp_v, None)
jitumv_p.def_transpose_rule(_jitumv_transpose_rules)
jitumv_p.def_batching_rule(_jitumv_batching)
jitumv_p.def_call(jitumv_p_call)
jitumv_p.def_tags('jit_uniform', 'float')
jitumv_p.def_benchmark_data(_jitumv_benchmark_data)


def _jitumm_numba_kernel_generator(
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    import numba

    if corder:
        if transpose:
            # JIT Matrix.T @ B
            # - JIT matrix: [k, m]
            # - B: [k, n]
            @numba.njit(fastmath=True)
            def kernel_impl(w_low, w_high, clen, B, seed, posts):
                m = posts.shape[0]
                n = posts.shape[1]
                k = B.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                seed0 = seed[0]
                clen0 = clen[0]
                np.random.seed(seed0)
                for i_m in range(m):
                    i_k = np.random.randint(0, clen0)
                    out = np.zeros(n, dtype=B.dtype)
                    while i_k < k:
                        out += B[i_k] * np.random.uniform(low=w_low0, high=w_high0)
                        i_k += np.random.randint(1, clen0)
                    posts[i_m] = out

        else:
            # JIT Matrix @ B
            # - JIT matrix: [m, k]
            # - B: [k, n]
            @numba.njit(fastmath=True)
            def kernel_impl(w_low, w_high, clen, B, seed, posts):
                m = posts.shape[0]
                n = posts.shape[1]
                k = B.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                seed0 = seed[0]
                clen0 = clen[0]
                np.random.seed(seed0)
                for i_m in range(m):
                    i_k = np.random.randint(0, clen0)
                    out = np.zeros(n, dtype=B.dtype)
                    while i_k < k:
                        out += B[i_k] * np.random.uniform(low=w_low0, high=w_high0)
                        i_k += np.random.randint(1, clen0)
                    posts[i_m] = out

    else:
        if transpose:
            # JIT Matrix.T @ B
            # - JIT matrix: [k, m]
            # - B: [k, n]
            @numba.njit(fastmath=True)
            def kernel_impl(w_low, w_high, clen, B, seed, posts):
                posts[:] = 0.
                m = posts.shape[0]
                k = B.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                seed0 = seed[0]
                clen0 = clen[0]
                np.random.seed(seed0)
                for i_k in range(k):
                    out = B[i_k]
                    i_m = np.random.randint(0, clen0)
                    while i_m < m:
                        posts[i_m] += out * np.random.uniform(low=w_low0, high=w_high0)
                        i_m += np.random.randint(1, clen0)

        else:
            # JIT Matrix @ B
            # - JIT matrix: [m, k]
            # - B: [k, n]
            @numba.njit(fastmath=True)
            def kernel_impl(w_low, w_high, clen, B, seed, posts):
                posts[:] = 0.
                m = posts.shape[0]
                k = B.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                seed0 = seed[0]
                clen0 = clen[0]
                np.random.seed(seed0)
                for i_k in range(k):
                    out = B[i_k]
                    i_m = np.random.randint(0, clen0)
                    while i_m < m:
                        posts[i_m] += out * np.random.uniform(low=w_low0, high=w_high0)
                        i_m += np.random.randint(1, clen0)

    def kernel(w_low, w_high, clen, B, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(w_low, w_high, clen, B, seed)

    return kernel


def _jitumm_warp_kernel_generator(
    w_low_info: jax.ShapeDtypeStruct,
    w_high_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    B_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    TITLE_SIZE: int,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    w_low_warp = jaxinfo_to_warpinfo(w_low_info)
    w_high_warp = jaxinfo_to_warpinfo(w_high_info)
    clen_warp = jaxinfo_to_warpinfo(clen_info)
    B_warp = jaxinfo_to_warpinfo(B_info)
    seed_warp = jaxinfo_to_warpinfo(seed_info)
    out_warp = jaxinfo_to_warpinfo(out_info)

    if corder:
        if transpose:
            # JIT Matrix.T @ B
            @warp.kernel
            def kernel_impl(
                w_low: w_low_warp,
                w_high: w_high_warp,
                clen: clen_warp,
                B: B_warp,
                seed: seed_warp,
                posts: out_warp,
            ):
                k = B.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                w_diff = w_high0 - w_low0
                clen0 = clen[0]
                seed0 = seed[0]
                i_m = warp.tid()
                state = warp.rand_init(seed0 + i_m)
                out = warp.tile_zeros(TITLE_SIZE, dtype=warp.float32)
                i_k = warp.randi(state, 0, clen0)
                while i_k < k:
                    w = warp.randf(state) * w_diff + w_low0
                    out += warp.tile_load(B[i_k], TITLE_SIZE) * w
                    i_k += warp.randi(state, 1, clen0)
                warp.tile_store(posts[i_m], out)

        else:
            # JIT Matrix @ B
            @warp.kernel
            def kernel_impl(
                w_low: w_low_warp,
                w_high: w_high_warp,
                clen: clen_warp,
                B: B_warp,
                seed: seed_warp,
                posts: out_warp,
            ):
                k = B.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                w_diff = w_high0 - w_low0
                clen0 = clen[0]
                seed0 = seed[0]
                i_m = warp.tid()
                state = warp.rand_init(seed0 + i_m)
                out = warp.tile_zeros(TITLE_SIZE, dtype=warp.float32)
                i_k = warp.randi(state, 0, clen0)
                while i_k < k:
                    w = warp.randf(state) * w_diff + w_low0
                    out += warp.tile_load(B[i_k], TITLE_SIZE) * w
                    i_k += warp.randi(state, 1, clen0)
                warp.tile_store(posts[i_m], out)

    else:
        if transpose:
            # JIT Matrix.T @ B
            @warp.kernel
            def kernel_impl(
                w_low: w_low_warp,
                w_high: w_high_warp,
                clen: clen_warp,
                B: B_warp,
                seed: seed_warp,
                posts: out_warp,
            ):
                m = posts.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                w_diff = w_high0 - w_low0
                clen0 = clen[0]
                seed0 = seed[0]
                i_k = warp.tid()
                state = warp.rand_init(seed0 + i_k)
                out = warp.tile_load(B[i_k], TITLE_SIZE)
                i_m = warp.randi(state, 0, clen0)
                while i_m < m:
                    w = warp.randf(state) * w_diff + w_low0
                    warp.tile_atomic_add(posts[i_m], out * w)
                    i_m += warp.randi(state, 1, clen0)

        else:
            # JIT Matrix @ B
            @warp.kernel
            def kernel_impl(
                w_low: w_low_warp,
                w_high: w_high_warp,
                clen: clen_warp,
                B: B_warp,
                seed: seed_warp,
                posts: out_warp,
            ):
                m = posts.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                w_diff = w_high0 - w_low0
                clen0 = clen[0]
                seed0 = seed[0]
                i_k = warp.tid()
                state = warp.rand_init(seed0 + i_k)
                out = warp.tile_load(B[i_k], TITLE_SIZE)
                i_m = warp.randi(state, 0, clen0)
                while i_m < m:
                    w = warp.randf(state) * w_diff + w_low0
                    warp.tile_atomic_add(posts[i_m], out * w)
                    i_m += warp.randi(state, 1, clen0)

    def kernel(w_low, w_high, clen, B, seed):
        tile = out_info.shape[0] if corder else B_info.shape[0]
        fn = jax_kernel(kernel_impl, launch_dims=[tile], num_outputs=1, output_dims={'posts': out_info.shape})
        return fn(w_low, w_high, clen, B, seed)

    return kernel


def _jitumm_pallas_kernel_generator(
    B_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add

    block_dim = generate_block_dim(B_info.shape[1], maximum=1024)

    if corder:
        if transpose:
            # JIT Matrix.T @ B
            # - JIT matrix: [k, m]
            # - B: [k, n]
            def kernel(w_low_ref, w_high_ref, clen_ref, B_ref, seed_ref, _, post_ref):
                k = B_ref.shape[0]
                w_low0 = w_low_ref[0]
                w_high0 = w_high_ref[0]
                clen0 = clen_ref[0]
                seed0 = seed_ref[0]
                i_m = pl.program_id(0)
                i_n_block = pl.program_id(1)
                i_n_start = block_dim * i_n_block
                i_n_indices = i_n_start + jnp.arange(block_dim)
                mask = i_n_indices < B_info.shape[1]

                def body(data):
                    i, rng, out = data
                    w = rng.uniform(w_low0, w_high0)
                    B_vals = jnp.where(mask, B_ref[i, i_n_indices], 0.)
                    out += B_vals * w
                    i += rng.random_integers(1, clen0)
                    return i, rng, out

                rng = PallasLFSR88RNG(seed0 + i_m)
                out = jnp.zeros(block_dim, dtype=post_ref.dtype)
                _, _, out = jax.lax.while_loop(
                    lambda data: data[0] < k,
                    body,
                    (rng.random_integers(0, clen0), rng, out)
                )
                post_ref[i_m, i_n_indices] = jnp.where(mask, out, post_ref[i_m, i_n_indices])

        else:
            # JIT Matrix @ B
            # - JIT matrix: [m, k]
            # - B: [k, n]
            def kernel(w_low_ref, w_high_ref, clen_ref, B_ref, seed_ref, _, post_ref):
                k = B_ref.shape[0]
                w_low0 = w_low_ref[0]
                w_high0 = w_high_ref[0]
                clen0 = clen_ref[0]
                seed0 = seed_ref[0]
                i_m = pl.program_id(0)
                i_n_block = pl.program_id(1)
                i_n_start = block_dim * i_n_block
                i_n_indices = i_n_start + jnp.arange(block_dim)
                mask = i_n_indices < B_info.shape[1]

                def body(data):
                    i, rng, out = data
                    w = rng.uniform(w_low0, w_high0)
                    B_vals = jnp.where(mask, B_ref[i, i_n_indices], 0.)
                    out += B_vals * w
                    i += rng.random_integers(1, clen0)
                    return i, rng, out

                rng = PallasLFSR88RNG(seed0 + i_m)
                out = jnp.zeros(block_dim, dtype=post_ref.dtype)
                _, _, out = jax.lax.while_loop(
                    lambda data: data[0] < k,
                    body,
                    (rng.random_integers(0, clen0), rng, out)
                )
                post_ref[i_m, i_n_indices] = jnp.where(mask, out, post_ref[i_m, i_n_indices])

    else:
        if transpose:
            # JIT Matrix.T @ B
            # - JIT matrix: [k, m]
            # - B: [k, n]
            def kernel(w_low_ref, w_high_ref, clen_ref, B_ref, seed_ref, _, post_ref):
                m = post_ref.shape[0]
                w_low0 = w_low_ref[0]
                w_high0 = w_high_ref[0]
                clen0 = clen_ref[0]
                seed0 = seed_ref[0]
                i_k = pl.program_id(0)
                i_n_block = pl.program_id(1)
                i_n_start = block_dim * i_n_block
                i_n_indices = i_n_start + jnp.arange(block_dim)
                mask = i_n_indices < B_info.shape[1]

                B_block = jnp.where(mask, B_ref[i_k, i_n_indices], 0.)

                def body(data):
                    i, rng = data
                    w = rng.uniform(w_low0, w_high0)
                    atomic_add(post_ref, (i, i_n_indices), B_block * w, mask=mask)
                    i += rng.random_integers(1, clen0)
                    return i, rng

                rng = PallasLFSR88RNG(seed0 + i_k)
                jax.lax.while_loop(
                    lambda data: data[0] < m,
                    body,
                    (rng.random_integers(0, clen0), rng)
                )

        else:
            # JIT Matrix @ B
            # - JIT matrix: [m, k]
            # - B: [k, n]
            def kernel(w_low_ref, w_high_ref, clen_ref, B_ref, seed_ref, _, post_ref):
                m = post_ref.shape[0]
                w_low0 = w_low_ref[0]
                w_high0 = w_high_ref[0]
                clen0 = clen_ref[0]
                seed0 = seed_ref[0]
                i_k = pl.program_id(0)
                i_n_block = pl.program_id(1)
                i_n_start = block_dim * i_n_block
                i_n_indices = i_n_start + jnp.arange(block_dim)
                mask = i_n_indices < B_info.shape[1]

                B_block = jnp.where(mask, B_ref[i_k, i_n_indices], 0.)

                def body(data):
                    i, rng = data
                    w = rng.uniform(w_low0, w_high0)
                    atomic_add(post_ref, (i, i_n_indices), B_block * w, mask=mask)
                    i += rng.random_integers(1, clen0)
                    return i, rng

                rng = PallasLFSR88RNG(seed0 + i_k)
                jax.lax.while_loop(
                    lambda data: data[0] < m,
                    body,
                    (rng.random_integers(0, clen0), rng)
                )

    tile = (out_info.shape[0] if corder else B_info.shape[0])
    grid = (tile, pl.cdiv(B_info.shape[1], block_dim))

    def run(w_low, w_high, clen, B, seed):
        fn = pl.pallas_call(
            kernel,
            grid=grid,
            input_output_aliases={5: 0},
            out_shape=kwargs['outs']
        )
        placeholder = jnp.zeros(kwargs['outs'][0].shape, kwargs['outs'][0].dtype)
        return fn(w_low, w_high, clen, B, seed, placeholder)

    return run


def _jitumm_jvp_wlow(w_dot, w_low, w_high, clen, B, seed, *, shape, transpose, corder, **kwargs):
    return jitumm_p_call(w_dot, w_high, clen, B, seed, shape=shape, transpose=transpose, corder=corder)


def _jitumm_jvp_whigh(w_dot, w_low, w_high, clen, B, seed, *, shape, transpose, corder, **kwargs):
    return jitumm_p_call(w_low, w_dot, clen, B, seed, shape=shape, transpose=transpose, corder=corder)


def _jitumm_jvp_B(B_dot, w_low, w_high, clen, B, seed, *, shape, transpose, corder, **kwargs):
    return jitumm_p_call(w_low, w_high, clen, B_dot, seed, shape=shape, transpose=transpose, corder=corder)


def _jitumm_transpose_rules(ct, w_low, w_high, clen, B, seed, *, shape, transpose, corder, **kwargs):
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)

    ct = ct[0]
    if ad.is_undefined_primal(B):
        dB = jitumm_p_call(
            w_low,
            w_high,
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder,
        )[0]
        return w_low, w_high, clen, dB, seed
    elif ad.is_undefined_primal(w_low):
        # Same affine decomposition as _jitumv_transpose_rules, now for matrix right operand B:
        #   Y = w_low * C(B) + (w_high - w_low) * U(B),
        #   U(B) = Y(0, 1), C(B) = Y(1, 1).
        # Hence:
        #   dL/dw_high = <ct, U(B)>
        #   dL/dw_low  = <ct, C(B) - U(B)>.
        ones = jnp.ones((1,), dtype=ct.dtype)
        zeros = jnp.zeros((1,), dtype=ct.dtype)
        u_basis = jitumm_p_call(
            zeros,
            ones,
            clen,
            B,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder
        )[0]
        c_basis = jitumm_p_call(
            ones,
            ones,
            clen,
            B,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder
        )[0]
        dw_low = jnp.expand_dims(jnp.sum(ct * (c_basis - u_basis)), axis=0)
        return dw_low, w_high, clen, B, seed
    elif ad.is_undefined_primal(w_high):
        zeros = jnp.zeros((1,), dtype=ct.dtype)
        ones = jnp.ones((1,), dtype=ct.dtype)
        u_basis = jitumm_p_call(
            zeros,
            ones,
            clen,
            B,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder
        )[0]
        dw_high = jnp.expand_dims(jnp.sum(ct * u_basis), axis=0)
        return w_low, dw_high, clen, B, seed
    else:
        raise NotImplementedError(
            'Transpose rules for jitc_matmat_uniform not implemented for '
            'non-undefined primals.'
        )


def _batching_axis1(args, axis=1, **kwargs):
    assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
    m, maybe_batch1, maybe_batch2 = args[3].shape
    B = args[3].reshape(m, maybe_batch1 * maybe_batch2)
    r = jitumm_p_call(
        args[0],
        args[1],
        args[2],
        B,
        args[4],
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
        corder=kwargs['corder'],
    )
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _jitumm_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        args = list(args)
        args[3] = jnp.transpose(args[3], (1, 0, 2))
        return _batching_axis1(args, **kwargs)

    elif tuple(axes) == (None, None, None, 1, None):
        return _batching_axis1(args, **kwargs)

    elif tuple(axes) == (None, None, None, 2, None):
        return _batching_axis1(args, axis=2, **kwargs)

    else:
        return general_batching_rule(jitumm_p, args, axes, **kwargs)


def _jitumm_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for corder in (True, False):
            w_low = jnp.zeros(1, dtype=dtype)
            w_high = jnp.ones(1, dtype=dtype)
            clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
            b_rows = n_post if not transpose else n_pre
            B = jnp.asarray(np.random.randn(b_rows, 10), dtype=dtype)
            seed = jnp.asarray(42, dtype=jnp.uint32)
            name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'}"
            configs.append(
                BenchmarkConfig(
                    name,
                    (w_low, w_high, clen, B, seed),
                    {'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder}
                )
            )
    return configs


def jitumm_p_call(
    w_low,
    w_high,
    clen,
    B,
    seed,
    *,
    shape: MatrixShape,
    transpose: bool,
    corder: bool,
    backend: Optional[str] = None,
):
    w_low = jnp.atleast_1d(w_low)
    w_high = jnp.atleast_1d(w_high)
    clen = jnp.atleast_1d(clen)

    assert len(shape) == 2, "The matrix shape should be a tuple of two integers."
    assert B.ndim == 2, "The input matrix B should be a 2D array."
    assert seed.ndim == 1, "The seed should be a 1D array."
    assert w_low.ndim == 1, "The weight should be a 1D array."
    assert w_high.ndim == 1, "The weight should be a 1D array."
    assert clen.ndim == 1, "The clen should be a 1D array."
    assert w_low.shape == (1,), "The weight should be a scalar."
    assert w_high.shape == (1,), "The weight should be a scalar."
    assert clen.shape == (1,), "The clen should be a scalar."
    assert seed.shape == (1,), "The seed should be a scalar."
    if transpose:
        assert shape[0] == B.shape[0], f"The matrix shape and B shape do not match. {B.shape} @ {shape}"
    else:
        assert shape[1] == B.shape[0], f"The matrix shape and B shape do not match. {shape} @ {B.shape}"
    assert jnp.issubdtype(w_low.dtype, jnp.floating), 'Weights must be a floating-point type.'
    assert w_low.dtype == w_high.dtype, "w_low and w_high must have the same dtype."

    out_info = (
        jax.ShapeDtypeStruct([shape[1], B.shape[1]], w_low.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0], B.shape[1]], w_low.dtype)
    )

    return jitumm_p(
        w_low,
        w_high,
        clen,
        B,
        seed,
        outs=[out_info],
        w_low_info=jax.ShapeDtypeStruct(w_low.shape, w_low.dtype),
        w_high_info=jax.ShapeDtypeStruct(w_high.shape, w_high.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        B_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        corder=corder,
        TITLE_SIZE=B.shape[1],
        backend=backend,
    )


jitumm_p = XLACustomKernel('float_jitumm')
jitumm_p.def_numba_kernel(_jitumm_numba_kernel_generator)
jitumm_p.def_warp_kernel(_jitumm_warp_kernel_generator)
jitumm_p.def_pallas_kernel('gpu', _jitumm_pallas_kernel_generator)
jitumm_p.def_pallas_kernel('tpu', _jitumm_pallas_kernel_generator)
jitumm_p.def_jvp_rule2(_jitumm_jvp_wlow, _jitumm_jvp_whigh, None, _jitumm_jvp_B, None)
jitumm_p.def_transpose_rule(_jitumm_transpose_rules)
jitumm_p.def_batching_rule(_jitumm_batching)
jitumm_p.def_call(jitumm_p_call)
jitumm_p.def_tags('jit_uniform', 'float')
jitumm_p.def_benchmark_data(_jitumm_benchmark_data)
