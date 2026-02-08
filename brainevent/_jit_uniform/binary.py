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

from typing import Optional, Sequence

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
from .float import jitumv_p_call, jitumm_p_call

__all__ = [
    "binary_jitumv",
    "binary_jitumv_p",
    "binary_jitumm",
    "binary_jitumm_p",
]


@namescope(static_argnames=("shape", "transpose", "corder"))
def binary_jitumv(
    w_low: Data,
    w_high: Data,
    prob: float,
    vector: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
) -> Data:
    u.fail_for_dimension_mismatch(w_low, w_high, "w_low and w_high must have the same dimension.")
    seed = _initialize_seed(seed)
    w_low, unitd = u.split_mantissa_unit(w_low)
    w_high = u.Quantity(w_high).to(unitd).mantissa
    vector, unitv = u.split_mantissa_unit(vector)
    clen = _initialize_conn_length(prob)
    res = binary_jitumv_p_call(
        w_low,
        w_high,
        clen,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder
    )[0]
    return u.maybe_decimal(res * unitd * unitv)


@namescope(static_argnames=("shape", "transpose", "corder"))
def binary_jitumm(
    w_low: Data,
    w_high: Data,
    prob: float,
    B: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
) -> Data:
    u.fail_for_dimension_mismatch(w_low, w_high, "w_low and w_high must have the same dimension.")
    seed = _initialize_seed(seed)
    w_low, unitd = u.split_mantissa_unit(w_low)
    w_high = u.Quantity(w_high).to(unitd).mantissa
    B, unitB = u.split_mantissa_unit(B)
    clen = _initialize_conn_length(prob)
    res = binary_jitumm_p_call(
        w_low,
        w_high,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder
    )[0]
    return u.maybe_decimal(res * unitd * unitB)


# Kernel generators for JIT connection SPMV

def _jitumv_numba_kernel_generator(
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    import numba

    if corder:
        if transpose:
            if vector_info.dtype == jnp.bool_:
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
                        out = np.asarray(0., dtype=posts.dtype)
                        while i_row < n_row:
                            w = np.random.uniform(low=w_low0, high=w_high0)
                            if vector[i_row]:
                                out += w
                            i_row += np.random.randint(1, clen0)
                        posts[i_col] = out
            else:
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
                        out = np.asarray(0., dtype=posts.dtype)
                        while i_row < n_row:
                            w = np.random.uniform(low=w_low0, high=w_high0)
                            if vector[i_row] > 0.:
                                out += w
                            i_row += np.random.randint(1, clen0)
                        posts[i_col] = out

        else:
            if vector_info.dtype == jnp.bool_:
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
                        out = np.asarray(0., dtype=posts.dtype)
                        while i_col < num_col:
                            w = np.random.uniform(low=w_low0, high=w_high0)
                            if vector[i_col]:
                                out += w
                            i_col += np.random.randint(1, clen0)
                        posts[i_row] = out
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
                        out = np.asarray(0., dtype=posts.dtype)
                        while i_col < num_col:
                            w = np.random.uniform(low=w_low0, high=w_high0)
                            if vector[i_col] > 0.:
                                out += w
                            i_col += np.random.randint(1, clen0)
                        posts[i_row] = out

    else:
        if transpose:
            if vector_info.dtype == jnp.bool_:
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
                            w = np.random.uniform(low=w_low0, high=w_high0)
                            if v:
                                posts[i_col] += w
                            i_col += np.random.randint(1, clen0)
            else:
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
                        v = vector[i_row] > 0.
                        i_col = np.random.randint(0, clen0)
                        while i_col < num_col:
                            w = np.random.uniform(low=w_low0, high=w_high0)
                            if v:
                                posts[i_col] += w
                            i_col += np.random.randint(1, clen0)

        else:
            if vector_info.dtype == jnp.bool_:
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
                            w = np.random.uniform(low=w_low0, high=w_high0)
                            if v:
                                posts[i_row] += w
                            i_row += np.random.randint(1, clen0)
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
                        v = vector[i_col] > 0.
                        i_row = np.random.randint(0, clen0)
                        while i_row < num_row:
                            w = np.random.uniform(low=w_low0, high=w_high0)
                            if v:
                                posts[i_row] += w
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
    corder: bool = True,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    w_low_warp = jaxinfo_to_warpinfo(w_low_info)
    w_high_warp = jaxinfo_to_warpinfo(w_high_info)
    clen_warp = jaxinfo_to_warpinfo(clen_info)
    vector_warp = jaxinfo_to_warpinfo(vector_info)
    seed_warp = jaxinfo_to_warpinfo(seed_info)
    out_warp = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if corder:
        if vector_info.dtype == jnp.bool_:
            @warp.kernel
            def kernel_impl(
                w_low: w_low_warp,
                w_high: w_high_warp,
                clen: clen_warp,
                vector: vector_warp,
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
                    w = warp.randf(state) * w_diff + w_low0
                    r = warp.where(vector[i_row], r + w, r)
                    i_row += warp.randi(state, 1, clen0)
                posts[i_col] = r

        else:
            @warp.kernel
            def kernel_impl(
                w_low: w_low_warp,
                w_high: w_high_warp,
                clen: clen_warp,
                vector: vector_warp,
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
                    w = warp.randf(state) * w_diff + w_low0
                    r += w * vector[i_row]
                    i_row += warp.randi(state, 1, clen0)
                posts[i_col] = r

    else:
        if vector_info.dtype == jnp.bool_:
            @warp.kernel
            def kernel_impl(
                w_low: w_low_warp,
                w_high: w_high_warp,
                clen: clen_warp,
                vector: vector_warp,
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
                if v:
                    state = warp.rand_init(seed0 + i_row)
                    i_col = warp.randi(state, 0, clen0)
                    while i_col < num_col:
                        w = warp.randf(state) * w_diff + w_low0
                        posts[i_col] += w
                        i_col += warp.randi(state, 1, clen0)
        else:
            @warp.kernel
            def kernel_impl(
                w_low: w_low_warp,
                w_high: w_high_warp,
                clen: clen_warp,
                vector: vector_warp,
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
                v = vector[i_row] > 0.
                if v:
                    state = warp.rand_init(seed0 + i_row)
                    i_col = warp.randi(state, 0, clen0)
                    while i_col < num_col:
                        w = warp.randf(state) * w_diff + w_low0
                        posts[i_col] += w
                        i_col += warp.randi(state, 1, clen0)

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
                i_rows, i_row_mask, rng, res = data
                v = jnp.where(i_row_mask, vector_ref[i_rows], False if vector_info.dtype == jnp.bool_ else 0.)
                if vector_info.dtype > jnp.bool_:
                    v = v > 0.
                w = rng.uniform(w_low, w_high)
                res = jnp.where(v, res + w, res)
                i_rows += rng.random_integers(1, clen)
                return i_rows, i_rows < num_row, rng, res

            rng = PallasLFSR88RNG(seed + i_cols)
            i_rows = rng.random_integers(0, clen)
            i_row_mask = i_rows < num_row
            out = jax.lax.while_loop(
                lambda data: jnp.sum(data[1]) > 0,
                body,
                (i_rows, i_row_mask, rng, jnp.zeros(block_size, dtype=post_ref.dtype))
            )[-1]
            post_ref[i_cols] = jnp.where(i_col_mask, out, post_ref[i_cols])

    else:
        def kernel(w_low_ref, w_high_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
            num_col = post_ref.shape[0]
            w_low = w_low_ref[0]
            w_high = w_high_ref[0]
            clen = clen_ref[0]
            seed = seed_ref[0]
            i_row = pl.program_id(0)
            v = vector_ref[i_row]

            @pl.when(v if vector_info.dtype == jnp.bool_ else v > 0.)
            def run():
                def body(data):
                    i, rng = data
                    atomic_add(post_ref, (i,), rng.uniform(w_low, w_high))
                    i += rng.random_integers(1, clen)
                    return i, rng

                rng = PallasLFSR88RNG(seed + i_row)
                jax.lax.while_loop(
                    lambda data: data[0] < num_col,
                    body,
                    (rng.random_integers(0, clen), rng)
                )

    def run(w_low, w_high, clen, vector, seed):
        fn = pl.pallas_call(
            kernel,
            grid=(pl.cdiv(dim, block_size),) if corder else (dim,),
            input_output_aliases={5: 0},
            out_shape=kwargs['outs']
        )
        placeholder = jnp.zeros(kwargs['outs'][0].shape, kwargs['outs'][0].dtype)
        return fn(w_low, w_high, clen, vector, seed, placeholder)

    return run


def _jitumv_jvp_v(v_dot, w_low, w_high, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    return jitumv_p_call(w_low, w_high, clen, v_dot, seed, shape=shape, transpose=transpose, corder=corder)


def _jitumv_jvp_wloc(w_dot, w_low, w_high, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    return binary_jitumv_p_call(w_dot, w_high, clen, vector, seed, shape=shape, transpose=transpose, corder=corder)


def _jitumv_jvp_wscale(w_dot, w_low, w_high, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    return binary_jitumv_p_call(w_low, w_dot, clen, vector, seed, shape=shape, transpose=transpose, corder=corder)


def _jitumv_transpose_rules(
    ct,
    w_low,
    w_high,
    clen,
    vector,
    seed,
    *,
    shape,
    transpose,
    corder,
    **kwargs
):
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)
    assert not ad.is_undefined_primal(w_low)
    assert not ad.is_undefined_primal(w_high)

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
    else:
        raise NotImplementedError(
            f"Transpose rule for {ct} not implemented "
            f"for event-driven COO matrix-vector product."
        )


def _jitumv_batching(
    args,
    axes,
    **kwargs
):
    if tuple(axes) == (None, None, None, 0, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = binary_jitumm_p_call(
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
        r = binary_jitumm_p_call(
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
        return general_batching_rule(binary_jitumv_p, args, axes, **kwargs)


def _binary_jitumv_benchmark_data(*, platform):
    import numpy as _np
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for corder in (True, False):
            for bool_event in (True, False):
                w_low = jnp.zeros(1, dtype=dtype)
                w_high = jnp.ones(1, dtype=dtype)
                clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
                v_size = n_post if not transpose else n_pre
                if bool_event:
                    vector = jnp.asarray(_np.random.rand(v_size) > 0.5, dtype=jnp.bool_)
                else:
                    vector = jnp.asarray(_np.random.rand(v_size), dtype=dtype)
                seed = jnp.asarray(42, dtype=jnp.uint32)
                name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'},{'bool' if bool_event else 'float'}"
                configs.append(BenchmarkConfig(name, (w_low, w_high, clen, vector, seed), {
                    'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder
                }))
    return configs


def binary_jitumv_p_call(
    w_low,
    w_high,
    clen,
    vector,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    corder: bool,
    backend: Optional[str] = None,
):
    w_low = jnp.atleast_1d(w_low)
    w_high = jnp.atleast_1d(w_high)
    clen = jnp.atleast_1d(clen)
    assert jnp.issubdtype(w_low.dtype, jnp.floating), 'Weights must be a floating-point type.'
    assert w_low.dtype == w_high.dtype, "w_low and w_high must have the same dtype."

    assert len(shape) == 2, "The matrix shape should be a tuple of two integers."
    assert w_low.shape == (1,), f"The weight shape should be (1,), but got {w_low.shape}."
    assert w_high.shape == (1,), f"The weight shape should be (1,), but got {w_high.shape}."
    assert clen.shape == (1,), f"The clen shape should be (1,), but got {clen.shape}."
    assert vector.ndim == 1, f"The vector should be a 1D array, but got {vector.ndim}D."
    assert seed.shape == (1,), f"The seed shape should be (1,), but got {seed.shape}."

    if transpose:
        assert shape[0] == len(vector), f"The matrix shape and vector length do not match. {vector.shape} @ {shape}"
    else:
        assert shape[1] == len(vector), f"The matrix shape and vector length do not match. {shape} @ {vector.shape}"

    out_info = (
        jax.ShapeDtypeStruct([shape[1]], w_low.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0]], w_low.dtype)
    )

    return binary_jitumv_p(
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


binary_jitumv_p = XLACustomKernel('binary_jitumv')
binary_jitumv_p.def_numba_kernel(_jitumv_numba_kernel_generator)
binary_jitumv_p.def_warp_kernel(_jitumv_warp_kernel_generator)
binary_jitumv_p.def_pallas_kernel('gpu', _jitumv_pallas_kernel_generator)
binary_jitumv_p.def_pallas_kernel('tpu', _jitumv_pallas_kernel_generator)
binary_jitumv_p.def_jvp_rule2(_jitumv_jvp_wloc, _jitumv_jvp_wscale, None, _jitumv_jvp_v, None)
binary_jitumv_p.def_transpose_rule(_jitumv_transpose_rules)
binary_jitumv_p.def_batching_rule(_jitumv_batching)
binary_jitumv_p.def_call(binary_jitumv_p_call)
binary_jitumv_p.def_tags('jit_uniform', 'binary')
binary_jitumv_p.def_benchmark_data(_binary_jitumv_benchmark_data)


def _jitumm_numba_kernel_generator(
    B_info: jax.ShapeDtypeStruct,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    import numba

    if corder:
        if transpose:
            if B_info.dtype == jnp.bool_:
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
                        out = np.zeros(n, dtype=posts.dtype)
                        while i_k < k:
                            w = np.random.uniform(low=w_low0, high=w_high0)
                            for j in range(B.shape[1]):
                                if B[i_k, j]:
                                    out[j] += w
                            i_k += np.random.randint(1, clen0)
                        posts[i_m] = out
            else:
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
                        out = np.zeros(n, dtype=posts.dtype)
                        while i_k < k:
                            w = np.random.uniform(low=w_low0, high=w_high0)
                            for j in range(B.shape[1]):
                                if B[i_k, j] > 0.:
                                    out[j] += w
                            i_k += np.random.randint(1, clen0)
                        posts[i_m] = out

        else:
            if B_info.dtype == jnp.bool_:
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
                        out = np.zeros(n, dtype=posts.dtype)
                        while i_k < k:
                            w = np.random.uniform(low=w_low0, high=w_high0)
                            for j in range(B.shape[1]):
                                if B[i_k, j]:
                                    out[j] += w
                            i_k += np.random.randint(1, clen0)
                        posts[i_m] = out
            else:
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
                        out = np.zeros(n, dtype=posts.dtype)
                        while i_k < k:
                            w = np.random.uniform(low=w_low0, high=w_high0)
                            for j in range(B.shape[1]):
                                if B[i_k, j] > 0.:
                                    out[j] += w
                            i_k += np.random.randint(1, clen0)
                        posts[i_m] = out

    else:
        if transpose:
            if B_info.dtype == jnp.bool_:
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
                        indices = np.where(B[i_k])[0]
                        i_m = np.random.randint(0, clen0)
                        while i_m < m:
                            w = np.random.uniform(low=w_low0, high=w_high0)
                            posts[i_m, indices] += w
                            i_m += np.random.randint(1, clen0)
            else:
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
                        indices = np.where(B[i_k] > 0.)[0]
                        i_m = np.random.randint(0, clen0)
                        while i_m < m:
                            w = np.random.uniform(low=w_low0, high=w_high0)
                            posts[i_m, indices] += w
                            i_m += np.random.randint(1, clen0)

        else:
            if B_info.dtype == jnp.bool_:
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
                        indices = np.where(B[i_k])[0]
                        i_m = np.random.randint(0, clen0)
                        while i_m < m:
                            w = np.random.uniform(low=w_low0, high=w_high0)
                            posts[i_m, indices] += w
                            i_m += np.random.randint(1, clen0)
            else:
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
                        indices = np.where(B[i_k] > 0.)[0]
                        i_m = np.random.randint(0, clen0)
                        while i_m < m:
                            w = np.random.uniform(low=w_low0, high=w_high0)
                            posts[i_m, indices] += w
                            i_m += np.random.randint(1, clen0)

    def kernel(w_low, w_high, clen, B, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(w_low, w_high, clen, B, seed)

    return kernel


def _jitumm_warp_kernel_generator(
    w_low_info: jax.ShapeDtypeStruct,
    w_high_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    B_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    TILE_SIZE = B_info.shape[1]
    w_low_warp = jaxinfo_to_warpinfo(w_low_info)
    w_high_warp = jaxinfo_to_warpinfo(w_high_info)
    clen_warp = jaxinfo_to_warpinfo(clen_info)
    B_warp = jaxinfo_to_warpinfo(B_info)
    seed_warp = jaxinfo_to_warpinfo(seed_info)
    out_warp = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if corder:
        if B_info.dtype == jnp.bool_:
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

                out = warp.tile_zeros(TILE_SIZE, dtype=w_low_warp.dtype)
                i_k = warp.randi(state, 0, clen0)
                while i_k < k:
                    w = warp.randf(state) * w_diff + w_low0
                    out += warp.tile_astype(warp.tile_load(B[i_k], TILE_SIZE), dtype=w_low_warp.dtype) * w
                    i_k += warp.randi(state, 1, clen0)
                warp.tile_store(posts[i_m], out)

        else:
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

                out = warp.tile_zeros(TILE_SIZE, dtype=w_low_warp.dtype)
                i_k = warp.randi(state, 0, clen0)
                while i_k < k:
                    w = warp.randf(state) * w_diff + w_low0
                    out += warp.tile_load(B[i_k], TILE_SIZE) * w
                    i_k += warp.randi(state, 1, clen0)
                warp.tile_store(posts[i_m], out)

    else:
        if B_info.dtype == jnp.bool_:
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

                out = warp.tile_astype(warp.tile_load(B[i_k], TILE_SIZE), dtype=w_low_warp.dtype)
                i_m = warp.randi(state, 0, clen0)
                while i_m < m:
                    w = warp.randf(state) * w_diff + w_low0
                    warp.tile_atomic_add(posts[i_m], out * w)
                    i_m += warp.randi(state, 1, clen0)

        else:
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

                out = warp.tile_load(B[i_k], TILE_SIZE)
                i_m = warp.randi(state, 0, clen0)
                while i_m < m:
                    w = warp.randf(state) * w_diff + w_low0
                    warp.tile_atomic_add(posts[i_m], out * w)
                    i_m += warp.randi(state, 1, clen0)

    def kernel(w_low, w_high, clen, B, seed):
        dim = out_info.shape[0] if corder else B_info.shape[0]
        fn = jax_kernel(kernel_impl, launch_dims=[dim], num_outputs=1, output_dims={'posts': out_info.shape})
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
                    events = jnp.where(mask, B_ref[i, i_n_indices], False if B_info.dtype == jnp.bool_ else 0.)
                    if B_info.dtype == jnp.bool_:
                        out = jnp.where(events, out + w, out)
                    else:
                        out = jnp.where(events > 0., out + w, out)
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
                    events = jnp.where(mask, B_ref[i, i_n_indices], False if B_info.dtype == jnp.bool_ else 0.)
                    if B_info.dtype == jnp.bool_:
                        out = jnp.where(events, out + w, out)
                    else:
                        out = jnp.where(events > 0., out + w, out)
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
                B_block = jnp.asarray(B_block, dtype=post_ref.dtype)

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
                B_block = jnp.asarray(B_block, dtype=post_ref.dtype)

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


def _jitumm_jvp_wloc(w_dot, w_low, w_high, clen, B, seed, *, shape, transpose, corder, **kwargs):
    return binary_jitumm_p_call(w_dot, w_high, clen, B, seed, shape=shape, transpose=transpose, corder=corder)


def _jitumm_jvp_wscale(w_dot, w_low, w_high, clen, B, seed, *, shape, transpose, corder, **kwargs):
    return binary_jitumm_p_call(w_low, w_dot, clen, B, seed, shape=shape, transpose=transpose, corder=corder)


def _jitumm_jvp_B(B_dot, w_low, w_high, clen, B, seed, *, shape, transpose, corder, **kwargs):
    return jitumm_p_call(w_low, w_high, clen, B_dot, seed, shape=shape, transpose=transpose, corder=corder)


def _jitumm_transpose_rules(
    ct,
    w_low,
    w_high,
    clen,
    B,
    seed,
    *,
    shape,
    transpose,
    corder,
    **kwargs
):
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)
    assert not ad.is_undefined_primal(w_low)
    assert not ad.is_undefined_primal(w_high)

    ct = ct[0]
    if ad.is_undefined_primal(B):
        r = jitumm_p_call(
            w_low,
            w_high,
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder,
        )[0]
        return w_low, w_high, clen, r, seed
    else:
        raise NotImplementedError(
            'Transpose rules for jitc_matmat_uniform not implemented for '
            'non-undefined primals.'
        )


def _batching_axis1(args, axis=1, **kwargs):
    assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
    m, maybe_batch1, maybe_batch2 = args[3].shape
    B = args[3].reshape(m, maybe_batch1 * maybe_batch2)
    r = binary_jitumm_p_call(
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
        return general_batching_rule(binary_jitumm_p, args, axes, **kwargs)


def _binary_jitumm_benchmark_data(*, platform):
    import numpy as _np
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for corder in (True, False):
            for bool_event in (True, False):
                w_low = jnp.zeros(1, dtype=dtype)
                w_high = jnp.ones(1, dtype=dtype)
                clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
                b_rows = n_post if not transpose else n_pre
                if bool_event:
                    B = jnp.asarray(_np.random.rand(b_rows, 10) > 0.5, dtype=jnp.bool_)
                else:
                    B = jnp.asarray(_np.random.rand(b_rows, 10), dtype=dtype)
                seed = jnp.asarray(42, dtype=jnp.uint32)
                name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'},{'bool' if bool_event else 'float'}"
                configs.append(BenchmarkConfig(name, (w_low, w_high, clen, B, seed), {
                    'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder
                }))
    return configs


def binary_jitumm_p_call(
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

    return binary_jitumm_p(
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
        backend=backend,
    )


binary_jitumm_p = XLACustomKernel('binary_jitumm')
binary_jitumm_p.def_numba_kernel(_jitumm_numba_kernel_generator)
binary_jitumm_p.def_warp_kernel(_jitumm_warp_kernel_generator)
binary_jitumm_p.def_pallas_kernel('gpu', _jitumm_pallas_kernel_generator)
binary_jitumm_p.def_pallas_kernel('tpu', _jitumm_pallas_kernel_generator)
binary_jitumm_p.def_jvp_rule2(_jitumm_jvp_wloc, _jitumm_jvp_wscale, None, _jitumm_jvp_B, None)
binary_jitumm_p.def_transpose_rule(_jitumm_transpose_rules)
binary_jitumm_p.def_batching_rule(_jitumm_batching)
binary_jitumm_p.def_call(binary_jitumm_p_call)
binary_jitumm_p.def_tags('jit_uniform', 'binary')
binary_jitumm_p.def_benchmark_data(_binary_jitumm_benchmark_data)
