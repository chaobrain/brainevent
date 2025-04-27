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

from typing import Optional, Sequence

import brainunit as u
import jax
import numpy as np
from jax import numpy as jnp
from jax.interpreters import ad

from ._jitc_float_uniform_impl import float_jitc_mv_uniform_p_call, float_jitc_mm_uniform_p_call
from ._jitc_util import _initialize_seed, _initialize_conn_length
from ._typing import Kernel, Data, MatrixShape
from ._xla_custom_op import XLACustomKernel
from ._xla_custom_op_numba import NumbaKernelGenerator, numba_environ
from ._xla_custom_op_util import general_batching_rule
from ._xla_custom_op_warp import dtype_to_warp_type, WarpKernelGenerator

__all__ = [
    "event_jitc_uniform_matvec",
    "event_jitc_uniform_matmat",
]


def event_jitc_uniform_matvec(
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
    res = event_jitc_mv_uniform_p_call(
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


def event_jitc_uniform_matmat(
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
    res = event_jitc_mm_uniform_p_call(
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

def _jitc_mv_uniform_cpu_kernel_generator(
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
) -> Kernel:
    import numba  # pylint: disable=import-outside-toplevel

    if corder:
        if transpose:
            if vector_info.dtype == jnp.bool_:
                def kernel(w_low, w_high, clen, vector, seed, _, posts):
                    n_col = posts.shape[0]
                    n_row = vector.shape[0]
                    w_low0 = w_low[0]
                    w_high0 = w_high[0]
                    clen0 = clen[0]  # Connection length (inverse of connection probability)
                    seed0 = seed[0]  # Random seed
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
                def kernel(w_low, w_high, clen, vector, seed, _, posts):
                    n_col = posts.shape[0]
                    n_row = vector.shape[0]
                    w_low0 = w_low[0]
                    w_high0 = w_high[0]
                    clen0 = clen[0]  # Connection length (inverse of connection probability)
                    seed0 = seed[0]  # Random seed
                    np.random.seed(seed0)
                    for i_col in range(n_col):
                        i_row = np.random.randint(0, clen0)
                        out = np.asarray(0., dtype=posts.dtype)
                        while i_row < n_row:
                            w = np.random.uniform(low=w_low0, high=w_high0)
                            if vector[i_row] != 0.:
                                out += w
                            i_row += np.random.randint(1, clen0)
                        posts[i_col] = out

        else:
            if vector_info.dtype == jnp.bool_:
                def kernel(w_low, w_high, clen, vector, seed, _, posts):
                    num_row = posts.shape[0]
                    num_col = vector.shape[0]
                    w_low0 = w_low[0]
                    w_high0 = w_high[0]
                    seed0 = seed[0]  # Random seed for reproducible matrix generation
                    clen0 = clen[0]  # Connection length parameter (controls sparsity)
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
                def kernel(w_low, w_high, clen, vector, seed, _, posts):
                    num_row = posts.shape[0]
                    num_col = vector.shape[0]
                    w_low0 = w_low[0]
                    w_high0 = w_high[0]
                    seed0 = seed[0]  # Random seed for reproducible matrix generation
                    clen0 = clen[0]  # Connection length parameter (controls sparsity)
                    np.random.seed(seed0)
                    for i_row in range(num_row):
                        i_col = np.random.randint(0, clen0)
                        out = np.asarray(0., dtype=posts.dtype)
                        while i_col < num_col:
                            w = np.random.uniform(low=w_low0, high=w_high0)
                            if vector[i_col] != 0.:
                                out += w
                            i_col += np.random.randint(1, clen0)
                        posts[i_row] = out

    else:
        if transpose:
            if vector_info.dtype == jnp.bool_:
                def kernel(w_low, w_high, clen, vector, seed, _, posts):
                    num_col = posts.shape[0]
                    num_row = vector.shape[0]
                    w_low0 = w_low[0]
                    w_high0 = w_high[0]
                    clen0 = clen[0]  # Controls sparsity - higher values mean fewer connections
                    seed0 = seed[0]  # Random seed for reproducible matrix generation
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
                def kernel(w_low, w_high, clen, vector, seed, _, posts):
                    num_col = posts.shape[0]
                    num_row = vector.shape[0]
                    w_low0 = w_low[0]
                    w_high0 = w_high[0]
                    clen0 = clen[0]  # Controls sparsity - higher values mean fewer connections
                    seed0 = seed[0]  # Random seed for reproducible matrix generation
                    np.random.seed(seed0)
                    for i_row in range(num_row):
                        v = vector[i_row] != 0.
                        i_col = np.random.randint(0, clen0)
                        while i_col < num_col:
                            w = np.random.uniform(low=w_low0, high=w_high0)
                            if v:
                                posts[i_col] += w
                            i_col += np.random.randint(1, clen0)

        else:
            if vector_info.dtype == jnp.bool_:
                def kernel(w_low, w_high, clen, vector, seed, _, posts):
                    num_row = posts.shape[0]
                    num_col = vector.shape[0]
                    w_low0 = w_low[0]
                    w_high0 = w_high[0]
                    clen0 = clen[0]  # Controls sparsity - higher values mean fewer connections
                    seed0 = seed[0]  # Random seed for reproducible matrix generation
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
                def kernel(w_low, w_high, clen, vector, seed, _, posts):
                    num_row = posts.shape[0]
                    num_col = vector.shape[0]
                    w_low0 = w_low[0]
                    w_high0 = w_high[0]
                    clen0 = clen[0]  # Controls sparsity - higher values mean fewer connections
                    seed0 = seed[0]  # Random seed for reproducible matrix generation
                    np.random.seed(seed0)
                    for i_col in range(num_col):
                        v = vector[i_col] != 0.
                        i_row = np.random.randint(0, clen0)
                        while i_row < num_row:
                            w = np.random.uniform(low=w_low0, high=w_high0)
                            if v:
                                posts[i_row] += w
                            i_row += np.random.randint(1, clen0)
    return numba.njit(**numba_environ.setting)(kernel)


def _jitc_mv_uniform_gpu_kernel_generator(
    w_low_info: jax.ShapeDtypeStruct,
    w_high_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
) -> Kernel:
    import warp

    w_low_dtype = dtype_to_warp_type(w_low_info.dtype)
    w_high_dtype = dtype_to_warp_type(w_high_info.dtype)
    clen_dtype = dtype_to_warp_type(clen_info.dtype)
    v_dtype = dtype_to_warp_type(vector_info.dtype)
    seed_dtype = dtype_to_warp_type(seed_info.dtype)

    if corder:
        if transpose:
            if vector_info.dtype == jnp.bool_:
                def kernel(
                    w_low: warp.array1d(dtype=w_low_dtype),
                    w_high: warp.array1d(dtype=w_high_dtype),
                    clen: warp.array1d(dtype=clen_dtype),
                    vector: warp.array1d(dtype=v_dtype),
                    seed: warp.array1d(dtype=seed_dtype),
                    _: warp.array1d(dtype=w_low_dtype),
                    posts: warp.array1d(dtype=w_low_dtype),
                ):
                    num_row = vector.shape[0]
                    w_low0 = w_low[0]
                    w_high0 = w_high[0]
                    w_diff = w_high0 - w_low0
                    clen0 = clen[0]  # Connection length parameter (controls sparsity)
                    seed0 = seed[0]  # Base random seed value
                    i_col = warp.tid()
                    r = float(0.0)
                    state = warp.rand_init(seed0, i_col)
                    i_row = warp.randi(state, 0, clen0)
                    while i_row < num_row:
                        w = warp.randf(state) * w_diff + w_low0
                        if vector[i_row]:
                            r += w
                        i_row += warp.randi(state, 1, clen0)
                    posts[i_col] = r
            else:
                def kernel(
                    w_low: warp.array1d(dtype=w_low_dtype),
                    w_high: warp.array1d(dtype=w_high_dtype),
                    clen: warp.array1d(dtype=clen_dtype),
                    vector: warp.array1d(dtype=v_dtype),
                    seed: warp.array1d(dtype=seed_dtype),
                    _: warp.array1d(dtype=w_low_dtype),
                    posts: warp.array1d(dtype=w_low_dtype),
                ):
                    num_row = vector.shape[0]
                    w_low0 = w_low[0]
                    w_high0 = w_high[0]
                    w_diff = w_high0 - w_low0
                    clen0 = clen[0]  # Connection length parameter (controls sparsity)
                    seed0 = seed[0]  # Base random seed value
                    i_col = warp.tid()
                    r = float(0.0)
                    state = warp.rand_init(seed0, i_col)
                    i_row = warp.randi(state, 0, clen0)
                    while i_row < num_row:
                        w = warp.randf(state) * w_diff + w_low0
                        if vector[i_row] != 0.:
                            r += w
                        i_row += warp.randi(state, 1, clen0)
                    posts[i_col] = r

        else:
            if vector_info.dtype == jnp.bool_:
                def kernel(
                    w_low: warp.array1d(dtype=w_low_dtype),
                    w_high: warp.array1d(dtype=w_high_dtype),
                    clen: warp.array1d(dtype=clen_dtype),
                    vector: warp.array1d(dtype=v_dtype),
                    seed: warp.array1d(dtype=seed_dtype),
                    _: warp.array1d(dtype=w_low_dtype),
                    posts: warp.array1d(dtype=w_low_dtype),
                ):
                    num_col = vector.shape[0]
                    w_low0 = w_low[0]
                    w_high0 = w_high[0]
                    w_diff = w_high0 - w_low0
                    clen0 = clen[0]  # Connection length parameter (controls sparsity)
                    seed0 = seed[0]  # Base random seed value
                    i_row = warp.tid()
                    r = float(0.0)
                    state = warp.rand_init(seed0, i_row)
                    i_col = warp.randi(state, 0, clen0)
                    while i_col < num_col:
                        w = (warp.randf(state) * w_diff + w_low0)
                        if vector[i_col]:
                            r += w
                        i_col += warp.randi(state, 1, clen0)
                    posts[i_row] = r
            else:
                def kernel(
                    w_low: warp.array1d(dtype=w_low_dtype),
                    w_high: warp.array1d(dtype=w_high_dtype),
                    clen: warp.array1d(dtype=clen_dtype),
                    vector: warp.array1d(dtype=v_dtype),
                    seed: warp.array1d(dtype=seed_dtype),
                    _: warp.array1d(dtype=w_low_dtype),
                    posts: warp.array1d(dtype=w_low_dtype),
                ):
                    num_col = vector.shape[0]
                    w_low0 = w_low[0]
                    w_high0 = w_high[0]
                    w_diff = w_high0 - w_low0
                    clen0 = clen[0]  # Connection length parameter (controls sparsity)
                    seed0 = seed[0]  # Base random seed value
                    i_row = warp.tid()
                    r = float(0.0)
                    state = warp.rand_init(seed0, i_row)
                    i_col = warp.randi(state, 0, clen0)
                    while i_col < num_col:
                        w = (warp.randf(state) * w_diff + w_low0)
                        if vector[i_col] != 0.:
                            r += w
                        i_col += warp.randi(state, 1, clen0)
                    posts[i_row] = r
    else:
        if transpose:
            if vector_info.dtype == jnp.bool_:
                def kernel(
                    w_low: warp.array1d(dtype=w_low_dtype),
                    w_high: warp.array1d(dtype=w_high_dtype),
                    clen: warp.array1d(dtype=clen_dtype),
                    vector: warp.array1d(dtype=v_dtype),
                    seed: warp.array1d(dtype=seed_dtype),
                    _: warp.array1d(dtype=w_low_dtype),
                    posts: warp.array1d(dtype=w_low_dtype),
                ):
                    num_col = posts.shape[0]
                    w_low0 = w_low[0]
                    w_high0 = w_high[0]
                    w_diff = w_high0 - w_low0
                    clen0 = clen[0]  # Connection length parameter (controls sparsity)
                    seed0 = seed[0]  # Base random seed value
                    i_row = warp.tid()
                    v = vector[i_row]
                    state = warp.rand_init(seed0, i_row)
                    i_col = warp.randi(state, 0, clen0)
                    while i_col < num_col:
                        w = warp.randf(state) * w_diff + w_low0
                        if v:
                            posts[i_col] += w
                        i_col += warp.randi(state, 1, clen0)
            else:
                def kernel(
                    w_low: warp.array1d(dtype=w_low_dtype),
                    w_high: warp.array1d(dtype=w_high_dtype),
                    clen: warp.array1d(dtype=clen_dtype),
                    vector: warp.array1d(dtype=v_dtype),
                    seed: warp.array1d(dtype=seed_dtype),
                    _: warp.array1d(dtype=w_low_dtype),
                    posts: warp.array1d(dtype=w_low_dtype),
                ):
                    num_col = posts.shape[0]
                    w_low0 = w_low[0]
                    w_high0 = w_high[0]
                    w_diff = w_high0 - w_low0
                    clen0 = clen[0]  # Connection length parameter (controls sparsity)
                    seed0 = seed[0]  # Base random seed value
                    i_row = warp.tid()
                    v = vector[i_row] != 0.
                    state = warp.rand_init(seed0, i_row)
                    i_col = warp.randi(state, 0, clen0)
                    while i_col < num_col:
                        w = warp.randf(state) * w_diff + w_low0
                        if v:
                            posts[i_col] += w
                        i_col += warp.randi(state, 1, clen0)

        else:
            if vector_info.dtype == jnp.bool_:
                def kernel(
                    w_low: warp.array1d(dtype=w_low_dtype),
                    w_high: warp.array1d(dtype=w_high_dtype),
                    clen: warp.array1d(dtype=clen_dtype),
                    vector: warp.array1d(dtype=v_dtype),
                    seed: warp.array1d(dtype=seed_dtype),
                    _: warp.array1d(dtype=w_low_dtype),
                    posts: warp.array1d(dtype=w_low_dtype),
                ):
                    num_row = posts.shape[0]
                    w_low0 = w_low[0]
                    w_high0 = w_high[0]
                    w_diff = w_high0 - w_low0
                    clen0 = clen[0]  # Connection length parameter (controls sparsity)
                    seed0 = seed[0]  # Base random seed value
                    i_col = warp.tid()
                    v = vector[i_col]
                    state = warp.rand_init(seed0, i_col)
                    i_row = warp.randi(state, 0, clen0)
                    while i_row < num_row:
                        w = warp.randf(state) * w_diff + w_low0
                        if v:
                            posts[i_row] += w
                        i_row += warp.randi(state, 1, clen0)
            else:
                def kernel(
                    w_low: warp.array1d(dtype=w_low_dtype),
                    w_high: warp.array1d(dtype=w_high_dtype),
                    clen: warp.array1d(dtype=clen_dtype),
                    vector: warp.array1d(dtype=v_dtype),
                    seed: warp.array1d(dtype=seed_dtype),
                    _: warp.array1d(dtype=w_low_dtype),
                    posts: warp.array1d(dtype=w_low_dtype),
                ):
                    num_row = posts.shape[0]
                    w_low0 = w_low[0]
                    w_high0 = w_high[0]
                    w_diff = w_high0 - w_low0
                    clen0 = clen[0]  # Connection length parameter (controls sparsity)
                    seed0 = seed[0]  # Base random seed value
                    i_col = warp.tid()
                    v = vector[i_col] != 0.
                    state = warp.rand_init(seed0, i_col)
                    i_row = warp.randi(state, 0, clen0)
                    while i_row < num_row:
                        w = warp.randf(state) * w_diff + w_low0
                        if v:
                            posts[i_row] += w
                        i_row += warp.randi(state, 1, clen0)

    return warp.kernel(kernel)


def _jitc_mv_uniform_jvp_v(
    v_dot,
    w_low,
    w_high,
    clen,
    vector,
    seed,
    _,
    *,
    shape,
    transpose,
    corder,
    **kwargs
):
    return float_jitc_mv_uniform_p_call(
        w_low,
        w_high,
        clen,
        v_dot,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder
    )


def _jitc_mv_uniform_jvp_wloc(
    w_dot,
    w_low,
    w_high,
    clen,
    vector,
    seed,
    _,
    *,
    shape,
    transpose,
    corder,
    **kwargs
):
    return event_jitc_mv_uniform_p_call(
        w_dot,
        w_high,
        clen,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder
    )


def _jitc_mv_uniform_jvp_wscale(
    w_dot,
    w_low,
    w_high,
    clen,
    vector,
    seed,
    _,
    *,
    shape,
    transpose,
    corder,
    **kwargs
):
    return event_jitc_mv_uniform_p_call(
        w_low,
        w_dot,
        clen,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder
    )


def _jitc_mv_uniform_transpose_rules(
    ct,
    w_low,
    w_high,
    clen,
    vector,
    seed,
    _,
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
        r = float_jitc_mv_uniform_p_call(
            w_low,
            w_high,
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder
        )[0]
        return w_low, w_high, clen, r, seed, _
    else:
        raise NotImplementedError(
            f"Transpose rule for {ct} not implemented "
            f"for event-driven COO matrix-vector product."
        )


def _jitc_mv_uniform_batching(
    args,
    axes,
    **kwargs
):
    if tuple(axes) == (None, None, None, 0, None, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = event_jitc_mm_uniform_p_call(
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
    elif tuple(axes) == (None, None, None, 1, None, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = event_jitc_mm_uniform_p_call(
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
        return general_batching_rule(
            event_jitc_mv_uniform_p,
            args,
            axes,
            **kwargs,
        )


def event_jitc_mv_uniform_p_call(
    w_low,
    w_high,
    clen,
    vector,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    corder: bool,
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

    if transpose:
        assert shape[0] == len(vector), f"The matrix shape and vector length do not match. {vector.shape} @ {shape}"
    else:
        assert shape[1] == len(vector), f"The matrix shape and vector length do not match. {shape} @ {vector.shape}"

    out_info = (
        jax.ShapeDtypeStruct([shape[1]], w_low.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0]], w_low.dtype)
    )

    return event_jitc_mv_uniform_p(
        w_low,
        w_high,
        clen,
        vector,
        seed,
        jnp.zeros(out_info.shape, out_info.dtype),
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
    )


event_jitc_mv_uniform_p = XLACustomKernel(
    'event_jitc_mv_uniform',
    cpu_kernel=NumbaKernelGenerator(
        _jitc_mv_uniform_cpu_kernel_generator,
        input_output_aliases={5: 0}
    ),
    gpu_kernel=WarpKernelGenerator(
        _jitc_mv_uniform_gpu_kernel_generator,
        dim=lambda out_info, vector_info, corder, **kwargs: (out_info.shape[0] if corder else vector_info.shape[0]),
        input_output_aliases={5: 0}
    )
)
event_jitc_mv_uniform_p.defjvp(
    _jitc_mv_uniform_jvp_wloc,
    _jitc_mv_uniform_jvp_wscale,
    None,
    _jitc_mv_uniform_jvp_v,
    None,
    None
)
event_jitc_mv_uniform_p.def_transpose_rule(_jitc_mv_uniform_transpose_rules)
event_jitc_mv_uniform_p.def_batching_rule(_jitc_mv_uniform_batching)


def _jitc_mm_uniform_cpu_kernel_generator(
    B_info: jax.ShapeDtypeStruct,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
) -> Kernel:
    import numba  # pylint: disable=import-outside-toplevel

    if corder:

        if transpose:
            # JIT Matrix.T @ B
            #
            # - JIT matrix: [k, m]
            # - B: [k, n]

            if B_info.dtype == jnp.bool_:
                def kernel(w_low, w_high, clen, B, seed, _, posts):
                    m = posts.shape[0]  # Number of rows in output matrix (columns in M)
                    n = posts.shape[1]  # Number of columns in output matrix (columns in B)
                    k = B.shape[0]  # Number of rows in B (rows in M)
                    w_low0 = w_low[0]
                    w_high0 = w_high[0]
                    seed0 = seed[0]  # Random seed for reproducible matrix generation
                    clen0 = clen[0]  # Connection length parameter (controls sparsity)
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
                def kernel(w_low, w_high, clen, B, seed, _, posts):
                    m = posts.shape[0]  # Number of rows in output matrix (columns in M)
                    n = posts.shape[1]  # Number of columns in output matrix (columns in B)
                    k = B.shape[0]  # Number of rows in B (rows in M)
                    w_low0 = w_low[0]
                    w_high0 = w_high[0]
                    seed0 = seed[0]  # Random seed for reproducible matrix generation
                    clen0 = clen[0]  # Connection length parameter (controls sparsity)
                    np.random.seed(seed0)
                    for i_m in range(m):
                        i_k = np.random.randint(0, clen0)
                        out = np.zeros(n, dtype=posts.dtype)
                        while i_k < k:
                            w = np.random.uniform(low=w_low0, high=w_high0)
                            for j in range(B.shape[1]):
                                if B[i_k, j] != 0.:
                                    out[j] += w
                            i_k += np.random.randint(1, clen0)
                        posts[i_m] = out

        else:
            # JIT Matrix @ B
            #
            # - JIT matrix: [m, k]
            # - B: [k, n]

            if B_info.dtype == jnp.bool_:
                def kernel(w_low, w_high, clen, B, seed, _, posts):
                    m = posts.shape[0]  # Number of rows in output matrix (rows in M)
                    n = posts.shape[1]  # Number of columns in output matrix (columns in B)
                    k = B.shape[0]  # Number of rows in B (columns in M)
                    w_low0 = w_low[0]
                    w_high0 = w_high[0]
                    seed0 = seed[0]  # Random seed for reproducible matrix generation
                    clen0 = clen[0]  # Connection length parameter (controls sparsity)
                    np.random.seed(seed0)  # Initialize random number generator with seed for reproducibility
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
                def kernel(w_low, w_high, clen, B, seed, _, posts):
                    m = posts.shape[0]  # Number of rows in output matrix (rows in M)
                    n = posts.shape[1]  # Number of columns in output matrix (columns in B)
                    k = B.shape[0]  # Number of rows in B (columns in M)
                    w_low0 = w_low[0]
                    w_high0 = w_high[0]
                    seed0 = seed[0]  # Random seed for reproducible matrix generation
                    clen0 = clen[0]  # Connection length parameter (controls sparsity)
                    np.random.seed(seed0)  # Initialize random number generator with seed for reproducibility
                    for i_m in range(m):
                        i_k = np.random.randint(0, clen0)
                        out = np.zeros(n, dtype=posts.dtype)
                        while i_k < k:
                            w = np.random.uniform(low=w_low0, high=w_high0)
                            for j in range(B.shape[1]):
                                if B[i_k, j] != 0.:
                                    out[j] += w
                            i_k += np.random.randint(1, clen0)
                        posts[i_m] = out

    else:
        if transpose:
            # JIT Matrix.T @ B
            #
            # - JIT matrix: [k, m]
            # - B: [k, n]

            if B_info.dtype == jnp.bool_:
                def kernel(w_low, w_high, clen, B, seed, _, posts):
                    m = posts.shape[0]  # Number of rows in output matrix (columns in M)
                    k = B.shape[0]  # Number of rows in B (rows in M)
                    w_low0 = w_low[0]
                    w_high0 = w_high[0]
                    seed0 = seed[0]  # Random seed for reproducible matrix generation
                    clen0 = clen[0]  # Connection length parameter (controls sparsity)
                    np.random.seed(seed0)  # Initialize random number generator with seed
                    for i_k in range(k):
                        indices = np.where(B[i_k])[0]
                        i_m = np.random.randint(0, clen0)
                        while i_m < m:
                            w = np.random.uniform(low=w_low0, high=w_high0)
                            posts[i_m, indices] += w
                            i_m += np.random.randint(1, clen0)
            else:
                def kernel(w_low, w_high, clen, B, seed, _, posts):
                    m = posts.shape[0]  # Number of rows in output matrix (columns in M)
                    k = B.shape[0]  # Number of rows in B (rows in M)
                    w_low0 = w_low[0]
                    w_high0 = w_high[0]
                    seed0 = seed[0]  # Random seed for reproducible matrix generation
                    clen0 = clen[0]  # Connection length parameter (controls sparsity)
                    np.random.seed(seed0)  # Initialize random number generator with seed
                    for i_k in range(k):
                        indices = np.where(B[i_k] != 0.)[0]
                        i_m = np.random.randint(0, clen0)
                        while i_m < m:
                            w = np.random.uniform(low=w_low0, high=w_high0)
                            posts[i_m, indices] += w
                            i_m += np.random.randint(1, clen0)

        else:
            # JIT Matrix @ B
            #
            # - JIT matrix: [m, k]
            # - B: [k, n]

            if B_info.dtype == jnp.bool_:
                def kernel(w_low, w_high, clen, B, seed, _, posts):
                    m = posts.shape[0]  # Number of rows in output matrix (rows in M)
                    k = B.shape[0]  # Number of rows in B (columns in M)
                    w_low0 = w_low[0]
                    w_high0 = w_high[0]
                    seed0 = seed[0]  # Random seed for reproducible matrix generation
                    clen0 = clen[0]  # Connection length parameter (controls sparsity)
                    np.random.seed(seed0)  # Initialize random number generator with seed
                    for i_k in range(k):
                        indices = np.where(B[i_k])[0]
                        i_m = np.random.randint(0, clen0)
                        while i_m < m:
                            w = np.random.uniform(low=w_low0, high=w_high0)
                            posts[i_m, indices] += w
                            i_m += np.random.randint(1, clen0)
            else:
                def kernel(w_low, w_high, clen, B, seed, _, posts):
                    m = posts.shape[0]  # Number of rows in output matrix (rows in M)
                    k = B.shape[0]  # Number of rows in B (columns in M)
                    w_low0 = w_low[0]
                    w_high0 = w_high[0]
                    seed0 = seed[0]  # Random seed for reproducible matrix generation
                    clen0 = clen[0]  # Connection length parameter (controls sparsity)
                    np.random.seed(seed0)  # Initialize random number generator with seed
                    for i_k in range(k):
                        indices = np.where(B[i_k] != 0.)[0]
                        i_m = np.random.randint(0, clen0)
                        while i_m < m:
                            w = np.random.uniform(low=w_low0, high=w_high0)
                            posts[i_m, indices] += w
                            i_m += np.random.randint(1, clen0)

    kernel = numba.njit(**numba_environ.setting)(kernel)
    return kernel


def _jitc_mm_uniform_gpu_kernel_generator(
    w_low_info: jax.ShapeDtypeStruct,
    w_high_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    B_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    TITLE_SIZE: int,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
) -> Kernel:
    import warp

    w_low_dtype = dtype_to_warp_type(w_low_info.dtype)
    w_high_dtype = dtype_to_warp_type(w_high_info.dtype)
    clen_dtype = dtype_to_warp_type(clen_info.dtype)
    B_dtype = dtype_to_warp_type(B_info.dtype)
    seed_dtype = dtype_to_warp_type(seed_info.dtype)

    if corder:
        if transpose:
            # JIT Matrix.T @ B
            def kernel(
                w_low: warp.array1d(dtype=w_low_dtype),
                w_high: warp.array1d(dtype=w_high_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                B: warp.array2d(dtype=B_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array2d(dtype=w_low_dtype),
                posts: warp.array2d(dtype=w_low_dtype),
            ):
                k = B.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                w_diff = w_high0 - w_low0
                clen0 = clen[0]
                seed0 = seed[0]

                i_m = warp.tid()
                state = warp.rand_init(seed0, i_m)

                out = warp.tile_zeros(TITLE_SIZE, dtype=w_low_dtype)
                i_k = warp.randi(state, 0, clen0)
                while i_k < k:
                    w = warp.randf(state) * w_diff + w_low0
                    out += warp.tile_load(B[i_k], TITLE_SIZE) * w
                    i_k += warp.randi(state, 1, clen0)
                warp.tile_store(posts[i_m], out)

        else:
            # JIT Matrix @ B
            def kernel(
                w_low: warp.array1d(dtype=w_low_dtype),
                w_high: warp.array1d(dtype=w_high_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                B: warp.array2d(dtype=B_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array2d(dtype=w_low_dtype),
                posts: warp.array2d(dtype=w_low_dtype),
            ):
                k = B.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                w_diff = w_high0 - w_low0
                clen0 = clen[0]
                seed0 = seed[0]

                i_m = warp.tid()
                state = warp.rand_init(seed0, i_m)

                out = warp.tile_zeros(TITLE_SIZE, dtype=w_low_dtype)
                i_k = warp.randi(state, 0, clen0)
                while i_k < k:
                    w = warp.randf(state) * w_diff + w_low0
                    out += warp.tile_load(B[i_k], TITLE_SIZE) * w
                    i_k += warp.randi(state, 1, clen0)
                warp.tile_store(posts[i_m], out)

    else:
        if transpose:
            # JIT Matrix.T @ B
            def kernel(
                w_low: warp.array1d(dtype=w_low_dtype),
                w_high: warp.array1d(dtype=w_high_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                B: warp.array2d(dtype=B_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array2d(dtype=w_low_dtype),
                posts: warp.array2d(dtype=w_low_dtype),
            ):
                m = posts.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                w_diff = w_high0 - w_low0
                clen0 = clen[0]
                seed0 = seed[0]

                i_k = warp.tid()
                state = warp.rand_init(seed0, i_k)

                out = warp.tile_load(B[i_k], TITLE_SIZE)
                i_m = warp.randi(state, 0, clen0)
                while i_m < m:
                    w = warp.randf(state) * w_diff + w_low0
                    warp.tile_atomic_add(posts[i_m], out * w)
                    i_m += warp.randi(state, 1, clen0)


        else:
            # JIT Matrix @ B
            def kernel(
                w_low: warp.array1d(dtype=w_low_dtype),
                w_high: warp.array1d(dtype=w_high_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                B: warp.array2d(dtype=B_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array2d(dtype=w_low_dtype),
                posts: warp.array2d(dtype=w_low_dtype),
            ):
                m = posts.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                w_diff = w_high0 - w_low0
                clen0 = clen[0]
                seed0 = seed[0]

                i_k = warp.tid()
                state = warp.rand_init(seed0, i_k)

                out = warp.tile_load(B[i_k], TITLE_SIZE)
                i_m = warp.randi(state, 0, clen0)
                while i_m < m:
                    w = warp.randf(state) * w_diff + w_low0
                    warp.tile_atomic_add(posts[i_m], out * w)
                    i_m += warp.randi(state, 1, clen0)

    kernel = warp.kernel(kernel)
    return kernel


def _jitc_mm_uniform_jvp_wloc(
    w_dot,
    w_low,
    w_high,
    clen,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    corder,
    **kwargs
):
    return event_jitc_mm_uniform_p_call(
        w_dot,
        w_high,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
    )


def _jitc_mm_uniform_jvp_wscale(
    w_dot,
    w_low,
    w_high,
    clen,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    corder,
    **kwargs
):
    return event_jitc_mm_uniform_p_call(
        w_low,
        w_dot,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
    )


def _jitc_mm_uniform_jvp_B(
    B_dot,
    w_low,
    w_high,
    clen,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    corder,
    **kwargs
):
    return float_jitc_mm_uniform_p_call(
        w_low,
        w_high,
        clen,
        B_dot,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
    )


def _jitc_mm_uniform_transpose_rules(
    ct,
    w_low,
    w_high,
    clen,
    B,
    seed,
    _,
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
        r = float_jitc_mm_uniform_p_call(
            w_low,
            w_high,
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder,
        )[0]

        return w_low, w_high, clen, r, seed, _

    else:
        raise NotImplementedError(
            'Transpose rules for jitc_matmat_uniform not implemented for '
            'non-undefined primals.'
        )


def _batching_axis0(args, axes, **kwargs):
    assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
    batch_size, m, n = args[3].shape
    B = jnp.transpose(args[3], (1, 0, 2)).reshape(m, batch_size * n)
    r = event_jitc_mm_uniform_p_call(
        args[0],
        args[1],
        args[2],
        B,
        args[4],
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
        corder=kwargs['corder'],
    )
    r = jnp.reshape(r[0], [r[0].shape[0], batch_size, n])
    return [r], [1]


def _jitc_mm_uniform_batching(
    args,
    axes,
    **kwargs
):
    if tuple(axes) == (None, None, None, 0, None, None):
        return _batching_axis0(args, axes, **kwargs)

    elif tuple(axes) == (None, None, None, 1, None, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        args = list(args)
        args[3] = jnp.transpose(args[3], (1, 0, 2))
        return _batching_axis0(args, axes, **kwargs)

    elif tuple(axes) == (None, None, None, 1, None, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        args = list(args)
        args[3] = jnp.transpose(args[3], (2, 0, 1))
        return _batching_axis0(args, axes, **kwargs)

    else:
        return general_batching_rule(
            event_jitc_mm_uniform_p,
            args,
            axes,
            **kwargs,
        )


def event_jitc_mm_uniform_p_call(
    w_low,
    w_high,
    clen,
    B,
    seed,
    *,
    shape: MatrixShape,
    transpose: bool,
    corder: bool,
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

    out_info = (
        jax.ShapeDtypeStruct([shape[1], B.shape[1]], w_low.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0], B.shape[1]], w_low.dtype)
    )

    return event_jitc_mm_uniform_p(
        w_low,
        w_high,
        clen,
        B,
        seed,
        jnp.zeros(out_info.shape, out_info.dtype),
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
        TITLE_SIZE=B.shape[1],  # Assuming B is [k, n], we want to process n columns at once
    )


event_jitc_mm_uniform_p = XLACustomKernel(
    'event_jitc_mm_uniform',
    cpu_kernel=NumbaKernelGenerator(
        _jitc_mm_uniform_cpu_kernel_generator,
        input_output_aliases={5: 0}
    ),
    gpu_kernel=WarpKernelGenerator(
        _jitc_mm_uniform_gpu_kernel_generator,
        tile=lambda out_info, B_info, corder, **kwargs: (out_info.shape[0] if corder else B_info.shape[0]),
        block_dim=256,
        input_output_aliases={5: 0}
    )
)
event_jitc_mm_uniform_p.defjvp(
    _jitc_mm_uniform_jvp_wloc,
    _jitc_mm_uniform_jvp_wscale,
    None,
    _jitc_mm_uniform_jvp_B,
    None,
    None
)
event_jitc_mm_uniform_p.def_transpose_rule(_jitc_mm_uniform_transpose_rules)
event_jitc_mm_uniform_p.def_batching_rule(_jitc_mm_uniform_batching)
