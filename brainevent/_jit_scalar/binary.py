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

from pathlib import Path
from typing import Optional, Sequence
import warnings

import brainunit as u
import jax
import numpy as np
from jax import numpy as jnp
from jax.interpreters import ad

from brainevent._data import _initialize_seed, _initialize_conn_length
from brainevent._misc import namescope
from brainevent._numba_random import get_numba_lfsr_seed, get_numba_lfsr_random_integers
from brainevent._op import XLACustomKernel, numba_kernel, general_batching_rule, BenchmarkConfig
from brainevent._op import load_cuda_file
from brainevent._typing import Data, MatrixShape
from .float import jitsmv_p_call, jitsmm_p_call, _dtype_sfx

__all__ = [
    "binary_jitsmv",
    "binary_jitsmv_p",
    "binary_jitsmm",
    "binary_jitsmm_p",
]


def _warn_corder_deprecated(corder: Optional[bool]) -> None:
    if corder is None:
        return
    warnings.warn(
        "corder is deprecated and ignored by the light JIT scalar implementation.",
        FutureWarning,
        stacklevel=3,
    )


def _normalize_chunk_size(n_cols: int, chunk_size: Optional[int], target_chunks: int) -> int:
    if chunk_size is None:
        target_chunks = int(target_chunks)
        if target_chunks <= 0:
            raise ValueError("target_chunks must be positive")
        chunk_size = max(1, (int(n_cols) + target_chunks - 1) // target_chunks)
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return chunk_size


def _light_options(kwargs):
    return {
        'chunk_size': kwargs.get('chunk_size', None),
        'target_chunks': kwargs.get('target_chunks', 4),
    }


@namescope(name="brainevent.binary_jitsmv", static_argnames=("shape", "transpose", "chunk_size", "target_chunks"))
def _binary_jitsmv_impl(
    weight: Data,
    prob: float,
    vector: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    backend: Optional[str] = None,
) -> Data:
    seed = _initialize_seed(seed)
    weight, unitd = u.split_mantissa_unit(weight)
    vector, unitv = u.split_mantissa_unit(vector)
    clen = _initialize_conn_length(prob)
    res = binary_jitsmv_p_call(
        weight,
        clen,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        chunk_size=chunk_size,
        target_chunks=target_chunks,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd * unitv)


def binary_jitsmv(
    weight: Data,
    prob: float,
    vector: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: Optional[bool] = None,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    backend: Optional[str] = None,
) -> Data:
    _warn_corder_deprecated(corder)
    return _binary_jitsmv_impl(
        weight,
        prob,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        chunk_size=chunk_size,
        target_chunks=target_chunks,
        backend=backend,
    )


binary_jitsmv.__doc__ = _binary_jitsmv_impl.__doc__


@namescope(name="brainevent.binary_jitsmm", static_argnames=("shape", "transpose", "chunk_size", "target_chunks"))
def _binary_jitsmm_impl(
    weight: Data,
    prob: float,
    B: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    backend: Optional[str] = None,
) -> Data:
    seed = _initialize_seed(seed)
    weight, unitd = u.split_mantissa_unit(weight)
    B, unitB = u.split_mantissa_unit(B)
    clen = _initialize_conn_length(prob)
    res = binary_jitsmm_p_call(
        weight,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        chunk_size=chunk_size,
        target_chunks=target_chunks,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd * unitB)


def binary_jitsmm(
    weight: Data,
    prob: float,
    B: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: Optional[bool] = None,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    backend: Optional[str] = None,
) -> Data:
    _warn_corder_deprecated(corder)
    return _binary_jitsmm_impl(
        weight,
        prob,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        chunk_size=chunk_size,
        target_chunks=target_chunks,
        backend=backend,
    )


binary_jitsmm.__doc__ = _binary_jitsmm_impl.__doc__


# Kernel generators for JIT connection SPMV

def _jitsmv_numba_kernel_generator(
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool = False,
    **kwargs
):
    import numba

    _lfsr_seed = get_numba_lfsr_seed()
    _lfsr_random_integers = get_numba_lfsr_random_integers()

    if transpose:
        if vector_info.dtype == jnp.bool_:
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, vector, seed, posts):
                posts[:] = 0.
                n_rows = vector.shape[0]
                n_cols = posts.shape[0]
                w = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                for row in range(n_rows):
                    if vector[row]:
                        state = _lfsr_seed(seed0 + row * n_cols)
                        col = _lfsr_random_integers(state, 0, clen0 - 1)
                        while col < n_cols:
                            posts[col] += w
                            col += _lfsr_random_integers(state, 1, clen0 - 1)
        else:
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, vector, seed, posts):
                posts[:] = 0.
                n_rows = vector.shape[0]
                n_cols = posts.shape[0]
                w = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                for row in range(n_rows):
                    if vector[row] > 0.:
                        state = _lfsr_seed(seed0 + row * n_cols)
                        col = _lfsr_random_integers(state, 0, clen0 - 1)
                        while col < n_cols:
                            posts[col] += w
                            col += _lfsr_random_integers(state, 1, clen0 - 1)
    else:
        if vector_info.dtype == jnp.bool_:
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, vector, seed, posts):
                n_rows = posts.shape[0]
                n_cols = vector.shape[0]
                w = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                for row in range(n_rows):
                    state = _lfsr_seed(seed0 + row * n_cols)
                    col = _lfsr_random_integers(state, 0, clen0 - 1)
                    out = np.asarray(0., dtype=posts.dtype)
                    while col < n_cols:
                        if vector[col]:
                            out += w
                        col += _lfsr_random_integers(state, 1, clen0 - 1)
                    posts[row] = out
        else:
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, vector, seed, posts):
                n_rows = posts.shape[0]
                n_cols = vector.shape[0]
                w = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                for row in range(n_rows):
                    state = _lfsr_seed(seed0 + row * n_cols)
                    col = _lfsr_random_integers(state, 0, clen0 - 1)
                    out = np.asarray(0., dtype=posts.dtype)
                    while col < n_cols:
                        if vector[col] > 0.:
                            out += w
                        col += _lfsr_random_integers(state, 1, clen0 - 1)
                    posts[row] = out

    def kernel(weight, clen, vector, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(weight, clen, vector, seed)

    return kernel


_spike_sfx = {
    np.dtype('bool'): '_bool',
    np.dtype('int8'): '_bool',
    np.dtype('float32'): '_float',
    np.dtype('float16'): '_float',
    np.dtype('float64'): '_float',
    np.dtype('bfloat16'): '_float',
}


def _binary_jitsmv_cuda_kernel(
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    shape: MatrixShape,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    **kwargs
):
    if np.dtype(kwargs['weight_info'].dtype) != np.dtype('float32'):
        raise NotImplementedError("light binary_jitsmv currently supports float32 weights only")

    load_cuda_file(
        Path(__file__).parent.joinpath('binary_jitsmv.cu'),
        name='binary_jitsmv',
    )
    event_size = int(vector_info.shape[0])
    packed_words = (event_size + 31) // 32
    packed_info = jax.ShapeDtypeStruct((packed_words,), jnp.uint32)
    chunk_size_value = _normalize_chunk_size(int(shape[1]), chunk_size, target_chunks)
    compute_name = 'binary_jitsmv.trans_f32' if transpose else 'binary_jitsmv.notrans_f32'

    def kernel(weight, clen, vector, seed):
        active = vector if vector.dtype == jnp.bool_ else (vector > 0).astype(jnp.int8)
        packed = jax.ffi.ffi_call(
            'binary_jitsmv.pack_bool',
            packed_info,
        )(active)
        return jax.ffi.ffi_call(
            compute_name,
            kwargs['outs'],
        )(
            weight,
            clen,
            seed,
            packed,
            vector_size=np.int32(event_size),
            chunk_size=np.int32(chunk_size_value),
        )

    return kernel


def _jitsmv_jvp_v(v_dot, weight, clen, vector, seed, *, shape, transpose, **kwargs):
    return jitsmv_p_call(
        weight, clen, v_dot, seed,
        shape=shape,
        transpose=transpose,
        **_light_options(kwargs),
        backend=kwargs['backend'],
    )


def _jitsmv_jvp_weight(w_dot, weight, clen, vector, seed, *, shape, transpose, **kwargs):
    return binary_jitsmv_p_call(
        w_dot, clen, vector, seed,
        shape=shape,
        transpose=transpose,
        **_light_options(kwargs),
        backend=kwargs['backend'],
    )


def _jitsmv_transpose_rules(ct, weight, clen, vector, seed, *, shape, transpose, **kwargs):
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)

    ct = ct[0]
    if ad.is_undefined_primal(vector):
        r = jitsmv_p_call(
            weight,
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            **_light_options(kwargs),
            backend=kwargs['backend'],
        )[0]
        return weight, clen, r, seed
    elif ad.is_undefined_primal(weight):
        ones = jnp.ones((1,), dtype=ct.dtype)
        basis = binary_jitsmv_p_call(
            ones,
            clen,
            vector,
            seed,
            shape=shape,
            transpose=transpose,
            **_light_options(kwargs),
            backend=kwargs['backend'],
        )[0]
        dweight = jnp.expand_dims(jnp.sum(ct * basis), axis=0)
        return dweight, clen, vector, seed
    else:
        raise NotImplementedError(
            f"Transpose rule for {ct} not implemented "
            f"for event-driven COO matrix-vector product."
        )


def _jitsmv_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, 0, None):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = binary_jitsmm_p_call(
            args[0],
            args[1],
            args[2].T,
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            **_light_options(kwargs),
            backend=kwargs['backend'],
        )
        return r, [1]
    elif tuple(axes) == (None, None, 1, None):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = binary_jitsmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            **_light_options(kwargs),
            backend=kwargs['backend'],
        )
        return r, [1]
    else:
        return general_batching_rule(binary_jitsmv_p, args, axes, **kwargs)


def _binary_jitsmv_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for bool_event in (True, False):
            weight = jnp.ones(1, dtype=dtype)
            clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
            v_size = n_post if not transpose else n_pre
            if bool_event:
                vector = jnp.asarray(np.random.rand(v_size) > 0.5, dtype=jnp.bool_)
            else:
                vector = jnp.asarray(np.random.rand(v_size), dtype=dtype)
            seed = jnp.asarray(42, dtype=jnp.uint32)
            name = f"{'T' if transpose else 'NT'},{'bool' if bool_event else 'float'}"
            configs.append(BenchmarkConfig(name, (weight, clen, vector, seed), {
                'shape': (n_pre, n_post), 'transpose': transpose
            }))
    return configs


def binary_jitsmv_p_call(
    weight,
    clen,
    vector,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    backend: Optional[str] = None,
):
    weight = jnp.atleast_1d(weight)
    clen = jnp.atleast_1d(clen)
    seed = jnp.atleast_1d(seed)
    assert jnp.issubdtype(weight.dtype, jnp.floating), 'Weights must be a floating-point type.'
    if np.dtype(weight.dtype) != np.dtype('float32'):
        raise NotImplementedError("light binary_jitsmv currently supports float32 weights only")

    assert len(shape) == 2, "The matrix shape should be a tuple of two integers."
    assert weight.shape == (1,), f"The weight shape should be (1,), but got {weight.shape}."
    assert clen.shape == (1,), f"The clen shape should be (1,), but got {clen.shape}."
    assert vector.ndim == 1, f"The vector should be a 1D array, but got {vector.ndim}D."
    assert seed.shape == (1,), f"The seed shape should be (1,), but got {seed.shape}."

    if transpose:
        assert shape[0] == len(vector), f"The matrix shape and vector length do not match. {vector.shape} @ {shape}"
    else:
        assert shape[1] == len(vector), f"The matrix shape and vector length do not match. {shape} @ {vector.shape}"

    chunk_size_value = _normalize_chunk_size(int(shape[1]), chunk_size, target_chunks)
    out_info = (
        jax.ShapeDtypeStruct([shape[1]], weight.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0]], weight.dtype)
    )

    return binary_jitsmv_p(
        weight,
        clen,
        vector,
        seed,
        outs=[out_info],
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        vector_info=jax.ShapeDtypeStruct(vector.shape, vector.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        chunk_size=chunk_size_value,
        target_chunks=target_chunks,
        backend=backend,
    )


binary_jitsmv_p = XLACustomKernel('binary_jitsmv')
binary_jitsmv_p.def_cuda_raw_kernel(_binary_jitsmv_cuda_kernel, asdefault=True)
binary_jitsmv_p.def_jvp_rule2(_jitsmv_jvp_weight, None, _jitsmv_jvp_v, None)
binary_jitsmv_p.def_transpose_rule(_jitsmv_transpose_rules)
binary_jitsmv_p.def_batching_rule(_jitsmv_batching)
binary_jitsmv_p.def_call(binary_jitsmv_p_call)
binary_jitsmv_p.def_tags('jit_scalar', 'binary')
binary_jitsmv_p.def_benchmark_data(_binary_jitsmv_benchmark_data)


def _jitsmm_numba_kernel_generator(
    B_info: jax.ShapeDtypeStruct,
    transpose: bool = False,
    **kwargs
):
    import numba

    _lfsr_seed = get_numba_lfsr_seed()
    _lfsr_random_integers = get_numba_lfsr_random_integers()

    if transpose:
        if B_info.dtype == jnp.bool_:
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, B, seed, posts):
                posts[:] = 0.
                n_rows = B.shape[0]
                n_cols = posts.shape[0]
                n_batch = B.shape[1]
                w = weight[0]
                seed0 = seed[0]
                clen0 = clen[0]
                for row in range(n_rows):
                    state = _lfsr_seed(seed0 + row * n_cols)
                    col = _lfsr_random_integers(state, 0, clen0 - 1)
                    while col < n_cols:
                        for j in range(n_batch):
                            if B[row, j]:
                                posts[col, j] += w
                        col += _lfsr_random_integers(state, 1, clen0 - 1)
        else:
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, B, seed, posts):
                posts[:] = 0.
                n_rows = B.shape[0]
                n_cols = posts.shape[0]
                n_batch = B.shape[1]
                w = weight[0]
                seed0 = seed[0]
                clen0 = clen[0]
                for row in range(n_rows):
                    state = _lfsr_seed(seed0 + row * n_cols)
                    col = _lfsr_random_integers(state, 0, clen0 - 1)
                    while col < n_cols:
                        for j in range(n_batch):
                            if B[row, j] > 0.:
                                posts[col, j] += w
                        col += _lfsr_random_integers(state, 1, clen0 - 1)
    else:
        if B_info.dtype == jnp.bool_:
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, B, seed, posts):
                n_rows = posts.shape[0]
                n_cols = B.shape[0]
                n_batch = B.shape[1]
                w = weight[0]
                seed0 = seed[0]
                clen0 = clen[0]
                for row in range(n_rows):
                    state = _lfsr_seed(seed0 + row * n_cols)
                    col = _lfsr_random_integers(state, 0, clen0 - 1)
                    out = np.zeros(n_batch, dtype=posts.dtype)
                    while col < n_cols:
                        for j in range(n_batch):
                            if B[col, j]:
                                out[j] += w
                        col += _lfsr_random_integers(state, 1, clen0 - 1)
                    posts[row] = out
        else:
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, B, seed, posts):
                n_rows = posts.shape[0]
                n_cols = B.shape[0]
                n_batch = B.shape[1]
                w = weight[0]
                seed0 = seed[0]
                clen0 = clen[0]
                for row in range(n_rows):
                    state = _lfsr_seed(seed0 + row * n_cols)
                    col = _lfsr_random_integers(state, 0, clen0 - 1)
                    out = np.zeros(n_batch, dtype=posts.dtype)
                    while col < n_cols:
                        for j in range(n_batch):
                            if B[col, j] > 0.:
                                out[j] += w
                        col += _lfsr_random_integers(state, 1, clen0 - 1)
                    posts[row] = out

    def kernel(weight, clen, B, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(weight, clen, B, seed)

    return kernel


def _binary_jitsmm_cuda_kernel(
    B_info: jax.ShapeDtypeStruct,
    transpose: bool,
    shape: MatrixShape,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    **kwargs
):
    if np.dtype(kwargs['weight_info'].dtype) != np.dtype('float32'):
        raise NotImplementedError("light binary_jitsmm currently supports float32 weights only")
    if int(B_info.shape[1]) > 32:
        raise NotImplementedError("light binary_jitsmm currently supports at most 32 columns")

    load_cuda_file(
        Path(__file__).parent.joinpath('binary_jitsmm.cu'),
        name='binary_jitsmm',
    )
    event_rows = int(B_info.shape[0])
    n_cols = int(B_info.shape[1])
    n_words = (event_rows + 31) // 32
    packed_info = jax.ShapeDtypeStruct((n_cols, n_words), jnp.uint32)
    chunk_size_value = _normalize_chunk_size(int(shape[1]), chunk_size, target_chunks)
    compute_name = 'binary_jitsmm.trans_f32' if transpose else 'binary_jitsmm.notrans_f32'

    def kernel(weight, clen, B, seed):
        active = B if B.dtype == jnp.bool_ else (B > 0).astype(jnp.int8)
        packed = jax.ffi.ffi_call(
            'binary_jitsmm.pack',
            packed_info,
        )(
            active,
            k=np.int32(event_rows),
            n=np.int32(n_cols),
            n_words=np.int32(n_words),
        )
        return jax.ffi.ffi_call(
            compute_name,
            kwargs['outs'],
        )(
            weight,
            clen,
            seed,
            packed,
            m=np.int32(shape[0]),
            k=np.int32(shape[1]),
            n=np.int32(n_cols),
            n_words=np.int32(n_words),
            chunk_size=np.int32(chunk_size_value),
        )

    return kernel


def _jitsmm_jvp_weight(w_dot, weight, clen, B, seed, *, shape, transpose, **kwargs):
    return binary_jitsmm_p_call(
        w_dot, clen, B, seed,
        shape=shape,
        transpose=transpose,
        **_light_options(kwargs),
        backend=kwargs['backend'],
    )


def _jitsmm_jvp_B(B_dot, weight, clen, B, seed, *, shape, transpose, **kwargs):
    return jitsmm_p_call(
        weight, clen, B_dot, seed,
        shape=shape,
        transpose=transpose,
        **_light_options(kwargs),
        backend=kwargs['backend'],
    )


def _jitsmm_transpose_rules(ct, weight, clen, B, seed, *, shape, transpose, **kwargs):
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)

    ct = ct[0]
    if ad.is_undefined_primal(B):
        r = jitsmm_p_call(
            weight,
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            **_light_options(kwargs),
            backend=kwargs['backend'],
        )[0]
        return weight, clen, r, seed
    elif ad.is_undefined_primal(weight):
        ones = jnp.ones((1,), dtype=ct.dtype)
        basis = binary_jitsmm_p_call(
            ones,
            clen,
            B,
            seed,
            shape=shape,
            transpose=transpose,
            **_light_options(kwargs),
            backend=kwargs['backend'],
        )[0]
        dweight = jnp.expand_dims(jnp.sum(ct * basis), axis=0)
        return dweight, clen, B, seed
    else:
        raise NotImplementedError(
            'Transpose rules for jitc_matmat_scalar not implemented for '
            'non-undefined primals.'
        )


def _batching_axis1(args, axis=1, **kwargs):
    assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
    m, maybe_batch1, maybe_batch2 = args[2].shape
    B = args[2].reshape(m, maybe_batch1 * maybe_batch2)
    r = binary_jitsmm_p_call(
        args[0],
        args[1],
        B,
        args[3],
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
        **_light_options(kwargs),
        backend=kwargs['backend'],
    )
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _jitsmm_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, 0, None):
        assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
        args = list(args)
        args[2] = jnp.transpose(args[2], (1, 0, 2))
        return _batching_axis1(args, **kwargs)

    elif tuple(axes) == (None, None, 1, None):
        return _batching_axis1(args, **kwargs)

    elif tuple(axes) == (None, None, 2, None):
        return _batching_axis1(args, axis=2, **kwargs)

    else:
        return general_batching_rule(binary_jitsmm_p, args, axes, **kwargs)


def _binary_jitsmm_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for bool_event in (True, False):
            weight = jnp.ones(1, dtype=dtype)
            clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
            b_rows = n_post if not transpose else n_pre
            if bool_event:
                B = jnp.asarray(np.random.rand(b_rows, 10) > 0.5, dtype=jnp.bool_)
            else:
                B = jnp.asarray(np.random.rand(b_rows, 10), dtype=dtype)
            seed = jnp.asarray(42, dtype=jnp.uint32)
            name = f"{'T' if transpose else 'NT'},{'bool' if bool_event else 'float'}"
            configs.append(BenchmarkConfig(name, (weight, clen, B, seed), {
                'shape': (n_pre, n_post), 'transpose': transpose
            }))
    return configs


def binary_jitsmm_p_call(
    weight,
    clen,
    B,
    seed,
    *,
    shape: MatrixShape,
    transpose: bool,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    backend: Optional[str] = None,
):
    weight = jnp.atleast_1d(weight)
    clen = jnp.atleast_1d(clen)
    seed = jnp.atleast_1d(seed)

    assert len(shape) == 2, "The matrix shape should be a tuple of two integers."
    assert B.ndim == 2, "The input matrix B should be a 2D array."
    assert seed.ndim == 1, "The seed should be a 1D array."
    assert weight.ndim == 1, "The weight should be a 1D array."
    assert clen.ndim == 1, "The clen should be a 1D array."
    assert weight.shape == (1,), "The weight should be a scalar."
    assert clen.shape == (1,), "The clen should be a scalar."
    assert seed.shape == (1,), "The seed should be a scalar."
    if B.shape[1] > 32:
        raise NotImplementedError("light binary_jitsmm currently supports at most 32 columns")
    if transpose:
        assert shape[0] == B.shape[0], f"The matrix shape and B shape do not match. {B.shape} @ {shape}"
    else:
        assert shape[1] == B.shape[0], f"The matrix shape and B shape do not match. {shape} @ {B.shape}"
    assert jnp.issubdtype(weight.dtype, jnp.floating), 'Weights must be a floating-point type.'
    if np.dtype(weight.dtype) != np.dtype('float32'):
        raise NotImplementedError("light binary_jitsmm currently supports float32 weights only")

    chunk_size_value = _normalize_chunk_size(int(shape[1]), chunk_size, target_chunks)
    out_info = (
        jax.ShapeDtypeStruct([shape[1], B.shape[1]], weight.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0], B.shape[1]], weight.dtype)
    )

    return binary_jitsmm_p(
        weight,
        clen,
        B,
        seed,
        outs=[out_info],
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        B_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        chunk_size=chunk_size_value,
        target_chunks=target_chunks,
        backend=backend,
    )


binary_jitsmm_p = XLACustomKernel('binary_jitsmm')
binary_jitsmm_p.def_cuda_raw_kernel(_binary_jitsmm_cuda_kernel, asdefault=True)
binary_jitsmm_p.def_jvp_rule2(_jitsmm_jvp_weight, None, _jitsmm_jvp_B, None)
binary_jitsmm_p.def_transpose_rule(_jitsmm_transpose_rules)
binary_jitsmm_p.def_batching_rule(_jitsmm_batching)
binary_jitsmm_p.def_call(binary_jitsmm_p_call)
binary_jitsmm_p.def_tags('jit_scalar', 'binary')
binary_jitsmm_p.def_benchmark_data(_binary_jitsmm_benchmark_data)
