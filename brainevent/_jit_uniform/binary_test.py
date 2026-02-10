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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Keep GPU matmul reference numerics stable (avoid TF32 drift in dense @ B checks).
if jax.default_backend() == 'gpu' and jax.config.jax_default_matmul_precision is None:
    jax.config.update('jax_default_matmul_precision', 'highest')

from brainevent._jit_uniform.binary import (
    binary_jitumv,
    binary_jitumv_p,
    binary_jitumm,
    binary_jitumm_p,
)
from brainevent._jit_uniform.float import jitumv, jitumm
from brainevent._test_util import allclose

platform = jax.default_backend()
JITUMV_IMPLEMENTATIONS = tuple(binary_jitumv_p.available_backends(platform))
JITUMM_IMPLEMENTATIONS = tuple(binary_jitumm_p.available_backends(platform))


if platform == 'cpu':
    SHAPES = ((20, 30), (100, 50))
else:
    SHAPES = ((20, 30), (100, 50), (400, 300))


def _implementation_params(implementations, op_name: str):
    if implementations:
        return [pytest.param(impl, id=impl) for impl in implementations]
    return [
        pytest.param(
            None,
            marks=pytest.mark.skip(reason=f'No {op_name} implementations on platform={platform}'),
            id=f'no-{op_name}',
        )
    ]


JITUMV_PARAMS = _implementation_params(JITUMV_IMPLEMENTATIONS, 'binary_jitumv')
JITUMM_PARAMS = _implementation_params(JITUMM_IMPLEMENTATIONS, 'binary_jitumm')


def _binary_events(x, dtype=jnp.float32):
    return jnp.asarray(jnp.asarray(x) > 0, dtype=dtype)


def _sample_vector(size: int, event_dtype, seed: int):
    rng = np.random.RandomState(seed)
    if event_dtype is bool:
        return jnp.asarray(rng.rand(size) > 0.5)
    raw = rng.rand(size).astype(np.float32)
    mask = rng.rand(size) > 0.5
    return jnp.asarray(raw * mask)


def _sample_matrix(rows: int, cols: int, event_dtype, seed: int):
    rng = np.random.RandomState(seed)
    if event_dtype is bool:
        return jnp.asarray(rng.rand(rows, cols) > 0.5)
    raw = rng.rand(rows, cols).astype(np.float32)
    mask = rng.rand(rows, cols) > 0.5
    return jnp.asarray(raw * mask)


def _sample_cotangent(shape, seed: int):
    rng = np.random.RandomState(seed)
    return jnp.asarray(rng.randn(*shape).astype(np.float32))


@pytest.mark.parametrize('implementation', JITUMV_PARAMS)
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('event_dtype', [bool, float])
@pytest.mark.parametrize('prob', [0.1, 0.2])
def test_binary_jitumv_forward_matches_reference(implementation, shape, transpose, corder, event_dtype, prob):
    seed = 123
    w_low = jnp.asarray(-1.5, dtype=jnp.float32)
    w_high = jnp.asarray(1.5, dtype=jnp.float32)
    event_size = shape[0] if transpose else shape[1]
    vector = _sample_vector(event_size, event_dtype, seed + 7)
    vector_ref = _binary_events(vector, dtype=jnp.float32)

    y = binary_jitumv(
        w_low,
        w_high,
        prob,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=implementation,
    )
    y_ref = jitumv(
        w_low,
        w_high,
        prob,
        vector_ref,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=implementation,
    )
    assert allclose(y, y_ref, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((w_low, w_high, vector, vector_ref, y, y_ref))


@pytest.mark.parametrize('implementation', JITUMM_PARAMS)
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('event_dtype', [bool, float])
@pytest.mark.parametrize('prob', [0.1, 0.2])
@pytest.mark.parametrize('k', [5, 10])
def test_binary_jitumm_forward_matches_reference(implementation, shape, transpose, corder, event_dtype, prob, k):
    seed = 123
    w_low = jnp.asarray(-1.5, dtype=jnp.float32)
    w_high = jnp.asarray(1.5, dtype=jnp.float32)
    rows = shape[0] if transpose else shape[1]
    matrix = _sample_matrix(rows, k, event_dtype, seed + 11)
    matrix_ref = _binary_events(matrix, dtype=jnp.float32)

    y = binary_jitumm(
        w_low,
        w_high,
        prob,
        matrix,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=implementation,
    )
    y_ref = jitumm(
        w_low,
        w_high,
        prob,
        matrix_ref,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=implementation,
    )
    assert allclose(y, y_ref, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((w_low, w_high, matrix, matrix_ref, y, y_ref))


@pytest.mark.parametrize('implementation', JITUMV_PARAMS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
def test_binary_jitumv_thresholds_float_events(implementation, shape, transpose, corder):
    seed = 123
    prob = 0.1
    w_low = jnp.asarray(-1.5, dtype=jnp.float32)
    w_high = jnp.asarray(1.5, dtype=jnp.float32)
    size = shape[0] if transpose else shape[1]
    vector = _sample_vector(size, float, seed + 17)
    vector_binary = _binary_events(vector, dtype=jnp.float32)

    y_float = binary_jitumv(
        w_low,
        w_high,
        prob,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=implementation,
    )
    y_binary = binary_jitumv(
        w_low,
        w_high,
        prob,
        vector_binary,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=implementation,
    )
    assert allclose(y_float, y_binary, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((w_low, w_high, vector, vector_binary, y_float, y_binary))


@pytest.mark.parametrize('implementation', JITUMM_PARAMS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('k', [10])
def test_binary_jitumm_thresholds_float_events(implementation, shape, transpose, corder, k):
    seed = 123
    prob = 0.1
    w_low = jnp.asarray(-1.5, dtype=jnp.float32)
    w_high = jnp.asarray(1.5, dtype=jnp.float32)
    rows = shape[0] if transpose else shape[1]
    matrix = _sample_matrix(rows, k, float, seed + 23)
    matrix_binary = _binary_events(matrix, dtype=jnp.float32)

    y_float = binary_jitumm(
        w_low,
        w_high,
        prob,
        matrix,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=implementation,
    )
    y_binary = binary_jitumm(
        w_low,
        w_high,
        prob,
        matrix_binary,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=implementation,
    )
    assert allclose(y_float, y_binary, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((w_low, w_high, matrix, matrix_binary, y_float, y_binary))


@pytest.mark.parametrize('implementation', JITUMV_PARAMS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
def test_binary_jitumv_jvp_and_vjp_match_reference(implementation, transpose, corder):
    shape = (20, 30)
    seed = 123
    prob = 0.1
    vector_size = shape[0] if transpose else shape[1]
    vector = _binary_events(_sample_vector(vector_size, float, seed + 29), dtype=jnp.float32)

    def f_binary(wl, wh, v):
        return binary_jitumv(
            wl,
            wh,
            prob,
            v,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )

    def f_ref(wl, wh, v):
        return jitumv(
            wl,
            wh,
            prob,
            v,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )

    primals = (
        jnp.asarray(-1.5, dtype=jnp.float32),
        jnp.asarray(1.5, dtype=jnp.float32),
        vector,
    )
    tangents = (
        jnp.asarray(0.5, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.ones_like(vector),
    )

    out1, jvp1 = jax.jvp(f_binary, primals, tangents)
    out2, jvp2 = jax.jvp(f_ref, primals, tangents)
    assert allclose(out1, out2, rtol=1e-2, atol=1e-2)
    assert allclose(jvp1, jvp2, rtol=1e-2, atol=1e-2)

    g_v1 = jax.grad(lambda v: f_binary(primals[0], primals[1], v).sum())(primals[2])
    g_v2 = jax.grad(lambda v: f_ref(primals[0], primals[1], v).sum())(primals[2])
    assert allclose(g_v1, g_v2, rtol=1e-2, atol=1e-2)
    jax.block_until_ready(
        (vector, primals[0], primals[1], tangents[0], tangents[1], tangents[2], out1, jvp1, out2, jvp2, g_v1, g_v2))


@pytest.mark.parametrize('implementation', JITUMM_PARAMS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
def test_binary_jitumm_jvp_matches_reference(implementation, transpose, corder):
    shape = (20, 30)
    seed = 123
    prob = 0.1
    k = 8
    rows = shape[0] if transpose else shape[1]
    matrix = _binary_events(_sample_matrix(rows, k, float, seed + 31), dtype=jnp.float32)

    def f_binary(wl, wh, B):
        return binary_jitumm(
            wl,
            wh,
            prob,
            B,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )

    def f_ref(wl, wh, B):
        return jitumm(
            wl,
            wh,
            prob,
            B,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )

    primals = (
        jnp.asarray(-1.5, dtype=jnp.float32),
        jnp.asarray(1.5, dtype=jnp.float32),
        matrix,
    )
    tangents = (
        jnp.asarray(0.5, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.ones_like(matrix),
    )
    out1, jvp1 = jax.jvp(f_binary, primals, tangents)
    out2, jvp2 = jax.jvp(f_ref, primals, tangents)
    assert allclose(out1, out2, rtol=1e-2, atol=1e-2)
    assert allclose(jvp1, jvp2, rtol=1e-2, atol=1e-2)

    g_B1 = jax.grad(lambda B: f_binary(primals[0], primals[1], B).sum())(primals[2])
    g_B2 = jax.grad(lambda B: f_ref(primals[0], primals[1], B).sum())(primals[2])
    assert allclose(g_B1, g_B2, rtol=1e-2, atol=1e-2)
    jax.block_until_ready(
        (matrix, primals[0], primals[1], tangents[0], tangents[1], tangents[2], out1, jvp1, out2, jvp2, g_B1, g_B2))


@pytest.mark.parametrize('implementation', JITUMV_PARAMS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('event_dtype', [bool, float])
def test_binary_jitumv_grad_w_bounds_match_reference_and_finite_difference(
    implementation,
    transpose,
    corder,
    event_dtype,
):
    shape = (20, 30)
    seed = 123
    prob = 0.1
    eps = jnp.asarray(1e-3, dtype=jnp.float32)
    vector_size = shape[0] if transpose else shape[1]
    vector = _sample_vector(vector_size, event_dtype, seed + 47)
    cotangent = _sample_cotangent((shape[1] if transpose else shape[0],), seed + 99)

    def scalar_binary(wl, wh):
        out = binary_jitumv(
            wl,
            wh,
            prob,
            vector,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )
        return jnp.sum(out * cotangent)

    w_low = jnp.asarray(-1.5, dtype=jnp.float32)
    w_high = jnp.asarray(1.5, dtype=jnp.float32)

    grad_w_low = jax.grad(scalar_binary, argnums=0)(w_low, w_high)
    grad_w_high = jax.grad(scalar_binary, argnums=1)(w_low, w_high)

    fd_w_low = (scalar_binary(w_low + eps, w_high) - scalar_binary(w_low - eps, w_high)) / (2.0 * eps)
    fd_w_high = (scalar_binary(w_low, w_high + eps) - scalar_binary(w_low, w_high - eps)) / (2.0 * eps)

    assert allclose(grad_w_low, fd_w_low, rtol=1e-2, atol=1e-2)
    assert allclose(grad_w_high, fd_w_high, rtol=1e-2, atol=1e-2)
    jax.block_until_ready((eps, vector, cotangent, w_low, w_high, grad_w_low, grad_w_high, fd_w_low, fd_w_high))


@pytest.mark.parametrize('implementation', JITUMM_PARAMS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('event_dtype', [bool, float])
def test_binary_jitumm_grad_w_bounds_match_reference_and_finite_difference(
    implementation,
    transpose,
    corder,
    event_dtype,
):
    shape = (20, 30)
    seed = 123
    prob = 0.1
    eps = jnp.asarray(1e-3, dtype=jnp.float32)
    k = 8
    rows = shape[0] if transpose else shape[1]
    matrix = _sample_matrix(rows, k, event_dtype, seed + 53)
    out_rows = shape[1] if transpose else shape[0]
    cotangent = _sample_cotangent((out_rows, k), seed + 101)

    def scalar_binary(wl, wh):
        out = binary_jitumm(
            wl,
            wh,
            prob,
            matrix,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )
        return jnp.sum(out * cotangent)

    w_low = jnp.asarray(-1.5, dtype=jnp.float32)
    w_high = jnp.asarray(1.5, dtype=jnp.float32)

    grad_w_low = jax.grad(scalar_binary, argnums=0)(w_low, w_high)
    grad_w_high = jax.grad(scalar_binary, argnums=1)(w_low, w_high)

    fd_w_low = (scalar_binary(w_low + eps, w_high) - scalar_binary(w_low - eps, w_high)) / (2.0 * eps)
    fd_w_high = (scalar_binary(w_low, w_high + eps) - scalar_binary(w_low, w_high - eps)) / (2.0 * eps)

    assert allclose(grad_w_low, fd_w_low, rtol=1e-2, atol=1e-2)
    assert allclose(grad_w_high, fd_w_high, rtol=1e-2, atol=1e-2)
    jax.block_until_ready((eps, matrix, cotangent, w_low, w_high, grad_w_low, grad_w_high, fd_w_low, fd_w_high))


@pytest.mark.parametrize('implementation', JITUMV_PARAMS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
def test_binary_jitumv_vmap_matches_reference(implementation, transpose, corder):
    shape = (20, 30)
    seed = 123
    prob = 0.1
    batch = 6
    event_size = shape[0] if transpose else shape[1]
    vectors = _sample_matrix(batch, event_size, float, seed + 41)

    f_binary = jax.vmap(
        lambda v: binary_jitumv(
            jnp.asarray(-1.5, dtype=jnp.float32),
            jnp.asarray(1.5, dtype=jnp.float32),
            prob,
            v,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )
    )
    f_ref = jax.vmap(
        lambda v: jitumv(
            jnp.asarray(-1.5, dtype=jnp.float32),
            jnp.asarray(1.5, dtype=jnp.float32),
            prob,
            _binary_events(v, dtype=jnp.float32),
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )
    )
    y_binary = f_binary(vectors)
    y_ref = f_ref(vectors)
    assert allclose(y_binary, y_ref, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((vectors, y_binary, y_ref))


@pytest.mark.parametrize('implementation', JITUMM_PARAMS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
def test_binary_jitumm_vmap_matches_reference(implementation, transpose, corder):
    shape = (20, 30)
    seed = 123
    prob = 0.1
    batch = 4
    k = 7
    rows = shape[0] if transpose else shape[1]
    matrices = _sample_matrix(batch * rows, k, float, seed + 43).reshape(batch, rows, k)

    f_binary = jax.vmap(
        lambda B: binary_jitumm(
            jnp.asarray(-1.5, dtype=jnp.float32),
            jnp.asarray(1.5, dtype=jnp.float32),
            prob,
            B,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )
    )
    f_ref = jax.vmap(
        lambda B: jitumm(
            jnp.asarray(-1.5, dtype=jnp.float32),
            jnp.asarray(1.5, dtype=jnp.float32),
            prob,
            _binary_events(B, dtype=jnp.float32),
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )
    )
    y_binary = f_binary(matrices)
    y_ref = f_ref(matrices)
    assert allclose(y_binary, y_ref, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((matrices, y_binary, y_ref))
