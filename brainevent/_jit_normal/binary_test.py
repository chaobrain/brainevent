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

from brainevent._jit_normal.binary import (
    binary_jitnmv,
    binary_jitnmv_p,
    binary_jitnmm_p,
    binary_jitnmm_p_call,
)
from brainevent._jit_normal.float import jitn, jitnmv, jitnmm
from brainevent._jitc_matrix import _initialize_conn_length, _initialize_seed
from brainevent._test_util import allclose

platform = jax.default_backend()
JITNMV_IMPLEMENTATIONS = tuple(binary_jitnmv_p.available_backends(platform))
JITNMM_IMPLEMENTATIONS = tuple(binary_jitnmm_p.available_backends(platform))

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


JITNMV_PARAMS = _implementation_params(JITNMV_IMPLEMENTATIONS, 'binary_jitnmv')
JITNMM_PARAMS = _implementation_params(JITNMM_IMPLEMENTATIONS, 'binary_jitnmm')


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


def _call_binary_jitnmm(w_loc, w_scale, prob, matrix, seed, *, shape, transpose, corder, implementation):
    return binary_jitnmm_p_call(
        w_loc,
        w_scale,
        _initialize_conn_length(prob),
        matrix,
        _initialize_seed(seed),
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=implementation,
    )[0]


@pytest.mark.parametrize('implementation', JITNMV_PARAMS)
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('event_dtype', [bool, float])
@pytest.mark.parametrize('prob', [0.1, 0.2])
def test_binary_jitnmv_forward_matches_reference(implementation, shape, transpose, corder, event_dtype, prob):
    seed = 123
    w_loc = jnp.asarray(1.5, dtype=jnp.float32)
    w_scale = jnp.asarray(0.1, dtype=jnp.float32)
    event_size = shape[0] if transpose else shape[1]
    vector = _sample_vector(event_size, event_dtype, seed + 7)
    vector_ref = _binary_events(vector, dtype=jnp.float32)

    y = binary_jitnmv(
        w_loc,
        w_scale,
        prob,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=implementation,
    )
    y_ref = jitnmv(
        w_loc,
        w_scale,
        prob,
        vector_ref,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
    )
    assert allclose(y, y_ref, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((w_loc, w_scale, vector, vector_ref, y, y_ref))


@pytest.mark.parametrize('implementation', JITNMM_PARAMS)
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('event_dtype', [bool, float])
@pytest.mark.parametrize('prob', [0.1, 0.2])
@pytest.mark.parametrize('k', [5, 10])
def test_binary_jitnmm_forward_matches_reference(implementation, shape, transpose, corder, event_dtype, prob, k):
    seed = 123
    w_loc = jnp.asarray([1.5], dtype=jnp.float32)
    w_scale = jnp.asarray([0.1], dtype=jnp.float32)
    rows = shape[0] if transpose else shape[1]
    matrix = _sample_matrix(rows, k, event_dtype, seed + 11)
    matrix_ref = _binary_events(matrix, dtype=jnp.float32)

    y = _call_binary_jitnmm(
        w_loc,
        w_scale,
        prob,
        matrix,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        implementation=implementation,
    )
    y_ref = jitnmm(
        w_loc[0],
        w_scale[0],
        prob,
        matrix_ref,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
    )
    assert allclose(y, y_ref, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((w_loc, w_scale, matrix, matrix_ref, y, y_ref))


@pytest.mark.parametrize('implementation', JITNMV_PARAMS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
def test_binary_jitnmv_thresholds_float_events(implementation, shape, transpose, corder):
    seed = 123
    prob = 0.1
    w_loc = jnp.asarray(1.5, dtype=jnp.float32)
    w_scale = jnp.asarray(0.1, dtype=jnp.float32)
    size = shape[0] if transpose else shape[1]
    vector = _sample_vector(size, float, seed + 17)
    vector_binary = _binary_events(vector, dtype=jnp.float32)

    y_float = binary_jitnmv(
        w_loc,
        w_scale,
        prob,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=implementation,
    )
    y_binary = binary_jitnmv(
        w_loc,
        w_scale,
        prob,
        vector_binary,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=implementation,
    )
    assert allclose(y_float, y_binary, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((w_loc, w_scale, vector, vector_binary, y_float, y_binary))


@pytest.mark.parametrize('implementation', JITNMM_PARAMS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('k', [10])
def test_binary_jitnmm_thresholds_float_events(implementation, shape, transpose, corder, k):
    seed = 123
    prob = 0.1
    w_loc = jnp.asarray([1.5], dtype=jnp.float32)
    w_scale = jnp.asarray([0.1], dtype=jnp.float32)
    rows = shape[0] if transpose else shape[1]
    matrix = _sample_matrix(rows, k, float, seed + 23)
    matrix_binary = _binary_events(matrix, dtype=jnp.float32)

    y_float = _call_binary_jitnmm(
        w_loc,
        w_scale,
        prob,
        matrix,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        implementation=implementation,
    )
    y_binary = _call_binary_jitnmm(
        w_loc,
        w_scale,
        prob,
        matrix_binary,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        implementation=implementation,
    )
    assert allclose(y_float, y_binary, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((w_loc, w_scale, matrix, matrix_binary, y_float, y_binary))


@pytest.mark.parametrize('implementation', JITNMV_PARAMS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
def test_binary_jitnmv_jvp_and_vjp_match_reference(implementation, transpose, corder):
    shape = (20, 30)
    seed = 123
    prob = 0.1
    vector_size = shape[0] if transpose else shape[1]
    vector = _binary_events(_sample_vector(vector_size, float, seed + 29), dtype=jnp.float32)

    def f_binary(w_loc, w_scale, v):
        return binary_jitnmv(
            w_loc,
            w_scale,
            prob,
            v,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )

    def f_ref(w_loc, w_scale, v):
        return jitnmv(
            w_loc,
            w_scale,
            prob,
            v,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
        )

    primals = (jnp.asarray(1.5, dtype=jnp.float32), jnp.asarray(0.1, dtype=jnp.float32), vector)
    tangents = (jnp.asarray(1.0, dtype=jnp.float32), jnp.asarray(0.5, dtype=jnp.float32), jnp.ones_like(vector))

    out1, jvp1 = jax.jvp(f_binary, primals, tangents)
    out2, jvp2 = jax.jvp(f_ref, primals, tangents)
    assert allclose(out1, out2, rtol=1e-4, atol=1e-4)
    assert allclose(jvp1, jvp2, rtol=1e-4, atol=1e-4)

    def scalar_binary(w_loc, w_scale, v):
        return f_binary(w_loc, w_scale, v).sum()

    def scalar_ref(w_loc, w_scale, v):
        return f_ref(w_loc, w_scale, v).sum()

    (g_wloc1, g_wscale1, g_v1) = jax.grad(scalar_binary, argnums=(0, 1, 2))(*primals)
    (g_wloc2, g_wscale2, g_v2) = jax.grad(scalar_ref, argnums=(0, 1, 2))(*primals)
    assert allclose(g_wloc1, g_wloc2, rtol=1e-4, atol=1e-4)
    assert allclose(g_wscale1, g_wscale2, rtol=1e-4, atol=1e-4)
    assert allclose(g_v1, g_v2, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((vector, primals[0], primals[1], tangents[0], tangents[1], tangents[2],
                           out1, jvp1, out2, jvp2, g_wloc1, g_wscale1, g_v1, g_wloc2, g_wscale2, g_v2))


@pytest.mark.parametrize('implementation', JITNMM_PARAMS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
def test_binary_jitnmm_jvp_matches_reference(implementation, transpose, corder):
    shape = (20, 30)
    seed = 123
    prob = 0.1
    k = 8
    rows = shape[0] if transpose else shape[1]
    matrix = _binary_events(_sample_matrix(rows, k, float, seed + 31), dtype=jnp.float32)

    def f_binary(w_loc, w_scale, B):
        return _call_binary_jitnmm(
            jnp.atleast_1d(w_loc),
            jnp.atleast_1d(w_scale),
            prob,
            B,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            implementation=implementation,
        )

    def f_ref(w_loc, w_scale, B):
        return jitnmm(
            w_loc,
            w_scale,
            prob,
            B,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
        )

    primals = (jnp.asarray(1.5, dtype=jnp.float32), jnp.asarray(0.1, dtype=jnp.float32), matrix)
    tangents = (jnp.asarray(1.0, dtype=jnp.float32), jnp.asarray(0.5, dtype=jnp.float32), jnp.ones_like(matrix))
    out1, jvp1 = jax.jvp(f_binary, primals, tangents)
    out2, jvp2 = jax.jvp(f_ref, primals, tangents)
    assert allclose(out1, out2, rtol=1e-4, atol=1e-4)
    assert allclose(jvp1, jvp2, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((matrix, primals[0], primals[1], tangents[0], tangents[1], tangents[2],
                           out1, jvp1, out2, jvp2))


@pytest.mark.parametrize('implementation', JITNMV_PARAMS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
def test_binary_jitnmv_vmap_matches_reference(implementation, transpose, corder):
    shape = (20, 30)
    seed = 123
    prob = 0.1
    batch = 6
    event_size = shape[0] if transpose else shape[1]
    vectors = _sample_matrix(batch, event_size, float, seed + 41)

    f_binary = jax.vmap(
        lambda v: binary_jitnmv(
            jnp.asarray(1.5, dtype=jnp.float32),
            jnp.asarray(0.1, dtype=jnp.float32),
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
        lambda v: jitnmv(
            jnp.asarray(1.5, dtype=jnp.float32),
            jnp.asarray(0.1, dtype=jnp.float32),
            prob,
            _binary_events(v, dtype=jnp.float32),
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
        )
    )
    result_binary = f_binary(vectors)
    result_ref = f_ref(vectors)
    assert allclose(result_binary, result_ref, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((vectors, result_binary, result_ref))


@pytest.mark.parametrize('implementation', JITNMM_PARAMS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
def test_binary_jitnmm_vmap_matches_reference(implementation, transpose, corder):
    shape = (20, 30)
    seed = 123
    prob = 0.1
    batch = 4
    k = 7
    rows = shape[0] if transpose else shape[1]
    matrices = _sample_matrix(batch * rows, k, float, seed + 43).reshape(batch, rows, k)

    f_binary = jax.vmap(
        lambda B: _call_binary_jitnmm(
            jnp.asarray([1.5], dtype=jnp.float32),
            jnp.asarray([0.1], dtype=jnp.float32),
            prob,
            B,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            implementation=implementation,
        )
    )
    f_ref = jax.vmap(
        lambda B: jitnmm(
            jnp.asarray(1.5, dtype=jnp.float32),
            jnp.asarray(0.1, dtype=jnp.float32),
            prob,
            _binary_events(B, dtype=jnp.float32),
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
        )
    )
    result_binary = f_binary(matrices)
    result_ref = f_ref(matrices)
    assert allclose(result_binary, result_ref, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((matrices, result_binary, result_ref))


# ---- Gradient VJP: binary_jitnmv w.r.t. w_loc ----

@pytest.mark.parametrize('implementation', JITNMV_PARAMS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_jitnmv_vjp_wloc(implementation, shape, corder, transpose):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    vec_size = shape[0] if transpose else shape[1]
    vector = _binary_events(_sample_vector(vec_size, float, seed + 7), dtype=jnp.float32)
    w_loc_arr = jnp.array([w_loc])
    mask = jitn(1., 0., prob, seed, shape=shape, transpose=transpose, corder=corder)
    z_mask = jitn(0., 1., prob, seed, shape=shape, transpose=transpose, corder=corder)

    def f_fn(wl):
        return binary_jitnmv(wl, w_scale, prob, vector, seed, shape=shape, transpose=transpose, corder=corder,
                             backend=implementation).sum()

    def f_ref(wl):
        M = wl * mask + w_scale * z_mask
        return (M @ vector).sum()

    grad1 = jax.grad(f_fn)(w_loc_arr)
    grad2 = jax.grad(f_ref)(w_loc_arr)
    assert allclose(grad1, grad2, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((vector, w_loc_arr, mask, z_mask, grad1, grad2))


# ---- Gradient VJP: binary_jitnmv w.r.t. w_scale ----

@pytest.mark.parametrize('implementation', JITNMV_PARAMS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_jitnmv_vjp_wscale(implementation, shape, corder, transpose):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    vec_size = shape[0] if transpose else shape[1]
    vector = _binary_events(_sample_vector(vec_size, float, seed + 7), dtype=jnp.float32)
    w_scale_arr = jnp.array([w_scale])
    mask = jitn(1., 0., prob, seed, shape=shape, transpose=transpose, corder=corder)
    z_mask = jitn(0., 1., prob, seed, shape=shape, transpose=transpose, corder=corder)

    def f_fn(ws):
        return binary_jitnmv(w_loc, ws, prob, vector, seed, shape=shape, transpose=transpose, corder=corder,
                             backend=implementation).sum()

    def f_ref(ws):
        M = w_loc * mask + ws * z_mask
        return (M @ vector).sum()

    grad1 = jax.grad(f_fn)(w_scale_arr)
    grad2 = jax.grad(f_ref)(w_scale_arr)
    assert allclose(grad1, grad2, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((vector, w_scale_arr, mask, z_mask, grad1, grad2))


# ---- End-to-end VJP: binary_jitnmv w.r.t. w_loc with loss ----

@pytest.mark.parametrize('implementation', JITNMV_PARAMS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_jitnmv_vjp_wloc_with_loss(implementation, shape, corder, transpose):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    vec_size = shape[0] if transpose else shape[1]
    out_size = shape[1] if transpose else shape[0]
    vector = _binary_events(_sample_vector(vec_size, float, seed + 7), dtype=jnp.float32)
    target = jnp.asarray(np.random.rand(out_size))
    w_loc_arr = jnp.array([w_loc])
    mask = jitn(1., 0., prob, seed, shape=shape, transpose=transpose, corder=corder)
    z_mask = jitn(0., 1., prob, seed, shape=shape, transpose=transpose, corder=corder)

    def loss_fn(wl):
        out = binary_jitnmv(wl, w_scale, prob, vector, seed, shape=shape, transpose=transpose, corder=corder,
                            backend=implementation)
        return jnp.sum((out - target) ** 2)

    def loss_ref(wl):
        M = wl * mask + w_scale * z_mask
        out = M @ vector
        return jnp.sum((out - target) ** 2)

    grad1 = jax.grad(loss_fn)(w_loc_arr)
    grad2 = jax.grad(loss_ref)(w_loc_arr)
    assert allclose(grad1, grad2, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((vector, target, w_loc_arr, mask, z_mask, grad1, grad2))


# ---- End-to-end VJP: binary_jitnmv w.r.t. w_scale with loss ----

@pytest.mark.parametrize('implementation', JITNMV_PARAMS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_jitnmv_vjp_wscale_with_loss(implementation, shape, corder, transpose):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    vec_size = shape[0] if transpose else shape[1]
    out_size = shape[1] if transpose else shape[0]
    vector = _binary_events(_sample_vector(vec_size, float, seed + 7), dtype=jnp.float32)
    target = jnp.asarray(np.random.rand(out_size))
    w_scale_arr = jnp.array([w_scale])
    mask = jitn(1., 0., prob, seed, shape=shape, transpose=transpose, corder=corder)
    z_mask = jitn(0., 1., prob, seed, shape=shape, transpose=transpose, corder=corder)

    def loss_fn(ws):
        out = binary_jitnmv(w_loc, ws, prob, vector, seed, shape=shape, transpose=transpose, corder=corder,
                            backend=implementation)
        return jnp.sum((out - target) ** 2)

    def loss_ref(ws):
        M = w_loc * mask + ws * z_mask
        out = M @ vector
        return jnp.sum((out - target) ** 2)

    grad1 = jax.grad(loss_fn)(w_scale_arr)
    grad2 = jax.grad(loss_ref)(w_scale_arr)
    assert allclose(grad1, grad2, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((vector, target, w_scale_arr, mask, z_mask, grad1, grad2))


# ---- Gradient VJP: binary_jitnmm w.r.t. w_loc ----

@pytest.mark.parametrize('implementation', JITNMM_PARAMS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_jitnmm_vjp_wloc(implementation, shape, corder, transpose):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    k = 10
    mat_rows = shape[0] if transpose else shape[1]
    B = _binary_events(_sample_matrix(mat_rows, k, float, seed + 31), dtype=jnp.float32)
    w_loc_arr = jnp.array([w_loc])
    mask = jitn(1., 0., prob, seed, shape=shape, transpose=transpose, corder=corder)
    z_mask = jitn(0., 1., prob, seed, shape=shape, transpose=transpose, corder=corder)

    def f_fn(wl):
        return _call_binary_jitnmm(wl, jnp.array([w_scale]), prob, B, seed,
                                   shape=shape, transpose=transpose, corder=corder, implementation=implementation).sum()

    def f_ref(wl):
        M = wl * mask + w_scale * z_mask
        return (M @ B).sum()

    grad1 = jax.grad(f_fn)(w_loc_arr)
    grad2 = jax.grad(f_ref)(w_loc_arr)
    assert allclose(grad1, grad2, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((B, w_loc_arr, mask, z_mask, grad1, grad2))


# ---- Gradient VJP: binary_jitnmm w.r.t. w_scale ----

@pytest.mark.parametrize('implementation', JITNMM_PARAMS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_jitnmm_vjp_wscale(implementation, shape, corder, transpose):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    k = 10
    mat_rows = shape[0] if transpose else shape[1]
    B = _binary_events(_sample_matrix(mat_rows, k, float, seed + 31), dtype=jnp.float32)
    w_scale_arr = jnp.array([w_scale])
    mask = jitn(1., 0., prob, seed, shape=shape, transpose=transpose, corder=corder)
    z_mask = jitn(0., 1., prob, seed, shape=shape, transpose=transpose, corder=corder)

    def f_fn(ws):
        return _call_binary_jitnmm(jnp.array([w_loc]), ws, prob, B, seed,
                                   shape=shape, transpose=transpose, corder=corder, implementation=implementation).sum()

    def f_ref(ws):
        M = w_loc * mask + ws * z_mask
        return (M @ B).sum()

    grad1 = jax.grad(f_fn)(w_scale_arr)
    grad2 = jax.grad(f_ref)(w_scale_arr)
    assert allclose(grad1, grad2, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((B, w_scale_arr, mask, z_mask, grad1, grad2))


# ---- End-to-end VJP: binary_jitnmm w.r.t. w_loc with loss ----

@pytest.mark.parametrize('implementation', JITNMM_PARAMS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_jitnmm_vjp_wloc_with_loss(implementation, shape, corder, transpose):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    k = 10
    mat_rows = shape[0] if transpose else shape[1]
    out_rows = shape[1] if transpose else shape[0]
    B = _binary_events(_sample_matrix(mat_rows, k, float, seed + 31), dtype=jnp.float32)
    target = jnp.asarray(np.random.rand(out_rows, k))
    w_loc_arr = jnp.array([w_loc])
    mask = jitn(1., 0., prob, seed, shape=shape, transpose=transpose, corder=corder)
    z_mask = jitn(0., 1., prob, seed, shape=shape, transpose=transpose, corder=corder)

    def loss_fn(wl):
        out = _call_binary_jitnmm(wl, jnp.array([w_scale]), prob, B, seed,
                                  shape=shape, transpose=transpose, corder=corder, implementation=implementation)
        return jnp.sum((out - target) ** 2)

    def loss_ref(wl):
        M = wl * mask + w_scale * z_mask
        out = M @ B
        return jnp.sum((out - target) ** 2)

    grad1 = jax.grad(loss_fn)(w_loc_arr)
    grad2 = jax.grad(loss_ref)(w_loc_arr)
    assert allclose(grad1, grad2, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((B, target, w_loc_arr, mask, z_mask, grad1, grad2))


# ---- End-to-end VJP: binary_jitnmm w.r.t. w_scale with loss ----

@pytest.mark.parametrize('implementation', JITNMM_PARAMS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_jitnmm_vjp_wscale_with_loss(implementation, shape, corder, transpose):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    k = 10
    mat_rows = shape[0] if transpose else shape[1]
    out_rows = shape[1] if transpose else shape[0]
    B = _binary_events(_sample_matrix(mat_rows, k, float, seed + 31), dtype=jnp.float32)
    target = jnp.asarray(np.random.rand(out_rows, k))
    w_scale_arr = jnp.array([w_scale])
    mask = jitn(1., 0., prob, seed, shape=shape, transpose=transpose, corder=corder)
    z_mask = jitn(0., 1., prob, seed, shape=shape, transpose=transpose, corder=corder)

    def loss_fn(ws):
        out = _call_binary_jitnmm(jnp.array([w_loc]), ws, prob, B, seed,
                                  shape=shape, transpose=transpose, corder=corder, implementation=implementation)
        return jnp.sum((out - target) ** 2)

    def loss_ref(ws):
        M = w_loc * mask + ws * z_mask
        out = M @ B
        return jnp.sum((out - target) ** 2)

    grad1 = jax.grad(loss_fn)(w_scale_arr)
    grad2 = jax.grad(loss_ref)(w_scale_arr)
    assert allclose(grad1, grad2, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((B, target, w_scale_arr, mask, z_mask, grad1, grad2))
