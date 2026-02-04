# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._jitc_normal.binary import (
    binary_jitc_mv_normal_p_call,
    binary_jitc_mm_normal_p_call,
)
from brainevent._jitc_normal.float import (
    float_jitc_normal_matrix_p_call,
    float_jitc_mv_normal_p_call,
    float_jitc_mm_normal_p_call,
)


def _params(dtype=jnp.float32):
    w_loc = jnp.asarray([0.35], dtype=dtype)
    w_scale = jnp.asarray([0.6], dtype=dtype)
    clen = jnp.asarray([3], dtype=jnp.int32)
    seed = jnp.asarray([123], dtype=jnp.int32)
    return w_loc, w_scale, clen, seed


def _dense_matrix(w_loc, w_scale, clen, seed, shape, transpose, corder):
    return float_jitc_normal_matrix_p_call(
        w_loc,
        w_scale,
        clen,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
    )[0]


@pytest.mark.parametrize("corder", [True, False])
@pytest.mark.parametrize("transpose", [False, True])
def test_float_mv_forward_jvp_vjp(corder, transpose):
    shape = (4, 5)
    w_loc, w_scale, clen, seed = _params()
    mat = _dense_matrix(w_loc, w_scale, clen, seed, shape, transpose, corder)
    rng = np.random.default_rng(0)
    vector = jnp.asarray(rng.normal(size=(mat.shape[1],)).astype(np.float32))

    out = float_jitc_mv_normal_p_call(
        w_loc,
        w_scale,
        clen,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
    )[0]
    expected = mat @ vector
    assert jnp.allclose(out, expected, rtol=1e-5, atol=1e-5)

    v_dot = jnp.asarray(rng.normal(size=vector.shape).astype(np.float32))
    f = lambda v: float_jitc_mv_normal_p_call(
        w_loc,
        w_scale,
        clen,
        v,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
    )[0].sum()

    _, jvp_val = jax.jvp(f, (vector,), (v_dot,))
    expected_jvp = (mat @ v_dot).sum()
    assert jnp.allclose(jvp_val, expected_jvp, rtol=1e-5, atol=1e-5)

    grad = jax.grad(f)(vector)
    expected_grad = mat.T @ jnp.ones((mat.shape[0],), dtype=mat.dtype)
    assert jnp.allclose(grad, expected_grad, rtol=1e-5, atol=1e-5)


def test_float_mv_batching_axes():
    shape = (4, 5)
    w_loc, w_scale, clen, seed = _params()
    mat = _dense_matrix(w_loc, w_scale, clen, seed, shape, transpose=False, corder=True)
    rng = np.random.default_rng(1)

    vectors = jnp.asarray(rng.normal(size=(3, mat.shape[1])).astype(np.float32))
    out = jax.vmap(lambda v: float_jitc_mv_normal_p_call(
        w_loc, w_scale, clen, v, seed, shape=shape, transpose=False, corder=True
    )[0])(vectors)
    expected = (mat @ vectors.T).T
    assert jnp.allclose(out, expected, rtol=1e-5, atol=1e-5)

    vectors_t = jnp.asarray(rng.normal(size=(mat.shape[1], 3)).astype(np.float32))
    out_t = jax.vmap(
        lambda v: float_jitc_mv_normal_p_call(
            w_loc, w_scale, clen, v, seed, shape=shape, transpose=False, corder=True
        )[0],
        in_axes=1,
        out_axes=1,
    )(vectors_t)
    expected_t = mat @ vectors_t
    assert jnp.allclose(out_t, expected_t, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("corder", [True, False])
@pytest.mark.parametrize("transpose", [False, True])
def test_float_mm_forward_jvp_vjp(corder, transpose):
    shape = (4, 5)
    w_loc, w_scale, clen, seed = _params()
    mat = _dense_matrix(w_loc, w_scale, clen, seed, shape, transpose, corder)
    rng = np.random.default_rng(2)
    B = jnp.asarray(rng.normal(size=(mat.shape[1], 3)).astype(np.float32))

    out = float_jitc_mm_normal_p_call(
        w_loc,
        w_scale,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
    )[0]
    expected = mat @ B
    assert jnp.allclose(out, expected, rtol=1e-5, atol=1e-5)

    B_dot = jnp.asarray(rng.normal(size=B.shape).astype(np.float32))
    f = lambda b: float_jitc_mm_normal_p_call(
        w_loc,
        w_scale,
        clen,
        b,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
    )[0].sum()

    _, jvp_val = jax.jvp(f, (B,), (B_dot,))
    expected_jvp = (mat @ B_dot).sum()
    assert jnp.allclose(jvp_val, expected_jvp, rtol=1e-5, atol=1e-5)

    grad = jax.grad(f)(B)
    expected_grad = mat.T @ jnp.ones((mat.shape[0], B.shape[1]), dtype=mat.dtype)
    assert jnp.allclose(grad, expected_grad, rtol=1e-5, atol=1e-5)


def test_float_mm_batching_axes():
    shape = (4, 5)
    w_loc, w_scale, clen, seed = _params()
    mat = _dense_matrix(w_loc, w_scale, clen, seed, shape, transpose=False, corder=True)
    rng = np.random.default_rng(3)

    B_batch = jnp.asarray(rng.normal(size=(2, mat.shape[1], 3)).astype(np.float32))
    out = jax.vmap(lambda b: float_jitc_mm_normal_p_call(
        w_loc, w_scale, clen, b, seed, shape=shape, transpose=False, corder=True
    )[0])(B_batch)
    expected = jnp.stack([mat @ b for b in B_batch])
    assert jnp.allclose(out, expected, rtol=1e-5, atol=1e-5)

    B_batch2 = jnp.asarray(rng.normal(size=(mat.shape[1], 3, 2)).astype(np.float32))
    out2 = jax.vmap(
        lambda b: float_jitc_mm_normal_p_call(
            w_loc, w_scale, clen, b, seed, shape=shape, transpose=False, corder=True
        )[0],
        in_axes=2,
        out_axes=2,
    )(B_batch2)
    expected2 = jnp.stack([mat @ B_batch2[..., i] for i in range(B_batch2.shape[2])], axis=2)
    assert jnp.allclose(out2, expected2, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("corder", [True, False])
@pytest.mark.parametrize("transpose", [False, True])
def test_binary_mv_forward_jvp_vjp(corder, transpose):
    shape = (4, 5)
    w_loc, w_scale, clen, seed = _params()
    mat = _dense_matrix(w_loc, w_scale, clen, seed, shape, transpose, corder)
    rng = np.random.default_rng(4)
    vector = (rng.random(size=(mat.shape[1],)) < 0.5).astype(np.float32)
    vector = jnp.asarray(vector)

    out = binary_jitc_mv_normal_p_call(
        w_loc,
        w_scale,
        clen,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
    )[0]
    expected = mat @ vector
    assert jnp.allclose(out, expected, rtol=1e-5, atol=1e-5)

    v_dot = jnp.asarray(rng.normal(size=vector.shape).astype(np.float32))
    f = lambda v: binary_jitc_mv_normal_p_call(
        w_loc,
        w_scale,
        clen,
        v,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
    )[0].sum()

    _, jvp_val = jax.jvp(f, (vector,), (v_dot,))
    expected_jvp = (mat @ v_dot).sum()
    assert jnp.allclose(jvp_val, expected_jvp, rtol=1e-5, atol=1e-5)

    grad = jax.grad(f)(vector)
    expected_grad = mat.T @ jnp.ones((mat.shape[0],), dtype=mat.dtype)
    assert jnp.allclose(grad, expected_grad, rtol=1e-5, atol=1e-5)


def test_binary_mv_batching_axes():
    shape = (4, 5)
    w_loc, w_scale, clen, seed = _params()
    mat = _dense_matrix(w_loc, w_scale, clen, seed, shape, transpose=False, corder=True)
    rng = np.random.default_rng(5)

    vectors = (rng.random(size=(3, mat.shape[1])) < 0.4).astype(np.float32)
    vectors = jnp.asarray(vectors)
    out = jax.vmap(lambda v: binary_jitc_mv_normal_p_call(
        w_loc, w_scale, clen, v, seed, shape=shape, transpose=False, corder=True
    )[0])(vectors)
    expected = (mat @ vectors.T).T
    assert jnp.allclose(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("corder", [True, False])
@pytest.mark.parametrize("transpose", [False, True])
def test_binary_mm_forward(corder, transpose):
    shape = (4, 5)
    w_loc, w_scale, clen, seed = _params()
    mat = _dense_matrix(w_loc, w_scale, clen, seed, shape, transpose, corder)
    rng = np.random.default_rng(6)
    B = (rng.random(size=(mat.shape[1], 3)) < 0.4).astype(np.float32)
    B = jnp.asarray(B)

    out = binary_jitc_mm_normal_p_call(
        w_loc,
        w_scale,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
    )[0]
    expected = mat @ B
    assert jnp.allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_binary_mm_batching_axes():
    shape = (4, 5)
    w_loc, w_scale, clen, seed = _params()
    mat = _dense_matrix(w_loc, w_scale, clen, seed, shape, transpose=False, corder=True)
    rng = np.random.default_rng(7)

    B_batch = (rng.random(size=(2, mat.shape[1], 3)) < 0.5).astype(np.float32)
    B_batch = jnp.asarray(B_batch)
    out = jax.vmap(lambda b: binary_jitc_mm_normal_p_call(
        w_loc, w_scale, clen, b, seed, shape=shape, transpose=False, corder=True
    )[0])(B_batch)
    expected = jnp.stack([mat @ b for b in B_batch])
    assert jnp.allclose(out, expected, rtol=1e-5, atol=1e-5)
