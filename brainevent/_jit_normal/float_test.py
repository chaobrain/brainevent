# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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


import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._jit_normal.float import jitn, jitn_p, jitnmv, jitnmv_p, jitnmm, jitnmm_p

platform = jax.default_backend()
JITN_IMPLEMENTATIONS = tuple(jitn_p.available_backends(platform))
JITNMV_IMPLEMENTATIONS = tuple(jitnmv_p.available_backends(platform))
JITNMM_IMPLEMENTATIONS = tuple(jitnmm_p.available_backends(platform))


# ---- Forward: jitnmv (matrix @ vector, transpose=False) ----

@pytest.mark.parametrize("implementation", JITNMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitnmv_forward(implementation, shape, corder):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    vector = jnp.asarray(np.random.rand(shape[1]))
    dense = jitn(w_loc, w_scale, prob, seed, shape=shape, corder=corder)
    out = jitnmv(w_loc, w_scale, prob, vector, seed, shape=shape, corder=corder, backend=implementation)
    expected = dense @ vector
    assert jnp.allclose(out, expected, rtol=1e-4, atol=1e-4)


# ---- Forward: jitnmv (vector @ matrix, transpose=True) ----

@pytest.mark.parametrize("implementation", JITNMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitnmv_transpose_forward(implementation, shape, corder):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    vector = jnp.asarray(np.random.rand(shape[0]))
    dense = jitn(w_loc, w_scale, prob, seed, shape=shape, transpose=True, corder=corder)
    out = jitnmv(w_loc, w_scale, prob, vector, seed, shape=shape, transpose=True, corder=corder, backend=implementation)
    expected = dense @ vector
    assert jnp.allclose(out, expected, rtol=1e-4, atol=1e-4)


# ---- Forward: jitnmm (matrix @ matrix, transpose=False) ----

@pytest.mark.parametrize("implementation", JITNMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('k', [10])
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitnmm_forward(implementation, k, shape, corder):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    B = jnp.asarray(np.random.rand(shape[1], k))
    dense = jitn(w_loc, w_scale, prob, seed, shape=shape, corder=corder)
    out = jitnmm(w_loc, w_scale, prob, B, seed, shape=shape, corder=corder, backend=implementation)
    expected = dense @ B
    assert jnp.allclose(out, expected, rtol=1e-4, atol=1e-4)


# ---- Forward: jitnmm (matrix.T @ matrix, transpose=True) ----

@pytest.mark.parametrize("implementation", JITNMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('k', [10])
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitnmm_transpose_forward(implementation, k, shape, corder):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    B = jnp.asarray(np.random.rand(shape[0], k))
    dense = jitn(w_loc, w_scale, prob, seed, shape=shape, transpose=True, corder=corder)
    out = jitnmm(w_loc, w_scale, prob, B, seed, shape=shape, transpose=True, corder=corder, backend=implementation)
    expected = dense @ B
    assert jnp.allclose(out, expected, rtol=1e-4, atol=1e-4)


# ---- Gradient JVP: jitnmv ----

@pytest.mark.parametrize("implementation", JITNMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitnmv_jvp(implementation, shape, corder, transpose):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    vec_size = shape[0] if transpose else shape[1]
    x = jnp.asarray(np.random.rand(vec_size))
    dense = jitn(w_loc, w_scale, prob, seed, shape=shape, transpose=transpose, corder=corder)

    def f_fn(x):
        return jitnmv(w_loc, w_scale, prob, x, seed, shape=shape, transpose=transpose, corder=corder, backend=implementation).sum()

    def f_dense(x):
        return (dense @ x).sum()

    out1, jvp1 = jax.jvp(f_fn, (x,), (jnp.ones_like(x),))
    out2, jvp2 = jax.jvp(f_dense, (x,), (jnp.ones_like(x),))
    assert jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(jvp1, jvp2, rtol=1e-4, atol=1e-4)


# ---- Gradient VJP: jitnmv ----

@pytest.mark.parametrize("implementation", JITNMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitnmv_vjp(implementation, shape, corder, transpose):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    vec_size = shape[0] if transpose else shape[1]
    x = jnp.asarray(np.random.rand(vec_size))
    dense = jitn(w_loc, w_scale, prob, seed, shape=shape, transpose=transpose, corder=corder)

    def f_fn(x):
        return jitnmv(w_loc, w_scale, prob, x, seed, shape=shape, transpose=transpose, corder=corder, backend=implementation).sum()

    def f_dense(x):
        return (dense @ x).sum()

    out1, (vjp1,) = jax.value_and_grad(f_fn, argnums=(0,))(x)
    out2, (vjp2,) = jax.value_and_grad(f_dense, argnums=(0,))(x)
    assert jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(vjp1, vjp2, rtol=1e-4, atol=1e-4)


# ---- Gradient JVP: jitnmm ----

@pytest.mark.parametrize("implementation", JITNMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('k', [10])
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitnmm_jvp(implementation, k, shape, corder, transpose):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    mat_rows = shape[0] if transpose else shape[1]
    x = jnp.asarray(np.random.rand(mat_rows, k))
    dense = jitn(w_loc, w_scale, prob, seed, shape=shape, transpose=transpose, corder=corder)

    def f_fn(x):
        return jitnmm(w_loc, w_scale, prob, x, seed, shape=shape, transpose=transpose, corder=corder, backend=implementation).sum()

    def f_dense(x):
        return (dense @ x).sum()

    out1, jvp1 = jax.jvp(f_fn, (x,), (jnp.ones_like(x),))
    out2, jvp2 = jax.jvp(f_dense, (x,), (jnp.ones_like(x),))
    assert jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(jvp1, jvp2, rtol=1e-4, atol=1e-4)


# ---- Gradient VJP: jitnmm ----

@pytest.mark.parametrize("implementation", JITNMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('k', [10])
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitnmm_vjp(implementation, k, shape, corder, transpose):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    mat_rows = shape[0] if transpose else shape[1]
    x = jnp.asarray(np.random.rand(mat_rows, k))
    dense = jitn(w_loc, w_scale, prob, seed, shape=shape, transpose=transpose, corder=corder)

    def f_fn(x):
        return jitnmm(w_loc, w_scale, prob, x, seed, shape=shape, transpose=transpose, corder=corder, backend=implementation).sum()

    def f_dense(x):
        return (dense @ x).sum()

    out1, (vjp1,) = jax.value_and_grad(f_fn, argnums=(0,))(x)
    out2, (vjp2,) = jax.value_and_grad(f_dense, argnums=(0,))(x)
    assert jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(vjp1, vjp2, rtol=1e-4, atol=1e-4)


# ---- Batching: jitnmv over vectors ----

@pytest.mark.parametrize("implementation", JITNMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('batch_size', [10])
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitnmv_vmap_over_vectors(implementation, batch_size, shape, corder):
    w_loc, w_scale, prob, seed = 1.05, 0.1, 0.1, 123
    vectors = brainstate.random.rand(batch_size, shape[1])

    def f(vector):
        return jitnmv(w_loc, w_scale, prob, vector, seed, shape=shape, corder=corder, backend=implementation)

    results = jax.vmap(f)(vectors)
    assert results.shape == (batch_size, shape[0])

    results_loop = brainstate.transform.for_loop(f, vectors)
    assert results_loop.shape == (batch_size, shape[0])

    assert jnp.allclose(results, results_loop, rtol=1e-4, atol=1e-4)


# ---- Batching: jitnmv over vectors (transpose) ----

@pytest.mark.parametrize("implementation", JITNMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('batch_size', [10])
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitnmv_transpose_vmap_over_vectors(implementation, batch_size, shape, corder):
    w_loc, w_scale, prob, seed = 1.05, 0.1, 0.1, 123
    vectors = brainstate.random.rand(batch_size, shape[0])

    def f(vector):
        return jitnmv(w_loc, w_scale, prob, vector, seed, shape=shape, transpose=True, corder=corder, backend=implementation)

    results = jax.vmap(f)(vectors)
    assert results.shape == (batch_size, shape[1])

    results_loop = brainstate.transform.for_loop(f, vectors)
    assert results_loop.shape == (batch_size, shape[1])

    assert jnp.allclose(results, results_loop, rtol=1e-4, atol=1e-4)


# ---- Batching: jitnmv over w_loc ----

@pytest.mark.parametrize("implementation", JITNMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('batch_size', [10])
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitnmv_vmap_over_wloc(implementation, batch_size, shape, corder):
    w_scale, prob, seed = 0.1, 0.1, 123
    w_locs = brainstate.random.rand(batch_size)
    vector = brainstate.random.rand(shape[1])

    def f(w_loc):
        return jitnmv(w_loc, w_scale, prob, vector, seed, shape=shape, corder=corder, backend=implementation)

    results = jax.vmap(f)(w_locs)
    assert results.shape == (batch_size, shape[0])

    results_loop = brainstate.transform.for_loop(f, w_locs)
    assert results_loop.shape == (batch_size, shape[0])

    assert jnp.allclose(results, results_loop, rtol=1e-4, atol=1e-4)


# ---- Batching: jitnmm over matrices ----

@pytest.mark.parametrize("implementation", JITNMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('batch_size', [10])
@pytest.mark.parametrize('k', [5])
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitnmm_vmap_over_matrices(implementation, batch_size, k, shape, corder):
    w_loc, w_scale, prob, seed = 1.05, 0.1, 0.1, 123
    matrices = brainstate.random.rand(batch_size, shape[1], k)

    def f(mat):
        return jitnmm(w_loc, w_scale, prob, mat, seed, shape=shape, corder=corder, backend=implementation)

    outs = jax.vmap(f)(matrices)
    assert outs.shape == (batch_size, shape[0], k)

    outs_loop = brainstate.transform.for_loop(f, matrices)
    assert outs_loop.shape == (batch_size, shape[0], k)

    assert jnp.allclose(outs, outs_loop, rtol=1e-4, atol=1e-4)


# ---- Batching: jitnmm over matrices (transpose) ----

@pytest.mark.parametrize("implementation", JITNMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('batch_size', [10])
@pytest.mark.parametrize('k', [5])
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitnmm_transpose_vmap_over_matrices(implementation, batch_size, k, shape, corder):
    w_loc, w_scale, prob, seed = 1.05, 0.1, 0.1, 123
    matrices = brainstate.random.rand(batch_size, shape[0], k)

    def f(mat):
        return jitnmm(w_loc, w_scale, prob, mat, seed, shape=shape, transpose=True, corder=corder, backend=implementation)

    outs = jax.vmap(f)(matrices)
    assert outs.shape == (batch_size, shape[1], k)

    outs_loop = brainstate.transform.for_loop(f, matrices)
    assert outs_loop.shape == (batch_size, shape[1], k)

    assert jnp.allclose(outs, outs_loop, rtol=1e-4, atol=1e-4)


# ---- Batching: jitnmm over w_loc ----

@pytest.mark.parametrize("implementation", JITNMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('batch_size', [10])
@pytest.mark.parametrize('k', [5])
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitnmm_vmap_over_wloc(implementation, batch_size, k, shape, corder):
    w_scale, prob, seed = 0.1, 0.1, 123
    w_locs = brainstate.random.rand(batch_size)
    matrix = brainstate.random.rand(shape[1], k)

    def f(w_loc):
        return jitnmm(w_loc, w_scale, prob, matrix, seed, shape=shape, corder=corder, backend=implementation)

    results = jax.vmap(f)(w_locs)
    assert results.shape == (batch_size, shape[0], k)

    results_loop = brainstate.transform.for_loop(f, w_locs)
    assert results_loop.shape == (batch_size, shape[0], k)

    assert jnp.allclose(results, results_loop, rtol=1e-4, atol=1e-4)


# ---- Batching: jitn over w_loc ----

@pytest.mark.parametrize("implementation", JITN_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(100, 50)])
def test_jitn_vmap_over_wloc(implementation, shape):
    w_scale, prob, seed = 0.1, 0.1, 123

    def f(w_loc):
        return jitn(w_loc, w_scale, prob, seed, shape=shape, backend=implementation)

    w_locs = brainstate.random.rand(10)
    results = jax.vmap(f)(w_locs)
    assert results.shape == (10,) + shape

    results_loop = brainstate.transform.for_loop(f, w_locs)
    assert results_loop.shape == (10,) + shape

    assert jnp.allclose(results, results_loop, rtol=1e-4, atol=1e-4)


# ---- Batching: jitn over prob ----

@pytest.mark.parametrize("implementation", JITN_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(100, 50)])
def test_jitn_vmap_over_prob(implementation, shape):
    w_loc, w_scale, seed = 1.5, 0.1, 123

    def f(prob):
        return jitn(w_loc, w_scale, prob, seed, shape=shape, backend=implementation)

    probs = brainstate.random.rand(10)
    results = jax.vmap(f)(probs)
    assert results.shape == (10,) + shape

    results_loop = brainstate.transform.for_loop(f, probs)
    assert results_loop.shape == (10,) + shape

    assert jnp.allclose(results, results_loop, rtol=1e-4, atol=1e-4)


# ---- Batching: jitn over seed ----

@pytest.mark.parametrize("implementation", JITN_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(100, 50)])
def test_jitn_vmap_over_seed(implementation, shape):
    w_loc, w_scale, prob = 1.5, 0.1, 0.1

    def f(seed):
        return jitn(w_loc, w_scale, prob, seed, shape=shape, backend=implementation)

    seeds = brainstate.random.randint(0, 100000, 10)
    results = jax.vmap(f)(seeds)
    assert results.shape == (10,) + shape

    results_loop = brainstate.transform.for_loop(f, seeds)
    assert results_loop.shape == (10,) + shape

    assert jnp.allclose(results, results_loop, rtol=1e-4, atol=1e-4)


# ---- Gradient VJP: jitnmv w.r.t. w_loc ----

@pytest.mark.parametrize("implementation", JITNMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitnmv_vjp_wloc(implementation, shape, corder, transpose):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    vec_size = shape[0] if transpose else shape[1]
    vector = jnp.asarray(np.random.rand(vec_size))
    w_loc_arr = jnp.array([w_loc])
    # Build fixed dense components: mask and Z*mask
    mask = jitn(1., 0., prob, seed, shape=shape, transpose=transpose, corder=corder)
    z_mask = jitn(0., 1., prob, seed, shape=shape, transpose=transpose, corder=corder)

    def f_fn(wl):
        return jitnmv(wl, w_scale, prob, vector, seed, shape=shape, transpose=transpose, corder=corder, backend=implementation).sum()

    def f_ref(wl):
        M = wl * mask + w_scale * z_mask
        return (M @ vector).sum()

    grad1 = jax.grad(f_fn)(w_loc_arr)
    grad2 = jax.grad(f_ref)(w_loc_arr)
    assert jnp.allclose(grad1, grad2, rtol=1e-4, atol=1e-4)


# ---- Gradient VJP: jitnmv w.r.t. w_scale ----

@pytest.mark.parametrize("implementation", JITNMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitnmv_vjp_wscale(implementation, shape, corder, transpose):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    vec_size = shape[0] if transpose else shape[1]
    vector = jnp.asarray(np.random.rand(vec_size))
    w_scale_arr = jnp.array([w_scale])
    mask = jitn(1., 0., prob, seed, shape=shape, transpose=transpose, corder=corder)
    z_mask = jitn(0., 1., prob, seed, shape=shape, transpose=transpose, corder=corder)

    def f_fn(ws):
        return jitnmv(w_loc, ws, prob, vector, seed, shape=shape, transpose=transpose, corder=corder, backend=implementation).sum()

    def f_ref(ws):
        M = w_loc * mask + ws * z_mask
        return (M @ vector).sum()

    grad1 = jax.grad(f_fn)(w_scale_arr)
    grad2 = jax.grad(f_ref)(w_scale_arr)
    assert jnp.allclose(grad1, grad2, rtol=1e-4, atol=1e-4)


# ---- End-to-end VJP: jitnmv w.r.t. w_loc with loss ----

@pytest.mark.parametrize("implementation", JITNMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitnmv_vjp_wloc_with_loss(implementation, shape, corder, transpose):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    vec_size = shape[0] if transpose else shape[1]
    out_size = shape[1] if transpose else shape[0]
    vector = jnp.asarray(np.random.rand(vec_size))
    target = jnp.asarray(np.random.rand(out_size))
    w_loc_arr = jnp.array([w_loc])
    mask = jitn(1., 0., prob, seed, shape=shape, transpose=transpose, corder=corder)
    z_mask = jitn(0., 1., prob, seed, shape=shape, transpose=transpose, corder=corder)

    def loss_fn(wl):
        out = jitnmv(wl, w_scale, prob, vector, seed, shape=shape, transpose=transpose, corder=corder, backend=implementation)
        return jnp.sum((out - target) ** 2)

    def loss_ref(wl):
        M = wl * mask + w_scale * z_mask
        out = M @ vector
        return jnp.sum((out - target) ** 2)

    grad1 = jax.grad(loss_fn)(w_loc_arr)
    grad2 = jax.grad(loss_ref)(w_loc_arr)
    assert jnp.allclose(grad1, grad2, rtol=1e-4, atol=1e-4)


# ---- End-to-end VJP: jitnmv w.r.t. w_scale with loss ----

@pytest.mark.parametrize("implementation", JITNMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitnmv_vjp_wscale_with_loss(implementation, shape, corder, transpose):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    vec_size = shape[0] if transpose else shape[1]
    out_size = shape[1] if transpose else shape[0]
    vector = jnp.asarray(np.random.rand(vec_size))
    target = jnp.asarray(np.random.rand(out_size))
    w_scale_arr = jnp.array([w_scale])
    mask = jitn(1., 0., prob, seed, shape=shape, transpose=transpose, corder=corder)
    z_mask = jitn(0., 1., prob, seed, shape=shape, transpose=transpose, corder=corder)

    def loss_fn(ws):
        out = jitnmv(w_loc, ws, prob, vector, seed, shape=shape, transpose=transpose, corder=corder, backend=implementation)
        return jnp.sum((out - target) ** 2)

    def loss_ref(ws):
        M = w_loc * mask + ws * z_mask
        out = M @ vector
        return jnp.sum((out - target) ** 2)

    grad1 = jax.grad(loss_fn)(w_scale_arr)
    grad2 = jax.grad(loss_ref)(w_scale_arr)
    assert jnp.allclose(grad1, grad2, rtol=1e-4, atol=1e-4)


# ---- Gradient VJP: jitnmm w.r.t. w_loc ----

@pytest.mark.parametrize("implementation", JITNMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitnmm_vjp_wloc(implementation, shape, corder, transpose):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    k = 10
    mat_rows = shape[0] if transpose else shape[1]
    B = jnp.asarray(np.random.rand(mat_rows, k))
    w_loc_arr = jnp.array([w_loc])
    mask = jitn(1., 0., prob, seed, shape=shape, transpose=transpose, corder=corder)
    z_mask = jitn(0., 1., prob, seed, shape=shape, transpose=transpose, corder=corder)

    def f_fn(wl):
        return jitnmm(wl, w_scale, prob, B, seed, shape=shape, transpose=transpose, corder=corder, backend=implementation).sum()

    def f_ref(wl):
        M = wl * mask + w_scale * z_mask
        return (M @ B).sum()

    grad1 = jax.grad(f_fn)(w_loc_arr)
    grad2 = jax.grad(f_ref)(w_loc_arr)
    assert jnp.allclose(grad1, grad2, rtol=1e-4, atol=1e-4)


# ---- Gradient VJP: jitnmm w.r.t. w_scale ----

@pytest.mark.parametrize("implementation", JITNMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitnmm_vjp_wscale(implementation, shape, corder, transpose):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    k = 10
    mat_rows = shape[0] if transpose else shape[1]
    B = jnp.asarray(np.random.rand(mat_rows, k))
    w_scale_arr = jnp.array([w_scale])
    mask = jitn(1., 0., prob, seed, shape=shape, transpose=transpose, corder=corder)
    z_mask = jitn(0., 1., prob, seed, shape=shape, transpose=transpose, corder=corder)

    def f_fn(ws):
        return jitnmm(w_loc, ws, prob, B, seed, shape=shape, transpose=transpose, corder=corder, backend=implementation).sum()

    def f_ref(ws):
        M = w_loc * mask + ws * z_mask
        return (M @ B).sum()

    grad1 = jax.grad(f_fn)(w_scale_arr)
    grad2 = jax.grad(f_ref)(w_scale_arr)
    assert jnp.allclose(grad1, grad2, rtol=1e-4, atol=1e-4)


# ---- End-to-end VJP: jitnmm w.r.t. w_loc with loss ----

@pytest.mark.parametrize("implementation", JITNMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitnmm_vjp_wloc_with_loss(implementation, shape, corder, transpose):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    k = 10
    mat_rows = shape[0] if transpose else shape[1]
    out_rows = shape[1] if transpose else shape[0]
    B = jnp.asarray(np.random.rand(mat_rows, k))
    target = jnp.asarray(np.random.rand(out_rows, k))
    w_loc_arr = jnp.array([w_loc])
    mask = jitn(1., 0., prob, seed, shape=shape, transpose=transpose, corder=corder)
    z_mask = jitn(0., 1., prob, seed, shape=shape, transpose=transpose, corder=corder)

    def loss_fn(wl):
        out = jitnmm(wl, w_scale, prob, B, seed, shape=shape, transpose=transpose, corder=corder, backend=implementation)
        return jnp.sum((out - target) ** 2)

    def loss_ref(wl):
        M = wl * mask + w_scale * z_mask
        out = M @ B
        return jnp.sum((out - target) ** 2)

    grad1 = jax.grad(loss_fn)(w_loc_arr)
    grad2 = jax.grad(loss_ref)(w_loc_arr)
    assert jnp.allclose(grad1, grad2, rtol=1e-4, atol=1e-4)


# ---- End-to-end VJP: jitnmm w.r.t. w_scale with loss ----

@pytest.mark.parametrize("implementation", JITNMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitnmm_vjp_wscale_with_loss(implementation, shape, corder, transpose):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    k = 10
    mat_rows = shape[0] if transpose else shape[1]
    out_rows = shape[1] if transpose else shape[0]
    B = jnp.asarray(np.random.rand(mat_rows, k))
    target = jnp.asarray(np.random.rand(out_rows, k))
    w_scale_arr = jnp.array([w_scale])
    mask = jitn(1., 0., prob, seed, shape=shape, transpose=transpose, corder=corder)
    z_mask = jitn(0., 1., prob, seed, shape=shape, transpose=transpose, corder=corder)

    def loss_fn(ws):
        out = jitnmm(w_loc, ws, prob, B, seed, shape=shape, transpose=transpose, corder=corder, backend=implementation)
        return jnp.sum((out - target) ** 2)

    def loss_ref(ws):
        M = w_loc * mask + ws * z_mask
        out = M @ B
        return jnp.sum((out - target) ** 2)

    grad1 = jax.grad(loss_fn)(w_scale_arr)
    grad2 = jax.grad(loss_ref)(w_scale_arr)
    assert jnp.allclose(grad1, grad2, rtol=1e-4, atol=1e-4)
