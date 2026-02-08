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
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._jit_scalar.float import jits, jits_p, jitsmv, jitsmv_p, jitsmm, jitsmm_p

platform = jax.default_backend()
JITS_IMPLEMENTATIONS = tuple(jits_p.available_backends(platform))
JITSMV_IMPLEMENTATIONS = tuple(jitsmv_p.available_backends(platform))
JITSMM_IMPLEMENTATIONS = tuple(jitsmm_p.available_backends(platform))


# ---- jits: transpose symmetry ----

@pytest.mark.parametrize("implementation", JITS_IMPLEMENTATIONS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
def test_jits_transpose_symmetry(implementation, transpose, corder):
    out1 = jits(1.5, 0.1, 123, shape=(100, 50), transpose=transpose, corder=corder, backend=implementation)
    out2 = jits(1.5, 0.1, 123, shape=(100, 50), transpose=not transpose, corder=not corder, backend=implementation)
    assert jnp.allclose(out1, out2.T)


# ---- Forward: jitsmv (transpose=False) ----

@pytest.mark.parametrize("implementation", JITSMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(100, 200), (20, 100)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitsmv_forward(implementation, shape, corder):
    weight, prob, seed = 1.5, 0.1, 1234
    vector = jnp.asarray(np.random.rand(shape[1]))
    dense = jits(weight, prob, seed, shape=shape, corder=corder)
    out = jitsmv(weight, prob, vector, seed=seed, shape=shape, corder=corder, backend=implementation)
    expected = dense @ vector
    assert jnp.allclose(out, expected, rtol=1e-4, atol=1e-4)


# ---- Forward: jitsmv (transpose=True) ----

@pytest.mark.parametrize("implementation", JITSMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(100, 200), (20, 100)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitsmv_transpose_forward(implementation, shape, corder):
    weight, prob, seed = 1.5, 0.1, 1234
    vector = jnp.asarray(np.random.rand(shape[0]))
    dense = jits(weight, prob, seed, shape=shape, transpose=True, corder=corder)
    out = jitsmv(weight, prob, vector, seed=seed, shape=shape, transpose=True, corder=corder, backend=implementation)
    expected = dense @ vector
    assert jnp.allclose(out, expected, rtol=1e-4, atol=1e-4)


# ---- Forward: jitsmv zero weight ----

@pytest.mark.parametrize("implementation", JITSMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
def test_jitsmv_zero_weight(implementation, transpose, corder):
    shape = (2, 3)
    v = brainstate.random.rand(shape[0]) if transpose else brainstate.random.rand(shape[1])
    result = jitsmv(0.0, 0.5, v, seed=1234, shape=shape, transpose=transpose, corder=corder, backend=implementation)
    expected = jnp.zeros(shape[1]) if transpose else jnp.zeros(shape[0])
    assert jnp.allclose(result, expected)


# ---- Forward: jitsmm (transpose=False) ----

@pytest.mark.parametrize("implementation", JITSMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(100, 200), (20, 100)])
@pytest.mark.parametrize('k', [10])
@pytest.mark.parametrize('corder', [True, False])
def test_jitsmm_forward(implementation, shape, k, corder):
    weight, prob, seed = 1.5, 0.1, 1234
    B = jnp.asarray(np.random.rand(shape[1], k))
    dense = jits(weight, prob, seed, shape=shape, corder=corder)
    out = jitsmm(weight, prob, B, seed=seed, shape=shape, corder=corder, backend=implementation)
    expected = dense @ B
    assert jnp.allclose(out, expected, rtol=1e-4, atol=1e-4)


# ---- Forward: jitsmm (transpose=True) ----

@pytest.mark.parametrize("implementation", JITSMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(100, 200), (20, 100)])
@pytest.mark.parametrize('k', [10])
@pytest.mark.parametrize('corder', [True, False])
def test_jitsmm_transpose_forward(implementation, shape, k, corder):
    weight, prob, seed = 1.5, 0.1, 1234
    B = jnp.asarray(np.random.rand(shape[0], k))
    dense = jits(weight, prob, seed, shape=shape, transpose=True, corder=corder)
    out = jitsmm(weight, prob, B, seed=seed, shape=shape, transpose=True, corder=corder, backend=implementation)
    expected = dense @ B
    assert jnp.allclose(out, expected, rtol=1e-4, atol=1e-4)


# ---- Gradient JVP: jitsmv ----

@pytest.mark.parametrize("implementation", JITSMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitsmv_jvp(implementation, shape, corder, transpose):
    weight, prob, seed = 1.5, 0.1, 1234
    vec_size = shape[0] if transpose else shape[1]
    x = jnp.asarray(np.random.rand(vec_size))
    dense = jits(1.0, prob, seed, shape=shape, transpose=transpose, corder=corder)

    def f_fn(x, w):
        return jitsmv(w, prob, x, seed=seed, shape=shape, transpose=transpose, corder=corder, backend=implementation).sum()

    def f_dense(x, w):
        return (dense * w @ x).sum()

    out1, jvp1 = jax.jvp(f_fn, (x, jnp.array(weight)), (jnp.ones_like(x), jnp.array(1.0)))
    out2, jvp2 = jax.jvp(f_dense, (x, jnp.array(weight)), (jnp.ones_like(x), jnp.array(1.0)))
    assert jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(jvp1, jvp2, rtol=1e-4, atol=1e-4)


# ---- Gradient VJP: jitsmv ----

@pytest.mark.parametrize("implementation", JITSMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitsmv_vjp(implementation, shape, corder, transpose):
    weight, prob, seed = 1.5, 0.1, 1234
    vec_size = shape[0] if transpose else shape[1]
    x = jnp.asarray(np.random.rand(vec_size))
    dense = jits(1.0, prob, seed, shape=shape, transpose=transpose, corder=corder)

    def f_fn(x, w):
        return jitsmv(w, prob, x, seed=seed, shape=shape, transpose=transpose, corder=corder, backend=implementation).sum()

    def f_dense(x, w):
        return (dense * w @ x).sum()

    out1, (vjp_x1, vjp_w1) = jax.value_and_grad(f_fn, argnums=(0, 1))(x, jnp.array(weight))
    out2, (vjp_x2, vjp_w2) = jax.value_and_grad(f_dense, argnums=(0, 1))(x, jnp.array(weight))
    assert jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(vjp_w1, vjp_w2, rtol=1e-4, atol=1e-4)


# ---- Gradient JVP: jitsmm ----

@pytest.mark.parametrize("implementation", JITSMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('k', [10])
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitsmm_jvp(implementation, k, shape, corder, transpose):
    weight, prob, seed = 1.5, 0.1, 1234
    mat_rows = shape[0] if transpose else shape[1]
    X = jnp.asarray(np.random.rand(mat_rows, k))
    dense = jits(1.0, prob, seed, shape=shape, transpose=transpose, corder=corder)

    def f_fn(X, w):
        return jitsmm(w, prob, X, seed=seed, shape=shape, transpose=transpose, corder=corder, backend=implementation).sum()

    def f_dense(X, w):
        return (dense * w @ X).sum()

    out1, jvp1 = jax.jvp(f_fn, (X, jnp.array(weight)), (jnp.ones_like(X), jnp.array(1.0)))
    out2, jvp2 = jax.jvp(f_dense, (X, jnp.array(weight)), (jnp.ones_like(X), jnp.array(1.0)))
    assert jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(jvp1, jvp2, rtol=1e-4, atol=1e-4)


# ---- Gradient VJP: jitsmm ----

@pytest.mark.parametrize("implementation", JITSMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('k', [10])
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitsmm_vjp(implementation, k, shape, corder, transpose):
    weight, prob, seed = 1.5, 0.1, 1234
    mat_rows = shape[0] if transpose else shape[1]
    X = jnp.asarray(np.random.rand(mat_rows, k))
    dense = jits(1.0, prob, seed, shape=shape, transpose=transpose, corder=corder)

    def f_fn(X, w):
        return jitsmm(w, prob, X, seed=seed, shape=shape, transpose=transpose, corder=corder, backend=implementation).sum()

    def f_dense(X, w):
        return (dense * w @ X).sum()

    out1, (vjp_x1, vjp_w1) = jax.value_and_grad(f_fn, argnums=(0, 1))(X, jnp.array(weight))
    out2, (vjp_x2, vjp_w2) = jax.value_and_grad(f_dense, argnums=(0, 1))(X, jnp.array(weight))
    assert jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(vjp_w1, vjp_w2, rtol=1e-4, atol=1e-4)


# ---- Batching: jitsmv over vectors ----

@pytest.mark.parametrize("implementation", JITSMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('batch_size', [10])
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitsmv_vmap_over_vectors(implementation, batch_size, shape, corder):
    weight, prob, seed = 1.05, 0.1, 123
    vectors = brainstate.random.rand(batch_size, shape[1])

    def f(vector):
        return jitsmv(weight, prob, vector, seed=seed, shape=shape, corder=corder, backend=implementation)

    results = jax.vmap(f)(vectors)
    assert results.shape == (batch_size, shape[0])

    results_loop = brainstate.transform.for_loop(f, vectors)
    assert results_loop.shape == (batch_size, shape[0])

    assert jnp.allclose(results, results_loop, rtol=1e-4, atol=1e-4)


# ---- Batching: jitsmv over vectors (transpose) ----

@pytest.mark.parametrize("implementation", JITSMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('batch_size', [10])
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitsmv_transpose_vmap_over_vectors(implementation, batch_size, shape, corder):
    weight, prob, seed = 1.05, 0.1, 123
    vectors = brainstate.random.rand(batch_size, shape[0])

    def f(vector):
        return jitsmv(weight, prob, vector, seed=seed, shape=shape, transpose=True, corder=corder, backend=implementation)

    results = jax.vmap(f)(vectors)
    assert results.shape == (batch_size, shape[1])

    results_loop = brainstate.transform.for_loop(f, vectors)
    assert results_loop.shape == (batch_size, shape[1])

    assert jnp.allclose(results, results_loop, rtol=1e-4, atol=1e-4)


# ---- Batching: jitsmv over weight ----

@pytest.mark.parametrize("implementation", JITSMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('batch_size', [10])
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitsmv_vmap_over_weight(implementation, batch_size, shape, corder):
    prob, seed = 0.1, 123
    weights = brainstate.random.rand(batch_size)
    vector = brainstate.random.rand(shape[1])

    def f(w):
        return jitsmv(w, prob, vector, seed=seed, shape=shape, corder=corder, backend=implementation)

    results = jax.vmap(f)(weights)
    assert results.shape == (batch_size, shape[0])

    results_loop = brainstate.transform.for_loop(f, weights)
    assert results_loop.shape == (batch_size, shape[0])

    assert jnp.allclose(results, results_loop, rtol=1e-4, atol=1e-4)


# ---- Batching: jitsmm over matrices ----

@pytest.mark.parametrize("implementation", JITSMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('batch_size', [10])
@pytest.mark.parametrize('k', [5])
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitsmm_vmap_over_matrices(implementation, batch_size, k, shape, corder):
    weight, prob, seed = 1.05, 0.1, 123
    matrices = brainstate.random.rand(batch_size, shape[1], k)

    def f(mat):
        return jitsmm(weight, prob, mat, seed=seed, shape=shape, corder=corder, backend=implementation)

    outs = jax.vmap(f)(matrices)
    assert outs.shape == (batch_size, shape[0], k)

    outs_loop = brainstate.transform.for_loop(f, matrices)
    assert outs_loop.shape == (batch_size, shape[0], k)

    assert jnp.allclose(outs, outs_loop, rtol=1e-4, atol=1e-4)


# ---- Batching: jitsmm over matrices (transpose) ----

@pytest.mark.parametrize("implementation", JITSMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('batch_size', [10])
@pytest.mark.parametrize('k', [5])
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitsmm_transpose_vmap_over_matrices(implementation, batch_size, k, shape, corder):
    weight, prob, seed = 1.05, 0.1, 123
    matrices = brainstate.random.rand(batch_size, shape[0], k)

    def f(mat):
        return jitsmm(weight, prob, mat, seed=seed, shape=shape, transpose=True, corder=corder, backend=implementation)

    outs = jax.vmap(f)(matrices)
    assert outs.shape == (batch_size, shape[1], k)

    outs_loop = brainstate.transform.for_loop(f, matrices)
    assert outs_loop.shape == (batch_size, shape[1], k)

    assert jnp.allclose(outs, outs_loop, rtol=1e-4, atol=1e-4)


# ---- Batching: jitsmm over weight ----

@pytest.mark.parametrize("implementation", JITSMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('batch_size', [10])
@pytest.mark.parametrize('k', [5])
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitsmm_vmap_over_weight(implementation, batch_size, k, shape, corder):
    prob, seed = 0.1, 123
    weights = brainstate.random.rand(batch_size)
    matrix = brainstate.random.rand(shape[1], k)

    def f(w):
        return jitsmm(w, prob, matrix, seed=seed, shape=shape, corder=corder, backend=implementation)

    results = jax.vmap(f)(weights)
    assert results.shape == (batch_size, shape[0], k)

    results_loop = brainstate.transform.for_loop(f, weights)
    assert results_loop.shape == (batch_size, shape[0], k)

    assert jnp.allclose(results, results_loop, rtol=1e-4, atol=1e-4)


# ---- Batching: jits over weight ----

@pytest.mark.parametrize("implementation", JITS_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(100, 50)])
def test_jits_vmap_over_weight(implementation, shape):
    prob, seed = 0.1, 123

    def f(weight):
        return jits(weight, prob, seed, shape=shape, backend=implementation)

    weights = brainstate.random.rand(10)
    results = jax.vmap(f)(weights)
    assert results.shape == (10,) + shape

    results_loop = brainstate.transform.for_loop(f, weights)
    assert results_loop.shape == (10,) + shape

    assert jnp.allclose(results, results_loop, rtol=1e-4, atol=1e-4)


# ---- Batching: jits over prob ----

@pytest.mark.parametrize("implementation", JITS_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(100, 50)])
def test_jits_vmap_over_prob(implementation, shape):
    weight, seed = 1.5, 123

    def f(prob):
        return jits(weight, prob, seed, shape=shape, backend=implementation)

    probs = brainstate.random.rand(10)
    results = jax.vmap(f)(probs)
    assert results.shape == (10,) + shape

    results_loop = brainstate.transform.for_loop(f, probs)
    assert results_loop.shape == (10,) + shape

    assert jnp.allclose(results, results_loop, rtol=1e-4, atol=1e-4)


# ---- Batching: jits over seed ----

@pytest.mark.parametrize("implementation", JITS_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(100, 50)])
def test_jits_vmap_over_seed(implementation, shape):
    weight, prob = 1.5, 0.1

    def f(seed):
        return jits(weight, prob, seed, shape=shape, backend=implementation)

    seeds = brainstate.random.randint(0, 100000, 10)
    results = jax.vmap(f)(seeds)
    assert results.shape == (10,) + shape

    results_loop = brainstate.transform.for_loop(f, seeds)
    assert results_loop.shape == (10,) + shape

    assert jnp.allclose(results, results_loop, rtol=1e-4, atol=1e-4)
