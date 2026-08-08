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

# Keep GPU matmul reference numerics stable (avoid TF32 drift in dense @ B checks).
if jax.default_backend() == 'gpu' and jax.config.jax_default_matmul_precision is None:
    jax.config.update('jax_default_matmul_precision', 'highest')

from brainevent._jit_scalar.float import jits, jits_p, jitsmv, jitsmv_p, jitsmm, jitsmm_p
from brainevent._jit_scalar._test_util import dense_scalar_reference

platform = jax.default_backend()
JITS_IMPLEMENTATIONS = tuple(jits_p.available_backends(platform))
JITSMV_IMPLEMENTATIONS = tuple(jitsmv_p.available_backends(platform))
JITSMM_IMPLEMENTATIONS = tuple(jitsmm_p.available_backends(platform))


@pytest.fixture(autouse=True)
def _seed_rng():
    """Seed the global NumPy RNG so unseeded ``np.random`` probe draws are
    deterministic; keeps tolerance-sensitive autodiff-vs-finite-difference
    gradient checks from being order-dependently flaky."""
    np.random.seed(0x5EED)


# ---- jits: transpose symmetry ----

@pytest.mark.parametrize("implementation", JITS_IMPLEMENTATIONS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
def test_jits_transpose_symmetry(implementation, transpose, corder):
    out1 = jits(1.5, 0.1, 123, shape=(100, 50), transpose=transpose, corder=corder,
                backend=implementation)
    out2 = jits(1.5, 0.1, 123, shape=(100, 50), transpose=not transpose, corder=not corder,
                backend=implementation)
    assert jnp.allclose(out1, out2.T)
    jax.block_until_ready((out1, out2))


# ---- Forward: jitsmv (transpose=False) vs the dense matrix ----

@pytest.mark.parametrize("implementation", JITSMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(100, 200), (20, 100)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitsmv_forward(implementation, shape, corder):
    weight, prob, seed = 1.5, 0.1, 1234
    vector = jnp.asarray(np.random.rand(shape[1]))
    dense = jits(weight, prob, seed, shape=shape, corder=corder, backend=implementation)
    out = jitsmv(weight, prob, vector, seed=seed, shape=shape, corder=corder, backend=implementation)
    expected = dense @ vector
    assert jnp.allclose(out, expected, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((vector, dense, out, expected))


# ---- Forward: jitsmv (transpose=True) ----

@pytest.mark.parametrize("implementation", JITSMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(100, 200), (20, 100)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitsmv_transpose_forward(implementation, shape, corder):
    weight, prob, seed = 1.5, 0.1, 1234
    vector = jnp.asarray(np.random.rand(shape[0]))
    dense = jits(weight, prob, seed, shape=shape, transpose=True, corder=corder, backend=implementation)
    out = jitsmv(weight, prob, vector, seed=seed, shape=shape, transpose=True, corder=corder, backend=implementation)
    expected = dense @ vector
    assert jnp.allclose(out, expected, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((vector, dense, out, expected))


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
    jax.block_until_ready((v, result, expected))


# ---- Forward: jitsmm (transpose=False) vs the dense matrix ----

@pytest.mark.parametrize("implementation", JITSMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(100, 200), (20, 100)])
@pytest.mark.parametrize('k', [10])
@pytest.mark.parametrize('corder', [True, False])
def test_jitsmm_forward(implementation, shape, k, corder):
    weight, prob, seed = 1.5, 0.1, 1234
    B = jnp.asarray(np.random.rand(shape[1], k))
    dense = jits(weight, prob, seed, shape=shape, corder=corder, backend=implementation)
    out = jitsmm(weight, prob, B, seed=seed, shape=shape, corder=corder, backend=implementation)
    expected = dense @ B
    assert jnp.allclose(out, expected, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((B, dense, out, expected))


# ---- Forward: jitsmm (transpose=True) ----

@pytest.mark.parametrize("implementation", JITSMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(100, 200), (20, 100)])
@pytest.mark.parametrize('k', [10])
@pytest.mark.parametrize('corder', [True, False])
def test_jitsmm_transpose_forward(implementation, shape, k, corder):
    weight, prob, seed = 1.5, 0.1, 1234
    B = jnp.asarray(np.random.rand(shape[0], k))
    dense = jits(weight, prob, seed, shape=shape, transpose=True, corder=corder, backend=implementation)
    out = jitsmm(weight, prob, B, seed=seed, shape=shape, transpose=True, corder=corder, backend=implementation)
    expected = dense @ B
    assert jnp.allclose(out, expected, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((B, dense, out, expected))


# ---- Gradient JVP: jitsmv ----

@pytest.mark.parametrize("implementation", JITSMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitsmv_jvp(implementation, shape, corder, transpose):
    weight, prob, seed = 1.5, 0.1, 1234
    vec_size = shape[0] if transpose else shape[1]
    x = jnp.asarray(np.random.rand(vec_size))
    dense = jits(1.0, prob, seed, shape=shape, transpose=transpose, corder=corder, backend=implementation)

    def f_fn(x, w):
        return jitsmv(w, prob, x, seed=seed, shape=shape, transpose=transpose, corder=corder,
                      backend=implementation).sum()

    def f_dense(x, w):
        return (dense * w @ x).sum()

    w_arr = jnp.array(weight)
    t_x = jnp.ones_like(x)
    t_w = jnp.array(1.0)
    out1, jvp1 = jax.jvp(f_fn, (x, w_arr), (t_x, t_w))
    out2, jvp2 = jax.jvp(f_dense, (x, w_arr), (t_x, t_w))
    assert jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(jvp1, jvp2, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((x, dense, w_arr, t_x, t_w, out1, jvp1, out2, jvp2))


# ---- Gradient VJP: jitsmv ----

@pytest.mark.parametrize("implementation", JITSMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitsmv_vjp(implementation, shape, corder, transpose):
    weight, prob, seed = 1.5, 0.1, 1234
    vec_size = shape[0] if transpose else shape[1]
    x = jnp.asarray(np.random.rand(vec_size))
    dense = jits(1.0, prob, seed, shape=shape, transpose=transpose, corder=corder, backend=implementation)

    def f_fn(x, w):
        return jitsmv(w, prob, x, seed=seed, shape=shape, transpose=transpose, corder=corder,
                      backend=implementation).sum()

    def f_dense(x, w):
        return (dense * w @ x).sum()

    w_arr = jnp.array(weight)
    out1, (vjp_x1, vjp_w1) = jax.value_and_grad(f_fn, argnums=(0, 1))(x, w_arr)
    out2, (vjp_x2, vjp_w2) = jax.value_and_grad(f_dense, argnums=(0, 1))(x, w_arr)
    assert jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(vjp_w1, vjp_w2, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((x, dense, w_arr, out1, vjp_x1, vjp_w1, out2, vjp_x2, vjp_w2))


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
    dense = jits(1.0, prob, seed, shape=shape, transpose=transpose, corder=corder, backend=implementation)

    def f_fn(X, w):
        return jitsmm(w, prob, X, seed=seed, shape=shape, transpose=transpose, corder=corder,
                      backend=implementation).sum()

    def f_dense(X, w):
        return (dense * w @ X).sum()

    w_arr = jnp.array(weight)
    t_X = jnp.ones_like(X)
    t_w = jnp.array(1.0)
    out1, jvp1 = jax.jvp(f_fn, (X, w_arr), (t_X, t_w))
    out2, jvp2 = jax.jvp(f_dense, (X, w_arr), (t_X, t_w))
    assert jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(jvp1, jvp2, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((X, dense, w_arr, t_X, t_w, out1, jvp1, out2, jvp2))


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
    dense = jits(1.0, prob, seed, shape=shape, transpose=transpose, corder=corder, backend=implementation)

    def f_fn(X, w):
        return jitsmm(w, prob, X, seed=seed, shape=shape, transpose=transpose, corder=corder,
                      backend=implementation).sum()

    def f_dense(X, w):
        return (dense * w @ X).sum()

    w_arr = jnp.array(weight)
    out1, (vjp_x1, vjp_w1) = jax.value_and_grad(f_fn, argnums=(0, 1))(X, w_arr)
    out2, (vjp_x2, vjp_w2) = jax.value_and_grad(f_dense, argnums=(0, 1))(X, w_arr)
    assert jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(vjp_w1, vjp_w2, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((X, dense, w_arr, out1, vjp_x1, vjp_w1, out2, vjp_x2, vjp_w2))


# ---- Batching: jitsmv over vectors == jitsmm ----
# vmap over the vector axis is a matrix-matrix operation; matvec and matmat draw
# the same matrix, so the two agree column by column.

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

    expected = jitsmm(weight, prob, jnp.asarray(vectors).T, seed=seed, shape=shape, corder=corder,
                      backend=implementation).T
    assert jnp.allclose(results, expected, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((vectors, results, expected))


# ---- Batching: jitsmv over vectors (transpose) == jitsmm (mm) ----

@pytest.mark.parametrize("implementation", JITSMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('batch_size', [10])
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitsmv_transpose_vmap_over_vectors(implementation, batch_size, shape, corder):
    weight, prob, seed = 1.05, 0.1, 123
    vectors = brainstate.random.rand(batch_size, shape[0])

    def f(vector):
        return jitsmv(weight, prob, vector, seed=seed, shape=shape, transpose=True, corder=corder,
                      backend=implementation)

    results = jax.vmap(f)(vectors)
    assert results.shape == (batch_size, shape[1])

    expected = jitsmm(weight, prob, jnp.asarray(vectors).T, seed=seed, shape=shape, transpose=True,
                      corder=corder, backend=implementation).T
    assert jnp.allclose(results, expected, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((vectors, results, expected))


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
    jax.block_until_ready((weights, vector, results, results_loop))


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
    jax.block_until_ready((matrices, outs, outs_loop))


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
    jax.block_until_ready((matrices, outs, outs_loop))


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
    jax.block_until_ready((weights, matrix, results, results_loop))


# ---- Batching: jits over weight / prob / seed ----

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
    jax.block_until_ready((weights, results, results_loop))


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
    jax.block_until_ready((probs, results, results_loop))


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
    jax.block_until_ready((seeds, results, results_loop))


# ---- One matrix: vmap over vectors agrees with a loop, and with the numpy walk ----
# ``vmap(jitsmv)`` forwards to ``jitsmm``; before unification the two drew
# different matrices, so this equality did not hold.

@pytest.mark.parametrize("implementation", JITSMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30)])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('corder', [True, False])
def test_vmap_jitsmv_matches_loop(implementation, shape, transpose, corder):
    prob, seed, batch = 0.2, 123, 5
    k = shape[0] if transpose else shape[1]
    vectors = np.random.randn(batch, k).astype(np.float32)

    def f(v):
        return jitsmv(1.5, prob, v, seed, shape=shape,
                   transpose=transpose, corder=corder, backend=implementation)

    batched = jax.vmap(f)(jnp.asarray(vectors))
    looped = jnp.stack([f(jnp.asarray(vectors[i])) for i in range(batch)])
    assert jnp.allclose(batched, looped, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((batched, looped))


@pytest.mark.parametrize("implementation", JITS_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(13, 17), (33, 33), (7, 5)])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('corder', [True, False])
def test_jits_matches_numpy_reference(implementation, shape, transpose, corder):
    # An independent pure-numpy replay of the 32-lane walk -- this pins the drawn
    # matrix itself, not merely the agreement of the kernels with each other.
    prob, seed = 0.2, 123
    actual = np.asarray(jits(1.5, prob, seed, shape=shape, transpose=transpose,
                         corder=corder, backend=implementation))
    expected = dense_scalar_reference(1.5, prob, seed, shape=shape,
                       transpose=transpose, corder=corder)
    assert np.array_equal(actual != 0, expected != 0)
    assert np.allclose(actual, expected, rtol=1e-4, atol=1e-4)


# ---- Public interface: back to the v0.1.2 parameter lists ----
# ``matrix_mode`` was a 0.2.0-only keyword; with one matrix it is gone, and these
# signatures must again be exactly the ones v0.1.2 shipped.

@pytest.mark.parametrize('fn,expected', [
    (jits, ('weight', 'prob', 'seed', 'shape', 'transpose', 'corder', 'backend')),
    (jitsmv, ('weight', 'prob', 'vector', 'seed', 'shape', 'transpose', 'corder', 'backend')),
    (jitsmm, ('weight', 'prob', 'B', 'seed', 'shape', 'transpose', 'corder', 'backend')),
])
def test_public_signature_matches_0_1_2(fn, expected):
    import inspect
    assert tuple(inspect.signature(fn).parameters) == tuple(
        p.strip() for part in expected for p in part.split(',')
    )


# ---- The drawn matrix depends only on the generated matrix's geometry ----
# ``chunk_size`` splits the *walked* dimension, never the caller's ``shape[1]``.
# The two ``(shape, transpose)`` pairs below describe the same matrix, so they
# must draw it identically -- this is what lets the matrix classes materialize
# with their own shape instead of a swapped one.

@pytest.mark.parametrize("implementation", JITS_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(12, 20), (33, 33), (7, 5), (64, 3)])
@pytest.mark.parametrize('corder', [True, False])
def test_generation_is_shape_pair_independent(implementation, shape, corder):
    m, n = shape
    prob, seed = 0.2, 123
    a = jits(1.5, prob, seed, shape=(m, n), transpose=False,
          corder=corder, backend=implementation)
    b = jits(1.5, prob, seed, shape=(n, m), transpose=True,
          corder=corder, backend=implementation)
    assert np.array_equal(np.asarray(a), np.asarray(b))


# ---- numba (CPU) and cuda_raw (GPU) must draw the *same* matrix ----
# The two backends reimplement the light-RNG walk independently; the whole point
# of the shared chunk/lane keying is that they agree bit for bit. Skipped unless
# both backends are actually available on this machine.

@pytest.mark.parametrize('shape', [(12, 20), (33, 33), (7, 5), (100, 250)])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('corder', [True, False])
def test_numba_and_cuda_draw_the_same_matrix(shape, transpose, corder):
    if 'numba' not in jits_p.available_backends('cpu') or 'cuda_raw' not in jits_p.available_backends('gpu'):
        pytest.skip('needs both a CPU numba backend and a CUDA device')
    prob, seed = 0.2, 123
    with jax.default_device(jax.devices('cpu')[0]):
        cpu = np.asarray(jits(1.5, prob, seed, shape=shape, transpose=transpose,
                          corder=corder, backend='numba'))
    with jax.default_device(jax.devices('cuda')[0]):
        gpu = np.asarray(jits(1.5, prob, seed, shape=shape, transpose=transpose,
                          corder=corder, backend='cuda_raw'))
    assert np.array_equal(cpu != 0, gpu != 0)
    assert np.array_equal(cpu, gpu)


@pytest.mark.parametrize('shape', [(20, 30), (33, 33)])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('corder', [True, False])
def test_numba_and_cuda_matvec_agree(shape, transpose, corder):
    if 'numba' not in jitsmv_p.available_backends('cpu') or 'cuda_raw' not in jitsmv_p.available_backends('gpu'):
        pytest.skip('needs both a CPU numba backend and a CUDA device')
    prob, seed = 0.2, 123
    k = shape[0] if transpose else shape[1]
    v = np.random.rand(k).astype(np.float32)
    with jax.default_device(jax.devices('cpu')[0]):
        cpu = np.asarray(jitsmv(1.5, prob, jnp.asarray(v), seed, shape=shape,
                            transpose=transpose, corder=corder, backend='numba'))
    with jax.default_device(jax.devices('cuda')[0]):
        gpu = np.asarray(jitsmv(1.5, prob, jnp.asarray(v), seed, shape=shape,
                            transpose=transpose, corder=corder, backend='cuda_raw'))
    # same matrix, so only the float summation order may differ
    assert np.allclose(cpu, gpu, rtol=1e-5, atol=1e-5)


# ---- The ``notrans`` and ``trans`` CUDA entry points draw ONE matrix ----
# They are separate kernels: ``notrans`` gathers (``acc += w * v[j]`` into
# ``output[row]``), ``trans`` scatters (``atomic_add(output[j], w * v[row])``).
# For the same seeded/walked dimensions they must replay the same stream, so one
# computes ``M @ v`` and the other ``M.T @ u`` for the *same* M. Recover M from
# each by feeding unit vectors and compare.
#
#   notrans entry:  jitsmv(shape=(a, b), transpose=False, corder=True )   -> M @ v
#   trans   entry:  jitsmv(shape=(b, a), transpose=False, corder=False)   -> M.T @ u

@pytest.mark.parametrize("implementation", JITSMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(12, 20), (17, 9), (33, 33)])
def test_notrans_and_trans_kernels_draw_one_matrix(implementation, shape):
    a, b = shape
    prob, seed = 0.2, 123
    eye_b = np.eye(b, dtype=np.float32)
    eye_a = np.eye(a, dtype=np.float32)
    via_notrans = np.stack(
        [np.asarray(jitsmv(1.5, prob, jnp.asarray(eye_b[j]), seed, shape=(a, b),
                       transpose=False, corder=True, backend=implementation))
         for j in range(b)], axis=1)
    via_trans = np.stack(
        [np.asarray(jitsmv(1.5, prob, jnp.asarray(eye_a[i]), seed, shape=(b, a),
                       transpose=False, corder=False, backend=implementation))
         for i in range(a)], axis=1).T
    assert via_notrans.shape == via_trans.shape == (a, b)
    assert np.array_equal(via_notrans != 0, via_trans != 0)
    assert np.allclose(via_notrans, via_trans, rtol=1e-5, atol=1e-5)
    # and it is the matrix the materialization operator writes out
    materialized = np.asarray(jits(1.5, prob, seed, shape=(a, b), transpose=False,
                              corder=True, backend=implementation))
    assert np.allclose(via_notrans, materialized, rtol=1e-5, atol=1e-5)
