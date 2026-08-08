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

from brainevent._jit_normal.float import jitn, jitn_p, jitnmv, jitnmv_p, jitnmm, jitnmm_p
from brainevent._jit_normal._test_util import dense_normal_reference

platform = jax.default_backend()
JITN_IMPLEMENTATIONS = tuple(jitn_p.available_backends(platform))
JITNMV_IMPLEMENTATIONS = tuple(jitnmv_p.available_backends(platform))
JITNMM_IMPLEMENTATIONS = tuple(jitnmm_p.available_backends(platform))
CPU_DEVICE = jax.devices('cpu')[0]
CPU_JITN_IMPLEMENTATIONS = tuple(jitn_p.available_backends('cpu'))
CPU_JITNMV_IMPLEMENTATIONS = tuple(jitnmv_p.available_backends('cpu'))
CPU_JITNMM_IMPLEMENTATIONS = tuple(jitnmm_p.available_backends('cpu'))

requires_cpu_jitn = pytest.mark.skipif(
    'numba' not in CPU_JITN_IMPLEMENTATIONS,
    reason='No jitn numba backend registered on CPU',
)
requires_cpu_jitnmv = pytest.mark.skipif(
    'numba' not in CPU_JITNMV_IMPLEMENTATIONS,
    reason='No jitnmv numba backend registered on CPU',
)
requires_cpu_jitnmm = pytest.mark.skipif(
    'numba' not in CPU_JITNMM_IMPLEMENTATIONS,
    reason='No jitnmm numba backend registered on CPU',
)


@pytest.fixture(autouse=True)
def _seed_rng():
    """Make the unseeded ``np.random`` draws in this module deterministic.

    Several tests validate an analytic (autodiff) gradient against a float32
    finite-difference estimate with a tight tolerance. The probe vectors are
    drawn from the global NumPy RNG, so without a fixed seed an unlucky draw on
    a small problem can push the finite-difference error past the tolerance,
    making the test order-dependently flaky. Seeding per test removes that
    dependence without changing the statistical nature of the checks.
    """
    np.random.seed(0x5EED)


# ---- Forward: jitnmv (matrix @ vector, transpose=False) ----

@requires_cpu_jitn
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('corder', [True, False])
def test_jitn_numba_matches_light_rng_reference(transpose, corder):
    shape = (13, 17)
    w_loc = jnp.asarray(1.5, dtype=jnp.float32)
    w_scale = jnp.asarray(0.15, dtype=jnp.float32)
    prob, seed = 0.2, 123
    with jax.default_device(CPU_DEVICE):
        actual = jitn(
            w_loc, w_scale, prob, seed,
            shape=shape, transpose=transpose, corder=corder,
            backend='numba',
        )
    expected = dense_normal_reference(
        w_loc, w_scale, prob, seed,
        shape=shape, transpose=transpose, corder=corder,
    )
    assert np.allclose(np.asarray(actual), expected, rtol=5e-6, atol=5e-6)


@requires_cpu_jitnmv
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('corder', [True, False])
def test_jitnmv_numba_matches_light_rng_reference(transpose, corder):
    shape = (13, 17)
    w_loc = jnp.asarray(1.5, dtype=jnp.float32)
    w_scale = jnp.asarray(0.15, dtype=jnp.float32)
    prob, seed = 0.2, 123
    vec_size = shape[0] if transpose else shape[1]
    vector = jnp.linspace(-0.3, 0.7, vec_size, dtype=jnp.float32)
    with jax.default_device(CPU_DEVICE):
        actual = jitnmv(
            w_loc, w_scale, prob, vector, seed,
            shape=shape, transpose=transpose, corder=corder, backend='numba',
        )
    dense = dense_normal_reference(
        w_loc, w_scale, prob, seed,
        shape=shape, transpose=transpose, corder=corder,
    )
    expected = dense @ np.asarray(vector)
    assert np.allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-5)


@requires_cpu_jitnmm
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('corder', [True, False])
def test_jitnmm_numba_matches_light_rng_reference(transpose, corder):
    shape = (13, 17)
    w_loc = jnp.asarray(1.5, dtype=jnp.float32)
    w_scale = jnp.asarray(0.15, dtype=jnp.float32)
    prob, seed = 0.2, 123
    b_rows = shape[0] if transpose else shape[1]
    B = jnp.reshape(jnp.linspace(-0.5, 0.8, b_rows * 3, dtype=jnp.float32), (b_rows, 3))
    with jax.default_device(CPU_DEVICE):
        actual = jitnmm(
            w_loc, w_scale, prob, B, seed,
            shape=shape, transpose=transpose, corder=corder, backend='numba',
        )
    dense = dense_normal_reference(
        w_loc, w_scale, prob, seed,
        shape=shape, transpose=transpose, corder=corder,
    )
    expected = dense @ np.asarray(B)
    assert np.allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("implementation", JITN_IMPLEMENTATIONS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
def test_jitn_transpose_symmetry(implementation, transpose, corder):
    out1 = jitn(1.5, 0.15, 0.1, 123, shape=(100, 50), transpose=transpose,
                corder=corder, backend=implementation)
    out2 = jitn(1.5, 0.15, 0.1, 123, shape=(100, 50), transpose=not transpose,
                corder=not corder, backend=implementation)
    assert jnp.allclose(out1, out2.T)
    jax.block_until_ready((out1, out2))


@pytest.mark.parametrize("implementation", JITNMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitnmv_forward(implementation, shape, corder):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    vector = jnp.asarray(np.random.rand(shape[1]))
    dense = jitn(w_loc, w_scale, prob, seed, shape=shape, corder=corder,
                 backend=implementation)
    out = jitnmv(w_loc, w_scale, prob, vector, seed, shape=shape, corder=corder, backend=implementation)
    expected = dense @ vector
    print(out - expected)
    assert jnp.allclose(out, expected, rtol=1e-3, atol=1e-3)
    jax.block_until_ready((vector, dense, out, expected))


# ---- Forward: jitnmv (vector @ matrix, transpose=True) ----

@pytest.mark.parametrize("implementation", JITNMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitnmv_transpose_forward(implementation, shape, corder):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    vector = jnp.asarray(np.random.rand(shape[0]))
    dense = jitn(w_loc, w_scale, prob, seed, shape=shape, transpose=True,
                 corder=corder, backend=implementation)
    out = jitnmv(w_loc, w_scale, prob, vector, seed, shape=shape, transpose=True, corder=corder, backend=implementation)
    expected = dense @ vector
    assert jnp.allclose(out, expected, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((vector, dense, out, expected))


# ---- Forward: jitnmm (matrix @ matrix, transpose=False) ----

@pytest.mark.parametrize("implementation", JITNMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('k', [10])
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitnmm_forward(implementation, k, shape, corder):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    B = jnp.asarray(np.random.rand(shape[1], k))
    dense = jitn(w_loc, w_scale, prob, seed, shape=shape, corder=corder,
                 backend=implementation)
    out = jitnmm(w_loc, w_scale, prob, B, seed, shape=shape, corder=corder, backend=implementation)
    expected = dense @ B
    assert jnp.allclose(out, expected, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((B, dense, out, expected))


# ---- Forward: jitnmm (matrix.T @ matrix, transpose=True) ----

@pytest.mark.parametrize("implementation", JITNMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('k', [10])
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitnmm_transpose_forward(implementation, k, shape, corder):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    B = jnp.asarray(np.random.rand(shape[0], k))
    dense = jitn(w_loc, w_scale, prob, seed, shape=shape, transpose=True,
                 corder=corder, backend=implementation)
    out = jitnmm(w_loc, w_scale, prob, B, seed, shape=shape, transpose=True, corder=corder, backend=implementation)
    expected = dense @ B
    assert jnp.allclose(out, expected, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((B, dense, out, expected))


# ---- Gradient JVP: jitnmv ----

@pytest.mark.parametrize("implementation", JITNMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitnmv_jvp(implementation, shape, corder, transpose):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    vec_size = shape[0] if transpose else shape[1]
    x = jnp.asarray(np.random.rand(vec_size))

    def f_fn(x):
        return jitnmv(
            w_loc, w_scale, prob, x, seed, shape=shape, transpose=transpose, corder=corder, backend=implementation
        ).sum()

    # Validate JVP via finite differences (avoids jitn vs jitnmv RNG mismatch)
    tangent = jnp.ones_like(x)
    out1, jvp1 = jax.jvp(f_fn, (x,), (tangent,))
    eps = 1e-2
    f_plus = f_fn(x + eps * tangent)
    f_minus = f_fn(x - eps * tangent)
    jvp_fd = (f_plus - f_minus) / (2 * eps)
    assert jnp.allclose(jvp1, jvp_fd, rtol=1e-2, atol=1e-2), (
        f"JVP mismatch: AD={float(jvp1)}, FD={float(jvp_fd)}"
    )
    jax.block_until_ready((x, tangent, out1, jvp1))


# ---- Gradient VJP: jitnmv ----

@pytest.mark.parametrize("implementation", JITNMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitnmv_vjp(implementation, shape, corder, transpose):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    vec_size = shape[0] if transpose else shape[1]
    x = jnp.asarray(np.random.rand(vec_size))

    def f_fn(x):
        return jitnmv(w_loc, w_scale, prob, x, seed, shape=shape, transpose=transpose, corder=corder,
                      backend=implementation).sum()

    # Validate VJP: for f(x) = sum(M @ x), grad_x = M^T @ ones
    # Check that dot(grad, tangent) = JVP(tangent) for a random tangent
    out1, (vjp1,) = jax.value_and_grad(f_fn, argnums=(0,))(x)
    tangent = jnp.asarray(np.random.rand(vec_size))
    _, jvp1 = jax.jvp(f_fn, (x,), (tangent,))
    dot_product = jnp.sum(vjp1 * tangent)
    assert jnp.allclose(dot_product, jvp1, rtol=1e-3, atol=1e-3), (
        f"VJP/JVP consistency mismatch: dot={float(dot_product)}, jvp={float(jvp1)}"
    )
    jax.block_until_ready((x, out1, vjp1, tangent, jvp1))


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
    dense = jitn(w_loc, w_scale, prob, seed, shape=shape, transpose=transpose,
                 corder=corder, backend=implementation)

    def f_mm(x):
        return jitnmm(w_loc, w_scale, prob, x, seed, shape=shape, transpose=transpose, corder=corder,
                      backend=implementation).sum()

    def f_dense(x):
        return (dense @ x).sum()

    tangent_mm = jnp.ones_like(x)
    out1, jvp1 = jax.jvp(f_mm, (x,), (tangent_mm,))
    out2, jvp2 = jax.jvp(f_dense, (x,), (tangent_mm,))
    assert jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(jvp1, jvp2, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((x, dense, tangent_mm, out1, jvp1, out2, jvp2))


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
    dense = jitn(w_loc, w_scale, prob, seed, shape=shape, transpose=transpose,
                 corder=corder, backend=implementation)

    def f_mm(x):
        return jitnmm(w_loc, w_scale, prob, x, seed, shape=shape, transpose=transpose, corder=corder,
                      backend=implementation).sum()

    def f_dense(x):
        return (dense @ x).sum()

    out1, (grad1,) = jax.value_and_grad(f_mm, argnums=(0,))(x)
    out2, (grad2,) = jax.value_and_grad(f_dense, argnums=(0,))(x)
    assert jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(grad1, grad2, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((x, dense, out1, grad1, out2, grad2))


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

    expected = jitnmm(w_loc, w_scale, prob, jnp.asarray(vectors).T, seed, shape=shape,
                      corder=corder, backend=implementation).T
    assert jnp.allclose(results, expected, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((vectors, results, expected))


# ---- Batching: jitnmv over vectors (transpose) ----

@pytest.mark.parametrize("implementation", JITNMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('batch_size', [10])
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitnmv_transpose_vmap_over_vectors(implementation, batch_size, shape, corder):
    w_loc, w_scale, prob, seed = 1.05, 0.1, 0.1, 123
    vectors = brainstate.random.rand(batch_size, shape[0])

    def f(vector):
        return jitnmv(w_loc, w_scale, prob, vector, seed, shape=shape, transpose=True, corder=corder,
                      backend=implementation)

    results = jax.vmap(f)(vectors)
    assert results.shape == (batch_size, shape[1])

    expected = jitnmm(w_loc, w_scale, prob, jnp.asarray(vectors).T, seed, shape=shape,
                      transpose=True, corder=corder, backend=implementation).T
    assert jnp.allclose(results, expected, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((vectors, results, expected))


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
    jax.block_until_ready((w_locs, vector, results, results_loop))


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
    jax.block_until_ready((matrices, outs, outs_loop))


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
        return jitnmm(w_loc, w_scale, prob, mat, seed, shape=shape, transpose=True, corder=corder,
                      backend=implementation)

    outs = jax.vmap(f)(matrices)
    assert outs.shape == (batch_size, shape[1], k)

    outs_loop = brainstate.transform.for_loop(f, matrices)
    assert outs_loop.shape == (batch_size, shape[1], k)

    assert jnp.allclose(outs, outs_loop, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((matrices, outs, outs_loop))


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
    jax.block_until_ready((w_locs, matrix, results, results_loop))


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
    jax.block_until_ready((w_locs, results, results_loop))


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
    jax.block_until_ready((probs, results, results_loop))


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
    jax.block_until_ready((seeds, results, results_loop))


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

    def f_fn(wl):
        return jitnmv(wl, w_scale, prob, vector, seed, shape=shape, transpose=transpose, corder=corder,
                      backend=implementation).sum()

    # Validate via finite differences (avoids jitn vs jitnmv RNG mismatch)
    grad1 = jax.grad(f_fn)(w_loc_arr)
    eps = 1e-2
    f_plus = f_fn(w_loc_arr + eps)
    f_minus = f_fn(w_loc_arr - eps)
    grad_fd = (f_plus - f_minus) / (2 * eps)
    assert jnp.allclose(grad1, grad_fd, rtol=1e-2, atol=1e-2), (
        f"w_loc grad mismatch: AD={float(grad1[0])}, FD={float(grad_fd)}"
    )
    jax.block_until_ready((vector, w_loc_arr, grad1))


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

    def f_fn(ws):
        return jitnmv(w_loc, ws, prob, vector, seed, shape=shape, transpose=transpose, corder=corder,
                      backend=implementation).sum()

    # Validate via finite differences (avoids jitn vs jitnmv RNG mismatch)
    grad1 = jax.grad(f_fn)(w_scale_arr)
    eps = 1e-2
    f_plus = f_fn(w_scale_arr + eps)
    f_minus = f_fn(w_scale_arr - eps)
    grad_fd = (f_plus - f_minus) / (2 * eps)
    assert jnp.allclose(grad1, grad_fd, rtol=1e-2, atol=1e-2), (
        f"w_scale grad mismatch: AD={float(grad1[0])}, FD={float(grad_fd)}"
    )
    jax.block_until_ready((vector, w_scale_arr, grad1))


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

    def loss_fn(wl):
        out = jitnmv(wl, w_scale, prob, vector, seed, shape=shape, transpose=transpose, corder=corder,
                     backend=implementation)
        return jnp.sum((out - target) ** 2)

    # Validate via finite differences (avoids jitn vs jitnmv RNG mismatch)
    grad1 = jax.grad(loss_fn)(w_loc_arr)
    eps = 1e-2
    f_plus = loss_fn(w_loc_arr + eps)
    f_minus = loss_fn(w_loc_arr - eps)
    grad_fd = (f_plus - f_minus) / (2 * eps)
    assert jnp.allclose(grad1, grad_fd, rtol=1e-2, atol=1e-2), (
        f"w_loc loss grad mismatch: AD={float(grad1[0])}, FD={float(grad_fd)}"
    )
    jax.block_until_ready((vector, target, w_loc_arr, grad1))


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

    def loss_fn(ws):
        out = jitnmv(w_loc, ws, prob, vector, seed, shape=shape, transpose=transpose, corder=corder,
                     backend=implementation)
        return jnp.sum((out - target) ** 2)

    # Validate via finite differences (avoids jitn vs jitnmv RNG mismatch)
    grad1 = jax.grad(loss_fn)(w_scale_arr)
    eps = 1e-2
    f_plus = loss_fn(w_scale_arr + eps)
    f_minus = loss_fn(w_scale_arr - eps)
    grad_fd = (f_plus - f_minus) / (2 * eps)
    assert jnp.allclose(grad1, grad_fd, rtol=1e-2, atol=1e-2), (
        f"w_scale loss grad mismatch: AD={float(grad1[0])}, FD={float(grad_fd)}"
    )
    jax.block_until_ready((vector, target, w_scale_arr, grad1))


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

    def f_fn(wl):
        return jitnmm(wl, w_scale, prob, B, seed, shape=shape, transpose=transpose, corder=corder,
                      backend=implementation).sum()

    # Validate via finite differences (avoids jitn vs jitnmm RNG mismatch)
    grad1 = jax.grad(f_fn)(w_loc_arr)
    eps = 1e-2
    f_plus = f_fn(w_loc_arr + eps)
    f_minus = f_fn(w_loc_arr - eps)
    grad_fd = (f_plus - f_minus) / (2 * eps)
    assert jnp.allclose(grad1, grad_fd, rtol=1e-2, atol=1e-2), (
        f"w_loc grad mismatch: AD={float(grad1[0])}, FD={float(grad_fd)}"
    )
    jax.block_until_ready((B, w_loc_arr, grad1))


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

    def f_fn(ws):
        return jitnmm(w_loc, ws, prob, B, seed, shape=shape, transpose=transpose, corder=corder,
                      backend=implementation).sum()

    # Validate via finite differences (avoids jitn vs jitnmm RNG mismatch)
    grad1 = jax.grad(f_fn)(w_scale_arr)
    eps = 1e-2
    f_plus = f_fn(w_scale_arr + eps)
    f_minus = f_fn(w_scale_arr - eps)
    grad_fd = (f_plus - f_minus) / (2 * eps)
    assert jnp.allclose(grad1, grad_fd, rtol=1e-2, atol=1e-2), (
        f"w_scale grad mismatch: AD={float(grad1[0])}, FD={float(grad_fd)}"
    )
    jax.block_until_ready((B, w_scale_arr, grad1))


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

    def loss_fn(wl):
        out = jitnmm(wl, w_scale, prob, B, seed, shape=shape, transpose=transpose, corder=corder,
                     backend=implementation)
        return jnp.sum((out - target) ** 2)

    # Validate via finite differences (avoids jitn vs jitnmm RNG mismatch)
    grad1 = jax.grad(loss_fn)(w_loc_arr)
    eps = 1e-2
    f_plus = loss_fn(w_loc_arr + eps)
    f_minus = loss_fn(w_loc_arr - eps)
    grad_fd = (f_plus - f_minus) / (2 * eps)
    assert jnp.allclose(grad1, grad_fd, rtol=1e-2, atol=1e-2), (
        f"w_loc loss grad mismatch: AD={float(grad1[0])}, FD={float(grad_fd)}"
    )
    jax.block_until_ready((B, target, w_loc_arr, grad1))


# ---- End-to-end VJP: jitnmm w.r.t. w_scale with loss ----

@pytest.mark.parametrize("implementation", JITNMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitnmm_vjp_wscale_with_loss(implementation, shape, corder, transpose):
    k = 10
    rng = np.random.RandomState(123)
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    mat_rows = shape[0] if transpose else shape[1]
    out_rows = shape[1] if transpose else shape[0]
    B = jnp.asarray(rng.rand(mat_rows, k))
    target = jnp.asarray(rng.rand(out_rows, k))
    w_scale_arr = jnp.array([w_scale])

    def loss_fn(ws):
        out = jitnmm(w_loc, ws, prob, B, seed, shape=shape, transpose=transpose, corder=corder, backend=implementation)
        return jnp.sum((out - target) ** 2)

    # Validate via finite differences (avoids jitn vs jitnmm RNG mismatch)
    grad1 = jax.grad(loss_fn)(w_scale_arr)
    eps = 1e-2
    f_plus = loss_fn(w_scale_arr + eps)
    f_minus = loss_fn(w_scale_arr - eps)
    grad_fd = (f_plus - f_minus) / (2 * eps)
    assert jnp.allclose(grad1, grad_fd, rtol=1e-2, atol=1e-2), (
        f"w_scale loss grad mismatch: AD={float(grad1[0])}, FD={float(grad_fd)}"
    )
    jax.block_until_ready((B, target, w_scale_arr, grad1))


# ---- One matrix: vmap over vectors agrees with a loop, and with the numpy walk ----
# ``vmap(jitnmv)`` forwards to ``jitnmm``; before unification the two drew
# different matrices, so this equality did not hold.

@pytest.mark.parametrize("implementation", JITNMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30)])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('corder', [True, False])
def test_vmap_jitnmv_matches_loop(implementation, shape, transpose, corder):
    prob, seed, batch = 0.2, 123, 5
    k = shape[0] if transpose else shape[1]
    vectors = np.random.randn(batch, k).astype(np.float32)

    def f(v):
        return jitnmv(1.5, 0.15, prob, v, seed, shape=shape,
                   transpose=transpose, corder=corder, backend=implementation)

    batched = jax.vmap(f)(jnp.asarray(vectors))
    looped = jnp.stack([f(jnp.asarray(vectors[i])) for i in range(batch)])
    assert jnp.allclose(batched, looped, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((batched, looped))


@pytest.mark.parametrize("implementation", JITN_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(13, 17), (33, 33), (7, 5)])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('corder', [True, False])
def test_jitn_matches_numpy_reference(implementation, shape, transpose, corder):
    # An independent pure-numpy replay of the 32-lane walk -- this pins the drawn
    # matrix itself, not merely the agreement of the kernels with each other.
    prob, seed = 0.2, 123
    actual = np.asarray(jitn(1.5, 0.15, prob, seed, shape=shape, transpose=transpose,
                         corder=corder, backend=implementation))
    expected = dense_normal_reference(1.5, 0.15, prob, seed, shape=shape,
                       transpose=transpose, corder=corder)
    assert np.array_equal(actual != 0, expected != 0)
    assert np.allclose(actual, expected, rtol=1e-4, atol=1e-4)


# ---- Public interface: back to the v0.1.2 parameter lists ----
# ``matrix_mode`` was a 0.2.0-only keyword; with one matrix it is gone, and these
# signatures must again be exactly the ones v0.1.2 shipped.

@pytest.mark.parametrize('fn,expected', [
    (jitn, ('w_loc, w_scale', 'prob', 'seed', 'shape', 'transpose', 'corder', 'backend')),
    (jitnmv, ('w_loc, w_scale', 'prob', 'vector', 'seed', 'shape', 'transpose', 'corder', 'backend')),
    (jitnmm, ('w_loc, w_scale', 'prob', 'B', 'seed', 'shape', 'transpose', 'corder', 'backend')),
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

@pytest.mark.parametrize("implementation", JITN_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(12, 20), (33, 33), (7, 5), (64, 3)])
@pytest.mark.parametrize('corder', [True, False])
def test_generation_is_shape_pair_independent(implementation, shape, corder):
    m, n = shape
    prob, seed = 0.2, 123
    a = jitn(1.5, 0.15, prob, seed, shape=(m, n), transpose=False,
          corder=corder, backend=implementation)
    b = jitn(1.5, 0.15, prob, seed, shape=(n, m), transpose=True,
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
    if 'numba' not in jitn_p.available_backends('cpu') or 'cuda_raw' not in jitn_p.available_backends('gpu'):
        pytest.skip('needs both a CPU numba backend and a CUDA device')
    prob, seed = 0.2, 123
    with jax.default_device(jax.devices('cpu')[0]):
        cpu = np.asarray(jitn(1.5, 0.15, prob, seed, shape=shape, transpose=transpose,
                          corder=corder, backend='numba'))
    with jax.default_device(jax.devices('cuda')[0]):
        gpu = np.asarray(jitn(1.5, 0.15, prob, seed, shape=shape, transpose=transpose,
                          corder=corder, backend='cuda_raw'))
    # The *structure* is bit-exact: it comes from the shared chunk/lane walk.
    assert np.array_equal(cpu != 0, gpu != 0)
    # The *weights* are not, and cannot be: the Acklam probit is a rational
    # expression whose float32 evaluation order differs between numba (x86) and
    # nvcc (PTX). Measured worst case is 1 ULP on 2 of ~5000 non-zeros.
    assert np.allclose(cpu, gpu, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('shape', [(20, 30), (33, 33)])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('corder', [True, False])
def test_numba_and_cuda_matvec_agree(shape, transpose, corder):
    if 'numba' not in jitnmv_p.available_backends('cpu') or 'cuda_raw' not in jitnmv_p.available_backends('gpu'):
        pytest.skip('needs both a CPU numba backend and a CUDA device')
    prob, seed = 0.2, 123
    k = shape[0] if transpose else shape[1]
    v = np.random.rand(k).astype(np.float32)
    with jax.default_device(jax.devices('cpu')[0]):
        cpu = np.asarray(jitnmv(1.5, 0.15, prob, jnp.asarray(v), seed, shape=shape,
                            transpose=transpose, corder=corder, backend='numba'))
    with jax.default_device(jax.devices('cuda')[0]):
        gpu = np.asarray(jitnmv(1.5, 0.15, prob, jnp.asarray(v), seed, shape=shape,
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
#   notrans entry:  jitnmv(shape=(a, b), transpose=False, corder=True )   -> M @ v
#   trans   entry:  jitnmv(shape=(b, a), transpose=False, corder=False)   -> M.T @ u

@pytest.mark.parametrize("implementation", JITNMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(12, 20), (17, 9), (33, 33)])
def test_notrans_and_trans_kernels_draw_one_matrix(implementation, shape):
    a, b = shape
    prob, seed = 0.2, 123
    eye_b = np.eye(b, dtype=np.float32)
    eye_a = np.eye(a, dtype=np.float32)
    via_notrans = np.stack(
        [np.asarray(jitnmv(1.5, 0.15, prob, jnp.asarray(eye_b[j]), seed, shape=(a, b),
                       transpose=False, corder=True, backend=implementation))
         for j in range(b)], axis=1)
    via_trans = np.stack(
        [np.asarray(jitnmv(1.5, 0.15, prob, jnp.asarray(eye_a[i]), seed, shape=(b, a),
                       transpose=False, corder=False, backend=implementation))
         for i in range(a)], axis=1).T
    assert via_notrans.shape == via_trans.shape == (a, b)
    assert np.array_equal(via_notrans != 0, via_trans != 0)
    assert np.allclose(via_notrans, via_trans, rtol=1e-5, atol=1e-5)
    # and it is the matrix the materialization operator writes out
    materialized = np.asarray(jitn(1.5, 0.15, prob, seed, shape=(a, b), transpose=False,
                              corder=True, backend=implementation))
    assert np.allclose(via_notrans, materialized, rtol=1e-5, atol=1e-5)
