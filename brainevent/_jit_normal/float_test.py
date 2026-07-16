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

from brainevent._jit_normal.float import jitn, jitn_p, jitnmv, jitnmv_p, jitnmm, jitnmm_p
from brainevent._jit_normal.csr import jitn_to_csr

PROB = 0.1
SEED = 123

platform = jax.default_backend()
JITN_IMPLEMENTATIONS = tuple(jitn_p.available_backends(platform))
JITNMV_IMPLEMENTATIONS = tuple(jitnmv_p.available_backends(platform))
JITNMM_IMPLEMENTATIONS = tuple(jitnmm_p.available_backends(platform))

pytestmark = pytest.mark.skipif(
    platform != 'gpu',
    reason='light JIT normal CSR/CUDA alignment tests require a GPU backend',
)


def _skip_if_no_backend(implementation):
    if implementation is None:
        pytest.skip(f"No _jit_normal backend is available on {platform!r}.")


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


@pytest.mark.parametrize("implementation", JITN_IMPLEMENTATIONS or (None,))
@pytest.mark.parametrize("matrix_mode", ["mv", "mm"])
def test_jitn_dense_mv_mm_transpose_matches_transposed_notrans(implementation, matrix_mode):
    _skip_if_no_backend(implementation)
    shape = (8, 11)
    dense = jitn(1.5, 0.15, PROB, SEED, shape=shape, matrix_mode=matrix_mode, backend=implementation)
    dense_t = jitn(
        1.5,
        0.15,
        PROB,
        SEED,
        shape=shape,
        matrix_mode=matrix_mode,
        transpose=True,
        backend=implementation,
    )
    assert jnp.allclose(dense_t, dense.T, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((dense, dense_t))


@pytest.mark.parametrize("implementation", JITNMM_IMPLEMENTATIONS or (None,))
@pytest.mark.parametrize("matrix_mode", ["mv", "mm"])
@pytest.mark.parametrize("transpose", [False, True])
def test_jitnmm_matrix_mode_mv_and_mm_forward_transpose(implementation, matrix_mode, transpose):
    _skip_if_no_backend(implementation)
    shape = (8, 11)
    batch = 3
    rhs_rows = shape[0] if transpose else shape[1]
    rhs = jnp.asarray(np.random.rand(rhs_rows, batch), dtype=jnp.float32)
    dense = jitn(
        1.5,
        0.15,
        PROB,
        SEED,
        shape=shape,
        matrix_mode=matrix_mode,
        transpose=transpose,
        backend=implementation,
    )
    out = jitnmm(
        1.5,
        0.15,
        PROB,
        rhs,
        SEED,
        shape=shape,
        matrix_mode=matrix_mode,
        transpose=transpose,
        backend=implementation,
    )
    assert jnp.allclose(out, dense @ rhs, rtol=1e-3, atol=1e-3)
    jax.block_until_ready((dense, out))


@pytest.mark.parametrize("implementation", JITN_IMPLEMENTATIONS or (None,))
@pytest.mark.parametrize("matrix_mode", ["mv", "mm"])
@pytest.mark.parametrize("transpose", [False, True])
def test_jitn_to_csr_mv_mm_transpose_matches_dense(implementation, matrix_mode, transpose):
    _skip_if_no_backend(implementation)
    shape = (8, 11)
    dense = jitn(
        1.5,
        0.15,
        PROB,
        SEED,
        shape=shape,
        matrix_mode=matrix_mode,
        transpose=transpose,
        backend=implementation,
    )
    csr = jitn_to_csr(
        1.5,
        0.15,
        PROB,
        SEED,
        shape=shape,
        matrix_mode=matrix_mode,
        transpose=transpose,
        backend=implementation,
    )
    assert jnp.allclose(csr.todense(), dense, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((dense, csr.data))


@pytest.mark.parametrize("implementation", JITN_IMPLEMENTATIONS or (None,))
def test_corder_deprecated_and_ignored(implementation):
    _skip_if_no_backend(implementation)
    shape = (8, 11)
    baseline = jitn(1.5, 0.15, PROB, SEED, shape=shape, backend=implementation)
    with pytest.warns(FutureWarning, match="corder.*deprecated.*ignored"):
        corder_true = jitn(1.5, 0.15, PROB, SEED, shape=shape, corder=True, backend=implementation)
    with pytest.warns(FutureWarning, match="corder.*deprecated.*ignored"):
        corder_false = jitn(1.5, 0.15, PROB, SEED, shape=shape, corder=False, backend=implementation)
    assert jnp.allclose(corder_true, baseline)
    assert jnp.allclose(corder_false, baseline)
    jax.block_until_ready((baseline, corder_true, corder_false))


# ---- Forward: jitnmv (matrix @ vector, transpose=False) ----

@pytest.mark.parametrize("implementation", JITNMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitnmv_forward(implementation, shape, corder):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    vector = jnp.asarray(np.random.rand(shape[1]))
    dense = jitn(w_loc, w_scale, prob, seed, shape=shape, corder=corder, backend=implementation)
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
    dense = jitn(w_loc, w_scale, prob, seed, shape=shape, transpose=True, corder=corder, backend=implementation)
    out = jitnmv(w_loc, w_scale, prob, vector, seed, shape=shape, transpose=True, corder=corder, backend=implementation)
    expected = dense @ vector
    assert jnp.allclose(out, expected, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((vector, dense, out, expected))


def _light_csr(w_loc, w_scale, *, shape, matrix_mode, corder, backend):
    with pytest.warns(FutureWarning, match="corder.*deprecated.*ignored"):
        return jitn_to_csr(
            w_loc,
            w_scale,
            PROB,
            SEED,
            shape=shape,
            corder=corder,
            matrix_mode=matrix_mode,
            backend=backend,
        )


def _light_csr_dense(w_loc, w_scale, *, shape, matrix_mode, corder, backend):
    return _light_csr(
        w_loc, w_scale, shape=shape, matrix_mode=matrix_mode, corder=corder, backend=backend
    ).todense()


# ---- Forward: jitnmm (matrix @ matrix, transpose=False) ----

@pytest.mark.parametrize("implementation", JITNMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('k', [10])
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitnmm_forward(implementation, k, shape, corder):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    B = jnp.asarray(np.random.rand(shape[1], k))
    out = jitnmm(w_loc, w_scale, prob, B, seed, shape=shape, corder=corder, backend=implementation)
    dense = _light_csr_dense(
        jnp.asarray(w_loc, dtype=jnp.float32), jnp.asarray(w_scale, dtype=jnp.float32),
        shape=shape, matrix_mode="mm", corder=corder, backend=implementation,
    )
    expected = dense @ B
    assert jnp.allclose(out, expected, rtol=1e-3, atol=1e-3)
    jax.block_until_ready(out)


# ---- Forward: jitnmm (matrix.T @ matrix, transpose=True) ----

@pytest.mark.parametrize("implementation", JITNMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('k', [10])
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('corder', [True, False])
def test_jitnmm_transpose_forward(implementation, k, shape, corder):
    w_loc, w_scale, prob, seed = 1.5, 0.15, 0.1, 123
    B = jnp.asarray(np.random.rand(shape[0], k))
    out = jitnmm(w_loc, w_scale, prob, B, seed, shape=shape, transpose=True, corder=corder, backend=implementation)
    dense = _light_csr_dense(
        jnp.asarray(w_loc, dtype=jnp.float32), jnp.asarray(w_scale, dtype=jnp.float32),
        shape=shape, matrix_mode="mm", corder=corder, backend=implementation,
    )
    expected = dense.T @ B
    assert jnp.allclose(out, expected, rtol=1e-3, atol=1e-3)
    jax.block_until_ready(out)


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
    dense = _light_csr_dense(
        jnp.asarray(w_loc, dtype=jnp.float32), jnp.asarray(w_scale, dtype=jnp.float32),
        shape=shape, matrix_mode="mm", corder=corder, backend=implementation,
    )
    op = dense.T if transpose else dense

    def f_mm(x):
        return jitnmm(w_loc, w_scale, prob, x, seed, shape=shape, transpose=transpose, corder=corder,
                      backend=implementation).sum()

    def f_dense(x):
        return (op @ x).sum()

    tangent_mm = jnp.ones_like(x)
    out1, jvp1 = jax.jvp(f_mm, (x,), (tangent_mm,))
    out2, jvp2 = jax.jvp(f_dense, (x,), (tangent_mm,))
    assert jnp.allclose(out1, out2, rtol=1e-3, atol=1e-3)
    assert jnp.allclose(jvp1, jvp2, rtol=1e-3, atol=1e-3)
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
    dense = _light_csr_dense(
        jnp.asarray(w_loc, dtype=jnp.float32), jnp.asarray(w_scale, dtype=jnp.float32),
        shape=shape, matrix_mode="mm", corder=corder, backend=implementation,
    )
    op = dense.T if transpose else dense

    def f_mm(x):
        return jitnmm(w_loc, w_scale, prob, x, seed, shape=shape, transpose=transpose, corder=corder,
                      backend=implementation).sum()

    def f_dense(x):
        return (op @ x).sum()

    out_mm, (grad_mm,) = jax.value_and_grad(f_mm, argnums=(0,))(x)
    out_dense, (grad_dense,) = jax.value_and_grad(f_dense, argnums=(0,))(x)
    assert jnp.allclose(out_mm, out_dense, rtol=1e-3, atol=1e-3)
    assert jnp.allclose(grad_mm, grad_dense, rtol=1e-3, atol=1e-3)
    jax.block_until_ready((x, dense, out_mm, grad_mm, out_dense, grad_dense))


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
    jax.block_until_ready((vectors, results, results_loop))


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

    results_loop = brainstate.transform.for_loop(f, vectors)
    assert results_loop.shape == (batch_size, shape[1])

    assert jnp.allclose(results, results_loop, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((vectors, results, results_loop))


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
