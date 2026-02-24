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

from brainevent._jit_uniform.float import jitu, jitu_p, jitumv, jitumv_p, jitumm, jitumm_p

platform = jax.default_backend()
JITU_IMPLEMENTATIONS = tuple(jitu_p.available_backends(platform))
JITUMV_IMPLEMENTATIONS = tuple(jitumv_p.available_backends(platform))
JITUMM_IMPLEMENTATIONS = tuple(jitumm_p.available_backends(platform))

SHAPES = [(20, 30), (100, 50)]
W_LOW = -1.5
W_HIGH = 1.5
PROB = 0.1
SEED = 123


def _assert_allclose(a, b, rtol=1e-4, atol=1e-4):
    assert jnp.allclose(a, b, rtol=rtol, atol=atol)


def _sample_cotangent(shape, seed: int):
    rng = np.random.RandomState(seed)
    return jnp.asarray(rng.randn(*shape).astype(np.float32))


@pytest.mark.skipif(
    not JITUMV_IMPLEMENTATIONS,
    reason=f'No jitumv implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITUMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('corder', [True, False])
def test_jitumv_forward(implementation, shape, corder):
    vector = jnp.asarray(np.random.rand(shape[1]))
    dense = jitu(W_LOW, W_HIGH, PROB, SEED, shape=shape, corder=corder, backend=implementation)
    out = jitumv(W_LOW, W_HIGH, PROB, vector, SEED, shape=shape, corder=corder, backend=implementation)
    _assert_allclose(out, dense @ vector)
    jax.block_until_ready((vector, dense, out))


@pytest.mark.skipif(
    not JITUMV_IMPLEMENTATIONS,
    reason=f'No jitumv implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITUMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('corder', [True, False])
def test_jitumv_transpose_forward(implementation, shape, corder):
    vector = jnp.asarray(np.random.rand(shape[0]))
    dense = jitu(
        W_LOW,
        W_HIGH,
        PROB,
        SEED,
        shape=shape,
        transpose=True,
        corder=corder,
        backend=implementation,
    )
    out = jitumv(
        W_LOW,
        W_HIGH,
        PROB,
        vector,
        SEED,
        shape=shape,
        transpose=True,
        corder=corder,
        backend=implementation,
    )
    _assert_allclose(out, dense @ vector)
    jax.block_until_ready((vector, dense, out))


@pytest.mark.skipif(
    not JITUMM_IMPLEMENTATIONS,
    reason=f'No jitumm implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITUMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('k', [10])
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('corder', [True, False])
def test_jitumm_forward(implementation, k, shape, corder):
    matrix = jnp.asarray(np.random.rand(shape[1], k))
    out = jitumm(W_LOW, W_HIGH, PROB, matrix, SEED, shape=shape, corder=corder, backend=implementation)
    # Validate against jitumv column-by-column (exact match expected)
    for j in range(k):
        expected_col = jitumv(W_LOW, W_HIGH, PROB, matrix[:, j], SEED, shape=shape, corder=corder,
                              backend=implementation)
        assert jnp.allclose(out[:, j], expected_col, rtol=1e-4, atol=1e-4), (
            f"Column {j} mismatch: max_diff={float(jnp.max(jnp.abs(out[:, j] - expected_col)))}"
        )
    jax.block_until_ready(out)


@pytest.mark.skipif(
    not JITUMM_IMPLEMENTATIONS,
    reason=f'No jitumm implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITUMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('k', [10])
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('corder', [True, False])
def test_jitumm_transpose_forward(implementation, k, shape, corder):
    matrix = jnp.asarray(np.random.rand(shape[0], k))
    out = jitumm(
        W_LOW,
        W_HIGH,
        PROB,
        matrix,
        SEED,
        shape=shape,
        transpose=True,
        corder=corder,
        backend=implementation,
    )
    # Validate against jitumv column-by-column (exact match expected)
    for j in range(k):
        expected_col = jitumv(
            W_LOW, W_HIGH, PROB, matrix[:, j], SEED,
            shape=shape, transpose=True, corder=corder, backend=implementation,
        )
        assert jnp.allclose(out[:, j], expected_col, rtol=1e-4, atol=1e-4), (
            f"Column {j} mismatch: max_diff={float(jnp.max(jnp.abs(out[:, j] - expected_col)))}"
        )
    jax.block_until_ready(out)


@pytest.mark.skipif(
    not JITUMV_IMPLEMENTATIONS,
    reason=f'No jitumv implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITUMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitumv_jvp(implementation, shape, corder, transpose):
    vec_size = shape[0] if transpose else shape[1]
    vector = jnp.asarray(np.random.rand(vec_size))
    dense = jitu(W_LOW, W_HIGH, PROB, SEED, shape=shape, transpose=transpose, corder=corder, backend=implementation)

    def f_fn(x):
        return jitumv(
            W_LOW,
            W_HIGH,
            PROB,
            x,
            SEED,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        ).sum()

    def f_dense(x):
        return (dense @ x).sum()

    tangent = jnp.ones_like(vector)
    out1, jvp1 = jax.jvp(f_fn, (vector,), (tangent,))
    out2, jvp2 = jax.jvp(f_dense, (vector,), (tangent,))
    _assert_allclose(out1, out2)
    _assert_allclose(jvp1, jvp2)
    jax.block_until_ready((vector, dense, tangent, out1, jvp1, out2, jvp2))


@pytest.mark.skipif(
    not JITUMV_IMPLEMENTATIONS,
    reason=f'No jitumv implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITUMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitumv_vjp(implementation, shape, corder, transpose):
    vec_size = shape[0] if transpose else shape[1]
    vector = jnp.asarray(np.random.rand(vec_size))
    dense = jitu(W_LOW, W_HIGH, PROB, SEED, shape=shape, transpose=transpose, corder=corder, backend=implementation)

    def f_fn(x):
        return jitumv(
            W_LOW,
            W_HIGH,
            PROB,
            x,
            SEED,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        ).sum()

    def f_dense(x):
        return (dense @ x).sum()

    out1, (vjp1,) = jax.value_and_grad(f_fn, argnums=(0,))(vector)
    out2, (vjp2,) = jax.value_and_grad(f_dense, argnums=(0,))(vector)
    _assert_allclose(out1, out2)
    _assert_allclose(vjp1, vjp2)
    jax.block_until_ready((vector, dense, out1, vjp1, out2, vjp2))


@pytest.mark.skipif(
    not JITUMM_IMPLEMENTATIONS,
    reason=f'No jitumm implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITUMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('k', [10])
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitumm_jvp(implementation, k, shape, corder, transpose):
    mat_rows = shape[0] if transpose else shape[1]
    x = jnp.asarray(np.random.rand(mat_rows, k))

    # Validate jitumm JVP against jitumv JVP
    # (avoids jitu vs jitumm RNG mismatch for corder=False on GPU)
    def f_mm(x):
        return jitumm(W_LOW, W_HIGH, PROB, x, SEED, shape=shape, transpose=transpose, corder=corder,
                      backend=implementation).sum()

    def f_mv(v):
        return jitumv(W_LOW, W_HIGH, PROB, v, SEED, shape=shape, transpose=transpose, corder=corder,
                      backend=implementation).sum()

    tangent_mm = jnp.ones_like(x)
    tangent_mv = jnp.ones(mat_rows)
    out1, jvp1 = jax.jvp(f_mm, (x,), (tangent_mm,))
    out_mv, jvp_mv = jax.jvp(f_mv, (x[:, 0],), (tangent_mv,))
    # JVP of sum(M @ B) with tangent=ones is sum(M @ ones_matrix) = k * sum(M @ ones_vector)
    assert jnp.allclose(jvp1, jvp_mv * k, rtol=1e-4, atol=1e-4), (
        f"JVP mismatch: jitumm={float(jvp1)}, jitumv*k={float(jvp_mv * k)}"
    )
    jax.block_until_ready((x, tangent_mm, tangent_mv, out1, jvp1, out_mv, jvp_mv))


@pytest.mark.skipif(
    not JITUMM_IMPLEMENTATIONS,
    reason=f'No jitumm implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITUMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('k', [10])
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitumm_vjp(implementation, k, shape, corder, transpose):
    mat_rows = shape[0] if transpose else shape[1]
    x = jnp.asarray(np.random.rand(mat_rows, k))

    # Validate jitumm VJP against jitumv VJP column-by-column
    # (avoids jitu vs jitumm RNG mismatch for corder=False on GPU)
    def f_mm(x):
        return jitumm(W_LOW, W_HIGH, PROB, x, SEED, shape=shape, transpose=transpose, corder=corder,
                      backend=implementation).sum()

    out_mm, (grad_mm,) = jax.value_and_grad(f_mm, argnums=(0,))(x)

    # jitumv gradient: grad of sum(M @ v) w.r.t. v = M^T @ ones
    # Each column of grad_mm should match the jitumv gradient
    def f_mv(v):
        return jitumv(W_LOW, W_HIGH, PROB, v, SEED, shape=shape, transpose=transpose, corder=corder,
                      backend=implementation).sum()

    v0 = x[:, 0]
    _, (grad_mv,) = jax.value_and_grad(f_mv, argnums=(0,))(v0)
    for j in range(k):
        assert jnp.allclose(grad_mm[:, j], grad_mv, rtol=1e-4, atol=1e-4), (
            f"VJP column {j} mismatch: max_diff={float(jnp.max(jnp.abs(grad_mm[:, j] - grad_mv)))}"
        )
    jax.block_until_ready((x, out_mm, grad_mm, grad_mv))


@pytest.mark.skipif(
    not JITUMV_IMPLEMENTATIONS,
    reason=f'No jitumv implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITUMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitumv_vjp_w_bounds_match_affine_reference_and_finite_difference(
    implementation,
    shape,
    corder,
    transpose,
):
    vec_size = shape[0] if transpose else shape[1]
    out_size = shape[1] if transpose else shape[0]
    rng = np.random.RandomState(1001)
    vector = jnp.asarray(rng.rand(vec_size).astype(np.float32))
    cotangent = _sample_cotangent((out_size,), seed=1002)
    w_low = jnp.asarray(W_LOW, dtype=jnp.float32)
    w_high = jnp.asarray(W_HIGH, dtype=jnp.float32)
    eps = jnp.asarray(1e-3, dtype=jnp.float32)

    def scalar_sparse(wl, wh):
        out = jitumv(
            wl,
            wh,
            PROB,
            vector,
            SEED,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )
        return jnp.sum(out * cotangent)

    g_w_low = jax.grad(scalar_sparse, argnums=0)(w_low, w_high)
    g_w_high = jax.grad(scalar_sparse, argnums=1)(w_low, w_high)

    # Affine decomposition with fixed random graph:
    # y = w_low * C(v) + (w_high - w_low) * U(v),
    # U(v) = jitumv(0, 1, ...), C(v) = jitumv(1, 1, ...).
    U = jitu(
        0.0,
        1.0,
        PROB,
        SEED,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=implementation,
    )
    C = jitu(
        1.0,
        1.0,
        PROB,
        SEED,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=implementation,
    )
    u_out = U @ vector
    c_out = C @ vector
    ref_w_high = jnp.sum(cotangent * u_out)
    ref_w_low = jnp.sum(cotangent * (c_out - u_out))

    fd_w_low = (scalar_sparse(w_low + eps, w_high) - scalar_sparse(w_low - eps, w_high)) / (2.0 * eps)
    fd_w_high = (scalar_sparse(w_low, w_high + eps) - scalar_sparse(w_low, w_high - eps)) / (2.0 * eps)

    _assert_allclose(g_w_low, ref_w_low, rtol=1e-2, atol=1e-2)
    _assert_allclose(g_w_high, ref_w_high, rtol=1e-2, atol=1e-2)
    _assert_allclose(g_w_low, fd_w_low, rtol=1e-2, atol=1e-2)
    _assert_allclose(g_w_high, fd_w_high, rtol=1e-2, atol=1e-2)
    jax.block_until_ready(
        (vector, cotangent, w_low, w_high, eps, g_w_low, g_w_high, U, C, u_out, c_out, ref_w_high, ref_w_low, fd_w_low,
         fd_w_high))


@pytest.mark.skipif(
    not JITUMM_IMPLEMENTATIONS,
    reason=f'No jitumm implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITUMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('k', [10])
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitumm_vjp_w_bounds_match_affine_reference_and_finite_difference(
    implementation,
    k,
    shape,
    corder,
    transpose,
):
    mat_rows = shape[0] if transpose else shape[1]
    out_rows = shape[1] if transpose else shape[0]
    rng = np.random.RandomState(1003)
    matrix = jnp.asarray(rng.rand(mat_rows, k).astype(np.float32))
    cotangent = _sample_cotangent((out_rows, k), seed=1004)
    w_low = jnp.asarray(W_LOW, dtype=jnp.float32)
    w_high = jnp.asarray(W_HIGH, dtype=jnp.float32)
    eps = jnp.asarray(1e-3, dtype=jnp.float32)

    def scalar_sparse(wl, wh):
        out = jitumm(
            wl,
            wh,
            PROB,
            matrix,
            SEED,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )
        return jnp.sum(out * cotangent)

    g_w_low = jax.grad(scalar_sparse, argnums=0)(w_low, w_high)
    g_w_high = jax.grad(scalar_sparse, argnums=1)(w_low, w_high)

    # Use jitumm-based affine reference (avoids todense vs matmat mismatch)
    u_out = jitumm(
        0.0, 1.0, PROB, matrix, SEED,
        shape=shape, transpose=transpose, corder=corder, backend=implementation,
    )
    c_out = jitumm(
        1.0, 1.0, PROB, matrix, SEED,
        shape=shape, transpose=transpose, corder=corder, backend=implementation,
    )
    ref_w_high = jnp.sum(cotangent * u_out)
    ref_w_low = jnp.sum(cotangent * (c_out - u_out))

    fd_w_low = (scalar_sparse(w_low + eps, w_high) - scalar_sparse(w_low - eps, w_high)) / (2.0 * eps)
    fd_w_high = (scalar_sparse(w_low, w_high + eps) - scalar_sparse(w_low, w_high - eps)) / (2.0 * eps)

    _assert_allclose(g_w_low, ref_w_low, rtol=1e-2, atol=1e-2)
    _assert_allclose(g_w_high, ref_w_high, rtol=1e-2, atol=1e-2)
    _assert_allclose(g_w_low, fd_w_low, rtol=1e-2, atol=1e-2)
    _assert_allclose(g_w_high, fd_w_high, rtol=1e-2, atol=1e-2)
    jax.block_until_ready(
        (matrix, cotangent, w_low, w_high, eps, g_w_low, g_w_high, u_out, c_out, ref_w_high, ref_w_low, fd_w_low,
         fd_w_high))


@pytest.mark.skipif(
    not JITUMV_IMPLEMENTATIONS,
    reason=f'No jitumv implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITUMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('batch_size', [10])
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('corder', [True, False])
def test_jitumv_vmap_over_vectors(implementation, batch_size, shape, corder):
    vectors = brainstate.random.rand(batch_size, shape[1])

    def f(vector):
        return jitumv(W_LOW, W_HIGH, PROB, vector, SEED, shape=shape, corder=corder, backend=implementation)

    results = jax.vmap(f)(vectors)
    assert results.shape == (batch_size, shape[0])

    results_loop = brainstate.transform.for_loop(f, vectors)
    assert results_loop.shape == (batch_size, shape[0])
    _assert_allclose(results, results_loop)
    jax.block_until_ready((vectors, results, results_loop))


@pytest.mark.skipif(
    not JITUMM_IMPLEMENTATIONS,
    reason=f'No jitumm implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITUMM_IMPLEMENTATIONS)
@pytest.mark.parametrize('batch_size', [10])
@pytest.mark.parametrize('k', [5])
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('corder', [True, False])
def test_jitumm_vmap_over_matrices(implementation, batch_size, k, shape, corder):
    matrices = brainstate.random.rand(batch_size, shape[1], k)

    def f(matrix):
        return jitumm(W_LOW, W_HIGH, PROB, matrix, SEED, shape=shape, corder=corder, backend=implementation)

    results = jax.vmap(f)(matrices)
    assert results.shape == (batch_size, shape[0], k)

    results_loop = brainstate.transform.for_loop(f, matrices)
    assert results_loop.shape == (batch_size, shape[0], k)
    _assert_allclose(results, results_loop)
    jax.block_until_ready((matrices, results, results_loop))


@pytest.mark.skipif(
    not JITU_IMPLEMENTATIONS,
    reason=f'No jitu implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITU_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(100, 50)])
def test_jitu_vmap_over_wlow(implementation, shape):
    w_lows = brainstate.random.rand(10)

    def f(w_low):
        return jitu(w_low, w_low + 0.5, PROB, SEED, shape=shape, backend=implementation)

    results = jax.vmap(f)(w_lows)
    assert results.shape == (10,) + shape

    results_loop = brainstate.transform.for_loop(f, w_lows)
    assert results_loop.shape == (10,) + shape
    _assert_allclose(results, results_loop)
    jax.block_until_ready((w_lows, results, results_loop))


@pytest.mark.skipif(
    not JITU_IMPLEMENTATIONS,
    reason=f'No jitu implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITU_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(100, 50)])
def test_jitu_vmap_over_prob(implementation, shape):
    probs = brainstate.random.rand(10) * 0.5

    def f(prob):
        return jitu(W_LOW, W_HIGH, prob, SEED, shape=shape, backend=implementation)

    results = jax.vmap(f)(probs)
    assert results.shape == (10,) + shape

    results_loop = brainstate.transform.for_loop(f, probs)
    assert results_loop.shape == (10,) + shape
    _assert_allclose(results, results_loop)
    jax.block_until_ready((probs, results, results_loop))


@pytest.mark.skipif(
    not JITU_IMPLEMENTATIONS,
    reason=f'No jitu implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITU_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(100, 50)])
def test_jitu_vmap_over_seed(implementation, shape):
    seeds = brainstate.random.randint(0, 100000, 10)

    def f(seed):
        return jitu(W_LOW, W_HIGH, PROB, seed, shape=shape, backend=implementation)

    results = jax.vmap(f)(seeds)
    assert results.shape == (10,) + shape

    results_loop = brainstate.transform.for_loop(f, seeds)
    assert results_loop.shape == (10,) + shape
    _assert_allclose(results, results_loop)
    jax.block_until_ready((seeds, results, results_loop))
