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
from brainevent._jit_uniform._test_util import dense_uniform_reference
from brainevent._test_util import requires_gpu

platform = jax.default_backend()
JITU_IMPLEMENTATIONS = tuple(jitu_p.available_backends(platform))
JITUMV_IMPLEMENTATIONS = tuple(jitumv_p.available_backends(platform))
JITUMM_IMPLEMENTATIONS = tuple(jitumm_p.available_backends(platform))
CPU_DEVICE = jax.devices('cpu')[0]
CPU_JITU_IMPLEMENTATIONS = tuple(jitu_p.available_backends('cpu'))
CPU_JITUMV_IMPLEMENTATIONS = tuple(jitumv_p.available_backends('cpu'))
CPU_JITUMM_IMPLEMENTATIONS = tuple(jitumm_p.available_backends('cpu'))

requires_cpu_jitu = pytest.mark.skipif(
    'numba' not in CPU_JITU_IMPLEMENTATIONS,
    reason='No jitu numba backend registered on CPU',
)
requires_cpu_jitumv = pytest.mark.skipif(
    'numba' not in CPU_JITUMV_IMPLEMENTATIONS,
    reason='No jitumv numba backend registered on CPU',
)
requires_cpu_jitumm = pytest.mark.skipif(
    'numba' not in CPU_JITUMM_IMPLEMENTATIONS,
    reason='No jitumm numba backend registered on CPU',
)


@pytest.fixture(autouse=True)
def _seed_rng():
    """Seed the global NumPy RNG so unseeded ``np.random`` probe draws are
    deterministic; keeps tolerance-sensitive autodiff-vs-finite-difference
    gradient checks from being order-dependently flaky."""
    np.random.seed(0x5EED)


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


@requires_cpu_jitu
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('corder', [True, False])
def test_jitu_numba_matches_light_rng_reference(transpose, corder):
    shape = (13, 17)
    with jax.default_device(CPU_DEVICE):
        actual = jitu(
            W_LOW, W_HIGH, 0.2, SEED,
            shape=shape, transpose=transpose, corder=corder,
            backend='numba',
        )
    expected = dense_uniform_reference(
        W_LOW, W_HIGH, 0.2, SEED,
        shape=shape, transpose=transpose, corder=corder,
    )
    assert np.allclose(np.asarray(actual), expected, rtol=1e-6, atol=1e-6)


@requires_cpu_jitumv
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('corder', [True, False])
def test_jitumv_numba_matches_light_rng_reference(transpose, corder):
    shape = (13, 17)
    vec_size = shape[0] if transpose else shape[1]
    vector = jnp.linspace(-0.3, 0.7, vec_size, dtype=jnp.float32)
    with jax.default_device(CPU_DEVICE):
        actual = jitumv(
            W_LOW, W_HIGH, 0.2, vector, SEED,
            shape=shape, transpose=transpose, corder=corder, backend='numba',
        )
    dense = dense_uniform_reference(
        W_LOW, W_HIGH, 0.2, SEED,
        shape=shape, transpose=transpose, corder=corder,
    )
    expected = dense @ np.asarray(vector)
    assert np.allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-5)


@requires_cpu_jitumm
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('corder', [True, False])
def test_jitumm_numba_matches_light_rng_reference(transpose, corder):
    shape = (13, 17)
    b_rows = shape[0] if transpose else shape[1]
    B = jnp.reshape(jnp.linspace(-0.5, 0.8, b_rows * 3, dtype=jnp.float32), (b_rows, 3))
    with jax.default_device(CPU_DEVICE):
        actual = jitumm(
            W_LOW, W_HIGH, 0.2, B, SEED,
            shape=shape, transpose=transpose, corder=corder, backend='numba',
        )
    dense = dense_uniform_reference(
        W_LOW, W_HIGH, 0.2, SEED,
        shape=shape, transpose=transpose, corder=corder,
    )
    expected = dense @ np.asarray(B)
    assert np.allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    not JITUMV_IMPLEMENTATIONS,
    reason=f'No jitumv implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITUMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('corder', [True, False])
def test_jitumv_forward(implementation, shape, corder):
    vector = jnp.asarray(np.random.rand(shape[1]))
    dense = jitu(W_LOW, W_HIGH, PROB, SEED, shape=shape, corder=corder,
                 backend=implementation)
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
    dense = jitu(W_LOW, W_HIGH, PROB, SEED, shape=shape, corder=corder,
                 backend=implementation)
    out = jitumm(W_LOW, W_HIGH, PROB, matrix, SEED, shape=shape, corder=corder, backend=implementation)
    expected = dense @ matrix
    _assert_allclose(out, expected)
    jax.block_until_ready((matrix, dense, out, expected))


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
    dense = jitu(
        W_LOW, W_HIGH, PROB, SEED, shape=shape, transpose=True, corder=corder,
        backend=implementation,
    )
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
    expected = dense @ matrix
    _assert_allclose(out, expected)
    jax.block_until_ready((matrix, dense, out, expected))


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
    dense = jitu(W_LOW, W_HIGH, PROB, SEED, shape=shape, transpose=transpose, corder=corder,
                 backend=implementation)

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
    dense = jitu(W_LOW, W_HIGH, PROB, SEED, shape=shape, transpose=transpose, corder=corder,
                 backend=implementation)

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
    dense = jitu(W_LOW, W_HIGH, PROB, SEED, shape=shape, transpose=transpose,
                 corder=corder, backend=implementation)

    def f_mm(x):
        return jitumm(W_LOW, W_HIGH, PROB, x, SEED, shape=shape, transpose=transpose, corder=corder,
                      backend=implementation).sum()

    def f_dense(x):
        return (dense @ x).sum()

    tangent_mm = jnp.ones_like(x)
    out1, jvp1 = jax.jvp(f_mm, (x,), (tangent_mm,))
    out2, jvp2 = jax.jvp(f_dense, (x,), (tangent_mm,))
    _assert_allclose(out1, out2)
    _assert_allclose(jvp1, jvp2)
    jax.block_until_ready((x, dense, tangent_mm, out1, jvp1, out2, jvp2))


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
    dense = jitu(W_LOW, W_HIGH, PROB, SEED, shape=shape, transpose=transpose,
                 corder=corder, backend=implementation)

    def f_mm(x):
        return jitumm(W_LOW, W_HIGH, PROB, x, SEED, shape=shape, transpose=transpose, corder=corder,
                      backend=implementation).sum()

    def f_dense(x):
        return (dense @ x).sum()

    out1, (grad1,) = jax.value_and_grad(f_mm, argnums=(0,))(x)
    out2, (grad2,) = jax.value_and_grad(f_dense, argnums=(0,))(x)
    _assert_allclose(out1, out2)
    _assert_allclose(grad1, grad2)
    jax.block_until_ready((x, dense, out1, grad1, out2, grad2))


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

    expected = jitumm(W_LOW, W_HIGH, PROB, jnp.asarray(vectors).T, SEED, shape=shape,
                      corder=corder, backend=implementation).T
    _assert_allclose(results, expected)
    jax.block_until_ready((vectors, results, expected))


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


# ---- One matrix: vmap over vectors agrees with a loop, and with the numpy walk ----
# ``vmap(jitumv)`` forwards to ``jitumm``; before unification the two drew
# different matrices, so this equality did not hold.

@pytest.mark.parametrize("implementation", JITUMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(20, 30)])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('corder', [True, False])
def test_vmap_jitumv_matches_loop(implementation, shape, transpose, corder):
    prob, seed, batch = 0.2, 123, 5
    k = shape[0] if transpose else shape[1]
    vectors = np.random.randn(batch, k).astype(np.float32)

    def f(v):
        return jitumv(-1.5, 1.5, prob, v, seed, shape=shape,
                   transpose=transpose, corder=corder, backend=implementation)

    batched = jax.vmap(f)(jnp.asarray(vectors))
    looped = jnp.stack([f(jnp.asarray(vectors[i])) for i in range(batch)])
    assert jnp.allclose(batched, looped, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((batched, looped))


@pytest.mark.parametrize("implementation", JITU_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(13, 17), (33, 33), (7, 5)])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('corder', [True, False])
def test_jitu_matches_numpy_reference(implementation, shape, transpose, corder):
    # An independent pure-numpy replay of the 32-lane walk -- this pins the drawn
    # matrix itself, not merely the agreement of the kernels with each other.
    prob, seed = 0.2, 123
    actual = np.asarray(jitu(-1.5, 1.5, prob, seed, shape=shape, transpose=transpose,
                         corder=corder, backend=implementation))
    expected = dense_uniform_reference(-1.5, 1.5, prob, seed, shape=shape,
                       transpose=transpose, corder=corder)
    assert np.array_equal(actual != 0, expected != 0)
    assert np.allclose(actual, expected, rtol=1e-4, atol=1e-4)


# ---- Public interface: back to the v0.1.2 parameter lists ----
# ``matrix_mode`` was a 0.2.0-only keyword; with one matrix it is gone, and these
# signatures must again be exactly the ones v0.1.2 shipped.

@pytest.mark.parametrize('fn,expected', [
    (jitu, ('w_low, w_high', 'prob', 'seed', 'shape', 'transpose', 'corder', 'backend')),
    (jitumv, ('w_low, w_high', 'prob', 'vector', 'seed', 'shape', 'transpose', 'corder', 'backend')),
    (jitumm, ('w_low, w_high', 'prob', 'B', 'seed', 'shape', 'transpose', 'corder', 'backend')),
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

@pytest.mark.parametrize("implementation", JITU_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(12, 20), (33, 33), (7, 5), (64, 3)])
@pytest.mark.parametrize('corder', [True, False])
def test_generation_is_shape_pair_independent(implementation, shape, corder):
    m, n = shape
    prob, seed = 0.2, 123
    a = jitu(-1.5, 1.5, prob, seed, shape=(m, n), transpose=False,
          corder=corder, backend=implementation)
    b = jitu(-1.5, 1.5, prob, seed, shape=(n, m), transpose=True,
          corder=corder, backend=implementation)
    assert np.array_equal(np.asarray(a), np.asarray(b))


# ---- jitu: transpose symmetry ----
# ``transpose`` and ``corder`` flipped together give the transposed matrix; with
# only ``transpose`` flipped they are genuinely different matrices (documented).

@pytest.mark.parametrize("implementation", JITU_IMPLEMENTATIONS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
def test_jitu_transpose_symmetry(implementation, transpose, corder):
    out1 = jitu(-1.5, 1.5, 0.1, 123, shape=(100, 50), transpose=transpose, corder=corder,
                backend=implementation)
    out2 = jitu(-1.5, 1.5, 0.1, 123, shape=(100, 50), transpose=not transpose, corder=not corder,
                backend=implementation)
    assert jnp.allclose(out1, out2.T)
    jax.block_until_ready((out1, out2))


# ---- numba (CPU) and cuda_raw (GPU) must draw the *same* matrix ----
# The two backends reimplement the light-RNG walk independently; the whole point
# of the shared chunk/lane keying is that they agree bit for bit. Skipped unless
# both backends are actually available on this machine.

@pytest.mark.parametrize('shape', [(12, 20), (33, 33), (7, 5), (100, 250)])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('corder', [True, False])
@requires_gpu
def test_numba_and_cuda_draw_the_same_matrix(shape, transpose, corder):
    if 'numba' not in jitu_p.available_backends('cpu') or 'cuda_raw' not in jitu_p.available_backends('gpu'):
        pytest.skip('needs both a CPU numba backend and a CUDA device')
    prob, seed = 0.2, 123
    with jax.default_device(jax.devices('cpu')[0]):
        cpu = np.asarray(jitu(-1.5, 1.5, prob, seed, shape=shape, transpose=transpose,
                          corder=corder, backend='numba'))
    with jax.default_device(jax.devices('cuda')[0]):
        gpu = np.asarray(jitu(-1.5, 1.5, prob, seed, shape=shape, transpose=transpose,
                          corder=corder, backend='cuda_raw'))
    assert np.array_equal(cpu != 0, gpu != 0)
    assert np.array_equal(cpu, gpu)


@pytest.mark.parametrize('shape', [(20, 30), (33, 33)])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('corder', [True, False])
@requires_gpu
def test_numba_and_cuda_matvec_agree(shape, transpose, corder):
    if 'numba' not in jitumv_p.available_backends('cpu') or 'cuda_raw' not in jitumv_p.available_backends('gpu'):
        pytest.skip('needs both a CPU numba backend and a CUDA device')
    prob, seed = 0.2, 123
    k = shape[0] if transpose else shape[1]
    v = np.random.rand(k).astype(np.float32)
    with jax.default_device(jax.devices('cpu')[0]):
        cpu = np.asarray(jitumv(-1.5, 1.5, prob, jnp.asarray(v), seed, shape=shape,
                            transpose=transpose, corder=corder, backend='numba'))
    with jax.default_device(jax.devices('cuda')[0]):
        gpu = np.asarray(jitumv(-1.5, 1.5, prob, jnp.asarray(v), seed, shape=shape,
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
#   notrans entry:  jitumv(shape=(a, b), transpose=False, corder=True )   -> M @ v
#   trans   entry:  jitumv(shape=(b, a), transpose=False, corder=False)   -> M.T @ u

@pytest.mark.parametrize("implementation", JITUMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', [(12, 20), (17, 9), (33, 33)])
def test_notrans_and_trans_kernels_draw_one_matrix(implementation, shape):
    a, b = shape
    prob, seed = 0.2, 123
    eye_b = np.eye(b, dtype=np.float32)
    eye_a = np.eye(a, dtype=np.float32)
    via_notrans = np.stack(
        [np.asarray(jitumv(-1.5, 1.5, prob, jnp.asarray(eye_b[j]), seed, shape=(a, b),
                       transpose=False, corder=True, backend=implementation))
         for j in range(b)], axis=1)
    via_trans = np.stack(
        [np.asarray(jitumv(-1.5, 1.5, prob, jnp.asarray(eye_a[i]), seed, shape=(b, a),
                       transpose=False, corder=False, backend=implementation))
         for i in range(a)], axis=1).T
    assert via_notrans.shape == via_trans.shape == (a, b)
    assert np.array_equal(via_notrans != 0, via_trans != 0)
    assert np.allclose(via_notrans, via_trans, rtol=1e-5, atol=1e-5)
    # and it is the matrix the materialization operator writes out
    materialized = np.asarray(jitu(-1.5, 1.5, prob, seed, shape=(a, b), transpose=False,
                              corder=True, backend=implementation))
    assert np.allclose(via_notrans, materialized, rtol=1e-5, atol=1e-5)
