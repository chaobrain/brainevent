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

from brainevent._jit_uniform.csr import jitu_to_csr
from brainevent._jit_uniform.float import jitu, jitu_p, jitumv, jitumv_p, jitumm, jitumm_p

platform = jax.default_backend()
JITU_IMPLEMENTATIONS = tuple(jitu_p.available_backends(platform))
JITUMV_IMPLEMENTATIONS = tuple(jitumv_p.available_backends(platform))
JITUMM_IMPLEMENTATIONS = tuple(jitumm_p.available_backends(platform))


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


def _light_csr(w_low, w_high, *, shape, matrix_mode, corder, backend, transpose=False):
    with pytest.warns(FutureWarning, match="corder.*ignored"):
        return jitu_to_csr(
            w_low,
            w_high,
            PROB,
            SEED,
            shape=shape,
            corder=corder,
            matrix_mode=matrix_mode,
            transpose=transpose,
            backend=backend,
        )


def _light_csr_dense(w_low, w_high, *, shape, matrix_mode, corder, backend, transpose=False):
    return _light_csr(
        w_low,
        w_high,
        shape=shape,
        matrix_mode=matrix_mode,
        corder=corder,
        backend=backend,
        transpose=transpose,
    ).todense()


pytestmark = pytest.mark.skipif(
    platform != 'gpu',
    reason='light JIT uniform CSR/CUDA alignment tests require a GPU backend',
)


@pytest.mark.skipif(
    not JITUMV_IMPLEMENTATIONS,
    reason=f'No jitumv implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITUMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('corder', [True, False])
def test_jitumv_forward(implementation, shape, corder):
    vector = jnp.asarray(np.random.rand(shape[1]))
    dense = _light_csr_dense(
        W_LOW, W_HIGH, shape=shape, matrix_mode="mv", corder=corder, backend=implementation,
    )
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
    dense = _light_csr_dense(
        W_LOW, W_HIGH, shape=shape, matrix_mode="mv", corder=corder, backend=implementation,
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
    _assert_allclose(out, dense.T @ vector)
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
    dense = _light_csr_dense(
        W_LOW, W_HIGH, shape=shape, matrix_mode="mm", corder=corder, backend=implementation,
    )
    out = jitumm(W_LOW, W_HIGH, PROB, matrix, SEED, shape=shape, corder=corder, backend=implementation)
    _assert_allclose(out, dense @ matrix)
    jax.block_until_ready((matrix, dense, out))


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
    dense = _light_csr_dense(
        W_LOW, W_HIGH, shape=shape, matrix_mode="mm", corder=corder, backend=implementation,
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
    _assert_allclose(out, dense.T @ matrix)
    jax.block_until_ready((matrix, dense, out))


@pytest.mark.skipif(
    not JITU_IMPLEMENTATIONS,
    reason=f'No jitu implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITU_IMPLEMENTATIONS)
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('matrix_mode', ["mv", "mm"])
@pytest.mark.parametrize('transpose', [False, True])
def test_jitu_matrix_mode_matches_light_csr_dense(implementation, shape, corder, matrix_mode, transpose):
    dense = jitu(
        W_LOW,
        W_HIGH,
        PROB,
        SEED,
        shape=shape,
        transpose=transpose,
        corder=corder,
        matrix_mode=matrix_mode,
        backend=implementation,
    )
    expected = _light_csr_dense(
        W_LOW, W_HIGH, shape=shape, matrix_mode=matrix_mode, corder=corder, backend=implementation,
        transpose=transpose,
    )
    _assert_allclose(dense, expected)
    jax.block_until_ready((dense, expected))


@pytest.mark.skipif(
    not JITU_IMPLEMENTATIONS,
    reason=f'No jitu implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITU_IMPLEMENTATIONS)
@pytest.mark.parametrize('matrix_mode', ["mv", "mm"])
def test_jitu_transpose_matches_notrans_transpose(implementation, matrix_mode):
    shape = (20, 30)
    notrans = jitu(
        W_LOW, W_HIGH, PROB, SEED,
        shape=shape, matrix_mode=matrix_mode, backend=implementation,
    )
    trans = jitu(
        W_LOW, W_HIGH, PROB, SEED,
        shape=shape, transpose=True, matrix_mode=matrix_mode, backend=implementation,
    )
    _assert_allclose(trans, notrans.T)
    jax.block_until_ready((notrans, trans))


@pytest.mark.skipif(
    not JITUMV_IMPLEMENTATIONS,
    reason=f'No jitumv implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITUMV_IMPLEMENTATIONS)
def test_float_corder_warns_and_is_ignored(implementation):
    shape = (20, 30)
    vector = jnp.asarray(np.random.rand(shape[1]).astype(np.float32))
    expected = jitumv(W_LOW, W_HIGH, PROB, vector, SEED, shape=shape, backend=implementation)
    with pytest.warns(FutureWarning, match="corder.*ignored"):
        out_c = jitumv(W_LOW, W_HIGH, PROB, vector, SEED, shape=shape, corder=True, backend=implementation)
    with pytest.warns(FutureWarning, match="corder.*ignored"):
        out_f = jitumv(W_LOW, W_HIGH, PROB, vector, SEED, shape=shape, corder=False, backend=implementation)
    _assert_allclose(out_c, expected)
    _assert_allclose(out_c, out_f)
    jax.block_until_ready((vector, expected, out_c, out_f))


@pytest.mark.skipif(
    not JITU_IMPLEMENTATIONS or not JITUMV_IMPLEMENTATIONS or not JITUMM_IMPLEMENTATIONS,
    reason=f'No float CUDA implementation on platform={platform}',
)
def test_float_cuda_non_f32_weights_raise():
    shape = (20, 30)
    w_low = jnp.asarray(-1.0, dtype=jnp.float16)
    w_high = jnp.asarray(1.0, dtype=jnp.float16)
    vector = jnp.ones(shape[1], dtype=jnp.float16)
    matrix = jnp.ones((shape[1], 4), dtype=jnp.float16)

    with pytest.raises(NotImplementedError, match="float32"):
        jitu(w_low, w_high, PROB, SEED, shape=shape, backend='cuda_raw')
    with pytest.raises(NotImplementedError, match="float32"):
        jitumv(w_low, w_high, PROB, vector, SEED, shape=shape, backend='cuda_raw')
    with pytest.raises(NotImplementedError, match="float32"):
        jitumm(w_low, w_high, PROB, matrix, SEED, shape=shape, backend='cuda_raw')


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
    dense = _light_csr_dense(
        W_LOW, W_HIGH, shape=shape, matrix_mode="mm", corder=corder, backend=implementation,
    )
    op = dense.T if transpose else dense

    def f_mm(x):
        return jitumm(W_LOW, W_HIGH, PROB, x, SEED, shape=shape, transpose=transpose, corder=corder,
                      backend=implementation).sum()

    def f_dense(x):
        return (op @ x).sum()

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
    dense = _light_csr_dense(
        W_LOW, W_HIGH, shape=shape, matrix_mode="mm", corder=corder, backend=implementation,
    )
    op = dense.T if transpose else dense

    def f_mm(x):
        return jitumm(W_LOW, W_HIGH, PROB, x, SEED, shape=shape, transpose=transpose, corder=corder,
                      backend=implementation).sum()

    def f_dense(x):
        return (op @ x).sum()

    out_mm, (grad_mm,) = jax.value_and_grad(f_mm, argnums=(0,))(x)
    out_dense, (grad_dense,) = jax.value_and_grad(f_dense, argnums=(0,))(x)
    _assert_allclose(out_mm, out_dense)
    _assert_allclose(grad_mm, grad_dense)
    jax.block_until_ready((x, dense, out_mm, grad_mm, out_dense, grad_dense))


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
