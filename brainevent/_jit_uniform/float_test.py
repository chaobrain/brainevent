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

from brainevent._jit_uniform.float import jitu, jitu_p, jitumv, jitumv_p, jitumm, jitumm_p

platform = jax.default_backend()
JITU_IMPLEMENTATIONS = tuple(jitu_p.available_backends(platform))
JITUMV_IMPLEMENTATIONS = tuple(jitumv_p.available_backends(platform))
JITUMM_IMPLEMENTATIONS = tuple(jitumm_p.available_backends(platform))

JITU_PARAMS = JITU_IMPLEMENTATIONS or (None,)
JITUMV_PARAMS = JITUMV_IMPLEMENTATIONS or (None,)
JITUMM_PARAMS = JITUMM_IMPLEMENTATIONS or (None,)

SHAPES = [(20, 30), (100, 50)]
W_LOW = -1.5
W_HIGH = 1.5
PROB = 0.1
SEED = 123


def _assert_allclose(a, b, rtol=1e-4, atol=1e-4):
    assert jnp.allclose(a, b, rtol=rtol, atol=atol)


@pytest.mark.skipif(
    not JITUMV_IMPLEMENTATIONS,
    reason=f'No jitumv implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITUMV_PARAMS)
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('corder', [True, False])
def test_jitumv_forward(implementation, shape, corder):
    vector = jnp.asarray(np.random.rand(shape[1]))
    dense = jitu(W_LOW, W_HIGH, PROB, SEED, shape=shape, corder=corder, backend=implementation)
    out = jitumv(W_LOW, W_HIGH, PROB, vector, SEED, shape=shape, corder=corder, backend=implementation)
    _assert_allclose(out, dense @ vector)


@pytest.mark.skipif(
    not JITUMV_IMPLEMENTATIONS,
    reason=f'No jitumv implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITUMV_PARAMS)
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


@pytest.mark.skipif(
    not JITUMM_IMPLEMENTATIONS,
    reason=f'No jitumm implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITUMM_PARAMS)
@pytest.mark.parametrize('k', [10])
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('corder', [True, False])
def test_jitumm_forward(implementation, k, shape, corder):
    matrix = jnp.asarray(np.random.rand(shape[1], k))
    dense = jitu(W_LOW, W_HIGH, PROB, SEED, shape=shape, corder=corder, backend=implementation)
    out = jitumm(W_LOW, W_HIGH, PROB, matrix, SEED, shape=shape, corder=corder, backend=implementation)
    _assert_allclose(out, dense @ matrix)


@pytest.mark.skipif(
    not JITUMM_IMPLEMENTATIONS,
    reason=f'No jitumm implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITUMM_PARAMS)
@pytest.mark.parametrize('k', [10])
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('corder', [True, False])
def test_jitumm_transpose_forward(implementation, k, shape, corder):
    matrix = jnp.asarray(np.random.rand(shape[0], k))
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
    _assert_allclose(out, dense @ matrix)


@pytest.mark.skipif(
    not JITUMV_IMPLEMENTATIONS,
    reason=f'No jitumv implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITUMV_PARAMS)
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

    out1, jvp1 = jax.jvp(f_fn, (vector,), (jnp.ones_like(vector),))
    out2, jvp2 = jax.jvp(f_dense, (vector,), (jnp.ones_like(vector),))
    _assert_allclose(out1, out2)
    _assert_allclose(jvp1, jvp2)


@pytest.mark.skipif(
    not JITUMV_IMPLEMENTATIONS,
    reason=f'No jitumv implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITUMV_PARAMS)
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


@pytest.mark.skipif(
    not JITUMM_IMPLEMENTATIONS,
    reason=f'No jitumm implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITUMM_PARAMS)
@pytest.mark.parametrize('k', [10])
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitumm_jvp(implementation, k, shape, corder, transpose):
    mat_rows = shape[0] if transpose else shape[1]
    matrix = jnp.asarray(np.random.rand(mat_rows, k))
    dense = jitu(W_LOW, W_HIGH, PROB, SEED, shape=shape, transpose=transpose, corder=corder, backend=implementation)

    def f_fn(x):
        return jitumm(
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

    out1, jvp1 = jax.jvp(f_fn, (matrix,), (jnp.ones_like(matrix),))
    out2, jvp2 = jax.jvp(f_dense, (matrix,), (jnp.ones_like(matrix),))
    _assert_allclose(out1, out2)
    _assert_allclose(jvp1, jvp2)


@pytest.mark.skipif(
    not JITUMM_IMPLEMENTATIONS,
    reason=f'No jitumm implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITUMM_PARAMS)
@pytest.mark.parametrize('k', [10])
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_jitumm_vjp(implementation, k, shape, corder, transpose):
    mat_rows = shape[0] if transpose else shape[1]
    matrix = jnp.asarray(np.random.rand(mat_rows, k))
    dense = jitu(W_LOW, W_HIGH, PROB, SEED, shape=shape, transpose=transpose, corder=corder, backend=implementation)

    def f_fn(x):
        return jitumm(
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

    out1, (vjp1,) = jax.value_and_grad(f_fn, argnums=(0,))(matrix)
    out2, (vjp2,) = jax.value_and_grad(f_dense, argnums=(0,))(matrix)
    _assert_allclose(out1, out2)
    _assert_allclose(vjp1, vjp2)


@pytest.mark.skipif(
    not JITUMV_IMPLEMENTATIONS,
    reason=f'No jitumv implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITUMV_PARAMS)
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


@pytest.mark.skipif(
    not JITUMM_IMPLEMENTATIONS,
    reason=f'No jitumm implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITUMM_PARAMS)
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


@pytest.mark.skipif(
    not JITU_IMPLEMENTATIONS,
    reason=f'No jitu implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITU_PARAMS)
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


@pytest.mark.skipif(
    not JITU_IMPLEMENTATIONS,
    reason=f'No jitu implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITU_PARAMS)
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


@pytest.mark.skipif(
    not JITU_IMPLEMENTATIONS,
    reason=f'No jitu implementation on platform={platform}',
)
@pytest.mark.parametrize('implementation', JITU_PARAMS)
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
