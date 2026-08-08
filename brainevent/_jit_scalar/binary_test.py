# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Keep GPU matmul reference numerics stable (avoid TF32 drift in dense @ B checks).
if jax.default_backend() == 'gpu' and jax.config.jax_default_matmul_precision is None:
    jax.config.update('jax_default_matmul_precision', 'highest')

from brainevent._jit_scalar.binary import (
    binary_jitsmv,
    binary_jitsmv_p,
    binary_jitsmm,
    binary_jitsmm_p,
)
from brainevent._jit_scalar.float import jitsmv, jitsmm
from brainevent._test_util import allclose

platform = jax.default_backend()
JITSMV_IMPLEMENTATIONS = tuple(binary_jitsmv_p.available_backends(platform))
JITSMM_IMPLEMENTATIONS = tuple(binary_jitsmm_p.available_backends(platform))

if platform == 'cpu':
    SHAPES = ((20, 30), (100, 50))
else:
    SHAPES = ((20, 30), (100, 50), (400, 300))


def _implementation_params(implementations, op_name: str):
    if implementations:
        return [pytest.param(impl, id=impl) for impl in implementations]
    return [
        pytest.param(
            None,
            marks=pytest.mark.skip(reason=f'No {op_name} implementations on platform={platform}'),
            id=f'no-{op_name}',
        )
    ]


JITSMV_PARAMS = _implementation_params(JITSMV_IMPLEMENTATIONS, 'binary_jitsmv')
JITSMM_PARAMS = _implementation_params(JITSMM_IMPLEMENTATIONS, 'binary_jitsmm')


def _binary_events(x, dtype=jnp.float32):
    return jnp.asarray(jnp.asarray(x) > 0, dtype=dtype)


def _sample_vector(size: int, event_dtype, seed: int):
    rng = np.random.RandomState(seed)
    if event_dtype is bool:
        return jnp.asarray(rng.rand(size) > 0.5)
    raw = rng.rand(size).astype(np.float32)
    mask = rng.rand(size) > 0.5
    return jnp.asarray(raw * mask)


def _sample_matrix(rows: int, cols: int, event_dtype, seed: int):
    rng = np.random.RandomState(seed)
    if event_dtype is bool:
        return jnp.asarray(rng.rand(rows, cols) > 0.5)
    raw = rng.rand(rows, cols).astype(np.float32)
    mask = rng.rand(rows, cols) > 0.5
    return jnp.asarray(raw * mask)


@pytest.mark.parametrize('implementation', JITSMV_PARAMS)
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('event_dtype', [bool, float])
@pytest.mark.parametrize('prob', [0.1, 0.2])
def test_binary_jitsmv_forward_matches_reference(implementation, shape, transpose, corder, event_dtype, prob):
    seed = 123
    weight = jnp.asarray(1.5, dtype=jnp.float32)
    event_size = shape[0] if transpose else shape[1]
    vector = _sample_vector(event_size, event_dtype, seed + 7)
    vector_ref = _binary_events(vector, dtype=jnp.float32)

    y = binary_jitsmv(
        weight,
        prob,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=implementation,
    )
    y_ref = jitsmv(
        weight,
        prob,
        vector_ref,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=implementation,
    )
    assert allclose(y, y_ref, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((weight, vector, vector_ref, y, y_ref))


@pytest.mark.parametrize('implementation', JITSMM_PARAMS)
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('event_dtype', [bool, float])
@pytest.mark.parametrize('prob', [0.1, 0.2])
@pytest.mark.parametrize('k', [5, 10])
def test_binary_jitsmm_forward_matches_reference(implementation, shape, transpose, corder, event_dtype, prob, k):
    seed = 123
    weight = jnp.asarray(1.5, dtype=jnp.float32)
    rows = shape[0] if transpose else shape[1]
    matrix = _sample_matrix(rows, k, event_dtype, seed + 11)
    matrix_ref = _binary_events(matrix, dtype=jnp.float32)

    y = binary_jitsmm(
        weight,
        prob,
        matrix,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=implementation,
    )
    y_ref = jitsmm(
        weight,
        prob,
        matrix_ref,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=implementation,
    )
    assert allclose(y, y_ref, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((weight, matrix, matrix_ref, y, y_ref))


@pytest.mark.parametrize('implementation', JITSMV_PARAMS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
def test_binary_jitsmv_thresholds_float_events(implementation, shape, transpose, corder):
    seed = 123
    prob = 0.1
    weight = jnp.asarray(1.5, dtype=jnp.float32)
    size = shape[0] if transpose else shape[1]
    vector = _sample_vector(size, float, seed + 17)
    vector_binary = _binary_events(vector, dtype=jnp.float32)

    y_float = binary_jitsmv(
        weight,
        prob,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=implementation,
    )
    y_binary = binary_jitsmv(
        weight,
        prob,
        vector_binary,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=implementation,
    )
    assert allclose(y_float, y_binary, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((weight, vector, vector_binary, y_float, y_binary))


@pytest.mark.parametrize('implementation', JITSMM_PARAMS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('k', [10])
def test_binary_jitsmm_thresholds_float_events(implementation, shape, transpose, corder, k):
    seed = 123
    prob = 0.1
    weight = jnp.asarray(1.5, dtype=jnp.float32)
    rows = shape[0] if transpose else shape[1]
    matrix = _sample_matrix(rows, k, float, seed + 23)
    matrix_binary = _binary_events(matrix, dtype=jnp.float32)

    y_float = binary_jitsmm(
        weight,
        prob,
        matrix,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=implementation,
    )
    y_binary = binary_jitsmm(
        weight,
        prob,
        matrix_binary,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=implementation,
    )
    assert allclose(y_float, y_binary, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((weight, matrix, matrix_binary, y_float, y_binary))


@pytest.mark.parametrize('implementation', JITSMV_PARAMS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
def test_binary_jitsmv_jvp_and_vjp_match_reference(implementation, transpose, corder):
    shape = (20, 30)
    seed = 123
    prob = 0.1
    vector_size = shape[0] if transpose else shape[1]
    vector = _binary_events(_sample_vector(vector_size, float, seed + 29), dtype=jnp.float32)

    def f_binary(w, v):
        return binary_jitsmv(
            w,
            prob,
            v,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )

    def f_ref(w, v):
        return jitsmv(
            w,
            prob,
            v,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )

    primals = (jnp.asarray(1.5, dtype=jnp.float32), vector)
    tangents = (jnp.asarray(1.0, dtype=jnp.float32), jnp.ones_like(vector))

    out1, jvp1 = jax.jvp(f_binary, primals, tangents)
    out2, jvp2 = jax.jvp(f_ref, primals, tangents)
    assert allclose(out1, out2, rtol=1e-4, atol=1e-4)
    assert allclose(jvp1, jvp2, rtol=1e-4, atol=1e-4)

    g_w1, g_v1 = jax.grad(lambda w, v: f_binary(w, v).sum(), argnums=(0, 1))(*primals)
    g_w2, g_v2 = jax.grad(lambda w, v: f_ref(w, v).sum(), argnums=(0, 1))(*primals)
    assert allclose(g_w1, g_w2, rtol=1e-4, atol=1e-4)
    assert allclose(g_v1, g_v2, rtol=1e-4, atol=1e-4)
    jax.block_until_ready(
        (vector, primals[0], tangents[0], tangents[1], out1, jvp1, out2, jvp2, g_w1, g_v1, g_w2, g_v2))


@pytest.mark.parametrize('implementation', JITSMM_PARAMS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
def test_binary_jitsmm_jvp_matches_reference(implementation, transpose, corder):
    shape = (20, 30)
    seed = 123
    prob = 0.1
    k = 8
    rows = shape[0] if transpose else shape[1]
    matrix = _binary_events(_sample_matrix(rows, k, float, seed + 31), dtype=jnp.float32)

    def f_binary(w, B):
        return binary_jitsmm(
            w,
            prob,
            B,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )

    def f_ref(w, B):
        return jitsmm(
            w,
            prob,
            B,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )

    primals = (jnp.asarray(1.5, dtype=jnp.float32), matrix)
    tangents = (jnp.asarray(1.0, dtype=jnp.float32), jnp.ones_like(matrix))
    out1, jvp1 = jax.jvp(f_binary, primals, tangents)
    out2, jvp2 = jax.jvp(f_ref, primals, tangents)
    assert allclose(out1, out2, rtol=1e-4, atol=1e-4)
    assert allclose(jvp1, jvp2, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((matrix, primals[0], tangents[0], tangents[1], out1, jvp1, out2, jvp2))


@pytest.mark.parametrize('implementation', JITSMV_PARAMS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
def test_binary_jitsmv_vmap_matches_reference(implementation, transpose, corder):
    shape = (20, 30)
    seed = 123
    prob = 0.1
    batch = 6
    event_size = shape[0] if transpose else shape[1]
    vectors = _sample_matrix(batch, event_size, float, seed + 41)

    f_binary = jax.vmap(
        lambda v: binary_jitsmv(
            jnp.asarray(1.5, dtype=jnp.float32),
            prob,
            v,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )
    )
    f_ref = jax.vmap(
        lambda v: jitsmv(
            jnp.asarray(1.5, dtype=jnp.float32),
            prob,
            _binary_events(v, dtype=jnp.float32),
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )
    )
    result_binary = f_binary(vectors)
    result_ref = f_ref(vectors)
    assert allclose(result_binary, result_ref, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((vectors, result_binary, result_ref))


@pytest.mark.parametrize('implementation', JITSMM_PARAMS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
def test_binary_jitsmm_vmap_matches_reference(implementation, transpose, corder):
    shape = (20, 30)
    seed = 123
    prob = 0.1
    batch = 4
    k = 7
    rows = shape[0] if transpose else shape[1]
    matrices = _sample_matrix(batch * rows, k, float, seed + 43).reshape(batch, rows, k)

    f_binary = jax.vmap(
        lambda B: binary_jitsmm(
            jnp.asarray(1.5, dtype=jnp.float32),
            prob,
            B,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )
    )
    f_ref = jax.vmap(
        lambda B: jitsmm(
            jnp.asarray(1.5, dtype=jnp.float32),
            prob,
            _binary_events(B, dtype=jnp.float32),
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )
    )
    result_binary = f_binary(matrices)
    result_ref = f_ref(matrices)
    assert allclose(result_binary, result_ref, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((matrices, result_binary, result_ref))


# ---- One matrix: matmat is n independent matvecs, and vmap agrees with a loop ----
# Regression guards for the pre-unification defects: ``binary_jitsmm`` drew the
# AW-T4 matrix while ``binary_jitsmv`` drew the 32-lane one, so both
# ``binary_jitsmm(...)[:, c]`` and ``vmap(binary_jitsmv)`` silently disagreed
# with a plain loop over ``binary_jitsmv``.

@pytest.mark.parametrize("implementation", JITSMM_PARAMS)
@pytest.mark.parametrize('shape', [(20, 30), (33, 33)])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('corder', [True, False])
def test_binary_jitsmm_batch1_matches_binary_jitsmv(implementation, shape, transpose, corder):
    prob, seed = 0.2, 123
    k = shape[0] if transpose else shape[1]
    spikes = np.random.rand(k, 1) < 0.5
    mm = binary_jitsmm(1.5, prob, jnp.asarray(spikes), seed, shape=shape,
                    transpose=transpose, corder=corder, backend=implementation)
    mv = binary_jitsmv(1.5, prob, jnp.asarray(spikes[:, 0]), seed, shape=shape,
                    transpose=transpose, corder=corder, backend=implementation)
    assert allclose(mm[:, 0], mv, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((mm, mv))


@pytest.mark.parametrize("implementation", JITSMM_PARAMS)
@pytest.mark.parametrize('shape', [(20, 30)])
@pytest.mark.parametrize('n', [1, 4, 33])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('corder', [True, False])
def test_binary_jitsmm_columns_are_independent_mvs(implementation, shape, n, transpose, corder):
    prob, seed = 0.2, 123
    k = shape[0] if transpose else shape[1]
    spikes = np.random.rand(k, n) < 0.5
    mm = binary_jitsmm(1.5, prob, jnp.asarray(spikes), seed, shape=shape,
                    transpose=transpose, corder=corder, backend=implementation)
    for c in range(n):
        mv = binary_jitsmv(1.5, prob, jnp.asarray(spikes[:, c]), seed, shape=shape,
                        transpose=transpose, corder=corder, backend=implementation)
        assert allclose(mm[:, c], mv, rtol=1e-4, atol=1e-4)
    jax.block_until_ready(mm)


@pytest.mark.parametrize("implementation", JITSMV_PARAMS)
@pytest.mark.parametrize('shape', [(20, 30)])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('corder', [True, False])
def test_vmap_binary_jitsmv_matches_loop(implementation, shape, transpose, corder):
    prob, seed, batch = 0.2, 123, 5
    k = shape[0] if transpose else shape[1]
    spikes = np.random.rand(batch, k) < 0.5

    def f(v):
        return binary_jitsmv(1.5, prob, v, seed, shape=shape,
                          transpose=transpose, corder=corder, backend=implementation)

    batched = jax.vmap(f)(jnp.asarray(spikes))
    looped = jnp.stack([f(jnp.asarray(spikes[i])) for i in range(batch)])
    assert allclose(batched, looped, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((batched, looped))


# ---- Same check on the event-driven path ----
# ``binary_jitsmv``'s notrans/trans kernels must replay the same stream too.

@pytest.mark.parametrize("implementation", JITSMV_PARAMS)
@pytest.mark.parametrize('shape', [(12, 20), (17, 9)])
def test_binary_notrans_and_trans_kernels_draw_one_matrix(implementation, shape):
    a, b = shape
    prob, seed = 0.2, 123
    eye_b = np.eye(b, dtype=bool)
    eye_a = np.eye(a, dtype=bool)
    via_notrans = np.stack(
        [np.asarray(binary_jitsmv(1.5, prob, jnp.asarray(eye_b[j]), seed, shape=(a, b),
                              transpose=False, corder=True, backend=implementation))
         for j in range(b)], axis=1)
    via_trans = np.stack(
        [np.asarray(binary_jitsmv(1.5, prob, jnp.asarray(eye_a[i]), seed, shape=(b, a),
                              transpose=False, corder=False, backend=implementation))
         for i in range(a)], axis=1).T
    assert np.array_equal(via_notrans != 0, via_trans != 0)
    assert np.allclose(via_notrans, via_trans, rtol=1e-5, atol=1e-5)
