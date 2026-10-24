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
    # Activity contract: a numeric event is active iff it is non-zero.
    return jnp.asarray(jnp.asarray(x) != 0, dtype=dtype)


def _sample_vector(size: int, event_dtype, seed: int):
    rng = np.random.RandomState(seed)
    if event_dtype is bool:
        return jnp.asarray(rng.rand(size) > 0.5)
    # Signed values: negative events are active too.
    raw = rng.uniform(-1.0, 1.0, size).astype(np.float32)
    mask = rng.rand(size) > 0.5
    return jnp.asarray(raw * mask)


def _sample_matrix(rows: int, cols: int, event_dtype, seed: int):
    rng = np.random.RandomState(seed)
    if event_dtype is bool:
        return jnp.asarray(rng.rand(rows, cols) > 0.5)
    # Signed values: negative events are active too.
    raw = rng.uniform(-1.0, 1.0, (rows, cols)).astype(np.float32)
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


# ---------------------------------------------------------------------------
# Activity contract: a numeric event is active iff ``x != 0`` (negative values
# and NaN included), a boolean event iff ``x``; binary weight gradients use the
# same activity mask, never the raw event value (#198).
# ---------------------------------------------------------------------------

from contextlib import nullcontext as _nullcontext

import brainevent as _be
from brainevent._test_util import check_event_op_grads, jax_x64_enabled

_ACTIVITY_DTYPES = [
    jnp.bool_, jnp.int8, jnp.uint8, jnp.int16, jnp.int32,
    jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64,
]
_ACTIVITY_SHAPE = (20, 30)
_ACTIVITY_PROB = 0.3
_ACTIVITY_SEED = 123
_ACTIVITY_CLASSES = ['JITCScalarR', 'JITCScalarC']


def _activity_dtype_id(dtype):
    return np.dtype(dtype).name


def _activity_events(shape, dtype):
    """Events from ``[2, -1, 0, 0.5]`` (integers ``[2, -1, 0, 1]``) as ``dtype``.

    Floating dtypes also carry one NaN, which is active.
    """
    n = int(np.prod(shape))
    dtype = np.dtype(dtype)
    if dtype == np.bool_:
        base = np.asarray([True, True, False, True])
    elif np.issubdtype(dtype, np.integer):
        base = np.asarray([2, -1, 0, 1])
    else:
        base = np.asarray([2.0, -1.0, 0.0, 0.5])
    values = np.resize(base, n)
    values = values[np.random.RandomState(0).permutation(n)].reshape(shape).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        flat = values.reshape(-1)
        flat[np.flatnonzero(flat)[1]] = np.nan
    return values


def _activity_context(dtype):
    return jax_x64_enabled() if np.dtype(dtype) == np.float64 else _nullcontext()


def _activity_params():
    return jnp.asarray(1.5, dtype=jnp.float32)


def _activity_matrix(cls_name, params, corder, backend=None):
    cls = getattr(_be, cls_name)
    return cls(
        (params, _ACTIVITY_PROB, _ACTIVITY_SEED),
        shape=_ACTIVITY_SHAPE, corder=corder, backend=backend,
    )


def _activity_event_shape(op):
    m, n = _ACTIVITY_SHAPE
    return {'mv': (n,), 'vm': (m,), 'mm': (n, 3), 'mr': (3, m)}[op]


def _activity_apply(op, mat, events):
    return mat @ events if op in ('mv', 'mm') else events @ mat


def _check_activity_forward(implementation, cls_name, corder, dtype, op):
    host = _activity_events(_activity_event_shape(op), dtype)
    with _activity_context(dtype):
        params = _activity_params()
        dense = _activity_matrix(cls_name, params, corder).todense()
        mat = _activity_matrix(cls_name, params, corder, backend=implementation)
        got = np.asarray(_activity_apply(op, mat, _be.BinaryArray(jnp.asarray(host))))
        ref = np.asarray(_activity_apply(op, dense, jnp.asarray((host != 0).astype(np.float32))))
    assert got.shape == ref.shape
    assert np.allclose(got, ref, rtol=1e-4, atol=1e-4), (got, ref)


@pytest.mark.parametrize('implementation', JITSMV_PARAMS)
@pytest.mark.parametrize('cls_name', _ACTIVITY_CLASSES)
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('op', ['mv', 'vm'])
@pytest.mark.parametrize('dtype', _ACTIVITY_DTYPES, ids=_activity_dtype_id)
def test_binary_jitsmv_activity_all_dtypes_match_dense(implementation, cls_name, corder, op, dtype):
    _check_activity_forward(implementation, cls_name, corder, dtype, op)


@pytest.mark.parametrize('implementation', JITSMM_PARAMS)
@pytest.mark.parametrize('cls_name', _ACTIVITY_CLASSES)
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('op', ['mm', 'mr'])
@pytest.mark.parametrize('dtype', _ACTIVITY_DTYPES, ids=_activity_dtype_id)
def test_binary_jitsmm_activity_all_dtypes_match_dense(implementation, cls_name, corder, op, dtype):
    _check_activity_forward(implementation, cls_name, corder, dtype, op)


def _check_activity_grads(implementation, cls_name, corder, op):
    n = int(np.prod(_activity_event_shape(op)))
    values = np.resize(np.asarray([2.0, -1.0, 0.0], dtype=np.float32), n)
    x = jnp.asarray(values[np.random.RandomState(1).permutation(n)].reshape(_activity_event_shape(op)))

    def op_fn(params, v):
        mat = _activity_matrix(cls_name, params, corder, backend=implementation)
        return _activity_apply(op, mat, _be.BinaryArray(v))

    def dense_fn(params, z):
        return _activity_apply(op, _activity_matrix(cls_name, params, corder).todense(), z)

    check_event_op_grads(op_fn, dense_fn, _activity_params(), x, mode='binary', rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('implementation', JITSMV_PARAMS)
@pytest.mark.parametrize('cls_name', _ACTIVITY_CLASSES)
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('op', ['mv', 'vm'])
def test_binary_jitsmv_grads_use_activity_mask(implementation, cls_name, corder, op):
    _check_activity_grads(implementation, cls_name, corder, op)


@pytest.mark.parametrize('implementation', JITSMM_PARAMS)
@pytest.mark.parametrize('cls_name', _ACTIVITY_CLASSES)
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('op', ['mm', 'mr'])
def test_binary_jitsmm_grads_use_activity_mask(implementation, cls_name, corder, op):
    _check_activity_grads(implementation, cls_name, corder, op)


@pytest.mark.parametrize('implementation', JITSMV_PARAMS)
def test_issue_198_jitc_scalar_weight_grad_regression(implementation):
    """Issue #198 repro: grad must be 3.0 (mask), not 4.0 (raw event values)."""
    x = jnp.array([0., 2., 1.])

    def loss(w):
        mat = _be.JITCScalarR((w, 0.5, 123), shape=(3, 2), backend=implementation)
        return (_be.BinaryArray(x) @ mat).sum()

    w = jnp.array(1.5, dtype=jnp.float32)
    assert np.allclose(loss(w), 4.5)
    assert np.allclose(jax.grad(loss)(w), 3.0)
    assert np.allclose(jax.jvp(loss, (w,), (jnp.ones_like(w),))[1], 3.0)


@pytest.mark.parametrize('implementation', JITSMM_PARAMS)
def test_issue_198_jitc_scalar_mm_and_negative_weight_grad_regression(implementation):
    """Issue #198 mm repro (doubled batch) plus a negative event, which is active."""
    x = jnp.array([[0., 2., 1.], [0., 2., 1.], [-1., 0., 0.]])
    dense_mask = (np.asarray(x) != 0).astype(np.float32)

    def loss(w):
        mat = _be.JITCScalarR((w, 0.5, 123), shape=(3, 2), backend=implementation)
        return (_be.BinaryArray(x) @ mat).sum()

    def ref(w):
        return (jnp.asarray(dense_mask) @ _be.JITCScalarR((w, 0.5, 123), shape=(3, 2)).todense()).sum()

    w = jnp.array(1.5, dtype=jnp.float32)
    assert np.allclose(loss(w), ref(w))
    assert np.allclose(jax.grad(loss)(w), jax.grad(ref)(w))
    # Rows 0 and 1 contribute 3.0 each (the issue's mm number), the -1 row adds
    # its own (active) connections.
    assert np.allclose(jax.grad(ref)(w), 6.0 + float(np.asarray(_be.JITCScalarR((1.0, 0.5, 123), shape=(3, 2)).todense())[0].sum()))


@pytest.mark.skipif(
    platform != 'gpu' or 'cuda_raw' not in JITSMM_IMPLEMENTATIONS,
    reason='requires the cuda_raw binary_jitsmm backend on GPU',
)
@pytest.mark.parametrize('op', ['float', 'binary'])
def test_cuda_grid_guard_raises_python_exception_instead_of_aborting(op):
    """Host-side CUDA grid guards (``n > 65535`` grid dims) must raise, not ``abort()``.

    Runs in a subprocess: before the fix the float guard called ``abort()`` and
    killed the interpreter with SIGABRT (negative return code). The binary
    path trips the (new) bit-pack guard first.
    """
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        f"""
        import jax, jax.numpy as jnp
        from brainevent._jit_scalar.binary import binary_jitsmm
        from brainevent._jit_scalar.float import jitsmm
        op = {op!r}
        try:
            if op == 'float':
                B = jnp.zeros((8, 65536), dtype=jnp.float32)
                out = jitsmm(1.5, 0.1, B, 123, shape=(4, 8), transpose=False,
                             corder=True, backend='cuda_raw')
            else:
                B = jnp.zeros((8, 65536), dtype=jnp.bool_)
                out = binary_jitsmm(1.5, 0.1, B, 123, shape=(4, 8), transpose=False,
                                    corder=True, backend='cuda_raw')
            jax.block_until_ready(out)
        except Exception as e:
            print('RAISED', 'grid overflow' in str(e))
        else:
            print('NO_ERROR')
        """
    )
    proc = subprocess.run(
        [sys.executable, '-c', script], capture_output=True, text=True,
        timeout=900, env=os.environ.copy(),
    )
    assert proc.returncode >= 0, f'process killed by signal {-proc.returncode}:\n{proc.stderr[-2000:]}'
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert 'RAISED True' in proc.stdout, (proc.stdout, proc.stderr[-2000:])
