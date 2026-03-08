# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._coo.binary import binary_coomv, binary_coomm
from brainevent._coo.float import coomv, coomm, coomv_p, coomm_p

if jax.default_backend() != 'gpu':
    pytest.skip('Pallas GPU COO tests require a GPU backend.', allow_module_level=True)
if 'pallas' not in coomv_p.available_backends('gpu'):
    pytest.skip('Pallas backend is unavailable for coomv on GPU.', allow_module_level=True)
if 'pallas' not in coomm_p.available_backends('gpu'):
    pytest.skip('Pallas backend is unavailable for coomm on GPU.', allow_module_level=True)


def _make_indices(shape, nnz, mode, seed):
    m, n = shape
    rng = np.random.default_rng(seed)
    if nnz == 0:
        return jnp.empty((0,), dtype=jnp.int32), jnp.empty((0,), dtype=jnp.int32)

    if mode == 'uniform':
        row = rng.integers(0, m, size=nnz, dtype=np.int32)
        col = rng.integers(0, n, size=nnz, dtype=np.int32)
    elif mode == 'skewed_row':
        # Heavily concentrated rows to stress atomic contention and duplicate accumulation.
        hot = max(1, m // 16)
        hot_rows = rng.integers(0, hot, size=nnz, dtype=np.int32)
        cold_rows = rng.integers(0, m, size=nnz, dtype=np.int32)
        choose_hot = rng.random(nnz) < 0.85
        row = np.where(choose_hot, hot_rows, cold_rows).astype(np.int32)
        col = rng.integers(0, n, size=nnz, dtype=np.int32)
    else:
        raise ValueError(f'Unsupported mode: {mode}')

    return jnp.asarray(row), jnp.asarray(col)


def _expand_weights(weights, nnz):
    return jnp.full((nnz,), weights[0], dtype=weights.dtype) if weights.shape[0] == 1 else weights


def _dense_from_coo(weights, row, col, shape):
    dense = jnp.zeros(shape, dtype=weights.dtype)
    if row.shape[0] == 0:
        return dense
    return dense.at[(row, col)].add(_expand_weights(weights, row.shape[0]))


def _tol(dtype):
    if dtype == jnp.float32:
        return (1e-4, 1e-4)
    if dtype == jnp.float16:
        return (7e-2, 7e-2)
    return (3e-1, 3e-1)


def _tol_mm(dtype):
    # COO-MM accumulates many more atomic contributions than COO-MV.
    if dtype == jnp.float32:
        return (2e-2, 2e-2)
    return (8e-2, 8e-2)


@pytest.mark.parametrize('shape,nnz', [((1, 1), 0), ((17, 29), 129), ((257, 193), 2048)])
@pytest.mark.parametrize('mode', ['uniform', 'skewed_row'])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('homo_w', [False, True])
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float16, jnp.bfloat16])
def test_pallas_coomv_matches_dense(shape, nnz, mode, transpose, homo_w, dtype):
    row, col = _make_indices(shape, nnz, mode, seed=11)
    rng = np.random.default_rng(12)

    if homo_w:
        weights = jnp.asarray([rng.standard_normal()], dtype=dtype)
    else:
        weights = jnp.asarray(rng.standard_normal(nnz), dtype=dtype)

    v_len = shape[0] if transpose else shape[1]
    v = jnp.asarray(rng.standard_normal(v_len), dtype=dtype)

    out = coomv(weights, row, col, v, shape=shape, transpose=transpose, backend='pallas')

    dense = _dense_from_coo(weights, row, col, shape)
    ref = dense.T @ v if transpose else dense @ v

    rtol, atol = _tol(dtype)
    assert jnp.allclose(out, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize('shape,nnz', [((1, 1), 0), ((17, 29), 129), ((257, 193), 2048)])
@pytest.mark.parametrize('mode', ['uniform', 'skewed_row'])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('homo_w', [False, True])
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float16])
def test_pallas_coomm_matches_dense(shape, nnz, mode, transpose, homo_w, dtype):
    row, col = _make_indices(shape, nnz, mode, seed=21)
    rng = np.random.default_rng(22)

    if homo_w:
        weights = jnp.asarray([rng.standard_normal()], dtype=dtype)
    else:
        weights = jnp.asarray(rng.standard_normal(nnz), dtype=dtype)

    n_rhs = 32
    b_rows = shape[0] if transpose else shape[1]
    B = jnp.asarray(rng.standard_normal((b_rows, n_rhs)), dtype=dtype)

    out = coomm(weights, row, col, B, shape=shape, transpose=transpose, backend='pallas')

    dense = _dense_from_coo(weights, row, col, shape)
    ref = dense.T @ B if transpose else dense @ B

    rtol, atol = _tol_mm(dtype)
    assert jnp.allclose(out, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize('transpose', [False, True])
def test_pallas_binary_coo_matches_dense(transpose):
    shape = (33, 41)
    nnz = 511
    row, col = _make_indices(shape, nnz, mode='skewed_row', seed=31)
    rng = np.random.default_rng(32)
    weights = jnp.asarray(rng.standard_normal(nnz), dtype=jnp.float32)

    v_len = shape[0] if transpose else shape[1]
    bool_v = jnp.asarray(rng.random(v_len) > 0.7)
    float_v = jnp.asarray(rng.standard_normal(v_len), dtype=jnp.float32)

    dense = _dense_from_coo(weights, row, col, shape)

    out_bool = binary_coomv(weights, row, col, bool_v, shape=shape, transpose=transpose, backend='pallas')
    ref_bool = dense.T @ bool_v.astype(weights.dtype) if transpose else dense @ bool_v.astype(weights.dtype)
    assert jnp.allclose(out_bool, ref_bool, rtol=1e-5, atol=1e-5)

    out_float = binary_coomv(weights, row, col, float_v, shape=shape, transpose=transpose, backend='pallas')
    act = (float_v > 0).astype(weights.dtype)
    ref_float = dense.T @ act if transpose else dense @ act
    assert jnp.allclose(out_float, ref_float, rtol=1e-5, atol=1e-5)

    n_rhs = 24
    b_rows = shape[0] if transpose else shape[1]
    bool_B = jnp.asarray(rng.random((b_rows, n_rhs)) > 0.75)
    float_B = jnp.asarray(rng.standard_normal((b_rows, n_rhs)), dtype=jnp.float32)

    out_bool_mm = binary_coomm(weights, row, col, bool_B, shape=shape, transpose=transpose, backend='pallas')
    ref_bool_mm = dense.T @ bool_B.astype(weights.dtype) if transpose else dense @ bool_B.astype(weights.dtype)
    assert jnp.allclose(out_bool_mm, ref_bool_mm, rtol=5e-3, atol=5e-3)

    out_float_mm = binary_coomm(weights, row, col, float_B, shape=shape, transpose=transpose, backend='pallas')
    act_B = (float_B > 0).astype(weights.dtype)
    ref_float_mm = dense.T @ act_B if transpose else dense @ act_B
    assert jnp.allclose(out_float_mm, ref_float_mm, rtol=5e-3, atol=5e-3)


def test_pallas_empty_coo_outputs_zero():
    shape = (19, 23)
    row = jnp.empty((0,), dtype=jnp.int32)
    col = jnp.empty((0,), dtype=jnp.int32)
    weights = jnp.empty((0,), dtype=jnp.float32)

    v = jnp.ones((shape[1],), dtype=jnp.float32)
    B = jnp.ones((shape[1], 7), dtype=jnp.float32)

    out_mv = coomv(weights, row, col, v, shape=shape, transpose=False, backend='pallas')
    out_mm = coomm(weights, row, col, B, shape=shape, transpose=False, backend='pallas')

    assert out_mv.shape == (shape[0],)
    assert out_mm.shape == (shape[0], 7)
    assert jnp.all(out_mv == 0)
    assert jnp.all(out_mm == 0)


def test_pallas_determinism_with_atomics_tolerance():
    shape = (257, 193)
    nnz = 8192
    row, col = _make_indices(shape, nnz, mode='skewed_row', seed=41)
    rng = np.random.default_rng(42)

    weights = jnp.asarray(rng.standard_normal(nnz), dtype=jnp.float32)
    v = jnp.asarray(rng.standard_normal(shape[1]), dtype=jnp.float32)

    f = jax.jit(lambda w, r, c, x: coomv(w, r, c, x, shape=shape, transpose=False, backend='pallas'))
    y0 = f(weights, row, col, v)
    y1 = f(weights, row, col, v)
    y2 = f(weights, row, col, v)

    assert jnp.allclose(y0, y1, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(y1, y2, rtol=1e-4, atol=1e-4)


def test_pallas_input_validation_errors():
    weights = jnp.asarray([1.0], dtype=jnp.float32)

    with pytest.raises(ValueError, match='same length'):
        coomv(weights, jnp.array([0, 1], dtype=jnp.int32), jnp.array([0], dtype=jnp.int32),
              jnp.ones((2,), dtype=jnp.float32),
              shape=(2, 2), transpose=False, backend='pallas')

    with pytest.raises(ValueError, match='length must be 1 or nnz'):
        coomv(jnp.array([1.0, 2.0], dtype=jnp.float32), jnp.array([0], dtype=jnp.int32),
              jnp.array([1], dtype=jnp.int32),
              jnp.ones((2,), dtype=jnp.float32), shape=(2, 2), transpose=False, backend='pallas')

    with pytest.raises(ValueError, match='incompatible shape'):
        coomm(weights, jnp.array([0], dtype=jnp.int32), jnp.array([1], dtype=jnp.int32),
              jnp.ones((3, 2), dtype=jnp.float32),
              shape=(2, 2), transpose=False, backend='pallas')


@pytest.mark.skipif(not jax.config.read('jax_enable_x64'), reason='Requires JAX x64 enabled for true int64 indices.')
def test_pallas_int64_indices_parity():
    shape = (41, 37)
    nnz = 733
    row, col = _make_indices(shape, nnz, mode='uniform', seed=51)
    row = row.astype(jnp.int64)
    col = col.astype(jnp.int64)

    rng = np.random.default_rng(52)
    weights = jnp.asarray(rng.standard_normal(nnz), dtype=jnp.float32)
    v = jnp.asarray(rng.standard_normal(shape[1]), dtype=jnp.float32)
    B = jnp.asarray(rng.standard_normal((shape[1], 8)), dtype=jnp.float32)

    out_mv = coomv(weights, row, col, v, shape=shape, transpose=False, backend='pallas')
    out_mm = coomm(weights, row, col, B, shape=shape, transpose=False, backend='pallas')

    dense = _dense_from_coo(weights, row, col, shape)
    ref_mv = dense @ v
    ref_mm = dense @ B

    assert jnp.allclose(out_mv, ref_mv, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(out_mm, ref_mm, rtol=1e-4, atol=1e-4)
