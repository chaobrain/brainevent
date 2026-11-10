# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""CUDA integration through TCSR's public matrix operations and pytree API."""

from contextlib import nullcontext
import re

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent import BinaryArray, CSR, TCSR
from brainevent._test_util import jax_x64_enabled


def _cuda_available():
    try:
        return any('cuda' in d.client.platform_version.lower() for d in jax.devices('gpu'))
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(not _cuda_available(), reason='actual CUDA device required')


def _matrix(
    dtype=jnp.float32, mode='hetero', algorithm='weighted_float', *, batch=3,
):
    values = {'hetero': [1., -2., .5, 3., 4.], 'singleton': [1.5], 'scalar': 1.5}[mode]
    return TCSR(CSR(jnp.asarray(values, dtype), np.array([2, 0, 2, 1, 2], np.int32),
                    np.array([0, 3, 5], np.int32), shape=(2, 3),
                    backward_algorithm=algorithm))


def _events(matrix, transpose, batch=3):
    source = matrix.shape[0 if transpose else 1]
    shape = (batch, source) if transpose else (source, batch)
    return jnp.resize(jnp.array([0., -2., .5, 1., 0., 3.], jnp.float32), shape)


def _product(matrix, events, transpose):
    return BinaryArray(events) @ matrix if transpose else matrix @ BinaryArray(events)


def _assert_native_combined(run, *args):
    # The unoptimized module traces both resource branches. Only optimized
    # HLO proves that a real combined custom call survives resource selection.
    optimized = run.lower(*args).compile().as_text()
    assert re.search(r'custom_call_target="[^"]*tcsr_combined_', optimized), optimized


def _dense(matrix):
    rows = np.repeat(np.arange(matrix.shape[0]), np.diff(np.asarray(matrix.indptr)))
    cols = np.asarray(matrix.indices)
    weights = np.broadcast_to(np.asarray(u.get_mantissa(matrix.data)), (cols.size,))
    result = np.zeros(matrix.shape, np.float64)
    np.add.at(result, (rows, cols), weights)
    return result


_CASES = [
    (jnp.float16, True, 'hetero', 'binary'),
    (jnp.float16, False, 'scalar', 'weighted_float'),
    (jnp.bfloat16, True, 'singleton', 'weighted_float'),
    (jnp.bfloat16, False, 'hetero', 'binary'),
    (jnp.float32, True, 'scalar', 'binary'),
    (jnp.float32, False, 'singleton', 'weighted_float'),
    (jnp.float32, False, 'hetero', 'weighted_float'),
    (jnp.float64, True, 'hetero', 'weighted_float'),
    (jnp.float64, False, 'scalar', 'binary'),
]


@pytest.mark.parametrize('dtype,transpose,mode,algorithm', _CASES)
def test_public_cuda_forward_and_matrix_and_event_cotangents(dtype, transpose, mode, algorithm):
    with jax_x64_enabled() if dtype == jnp.float64 else nullcontext():
        matrix = _matrix(dtype, mode, algorithm)
        events = _events(matrix, transpose)
        run = jax.jit(lambda m, x: _product(m, x, transpose))
        # The first public mirror call prepares its geometry; subsequent JIT
        # calls carry it as leaves rather than reading traced structure.
        if not transpose:
            _product(matrix, events, transpose)
            assert matrix._cache.mirror is not None
            assert matrix._get_binary_mm_plan(events.shape[1], transpose=False) is not None
        result = run(matrix, events)
        dense = _dense(matrix)
        expected = (np.asarray(events) != 0) @ dense if transpose else dense @ (np.asarray(events) != 0)
        tolerance = .03 if dtype in (jnp.float16, jnp.bfloat16) else 1e-5
        np.testing.assert_allclose(np.asarray(result, dtype=float), expected, atol=tolerance)
        assert result.dtype == dtype
        _assert_native_combined(run, matrix, events)
        grad_matrix, grad_events = jax.jit(jax.grad(
            lambda m, x: _product(m, x, transpose).sum(), argnums=(0, 1), allow_int=True,
        ))(matrix, events)
        rows = np.repeat(np.arange(matrix.shape[0]), np.diff(np.asarray(matrix.indptr)))
        cols = np.asarray(matrix.indices)
        activity = (
            np.asarray(events) if algorithm == 'weighted_float'
            else np.asarray(events) != 0
        )
        expected_weights = activity[:, rows].sum(axis=0) if transpose else activity[cols].sum(axis=1)
        if mode != 'hetero':
            expected_weights = np.asarray(expected_weights.sum()).reshape(matrix.data.shape)
        expected_events = np.broadcast_to(dense.sum(axis=1), events.shape) if transpose else np.broadcast_to(dense.sum(axis=0)[:, None], events.shape)
        np.testing.assert_allclose(np.asarray(grad_matrix.data, dtype=float), expected_weights, atol=tolerance)
        np.testing.assert_allclose(np.asarray(grad_events), expected_events, atol=tolerance)
        assert grad_events.dtype == events.dtype
        assert grad_matrix.indices.dtype == jax.dtypes.float0
        assert grad_matrix._cache.tile_geometries


@pytest.mark.parametrize('transpose', [True, False])
def test_public_cuda_units(transpose):
    matrix = _matrix().apply(lambda w: w * u.mV)
    events = _events(matrix, transpose) * u.ms
    if not transpose:
        _product(matrix, events, transpose)
    result = jax.jit(lambda m, x: _product(m, x, transpose))(matrix, events)
    expected = (np.asarray(u.get_mantissa(events)) != 0) @ _dense(matrix) if transpose else _dense(matrix) @ (np.asarray(u.get_mantissa(events)) != 0)
    assert u.get_unit(result) == u.mV * u.ms
    np.testing.assert_allclose(u.get_mantissa(result), expected)


@pytest.mark.parametrize('shape,batch', [((0, 3), 3), ((3, 0), 3), ((0, 0), 3), ((3, 5), 3), ((2, 3), 0)])
@pytest.mark.parametrize('transpose', [True, False])
def test_public_cuda_empty_geometry(shape, batch, transpose):
    matrix = TCSR(CSR(jnp.empty((0,), jnp.float32), np.empty((0,), np.int32),
                      np.zeros(shape[0] + 1, np.int32), shape=shape))
    events = _events(matrix, transpose, batch)
    result = jax.jit(lambda m, x: _product(m, x, transpose))(matrix, events)
    expected_shape = (batch, shape[1]) if transpose else (shape[0], batch)
    assert result.shape == expected_shape
    np.testing.assert_array_equal(result, np.zeros(expected_shape, np.float32))


@pytest.mark.parametrize('transpose', [True, False])
def test_public_cuda_tail_tiles_and_batched_transform(transpose):
    rows = np.array([0, 0, 128, 128, 256, 256])
    indices = np.array([258, 0, 129, 128, 1, 258], np.int32)
    indptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=257))].astype(np.int32)
    matrix = TCSR(CSR(jnp.arange(1, 7, dtype=jnp.float32), indices, indptr,
                      shape=(257, 259)))
    events = _events(matrix, transpose)
    if not transpose:
        _product(matrix, events, transpose)
    plan = matrix._get_binary_mm_plan(3, transpose=transpose)
    assert matrix.shape[1 if transpose else 0] % plan.tile_width
    batched = jnp.stack([events, -events])
    result = jax.jit(jax.vmap(lambda x: _product(matrix, x, transpose)))(batched)
    expected = (np.asarray(events) != 0) @ _dense(matrix) if transpose else _dense(matrix) @ (np.asarray(events) != 0)
    np.testing.assert_allclose(result, np.stack([expected, expected]))


def test_public_batch_beyond_combined_limit_uses_parent(monkeypatch):
    matrix = _matrix()
    events = BinaryArray(jnp.zeros((65536, 2), jnp.bool_))
    calls = []

    def inherited(self, other):
        calls.append(other)
        return 'parent'

    monkeypatch.setattr(CSR, '__rmatmul__', inherited)
    assert events @ matrix == 'parent'
    assert len(calls) == 1 and calls[0] is events


def test_public_left_then_right_uses_automatic_and_lazy_geometries(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('normal calls cannot require explicit preparation')

    monkeypatch.setattr(TCSR, 'prepare_binary_mm', forbidden)
    matrix = _matrix()
    primary = dict(matrix._cache.tile_geometries)
    assert primary and all(key[0] for key in primary)
    assert matrix._cache.mirror is None
    left = _events(matrix, True)
    direct = jax.jit(lambda m, x: _product(m, x, True))
    np.testing.assert_allclose(direct(matrix, left), (np.asarray(left) != 0) @ _dense(matrix))
    _assert_native_combined(direct, matrix, left)
    assert matrix._cache.mirror is None
    right = _events(matrix, False)
    np.testing.assert_allclose(_product(matrix, right, False), _dense(matrix) @ (np.asarray(right) != 0))
    assert matrix._cache.mirror is not None
    assert any(not key[0] for key in matrix._cache.tile_geometries)
    assert all(matrix._cache.tile_geometries[key] is geometry for key, geometry in primary.items())
    mirror = jax.jit(lambda m, x: _product(m, x, False))
    np.testing.assert_allclose(mirror(matrix, right), _dense(matrix) @ (np.asarray(right) != 0))
    _assert_native_combined(mirror, matrix, right)


@pytest.mark.parametrize('transpose', [True, False])
def test_public_same_instance_supports_batch_boundaries_without_replanning(transpose):
    matrix = _matrix()
    if not transpose:
        _product(matrix, _events(matrix, False), False)
    geometries = dict(matrix._cache.tile_geometries)
    for batch in (0, 1, 187, 188, 189, 65535):
        events = _events(matrix, transpose, batch)
        result = _product(matrix, events, transpose)
        expected = ((np.asarray(events) != 0) @ _dense(matrix) if transpose else
                    _dense(matrix) @ (np.asarray(events) != 0))
        np.testing.assert_allclose(result, expected)
        assert matrix._cache.tile_geometries.keys() == geometries.keys()
        assert all(matrix._cache.tile_geometries[key] is geometry for key, geometry in geometries.items())
