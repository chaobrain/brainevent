# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""CUDA integration through TCSR's public matrix operations and pytree API."""

from contextlib import nullcontext
import re
import warnings

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent import BinaryArray, CSR, TCSR
from brainevent._error import PerformanceWarning
from brainevent._tcsr import main as tcsr_main
from brainevent._tcsr.binary.combined.planner import tile_schedule
from brainevent._test_util import jax_x64_enabled


def _cuda_available():
    try:
        return any('cuda' in d.client.platform_version.lower() for d in jax.devices('gpu'))
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(not _cuda_available(), reason='actual CUDA device required')


def _matrix(dtype=jnp.float32, mode='hetero', *, batch=3):
    values = {'hetero': [1., -2., .5, 3., 4.], 'singleton': [1.5], 'scalar': 1.5}[mode]
    return TCSR(CSR(jnp.asarray(values, dtype), np.array([2, 0, 2, 1, 2], np.int32),
                    np.array([0, 3, 5], np.int32), shape=(2, 3)))


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
    (jnp.float16, True, 'hetero'),
    (jnp.float16, False, 'scalar'),
    (jnp.bfloat16, True, 'singleton'),
    (jnp.bfloat16, False, 'hetero'),
    (jnp.float32, True, 'scalar'),
    (jnp.float32, False, 'singleton'),
    (jnp.float32, False, 'hetero'),
    (jnp.float64, True, 'hetero'),
    (jnp.float64, False, 'scalar'),
]


@pytest.mark.parametrize('dtype,transpose,mode', _CASES)
def test_public_cuda_forward_and_matrix_and_event_cotangents(dtype, transpose, mode):
    with jax_x64_enabled() if dtype == jnp.float64 else nullcontext():
        matrix = _matrix(dtype, mode)
        events = _events(matrix, transpose)
        run = jax.jit(lambda m, x: _product(m, x, transpose))
        # The first eager call prepares its tiles; subsequent JIT calls carry
        # them as leaves rather than reading traced structure.
        _product(matrix, events, transpose)
        assert (matrix._cache.mirror is not None) == (not transpose)
        assert matrix._get_tile_geometry(events.shape[1], transpose=transpose) is not None
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
        # Binary events: the weight gradient is the exact activity-mask derivative.
        activity = np.asarray(events) != 0
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
    _product(matrix, events, transpose)
    plan = matrix._get_tile_geometry(3, transpose=transpose)
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


def test_public_left_then_right_prepare_tiles_lazily(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('normal eager calls cannot require explicit preparation')

    monkeypatch.setattr(TCSR, 'prepare_binary_mm', forbidden)
    matrix = _matrix()
    assert matrix._cache.tile_geometries == {} and matrix._cache.mirror is None
    left = _events(matrix, True)
    np.testing.assert_allclose(_product(matrix, left, True), (np.asarray(left) != 0) @ _dense(matrix))
    primary = dict(matrix._cache.tile_geometries)
    assert list(primary) == [(True, tile_schedule(3, 3).tile_width, np.dtype('int32'))]
    assert matrix._cache.mirror is None
    direct = jax.jit(lambda m, x: _product(m, x, True))
    np.testing.assert_allclose(direct(matrix, left), (np.asarray(left) != 0) @ _dense(matrix))
    _assert_native_combined(direct, matrix, left)
    right = _events(matrix, False)
    np.testing.assert_allclose(_product(matrix, right, False), _dense(matrix) @ (np.asarray(right) != 0))
    assert matrix._cache.mirror is not None
    assert sorted(key[0] for key in matrix._cache.tile_geometries) == [False, True]
    assert all(matrix._cache.tile_geometries[key] is geometry for key, geometry in primary.items())
    mirror = jax.jit(lambda m, x: _product(m, x, False))
    np.testing.assert_allclose(mirror(matrix, right), _dense(matrix) @ (np.asarray(right) != 0))
    _assert_native_combined(mirror, matrix, right)


@pytest.mark.parametrize('transpose', [True, False])
def test_public_batches_add_only_the_widths_they_run(transpose):
    matrix = TCSR(CSR(jnp.arange(1, 6, dtype=jnp.float32), np.array([0, 999, 5, 500, 998], np.int32),
                      np.array([0, 2, 5], np.int32), shape=(2, 1000)))
    target = matrix.shape[1] if transpose else matrix.shape[0]
    for batch in (0, 1, 2, 187, 188, 189, 1000):
        events = _events(matrix, transpose, batch)
        result = _product(matrix, events, transpose)
        expected = ((np.asarray(events) != 0) @ _dense(matrix) if transpose else
                    _dense(matrix) @ (np.asarray(events) != 0))
        np.testing.assert_allclose(result, expected)
    widths = {tile_schedule(target, batch).tile_width for batch in (1, 2, 187, 188, 189, 1000)}
    assert {key[1] for key in matrix._cache.tile_geometries if key[0] == transpose} == widths
    assert all(key[0] == transpose for key in matrix._cache.tile_geometries)


@pytest.mark.parametrize('transpose', [True, False])
def test_traced_unprepared_matrix_warns_and_prepared_matrix_runs_tiles(transpose):
    tcsr_main._UNPREPARED_WARNED.clear()
    matrix = _matrix()
    events = _events(matrix, transpose)
    run = jax.jit(lambda m, x: _product(m, x, transpose))
    with pytest.warns(PerformanceWarning, match='prepare_binary_mm'):
        text = run.lower(matrix, events).compile().as_text()
    assert not re.search(r'custom_call_target="[^"]*tcsr_combined_', text)
    prepared = matrix.prepare_binary_mm(3, transpose=transpose)
    with warnings.catch_warnings():
        warnings.simplefilter('error', PerformanceWarning)
        _assert_native_combined(run, prepared, events)
        result = run(prepared, events)
    expected = (np.asarray(events) != 0) @ _dense(matrix) if transpose else _dense(matrix) @ (np.asarray(events) != 0)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize('cols,dtype,threads', [
    (100000, jnp.float32, 256),
    (400003, jnp.float32, 512),
    (2000003, jnp.float32, 1024),
    (1200001, jnp.float64, 512),
])
@pytest.mark.parametrize('mode', ['direct', 'homo', 'mirror'])
def test_public_adaptive_scatter_tiers(cols, dtype, threads, mode):
    """Every block-size tier, tile boundaries and the last column, via public products."""
    batch = 2
    schedule = tile_schedule(cols, batch, 8 if dtype == jnp.float64 else 4)
    assert schedule.block_size == threads
    width = schedule.tile_width
    ptr = np.array([0, 3, 5, 5, 8], np.int64)
    indices = np.array([cols - 1, 0, width - 1, width, width, cols - 1, width + 1, 0], np.int32)
    values = np.array([0.5] if mode == 'homo' else [1, 2, -0.5, 0.25, 1.5, -1, 2, 0.5])
    events = np.array([[1, -2, 3, 0.5], [0, 0, 1, 0]], np.float32)  # (batch, 4)
    with jax_x64_enabled() if dtype == jnp.float64 else nullcontext():
        wide = CSR(jnp.asarray(values, dtype), jnp.asarray(indices), jnp.asarray(ptr.astype(np.int32)),
                   shape=(4, cols))
        dense = np.zeros((4, cols))
        np.add.at(dense, (np.repeat(np.arange(4), np.diff(ptr)), indices),
                  np.broadcast_to(values, indices.shape))
        if mode == 'mirror':
            # W @ x on the (cols, 4) transpose traverses this structure as its mirror.
            matrix = TCSR(wide.T.tocsr())
            run = jax.jit(lambda m, x: m @ BinaryArray(x))
            operand, expected = jnp.asarray(events.T), (dense.T @ (events.T != 0))
        else:
            matrix = TCSR(wide)
            run = jax.jit(lambda m, x: BinaryArray(x) @ m)
            operand, expected = jnp.asarray(events), ((events != 0) @ dense)
        matrix = matrix.prepare_binary_mm(batch, transpose=mode != 'mirror')
        _assert_native_combined(run, matrix, operand)
        np.testing.assert_array_equal(np.asarray(run(matrix, operand), np.float64), expected)
        # Reuse output/scratch on another step with no active events.
        np.testing.assert_array_equal(np.asarray(run(matrix, jnp.zeros_like(operand))), np.zeros_like(expected))
