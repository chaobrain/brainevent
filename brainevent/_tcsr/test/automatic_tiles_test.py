# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Lazy per-width tile geometry, mirror preparation and backend gating."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent import BinaryArray, BitPackedBinary, CompactBinary, CSR, TCSR, WeightedArray
from brainevent._csr import main as csr_main
from brainevent._error import PerformanceWarning
from brainevent._tcsr import main as tcsr_main
from brainevent._tcsr.binary.combined.planner import tile_schedule


@pytest.fixture(autouse=True)
def _fresh_warning_registry():
    tcsr_main._UNPREPARED_WARNED.clear()
    yield
    tcsr_main._UNPREPARED_WARNED.clear()


@pytest.fixture
def cuda_reachable(monkeypatch):
    """Let CPU tests exercise the CUDA-only preparation path; dispatch still lowers for CPU."""
    monkeypatch.setattr(tcsr_main, '_cuda_available', lambda: True)
    monkeypatch.setattr(tcsr_main, '_structure_device', lambda indices: None)


def _matrix(backend='jax_raw'):
    return TCSR(CSR(
        jnp.array([1., 2., 3.], jnp.float32),
        jnp.array([0, 2, 1], jnp.int32), jnp.array([0, 2, 3], jnp.int32),
        shape=(2, 3), backend=backend,
    ))


def _forbid(monkeypatch, *targets):
    def forbidden(*args, **kwargs):
        raise AssertionError('unexpected host preparation')
    for owner, name in targets:
        monkeypatch.setattr(owner, name, forbidden)


def test_construction_prepares_no_tiles_and_no_mirror(monkeypatch):
    _forbid(monkeypatch, (tcsr_main, 'build_tile_geometries'))
    matrix = _matrix()
    assert matrix._cache.tile_geometries == {}
    assert matrix._cache.mirror is None
    # The config holds no structure state, so it adds no pytree leaves.
    assert jax.tree.leaves(matrix.config) == []
    assert len(jax.tree.leaves(matrix)) == 3
    for batch in (1, 188, 65535):
        assert matrix._get_tile_geometry(batch, transpose=True) is None


@pytest.mark.parametrize('side', ['left', 'right'])
def test_parent_backends_never_build_mirror_or_tiles(monkeypatch, side):
    """A product on a non-CUDA backend must not pay for the mirror or tile memory."""
    matrix = _matrix('jax_raw')
    _forbid(monkeypatch, (tcsr_main, 'build_tile_geometries'))
    events = jnp.array([[True, False, True], [False, True, True]])
    if side == 'left':
        result = BinaryArray(events.T[:2]) @ matrix
        expected = events.T[:2] @ matrix.todense()
    else:
        result = matrix @ BinaryArray(events.T)
        expected = matrix.todense() @ events.T
    np.testing.assert_allclose(result, expected)
    assert matrix._cache.mirror is None
    assert matrix._cache.tile_geometries == {}


@pytest.mark.parametrize('side', ['left', 'right'])
def test_default_backend_without_cuda_builds_nothing(monkeypatch, side):
    matrix = _matrix(None)
    # Both TCSR tiles and the inherited CSR mirror route ask whether CUDA exists.
    monkeypatch.setattr(tcsr_main, '_cuda_available', lambda: False)
    monkeypatch.setattr(csr_main, '_cuda_available', lambda: False)
    _forbid(monkeypatch, (tcsr_main, 'build_tile_geometries'))
    if side == 'left':
        BinaryArray(jnp.ones((2, 2), bool)) @ matrix
    else:
        matrix @ BinaryArray(jnp.ones((3, 2), bool))
    assert matrix._cache.mirror is None and matrix._cache.tile_geometries == {}


def test_cpu_committed_structure_builds_nothing_even_with_cuda(monkeypatch):
    matrix = jax.device_put(_matrix(None), jax.devices('cpu')[0])
    monkeypatch.setattr(tcsr_main, '_cuda_available', lambda: True)
    _forbid(monkeypatch, (tcsr_main, 'build_tile_geometries'))
    assert next(iter(matrix.indices.devices())).platform == 'cpu'
    BinaryArray(jax.device_put(jnp.ones((2, 2), bool), jax.devices('cpu')[0])) @ matrix
    assert matrix._cache.tile_geometries == {}


def test_eager_product_prepares_exactly_the_width_it_runs(cuda_reachable):
    matrix = _matrix(None)
    events = jnp.array([[True, False], [True, True]])
    np.testing.assert_allclose(BinaryArray(events) @ matrix, events @ matrix.todense())
    width = tile_schedule(3, 2).tile_width
    assert list(matrix._cache.tile_geometries) == [(True, width, np.dtype('int32'))]
    assert matrix._cache.mirror is None
    first = matrix._get_tile_geometry(2, transpose=True)
    BinaryArray(events) @ matrix
    assert matrix._get_tile_geometry(2, transpose=True) is first
    assert len(matrix._cache.tile_geometries) == 1


def test_right_product_prepares_mirror_and_one_mirror_width(cuda_reachable):
    matrix = _matrix(None)
    events = jnp.array([[True, False], [False, True], [True, True]])
    np.testing.assert_allclose(matrix @ BinaryArray(events), matrix.todense() @ events)
    assert matrix._cache.mirror is not None
    keys = list(matrix._cache.tile_geometries)
    assert keys == [(False, tile_schedule(2, 2).tile_width, np.dtype('int32'))]


def test_lazy_prepare_covers_every_batch_and_accumulator_width():
    matrix = TCSR(CSR(
        jnp.array([1., 2., 3.], jnp.float32),
        jnp.array([0, 100, 24575], jnp.int32), jnp.array([0, 2, 3], jnp.int32),
        shape=(2, 24576), backend='jax_raw',
    ))
    widths = set()
    for batch in (1, 187, 188, 189, 65535):
        for dtype, accumulator_bytes in ((jnp.float32, 4), (jnp.float64, 8)):
            plan = matrix._get_tile_geometry(batch, transpose=True, dtype=dtype, prepare=True)
            assert plan.tile_width == tile_schedule(24576, batch, accumulator_bytes).tile_width
            assert matrix._cache.tile_geometries[plan.key] is plan
            widths.add(plan.tile_width)
    # Only the widths that ran exist; same-width batches share one geometry.
    assert {key[1] for key in matrix._cache.tile_geometries} == widths
    assert matrix._get_tile_geometry(0, transpose=True, prepare=True) is None
    assert matrix._cache.mirror is None


def test_shared_values_reuse_lazily_prepared_geometry(monkeypatch):
    matrix = _matrix()
    sibling = matrix.with_data(matrix.data * 2)
    plan = matrix._get_tile_geometry(2, transpose=False, prepare=True)
    _forbid(monkeypatch, (tcsr_main, 'build_tile_geometries'))
    assert sibling._cache is matrix._cache
    assert sibling._get_tile_geometry(2, transpose=False, prepare=True) is plan


def test_traced_unprepared_matrix_warns_and_uses_parent(monkeypatch, cuda_reachable):
    matrix = _matrix(None)
    expected = np.asarray(matrix.todense()) @ np.ones((3, 2), bool)
    _forbid(monkeypatch, (tcsr_main, 'build_tile_geometries'), (jax, 'device_get'))
    run = jax.jit(lambda m, x: m @ BinaryArray(x))
    with pytest.warns(PerformanceWarning, match='prepare_binary_mm'):
        result = run(matrix, jnp.ones((3, 2), bool))
    np.testing.assert_array_equal(result, expected)
    assert matrix._cache.mirror is None
    # The warning is emitted once per shape, batch and direction.
    jax.clear_caches()
    with warnings.catch_warnings():
        warnings.simplefilter('error', PerformanceWarning)
        run(matrix, jnp.ones((3, 2), bool))


def test_prepared_matrix_runs_under_jit_without_host_work(monkeypatch, cuda_reachable):
    matrix = _matrix(None).prepare_binary_mm(2)
    _forbid(monkeypatch, (tcsr_main, 'build_tile_geometries'), (jax, 'device_get'))
    left, right = jnp.ones((2, 2), bool), jnp.ones((3, 2), bool)
    with warnings.catch_warnings():
        warnings.simplefilter('error', PerformanceWarning)
        actual = jax.jit(lambda m, l, r: (BinaryArray(l) @ m, m @ BinaryArray(r)))(matrix, left, right)
    dense = np.asarray(matrix.todense())
    np.testing.assert_allclose(actual[0], left @ dense)
    np.testing.assert_allclose(actual[1], dense @ right)


def test_concrete_closure_prepares_while_tracing(cuda_reachable):
    matrix = _matrix(None)
    result = jax.jit(lambda x: matrix @ BinaryArray(x))(jnp.ones((3, 2), bool))
    np.testing.assert_array_equal(result, [[3., 3.], [3., 3.]])
    assert matrix._cache.mirror is not None
    assert matrix._get_tile_geometry(2, transpose=False) is not None
    assert all(not isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(matrix))


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('kind', ['numeric', 'weighted', 'vector', 'half_events', 'too_many_batches'])
def test_non_tcsr_operands_reach_correct_parent_operator_unchanged(monkeypatch, side, kind):
    matrix = _matrix()
    source = matrix.shape[0 if side == 'left' else 1]
    batch = 65536 if kind == 'too_many_batches' else 2
    shape = (batch, source) if side == 'left' else (source, batch)
    values = jnp.ones(shape, jnp.float16 if kind == 'half_events' else jnp.float32)
    if kind == 'numeric':
        operand = values
    elif kind == 'weighted':
        operand = WeightedArray(values)
    elif kind == 'vector':
        operand = BinaryArray(jnp.ones(source, bool))
    else:
        operand = BinaryArray(values)
    method = '__rmatmul__' if side == 'left' else '__matmul__'
    calls = []
    sentinel = object()

    def parent(self, other):
        assert self is matrix
        calls.append(other)
        return sentinel

    monkeypatch.setattr(CSR, method, parent)
    assert getattr(matrix, method)(operand) is sentinel
    assert len(calls) == 1 and calls[0] is operand


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('wrapper', [BinaryArray, BitPackedBinary, CompactBinary.from_array])
def test_all_csr_binary_wrappers_keep_value_semantics_without_prepare(side, wrapper):
    matrix = _matrix()
    values = (jnp.array([[True, False], [True, True]]) if side == 'left' else
              jnp.array([[True, False], [False, True], [True, True]]))
    operand = wrapper(values)
    result = operand @ matrix if side == 'left' else matrix @ operand
    expected = values @ matrix.todense() if side == 'left' else matrix.todense() @ values
    np.testing.assert_array_equal(result, expected)


def test_stored_event_product_is_parent_owned():
    assert TCSR._stored_event_product is CSR._stored_event_product


def test_failed_tile_preparation_does_not_publish_partial_geometries(monkeypatch):
    matrix = _matrix()
    primary = matrix._get_tile_geometry(2, transpose=True, prepare=True)
    real = tcsr_main.build_tile_geometries

    def fail(*args, **kwargs):
        assert kwargs['transpose'] is False
        raise RuntimeError('injected mirror preparation failure')

    monkeypatch.setattr(tcsr_main, 'build_tile_geometries', fail)
    with pytest.raises(RuntimeError, match='injected'):
        matrix._get_tile_geometry(2, transpose=False, prepare=True)
    # The mirror was built before the failure and stays cached.
    assert matrix._cache.mirror is not None
    assert list(matrix._cache.tile_geometries.values()) == [primary]
    monkeypatch.setattr(tcsr_main, 'build_tile_geometries', real)
    assert matrix._get_tile_geometry(2, transpose=False, prepare=True) is not None


def test_orphaned_mirror_geometry_errors_before_parent_can_replace_it():
    matrix = _matrix().prepare_binary_mm(2, transpose=False)
    matrix._cache.mirror = None
    with pytest.raises(ValueError, match='matching mirror'):
        matrix._get_tile_geometry(2, transpose=False, prepare=True)
