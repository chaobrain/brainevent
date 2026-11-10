# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Automatic primary geometry and parent-owned lazy mirror preparation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent import BinaryArray, BitPackedBinary, CompactBinary, CSR, TCSR, WeightedArray
from brainevent._tcsr import main as tcsr_main


def _matrix():
    return TCSR(CSR(
        jnp.array([1., 2., 3.], jnp.float32),
        jnp.array([0, 2, 1], jnp.int32), jnp.array([0, 2, 3], jnp.int32),
        shape=(2, 3), backend='jax_raw',
    ))


def test_construction_prepares_all_primary_batches_without_building_mirror():
    matrix = _matrix()
    assert matrix._cache.tile_geometries
    assert all(key[0] for key in matrix._cache.tile_geometries)
    assert matrix._cache.mirror is None
    for batch in (0, 1, 187, 188, 189, 65535):
        for dtype in (jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64):
            assert matrix._get_binary_mm_plan(batch, transpose=True, dtype=dtype) is not None
    assert matrix._cache.mirror is None


def test_mirror_hook_runs_after_parent_and_prepares_once_for_all_batches(monkeypatch):
    matrix = _matrix()
    real = tcsr_main.build_tile_geometries
    calls = []

    def prepare(indices, indptr, **kwargs):
        assert matrix._cache.mirror is not None
        calls.append(kwargs['transpose'])
        return real(indices, indptr, **kwargs)

    monkeypatch.setattr(tcsr_main, 'build_tile_geometries', prepare)
    mirror = matrix._mirror()
    assert mirror is matrix._cache.mirror
    assert calls == [False]
    original = dict(matrix._cache.tile_geometries)
    for batch in (0, 1, 187, 188, 189, 65535):
        for dtype in (jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64):
            assert matrix._get_binary_mm_plan(batch, transpose=False, dtype=dtype) is not None
        assert matrix._mirror() is mirror
    assert calls == [False]
    assert all(matrix._cache.tile_geometries[k] is geometry for k, geometry in original.items())


def test_existing_parent_mirror_without_tiles_gets_lazy_tiles():
    matrix = _matrix()
    mirror = CSR._mirror(matrix)
    assert mirror is not None
    assert not any(not key[0] for key in matrix._cache.tile_geometries)
    assert matrix._mirror() is mirror
    assert any(not key[0] for key in matrix._cache.tile_geometries)


def test_shared_values_reuse_first_mirror_geometry_initialization(monkeypatch):
    matrix = _matrix()
    sibling = matrix.with_data(matrix.data * 2)
    matrix._mirror()

    def forbidden(*args, **kwargs):
        raise AssertionError('same-structure siblings must share mirror preparation')

    monkeypatch.setattr(tcsr_main, 'build_tile_geometries', forbidden)
    assert sibling._mirror() is matrix._cache.mirror
    assert sibling._cache is matrix._cache
    assert sibling._get_binary_mm_plan(65535, transpose=False) is matrix._get_binary_mm_plan(65535, transpose=False)


def test_traced_matrix_without_mirror_uses_parent_without_host_preparation(monkeypatch):
    matrix = _matrix()
    expected = np.asarray(matrix.todense()) @ np.ones((3, 2), bool)

    def forbidden(*args, **kwargs):
        raise AssertionError('traced missing mirror must not prepare on the host')

    monkeypatch.setattr(tcsr_main, 'build_tile_geometries', forbidden)
    monkeypatch.setattr(jax, 'device_get', forbidden)
    result = jax.jit(lambda m, x: m @ BinaryArray(x))(matrix, jnp.ones((3, 2), bool))
    np.testing.assert_array_equal(result, expected)
    assert matrix._cache.mirror is None


def test_concrete_structure_closure_can_prepare_first_mirror_while_tracing():
    matrix = _matrix()
    def run(x):
        matrix._mirror()
        return matrix @ BinaryArray(x)

    result = jax.jit(run)(jnp.ones((3, 2), bool))
    np.testing.assert_array_equal(result, [[3., 3.], [3., 3.]])
    assert matrix._cache.mirror is not None
    assert matrix._get_binary_mm_plan(2, transpose=False) is not None
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


def test_wide_primary_has_geometry_for_distinct_batch_and_accumulator_widths():
    matrix = TCSR(CSR(
        jnp.array([1., 2., 3.], jnp.float32),
        jnp.array([0, 100, 24575], jnp.int32), jnp.array([0, 2, 3], jnp.int32),
        shape=(2, 24576), backend='jax_raw',
    ))
    from brainevent._tcsr.binary.combined.planner import tile_schedule
    geometries = dict(matrix._cache.tile_geometries)
    for batch in (0, 1, 187, 188, 189, 65535):
        for dtype, accumulator_bytes in ((jnp.float32, 4), (jnp.float64, 8)):
            plan = matrix._get_binary_mm_plan(batch, transpose=True, dtype=dtype)
            assert plan.tile_width == tile_schedule(24576, batch, accumulator_bytes).tile_width
            assert matrix._cache.tile_geometries[plan.key] is plan
    assert len(geometries) > 2
    assert matrix._cache.mirror is None
    assert matrix._cache.tile_geometries.keys() == geometries.keys()


def test_failed_mirror_preparation_does_not_publish_partial_geometries(monkeypatch):
    matrix = _matrix()
    primary = dict(matrix._cache.tile_geometries)
    real = tcsr_main.build_tile_geometries

    def fail(*args, **kwargs):
        assert kwargs['transpose'] is False
        raise RuntimeError('injected mirror preparation failure')

    monkeypatch.setattr(tcsr_main, 'build_tile_geometries', fail)
    with pytest.raises(RuntimeError, match='injected'):
        matrix._mirror()
    assert matrix._cache.mirror is not None
    assert matrix._cache.tile_geometries.keys() == primary.keys()
    assert all(matrix._cache.tile_geometries[key] is geometry for key, geometry in primary.items())
    monkeypatch.setattr(tcsr_main, 'build_tile_geometries', real)
    matrix._mirror()
    assert matrix._get_binary_mm_plan(2, transpose=False) is not None


def test_orphaned_mirror_geometry_errors_before_parent_can_replace_it():
    matrix = _matrix().build_weight_indices()
    matrix._cache.mirror = None
    with pytest.raises(ValueError, match='matching mirror'):
        matrix @ BinaryArray(jnp.ones((3, 2), bool))
