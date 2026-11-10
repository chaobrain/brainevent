# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Preparation invariants, including empty structures and traced calls."""

from dataclasses import FrozenInstanceError

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._csr.main import CSR
from brainevent._test_util import jax_x64_enabled
from brainevent._tcsr.binary.combined.planner import tile_schedule
from brainevent._tcsr.preprocess import (
    TilePlan, build_binary_mm_plan, ensure_preparation_context, sort_csr, validate_preparation,
)


def _plan(**overrides):
    args = dict(indices=jnp.array([0, 2, 2, 1], jnp.int32),
                indptr=jnp.array([0, 3, 4], jnp.int32),
                shape=(2, 3), transpose=True, batch=2, value_dtype=jnp.float32)
    args.update(overrides)
    return build_binary_mm_plan(**args)


def test_stable_sort_preserves_source_duplicates_and_units():
    source = CSR(jnp.array([10., 20., 30., 40., 50.]) * u.mV,
                 jnp.array([2, 0, 2, 1, 0]), jnp.array([0, 3, 5]), shape=(2, 3))
    data, indices = sort_csr(source)
    np.testing.assert_array_equal(indices, [0, 2, 2, 0, 1])
    np.testing.assert_array_equal(u.get_mantissa(data), [20., 10., 30., 50., 40.])
    assert u.get_unit(data) == u.mV
    np.testing.assert_array_equal(source.indices, [2, 0, 2, 1, 0])
    np.testing.assert_array_equal(u.get_mantissa(source.data), [10., 20., 30., 40., 50.])


@pytest.mark.parametrize('data', [jnp.array(2.), jnp.array([2.])])
def test_shared_weight_is_not_expanded(data):
    source = CSR(data, jnp.array([1, 0]), jnp.array([0, 2]), shape=(1, 2))
    sorted_data, indices = sort_csr(source)
    assert sorted_data is source.data
    np.testing.assert_array_equal(indices, [0, 1])


def test_sort_empty_structure():
    source = CSR(jnp.empty(0), jnp.empty(0, jnp.int32), jnp.zeros(3, jnp.int32), shape=(2, 0))
    data, indices = sort_csr(source)
    assert data.shape == indices.shape == (0,)


@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('batch', [0, 1, 2, 65535])
@pytest.mark.parametrize('dtype', [jnp.float16, jnp.bfloat16, jnp.float32])
def test_plan_matches_adaptive_geometry(transpose, batch, dtype):
    shape = (2, 3) if transpose else (3, 2)
    plan = _plan(shape=shape, transpose=transpose, batch=batch, value_dtype=dtype)
    schedule = tile_schedule(3, batch, 4)
    assert plan.key == (transpose, batch, np.dtype(dtype))
    assert plan.shape == shape
    assert plan.nnz == 4
    assert plan.tile_width == schedule.tile_width
    assert plan.local_targets.dtype == jnp.uint16
    assert plan.tile_offsets.dtype == jnp.int32
    np.testing.assert_array_equal(plan.local_targets, np.array([0, 2, 2, 1]) % plan.tile_width)
    offsets = np.asarray(plan.tile_offsets).reshape(2, schedule.tile_count + 1)
    np.testing.assert_array_equal(offsets[:, 0], [0, 3])
    np.testing.assert_array_equal(offsets[:, -1], [3, 4])


@pytest.mark.parametrize('shape', [(0, 0), (2, 0), (0, 2)])
@pytest.mark.parametrize('transpose', [True, False])
def test_plan_zero_dimensions(shape, transpose):
    source = shape[0] if transpose else shape[1]
    plan = _plan(indices=jnp.empty(0, jnp.int32), indptr=jnp.zeros(source + 1, jnp.int32),
                 shape=shape, transpose=transpose)
    assert plan.local_targets.shape == (0,)
    assert np.all(np.asarray(plan.tile_offsets) == 0)


def test_float64_and_int64_offsets_preserved():
    with jax_x64_enabled():
        plan = _plan(indptr=jnp.array([0, 3, 4], jnp.int64), value_dtype=jnp.float64)
        assert plan.tile_offsets.dtype == jnp.int64
        assert plan.offset_dtype == np.dtype('int64')
        assert plan.tile_width == tile_schedule(3, 2, 8).tile_width


def test_metadata_reconstruction_accepts_placeholder_leaves():
    plan = _plan()
    restored = TilePlan.from_metadata(plan.metadata, None, object())
    assert restored.metadata == plan.metadata
    assert restored.local_targets is None
    with pytest.raises(FrozenInstanceError):
        plan.batch = 3


@pytest.mark.parametrize('overrides, message', [
    ({'batch': True}, 'batch'),
    ({'batch': 1.5}, 'batch'),
    ({'batch': -1}, 'dimensions'),
    ({'batch': 65536}, 'dimensions'),
    ({'transpose': 1}, 'transpose'),
    ({'shape': (True, 3)}, 'shape'),
    ({'shape': (2.5, 3)}, 'shape'),
    ({'shape': (1, 3)}, 'indptr'),
    ({'value_dtype': jnp.int32}, 'values'),
    ({'indices': jnp.array([2, 0, 2, 1])}, 'sorted'),
    ({'indices': jnp.array([0, 2, 3, 1])}, 'range'),
    ({'indices': jnp.array([0., 2., 2., 1.])}, 'int32'),
])
def test_invalid_plan_inputs(overrides, message):
    with pytest.raises((ValueError, TypeError), match=message):
        _plan(**overrides)


def test_prepare_rejects_jit_even_for_closed_concrete_structure():
    with pytest.raises(ValueError, match='outside JIT'):
        jax.jit(lambda x: (_plan(), x)[1])(jnp.array(1.))


def test_sort_rejects_jit():
    source = CSR(jnp.array(1.), jnp.array([1, 0]), jnp.array([0, 2]), shape=(1, 2))
    with pytest.raises(ValueError, match='outside JIT'):
        jax.jit(lambda x: (sort_csr(source), x)[1])(jnp.array(1.))


def test_preparation_rejects_traced_structure():
    with pytest.raises(ValueError, match='outside JIT'):
        jax.jit(ensure_preparation_context)(jnp.array([1], jnp.int32))


def test_validate_static_arguments():
    assert validate_preparation(np.int64(3), jnp.float32) == 3
    with pytest.raises(ValueError, match='f16'):
        validate_preparation(3, jnp.int32)


def test_int64_offsets_must_not_be_narrowed(monkeypatch):
    from brainevent._tcsr.preprocess import preprocess

    original_asarray = preprocess.jnp.asarray

    def narrow_offsets(array, *args, **kwargs):
        if getattr(array, 'dtype', None) == np.dtype('int64'):
            return original_asarray(array, dtype=jnp.int32)
        return original_asarray(array, *args, **kwargs)

    with jax_x64_enabled():
        ptr = jnp.array([0, 3, 4], jnp.int64)
        monkeypatch.setattr(preprocess.jnp, 'asarray', narrow_offsets)
        with pytest.raises(ValueError, match='preserve int64'):
            _plan(indptr=ptr)
