# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Complete automatic tile geometry, independent of batch and weight dtype."""

from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._test_util import jax_x64_enabled
from brainevent._tcsr.binary.combined.planner import INT32_MAX, tile_schedule
from brainevent._tcsr.preprocess import TileGeometry, build_tile_geometries, geometry_widths


def _geometries(**updates):
    arguments = dict(indices=jnp.array([0, 2, 2, 1], jnp.int32),
                     indptr=jnp.array([0, 3, 4], jnp.int32),
                     shape=(2, 3), transpose=True)
    arguments.update(updates)
    return build_tile_geometries(**arguments)


@pytest.mark.parametrize('target', [0, 1, 32, 1000, 24576, 1000000])
def test_finite_width_set_covers_every_batch_and_accumulator(target):
    widths = geometry_widths(target)
    assert widths == tuple(sorted(set(widths)))
    for size in (4, 8):
        assert {tile_schedule(target, batch, size).tile_width for batch in range(65536)} <= set(widths)


@pytest.mark.parametrize('target', [True, np.bool_(False), 1.5, -1, INT32_MAX + 1])
def test_invalid_width_dimensions_are_rejected(target):
    with pytest.raises(ValueError, match='target'):
        geometry_widths(target)


def test_width_cache_does_not_hide_invalid_argument_types():
    geometry_widths(np.int32(1))
    with pytest.raises(ValueError, match='target'):
        geometry_widths(True)


@pytest.mark.parametrize('transpose', [True, False])
def test_geometry_table_preserves_duplicate_slots_and_all_boundaries(transpose):
    shape = (2, 3) if transpose else (3, 2)
    geometries = _geometries(shape=shape, transpose=transpose)
    assert len(geometries) == len(geometry_widths(3))
    for key, geometry in geometries.items():
        assert key == (transpose, geometry.tile_width, np.dtype('int32'))
        assert geometry.shape == shape and geometry.nnz == 4
        assert geometry.offset_dtype == np.dtype('int32')
        assert geometry.local_targets.dtype == jnp.uint16
        np.testing.assert_array_equal(geometry.local_targets, np.array([0, 2, 2, 1]) % geometry.tile_width)
        tiles = (3 + geometry.tile_width - 1) // geometry.tile_width
        offsets = np.asarray(geometry.tile_offsets).reshape((2, tiles + 1))
        bounds = np.arange(tiles + 1) * geometry.tile_width
        np.testing.assert_array_equal(offsets[0], np.searchsorted([0, 2, 2], bounds))
        np.testing.assert_array_equal(offsets[1], 3 + np.searchsorted([1], bounds))
        with pytest.raises(FrozenInstanceError):
            geometry.tile_width = 1


def test_batches_and_value_dtypes_select_shared_geometry_objects():
    geometries = _geometries()
    for batch in (0, 1, 2, 187, 188, 189, 65535):
        for size in (4, 8):
            width = tile_schedule(3, batch, size).tile_width
            assert geometries[(True, width, np.dtype('int32'))].tile_width == width
    same_width = tile_schedule(3, 188, 4).tile_width
    assert same_width == tile_schedule(3, 65535, 8).tile_width
    assert len(geometries) < 189 * 2


@pytest.mark.parametrize('shape', [(0, 0), (2, 0), (0, 2), (2, 3)])
@pytest.mark.parametrize('transpose', [True, False])
def test_empty_geometry_table(shape, transpose):
    source = shape[0] if transpose else shape[1]
    geometries = _geometries(indices=jnp.empty((0,), jnp.int32), indptr=jnp.zeros(source + 1, jnp.int32),
                             shape=shape, transpose=transpose)
    for geometry in geometries.values():
        assert geometry.local_targets.shape == (0,)
        assert np.all(np.asarray(geometry.tile_offsets) == 0)


def test_int64_offsets_are_preserved_without_weight_dtype_binding():
    with jax_x64_enabled():
        geometries = _geometries(indptr=jnp.array([0, 3, 4], jnp.int64))
        for geometry in geometries.values():
            assert geometry.offset_dtype == np.dtype('int64')
            assert geometry.tile_offsets.dtype == jnp.int64
            assert not hasattr(geometry, 'batch') and not hasattr(geometry, 'value_dtype')


def test_metadata_restore_keeps_none_float0_and_other_placeholders():
    geometry = next(iter(_geometries().values()))
    restored = TileGeometry.from_metadata(geometry.metadata, None, object())
    assert restored.local_targets is None and restored.metadata == geometry.metadata
    differential = np.zeros((4,), dtype=jax.dtypes.float0)
    restored = TileGeometry.from_metadata(geometry.metadata, differential, differential)
    assert restored.local_targets is differential and restored.offset_dtype == np.dtype('int32')


def test_tiles_are_placed_with_committed_structure():
    cpu = jax.devices('cpu')[0]
    idx = jax.device_put(np.array([0, 2, 2, 1], np.int32), cpu)
    ptr = jax.device_put(np.array([0, 3, 4], np.int32), cpu)
    geometries = _geometries(indices=idx, indptr=ptr)
    for geometry in geometries.values():
        assert geometry.local_targets.devices() == {cpu}
        assert geometry.tile_offsets.devices() == {cpu}
        assert geometry.local_targets.committed


@pytest.mark.parametrize('updates, error, message', [
    ({'transpose': 1}, ValueError, 'transpose'),
    ({'shape': (2,)}, ValueError, 'shape'),
    ({'shape': (True, 3)}, ValueError, 'shape'),
    ({'shape': (2.5, 3)}, ValueError, 'shape'),
    ({'shape': (-1, 3)}, ValueError, 'shape'),
    ({'shape': (2, INT32_MAX + 1)}, ValueError, 'shape'),
    ({'shape': (1, 3)}, ValueError, 'indptr'),
    ({'indices': np.array([0, 2, 2, 1], np.int64)}, TypeError, 'indices'),
    ({'indices': np.array([[0, 2, 2, 1]], np.int32)}, TypeError, 'indices'),
    ({'indptr': np.array([0., 3., 4.])}, TypeError, 'indptr'),
    ({'indptr': np.array([[0, 3, 4]], np.int32)}, TypeError, 'indptr'),
    ({'indptr': np.array([1, 3, 4], np.int32)}, ValueError, 'pointers'),
    ({'indptr': np.array([0, 4, 3], np.int32)}, ValueError, 'pointers'),
    ({'indices': np.array([-1, 2, 2, 1], np.int32)}, ValueError, 'range'),
    ({'indices': np.array([0, 2, 3, 1], np.int32)}, ValueError, 'range'),
    ({'indices': np.array([2, 0, 2, 1], np.int32)}, ValueError, 'sorted'),
])
def test_invalid_geometry_inputs(updates, error, message):
    with pytest.raises(error, match=message):
        _geometries(**updates)


def test_geometry_preparation_rejects_jit_but_supports_explicit_compile_time_eval():
    with pytest.raises(ValueError, match='outside JIT'):
        jax.jit(lambda x: (_geometries(), x)[1])(jnp.array(1.))
    arrays = (jnp.array([0, 2, 2, 1], jnp.int32), jnp.array([0, 3, 4], jnp.int32))

    def initialize(x):
        with jax.ensure_compile_time_eval():
            geometries = _geometries(indices=arrays[0], indptr=arrays[1])
        assert all(not isinstance(g.local_targets, jax.core.Tracer) for g in geometries.values())
        return x

    np.testing.assert_array_equal(jax.jit(initialize)(jnp.array(2.)), 2.)


def test_compile_time_context_never_allows_traced_structure():
    pointers = jnp.array([0, 3, 4], jnp.int32)

    def prepare(indices):
        with jax.ensure_compile_time_eval():
            _geometries(indices=indices, indptr=pointers)
        return indices

    with pytest.raises(ValueError, match='outside JIT'):
        jax.jit(prepare)(jnp.array([0, 2, 2, 1], jnp.int32))


def test_explicit_int64_offsets_cannot_be_silently_narrowed(monkeypatch):
    from brainevent._tcsr.preprocess import preprocess

    original = preprocess.jax.device_put

    def narrow(array, placement=None):
        if array.dtype == np.dtype('int64'):
            array = array.astype(np.int32)
        return original(array, placement)

    with jax_x64_enabled():
        monkeypatch.setattr(preprocess.jax, 'device_put', narrow)
        with pytest.raises(ValueError, match='preserve int64'):
            _geometries(indptr=np.array([0, 3, 4], np.int64))
