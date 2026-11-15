# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._error import PerformanceWarning
from brainevent._structure import StructureCache, is_concrete, warn_mirror_fallback


def test_is_concrete_flags_tracers_only():
    assert is_concrete(jnp.arange(3), np.arange(3), 1.0)
    seen = []
    jax.jit(lambda x: seen.append(is_concrete(jnp.arange(3), x)) or x)(jnp.ones(2))
    assert seen == [False]


def test_structure_cache_set_mirror_does_not_read_array_values():
    class NoHostRead:
        def __array__(self, *args, **kwargs):
            raise AssertionError('storing a mirror must not scan its values')

    cache = StructureCache()
    mirror = cache.set_mirror([NoHostRead(), NoHostRead(), NoHostRead()])
    assert isinstance(mirror, tuple) and cache.mirror is mirror
    assert cache.diag_plan is None


def test_copies_share_structure_until_the_mirror_is_replaced():
    mirror = (object(), object(), object())
    diag_plan = (object(), object())
    cache = StructureCache(mirror=mirror, diag_plan=diag_plan)
    copied = cache.copy()
    assert copied.mirror is cache.mirror and copied.diag_plan is cache.diag_plan
    replacement = copied.set_mirror([object(), object(), object()])
    assert copied.mirror is replacement and replacement is not mirror
    # Replacing the copy's mirror neither touches the source nor the diag plan.
    assert cache.mirror is not replacement and cache.mirror == mirror
    assert copied.diag_plan is cache.diag_plan


def test_structure_cache_keeps_masks_separate_from_mm_launch_plans():
    assert set(StructureCache.__slots__) == {
        'mirror', 'diag_plan', 'tile_geometries', 'primary_mm_mask', 'mirror_mm_mask',
    }
    with pytest.raises(TypeError):
        StructureCache(mirror_preparation=(object(), 7))


def test_structure_cache_copies_masks_and_rebuilds_only_the_replaced_mirror():
    primary = jnp.array([7], jnp.uint32)
    lengths = np.zeros(64, np.int32)
    lengths[0] = 65
    mirror = (jnp.asarray(np.r_[0, lengths.cumsum()], jnp.int32),
              jnp.zeros(65, jnp.int32), jnp.arange(65, dtype=jnp.int32))
    cache = StructureCache(primary_mm_mask=primary)
    cache.set_mirror(mirror)
    np.testing.assert_array_equal(cache.mirror_mm_mask, [1, 0])
    copied = cache.copy()
    assert copied.primary_mm_mask is primary
    assert copied.mirror_mm_mask is cache.mirror_mm_mask

    lengths[0], lengths[33] = 0, 65
    copied.set_mirror((jnp.asarray(np.r_[0, lengths.cumsum()], jnp.int32), *mirror[1:]))
    np.testing.assert_array_equal(copied.mirror_mm_mask, [0, 2])
    np.testing.assert_array_equal(cache.mirror_mm_mask, [1, 0])
    assert copied.primary_mm_mask is primary


def test_warn_mirror_fallback_warns_once_per_owner_and_operation():
    owner = 'StructureTestOwner'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        warn_mirror_fallback(owner, 'op-a')
        warn_mirror_fallback(owner, 'op-a')
        warn_mirror_fallback(owner, 'op-b')
    perf = [w for w in caught if issubclass(w.category, PerformanceWarning)]
    assert len(perf) == 2
    assert 'build_weight_indices' in str(perf[0].message)


def test_tile_geometry_containers_are_copied_and_mirror_replacement_is_selective():
    primary, mirror = object(), object()
    entries = {(True, 32, np.dtype('int32')): primary,
               (False, 32, np.dtype('int32')): mirror}
    cache = StructureCache(tile_geometries=entries)
    assert cache.tile_geometries is not entries
    copied = cache.copy()
    assert copied is not cache
    assert copied.tile_geometries is not cache.tile_geometries
    assert copied.tile_geometries[(True, 32, np.dtype('int32'))] is primary
    copied.set_mirror((None, None, None))
    assert copied.tile_geometries == {(True, 32, np.dtype('int32')): primary}
    assert cache.tile_geometries == entries
