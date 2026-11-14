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


def test_structure_cache_holds_no_mm_preparation():
    # The hybrid MM is planned from shapes at lowering; no mirror plan is cached.
    assert set(StructureCache.__slots__) == {'mirror', 'diag_plan', 'tile_geometries'}
    with pytest.raises(TypeError):
        StructureCache(mirror_preparation=(object(), 7))


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
