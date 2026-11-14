# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

import warnings

import jax
import jax.numpy as jnp
import numpy as np

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


def test_mirror_preparation_is_shared_by_copies_and_invalidated_on_replacement(
):
    preparation = (object(), 7)
    mirror = (object(), object(), object())
    cache = StructureCache(mirror=mirror, mirror_preparation=preparation)
    copied = cache.copy()
    assert copied.mirror_preparation is cache.mirror_preparation is preparation
    copied.set_mirror((object(), object(), object()))
    assert copied.mirror_preparation is None
    assert cache.mirror_preparation is preparation


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
