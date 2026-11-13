# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Shared CSR cache lifecycle contracts needed by tiled subclasses."""

import jax.numpy as jnp
import numpy as np

from brainevent import CSR
from brainevent._csr.config import CSRConfig


def _matrix():
    return CSR(
        jnp.array([1., 2.]), jnp.array([1, 0], jnp.int32),
        jnp.array([0, 1, 2], jnp.int32), shape=(2, 2),
        backend='jax_raw', config=CSRConfig(),
    )


def test_mirror_replacement_keeps_primary_tiles_and_detaches_siblings():
    matrix = _matrix().build_weight_indices()
    primary, mirror_plan = object(), object()
    matrix._cache.tile_geometries.update({(True, 2, 'float32'): primary, (False, 2, 'float32'): mirror_plan})
    sibling = matrix.with_data(matrix.data * 2)
    matrix.set_buffer('csc', tuple(jnp.array(a) for a in matrix.buffers['csc']))
    assert matrix._cache.tile_geometries == {(True, 2, 'float32'): primary}
    assert sibling._cache.tile_geometries[(False, 2, 'float32')] is mirror_plan


def test_build_weight_indices_fresh_cache_keeps_primary_tiles(monkeypatch):
    matrix = _matrix()
    primary = object()
    matrix._cache.tile_geometries[(True, 2, 'float32')] = primary
    mirror = matrix._weight_indices()
    matrix._cache.mirror = None
    monkeypatch.setattr(CSR, '_mirror', lambda self: None)
    monkeypatch.setattr(CSR, '_weight_indices', lambda self: mirror)
    prepared = matrix.build_weight_indices()
    assert prepared._cache is not matrix._cache
    assert prepared._cache.tile_geometries == matrix._cache.tile_geometries
    assert prepared._cache.tile_geometries is not matrix._cache.tile_geometries
    assert matrix._cache.mirror is None
    assert prepared._cache.mirror is not None


def test_diag_add_creates_fresh_cache_without_tiles():
    matrix = _matrix()
    matrix._cache.tile_geometries[(True, 2, 'float32')] = object()
    result = matrix.diag_add(jnp.ones(2))
    assert result._cache.tile_geometries == {}
    assert matrix._cache.tile_geometries
    np.testing.assert_allclose(result.todense(), jnp.array([[1., 1.], [2., 1.]]))
