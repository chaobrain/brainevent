# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""FixedNum objects carry prepared CSR configuration for indexed products."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent import FixedNumPerPre, FixedNumPerPost
from brainevent._csr.config import CSRConfig


def _matrix(cls):
    shape = (2, 3) if cls is FixedNumPerPre else (3, 2)
    return cls(
        jnp.ones((2, 2)), jnp.array([[0, 2], [1, 2]], jnp.int32),
        shape=shape, backend='jax_raw', config=CSRConfig(tpr_threshold=5),
    )


@pytest.mark.parametrize('cls', [FixedNumPerPre, FixedNumPerPost])
def test_fcn_config_shares_and_survives_jit_without_reading_persistence(monkeypatch, cls):
    from brainevent._csr.binary import initialize

    matrix = _matrix(cls)

    def no_load(*args, **kwargs):
        raise AssertionError('data-only reconstruction must not resolve configuration')

    monkeypatch.setattr(initialize, 'load_csr_config', no_load)
    assert matrix.T.config is matrix.config
    assert matrix.with_data(matrix.data * 2).config is matrix.config
    result = jax.jit(lambda value: value.with_data(value.data * 2))(matrix)
    assert result.config.shape == (2, 3)
    assert result.config.nnz == 4
    assert result.config.tpr_threshold == 5
    np.testing.assert_array_equal(result.data, matrix.data * 2)
    converted = matrix.tocsr()
    expected = matrix.config if cls is FixedNumPerPre else matrix.config.swap_layouts()
    assert converted.config.primary is expected.primary
    assert converted.config.mirror is expected.mirror
    assert converted.config.shape == expected.shape
    assert converted.config.nnz == expected.nnz
    np.testing.assert_allclose(converted.todense(), matrix.todense())


@pytest.mark.parametrize('cls', [FixedNumPerPre, FixedNumPerPost])
def test_indexed_products_receive_the_prepared_mirror_config(monkeypatch, cls):
    from brainevent._fcn import main

    matrix = _matrix(cls)
    configs = []

    def record(*args, config, **kwargs):
        configs.append(config)
        return jnp.zeros((2,))

    monkeypatch.setattr(main, 'binary_csrmv_indexed', record)
    monkeypatch.setattr(main, 'binary_csrmm_indexed', record)
    transpose = cls is FixedNumPerPost
    matrix._binary_matvec(jnp.ones(3, bool), transpose_W=transpose)
    matrix._binary_matmat(jnp.ones((3, 2), bool), transpose_W=transpose)
    assert len(configs) == 2
    assert all(config.primary is matrix.config.mirror for config in configs)
    assert all(config.shape == matrix.config.shape[::-1] for config in configs)
    assert all(config.tpr_threshold == 5 for config in configs)


def test_fcn_slice_invalidates_traced_plan_and_rebuild_restores_it():
    matrix = _matrix(FixedNumPerPre)
    eager = matrix.slice_rows([0])
    assert eager.config.shape == (1, 3)
    assert eager.config.nnz == 2
    traced = jax.jit(lambda value: value.slice_rows(jnp.array([0])))(matrix)
    assert traced.config.primary is None
    assert traced.config.mirror is None
    prepared = traced.with_rebuilt_config()
    assert prepared.config.shape == (1, 3)
    assert prepared.config.tpr_threshold == 5


@pytest.mark.parametrize('cls', [FixedNumPerPre, FixedNumPerPost])
def test_with_rebuilt_config_shares_matrix_storage_without_reloading(monkeypatch, cls):
    from brainevent._csr.binary import initialize

    matrix = _matrix(cls).build_weight_indices()
    original_config = matrix.config
    original_cache = matrix._cache

    def no_load(*args, **kwargs):
        raise AssertionError('rebuilding configuration must retain existing tuning')

    monkeypatch.setattr(initialize, 'load_csr_config', no_load)
    rebuilt = matrix.with_rebuilt_config()

    assert type(rebuilt) is cls
    assert rebuilt is not matrix
    assert rebuilt.data is matrix.data
    assert rebuilt.indices is matrix.indices
    assert rebuilt._cache is original_cache
    assert matrix._cache is original_cache
    assert matrix.config is original_config
    assert rebuilt.config is not original_config
    assert rebuilt.config.mv == original_config.mv
    assert rebuilt.config.shape == original_config.shape == (2, 3)
    assert rebuilt.config.nnz == original_config.nnz == 4
    assert rebuilt.config.primary is not None
    assert rebuilt.config.mirror is not None
    assert rebuilt.config.primary is not original_config.primary
    assert rebuilt.config.mirror is not original_config.mirror
    np.testing.assert_array_equal(rebuilt.config.primary.long_mask, original_config.primary.long_mask)
    np.testing.assert_array_equal(rebuilt.config.mirror.long_mask, original_config.mirror.long_mask)
