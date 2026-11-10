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
def test_fcn_construction_and_rebuild_keep_mirror_statistics_lazy(cls):
    matrix = _matrix(cls)
    original_config = matrix.config
    assert matrix.config.primary is not None
    assert matrix.config.mirror is None
    assert matrix.config.mirror_max_row_nnz is None
    assert matrix._cache.mirror is None

    rebuilt = matrix.with_rebuilt_config()
    assert matrix.config is original_config
    assert rebuilt.config.primary is not None
    assert rebuilt.config.mirror is None
    assert rebuilt._cache.mirror is None


@pytest.mark.parametrize('cls', [FixedNumPerPre, FixedNumPerPost])
def test_first_fcn_mirror_prepares_once_and_siblings_reuse_statistics(
    monkeypatch, cls,
):
    from brainevent._csr import config as config_mod

    matrix = _matrix(cls)
    sibling = matrix.with_data(matrix.data * 2)
    transpose = matrix.T
    primary = matrix.config.primary
    calls = []
    prepare = config_mod._structure_plan

    def record(lengths):
        calls.append(np.asarray(lengths).copy())
        return prepare(lengths)

    monkeypatch.setattr(config_mod, '_structure_plan', record)
    mirror = matrix._mirror()
    assert matrix.config.primary is primary
    assert matrix.config.mirror_max_row_nnz == 2
    assert matrix.config.mirror is not None
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], [1, 1, 2])
    assert sibling._mirror() is mirror
    assert transpose._mirror() is mirror
    assert matrix._mirror() is mirror
    assert len(calls) == 1
    assert sibling.config.mirror is matrix.config.mirror
    assert transpose.config.mirror is matrix.config.mirror


@pytest.mark.parametrize('cls', [FixedNumPerPre, FixedNumPerPost])
def test_fcn_pytree_binds_shared_statistics_without_preparing(
    monkeypatch, cls,
):
    from brainevent._csr import config as config_mod

    matrix = _matrix(cls)
    sibling = matrix.with_data(matrix.data * 2)
    matrix._mirror()

    def no_prepare(*args, **kwargs):
        raise AssertionError(
            'pytree reconstruction must reuse mirror statistics',
        )

    monkeypatch.setattr(config_mod, '_structure_plan', no_prepare)
    monkeypatch.setattr(CSRConfig, 'prepare_structure', no_prepare)
    leaves, tree = jax.tree_util.tree_flatten(sibling)
    restored = jax.tree_util.tree_unflatten(tree, leaves)
    assert sibling.config.mirror is matrix.config.mirror
    assert restored.config.mirror_max_row_nnz == 2
    assert restored._cache.mirror_preparation[0] is restored.config.mirror
    np.testing.assert_array_equal(
        restored.config.mirror.long_mask, matrix.config.mirror.long_mask,
    )
    np.testing.assert_array_equal(
        jax.jit(lambda value: value.with_data(value.data * 2))(restored).data,
        restored.data * 2,
    )


@pytest.mark.parametrize('cls', [FixedNumPerPre, FixedNumPerPost])
def test_fcn_mirror_replacement_detaches_prepared_statistics(cls):
    matrix = _matrix(cls).build_weight_indices()
    sibling = matrix.with_data(matrix.data * 2)
    original_preparation = sibling._cache.mirror_preparation
    primary = matrix.config.primary
    replacement = tuple(jnp.array(array) for array in matrix.buffers['csc'])
    matrix.set_buffer('csc', replacement)
    assert matrix.config.primary is primary
    assert matrix._cache is not sibling._cache
    assert sibling._cache.mirror_preparation is original_preparation
    assert matrix._cache.mirror_preparation[0] is matrix.config.mirror
    assert matrix.config.mirror is not sibling.config.mirror
    assert (matrix.config.mirror_max_row_nnz
            == sibling.config.mirror_max_row_nnz == 2)


@pytest.mark.parametrize('cls', [FixedNumPerPre, FixedNumPerPost])
def test_fcn_traced_missing_mirror_does_not_prepare_statistics(
    monkeypatch, cls,
):
    from brainevent._csr import config as config_mod

    matrix = _matrix(cls)

    def no_prepare(*args, **kwargs):
        raise AssertionError(
            'traced missing mirror must not prepare on the host',
        )

    monkeypatch.setattr(config_mod, '_structure_plan', no_prepare)

    @jax.jit
    def check(value):
        assert value._mirror() is None
        assert value.config.mirror is None
        return value.data * 2

    np.testing.assert_array_equal(check(matrix), matrix.data * 2)
    assert matrix._cache.mirror is None


@pytest.mark.parametrize('cls', [FixedNumPerPre, FixedNumPerPost])
def test_fcn_closure_prepares_first_mirror_during_tracing(cls):
    matrix = _matrix(cls)
    transpose = cls is FixedNumPerPost
    values = jnp.array([True, False, True])
    product = jax.jit(lambda events: matrix._binary_matvec(
        events, transpose_W=transpose,
    ))
    expected = (matrix.todense().T @ values if transpose
                else matrix.todense() @ values)
    np.testing.assert_allclose(product(values), expected)
    assert matrix.config.mirror is not None
    assert matrix.config.mirror_max_row_nnz == 2
    assert matrix._cache.mirror_preparation[0] is matrix.config.mirror


@pytest.mark.parametrize('cls', [FixedNumPerPre, FixedNumPerPost])
def test_fcn_failed_mirror_statistics_can_be_retried(monkeypatch, cls):
    from brainevent._csr import config as config_mod

    matrix = _matrix(cls)
    original_config = matrix.config
    prepare = config_mod._structure_plan

    def fail(*args, **kwargs):
        raise RuntimeError('injected mirror statistics failure')

    monkeypatch.setattr(config_mod, '_structure_plan', fail)
    with pytest.raises(
        RuntimeError, match='injected mirror statistics failure',
    ):
        matrix._mirror()
    mirror = matrix._cache.mirror
    assert mirror is not None
    assert matrix.config is original_config
    assert matrix._cache.mirror_preparation is None
    monkeypatch.setattr(config_mod, '_structure_plan', prepare)
    assert matrix._mirror() is mirror
    assert matrix.config.mirror_max_row_nnz == 2
    assert matrix._cache.mirror_preparation[0] is matrix.config.mirror


@pytest.mark.parametrize('cls', [FixedNumPerPre, FixedNumPerPost])
@pytest.mark.parametrize('backend', ['cuda_raw', 'jax_raw', 'numba'])
def test_fcn_traced_carried_mirror_without_statistics_selects_mm_route(
    monkeypatch, cls, backend,
):
    from brainevent._fcn import main

    matrix = jax.jit(lambda value: value.build_weight_indices())(_matrix(cls))
    assert matrix._cache.mirror is not None
    assert matrix.config.mirror is None
    matrix.backend = backend
    calls = []

    def indexed(*args, **kwargs):
        calls.append(('indexed', kwargs['backend']))
        return jnp.zeros((2, 2), dtype=jnp.float32)

    def gather(*args, **kwargs):
        calls.append(('gather', kwargs['backend']))
        return jnp.zeros((2, 2), dtype=jnp.float32)

    monkeypatch.setattr(main, 'binary_csrmm_indexed', indexed)
    monkeypatch.setattr(main, 'binary_fcnmm', gather)
    events = jnp.ones((3, 2), dtype=jnp.bool_)
    jax.jit(lambda value: value._binary_matmat(
        events, transpose_W=cls is FixedNumPerPost,
    ))(matrix)
    assert calls == ([('gather', 'jax_raw')] if backend == 'cuda_raw'
                     else [('indexed', backend)])


@pytest.mark.skipif(
    jax.default_backend() != 'gpu', reason='native mirror MM requires a GPU',
)
@pytest.mark.parametrize('cls', [FixedNumPerPre, FixedNumPerPost])
@pytest.mark.parametrize('backend', ['cuda_raw', None])
def test_fcn_cuda_mm_with_mirror_built_under_jit(cls, backend):
    matrix = _matrix(cls)
    matrix.backend = backend
    matrix = matrix.with_rebuilt_config()
    expected_matrix = (matrix.todense().T if cls is FixedNumPerPost
                       else matrix.todense())
    prepared = jax.jit(lambda value: value.build_weight_indices())(matrix)
    assert prepared._cache.mirror is not None
    assert prepared.config.mirror is None
    events = jnp.array([[True, False], [False, True], [True, True]])
    product = jax.jit(lambda value, spikes: value._binary_matmat(
        spikes, transpose_W=cls is FixedNumPerPost))
    np.testing.assert_allclose(
        product(prepared, events), expected_matrix @ events,
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
