# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""FixedNum objects carry a structure-free CSR config for indexed products."""

import re
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent import FixedNumPerPre, FixedNumPerPost, PerformanceWarning
from brainevent._csr.config import CSRConfig
from brainevent._test_util import requires_gpu


def _matrix(cls, backend='jax_raw'):
    shape = (2, 3) if cls is FixedNumPerPre else (3, 2)
    return cls(
        jnp.array([[1., 2.], [3., 4.]]), jnp.array([[0, 2], [1, 2]], jnp.int32),
        shape=shape, backend=backend, config=CSRConfig(tpr_threshold=5),
    )


def _unfavorable(matrix):
    """``transpose_W`` of the mirror-driven direction and its dense operator."""
    transpose = isinstance(matrix, FixedNumPerPost)
    dense = np.asarray(matrix.todense())
    return transpose, dense.T if transpose else dense


def _forbid(monkeypatch, owner, name, message):
    def forbidden(*args, **kwargs):
        raise AssertionError(message)

    monkeypatch.setattr(owner, name, forbidden)


@pytest.mark.parametrize('cls', [FixedNumPerPre, FixedNumPerPost])
def test_fcn_construction_keeps_mirror_lazy_and_config_leafless(cls):
    matrix = _matrix(cls)
    assert matrix._cache.mirror is None
    assert matrix.config.tpr_threshold == 5
    assert jax.tree_util.tree_leaves(matrix.config) == []
    # Only ``data`` and ``indices``: the config adds no pytree leaves.
    assert len(jax.tree_util.tree_leaves(matrix)) == 2


@pytest.mark.parametrize('cls', [FixedNumPerPre, FixedNumPerPost])
def test_first_fcn_mirror_is_built_once_and_shared_by_siblings(monkeypatch, cls):
    from brainevent._fcn import main

    matrix = _matrix(cls)
    config = matrix.config
    sibling = matrix.with_data(matrix.data * 2)
    transpose = matrix.T
    calls = []
    build = main.fixed_conn_num_csc_structure

    def record(*args, **kwargs):
        calls.append(kwargs.get('shape'))
        return build(*args, **kwargs)

    monkeypatch.setattr(main, 'fixed_conn_num_csc_structure', record)
    mirror = matrix._mirror()
    assert calls == [(2, 3)]
    assert sibling._mirror() is mirror
    assert transpose._mirror() is mirror
    assert matrix._mirror() is mirror
    assert len(calls) == 1
    # Building the mirror leaves every config untouched.
    assert matrix.config is config
    assert sibling.config is config
    assert transpose.config is config
    indptr, indices, perm = mirror
    np.testing.assert_array_equal(indptr, [0, 1, 2, 4])
    np.testing.assert_array_equal(indices, [0, 1, 0, 1])
    np.testing.assert_array_equal(perm, [0, 2, 1, 3])


@pytest.mark.parametrize('cls', [FixedNumPerPre, FixedNumPerPost])
def test_fcn_pytree_round_trip_restores_mirror_and_config_without_preparing(monkeypatch, cls):
    from brainevent._csr.binary import initialize
    from brainevent._fcn import main

    matrix = _matrix(cls)
    sibling = matrix.with_data(matrix.data * 2)
    mirror = matrix._mirror()
    _forbid(monkeypatch, initialize, 'load_csr_config', 'pytree reconstruction must not select a config')
    _forbid(monkeypatch, main, 'fixed_conn_num_csc_structure', 'pytree reconstruction must reuse the mirror')
    leaves, tree = jax.tree_util.tree_flatten(sibling)
    # data, indices and the three mirror arrays; the config has no leaves.
    assert len(leaves) == 5
    restored = jax.tree_util.tree_unflatten(tree, leaves)
    assert type(restored) is cls
    assert restored._cache is not sibling._cache
    assert all(new is old for new, old in zip(restored._cache.mirror, mirror, strict=True))
    assert restored.config.mv == matrix.config.mv
    assert jax.tree_util.tree_structure(restored) == tree
    np.testing.assert_array_equal(restored.todense(), sibling.todense())
    children, aux = sibling.tree_flatten()
    with pytest.raises(ValueError, match='leaves'):
        cls.tree_unflatten(aux, children[:-1])
    result = jax.jit(lambda value: value.with_data(value.data * 2))(restored)
    np.testing.assert_array_equal(result.data, restored.data * 2)
    assert result.config.mv == matrix.config.mv
    for new, old in zip(result._cache.mirror, mirror, strict=True):
        np.testing.assert_array_equal(new, old)


@pytest.mark.parametrize('cls', [FixedNumPerPre, FixedNumPerPost])
def test_fcn_mirror_replacement_copies_cache_and_leaves_siblings(cls):
    matrix = _matrix(cls).build_weight_indices()
    sibling = matrix.with_data(matrix.data * 2)
    shared = matrix._cache
    config = matrix.config
    original = sibling._cache.mirror
    replacement = tuple(jnp.array(array) for array in matrix.buffers['csc'])
    matrix.set_buffer('csc', replacement)
    assert matrix._cache is not shared
    assert sibling._cache is shared
    assert sibling._cache.mirror is original
    assert all(new is given for new, given in zip(matrix._cache.mirror, replacement, strict=True))
    assert matrix.config is config
    assert sibling.config is config
    # The replacement drives the indexed product.
    transpose, dense = _unfavorable(matrix)
    events = jnp.array([True, False, True])
    np.testing.assert_allclose(matrix._binary_matvec(events, transpose_W=transpose), dense @ events)


@pytest.mark.parametrize('cls', [FixedNumPerPre, FixedNumPerPost])
def test_fcn_traced_missing_mirror_is_not_built(monkeypatch, cls):
    from brainevent._fcn import main

    matrix = _matrix(cls)
    _forbid(monkeypatch, main, 'fixed_conn_num_csc_structure',
            'a traced missing mirror must not be built on the host')

    @jax.jit
    def check(value):
        assert value._mirror() is None
        return value.data * 2

    np.testing.assert_array_equal(check(matrix), matrix.data * 2)
    assert matrix._cache.mirror is None


@pytest.mark.parametrize('cls', [FixedNumPerPre, FixedNumPerPost])
def test_fcn_closure_builds_first_mirror_during_tracing(cls):
    matrix = _matrix(cls)
    config = matrix.config
    transpose, dense = _unfavorable(matrix)
    values = jnp.array([True, False, True])
    product = jax.jit(lambda events: matrix._binary_matvec(events, transpose_W=transpose))
    np.testing.assert_allclose(product(values), dense @ values)
    mirror = matrix._cache.mirror
    assert mirror is not None
    assert not any(isinstance(array, jax.core.Tracer) for array in mirror)
    assert matrix.config is config


@pytest.mark.parametrize('cls', [FixedNumPerPre, FixedNumPerPost])
@pytest.mark.parametrize('backend', ['cuda_raw', 'jax_raw', 'numba', None])
def test_fcn_traced_carried_mirror_takes_indexed_route_for_every_backend(monkeypatch, cls, backend):
    from brainevent._fcn import main

    matrix = jax.jit(lambda value: value.build_weight_indices())(_matrix(cls))
    assert matrix._cache.mirror is not None
    matrix.backend = backend
    calls = []

    def indexed(name):
        def record(*args, config, **kwargs):
            calls.append((name, kwargs['backend'], config.mv))
            return jnp.zeros((2, 2), dtype=jnp.float32)
        return record

    def gather(*args, **kwargs):
        calls.append(('gather', kwargs['backend']))
        return jnp.zeros((2, 2), dtype=jnp.float32)

    monkeypatch.setattr(main, 'binary_csrmv_indexed', indexed('mv'))
    monkeypatch.setattr(main, 'binary_csrmm_indexed', indexed('mm'))
    monkeypatch.setattr(main, 'binary_fcnmv', gather)
    monkeypatch.setattr(main, 'binary_fcnmm', gather)
    transpose, _ = _unfavorable(matrix)
    events = jnp.ones((3, 2), dtype=jnp.bool_)
    jax.jit(lambda value: (value._binary_matvec(events[:, 0], transpose_W=transpose),
                           value._binary_matmat(events, transpose_W=transpose)))(matrix)
    schedule = matrix.config.mv
    assert calls == [('mv', backend, schedule), ('mm', backend, schedule)]


@pytest.mark.parametrize('cls', [FixedNumPerPre, FixedNumPerPost])
@pytest.mark.parametrize('backend', ['cuda_raw', 'jax_raw', 'numba', None])
def test_fcn_traced_structure_without_mirror_gathers_with_gather_backend(monkeypatch, cls, backend):
    from brainevent._fcn import main

    matrix = _matrix(cls)
    matrix.backend = backend
    expected = 'jax_raw' if backend == 'cuda_raw' else backend
    if backend is None and jax.default_backend() == 'gpu':
        expected = 'jax_raw'
    calls = []

    def gather(*args, **kwargs):
        calls.append((kwargs['transpose'], kwargs['backend']))
        return jnp.zeros((2, 2), dtype=jnp.float32)

    _forbid(monkeypatch, main, 'binary_csrmv_indexed', 'no mirror is available')
    _forbid(monkeypatch, main, 'binary_csrmm_indexed', 'no mirror is available')
    monkeypatch.setattr(main, 'binary_fcnmv', gather)
    monkeypatch.setattr(main, 'binary_fcnmm', gather)
    transpose, _ = _unfavorable(matrix)
    events = jnp.ones((3, 2), dtype=jnp.bool_)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PerformanceWarning)
        jax.jit(lambda value: (value._binary_matvec(events[:, 0], transpose_W=transpose),
                               value._binary_matmat(events, transpose_W=transpose)))(matrix)
    assert calls == [(False, expected), (False, expected)]
    assert matrix._cache.mirror is None


def _hybrid_mm_targets(function, *args):
    text = jax.jit(function).lower(*args).compile().as_text()
    return [t for t in re.findall(r'custom_call_target="([^"]+)"', text) if t.startswith('csr_hybrid_mm')]


@requires_gpu
@pytest.mark.parametrize('cls', [FixedNumPerPre, FixedNumPerPost])
@pytest.mark.parametrize('backend', ['cuda_raw', None])
def test_fcn_cuda_mm_with_mirror_built_under_jit(cls, backend):
    matrix = _matrix(cls, backend=backend)
    transpose, dense = _unfavorable(matrix)
    events = jnp.array([[True, False], [False, True], [True, True]])
    expected = dense @ np.asarray(events, np.float32)

    # A mirror computed inside jit and carried out as pytree leaves.
    carried = jax.jit(lambda value: value.build_weight_indices())(matrix)
    assert carried._cache.mirror is not None
    product = lambda value, spikes: value._binary_matmat(spikes, transpose_W=transpose)
    assert _hybrid_mm_targets(product, carried, events)
    np.testing.assert_allclose(jax.jit(product)(carried, events), expected, rtol=1e-6)

    # The whole matrix built inside jit from traced arrays, as brainspike does.
    def built(data, indices, spikes):
        traced = cls(data, indices, shape=matrix.shape, backend=backend,
                     config=matrix.config, precompute_weight_indices=True)
        return traced._binary_matmat(spikes, transpose_W=transpose)

    assert _hybrid_mm_targets(built, matrix.data, matrix.indices, events)
    np.testing.assert_allclose(jax.jit(built)(matrix.data, matrix.indices, events), expected, rtol=1e-6)


@pytest.mark.parametrize('cls', [FixedNumPerPre, FixedNumPerPost])
def test_fcn_config_shares_and_survives_jit_without_reading_persistence(monkeypatch, cls):
    from brainevent._csr.binary import initialize

    matrix = _matrix(cls)
    _forbid(monkeypatch, initialize, 'load_csr_config', 'data-only reconstruction must not resolve configuration')
    assert matrix.T.config is matrix.config
    assert matrix.with_data(matrix.data * 2).config is matrix.config
    result = jax.jit(lambda value: value.with_data(value.data * 2))(matrix)
    assert type(result) is cls
    assert result.config.mv == matrix.config.mv
    assert result.config.tpr_threshold == 5
    np.testing.assert_array_equal(result.data, matrix.data * 2)
    converted = matrix.tocsr()
    assert converted.config is matrix.config
    np.testing.assert_allclose(converted.todense(), matrix.todense())
    assert matrix.tocsc().config is matrix.config


@pytest.mark.parametrize('cls', [FixedNumPerPre, FixedNumPerPost])
def test_equal_fcn_schedules_share_one_executable(cls):
    matrix = _matrix(cls)
    run = jax.jit(lambda value: value.data * 2)
    run(matrix)
    # A fresh config with the same scheduler is the same tree structure.
    same = cls(matrix.data, matrix.indices, shape=matrix.shape, backend=matrix.backend,
               config=CSRConfig(tpr_threshold=5))
    assert same.config is not matrix.config
    assert jax.tree_util.tree_structure(same) == jax.tree_util.tree_structure(matrix)
    run(same)
    assert run._cache_size() == 1
    other = cls(matrix.data, matrix.indices, shape=matrix.shape, backend=matrix.backend,
                config=CSRConfig(tpr_threshold=6))
    run(other)
    assert run._cache_size() == 2


@pytest.mark.parametrize('cls', [FixedNumPerPre, FixedNumPerPost])
def test_indexed_products_receive_the_matrix_config(monkeypatch, cls):
    from brainevent._fcn import main

    matrix = _matrix(cls)
    configs = []

    def record(*args, config, **kwargs):
        configs.append((kwargs['shape'], config))
        return jnp.zeros((2,))

    monkeypatch.setattr(main, 'binary_csrmv_indexed', record)
    monkeypatch.setattr(main, 'binary_csrmm_indexed', record)
    transpose, _ = _unfavorable(matrix)
    matrix._binary_matvec(jnp.ones(3, bool), transpose_W=transpose)
    matrix._binary_matmat(jnp.ones((3, 2), bool), transpose_W=transpose)
    assert len(configs) == 2
    # The mirror traversal has the reversed stored shape but the same config.
    assert all(shape == (3, 2) and config is matrix.config for shape, config in configs)


@pytest.mark.parametrize('cls', [FixedNumPerPre, FixedNumPerPost])
def test_fcn_slices_keep_config(monkeypatch, cls):
    from brainevent._csr.binary import initialize

    matrix = _matrix(cls)
    _forbid(monkeypatch, initialize, 'load_csr_config', 'a slice must keep the matrix config')
    eager = matrix.slice_rows([1])
    assert eager.config is matrix.config
    np.testing.assert_allclose(eager.todense(), matrix.todense()[1:2])
    if cls is FixedNumPerPre:
        traced = jax.jit(lambda value: value.slice_rows(jnp.array([1])))(matrix)
        assert type(traced) is FixedNumPerPre
        assert traced.config.mv == matrix.config.mv
        np.testing.assert_allclose(traced.todense(), matrix.todense()[1:2])
