# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Automatic sorted storage, tile lifecycle and inherited CSR contracts."""

from dataclasses import replace

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent import BinaryArray, CSC, CSR, TCSR, WeightedArray
from brainevent._csr.config import CSRConfig
from brainevent._tcsr import main as tcsr_main


def _matrix(*, backend='jax_raw', data=None, backward_algorithm='bptt'):
    if data is None:
        data = jnp.array([1., 2., 3.], dtype=jnp.float32)
    return TCSR(CSR(
        data, jnp.array([0, 2, 1], dtype=jnp.int32),
        jnp.array([0, 2, 3], dtype=jnp.int32), shape=(2, 3),
        backend=backend, backward_algorithm=backward_algorithm,
        config=CSRConfig(tpr_threshold=5),
    ))


def test_constructor_stably_sorts_duplicate_edges_without_mutating_source():
    source = CSR(
        jnp.array([10., 20., 30., 40., 50.]) * u.mV,
        jnp.array([2, 1, 1, 2, 0], dtype=jnp.int32),
        jnp.array([0, 3, 5], dtype=jnp.int32), shape=(2, 3),
        backend='jax_raw', config=CSRConfig(tpr_threshold=7),
    ).build_weight_indices().prepare_diag_add()
    data, indices, indptr = source.data, source.indices, source.indptr
    matrix = TCSR(source)
    np.testing.assert_array_equal(matrix.indices, [1, 1, 2, 0, 2])
    np.testing.assert_array_equal(u.get_mantissa(matrix.data), [20., 30., 10., 50., 40.])
    assert u.get_unit(matrix.data) == u.mV and matrix.nse == 5
    assert source.data is data and source.indices is indices and source.indptr is indptr
    np.testing.assert_array_equal(source.indices, [2, 1, 1, 2, 0])
    assert matrix.backend == 'jax_raw' and matrix.config.tpr_threshold == 7
    assert matrix._cache is not source._cache
    assert matrix._cache.mirror is None and matrix._cache.diag_plan is None
    assert matrix._cache.tile_geometries and all(k[0] for k in matrix._cache.tile_geometries)
    np.testing.assert_allclose(u.get_mantissa(matrix.todense()), u.get_mantissa(source.todense()))


@pytest.mark.parametrize('data', [jnp.array(2.), jnp.array([2.])])
def test_constructor_keeps_homogeneous_weight_shape(data):
    source = CSR(data, jnp.array([2, 0, 1], jnp.int32), jnp.array([0, 2, 3], jnp.int32),
                 shape=(2, 3), backend='jax_raw')
    matrix = TCSR(source, backend='numba')
    assert matrix.data.shape == data.shape and matrix.backend == 'numba'
    assert source.backend == 'jax_raw'
    np.testing.assert_array_equal(matrix.indices, [0, 2, 1])


@pytest.mark.parametrize('factory', [TCSR, TCSR.from_sorted_csr])
def test_source_must_be_csr(factory):
    with pytest.raises(TypeError, match='CSR'):
        factory(jnp.ones((2, 2)))


def test_sorted_existing_tcsr_reuses_geometries_without_host_reads(monkeypatch):
    source = _matrix().prepare_diag_add()
    source._mirror()

    def forbidden(*args, **kwargs):
        raise AssertionError('same-structure trusted assembly must not prepare arrays')

    monkeypatch.setattr(tcsr_main, 'sort_csr', forbidden)
    monkeypatch.setattr(jax, 'device_get', forbidden)
    monkeypatch.setattr(CSRConfig, 'prepare_structure', forbidden)
    result = TCSR.from_sorted_csr(source)
    assert result.indices is source.indices and result.data is source.data
    assert result.config is source.config and result._cache is not source._cache
    assert result._cache.tile_geometries is not source._cache.tile_geometries
    assert result._cache.mirror is source._cache.mirror
    assert result._cache.diag_plan is source._cache.diag_plan
    assert all(result._cache.tile_geometries[k] is p for k, p in source._cache.tile_geometries.items())


def test_sorted_plain_csr_factory_prepares_primary_and_keeps_mirror_lazy():
    source = CSR.fromdense(jnp.array([[1., 0., 2.], [0., 3., 0.]]), backend='jax_raw')
    result = TCSR.from_sorted_csr(source)
    assert type(result) is TCSR and result._cache.mirror is None
    assert result._get_binary_mm_plan(188, transpose=True) is not None
    assert source._cache.tile_plans == {}


def test_dense_factory_row_slice_diagonal_and_transpose_types():
    dense = jnp.array([[1., 0., 2.], [0., 3., 0.]])
    matrix = TCSR.fromdense(dense, backend='jax_raw',
                            precompute_weight_indices=True, config=CSRConfig(tpr_threshold=4))
    assert type(matrix) is TCSR and matrix.config.tpr_threshold == 4
    np.testing.assert_array_equal(matrix.todense(), dense)
    sliced = matrix.slice_rows([1, 0, 1])
    assert type(sliced) is TCSR and sliced.shape == (3, 3)
    assert sliced._cache is not matrix._cache
    assert sliced._get_binary_mm_plan(2, transpose=True) is not None
    assert not any(not k[0] for k in sliced._cache.tile_geometries)
    np.testing.assert_array_equal(sliced.todense(), dense[jnp.array([1, 0, 1])])
    augmented = matrix.diag_add(jnp.ones(2))
    assert type(augmented) is TCSR and augmented._cache is not matrix._cache
    assert augmented._get_binary_mm_plan(2, transpose=True) is not None
    np.testing.assert_array_equal(augmented.todense(), dense.at[0, 0].add(1).at[1, 1].add(1))
    assert type(matrix.T) is CSC and type(matrix.T.T) is CSR


def test_optional_prepare_compatibility_is_not_batch_specific():
    base = _matrix()
    primary = dict(base._cache.tile_geometries)
    result = base.prepare_binary_mm(3, transpose=False)
    assert base._cache.mirror is None
    assert result._cache.mirror is not None
    for key, geometry in primary.items():
        assert result._cache.tile_geometries[key] is geometry
    assert result._get_binary_mm_plan(4, transpose=False) is not None
    assert result._get_binary_mm_plan(65535, transpose=True, dtype=jnp.float64) is not None


def test_data_only_changes_share_geometries_and_obey_dtype_unit_contracts():
    matrix = _matrix()
    replaced = matrix.with_data(matrix.data * 2)
    assert type(replaced) is TCSR and replaced._cache is matrix._cache
    for incompatible in (jnp.ones(1, dtype=matrix.dtype), matrix.data.astype(jnp.float16), matrix.data * u.mV):
        with pytest.raises(AssertionError):
            matrix.with_data(incompatible)
    converted = matrix.apply(lambda data: data.astype(jnp.float16) * u.mV)
    assert converted._cache is matrix._cache
    assert converted._get_binary_mm_plan(2, transpose=True) is not None
    for result in (matrix * 2, matrix / 2, -matrix, matrix.prepare_diag_add(), matrix.with_rebuilt_config()):
        assert type(result) is TCSR and result._cache is matrix._cache


def test_mirror_replacement_invalidates_only_mirror_geometry():
    matrix = _matrix()
    matrix._mirror()
    sibling = matrix.with_data(matrix.data * 2)
    primary = matrix._get_binary_mm_plan(2, transpose=True)
    mirror_plan = matrix._get_binary_mm_plan(2, transpose=False)
    matrix.set_buffer('csc', tuple(jnp.array(a) for a in matrix.buffers['csc']))
    assert matrix._get_binary_mm_plan(2, transpose=True) is primary
    assert matrix._get_binary_mm_plan(2, transpose=False) is None
    assert sibling._get_binary_mm_plan(2, transpose=False) is mirror_plan
    matrix._mirror()
    assert matrix._get_binary_mm_plan(2, transpose=False) is not None
    assert matrix._get_binary_mm_plan(2, transpose=False) is not mirror_plan


def test_pytree_roundtrip_is_hashable_and_restores_all_geometries():
    matrix = _matrix().prepare_diag_add()
    matrix._mirror()
    children, aux = matrix.tree_flatten()
    hash(aux)
    restored = TCSR.tree_unflatten(aux, children)
    assert type(restored) is TCSR and restored._cache is not matrix._cache
    assert restored._cache.tile_geometries.keys() == matrix._cache.tile_geometries.keys()
    for key, plan in matrix._cache.tile_geometries.items():
        assert restored._cache.tile_geometries[key].local_targets is plan.local_targets
        assert restored._cache.tile_geometries[key].tile_offsets is plan.tile_offsets
    np.testing.assert_array_equal(restored.todense(), matrix.todense())
    with pytest.raises(ValueError, match='leaves'):
        TCSR.tree_unflatten(aux, children[:-1])


def test_pytree_placeholder_and_float0_restoration_never_prepares(monkeypatch):
    matrix = _matrix().prepare_diag_add()
    matrix._mirror()

    def forbidden(*args, **kwargs):
        raise AssertionError('pytree reconstruction must not prepare or inspect arrays')

    monkeypatch.setattr(tcsr_main, 'sort_csr', forbidden)
    monkeypatch.setattr(tcsr_main, 'build_tile_geometries', forbidden)
    monkeypatch.setattr(jax, 'device_get', forbidden)
    empty = jax.tree.map(lambda _: None, matrix)
    assert empty.data is None and empty.indices is None
    assert all(p.local_targets is None and p.tile_offsets is None for p in empty._cache.tile_geometries.values())
    differential = jax.tree.map(lambda a: np.zeros(a.shape, dtype=jax.dtypes.float0), matrix)
    assert differential.indices.dtype == jax.dtypes.float0
    for plan in differential._cache.tile_geometries.values():
        assert plan.local_targets.dtype == jax.dtypes.float0
        assert plan.offset_dtype == np.dtype('int32')


def test_jit_scan_and_vmap_preserve_tcsr_and_geometry_payload():
    matrix = _matrix()
    matrix._mirror()
    restored = jax.jit(lambda m: m)(matrix)
    assert type(restored) is TCSR
    assert restored._get_binary_mm_plan(2, transpose=False) is not None
    carried, outputs = jax.jit(lambda m: jax.lax.scan(
        lambda carry, _: (carry.apply(lambda d: d + 1), carry.data.sum()), m, jnp.arange(3),
    ))(matrix)
    assert type(carried) is TCSR
    np.testing.assert_allclose(carried.data, matrix.data + 3)
    np.testing.assert_allclose(outputs, [6., 9., 12.])
    stacked = jax.tree.map(lambda a: jnp.stack([a, a]), matrix)
    np.testing.assert_allclose(jax.vmap(lambda m: m.data.sum())(stacked), [6., 6.])
    events = BinaryArray(jnp.array([[True, False], [True, True]]))
    result = jax.vmap(lambda d: events @ matrix.with_data(d))(jnp.stack([matrix.data, matrix.data * 2]))
    np.testing.assert_allclose(result[1], result[0] * 2)


@pytest.mark.parametrize('homo', [False, True])
def test_matrix_parameter_gradient_preserves_canonical_order_and_float0(homo):
    matrix = _matrix(data=jnp.array([2.]) if homo else None)
    events = BinaryArray(jnp.array([[True, False], [True, True]]))
    gradient = jax.jit(jax.grad(lambda m: (events @ m).sum(), allow_int=True))(matrix)
    assert type(gradient) is TCSR
    np.testing.assert_allclose(gradient.data, [5.] if homo else [2., 2., 1.])
    assert gradient.indices.dtype == gradient.indptr.dtype == jax.dtypes.float0
    assert all(p.local_targets.dtype == jax.dtypes.float0 for p in gradient._cache.tile_geometries.values())


def test_cpu_event_numeric_mv_and_weighted_products_follow_csr():
    matrix = _matrix()
    dense = matrix.todense()
    left = jnp.array([[1., 0.], [-2., 3.]], jnp.float32)
    right = jnp.array([[1., 0.], [0., -2.], [4., 3.]], jnp.float32)
    np.testing.assert_allclose(BinaryArray(left) @ matrix, (left != 0) @ dense)
    np.testing.assert_allclose(matrix @ BinaryArray(right), dense @ (right != 0))
    np.testing.assert_allclose(WeightedArray(left) @ matrix, left @ dense)
    np.testing.assert_allclose(matrix @ WeightedArray(right), dense @ right)
    np.testing.assert_allclose(matrix @ BinaryArray(right[:, 0]), dense @ (right[:, 0] != 0))
    np.testing.assert_allclose(left @ matrix, left @ dense)
    np.testing.assert_allclose(matrix @ right, dense @ right)


def test_corrupt_geometry_reports_error():
    matrix = _matrix()
    plan = matrix._get_binary_mm_plan(2, transpose=True)
    matrix._cache.tile_geometries[plan.key] = replace(plan, tile_width=999)
    with pytest.raises(ValueError, match='geometry|geometr'):
        matrix._get_binary_mm_plan(2, transpose=True)


def test_hot_path_never_sorts_prepares_or_reads_host(monkeypatch):
    matrix = _matrix()
    matrix._mirror()
    left = jnp.array([[True, False], [True, True]])
    right = jnp.array([[True, False], [False, True], [True, True]])
    expected_left, expected_right = left @ matrix.todense(), matrix.todense() @ right

    def forbidden(*args, **kwargs):
        raise AssertionError('event hot path must only query existing geometry')

    monkeypatch.setattr(tcsr_main, 'sort_csr', forbidden)
    monkeypatch.setattr(tcsr_main, 'build_tile_geometries', forbidden)
    monkeypatch.setattr(TCSR, 'prepare_binary_mm', forbidden)
    monkeypatch.setattr(jax, 'device_get', forbidden)
    actual = jax.jit(lambda m, l, r: (BinaryArray(l) @ m, m @ BinaryArray(r)))(matrix, left, right)
    np.testing.assert_allclose(actual[0], expected_left)
    np.testing.assert_allclose(actual[1], expected_right)


def test_invalid_geometry_object_reports_error():
    matrix = _matrix()
    plan = matrix._get_binary_mm_plan(2, transpose=True)
    matrix._cache.tile_geometries[plan.key] = object()
    with pytest.raises(ValueError, match='Invalid|invalid'):
        matrix._get_binary_mm_plan(2, transpose=True)


@pytest.mark.parametrize('shape', [(0, 3), (2, 0), (2, 3)])
@pytest.mark.parametrize('batch', [0, 2])
def test_empty_dimensions_edges_and_batches_follow_inherited_csr(shape, batch):
    matrix = TCSR.fromdense(jnp.zeros(shape), backend='jax_raw')
    left = BinaryArray(jnp.ones((batch, shape[0]), dtype=bool))
    right = BinaryArray(jnp.ones((shape[1], batch), dtype=bool))
    assert matrix.nse == 0
    np.testing.assert_array_equal(left @ matrix, jnp.zeros((batch, shape[1])))
    np.testing.assert_array_equal(matrix @ right, jnp.zeros((shape[0], batch)))
