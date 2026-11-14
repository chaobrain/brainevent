# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Canonical CSR argument names and narrow legacy TCSR entry points."""

import inspect

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent import BinaryArray, CSC, CSR, TCSR
from brainevent._csr.config import CSRConfig


def _source():
    return CSR(
        jnp.array([2., 3., 5.], jnp.float32) * u.mV,
        np.array([2, 0, 1], np.int32), np.array([0, 2, 3], np.int32),
        shape=(2, 3), backend='jax_raw', config=CSRConfig(tpr_threshold=7),
    )


@pytest.mark.parametrize('factory', [TCSR, TCSR.from_sorted_csr])
def test_source_keyword_is_a_legacy_alias_for_csr(factory):
    source = _source()
    if factory is not TCSR:
        source = CSR.fromdense(source.todense(), backend='jax_raw')
    result = factory(source=source)
    assert isinstance(result, TCSR)
    np.testing.assert_array_equal(u.get_mantissa(result.todense()), u.get_mantissa(source.todense()))
    with pytest.raises(TypeError, match='csr.*source|source.*csr'):
        factory(source, source=source)


@pytest.mark.parametrize('name', ['fromcsr', 'fromcsc'])
@pytest.mark.parametrize('legacy_keyword', [False, True])
def test_compatibility_factories_preserve_values_and_parent_configuration(name, legacy_keyword):
    source = _source()
    argument = source if name == 'fromcsr' else source.tocsc()
    keyword = 'source' if legacy_keyword else ('csr' if name == 'fromcsr' else 'csc')
    result = getattr(TCSR, name)(**{keyword: argument})
    assert type(result) is TCSR
    assert result.backend == source.backend
    assert result.config.tpr_threshold == source.config.tpr_threshold
    assert result._cache.tile_geometries == {}  # tiles are prepared lazily
    np.testing.assert_array_equal(u.get_mantissa(result.todense()), u.get_mantissa(source.todense()))
    np.testing.assert_array_equal(result.indices, [0, 2, 1])
    assert u.get_unit(result.data) == u.mV


def test_dense_factory_accepts_canonical_mat_and_legacy_source_keywords():
    dense = jnp.array([[1., 0.], [2., 3.]])
    for kwargs in ({'mat': dense}, {'source': dense}):
        matrix = TCSR.fromdense(**kwargs, binary_backend='jax')
        assert matrix.backend == 'jax_raw'
        np.testing.assert_array_equal(matrix.todense(), dense)
    with pytest.raises(TypeError, match='mat.*source|source.*mat'):
        TCSR.fromdense(dense, source=dense)


@pytest.mark.parametrize('factory_name', ['constructor', 'from_sorted_csr', 'fromcsr', 'fromcsc', 'fromdense'])
def test_legacy_backend_name_is_normalized_and_conflicts_are_explicit(factory_name):
    source = _source()
    if factory_name == 'from_sorted_csr':
        source = CSR.fromdense(source.todense(), backend='jax_raw')
    factory = TCSR if factory_name == 'constructor' else getattr(TCSR, factory_name)
    argument = source
    if factory_name == 'fromcsc':
        argument = source.tocsc()
    elif factory_name == 'fromdense':
        argument = source.todense()
    for kwargs in ({'backend': 'jax'}, {'binary_backend': 'jax'},
                   {'backend': 'jax_raw', 'binary_backend': 'jax'}):
        matrix = factory(argument, **kwargs)
        assert matrix.backend == 'jax_raw'
        assert matrix.with_data(matrix.data * 2).backend == 'jax_raw'
    with pytest.raises(ValueError, match='backend.*binary_backend|binary_backend.*backend'):
        factory(argument, backend='jax_raw', binary_backend='cuda_raw')


def test_mirror_alias_uses_common_cache_without_new_storage():
    matrix = TCSR(_source())
    assert not matrix.has_tcsc_mirror
    result = matrix.materialize_tcsc_mirror()
    assert type(result) is TCSR
    assert result.has_tcsc_mirror
    assert result.data is matrix.data and result.indices is matrix.indices
    assert result._cache is matrix._cache
    assert result._get_tile_geometry(2, transpose=True, prepare=True) is matrix._get_tile_geometry(2, transpose=True)
    assert jax.jit(lambda m: m.has_tcsc_mirror)(result)
    assert type(result.T) is CSC and type(result.T.T) is CSR


def test_shared_methods_keep_parent_parameter_names():
    for name in ('fromdense', 'slice_rows', 'with_data', 'apply', 'dt2t',
                 'dt2t_transposed', 'delay_layout', 'update_on_pre',
                 'update_on_post', 'apply_on_pre', 'apply_on_post', 'solve'):
        parent = inspect.signature(getattr(CSR, name))
        subclass = inspect.signature(getattr(TCSR, name))
        assert set(parent.parameters) <= set(subclass.parameters), name


def test_compatibility_factories_reject_wrong_formats_and_duplicate_arguments():
    source = _source()
    with pytest.raises(TypeError, match='CSR'):
        TCSR.fromcsr(source.tocsc())
    with pytest.raises(TypeError, match='CSC'):
        TCSR.fromcsc(source)
    with pytest.raises(TypeError, match='csc.*source|source.*csc'):
        TCSR.fromcsc(source.tocsc(), source=source.tocsc())


def test_missing_matrix_arguments_fail_clearly():
    with pytest.raises(TypeError, match='CSR'):
        TCSR()
    with pytest.raises(TypeError, match='CSR'):
        TCSR.from_sorted_csr()
    with pytest.raises(TypeError, match='mat'):
        TCSR.fromdense()


def _require_cuda():
    try:
        devices = jax.devices('gpu')
    except RuntimeError:
        pytest.skip('CUDA device required')
    if not any('cuda' in device.client.platform_version.lower() for device in devices):
        pytest.skip('CUDA device required')


@pytest.mark.parametrize('factory_name', ['constructor', 'fromcsr', 'fromcsc', 'fromdense'])
def test_checked_factories_run_inherited_mm_for_backend_override(factory_name):
    _require_cuda()
    source = _source()
    argument = source.tocsc() if factory_name == 'fromcsc' else source
    if factory_name == 'fromdense':
        argument = source.todense()
    factory = TCSR if factory_name == 'constructor' else getattr(TCSR, factory_name)
    matrix = factory(argument, binary_backend='cuda_raw')
    assert matrix.backend == 'cuda_raw'
    if factory_name != 'fromdense':
        assert matrix.config is source.config
    assert matrix._cache.tile_geometries == {}  # prepared by the first product
    events = jnp.array([[True, False], [True, True]])
    np.testing.assert_allclose(
        u.get_mantissa(BinaryArray(events) @ matrix),
        np.asarray(events) @ np.asarray(u.get_mantissa(source.todense())),
    )
    assert [key[0] for key in matrix._cache.tile_geometries] == [True]
    right = jnp.array([[True, False], [False, True], [True, True]])
    np.testing.assert_allclose(
        u.get_mantissa(matrix @ BinaryArray(right)),
        np.asarray(u.get_mantissa(source.todense())) @ np.asarray(right),
    )
    assert sorted(key[0] for key in matrix._cache.tile_geometries) == [False, True]


@pytest.mark.parametrize('built_on', ['default', 'cpu'])
def test_trusted_backend_override_runs_cuda_mm_with_parent_configuration(built_on):
    _require_cuda()
    dense = jnp.array([[1., 2., 0.], [0., 3., 4.]])
    if built_on == 'cpu':
        # A CPU-built JAX configuration needs no rebuild on the GPU.
        cpu = jax.devices('cpu')[0]
        with jax.default_device(cpu):
            source = CSR.fromdense(jax.device_put(dense, cpu), backend='jax_raw')
        source = jax.device_put(source, jax.devices('gpu')[0])
    else:
        source = CSR.fromdense(dense, backend='jax_raw')
    matrix = TCSR.from_sorted_csr(source, backend='cuda_raw')
    assert type(matrix) is TCSR and matrix.config is source.config
    events = jnp.array([[True, False], [True, True]])
    np.testing.assert_allclose(BinaryArray(events) @ matrix, np.asarray(events) @ np.asarray(dense))
    right = jnp.array([[True, False], [False, True], [True, True]])
    np.testing.assert_allclose(matrix @ BinaryArray(right), np.asarray(dense) @ np.asarray(right))
