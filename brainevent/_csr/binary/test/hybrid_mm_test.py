"""Prepared SpMM adapter validation, layout and execution-device contracts."""

from dataclasses import replace
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent._csr.config as config_mod
from brainevent._csr.config import CSRConfig, CSRDeviceInfo
import brainevent._csr.binary.binary as binary_mod


def prepared(*, weighted=False, homo=False):
    indices = jnp.array([0, 2, 1], jnp.int32)
    indptr = jnp.array([0, 2, 3], jnp.int32)
    device = CSRDeviceInfo('gpu', 'TEST_GPU', l2_bytes=4096, sms=8)
    config = CSRConfig(device_info=device).prepare_structure(indices, indptr, shape=(2, 3))
    spec, mask = config.kernel_args()
    weights = jnp.ones((1 if homo else 3,), jnp.float32)
    events = jnp.array([[1, 0], [-2, 3]], jnp.float32)
    kwargs = dict(
        config_spec=spec, call_plan=config.mm_plan(2, weights.dtype, weighted=weighted),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(events.shape, events.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        out_info=jax.ShapeDtypeStruct((3, 2), weights.dtype), weighted=weighted,
    )
    return kwargs, (weights, indices, indptr, events, mask)


@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('permuted,homo', [(False, False), (False, True), (True, False), (True, True)])
def test_adapter_preserves_layout_dtype_and_plan(monkeypatch, weighted, permuted, homo):
    params, args = prepared(weighted=weighted, homo=homo)
    modules, calls = [], []

    def load_module(source, name, **kwargs):
        modules.append((source, name, kwargs))

    def ffi_call(target, infos, **kwargs):
        def call(*operands, **attrs):
            calls.append((target, infos, operands, attrs))
            return (jnp.arange(6, dtype=jnp.float32).reshape(2, 3),)
        return call

    monkeypatch.setattr(binary_mod, 'load_cuda_file', load_module)
    monkeypatch.setattr(jax.ffi, 'ffi_call', ffi_call)
    kernel = binary_mod._make_hybrid_mm_kernel(**params, permuted=permuted)
    if permuted:
        args = (*args[:3], jnp.array([2, 0, 1], jnp.int32), *args[3:])
    actual, = kernel(*args)
    np.testing.assert_array_equal(actual, np.arange(6, dtype=np.float32).reshape(2, 3).T)
    target, infos, operands, attrs = calls[0]
    np.testing.assert_array_equal(operands[-1], args[-2].T)
    assert len(modules) == 1
    source, module, load_kwargs = modules[0]
    mode = 'weighted' if weighted else 'binary'
    layout = 'col_major' if permuted and not homo else 'row'
    assert source.name == f'hybrid_batchedv3_spmm_{"col_major_" if layout == "col_major" else ""}{mode}.cu'
    assert re.fullmatch(rf'csr_hybrid_mm_{layout}_{mode}_[0-9a-f]{{12}}', module)
    assert load_kwargs == dict(extra_include_paths=[str(source.parent)], allow_cuda_graph=True)
    assert target.startswith(module + '.')
    assert len(operands) == (6 if permuted and not homo else 5)
    assert ('_homo' in target) == homo
    assert infos[1].shape == (params['call_plan'].workspace_words,)
    assert attrs['task_capacity'] == params['call_plan'].task_capacity
    assert attrs['expected_sms'] == 8
    assert attrs['expected_l2_bytes'] == 4096


def test_adapter_requires_complete_plan_and_matching_shape(monkeypatch):
    monkeypatch.setattr(binary_mod, 'load_cuda_file', lambda *args, **kwargs: pytest.fail('must reject before compilation'))
    params, _ = prepared()
    with pytest.raises(ValueError, match='complete prepared CSRConfig'):
        binary_mod._make_hybrid_mm_kernel(**dict(params, call_plan=None))
    with pytest.raises(ValueError, match='operand structure and batch'):
        binary_mod._make_hybrid_mm_kernel(**dict(params, call_plan=replace(params['call_plan'], batch=7)))
    with pytest.raises(ValueError, match='value dtype'):
        binary_mod._make_hybrid_mm_kernel(**dict(params, call_plan=replace(params['call_plan'], dtype=np.dtype(np.float64))))
    with pytest.raises(ValueError, match='prepared config and invocation'):
        binary_mod._make_hybrid_mm_kernel(**dict(params, call_plan=replace(params['call_plan'], workspace_words=1)))


def test_empty_adapter_does_not_require_preparation_or_cuda(monkeypatch):
    monkeypatch.setattr(binary_mod, 'load_cuda_file', lambda *args, **kwargs: pytest.fail('empty operation must not compile'))
    params, args = prepared()
    kernel = binary_mod._make_hybrid_mm_kernel(**dict(
        params, config_spec=None, call_plan=None,
        indices_info=jax.ShapeDtypeStruct((0,), jnp.int32),
    ))
    np.testing.assert_array_equal(kernel(*args)[0], np.zeros((3, 2)))


@pytest.mark.parametrize('mask', [np.zeros(0, np.uint32), np.zeros(1, np.int32)])
def test_adapter_rejects_mismatched_mask_before_ffi(monkeypatch, mask):
    params, args = prepared()
    monkeypatch.setattr(binary_mod, 'load_cuda_file', lambda *args, **kwargs: None)

    def ffi_call(*args, **kwargs):
        return lambda *args, **kwargs: pytest.fail('invalid prepared mask must not reach FFI')

    monkeypatch.setattr(jax.ffi, 'ffi_call', ffi_call)
    kernel = binary_mod._make_hybrid_mm_kernel(**params)
    with pytest.raises(ValueError, match='long-row mask'):
        kernel(*args[:-1], jnp.asarray(mask))


def test_cpu_preparation_does_not_load_cuda(monkeypatch):
    monkeypatch.setattr(config_mod, '_query_cuda_device_info', lambda *args: pytest.fail('CPU preparation must not query CUDA'))
    cpu = jax.devices('cpu')[0]
    assert config_mod.get_device_info(device=cpu) is None
    assert config_mod.get_device_info(np.arange(3)) is None
    assert config_mod.get_device_info() is None


def test_device_query_does_not_load_mm_module(monkeypatch):
    if jax.default_backend() != 'gpu':
        pytest.skip('CUDA device query requires a GPU')
    gpu = jax.devices('gpu')[0]
    config_mod._query_cuda_device_info.cache_clear()
    monkeypatch.setattr(
        binary_mod, 'load_cuda_file',
        lambda *args, **kwargs: pytest.fail('device query must not load the MM module'),
    )
    info = config_mod.get_device_info(device=gpu)
    assert info is not None
    assert info.device_kind == str(gpu.device_kind)
    assert info.device_id == gpu.id
    assert info.sms > 0 and info.l2_bytes > 0


@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('permuted,homo', [(False, False), (False, True), (True, False)])
def test_prepared_adapter_cuda_repeated_calls(weighted, permuted, homo):
    if jax.default_backend() != 'gpu':
        pytest.skip('CUDA execution requires a GPU')
    lengths = np.array([0, 3, 65, 513], np.int32)
    offsets = np.r_[0, np.cumsum(lengths)].astype(np.int32)
    indices_np = np.arange(int(offsets[-1]), dtype=np.int32) % 7
    indices, indptr = jnp.asarray(indices_np), jnp.asarray(offsets)
    device_info = config_mod.get_device_info(indices, indptr)
    config = CSRConfig(device_info=device_info).prepare_structure(indices, indptr, shape=(4, 7))
    spec, mask = config.kernel_args()
    weights = jnp.ones((1 if homo else indices.size,), jnp.float32) / 8
    if not homo:
        weights *= jnp.arange(indices.size) % 3 - 1
    events = jnp.array([[0, 1, -2], [1, 0, 2], [-2, 1, 0], [1, 0, -1]], jnp.float32)
    perm = jnp.arange(indices.size - 1, -1, -1, dtype=jnp.int32)
    params = dict(
        config_spec=spec, call_plan=config.mm_plan(3, jnp.float32, weighted),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(events.shape, events.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        out_info=jax.ShapeDtypeStruct((7, 3), jnp.float32),
        weighted=weighted, permuted=permuted,
    )
    kernel = jax.jit(binary_mod._make_hybrid_mm_kernel(**params))
    edge_rows = np.repeat(np.arange(4), lengths)
    for active in (True, False, True):
        current = events if active else jnp.zeros_like(events)
        operands = (weights, indices, indptr) + ((perm,) if permuted else ()) + (current, mask)
        actual, = kernel(*operands)
        factors = np.asarray(current)[edge_rows]
        if not weighted:
            factors = factors != 0
        values = np.full(indices.size, float(weights[0])) if homo else np.asarray(weights)[np.asarray(perm) if permuted else np.arange(indices.size)]
        expected = np.zeros((7, 3), np.float32)
        np.add.at(expected, indices_np, factors * values[:, None])
        np.testing.assert_array_equal(actual, expected)


def test_cuda_rejects_plan_prepared_for_other_device():
    if jax.default_backend() != 'gpu':
        pytest.skip('CUDA execution requires a GPU')
    params, args = prepared()
    actual_device = config_mod.get_device_info(*args)
    wrong_device = replace(actual_device, sms=actual_device.sms + 1)
    params['config_spec'] = replace(params['config_spec'], device_info=wrong_device)
    kernel = jax.jit(binary_mod._make_hybrid_mm_kernel(**params))
    with pytest.raises(Exception, match='prepared CSRConfig device does not match'):
        jax.block_until_ready(kernel(*args))


@pytest.mark.parametrize('platform_version', ['ROCm 7.0', 'Metal', ''])
def test_non_cuda_gpu_preparation_does_not_query_cuda(monkeypatch, platform_version):
    from types import SimpleNamespace
    device = SimpleNamespace(platform='gpu', client=SimpleNamespace(platform_version=platform_version))
    monkeypatch.setattr(config_mod, '_query_cuda_device_info', lambda *args: pytest.fail('non-CUDA device query'))
    assert config_mod.get_device_info(device=device) is None
