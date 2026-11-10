# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Actual-target resource limits and the native shared-memory opt-in policy."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest

from brainevent._tcsr import _resources as api


@pytest.fixture(autouse=True)
def _isolate_resource_caches():
    api._query_launch_resources.cache_clear()
    api._cuda_driver.cache_clear()
    yield
    api._query_launch_resources.cache_clear()
    api._cuda_driver.cache_clear()
    jax.clear_caches()


@pytest.mark.parametrize('resources,threads,shared,expected', [
    (None, 256, 4, False),
    (api.LaunchResources(128, 49152, 98304), 256, 4, False),
    (api.LaunchResources(1024, 49152, 0), 1024, 49152, True),
    (api.LaunchResources(1024, 16384, 98304), 256, 32768, False),
    (api.LaunchResources(1024, 49152, 65536), 512, 65536, False),
    (api.LaunchResources(1024, 49152, 98304), 512, 65536, True),
])
def test_limits_match_actual_native_allocation(resources, threads, shared, expected):
    assert api._supports(resources, threads, shared) is expected


def test_target_assignment_and_mesh_precede_backend_candidates():
    context = SimpleNamespace(module_context=SimpleNamespace(
        axis_context=SimpleNamespace(device_assignment=('actual',)), backend=None,
    ))
    assert api._target_devices(context) == ('actual',)
    context.module_context.axis_context = SimpleNamespace(mesh=SimpleNamespace(devices=np.array([['mesh0', 'mesh1']])))
    assert api._target_devices(context) == ('mesh0', 'mesh1')
    context.module_context.axis_context = SimpleNamespace()
    context.module_context.backend = SimpleNamespace(local_devices=lambda: ['candidate0', 'candidate1'])
    assert api._target_devices(context) == ('candidate0', 'candidate1')
    context.module_context.backend = None
    assert api._target_devices(context) == ()


def test_unknown_or_cpu_device_does_not_load_driver(monkeypatch):
    def forbidden():
        raise AssertionError('non-CUDA target loaded the driver')
    monkeypatch.setattr(api, '_cuda_driver', forbidden)
    assert api._query_launch_resources('no ordinal') is None
    class Device:
        local_hardware_id = 0
        platform = 'cpu'
    assert api._query_launch_resources(Device()) is None


@pytest.mark.parametrize('failure', ['device', 'attribute', 'loader', None])
def test_driver_queries_use_actual_local_ordinal_and_cache(monkeypatch, failure):
    calls = []
    class Device:
        local_hardware_id = 7
        platform = 'gpu'

    class Driver:
        def cuDeviceGet(self, output, ordinal):
            calls.append(('device', ordinal))
            output._obj.value = 123
            return int(failure == 'device')
        def cuDeviceGetAttribute(self, output, attribute, device):
            calls.append(('attribute', attribute, device))
            output._obj.value = {1: 1024, 8: 49152, 97: 98304}[attribute]
            return int(failure == 'attribute')

    def loader():
        if failure == 'loader':
            raise OSError('driver unavailable')
        return Driver()
    monkeypatch.setattr(api, '_cuda_driver', loader)
    device = Device()
    result = api._query_launch_resources(device)
    assert result == (api.LaunchResources(1024, 49152, 98304) if failure is None else None)
    assert api._query_launch_resources(device) is result
    assert sum(c[0] == 'device' for c in calls) <= 1
    if failure is None:
        assert calls == [('device', 7), ('attribute', 1, 123), ('attribute', 8, 123), ('attribute', 97, 123)]


@pytest.mark.parametrize('status', [0, 1])
def test_driver_loading_and_initialization(monkeypatch, status):
    class Call:
        def __call__(self, flags):
            assert flags == 0
            return status
    library = SimpleNamespace(cuInit=Call(), cuDeviceGet=Call(), cuDeviceGetAttribute=Call())
    monkeypatch.setattr(api.ctypes, 'CDLL', lambda name: library)
    if status:
        with pytest.raises(OSError, match='initialization'):
            api._cuda_driver()
    else:
        assert api._cuda_driver() is library
        assert library.cuDeviceGet.restype is api.ctypes.c_int


def test_cpu_lowering_never_queries_cuda(monkeypatch):
    def forbidden(device):
        raise AssertionError('CPU lowering queried CUDA capabilities')
    monkeypatch.setattr(api, '_query_launch_resources', forbidden)
    result = jax.jit(lambda: api.launch_supported(threads=256, shared_bytes=1024), backend='cpu')()
    assert not bool(result)
    assert next(iter(result.devices())).platform == 'cpu'


def test_cuda_lowering_uses_target_device_and_folds_resources(monkeypatch):
    try:
        gpu = jax.devices('gpu')[-1]
    except RuntimeError:
        pytest.skip('CUDA device required')
    if 'cuda' not in gpu.client.platform_version.lower():
        pytest.skip('CUDA device required')
    queried = []
    def query(device):
        queried.append(device)
        return api.LaunchResources(1024, 49152, 65536)
    monkeypatch.setattr(api, '_query_launch_resources', query)
    run = jax.jit(lambda: api.launch_supported(threads=512, shared_bytes=65536), device=gpu)
    assert not bool(run())
    assert queried and gpu in queried
    # Ordinary JIT lowering currently omits its device assignment even when
    # jit(device=...) was explicit. Such portable IR must fit all candidates.
    assert set(queried) in ({gpu}, set(gpu.client.local_devices()))


def test_cuda_lowering_with_assignment_checks_only_that_target(monkeypatch):
    context = SimpleNamespace(module_context=SimpleNamespace(
        axis_context=SimpleNamespace(device_assignment=('actual3',)),
        backend=SimpleNamespace(local_devices=lambda: ['default0']),
    ))
    queried = []
    def query(device):
        queried.append(device)
        return api.LaunchResources(1024, 49152, 98304)
    monkeypatch.setattr(api, '_query_launch_resources', query)
    monkeypatch.setattr(api.mlir, 'ir_constant', lambda value: bool(value))
    assert api._lower_cuda(context, threads=512, shared_bytes=65536) == [True]
    assert queried == ['actual3']
