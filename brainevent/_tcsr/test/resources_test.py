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
    yield
    api._query_launch_resources.cache_clear()
    jax.clear_caches()


SM86 = (8, 6)


@pytest.mark.parametrize('resources,threads,shared,expected', [
    (None, 256, 4, False),
    (api.LaunchResources(128, 49152, 98304, SM86), 256, 4, False),
    (api.LaunchResources(1024, 49152, 0, SM86), 1024, 49152, True),
    (api.LaunchResources(1024, 16384, 98304, SM86), 256, 32768, False),
    (api.LaunchResources(1024, 49152, 65536, SM86), 512, 65536, False),
    (api.LaunchResources(1024, 49152, 98304, SM86), 512, 65536, True),
    # The architecture floor: f64 tiles need the native double atomicAdd.
    (api.LaunchResources(1024, 49152, 49152, (5, 2)), 256, 4, False),
    (api.LaunchResources(1024, 49152, 49152, (6, 0)), 256, 4, True),
    # Turing: pre-sm_80 (shuffle fallback) and a 64 KiB opt-in ceiling.
    (api.LaunchResources(1024, 49152, 65536, (7, 5)), 256, 32768, True),
    (api.LaunchResources(1024, 49152, 65536, (7, 5)), 512, 65536, False),
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
    def forbidden(device, attributes):
        raise AssertionError('non-CUDA target loaded the driver')
    monkeypatch.setattr(api, 'device_attributes', forbidden)
    assert api._query_launch_resources('no ordinal') is None
    class Device:
        local_hardware_id = 0
        platform = 'cpu'
    assert api._query_launch_resources(Device()) is None


@pytest.mark.parametrize('failure', [True, False])
def test_driver_queries_are_cached_and_failures_mean_unsupported(monkeypatch, failure):
    calls = []
    class Device:
        local_hardware_id = 7
        platform = 'gpu'

    def query(device, attributes):
        calls.append(tuple(attributes))
        if failure:
            raise OSError('driver unavailable')
        return tuple({1: 1024, 8: 49152, 97: 98304, 75: 8, 76: 6}[a] for a in attributes)
    monkeypatch.setattr(api, 'device_attributes', query)
    device = Device()
    result = api._query_launch_resources(device)
    assert result == (None if failure else api.LaunchResources(1024, 49152, 98304, SM86))
    assert api._query_launch_resources(device) is result
    assert calls == [(1, 8, 97, 75, 76)]


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
        return api.LaunchResources(1024, 49152, 65536, SM86)
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
        return api.LaunchResources(1024, 49152, 98304, SM86)
    monkeypatch.setattr(api, '_query_launch_resources', query)
    monkeypatch.setattr(api.mlir, 'ir_constant', lambda value: bool(value))
    assert api._lower_cuda(context, threads=512, shared_bytes=65536) == [True]
    assert queried == ['actual3']


def test_static_decision_requires_one_process_and_agreeing_local_cuda_devices(monkeypatch):
    fits = api.LaunchResources(1024, 49152, 98304, SM86)
    small = api.LaunchResources(1024, 49152, 65536, SM86)
    table = {}
    monkeypatch.setattr(api, '_query_launch_resources', lambda device: table[device])

    def devices(kind):
        assert kind == 'cuda'
        if not table:
            raise RuntimeError('no CUDA backend')
        return list(table)

    monkeypatch.setattr(api.jax, 'devices', devices)
    decide = lambda: api.static_launch_supported(threads=512, shared_bytes=65536)
    assert decide() is None
    table.update(gpu0=fits, gpu1=fits)
    assert decide() is True
    table.update(gpu0=small, gpu1=small)
    assert decide() is False
    table.update(gpu1=fits)
    assert decide() is None
    table.update(gpu0=fits)
    monkeypatch.setattr(api.jax, 'process_count', lambda: 2)
    assert decide() is None
