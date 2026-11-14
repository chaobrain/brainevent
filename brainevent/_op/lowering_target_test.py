# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._op import XLACustomKernel
from brainevent._op import lowering_target as api
from brainevent._test_util import requires_gpu


@pytest.fixture(autouse=True)
def _isolate_l2_cache():
    api._device_l2_bytes.cache_clear()
    yield
    api._device_l2_bytes.cache_clear()


def test_target_assignment_and_mesh_precede_backend_candidates():
    context = SimpleNamespace(module_context=SimpleNamespace(
        axis_context=SimpleNamespace(device_assignment=('actual',)), backend=None,
    ))
    assert api.target_devices(context) == ('actual',)
    context.module_context.axis_context = SimpleNamespace(mesh=SimpleNamespace(devices=np.array([['mesh0', 'mesh1']])))
    assert api.target_devices(context) == ('mesh0', 'mesh1')
    context.module_context.axis_context = SimpleNamespace()
    context.module_context.backend = SimpleNamespace(local_devices=lambda: ['candidate0', 'candidate1'])
    assert api.target_devices(context) == ('candidate0', 'candidate1')
    context.module_context.backend = None
    assert api.target_devices(context) == ()


class _Device:
    """Hashable stand-in for a ``jax.Device``."""

    def __init__(self, platform, ordinal, version=''):
        self.platform = platform
        self.local_hardware_id = ordinal
        self.client = SimpleNamespace(platform_version=version)


def _cuda_device(ordinal):
    return _Device('gpu', ordinal, 'cuda 12080')


def test_l2_is_the_smallest_readable_target_and_is_cached(monkeypatch):
    sizes = {0: 6 << 20, 1: 40 << 20}
    calls = []

    def query(device, attributes):
        calls.append(device.local_hardware_id)
        assert attributes == (38,)
        if device.local_hardware_id not in sizes:
            raise OSError('driver unavailable')
        return (sizes[device.local_hardware_id],)

    monkeypatch.setattr(api, 'device_attributes', query)
    devices = [_cuda_device(0), _cuda_device(1), _cuda_device(2)]
    assert api.cuda_l2_bytes(devices) == 6 << 20
    assert api.cuda_l2_bytes(devices) == 6 << 20
    assert sorted(calls) == [0, 1, 2]


def test_l2_defaults_without_a_readable_cuda_target(monkeypatch):
    def forbidden(device, attributes):
        raise AssertionError('non-CUDA device queried the driver')

    monkeypatch.setattr(api, 'device_attributes', forbidden)
    rocm = _Device('gpu', 0, 'rocm 6.0')
    cpu = _Device('cpu', 0)
    assert api.cuda_l2_bytes([]) == api.DEFAULT_L2_BYTES
    assert api.cuda_l2_bytes([cpu, rocm, object()]) == api.DEFAULT_L2_BYTES


def _identity_primitive(name, generator):
    primitive = XLACustomKernel(name)
    primitive.def_kernel('jax_raw', 'cpu', generator)
    primitive.def_kernel('jax_raw', 'gpu', generator)
    return primitive


def test_only_generators_declaring_target_devices_receive_them():
    received = {}

    def with_devices(*, outs, target_devices):
        received['with'] = target_devices
        return lambda x: (x,)

    def without_devices(**kwargs):
        received['without'] = kwargs
        return lambda x: (x,)

    x = jnp.arange(3.0)
    for name, generator in (('lowering_target_with', with_devices), ('lowering_target_without', without_devices)):
        primitive = _identity_primitive(name, generator)
        out = jax.jit(lambda x: primitive(x, outs=[jax.ShapeDtypeStruct(x.shape, x.dtype)])[0])(x)
        np.testing.assert_array_equal(out, x)
    assert received['with'] and all(isinstance(d, jax.Device) for d in received['with'])
    assert 'target_devices' not in received['without']


@requires_gpu
def test_real_cuda_target_reports_its_l2_size():
    gpu = jax.devices('gpu')[0]
    if 'cuda' not in gpu.client.platform_version.lower():
        pytest.skip('CUDA device required')
    assert api.cuda_l2_bytes([gpu]) > 0
