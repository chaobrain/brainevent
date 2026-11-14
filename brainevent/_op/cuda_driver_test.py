# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

import types

import jax
import pytest

from brainevent._op import cuda_driver
from brainevent._op.cuda_driver import device_attributes


def test_device_without_ordinal_raises():
    with pytest.raises(OSError, match='no local CUDA ordinal'):
        device_attributes(types.SimpleNamespace(), (16,))


def test_missing_driver_raises(monkeypatch):
    monkeypatch.setattr(cuda_driver, '_LIBRARIES', ('libbrainevent-missing-cuda.so',))
    cuda_driver._driver.cache_clear()
    try:
        with pytest.raises(OSError, match='CUDA driver library not found'):
            device_attributes(types.SimpleNamespace(local_hardware_id=0), (16,))
    finally:
        cuda_driver._driver.cache_clear()


def test_gpu_attributes_match_jax():
    gpus = [d for d in jax.devices() if d.platform == 'gpu']
    if not gpus:
        pytest.skip('requires a CUDA device')
    sms, l2_bytes, major = device_attributes(gpus[0], (16, 38, 75))
    assert sms == gpus[0].core_count
    assert l2_bytes > 0 and major >= 3
