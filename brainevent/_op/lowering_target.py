# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""The devices a lowering targets, and their CUDA properties.

Kernel generators that size launches from device properties read them here,
at lowering time, instead of storing them on matrices at construction time.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

import jax
import numpy as np

from brainevent._op.cuda_driver import device_attributes

__all__ = ['target_devices', 'cuda_l2_bytes', 'cuda_available', 'DEFAULT_L2_BYTES']

#: L2 size assumed when no target CUDA device can be queried. Smaller values
#: give smaller batch chunks and scratch, so this errs on the side of memory.
DEFAULT_L2_BYTES = 4 << 20

# CUdevice_attribute id of the L2 cache size, from cuda.h.
_CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38


def cuda_available() -> bool:
    """Whether this process has a CUDA backend; never raises.

    Returns
    -------
    bool
        ``True`` when JAX can list at least one CUDA device.
    """
    try:
        return bool(jax.devices('cuda'))
    except RuntimeError:
        return False


def target_devices(context: Any) -> tuple[Any, ...]:
    """Find the devices a lowering targets, conservatively covering unknown ones.

    Parameters
    ----------
    context : jax.interpreters.mlir.LoweringRuleContext
        Context of the lowering rule.

    Returns
    -------
    tuple of jax.Device
        The explicit device assignment or mesh devices when the lowering has
        one; otherwise every local device of the lowering backend, since a
        portable lowering may run on any of them. Empty when even the backend
        is unknown.
    """
    module = context.module_context
    axes = module.axis_context
    assignment = getattr(axes, 'device_assignment', None)
    if assignment is not None:
        return tuple(assignment)
    mesh = getattr(axes, 'mesh', None)
    devices = getattr(mesh, 'devices', None)
    if devices is not None:
        return tuple(np.asarray(devices, dtype=object).ravel())
    backend = module.backend
    return () if backend is None else tuple(backend.local_devices())


@lru_cache(maxsize=None)
def _device_l2_bytes(device: Any) -> int | None:
    """L2 bytes of one local CUDA device, or ``None`` when it cannot be read."""
    if getattr(device, 'platform', None) != 'gpu' or getattr(device, 'local_hardware_id', None) is None:
        return None
    version = str(getattr(getattr(device, 'client', None), 'platform_version', '')).lower()
    if 'cuda' not in version:
        return None
    try:
        (l2_bytes,) = device_attributes(device, (_CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,))
    except OSError:
        return None
    return l2_bytes if l2_bytes > 0 else None


def cuda_l2_bytes(devices: Sequence[Any]) -> int:
    """The smallest L2 size among the CUDA devices a lowering may run on.

    Parameters
    ----------
    devices : sequence of jax.Device
        Target devices, usually from :func:`target_devices`.

    Returns
    -------
    int
        The minimum L2 size in bytes over the devices whose size can be read
        through the CUDA driver (cached per device), or
        :data:`DEFAULT_L2_BYTES` when none can.
    """
    sizes = [size for size in map(_device_l2_bytes, devices) if size is not None]
    return min(sizes) if sizes else DEFAULT_L2_BYTES
