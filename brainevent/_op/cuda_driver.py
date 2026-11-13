# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""CUDA driver API device queries through ``ctypes`` (no compiler toolchain)."""

from __future__ import annotations

import ctypes
from functools import lru_cache
from typing import Any, Sequence

__all__ = ['device_attributes']

_LIBRARIES = ('libcuda.so.1', 'libcuda.so', 'nvcuda.dll')


@lru_cache(maxsize=1)
def _driver() -> Any:
    """Load and initialize the CUDA driver library (cached after success)."""
    errors = []
    for name in _LIBRARIES:
        try:
            library = ctypes.CDLL(name)
        except OSError as exc:
            errors.append(str(exc))
            continue
        library.cuInit.argtypes = [ctypes.c_uint]
        library.cuInit.restype = ctypes.c_int
        library.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        library.cuDeviceGet.restype = ctypes.c_int
        library.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        library.cuDeviceGetAttribute.restype = ctypes.c_int
        status = library.cuInit(0)
        if status:
            raise OSError(f'cuInit failed with CUDA driver error {status}')
        return library
    raise OSError('CUDA driver library not found: ' + '; '.join(errors))


def device_attributes(device: Any, attributes: Sequence[int]) -> tuple[int, ...]:
    """Read ``CUdevice_attribute`` values of a JAX CUDA device.

    Parameters
    ----------
    device : jax.Device
        CUDA device; its ``local_hardware_id`` is the driver ordinal.
    attributes : sequence of int
        ``CUdevice_attribute`` ids from ``cuda.h``.

    Returns
    -------
    tuple of int
        One value per requested attribute, in order.

    Raises
    ------
    OSError
        The device has no local ordinal, or the driver is missing or fails.
    """
    ordinal = getattr(device, 'local_hardware_id', None)
    if ordinal is None:
        raise OSError(f'{device} has no local CUDA ordinal')
    driver = _driver()
    handle = ctypes.c_int()
    status = driver.cuDeviceGet(ctypes.byref(handle), int(ordinal))
    if status:
        raise OSError(f'cuDeviceGet({ordinal}) failed with CUDA driver error {status}')
    values = []
    for attribute in attributes:
        value = ctypes.c_int()
        status = driver.cuDeviceGetAttribute(ctypes.byref(value), int(attribute), handle.value)
        if status:
            raise OSError(f'cuDeviceGetAttribute({attribute}) failed with CUDA driver error {status}')
        values.append(int(value.value))
    return tuple(values)
