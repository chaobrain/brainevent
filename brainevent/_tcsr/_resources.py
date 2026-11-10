# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Compile-time launch capability checks for the actual CUDA target devices."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Any

import jax
import numpy as np
from jax._src import core
from jax.interpreters import ad, batching, mlir

from brainevent._compatible_import import Primitive, apply_primitive, init_zero


@dataclass(frozen=True)
class LaunchResources:
    """CUDA launch limits of one concrete device.

    Parameters
    ----------
    max_threads : int
        Maximum threads in a block.
    shared_bytes : int
        Shared-memory bytes available without opt-in.
    optin_shared_bytes : int
        Maximum shared-memory bytes available with opt-in.
    """

    max_threads: int
    shared_bytes: int
    optin_shared_bytes: int


@lru_cache(maxsize=1)
def _cuda_driver() -> Any:
    """Load the CUDA driver only when lowering a CUDA computation."""
    library = ctypes.CDLL('nvcuda.dll' if os.name == 'nt' else 'libcuda.so.1')
    library.cuInit.argtypes = [ctypes.c_uint]
    library.cuInit.restype = ctypes.c_int
    library.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    library.cuDeviceGet.restype = ctypes.c_int
    library.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    library.cuDeviceGetAttribute.restype = ctypes.c_int
    if library.cuInit(0):
        raise OSError('CUDA driver initialization failed.')
    return library


@lru_cache(maxsize=None)
def _query_launch_resources(device: Any) -> LaunchResources | None:
    """Cache immutable launch attributes using the device's local ordinal."""
    ordinal = getattr(device, 'local_hardware_id', None)
    if ordinal is None or getattr(device, 'platform', None) != 'gpu':
        return None
    try:
        driver = _cuda_driver()
        cuda_device = ctypes.c_int()
        if driver.cuDeviceGet(ctypes.byref(cuda_device), int(ordinal)):
            return None
        result = []
        # CUdevice_attribute values from the installed CUDA driver API:
        # MAX_THREADS_PER_BLOCK, MAX_SHARED_MEMORY_PER_BLOCK, and ..._OPTIN.
        for attribute in (1, 8, 97):
            value = ctypes.c_int()
            if driver.cuDeviceGetAttribute(ctypes.byref(value), attribute, cuda_device.value):
                return None
            result.append(value.value)
        return LaunchResources(*result)
    except (OSError, AttributeError):
        # A missing/unsupported driver probe is not evidence that a tile fits.
        return None


def _supports(resources: LaunchResources | None, threads: int, shared_bytes: int) -> bool:
    """Match the existing native launcher's actual allocation policy."""
    if resources is None or threads > resources.max_threads:
        return False
    if shared_bytes <= 48 * 1024:
        return shared_bytes <= resources.shared_bytes
    # The launcher requests the full 96 KiB ceiling even for a smaller tile.
    return resources.optin_shared_bytes >= 96 * 1024


def _target_devices(context: Any) -> tuple[Any, ...]:
    """Find the lowering assignment, conservatively covering unknown targets."""
    module = context.module_context
    axes = module.axis_context
    assignment = getattr(axes, 'device_assignment', None)
    if assignment is not None:
        return tuple(assignment)
    mesh = getattr(axes, 'mesh', None)
    devices = getattr(mesh, 'devices', None)
    if devices is not None:
        return tuple(np.asarray(devices, dtype=object).ravel())
    # A portable lowering may have no concrete assignment. Restrict the tile
    # to capabilities common to every possible local target of this backend.
    backend = module.backend
    return () if backend is None else tuple(backend.local_devices())


def _lower_false(context, *, threads, shared_bytes):
    """CPU, TPU and unknown platforms never query the CUDA driver."""
    return [mlir.ir_constant(np.bool_(False))]


def _lower_cuda(context, *, threads, shared_bytes):
    """Fold the actual CUDA target's cached limits to a boolean constant."""
    devices = _target_devices(context)
    resources = tuple(_query_launch_resources(device) for device in devices)
    supported = bool(devices) and all(
        _supports(resource, threads, shared_bytes) for resource in resources
    )
    return [mlir.ir_constant(np.bool_(supported))]


_launch_supported_p = Primitive('tcsr_launch_supported')
_launch_supported_p.def_impl(partial(apply_primitive, _launch_supported_p))
_launch_supported_p.def_abstract_eval(lambda **kwargs: core.ShapedArray((), np.dtype('bool')))
mlir.register_lowering(_launch_supported_p, _lower_false)
mlir.register_lowering(_launch_supported_p, _lower_cuda, platform='cuda')


def _jvp(primals, tangents, **kwargs):
    result = _launch_supported_p.bind(**kwargs)
    return result, init_zero(result)


ad.primitive_jvps[_launch_supported_p] = _jvp
batching.primitive_batchers[_launch_supported_p] = (
    lambda args, axes, **kwargs: (_launch_supported_p.bind(**kwargs), None)
)


def launch_supported(*, threads: int, shared_bytes: int) -> jax.Array:
    """Return whether the actual compilation target can launch this tile.

    Parameters
    ----------
    threads : int
        Planned threads per scatter block.
    shared_bytes : int
        Planned accumulation storage in bytes. Above 48 KiB the native
        launcher requests 96 KiB opt-in, and this check uses that ceiling.

    Returns
    -------
    jax.Array
        Scalar boolean, lowered to a constant for the target assignment.
        Unknown or unsupported resource information selects the CSR path.

    Notes
    -----
    This primitive is staged inside the dispatch program. It does not use
    the Python default backend or run a per-invocation resource probe.
    With no explicit assignment, every candidate local device must fit.
    """
    return _launch_supported_p.bind(threads=int(threads), shared_bytes=int(shared_bytes))
