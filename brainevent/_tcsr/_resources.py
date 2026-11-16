# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Compile-time launch capability checks for the actual CUDA target devices."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Any

import jax
import numpy as np
from jax._src import core
from jax.interpreters import ad, batching, mlir

from brainevent._compatible_import import Primitive, apply_primitive, init_zero
from brainevent._op.cuda_driver import device_attributes
from brainevent._op.lowering_target import target_devices as _target_devices


#: Oldest compute capability the combined kernels support. f64 tiles use the
#: native double ``atomicAdd`` (sm_60+); sm_80-only warp reductions have a
#: shuffle fallback, so nothing newer is required.
MIN_COMPUTE_CAPABILITY = (6, 0)


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
    compute_capability : tuple of int
        ``(major, minor)`` compute capability of the device.
    """

    max_threads: int
    shared_bytes: int
    optin_shared_bytes: int
    compute_capability: tuple[int, int]


# CUdevice_attribute ids: MAX_THREADS_PER_BLOCK, MAX_SHARED_MEMORY_PER_BLOCK,
# MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, COMPUTE_CAPABILITY_MAJOR / _MINOR.
_LAUNCH_ATTRIBUTES = (1, 8, 97, 75, 76)


@lru_cache(maxsize=None)
def _query_launch_resources(device: Any) -> LaunchResources | None:
    """Cache immutable launch attributes using the device's local ordinal."""
    if getattr(device, 'local_hardware_id', None) is None or getattr(device, 'platform', None) != 'gpu':
        return None
    try:
        max_threads, shared_bytes, optin_shared_bytes, major, minor = device_attributes(
            device, _LAUNCH_ATTRIBUTES)
    except OSError:
        # A missing/unsupported driver probe is not evidence that a tile fits.
        return None
    return LaunchResources(max_threads, shared_bytes, optin_shared_bytes, (major, minor))


def _supports(resources: LaunchResources | None, threads: int, shared_bytes: int) -> bool:
    """Match the native launcher's allocation policy and architecture floor."""
    if resources is None or threads > resources.max_threads:
        return False
    if tuple(resources.compute_capability) < MIN_COMPUTE_CAPABILITY:
        return False
    if shared_bytes <= 48 * 1024:
        return shared_bytes <= resources.shared_bytes
    # The launcher requests the full 96 KiB ceiling even for a smaller tile.
    return resources.optin_shared_bytes >= 96 * 1024


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


def static_launch_supported(*, threads: int, shared_bytes: int) -> bool | None:
    """Decide a launch at trace time when every possible local target agrees.

    Parameters
    ----------
    threads : int
        Planned threads per scatter block.
    shared_bytes : int
        Planned accumulation storage in bytes.

    Returns
    -------
    bool or None
        ``True`` or ``False`` when this single process sees CUDA devices
        that all give the same answer, so only the selected branch needs to
        be traced and lowered. ``None`` when the target is unknown at trace
        time (multiple processes, no local CUDA device, or devices that
        disagree); use :func:`launch_supported` to decide at lowering time.
    """
    if jax.process_count() != 1:
        return None
    try:
        devices = jax.devices('cuda')
    except RuntimeError:
        return None
    verdicts = {_supports(_query_launch_resources(device), threads, shared_bytes) for device in devices}
    return verdicts.pop() if len(verdicts) == 1 else None


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
