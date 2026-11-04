"""Prepared hybrid SpMM: device discovery and the common CUDA adapter."""

import hashlib
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from brainevent._compatible_import import match_varying_axes
from brainevent._op import load_cuda_file
from brainevent._op.util import cuda_event_abi, dtype_suffix

_ROOT = Path(__file__).parent / 'hybrid_mm'


def load_module(*, weighted: bool = False, permuted: bool = False) -> str:
    """Load a hybrid MM source and return its FFI module prefix.

    Parameters
    ----------
    weighted : bool, optional
        Load event-amplitude multiplication instead of binary activation.
    permuted : bool, optional
        Load the hetero mirror entry points reading ``weights[perm[j]]``.
        Shared weights use the ordinary module, even with mirror structure.

    Returns
    -------
    str
        Registered module prefix for the source's ``batched_spmm_v3_*``
        functions. Loading does not prepare a structure or a call plan.
    """
    mode = 'weighted' if weighted else 'binary'
    layout = 'col_major' if permuted else 'row'
    source = _ROOT / f'hybrid_batchedv3_spmm_{"col_major_" if permuted else ""}{mode}.cu'
    # Include nested common headers in both the target identity and the build
    # cache. A header-only change must not re-use a registered older target.
    dependencies = (source, *sorted(_ROOT.rglob('*.cuh')))
    digest = hashlib.sha256(b''.join(path.read_bytes() for path in dependencies)).hexdigest()[:12]
    module_name = f'csr_hybrid_mm_{layout}_{mode}_{digest}'
    load_cuda_file(
        source, name=module_name, extra_include_paths=[str(_ROOT)], allow_cuda_graph=False,
    )
    return module_name


@lru_cache(maxsize=None)
def _query_device(device):
    from brainevent._csr.config import CSRDeviceInfo

    module = load_module()
    dummy = jax.device_put(np.zeros((1,), np.uint8), device)
    info = jax.ffi.ffi_call(
        module + '.batched_spmm_v3_device_info',
        jax.ShapeDtypeStruct((2,), jnp.int32),
    )(dummy)
    sms, l2_bytes = map(int, np.asarray(jax.device_get(info)))
    return CSRDeviceInfo(
        platform='gpu', device_kind=str(device.device_kind),
        l2_bytes=l2_bytes, sms=sms, device_id=device.id,
    )


def get_device_info(*arrays, device=None):
    """Read and cache the preparation device's CUDA launch properties.

    Parameters
    ----------
    *arrays : array_like
        Concrete arrays whose placement identifies the preparation device.
    device : jax.Device or None, optional
        Explicit preparation device. Otherwise use the arrays' single device,
        leaving host arrays and tracers without device preparation.

    Returns
    -------
    CSRDeviceInfo or None
        The selected CUDA device's SM count and L2 size, or ``None`` without
        an unambiguous concrete GPU placement.

    Raises
    ------
    ValueError
        The arrays span multiple devices and no explicit device was supplied.
    """
    if device is None:
        devices = set()
        for array in arrays:
            if isinstance(array, jax.core.Tracer):
                continue
            if hasattr(array, 'devices'):
                devices.update(array.devices())
        if len(devices) > 1:
            raise ValueError('CSR preparation requires a single device; pass device explicitly.')
        if not devices:
            return None
        device = next(iter(devices))
    if device.platform != 'gpu':
        return None
    platform_version = str(getattr(getattr(device, 'client', None), 'platform_version', '')).lower()
    if 'cuda' not in platform_version:
        return None
    return _query_device(device)


def make_kernel(*, config_spec, call_plan, weight_info, vector_info,
                indices_info, indptr_info, outs=None, out_info=None,
                weighted=False, permuted=False, perm_info=None, **kwargs):
    """Lower a prepared hybrid SpMM call to its native BN layout.

    Parameters
    ----------
    config_spec : CSRKernelSpec
        Immutable scheduler, structure and preparation-device metadata.
    call_plan : MMCallPlan or None
        Purely derived launch and workspace dimensions for this invocation.
    weight_info, vector_info, indices_info, indptr_info : jax.ShapeDtypeStruct
        Mathematical operand shapes and dtypes.
    outs : sequence of jax.ShapeDtypeStruct or None, optional
        Primitive output metadata, used when ``out_info`` is omitted.
    out_info : jax.ShapeDtypeStruct or None, optional
        Logical NB output shape and dtype.
    weighted : bool, optional
        Multiply weights by active event amplitudes.
    permuted : bool, optional
        The returned callable receives an additional weight permutation.
    perm_info : jax.ShapeDtypeStruct or None, optional
        Permutation operand metadata for indexed callers.
    **kwargs
        Additional primitive metadata ignored by this adapter.

    Returns
    -------
    callable
        A kernel receiving math operands and a final prepared long-row mask.

    Raises
    ------
    ValueError
        A nonempty hybrid call lacks matching structure or device preparation.
    """
    out_info = out_info if out_info is not None else outs[0]
    batch, rows, cols = vector_info.shape[1], indptr_info.size - 1, out_info.shape[0]
    nnz = indices_info.size
    if not batch or not rows or not cols or not nnz:
        return lambda *args: (match_varying_axes(jnp.zeros(out_info.shape, out_info.dtype), *args),)
    if np.dtype(weight_info.dtype).name not in ('float16', 'bfloat16', 'float32', 'float64'):
        raise ValueError('Hybrid CSR MM supports only float16, bfloat16, float32, or float64 values.')
    device = None if config_spec is None else config_spec.device_info
    if (call_plan is None or device is None or device.platform != 'gpu'
            or device.sms <= 0 or device.l2_bytes <= 0):
        raise ValueError(
            'Hybrid CSR MM requires a complete prepared CSRConfig with structure and GPU device '
            'information. Prepare the config before jax.jit and pass it explicitly.'
        )
    if (call_plan.rows, call_plan.cols, call_plan.nnz, call_plan.batch) != (rows, cols, nnz, batch):
        raise ValueError('Hybrid CSR MM plan does not match the operand structure and batch shape.')
    if np.dtype(call_plan.dtype) != np.dtype(weight_info.dtype):
        raise ValueError('Hybrid CSR MM plan does not match the value dtype.')
    from brainevent._csr.config import CSRConfig
    config = CSRConfig.from_kernel_args(config_spec, None)
    expected_plan = config.mm_plan(batch, weight_info.dtype, weighted)
    if call_plan != expected_plan:
        raise ValueError('Hybrid CSR MM call plan does not match the prepared config and invocation.')
    native_permuted = permuted and weight_info.size != 1
    module = load_module(weighted=weighted, permuted=native_permuted)
    abi = cuda_event_abi(vector_info.dtype, weighted=weighted, value_dtype=weight_info.dtype)
    if native_permuted:
        symbol = 'batched_spmm_v3_col_major_' + ('weighted' if weighted else 'binary') + '_hetero'
    else:
        symbol = 'batched_spmm_v3' + ('_weighted' if weighted else '')
        symbol += '_homo' if weight_info.size == 1 else ''
    symbol += dtype_suffix(weight_info.dtype)
    if not weighted:
        symbol += abi.suffix
    infos = (
        jax.ShapeDtypeStruct((batch, cols), out_info.dtype),
        jax.ShapeDtypeStruct((call_plan.workspace_words,), jnp.uint32),
        jax.ShapeDtypeStruct((call_plan.short_rows_size,), jnp.int32),
    )
    attrs = dict(
        group=call_plan.group, long_threshold=call_plan.long_threshold,
        chunk=call_plan.chunk, long_tasks=call_plan.long_tasks,
        task_capacity=call_plan.task_capacity,
        expected_sms=device.sms, expected_l2_bytes=device.l2_bytes,
    )
    ffi = jax.ffi.ffi_call(module + '.' + symbol, infos, vmap_method='sequential')

    def kernel(*args):
        if permuted:
            weights, indices, indptr, perm, events, long_mask = args
        else:
            weights, indices, indptr, events, long_mask = args
            perm = None
        CSRConfig.from_kernel_args(config_spec, long_mask).require_prepared(rows=rows, cols=cols, nnz=nnz)
        operands = (weights, indices, indptr)
        if native_permuted:
            operands += (perm,)
        output = ffi(*operands, long_mask, abi.prepare(events).T, **attrs)[0]
        return (match_varying_axes(output.T, *args),)

    return kernel
