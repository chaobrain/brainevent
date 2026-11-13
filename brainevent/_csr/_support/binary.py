# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Shared helpers for direct and indexed CSR binary event primitives."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from brainevent._csr.config import CSRConfig, _bucket
from brainevent._misc import _static_integer
from brainevent._op import load_cuda_file
from brainevent._sddmm import csr_sddmm_dweight
from brainevent._sddmm import dispatch as _sddmm_dispatch

_HYBRID_MV_DIR = Path(__file__).parents[1] / 'binary' / 'hybrid_mv'


def _task_capacity(indices, indptr, max_row_nnz, config=None) -> int:
    """Hybrid task-queue capacity bound from static shapes and a row-length hint."""
    config = CSRConfig() if config is None else config
    if config.primary_max_row_nnz is not None:
        max_row_nnz = config.primary_max_row_nnz
    return config.mv_capacity(indices.shape[0], indptr.shape[0] - 1, max_row_nnz)


def _ignore_config_mask(generator):
    """Adapt a backend that does not consume the final structural mask operand."""
    def generate(**kwargs):
        kernel = generator(**kwargs)
        return lambda *args: kernel(*args[:-1])
    return generate


def _config_from_params(params, mask):
    """Restore the selected configuration without reading arrays or persistence."""
    spec = params.get('config_spec')
    return CSRConfig() if spec is None else CSRConfig.from_kernel_args(spec, mask)


def _batch_config_params(args, axes, params):
    """Invalidate a shared exact plan when indptr or its mask is mapped."""
    if axes[2] is None and axes[-1] is None:
        return params
    config = _config_from_params(params, args[-1]).parameter_snapshot()
    spec, _ = config.kernel_args()
    result = dict(params, config_spec=spec)
    if 'max_row_nnz' in params:
        result['max_row_nnz'] = None
    if 'task_capacity' in params:
        result['task_capacity'] = config.mv_capacity(params['indices_info'].size,
                                                     params['indptr_info'].size - 1)
    return result


def _accumulator_info(out_info):
    """Output metadata of a native hybrid scatter kernel for ``out_info``.

    The hybrid MV/MM kernels add every edge into a float32 output (float64 for
    float64 values), so half-precision sums are rounded once by the caller
    instead of once per edge.

    Parameters
    ----------
    out_info : jax.ShapeDtypeStruct
        Logical primitive output.

    Returns
    -------
    jax.ShapeDtypeStruct
        ``out_info`` with its accumulator dtype.
    """
    dtype = jnp.float64 if np.dtype(out_info.dtype).itemsize == 8 else jnp.float32
    return jax.ShapeDtypeStruct(out_info.shape, dtype)


def _hybrid_scratch(task_capacity: int, indptr_dtype):
    """Scratch operands and result types of the hybrid CUDA kernels.

    The kernels zero ``status`` and write ``task_begin``/``task_end`` before
    reading them, so the scratch is created inside the lowering for every call
    and never lives on a matrix.
    """
    infos = (
        jax.ShapeDtypeStruct((task_capacity,), indptr_dtype),
        jax.ShapeDtypeStruct((task_capacity,), indptr_dtype),
        jax.ShapeDtypeStruct((2,), jnp.int32),
    )
    return tuple(jnp.empty(info.shape, info.dtype) for info in infos), infos


def _check_max_row_nnz(max_row_nnz, config=None) -> None:
    """Validate a scalar hint without inspecting compressed arrays."""
    if max_row_nnz is None:
        return
    max_row_nnz = _static_integer(max_row_nnz, 'max_row_nnz')
    # Prepared configs store a power-of-two bucket of the longest row; a hint
    # whose bucket is smaller is certainly below the longest row.
    if (config is not None and config.primary_max_row_nnz is not None and config.nnz is not None
            and _bucket(max_row_nnz, config.nnz) < config.primary_max_row_nnz):
        raise ValueError('max_row_nnz is below the longest row in the prepared CSRConfig')


def _config_for_call(config, indices, indptr, shape):
    """Validate a supplied structural descriptor using only static dimensions."""
    config = CSRConfig() if config is None else config
    if not isinstance(config, CSRConfig):
        raise TypeError('config must be a CSRConfig or None')
    if config.primary is not None and (
        config.shape != tuple(shape)
        or config.nnz != indices.size or config.shape[0] + 1 != indptr.size
    ):
        raise ValueError('CSRConfig does not match the current CSR dimensions; prepare the config for this structure.')
    return config


def _hybrid_mv_kernel(kwargs, symbol: str, event_abi, *, permuted: bool):
    """Build the native hybrid scatter MV kernel of a transposed CSR event product.

    Parameters
    ----------
    kwargs : dict
        Primitive parameters, including ``outs``, ``task_capacity``,
        ``weighted`` and ``config_spec``.
    symbol : str
        FFI symbol inside the compiled module.
    event_abi : EventABI
        Event dtype adapter of the call.
    permuted : bool
        Use the column-major source, which reads ``weights[perm[k]]``.

    Returns
    -------
    callable
        ``kernel(weights, indices, indptr, [perm,] vector)``. A ``perm``
        operand is ignored unless ``permuted``.
    """
    schedule = _config_from_params(kwargs, None).mv
    mode = 'weighted' if kwargs.get('weighted', False) else 'binary'
    stem = f'hybrid_spmv_col_major_{mode}' if permuted else f'hybrid_spmv_{mode}'
    module = f'csr_{stem}{schedule.module_suffix()}'
    load_cuda_file(
        _HYBRID_MV_DIR / f'{stem}.cu',
        name=module,
        extra_include_paths=[str(_HYBRID_MV_DIR)],
        extra_cuda_cflags=schedule.compile_flags(),
        allow_cuda_graph=True,
    )
    out_info = kwargs['outs'][0]
    task_capacity = kwargs['task_capacity']
    first_scratch = 5 if permuted else 4

    def kernel(weights, indices, indptr, *rest):
        operands = (weights, indices, indptr) + (rest[:1] if permuted else ()) + (event_abi.prepare(rest[-1]),)
        scratch, scratch_info = _hybrid_scratch(task_capacity, indptr.dtype)
        out = jax.ffi.ffi_call(
            f'{module}.{symbol}',
            (_accumulator_info(out_info), *scratch_info),
            input_output_aliases={first_scratch + i: 1 + i for i in range(len(scratch))},
        )(*operands, *scratch, task_capacity=task_capacity)[0]
        return (out.astype(out_info.dtype),)

    return kernel


def _event_weight_cotangent(ct, weight_aval, events, indices, indptr, params, *,
                            primitive, unit_product, perm=None):
    """Weight cotangent of an event-driven CSR product ``W @ events``.

    The event type decides the derivative: binary activity selects the
    weight, weighted events multiply it by their amplitude. Inactive slots
    contribute exactly zero, even for a NaN/Inf cotangent.

    Parameters
    ----------
    ct : jax.Array
        Output cotangent, a vector or ``(out, batch)`` matrix.
    weight_aval : jax.core.ShapedArray
        Abstract weights, shape ``(1,)`` for a shared weight.
    events : jax.Array
        Event vector or ``(n, batch)`` matrix of the forward product.
    indices, indptr : jax.Array
        CSR structure in physical slot order.
    params : dict
        Forward primitive parameters (``transpose``, ``weighted``,
        ``backend`` and ``config_spec``).
    primitive : XLACustomKernel
        Forward primitive whose backend choice the sampled product inherits.
    unit_product : callable or None
        ``unit_product(w)`` evaluates the forward product with shared weight
        ``w`` of shape ``(1,)``. ``None`` sums the per-slot cotangents instead.
    perm : jax.Array or None, optional
        Weight index of each slot (indexed CSR); per-slot cotangents are
        scatter-added back through it.

    Returns
    -------
    jax.Array
        Cotangent with the shape and dtype of ``weight_aval``.
    """
    def per_slot(ct):
        return csr_sddmm_dweight(
            events, ct, indices, indptr,
            transpose=bool(params['transpose']), weighted=bool(params.get('weighted', False)),
            backend=_sddmm_dispatch._CSRBackend(primitive, params['backend'], params.get('config_spec')),
        )

    if weight_aval.shape[0] != 1:
        grad = per_slot(ct)
        if perm is None:
            return grad.astype(weight_aval.dtype)
        return jnp.zeros(weight_aval.shape, weight_aval.dtype).at[perm].add(grad)

    # A shared weight's cotangent is <ct, A_unit @ events>: the event-driven
    # forward touches active edges only. Its unit weight uses at least float32,
    # so half-precision event sums cannot overflow. ``ct * 0`` is exact for a
    # finite cotangent; a NaN/Inf one takes the per-slot select so it cannot
    # leak through an output that received only inactive events.
    acc = jnp.result_type(weight_aval.dtype, ct.dtype, jnp.float32)

    def unit_inner(ct):
        return jnp.vdot(ct.astype(acc), unit_product(jnp.ones((1,), acc)).astype(acc))

    def select_sum(ct):
        return jnp.sum(per_slot(ct).astype(acc))

    if unit_product is None:
        grad = select_sum(ct)
    else:
        grad = jax.lax.cond(jnp.all(jnp.isfinite(ct)), unit_inner, select_sum, ct)
    return grad.astype(weight_aval.dtype).reshape(weight_aval.shape)
