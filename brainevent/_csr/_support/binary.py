# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Shared helpers for direct and indexed CSR binary event primitives."""

import jax
import jax.numpy as jnp
import numpy as np

from brainevent._csr.config import CSRConfig


BACKWARD_ALGORITHMS = frozenset({'binary', 'weighted_float'})


def _validate_backward_algorithm(backward_algorithm: str) -> None:
    """Validate the weight-gradient rule for CSR event products."""
    if backward_algorithm not in BACKWARD_ALGORITHMS:
        raise ValueError(
            f"Unsupported CSR backward algorithm {backward_algorithm!r}. "
            "Supported algorithms: binary, weighted_float."
        )


def _weight_gradient_mode(weighted: bool, backward_algorithm: str) -> str:
    """Select event values or activity masks for weight cotangents."""
    return 'weighted' if weighted or backward_algorithm == 'weighted_float' else 'binary'


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
    result = dict(params, config_spec=spec, call_plan=None)
    if 'max_row_nnz' in params:
        result['max_row_nnz'] = None
    if 'task_capacity' in params:
        result['task_capacity'] = config.mv_capacity(params['indices_info'].size,
                                                     params['indptr_info'].size - 1)
    return result


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
    if isinstance(max_row_nnz, (bool, np.bool_)) or not isinstance(max_row_nnz, (int, np.integer)) or max_row_nnz < 0:
        raise ValueError('max_row_nnz must be a nonnegative integer')
    if (config is not None and config.primary_max_row_nnz is not None
            and max_row_nnz < config.primary_max_row_nnz):
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
