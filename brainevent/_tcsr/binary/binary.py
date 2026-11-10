# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""CUDA tiled binary MM with canonical CSR differentiation."""

from __future__ import annotations

import hashlib
import operator
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainevent._compatible_import import match_varying_axes
from brainevent._csr._support.binary import _config_for_call, _validate_backward_algorithm
from brainevent._csr.binary.binary import (
    _csrmm_jvp_B as _csr_jvp_B,
    _csrmm_transpose_rule as _csr_transpose_rule,
)
from brainevent._csr.config import CSRConfig
from brainevent._misc import namescope
from brainevent._op import XLACustomKernel, general_batching_rule, load_cuda_file, load_cuda_inline
from brainevent._op.util import cuda_event_abi
from brainevent._typing import Data, Indptr, Index, MatrixShape

from brainevent._tcsr.binary.combined.planner import INT32_MAX, tile_schedule

__all__ = ['binary_csrmm', 'binary_csrmm_p_call', 'binary_csrmm_p']

_ROOT = Path(__file__).parent / 'combined'
_VALUE_TYPES = {
    np.dtype(jnp.float16): ('f16', 0),
    np.dtype(jnp.bfloat16): ('bf16', 1),
    np.dtype(jnp.float32): ('f32', 2),
    np.dtype(jnp.float64): ('f64', 3),
}
_INDEX_DTYPES = (np.dtype('int32'), np.dtype('int64'))


def _metadata_array(value: Any, name: str) -> jax.Array:
    """Convert metadata without silently narrowing explicit int64 inputs."""
    dtype = getattr(value, 'dtype', None)
    result = jnp.asarray(value)
    if dtype is not None and np.dtype(dtype) != np.dtype(result.dtype):
        raise ValueError(f'JAX must preserve {name} dtype; enable x64 for int64 metadata.')
    return result


def _static_integer(value: Any, name: str) -> int:
    """Require a nonnegative static integer rather than a bool or tracer."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a nonnegative integer')
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f'{name} must be a nonnegative static integer') from exc
    if result < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return result


def _precheck_metadata(arguments: dict[str, Any]) -> None:
    """Check explicit int64 metadata before the public wrapper enters JIT."""
    names = ['indices', 'indptr', 'local_targets', 'tile_offsets']
    data = u.get_mantissa(arguments.get('data'))
    if not arguments.get('transpose', True) and getattr(data, 'size', 1) != 1:
        names.append('perm')
    for name in names:
        value = arguments.get(name)
        dtype = getattr(value, 'dtype', None)
        if dtype is not None and np.dtype(dtype) == np.dtype('int64'):
            _metadata_array(value, name)


def _validate_csrmm_call(
    weights: jax.Array, indices: jax.Array, indptr: jax.Array, B: jax.Array,
    local_targets: jax.Array, tile_offsets: jax.Array, perm: jax.Array,
    *, shape: MatrixShape, transpose: bool, tile_width: int,
) -> None:
    """Validate dimensions and typed metadata without reading their contents."""
    rows, cols = shape
    if rows > INT32_MAX or cols > INT32_MAX:
        raise ValueError('TCSR MM dimensions must fit int32')
    if indices.ndim != 1 or indices.dtype != jnp.int32:
        raise TypeError('canonical indices must be rank-one int32')
    if indptr.ndim != 1 or np.dtype(indptr.dtype) not in _INDEX_DTYPES:
        raise TypeError('canonical indptr must be rank-one int32/int64')
    if indptr.shape != (rows + 1,):
        raise ValueError('indptr length must equal shape[0] + 1')
    nnz = indices.size
    if nnz and (not rows or not cols):
        raise ValueError('nonempty CSR structure requires positive matrix dimensions')
    if indptr.dtype == jnp.int32 and nnz > INT32_MAX:
        raise ValueError('int32 indptr cannot address more than INT32_MAX edges')
    if weights.ndim != 1 or weights.shape not in ((1,), (nnz,)):
        raise ValueError('weights must be scalar, shape (1,), or shape (nnz,)')
    if np.dtype(weights.dtype) not in _VALUE_TYPES:
        raise TypeError('weights must be float16, bfloat16, float32 or float64')
    if B.ndim != 2:
        raise ValueError('events must be a rank-two NB matrix')
    if B.dtype not in (jnp.bool_, jnp.float32):
        raise TypeError('binary events must be bool or float32')
    source, target = (rows, cols) if transpose else (cols, rows)
    if B.shape[0] != source:
        raise ValueError('events shape does not match the selected CSR direction')
    batch = B.shape[1]
    if batch > 65535:
        raise ValueError('TCSR MM batch must not exceed 65535')
    schedule = tile_schedule(target, batch, 8 if weights.dtype == jnp.float64 else 4)
    if tile_width != schedule.tile_width:
        raise ValueError('tile_width does not match the prepared direction, batch and dtype')
    if local_targets.ndim != 1 or local_targets.dtype != jnp.uint16:
        raise TypeError('local_targets must be rank-one uint16')
    if local_targets.shape != (nnz,):
        raise ValueError('local_targets length must equal nnz')
    if tile_offsets.ndim != 1 or np.dtype(tile_offsets.dtype) not in _INDEX_DTYPES:
        raise TypeError('tile_offsets must be rank-one int32/int64 absolute offsets')
    if tile_offsets.size != source * (schedule.tile_count + 1):
        raise ValueError('tile_offsets length does not match the prepared tile geometry')
    if tile_offsets.dtype == jnp.int32 and nnz > INT32_MAX:
        raise ValueError('int32 tile_offsets cannot address more than INT32_MAX edges')
    if perm.ndim != 1 or perm.dtype != tile_offsets.dtype:
        raise TypeError('perm must be rank one and have the same dtype as tile_offsets')
    expected_perm = (nnz,) if not transpose and weights.size != 1 else (0,)
    if perm.shape != expected_perm:
        raise ValueError('col-major hetero perm must have length nnz; other modes use an empty perm')


@namescope(
    name='brainevent.tcsr_binary_csrmm',
    static_argnames=('shape', 'transpose', 'tile_width', 'backward_algorithm'),
    precheck=_precheck_metadata,
)
def binary_csrmm(
    data: Data, indices: Index, indptr: Indptr, B: Data, *,
    local_targets: Index, tile_offsets: Indptr, tile_width: int,
    shape: MatrixShape, perm: Index | None = None, transpose: bool = True,
    backend: str | None = None, backward_algorithm: str = 'weighted_float',
    config: CSRConfig | None = None,
) -> Data:
    """Multiply canonical CSR weights and binary events using CUDA tiles.

    Parameters
    ----------
    data : array_like or brainunit.Quantity
        Canonical weights, scalar, shape ``(1,)`` or ``(nnz,)``. Supported
        dtypes are float16, bfloat16, float32 and float64.
    indices, indptr : array_like
        Canonical CSR columns (int32) and row pointers (int32/int64).
    B : array_like or brainunit.Quantity
        Bool/float32 NB events. With ``shape=(M,N)``, shape is ``(M,batch)``
        for ``transpose=True`` and ``(N,batch)`` otherwise. Nonzero events,
        including negative values and NaN, are active.
    local_targets : array_like
        Prepared uint16 local targets, shape ``(nnz,)``.
    tile_offsets : array_like
        Flattened absolute int32/int64 edge boundaries of the selected
        canonical or mirror traversal.
    tile_width : int
        Actual width from the combined planner for this direction and batch.
    shape : tuple of int
        Shape ``(M,N)`` of the canonical matrix, regardless of direction.
    perm : array_like or None, optional
        Mirror-slot to canonical-weight permutation for col-major hetero.
        Must match the tile offsets' dtype. Ignored in direct/homo modes.
    transpose : bool, optional
        Compute ``W.T @ e(B)`` when True, otherwise ``W @ e(B)`` using
        mirror tiles, where ``e(x)=(x != 0)``. Default is True.
    backend : {None, 'cuda_raw'}, optional
        Forward implementation. Both select CUDA; no CPU kernel is registered.
    backward_algorithm : {'binary', 'weighted_float'}, optional
        Weight-gradient convention. ``binary`` uses the activity mask;
        ``weighted_float`` uses numeric event values. Does not change the
        forward. Default is ``'weighted_float'``.
    config : CSRConfig or None, optional
        Canonical CSR configuration for shared gradient dispatch.

    Returns
    -------
    jax.Array or brainunit.Quantity
        NB result of shape ``(N,batch)`` when transposed, otherwise
        ``(M,batch)``, with weight dtype and product units.

    Notes
    -----
    Prepare sorted tile metadata outside JIT. The caller guarantees that
    canonical CSR, mirror tiles and perm describe the same structure.
    Updating values reuses metadata; changing structure, direction, dtype or
    batch requires a matching plan. Input derivatives follow the shared CSR
    numerical straight-through convention, rather than differentiating the
    activity comparison.
    """
    data, unitd = u.split_mantissa_unit(data)
    B, unitb = u.split_mantissa_unit(B)
    result = binary_csrmm_p_call(
        data, indices, indptr, B, local_targets=local_targets, tile_offsets=tile_offsets,
        tile_width=tile_width, shape=shape, perm=perm, transpose=transpose,
        backend=backend, backward_algorithm=backward_algorithm, config=config,
    )[0]
    return u.maybe_decimal(result * (unitd * unitb))


def binary_csrmm_p_call(
    weights: Any, indices: Index, indptr: Indptr, B: Any, *,
    local_targets: Index, tile_offsets: Indptr, tile_width: int,
    shape: MatrixShape, perm: Index | None = None, transpose: bool = True,
    backend: str | None = None, backward_algorithm: str = 'weighted_float',
    config: CSRConfig | None = None,
) -> tuple[jax.Array, ...]:
    """Validate and bind the seven-operand CUDA tiled binary MM primitive.

    Parameters
    ----------
    weights, indices, indptr, B : array_like
        Unitless canonical CSR weights/structure and NB binary events, as
        described by :func:`binary_csrmm`.
    local_targets, tile_offsets : array_like
        Prepared dynamic tile arrays for the selected traversal.
    tile_width : int
        Actual prepared width, matching output length, batch and weight dtype.
    shape : tuple of int
        Canonical CSR shape, independent of the physical traversal.
    perm : array_like or None, optional
        Mirror-to-canonical permutation required for col-major hetero.
    transpose : bool, optional
        Select direct ``W.T @ e(B)`` or mirror ``W @ e(B)``.
    backend : {None, 'cuda_raw'}, optional
        Both select the only registered forward backend, CUDA.
    backward_algorithm : {'binary', 'weighted_float'}, optional
        Shared CSR weight-gradient convention. ``binary`` uses the activity
        mask; ``weighted_float`` uses numeric event values. Default is
        ``'weighted_float'``.
    config : CSRConfig or None, optional
        Canonical configuration snapshot used by shared gradient rules.

    Returns
    -------
    tuple of jax.Array
        One mathematical NB output. Native scratch is not exposed.

    Raises
    ------
    TypeError
        Operand dtype or rank is unsupported.
    ValueError
        Dimensions, backend, tile width, metadata sizes or config disagree.
        Array contents and permutation validity are guaranteed by preparation.
    """
    if backend not in (None, 'cuda_raw'):
        raise ValueError('TCSR binary MM supports only the cuda_raw backend')
    _validate_backward_algorithm(backward_algorithm)
    if not isinstance(transpose, (bool, np.bool_)):
        raise TypeError('transpose must be bool')
    if len(shape) != 2:
        raise ValueError('shape must contain two dimensions')
    shape = (_static_integer(shape[0], 'shape[0]'), _static_integer(shape[1], 'shape[1]'))
    tile_width = _static_integer(tile_width, 'tile_width')
    transpose = bool(transpose)
    weights, B = jnp.asarray(weights), jnp.asarray(B)
    if weights.ndim == 0:
        weights = weights.reshape((1,))
    indices = _metadata_array(indices, 'indices')
    indptr = _metadata_array(indptr, 'indptr')
    local_targets = _metadata_array(local_targets, 'local_targets')
    tile_offsets = _metadata_array(tile_offsets, 'tile_offsets')
    if transpose or weights.size == 1:
        permutation = jnp.empty((0,), dtype=tile_offsets.dtype)
    elif perm is None:
        raise ValueError('col-major hetero requires a mirror-to-canonical perm')
    else:
        permutation = _metadata_array(perm, 'perm')
    _validate_csrmm_call(
        weights, indices, indptr, B, local_targets, tile_offsets, permutation,
        shape=shape, transpose=transpose, tile_width=tile_width,
    )
    config_spec = None
    if config is not None:
        config_spec, _ = _config_for_call(config, indices, indptr, shape).kernel_args()
    target = shape[1] if transpose else shape[0]
    operands = (weights, indices, indptr, B, local_targets, tile_offsets, permutation)
    return tuple(binary_csrmm_p(
        *operands, outs=(jax.ShapeDtypeStruct((target, B.shape[1]), weights.dtype),),
        shape=shape, transpose=transpose, tile_width=tile_width, backend='cuda_raw',
        backward_algorithm=backward_algorithm, config_spec=config_spec,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
    ))


def _combined_ffi_target(
    weight_dtype: Any, event_dtype: Any, *, transpose: bool, homo: bool,
) -> tuple[str, str]:
    """Select a module family and its exact native binary symbol."""
    value = _VALUE_TYPES[np.dtype(weight_dtype)][0]
    event = 'bool' if event_dtype == jnp.bool_ else 'float'
    if homo:
        return 'direct', f'combined_homo_{value}_{event}'
    if transpose:
        return 'direct', f'combined_{value}_{event}'
    return 'col_major', f'combined_col_major_binary_hetero_{value}_{event}'


@lru_cache(maxsize=2)
def _load_combined_cuda(family: str) -> str:
    """Load a standalone-safe native module with all dependencies in its key."""
    source = _ROOT / 'combined_spmm_binary.cu'
    if family == 'col_major':
        source = _ROOT / 'col_major' / 'combined_spmm_col_major_binary.cu'
    dependencies = (source, _ROOT / 'combined_spmm_binary.cu',
                    _ROOT / 'planner.py', _ROOT / 'planner.h', *sorted(_ROOT.rglob('*.cuh')))
    digest = hashlib.sha256(b''.join(path.read_bytes() for path in dependencies)).hexdigest()[:12]
    module = f'tcsr_combined_{family}_{digest}'
    if family == 'direct':
        load_cuda_file(source, name=module, extra_include_paths=[str(_ROOT)], allow_cuda_graph=False)
    else:
        text = ('#define BE_SPMM_STANDALONE\n'
                f'#include "{_ROOT / "combined_spmm_binary.cu"}"\n'
                '#undef BE_SPMM_STANDALONE\n' + source.read_text())
        load_cuda_inline(module, text, extra_include_paths=[str(_ROOT), str(source.parent)],
                         allow_cuda_graph=False)
    return module


def _combined_ffi_outputs(rows: int, cols: int, batch: int, dtype: Any) -> tuple[Any, ...]:
    """Describe the BN result and invocation-local ordered-extraction buffers."""
    chunks = (rows + 4095) // 4096
    workspace_bytes = ((3 * batch + 1) // 2 * 8 + batch * chunks * 8) if rows and batch else 0
    return (
        jax.ShapeDtypeStruct((batch, cols), dtype),
        jax.ShapeDtypeStruct((max(1, batch * rows),), jnp.int32),
        jax.ShapeDtypeStruct((max(1, (workspace_bytes + 3) // 4),), jnp.uint32),
    )


def _binary_csrmm_cuda_kernel(
    *, weight_info: jax.ShapeDtypeStruct, vector_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct, shape: MatrixShape, transpose: bool,
    tile_width: int, outs: Any, **kwargs: Any,
) -> Callable[..., tuple[jax.Array, ...]]:
    """Lower one mathematical NB result to direct or mirror BN CUDA FFI."""
    rows, cols = shape if transpose else shape[::-1]
    batch = vector_info.shape[1]
    out_info = outs[0]
    if not rows or not cols or not batch or not indices_info.size:
        return lambda *args: (match_varying_axes(jnp.zeros(out_info.shape, out_info.dtype), *args),)
    homo = weight_info.size == 1
    family, symbol = _combined_ffi_target(
        weight_info.dtype, vector_info.dtype, transpose=transpose, homo=homo,
    )
    module = _load_combined_cuda(family)
    abi = cuda_event_abi(vector_info.dtype)
    ffi = jax.ffi.ffi_call(
        module + '.' + symbol, _combined_ffi_outputs(rows, cols, batch, weight_info.dtype),
        vmap_method='sequential',
    )

    def kernel(weights, indices, indptr, B, local_targets, tile_offsets, perm):
        operands = (weights, local_targets, tile_offsets)
        if not transpose and not homo:
            operands += (perm,)
        output = ffi(
            *operands, abi.prepare(B).T, tile_width=tile_width,
            value_type=_VALUE_TYPES[np.dtype(weight_info.dtype)][1],
        )[0]
        return (match_varying_axes(output.T, weights, indices, indptr, B,
                                   local_targets, tile_offsets, perm),)

    return kernel


def _csrmm_jvp_data(
    data_dot, data, indices, indptr, B, local_targets, tile_offsets, perm,
    *, shape, transpose, tile_width, config_spec=None, **kwargs,
):
    """Reuse the tiled forward for the linear contribution of the weights."""
    config = None if config_spec is None else CSRConfig.from_kernel_args(config_spec, None)
    return binary_csrmm_p_call(
        data_dot, indices, indptr, B, local_targets=local_targets, tile_offsets=tile_offsets,
        tile_width=tile_width, perm=perm, shape=shape, transpose=transpose,
        backend='cuda_raw', backward_algorithm=kwargs.get('backward_algorithm', 'weighted_float'), config=config,
    )


def _csrmm_jvp_B(
    B_dot, data, indices, indptr, B, local_targets, tile_offsets, perm,
    *, shape, transpose, **kwargs,
):
    """Delegate numerical input tangents to the canonical CSR rule."""
    return _csr_jvp_B(B_dot, data, indices, indptr, B, None,
                      shape=shape, transpose=transpose, backend='cuda_raw')


def _csrmm_transpose_rule(
    ct, data, indices, indptr, B, local_targets, tile_offsets, perm,
    *, shape, transpose, config_spec=None, **kwargs,
):
    """Adapt canonical CSR cotangents and mark all tile operands nondifferentiable."""
    result = _csr_transpose_rule(
        ct, data, indices, indptr, B, None, shape=shape, transpose=transpose,
        backend='cuda_raw', config_spec=config_spec, weighted=False,
        backward_algorithm=kwargs.get('backward_algorithm', 'weighted_float'),
    )
    return (*result[:4], None, None, None)


def _csrmm_batching(args, axes, **kwargs):
    """Scan outer mapped axes without changing the prepared inner batch."""
    if axes[1] is not None or axes[2] is not None:
        kwargs = dict(kwargs, config_spec=None)
    return general_batching_rule(binary_csrmm_p.primitive, args, axes, **kwargs)


binary_csrmm_p = XLACustomKernel(
    'tcsr_combined_binary_csrmm',
    doc='CUDA tiled binary MM with canonical CSR AD and direct/mirror traversal.',
)
binary_csrmm_p.def_cuda_raw_kernel(_binary_csrmm_cuda_kernel, asdefault=True)
binary_csrmm_p.def_jvp_rule2(_csrmm_jvp_data, None, None, _csrmm_jvp_B, None, None, None)
binary_csrmm_p.def_transpose_rule(_csrmm_transpose_rule)
binary_csrmm_p.def_batching_rule(_csrmm_batching)
binary_csrmm_p.def_call(binary_csrmm_p_call)
binary_csrmm_p.def_tags('tcsr', 'binary', 'mm')
