# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-

from typing import Optional, Tuple, Union

import brainunit as u
import jax

from brainevent._misc import namescope

__all__ = [
    'compact_binary_fcnmv',
    'compact_binary_fcnmv_p',
    'compact_binary_fcnmm',
    'compact_binary_fcnmm_p',
]

_COMPACT_REMOVED_MESSAGE = (
    'Compact binary_fcnmv / compact_binary_fcnmm operators are no longer supported; '
    'the related operators have been removed. '
    'Recommended alternatives: binary_fcnmv / binary_fcnmm or '
    'bitpack_binary_fcnmv / bitpack_binary_fcnmm.'
)


def _raise_compact_removed() -> None:
    raise RuntimeError(_COMPACT_REMOVED_MESSAGE)


class _RemovedCompactPrimitive:
    """Compatibility stub for removed compact FCN primitives."""

    def __init__(self, name: str):
        self.name = name

    def available_backends(self, platform: str) -> Tuple[str, ...]:
        del platform
        return ()

    def __call__(self, *args, **kwargs):
        del args, kwargs
        _raise_compact_removed()


compact_binary_fcnmv_p = _RemovedCompactPrimitive('compact_binary_fcnmv')
compact_binary_fcnmm_p = _RemovedCompactPrimitive('compact_binary_fcnmm')


@namescope(static_argnames=['shape', 'transpose'])
def compact_binary_fcnmv(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    packed_spikes: jax.Array,
    active_ids: jax.Array,
    n_active: jax.Array,
    spikes: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    backend: Optional[str] = None,
    col_weights: Optional[Union[jax.Array, u.Quantity]] = None,
    col_indices: Optional[jax.Array] = None,
    col_indptr: Optional[jax.Array] = None,
) -> Union[jax.Array, u.Quantity]:
    del (
        weights, indices, packed_spikes, active_ids, n_active, spikes,
        shape, transpose, backend, col_weights, col_indices, col_indptr,
    )
    _raise_compact_removed()


@namescope(static_argnames=['shape', 'transpose', 'pack_axis'])
def compact_binary_fcnmm(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    packed_spikes: jax.Array,
    active_ids: jax.Array,
    n_active: jax.Array,
    spikes: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    pack_axis: int = 1,
    backend: Optional[str] = None,
) -> Union[jax.Array, u.Quantity]:
    del (
        weights, indices, packed_spikes, active_ids, n_active, spikes,
        shape, transpose, pack_axis, backend,
    )
    _raise_compact_removed()


'''

def _compact_binary_fcnmv_explicit_scatter_kernel(
    scatter_kind: str,
    transpose: bool,
    col_weight_info: jax.ShapeDtypeStruct,
    col_indices_info: jax.ShapeDtypeStruct,
    col_indptr_info: jax.ShapeDtypeStruct,
    **kwargs,
):
    load_cuda_file(
        _COMPACT_CUDA_DIR / 'compact_binary_fcnmv.cu',
        name='fcn_compact_binary_mv',
    )
    if scatter_kind == '2d_and_atomic':
        load_cuda_file(
            _COMPACT_CUDA_DIR / 'compact_binary_fcnmv_2d_and_atomic.cu',
            name='fcn_compact_binary_mv',
        )
    load_cuda_file(
        _COMPACT_CUDA_DIR / 'compact_binary_fcnmv_T.cu',
        name='fcn_compact_binary_mv_t',
    )
    out_info = kwargs['outs']
    weight_info = kwargs['weight_info']
    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16',
    }
    row_sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')
    col_sfx = _dtype_sfx.get(jnp.dtype(col_weight_info.dtype), row_sfx)
    row_homo = weight_info.size == 1
    col_homo = col_weight_info.size == 1
    col_indices_size = 1
    for dim in col_indices_info.shape:
        col_indices_size *= dim
    col_indptr_size = 1
    for dim in col_indptr_info.shape:
        col_indptr_size *= dim
    col_weight_size = 1
    for dim in col_weight_info.shape:
        col_weight_size *= dim
    use_col_scatter = (
        (not transpose)
        and col_indices_size > 0
        and col_indptr_size > 0
        and col_weight_size > 0
    )
    use_2d_and_atomic = (
        scatter_kind == '2d_and_atomic'
        and transpose
        and not use_col_scatter
        and row_homo
        and row_sfx == '_f32'
    )

    if use_2d_and_atomic:
        kernel_name = 'fcn_compact_binary_mv.compact_binary_fcnmv_scatter_2d_and_atomic_homo_f32'
    elif use_col_scatter:
        mode_sfx = '_homo' if col_homo else '_hetero'
        kernel_name = f'fcn_compact_binary_mv_t.compact_binary_fcnmv_scatter{mode_sfx}{col_sfx}'
    elif transpose:
        mode_sfx = '_homo' if row_homo else '_hetero'
        if scatter_kind == '2d_and_atomic':
            kernel_name = f'fcn_compact_binary_mv.compact_binary_fcnmv_scatter{mode_sfx}{row_sfx}'
        else:
            kernel_name = f'fcn_compact_binary_mv.compact_binary_fcnmv_scatter_{scatter_kind}{mode_sfx}{row_sfx}'
    else:
        mode_sfx = '_homo' if row_homo else '_hetero'
        kernel_name = f'fcn_compact_binary_mv.compact_binary_fcnmv_gather{mode_sfx}{row_sfx}'

    def kernel(weights, indices, packed, active_ids, n_active, spikes, col_weights, col_indices, col_indptr):
        del spikes
        if use_col_scatter:
            return jax.ffi.ffi_call(
                kernel_name, out_info
            )(col_weights, col_indices, col_indptr, packed, active_ids, n_active)
        return jax.ffi.ffi_call(
            kernel_name, out_info
        )(weights, indices, packed, active_ids, n_active)

    return kernel


def compact_tpr_kernel(*args, **kwargs):
    return _compact_binary_fcnmv_explicit_scatter_kernel('tpr', *args, **kwargs)


def compact_wpr_kernel(*args, **kwargs):
    return _compact_binary_fcnmv_explicit_scatter_kernel('wpr', *args, **kwargs)


def compact_bpr_kernel(*args, **kwargs):
    return _compact_binary_fcnmv_explicit_scatter_kernel('bpr', *args, **kwargs)


def compact_2d_and_atomic_kernel(*args, **kwargs):
    return _compact_binary_fcnmv_explicit_scatter_kernel('2d_and_atomic', *args, **kwargs)
    '''