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
