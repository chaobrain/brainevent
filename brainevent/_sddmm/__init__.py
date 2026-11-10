# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""One facade for sampled matrix/vector products and event weight gradients."""

import jax
import jax.numpy as jnp

from . import dispatch as _dispatch
from .dispatch import sddmm
from ._mv.sddmv import csr_sddmv_p
from ._mm.sddmm import csr_sddmm_p

__all__ = [
    'sddmm',
    'csr_sddmm_dweight', 'csr_sddmm_p',
    'csr_sddmv_dweight', 'csr_sddmv_p',
]


def csr_sddmm_dweight(
    events: jax.Array, ct: jax.Array, indices: jax.Array, indptr: jax.Array,
    *, transpose: bool, weighted: bool = False, backend: str | None = None, config=None,
) -> jax.Array:
    """Compute CSR event weight gradients through batch-based dispatch.

    Parameters
    ----------
    events, ct : jax.Array
        Event and cotangent vectors, or matrices with batch on the last axis.
        For transposed CSR products their leading sizes are rows and columns;
        otherwise columns and rows. Matrix batch sizes must match.
    indices, indptr : jax.Array
        Int32 CSR columns and int32/int64 row offsets, in physical slot order.
    transpose : bool
        Whether the forward product is ``W.T @ events``.
    weighted : bool, optional
        Use event amplitudes instead of binary activity alone.
        CUDA tests weighted activity after casting events to ``ct.dtype``;
        JAX tests the original events before casting.
    backend : str or None, optional
        Forwarded to the chosen MV/MM implementation. GPU auto uses CUDA for
        supported transposed CSR events and JAX otherwise. ``jax_raw`` forces
        differentiable JAX in both directions without binding an event primitive.
        Explicit ``cusparse`` uses cuSPARSE SDDMM on GPU in either direction,
        with exact selection for nonfinite cotangents and no derivative rules.
        Half types compute in float32; weighted activity is tested after casting.
    config : CSRConfig or None, optional
        Explicit scheduler snapshot. Defaults do not read persisted settings.

    Returns
    -------
    jax.Array
        One cotangent per CSR slot, summed over batch in ``ct.dtype``.
        Inactive events contribute zero even with NaN/Inf cotangents.
        No canonical-weight permutation or shared-weight reduction is applied.
    """
    if not isinstance(transpose, bool) or not isinstance(weighted, bool):
        raise TypeError('transpose and weighted must be bool')
    events, ct = map(jnp.asarray, (events, ct))
    if events.ndim not in (1, 2) or ct.ndim != events.ndim:
        raise ValueError('events and ct must both be vectors or rank-2 matrices')
    if events.ndim == 1:
        A, B = (events, ct) if transpose else (ct, events)
    else:
        A, B = (events, ct.T) if transpose else (ct, events.T)
    return _dispatch.sddmm(A, B, indices, indptr, mode='weighted' if weighted else 'binary',
                           transpose=transpose, backend=backend, config=config)


def csr_sddmv_dweight(
    events: jax.Array, ct: jax.Array, indices: jax.Array, indptr: jax.Array,
    *, transpose: bool, weighted: bool = False, backend: str | None = None, config=None,
) -> jax.Array:
    """Vector entry into the unified CSR sampled dispatcher.

    Parameters
    ----------
    events, ct : jax.Array
        Rank-one event and cotangent arrays.
    indices, indptr : jax.Array
        CSR columns and row offsets.
    transpose : bool
        Direction of the forward CSR product.
    weighted : bool, optional
        Use event amplitudes rather than binary selection.
    backend : str or None, optional
        Leaf backend override, as in :func:`csr_sddmm_dweight`.
    config : CSRConfig or None, optional
        Explicit scheduler snapshot, as in :func:`csr_sddmm_dweight`.

    Returns
    -------
    jax.Array
        Per-slot vector weight cotangents in ``ct.dtype``.
    """
    if jnp.ndim(events) != 1 or jnp.ndim(ct) != 1:
        raise ValueError('events and ct must be rank 1')
    return csr_sddmm_dweight(events, ct, indices, indptr, transpose=transpose,
                             weighted=weighted, backend=backend, config=config)


# Registry calls also enter the facade, never bypassing batch dispatch.
csr_sddmv_p.def_call(csr_sddmv_dweight)
csr_sddmm_p.def_call(csr_sddmm_dweight)
