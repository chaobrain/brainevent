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

"""Structure-derived caches shared by the CSR/CSC and FixedNum formats.

A sparse matrix stores one compression of its structure (the *primary*
layout). Some operations need the other compression (the *mirror*: the CSC
view of a CSR matrix, or the column-major view of a fixed-connection matrix)
plus a permutation back to the stored ``data`` order. The mirror depends only
on the structure, so every matrix that shares a structure (``with_data``,
``apply``, ``transpose``, data-only arithmetic) shares one
:class:`StructureCache` by reference.

The names are orientation-free: ``CSR(W).T`` is a ``CSC`` over the very same
arrays, so its primary/mirror layouts are the same objects and nothing has to
be re-keyed.

A cache is only written with concrete arrays (see :func:`is_concrete`): a
matrix whose structure is traced either carries its mirror as pytree leaves
or uses a mirror-free fallback, so no tracer is ever stored on an object that
outlives its trace.
"""

import warnings
from typing import Optional, Tuple

import numpy as np

from brainevent._compatible_import import Tracer
from brainevent._error import PerformanceWarning

__all__ = [
    'StructureCache',
    'is_concrete',
    'nnz_hint',
    'warn_mirror_fallback',
]


def is_concrete(*arrays) -> bool:
    """Return ``True`` when none of ``arrays`` is a JAX tracer."""
    return not any(isinstance(a, Tracer) for a in arrays)


def nnz_hint(indptr) -> Optional[int]:
    """Power-of-two upper bound on the longest line of ``indptr``.

    It sizes the ``cuda_raw`` hybrid task queue (see
    :func:`brainevent._csr.hybrid_config.hybrid_capacity_bound`). Rounding up
    to a power of two keeps the hint, which lives in the pytree metadata, the
    same across random draws of one connectivity family, so re-drawing a
    matrix does not force a recompilation.

    Returns ``None`` for a traced ``indptr``.
    """
    if not is_concrete(indptr):
        return None
    ptr = np.asarray(indptr)
    if ptr.size < 2:
        return 0
    longest = int(np.diff(ptr).max())
    return 0 if longest <= 0 else 1 << (longest - 1).bit_length()


class StructureCache:
    """Mutable holder of arrays derived from one sparse structure.

    Attributes
    ----------
    primary_hint : int or None
        :func:`nnz_hint` of the stored ``indptr`` (``None`` when unknown).
    mirror : tuple of jax.Array or None
        ``(indptr, indices, perm)`` of the other compression; slot ``j`` of
        the mirror holds stored element ``perm[j]``.
    mirror_hint : int or None
        :func:`nnz_hint` of the mirror ``indptr``.
    diag_plan : tuple of jax.Array or None
        The ``diag_add`` relocation plan (CSR/CSC only).
    """
    __slots__ = ('primary_hint', 'mirror', 'mirror_hint', 'diag_plan')

    def __init__(
        self,
        primary_hint: Optional[int] = None,
        mirror: Optional[Tuple] = None,
        mirror_hint: Optional[int] = None,
        diag_plan: Optional[Tuple] = None,
    ):
        self.primary_hint = primary_hint
        self.mirror = None if mirror is None else tuple(mirror)
        self.mirror_hint = mirror_hint
        self.diag_plan = None if diag_plan is None else tuple(diag_plan)

    def set_mirror(self, mirror: Tuple) -> Tuple:
        """Store ``mirror`` (and its hint) and return it as a tuple."""
        self.mirror = tuple(mirror)
        self.mirror_hint = nnz_hint(self.mirror[0])
        return self.mirror


_WARNED: set = set()


def warn_mirror_fallback(owner: str, operation: str) -> None:
    """Warn once per ``(owner, operation)`` that a mirror-free path is used."""
    key = (owner, operation)
    if key in _WARNED:
        return
    _WARNED.add(key)
    warnings.warn(
        f"{owner}: {operation} runs without the cached mirror layout because the "
        f"matrix structure is traced, so it uses a slower non-event-driven path. "
        f"Call `M = M.build_weight_indices()` before `jax.jit` (or pass "
        f"`precompute_weight_indices=True` at construction) to keep it event-driven.",
        PerformanceWarning,
        stacklevel=3,
    )
