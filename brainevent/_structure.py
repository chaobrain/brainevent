# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

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
from typing import Any, Mapping, Optional, Tuple

from brainevent._compatible_import import Tracer
from brainevent._error import PerformanceWarning

__all__ = [
    'StructureCache',
    'is_concrete',
    'warn_mirror_fallback',
]


def is_concrete(*arrays) -> bool:
    """Return ``True`` when none of ``arrays`` is a JAX tracer."""
    return not any(isinstance(a, Tracer) for a in arrays)


class StructureCache:
    """Mutable holder of arrays derived from one sparse structure.

    Attributes
    ----------
    mirror : tuple of jax.Array or None
        ``(indptr, indices, perm)`` of the other compression; slot ``j`` of
        the mirror holds stored element ``perm[j]``.
    diag_plan : tuple of jax.Array or None
        The ``diag_add`` relocation plan (CSR/CSC only).
    mirror_preparation : tuple or None
        The mirror's immutable MM structure plan and longest compressed row.
        Shared by same-structure objects after the mirror is prepared.
    tile_geometries : dict
        Immutable tiled-MM geometry keyed by direction, actual tile width and
        offset dtype, independent of batch and value dtype.
        ``transpose=True`` addresses the primary traversal; ``False`` uses
        mirror edge slots. Each cache owns its plan dictionary, while plan
        arrays may be shared by structure-preserving copies.
    """
    __slots__ = (
        'mirror', 'diag_plan', 'tile_geometries', 'mirror_preparation',
    )

    def __init__(
        self,
        mirror: Optional[Tuple] = None,
        diag_plan: Optional[Tuple] = None,
        *,
        tile_geometries: Optional[Mapping[Tuple, Any]] = None,
        mirror_preparation: Optional[Tuple[Any, int]] = None,
    ):
        self.mirror = None if mirror is None else tuple(mirror)
        self.mirror_preparation = mirror_preparation
        self.diag_plan = None if diag_plan is None else tuple(diag_plan)
        self.tile_geometries = {} if tile_geometries is None else dict(tile_geometries)

    def copy(self) -> 'StructureCache':
        """Copy cache containers while sharing immutable plan arrays.

        Returns
        -------
        StructureCache
            A cache for the same canonical structure with an independent
            tile-plan dictionary. No array values are read or copied.
        """
        return type(self)(
            mirror=self.mirror, diag_plan=self.diag_plan,
            tile_geometries=self.tile_geometries,
            mirror_preparation=self.mirror_preparation,
        )

    def set_mirror(self, mirror: Tuple) -> Tuple:
        """Replace the mirror and invalidate plans using its edge slots.

        Parameters
        ----------
        mirror : tuple
            ``(indptr, indices, perm)`` for this canonical structure.

        Returns
        -------
        tuple
            The stored mirror. Primary tile plans are preserved; mirror
            statistics and tile plans must be prepared again. No array values
            are read.
        """
        self.mirror = tuple(mirror)
        self.mirror_preparation = None
        self.tile_geometries = {
            key: geometry for key, geometry in self.tile_geometries.items() if key[0]
        }
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
