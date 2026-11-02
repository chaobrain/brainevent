# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""One facade for sampled matrix/vector products and event weight gradients."""

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from brainevent._misc import namescope

from . import dispatch as _dispatch
from .dispatch import sddmm
from ._mv.sddmv import csr_sddmv_p
from ._mm.sddmm import csr_sddmm_p

__all__ = [
    'sddmm', 'sddmm_indices', 'sddmm_coo_indices', 'sddmm_bcoo',
    'csr_sddmm_dweight', 'csr_sddmm_p',
    'csr_sddmv_dweight', 'csr_sddmv_dweight_binary', 'csr_sddmv_dweight_weighted', 'csr_sddmv_p',
]


@namescope
def sddmm_indices(A: jax.Array, B: jax.Array, indices: jax.Array,
                  *, backend: str | None = None) -> BCOO:
    """Sample ``A @ B`` at COO coordinates through the common dispatcher.

    Parameters
    ----------
    A, B : jax.Array
        Dense matrices of shape ``(m, batch)`` and ``(batch, n)``.
    indices : jax.Array
        Integer COO coordinates of shape ``(nse, 2)``.
    backend : str or None, optional
        Leaf backend selection; ordinary products currently use JAX.

    Returns
    -------
    BCOO
        Shape ``(m, n)``, retaining the input coordinate order and duplicates.
        Matrix values are computed only at the sampled slots. The ordinary
        JAX implementation retains differentiation and shard_map support.
    """
    A, B, indices = map(jnp.asarray, (A, B, indices))
    # Preserve the original public wrapper's rank/shape contract.
    assert A.ndim == 2
    assert B.ndim == 2
    assert A.shape[1] == B.shape[0]
    assert indices.ndim == 2
    assert indices.shape[1] == 2
    values = _dispatch.sddmm(A, B, indices, backend=backend)
    return BCOO((values, indices), shape=(A.shape[0], B.shape[1]))


@namescope
def sddmm_coo_indices(A: jax.Array, B: jax.Array, pre_idx: jax.Array, post_idx: jax.Array,
                      *, backend: str | None = None) -> BCOO:
    """Sample a dense product at separate row and column coordinates.

    Parameters
    ----------
    A, B : jax.Array
        Dense matrices as in :func:`sddmm_indices`.
    pre_idx, post_idx : jax.Array
        Matching rank-one arrays of sampled row and column coordinates.
    backend : str or None, optional
        Leaf backend override.

    Returns
    -------
    BCOO
        Sampled values and coordinates, preserving the original API.
    """
    pre_idx, post_idx = map(jnp.asarray, (pre_idx, post_idx))
    assert pre_idx.ndim == 1
    assert post_idx.ndim == 1
    assert pre_idx.shape == post_idx.shape
    return sddmm_indices(A, B, jnp.stack([pre_idx, post_idx], axis=1), backend=backend)


@namescope
def sddmm_bcoo(A: jax.Array, B: jax.Array, sparsity_pattern: BCOO,
               *, backend: str | None = None) -> BCOO:
    """Sample a dense product using a BCOO coordinate pattern.

    Parameters
    ----------
    A, B : jax.Array
        Dense matrices as in :func:`sddmm_indices`.
    sparsity_pattern : BCOO
        Supplies the sparse coordinates; its numerical data are ignored.
    backend : str or None, optional
        Leaf backend override.

    Returns
    -------
    BCOO
        Sampled dense product in the supplied coordinate order.
    """
    return sddmm_indices(A, B, sparsity_pattern.indices, backend=backend)


def csr_sddmm_dweight(
    events: jax.Array, ct: jax.Array, indices: jax.Array, indptr: jax.Array,
    *, transpose: bool, weighted: bool = False, backend: str | None = None,
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
    backend : str or None, optional
        Forwarded to the chosen MV/MM implementation. MM currently uses JAX.

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
                           transpose=transpose, backend=backend)


def csr_sddmv_dweight(
    events: jax.Array, ct: jax.Array, indices: jax.Array, indptr: jax.Array,
    *, transpose: bool, weighted: bool = False, backend: str | None = None,
) -> jax.Array:
    """Compatibility vector entry into the unified CSR sampled dispatcher.

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

    Returns
    -------
    jax.Array
        Per-slot vector weight cotangents in ``ct.dtype``.
    """
    if jnp.ndim(events) != 1 or jnp.ndim(ct) != 1:
        raise ValueError('events and ct must be rank 1')
    return csr_sddmm_dweight(events, ct, indices, indptr, transpose=transpose,
                             weighted=weighted, backend=backend)


def csr_sddmv_dweight_binary(
    events: jax.Array, ct: jax.Array, indices: jax.Array, indptr: jax.Array,
    *, transpose: bool, backend: str | None = None,
) -> jax.Array:
    """Compute binary-event vector weight cotangents.

    Parameters
    ----------
    events, ct, indices, indptr : jax.Array
        Vector and CSR operands as in :func:`csr_sddmv_dweight`.
    transpose : bool
        Direction of the forward CSR product.
    backend : str or None, optional
        Leaf backend override.

    Returns
    -------
    jax.Array
        Selected cotangents in physical CSR slot order.
    """
    return csr_sddmv_dweight(events, ct, indices, indptr, transpose=transpose, backend=backend)


def csr_sddmv_dweight_weighted(
    events: jax.Array, ct: jax.Array, indices: jax.Array, indptr: jax.Array,
    *, transpose: bool, backend: str | None = None,
) -> jax.Array:
    """Compute value-weighted event vector weight cotangents.

    Parameters
    ----------
    events, ct, indices, indptr : jax.Array
        Vector and CSR operands as in :func:`csr_sddmv_dweight`.
    transpose : bool
        Direction of the forward CSR product.
    backend : str or None, optional
        Leaf backend override.

    Returns
    -------
    jax.Array
        Event-weighted cotangents in physical CSR slot order.
    """
    return csr_sddmv_dweight(events, ct, indices, indptr, transpose=transpose,
                             weighted=True, backend=backend)


# Registry calls also enter the facade, never bypassing batch dispatch.
csr_sddmv_p.def_call(csr_sddmv_dweight)
csr_sddmm_p.def_call(csr_sddmm_dweight)
