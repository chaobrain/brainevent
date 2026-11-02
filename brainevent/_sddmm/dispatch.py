# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Unified dispatch for sampled dense products and CSR event weight gradients.

Public wrappers in ``brainevent._sddmm`` enter this dispatcher, then the chosen
leaf selects its backend. Operands are ``A(rows, batch)`` and ``B(batch, cols)``:
batch one uses ``_mv``, all other sizes use ``_mm``. Vectors form a batch-one
outer product; batch zero yields zero per slot. An outer ``vmap`` axis is
independent of this contraction dimension.

Patterns are COO coordinates or CSR columns/offsets, retaining slot order and
duplicates. ``sddmm`` returns slot values; legacy ``sddmm_indices``,
``sddmm_coo_indices`` and ``sddmm_bcoo`` wrappers retain BCOO returns.
``float`` means ordinary multiplication; ``binary`` selects active cotangents;
``weighted`` multiplies active event amplitudes by cotangents. Event leaves
handle NaN/Inf cotangents without contributions from inactive events.

JAX supports both leaves on CPU/GPU/TPU. MV additionally uses hybrid CUDA for
transposed CSR events on GPU. MM currently defaults to JAX; its unregistered
CUDA placeholder raises ``NotImplementedError`` on explicit requests.
Ordinary JAX products retain differentiation and ``shard_map`` support; CSR
event primitives supply other operators' gradients and have no AD rules.
Existing CSR MV gradient rules and TCSR-specific adapters remain separate.
"""

import jax
import jax.numpy as jnp

from ._common import Mode, VALUE_DTYPES
from ._mv import sddmv as mv_backend
from ._mm import sddmm as mm_backend


def sddmm(
    A: jax.Array, B: jax.Array, indices: jax.Array, indptr: jax.Array | None = None,
    *, mode: Mode = 'float', transpose: bool = True, backend: str | None = None,
) -> jax.Array:
    """Sample a dense product through one MV/MM dispatcher.

    Parameters
    ----------
    A, B : jax.Array
        Dense operands of shape ``(rows, batch)`` and ``(batch, cols)``.
        One-dimensional operands are normalized to a column/row vector,
        giving an outer product with batch one. Batch is the contraction
        dimension, independent of an outer ``vmap`` axis.
    indices : jax.Array
        Integer COO coordinates ``(nse, 2)`` when ``indptr`` is absent,
        otherwise int32 CSR column indices ``(nse,)``.
    indptr : jax.Array or None, optional
        Int32/int64 CSR row offsets. Valid CSR structure and column bounds
        are required. No sorting or coalescing is performed.
    mode : {'float', 'binary', 'weighted'}, optional
        Ordinary sampled multiplication, binary-event selection, or
        value-weighted event selection. Event modes preserve original activity
        before converting amplitudes to the cotangent dtype.
    transpose : bool, optional
        In event modes, ``True`` identifies ``A`` as events and ``B.T`` as
        output cotangents; ``False`` identifies ``B.T`` as events and ``A`` as
        cotangents. It does not transpose the ordinary product ``A @ B``.
    backend : str or None, optional
        Forwarded to the selected leaf. ``jax_raw`` supports both leaves.
        MV can use hybrid CUDA for transposed CSR event products. MM currently
        uses JAX; an explicit ``cuda_raw`` request raises NotImplementedError.

    Returns
    -------
    jax.Array
        Shape ``(nse,)`` in physical slot order. Event results have the
        cotangent dtype; ordinary products retain JAX's multiplication dtype.

    Notes
    -----
    Batch one dispatches to ``_mv``; other sizes, including zero, to ``_mm``.
    Ordinary JAX products remain differentiable. CSR event primitives supply
    gradients to other operations and do not have derivative rules themselves.
    """
    if mode not in ('float', 'binary', 'weighted'):
        raise ValueError('mode must be float, binary or weighted')
    if not isinstance(transpose, bool):
        raise TypeError('transpose must be bool')
    A, B, indices = map(jnp.asarray, (A, B, indices))
    if A.ndim == 1:
        A = A[:, None]
    if B.ndim == 1:
        B = B[None, :]
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError('A and B must be vectors or rank-2 matrices')
    if A.shape[1] != B.shape[0]:
        raise ValueError('A and B contraction dimensions must match')
    if indptr is None:
        if indices.ndim != 2 or indices.shape[1] != 2:
            raise ValueError('COO indices must have shape (nse, 2)')
        if not jnp.issubdtype(indices.dtype, jnp.integer):
            raise TypeError('COO indices must be integers')
    else:
        indptr = jnp.asarray(indptr)
        if indices.ndim != 1 or indptr.ndim != 1:
            raise ValueError('CSR indices and indptr must be rank 1')
        if indices.dtype != jnp.int32:
            raise TypeError('indices must be int32 CSR column coordinates')
        if indptr.dtype not in (jnp.int32, jnp.int64):
            raise TypeError('indptr must be int32 or int64 CSR row offsets')
        if indptr.size != A.shape[0] + 1:
            raise ValueError('indptr must match the CSR row count plus one')
        if indices.size and (not A.shape[0] or not B.shape[1]):
            raise ValueError('nonempty CSR slots require positive row and column counts')
        if A.shape[0] > 2**31 - 1:
            raise ValueError('CSR row count exceeds the CUDA int32 limit')
        if indptr.dtype == jnp.int32 and indices.size > 2**31 - 1:
            raise ValueError('int32 indptr cannot address more than INT32_MAX slots')
    if mode != 'float':
        events, ct = (A, B) if transpose else (B, A)
        if ct.dtype not in VALUE_DTYPES:
            raise TypeError('ct must be float16, bfloat16, float32 or float64')
        if not (jnp.issubdtype(events.dtype, jnp.bool_) or jnp.issubdtype(events.dtype, jnp.integer)
                or jnp.issubdtype(events.dtype, jnp.floating)):
            raise TypeError('events must be boolean, integer or floating')
    if A.shape[1] == 1:
        return mv_backend.sampled(A[:, 0], B[0, :], indices, indptr,
                                   mode=mode, transpose=transpose, backend=backend)
    return mm_backend.sampled(A, B, indices, indptr,
                               mode=mode, transpose=transpose, backend=backend)
