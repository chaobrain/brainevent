# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Unified dispatch for sampled dense products and CSR event weight gradients.

CSR adapters in ``brainevent._sddmm`` enter this dispatcher, then the chosen
leaf selects its backend. Operands are ``A(rows, batch)`` and ``B(batch, cols)``:
batch one uses ``_mv``, all other sizes use ``_mm``. Vectors form a batch-one
outer product; batch zero yields zero per slot. An outer ``vmap`` axis is
independent of this contraction dimension.

Patterns are COO coordinates or CSR columns/offsets, retaining slot order and
duplicates. ``sddmm`` returns an array of slot values.
``float`` means ordinary multiplication; ``binary`` selects active cotangents;
``weighted`` multiplies active event amplitudes by cotangents. Event leaves
handle NaN/Inf cotangents without contributions from inactive events.
Weighted CUDA tests activity after casting events to the cotangent dtype;
JAX and binary products use the original event activity.

JAX supports both leaves on CPU/GPU/TPU. GPU auto uses hybrid CUDA for
transposed CSR events in both leaves, with JAX for the other direction.
MM batches larger than 65535 or column counts above INT32_MAX also use JAX
under auto; explicit CUDA requests reject these unsupported dimensions.
Ordinary JAX products retain differentiation and ``shard_map`` support; CSR
event primitives supply other operators' gradients and have no AD rules.
Explicit ``jax_raw`` calls execute JAX directly and retain differentiation for
CSR and COO inputs. CSR event gradient consumers pass their CSR structure and
forward backend selection. Their private adapter retains differentiable JAX
tangents while the primal uses the forward backend's available sampled path;
automatic GPU selection resolves the source CSR primitive at lowering time.
TCSR-specific adapters remain separate.
"""

from functools import partial

import jax
import jax.numpy as jnp

from ._common import Mode, VALUE_DTYPES, _CSRBackend
from ._mv import sddmv as mv_backend
from ._mm import sddmm as mm_backend


def sddmm(
    A: jax.Array, B: jax.Array, indices: jax.Array, indptr: jax.Array | None = None,
    *, mode: Mode = 'float', transpose: bool = True, backend: str | None | _CSRBackend = None,
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
        value-weighted event selection. Weighted CUDA tests activity after
        converting amplitudes to the cotangent dtype. JAX and binary products
        preserve original event activity.
    transpose : bool, optional
        In event modes, ``True`` identifies ``A`` as events and ``B.T`` as
        output cotangents; ``False`` identifies ``B.T`` as events and ``A`` as
        cotangents. It does not transpose the ordinary product ``A @ B``.
    backend : str or None, optional
        Forwarded to the selected leaf. ``jax_raw`` executes differentiable
        JAX directly for both CSR and COO inputs in either leaf.
        GPU auto uses hybrid CUDA for supported transposed CSR event products
        in both leaves. Explicit ``cuda_raw`` requires this direction and CSR
        event mode; other cases use ``jax_raw`` or ``auto``.

    Returns
    -------
    jax.Array
        Shape ``(nse,)`` in physical slot order. Event results have the
        cotangent dtype; ordinary products retain JAX's multiplication dtype.

    Notes
    -----
    Batch one dispatches to ``_mv``; other sizes, including zero, to ``_mm``.
    Ordinary products and explicit ``jax_raw`` event calls remain
    differentiable. CSR event calls using ``auto`` or ``cuda_raw`` bind
    primitives without derivative rules.
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
    if isinstance(backend, _CSRBackend):
        return _csr_sampled(A, B, indices, indptr, mode, transpose, backend)
    return _sampled(A, B, indices, indptr, mode, transpose, backend)


def _sampled(A, B, indices, indptr, mode, transpose, backend):
    """Choose the leaf after the dispatcher has normalized the operands."""
    if A.shape[1] == 1:
        return mv_backend.sampled(A[:, 0], B[0, :], indices, indptr,
                                   mode=mode, transpose=transpose, backend=backend)
    return mm_backend.sampled(A, B, indices, indptr,
                               mode=mode, transpose=transpose, backend=backend)


@partial(jax.custom_jvp, nondiff_argnums=(4, 5, 6))
def _csr_sampled(A, B, indices, indptr, mode, transpose, source):
    """Keep CSR derivatives while its primal follows the forward backend.

    Only a CUDA forward selects the native event gradient. Other explicit
    backends use JAX; an automatic request is resolved by the source CSR
    primitive on the actual lowering platform. Public sampled-event calls
    retain their existing primitive/AD contract.
    """
    native = (mode != 'float' and transpose and
              (A.shape[1] == 1 or (A.shape[1] <= 65535 and B.shape[1] <= 2**31 - 1)))
    if not native or source.requested not in (None, 'cuda_raw'):
        backend = 'jax_raw'
    else:
        backend = 'cuda_raw' if source.requested == 'cuda_raw' else source
    return _sampled(A, B, indices, indptr, mode, transpose, backend)


@_csr_sampled.defjvp
def _csr_sampled_jvp(mode, transpose, source, primals, tangents):
    A, B, indices, indptr = primals
    A_dot, B_dot, _, _ = tangents
    primal = _csr_sampled(A, B, indices, indptr, mode, transpose, source)
    # Preserve the CSR consumers' existing differentiable JAX rules. This
    # runs only under further differentiation, not in a first-order VJP.
    _, tangent = jax.jvp(
        lambda a, b: _sampled(a, b, indices, indptr, mode, transpose, 'jax_raw'),
        (A, B), (A_dot, B_dot),
    )
    return primal, tangent
