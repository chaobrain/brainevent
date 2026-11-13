# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

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
Weighted products test activity after casting events to the cotangent dtype
on every backend; binary products use the original event activity.

JAX supports both leaves on CPU/GPU/TPU. GPU auto uses hybrid CUDA for
transposed CSR events in both leaves, with JAX for the other direction.
MM batches larger than 65535 or column counts above INT32_MAX also use JAX
under auto; explicit CUDA requests reject these unsupported dimensions.
Explicit ``cusparse`` uses a shared cuSPARSE SDDMM FFI for CSR products on GPU
in both directions and all modes; automatic backend selection is unchanged.
Every product is differentiable and supports ``shard_map``. Explicit
``jax_raw`` calls and other pure JAX products differentiate directly. Calls
that bind a CSR event primitive or the cuSPARSE FFI, which have no AD rules of
their own, run the primal on that backend and take derivatives from the JAX
sampled product. CSR event gradient consumers pass their CSR structure and
forward backend selection; the primal then uses the forward backend's
available sampled path, including cuSPARSE. Automatic GPU selection resolves
the source CSR primitive at lowering time, honoring its global and
per-primitive backend choices. TCSR-specific adapters remain separate.
"""

from functools import partial

import jax
import jax.numpy as jnp

from brainevent._misc import _csr_to_coo
from brainevent._op.util import CUDA_VALUE_DTYPES
from brainevent._sddmm._common import Mode, _CSRBackend, _cusparse_sampled, check_jax_backend
from brainevent._sddmm._mv import sddmv as mv_backend
from brainevent._sddmm._mm import sddmm as mm_backend


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
        value-weighted event selection. Weighted products test activity after
        converting amplitudes to the cotangent dtype on every backend; binary
        products use the original event activity.
    transpose : bool, optional
        In event modes, ``True`` identifies ``A`` as events and ``B.T`` as
        output cotangents; ``False`` identifies ``B.T`` as events and ``A`` as
        cotangents. It does not transpose the ordinary product ``A @ B``.
    backend : str, None or _CSRBackend, optional
        Forwarded to the selected leaf. A ``_CSRBackend`` carries a CSR
        consumer's forward backend and scheduler snapshot; ``None`` uses
        default scheduler constants without reading persisted configuration.
        ``jax_raw`` executes differentiable
        JAX directly for both CSR and COO inputs in either leaf.
        GPU auto uses hybrid CUDA for supported transposed CSR event products
        in both leaves. Explicit ``cuda_raw`` requires this direction and CSR
        event mode; other cases use ``jax_raw`` or ``auto``.
        Explicit ``cusparse`` supports CSR products on GPU in either direction
        and all modes. Half types compute in float32. COO and patterns with
        more slots than rows times columns are unsupported. Nonfinite event
        cotangents use exact JAX selection; weighted activity is tested after
        casting to the cotangent dtype.

    Returns
    -------
    jax.Array
        Shape ``(nse,)`` in physical slot order. Event results have the
        cotangent dtype; ordinary products retain JAX's multiplication dtype.

    Notes
    -----
    Batch one dispatches to ``_mv``; other sizes, including zero, to ``_mm``.
    Every backend is differentiable to any order. CSR event calls using
    ``None``, ``auto`` or ``cuda_raw`` and explicit ``cusparse`` calls run the
    primal on the selected backend and take derivatives from the JAX sampled
    product; ``jax_raw`` and other JAX products differentiate directly.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import jax.numpy as jnp
        >>> from brainevent._sddmm import sddmm
        >>> events = jnp.array([1., 0.])
        >>> ct = jnp.array([2., 3.])
        >>> indices, indptr = jnp.array([0, 1], jnp.int32), jnp.array([0, 1, 2], jnp.int32)
        >>> sddmm(events, ct, indices, indptr, mode='weighted')
        Array([2., 0.], dtype=float32)
        >>> jax.grad(lambda c: sddmm(events, c, indices, indptr, mode='weighted').sum())(ct)
        Array([1., 0.], dtype=float32)
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
        if ct.dtype not in CUDA_VALUE_DTYPES:
            raise TypeError('ct must be float16, bfloat16, float32 or float64')
        if not (jnp.issubdtype(events.dtype, jnp.bool_) or jnp.issubdtype(events.dtype, jnp.integer)
                or jnp.issubdtype(events.dtype, jnp.floating)):
            raise TypeError('events must be boolean, integer or floating')
    if isinstance(backend, _CSRBackend):
        return _csr_sampled(A, B, indices, indptr, mode, transpose, backend)
    if indptr is not None and backend != 'jax_raw' and (mode != 'float' or backend == 'cusparse'):
        # This call binds a native primitive or the cuSPARSE FFI.
        return _native_sampled(A, B, indices, indptr, mode, transpose, backend, None)
    return _sampled(A, B, indices, indptr, mode, transpose, backend)


def _leaf(A):
    """Batch one uses the vector leaf; every other size the matrix leaf."""
    return mv_backend if A.shape[1] == 1 else mm_backend


def _sampled(A, B, indices, indptr, mode, transpose, backend, *, config_spec=None):
    """Execute a normalized ``A (rows, batch)``, ``B (batch, cols)`` product on its leaf.

    CSR event products bind the leaf primitive unless ``jax_raw`` is explicit,
    so explicit JAX calls stay differentiable. Other event products run the
    leaf's JAX kernel; ordinary products sample at COO coordinates.
    """
    check_jax_backend(backend)
    leaf = _leaf(A)
    if backend == 'cusparse':
        if indptr is None:
            raise NotImplementedError('cusparse sampled products require CSR indices and indptr')
        if mode == 'float':
            return _cusparse_sampled(A, B, indices, indptr, mode=mode, transpose=transpose)
    if mode != 'float':
        events, ct = (A, B.T) if transpose else (B.T, A)
        if leaf is mv_backend:
            events, ct = events[:, 0], ct[:, 0]
        weighted = mode == 'weighted'
        if indptr is not None and backend != 'jax_raw':
            if backend == 'cuda_raw' and not transpose:
                raise ValueError(f'cuda_raw {leaf.NAME} requires transpose=True; use jax_raw or auto')
            return leaf.bind_events(events, ct, indices, indptr, transpose=transpose,
                                    weighted=weighted, backend=backend, config_spec=config_spec)
        if backend != 'cuda_raw':
            return leaf._jax_kernel(events, ct, indices, indptr, transpose=transpose, weighted=weighted)[0]
    if backend == 'cuda_raw':
        raise NotImplementedError(
            f'{leaf.LABEL} CUDA supports transposed CSR event products; use backend="jax_raw"')
    if indptr is not None:
        rows, cols = _csr_to_coo(indices, indptr)
        indices = jnp.stack([rows, cols], axis=1)
    return leaf.float_product(A, B, indices)


def _csr_sampled(A, B, indices, indptr, mode, transpose, source):
    """Run a CSR consumer's sampled product on its forward backend.

    Explicit cuSPARSE requests keep their backend. CUDA forwards select native
    event gradients where supported; other explicit backends use JAX.
    Automatic requests resolve the source CSR primitive on the actual lowering
    platform, including its cuSPARSE choice.
    """
    native = mode != 'float' and transpose and _leaf(A).native_dims((B.shape[1], A.shape[1]))
    if source.requested == 'cusparse':
        backend = 'cusparse'
    elif source.requested is None and mode != 'float':
        backend = source
    elif native and source.requested == 'cuda_raw':
        backend = 'cuda_raw'
    else:
        return _sampled(A, B, indices, indptr, mode, transpose, 'jax_raw')
    return _native_sampled(A, B, indices, indptr, mode, transpose, backend, source.config_spec)


@partial(jax.custom_jvp, nondiff_argnums=(4, 5, 6, 7))
def _native_sampled(A, B, indices, indptr, mode, transpose, backend, config_spec):
    """Run a native sampled product whose derivatives are the JAX product's.

    Native event primitives and the cuSPARSE FFI have no AD rules of their
    own. The primal keeps the selected backend; the tangent is the JAX
    sampled product's, which is linear in the tangents and differentiable
    again, so reverse mode and higher orders follow.
    """
    return _sampled(A, B, indices, indptr, mode, transpose, backend, config_spec=config_spec)


@_native_sampled.defjvp
def _native_sampled_jvp(mode, transpose, backend, config_spec, primals, tangents):
    A, B, indices, indptr = primals
    A_dot, B_dot, _, _ = tangents
    primal = _native_sampled(A, B, indices, indptr, mode, transpose, backend, config_spec)
    _, tangent = jax.jvp(
        lambda a, b: _sampled(a, b, indices, indptr, mode, transpose, 'jax_raw'),
        (A, B), (A_dot, B_dot),
    )
    return primal, tangent
