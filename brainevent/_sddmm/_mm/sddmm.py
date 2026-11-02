# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Callable matrix SDDMM with a JAX backend and a future CUDA entry point."""

from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import sparse

from brainevent._compatible_import import varying_axes_of
from brainevent._op import XLACustomKernel
from brainevent._op.util import event_active, select_active
from .._common import Mode, check_jax_backend, coordinates


def _sampled_jax(A: jax.Array, B: jax.Array, rows: jax.Array, cols: jax.Array) -> jax.Array:
    """The old generic SDDMM implementation, including per-shard derivatives."""
    indices = jnp.stack([rows, cols], axis=1)
    if varying_axes_of(A, B, indices):
        left = A.at[rows, :].get(mode='fill', fill_value=0)
        right = B.at[:, cols].get(mode='fill', fill_value=0)
        return jax.lax.dot_general(
            left, right, dimension_numbers=(((1,), (0,)), ((0,), (1,))),
            precision=jax.lax.Precision.HIGHEST,
        )
    return sparse.bcoo_dot_general_sampled(
        A, B, indices, dimension_numbers=(((1,), (0,)), ((), ())),
    )


def _event_jax(A: jax.Array, B: jax.Array, rows: jax.Array, cols: jax.Array,
               *, transpose: bool, mode: Mode) -> jax.Array:
    events, ct = (A, B.T) if transpose else (B.T, A)
    unit, out = (rows, cols) if transpose else (cols, rows)

    def sampled(ct):
        factors = events.astype(ct.dtype) if mode == 'weighted' else event_active(events).astype(ct.dtype)
        left, right = (factors, ct.T) if transpose else (ct, factors.T)
        return _sampled_jax(left, right, rows, cols)

    def selected(ct):
        x = events.at[unit, :].get(mode='fill', fill_value=0)
        values = ct.at[out, :].get(mode='fill', fill_value=0)
        if mode == 'weighted':
            values = x.astype(ct.dtype) * values
        return jnp.sum(select_active(event_active(x), values), axis=1)

    return jax.lax.cond(jnp.all(jnp.isfinite(ct)), sampled, selected, ct)


def _jax_kernel(*, transpose: bool, weighted: bool, **kwargs: Any):
    """Lower the CSR event primitive to the same JAX sampled calculation."""
    def kernel(events, ct, indices, indptr):
        if events.shape[1] == 0 or indices.size == 0:
            return (jnp.zeros(indices.shape, ct.dtype),)
        rows, cols = coordinates(indices, indptr)
        A, B = (events, ct.T) if transpose else (ct, events.T)
        return (_event_jax(A, B, rows, cols, transpose=transpose,
                           mode='weighted' if weighted else 'binary'),)
    return kernel


def _cuda_kernel(**kwargs: Any):
    """Reserved for the future matrix CUDA kernel; not registered yet."""
    raise NotImplementedError('matrix SDDMM CUDA backend is not implemented; use backend="jax_raw"')


def sampled(A: jax.Array, B: jax.Array, indices: jax.Array, indptr: jax.Array | None,
            *, mode: Mode, transpose: bool, backend: str | None) -> jax.Array:
    """Execute a normalized matrix sampled product.

    Parameters
    ----------
    A, B : jax.Array
        Dense operands of shape ``(rows, batch)`` and ``(batch, cols)``.
    indices, indptr : jax.Array or None
        COO coordinates, or CSR columns and row offsets.
    mode : {'float', 'binary', 'weighted'}
        Ordinary multiplication or event selection semantics.
    transpose : bool
        Events are on the left when true and on the right otherwise.
    backend : str or None
        ``None``, ``auto`` and ``jax_raw`` use JAX. CUDA is a future extension.

    Returns
    -------
    jax.Array
        One result per physical sparse slot.
    """
    check_jax_backend(backend)
    if backend == 'cuda_raw':
        return _cuda_kernel()
    if mode != 'float' and indptr is not None:
        events, ct = (A, B.T) if transpose else (B.T, A)
        return csr_sddmm_p(
            events, ct, indices, indptr,
            outs=[jax.ShapeDtypeStruct(indices.shape, ct.dtype)],
            transpose=transpose, weighted=mode == 'weighted', backend='jax_raw',
        )[0]
    rows, cols = coordinates(indices, indptr)
    if mode == 'float':
        return _sampled_jax(A, B, rows, cols)
    return _event_jax(A, B, rows, cols, transpose=transpose, mode=mode)


csr_sddmm_p = XLACustomKernel('csr_sddmm')
for _platform in ('cpu', 'gpu', 'tpu'):
    csr_sddmm_p.def_kernel('jax_raw', _platform, _jax_kernel)
csr_sddmm_p.def_tags('csr', 'sddmm')
