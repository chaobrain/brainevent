# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Vector leaf for sampled products and CSR event weight gradients.

Call through ``brainevent._sddmm`` and its unified dispatcher. For a CSR slot
``(row, col)``, transposed products use ``events[row]`` and ``ct[col]``;
nontransposed products use ``events[col]`` and ``ct[row]``. Binary mode selects
the cotangent; weighted mode multiplies it by the event cast to ``ct.dtype``.
Inactive events contribute zero even with NaN/Inf cotangents. Output preserves
physical slot order, duplicates and ``ct.dtype``.

Cotangents support float16/bfloat16/float32/float64; events support bool,
signed/unsigned integers and floating types. CSR columns are int32 and offsets
int32/int64. Enable JAX x64 for 64-bit operands; callers supply valid CSR bounds.
GPU ``auto`` uses hybrid CUDA for transposed CSR events and JAX otherwise.
``cuda_raw`` requires transposed CSR events; ``jax_raw`` supports both
directions on CPU/GPU/TPU. Ordinary outer products use differentiable JAX.

JAX uses sampled multiplication for finite cotangents and exact selection
otherwise. Mixed floating event types with nonfinite cotangents also use exact
selection in the CUDA adapter, preserving activity before narrowing. Each
``.cu`` owns its kernels and FFI bindings; output is zeroed per call, and private
scratch capacity uses the same captured ``HybridConfig`` as compilation.

The CSR event primitive has no AD rule. Consumers handle shared-weight sums
and indexed scatter-add into canonical weight order. Summing selected slot
products avoids overflow from an intermediate unit-weight forward reduction.
Existing CSR MV gradient entry points remain separate. Regression coverage is
in ``test/sddmv_test.py`` (types, NaN/Inf, empty/heavy rows, vmap and shard_map).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.experimental import sparse

from brainevent._compatible_import import match_varying_axes, varying_axes_of
from brainevent._misc import _csr_to_coo
from brainevent._op import XLACustomKernel, load_cuda_file
from brainevent._op.util import event_active, select_active
from .._common import Mode, check_jax_backend, coordinates

if TYPE_CHECKING:
    from brainevent._csr.binary.hybrid_config import HybridConfig


def get_hybrid_config() -> HybridConfig:
    """Resolve CSR tuning lazily, avoiding the CSR/package import cycle."""
    from brainevent._csr.binary.hybrid_config import get_hybrid_config as resolve
    return resolve()


_DIRECTORY = Path(__file__).parent
_VALUE_SUFFIXES = {
    jnp.dtype(jnp.float16): 'f16', jnp.dtype(jnp.bfloat16): 'bf16',
    jnp.dtype(jnp.float32): 'f32', jnp.dtype(jnp.float64): 'f64',
}
_BINARY_TARGET_SUFFIXES = {
    (value_dtype, event_dtype): f'{suffix}_{event_suffix}'
    for value_dtype, suffix in _VALUE_SUFFIXES.items()
    for event_dtype, event_suffix in [(jnp.dtype(jnp.bool_), 'bool'), (jnp.dtype(jnp.float32), 'float')]
}
_BINARY_TARGET_SUFFIXES.update({
    (jnp.dtype(jnp.float64), jnp.dtype(jnp.int8)): 'f64_int8',
    (jnp.dtype(jnp.float64), jnp.dtype(jnp.float64)): 'f64_double',
})


@lru_cache(maxsize=None)
def _load_hybrid(weighted: bool, config: HybridConfig) -> str:
    from brainevent._csr.binary.hybrid_config import compile_flags_for_config, module_suffix_for_config
    kind = 'weighted' if weighted else 'binary'
    source = 'hybrid_sddmv_prezeroed_weighted.cu' if weighted else 'hybrid_sddmv_prezeroed.cu'
    name = f'csr_sddmv_hybrid_{kind}' + module_suffix_for_config(config)
    load_cuda_file(
        _DIRECTORY / source, name=name,
        extra_cuda_cflags=compile_flags_for_config(config),
        allow_cuda_graph=False,
    )
    return name


def _exact(events: jax.Array, ct: jax.Array, indices: jax.Array, indptr: jax.Array,
           *, transpose: bool, weighted: bool) -> jax.Array:
    """Select original event activity before reducing or narrowing amplitudes."""
    row, col = _csr_to_coo(indices, indptr)
    unit, out = (row, col) if transpose else (col, row)
    x = events[unit]
    value = x.astype(ct.dtype) * ct[out] if weighted else ct[out]
    return select_active(event_active(x), value)


def _sampled_jax(A: jax.Array, B: jax.Array, rows: jax.Array, cols: jax.Array) -> jax.Array:
    """Rank-one sampled outer product, retaining JAX derivatives and fill rules."""
    indices = jnp.stack([rows, cols], axis=1)
    if varying_axes_of(A, B, indices):
        return (A.at[rows].get(mode='fill', fill_value=0)
                * B.at[cols].get(mode='fill', fill_value=0))
    return sparse.bcoo_dot_general_sampled(
        A, B, indices, dimension_numbers=(((), ()), ((), ())),
    )


def _event_jax(A: jax.Array, B: jax.Array, rows: jax.Array, cols: jax.Array,
               *, transpose: bool, mode: Mode) -> jax.Array:
    events, ct = (A, B) if transpose else (B, A)
    unit, out = (rows, cols) if transpose else (cols, rows)

    def sampled(ct):
        factors = events.astype(ct.dtype) if mode == 'weighted' else event_active(events).astype(ct.dtype)
        left, right = (factors, ct) if transpose else (ct, factors)
        return _sampled_jax(left, right, rows, cols)

    def selected(ct):
        x = events.at[unit].get(mode='fill', fill_value=0)
        value = ct.at[out].get(mode='fill', fill_value=0)
        if mode == 'weighted':
            value = x.astype(ct.dtype) * value
        return select_active(event_active(x), value)

    return jax.lax.cond(jnp.all(jnp.isfinite(ct)), sampled, selected, ct)


def _jax_kernel(*, transpose: bool, weighted: bool, **kwargs: Any):
    """CSR event primitive lowered directly to this leaf's JAX implementation."""
    def kernel(events, ct, indices, indptr):
        if not indices.size:
            return (jnp.zeros((0,), ct.dtype),)
        row, col = coordinates(indices, indptr)
        A, B = (events, ct) if transpose else (ct, events)
        return (_event_jax(A, B, row, col, transpose=transpose,
                           mode='weighted' if weighted else 'binary'),)
    return kernel


def _cuda_kernel(*, transpose: bool, weighted: bool, config: HybridConfig,
                 ct_info: jax.ShapeDtypeStruct, indices_info: jax.ShapeDtypeStruct,
                 indptr_info: jax.ShapeDtypeStruct, **kwargs: Any):
    if not transpose:
        raise ValueError('cuda_raw SDDMV requires transpose=True; use jax_raw or auto')
    nnz, rows = indices_info.size, indptr_info.size - 1
    if not nnz:
        return lambda events, ct, indices, indptr: (jnp.zeros((0,), ct.dtype),)
    from brainevent._csr.binary.hybrid_config import hybrid_capacity_bound
    capacity = hybrid_capacity_bound(nnz, rows, config=config)
    module = _load_hybrid(weighted, config)
    value_suffix = _VALUE_SUFFIXES[jnp.dtype(ct_info.dtype)]

    def kernel(events, ct, indices, indptr):
        def hybrid(events, ct, indices, indptr):
            if weighted:
                vector = events.astype(ct.dtype)
                target = f'{module}.hybrid_sddmv_weighted_{value_suffix}'
            else:
                vector = events
                pair = (jnp.dtype(ct.dtype), jnp.dtype(vector.dtype))
                if pair not in _BINARY_TARGET_SUFFIXES:
                    vector = event_active(vector)
                suffix = _BINARY_TARGET_SUFFIXES[(jnp.dtype(ct.dtype), jnp.dtype(vector.dtype))]
                target = f'{module}.hybrid_sddmv_binary_{suffix}'
            infos = (
                jax.ShapeDtypeStruct((nnz,), ct.dtype),
                jax.ShapeDtypeStruct((capacity,), indptr.dtype),
                jax.ShapeDtypeStruct((capacity,), indptr.dtype),
                jax.ShapeDtypeStruct((2,), jnp.int32),
            )
            # Only private, per-call output/scratch operands are aliased.
            zeroed = jnp.zeros((nnz,), ct.dtype)
            begin = jnp.empty((capacity,), indptr.dtype)
            end = jnp.empty((capacity,), indptr.dtype)
            status = jnp.empty((2,), jnp.int32)
            out = jax.ffi.ffi_call(
                target, infos, input_output_aliases={4: 0, 5: 1, 6: 2, 7: 3},
                vmap_method='sequential',
            )(ct, indices, indptr, vector, zeroed, begin, end, status,
              task_capacity=capacity)[0]
            return match_varying_axes(out, events, ct, indices, indptr)

        if weighted and events.dtype != ct.dtype and jnp.issubdtype(events.dtype, jnp.floating):
            # A tiny active event can round to zero in ct.dtype. The native
            # gate then differs from the original mask for NaN/Inf ct only.
            out = jax.lax.cond(
                jnp.all(jnp.isfinite(ct)), hybrid,
                lambda e, c, i, p: _exact(e, c, i, p, transpose=True, weighted=True),
                events, ct, indices, indptr,
            )
        else:
            out = hybrid(events, ct, indices, indptr)
        return (out,)
    return kernel


def _auto_gpu_kernel(*, transpose: bool, **kwargs: Any):
    factory = _cuda_kernel if transpose else _jax_kernel
    return factory(transpose=transpose, **kwargs)


def _csr_call(events, ct, indices, indptr, *, transpose, weighted, backend):
    """Bind normalized CSR events; validation belongs to the dispatcher."""
    return csr_sddmv_p(
        events, ct, indices, indptr,
        outs=[jax.ShapeDtypeStruct(indices.shape, ct.dtype)],
        transpose=transpose, weighted=weighted, backend=backend,
        ct_info=jax.ShapeDtypeStruct(ct.shape, ct.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        config=get_hybrid_config(),
    )[0]


def sampled(A: jax.Array, B: jax.Array, indices: jax.Array, indptr: jax.Array | None,
            *, mode: Mode, transpose: bool, backend: str | None) -> jax.Array:
    """Execute a normalized rank-one sampled product.

    Parameters
    ----------
    A, B : jax.Array
        Left and right vectors of a sampled outer product.
    indices, indptr : jax.Array or None
        COO coordinates, or CSR columns and row offsets.
    mode : {'float', 'binary', 'weighted'}
        Ordinary multiplication or event-selection semantics.
    transpose : bool
        Events are on the left when true, on the right otherwise.
    backend : str or None
        JAX supports all forms; CUDA supports transposed CSR events.

    Returns
    -------
    jax.Array
        One value per physical sparse slot.
    """
    check_jax_backend(backend)
    if mode != 'float' and indptr is not None:
        if backend == 'cuda_raw' and not transpose:
            raise ValueError('cuda_raw SDDMV requires transpose=True; use jax_raw or auto')
        events, ct = (A, B) if transpose else (B, A)
        return _csr_call(events, ct, indices, indptr, transpose=transpose,
                         weighted=mode == 'weighted', backend=backend)
    if backend == 'cuda_raw':
        raise NotImplementedError('MV CUDA supports transposed CSR event products; use backend="jax_raw"')
    rows, cols = coordinates(indices, indptr)
    if mode == 'float':
        return _sampled_jax(A, B, rows, cols)
    return _event_jax(A, B, rows, cols, transpose=transpose, mode=mode)


csr_sddmv_p = XLACustomKernel('csr_sddmv')
csr_sddmv_p.def_cuda_raw_kernel(_cuda_kernel)
csr_sddmv_p.def_kernel('auto', 'gpu', _auto_gpu_kernel, asdefault=True)
for _platform in ('cpu', 'gpu', 'tpu'):
    csr_sddmv_p.def_kernel('jax_raw', _platform, _jax_kernel)
for _platform in ('cpu', 'tpu'):
    csr_sddmv_p.def_kernel('auto', _platform, _jax_kernel)
csr_sddmv_p.def_tags('csr', 'sddmv')
