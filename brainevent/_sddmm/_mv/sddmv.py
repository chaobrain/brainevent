# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

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
directions on CPU/GPU/TPU. Ordinary outer products default to differentiable JAX.
Explicit ``cusparse`` uses the shared cuSPARSE SDDMM FFI for CSR products in
either direction, including ordinary outer products. Half types compute in
float32; weighted activity is tested after casting, as with native CUDA.

JAX masks each slot's cotangent where its event factor is zero, without a
branch on global finiteness. Every backend tests weighted activity after
casting events to the cotangent dtype, so values cast to zero are skipped. Each
``.cu`` owns its kernels and FFI bindings. The output dW is created and zeroed
in JAX before each FFI call (the kernels write only active rows), and private
scratch capacity uses the same captured CSR scheduler snapshot as compilation.

The CSR event primitive has no AD rule; the dispatcher wraps native calls in
a custom JVP taken from the JAX product. Consumers handle shared-weight sums
and indexed scatter-add into canonical weight order. Summing selected slot
products avoids overflow from an intermediate unit-weight forward reduction.
Explicit ``jax_raw`` calls execute JAX directly and retain differentiation
for both CSR and COO inputs; CSR consumers pass their original structure.
Regression coverage is in ``sddmv_test.py`` (types, NaN/Inf, empty/heavy rows,
vmap and shard_map).
"""

from __future__ import annotations

from functools import lru_cache, partial
from pathlib import Path
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.experimental import sparse

from brainevent._compatible_import import match_varying_axes, varying_axes_of
from brainevent._misc import _csr_to_coo
from brainevent._op import XLACustomKernel, load_cuda_file
from brainevent._op.util import event_active
from brainevent._sddmm._common import (
    BINARY_TARGET_SUFFIXES, VALUE_SUFFIXES, Mode, _CSRBackend, check_jax_backend,
    _cusparse_event_kernel, _cusparse_sampled,
)

if TYPE_CHECKING:
    from brainevent._csr.config import CSRKernelSpec


_DIRECTORY = Path(__file__).parent


def _load_hybrid(weighted: bool, config_spec: CSRKernelSpec) -> str:
    """Share compiled modules across structures with the same scheduler."""
    return _load_hybrid_cached(
        bool(weighted), config_spec.block_size, config_spec.fixed_scatter_blocks,
        config_spec.tpr_threshold, config_spec.task_nnz,
    )


@lru_cache(maxsize=128)
def _load_hybrid_cached(weighted: bool, block_size: int, fixed_scatter_blocks: int,
                        tpr_threshold: int, task_nnz: int) -> str:
    from brainevent._csr.config import CSRKernelSpec
    config_spec = CSRKernelSpec(block_size, fixed_scatter_blocks, tpr_threshold, task_nnz)
    kind = 'weighted' if weighted else 'binary'
    source = f'hybrid_sddmv_{kind}.cu'
    name = f'csr_sddmv_hybrid_{kind}' + config_spec.module_suffix()
    load_cuda_file(
        _DIRECTORY / source, name=name,
        extra_cuda_cflags=config_spec.compile_flags(),
        extra_include_paths=[str(_DIRECTORY)],
        allow_cuda_graph=True,
    )
    return name


def _jax_kernel(events: jax.Array, ct: jax.Array, indices: jax.Array,
                indptr: jax.Array | None, *, transpose: bool, weighted: bool,
                **kwargs: Any) -> tuple[jax.Array]:
    """Compute event slot values for CSR lowering or direct CSR/COO calls.

    Branch-free: each slot masks its own cotangent instead of testing the
    whole cotangent for finiteness first. Weighted activity is tested after
    the cast to ``ct.dtype``, as in the native kernels.
    """
    if indptr is not None:
        rows, cols = _csr_to_coo(indices, indptr)
    else:
        rows, cols = indices[:, 0], indices[:, 1]
    unit, out = (rows, cols) if transpose else (cols, rows)
    factors = events.astype(ct.dtype) if weighted else event_active(events).astype(ct.dtype)
    gather = partial(_gather_slots, csr=indptr is not None)
    return (_masked_product(gather(factors, unit), gather(ct, out)),)


def _gather_slots(values: jax.Array, index: jax.Array, *, csr: bool) -> jax.Array:
    """Gather one leading-axis entry per slot; out-of-range slots read zero.

    CSR coordinates are non-negative, so their gathers skip the negative-index
    wrap, a per-slot select that otherwise costs about a quarter of the
    kernel. COO keeps NumPy's wrapping of negative coordinates.
    """
    return values.at[index].get(mode='fill', fill_value=0, wrap_negative_indices=not csr)


def _masked_product(factor: jax.Array, ct: jax.Array) -> jax.Array:
    """Multiply slot factors by cotangents; zero factors drop NaN/Inf cotangents.

    For finite cotangents this is the plain product (including its sign of
    zero), and its derivative with respect to ``factor`` stays ``ct``.
    """
    return factor * jnp.where((factor != 0) | jnp.isfinite(ct), ct, jnp.zeros_like(ct))


def _cuda_kernel(*, transpose: bool, weighted: bool, config_spec: CSRKernelSpec,
                 ct_info: jax.ShapeDtypeStruct, indices_info: jax.ShapeDtypeStruct,
                 indptr_info: jax.ShapeDtypeStruct, **kwargs: Any):
    if not transpose:
        raise ValueError('cuda_raw SDDMV requires transpose=True; use jax_raw or auto')
    nnz, rows = indices_info.size, indptr_info.size - 1
    if not nnz:
        return lambda events, ct, indices, indptr: (jnp.zeros((0,), ct.dtype),)
    capacity = config_spec.mv_capacity(nnz, rows)
    module = _load_hybrid(weighted, config_spec)
    value_dtype, offset_dtype = jnp.dtype(ct_info.dtype), jnp.dtype(indptr_info.dtype)
    infos = [
        jax.ShapeDtypeStruct((nnz,), value_dtype),  # dW
        jax.ShapeDtypeStruct((capacity,), offset_dtype),
        jax.ShapeDtypeStruct((capacity,), offset_dtype),
    ]
    if weighted:
        infos.append(jax.ShapeDtypeStruct((capacity,), value_dtype))
    infos.append(jax.ShapeDtypeStruct((2,), jnp.int32))

    def kernel(events, ct, indices, indptr):
        if weighted:
            vector = events.astype(value_dtype)
            target = f'{module}.hybrid_sddmv_weighted_{VALUE_SUFFIXES[value_dtype]}'
        else:
            vector = events
            if (value_dtype, jnp.dtype(vector.dtype)) not in BINARY_TARGET_SUFFIXES:
                vector = event_active(vector)
            target = f'{module}.hybrid_sddmv_binary_{BINARY_TARGET_SUFFIXES[(value_dtype, jnp.dtype(vector.dtype))]}'
        # Kernels write only active slots, so dW must arrive zeroed.
        # Operand 4 is aliased to output 0.
        dweight = jnp.zeros((nnz,), value_dtype)
        # Task/status scratch is private per call; the kernel resets status.
        scratch = [jnp.empty(info.shape, info.dtype) for info in infos[1:]]
        out = jax.ffi.ffi_call(
            target, infos, input_output_aliases={4 + k: k for k in range(len(infos))},
            vmap_method='sequential',
        )(ct, indices, indptr, vector, dweight, *scratch,
          task_capacity=capacity)[0]
        return (match_varying_axes(out, events, ct, indices, indptr),)
    return kernel


def _auto_gpu_kernel(*, transpose: bool, csr_primitive=None, **kwargs: Any):
    backend = None if csr_primitive is None else csr_primitive._resolve_backend('gpu')
    if backend == 'cusparse':
        return _cusparse_event_kernel(transpose=transpose, **kwargs)
    native = backend in (None, 'cuda_raw')
    if transpose and native:
        return _cuda_kernel(transpose=transpose, **kwargs)
    return partial(_jax_kernel, transpose=transpose, **kwargs)


def _csr_call(events, ct, indices, indptr, *, transpose, weighted, backend, config_spec=None):
    """Bind normalized CSR events; validation belongs to the dispatcher."""
    source = backend.primitive if isinstance(backend, _CSRBackend) else None
    if config_spec is None and isinstance(backend, _CSRBackend):
        config_spec = backend.config_spec
    if config_spec is None:
        from brainevent._csr.config import CSRConfig
        config_spec = CSRConfig().kernel_args()[0]
    return csr_sddmv_p(
        events, ct, indices, indptr,
        outs=[jax.ShapeDtypeStruct(indices.shape, ct.dtype)],
        transpose=transpose, weighted=weighted, backend='auto' if source is not None else backend,
        csr_primitive=source,
        ct_info=jax.ShapeDtypeStruct(ct.shape, ct.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        config_spec=config_spec,
    )[0]


def sampled(A: jax.Array, B: jax.Array, indices: jax.Array, indptr: jax.Array | None,
            *, mode: Mode, transpose: bool, backend: str | None | _CSRBackend,
            config_spec=None) -> jax.Array:
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
        ``cusparse`` supports CSR products on GPU in both directions and all
        modes, with float32 computation for half types. COO and patterns with
        more slots than rows times columns are unsupported. Nonfinite event
        cotangents use exact JAX selection. Native backends have no AD rules
        here; :func:`brainevent._sddmm.sddmm` supplies them.
    config_spec : CSRKernelSpec or None, optional
        Explicit scheduler metadata. Defaults never read persisted settings.

    Returns
    -------
    jax.Array
        One value per physical sparse slot.
    """
    check_jax_backend(backend)
    if backend == 'cusparse':
        if indptr is None:
            raise NotImplementedError('cusparse sampled products require CSR indices and indptr')
        if mode == 'float':
            return _cusparse_sampled(A[:, None], B[None, :], indices, indptr,
                                      mode=mode, transpose=transpose)
    # Explicit JAX calls stay differentiable without binding the event primitive.
    if mode != 'float' and indptr is not None and backend != 'jax_raw':
        if backend == 'cuda_raw' and not transpose:
            raise ValueError('cuda_raw SDDMV requires transpose=True; use jax_raw or auto')
        events, ct = (A, B) if transpose else (B, A)
        return _csr_call(events, ct, indices, indptr, transpose=transpose,
                         weighted=mode == 'weighted', backend=backend, config_spec=config_spec)
    if backend == 'cuda_raw':
        raise NotImplementedError('MV CUDA supports transposed CSR event products; use backend="jax_raw"')
    if mode != 'float':
        events, ct = (A, B) if transpose else (B, A)
        return _jax_kernel(events, ct, indices, indptr,
                           transpose=transpose, weighted=mode == 'weighted')[0]
    if indptr is not None:
        rows, cols = _csr_to_coo(indices, indptr)
        indices = jnp.stack([rows, cols], axis=1)
    if varying_axes_of(A, B, indices):
        return (A.at[indices[:, 0]].get(mode='fill', fill_value=0)
                * B.at[indices[:, 1]].get(mode='fill', fill_value=0))
    return sparse.bcoo_dot_general_sampled(
        A, B, indices, dimension_numbers=(((), ()), ((), ())),
    )


csr_sddmv_p = XLACustomKernel('csr_sddmv')
csr_sddmv_p.def_cuda_raw_kernel(_cuda_kernel)
csr_sddmv_p.def_kernel('cusparse', 'gpu', _cusparse_event_kernel)
csr_sddmv_p.def_kernel('auto', 'gpu', _auto_gpu_kernel, asdefault=True)
for _platform in ('cpu', 'gpu', 'tpu'):
    csr_sddmv_p.def_kernel('jax_raw', _platform, lambda **kwargs: partial(_jax_kernel, **kwargs))
for _platform in ('cpu', 'tpu'):
    csr_sddmv_p.def_kernel('auto', _platform, lambda **kwargs: partial(_jax_kernel, **kwargs))
csr_sddmv_p.def_tags('csr', 'sddmv')
