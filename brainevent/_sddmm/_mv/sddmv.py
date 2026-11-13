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
from brainevent._op import load_cuda_file
from brainevent._op.util import event_factor
from brainevent._sddmm import _common
from brainevent._sddmm._common import native_event_target

if TYPE_CHECKING:
    from brainevent._csr.config import CSRKernelSpec, MVSchedule


_DIRECTORY = Path(__file__).parent
NAME, LABEL = 'SDDMV', 'MV'


def _load_hybrid(weighted: bool, config_spec: CSRKernelSpec) -> str:
    """Share compiled modules across structures with the same scheduler."""
    return _load_hybrid_cached(bool(weighted), config_spec.mv)


@lru_cache(maxsize=128)
def _load_hybrid_cached(weighted: bool, schedule: MVSchedule) -> str:
    kind = 'weighted' if weighted else 'binary'
    name = f'csr_sddmv_hybrid_{kind}' + schedule.module_suffix()
    load_cuda_file(
        _DIRECTORY / f'hybrid_sddmv_{kind}.cu', name=name,
        extra_cuda_cflags=schedule.compile_flags(),
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
    factors = event_factor(events, dtype=ct.dtype, weighted=weighted)
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
        target, vector = native_event_target(
            f'hybrid_sddmv_{"weighted" if weighted else "binary"}', value_dtype, events, weighted=weighted)
        # Kernels write only active slots, so dW must arrive zeroed.
        # Operand 4 is aliased to output 0.
        dweight = jnp.zeros((nnz,), value_dtype)
        # Task/status scratch is private per call; the kernel resets status.
        scratch = [jnp.empty(info.shape, info.dtype) for info in infos[1:]]
        out = jax.ffi.ffi_call(
            f'{module}.{target}', infos, input_output_aliases={4 + k: k for k in range(len(infos))},
            vmap_method='sequential',
        )(ct, indices, indptr, vector, dweight, *scratch,
          task_capacity=capacity)[0]
        return (match_varying_axes(out, events, ct, indices, indptr),)
    return kernel


def native_dims(ct_shape: tuple[int, ...]) -> bool:
    """Every rank-one cotangent fits the native CUDA kernel."""
    return True


def bind_events(events: jax.Array, ct: jax.Array, indices: jax.Array, indptr: jax.Array, *,
                config_spec: CSRKernelSpec | None = None, **kwargs: Any) -> jax.Array:
    """Bind rank-one CSR events, defaulting to the unpersisted scheduler constants."""
    if config_spec is None:
        from brainevent._csr.config import CSRConfig
        config_spec = CSRConfig().kernel_args()[0]
    return _common.bind_events(csr_sddmv_p, events, ct, indices, indptr,
                               config_spec=config_spec, **kwargs)


def float_product(A: jax.Array, B: jax.Array, indices: jax.Array) -> jax.Array:
    """Sample the outer product of ``A (rows, 1)`` and ``B (1, cols)`` at COO coordinates."""
    A, B = A[:, 0], B[0, :]
    if varying_axes_of(A, B, indices):
        return (A.at[indices[:, 0]].get(mode='fill', fill_value=0)
                * B.at[indices[:, 1]].get(mode='fill', fill_value=0))
    return sparse.bcoo_dot_general_sampled(
        A, B, indices, dimension_numbers=(((), ()), ((), ())),
    )


csr_sddmv_p = _common.event_primitive('csr_sddmv', 'sddmv', _cuda_kernel, _jax_kernel, native_dims)
