# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Run forward-only delay steps with explicit ring and workspace state."""

from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any, Callable, NamedTuple

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import batching

from brainevent._op import XLACustomKernel, load_cuda_file
from brainevent._typing import Data, MatrixShape

__all__ = ["BINARY_BACKENDS", "binary_csrmv_delay_step", "split_region_delay"]

BINARY_BACKENDS = frozenset({"jax", "cusparse", "cuda_raw"})
_INDEX_DTYPES = (np.dtype("int32"), np.dtype("int64"))


class _DelayCSRInputs(NamedTuple):
    """Hold the original segmented CSR and hybrid task counts."""

    data: Data
    indices: jax.Array
    indptr: jax.Array
    delay_count: jax.Array
    row_task_count: jax.Array


class _SplitCSR(NamedTuple):
    """Hold packed per-delay matrices, including units on sub_values."""

    sub_indptr: jax.Array
    sub_base: jax.Array
    sub_indices: jax.Array
    sub_values: Data


class _DelayWorkspace(NamedTuple):
    """Carry all hybrid task state between successive steps."""

    task_capacity: int
    task_begin: jax.Array
    task_end: jax.Array
    task_out_off: jax.Array
    status: jax.Array


class _CusparseWorkspace(NamedTuple):
    """Carry the shared SpMV output vector between successive steps."""

    scratch: jax.Array


_Inputs = _DelayCSRInputs | _SplitCSR
_Workspace = _DelayWorkspace | _CusparseWorkspace


def _validate_backend(backend: str | None) -> None:
    if backend is not None and backend not in BINARY_BACKENDS:
        raise ValueError(f"Unsupported delay backend {backend!r}; use {sorted(BINARY_BACKENDS)}")


def _check(array: Any, shape: tuple[int, ...], dtypes: Any, name: str) -> None:
    """Check static ABI metadata without copying device values to the host."""
    if getattr(array, "shape", None) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    allowed = dtypes if isinstance(dtypes, tuple) else (dtypes,)
    if array.dtype not in allowed:
        raise TypeError(f"{name} has unsupported dtype {array.dtype}")
    if array.dtype == np.int64 and not jax.config.jax_enable_x64:
        raise RuntimeError("enable jax_enable_x64 before using int64 delay arrays")


@lru_cache(maxsize=None)
def _load(stem: str) -> str:
    source = Path(__file__).with_name(stem + ".cu")
    module = "tcsr_delay_" + stem + "_" + hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    load_cuda_file(
        source, name=module, allow_cuda_graph=False,
        extra_ldflags=["-lcusparse"] if stem == "region_delay_cusparse" else [],
    )
    return module


def split_region_delay(
    data: Data, indices: jax.Array, indptr: jax.Array, delay_count: jax.Array,
) -> _SplitCSR:
    """Split a segmented CSR once into packed cuSPARSE matrices on the GPU.

    Parameters
    ----------
    data : array_like or brainunit.Quantity
        Float32 weights of length one or nnz, with optional units.
    indices : jax.Array
        Int32/int64 target columns of shape (nnz,).
    indptr : jax.Array
        Int32/int64 row pointers of shape (M + 1,).
    delay_count : jax.Array
        Validated counts of shape (M, K), with the indptr dtype.

    Returns
    -------
    _SplitCSR
        Row pointers (K, M + 1), bases (K + 1,), packed columns (nnz,)
        and packed float32 values (nnz,). Values retain the input units.

    Raises
    ------
    ValueError
        If shapes are inconsistent or no GPU is available for execution.
    TypeError
        If array dtypes are unsupported.
    RuntimeError
        If int64 outputs require enabling JAX x64.

    Notes
    -----
    Inputs must already be segmented and validated. Shared weights are
    expanded. All three integer outputs use int32 when nnz fits, else int64.
    """
    weights, unit = u.split_mantissa_unit(data)
    rows, segments, nnz = indptr.size - 1, delay_count.shape[-1], indices.size
    _check(weights, (weights.size,), np.float32, "weights")
    _check(indices, (nnz,), _INDEX_DTYPES, "indices")
    _check(indptr, (rows + 1,), _INDEX_DTYPES, "indptr")
    _check(delay_count, (rows, segments), indptr.dtype, "delay_count")
    if rows < 0 or segments == 0 or weights.size not in (1, nnz):
        raise ValueError("invalid split dimensions or weight count")
    platform = next(iter(indices.devices())).platform
    if "cusparse" not in binary_csrmv_delay_p.available_backends(platform):
        raise ValueError(f"delay backend 'cusparse' is unavailable on {platform}")
    dtype = jnp.int32 if nnz <= np.iinfo(np.int32).max else jnp.int64
    if dtype == jnp.int64 and not jax.config.jax_enable_x64:
        raise RuntimeError("enable jax_enable_x64 before constructing int64 split outputs")
    outputs = (
        jax.ShapeDtypeStruct((segments, rows + 1), dtype),
        jax.ShapeDtypeStruct((segments + 1,), dtype),
        jax.ShapeDtypeStruct((nnz,), dtype),
        jax.ShapeDtypeStruct((nnz,), jnp.float32),
    )
    ptr, base, columns, values = jax.ffi.ffi_call(
        _load("region_delay_split") + ".region_delay_split_f32", outputs,
    )(weights, indices, indptr, delay_count)
    return _SplitCSR(ptr, base, columns, u.maybe_decimal(values * unit))


def _consume_gpu(buffer: jax.Array, head: jax.Array) -> tuple[jax.Array, jax.Array]:
    outputs = (
        jax.ShapeDtypeStruct((buffer.shape[1],), jnp.float32),
        jax.ShapeDtypeStruct(buffer.shape, buffer.dtype),
    )
    y, ring = jax.ffi.ffi_call(
        _load("binary_csrmv_hybrid_region_delay") + ".consume_region_delay_f32",
        outputs, input_output_aliases={0: 1},
    )(buffer, head)
    return y, ring


def _binary_csrmv_jax_kernel(shape: MatrixShape, **kwargs: Any) -> Callable[..., Any]:
    rows, cols = shape

    def _wrapped_offsets(offsets: jax.Array, head: jax.Array, ring_size: int) -> jax.Array:
        """Add a head offset without overflowing a valid int32 ring index."""
        base = head[0].astype(offsets.dtype) * cols
        tail = ring_size - base
        wraps = offsets >= tail
        # Each selected addend is safe even when base + offsets would overflow.
        return jnp.where(wraps, offsets - tail, offsets) + jnp.where(wraps, 0, base)

    def kernel(data: Any, indices: Any, indptr: Any, events: Any, counts: Any,
               offsets: Any, row_tasks: Any, head: Any, buffer: Any,
               begin: Any, end: Any, out_off: Any, status: Any) -> Any:
        nnz, segments = indices.size, offsets.size
        if nnz:
            row_ids = jnp.repeat(
                jnp.arange(rows, dtype=indptr.dtype), jnp.diff(indptr),
                total_repeat_length=nnz,
            )
            segment_ids = jnp.repeat(
                jnp.tile(jnp.arange(segments, dtype=jnp.int32), rows), counts.reshape(-1),
                total_repeat_length=nnz,
            )
            bases = _wrapped_offsets(offsets, head, buffer.size)
            targets = bases[segment_ids] + indices.astype(offsets.dtype)
            weights = data[0] if data.size == 1 else data
            values = jnp.where(events[row_ids] > 0, weights, jnp.float32(0))
            buffer = buffer.reshape(-1).at[targets].add(values).reshape(buffer.shape)
        y = buffer[head[0]]
        buffer = buffer.at[head[0]].set(0)
        return y, buffer, (head + 1) % buffer.shape[0], begin, end, out_off, status

    return kernel


def _binary_csrmv_cuda_kernel(**kwargs: Any) -> Callable[..., Any]:
    module = _load("binary_csrmv_hybrid_region_delay")
    # Whole-step outs: y, ring, head, begin, end, out_off, status.
    projection_outs = (kwargs["outs"][1], *kwargs["outs"][3:])

    def kernel(data: Any, indices: Any, indptr: Any, events: Any, counts: Any,
               offsets: Any, row_tasks: Any, head: Any, buffer: Any,
               begin: Any, end: Any, out_off: Any, status: Any) -> Any:
        ring, begin, end, out_off, status = jax.ffi.ffi_call(
            module + ".region_delay_f32", projection_outs,
            input_output_aliases={8: 0, 9: 1, 10: 2, 11: 3, 12: 4},
        )(data, indices, indptr, events > 0, counts, offsets, row_tasks,
          head, buffer, begin, end, out_off, status)
        y, ring = _consume_gpu(ring, head)
        return y, ring, (head + 1) % ring.shape[0], begin, end, out_off, status

    return kernel


def _binary_csrmv_cusparse_kernel(**kwargs: Any) -> Callable[..., Any]:
    module = _load("region_delay_cusparse")
    projection_outs = (kwargs["outs"][1], kwargs["outs"][3])

    def kernel(ptr: Any, base: Any, indices: Any, data: Any, events: Any,
               offsets: Any, head: Any, buffer: Any, scratch: Any) -> Any:
        ring, scratch = buffer, scratch
        if indices.size and buffer.shape[1]:
            ring, scratch = jax.ffi.ffi_call(
                module + ".region_delay_cusparse_f32", projection_outs,
                input_output_aliases={7: 0, 8: 1},
            )(ptr, base, indices, data, (events > 0).astype(jnp.float32),
              offsets, head, buffer, scratch)
        y, ring = _consume_gpu(ring, head)
        return y, ring, (head + 1) % ring.shape[0], scratch

    return kernel


def binary_csrmv_delay_step(
    inputs: _Inputs, events: Data, *, shape: MatrixShape,
    delay_off: jax.Array, buffer: jax.Array, head: jax.Array,
    workspace: _Workspace, backend: str,
) -> tuple[Data, jax.Array, jax.Array, _Workspace]:
    """Project positive events, consume the current slot and advance one step.

    Parameters
    ----------
    inputs : _DelayCSRInputs or _SplitCSR
        Validated, fixed CSR inputs for CUDA/JAX, or split inputs for cuSPARSE.
    events : array_like
        Bool/float32 vector (M,); only entries greater than zero activate.
    shape : tuple of int
        Source and target counts (M, N).
    delay_off : jax.Array
        Element offsets (K,), with int32/int64 precision appropriate to L*N.
    buffer : jax.Array
        Float32 ring (L, N), containing pending contributions.
    head : jax.Array
        Int32 current slot (1,).
    workspace : _DelayWorkspace or _CusparseWorkspace
        Corresponding backend state, passed back in full after execution.
    backend : str
        Resolved jax, cuda_raw or cusparse backend; fixed at initialization.

    Returns
    -------
    y : jax.Array or brainunit.Quantity
        Current arrivals (N,), with weight units when present.
    new_buffer : jax.Array
        Ring after projection and clearing the consumed slot.
    new_head : jax.Array
        Next slot (1,), int32.
    new_workspace : _DelayWorkspace or _CusparseWorkspace
        Updated task descriptors/status or cuSPARSE scratch.

    Raises
    ------
    ValueError
        If static dimensions or the backend are invalid.
    TypeError
        If input/workspace types or array dtypes do not match the backend.
    RuntimeError
        If int64 arrays require enabling JAX x64.
    NotImplementedError
        If differentiation or vmap is requested.

    Notes
    -----
    This explicit-state entry supports JIT. CSR values and metadata must be
    validated before use; it does not read device values back on every step.
    CUDA status is returned without a host synchronization. Callers must
    use returned state arrays and keep head and delay offsets in range.
    """
    _validate_backend(backend)
    if backend is None:
        raise ValueError("backend must be resolved before preparing delay inputs")
    rows, cols = shape
    if rows < 0 or cols < 0 or buffer.ndim != 2 or buffer.shape[0] < 1:
        raise ValueError("invalid delay dimensions")
    _check(events, (rows,), (np.dtype("bool"), np.dtype("float32")), "events")
    _check(buffer, (buffer.shape[0], cols), np.float32, "buffer")
    _check(head, (1,), np.int32, "head")
    _check(delay_off, (delay_off.size,), _INDEX_DTYPES, "delay_off")
    if delay_off.size == 0:
        raise ValueError("delay_off must contain at least one segment")
    if buffer.size > np.iinfo(np.int32).max and delay_off.dtype != np.int64:
        raise ValueError("large ring buffers require int64 delay_off")
    # Keep float events inside the opaque primitive: comparing here would let
    # AD silently return zero instead of rejecting an unsupported gradient.
    vector = jnp.asarray(events)
    work_arrays: tuple[jax.Array, ...]
    operands: tuple[Any, ...]
    if backend == "cusparse":
        if not isinstance(inputs, _SplitCSR) or not isinstance(workspace, _CusparseWorkspace):
            raise TypeError("cusparse requires _SplitCSR and _CusparseWorkspace")
        data, unit = u.split_mantissa_unit(inputs.sub_values)
        ptr, base, indices = inputs[:3]
        _check(ptr, (delay_off.size, rows + 1), _INDEX_DTYPES, "sub_indptr")
        _check(base, (delay_off.size + 1,), ptr.dtype, "sub_base")
        _check(indices, (indices.size,), ptr.dtype, "sub_indices")
        _check(data, (indices.size,), np.float32, "sub_values")
        _check(workspace.scratch, (cols,), np.float32, "scratch")
        work_arrays = (workspace.scratch,)
        operands = (*inputs[:3], data, vector, delay_off, head, buffer, *work_arrays)
    else:
        if not isinstance(inputs, _DelayCSRInputs) or not isinstance(workspace, _DelayWorkspace):
            raise TypeError("cuda_raw/jax require _DelayCSRInputs and _DelayWorkspace")
        data, unit = u.split_mantissa_unit(inputs.data)
        indices, indptr, counts, row_tasks = inputs[1:]
        _check(data, (data.size,), np.float32, "weights")
        if data.size not in (1, indices.size):
            raise ValueError("weights must have length one or nnz")
        _check(indices, (indices.size,), _INDEX_DTYPES, "indices")
        _check(indptr, (rows + 1,), _INDEX_DTYPES, "indptr")
        _check(counts, (rows, delay_off.size), indptr.dtype, "delay_count")
        _check(row_tasks, (rows,), np.int32, "row_task_count")
        capacity = workspace.task_begin.size
        _check(workspace.task_begin, (capacity,), indptr.dtype, "task_begin")
        _check(workspace.task_end, (capacity,), indptr.dtype, "task_end")
        _check(workspace.task_out_off, (capacity,), delay_off.dtype, "task_out_off")
        _check(workspace.status, (2,), np.int32, "status")
        work_arrays = workspace[1:]
        operands = (data, indices, indptr, vector, counts, delay_off, row_tasks,
                    head, buffer, *work_arrays)
    outs = (jax.ShapeDtypeStruct((cols,), jnp.float32),
            *(jax.ShapeDtypeStruct(a.shape, a.dtype) for a in (buffer, head, *work_arrays)))
    y, ring, position, *work = binary_csrmv_delay_p(
        *operands, outs=outs, shape=shape, backend=backend,
    )
    updated: _Workspace
    if isinstance(workspace, _CusparseWorkspace):
        updated = _CusparseWorkspace(work[0])
    else:
        updated = _DelayWorkspace(workspace.task_begin.size, *work)
    return u.maybe_decimal(y * unit), ring, position, updated


binary_csrmv_delay_p = XLACustomKernel("tcsr_binary_csrmv_delay")
del batching.primitive_batchers[binary_csrmv_delay_p.primitive]
binary_csrmv_delay_p.def_kernel("jax", "cpu", _binary_csrmv_jax_kernel, asdefault=True)
binary_csrmv_delay_p.def_kernel("jax", "gpu", _binary_csrmv_jax_kernel)
binary_csrmv_delay_p.def_kernel("jax", "tpu", _binary_csrmv_jax_kernel, asdefault=True)
binary_csrmv_delay_p.def_kernel("cusparse", "gpu", _binary_csrmv_cusparse_kernel)
binary_csrmv_delay_p.def_cuda_raw_kernel(_binary_csrmv_cuda_kernel, asdefault=True)
binary_csrmv_delay_p.def_call(binary_csrmv_delay_step)
binary_csrmv_delay_p.def_tags("csr", "binary", "delay")
