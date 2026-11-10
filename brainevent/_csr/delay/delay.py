# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Region-delay delivery with one primitive and CUDA/JAX backends.

Projection adds to every slot without clearing or advancing the current head.
Task queues are auxiliary storage, with zero tangents.
"""

from functools import lru_cache
import hashlib
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._compatible_import import init_zero, match_varying_axes
from brainevent._op import XLACustomKernel, general_batching_rule, load_cuda_file
from brainevent._op.util import cuda_event_abi

from ._util import _checked_result, _head_out_of_bounds


__all__ = [
    "HybridWorkspace",
    "WeightedHybridWorkspace",
    "csr_delay_deliver_layout_p_call",
    "csr_delay_deliver_layout_p",
]


_I32_MAX = int(np.iinfo(np.int32).max)
TPR_THRESHOLD = 128
TASK_NNZ = 4096
PACKAGE = Path(__file__).resolve().parent
TRAITS = PACKAGE / "_common" / "op_traits.cuh"
HEADER = PACKAGE / "hybrid" / "hybrid_delay_spmv.cuh"
VALUE_SUFFIX = {jnp.dtype(jnp.float16): "f16", jnp.dtype(jnp.bfloat16): "bf16",
                jnp.dtype(jnp.float32): "f32", jnp.dtype(jnp.float64): "f64"}


class HybridWorkspace(NamedTuple):
    """Group the hybrid operator's four mutable workspace arrays.

    Parameters
    ----------
    task_begin, task_end : jax.Array
        Edge offsets of shape (capacity,), with the indptr dtype.
    task_out_off : jax.Array
        Ring offsets of shape (capacity,), with the delay_off dtype.
    status : jax.Array
        Int32 pair: task count and error code; reset by each CUDA call.
    """

    task_begin: jax.Array
    task_end: jax.Array
    task_out_off: jax.Array
    status: jax.Array


class WeightedHybridWorkspace(NamedTuple):
    """Hold hybrid tasks with one event value per task.

    Parameters
    ----------
    task_begin, task_end, task_out_off : jax.Array
        Same shapes and dtypes as in HybridWorkspace. Ends retain delay segment
        boundaries; task_out_off stores the destination ring offset.
    task_value : jax.Array
        Event values of shape (capacity,), in the weights/ring dtype.
    status : jax.Array
        Int32 pair: task count and error code; reset on every CUDA call.
    """

    task_begin: jax.Array
    task_end: jax.Array
    task_out_off: jax.Array
    task_value: jax.Array
    status: jax.Array


@lru_cache(maxsize=None)
def _load(stem: str) -> str:
    source = HEADER.parent / (stem + ".cu")
    dependencies = [source, TRAITS, TRAITS.with_name("cuda_traits.cuh"), HEADER,
                    HEADER.with_name("hybrid_delay_validation.cuh")]
    name = stem + "_" + hashlib.sha256(b"".join(path.read_bytes() for path in dependencies)).hexdigest()[:16]
    load_cuda_file(source, name=name, allow_cuda_graph=True,
                   extra_include_paths=[str(source.parent), str(PACKAGE)])
    return name


def _input_error_code(ring, events, data, indices, indptr, counts, offsets, head, tasks, *workspace):
    """Validate prepared values once, before either delivery implementation."""
    lengths = jnp.diff(indptr)
    # Checking only a sum (or the sign of diff(indptr)) can accept int32
    # wraparound. Each nonnegative segment prefix must stay inside its row.
    segment_ends = jnp.cumsum(counts, axis=1)
    expected_tasks = jnp.sum(counts // TASK_NNZ + (counts % TASK_NNZ != 0), axis=1)
    expected_tasks = jnp.where(lengths > TPR_THRESHOLD, expected_tasks, 0)
    invalid_metadata = (
        (indptr[0] != 0) | (indptr[-1] != indices.size)
        | jnp.any(indptr < 0) | jnp.any(indptr > indices.size)
        | jnp.any(indptr[1:] < indptr[:-1])
        | jnp.any(counts < 0) | jnp.any(segment_ends < 0)
        | jnp.any(segment_ends > lengths[:, None])
        | jnp.any(segment_ends[:, -1] != lengths)
        | jnp.any(tasks != expected_tasks)
        | jnp.any(indices < 0) | jnp.any(indices >= ring.shape[1]))
    if ring.shape[1]:
        invalid_metadata |= jnp.any(
            (offsets < 0) | (offsets >= ring.size) | (offsets % ring.shape[1] != 0))
    else:
        invalid_metadata |= jnp.any(offsets != 0)
    active_tasks = jnp.sum(jnp.where(events != 0, tasks, 0))
    code = jnp.where(active_tasks > workspace[0].size, jnp.int32(1), jnp.int32(0))
    code = jnp.where(invalid_metadata, jnp.int32(3), code)
    return jnp.where(_head_out_of_bounds(head[0], ring.shape[0]), jnp.int32(2), code)


def _validated_kernel(kernel_generator):
    """Give all primitive bindings the same input gate, including raw binds."""
    def generate(*, check_status, **params):
        kernel = kernel_generator(**params)

        def checked(*args):
            code = _input_error_code(*args)

            def align(values):
                # Match the primitive's abstract outputs, including raw binds
                # whose operands carry different manual shard axes.
                return tuple(match_varying_axes(value, *args) for value in values)

            def reject(_):
                # No backend executes on invalid input, even when the caller
                # requests an error code instead of an exception.
                return align((args[0], *(jnp.zeros_like(x) for x in args[9:-1]),
                              jnp.stack((jnp.int32(0), code))))

            result = jax.lax.cond(code == 0, lambda _: align(kernel(*args)), reject, operand=None)
            # One reporter covers input errors and native execution status.
            # Its callback owns every leaf, so consuming only status or a
            # nonempty queue retains the check even when the ring is unused.
            return _checked_result(result, result[-1][1], head_only=not check_status)

        return checked

    return generate


# Backend generators only deliver already-validated operands. JAX transforms
# operate on the shared primitive's rules, never directly on an ffi_call.
def _cuda_kernel(*, weighted, outs, **kwargs):
    def kernel(ring, events, data, indices, indptr, counts, offsets, head, tasks, *workspace):
        stem = "hybrid_delay_spmv_weighted" if weighted else "hybrid_delay_spmv_binary"
        abi = cuda_event_abi(events.dtype, weighted=weighted, value_dtype=ring.dtype)
        value = VALUE_SUFFIX[jnp.dtype(ring.dtype)]
        return jax.ffi.ffi_call(
            _load(stem) + f".region_delay_{value}" + abi.suffix, outs,
            input_output_aliases={8 + i: i for i in range(len(outs))},
        )(data, indices, indptr, abi.prepare(events), counts, offsets, tasks, head, ring, *workspace)

    return kernel


def _jax_kernel(*, weighted, **kwargs):
    """Implement the hybrid prepared ABI with JAX gathers and scatter-add."""
    def kernel(ring, events, data, indices, indptr, counts, offsets, head, tasks, *workspace):
        result = ring
        if indices.size and ring.shape[1]:
            _, row_ids, slots, _ = _edge_locations(
                (ring, events, data, indices, indptr, counts, offsets, head))
            x = events[row_ids]
            weights = jnp.broadcast_to(data, indices.shape)
            values = weights * x.astype(ring.dtype) if weighted else weights
            # Inactive rows must not introduce NaN/Inf through zero products.
            values = jnp.where(x != 0, values, jnp.zeros_like(values))
            result = ring.at[slots, indices].add(values)
        # JAX needs no task queue; retain the prepared auxiliary-output ABI.
        return (result, *(jnp.zeros_like(value) for value in workspace))

    return kernel


def _aval(x):
    return x.aval if ad.is_undefined_primal(x) else x


def _edge_groups(counts, nnz):
    """Expand prepared segment counts for JAX delivery and differentiation."""
    dtype = jnp.int64 if counts.size > _I32_MAX or nnz > _I32_MAX else jnp.int32
    return jnp.repeat(jnp.arange(counts.size, dtype=dtype), counts.reshape(-1),
                      total_repeat_length=nnz)


def _edge_locations(args):
    ring, events, data, indices, indptr, segments, offsets = args[:7]
    nnz = indices.size
    rows, n_segments = segments.shape
    groups = _edge_groups(segments, nnz)
    row_ids, delay_ids = groups // n_segments, groups % n_segments
    head = args[7]
    slots, cols = _aval(ring).shape
    if nnz == 0:
        return rows, row_ids, jnp.zeros((0,), jnp.int32), head
    # Wrap in slot units: head * cols + offset can overflow even when the
    # ring fits int32. Select the subtrahend first so head + delay never
    # overflows either, including when the slot count is near INT32_MAX.
    current_head = head[0].astype(offsets.dtype)
    delays = offsets[delay_ids] // cols
    remaining = jnp.asarray(slots, offsets.dtype) - delays
    slot_ids = current_head - jnp.where(current_head >= remaining, remaining, -delays)
    return rows, row_ids, slot_ids, head


def _project_tangent(args, *, backend, events=None, data=None, weighted):
    """Bind derivative operands directly, with a fresh zero ring."""
    operands = list(args)
    operands[0] = jnp.zeros_like(args[0])
    if events is not None:
        operands[1] = events
    if data is not None:
        operands[2] = data
    workspace = args[9:]
    if weighted:
        # Binary event tangents use weighted delivery, including task_value.
        workspace = (*workspace[:3], jnp.zeros(workspace[0].shape, operands[0].dtype), workspace[-1])
        operands[9:] = workspace
    outputs = tuple(jax.ShapeDtypeStruct(a.shape, a.dtype) for a in (operands[0], *workspace))
    return csr_delay_deliver_layout_p(
        *operands, outs=outputs, backend=backend, weighted=weighted, check_status=False)[0]


def _projection_jvp(primals, tangents, *, weighted, backend=None, **params):
    result = csr_delay_deliver_layout_p.primitive.bind(
        *primals, backend=backend, weighted=weighted, **params)
    ring_dot = init_zero(result[0])
    if type(tangents[0]) is not ad.Zero:
        ring_dot = tangents[0]
    if type(tangents[1]) is not ad.Zero:
        # Weighted is exact; binary follows the main library's straight-through
        # rule on floating events. Boolean events have float0/Zero tangents.
        ring_dot = ad.add_tangents(ring_dot, _project_tangent(
            primals, backend=backend, events=tangents[1], weighted=True))
    if type(tangents[2]) is not ad.Zero:
        ring_dot = ad.add_tangents(ring_dot, _project_tangent(
            primals, backend=backend, data=tangents[2], weighted=weighted))
    return result, [ring_dot, *(init_zero(x) for x in result[1:])]


def _projection_transpose(cts, *args, weighted, **kwargs):
    ring, events, data, indices, indptr, segments, offsets = args[:7]
    workspace_start = 9
    for i in range(3, workspace_start):
        if ad.is_undefined_primal(args[i]):
            raise ValueError("region_delay is not differentiable with respect to structure or head")
    ct = cts[0]
    result = [None] * len(args)
    if ad.is_undefined_primal(ring):
        result[0] = ad.Zero(ring.aval) if type(ct) is ad.Zero else ct
    if ad.is_undefined_primal(events) or ad.is_undefined_primal(data):
        if type(ct) is not ad.Zero:
            rows, row_ids, slot_ids, _ = _edge_locations(args)
            g = ct[slot_ids, indices]
        if ad.is_undefined_primal(data):
            if type(ct) is ad.Zero:
                result[2] = ad.Zero(data.aval)
            else:
                x = events[row_ids]
                value = x.astype(g.dtype) * g if weighted else g
                value = jnp.where(x != 0, value, jnp.zeros_like(value))
                if data.aval.shape[0] == 1:
                    value = value.sum().reshape(data.aval.shape)
                result[2] = value.astype(data.aval.dtype)
        if ad.is_undefined_primal(events):
            if type(ct) is ad.Zero:
                result[1] = ad.Zero(events.aval)
            else:
                w = data[0] if data.shape[0] == 1 else data
                result[1] = jax.ops.segment_sum(w * g, row_ids, num_segments=rows).astype(events.aval.dtype)
    # Workspace is not a differentiable output or state. Its incoming
    # values never contribute to the ring; ignore its output cotangents too.
    for i in range(workspace_start, len(args)):
        if ad.is_undefined_primal(args[i]):
            result[i] = ad.Zero(args[i].aval)
    return result


def _batching_rule(args, axes, **kwargs):
    return general_batching_rule(csr_delay_deliver_layout_p, args, axes, **kwargs)


def _normalize_flags(weighted, check_status):
    """Share the flag contract across the call and abstract-evaluation boundaries."""
    if not isinstance(weighted, (bool, np.bool_)) or not isinstance(check_status, (bool, np.bool_)):
        raise TypeError("weighted and check_status must be bool")
    return bool(weighted), bool(check_status)


def _projection_abstract_eval(*operands, outs, weighted, check_status, backend=None, **kwargs):
    """Check every binding path once, including class calls and AD re-entry."""
    if backend not in (None, "cuda_raw", "jax_raw"):
        raise ValueError("backend must be 'cuda_raw', 'jax_raw' or None")
    weighted, check_status = _normalize_flags(weighted, check_status)
    if len(operands) != (14 if weighted else 13):
        raise ValueError("CUDA/JAX requires row_task_count followed by the hybrid workspace arrays")
    ring, events, data, indices, indptr, segments, offsets, head, *workspace = operands
    if indptr.ndim != 1 or indptr.size < 1:
        raise ValueError("indptr must be 1-D with at least one row pointer")
    if indices.ndim != 1:
        raise ValueError("indices must have shape (nnz,)")
    if data.ndim != 1 or data.size not in (1, indices.size):
        raise ValueError("data must have shape (1,) or (nnz,)")
    for name, value in (("indices", indices), ("indptr", indptr), ("offsets", offsets)):
        if value.dtype not in (jnp.int32, jnp.int64):
            raise TypeError(f"{name} must use int32 or int64")
    if offsets.ndim != 1 or offsets.size == 0:
        raise ValueError("offsets must have shape (positive K,)")
    rows, n_segments = indptr.size - 1, offsets.size
    if segments.dtype != indptr.dtype:
        raise TypeError("segments must have the indptr dtype")
    if segments.shape != (rows, n_segments):
        raise ValueError(f"segments must have shape ({rows}, {n_segments})")
    tasks, *mutable = workspace
    if tasks.shape != (rows,):
        raise ValueError(f"row_task_count must have shape ({rows},)")
    if tasks.dtype != jnp.int32:
        raise TypeError("row_task_count must be int32")

    if ring.ndim != 2 or ring.shape[0] == 0:
        raise ValueError("buffer must have shape (positive L, N)")
    if head.shape != (1,) or head.dtype != jnp.int32:
        raise TypeError("head must be int32[1], with 0 <= head < L")
    if events.shape != (rows,):
        raise ValueError(f"events must have shape ({rows},)")
    if ring.dtype not in VALUE_SUFFIX or data.dtype != ring.dtype:
        raise TypeError("require float16/bfloat16/float32/float64 weights and ring of one dtype")
    # Reuse the public event ABI's dtype contract; conversion belongs in the
    # CUDA lowering so the original values survive JAX and differentiation.
    cuda_event_abi(events.dtype, weighted=weighted, value_dtype=ring.dtype)
    if indptr.dtype == jnp.int32 and indices.size > _I32_MAX:
        raise ValueError("int32 indptr cannot address more than INT32_MAX edges; use int64")
    if max(rows, n_segments, *ring.shape) > _I32_MAX:
        raise ValueError("row count, segment count and ring dimensions must fit int32")
    if ring.size > np.iinfo(np.int64).max // ring.dtype.itemsize:
        raise ValueError("ring buffer byte size overflows int64")
    if offsets.dtype == jnp.int32 and ring.size > _I32_MAX:
        raise ValueError("use int64 offsets for a ring larger than INT32_MAX elements")
    capacity = mutable[0].size
    if capacity > _I32_MAX:
        raise ValueError("task capacity must fit int32")
    dtypes: tuple[np.dtype, ...] = (indptr.dtype, indptr.dtype, offsets.dtype)
    if weighted:
        dtypes += (ring.dtype,)
    for name, value, dtype in zip(
            ("task_begin", "task_end", "task_out_off", "task_value"), mutable[:-1], dtypes):
        if value.shape != (capacity,):
            raise ValueError(f"{name} must have shape ({capacity},)")
        if value.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}")
    if mutable[-1].shape != (2,) or mutable[-1].dtype != jnp.int32:
        raise TypeError("status must be int32[2]")
    expected = (ring, *mutable)
    if len(outs) != len(expected) or any(
            out.shape != value.shape or out.dtype != value.dtype
            for out, value in zip(outs, expected)):
        raise ValueError("outs must match the ring and mutable workspace shapes and dtypes")
    return csr_delay_deliver_layout_p._abstract_eval(*operands, outs=outs, **kwargs)


def csr_delay_deliver_layout_p_call(
    ring,
    events,
    data,
    indices,
    indptr,
    segments,
    offsets,
    head,
    *workspace,
    weighted: bool = False,
    check_status: bool = True,
    backend: str | None = None,
):
    """Validate prepared operands and bind the region-delay primitive.

    Parameters
    ----------
    ring : array_like
        Arrival buffer of shape ``(L, N)``, with positive ``L``. Values use
        float16, bfloat16, float32 or float64.
    events : array_like
        Events of shape ``(M,)``, with a boolean, integer or floating dtype.
        Binary delivery uses nonzero activity. Weighted values are converted
        to the ring dtype inside the execution backend.
    data : array_like
        Weights with the ring dtype: a scalar, ``(1,)`` or ``(nnz,)``.
    indices : array_like
        Column indices of shape ``(nnz,)``, int32 or int64.
    indptr : array_like
        Row pointers of shape ``(M + 1,)``, int32 or int64.
    segments : array_like
        Row/delay counts of shape ``(M, K)``, with the indptr dtype.
    offsets : array_like
        Ring element offsets of shape ``(K,)``, int32 or int64, with
        positive ``K``. Use int64 when the ring exceeds INT32_MAX elements.
    head : array_like
        Current slot, int32 with shape ``(1,)`` and ``0 <= head[0] < L``.
        This is a slot index, not a global timestep; it is not reduced modulo L.
    *workspace : array_like
        Int32 row task counts of shape ``(M,)``, followed by the
        flattened :class:`HybridWorkspace` or :class:`WeightedHybridWorkspace`
        arrays.
        All storage is prepared by the caller and reused across steps. JAX
        returns zero task arrays and status ``[0, error_code]`` without using
        the queues; their shapes and dtypes remain the same as CUDA.
    weighted : bool, optional
        Multiply weights by event values instead of the nonzero activity mask.
    check_status : bool, optional
        Raise on task/metadata errors, default True. With False, invalid inputs
        return an unchanged ring, zero task arrays and status ``[0, error_code]``;
        native execution failures also remain visible in status.
        Invalid heads are always reported, including when only status or a
        nonempty task array is consumed.
    backend : str or None, optional
        ``'cuda_raw'`` or ``'jax_raw'``. None uses the primitive's default
        for the execution platform: CUDA on GPU, JAX on CPU/TPU. JAX also
        runs on GPU using the same prepared layout. CUDA requires a GPU.

    Returns
    -------
    tuple of jax.Array
        Updated ring followed by mutable workspace arrays, in input order;
        row task counts are not returned. All existing ring values, including
        the current slot, are preserved before contributions are added.

    Raises
    ------
    TypeError
        If operand dtypes or static flags are incompatible with the backend.
    ValueError
        If the backend, shapes or static workspace capacity is invalid.
    RuntimeError
        If execution rejects a head or checked metadata/capacity, or reports a
        native execution error. Eager calls and JIT use the same input gate;
        an asynchronous error can surface when the result is synchronized.

    Notes
    -----
    This unitless entry point infers ``outs`` and performs shape/dtype checks
    without copying connectivity to the host. Connectivity and segment contents
    must already be prepared consistently, as by ``delaycsr``. Both backends
    share a JAX input gate for head, metadata and active-task capacity. Under
    JIT these values are checked before delivery at execution time, including
    metadata for inactive rows. JIT, JVP,
    transpose and batching use the existing primitive rules.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent import csr_delay_deliver_layout_p
        >>> queue = jnp.zeros(0, jnp.int32)
        >>> result = csr_delay_deliver_layout_p.call(  # doctest: +SKIP
        ...     jnp.zeros((2, 1), jnp.float32), jnp.array([True]),
        ...     jnp.array([2.], jnp.float32), jnp.array([0], jnp.int32),
        ...     jnp.array([0, 1], jnp.int32), jnp.array([[1]], jnp.int32),
        ...     jnp.array([1], jnp.int32), jnp.array([0], jnp.int32),
        ...     jnp.zeros(1, jnp.int32), queue, queue, queue,
        ...     jnp.zeros(2, jnp.int32))
        >>> result[0]  # doctest: +SKIP
        Array([[0.],
               [2.]], dtype=float32)
    """
    # Python's 0/False and 1/True share cache keys, so reject numeric flags
    # before JAX can reuse an abstract evaluation from a valid boolean call.
    weighted, check_status = _normalize_flags(weighted, check_status)
    ring, events, data, indices, indptr, segments, offsets, head = map(
        jnp.asarray, (ring, events, data, indices, indptr, segments, offsets, head))
    workspace = tuple(jnp.asarray(value) for value in workspace)
    if data.ndim == 0:
        data = data.reshape(1)

    outputs = tuple(jax.ShapeDtypeStruct(value.shape, value.dtype) for value in (ring, *workspace[1:]))
    return csr_delay_deliver_layout_p(
        ring, events, data, indices, indptr, segments, offsets, head, *workspace,
        outs=outputs, backend=backend, weighted=weighted, check_status=check_status)


csr_delay_deliver_layout_p = XLACustomKernel(
    "csr_delay_deliver_layout",
    doc="""Add delayed events to a ring through a prepared layout.

    Operands are (ring, events, data, indices, indptr, segments, offsets,
    head, row_task_count, *workspace). segments contains row/delay counts;
    head is int32[1]. Both backends use the hybrid workspace layout.

    Supply outs for the ring and mutable workspace arrays, plus static
    backend ('cuda_raw', 'jax_raw' or None), weighted and check_status parameters.
    None uses the default registered for the execution platform.
    Outputs preserve every ring slot before delivery.
    The class multiplication entry and derivative rules bind this primitive
    directly. Use csr_delay_deliver_layout_p_call or this primitive's call
    method to validate prepared operands and infer outs automatically.
    Shape/dtype errors in raw calls surface during compilation;
    invalid heads and checked device errors surface during execution.
    """,
)
csr_delay_deliver_layout_p.primitive.def_abstract_eval(_projection_abstract_eval)
csr_delay_deliver_layout_p.def_kernel("cuda_raw", "gpu", _validated_kernel(_cuda_kernel), asdefault=True)
for _platform in ("cpu", "gpu", "tpu"):
    csr_delay_deliver_layout_p.def_kernel("jax_raw", _platform, _validated_kernel(_jax_kernel))
csr_delay_deliver_layout_p.def_jvp_rule(_projection_jvp)
csr_delay_deliver_layout_p.def_transpose_rule(_projection_transpose)
csr_delay_deliver_layout_p.def_batching_rule(_batching_rule)
csr_delay_deliver_layout_p.def_call(csr_delay_deliver_layout_p_call)
csr_delay_deliver_layout_p.def_tags("csr", "delay")
