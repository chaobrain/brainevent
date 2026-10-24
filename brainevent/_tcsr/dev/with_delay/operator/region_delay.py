"""Prepare delayed CSR inputs and wrap the three CUDA operators.

Input preparation runs on the CPU before testing or benchmarking. Tests own
workspace allocation and time stepping; the cuSPARSE operator owns the K loop.
"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from brainevent._op import load_cuda_file


IntArray = NDArray[np.int32] | NDArray[np.int64]
_I32_MAX = int(np.iinfo(np.int32).max)
_I64_MAX = int(np.iinfo(np.int64).max)
TPR_THRESHOLD = 128
TASK_NNZ = 4096


@dataclass(frozen=True)
class DelayInputs:
    """Hold immutable inputs shared by native hybrid and split preparation.

    Parameters
    ----------
    weights, indices, indptr : numpy.ndarray
        CSR operands sorted within each row by (delay, column).
    delay_count : numpy.ndarray
        Segment counts of shape (M, K), with the indptr dtype.
    delay_off : numpy.ndarray
        Ring element offsets, independent of the CSR index dtype.
    row_task_count : numpy.ndarray
        Int32 task counts for the default hybrid scheduling configuration.
    events : numpy.ndarray
        Float32 binary events of logical shape (M, T), stored column-major.
    shape : tuple of int
        Source and target counts (M, N).
    delay_steps : tuple of int
        Increasing distinct delay values, including explicitly empty segments.
    buffer_slots : int
        Ring length, greater than every delay.
    task_capacity : int
        Maximum task count with all rows active.

    Notes
    -----
    Construct with ``prepare_delay_inputs`` and treat the arrays as read-only.
    Native runners copy the inputs once and retain their own GPU allocations.
    """

    weights: NDArray[np.float32]
    indices: IntArray
    indptr: IntArray
    delay_count: IntArray
    delay_off: IntArray
    row_task_count: NDArray[np.int32]
    events: NDArray[np.float32]
    shape: tuple[int, int]
    delay_steps: tuple[int, ...]
    buffer_slots: int
    task_capacity: int

    @property
    def duration_steps(self) -> int:
        """Return the number of event columns.

        Returns
        -------
        int
            Number of complete projection steps in each run.
        """
        return self.events.shape[1]


def validate_delay_steps(values: Any) -> tuple[int, ...]:
    """Validate an explicitly ordered set of delay values.

    Parameters
    ----------
    values : array-like
        Nonempty, strictly increasing nonnegative integer values.

    Returns
    -------
    tuple of int
        Delay values with a representable int32 ring length.

    Raises
    ------
    ValueError
        If values are not ordered, integral, nonnegative or in range.
    """
    array = np.asarray(values)
    if (array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iu"
            or np.any(array < 0) or np.any(array >= _I32_MAX)
            or np.any(array[1:] <= array[:-1])):
        raise ValueError("delay_steps must be increasing nonnegative integers below INT32_MAX")
    return tuple(int(value) for value in array)


def prepare_delay_inputs(
    weights: Any,
    indices: Any,
    indptr: Any,
    edge_delays: Any,
    events: Any,
    *,
    columns: int,
    delay_steps: Any = None,
    buffer_slots: int | None = None,
) -> DelayInputs:
    """Sort independently delayed edges and prepare both native operator inputs.

    Parameters
    ----------
    weights : array-like
        Float32 vector with one value per edge or one shared value.
    indices, indptr : array-like
        Int32/int64 CSR column indices and zero-based row pointers.
    edge_delays : array-like
        One nonnegative integer delay per edge, aligned with the original CSR.
    events : array-like
        Bool or float32 array of shape (M, T), with T positive. Nonzero values
        become binary activity. Each column is one step for both operators.
    columns : int
        Target count N, including targets without incoming edges.
    delay_steps : array-like or None, optional
        Explicit increasing delay values. Retain empty segments when supplied;
        otherwise derive the values from edges, using (0,) for an empty graph.
    buffer_slots : int or None, optional
        Ring length, defaulting to one plus the largest delay.

    Returns
    -------
    DelayInputs
        Contiguous CSR and metadata arrays, plus column-contiguous events.

    Raises
    ------
    TypeError
        If weights, indices, pointers or events have unsupported dtypes.
    ValueError
        If CSR, delays, dimensions, events or task capacities are invalid.

    Notes
    -----
    No per-edge delays are retained after sorting. Hybrid uses the prepared
    arrays directly; split consumes weights, indices, indptr and delay_count.
    Preprocessing belongs outside the benchmark's timed loop.
    """
    w, cols, ptr, delays, event_array = map(np.asarray, (weights, indices, indptr, edge_delays, events))
    if w.dtype != np.float32 or cols.dtype not in (np.dtype("int32"), np.dtype("int64")):
        raise TypeError("require float32 weights and int32/int64 indices")
    if ptr.dtype not in (np.dtype("int32"), np.dtype("int64")):
        raise TypeError("indptr must be int32/int64")
    if event_array.dtype not in (np.dtype("bool"), np.dtype("float32")):
        raise TypeError("events must be bool or float32")
    if w.ndim != 1 or cols.ndim != 1 or ptr.ndim != 1 or ptr.size == 0:
        raise ValueError("CSR arrays must be vectors and indptr must be nonempty")
    if w.size not in (1, cols.size):
        raise ValueError("weights must have length one or nnz")
    if ptr[0] != 0 or int(ptr[-1]) != cols.size or np.any(ptr < 0) or np.any(ptr[1:] < ptr[:-1]):
        raise ValueError("invalid CSR row pointers")
    rows = int(ptr.size - 1)
    if type(columns) is not int or not 0 <= columns <= _I32_MAX or rows > _I32_MAX:
        raise ValueError("source/target counts must fit nonnegative int32")
    if np.any(cols < 0) or np.any(cols >= columns):
        raise ValueError("column index outside target range")
    if (delays.shape != cols.shape or delays.dtype.kind not in "iu"
            or np.any(delays < 0) or np.any(delays >= _I32_MAX)):
        raise ValueError("edge_delays must contain one nonnegative integer per edge")
    if (event_array.ndim != 2 or event_array.shape[0] != rows
            or not 1 <= event_array.shape[1] <= _I32_MAX):
        raise ValueError("events must have shape (rows, positive int32 duration_steps)")
    steps = validate_delay_steps(
        delay_steps if delay_steps is not None else np.unique(delays) if delays.size else (0,)
    )
    if np.any(~np.isin(delays, steps)):
        raise ValueError("edge delay absent from delay_steps")
    slots = steps[-1] + 1 if buffer_slots is None else buffer_slots
    if type(slots) is not int or not steps[-1] < slots <= _I32_MAX:
        raise ValueError("buffer_slots must exceed every delay and fit int32")
    if slots * columns > _I64_MAX // np.dtype("float32").itemsize:
        raise ValueError("ring buffer byte size overflows int64")

    counts = np.zeros((rows, len(steps)), dtype=ptr.dtype)
    sorted_cols = cols.copy()
    sorted_weights = w.copy()
    for row in range(rows):
        begin, end = int(ptr[row]), int(ptr[row + 1])
        segment = np.searchsorted(steps, delays[begin:end])
        counts[row] = np.bincount(segment, minlength=len(steps))
        order = np.lexsort((cols[begin:end], segment))
        sorted_cols[begin:end] = cols[begin:end][order]
        if w.size != 1:
            sorted_weights[begin:end] = w[begin:end][order]
    tasks = (counts.astype(np.int64) // TASK_NNZ + (counts % TASK_NNZ != 0)).sum(axis=1)
    tasks[np.diff(ptr) <= TPR_THRESHOLD] = 0
    capacity = int(tasks.sum())
    if capacity > _I32_MAX:
        raise ValueError("hybrid task capacity exceeds int32")
    offset_values = [step * columns for step in steps]
    offsets: IntArray = (np.asarray(offset_values, dtype=np.int32) if slots * columns <= _I32_MAX
                         else np.asarray(offset_values, dtype=np.int64))
    return DelayInputs(
        np.ascontiguousarray(sorted_weights), np.ascontiguousarray(sorted_cols),
        np.ascontiguousarray(ptr), counts,
        offsets,
        tasks.astype(np.int32), np.asfortranarray(event_array != 0, dtype=np.float32),
        (rows, columns), steps, slots, capacity,
    )


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


class SplitCSR(NamedTuple):
    """Group the four device arrays returned directly by the split operator.

    Parameters
    ----------
    sub_indptr : jax.Array
        Zero-based row pointers of shape (K, M + 1).
    sub_base : jax.Array
        Packed segment boundaries of shape (K + 1,), ending at nnz.
    sub_indices : jax.Array
        Packed column indices of shape (nnz,), with the row-pointer dtype.
    sub_values : jax.Array
        Packed float32 weights of shape (nnz,), including expanded shared weights.
    """

    sub_indptr: jax.Array
    sub_base: jax.Array
    sub_indices: jax.Array
    sub_values: jax.Array


@lru_cache(maxsize=None)
def _load(stem: str) -> str:
    source = Path(__file__).with_name(stem + ".cu")
    name = stem + "_" + hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    load_cuda_file(source, name=name, allow_cuda_graph=False,
                   extra_ldflags=["-lcusparse"] if stem == "region_delay_cusparse" else [])
    return name


def split_region_delay(
    weights: jax.Array, indices: jax.Array, indptr: jax.Array, delay_count: jax.Array,
) -> SplitCSR:
    """Call CUDA split once and retain all four outputs for cuSPARSE steps.

    Parameters
    ----------
    weights, indices, indptr : jax.Array
        Prepared CSR, sorted within each row by delay then column. Weights
        are float32 with length one or nnz; indices and indptr are int32/int64.
    delay_count : jax.Array
        Counts of shape (M, K), with the indptr dtype and the same segment order.

    Returns
    -------
    SplitCSR
        Segment-major CSR arrays. Integer outputs use int32 when nnz fits,
        otherwise int64, which requires jax_enable_x64=True.
    """
    rows, segments, nnz = indptr.size - 1, delay_count.shape[1], indices.size
    dtype = jnp.int32 if nnz <= 2**31 - 1 else jnp.int64
    outputs = (
        jax.ShapeDtypeStruct((segments, rows + 1), dtype),
        jax.ShapeDtypeStruct((segments + 1,), dtype),
        jax.ShapeDtypeStruct((nnz,), dtype),
        jax.ShapeDtypeStruct((nnz,), jnp.float32),
    )
    result = jax.ffi.ffi_call(_load("region_delay_split") + ".region_delay_split_f32", outputs)(
        weights, indices, indptr, delay_count)
    return SplitCSR(*result)


def project_region_delay(
    weights: jax.Array, indices: jax.Array, indptr: jax.Array, events: jax.Array,
    delay_count: jax.Array, delay_off: jax.Array, row_task_count: jax.Array,
    head: jax.Array, buffer: jax.Array, workspace: HybridWorkspace,
) -> tuple[jax.Array, HybridWorkspace]:
    """Call the hybrid CUDA projection once for the current event column.

    Parameters
    ----------
    weights, indices, indptr : jax.Array
        Prepared CSR operands, as for split_region_delay.
    events : jax.Array
        Bool/float32 vector of length M; any nonzero value activates a row.
    delay_count : jax.Array
        Per-row segment counts of shape (M, K), with the indptr dtype.
    delay_off : jax.Array
        Int32/int64 element offsets delay_steps * N, of shape (K,).
    row_task_count : jax.Array
        Int32 task counts (M,) for CUDA defaults: short rows <=128, task size 4096.
    head : jax.Array
        Int32 vector (1,) containing the current ring slot.
    buffer : jax.Array
        Float32 ring (L, N); existing contributions are retained.
    workspace : HybridWorkspace
        Preallocated task arrays and status, reused by successive calls.

    Returns
    -------
    buffer : jax.Array
        Updated ring, aliased with the input buffer.
    workspace : HybridWorkspace
        Updated tasks and status. A nonzero status[1] invalidates the result.
    """
    outputs = tuple(jax.ShapeDtypeStruct(a.shape, a.dtype) for a in (buffer, *workspace))
    result = jax.ffi.ffi_call(
        _load("binary_csrmv_hybrid_region_delay") + ".region_delay_f32", outputs,
        input_output_aliases={8: 0, 9: 1, 10: 2, 11: 3, 12: 4},
    )(weights, indices, indptr, events, delay_count, delay_off, row_task_count, head, buffer, *workspace)
    return result[0], HybridWorkspace(*result[1:])


def project_region_delay_cusparse(
    split: SplitCSR, events: jax.Array, delay_off: jax.Array,
    head: jax.Array, buffer: jax.Array, scratch: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Pass the full split result to one whole-step cuSPARSE CUDA call.

    Parameters
    ----------
    split : SplitCSR
        Unmodified outputs of split_region_delay, reused across steps.
    events : jax.Array
        Bool/float32 vector of length M; converted to float32 binary activity.
    delay_off : jax.Array
        Int32/int64 element offsets (K,), shared with hybrid.
    head : jax.Array
        Int32 current slot (1,), shared by all K sub-matrices.
    buffer : jax.Array
        Float32 ring (L, N); existing contributions are retained.
    scratch : jax.Array
        Float32 workspace (N,), reused by each sub-matrix SpMV.

    Returns
    -------
    buffer : jax.Array
        Ring after all K sub-matrices have been projected.
    scratch : jax.Array
        Last nonempty sub-matrix's SpMV output, or unchanged for an empty graph.

    Notes
    -----
    The CUDA operator owns the segment loop. Its FFI entry creates a temporary
    StepPlan and reads back split boundaries; this wrapper is for correctness
    tests. Native benchmarks reuse a plan prepared before timing.
    """
    outputs = tuple(jax.ShapeDtypeStruct(a.shape, a.dtype) for a in (buffer, scratch))
    ring, work = jax.ffi.ffi_call(
        _load("region_delay_cusparse") + ".region_delay_cusparse_f32", outputs,
        input_output_aliases={7: 0, 8: 1},
    )(*split, (events != 0).astype(jnp.float32), delay_off, head, buffer, scratch)
    return ring, work
