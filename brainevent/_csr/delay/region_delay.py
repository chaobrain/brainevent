# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Stateful delayed CSR: initialization, sorting, ring and single-step reads.

CUDA operators and their JAX transformation rules live in ``delay.py``.
"""

from copy import copy
import operator
from typing import Any
import warnings

import jax
import jax.numpy as jnp
from jax.core import Tracer
import numpy as np
import brainunit as u

from brainevent._event import as_event_operand
from brainevent._error import PerformanceWarning
from brainevent._misc import build_sub_csr, normalize_row_index

from brainevent._csr.main import CSR
from brainevent._csr.config import _prepare_csr_config
from brainevent._csr.delay._util import _checked_head
from brainevent._csr.delay.delay import (
    TASK_NNZ, TPR_THRESHOLD, VALUE_SUFFIX, HybridWorkspace, WeightedHybridWorkspace,
    csr_delay_deliver_layout_p,
)


_I32_MAX = int(np.iinfo(np.int32).max)
_I64_MAX = int(np.iinfo(np.int64).max)


def _time_slot(t: Any, slots: int) -> jax.Array:
    """Reduce global time before converting to the int32 CUDA slot type."""
    if isinstance(t, (int, np.integer)) and not isinstance(t, (bool, np.bool_)):
        return jnp.asarray(int(t) % slots, jnp.int32)
    value = jnp.asarray(t)
    if value.ndim != 0 or value.dtype.kind not in "iu":
        raise TypeError("t must be an integer scalar timestep")
    return (value % slots).astype(jnp.int32)


def _owned_data(value: Any, *, concrete: bool = False) -> Any:
    """Own concrete initialization buffers and NumPy replacement weights.

    Traced/JAX replacement weights remain device values; no host transfer is
    introduced into a JIT or differentiation path.
    """
    mantissa, unit = u.split_mantissa_unit(value)
    if concrete or isinstance(mantissa, np.ndarray):
        mantissa = jax.device_put(np.array(mantissa, copy=True))
    else:
        mantissa = jnp.asarray(mantissa)
    return u.maybe_decimal(mantissa * unit)


def _prepared_operand(value: Any, name: str) -> Any:
    """Preserve device operands and reject implicit narrowing of typed inputs."""
    if not jax.config.x64_enabled and getattr(value, "dtype", None) in (
            np.dtype("int64"), np.dtype("float64")):
        raise RuntimeError(f"enable jax_enable_x64 before supplying int64/float64 {name}")
    return _owned_data(value)


def _resolve_events(events: Any, weighted: bool | None) -> tuple[Any, bool]:
    """Resolve event wrappers and an optional mode before preparing delivery."""
    if weighted is not None and not isinstance(weighted, (bool, np.bool_)):
        raise TypeError("weighted must be bool or None")
    event = as_event_operand(events)
    if event is not None:
        events, mode = event
        if weighted is not None and bool(weighted) != mode:
            raise TypeError("event wrapper mode disagrees with weighted")
        weighted = mode
    return events, bool(weighted)


def _clear_ring_slot(ring: Any, head: jax.Array) -> jax.Array | u.Quantity:
    """Clear one slot with JAX, preserving the other slots, dtype and unit."""
    values, unit = u.split_mantissa_unit(ring)
    cleared = jnp.asarray(values).at[head].set(0)
    return u.Quantity(cleared, unit=unit) if isinstance(ring, u.Quantity) else cleared


def _require_concrete(method: str, *values: Any) -> None:
    """Reject host-dependent construction on traced operands."""
    if any(isinstance(value, Tracer) for value in jax.tree_util.tree_leaves(values)):
        raise RuntimeError(f"delaycsr.{method} requires concrete inputs outside jax.jit")


def _checked_indptr(lengths: Any, dtype: Any) -> np.ndarray:
    """Check the new edge count before accumulating into an offset dtype."""
    total = sum(int(value) for value in lengths)
    if total > _I64_MAX:
        raise ValueError("new CSR edge count exceeds int64")
    dtype = np.dtype(dtype)
    if total > _I32_MAX:
        dtype = np.dtype("int64")
    if dtype == np.dtype("int64") and not jax.config.x64_enabled:
        raise RuntimeError("enable jax_enable_x64 for int64 indptr")
    pointers = np.zeros(len(lengths) + 1, dtype=dtype)
    np.cumsum(np.asarray(lengths, dtype=np.int64), out=pointers[1:])
    return pointers


@jax.tree_util.register_pytree_node_class
class delaycsr(CSR):
    """CSR with owned delay metadata, ring and task workspace.

    Parameters
    ----------
    data : CSR, tuple of arrays or array_like
        Source CSR, ``(weights, indices, indptr)``, or weights supplied with
        separate indices and row pointers. Weights use float16, bfloat16,
        float32 or float64, with one shared value or one value per edge.
        Quantity weights retain their units and canonical storage dtype.
    indices, indptr : array_like or None, optional
        Int32 or int64 column indices and row pointers. Omit these when
        supplying a CSR or a three-array tuple.
    edge_delays : int or array_like
        Concrete nonnegative integer delay shared by all edges, or one delay
        per edge in the input CSR order. A one-element array also broadcasts.
        Initialization sorts each row by (delay, column).
    shape : tuple of int or None, optional
        ``(n_pre, n_post)``, required unless data is a CSR object.
    delay_steps : array_like or None, optional
        Increasing distinct delay values covering every edge delay. Defaults
        to the distinct edge delays, or ``(0,)`` for an empty graph.
    n_slots : int or None, optional
        Ring length, larger than every delay. Defaults to maximum delay + 1.
    sorted_by_delay : bool, optional
        Validate the existing (delay, column) row order instead of sorting.
    weighted : bool, optional
        Multiply by event values when True; use nonzero activity when False.
    ring : array_like or None, optional
        Initial buffer with shape ``(n_slots, n_post)``. Its floating
        dtype determines delivery precision, independently of the weights.
        Units must be compatible with the delivered weight/event product;
        delivery preserves the initialized ring's unit. Defaults to zeros
        in the weights' dtype and unit, for dimensionless events.
    head : int or jax.Array, optional
        Initial scalar slot in ``[0, n_slots)``, default zero.
    backend : str or None, optional
        ``'cuda_raw'`` or ``'jax_raw'``. None inherits the source CSR backend,
        or uses the execution platform's default when no backend is set.

    Notes
    -----
    ``forward(x, t)`` sets ``head = t % L`` and returns the next timestep.
    ``x @ W`` uses the current head without advancing it. A read never clears
    its slot: the following stateful step uses the same JAX clearing operation
    as :meth:`clear_head`.
    :meth:`deliver` only delivers into an explicit ring.

    Use :meth:`from_prepared` when the CSR is already sorted and its delay
    counts, offsets and hybrid task schedule are supplied by the caller.
    That entry skips sorting and metadata preparation entirely.

    This is a mutable Python object. For JIT/scan, pass it as a pytree argument
    and return the updated object as part of the result/carry. Do not jit a
    bound method with hidden Python state; these methods do not detect that
    unsupported usage. Operators support JVP, reverse-mode AD and vmap;
    binary events use the main library's straight-through rule.

    Inherited queries such as ``sum`` and ``todense`` describe static weights,
    not delayed arrivals. ``tocsc``, ``tocoo`` and ``transpose``/``T`` export
    ordinary static matrices without delay state; ``self.T @ x`` is therefore
    an instantaneous product. ``tocsr`` returns this delaycsr unchanged.
    Addition/subtraction return static dense arrays. ``solve`` solves only the
    static weight system, subject to its usual dtype/backend restrictions.

    ``apply``, compatible ``apply2`` and ``with_data`` preserve existing arrivals
    while replacing weights for future delivery. ``slice_rows`` and ``diag_add``
    instead create fresh zero execution state. Plasticity and ``dt2t`` consume
    supplied signals without aligning their history by delay, and require int32
    column indices. ``delay_layout`` reinitializes a layout in its default binary
    mode with zero state. ``with_rebuilt_config`` retains the inherited plain-CSR
    return behavior; it does not preserve this object's delay functionality.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from brainevent import delaycsr
        >>> matrix = delaycsr(
        ...     (np.array([1., 2.], np.float32), np.array([0, 1], np.int32),
        ...      np.array([0, 2], np.int32)),
        ...     shape=(1, 2), edge_delays=2, backend='jax_raw')
        >>> matrix.delay_steps
        (2,)
        >>> matrix.L
        3
    """

    data: jax.Array | u.Quantity
    indices: jax.Array
    indptr: jax.Array
    workspace: HybridWorkspace | WeightedHybridWorkspace

    def __init__(
        self, data: Any, indices: Any = None, indptr: Any = None, *,
        edge_delays: Any, shape: tuple[int, int] | None = None,
        delay_steps: Any = None, n_slots: int | None = None,
        sorted_by_delay: bool = False, weighted: bool = False,
        ring: Any = None, head: Any = 0, backend: str | None = None,
    ):
        if isinstance(data, CSR):
            if indices is not None or indptr is not None:
                raise ValueError("a CSR source already supplies indices and indptr")
            if shape is not None and tuple(shape) != data.shape:
                raise ValueError("shape disagrees with the CSR source")
            shape = data.shape
            if backend is None:
                backend = data.backend
            data, indices, indptr = data.data, data.indices, data.indptr
        elif indices is None and indptr is None:
            if not isinstance(data, (tuple, list)) or len(data) != 3:
                raise TypeError("supply CSR or (weights, indices, indptr)")
            data, indices, indptr = data
        if backend not in (None, "cuda_raw", "jax_raw"):
            raise ValueError("delaycsr backend must be 'cuda_raw', 'jax_raw' or None")
        if (shape is None or len(shape) != 2 or any(
                isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
                or not 0 <= n <= _I32_MAX for n in shape)):
            raise ValueError("shape must contain two nonnegative int32 dimensions")
        if not isinstance(weighted, (bool, np.bool_)):
            raise TypeError("weighted must be bool")
        if not isinstance(sorted_by_delay, (bool, np.bool_)):
            raise TypeError("sorted_by_delay must be bool")
        # A source CSR can itself alias caller-owned CPU NumPy buffers. Take
        # owned snapshots even in that case, before building any metadata.
        weight_values, weight_unit = u.split_mantissa_unit(data)
        weights = np.atleast_1d(np.array(weight_values, copy=True))
        columns, pointers = np.array(indices, copy=True), np.array(indptr, copy=True)
        if jnp.dtype(weights.dtype) not in VALUE_SUFFIX:
            raise TypeError("weights must be float16/bfloat16/float32/float64")
        if columns.dtype not in (np.dtype("int32"), np.dtype("int64")):
            raise TypeError("indices must be int32/int64")
        if pointers.dtype not in (np.dtype("int32"), np.dtype("int64")):
            raise TypeError("indptr must be int32/int64")
        if not jax.config.x64_enabled and any(
                a.dtype in (np.dtype("int64"), np.dtype("float64"))
                for a in (weights, columns, pointers)):
            raise RuntimeError("enable jax_enable_x64 before constructing int64/float64 delaycsr")
        # The inherited CSR validates the array dimensions, bounds and pointers.
        columns_device = jax.device_put(columns)
        super().__init__(u.maybe_decimal(jax.device_put(weights) * weight_unit),
                         columns_device, jax.device_put(pointers),
                         shape=(int(shape[0]), int(shape[1])), backend=backend,
                         indptr_dtype=pointers.dtype)
        # The native operator permits int64 column indices independently of indptr.
        self.indices = columns_device
        delays = self._validate_edge_delays(edge_delays)
        self.delay_steps = self._validate_delay_steps(
            delay_steps if delay_steps is not None else np.unique(delays) if delays.size else (0,))
        self._check_delay_segments(delays)
        self.L = self.delay_steps[-1] + 1 if n_slots is None else n_slots
        if type(self.L) is not int or not self.delay_steps[-1] < self.L <= _I32_MAX:
            raise ValueError("n_slots must exceed every delay and fit int32")
        ring_values = None if ring is None else u.get_mantissa(ring)
        ring_dtype: np.dtype = self.dtype if ring_values is None else getattr(ring_values, "dtype", None)
        if ring_values is not None and not (hasattr(ring_values, "shape") and hasattr(ring_values, "dtype")):
            ring_values = np.asarray(ring_values)
            # Python literals follow JAX's default precision; only explicitly
            # typed float64 buffers require x64 to preserve their dtype.
            ring_dtype = jax.dtypes.canonicalize_dtype(ring_values.dtype)
        if ring_values is not None and ring_values.shape != (self.L, self.shape[1]):
            raise ValueError("ring must have shape (L, N)")
        if jnp.dtype(ring_dtype) not in VALUE_SUFFIX:
            raise TypeError("ring must use float16/bfloat16/float32/float64")
        if not jax.config.x64_enabled and ring_dtype == np.dtype("float64"):
            raise RuntimeError("enable jax_enable_x64 before constructing a float64 ring")
        if self.L * self.shape[1] > _I64_MAX // ring_dtype.itemsize:
            raise ValueError("ring buffer byte size overflows int64")
        if not jax.config.x64_enabled and self.L * self.shape[1] > _I32_MAX:
            raise RuntimeError("enable jax_enable_x64 for int64 ring offsets")
        self.ring = (u.maybe_decimal(jnp.zeros((self.L, self.shape[1]), ring_dtype) * weight_unit)
                     if ring is None else _owned_data(ring, concrete=True))
        self.weighted = bool(weighted)
        if sorted_by_delay:
            self._check_delay_order(delays)
            self._build_delay_metadata(delays)
        else:
            self._sort_by_delay(delays)
        self.head = head

    @classmethod
    def from_prepared(
        cls, data: Any, indices: Any = None, indptr: Any = None, *,
        shape: tuple[int, int] | None = None, delay_steps: Any,
        delay_count: Any, delay_off: Any, row_task_count: Any, capacity: int,
        n_slots: int | None = None, weighted: bool = False,
        ring: Any = None, head: Any = 0,
        workspace: HybridWorkspace | WeightedHybridWorkspace | None = None,
        backend: str | None = None, validate: str = "static",
    ) -> 'delaycsr':
        """Assemble an already sorted CSR and its caller-prepared delay layout.

        Parameters
        ----------
        data : CSR, tuple of arrays or array_like
            Sorted source CSR, ``(weights, indices, indptr)``, or weights with
            separate indices and row pointers. Each row contains consecutive
            delay segments in ``delay_steps`` order, with ascending columns
            within each segment. Weights already follow this edge order and
            have shape ``(1,)`` or ``(nnz,)``; a scalar is normalized to ``(1,)``.
        indices, indptr : array_like or None, optional
            Int32/int64 column indices and row pointers, as in the constructor.
        shape : tuple of int or None, optional
            ``(M, N)``, required unless data is a CSR object.
        delay_steps : tuple of int
            Static, increasing nonnegative delays, with positive length ``K``.
            Supply host integers, not a JAX array or tracer.
        delay_count : array_like
            Nonnegative counts of shape ``(M, K)``, with the indptr dtype.
            Each row must sum to its CSR row length.
        delay_off : array_like
            Int32/int64 element offsets of shape ``(K,)`` equal to
            ``delay_steps[k] * N``. Int64 is required when ``L * N`` exceeds
            INT32_MAX, independently of the column-index dtype.
        row_task_count : array_like
            Int32 counts of shape ``(M,)``. Rows with at most 128 edges use
            zero tasks. Longer rows sum ``ceil(count / 4096)`` over their
            delay segments, including empty segments as zero.
        capacity : int
            Static workspace capacity, at least the sum of all row task
            counts and no larger than INT32_MAX. Zero is supported.
        n_slots : int or None, optional
            Positive ring length ``L``, exceeding every delay. Defaults to
            ``delay_steps[-1] + 1``.
        weighted : bool, optional
            Multiply by event values rather than nonzero activity.
        ring : array_like or brainunit.Quantity or None, optional
            Initial ``(L, N)`` ring. Its floating dtype determines execution
            precision and its unit is preserved. Defaults to zeros in the
            weights' dtype and unit, for dimensionless events.
        head : int or jax.Array, optional
            Initial scalar slot in ``[0, L)``, default zero. The existing
            slot-range check is retained, including for traced heads.
        workspace : HybridWorkspace or WeightedHybridWorkspace or None, optional
            Optional task storage with arrays of length ``capacity``. Begin
            and end arrays use the indptr dtype, output offsets use the
            delay_off dtype, weighted values use the ring dtype, and status
            is int32 with shape ``(2,)``. Omit to allocate fresh storage.
        backend : str or None, optional
            ``'cuda_raw'`` or ``'jax_raw'``. None inherits a source CSR backend
            or uses the execution platform's default.
        validate : {'static', 'full'}, optional
            Static checks inspect shapes, dtypes and host parameters without
            reading connectivity or metadata values. Full checks additionally
            copy concrete layout arrays to the host to verify consistency,
            including segment column order and worst-case task capacity.

        Returns
        -------
        delaycsr
            A complete stateful CSR using the supplied prepared layout.

        Notes
        -----
        No sorting, segment counting, task-count preparation or ordinary CSR
        initialization is performed. JAX operands are reused; NumPy operands
        receive owned snapshots. No per-edge delays are needed or stored, so
        even full validation cannot verify the original edge-delay labels.
        The delivery primitive retains its runtime safety checks. Static
        construction supports traced operands under JIT and differentiation;
        full validation requires concrete layout arrays.

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> from brainevent import delaycsr
            >>> matrix = delaycsr.from_prepared(
            ...     (jnp.array([2.], jnp.float32), jnp.array([1], jnp.int32),
            ...      jnp.array([0, 1], jnp.int32)), shape=(1, 2),
            ...     delay_steps=(1,), delay_count=jnp.array([[1]], jnp.int32),
            ...     delay_off=jnp.array([2], jnp.int32),
            ...     row_task_count=jnp.array([0], jnp.int32), capacity=0,
            ...     backend='jax_raw')
            >>> matrix.L
            2
        """
        if isinstance(data, CSR):
            if indices is not None or indptr is not None:
                raise ValueError("a CSR source already supplies indices and indptr")
            if shape is not None and tuple(shape) != data.shape:
                raise ValueError("shape disagrees with the CSR source")
            shape = data.shape
            if backend is None:
                backend = data.backend
            data, indices, indptr = data.data, data.indices, data.indptr
        elif indices is None and indptr is None:
            if not isinstance(data, (tuple, list)) or len(data) != 3:
                raise TypeError("supply CSR or (weights, indices, indptr)")
            data, indices, indptr = data
        if backend not in (None, "cuda_raw", "jax_raw"):
            raise ValueError("delaycsr backend must be 'cuda_raw', 'jax_raw' or None")
        if (shape is None or len(shape) != 2 or any(
                isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
                or not 0 <= n <= _I32_MAX for n in shape)):
            raise ValueError("shape must contain two nonnegative int32 dimensions")
        if not isinstance(weighted, (bool, np.bool_)):
            raise TypeError("weighted must be bool")
        if validate not in ("static", "full"):
            raise ValueError("validate must be 'static' or 'full'")
        if isinstance(delay_steps, (jax.Array, Tracer)):
            raise TypeError("delay_steps must contain static host integers")
        steps = cls._validate_delay_steps(delay_steps)
        slots = steps[-1] + 1 if n_slots is None else n_slots
        if type(slots) is not int or not steps[-1] < slots <= _I32_MAX:
            raise ValueError("n_slots must exceed every delay and fit int32")
        if type(capacity) is not int or not 0 <= capacity <= _I32_MAX:
            raise ValueError("capacity must be a nonnegative Python int fitting int32")

        weights = _prepared_operand(data, "weights")
        if weights.ndim == 0:
            weights = weights.reshape(1)
        columns = jnp.asarray(_prepared_operand(indices, "indices"))
        pointers = jnp.asarray(_prepared_operand(indptr, "indptr"))
        counts = jnp.asarray(_prepared_operand(delay_count, "delay_count"))
        offsets = jnp.asarray(_prepared_operand(delay_off, "delay_off"))
        tasks = jnp.asarray(_prepared_operand(row_task_count, "row_task_count"))
        if columns.ndim != 1 or pointers.shape != (shape[0] + 1,):
            raise ValueError("indices must have shape (nnz,) and indptr shape (M + 1,)")
        if weights.shape not in ((1,), columns.shape):
            raise ValueError("weights must have shape (1,) or (nnz,)")
        if weights.dtype not in VALUE_SUFFIX:
            raise TypeError("weights must be float16/bfloat16/float32/float64")
        for name, array in (("indices", columns), ("indptr", pointers), ("delay_off", offsets)):
            if array.dtype not in (jnp.int32, jnp.int64):
                raise TypeError(f"{name} must be int32/int64")
        if pointers.dtype == jnp.int32 and columns.size > _I32_MAX:
            raise ValueError("int32 indptr cannot address more than INT32_MAX edges")
        if counts.shape != (shape[0], len(steps)) or offsets.shape != (len(steps),):
            raise ValueError("delay_count must have shape (M, K) and delay_off shape (K,)")
        if counts.dtype != pointers.dtype:
            raise TypeError("delay_count must have the indptr dtype")
        if tasks.shape != (shape[0],):
            raise ValueError("row_task_count must have shape (M,)")
        if tasks.dtype != jnp.int32:
            raise TypeError("row_task_count must be int32")

        ring = None if ring is None else _prepared_operand(ring, "ring")
        ring_dtype = weights.dtype if ring is None else ring.dtype
        if ring is not None and ring.shape != (slots, shape[1]):
            raise ValueError("ring must have shape (L, N)")
        if ring_dtype not in VALUE_SUFFIX:
            raise TypeError("ring must use float16/bfloat16/float32/float64")
        if slots * shape[1] > _I64_MAX // ring_dtype.itemsize:
            raise ValueError("ring buffer byte size overflows int64")
        if not jax.config.x64_enabled and slots * shape[1] > _I32_MAX:
            raise RuntimeError("enable jax_enable_x64 for int64 ring offsets")
        if offsets.dtype == jnp.int32 and slots * shape[1] > _I32_MAX:
            raise ValueError("use int64 delay_off for a ring larger than INT32_MAX elements")

        result = object.__new__(cls)
        result.__dict__.update(CSR._from_parts(
            weights, columns, pointers, shape=shape, backend=backend).__dict__)
        result.delay_steps, result.L, result.capacity = steps, slots, capacity
        result.delay_count, result.delay_off, result.row_task_count = counts, offsets, tasks
        result.weighted = bool(weighted)
        result.head = head
        if validate == "full":
            result._check_prepared_values()
        workspace_type = WeightedHybridWorkspace if weighted else HybridWorkspace
        if workspace is not None:
            if not isinstance(workspace, workspace_type):
                raise TypeError(f"workspace must be {workspace_type.__name__}")
            arrays = tuple(jnp.asarray(_prepared_operand(value, "workspace")) for value in workspace)
            dtypes: tuple[np.dtype, ...] = (pointers.dtype, pointers.dtype, offsets.dtype)
            if weighted:
                dtypes += (ring_dtype,)
            shapes = ((capacity,),) * len(dtypes) + ((2,),)
            for value, dtype, expected_shape in zip(arrays, (*dtypes, jnp.dtype(jnp.int32)), shapes):
                if value.shape != expected_shape:
                    raise ValueError("workspace arrays must match capacity and status shape (2,)")
                if value.dtype != dtype:
                    raise TypeError("workspace dtypes must match indptr, delay_off and ring")
            if any(value is not original for value, original in zip(arrays, workspace)):
                workspace = workspace_type(*arrays)
        result.ring = (u.maybe_decimal(jnp.zeros((slots, shape[1]), ring_dtype) * u.get_unit(weights))
                       if ring is None else ring)
        result.workspace = (result._make_workspace(ring_dtype, result.weighted)
                            if workspace is None else workspace)
        return result

    def _check_prepared_values(self) -> None:
        """Validate concrete layout contents without replacing prepared arrays."""
        arrays = (self.indices, self.indptr, self.delay_count, self.delay_off, self.row_task_count)
        if any(isinstance(value, Tracer) for value in arrays):
            raise ValueError("validate='full' requires concrete layout arrays; use 'static' under JIT")
        columns, pointers, counts, offsets, tasks = (np.asarray(value) for value in arrays)
        if (pointers[0] != 0 or pointers[-1] != columns.size
                or np.any(pointers < 0) or np.any(pointers > columns.size)
                or np.any(pointers[1:] < pointers[:-1])):
            raise ValueError("inconsistent prepared CSR indptr")
        if np.any(columns < 0) or np.any(columns >= self.shape[1]):
            raise ValueError("prepared indices must be within the column bounds")
        if tuple(int(value) for value in offsets) != tuple(
                step * self.shape[1] for step in self.delay_steps):
            raise ValueError("delay_off must equal delay_steps * N")
        total_tasks = 0
        for row in range(self.shape[0]):
            first, last = int(pointers[row]), int(pointers[row + 1])
            start, expected_tasks = first, 0
            for value in counts[row]:
                count = int(value)
                end = start + count
                if count < 0 or end > last:
                    raise ValueError("delay_count must be nonnegative and stay within each CSR row")
                if count > 1 and np.any(columns[start + 1:end] < columns[start:end - 1]):
                    raise ValueError("prepared columns must be sorted within each delay segment")
                expected_tasks += count // TASK_NNZ + (count % TASK_NNZ != 0)
                start = end
            if start != last:
                raise ValueError("delay_count must sum to each CSR row length")
            if last - first <= TPR_THRESHOLD:
                expected_tasks = 0
            if int(tasks[row]) != expected_tasks:
                raise ValueError("row_task_count disagrees with the hybrid task layout")
            total_tasks += expected_tasks
        if total_tasks > self.capacity:
            raise ValueError("capacity must cover the sum of all row_task_count values")

    @staticmethod
    def _validate_delay_steps(values: Any) -> tuple[int, ...]:
        steps = np.asarray(values)
        if (steps.ndim != 1 or steps.size == 0 or steps.dtype.kind not in "iu"
                or np.any(steps < 0) or np.any(steps >= _I32_MAX)
                or np.any(steps[1:] <= steps[:-1])):
            raise ValueError("delay_steps must be increasing nonnegative integers below INT32_MAX")
        return tuple(int(value) for value in steps)

    def _validate_edge_delays(self, edge_delays: Any) -> np.ndarray:
        delays = np.array(edge_delays, copy=True)
        if (delays.dtype.kind not in "iu" or np.any(delays < 0)
                or np.any(delays >= _I32_MAX)):
            raise ValueError("edge_delays must be nonnegative integers below INT32_MAX")
        if delays.size == 1 and delays.shape != self.indices.shape:
            delays = np.full(self.indices.shape, delays.item(), dtype=delays.dtype)
        if delays.shape != self.indices.shape:
            raise ValueError("edge_delays must be a scalar or contain one integer per edge")
        return delays

    def _check_delay_order(self, delays: np.ndarray) -> None:
        columns, pointers = np.asarray(self.indices), np.asarray(self.indptr)
        for row in range(self.shape[0]):
            first, last = int(pointers[row]), int(pointers[row + 1])
            d, c = delays[first:last], columns[first:last]
            if np.any((d[1:] < d[:-1]) | ((d[1:] == d[:-1]) & (c[1:] < c[:-1]))):
                raise ValueError(f"sorted_by_delay=True requires (delay, column) order in row {row}")

    def _check_delay_segments(self, delays: np.ndarray) -> None:
        if np.any(~np.isin(delays, self.delay_steps)):
            raise ValueError("edge delay absent from delay_steps")

    def sort_by_delay(self, edge_delays: Any) -> None:
        """Sort each row by (delay, column) and overwrite this object's CSR.

        Parameters
        ----------
        edge_delays : int or array_like
            Concrete integer delay shared by all edges, or delays aligned
            with the current CSR edge order. A one-element array broadcasts.
            Each delay must be present in ``self.delay_steps``.

        Notes
        -----
        Per-edge weights follow the column permutation; shared weights retain
        one value. This initialization-time method rebuilds delay metadata and
        workspace, invalidates edge-order caches, and stores no per-edge delays.
        """
        delays = self._validate_edge_delays(edge_delays)
        self._check_delay_segments(delays)
        self._sort_by_delay(delays)

    def _sort_by_delay(self, delays: np.ndarray) -> None:
        """Sort an owned, validated delay array and rebuild prepared metadata."""
        values, unit = u.split_mantissa_unit(self.data)
        columns, weights = np.asarray(self.indices).copy(), np.asarray(values).copy()
        pointers = np.asarray(self.indptr)
        for row in range(self.shape[0]):
            first, last = int(pointers[row]), int(pointers[row + 1])
            order = np.lexsort((columns[first:last], delays[first:last]))
            columns[first:last] = columns[first:last][order]
            delays[first:last] = delays[first:last][order]
            if weights.size != 1:
                weights[first:last] = weights[first:last][order]
        self.indices = jax.device_put(columns)
        self.data = u.maybe_decimal(jax.device_put(weights) * unit)
        self._cache = type(self._cache)()
        self._build_delay_metadata(delays)

    def _build_delay_metadata(self, delays: np.ndarray) -> None:
        """Build segment counts and task storage directly on this CSR object."""
        pointers = np.asarray(self.indptr)
        counts = np.zeros((self.shape[0], len(self.delay_steps)), dtype=pointers.dtype)
        for row in range(self.shape[0]):
            first, last = int(pointers[row]), int(pointers[row + 1])
            segments = np.searchsorted(self.delay_steps, delays[first:last])
            counts[row] = np.bincount(segments, minlength=len(self.delay_steps))
        tasks = (counts.astype(np.int64) // TASK_NNZ + (counts % TASK_NNZ != 0)).sum(axis=1)
        tasks[np.diff(pointers) <= TPR_THRESHOLD] = 0
        self.capacity = int(tasks.sum())
        if self.capacity > _I32_MAX:
            raise ValueError("hybrid task capacity exceeds int32")
        offset_dtype = np.int32 if self.L * self.shape[1] <= _I32_MAX else np.int64
        self.delay_count = jnp.asarray(counts)
        self.delay_off = jnp.asarray(np.asarray(
            [step * self.shape[1] for step in self.delay_steps], dtype=offset_dtype))
        self.row_task_count = jnp.asarray(tasks.astype(np.int32))
        self.workspace = self._make_workspace(self.ring.dtype, self.weighted)

    def _make_workspace(self, dtype: Any, weighted: bool) -> HybridWorkspace | WeightedHybridWorkspace:
        """Create task storage for an explicit execution dtype and event mode."""
        ranges = (jnp.zeros(self.capacity, self.indptr.dtype),
                  jnp.zeros(self.capacity, self.indptr.dtype),
                  jnp.zeros(self.capacity, self.delay_off.dtype))
        status = jnp.zeros(2, jnp.int32)
        return (WeightedHybridWorkspace(*ranges, jnp.zeros(self.capacity, dtype), status)
                if weighted else HybridWorkspace(*ranges, status))

    def build_weight_indices(self) -> CSR:
        """Export a plain CSR with its column-access indices prepared.

        Returns
        -------
        CSR
            Static weights and structure without delay metadata or execution
            state. The source delaycsr's delay state is unchanged.

        Warns
        -----
        UserWarning
            The returned CSR is for static data extraction, not delay delivery.
        """
        warnings.warn(
            "build_weight_indices() returns a plain CSR without delay information, "
            "for static weight/structure extraction only; it cannot deliver delayed events. "
            "The source delaycsr's delay state is unchanged.",
            UserWarning, stacklevel=2,
        )
        return super().build_weight_indices()

    @classmethod
    # Static CSR can infer structure; delay construction requires an explicit template.
    def fromdense(cls, mat: Any, *, template: 'delaycsr') -> 'delaycsr':  # type: ignore[override]
        """Replace template-edge values from a dense matrix without changing delay.

        Parameters
        ----------
        mat : array_like or brainunit.Quantity
            Dense weights with the template's shape, dtype and unit. Values
            outside its stored coordinates must be zero. Stored edges remain
            present when their new value is zero.
        template : delaycsr
            Initialized object supplying the current edge order, delay layout
            and execution state. Repeated coordinates are not representable
            by one dense value and are rejected.

        Returns
        -------
        delaycsr
            A copy with per-edge weights, retaining the template's layout and
            historical ring/head/workspace. The template is unchanged.

        Raises
        ------
        ValueError
            Shape differs, a new coordinate is nonzero, or coordinates repeat.
        TypeError
            The template, weight dtype or unit is incompatible.
        RuntimeError
            Inputs needed for validation are traced. Call outside JIT.

        Warns
        -----
        UserWarning
            Only numerical updates can reuse the template's delay information.

        Notes
        -----
        A classmethod cannot infer an instance's delay state: always pass
        ``template`` explicitly. Changes to connectivity require new structure
        and delays; interpreting them as a value update would give wrong
        arrival times. Use ``with_data`` for traced per-edge updates.
        """
        if not isinstance(template, cls) or not hasattr(template, "delay_count"):
            raise TypeError("fromdense requires an initialized delaycsr template")
        template._concrete_layout("fromdense", mat)
        if isinstance(mat, u.sparse.SparseMatrix):
            raise TypeError("fromdense expects a dense weight array")
        if hasattr(mat, "dtype") and mat.dtype != template.dtype:
            raise TypeError("fromdense weights must keep the template dtype")
        mat = u.math.asarray(mat)
        if mat.shape != template.shape:
            raise ValueError("fromdense matrix shape must match the template")
        template._check_weight_type(mat)
        pointers = np.asarray(template.indptr)
        rows = np.repeat(np.arange(template.shape[0], dtype=np.int64), np.diff(pointers))
        cols = np.asarray(template.indices).astype(np.int64, copy=False)
        coordinates = rows * template.shape[1] + cols
        if np.unique(coordinates).size != coordinates.size:
            raise ValueError("fromdense cannot recover per-edge values for duplicate coordinates")
        nonzero_rows, nonzero_cols = np.nonzero(np.asarray(u.get_mantissa(mat)))
        nonzero = nonzero_rows.astype(np.int64) * template.shape[1] + nonzero_cols
        if np.any(~np.isin(nonzero, coordinates)):
            raise ValueError("fromdense cannot add nonzero coordinates outside the template structure")
        warnings.warn(
            "fromdense only updates template-edge values and reuses existing delay information. "
            "Connectivity changes invalidate that correspondence and would give wrong arrival "
            "times; provide new structure and delays instead. Zero values retain stored edges.",
            UserWarning, stacklevel=2,
        )
        return template._same_structure(mat[jnp.asarray(rows, jnp.int32), template.indices])

    def slice_rows(self, index: Any) -> 'delaycsr':
        """Select complete pre rows and create a delaycsr with fresh execution state.

        Parameters
        ----------
        index : int, slice or one-dimensional integer array_like
            Concrete rows, preserving selection order and duplicates. Negative
            indices and empty selections are supported.

        Returns
        -------
        delaycsr
            Selected connectivity and delay segments, with zero ring/head and
            new workspace/configuration. Ring dtype and unit are preserved.

        Raises
        ------
        IndexError
            A row is non-integral, out of bounds or not one-dimensional.
        RuntimeError
            Layout/selection is traced, or int64 offsets need JAX x64.
        ValueError
            Edge or task capacity exceeds the supported range.

        Notes
        -----
        The original ring aggregates history from every pre row; selected-row
        history cannot be separated from it. This operation starts a new
        subnetwork, without transferring any original arrivals.
        """
        self._concrete_layout("slice_rows", index)
        if isinstance(index, slice):
            rows = normalize_row_index(index, self.shape[0])
        else:
            raw = np.asarray(index)
            if isinstance(index, (list, tuple)) and raw.size == 0:
                raw = np.asarray(index, dtype=np.int32)
            if raw.ndim > 1 or raw.dtype.kind not in "iu":
                raise IndexError("row selection must contain one-dimensional integer indices")
            if (np.any(raw >= self.shape[0])
                    or (raw.dtype.kind == "i" and np.any(raw < -self.shape[0]))):
                raise IndexError("row index out of bounds")
            rows = normalize_row_index(raw.astype(np.int32), self.shape[0])
        rows = jnp.atleast_1d(rows)
        if rows.size > _I32_MAX:
            raise ValueError("selected row count exceeds int32")
        host_rows = np.asarray(rows)
        pointers = np.asarray(self.indptr)
        lengths = pointers[host_rows + 1] - pointers[host_rows]
        checked_pointers = _checked_indptr(lengths, self.indptr.dtype)
        # The shared helper accumulates in its input pointer dtype. Promote
        # before calling it, after checking the complete new edge count.
        data, indices, indptr, shape = build_sub_csr(
            self.data, self.indices, self.indptr.astype(checked_pointers.dtype), rows, self.shape[1])
        tasks = self.row_task_count[rows]
        capacity = sum(int(value) for value in np.asarray(tasks))
        if capacity > _I32_MAX:
            raise ValueError("hybrid task capacity exceeds int32")
        counts = self.delay_count[rows].astype(indptr.dtype)
        config = _prepare_csr_config(
            indices, indptr, shape=shape, backend=self.backend, config=self.config)
        result = type(self).from_prepared(
            data, indices, indptr, shape=shape, delay_steps=self.delay_steps,
            delay_count=counts, delay_off=self.delay_off, row_task_count=tasks,
            capacity=capacity, n_slots=self.L, weighted=self.weighted,
            ring=self._empty_ring(self.L), head=0, backend=self.backend,
        )
        result.config = config
        result.backward_algorithm = self.backward_algorithm
        return result

    # Requiring increment delays deliberately excludes the parent's no-delay call.
    def diag_add(self, other: Any, *, diag_delays: Any) -> 'delaycsr':  # type: ignore[override]
        """Add independently delayed diagonal edges through full initialization.

        Parameters
        ----------
        other : one-dimensional array_like or brainunit.Quantity
            Diagonal increments of length ``min(shape)`` with compatible units.
            Values are converted to the weight unit and dtypes are promoted.
        diag_delays : int or one-dimensional integer array_like
            Required nonnegative delay for each increment, or one scalar
            broadcast to all diagonal positions. Old edge delays are unchanged.

        Returns
        -------
        delaycsr
            Fully rebuilt object with one additional edge per diagonal value,
            including zeros and repeated coordinates. The new ring/head and
            workspace are zero, with the original ring's dtype and unit.

        Raises
        ------
        ValueError
            Value/delay shapes or delay ranges are invalid, or capacity overflows.
        TypeError
            Values, units or promoted weight dtype are incompatible.
        RuntimeError
            Inputs are traced, or wide values/offsets require JAX x64.

        Warns
        -----
        PerformanceWarning
            Full host preparation, sorting and storage allocation can be costly.

        Notes
        -----
        Each increment is a new edge, never a retiming of an old diagonal edge.
        This method calls the ordinary initializer outside JIT and resets
        execution history. It is not intended for per-timestep weight updates.
        """
        self._concrete_layout("diag_add", self.data, other, diag_delays)
        if isinstance(other, u.sparse.SparseMatrix):
            raise TypeError("diag_add expects a one-dimensional diagonal array")
        explicit_dtype = getattr(other, "dtype", None)
        if not jax.config.x64_enabled and explicit_dtype in (
                np.dtype("int64"), np.dtype("uint64"), np.dtype("float64")):
            raise RuntimeError("enable jax_enable_x64 before supplying wide diagonal values")
        diagonal = u.math.asarray(other)
        size = min(self.shape)
        if diagonal.shape != (size,):
            raise ValueError("diag_add values must have shape (min(shape),)")
        delays = np.asarray(diag_delays)
        if (delays.dtype.kind not in "iu" or np.any(delays < 0)
                or np.any(delays >= _I32_MAX)):
            raise ValueError("diag_delays must be nonnegative integers below INT32_MAX")
        if delays.ndim == 0:
            delays = np.full((size,), int(delays), dtype=np.int64)
        elif delays.shape != (size,):
            raise ValueError("diag_delays must be a scalar or have shape (min(shape),)")
        else:
            delays = delays.astype(np.int64)
        converted = u.Quantity(diagonal).to(u.get_unit(self.data)).mantissa
        dtype = jnp.result_type(self.dtype, converted.dtype)
        if dtype not in VALUE_SUFFIX:
            raise TypeError("diag_add requires a supported floating weight dtype")
        warnings.warn(
            "delaycsr.diag_add() fully rebuilds through __init__: host data transfers, "
            "per-edge expansion, full sorting, configuration preparation and ring/workspace "
            "allocation may have significant performance and memory costs. Do not call it "
            "every timestep. The new ring/head/workspace are reset.",
            PerformanceWarning, stacklevel=2,
        )
        weights = np.broadcast_to(np.asarray(u.get_mantissa(self.data)), (self.nse,)).astype(dtype)
        increments = np.asarray(converted, dtype=dtype)
        pointers, columns = np.asarray(self.indptr), np.asarray(self.indices)
        old_delays = np.concatenate([
            np.repeat(np.asarray(self.delay_steps, dtype=np.int64), row)
            for row in np.asarray(self.delay_count)
        ]) if self.shape[0] else np.empty((0,), dtype=np.int64)
        lengths = np.diff(pointers).astype(np.int64)
        lengths[:size] += 1
        indptr = _checked_indptr(lengths, self.indptr.dtype)
        new_weights, new_columns, new_delays = [], [], []
        for row in range(self.shape[0]):
            first, last = int(pointers[row]), int(pointers[row + 1])
            new_weights.append(weights[first:last])
            new_columns.append(columns[first:last])
            new_delays.append(old_delays[first:last])
            if row < size:
                new_weights.append(increments[row:row + 1])
                new_columns.append(np.asarray([row], dtype=columns.dtype))
                new_delays.append(delays[row:row + 1])
        weights = np.concatenate(new_weights) if new_weights else np.empty((0,), dtype=dtype)
        columns = np.concatenate(new_columns) if new_columns else np.empty((0,), dtype=columns.dtype)
        edge_delays = np.concatenate(new_delays) if new_delays else np.empty((0,), dtype=np.int64)
        steps = tuple(sorted(set(self.delay_steps).union(int(value) for value in delays)))
        slots = max(self.L, int(delays.max()) + 1) if size else self.L
        result = type(self)(
            (u.maybe_decimal(weights * u.get_unit(self.data)), columns, indptr),
            edge_delays=edge_delays, shape=self.shape, delay_steps=steps,
            n_slots=slots, sorted_by_delay=False, weighted=self.weighted,
            ring=self._empty_ring(slots), head=0, backend=self.backend,
        )
        result.backward_algorithm = self.backward_algorithm
        return result

    @property
    def weights(self) -> jax.Array | u.Quantity:
        """jax.Array or Quantity: CSR weights in the initialized delay order."""
        return self.data

    @property
    def head(self) -> jax.Array:
        """jax.Array: Current scalar int32 slot in ``[0, self.L)``."""
        return self._head

    @head.setter
    def head(self, value: Any) -> None:
        if not (isinstance(value, int) and not isinstance(value, bool)):
            # Keep host integers wide even when JAX x64 is disabled. JAX
            # arrays and tracers likewise retain their dtype until checked.
            if not isinstance(value, (jax.Array, Tracer)):
                value = np.asarray(value)
            if value.ndim != 0 or value.dtype.kind not in "iu":
                raise TypeError("head must be an integer scalar slot")
        self._head = jnp.asarray(_checked_head(value, self.L), jnp.int32)

    def forward(self, x: Any, t: Any) -> jax.Array | u.Quantity:
        """Set the current time, clear its slot, deliver events and read next.

        Parameters
        ----------
        x : array_like, BinaryArray or WeightedArray
            Single-step events with shape ``(shape[0],)``. Event wrappers must
            agree with this object's weighted mode.
        t : int or jax.Array
            Global integer timestep, reduced modulo the ring length.

        Returns
        -------
        jax.Array
            The next ring slot, with shape ``(shape[1],)``. The updated ring,
            head and workspace are stored on this object.

        Notes
        -----
        For JIT or scan, pass this object as a pytree argument and return the
        updated object. Closing over this mutable object is unsupported.
        """
        head = _time_slot(t, self.L)
        # Modulo already guarantees the slot range. Avoid reading a device
        # scalar back to the host just to validate this derived head again.
        self._head = head
        return self.__rmatmul__(x)

    def __rmatmul__(self, x: Any) -> jax.Array | u.Quantity:
        """Clear the current slot, deliver events and read the next slot.

        Parameters
        ----------
        x : array_like, BinaryArray or WeightedArray
            Single-step events in this object's weighted mode.

        Returns
        -------
        jax.Array
            The next ring slot after delivery. The head is not advanced.
        """
        events, weighted = _resolve_events(x, self.weighted)
        ring = _clear_ring_slot(self.ring, self.head)
        self.ring, self.workspace = self._project(
            ring, events, self.head, self.workspace, weighted=weighted, backend=self.backend)
        return self.read_spike()

    def _project(
        self, ring: Any, events: Any, head: jax.Array,
        workspace: HybridWorkspace | WeightedHybridWorkspace, *, weighted: bool, backend: str | None,
    ) -> tuple[jax.Array | u.Quantity, HybridWorkspace | WeightedHybridWorkspace]:
        """Add into an explicit ring and return results without changing this object."""
        if ring.shape != (self.L, self.shape[1]):
            raise ValueError("ring must have shape (L, N)")
        workspace_type = WeightedHybridWorkspace if weighted else HybridWorkspace
        if not isinstance(workspace, workspace_type):
            raise RuntimeError(f"hybrid workspace must be {workspace_type.__name__}")
        weights, weight_unit = u.split_mantissa_unit(self.data)
        events, event_unit = u.split_mantissa_unit(events)
        ring_values, ring_unit = u.split_mantissa_unit(ring)
        ring_values = jnp.asarray(ring_values)
        contribution_unit = weight_unit * event_unit
        if contribution_unit != ring_unit:
            # Convert contributions, not stored arrivals: the ring's unit is
            # part of its pytree structure and must stay fixed across steps.
            # Scale in at least float32 before narrowing to execution storage.
            conversion_dtype = jnp.result_type(weights.dtype, ring_values.dtype, jnp.float32)
            weights = u.Quantity(jnp.asarray(weights, dtype=conversion_dtype),
                                 unit=contribution_unit).to_decimal(ring_unit)
        # Preserve original events at the differentiable boundary. Only CUDA
        # lowering adapts its event ABI; the ring sets the value precision.
        weights = jnp.asarray(weights, dtype=ring_values.dtype)
        events = jnp.asarray(events)
        operands = (ring_values, events, weights, self.indices, self.indptr, self.delay_count,
                    self.delay_off, head.reshape(1), self.row_task_count, *workspace)
        outputs = tuple(jax.ShapeDtypeStruct(a.shape, a.dtype) for a in (ring_values, *workspace))
        result = csr_delay_deliver_layout_p(
            *operands, outs=outputs, backend=backend, weighted=weighted, check_status=True)
        updated = u.Quantity(result[0], unit=ring_unit) if isinstance(ring, u.Quantity) else result[0]
        return updated, workspace_type(*result[1:])

    def clear_head(self) -> jax.Array | u.Quantity:
        """Clear the consumed current slot using JAX, without advancing head.

        Returns
        -------
        jax.Array
            The updated complete ring, also stored on this object. Old values
            in the current slot have zero derivative; other slots are preserved.
        """
        self.ring = _clear_ring_slot(self.ring, self.head)
        return self.ring

    def deliver(self, ring: Any, events: Any, t: Any, *,
                weighted: bool | None = None, backend: str | None = None) -> jax.Array | u.Quantity:
        """Deliver into an external ring using this object's prepared layout.

        Parameters
        ----------
        ring : jax.Array or brainunit.Quantity
            Arrival buffer of shape ``(L, shape[1])``. Its floating dtype
            determines delivery precision; units must be compatible with the
            product of weight and event units and are preserved on return.
        events : array_like, BinaryArray or WeightedArray
            Single-step boolean, integer or floating events. Original event
            dtypes are retained for differentiation.
        t : int or jax.Array
            Global integer time step, reduced modulo ``L``.
        weighted : bool or None, optional
            Event mode. None infers it from the wrapper, defaulting to binary.
        backend : str or None, optional
            ``cuda_raw`` or ``jax_raw`` override for this call. None uses
            the inherited CSR backend, which may itself select the platform
            default. The override leaves this object's backend unchanged.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Full ring with delayed contributions added, preserving the input
            ring's dtype and unit. No slot is cleared and this object's ring,
            head and workspace are unchanged.
        """
        events, weighted = _resolve_events(events, weighted)
        ring = ring if isinstance(ring, u.Quantity) else u.math.asarray(ring)
        workspace = self._make_workspace(ring.dtype, weighted)
        result, _ = self._project(
            ring, events, _time_slot(t, self.L), workspace, weighted=weighted,
            backend=self.backend if backend is None else backend)
        return result

    def read_spike(self, target_t: Any = None) -> jax.Array | u.Quantity:
        """Read a ring slot without changing this object's state.

        Parameters
        ----------
        target_t : int or jax.Array or None, optional
            Global timestep reduced modulo the ring length. None reads the
            slot immediately after the current head.

        Returns
        -------
        jax.Array
            The selected ring slot, with shape ``(shape[1],)``.
        """
        slot = ((self.head + 1) % self.L if target_t is None
                else _time_slot(target_t, self.L))
        return self.ring[slot]

    def with_data(self, data: Any) -> 'delaycsr':
        """Return a copy with weights in this object's current edge order.

        Parameters
        ----------
        data : array
            New weights aligned with ``self.data`` after delay initialization
            and sorting. Shape, dtype and unit must match ``self.data``.
            Shared weights retain shape ``(1,)``; per-edge weights retain
            shape ``(nnz,)``. CSR and delaycsr objects are not weight arrays.

        Returns
        -------
        delaycsr
            A new object holding the supplied weights and sharing this
            object's prepared structure, ring, head and workspace. The
            original object is unchanged.

        Raises
        ------
        TypeError
            If called without an initialized delaycsr, if data is not a
            weight array, or if its dtype or unit differs.
        ValueError
            If the new weight shape differs from ``self.data.shape``.

        Notes
        -----
        Initialize a delaycsr before updating its weights. This method never
        sorts or maps weights from the original CSR order. The caller must
        supply weights in the current ``self.data`` order; an array alone
        cannot identify which edge each value belongs to. Traced weights
        are supported under JIT and automatic differentiation.

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> from brainevent import delaycsr
            >>> w = delaycsr(
            ...     (jnp.array([1., 2.], jnp.float32), jnp.array([0, 1], jnp.int32),
            ...      jnp.array([0, 2], jnp.int32)),
            ...     shape=(1, 2), edge_delays=jnp.array([2, 1], jnp.int32))
            >>> w.data
            Array([2., 1.], dtype=float32)
            >>> updated = w.with_data(jnp.array([10., 20.], jnp.float32))
            >>> updated.data
            Array([10., 20.], dtype=float32)
        """
        if not isinstance(self, delaycsr) or not hasattr(self, "delay_count"):
            raise TypeError("with_data requires an initialized delaycsr; initialize delaycsr first")
        if isinstance(data, CSR) or not (hasattr(data, "shape") and hasattr(data, "dtype")):
            raise TypeError("with_data expects a weight array in the current delaycsr.data order")
        if data.shape != self.data.shape:
            raise ValueError(f"with_data weights must have shape {self.data.shape}; got {data.shape}")
        if data.dtype != self.data.dtype:
            raise TypeError(f"with_data weights must have dtype {self.data.dtype}; got {data.dtype}")
        if u.get_unit(data) != u.get_unit(self.data):
            raise TypeError("with_data weights must have the same unit as delaycsr.data")
        return self._replace_data(_owned_data(data))

    def __matmul__(self, other: Any):
        """Reject the unimplemented ``delaycsr @ other`` operation.

        Parameters
        ----------
        other : Any
            Right-hand operand.

        Raises
        ------
        NotImplementedError
            Always. For delayed delivery use pre events with ``events @ self``
            or ``forward(events, t)``; these are different product directions.
        """
        raise NotImplementedError(
            "delaycsr @ other is not implemented. For delayed delivery, "
            "use events @ delaycsr or forward(events, t).")

    def __getitem__(self, index: Any):
        """Reject indexing until delaycsr has a supported indexing contract.

        Parameters
        ----------
        index : Any
            Requested selector.

        Raises
        ------
        NotImplementedError
            Always. Use ``slice_rows`` for a fresh delay subnetwork or
            ``todense()[index]`` to explicitly extract static dense weights.
        """
        raise NotImplementedError(
            "delaycsr.__getitem__ is not implemented. Use slice_rows(index) for a delaycsr "
            "subgraph, or todense()[index] to extract static weights explicitly.")

    def prepare_diag_add(self):
        """Reject the parent diagonal planner, which delay rebuilding does not use.

        Raises
        ------
        NotImplementedError
            Always. ``diag_add(..., diag_delays=...)`` rebuilds through the
            ordinary initializer outside JIT, without a parent diagonal plan.
        """
        raise NotImplementedError(
            "delaycsr.prepare_diag_add() is not implemented. "
            "diag_add(..., diag_delays=...) rebuilds through __init__ outside JIT "
            "and does not use the parent CSR diagonal plan.")

    def apply(self, fn: Any) -> 'delaycsr':
        """Transform stored weights while retaining their shape, dtype and unit.

        Parameters
        ----------
        fn : callable
            Function applied to the current weight buffer.

        Returns
        -------
        delaycsr
            A copy retaining connectivity, delay layout and execution history.

        Raises
        ------
        ValueError
            The physical weight-buffer shape changes.
        TypeError
            The dtype or unit changes. No ring/workspace conversion is done.
        """
        return self._same_structure(self._apply_data(fn))

    def apply2(self, other: Any, fn: Any, *, reverse: bool = False):
        """Apply a binary weight operation with checked delay-layout compatibility.

        Parameters
        ----------
        other : array_like, brainunit.Quantity or delaycsr
            Numeric operand, or a delaycsr sharing this exact layout.
        fn : callable
            Binary operation supported by the parent CSR implementation.
        reverse : bool, optional
            Reverse numerical operands; the result still uses this object's
            layout and history, without combining the other object's ring.

        Returns
        -------
        delaycsr or array
            Sparse results retain the weight dtype/unit and have one shared
            or one per-edge value. Addition/subtraction return static arrays.

        Raises
        ------
        TypeError
            A sparse operand lacks delay metadata, or sparse-result dtype/unit
            changes.
        ValueError
            A shared delay layout cannot be proven or result shape is invalid.

        Notes
        -----
        Equal values in independently built layout arrays do not establish
        compatibility. Identity checks avoid host comparisons of JAX tracers.
        """
        self._check_sparse_operand(other)
        return super().apply2(other, fn, reverse=reverse)

    def _binary_op(self, other, op):
        self._check_sparse_operand(other)
        if isinstance(other, delaycsr) and op in (operator.add, operator.sub):
            other = other.todense()
        return super()._binary_op(other, op)

    def _binary_rop(self, other, op):
        self._check_sparse_operand(other)
        if isinstance(other, delaycsr) and op in (operator.add, operator.sub):
            other = other.todense()
        return super()._binary_rop(other, op)

    def update_on_pre(self, pre_spike, post_trace, w_min=None, w_max=None) -> 'delaycsr':
        """Delegate the presynaptic weight update with int32 column indices.

        Parameters
        ----------
        pre_spike : array_like
            Current presynaptic activity, shape ``(shape[0],)``.
        post_trace : array_like or brainunit.Quantity
            Supplied postsynaptic trace, shape ``(shape[1],)``.
        w_min, w_max : scalar or None, optional
            Parent update's clipping bounds, compatible with weight units.

        Returns
        -------
        delaycsr
            Updated weights with the same delay layout and historical state.

        Raises
        ------
        TypeError
            Original column indices are int64, on any backend.

        Notes
        -----
        Signals are not automatically aligned to each edge's delay. Weights
        must be per-edge as required by the parent plasticity operation.
        """
        self._require_int32_indices("update_on_pre")
        return super().update_on_pre(pre_spike, post_trace, w_min, w_max)

    def update_on_post(self, pre_trace, post_spike, w_min=None, w_max=None) -> 'delaycsr':
        """Delegate the postsynaptic weight update with int32 column indices.

        Parameters
        ----------
        pre_trace : array_like or brainunit.Quantity
            Supplied presynaptic trace, shape ``(shape[0],)``.
        post_spike : array_like
            Current postsynaptic activity, shape ``(shape[1],)``.
        w_min, w_max : scalar or None, optional
            Parent update's weight clipping bounds.

        Returns
        -------
        delaycsr
            Updated weights retaining delay layout and execution history.

        Raises
        ------
        TypeError
            Original column indices are int64, before any mirror preparation.

        Notes
        -----
        Supplied signals are not automatically aligned by edge delay.
        """
        self._require_int32_indices("update_on_post")
        return super().update_on_post(pre_trace, post_spike, w_min, w_max)

    def apply_on_pre(self, rule, pre_spike, *, edge_state=None, pre_vals=None, post_vals=None, params=None):
        """Apply a supplied presynaptic edge rule with int32 column indices.

        Parameters
        ----------
        rule : callable
            Parent CSR plasticity rule on one edge state.
        pre_spike : array_like
            Supplied presynaptic activity.
        edge_state : pytree or None, optional
            Per-edge state in the current delay-sorted order; None uses weights.
        pre_vals, post_vals : pytree or None, optional
            Supplied per-neuron values, without automatic delay alignment.
        params : Any, optional
            Rule parameters.

        Returns
        -------
        pytree
            New edge state; use ``with_data`` explicitly to install new weights.

        Raises
        ------
        TypeError
            Original column indices are int64.
        """
        self._require_int32_indices("apply_on_pre")
        return super().apply_on_pre(
            rule, pre_spike, edge_state=edge_state, pre_vals=pre_vals, post_vals=post_vals, params=params)

    def apply_on_post(self, rule, post_spike, *, edge_state=None, pre_vals=None, post_vals=None, params=None):
        """Apply a supplied postsynaptic edge rule with int32 column indices.

        Parameters
        ----------
        rule : callable
            Parent CSR plasticity rule on one edge state.
        post_spike : array_like
            Supplied postsynaptic activity.
        edge_state : pytree or None, optional
            Per-edge state in the current delay-sorted order; None uses weights.
        pre_vals, post_vals : pytree or None, optional
            Supplied per-neuron values, without automatic delay alignment.
        params : Any, optional
            Rule parameters.

        Returns
        -------
        pytree
            New edge state, without replacing this object's weights or history.

        Raises
        ------
        TypeError
            Original column indices are int64.
        """
        self._require_int32_indices("apply_on_post")
        return super().apply_on_post(
            rule, post_spike, edge_state=edge_state, pre_vals=pre_vals, post_vals=post_vals, params=params)

    def dt2t(self, y_dim_arr, w_dim_arr):
        """Map supplied row values to stored edges with int32 column indices.

        Parameters
        ----------
        y_dim_arr : array_like or brainunit.Quantity
            Supplied row values, shape ``(shape[0],)``.
        w_dim_arr : array_like or brainunit.Quantity
            Per-edge values, shape ``(nnz,)``.

        Returns
        -------
        array or brainunit.Quantity
            Parent static edge mapping; no history is read or delay-aligned.

        Raises
        ------
        TypeError
            Original column indices are int64.
        """
        self._require_int32_indices("dt2t")
        return super().dt2t(y_dim_arr, w_dim_arr)

    def dt2t_transposed(self, y_dim_arr, w_dim_arr):
        """Map supplied column values to stored edges with int32 indices.

        Parameters
        ----------
        y_dim_arr : array_like or brainunit.Quantity
            Supplied column values, shape ``(shape[1],)``.
        w_dim_arr : array_like or brainunit.Quantity
            Per-edge values, shape ``(nnz,)``.

        Returns
        -------
        array or brainunit.Quantity
            Parent static edge mapping, without delay-history alignment.

        Raises
        ------
        TypeError
            Original column indices are int64.
        """
        self._require_int32_indices("dt2t_transposed")
        return super().dt2t_transposed(y_dim_arr, w_dim_arr)

    def _check_weight_type(self, data):
        if data.dtype != self.dtype:
            raise TypeError(f"replacement weights must keep dtype {self.dtype}; got {data.dtype}")
        if u.get_unit(data) != u.get_unit(self.data):
            raise TypeError("replacement weights must keep the same unit as delaycsr.data")

    def _same_structure(self, data):
        # Reject typed precision changes before JAX can canonicalize host input.
        if hasattr(data, "dtype"):
            self._check_weight_type(data)
        data = u.math.atleast_1d(_owned_data(data))
        self._check_weight_type(data)
        if data.shape not in ((1,), (self.nse,)):
            raise ValueError("replacement weights must have length 1 or nnz")
        return self._replace_data(data)

    def _check_sparse_operand(self, other):
        if not isinstance(other, u.sparse.SparseMatrix):
            return
        if not isinstance(other, delaycsr):
            raise TypeError("sparse weight operations require a delaycsr with the same delay layout")
        same_static = all(getattr(self, name) == getattr(other, name)
                          for name in ("shape", "L", "delay_steps", "weighted"))
        same_arrays = all(getattr(self, name) is getattr(other, name)
                          for name in ("indices", "indptr", "delay_count", "delay_off"))
        if not same_static or not same_arrays:
            raise ValueError("sparse operands must share the same delay layout and edge order")

    def _require_int32_indices(self, method):
        if self.indices.dtype != jnp.int32:
            raise TypeError(f"delaycsr.{method} requires int32 indices; got {self.indices.dtype}")

    def _concrete_layout(self, method, *extra):
        _require_concrete(method, self.indices, self.indptr, self.delay_count,
                          self.delay_off, self.row_task_count, *extra)
        self._check_prepared_values()

    def _empty_ring(self, slots):
        values, unit = u.split_mantissa_unit(self.ring)
        size = slots * self.shape[1]
        if size > _I64_MAX // values.dtype.itemsize:
            raise ValueError("ring buffer byte size overflows int64")
        if size > _I32_MAX and not jax.config.x64_enabled:
            raise RuntimeError("enable jax_enable_x64 for int64 ring offsets")
        ring = jnp.zeros((slots, self.shape[1]), dtype=values.dtype)
        return u.Quantity(ring, unit=unit) if isinstance(self.ring, u.Quantity) else ring

    def _replace_data(self, data):
        """Assemble a same-structure copy after caller-specific validation."""
        result = copy(self)
        result.data = data
        return result

    @classmethod
    def _from_parts(cls, data, indices, indptr, **kwargs):
        # Structural CSR transformations have no delay permutation. Return a
        # plain CSR instead of constructing a delaycsr missing its delay state.
        return CSR._from_parts(data, indices, indptr, **kwargs)

    def tree_flatten(self):
        """Flatten the CSR structure and delay state for JAX transformations.

        Returns
        -------
        tuple
            Dynamic children and static auxiliary metadata for reconstruction.
        """
        base, base_aux = super().tree_flatten()
        return (base, self.delay_count, self.delay_off, self.row_task_count,
                self.ring, self.head, self.workspace), (
                    base_aux, self.delay_steps, self.capacity, self.L, self.weighted)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct delay state without initializing or sorting its arrays.

        Parameters
        ----------
        aux_data : tuple
            Static metadata returned by :meth:`tree_flatten`.
        children : tuple
            Dynamic arrays and workspace returned by :meth:`tree_flatten`.

        Returns
        -------
        delaycsr
            Reconstructed CSR with its prepared layout and ring state.
        """
        base_aux, steps, capacity, slots, weighted = aux_data
        base, counts, offsets, tasks, ring, head, workspace = children
        result = object.__new__(cls)
        result.__dict__.update(CSR.tree_unflatten(base_aux, base).__dict__)
        result.delay_steps, result.capacity, result.L, result.weighted = steps, capacity, slots, weighted
        result.delay_count, result.delay_off, result.row_task_count = counts, offsets, tasks
        result.ring, result._head, result.workspace = ring, head, workspace
        return result


def csr_delay_deliver_layout(ring, events, layout, t, *, weighted=None, backend=None):
    """Deliver into a ring using an existing ``delaycsr`` prepared layout.

    Parameters
    ----------
    ring : jax.Array or brainunit.Quantity
        Arrival buffer of shape ``(layout.L, layout.shape[1])``. Its dtype
        determines delivery precision; units must be compatible with weight
        times events and are preserved on return.
    events : array_like, BinaryArray or WeightedArray
        Single-step boolean, integer or floating presynaptic events.
    layout : delaycsr
        Existing prepared delay CSR object.
    t : int or jax.Array
        Global integer time step.
    weighted : bool or None, optional
        Event mode, inferred from the wrapper when omitted.
    backend : str or None, optional
        ``cuda_raw`` or ``jax_raw`` override for this call. None uses the
        layout's inherited CSR backend, without changing it.

    Returns
    -------
    jax.Array or brainunit.Quantity
        Updated full ring with the input ring's dtype and unit. Delivery
        preserves all existing slots and does not mutate the layout object's
        owned state.
    """
    if not isinstance(layout, delaycsr):
        raise TypeError("layout must be an existing delaycsr")
    return layout.deliver(ring, events, t, weighted=weighted, backend=backend)
