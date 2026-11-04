"""Stateful delayed CSR: initialization, sorting, ring and single-step reads.

CUDA operators and their JAX transformation rules live in ``delay.py``.
"""

from copy import copy
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import brainunit as u

from brainevent._event import as_event_operand

from ..main import CSR
from ._util import _checked_result, _check_projection, _raise_if_bad_head
from .delay import (
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


@jax.tree_util.register_pytree_node_class
class delaycsr(CSR):
    """CSR with owned delay metadata, ring and task workspace.

    Parameters
    ----------
    data : CSR, tuple of arrays or array_like
        Source CSR, ``(weights, indices, indptr)``, or weights supplied with
        separate indices and row pointers. Weights use float16, bfloat16,
        float32 or float64, with one shared value or one value per edge.
    indices, indptr : array_like or None, optional
        Int32 or int64 column indices and row pointers. Omit these when
        supplying a CSR or a three-array tuple.
    edge_delays : array_like
        Concrete nonnegative integer delays, one per edge in the input CSR
        order. Initialization sorts each row by (delay, column).
    shape : tuple of int or None, optional
        ``(n_pre, n_post)``, required unless data is a CSR object.
    delay_steps : array_like or None, optional
        Increasing distinct delay values covering every edge delay. Defaults
        to the distinct edge delays, or ``(0,)`` for an empty graph.
    buffer_slots : int or None, optional
        Ring length, larger than every delay. Defaults to maximum delay + 1.
    sorted_by_delay : bool, optional
        Validate the existing (delay, column) row order instead of sorting.
    weighted : bool, optional
        Multiply by event values when True; use nonzero activity when False.
    ring : array_like or None, optional
        Initial buffer with shape ``(buffer_slots, n_post)`` and the weight
        dtype. Defaults to zeros.
    head : int or jax.Array, optional
        Initial scalar slot in ``[0, buffer_slots)``, default zero.
    backend : str or None, optional
        ``'cuda_raw'`` or ``'jax_raw'``. None inherits the source CSR backend,
        or uses the execution platform's default when no backend is set.

    Notes
    -----
    ``forward(x, t)`` sets ``head = t % L`` and returns the next timestep.
    ``x @ W`` uses the current head without advancing it. A read never clears
    its slot: the following stateful step clears it with :meth:`clear_head`.
    :meth:`csr_delay_deliver_layout` only delivers into an explicit ring.

    This is a mutable Python object. For JIT/scan, pass it as a pytree argument
    and return the updated object as part of the result/carry. Do not jit a
    bound method with hidden Python state. Operators support JVP, reverse-mode
    AD and vmap; binary events use the main library's straight-through rule.
    """

    data: jax.Array
    indices: jax.Array
    indptr: jax.Array
    workspace: HybridWorkspace | WeightedHybridWorkspace

    def __init__(
        self, data: Any, indices: Any = None, indptr: Any = None, *,
        edge_delays: Any, shape: tuple[int, int] | None = None,
        delay_steps: Any = None, buffer_slots: int | None = None,
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
        weights = np.atleast_1d(np.asarray(data))
        columns, pointers = np.asarray(indices), np.asarray(indptr)
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
        super().__init__(weights, columns, pointers, shape=(int(shape[0]), int(shape[1])), backend=backend,
                         indptr_dtype=pointers.dtype)
        # The native operator permits int64 column indices independently of indptr.
        self.indices = jnp.asarray(columns)
        self.data = jnp.atleast_1d(self.data)
        delays = self._validate_edge_delays(edge_delays)
        self.delay_steps = self._validate_delay_steps(
            delay_steps if delay_steps is not None else np.unique(delays) if delays.size else (0,))
        self.L = self.delay_steps[-1] + 1 if buffer_slots is None else buffer_slots
        if type(self.L) is not int or not self.delay_steps[-1] < self.L <= _I32_MAX:
            raise ValueError("buffer_slots must exceed every delay and fit int32")
        if self.L * self.shape[1] > _I64_MAX // self.data.dtype.itemsize:
            raise ValueError("ring buffer byte size overflows int64")
        if not jax.config.x64_enabled and self.L * self.shape[1] > _I32_MAX:
            raise RuntimeError("enable jax_enable_x64 for int64 ring offsets")
        self.weighted = bool(weighted)
        if sorted_by_delay:
            self._check_delay_order(delays)
            self._build_delay_metadata(delays)
        else:
            self.sort_by_delay(delays)
        self.ring = (jnp.zeros((self.L, self.shape[1]), self.dtype) if ring is None
                     else jnp.asarray(ring))
        if self.ring.shape != (self.L, self.shape[1]) or self.ring.dtype != self.dtype:
            raise ValueError("ring must have shape (L, N) and the weights dtype")
        self.head = head

    @staticmethod
    def _validate_delay_steps(values: Any) -> tuple[int, ...]:
        steps = np.asarray(values)
        if (steps.ndim != 1 or steps.size == 0 or steps.dtype.kind not in "iu"
                or np.any(steps < 0) or np.any(steps >= _I32_MAX)
                or np.any(steps[1:] <= steps[:-1])):
            raise ValueError("delay_steps must be increasing nonnegative integers below INT32_MAX")
        return tuple(int(value) for value in steps)

    def _validate_edge_delays(self, edge_delays: Any) -> np.ndarray:
        delays = np.asarray(edge_delays)
        if (delays.shape != self.indices.shape or delays.dtype.kind not in "iu"
                or np.any(delays < 0) or np.any(delays >= _I32_MAX)):
            raise ValueError("edge_delays must contain one nonnegative integer per edge")
        return delays

    def _check_delay_order(self, delays: np.ndarray) -> None:
        columns, pointers = np.asarray(self.indices), np.asarray(self.indptr)
        for row in range(self.shape[0]):
            first, last = int(pointers[row]), int(pointers[row + 1])
            d, c = delays[first:last], columns[first:last]
            if np.any((d[1:] < d[:-1]) | ((d[1:] == d[:-1]) & (c[1:] < c[:-1]))):
                raise ValueError(f"sorted_by_delay=True requires (delay, column) order in row {row}")

    def sort_by_delay(self, edge_delays: Any) -> None:
        """Sort each row by (delay, column) and overwrite this object's CSR.

        Parameters
        ----------
        edge_delays : array_like
            Concrete integer delays aligned with the current CSR edge order.
            Each delay must be present in ``self.delay_steps``.

        Notes
        -----
        Per-edge weights follow the column permutation; shared weights retain
        one value. This initialization-time method rebuilds delay metadata and
        workspace, invalidates edge-order caches, and stores no per-edge delays.
        """
        delays = self._validate_edge_delays(edge_delays).copy()
        if np.any(~np.isin(delays, self.delay_steps)):
            raise ValueError("edge delay absent from delay_steps")
        columns, weights = np.asarray(self.indices).copy(), np.asarray(self.data).copy()
        pointers = np.asarray(self.indptr)
        for row in range(self.shape[0]):
            first, last = int(pointers[row]), int(pointers[row + 1])
            order = np.lexsort((columns[first:last], delays[first:last]))
            columns[first:last] = columns[first:last][order]
            delays[first:last] = delays[first:last][order]
            if weights.size != 1:
                weights[first:last] = weights[first:last][order]
        self.indices, self.data = jnp.asarray(columns), jnp.asarray(weights)
        self._cache = type(self._cache)()
        self._build_delay_metadata(delays)

    def _build_delay_metadata(self, delays: np.ndarray) -> None:
        """Build segment counts and task storage directly on this CSR object."""
        if np.any(~np.isin(delays, self.delay_steps)):
            raise ValueError("edge delay absent from delay_steps")
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
        self._allocate_workspace()

    def _allocate_workspace(self) -> None:
        ranges = (jnp.zeros(self.capacity, self.indptr.dtype),
                  jnp.zeros(self.capacity, self.indptr.dtype),
                  jnp.zeros(self.capacity, self.delay_off.dtype))
        status = jnp.zeros(2, jnp.int32)
        self.workspace = (WeightedHybridWorkspace(*ranges, jnp.zeros(self.capacity, self.dtype), status)
                          if self.weighted else HybridWorkspace(*ranges, status))

    @property
    def weights(self) -> jax.Array:
        """jax.Array: Inherited CSR weights in the initialized delay order."""
        return self.data

    @property
    def head(self) -> jax.Array:
        """jax.Array: Current scalar int32 slot in ``[0, self.L)``."""
        return self._head

    @head.setter
    def head(self, value: Any) -> None:
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            _raise_if_bad_head(value, self.L)
        else:
            value = jnp.asarray(value)
            if value.ndim != 0 or value.dtype.kind not in "iu":
                raise TypeError("head must be an integer scalar slot")
            if isinstance(value, jax.core.Tracer):
                value = _checked_result(value, jnp.where((value < 0) | (value >= self.L), 2, 0))
            else:
                _raise_if_bad_head(value, self.L)
        self._head = jnp.asarray(value, jnp.int32)

    def _check_trace_state(self, *values: Any) -> None:
        if (not isinstance(self.ring, jax.core.Tracer)
                and any(isinstance(v, jax.core.Tracer) for v in jax.tree.leaves(values))):
            raise RuntimeError("pass delaycsr as a JIT/scan argument and return its updated state; "
                               "do not jit a bound method or close over mutable delaycsr")

    def forward(self, x: Any, t: Any) -> jax.Array:
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
        self._check_trace_state(x, t, head)
        # Modulo already guarantees the slot range. Avoid reading a device
        # scalar back to the host just to validate this derived head again.
        self._head = head
        return self.__rmatmul__(x)

    def __rmatmul__(self, x: Any) -> jax.Array:
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
        self._project(x, clear_current=True)
        return self.read_spike()

    def _project(self, x: Any, *, clear_current: bool = False) -> jax.Array:
        """Run the existing prepared projection, optionally consuming owned head."""
        event = as_event_operand(x)
        if event is not None:
            x, weighted = event
            if weighted != self.weighted:
                raise TypeError("event wrapper mode disagrees with delaycsr.weighted")
        x = jnp.asarray(x)
        if clear_current:
            self._check_trace_state(x, self.head)
        if x.shape != (self.shape[0],):
            raise ValueError(f"events must have shape ({self.shape[0]},)")
        if self.ring.shape != (self.L, self.shape[1]):
            raise ValueError("ring must have shape (L, N)")
        head = self.head.reshape(1)
        if self.backend not in (None, "cuda_raw", "jax_raw"):
            raise ValueError("CSR backend must be 'cuda_raw', 'jax_raw' or None")
        workspace_type = WeightedHybridWorkspace if self.weighted else HybridWorkspace
        if not isinstance(self.workspace, workspace_type):
            raise RuntimeError(f"hybrid workspace must be {workspace_type.__name__}")
        workspace = self.workspace
        operands = (self.ring, x, self.data, self.indices, self.indptr, self.delay_count,
                    self.delay_off, head, self.row_task_count, *workspace)
        _check_projection(self.ring, x, self.data, self.indices, self.indptr,
                          head, workspace, weighted=self.weighted)
        outputs = tuple(jax.ShapeDtypeStruct(a.shape, a.dtype) for a in (self.ring, *workspace))
        if clear_current:
            self.clear_head()
            operands = (self.ring, *operands[1:])
        result = csr_delay_deliver_layout_p(
            *operands, outs=outputs, backend=self.backend, weighted=self.weighted, check_status=True)
        if clear_current:
            self._check_trace_state(result[0])
        self.ring = result[0]
        self.workspace = type(workspace)(*result[1:])
        return self.ring

    def clear_head(self) -> jax.Array:
        """Clear the consumed current slot using JAX, without advancing head.

        Returns
        -------
        jax.Array
            The updated complete ring, also stored on this object. Old values
            in the current slot have zero derivative; other slots are preserved.
        """
        self._check_trace_state(self.head)
        cleared = self.ring.at[self.head].set(0)
        self._check_trace_state(cleared)
        self.ring = cleared
        return self.ring

    def csr_delay_deliver_layout(self, ring: Any, events: Any, t: Any, *,
                                 weighted: bool | None = None, backend: str | None = None) -> jax.Array:
        """Deliver into an external ring using this object's prepared layout.

        Parameters
        ----------
        ring : jax.Array
            Arrival buffer of shape ``(L, shape[1])``, in the weight dtype.
        events : array_like, BinaryArray or WeightedArray
            Single-step events, subject to the existing backend dtype rules.
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
        jax.Array
            Full ring with delayed contributions added. No slot is cleared and
            this object's ring, head and workspace are unchanged.
        """
        if weighted is not None and not isinstance(weighted, (bool, np.bool_)):
            raise TypeError("weighted must be bool or None")
        event = as_event_operand(events)
        if event is not None:
            events, mode = event
            if weighted is not None and bool(weighted) != mode:
                raise TypeError("event wrapper mode disagrees with weighted")
            weighted = mode
        current = copy(self)
        current.ring = jnp.asarray(ring)
        current._head = _time_slot(t, self.L)
        current.weighted = bool(weighted)
        current.backend = self.backend if backend is None else backend
        if current.backend not in (None, "cuda_raw", "jax_raw"):
            raise ValueError("backend must be 'cuda_raw', 'jax_raw' or None")
        current._allocate_workspace()
        return current._project(events)

    def read_spike(self, target_t: Any = None) -> jax.Array:
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
        return self._same_structure(data)

    def _same_structure(self, data):
        # with_data and data-only arithmetic receive weights in current order.
        data = jnp.atleast_1d(jnp.asarray(data))
        if data.dtype != self.dtype or data.shape not in ((1,), (self.nse,)):
            raise ValueError("replacement weights must keep the ring dtype and have length 1 or nnz")
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
    ring : jax.Array
        Arrival buffer of shape ``(layout.L, layout.shape[1])``.
    events : array_like, BinaryArray or WeightedArray
        Single-step presynaptic events.
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
    jax.Array
        Updated full ring. Delivery preserves all existing slots and does not
        mutate the layout object's owned state.
    """
    if not isinstance(layout, delaycsr):
        raise TypeError("layout must be an existing delaycsr")
    return layout.csr_delay_deliver_layout(ring, events, t, weighted=weighted, backend=backend)
