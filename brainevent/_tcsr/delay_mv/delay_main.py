"""Prepare segmented CSR metadata and own Python-loop delay state."""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import brainunit as u

from brainevent._csr.main import CSR as PlainCSR
from brainevent._event import BinaryArray
from brainevent._typing import Data
from brainevent.config import get_backend

from . import binary_delay
from .binary_delay import (
    _CusparseWorkspace, _DelayCSRInputs, _DelayWorkspace, _Inputs, _Workspace,
)

__all__ = ["DelayMetadata", "prepare_delay_metadata", "delay_tcsr"]

_I32_MAX = int(np.iinfo(np.int32).max)
_I64_MAX = int(np.iinfo(np.int64).max)
_INDEX_DTYPES = (np.dtype("int32"), np.dtype("int64"))
_TPR_THRESHOLD = 128
_TASK_NNZ = 4096


class DelayMetadata(NamedTuple):
    """Hold precomputed segmentation and task allocation metadata.

    Parameters
    ----------
    delay_steps : tuple of int
        Strictly increasing nonnegative kernel delays, including unused bins.
    delay_count : array_like
        Nonnegative counts (M, K), with the CSR row-pointer dtype.
    delay_off : array_like
        Int32/int64 element offsets delay_steps * N, sized for the full ring.
    row_task_count : array_like
        Int32 task counts (M,), using threshold 128 and task size 4096.
    task_capacity : int
        Sum of all row task counts, fitting nonnegative int32.
    buffer_slots : int
        Ring length, strictly greater than every kernel delay.

    Notes
    -----
    The caller guarantees CSR edges already follow these delay segments.
    Initialization checks metadata consistency without reordering any edges.
    """

    delay_steps: tuple[int, ...]
    delay_count: Any
    delay_off: Any
    row_task_count: Any
    task_capacity: int
    buffer_slots: int


def prepare_delay_metadata(
    edge_delays: Any,
    indptr: Any,
    *,
    n_post: int,
    buffer_slots: int | None = None,
) -> tuple[tuple[int, ...], jax.Array, jax.Array]:
    """Convert already segmented edge delays into shared kernel metadata.

    Parameters
    ----------
    edge_delays : array_like
        Nonnegative integer steps, one per CSR edge, nondecreasing within
        each row. The input order is preserved.
    indptr : array_like
        Int32/int64 CSR row pointers starting at zero and ending at nnz.
    n_post : int
        Number of target neurons, including unconnected targets.
    buffer_slots : int or None, optional
        Ring length, greater than every delay; defaults to max_delay + 1.

    Returns
    -------
    delay_steps : tuple of int
        Sorted distinct delays, or (0,) for an empty graph.
    delay_count : jax.Array
        Counts of shape (M, K), with the indptr dtype.
    delay_off : jax.Array
        Element offsets delay_steps * n_post; int32 when the full ring fits,
        otherwise int64.

    Raises
    ------
    TypeError
        If delay or pointer dtypes are unsupported.
    ValueError
        If dimensions, pointers, delay ordering or the ring length are invalid.
    RuntimeError
        If int64 metadata is required but JAX x64 is disabled.

    Notes
    -----
    Conversion runs on the CPU during initialization. No CSR entries are
    reordered, and every row uses the same delay_steps ordering.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from brainevent._tcsr.delay_mv.delay_main import prepare_delay_metadata
        >>> steps, counts, offsets = prepare_delay_metadata(
        ...     [0, 2, 2, 0, 5], np.array([0, 3, 5], dtype=np.int32), n_post=10)
        >>> steps
        (0, 2, 5)
        >>> counts.tolist()
        [[1, 2, 0], [1, 0, 1]]
        >>> offsets.tolist()
        [0, 20, 50]
    """
    delays, ptr = np.asarray(edge_delays), np.asarray(indptr)
    if ptr.dtype not in _INDEX_DTYPES or delays.dtype.kind not in "iu":
        raise TypeError("require integer edge_delays and int32/int64 indptr")
    if ptr.ndim != 1 or ptr.size == 0 or delays.ndim != 1:
        raise ValueError("edge_delays and indptr must be vectors; indptr must be nonempty")
    if (ptr[0] != 0 or int(ptr[-1]) != delays.size or np.any(ptr < 0)
            or np.any(ptr[1:] < ptr[:-1])):
        raise ValueError("indptr must be nondecreasing and span all edge_delays")
    if (not isinstance(n_post, (int, np.integer)) or isinstance(n_post, (bool, np.bool_))
            or not 0 <= n_post <= _I32_MAX or ptr.size - 1 > _I32_MAX):
        raise ValueError("source and target counts must fit nonnegative int32")
    if np.any(delays < 0) or np.any(delays >= _I32_MAX):
        raise ValueError("edge_delays must be nonnegative and below INT32_MAX")

    steps = tuple(int(d) for d in np.unique(delays)) if delays.size else (0,)
    slots = steps[-1] + 1 if buffer_slots is None else buffer_slots
    if (not isinstance(slots, (int, np.integer)) or isinstance(slots, (bool, np.bool_))
            or not steps[-1] < slots <= _I32_MAX):
        raise ValueError("buffer_slots must exceed every delay and fit positive int32")
    ring_size = int(slots) * int(n_post)
    if ring_size > _I64_MAX // np.dtype(np.float32).itemsize:
        raise ValueError("ring buffer byte size overflows int64")
    offset_dtype = np.int32 if ring_size <= _I32_MAX else np.int64
    if not jax.config.jax_enable_x64 and (ptr.dtype == np.int64 or offset_dtype == np.int64):
        raise RuntimeError("enable jax_enable_x64 before constructing int64 delay metadata")

    counts = np.zeros((ptr.size - 1, len(steps)), dtype=ptr.dtype)
    for row in range(ptr.size - 1):
        row_delays = delays[int(ptr[row]):int(ptr[row + 1])]
        if np.any(row_delays[1:] < row_delays[:-1]):
            raise ValueError("edge_delays must be nondecreasing within each CSR row")
        segments = np.searchsorted(steps, row_delays)
        counts[row] = np.bincount(segments, minlength=len(steps))
    offsets = np.asarray([d * int(n_post) for d in steps], dtype=offset_dtype)
    return steps, jnp.asarray(counts), jnp.asarray(offsets)


class delay_tcsr:
    """Own a segmented CSR projection and its persistent delay state.

    Parameters
    ----------
    source : brainevent.CSR
        CSR already ordered by delay within each row, with float32 weights.
        A scalar or length-one weight is shared; weight units are retained.
    edge_delays : array_like or None, optional
        Nonnegative integer steps aligned with the source edges. Mutually
        exclusive with delay_metadata; exactly one input must be supplied.
    delay_metadata : DelayMetadata or None, optional
        Precomputed segmentation and task counts. No edge-delay expansion or
        sorting is performed. Explicit buffer_slots must match the metadata.
    buffer_slots : int or None, optional
        Ring length, defaulting to max_delay + 1.
    binary_backend : str or None, optional
        jax, cusparse or cuda_raw; None resolves the platform default at init.

    Attributes
    ----------
    buffer : jax.Array
        Float32 ring of shape (L, N), initially zero.
    head : jax.Array
        Device int32 array of shape (1,), initially zero.
    delay_steps, delay_count, delay_off
        Prepared metadata shared by every forward step.

    Notes
    -----
    Each x @ W consumes one time step, including when x is all zero. Call
    from a Python loop; implicit member updates under outer jit/scan are not
    supported. Backend, structure and weights remain fixed for this instance;
    reconstruct it when changing them. cuSPARSE splits the source once at init.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from brainevent._csr.main import CSR
        >>> from brainevent._tcsr.delay_mv.delay_main import delay_tcsr
        >>> source = CSR((np.array([2.0], np.float32), np.array([0], np.int32),
        ...               np.array([0, 1], np.int32)), shape=(1, 1))
        >>> projection = delay_tcsr(source, np.array([1]), binary_backend="jax")
        >>> (np.array([1.0], np.float32) @ projection).tolist()
        [0.0]
        >>> (np.array([0.0], np.float32) @ projection).tolist()
        [2.0]
    """

    __array_priority__ = 1000

    def __init__(
        self,
        source: PlainCSR,
        edge_delays: Any = None,
        *,
        delay_metadata: DelayMetadata | None = None,
        buffer_slots: int | None = None,
        binary_backend: str | None = None,
    ) -> None:
        """Validate the source and allocate metadata, ring and task workspace.

        Parameters
        ----------
        source : brainevent.CSR
            Presegmented CSR with float32 shared or per-edge weights.
        edge_delays : array_like or None, optional
            Integer delays in the same order as source edges.
        delay_metadata : DelayMetadata or None, optional
            Prepared segmentation and tasks, mutually exclusive with edge_delays.
        buffer_slots : int or None, optional
            Ring length, greater than the maximum delay.
        binary_backend : str or None, optional
            Forward backend selection.

        Raises
        ------
        TypeError
            If the source, index dtype or weight dtype is unsupported.
        ValueError
            If CSR dimensions, weight count, delays or backend are invalid.
        RuntimeError
            If int64 arrays require enabling JAX x64.
        """
        def _resolve_backend(backend: str | None, platform: str) -> str:
            """Resolve setup-time selection using the primitive's registered defaults."""
            primitive = binary_delay.binary_csrmv_delay_p
            available = primitive.available_backends(platform)
            if backend is not None:
                if backend not in available:
                    raise ValueError(f"delay backend {backend!r} is unavailable on {platform}")
                return backend
            selected = get_backend(platform)
            if selected in available:
                return str(selected)
            if selected is not None:
                primitive._warn_backend_once(
                    f"Global backend {selected!r} is unavailable for delay on {platform}; "
                    "using the primitive default."
                )
            return str(primitive.get_default(platform))

        if not isinstance(source, PlainCSR):
            raise TypeError("delay_tcsr requires a PlainCSR source")
        if (edge_delays is None) == (delay_metadata is None):
            raise ValueError("supply exactly one of edge_delays and delay_metadata")
        binary_delay._validate_backend(binary_backend)
        if len(source.shape) != 2 or any(
            not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_))
            or not 0 <= n <= _I32_MAX for n in source.shape
        ):
            raise ValueError("source.shape must contain two nonnegative int32 dimensions")
        rows, cols = map(int, source.shape)
        indices = source.indices
        if indices.dtype not in _INDEX_DTYPES:
            raise TypeError("indices must be int32/int64")
        if indices.dtype == np.int64 and not jax.config.jax_enable_x64:
            raise RuntimeError("enable jax_enable_x64 before using int64 indices")
        if indices.ndim != 1:
            raise ValueError("indices must be a vector of valid target columns")
        # Reduce on-device: a full host readback would copy gigabytes for large CSR.
        invalid_indices = jax.jit(lambda a: jnp.any((a < 0) | (a >= cols)))(indices)
        if bool(invalid_indices):
            raise ValueError("indices must be a vector of valid target columns")
        if source.indptr.shape != (rows + 1,) or int(source.indptr[-1]) != indices.size:
            raise ValueError("indptr shape or final offset does not match the CSR")
        if source.data.dtype != np.float32:
            raise TypeError("weights must use float32")
        if source.data.ndim > 1 or source.data.size not in (1, indices.size):
            raise ValueError("weights must be a scalar, length one or a vector of length nnz")

        self.data = source.data.reshape(1) if source.data.ndim == 0 else source.data
        self.indices, self.indptr = source.indices, source.indptr
        self.shape = (rows, cols)
        device = next(iter(source.indptr.devices())) if isinstance(source.indptr, jax.Array) else jax.devices()[0]
        self.binary_backend = _resolve_backend(binary_backend, device.platform)
        self._inputs: _Inputs
        self._workspace: _Workspace
        with jax.default_device(device):
            if delay_metadata is None:
                self.delay_steps, self.delay_count, self.delay_off = prepare_delay_metadata(
                    edge_delays, source.indptr, n_post=cols, buffer_slots=buffer_slots,
                )
                self.buffer_slots = self.delay_steps[-1] + 1 if buffer_slots is None else int(buffer_slots)
                counts = np.asarray(self.delay_count, dtype=np.int64)
                tasks = (counts // _TASK_NNZ + (counts % _TASK_NNZ != 0)).sum(axis=1)
                tasks[np.diff(np.asarray(self.indptr)) <= _TPR_THRESHOLD] = 0
                capacity = int(tasks.sum())
            else:
                if not isinstance(delay_metadata, DelayMetadata):
                    raise TypeError("delay_metadata must be a DelayMetadata instance")
                meta = delay_metadata
                steps, slots = tuple(meta.delay_steps), meta.buffer_slots
                if (not steps or any(isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer))
                                     or not 0 <= d < _I32_MAX for d in steps)
                        or any(a >= b for a, b in zip(steps, steps[1:]))):
                    raise ValueError("delay_steps must be strictly increasing nonnegative integers")
                if (isinstance(slots, (bool, np.bool_)) or not isinstance(slots, (int, np.integer))
                        or not steps[-1] < slots <= _I32_MAX):
                    raise ValueError("buffer_slots must exceed every delay and fit positive int32")
                if buffer_slots is not None and (isinstance(buffer_slots, (bool, np.bool_))
                        or not isinstance(buffer_slots, (int, np.integer)) or buffer_slots != slots):
                    raise ValueError("buffer_slots must match delay_metadata")
                ring_size = int(slots) * cols
                if ring_size > _I64_MAX // np.dtype(np.float32).itemsize:
                    raise ValueError("ring buffer byte size overflows int64")
                ptr = np.asarray(self.indptr)
                counts, offsets, tasks = map(np.asarray, (meta.delay_count, meta.delay_off, meta.row_task_count))
                offset_dtype = np.dtype("int32" if ring_size <= _I32_MAX else "int64")
                if ptr.dtype not in _INDEX_DTYPES or ptr[0] != 0 or np.any(ptr[1:] < ptr[:-1]):
                    raise ValueError("indptr must be nondecreasing and start at zero")
                if (counts.dtype != ptr.dtype or offsets.dtype != offset_dtype or tasks.dtype != np.int32):
                    raise TypeError("delay metadata has incompatible integer dtypes")
                if not jax.config.jax_enable_x64 and (ptr.dtype == np.int64 or offset_dtype == np.int64):
                    raise RuntimeError("enable jax_enable_x64 before using int64 delay metadata")
                if (counts.shape != (rows, len(steps)) or np.any(counts < 0)
                        or np.any(counts > indices.size) or not np.array_equal(counts.sum(axis=1), np.diff(ptr))):
                    raise ValueError("delay_count must partition every CSR row")
                if not np.array_equal(offsets, np.asarray([int(d) * cols for d in steps], dtype=offset_dtype)):
                    raise ValueError("delay_off must equal delay_steps * n_post")
                expected_tasks = (counts // _TASK_NNZ + (counts % _TASK_NNZ != 0)).sum(axis=1)
                expected_tasks[np.diff(ptr) <= _TPR_THRESHOLD] = 0
                if tasks.shape != (rows,) or not np.array_equal(tasks, expected_tasks):
                    raise ValueError("row_task_count does not match delay segments")
                capacity = meta.task_capacity
                if (isinstance(capacity, (bool, np.bool_)) or not isinstance(capacity, (int, np.integer))
                        or capacity != int(tasks.sum(dtype=np.int64))):
                    raise ValueError("task_capacity must equal the sum of row_task_count")
                self.delay_steps = tuple(int(d) for d in steps)
                self.delay_count, self.delay_off = jnp.asarray(counts), jnp.asarray(offsets)
                self.buffer_slots = int(slots)
            if not 0 <= capacity <= _I32_MAX:
                raise ValueError("delay task capacity exceeds int32")
            self.row_task_count = jnp.asarray(tasks, dtype=jnp.int32)
            # Preserve public source fields; prepare device operands only once.
            indices_device, indptr_device = jnp.asarray(self.indices), jnp.asarray(self.indptr)
            data_device = u.maybe_decimal(jnp.asarray(u.get_mantissa(self.data)) * u.get_unit(self.data))
            if self.binary_backend == "cusparse":
                if np.any(counts.sum(axis=0) > rows * cols):
                    raise ValueError("cuSPARSE delay segment nnz exceeds M*N")
                self._inputs = binary_delay.split_region_delay(
                    data_device, indices_device, indptr_device, self.delay_count,
                )
                self._workspace = _CusparseWorkspace(jnp.zeros(cols, dtype=jnp.float32))
            else:
                self._inputs = _DelayCSRInputs(
                    data_device, indices_device, indptr_device, self.delay_count, self.row_task_count,
                )
                self._workspace = _DelayWorkspace(
                    capacity,
                    jnp.zeros(capacity, dtype=self.indptr.dtype),
                    jnp.zeros(capacity, dtype=self.indptr.dtype),
                    jnp.zeros(capacity, dtype=self.delay_off.dtype),
                    jnp.zeros(2, dtype=jnp.int32),
                )
            self.buffer = jnp.zeros((self.buffer_slots, cols), dtype=jnp.float32)
            self.head = jnp.zeros(1, dtype=jnp.int32)

    def __rmatmul__(self, other: Any) -> Data:
        """Delegate one positive-event step and retain its returned state.

        Parameters
        ----------
        other : numpy.ndarray, jax.Array, or brainevent.BinaryArray
            Bool/float32 vector of length M; only positive entries activate.

        Returns
        -------
        y : jax.Array or brainunit.Quantity
            Current arrivals of shape (N,), with weight units when applicable.

        Raises
        ------
        TypeError
            If events are not bool or float32.
        ValueError
            If events are not a vector of length M.
        NotImplementedError
            If member updates are traced by an outer JAX transform.
        """
        events = other.value if isinstance(other, BinaryArray) else other
        if isinstance(events, jax.core.Tracer):
            raise NotImplementedError("delay_tcsr member updates require a Python loop")
        if getattr(events, "dtype", None) not in (np.dtype("bool"), np.dtype("float32")):
            raise TypeError("events must be a bool or float32 array")
        if events.shape != (self.shape[0],):
            raise ValueError("events must be a vector of length M")
        y, buffer, head, workspace = binary_delay.binary_csrmv_delay_step(
            self._inputs, events,
            shape=self.shape, delay_off=self.delay_off, buffer=self.buffer, head=self.head,
            workspace=self._workspace, backend=self.binary_backend,
        )
        if isinstance(buffer, jax.core.Tracer) or isinstance(head, jax.core.Tracer):
            raise NotImplementedError("delay_tcsr member updates require a Python loop")
        self.buffer, self.head, self._workspace = buffer, head, workspace
        return y
