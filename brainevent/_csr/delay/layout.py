# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Per-edge-delay delivery on a precomputed delay-major CSR layout (#203, #205)."""

import operator
from typing import Any, Iterator, Optional, Tuple

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

try:  # private, so guarded: without it ``_to_device`` is plain ``jnp.asarray``
    from jax._src.core import trace_state_clean as _trace_state_clean
except ImportError:  # pragma: no cover
    _trace_state_clean = None  # type: ignore[assignment]

from brainevent._compatible_import import Tracer
from brainevent._misc import namescope
from brainevent._typing import Data, MatrixShape
from .delay import csr_delay_deliver_layout_p_call

__all__ = [
    'CSRDelayLayout',
    'csr_delay_deliver_layout',
]

# CSRDelayLayout.build sorts one chunk of whole rows at a time (#229), so its host
# temporaries scale with the chunk, not with ``nse``.
#: Edges per chunk; a longer row is a chunk of its own.
_BUILD_CHUNK_EDGES = 1 << 20
#: Chunks whose keys ``(row - r0) * n_slots + delay`` span at most this many values sort
#: ``uint16`` keys, for which NumPy's stable sort is a radix sort.
_UINT16_KEYS = 1 << 16
#: Wider chunks sort ``int32`` keys up to this many values, ``int64`` keys beyond.
_INT32_KEYS = 1 << 31
#: ``perm`` is ``int32`` for fewer stored edges than this, ``int64`` otherwise.
_INT32_PERM = 1 << 31
#: Ring length bound: delays and the time slot are ``int32`` in every kernel.
_MAX_SLOTS = (1 << 31) - 1


def _row_chunks(indptr: np.ndarray, n_slots: int) -> Iterator[Tuple[int, int]]:
    """Yield ``(r0, r1)``: chunks of whole rows for :meth:`CSRDelayLayout.build`.

    A chunk holds at most ``_BUILD_CHUNK_EDGES`` edges, unless it is a single row, and
    at most ``_UINT16_KEYS // n_slots`` rows (at least one).
    """
    n_pre = indptr.shape[0] - 1
    max_rows = max(1, _UINT16_KEYS // n_slots)
    r0 = 0
    while r0 < n_pre:
        # the last row boundary within _BUILD_CHUNK_EDGES edges of the chunk's start
        r1 = int(np.searchsorted(indptr, indptr[r0] + _BUILD_CHUNK_EDGES, side='right')) - 1
        r1 = min(max(r1, r0 + 1), r0 + max_rows, n_pre)
        yield r0, r1
        r0 = r1


def _sort_keys(values: np.ndarray, n_values: int) -> np.ndarray:
    """``values`` (in ``[0, n_values)``) in the narrowest key dtype; ``uint16`` sorts by radix."""
    dtype = np.uint16 if n_values <= _UINT16_KEYS else np.int32 if n_values <= _INT32_KEYS else np.int64
    return values.astype(dtype, copy=False)


def _rows_sorted(columns: np.ndarray, row_starts: np.ndarray) -> bool:
    """Whether ``columns`` are non-decreasing within every row (row ``k`` starts at ``row_starts[k]``)."""
    descending = np.diff(columns) < 0
    starts = row_starts[(row_starts > 0) & (row_starts < columns.shape[0])]
    descending[starts - 1] = False  # a new row may restart at any column
    return not descending.any()


@jax.jit
def _take(data, perm):
    """``data[perm]`` as one compiled gather; eager indexing first materialises the
    wrapped (non-negative) copy of ``perm``."""
    return data[perm]


def _to_device(x: Any) -> jax.Array:
    """``jnp.asarray(x)``; outside a trace a NumPy array is copied, then ``device_put``.

    On JAX 0.12, ``jnp.asarray(ndarray)`` without a dtype compiles one identity XLA
    program per new shape (``lax.stage``, #251). ``device_put`` compiles nothing and
    gives the same dtype, weak type and bits. The host copy comes first because on
    jax/jaxlib 0.12.0.1 CPU ``device_put`` zero-copies a 64-byte-aligned NumPy buffer
    even with ``may_alias=False`` (#254): without it, a caller who reuses its buffer
    after ``build`` would change the layout. Inside a trace it stays ``jnp.asarray`` (a
    constant), since a traced ``device_put`` would add a copy to the program.
    """
    if isinstance(x, np.ndarray) and _trace_state_clean is not None and _trace_state_clean():
        return jax.device_put(np.array(x, copy=True))
    return jnp.asarray(x)


def _host_time(t: Any, n_slots: int) -> Any:
    """Reduce a host integer time modulo ``n_slots`` before it can overflow ``int32``."""
    if isinstance(t, (bool, np.bool_)):
        raise TypeError(f't must be an integer scalar; got {type(t).__name__}.')
    if isinstance(t, (int, np.integer)):
        return np.int32(int(t) % n_slots)
    if isinstance(t, np.ndarray) and t.ndim == 0 and np.issubdtype(t.dtype, np.integer):
        return np.int32(int(t) % n_slots)
    return t


@jax.tree_util.register_pytree_node_class
class CSRDelayLayout:
    """CSR connectivity with per-edge delays, reordered for fast delivery.

    The edges of every row are sorted by delay, then by column (equal pairs
    keep their original order), and ``delay_ptr[i, d]`` records where the
    edges of delay ``d`` start in row ``i``. :func:`csr_delay_deliver_layout`
    uses it to deliver all active rows delay by delay, so that concurrent
    writes share one ring row.

    Build it once with :meth:`build` (or :meth:`CSR.delay_layout`). The
    weights live in layout order: train or update ``layout.data`` directly,
    use :meth:`with_data` to load weights given in the original edge order
    and :meth:`original_data` to read them back in that order. A layout is
    immutable and owns no ring: the caller keeps the ring and the time step.

    Parameters
    ----------
    data : jax.Array or brainunit.Quantity
        Weights in layout order, shape ``(1,)`` (homogeneous) or ``(nse,)``.
    indices : jax.Array
        Column indices in layout order, shape ``(nse,)``, ``int32``.
    indptr : jax.Array
        Row pointers, shape ``(n_pre + 1,)``, ``int32`` or ``int64``.
    delays : jax.Array
        Delays in layout order, shape ``(nse,)``, ``int32``, in ``[0, L)``.
    delay_ptr : jax.Array
        Shape ``(n_pre, L)``, same dtype as ``indptr``: first position of
        delay ``d`` in row ``i``.
    perm : jax.Array or None, optional
        Shape ``(nse,)``: ``perm[k]`` is the original index of the edge at
        layout position ``k``. ``None`` (default) means the identity.
    shape : tuple of int
        ``(n_pre, n_post)``.

    Attributes
    ----------
    n_slots : int
        Ring length ``L`` the layout was built for.
    nse : int
        Number of stored edges.

    See Also
    --------
    csr_delay_deliver_layout : Delivery on a layout.
    CSR.delay_layout : Build a layout from a :class:`CSR`.

    Notes
    -----
    Every array is a pytree leaf, so a layout can be passed through
    ``jax.jit``; ``shape`` is static. ``delay_ptr`` costs
    ``n_pre * L * indptr.itemsize`` bytes.

    The constructor stores its arguments as they are, so it also works inside
    a trace; it does not validate them. Arrays that :meth:`build` did not make
    must satisfy its postconditions: rows sorted by delay, delays in
    ``[0, L)``, ``delay_ptr`` consistent with ``indptr`` and ``delays``, and
    column indices in ``[0, n_post)``. The ``cuda_raw`` backend reads
    ``delay_ptr`` and ``indices`` without bounds checks.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent import CSRDelayLayout
        >>> # row 0 -> {0 (delay 2), 2 (delay 1)}, row 1 -> {1 (delay 1)}
        >>> layout = CSRDelayLayout.build(
        ...     jnp.array([1., 2., 3.]), jnp.array([0, 2, 1], dtype=jnp.int32),
        ...     jnp.array([0, 2, 3], dtype=jnp.int32), jnp.array([2, 1, 1], dtype=jnp.int32),
        ...     n_slots=3, shape=(2, 3))
        >>> layout.indices
        Array([2, 0, 1], dtype=int32)
        >>> layout.original_data()
        Array([1., 2., 3.], dtype=float32)
    """

    def __init__(self, data, indices, indptr, delays, delay_ptr, perm=None, *, shape: MatrixShape):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.delays = delays
        self.delay_ptr = delay_ptr
        self.perm = perm
        self.shape: Tuple[int, int] = (int(shape[0]), int(shape[1]))

    @property
    def n_slots(self) -> int:
        """Ring length ``L`` the layout was built for."""
        return int(np.shape(self.delay_ptr)[1])

    @property
    def nse(self) -> int:
        """Number of stored edges."""
        return int(np.shape(self.indices)[0])

    def __repr__(self) -> str:
        return f'{type(self).__name__}(shape={self.shape}, nse={self.nse}, n_slots={self.n_slots})'

    def tree_flatten(self):
        return (self.data, self.indices, self.indptr, self.delays, self.delay_ptr, self.perm), self.shape

    @classmethod
    def tree_unflatten(cls, shape, children):
        return cls(*children, shape=shape)

    @classmethod
    def build(
        cls,
        data: Data,
        indices: Any,
        indptr: Any,
        delays: Any,
        *,
        n_slots: int,
        shape: MatrixShape,
    ) -> 'CSRDelayLayout':
        """Build a layout from a CSR matrix and its per-edge delays.

        Validates the structure once and sorts it on the host with NumPy, in
        chunks of whole rows (vectorized within a chunk). Delivery does no
        further validation.

        Parameters
        ----------
        data : jax.Array or brainunit.Quantity
            Weights in the original edge order: scalar, shape ``(1,)`` or
            ``(nse,)``. May be traced.
        indices : array_like
            CSR column indices, shape ``(nse,)``, each in ``[0, n_post)``.
            Must be concrete.
        indptr : array_like
            CSR row pointers, shape ``(n_pre + 1,)``. Must be concrete.
        delays : array_like or int
            Integer delays in steps, shape ``(nse,)`` or a scalar, each in
            ``[0, n_slots)``. Delay ``0`` arrives in the current slot
            ``t % n_slots``. Must be concrete.
        n_slots : int
            Ring length ``L >= 1`` (any integer-like value).
        shape : tuple of int
            ``(n_pre, n_post)``.

        Returns
        -------
        CSRDelayLayout
            The reordered connectivity.

        Raises
        ------
        TypeError
            If ``indices``, ``indptr`` or ``delays`` are traced, ``delays`` is
            not an integer array, or ``n_slots`` is not an integer.
        ValueError
            If a delay is outside ``[0, n_slots)``, a column is outside
            ``[0, n_post)``, ``n_slots < 1``, ``indptr`` is malformed, or the
            operand shapes are inconsistent.

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> from brainevent import CSRDelayLayout
            >>> layout = CSRDelayLayout.build(
            ...     jnp.array([0.5]), jnp.array([0, 1], dtype=jnp.int32),
            ...     jnp.array([0, 1, 2], dtype=jnp.int32), 1, n_slots=2, shape=(2, 2))
            >>> layout.delay_ptr
            Array([[0, 0],
                   [1, 1]], dtype=int32)
        """
        for name, x in (('indices', indices), ('indptr', indptr), ('delays', delays)):
            if isinstance(x, Tracer):
                raise TypeError(f'CSRDelayLayout.build needs a concrete `{name}`; build the layout outside jit.')
        try:
            n_slots = operator.index(n_slots)
        except TypeError:
            raise TypeError(f'n_slots must be an integer; got {n_slots!r}.') from None
        if not 1 <= n_slots <= _MAX_SLOTS:
            raise ValueError(f'n_slots must lie in [1, 2**31 - 1]; got {n_slots}.')
        if len(shape) != 2:
            raise ValueError(f'shape must be (n_pre, n_post); got {tuple(shape)}.')
        n_pre, n_post = operator.index(shape[0]), operator.index(shape[1])
        if n_pre < 0 or n_post < 0:
            raise ValueError(f'shape must be non-negative; got {(n_pre, n_post)}.')

        indices_np = np.asarray(indices)
        indptr_np = np.asarray(indptr)
        if indices_np.ndim != 1 or indptr_np.ndim != 1 or indptr_np.shape[0] != n_pre + 1:
            raise ValueError(
                f'indices must be 1-D and indptr must have shape ({n_pre + 1},); '
                f'got {indices_np.shape} and {indptr_np.shape}.'
            )
        for name, arr in (('indices', indices_np), ('indptr', indptr_np)):
            if not np.issubdtype(arr.dtype, np.integer):
                raise TypeError(f'{name} must be an integer array; got dtype {arr.dtype}.')
        nse = indices_np.shape[0]
        offsets = indptr_np.astype(np.int64)
        if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
            raise ValueError('indptr must start at 0 and be non-decreasing.')
        if int(offsets[-1]) != nse:
            raise ValueError(f'indptr[-1] = {int(offsets[-1])} does not match nse = {nse}.')
        if nse and (int(indices_np.min()) < 0 or int(indices_np.max()) >= min(n_post, _MAX_SLOTS + 1)):
            raise ValueError(
                f'column indices must lie in [0, {n_post}); '
                f'got values in [{int(indices_np.min())}, {int(indices_np.max())}].'
            )

        delays_np = np.asarray(delays)
        if not np.issubdtype(delays_np.dtype, np.integer):
            raise TypeError(f'delays must be an integer array; got dtype {delays_np.dtype}.')
        if delays_np.size == 1 and delays_np.shape != (nse,):
            delays_np = np.broadcast_to(delays_np.reshape(()), (nse,))
        if delays_np.shape != (nse,):
            raise ValueError(f'delays must have shape ({nse},) or be a scalar; got {delays_np.shape}.')
        if nse:
            lo, hi = int(delays_np.min()), int(delays_np.max())
            if lo < 0 or hi >= n_slots:
                raise ValueError(
                    f'delays must satisfy 0 <= d < L = {n_slots} (the ring length); '
                    f'got values in [{lo}, {hi}].'
                )

        wide = nse >= _INT32_PERM
        if wide and not jax.config.jax_enable_x64:
            raise ValueError(f'a layout of nse = {nse} >= 2**31 edges needs int64 offsets; enable jax_enable_x64.')
        ptr_dtype = np.int64 if (wide or indptr_np.dtype == np.int64) else np.int32
        perm_dtype = np.int64 if wide else np.int32

        data_arr = data if isinstance(data, u.Quantity) else _to_device(data)
        if np.ndim(data_arr) == 0:
            data_arr = data_arr.reshape(1)
        if np.ndim(data_arr) != 1 or np.shape(data_arr)[0] not in (1, nse):
            raise ValueError(f'data must be a scalar, shape (1,) or ({nse},); got {np.shape(data_arr)}.')

        # Sort the edges of every row by (delay, column); equal pairs keep their original
        # order. Rows are contiguous, so a chunk of whole rows [r0, r1) keeps its positions
        # [indptr[r0], indptr[r1]) and is sorted on its own: a stable sort by the keys
        # (row - r0) * n_slots + delay, after a stable sort by column when a row's columns
        # are not already ascending (an LSD radix pass). The outputs are written in their
        # final dtypes; no array of nse int64 values is made.
        perm = np.empty(nse, perm_dtype)
        indices_out = np.empty(nse, np.int32)
        delays_out = np.empty(nse, np.int32)
        delay_ptr = np.empty(n_pre * n_slots, ptr_dtype)
        for r0, r1 in _row_chunks(offsets, n_slots):
            start, stop = int(offsets[r0]), int(offsets[r1])
            n_keys = (r1 - r0) * n_slots
            row_keys = _sort_keys(np.arange(r1 - r0, dtype=np.int64) * n_slots, n_keys)
            keys = np.repeat(row_keys, np.diff(offsets[r0:r1 + 1]))
            keys += delays_np[start:stop].astype(keys.dtype)
            columns = indices_np[start:stop]
            if _rows_sorted(columns, offsets[r0:r1] - start):
                order = np.argsort(keys, kind='stable')
            else:
                by_column = np.argsort(_sort_keys(columns, n_post), kind='stable')
                order = by_column[np.argsort(keys[by_column], kind='stable')]
            perm[start:stop] = order + start
            indices_out[start:stop] = columns[order]
            delays_out[start:stop] = delays_np[start:stop][order]
            # delay_ptr[i, d]: the first layout position of delay d in row i
            seg = np.empty(n_keys, np.int64)
            seg[0] = start
            np.cumsum(np.bincount(keys, minlength=n_keys)[:-1], out=seg[1:])
            seg[1:] += start
            delay_ptr[r0 * n_slots:r1 * n_slots] = seg

        # Move each host array to the device and drop it before the next one.
        indices_j = _to_device(indices_out)
        del indices_out
        delays_j = _to_device(delays_out)
        del delays_out
        perm_j = _to_device(perm)
        del perm
        if np.shape(data_arr)[0] == nse and nse > 1:
            data_arr = _take(data_arr, perm_j)
        return cls(
            data_arr,
            indices_j,
            _to_device(offsets.astype(ptr_dtype)),
            delays_j,
            _to_device(delay_ptr.reshape(n_pre, n_slots)),
            perm_j,
            shape=(n_pre, n_post),
        )

    def with_data(self, data: Data) -> 'CSRDelayLayout':
        """Return a copy holding ``data`` given in the original edge order.

        Parameters
        ----------
        data : jax.Array or brainunit.Quantity
            Scalar, shape ``(1,)`` or ``(nse,)``, in the order of the CSR
            the layout was built from. May be traced.

        Returns
        -------
        CSRDelayLayout
            The layout with ``data[perm]`` as its weights; the structure
            arrays are shared.

        Raises
        ------
        ValueError
            If ``data`` has another shape.

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> from brainevent import CSRDelayLayout
            >>> layout = CSRDelayLayout.build(
            ...     jnp.array([1., 2.]), jnp.array([0, 1], dtype=jnp.int32),
            ...     jnp.array([0, 2], dtype=jnp.int32), jnp.array([2, 1], dtype=jnp.int32),
            ...     n_slots=3, shape=(1, 2))
            >>> layout.with_data(jnp.array([10., 20.])).data
            Array([20., 10.], dtype=float32)
        """
        if not isinstance(data, u.Quantity):
            data = jnp.asarray(data)
        if data.ndim == 0:
            data = data.reshape(1)
        if data.ndim != 1 or data.shape[0] not in (1, self.nse):
            raise ValueError(f'data must be a scalar, shape (1,) or ({self.nse},); got {data.shape}.')
        if data.shape[0] == self.nse and self.nse > 1 and self.perm is not None:
            data = _take(data, self.perm)
        return CSRDelayLayout(
            data, self.indices, self.indptr, self.delays, self.delay_ptr, self.perm, shape=self.shape,
        )

    def original_data(self) -> Data:
        """Return the weights in the original edge order.

        Returns
        -------
        jax.Array or brainunit.Quantity
            ``data`` scattered back through ``perm`` (unchanged for
            homogeneous weights or an identity ``perm``).

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> from brainevent import CSRDelayLayout
            >>> layout = CSRDelayLayout.build(
            ...     jnp.array([1., 2.]), jnp.array([0, 1], dtype=jnp.int32),
            ...     jnp.array([0, 2], dtype=jnp.int32), jnp.array([2, 1], dtype=jnp.int32),
            ...     n_slots=3, shape=(1, 2))
            >>> layout.original_data()
            Array([1., 2.], dtype=float32)
        """
        if self.perm is None or np.shape(self.data)[0] != self.nse or self.nse <= 1:
            return self.data
        mantissa, unit = u.split_mantissa_unit(self.data)
        out = jnp.zeros_like(mantissa).at[self.perm].set(mantissa)
        return u.maybe_decimal(out * unit) if isinstance(self.data, u.Quantity) else out

    def deliver(self, ring, events, t, *, weighted: Optional[bool] = None, backend: Optional[str] = None):
        """Deliver events into ``ring``; see :func:`csr_delay_deliver_layout`.

        Parameters
        ----------
        ring : jax.Array or brainunit.Quantity
            Arrival buffer, shape ``(n_slots, shape[1])``.
        events : jax.Array, BinaryArray or WeightedArray
            Presynaptic events, shape ``(shape[0],)``.
        t : int or jax.Array
            Current integer time step (may be traced).
        weighted : bool or None, optional
            Binary or weighted delivery; ``None`` infers it from the event type.
        backend : str or None, optional
            Backend override.

        Returns
        -------
        jax.Array or brainunit.Quantity
            The updated ring.

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> from brainevent import CSRDelayLayout
            >>> layout = CSRDelayLayout.build(
            ...     jnp.array([1.]), jnp.array([1], dtype=jnp.int32),
            ...     jnp.array([0, 1], dtype=jnp.int32), 1, n_slots=2, shape=(1, 2))
            >>> layout.deliver(jnp.zeros((2, 2)), jnp.array([True]), 0, backend='jax_raw')
            Array([[0., 0.],
                   [0., 1.]], dtype=float32)
        """
        return csr_delay_deliver_layout(ring, events, self, t, weighted=weighted, backend=backend)


def _resolve_event_mode(events, weighted):
    """Unwrap an event wrapper and reconcile its kind with ``weighted``."""
    from brainevent._event import as_event_operand  # pylint: disable=import-outside-toplevel

    operand = as_event_operand(events)
    if operand is None:
        return events, bool(weighted)
    value, wrapper_weighted = operand
    if weighted is not None and bool(weighted) != wrapper_weighted:
        kind = 'WeightedArray' if wrapper_weighted else 'a binary event array'
        raise ValueError(
            f'weighted={weighted} contradicts the event type ({kind}); '
            f'omit `weighted` or pass a raw array.'
        )
    return value, wrapper_weighted


def csr_delay_deliver_layout(
    ring: Data,
    events: Any,
    layout: CSRDelayLayout,
    t: Any,
    *,
    weighted: Optional[bool] = None,
    backend: Optional[str] = None,
) -> Data:
    """Deliver spikes with per-edge delays using a precomputed layout.

    Adds, for every edge ``e`` of every active row::

        ring_new[(t + d_e) % L, indices[e]] += w_e * f(x)[row(e)]

    with ``f(x)`` the activity mask (binary mode, ``x != 0``, so NaN and
    negative values are active) or the event value (weighted mode). The ring
    is neither cleared nor read: the caller owns it and the time step.

    Parameters
    ----------
    ring : jax.Array or brainunit.Quantity
        Arrival buffer of shape ``(layout.n_slots, n_post)``, dtype
        ``float16``, ``bfloat16``, ``float32`` or ``float64``. The result
        keeps the ring's dtype and unit; the weights are converted to them
        (the unit of ``data * events`` must be compatible with the ring's).
    events : jax.Array, BinaryArray or WeightedArray
        Presynaptic events of shape ``(n_pre,)``; a :class:`WeightedArray`
        selects weighted delivery.
    layout : CSRDelayLayout
        Connectivity built by :meth:`CSRDelayLayout.build`.
    t : int or jax.Array
        Current integer time step; may be traced. Host integers of any size
        are reduced modulo ``L`` first, so they cannot overflow ``int32``.
    weighted : bool or None, optional
        ``False``: binary, ``True``: weighted, ``None`` (default): inferred
        from the event type (binary for a raw array).
    backend : str or None, optional
        ``'numba'`` (CPU default), ``'cuda_raw'`` (GPU default) or
        ``'jax_raw'``.

    Returns
    -------
    ring_new : jax.Array or brainunit.Quantity
        Updated ring, same shape, dtype and unit as ``ring``.

    Raises
    ------
    TypeError
        If ``layout`` is not a :class:`CSRDelayLayout` or ``t`` is not an
        integer scalar.
    ValueError
        If ``ring`` does not have shape ``(layout.n_slots, n_post)``, or
        ``weighted`` contradicts the event type.
    brainunit.UnitMismatchError
        If the unit of ``data * events`` is incompatible with the ring's.

    See Also
    --------
    CSRDelayLayout : The layout and how to build it.

    Notes
    -----
    On ``cuda_raw`` the active rows are compacted first, then a persistent
    grid processes the items ``(d, active row)`` in delay-major order, so
    concurrent writes share the ring row ``(t + d) mod L``. ``numba`` visits
    the active rows only; ``jax_raw`` is one scatter-add over all edges.

    The event type decides the derivatives. Binary: the weight gradient is
    the activity mask times the ring cotangent, and the event gradient is
    straight-through (as if weighted). Weighted: both are exact. Gradients
    with respect to ``layout.data`` are in layout order.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent import CSRDelayLayout, csr_delay_deliver_layout
        >>> layout = CSRDelayLayout.build(
        ...     jnp.array([0.5]), jnp.array([0, 2, 1], dtype=jnp.int32),
        ...     jnp.array([0, 2, 3], dtype=jnp.int32), jnp.array([1, 2, 1], dtype=jnp.int32),
        ...     n_slots=3, shape=(2, 3))
        >>> ring = jnp.zeros((3, 3), dtype=jnp.float32)
        >>> csr_delay_deliver_layout(ring, jnp.array([True, False]), layout, 0, backend='jax_raw')
        Array([[0. , 0. , 0. ],
               [0.5, 0. , 0. ],
               [0. , 0. , 0.5]], dtype=float32)
    """
    if not isinstance(layout, CSRDelayLayout):
        raise TypeError(f'layout must be a CSRDelayLayout; got {type(layout).__name__}.')
    events, weighted = _resolve_event_mode(events, weighted)
    ring_shape = np.shape(u.get_mantissa(ring))
    if ring_shape != (layout.n_slots, layout.shape[1]):
        raise ValueError(
            f'ring must have shape ({layout.n_slots}, {layout.shape[1]}) to match the layout; got {ring_shape}.'
        )
    return _csr_delay_deliver_layout_jit(
        ring, events, layout.data, layout.indices, layout.indptr, layout.delays, layout.delay_ptr,
        _host_time(t, layout.n_slots),
        shape=layout.shape, weighted=weighted, backend=backend,
    )


@namescope(static_argnames=('shape', 'weighted'))
def _csr_delay_deliver_layout_jit(
    ring, events, data, indices, indptr, delays, delay_ptr, t, *, shape, weighted, backend=None,
):
    data, data_unit = u.split_mantissa_unit(data)
    events, event_unit = u.split_mantissa_unit(events)
    ring_values, ring_unit = u.split_mantissa_unit(ring)
    ring_values = jnp.asarray(ring_values)
    data = jnp.asarray(data)
    contribution_unit = data_unit * event_unit
    if contribution_unit != ring_unit:
        # Convert the weights, not the ring: the ring's unit is part of its pytree
        # structure and stays fixed across steps, and converting the whole ring could
        # overflow a half-precision buffer. Scale in at least float32, then narrow.
        scale_dtype = jnp.result_type(data.dtype, ring_values.dtype, jnp.float32)
        data = u.Quantity(data.astype(scale_dtype), unit=contribution_unit).to_decimal(ring_unit)
    if data.dtype != ring_values.dtype:
        data = data.astype(ring_values.dtype)
    res = csr_delay_deliver_layout_p_call(
        ring_values, events, data, indices, indptr, delays, delay_ptr, t,
        shape=shape, weighted=weighted, backend=backend,
    )[0]
    return u.Quantity(res, unit=ring_unit) if isinstance(ring, u.Quantity) else res
