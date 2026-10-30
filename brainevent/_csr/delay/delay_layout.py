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

from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, Tuple

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._compatible_import import Tracer
from brainevent._misc import _check_csr_cuda_structure_dtypes, namescope
from brainevent._op import load_cuda_file, XLACustomKernel, general_batching_rule
from brainevent._op.benchmark import BenchmarkConfig
from brainevent._op.util import cuda_event_abi, dtype_suffix
from brainevent._typing import Data, MatrixShape
from .delay import (
    _check_concrete_delays,
    _jax_kernel,
    _normalize_operands,
    _numba_kernel_generator,
    _resolve_event_mode,
    _transpose_rule,
)

__all__ = [
    'CSRDelayLayout',
    'csr_delay_deliver_layout',
    'csr_delay_deliver_layout_p',
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


@jax.jit
def _take(data, perm):
    """``data[perm]`` as one compiled gather; eager indexing first materialises the
    wrapped (non-negative) copy of ``perm``."""
    return data[perm]


@jax.tree_util.register_pytree_node_class
class CSRDelayLayout:
    """CSR connectivity with per-edge delays, reordered for fast delivery.

    The edges of every row are sorted by delay (ties keep their original
    order) and ``delay_ptr[i, d]`` records where the edges of delay ``d``
    start in row ``i``. :func:`csr_delay_deliver_layout` uses this to
    deliver all active rows delay by delay, so that concurrent writes share
    one ring row; on the GPU this is several times faster than
    :func:`csr_delay_deliver` when the ring does not fit in the L2 cache.

    Build it once with :meth:`build` (or :meth:`CSR.delay_layout`). The
    weights live in layout order: train or update ``layout.data`` directly,
    use :meth:`with_data` to load weights given in the original edge order
    and :meth:`original_data` to read them back in that order.

    Parameters
    ----------
    data : jax.Array or brainunit.Quantity
        Weights in layout order, shape ``(1,)`` (homogeneous) or ``(nse,)``.
    indices : jax.Array
        Column indices in layout order, shape ``(nse,)``, ``int32``.
    indptr : jax.Array
        Row pointers, shape ``(n_pre + 1,)``, ``int32`` or ``int64``.
    delays : jax.Array
        Delays in layout order, shape ``(nse,)``, ``int32``, in ``[1, L)``.
    delay_ptr : jax.Array
        Shape ``(n_pre, L)``, same dtype as ``indptr``: first position of
        delay ``d`` in row ``i``.
    perm : jax.Array
        Shape ``(nse,)``: ``perm[k]`` is the original index of the edge at
        layout position ``k``.
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
    csr_delay_deliver : Delivery on a plain CSR (no precomputation).

    Notes
    -----
    Every array is a pytree leaf, so a layout can be passed through
    ``jax.jit``; ``shape`` is static. ``delay_ptr`` costs
    ``n_pre * L * indptr.itemsize`` bytes.

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

    def __init__(self, data, indices, indptr, delays, delay_ptr, perm, *, shape: MatrixShape):
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

        Runs on the host with NumPy, once per connectivity.

        Parameters
        ----------
        data : jax.Array or brainunit.Quantity
            Weights in the original edge order: scalar, shape ``(1,)`` or
            ``(nse,)``. May be traced.
        indices : array_like
            CSR column indices, shape ``(nse,)``. Must be concrete.
        indptr : array_like
            CSR row pointers, shape ``(n_pre + 1,)``. Must be concrete.
        delays : array_like or int
            Integer delays in steps, shape ``(nse,)`` or a scalar, each in
            ``[1, n_slots)``. Must be concrete.
        n_slots : int
            Ring length ``L``.
        shape : tuple of int
            ``(n_pre, n_post)``.

        Returns
        -------
        CSRDelayLayout
            The reordered connectivity.

        Raises
        ------
        TypeError
            If ``indices``, ``indptr`` or ``delays`` are traced, or
            ``delays`` is not an integer array.
        ValueError
            If a delay is outside ``[1, n_slots)``, ``n_slots < 2``, or the
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
        n_slots = int(n_slots)
        if n_slots < 2:
            raise ValueError(f'n_slots must be >= 2 (delays lie in [1, n_slots)); got {n_slots}.')
        if len(shape) != 2:
            raise ValueError(f'shape must be (n_pre, n_post); got {tuple(shape)}.')
        n_pre, n_post = int(shape[0]), int(shape[1])
        indices_np = np.asarray(indices)
        indptr_np = np.asarray(indptr)
        if indices_np.ndim != 1 or indptr_np.ndim != 1 or indptr_np.shape[0] != n_pre + 1:
            raise ValueError(
                f'indices must be 1-D and indptr must have shape ({n_pre + 1},); '
                f'got {indices_np.shape} and {indptr_np.shape}.'
            )
        nse = indices_np.shape[0]
        if int(indptr_np[-1]) != nse:
            raise ValueError(f'indptr[-1] = {int(indptr_np[-1])} does not match nse = {nse}.')
        delays_np = np.asarray(delays)
        if not np.issubdtype(delays_np.dtype, np.integer):
            raise TypeError(f'delays must be an integer array; got dtype {delays_np.dtype}.')
        if delays_np.size == 1 and delays_np.shape != (nse,):
            delays_np = np.broadcast_to(delays_np.reshape(()), (nse,))
        if delays_np.shape != (nse,):
            raise ValueError(f'delays must have shape ({nse},) or be a scalar; got {delays_np.shape}.')
        _check_concrete_delays(delays_np, n_slots)

        offsets = indptr_np.astype(np.int64)
        if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
            raise ValueError('indptr must start at 0 and be non-decreasing.')

        ptr_dtype = indptr_np.dtype if indptr_np.dtype in (np.int32, np.int64) else np.int32
        perm_dtype = np.int32 if nse < _INT32_PERM else np.int64
        data_arr = data if isinstance(data, u.Quantity) else jnp.asarray(data)
        if np.ndim(data_arr) == 0:
            data_arr = data_arr.reshape(1)
        if np.shape(data_arr)[0] not in (1, nse) or np.ndim(data_arr) != 1:
            raise ValueError(f'data must be a scalar, shape (1,) or ({nse},); got {np.shape(data_arr)}.')

        # Sort the edges of every row by delay; ties keep their original order. Rows are
        # contiguous, so a chunk of whole rows [r0, r1) keeps its positions
        # [indptr[r0], indptr[r1]) in the layout and is sorted on its own: a stable sort
        # of the keys (row - r0) * n_slots + delay. The outputs are written in their final
        # dtypes; no array of nse int64 values is made.
        perm = np.empty(nse, perm_dtype)
        indices_out = np.empty(nse, np.int32)
        delays_out = np.empty(nse, np.int32)
        delay_ptr = np.empty(n_pre * n_slots, ptr_dtype)
        for r0, r1 in _row_chunks(offsets, n_slots):
            start, stop = int(offsets[r0]), int(offsets[r1])
            n_keys = (r1 - r0) * n_slots
            key_dtype = np.uint16 if n_keys <= _UINT16_KEYS else np.int32 if n_keys <= _INT32_KEYS else np.int64
            row_keys = (np.arange(r1 - r0, dtype=np.int64) * n_slots).astype(key_dtype)
            keys = np.repeat(row_keys, np.diff(offsets[r0:r1 + 1]))
            keys += delays_np[start:stop].astype(key_dtype)
            order = np.argsort(keys, kind='stable')
            perm[start:stop] = order + start
            indices_out[start:stop] = indices_np[start:stop][order].astype(np.int32)
            delays_out[start:stop] = delays_np[start:stop][order].astype(np.int32)
            # delay_ptr[i, d]: the first layout position of delay d in row i
            seg = np.empty(n_keys, np.int64)
            seg[0] = start
            np.cumsum(np.bincount(keys, minlength=n_keys)[:-1], out=seg[1:])
            seg[1:] += start
            delay_ptr[r0 * n_slots:r1 * n_slots] = seg.astype(ptr_dtype)

        # Move each host array to the device and drop it before the next one.
        indices_j = jnp.asarray(indices_out)
        del indices_out
        delays_j = jnp.asarray(delays_out)
        del delays_out
        perm_j = jnp.asarray(perm)
        del perm
        if np.shape(data_arr)[0] == nse and nse != 1:
            data_arr = _take(data_arr, perm_j)
        return cls(
            data_arr,
            indices_j,
            jnp.asarray(indptr_np.astype(ptr_dtype)),
            delays_j,
            jnp.asarray(delay_ptr.reshape(n_pre, n_slots)),
            perm_j,
            shape=(n_pre, n_post),
        )

    def with_data(self, data: Data) -> 'CSRDelayLayout':
        """Return a copy holding ``data`` given in the original edge order.

        Parameters
        ----------
        data : jax.Array or brainunit.Quantity
            Scalar, shape ``(1,)`` or ``(nse,)``, in the order of the CSR
            the layout was built from.

        Returns
        -------
        CSRDelayLayout
            The layout with ``data[perm]`` as its weights.

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
        if data.shape[0] == self.nse and self.nse != 1:
            data = data[self.perm]
        return CSRDelayLayout(
            data, self.indices, self.indptr, self.delays, self.delay_ptr, self.perm, shape=self.shape,
        )

    def original_data(self) -> Data:
        """Return the weights in the original edge order.

        Returns
        -------
        jax.Array or brainunit.Quantity
            ``data`` scattered back through ``perm`` (unchanged for
            homogeneous weights).

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
        if np.shape(self.data)[0] != self.nse or self.nse == 1:
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

    Computes the same result as :func:`csr_delay_deliver` on the CSR the
    layout was built from::

        ring_new[(t + d_e) % L, indices[e]] += w_e * f(x)[row(e)]

    with ``f(x)`` the activity mask (binary mode, ``x != 0``) or the event
    value (weighted mode), but processes the active rows delay by delay.

    Parameters
    ----------
    ring : jax.Array or brainunit.Quantity
        Arrival buffer of shape ``(layout.n_slots, n_post)``, dtype
        ``float16``, ``bfloat16``, ``float32`` or ``float64``. When the
        weights or events carry units, ``ring`` must carry
        ``unit(data) * unit(events)``.
    events : jax.Array, BinaryArray or WeightedArray
        Presynaptic events of shape ``(n_pre,)``; a :class:`WeightedArray`
        selects weighted delivery.
    layout : CSRDelayLayout
        Connectivity built by :meth:`CSRDelayLayout.build`.
    t : int or jax.Array
        Current integer time step; may be traced.
    weighted : bool or None, optional
        ``False``: binary, ``True``: weighted, ``None`` (default): inferred
        from the event type.
    backend : str or None, optional
        ``'cuda_raw'`` (GPU default), ``'numba'`` (CPU default) or
        ``'jax_raw'``.

    Returns
    -------
    ring_new : jax.Array or brainunit.Quantity
        Updated ring, same shape, dtype and unit as ``ring``.

    Raises
    ------
    ValueError
        If ``ring`` does not have ``layout.n_slots`` rows.
    TypeError
        If ``t`` is not an integer scalar.

    See Also
    --------
    CSRDelayLayout : The layout and how to build it.
    csr_delay_deliver : Delivery without precomputation.

    Notes
    -----
    On ``cuda_raw`` the active rows are compacted first, then a persistent
    grid processes the items ``(d, active row)`` in delay-major order, so
    concurrent writes share the ring row ``(t + d) mod L``. ``numba`` and
    ``jax_raw`` run the :func:`csr_delay_deliver` kernels on the layout
    arrays. Derivatives follow :func:`csr_delay_deliver`: the gradient with
    respect to ``layout.data`` is in layout order.

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
    from brainevent._event import as_event_operand  # pylint: disable=import-outside-toplevel

    if not isinstance(layout, CSRDelayLayout):
        raise TypeError(f'layout must be a CSRDelayLayout; got {type(layout).__name__}.')
    events, weighted = _resolve_event_mode(events, weighted, as_event_operand(events))
    ring_shape = np.shape(u.get_mantissa(ring))
    if len(ring_shape) != 2 or ring_shape[0] != layout.n_slots:
        raise ValueError(
            f'ring must have shape ({layout.n_slots}, n_post) to match the layout; got {ring_shape}.'
        )
    return _csr_delay_deliver_layout_jit(
        ring, events, layout.data, layout.indices, layout.indptr, layout.delays, layout.delay_ptr, t,
        shape=layout.shape, weighted=bool(weighted), backend=backend,
    )


@namescope(static_argnames=("shape", "weighted"))
def _csr_delay_deliver_layout_jit(
    ring, events, data, indices, indptr, delays, delay_ptr, t, *, shape, weighted, backend=None,
):
    data, unitd = u.split_mantissa_unit(data)
    events, unite = u.split_mantissa_unit(events)
    out_unit = unitd * unite
    ring_q = ring if isinstance(ring, u.Quantity) else u.Quantity(ring)
    ring = jnp.asarray(ring_q.to_decimal(out_unit))
    data = jnp.asarray(data)
    if data.dtype != ring.dtype:
        data = data.astype(ring.dtype)
    res = csr_delay_deliver_layout_p_call(
        ring, events, data, indices, indptr, delays, delay_ptr, t,
        shape=shape, weighted=weighted, backend=backend,
    )[0]
    return u.maybe_decimal(res * out_unit)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

def _drop_delay_ptr(generator):
    """Adapt a ``csr_delay_deliver`` kernel generator (``delay_ptr`` unused)."""

    def make(**kwargs):
        inner = generator(**kwargs)

        def kernel(ring, events, data, indices, indptr, delays, delay_ptr, t):
            return inner(ring, events, data, indices, indptr, delays, t)

        return kernel

    return make


def _cuda_kernel(weighted: bool, weight_info, events_info, **kwargs):
    _check_csr_cuda_structure_dtypes(kwargs['indices_info'], kwargs['indptr_info'])
    load_cuda_file(
        Path(__file__).parent.joinpath('csr_delay_deliver_layout.cu'),
        name='csr_delay_deliver_layout',
    )
    w_dtype = weight_info.dtype
    if weighted:
        ev_sfx = '_weighted'

        def prepare(x):
            return x.astype(w_dtype) if x.dtype != w_dtype else x
    else:
        ev_sfx, prepare = cuda_event_abi(events_info.dtype)
    kernel_name = f'csr_delay_deliver_layout.csr_delay_deliver_layout{dtype_suffix(w_dtype)}{ev_sfx}'
    ring_info = kwargs['outs'][0]
    n_pre = kwargs['shape'][0]
    scratch_info = jax.ShapeDtypeStruct((n_pre + 1,), jnp.int32)

    def kernel(ring, events, data, indices, indptr, delays, delay_ptr, t):
        events = prepare(events)
        n_slots = ring.shape[0]
        t = jnp.reshape(jnp.remainder(t, n_slots), (1,)).astype(jnp.int32)
        ring_new, _ = jax.ffi.ffi_call(
            kernel_name, [ring_info, scratch_info], input_output_aliases={0: 0},
        )(ring, events, data, indices, indptr, delay_ptr.reshape(-1), t)
        return [ring_new]

    return kernel


# ---------------------------------------------------------------------------
# Autodiff
# ---------------------------------------------------------------------------

def _jvp_ring(ring_dot, *args, **kwargs):
    return [ring_dot]


def _jvp_events(events_dot, ring, events, data, indices, indptr, delays, delay_ptr, t, *, shape, backend, **kwargs):
    # Weighted: exact. Binary: straight-through (the weighted op on the raw tangent).
    return csr_delay_deliver_layout_p_call(
        jnp.zeros_like(ring), events_dot, data, indices, indptr, delays, delay_ptr, t,
        shape=shape, weighted=True, backend=backend,
    )


def _jvp_data(data_dot, ring, events, data, indices, indptr, delays, delay_ptr, t, *, shape, weighted, backend,
              **kwargs):
    return csr_delay_deliver_layout_p_call(
        jnp.zeros_like(ring), events, data_dot, indices, indptr, delays, delay_ptr, t,
        shape=shape, weighted=weighted, backend=backend,
    )


def _layout_transpose_rule(ct, ring, events, data, indices, indptr, delays, delay_ptr, t, **kwargs):
    if ad.is_undefined_primal(delay_ptr):
        raise ValueError('csr_delay_deliver_layout is not differentiable with respect to delay_ptr.')
    # The operation equals csr_delay_deliver on the layout arrays.
    ct_ring, ct_events, ct_data, *_ = _transpose_rule(ct, ring, events, data, indices, indptr, delays, t, **kwargs)
    return ct_ring, ct_events, ct_data, None, None, None, None, None


def _batching_rule(args, axes, **kwargs):
    return general_batching_rule(csr_delay_deliver_layout_p, args, axes, **kwargs)


# ---------------------------------------------------------------------------
# Benchmark data
# ---------------------------------------------------------------------------

def _benchmark_data(*, platform):
    n_pre, n_post, n_conn, n_slots, dtype = 1000, 1000, 100, 20, jnp.float32
    rng = np.random.default_rng(0)
    indptr = np.arange(n_pre + 1, dtype=np.int32) * n_conn
    indices = rng.integers(0, n_post, n_pre * n_conn, dtype=np.int32)
    delays = rng.integers(1, n_slots, n_pre * n_conn, dtype=np.int32)
    ring = jnp.zeros((n_slots, n_post), dtype=dtype)
    t = jnp.asarray(3, dtype=jnp.int32)
    configs = []
    for weighted in (False, True):
        for homo in (True, False):
            data = jnp.ones(1, dtype) if homo else jnp.asarray(rng.random(n_pre * n_conn), dtype)
            lay = CSRDelayLayout.build(data, indices, indptr, delays, n_slots=n_slots, shape=(n_pre, n_post))
            if weighted:
                events = jnp.asarray(rng.random(n_pre) * (rng.random(n_pre) < 0.05), dtype)
            else:
                events = jnp.asarray(rng.random(n_pre) < 0.05)
            name = f"{'weighted' if weighted else 'binary'},{'homo' if homo else 'hetero'}"
            configs.append(BenchmarkConfig(
                name,
                (ring, events, lay.data, lay.indices, lay.indptr, lay.delays, lay.delay_ptr, t),
                {'shape': (n_pre, n_post), 'weighted': weighted},
            ))
    return configs


# ---------------------------------------------------------------------------
# Primitive call
# ---------------------------------------------------------------------------

def csr_delay_deliver_layout_p_call(
    ring,
    events,
    data,
    indices,
    indptr,
    delays,
    delay_ptr,
    t,
    *,
    shape: Sequence[int],
    weighted: bool = False,
    backend: Optional[str] = None,
):
    """Validate operands and bind :data:`csr_delay_deliver_layout_p` (no units).

    Parameters
    ----------
    ring : jax.Array
        Arrival buffer, shape ``(L, n_post)``.
    events : jax.Array
        Events, shape ``(n_pre,)``.
    data : jax.Array
        Weights in layout order: shape ``(1,)`` or ``(nse,)``; same dtype as
        ``ring``.
    indices, indptr, delays : jax.Array
        Layout arrays (see :class:`CSRDelayLayout`).
    delay_ptr : jax.Array
        Shape ``(n_pre, L)``, same dtype as ``indptr``.
    t : jax.Array or int
        Integer scalar time step.
    shape : sequence of int
        ``(n_pre, n_post)``.
    weighted : bool, optional
        Binary (``False``) or weighted (``True``) delivery.
    backend : str or None, optional
        Backend name or ``None`` for the default.

    Returns
    -------
    list of jax.Array
        One-element list holding the updated ring.

    Raises
    ------
    TypeError, AssertionError
        As :func:`brainevent._csr.delay.delay.csr_delay_deliver_p_call`, plus a
        mismatched ``delay_ptr``.

    See Also
    --------
    csr_delay_deliver_layout : Unit-aware wrapper.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent import CSRDelayLayout
        >>> from brainevent._csr.delay.delay_layout import csr_delay_deliver_layout_p_call
        >>> lay = CSRDelayLayout.build(
        ...     jnp.array([2.0]), jnp.array([1], dtype=jnp.int32),
        ...     jnp.array([0, 1], dtype=jnp.int32), 3, n_slots=4, shape=(1, 2))
        >>> (out,) = csr_delay_deliver_layout_p_call(
        ...     jnp.zeros((4, 2)), jnp.array([1.0]), lay.data, lay.indices, lay.indptr,
        ...     lay.delays, lay.delay_ptr, 2, shape=(1, 2), backend='jax_raw')
        >>> out[1]
        Array([0., 2.], dtype=float32)
    """
    shape = tuple(int(s) for s in shape)
    delay_ptr = jnp.asarray(delay_ptr)
    ring = jnp.asarray(ring)
    indptr = jnp.asarray(indptr)
    assert delay_ptr.shape == (shape[0], ring.shape[0]), (
        f'delay_ptr must have shape ({shape[0]}, {ring.shape[0]}); got {delay_ptr.shape}.'
    )
    assert delay_ptr.dtype == indptr.dtype, (
        f'delay_ptr dtype {delay_ptr.dtype} must equal indptr dtype {indptr.dtype}.'
    )
    (ring, events, data, indices, indptr, delays, t), infos = _normalize_operands(
        ring, events, data, indices, indptr, delays, t, shape=shape,
    )
    return csr_delay_deliver_layout_p(
        ring, events, data, indices, indptr, delays, delay_ptr, t,
        outs=[jax.ShapeDtypeStruct(ring.shape, ring.dtype)],
        shape=shape,
        weighted=bool(weighted),
        backend=backend,
        **infos,
    )


csr_delay_deliver_layout_p = XLACustomKernel(
    'csr_delay_deliver_layout',
    doc="""
Low-level XLA custom-kernel primitive for ``csr_delay_deliver_layout``.

Same operation as ``csr_delay_deliver_p`` on a :class:`CSRDelayLayout`.
Operands are ``(ring, events, data, indices, indptr, delays, delay_ptr,
t)``; static parameters are ``shape`` and ``weighted``.

Backends: ``cuda_raw`` (GPU, default; delay-major, in place with atomic
adds), ``numba`` (CPU, in place) and ``jax_raw`` (all platforms), the
latter two being the ``csr_delay_deliver`` kernels. JVP, transpose and
batching rules are registered.

See Also
--------
csr_delay_deliver_layout : High-level user-facing function wrapper.
"""
)
csr_delay_deliver_layout_p.def_numba_kernel(_drop_delay_ptr(_numba_kernel_generator))
csr_delay_deliver_layout_p.def_cuda_raw_kernel(_cuda_kernel, asdefault=True)
csr_delay_deliver_layout_p.def_kernel('jax_raw', 'cpu', _drop_delay_ptr(_jax_kernel))
csr_delay_deliver_layout_p.def_kernel('jax_raw', 'gpu', _drop_delay_ptr(_jax_kernel))
csr_delay_deliver_layout_p.def_kernel('jax_raw', 'tpu', _drop_delay_ptr(_jax_kernel))
csr_delay_deliver_layout_p.def_jvp_rule2(_jvp_ring, _jvp_events, _jvp_data, None, None, None, None, None)
csr_delay_deliver_layout_p.def_transpose_rule(_layout_transpose_rule)
csr_delay_deliver_layout_p.def_batching_rule(_batching_rule)
csr_delay_deliver_layout_p.def_call(csr_delay_deliver_layout_p_call)
csr_delay_deliver_layout_p.def_tags('csr', 'delay')
csr_delay_deliver_layout_p.def_benchmark_data(_benchmark_data)
