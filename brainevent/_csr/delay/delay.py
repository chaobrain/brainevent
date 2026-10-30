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

"""Fused per-edge-delay spike delivery into a post-side arrival ring (#203)."""

from pathlib import Path
from typing import Any, Optional, Sequence

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._compatible_import import Tracer
from brainevent._misc import _check_csr_cuda_structure_dtypes, _check_csr_structure_dtypes, namescope
from brainevent._op import load_cuda_file, numba_kernel, XLACustomKernel, general_batching_rule
from brainevent._op.benchmark import BenchmarkConfig
from brainevent._op.util import cuda_event_abi, dtype_suffix, event_active, numba_event_mask, select_active
from brainevent._typing import Data, Indptr, Index, MatrixShape

__all__ = [
    'csr_delay_deliver',
    'csr_delay_deliver_p',
]

_RING_DTYPES = (jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64)


def _check_concrete_delays(delays: Any, n_slots: int) -> None:
    """Raise ``ValueError`` if concrete ``delays`` fall outside ``[1, n_slots)``."""
    if isinstance(delays, Tracer) or np.size(delays) == 0:
        return
    if isinstance(delays, jax.Array):
        # Concrete arrays closed over by an outer ``jit`` are still checkable;
        # reduce on device to avoid copying the whole array to the host.
        with jax.ensure_compile_time_eval():
            lo, hi = int(jnp.min(delays)), int(jnp.max(delays))
    else:
        arr = np.asarray(delays)
        lo, hi = int(arr.min()), int(arr.max())
    if lo < 1 or hi >= n_slots:
        raise ValueError(
            f'delays must satisfy 1 <= d < L = {n_slots} (the ring length); '
            f'got values in [{lo}, {hi}].'
        )


def csr_delay_deliver(
    ring: Data,
    events: Any,
    data: Data,
    indices: Index,
    indptr: Indptr,
    delays: Any,
    t: Any,
    *,
    shape: MatrixShape,
    weighted: Optional[bool] = None,
    backend: Optional[str] = None,
) -> Data:
    """Deliver spikes through a CSR matrix with per-edge delays into a ring buffer.

    For every edge ``e`` of every active presynaptic row ``i``, adds the edge
    weight into the post-side arrival ring at the slot the spike will arrive
    in, ``(t + delays[e]) mod L``::

        ring_new[(t + d_e) % L, indices[e]] += w_e * f(x)[i]

    where ``f(x)`` is the activity mask of the events (binary mode; a
    numeric event is active iff ``x != 0``) or the event value itself
    (weighted mode). Only the rows of active events are visited.

    Parameters
    ----------
    ring : jax.Array or brainunit.Quantity
        Arrival buffer of shape ``(L, n_post)`` with a floating-point dtype
        (``float16``, ``bfloat16``, ``float32`` or ``float64``). Its dtype is
        the output dtype. When ``data`` or ``events`` carry units, ``ring``
        must carry ``unit(data) * unit(events)``.
    events : jax.Array, BinaryArray or WeightedArray
        Presynaptic events of shape ``(n_pre,)``. Event wrappers are
        unwrapped; a :class:`WeightedArray` selects weighted delivery. Any
        boolean, integer or floating dtype.
    data : jax.Array or brainunit.Quantity
        Synaptic weights: a scalar / shape ``(1,)`` (homogeneous) or shape
        ``(nse,)`` (heterogeneous). Cast to ``ring.dtype``.
    indices : jax.Array
        CSR column (postsynaptic) indices, shape ``(nse,)``, ``int32``.
    indptr : jax.Array
        CSR row pointers, shape ``(n_pre + 1,)``, ``int32`` or ``int64``.
    delays : jax.Array or int
        Integer per-edge delays in steps, shape ``(nse,)``, or a scalar
        broadcast to every edge.
    t : int or jax.Array
        Current integer time step. May be traced (e.g. the step counter of
        a :func:`jax.lax.scan`).
    shape : tuple of int
        ``(n_pre, n_post)``, the logical shape of the CSR matrix.
    weighted : bool or None, optional
        ``False``: binary delivery, each active row adds ``w_e``.
        ``True``: weighted delivery, each row adds ``x[i] * w_e``.
        ``None`` (default): ``True`` for a :class:`WeightedArray`,
        ``False`` otherwise.
    backend : str or None, optional
        ``'jax_raw'``, ``'numba'`` (CPU) or ``'cuda_raw'`` (GPU), or ``None``
        for the platform default.

    Returns
    -------
    ring_new : jax.Array or brainunit.Quantity
        Updated ring of shape ``(L, n_post)``, same dtype and unit as
        ``ring``.

    Raises
    ------
    ValueError
        If ``delays`` is concrete and any value is outside ``[1, L)``.
    TypeError
        If ``t`` or ``delays`` is not an integer array.
    AssertionError
        On shape or dtype mismatches of the CSR operands.

    See Also
    --------
    binary_csrmv : Scalar-delay (no ring) event-driven delivery.
    CSR.delay_deliver : Method form on a :class:`CSR` matrix.

    Notes
    -----
    ``mod`` is the floor modulo, so the slot is in ``[0, L)`` for any
    ``t`` (``t`` may grow without bound). ``t`` and each ``d`` are reduced
    modulo ``L`` separately, so ``t + d`` never overflows.

    A delay of ``0`` would write into the slot being read in the current
    step and a delay ``>= L`` would alias a later arrival, so concrete
    delays are validated to lie in ``[1, L)``. When ``delays`` is traced
    the check is impossible; out-of-range values then wrap modulo ``L``
    identically on every backend.

    The output is linear in ``ring`` and ``data`` (and, in weighted mode,
    in ``events``). With ``s_e = (t + d_e) mod L`` and output cotangent
    ``g``: ``d ring = g``, ``d data[e] = f(x)[row(e)] * g[s_e, indices[e]]``
    (summed for homogeneous weights) and
    ``d x[i] = sum_{e in row i} w_e * g[s_e, indices[e]]``, which is exact in
    weighted mode and the straight-through estimator in binary mode.
    ``delays``, ``t``, ``indices`` and ``indptr`` are not differentiable.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent import csr_delay_deliver
        >>> # 2 presynaptic, 3 postsynaptic neurons; row 0 -> {0, 2}, row 1 -> {1}
        >>> indices = jnp.array([0, 2, 1], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 2, 3], dtype=jnp.int32)
        >>> delays = jnp.array([1, 2, 1], dtype=jnp.int32)
        >>> ring = jnp.zeros((3, 3), dtype=jnp.float32)
        >>> spikes = jnp.array([True, False])
        >>> csr_delay_deliver(ring, spikes, jnp.array([0.5]), indices, indptr,
        ...                   delays, 0, shape=(2, 3), backend='jax_raw')
        Array([[0. , 0. , 0. ],
               [0.5, 0. , 0. ],
               [0. , 0. , 0.5]], dtype=float32)
    """
    from brainevent._event import as_event_operand  # pylint: disable=import-outside-toplevel

    events, weighted = _resolve_event_mode(events, weighted, as_event_operand(events))
    # Validate outside the jitted body so that concrete delays are visible.
    ring_shape = np.shape(u.get_mantissa(ring))
    if len(ring_shape) == 2:
        _check_concrete_delays(delays, ring_shape[0])
    return _csr_delay_deliver_jit(
        ring, events, data, indices, indptr, delays, t,
        shape=tuple(shape), weighted=bool(weighted), backend=backend,
    )


def _resolve_event_mode(events, weighted, operand):
    """Unwrap an event wrapper and reconcile its kind with ``weighted``."""
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


@namescope(static_argnames=("shape", "weighted"))
def _csr_delay_deliver_jit(ring, events, data, indices, indptr, delays, t, *, shape, weighted, backend=None):
    data, unitd = u.split_mantissa_unit(data)
    events, unite = u.split_mantissa_unit(events)
    out_unit = unitd * unite
    ring_q = ring if isinstance(ring, u.Quantity) else u.Quantity(ring)
    ring = ring_q.to_decimal(out_unit)

    if jnp.ndim(ring) != 2:
        raise ValueError(f'ring must be 2-D (L, n_post); got shape {jnp.shape(ring)}.')
    data = jnp.asarray(data)
    ring = jnp.asarray(ring)
    if data.dtype != ring.dtype:
        data = data.astype(ring.dtype)
    res = csr_delay_deliver_p_call(
        ring, events, data, indices, indptr, delays, t,
        shape=shape, weighted=weighted, backend=backend,
    )[0]
    return u.maybe_decimal(res * out_unit)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

def _slots(t, delays, n_slots):
    """Floor-modulo arrival slot of every edge, without int overflow."""
    return (jnp.remainder(t, n_slots) + jnp.remainder(delays, n_slots)) % n_slots


def _row_ids(indptr, n_pre, nse):
    return jnp.repeat(jnp.arange(n_pre, dtype=indptr.dtype), jnp.diff(indptr), total_repeat_length=nse)


def _jax_kernel(shape: MatrixShape, weighted: bool, weight_info, **kwargs):
    n_pre = shape[0]
    nse = kwargs['indices_info'].size
    is_homo = weight_info.size == 1

    def kernel(ring, events, data, indices, indptr, delays, t):
        n_slots = ring.shape[0]
        row_ids = _row_ids(indptr, n_pre, nse)
        x = events[row_ids]
        active = event_active(x)
        w = data[0] if is_homo else data
        if weighted:
            contrib = w * x.astype(ring.dtype)
        else:
            contrib = jnp.broadcast_to(w, (nse,))
        contrib = select_active(active, contrib.astype(ring.dtype))
        slot = _slots(t, delays, n_slots)
        return (ring.at[slot, indices].add(contrib),)

    return kernel


def _numba_kernel_generator(weighted: bool, weight_info, **kwargs):
    import numba  # pylint: disable=import-outside-toplevel

    is_homo = weight_info.size == 1
    if jnp.dtype(weight_info.dtype) not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise TypeError(
            f'The numba backend of csr_delay_deliver supports float32/float64 rings only; got '
            f'{jnp.dtype(weight_info.dtype)}. Use backend="jax_raw" (or cuda_raw on GPU).'
        )

    ring_dtype = np.dtype(weight_info.dtype).type

    # No fastmath: ``x != 0`` must treat NaN as active. Rows collide on
    # postsynaptic targets, so the row loop is serial. Each active row is
    # processed in two passes: first its flat ring offsets are computed into
    # a scratch buffer, then the weights are scattered. Decoupling the index
    # loads from the random read-modify-writes roughly halves the time on a
    # ring larger than the cache.
    #
    # Performance notes (measured on a 10 MB ring, see
    # examples/csr_delay_deliver_benchmark.py):
    # - Every array is indexed with *unsigned* integers. Numba emits a
    #   negative-index wraparound select for each signed index, which blocks
    #   vectorisation of the offset pass and adds work to the scatter; unsigned
    #   indexing removes it (~20% of the step).
    # - The offset pass is branch-free for delays in ``[0, L)``; out-of-range
    #   delays only set a flag and the row's offsets are then recomputed with a
    #   floor modulo.
    # - The scatter adds ``w`` (or ``w * ring_dtype(x)``) in the ring dtype, so
    #   no float64 promotion and the same rounding as the jax_raw path.
    @numba.njit
    def deliver(ring, events, weights, indices, indptr, delays, t, out):
        n_slots, n_post = out.shape
        flat = out.reshape(-1)
        t_mod = t[0] % n_slots
        w0 = weights[0]
        buf = np.empty(0, np.uint64)
        n_buf = np.uint64(0)
        for i in range(indptr.shape[0] - 1):
            x = events[i]
            if x != 0:
                start = np.uint64(indptr[i])
                m = np.uint64(indptr[i + 1]) - start
                if m > n_buf:
                    buf = np.empty(m, np.uint64)
                    n_buf = m
                out_of_range = False
                for k in range(m):
                    p = start + k
                    d = delays[p]
                    out_of_range |= (d < 0) | (d >= n_slots)
                    s = t_mod + d
                    s = s - n_slots if s >= n_slots else s
                    buf[k] = s * n_post + indices[p]
                if out_of_range:
                    for k in range(m):
                        p = start + k
                        s = (t_mod + delays[p] % n_slots) % n_slots
                        buf[k] = s * n_post + indices[p]
                if weighted:
                    f = ring_dtype(x)
                    for k in range(m):
                        w = w0 if is_homo else weights[start + k]
                        flat[buf[k]] += w * f
                else:
                    for k in range(m):
                        flat[buf[k]] += w0 if is_homo else weights[start + k]

    def kernel(ring, events, data, indices, indptr, delays, t):
        if numba_event_mask(events.dtype):
            # Numba has no float16/bfloat16: read binary events as their mask
            # and weighted events as float32 (exact for both).
            events = events.astype(jnp.float32) if weighted else event_active(events)
        return numba_kernel(deliver, outs=kwargs['outs'], input_output_aliases={0: 0})(
            ring, events, data, indices, indptr, delays, jnp.reshape(t, (1,)),
        )

    return kernel


def _cuda_kernel(weighted: bool, weight_info, events_info, **kwargs):
    _check_csr_cuda_structure_dtypes(kwargs['indices_info'], kwargs['indptr_info'])
    load_cuda_file(
        Path(__file__).parent.joinpath('csr_delay_deliver.cu'),
        name='csr_delay_deliver',
    )
    w_dtype = weight_info.dtype
    if weighted:
        ev_sfx = '_weighted'

        def prepare(x):
            return x.astype(w_dtype) if x.dtype != w_dtype else x
    else:
        ev_sfx, prepare = cuda_event_abi(events_info.dtype)
    kernel_name = f'csr_delay_deliver.csr_delay_deliver{dtype_suffix(w_dtype)}{ev_sfx}'
    out_info = kwargs['outs']

    def kernel(ring, events, data, indices, indptr, delays, t):
        events = prepare(events)
        # The kernel takes int32: floor-reduce wider integers modulo L first so
        # that values >= 2**31 pick the same slot as the jax_raw / numba paths.
        n_slots = ring.shape[0]
        if delays.dtype != jnp.int32:
            delays = jnp.remainder(delays, n_slots).astype(jnp.int32)
        t = jnp.reshape(jnp.remainder(t, n_slots), (1,)).astype(jnp.int32)
        return jax.ffi.ffi_call(kernel_name, out_info, input_output_aliases={0: 0})(
            ring, events, data, indices, indptr, delays, t,
        )

    return kernel


# ---------------------------------------------------------------------------
# Autodiff
# ---------------------------------------------------------------------------

def _jvp_ring(ring_dot, ring, events, data, indices, indptr, delays, t, **kwargs):
    return [ring_dot]


def _jvp_events(events_dot, ring, events, data, indices, indptr, delays, t, *, shape, backend, **kwargs):
    # Weighted: exact. Binary: straight-through (the weighted op on the raw tangent).
    return csr_delay_deliver_p_call(
        jnp.zeros_like(ring), events_dot, data, indices, indptr, delays, t,
        shape=shape, weighted=True, backend=backend,
    )


def _jvp_data(data_dot, ring, events, data, indices, indptr, delays, t, *, shape, weighted, backend, **kwargs):
    return csr_delay_deliver_p_call(
        jnp.zeros_like(ring), events, data_dot, indices, indptr, delays, t,
        shape=shape, weighted=weighted, backend=backend,
    )


def _aval(x):
    return x.aval if ad.is_undefined_primal(x) else x


def _transpose_rule(ct, ring, events, data, indices, indptr, delays, t, *, shape, weighted, **kwargs):
    for name, x in (('indices', indices), ('indptr', indptr), ('delays', delays), ('t', t)):
        if ad.is_undefined_primal(x):
            raise ValueError(f'csr_delay_deliver is not differentiable with respect to {name}.')
    ct = ct[0]
    ct_ring = ct_events = ct_data = None

    if ad.is_undefined_primal(ring):
        ct_ring = ad.Zero(ring.aval) if type(ct) is ad.Zero else ct

    if ad.is_undefined_primal(events) or ad.is_undefined_primal(data):
        n_pre = shape[0]
        nse = indices.shape[0]
        n_slots = _aval(ring).shape[0]
        if type(ct) is not ad.Zero:
            row_ids = _row_ids(indptr, n_pre, nse)
            g = ct[_slots(t, delays, n_slots), indices]
        if ad.is_undefined_primal(data):
            if type(ct) is ad.Zero:
                ct_data = ad.Zero(data.aval)
            else:
                # Select, not ``f * g``: inactive edges get exactly 0 even when the
                # ring cotangent at their slot is NaN/Inf.
                x = events[row_ids]
                value = x.astype(g.dtype) * g if weighted else g
                ct_data = select_active(event_active(x), value)
                if data.aval.shape[0] == 1:
                    ct_data = jnp.sum(ct_data).reshape(data.aval.shape)
                ct_data = ct_data.astype(data.aval.dtype)
        if ad.is_undefined_primal(events):
            if type(ct) is ad.Zero:
                ct_events = ad.Zero(events.aval)
            else:
                w = data[0] if data.shape[0] == 1 else data
                ct_events = jax.ops.segment_sum(w * g, row_ids, num_segments=n_pre)
                ct_events = ct_events.astype(events.aval.dtype)
    return ct_ring, ct_events, ct_data, None, None, None, None


def _batching_rule(args, axes, **kwargs):
    return general_batching_rule(csr_delay_deliver_p, args, axes, **kwargs)


# ---------------------------------------------------------------------------
# Benchmark data
# ---------------------------------------------------------------------------

def _benchmark_data(*, platform):
    n_pre, n_post, n_conn, n_slots, dtype = 1000, 1000, 100, 20, jnp.float32
    rng = np.random.default_rng(0)
    indptr = jnp.asarray(np.arange(n_pre + 1, dtype=np.int32) * n_conn)
    indices = jnp.asarray(rng.integers(0, n_post, n_pre * n_conn, dtype=np.int32))
    delays = jnp.asarray(rng.integers(1, n_slots, n_pre * n_conn, dtype=np.int32))
    ring = jnp.zeros((n_slots, n_post), dtype=dtype)
    t = jnp.asarray(3, dtype=jnp.int32)
    configs = []
    for weighted in (False, True):
        for homo in (True, False):
            data = jnp.ones(1, dtype) if homo else jnp.asarray(rng.random(n_pre * n_conn), dtype)
            if weighted:
                events = jnp.asarray(rng.random(n_pre) * (rng.random(n_pre) < 0.05), dtype)
            else:
                events = jnp.asarray(rng.random(n_pre) < 0.05)
            name = f"{'weighted' if weighted else 'binary'},{'homo' if homo else 'hetero'}"
            configs.append(BenchmarkConfig(
                name,
                (ring, events, data, indices, indptr, delays, t),
                {'shape': (n_pre, n_post), 'weighted': weighted},
            ))
    return configs


# ---------------------------------------------------------------------------
# Primitive call
# ---------------------------------------------------------------------------

def csr_delay_deliver_p_call(
    ring,
    events,
    data,
    indices,
    indptr,
    delays,
    t,
    *,
    shape: Sequence[int],
    weighted: bool = False,
    backend: Optional[str] = None,
):
    """Validate operands and bind :data:`csr_delay_deliver_p` (no units).

    Parameters
    ----------
    ring : jax.Array
        Arrival buffer, shape ``(L, n_post)``, dtype ``float16``,
        ``bfloat16``, ``float32`` or ``float64``.
    events : jax.Array
        Events, shape ``(n_pre,)``.
    data : jax.Array
        Weights: scalar, shape ``(1,)`` or ``(nse,)``; same dtype as ``ring``.
    indices : jax.Array
        CSR column indices, shape ``(nse,)``, ``int32``.
    indptr : jax.Array
        CSR row pointers, shape ``(n_pre + 1,)``, ``int32`` or ``int64``.
    delays : jax.Array or int
        Integer delays, shape ``(nse,)`` or scalar (broadcast).
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
    TypeError
        If ``t`` or ``delays`` is not an integer array, or ``ring`` has an
        unsupported dtype.
    AssertionError
        On shape or dtype mismatches.

    See Also
    --------
    csr_delay_deliver : Unit-aware wrapper with delay validation.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._csr.delay.delay import csr_delay_deliver_p_call
        >>> ring = jnp.zeros((4, 2), dtype=jnp.float32)
        >>> (out,) = csr_delay_deliver_p_call(
        ...     ring, jnp.array([1.0]), jnp.array([2.0]),
        ...     jnp.array([1], dtype=jnp.int32), jnp.array([0, 1], dtype=jnp.int32),
        ...     3, 2, shape=(1, 2), backend='jax_raw')
        >>> out[1]
        Array([0., 2.], dtype=float32)
    """
    operands, infos = _normalize_operands(ring, events, data, indices, indptr, delays, t, shape=shape)
    return csr_delay_deliver_p(
        *operands,
        outs=[jax.ShapeDtypeStruct(operands[0].shape, operands[0].dtype)],
        shape=tuple(int(s) for s in shape),
        weighted=bool(weighted),
        backend=backend,
        **infos,
    )


def _normalize_operands(ring, events, data, indices, indptr, delays, t, *, shape):
    """Validate and normalise the operands of a delay-delivery primitive.

    Returns the operands ``(ring, events, data, indices, indptr, delays, t)``
    as arrays (scalar data reshaped to ``(1,)``, scalar delays broadcast)
    and the ``*_info`` keyword parameters of the primitive.
    """
    shape = tuple(int(s) for s in shape)
    assert len(shape) == 2, f'shape must be (n_pre, n_post); got {shape}.'
    n_pre, n_post = shape
    indices = jnp.asarray(indices)
    indptr = jnp.asarray(indptr)
    assert indptr.ndim == 1 and indices.ndim == 1, 'indices and indptr must be 1-D.'
    _check_csr_structure_dtypes(indices, indptr)
    assert indptr.shape[0] == n_pre + 1, f'indptr shape {indptr.shape} does not match shape {shape}.'
    nse = indices.shape[0]

    ring = jnp.asarray(ring)
    assert ring.ndim == 2 and ring.shape[1] == n_post, (
        f'ring must have shape (L, {n_post}); got {ring.shape}.'
    )
    if not any(ring.dtype == jnp.dtype(d) for d in _RING_DTYPES):
        raise TypeError(f'ring dtype must be float16/bfloat16/float32/float64; got {ring.dtype}.')

    events = jnp.asarray(events)
    assert events.shape == (n_pre,), f'events must have shape ({n_pre},); got {events.shape}.'

    data = jnp.asarray(data)
    if data.ndim == 0:
        data = data.reshape(1)
    assert data.ndim == 1 and data.shape[0] in (1, nse), (
        f'data must be a scalar, shape (1,) or ({nse},); got {data.shape}.'
    )
    assert data.dtype == ring.dtype, f'data dtype {data.dtype} must equal ring dtype {ring.dtype}.'

    delays = jnp.asarray(delays)
    if not jnp.issubdtype(delays.dtype, jnp.integer):
        raise TypeError(f'delays must be an integer array; got dtype {delays.dtype}.')
    if delays.size == 1 and delays.shape != (nse,):
        delays = jnp.broadcast_to(delays.reshape(()), (nse,))
    assert delays.shape == (nse,), f'delays must have shape ({nse},) or be a scalar; got {delays.shape}.'

    t = jnp.asarray(t)
    if not jnp.issubdtype(t.dtype, jnp.integer):
        raise TypeError(f't must be an integer scalar; got dtype {t.dtype}.')
    assert t.ndim == 0, f't must be a scalar; got shape {t.shape}.'

    return (ring, events, data, indices, indptr, delays, t), dict(
        ring_info=jax.ShapeDtypeStruct(ring.shape, ring.dtype),
        events_info=jax.ShapeDtypeStruct(events.shape, events.dtype),
        weight_info=jax.ShapeDtypeStruct(data.shape, data.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
    )


csr_delay_deliver_p = XLACustomKernel(
    'csr_delay_deliver',
    doc="""
Low-level XLA custom-kernel primitive for ``csr_delay_deliver``.

Adds ``w_e * f(x)[row(e)]`` into ``ring[(t + delays[e]) % L, indices[e]]``
for every edge of every active row of a CSR matrix (fused per-edge-delay
spike delivery). Operands are ``(ring, events, data, indices, indptr,
delays, t)``; static parameters are ``shape`` and ``weighted``.

Backends: ``jax_raw`` (all platforms), ``numba`` (CPU, in place),
``cuda_raw`` (GPU, default, in place with atomic adds). JVP, transpose and
batching rules are registered.

Available backends can be queried with
``csr_delay_deliver_p.available_backends(platform)``.

See Also
--------
csr_delay_deliver : High-level user-facing function wrapper.
"""
)
csr_delay_deliver_p.def_numba_kernel(_numba_kernel_generator)
csr_delay_deliver_p.def_cuda_raw_kernel(_cuda_kernel, asdefault=True)
csr_delay_deliver_p.def_kernel('jax_raw', 'cpu', _jax_kernel)
csr_delay_deliver_p.def_kernel('jax_raw', 'gpu', _jax_kernel)
csr_delay_deliver_p.def_kernel('jax_raw', 'tpu', _jax_kernel)
csr_delay_deliver_p.def_jvp_rule2(_jvp_ring, _jvp_events, _jvp_data, None, None, None, None)
csr_delay_deliver_p.def_transpose_rule(_transpose_rule)
csr_delay_deliver_p.def_batching_rule(_batching_rule)
csr_delay_deliver_p.def_call(csr_delay_deliver_p_call)
csr_delay_deliver_p.def_tags('csr', 'delay')
csr_delay_deliver_p.def_benchmark_data(_benchmark_data)
