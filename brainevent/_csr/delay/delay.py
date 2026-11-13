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

"""The ``csr_delay_deliver_layout`` primitive: kernels, autodiff and batching rules.

One step adds ``w_e * f(x)[row(e)]`` into ``ring[(t + delays[e]) % L, indices[e]]``
for every edge ``e`` of every active row, on the arrays of a
:class:`~brainevent.CSRDelayLayout`. ``f(x)`` is the activity mask ``x != 0``
(binary) or the event value (weighted). Everything here is unit-free; units are
handled by :func:`brainevent.csr_delay_deliver_layout`.
"""

from pathlib import Path
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._misc import _check_csr_cuda_structure_dtypes, _check_csr_structure_dtypes
from brainevent._op import XLACustomKernel, general_batching_rule, load_cuda_file, numba_kernel
from brainevent._op.benchmark import BenchmarkConfig
from brainevent._op.util import cuda_event_abi, dtype_suffix, event_active, numba_event_mask, select_active

__all__ = [
    'csr_delay_deliver_layout_p',
    'csr_delay_deliver_layout_p_call',
]

_RING_DTYPES = tuple(jnp.dtype(d) for d in (jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64))
_NUMBA_DTYPES = (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64))


# ---------------------------------------------------------------------------
# Shared math
# ---------------------------------------------------------------------------

def _slots(t, delays, n_slots):
    """Floor-modulo arrival slot of every edge, without integer overflow."""
    return (jnp.remainder(t, n_slots) + jnp.remainder(delays, n_slots)) % n_slots


def _row_ids(indptr, n_pre, nse):
    return jnp.repeat(jnp.arange(n_pre, dtype=indptr.dtype), jnp.diff(indptr), total_repeat_length=nse)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

def _jax_kernel(shape, weighted: bool, weight_info, **kwargs):
    n_pre = shape[0]
    nse = kwargs['indices_info'].shape[0]
    is_homo = weight_info.shape[0] == 1

    def kernel(ring, events, data, indices, indptr, delays, delay_ptr, t):
        if nse == 0:
            return (ring,)
        row_ids = _row_ids(indptr, n_pre, nse)
        x = events[row_ids]
        w = data[0] if is_homo else data
        contrib = w * x.astype(ring.dtype) if weighted else jnp.broadcast_to(w, (nse,))
        # Select, not multiply: an inactive edge adds exactly 0, even for a NaN/Inf weight.
        contrib = select_active(event_active(x), contrib.astype(ring.dtype))
        return (ring.at[_slots(t, delays, ring.shape[0]), indices].add(contrib),)

    return kernel


def _numba_kernel_generator(shape, weighted: bool, weight_info, **kwargs):
    import numba  # pylint: disable=import-outside-toplevel

    nse = kwargs['indices_info'].shape[0]
    is_homo = weight_info.shape[0] == 1
    ring_dtype = jnp.dtype(weight_info.dtype)
    # Numba has no float16/bfloat16: such rings accumulate in float32 and are
    # rounded back once per step.
    acc_dtype = ring_dtype if ring_dtype in _NUMBA_DTYPES else jnp.dtype(jnp.float32)
    acc_type = np.dtype(acc_dtype).type
    acc_info = jax.ShapeDtypeStruct(kwargs['outs'][0].shape, acc_dtype)

    # No fastmath: ``x != 0`` must treat NaN as active. Rows collide on
    # postsynaptic targets, so the row loop is serial. Each active row is
    # processed in two passes: its flat ring offsets are computed into a
    # scratch buffer, then the weights are scattered. Decoupling the index
    # loads from the random read-modify-writes roughly halves the time on a
    # ring larger than the cache.
    #
    # Performance notes (examples/csr_delay_deliver_benchmark.py, 10 MB ring):
    # - Every array is indexed with *unsigned* integers. Numba emits a
    #   negative-index wraparound select for each signed index, which blocks
    #   vectorisation of the offset pass and adds work to the scatter.
    # - The offset pass is branch-free for delays in ``[0, L)``; an
    #   out-of-range delay (only possible through the raw constructor) sets a
    #   flag and the row's offsets are recomputed with a floor modulo.
    @numba.njit
    def deliver(ring, events, weights, indices, indptr, delays, t, out):
        n_slots, n_post = out.shape
        flat = out.reshape(-1)
        t_mod = t[0]
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
                    f = acc_type(x)
                    for k in range(m):
                        w = w0 if is_homo else weights[start + k]
                        flat[buf[k]] += w * f
                else:
                    for k in range(m):
                        flat[buf[k]] += w0 if is_homo else weights[start + k]

    call = numba_kernel(deliver, outs=[acc_info], input_output_aliases={0: 0})

    def kernel(ring, events, data, indices, indptr, delays, delay_ptr, t):
        if nse == 0:
            return (ring,)
        if numba_event_mask(events.dtype):
            # Exact for both modes: binary kernels test activity only, and
            # float16/bfloat16 values are exact in float32.
            events = events.astype(jnp.float32) if weighted else event_active(events)
        t = jnp.reshape(jnp.remainder(t, ring.shape[0]), (1,))
        out = call(ring.astype(acc_dtype), events, data.astype(acc_dtype), indices, indptr, delays, t)
        return (out[0].astype(ring_dtype),)

    return kernel


def _cuda_kernel(shape, weighted: bool, weight_info, events_info, **kwargs):
    _check_csr_cuda_structure_dtypes(kwargs['indices_info'], kwargs['indptr_info'])
    load_cuda_file(Path(__file__).parent.joinpath('csr_delay_deliver_layout.cu'), name='csr_delay_deliver_layout')
    w_dtype = weight_info.dtype
    abi = cuda_event_abi(events_info.dtype, weighted=weighted, value_dtype=w_dtype)
    kernel_name = f'csr_delay_deliver_layout.csr_delay_deliver_layout{dtype_suffix(w_dtype)}{abi.suffix}'
    ring_info = kwargs['outs'][0]
    scratch_info = jax.ShapeDtypeStruct((shape[0] + 1,), jnp.int32)

    def kernel(ring, events, data, indices, indptr, delays, delay_ptr, t):
        # ``prepare`` runs here, after ``bind``, so it cannot cut the event gradient.
        events = abi.prepare(events)
        t = jnp.reshape(jnp.remainder(t, ring.shape[0]), (1,)).astype(jnp.int32)
        ring_new, _ = jax.ffi.ffi_call(kernel_name, [ring_info, scratch_info], input_output_aliases={0: 0})(
            ring, events, data, indices, indptr, delay_ptr.reshape(-1), t,
        )
        return (ring_new,)

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


def _aval(x):
    return x.aval if ad.is_undefined_primal(x) else x


def _transpose_rule(ct, ring, events, data, indices, indptr, delays, delay_ptr, t, *, shape, weighted, **kwargs):
    for name, x in (('indices', indices), ('indptr', indptr), ('delays', delays), ('delay_ptr', delay_ptr), ('t', t)):
        if ad.is_undefined_primal(x):
            raise ValueError(f'csr_delay_deliver_layout is not differentiable with respect to {name}.')
    ct = ct[0]
    ct_ring = ct_events = ct_data = None

    if ad.is_undefined_primal(ring):
        ct_ring = ad.Zero(ring.aval) if type(ct) is ad.Zero else ct

    if ad.is_undefined_primal(events) or ad.is_undefined_primal(data):
        n_pre = shape[0]
        nse = indices.shape[0]
        if type(ct) is not ad.Zero:
            row_ids = _row_ids(indptr, n_pre, nse)
            g = ct[_slots(t, delays, _aval(ring).shape[0]), indices]
        if ad.is_undefined_primal(data):
            if type(ct) is ad.Zero:
                ct_data = ad.Zero(data.aval)
            else:
                # Select, not ``f * g``: an inactive edge gets exactly 0 even when
                # the ring cotangent at its slot is NaN/Inf.
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
    return ct_ring, ct_events, ct_data, None, None, None, None, None


def _batching_rule(args, axes, **kwargs):
    return general_batching_rule(csr_delay_deliver_layout_p, args, axes, **kwargs)


# ---------------------------------------------------------------------------
# Benchmark data
# ---------------------------------------------------------------------------

def _benchmark_data(*, platform):
    from .layout import CSRDelayLayout  # pylint: disable=import-outside-toplevel

    n_pre, n_post, n_conn, n_slots, dtype = 1000, 1000, 100, 20, jnp.float32
    rng = np.random.default_rng(0)
    indptr = np.arange(n_pre + 1, dtype=np.int32) * n_conn
    indices = rng.integers(0, n_post, n_pre * n_conn, dtype=np.int32)
    delays = rng.integers(0, n_slots, n_pre * n_conn, dtype=np.int32)
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

    Only static properties (shapes and dtypes) are checked; the layout arrays
    are trusted, as :meth:`brainevent.CSRDelayLayout.build` validated them.

    Parameters
    ----------
    ring : jax.Array
        Arrival buffer, shape ``(L, n_post)``, dtype ``float16``,
        ``bfloat16``, ``float32`` or ``float64``.
    events : jax.Array
        Events, shape ``(n_pre,)``, boolean, integer or floating point.
    data : jax.Array
        Weights in layout order: scalar, shape ``(1,)`` or ``(nse,)``; same
        dtype as ``ring``.
    indices : jax.Array
        Column indices in layout order, shape ``(nse,)``, ``int32``.
    indptr : jax.Array
        Row pointers, shape ``(n_pre + 1,)``, ``int32`` or ``int64``.
    delays : jax.Array or int
        Integer delays in layout order, shape ``(nse,)`` or scalar.
    delay_ptr : jax.Array
        Shape ``(n_pre, L)``, same dtype as ``indptr``.
    t : jax.Array or int
        Integer scalar time step.
    shape : sequence of int
        ``(n_pre, n_post)``.
    weighted : bool, optional
        Binary (``False``, default) or weighted (``True``) delivery.
    backend : str or None, optional
        ``'numba'``, ``'cuda_raw'``, ``'jax_raw'`` or ``None`` for the
        platform default.

    Returns
    -------
    list of jax.Array
        One-element list holding the updated ring.

    Raises
    ------
    TypeError
        If a dtype is unsupported or ``t``/``delays`` is not an integer.
    ValueError
        If the operand shapes are inconsistent.

    See Also
    --------
    brainevent.csr_delay_deliver_layout : Unit-aware wrapper.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent import CSRDelayLayout
        >>> from brainevent._csr.delay import csr_delay_deliver_layout_p_call
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
    if len(shape) != 2:
        raise ValueError(f'shape must be (n_pre, n_post); got {shape}.')
    n_pre, n_post = shape
    indices = jnp.asarray(indices)
    indptr = jnp.asarray(indptr)
    if indptr.ndim != 1 or indices.ndim != 1:
        raise ValueError(f'indices and indptr must be 1-D; got {indices.shape} and {indptr.shape}.')
    _check_csr_structure_dtypes(indices, indptr)
    if indptr.shape[0] != n_pre + 1:
        raise ValueError(f'indptr shape {indptr.shape} does not match shape {shape}.')
    nse = indices.shape[0]

    ring = jnp.asarray(ring)
    if ring.ndim != 2 or ring.shape[1] != n_post:
        raise ValueError(f'ring must have shape (L, {n_post}); got {ring.shape}.')
    if ring.dtype not in _RING_DTYPES:
        raise TypeError(f'ring dtype must be float16/bfloat16/float32/float64; got {ring.dtype}.')
    n_slots = ring.shape[0]

    events = jnp.asarray(events)
    if events.shape != (n_pre,):
        raise ValueError(f'events must have shape ({n_pre},); got {events.shape}.')

    data = jnp.asarray(data)
    if data.ndim == 0:
        data = data.reshape(1)
    if data.ndim != 1 or data.shape[0] not in (1, nse):
        raise ValueError(f'data must be a scalar, shape (1,) or ({nse},); got {data.shape}.')
    if data.dtype != ring.dtype:
        raise TypeError(f'data dtype {data.dtype} must equal ring dtype {ring.dtype}.')

    delays = jnp.asarray(delays)
    if not jnp.issubdtype(delays.dtype, jnp.integer):
        raise TypeError(f'delays must be an integer array; got dtype {delays.dtype}.')
    if delays.size == 1 and delays.shape != (nse,):
        delays = jnp.broadcast_to(delays.reshape(()), (nse,))
    if delays.shape != (nse,):
        raise ValueError(f'delays must have shape ({nse},) or be a scalar; got {delays.shape}.')

    delay_ptr = jnp.asarray(delay_ptr)
    if delay_ptr.shape != (n_pre, n_slots):
        raise ValueError(f'delay_ptr must have shape ({n_pre}, {n_slots}); got {delay_ptr.shape}.')
    if delay_ptr.dtype != indptr.dtype:
        raise TypeError(f'delay_ptr dtype {delay_ptr.dtype} must equal indptr dtype {indptr.dtype}.')

    t = jnp.asarray(t)
    if t.ndim != 0 or not jnp.issubdtype(t.dtype, jnp.integer):
        raise TypeError(f't must be an integer scalar; got shape {t.shape} and dtype {t.dtype}.')

    return csr_delay_deliver_layout_p(
        ring, events, data, indices, indptr, delays, delay_ptr, t,
        outs=[jax.ShapeDtypeStruct(ring.shape, ring.dtype)],
        shape=shape,
        weighted=bool(weighted),
        backend=backend,
        ring_info=jax.ShapeDtypeStruct(ring.shape, ring.dtype),
        events_info=jax.ShapeDtypeStruct(events.shape, events.dtype),
        weight_info=jax.ShapeDtypeStruct(data.shape, data.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
    )


csr_delay_deliver_layout_p = XLACustomKernel(
    'csr_delay_deliver_layout',
    doc="""
Low-level XLA custom-kernel primitive for ``csr_delay_deliver_layout``.

Adds ``w_e * f(x)[row(e)]`` into ``ring[(t + delays[e]) % L, indices[e]]``
for every edge of every active row of a :class:`CSRDelayLayout`. Operands
are ``(ring, events, data, indices, indptr, delays, delay_ptr, t)``; static
parameters are ``shape`` and ``weighted``.

Backends: ``numba`` (CPU, default; event-driven, in place), ``cuda_raw``
(GPU, default; active rows compacted, then a delay-major persistent grid with
atomic adds, in place) and ``jax_raw`` (all platforms; one scatter-add over
all edges). JVP, transpose and batching rules are registered.

See Also
--------
csr_delay_deliver_layout : High-level user-facing function wrapper.
"""
)
csr_delay_deliver_layout_p.def_numba_kernel(_numba_kernel_generator, asdefault=True)
csr_delay_deliver_layout_p.def_cuda_raw_kernel(_cuda_kernel, asdefault=True)
csr_delay_deliver_layout_p.def_kernel('jax_raw', 'cpu', _jax_kernel)
csr_delay_deliver_layout_p.def_kernel('jax_raw', 'gpu', _jax_kernel)
csr_delay_deliver_layout_p.def_kernel('jax_raw', 'tpu', _jax_kernel)
csr_delay_deliver_layout_p.def_jvp_rule2(_jvp_ring, _jvp_events, _jvp_data, None, None, None, None, None)
csr_delay_deliver_layout_p.def_transpose_rule(_transpose_rule)
csr_delay_deliver_layout_p.def_batching_rule(_batching_rule)
csr_delay_deliver_layout_p.def_call(csr_delay_deliver_layout_p_call)
csr_delay_deliver_layout_p.def_tags('csr', 'delay')
csr_delay_deliver_layout_p.def_benchmark_data(_benchmark_data)
