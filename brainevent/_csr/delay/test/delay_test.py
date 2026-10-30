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

"""Tests for ``csr_delay_deliver`` (fused per-edge-delay spike delivery, #203)."""

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
from brainevent import BinaryArray, CSR, WeightedArray, csr_delay_deliver, csr_delay_deliver_p
from brainevent._csr.delay.delay import _benchmark_data, csr_delay_deliver_p_call
from brainevent._test_util import check_event_op_grads, jax_x64_enabled, requires_gpu

PLATFORM = jax.default_backend()
BACKENDS = tuple(csr_delay_deliver_p.available_backends(PLATFORM))
# float16/bfloat16 events have no numba type; the numba kernel reads them
# through their mask (binary) or as float32 (weighted).
EVENT_DTYPES = ('bool', 'int8', 'int32', 'float32', 'float16', 'bfloat16')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_csr(n_pre, n_post, max_conn, n_slots, seed=0, indptr_dtype=np.int32):
    """Random CSR with repeated targets inside rows, empty rows and delays in [1, L)."""
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, max_conn + 1, n_pre)
    if n_pre > 1:
        counts[0] = 0  # an empty row
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(indptr_dtype)
    nse = int(indptr[-1])
    indices = rng.integers(0, n_post, nse).astype(np.int32)
    delays = rng.integers(1, n_slots, nse).astype(np.int32)
    weights = rng.normal(size=nse).astype(np.float32)
    return weights, indices, indptr, delays


def make_events(n_pre, dtype, seed=1):
    rng = np.random.default_rng(seed)
    if dtype == 'bool':
        return rng.random(n_pre) < 0.5
    vals = rng.choice(np.array([2.0, -1.0, 0.0, 0.5, 0.0, 3.0]), n_pre)
    if jnp.dtype(dtype).kind == 'i':
        vals = np.round(vals * 2)
    return vals.astype(jnp.dtype(dtype))


def reference(ring, events, data, indices, indptr, delays, t, weighted=False):
    """Dense numpy loop: ring[(t + d_e) % L, indices[e]] += w_e * f(x)[row(e)]."""
    out = np.array(ring, dtype=np.float64)
    n_slots = out.shape[0]
    data = np.asarray(data, dtype=np.float64).reshape(-1)
    delays = np.broadcast_to(np.asarray(delays), np.asarray(indices).shape)
    events = np.asarray(events)
    for i in range(len(indptr) - 1):
        x = events[i]
        if x != 0:
            f = float(x) if weighted else 1.0
            for e in range(indptr[i], indptr[i + 1]):
                w = data[0] if data.size == 1 else data[e]
                out[(int(t) + int(delays[e])) % n_slots, indices[e]] += w * f
    return out


def dense_delivery_tensor(data, indices, indptr, delays, t, n_slots, n_post):
    """Differentiable dense ``M[s, post, pre]`` so that ``ring_new = ring + M @ z``."""
    indptr = np.asarray(indptr)
    rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    slots = (int(t) + np.asarray(delays)) % n_slots
    w = jnp.broadcast_to(data.reshape(-1), (len(rows),)) if data.size == 1 else data
    m = jnp.zeros((n_slots, n_post, len(indptr) - 1), dtype=data.dtype)
    return m.at[slots, np.asarray(indices), rows].add(w)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

class TestForward:
    @pytest.mark.parametrize('backend', BACKENDS)
    @pytest.mark.parametrize('homo', [True, False])
    @pytest.mark.parametrize('weighted', [False, True])
    @pytest.mark.parametrize('ev_dtype', EVENT_DTYPES)
    def test_matches_reference(self, backend, homo, weighted, ev_dtype):
        n_pre, n_post, n_slots = 13, 7, 5
        w, indices, indptr, delays = make_csr(n_pre, n_post, 9, n_slots)
        data = np.array([0.7], np.float32) if homo else w
        events = make_events(n_pre, ev_dtype)
        ring = np.random.default_rng(2).normal(size=(n_slots, n_post)).astype(np.float32)
        for t in (0, 3, 7, 1_000_003):
            out = csr_delay_deliver(
                jnp.asarray(ring), jnp.asarray(events), jnp.asarray(data), jnp.asarray(indices),
                jnp.asarray(indptr), jnp.asarray(delays), t,
                shape=(n_pre, n_post), weighted=weighted, backend=backend,
            )
            expected = reference(ring, events, data, indices, indptr, delays, t, weighted)
            np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_collisions(self, backend):
        # Row 0: two edges to post 1 with the same delay; row 1: one edge to post 1, same slot.
        indices = jnp.array([1, 1, 0, 1], dtype=jnp.int32)
        indptr = jnp.array([0, 3, 4], dtype=jnp.int32)
        delays = jnp.array([2, 2, 1, 2], dtype=jnp.int32)
        data = jnp.array([1.0, 2.0, 4.0, 8.0], dtype=jnp.float32)
        ring = jnp.zeros((3, 2), jnp.float32)
        out = csr_delay_deliver(ring, jnp.array([True, True]), data, indices, indptr, delays, 0,
                                shape=(2, 2), backend=backend)
        expected = np.array([[0, 0], [4, 0], [0, 11]], np.float32)
        np.testing.assert_array_equal(np.asarray(out), expected)

    @pytest.mark.parametrize('backend', BACKENDS)
    @pytest.mark.parametrize('t', [-1, 4, 5, 2 ** 31 - 1])
    def test_wraparound(self, backend, t):
        n_slots = 5
        indices = jnp.array([0, 1, 2], dtype=jnp.int32)
        indptr = jnp.array([0, 3], dtype=jnp.int32)
        delays = jnp.array([1, 4, 3], dtype=jnp.int32)
        ring = jnp.zeros((n_slots, 3), jnp.float32)
        out = csr_delay_deliver(ring, jnp.array([1.0]), jnp.array([1.0]), indices, indptr, delays,
                                jnp.asarray(t, jnp.int32), shape=(1, 3), backend=backend)
        expected = reference(np.zeros((n_slots, 3)), [1.0], [1.0], np.asarray(indices),
                             np.asarray(indptr), np.asarray(delays), t)
        np.testing.assert_array_equal(np.asarray(out), expected)

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_scalar_delay_and_binary_array(self, backend):
        n_pre, n_post, n_slots = 9, 6, 4
        w, indices, indptr, _ = make_csr(n_pre, n_post, 5, n_slots)
        events = make_events(n_pre, 'bool')
        ring = jnp.zeros((n_slots, n_post), jnp.float32)
        out = csr_delay_deliver(ring, BinaryArray(jnp.asarray(events)), jnp.asarray(w), jnp.asarray(indices),
                                jnp.asarray(indptr), 2, 3, shape=(n_pre, n_post), backend=backend)
        expected = reference(np.zeros((n_slots, n_post)), events, w, indices, indptr, 2, 3)
        np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize('backend', BACKENDS)
    @pytest.mark.parametrize('ev_dtype', ('int8', 'float32'))
    def test_weighted_array_selects_weighted_mode(self, backend, ev_dtype):
        n_pre, n_post, n_slots = 9, 6, 4
        w, indices, indptr, delays = make_csr(n_pre, n_post, 5, n_slots)
        events = make_events(n_pre, ev_dtype)
        ring = jnp.zeros((n_slots, n_post), jnp.float32)
        args = (jnp.asarray(w), jnp.asarray(indices), jnp.asarray(indptr), jnp.asarray(delays), 1)
        expected = reference(np.zeros((n_slots, n_post)), events, w, indices, indptr, delays, 1, True)
        for weighted in (None, True):
            out = csr_delay_deliver(ring, WeightedArray(jnp.asarray(events)), *args,
                                    shape=(n_pre, n_post), weighted=weighted, backend=backend)
            np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-5, atol=1e-5)
        csr = CSR((jnp.asarray(w), jnp.asarray(indices), jnp.asarray(indptr)), shape=(n_pre, n_post))
        out = csr.delay_deliver(ring, WeightedArray(jnp.asarray(events)), jnp.asarray(delays), 1, backend=backend)
        np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-5, atol=1e-5)

    def test_contradictory_weighted_flag_raises(self):
        w, indices, indptr, delays = make_csr(4, 3, 2, 3)
        args = (jnp.asarray(w), jnp.asarray(indices), jnp.asarray(indptr), jnp.asarray(delays), 0)
        ring = jnp.zeros((3, 3), jnp.float32)
        with pytest.raises(ValueError, match='contradicts'):
            csr_delay_deliver(ring, WeightedArray(jnp.ones(4)), *args, shape=(4, 3), weighted=False)
        with pytest.raises(ValueError, match='contradicts'):
            csr_delay_deliver(ring, BinaryArray(jnp.ones(4, bool)), *args, shape=(4, 3), weighted=True)

    def test_weighted_array_grads_are_exact(self):
        n_pre, n_post, n_slots = 6, 5, 4
        w, indices, indptr, delays = make_csr(n_pre, n_post, 3, n_slots)
        events = jnp.asarray(make_events(n_pre, 'float32'))
        ring = jnp.zeros((n_slots, n_post), jnp.float32)
        args = (jnp.asarray(indices), jnp.asarray(indptr), jnp.asarray(delays), 2)

        def loss(data, x, wrap):
            ev = WeightedArray(x) if wrap else x
            out = csr_delay_deliver(ring, ev, data, *args, shape=(n_pre, n_post),
                                    weighted=None if wrap else True, backend='jax_raw')
            return jnp.sum(out ** 2)

        g_wrap = jax.grad(loss, argnums=(0, 1))(jnp.asarray(w), events, True)
        g_raw = jax.grad(loss, argnums=(0, 1))(jnp.asarray(w), events, False)
        for a, b in zip(g_wrap, g_raw):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6)

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_empty_matrix_and_no_spikes(self, backend):
        ring = jnp.ones((3, 4), jnp.float32)
        out = csr_delay_deliver(ring, jnp.ones(2, bool), jnp.ones(1, jnp.float32),
                                jnp.zeros(0, jnp.int32), jnp.zeros(3, jnp.int32), jnp.zeros(0, jnp.int32),
                                1, shape=(2, 4), backend=backend)
        np.testing.assert_array_equal(np.asarray(out), np.ones((3, 4)))
        w, indices, indptr, delays = make_csr(5, 4, 3, 3)
        out = csr_delay_deliver(ring, jnp.zeros(5, jnp.float32), jnp.asarray(w), jnp.asarray(indices),
                                jnp.asarray(indptr), jnp.asarray(delays), 1, shape=(5, 4), backend=backend)
        np.testing.assert_array_equal(np.asarray(out), np.ones((3, 4)))

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_nan_event_is_active(self, backend):
        indices = jnp.array([0], dtype=jnp.int32)
        indptr = jnp.array([0, 1], dtype=jnp.int32)
        out = csr_delay_deliver(jnp.zeros((2, 1)), jnp.array([jnp.nan]), jnp.array([1.5]), indices, indptr,
                                jnp.array([1], jnp.int32), 0, shape=(1, 1), backend=backend)
        np.testing.assert_array_equal(np.asarray(out), [[0.0], [1.5]])

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_csr_method(self, backend):
        n_pre, n_post, n_slots = 8, 5, 4
        w, indices, indptr, delays = make_csr(n_pre, n_post, 4, n_slots)
        csr = CSR((jnp.asarray(w), jnp.asarray(indices), jnp.asarray(indptr)), shape=(n_pre, n_post))
        events = make_events(n_pre, 'float32')
        ring = jnp.zeros((n_slots, n_post), jnp.float32)
        for weighted in (False, True):
            out = csr.delay_deliver(ring, jnp.asarray(events), jnp.asarray(delays), 2,
                                    weighted=weighted, backend=backend)
            expected = reference(np.zeros((n_slots, n_post)), events, w, indices, indptr, delays, 2, weighted)
            np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_jit_matches_eager(self, backend):
        n_pre, n_post, n_slots = 10, 6, 4
        w, indices, indptr, delays = make_csr(n_pre, n_post, 5, n_slots)
        events = jnp.asarray(make_events(n_pre, 'bool'))
        ring = jnp.ones((n_slots, n_post), jnp.float32)

        @jax.jit
        def f(ring, events, t):
            return csr_delay_deliver(ring, events, jnp.asarray(w), jnp.asarray(indices), jnp.asarray(indptr),
                                     jnp.asarray(delays), t, shape=(n_pre, n_post), backend=backend)

        out = f(ring, events, jnp.int32(6))
        expected = reference(np.ones((n_slots, n_post)), np.asarray(events), w, indices, indptr, delays, 6)
        np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-5, atol=1e-5)
        # The input ring is not clobbered by the in-place backends.
        np.testing.assert_array_equal(np.asarray(ring), np.ones((n_slots, n_post)))


# ---------------------------------------------------------------------------
# A mini network: deliver, read slot t % L, clear it, inside lax.scan
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('weighted', [False, True])
def test_scan_network(backend, weighted):
    n, n_slots, n_steps = 20, 6, 15
    w, indices, indptr, delays = make_csr(n, n, 6, n_slots, seed=3)
    rng = np.random.default_rng(4)
    spikes = (rng.random((n_steps, n)) < 0.3) * rng.normal(size=(n_steps, n))
    spikes = spikes.astype(np.float32)

    def step(carry, x):
        ring, t = carry
        ring = csr_delay_deliver(ring, x, jnp.asarray(w), jnp.asarray(indices), jnp.asarray(indptr),
                                 jnp.asarray(delays), t, shape=(n, n), weighted=weighted, backend=backend)
        slot = t % n_slots
        arrived = ring[slot]
        ring = ring.at[slot].set(0.)
        return (ring, t + 1), arrived

    (ring, t), arrived = jax.jit(lambda s: jax.lax.scan(step, (jnp.zeros((n_slots, n)), jnp.int32(0)), s))(
        jnp.asarray(spikes))

    ref_ring = np.zeros((n_slots, n))
    ref_arrived = []
    for k in range(n_steps):
        ref_ring = reference(ref_ring, spikes[k], w, indices, indptr, delays, k, weighted)
        ref_arrived.append(ref_ring[k % n_slots].copy())
        ref_ring[k % n_slots] = 0.
    assert int(t) == n_steps
    np.testing.assert_allclose(np.asarray(arrived), np.stack(ref_arrived), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(ring), ref_ring, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def setup_method(self):
        self.indices = jnp.array([0, 1], dtype=jnp.int32)
        self.indptr = jnp.array([0, 2], dtype=jnp.int32)
        self.ring = jnp.zeros((4, 2), jnp.float32)

    def call(self, delays, t=0, ring=None, **kw):
        return csr_delay_deliver(self.ring if ring is None else ring, jnp.array([True]), jnp.array([1.0]),
                                 self.indices, self.indptr, delays, t, shape=(1, 2), backend='jax_raw', **kw)

    @pytest.mark.parametrize('bad', [[0, 1], [1, 4], [-1, 2]])
    def test_invalid_concrete_delays_raise(self, bad):
        with pytest.raises(ValueError, match='1 <= d < L'):
            self.call(jnp.array(bad, jnp.int32))
        with pytest.raises(ValueError, match='1 <= d < L'):
            self.call(np.array(bad, np.int32))
        with pytest.raises(ValueError, match='1 <= d < L'):
            self.call(bad[0] if bad[0] != 1 else bad[1])

    def test_traced_delays_wrap_modulo_l(self):
        delays = jnp.array([0, 6], jnp.int32)  # 0 -> slot t, 6 -> slot (t + 2) % 4
        out = jax.jit(lambda d: self.call(d, t=1))(delays)
        expected = reference(np.zeros((4, 2)), [True], [1.0], [0, 1], [0, 2], [0, 6], 1)
        np.testing.assert_array_equal(np.asarray(out), expected)

    def test_type_errors(self):
        d = jnp.array([1, 2], jnp.int32)
        with pytest.raises(TypeError, match='t must be an integer'):
            self.call(d, t=1.0)
        with pytest.raises(TypeError, match='delays must be an integer'):
            csr_delay_deliver_p_call(self.ring, jnp.array([True]), jnp.array([1.0]), self.indices, self.indptr,
                                     jnp.array([1.0, 2.0]), 0, shape=(1, 2))
        with pytest.raises(TypeError, match='ring dtype'):
            csr_delay_deliver_p_call(self.ring.astype(jnp.int32), jnp.array([True]), jnp.array([1], jnp.int32),
                                     self.indices, self.indptr, d, 0, shape=(1, 2))

    def test_shape_errors(self):
        d = jnp.array([1, 2], jnp.int32)
        with pytest.raises(ValueError, match='2-D'):
            self.call(d, ring=jnp.zeros(4))
        with pytest.raises(AssertionError, match='ring must have shape'):
            self.call(d, ring=jnp.zeros((4, 3)))
        with pytest.raises(AssertionError, match='delays must have shape'):
            self.call(jnp.array([1, 2, 3], jnp.int32))
        with pytest.raises(AssertionError, match='t must be a scalar'):
            self.call(d, t=jnp.array([1, 2]))


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

def test_units():
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    delays = jnp.array([1, 2], jnp.int32)
    ring = jnp.zeros((3, 2)) * u.mV
    out = csr_delay_deliver(ring, jnp.array([True]), jnp.array([1.0, 2.0]) * u.mV, indices, indptr, delays, 0,
                            shape=(1, 2), backend='jax_raw')
    assert u.get_unit(out) == u.mV
    np.testing.assert_allclose(np.asarray(out.to_decimal(u.mV)), [[0, 0], [1, 0], [0, 2]])
    # The ring is converted to the delivered unit (V -> mV).
    out = csr_delay_deliver(jnp.ones((3, 2)) * u.volt, jnp.array([True]), jnp.array([1.0]) * u.mV, indices,
                            indptr, delays, 0, shape=(1, 2), backend='jax_raw')
    np.testing.assert_allclose(np.asarray(out.to_decimal(u.mV)), [[1000, 1000], [1001, 1000], [1000, 1001]])
    with pytest.raises(Exception):
        csr_delay_deliver(jnp.zeros((3, 2)), jnp.array([True]), jnp.array([1.0]) * u.mV, indices, indptr,
                          delays, 0, shape=(1, 2), backend='jax_raw')


# ---------------------------------------------------------------------------
# Autodiff
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('mode', ['binary', 'weighted'])
def test_grads_vs_dense(backend, homo, mode):
    n_pre, n_post, n_slots, t = 9, 6, 4, 5
    w, indices, indptr, delays = make_csr(n_pre, n_post, 6, n_slots, seed=5)
    data = jnp.array([0.8], jnp.float32) if homo else jnp.asarray(w)
    ring = jnp.asarray(np.random.default_rng(6).normal(size=(n_slots, n_post)), jnp.float32)
    x = jnp.asarray(make_events(n_pre, 'float32', seed=7))
    params = {'ring': ring, 'data': data}

    def op(p, ev):
        return csr_delay_deliver(p['ring'], ev, p['data'], jnp.asarray(indices), jnp.asarray(indptr),
                                 jnp.asarray(delays), t, shape=(n_pre, n_post),
                                 weighted=(mode == 'weighted'), backend=backend)

    def dense_op(p, z):
        m = dense_delivery_tensor(p['data'], indices, indptr, delays, t, n_slots, n_post)
        return p['ring'] + jnp.einsum('spi,i->sp', m, z)

    check_event_op_grads(op, dense_op, params, x, mode=mode)


@pytest.mark.parametrize('backend', BACKENDS)
def test_grad_traced_t_in_scan(backend):
    n, n_slots, n_steps = 8, 4, 6
    w, indices, indptr, delays = make_csr(n, n, 4, n_slots, seed=8)
    spikes = jnp.asarray(np.random.default_rng(9).random((n_steps, n)) < 0.5)

    def loss(data, op):
        def step(carry, x):
            ring, t = carry
            ring = op(ring, x, data, t)
            out = ring[t % n_slots]
            return (ring.at[t % n_slots].set(0.), t + 1), out

        _, outs = jax.lax.scan(step, (jnp.zeros((n_slots, n)), jnp.int32(0)), spikes)
        return jnp.sum(jnp.sin(outs))

    def op_prim(ring, x, data, t):
        return csr_delay_deliver(ring, x, data, jnp.asarray(indices), jnp.asarray(indptr), jnp.asarray(delays),
                                 t, shape=(n, n), backend=backend)

    rows = np.repeat(np.arange(n), np.diff(indptr))

    def op_ref(ring, x, data, t):
        slots = (t + jnp.asarray(delays)) % n_slots
        return ring.at[slots, jnp.asarray(indices)].add(data * x[rows].astype(data.dtype))

    data = jnp.asarray(w)
    np.testing.assert_allclose(jax.grad(loss)(data, op_prim), jax.grad(loss)(data, op_ref), rtol=1e-5, atol=1e-5)


def test_non_differentiable_operands_raise():
    indices = jnp.array([0], dtype=jnp.int32)
    indptr = jnp.array([0, 1], dtype=jnp.int32)
    from jax.interpreters import ad
    from brainevent._csr.delay.delay import _transpose_rule
    ct = [jnp.ones((2, 1))]
    with pytest.raises(ValueError, match='delays'):
        _transpose_rule(ct, jnp.zeros((2, 1)), jnp.ones(1), jnp.ones(1), indices, indptr,
                        ad.UndefinedPrimal(jax.core.ShapedArray((1,), jnp.float32)), jnp.int32(0),
                        shape=(1, 1), weighted=False)


# ---------------------------------------------------------------------------
# vmap
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('batched', ['events', 'ring', 'both'])
def test_vmap(backend, batched):
    n_pre, n_post, n_slots, n_batch = 7, 5, 4, 3
    w, indices, indptr, delays = make_csr(n_pre, n_post, 4, n_slots, seed=10)
    rng = np.random.default_rng(11)
    events = (rng.random((n_batch, n_pre)) < 0.5).astype(np.float32)
    rings = rng.normal(size=(n_batch, n_slots, n_post)).astype(np.float32)

    def f(ring, ev):
        return csr_delay_deliver(ring, ev, jnp.asarray(w), jnp.asarray(indices), jnp.asarray(indptr),
                                 jnp.asarray(delays), 2, shape=(n_pre, n_post), backend=backend)

    if batched == 'events':
        out = jax.vmap(f, in_axes=(None, 0))(jnp.asarray(rings[0]), jnp.asarray(events))
        refs = [reference(rings[0], events[b], w, indices, indptr, delays, 2) for b in range(n_batch)]
    elif batched == 'ring':
        out = jax.vmap(f, in_axes=(0, None))(jnp.asarray(rings), jnp.asarray(events[0]))
        refs = [reference(rings[b], events[0], w, indices, indptr, delays, 2) for b in range(n_batch)]
    else:
        out = jax.vmap(f)(jnp.asarray(rings), jnp.asarray(events))
        refs = [reference(rings[b], events[b], w, indices, indptr, delays, 2) for b in range(n_batch)]
    np.testing.assert_allclose(np.asarray(out), np.stack(refs), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# float64 / int64 under x64
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('weighted', [False, True])
def test_float64_x64(backend, weighted):
    with jax_x64_enabled():
        n_pre, n_post, n_slots = 11, 6, 5
        w, indices, indptr, delays = make_csr(n_pre, n_post, 6, n_slots, seed=12, indptr_dtype=np.int64)
        w = w.astype(np.float64) + 1e-9
        events = make_events(n_pre, 'float64')
        ring = np.random.default_rng(13).normal(size=(n_slots, n_post))
        out = csr_delay_deliver(jnp.asarray(ring), jnp.asarray(events), jnp.asarray(w), jnp.asarray(indices),
                                jnp.asarray(indptr), jnp.asarray(delays, jnp.int64), jnp.int64(9),
                                shape=(n_pre, n_post), weighted=weighted, backend=backend)
        assert out.dtype == jnp.float64
        expected = reference(ring, events, w, indices, indptr, delays, 9, weighted)
        np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Benchmark data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('backend', BACKENDS)
def test_int64_time_and_delays_beyond_int32(backend):
    """``t`` and traced ``delays`` >= 2**31 wrap modulo ``L`` like the int64 formula on every backend."""
    with jax_x64_enabled():
        n_pre, n_post, n_slots = 20, 12, 7
        w, indices, indptr, delays = make_csr(n_pre, n_post, 5, n_slots, seed=21)
        big_delays = delays.astype(np.int64) + np.int64(3) * 2 ** 31
        events = make_events(n_pre, 'bool')
        t = np.int64(2) ** 33 + 5
        fn = jax.jit(lambda d, tt: csr_delay_deliver(
            jnp.zeros((n_slots, n_post), jnp.float32), jnp.asarray(events), jnp.asarray(w), jnp.asarray(indices),
            jnp.asarray(indptr), d, tt, shape=(n_pre, n_post), backend=backend))
        out = fn(jnp.asarray(big_delays), jnp.asarray(t, jnp.int64))
        expected = reference(np.zeros((n_slots, n_post)), events, w, indices, indptr, big_delays, t)
        np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('t', [-3, 0, 4, 2 ** 31 - 1])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('homo', [False, True])
def test_traced_out_of_range_delays_in_some_rows(backend, t, weighted, homo):
    """Rows with delays outside ``[0, L)`` take the floor-modulo fix-up; in-range rows stay on the fast path."""
    n_pre, n_post, n_slots = 9, 6, 5
    w, indices, indptr, delays = make_csr(n_pre, n_post, 8, n_slots, seed=31)
    rng = np.random.default_rng(32)
    delays = delays.copy()
    rows = np.repeat(np.arange(n_pre), np.diff(indptr))
    odd = (rows % 2) == 1  # odd rows get out-of-range delays, even rows keep [1, L)
    delays[odd] = rng.choice(np.array([-1, -n_slots, -7, 0, n_slots, 2 * n_slots + 3, 999], np.int32), odd.sum())
    data = np.array([0.7], np.float32) if homo else w
    events = make_events(n_pre, 'float32', seed=33)
    events[1] = np.nan  # an odd (fix-up) row with a NaN event is active
    ring = rng.normal(size=(n_slots, n_post)).astype(np.float32)
    fn = jax.jit(lambda d: csr_delay_deliver(
        jnp.asarray(ring), jnp.asarray(events), jnp.asarray(data), jnp.asarray(indices), jnp.asarray(indptr),
        d, jnp.asarray(t, jnp.int32), shape=(n_pre, n_post), weighted=weighted, backend=backend))
    out = np.asarray(fn(jnp.asarray(delays)))
    expected = reference(ring, events, data, indices, indptr, delays, t, weighted)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.uint32])
def test_unsigned_delays(backend, dtype):
    n_pre, n_post, n_slots = 12, 7, 6
    w, indices, indptr, delays = make_csr(n_pre, n_post, 6, n_slots, seed=34)
    delays = (delays + (np.arange(delays.size) % 3) * n_slots).astype(dtype)  # some >= L
    events = make_events(n_pre, 'bool', seed=35)
    fn = jax.jit(lambda d: csr_delay_deliver(
        jnp.zeros((n_slots, n_post), jnp.float32), jnp.asarray(events), jnp.asarray(w), jnp.asarray(indices),
        jnp.asarray(indptr), d, 11, shape=(n_pre, n_post), backend=backend))
    out = np.asarray(fn(jnp.asarray(delays)))
    expected = reference(np.zeros((n_slots, n_post)), events, w, indices, indptr, delays.astype(np.int64), 11)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif('numba' not in BACKENDS, reason='numba backend unavailable')
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('homo', [False, True])
def test_numba_bitwise_matches_jax_raw(weighted, homo):
    """The numba scatter accumulates in the ring dtype, in edge order, like the jax_raw scatter-add."""
    n_pre, n_post, n_slots = 40, 9, 7
    w, indices, indptr, delays = make_csr(n_pre, n_post, 30, n_slots, seed=36)
    data = np.array([0.3], np.float32) if homo else w
    events = make_events(n_pre, 'float32', seed=37)
    ring = np.random.default_rng(38).normal(size=(n_slots, n_post)).astype(np.float32)
    outs = [np.asarray(csr_delay_deliver(
        jnp.asarray(ring), jnp.asarray(events), jnp.asarray(data), jnp.asarray(indices), jnp.asarray(indptr),
        jnp.asarray(delays), 5, shape=(n_pre, n_post), weighted=weighted, backend=b)) for b in ('numba', 'jax_raw')]
    np.testing.assert_array_equal(outs[0], outs[1])


def test_benchmark_data_runs():
    configs = _benchmark_data(platform=PLATFORM)
    assert len(configs) == 4
    for cfg in configs:
        out = csr_delay_deliver_p.call(*cfg.args, backend='jax_raw', **cfg.kernel_kwargs)[0]
        ring, events, data, indices, indptr, delays, t = (np.asarray(a) for a in cfg.args)
        expected = reference(ring, events, data, indices, indptr, delays, int(t), cfg.kernel_kwargs['weighted'])
        np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-4, atol=1e-4)


def test_exported():
    assert brainevent.csr_delay_deliver is csr_delay_deliver
    assert brainevent.csr_delay_deliver_p is csr_delay_deliver_p
    assert 'jax_raw' in BACKENDS


# ---------------------------------------------------------------------------
# GPU-specific
# ---------------------------------------------------------------------------

@requires_gpu
class TestGPU:
    def test_default_backend_is_cuda_raw(self):
        assert csr_delay_deliver_p.get_default('gpu') == 'cuda_raw'

    @pytest.mark.parametrize('max_conn', [8, 120, 800])  # thread / warp / block variants
    @pytest.mark.parametrize('homo', [True, False])
    @pytest.mark.parametrize('mode', ['bool', 'float', 'int8', 'weighted'])
    def test_launch_variants(self, max_conn, homo, mode):
        n_pre, n_post, n_slots = 300, 200, 17
        w, indices, indptr, delays = make_csr(n_pre, n_post, max_conn, n_slots, seed=max_conn)
        data = np.array([0.3], np.float32) if homo else w
        weighted = mode == 'weighted'
        events = make_events(n_pre, 'float32' if weighted else mode, seed=14)
        ring = np.random.default_rng(15).normal(size=(n_slots, n_post)).astype(np.float32)
        out = csr_delay_deliver(jnp.asarray(ring), jnp.asarray(events), jnp.asarray(data), jnp.asarray(indices),
                                jnp.asarray(indptr), jnp.asarray(delays), 40, shape=(n_pre, n_post),
                                weighted=weighted, backend='cuda_raw')
        expected = reference(ring, events, data, indices, indptr, delays, 40, weighted)
        np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-4, atol=1e-3)

    @pytest.mark.parametrize('dtype', [jnp.float16, jnp.bfloat16])
    @pytest.mark.parametrize('weighted', [False, True])
    def test_half_precision(self, dtype, weighted):
        n_pre, n_post, n_slots = 40, 30, 6
        w, indices, indptr, delays = make_csr(n_pre, n_post, 10, n_slots, seed=16)
        events = make_events(n_pre, 'float32', seed=17)
        args = (jnp.zeros((n_slots, n_post), dtype), jnp.asarray(events), jnp.asarray(w, dtype),
                jnp.asarray(indices), jnp.asarray(indptr), jnp.asarray(delays), 3)
        out = csr_delay_deliver(*args, shape=(n_pre, n_post), weighted=weighted, backend='cuda_raw')
        assert out.dtype == dtype
        expected = reference(np.zeros((n_slots, n_post)), events, np.asarray(jnp.asarray(w, dtype), np.float64),
                             indices, indptr, delays, 3, weighted)
        np.testing.assert_allclose(np.asarray(out, np.float32), expected, rtol=5e-2, atol=1e-1)

    def test_int64_indptr_and_f64(self):
        with jax_x64_enabled():
            n_pre, n_post, n_slots = 50, 40, 9
            w, indices, indptr, delays = make_csr(n_pre, n_post, 60, n_slots, seed=18, indptr_dtype=np.int64)
            events = make_events(n_pre, 'bool')
            out = csr_delay_deliver(jnp.zeros((n_slots, n_post), jnp.float64), jnp.asarray(events),
                                    jnp.asarray(w, jnp.float64), jnp.asarray(indices), jnp.asarray(indptr),
                                    jnp.asarray(delays), 4, shape=(n_pre, n_post), backend='cuda_raw')
            expected = reference(np.zeros((n_slots, n_post)), events, w.astype(np.float64), indices, indptr,
                                 delays, 4)
            np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-10, atol=1e-10)

    def test_matches_jax_raw_large(self):
        n_pre, n_post, n_slots = 2000, 1500, 50
        w, indices, indptr, delays = make_csr(n_pre, n_post, 200, n_slots, seed=19)
        events = jnp.asarray(np.random.default_rng(20).random(n_pre) < 0.05)
        args = (jnp.zeros((n_slots, n_post), jnp.float32), events, jnp.asarray(w), jnp.asarray(indices),
                jnp.asarray(indptr), jnp.asarray(delays), 123)
        a = csr_delay_deliver(*args, shape=(n_pre, n_post), backend='cuda_raw')
        b = csr_delay_deliver(*args, shape=(n_pre, n_post), backend='jax_raw')
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-4)


def test_scalar_and_mismatched_dtype_data():
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    delays = jnp.array([1, 2], jnp.int32)
    ring = jnp.zeros((3, 2), jnp.float32)
    for data in (0.5, jnp.float16(0.5), jnp.array([0.5, 0.5], jnp.float16)):
        out = csr_delay_deliver(ring, jnp.array([True]), data, indices, indptr, delays, 0,
                                shape=(1, 2), backend='jax_raw')
        assert out.dtype == jnp.float32
        np.testing.assert_array_equal(np.asarray(out), [[0, 0], [0.5, 0], [0, 0.5]])


def test_transpose_rule_symbolic_zero_cotangent():
    from jax.interpreters import ad
    from brainevent._csr.delay.delay import _transpose_rule
    indices = jnp.array([0], dtype=jnp.int32)
    indptr = jnp.array([0, 1], dtype=jnp.int32)
    ring_aval = jax.core.ShapedArray((2, 1), jnp.float32)
    vec_aval = jax.core.ShapedArray((1,), jnp.float32)
    ct = [ad.Zero(ring_aval)]
    args = (indices, indptr, jnp.array([1], jnp.int32), jnp.int32(0))
    res = _transpose_rule(ct, ad.UndefinedPrimal(ring_aval), jnp.ones(1), ad.UndefinedPrimal(vec_aval), *args,
                          shape=(1, 1), weighted=False)
    assert type(res[0]) is ad.Zero and type(res[2]) is ad.Zero
    res = _transpose_rule(ct, jnp.zeros((2, 1)), ad.UndefinedPrimal(vec_aval), jnp.ones(1), *args,
                          shape=(1, 1), weighted=True)
    assert type(res[1]) is ad.Zero


def test_concrete_delays_closed_over_by_jit_are_validated():
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    bad = jnp.array([1, 9], jnp.int32)

    @jax.jit
    def f(ring):
        return csr_delay_deliver(ring, jnp.array([True]), jnp.array([1.0]), indices, indptr, bad, 0,
                                 shape=(1, 2), backend='jax_raw')

    with pytest.raises(ValueError, match='1 <= d < L'):
        f(jnp.zeros((4, 2)))


@pytest.mark.parametrize('dtype', [jnp.float16, jnp.bfloat16])
def test_half_precision_jax_raw_and_numba_rejects(dtype):
    indices = jnp.array([0, 1, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 3], dtype=jnp.int32)
    delays = jnp.array([1, 2, 2], jnp.int32)
    args = (jnp.zeros((3, 2), dtype), jnp.array([True]), jnp.array([0.5, 0.25, 0.25], dtype), indices, indptr,
            delays, 0)
    out = csr_delay_deliver(*args, shape=(1, 2), backend='jax_raw')
    assert out.dtype == dtype
    np.testing.assert_array_equal(np.asarray(out, np.float32), [[0, 0], [0.5, 0], [0, 0.5]])
    if 'numba' in BACKENDS:
        with pytest.raises(brainevent.KernelCompilationError, match='float32/float64'):
            csr_delay_deliver(*args, shape=(1, 2), backend='numba')


# ---------------------------------------------------------------------------
# Inactive rows add nothing, whatever the weight (NaN/Inf).
# docs/specs/2026-10-24-inactive-row-nan-isolation.md
# ---------------------------------------------------------------------------

def _nonfinite_delay_problem(homo, bad, weighted, seed=21):
    n_pre, n_post, n_slots = 8, 5, 4
    w, indices, indptr, delays = make_csr(n_pre, n_post, 4, n_slots, seed=seed)
    events = make_events(n_pre, 'float32', seed=seed)
    events[1::2] = 0.0
    events[0::2] = np.where(events[0::2] == 0, 2.0, events[0::2])
    rows = np.repeat(np.arange(n_pre), np.diff(indptr))
    inactive = events[rows] == 0
    bad_value = {'nan': np.nan, '+inf': np.inf, '-inf': -np.inf}[bad]
    if homo:
        data = np.array([bad_value], np.float32)
    else:
        data = w.copy()
        data[inactive] = bad_value
    ev = jnp.asarray(events) if weighted else jnp.asarray(events != 0)
    ring = np.random.default_rng(seed).normal(size=(n_slots, n_post)).astype(np.float32)
    return ring, ev, data, indices, indptr, delays, rows, inactive, (n_pre, n_post)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('homo', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('bad', ['nan', '+inf', '-inf'])
def test_inactive_rows_with_nonfinite_weights_add_nothing(backend, homo, weighted, bad):
    ring, ev, data, indices, indptr, delays, rows, inactive, shape = _nonfinite_delay_problem(homo, bad, weighted)
    t = 3
    out = np.asarray(csr_delay_deliver(
        jnp.asarray(ring), ev, jnp.asarray(data), jnp.asarray(indices), jnp.asarray(indptr),
        jnp.asarray(delays), t, shape=shape, weighted=weighted, backend=backend,
    ), np.float64)
    expected = np.array(ring, np.float64)
    x = np.asarray(ev, np.float64)
    w = np.broadcast_to(np.asarray(data, np.float64), indices.shape)
    for e in np.flatnonzero(~inactive):
        f = x[rows[e]] if weighted else 1.0
        expected[(t + delays[e]) % ring.shape[0], indices[e]] += w[e] * f
    if not homo:
        assert np.isfinite(out).all()
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('homo', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
def test_weight_cotangent_of_inactive_rows_is_exact_zero(backend, homo, weighted):
    ring, ev, data, indices, indptr, delays, rows, inactive, shape = _nonfinite_delay_problem(False, 'nan', weighted)
    data = jnp.asarray(np.where(np.isfinite(data), data, 1.0).astype(np.float32))
    if homo:
        data = data[:1]
        # Only inactive edges see a NaN cotangent: the homo cotangent stays finite.
        inactive_slots = np.zeros(ring.shape, bool)
        inactive_slots[(3 + delays[inactive]) % ring.shape[0], indices[inactive]] = True
        active_slots = np.zeros(ring.shape, bool)
        active_slots[(3 + delays[~inactive]) % ring.shape[0], indices[~inactive]] = True
        ct = np.where(inactive_slots & ~active_slots, np.nan, 1.0).astype(np.float32)
    else:
        ct = np.full(ring.shape, np.nan, np.float32)

    def f(d):
        return csr_delay_deliver(
            jnp.asarray(ring), ev, d, jnp.asarray(indices), jnp.asarray(indptr),
            jnp.asarray(delays), 3, shape=shape, weighted=weighted, backend=backend,
        )

    _, vjp = jax.vjp(f, data)
    (ct_data,) = vjp(jnp.asarray(ct))
    ct_data = np.asarray(ct_data)
    if homo:
        assert np.isfinite(ct_data).all()
    else:
        np.testing.assert_array_equal(ct_data[inactive], 0.0)
        assert np.isnan(ct_data[~inactive]).all()
