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

"""Tests for ``csr_delay_deliver_layout`` and its primitive: forward, units, autodiff, transforms."""

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.interpreters import ad

import brainevent
from brainevent import BinaryArray, CSRDelayLayout, WeightedArray, csr_delay_deliver_layout
from brainevent._csr.delay import csr_delay_deliver_layout_p, csr_delay_deliver_layout_p_call
from brainevent._csr.delay.delay import _benchmark_data, _transpose_rule
from brainevent._test_util import jax_x64_enabled, requires_gpu
from ._reference import (
    BACKENDS, EVENT_DTYPES, L, N_POST, N_PRE, PLATFORM, build, jax_reference, make_csr, make_events, random_ring,
    reference, x64_disabled,
)

TOL = dict(rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

class TestForward:
    @pytest.mark.parametrize('backend', BACKENDS)
    @pytest.mark.parametrize('homo', [True, False])
    @pytest.mark.parametrize('weighted', [False, True])
    @pytest.mark.parametrize('ev_dtype', EVENT_DTYPES)
    def test_matches_reference(self, backend, homo, weighted, ev_dtype):
        if weighted and ev_dtype == 'bool':
            pytest.skip('weighted delivery needs numeric events')
        data, idx, ipt, dl, lay = build(homo=homo)
        ev = make_events(N_PRE, ev_dtype)
        ring = random_ring()
        out = csr_delay_deliver_layout(ring, jnp.asarray(ev), lay, 4, weighted=weighted, backend=backend)
        assert out.dtype == ring.dtype
        np.testing.assert_allclose(np.asarray(out), reference(ring, ev, data, idx, ipt, dl, 4, weighted=weighted),
                                   **TOL)

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_default_backend_and_int64_indptr(self, backend):
        with jax_x64_enabled():
            data, idx, ipt, dl, lay = build(indptr_dtype=np.int64)
            assert lay.indptr.dtype == np.int64
            ev = make_events(N_PRE, 'bool')
            ring = random_ring()
            out = csr_delay_deliver_layout(ring, jnp.asarray(ev), lay, 1, backend=backend)
            np.testing.assert_allclose(np.asarray(out), reference(ring, ev, data, idx, ipt, dl, 1), **TOL)
        default = csr_delay_deliver_layout(ring, jnp.asarray(ev), build()[4], 1)
        np.testing.assert_allclose(np.asarray(default), reference(ring, ev, data, idx, ipt, dl, 1), **TOL)

    @pytest.mark.parametrize('backend', BACKENDS)
    @pytest.mark.parametrize('t', [0, 6, 7, 123, -5, 2 ** 31 + 3, 2 ** 40 + 3, -2 ** 40, np.int64(2 ** 33 + 1),
                                   np.array(2 ** 35 + 2)])
    def test_host_time_steps_beyond_int32(self, backend, t):
        # x64 off: host integers are reduced modulo L before they reach int32
        data, idx, ipt, dl, lay = build()
        ev = make_events(N_PRE, 'bool')
        ring = random_ring()
        with x64_disabled():
            out = csr_delay_deliver_layout(ring, jnp.asarray(ev), lay, t, backend=backend)
        np.testing.assert_allclose(np.asarray(out), reference(ring, ev, data, idx, ipt, dl, int(t)), **TOL)

    @pytest.mark.parametrize('backend', BACKENDS)
    @pytest.mark.parametrize('t', [-5, 2 ** 40 + 3])
    def test_device_int64_time_steps(self, backend, t):
        data, idx, ipt, dl, lay = build()
        ev = make_events(N_PRE, 'bool')
        ring = random_ring()
        with jax_x64_enabled():
            f = jax.jit(lambda tt: csr_delay_deliver_layout(ring, jnp.asarray(ev), lay, tt, backend=backend))
            out = f(jnp.asarray(t, jnp.int64))
        np.testing.assert_allclose(np.asarray(out), reference(ring, ev, data, idx, ipt, dl, t), **TOL)

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_zero_delays_and_one_slot(self, backend):
        w, idx, ipt, dl = make_csr(N_PRE, N_POST, 9, L, seed=2)
        ev = make_events(N_PRE, 'float32')
        lay = CSRDelayLayout.build(jnp.asarray(w), idx, ipt, 0, n_slots=L, shape=(N_PRE, N_POST))
        ring = random_ring()
        out = csr_delay_deliver_layout(ring, jnp.asarray(ev), lay, 3, backend=backend)
        np.testing.assert_allclose(np.asarray(out), reference(ring, ev, w, idx, ipt, 0, 3), **TOL)
        one = CSRDelayLayout.build(jnp.asarray(w), idx, ipt, 0, n_slots=1, shape=(N_PRE, N_POST))
        ring1 = random_ring(n_slots=1)
        out = csr_delay_deliver_layout(ring1, WeightedArray(jnp.asarray(ev)), one, 12, backend=backend)
        np.testing.assert_allclose(np.asarray(out), reference(ring1, ev, w, idx, ipt, 0, 12, weighted=True), **TOL)

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_collisions_single_delay_and_empty_row(self, backend):
        # Every edge of row 0 hits the same (slot, post); row 1 is empty; row 2 has one delay.
        idx = np.array([3, 3, 3, 1, 2], np.int32)
        ipt = np.array([0, 3, 3, 5], np.int32)
        dl = np.array([2, 2, 2, 4, 4], np.int32)
        w = jnp.asarray([1., 2., 4., 8., 16.], jnp.float32)
        lay = CSRDelayLayout.build(w, idx, ipt, dl, n_slots=5, shape=(3, 5))
        ring = jnp.zeros((5, 5), jnp.float32)
        out = csr_delay_deliver_layout(ring, jnp.array([True, True, True]), lay, 1, backend=backend)
        expected = np.zeros((5, 5), np.float32)
        expected[3, 3] = 7.
        expected[0, 1] = 8.
        expected[0, 2] = 16.
        np.testing.assert_array_equal(np.asarray(out), expected)
        none = csr_delay_deliver_layout(ring, jnp.zeros(3, bool), lay, 1, backend=backend)
        np.testing.assert_array_equal(np.asarray(none), 0.)

    @pytest.mark.parametrize('backend', BACKENDS)
    @pytest.mark.parametrize('n_pre,n_post', [(0, 0), (0, 4), (3, 0), (3, 4)])
    @pytest.mark.parametrize('weighted', [False, True])
    def test_empty_graphs(self, backend, n_pre, n_post, weighted):
        lay = CSRDelayLayout.build(jnp.zeros((0,)), np.zeros(0, np.int32), np.zeros(n_pre + 1, np.int32),
                                   np.zeros(0, np.int32), n_slots=3, shape=(n_pre, n_post))
        ring = jnp.arange(3 * n_post, dtype=jnp.float32).reshape(3, n_post)
        ev = jnp.ones(n_pre, jnp.float32)

        def f(r, w, x):
            return csr_delay_deliver_layout(r, x, lay.with_data(w), 2, weighted=weighted, backend=backend)

        np.testing.assert_array_equal(np.asarray(jax.jit(f)(ring, lay.data, ev)), np.asarray(ring))
        dr, dw, dx = jax.grad(lambda r, w, x: f(r, w, x).sum(), argnums=(0, 1, 2))(ring, lay.data, ev)
        np.testing.assert_array_equal(dr, 1.)
        assert dw.shape == (0,) and dx.shape == (n_pre,)
        np.testing.assert_array_equal(dx, 0.)

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_event_wrappers(self, backend):
        data, idx, ipt, dl, lay = build()
        ev = make_events(N_PRE, 'float32')
        ring = random_ring()
        binary = csr_delay_deliver_layout(ring, BinaryArray(jnp.asarray(ev)), lay, 2, backend=backend)
        np.testing.assert_allclose(np.asarray(binary), reference(ring, ev, data, idx, ipt, dl, 2), **TOL)
        weighted = lay.deliver(ring, WeightedArray(jnp.asarray(ev)), 2, backend=backend)
        np.testing.assert_allclose(np.asarray(weighted), reference(ring, ev, data, idx, ipt, dl, 2, weighted=True),
                                   **TOL)
        with pytest.raises(ValueError, match='contradicts'):
            csr_delay_deliver_layout(ring, WeightedArray(jnp.asarray(ev)), lay, 2, weighted=False)
        with pytest.raises(ValueError, match='contradicts'):
            csr_delay_deliver_layout(ring, BinaryArray(jnp.asarray(ev)), lay, 2, weighted=True)

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_nan_isolation(self, backend):
        data, idx, ipt, dl, lay = build()
        ev = np.asarray(make_events(N_PRE, 'bool'))
        rows = np.repeat(np.arange(N_PRE), np.diff(ipt))
        bad = np.asarray(data).copy()
        bad[~ev[rows]] = np.nan
        bad[np.flatnonzero(~ev[rows])[::2]] = np.inf
        out = csr_delay_deliver_layout(random_ring(), jnp.asarray(ev), lay.with_data(jnp.asarray(bad)), 3,
                                       backend=backend)
        ref = reference(random_ring(), ev, np.where(ev[rows], bad, 0.), idx, ipt, dl, 3)
        assert np.all(np.isfinite(np.asarray(out)))
        np.testing.assert_allclose(np.asarray(out), ref, **TOL)
        # NaN in an active row propagates; NaN events are active.
        bad_active = np.asarray(data).copy()
        bad_active[np.flatnonzero(ev[rows])[0]] = np.nan
        out = csr_delay_deliver_layout(random_ring(), jnp.asarray(ev), lay.with_data(jnp.asarray(bad_active)), 3,
                                       backend=backend)
        assert np.isnan(np.asarray(out)).sum() == 1
        nan_ev = np.where(ev, 1., 0.).astype(np.float32)
        nan_ev[np.flatnonzero(~ev)[0]] = np.nan
        out = csr_delay_deliver_layout(random_ring(), jnp.asarray(nan_ev), lay, 3, backend=backend)
        ref = reference(random_ring(), np.where(np.isnan(nan_ev), 1., nan_ev), data, idx, ipt, dl, 3)
        np.testing.assert_allclose(np.asarray(out), ref, **TOL)

    @pytest.mark.parametrize('backend', BACKENDS)
    @pytest.mark.parametrize('dtype', ['float16', 'bfloat16', 'float64'])
    def test_other_ring_dtypes(self, backend, dtype):
        with jax_x64_enabled():
            data, idx, ipt, dl, lay = build()
            ev = make_events(N_PRE, 'float32')
            ring = random_ring(dtype=dtype)
            for weighted in (False, True):
                events = jnp.asarray(ev).astype(dtype) if weighted else jnp.asarray(ev)
                out = csr_delay_deliver_layout(ring, events, lay, 2, weighted=weighted, backend=backend)
                assert out.dtype == jnp.dtype(dtype)
                w = np.asarray(data.astype(dtype), np.float64)
                ref = reference(np.asarray(ring, np.float64), ev, w, idx, ipt, dl, 2, weighted=weighted)
                tol = 1e-1 if dtype in ('float16', 'bfloat16') else 1e-6
                np.testing.assert_allclose(np.asarray(out, np.float64), ref, rtol=tol, atol=tol)

    def test_validation(self):
        _, _, _, _, lay = build()
        ev = jnp.asarray(make_events(N_PRE, 'bool'))
        kw = dict(shape=lay.shape, backend='jax_raw')
        with pytest.raises(ValueError, match='to match the layout'):
            csr_delay_deliver_layout(random_ring(n_slots=L + 1), ev, lay, 0)
        with pytest.raises(ValueError, match='to match the layout'):
            csr_delay_deliver_layout(random_ring(n_post=N_POST + 1), ev, lay, 0)
        with pytest.raises(TypeError, match='CSRDelayLayout'):
            csr_delay_deliver_layout(random_ring(), ev, object(), 0)
        with pytest.raises(TypeError, match='integer'):
            csr_delay_deliver_layout(random_ring(), ev, lay, 1.5, backend='jax_raw')
        with pytest.raises(TypeError, match='integer'):
            csr_delay_deliver_layout(random_ring(), ev, lay, True, backend='jax_raw')
        with pytest.raises(ValueError, match='events'):
            csr_delay_deliver_layout(random_ring(), ev[:-1], lay, 0, backend='jax_raw')
        with pytest.raises(TypeError, match='ring dtype'):
            csr_delay_deliver_layout(random_ring().astype(jnp.int32), ev, lay, 0, backend='jax_raw')
        args = (random_ring(), ev, lay.data, lay.indices, lay.indptr, lay.delays, lay.delay_ptr, 0)
        with pytest.raises(ValueError, match='delay_ptr'):
            csr_delay_deliver_layout_p_call(*args[:6], lay.delay_ptr[:, :-1], 0, **kw)
        with pytest.raises(TypeError, match='delay_ptr'):
            csr_delay_deliver_layout_p_call(*args[:6], lay.delay_ptr.astype(jnp.int16), 0, **kw)
        with pytest.raises(TypeError, match='data dtype'):
            csr_delay_deliver_layout_p_call(args[0], ev, lay.data.astype(jnp.float16), *args[3:], **kw)
        with pytest.raises(TypeError, match='delays'):
            csr_delay_deliver_layout_p_call(*args[:5], lay.delays.astype(jnp.float32), *args[6:], **kw)
        with pytest.raises(ValueError, match='delays'):
            csr_delay_deliver_layout_p_call(*args[:5], lay.delays[:-1], *args[6:], **kw)
        with pytest.raises(ValueError, match='data'):
            csr_delay_deliver_layout_p_call(args[0], ev, lay.data[:-1], *args[3:], **kw)
        with pytest.raises(ValueError, match='shape'):
            csr_delay_deliver_layout_p_call(*args, shape=(N_PRE,), backend='jax_raw')
        with pytest.raises(ValueError, match='indptr'):
            csr_delay_deliver_layout_p_call(*args, shape=(N_PRE + 1, N_POST), backend='jax_raw')
        with pytest.raises(ValueError, match='1-D'):
            csr_delay_deliver_layout_p_call(*args[:3], lay.indices[None], *args[4:], **kw)
        with pytest.raises(ValueError, match='ring must have shape'):
            csr_delay_deliver_layout_p_call(*args, shape=(N_PRE, N_POST + 1), backend='jax_raw')

    def test_registered_call_and_scalar_operands(self):
        _, _, _, _, lay = build()
        ev = jnp.asarray(make_events(N_PRE, 'bool'))
        ring = random_ring()
        homo = CSRDelayLayout.build(0.5, lay.indices, lay.indptr, 2, n_slots=L, shape=lay.shape)
        a = csr_delay_deliver_layout_p.call(ring, ev, jnp.float32(0.5), homo.indices, homo.indptr, 2,
                                            homo.delay_ptr, 3, shape=lay.shape, backend='jax_raw')[0]
        b = csr_delay_deliver_layout(ring, ev, homo, 3, backend='jax_raw')
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_benchmark_data(self):
        configs = _benchmark_data(platform=PLATFORM)
        assert len(configs) == 4
        for cfg in configs:
            out = csr_delay_deliver_layout_p_call(*cfg.args, **cfg.kernel_kwargs, backend='jax_raw')
            ref = csr_delay_deliver_layout_p_call(*cfg.args, **cfg.kernel_kwargs)
            np.testing.assert_allclose(np.asarray(out[0]), np.asarray(ref[0]), **TOL)

    def test_exports(self):
        assert brainevent.CSRDelayLayout is CSRDelayLayout
        assert brainevent._csr.CSRDelayLayout is CSRDelayLayout
        assert brainevent.csr_delay_deliver_layout_p is csr_delay_deliver_layout_p
        assert 'CSRDelayLayout' in brainevent.__all__ and 'delaycsr' not in brainevent.__all__
        assert not hasattr(brainevent, 'delaycsr')
        assert csr_delay_deliver_layout_p._defaults == {'cpu': 'numba', 'gpu': 'cuda_raw', 'tpu': 'jax_raw'}


# ---------------------------------------------------------------------------
# Units: the ring keeps its unit and dtype
# ---------------------------------------------------------------------------

class TestUnits:
    @pytest.mark.parametrize('backend', BACKENDS)
    def test_matching_units(self, backend):
        data, idx, ipt, dl, _ = build()
        lay = CSRDelayLayout.build(data * u.mS, idx, ipt, dl, n_slots=L, shape=(N_PRE, N_POST))
        ev = make_events(N_PRE, 'bool')
        out = csr_delay_deliver_layout(random_ring() * u.mS, jnp.asarray(ev), lay, 2, backend=backend)
        assert u.get_unit(out) == u.mS
        np.testing.assert_allclose(u.get_mantissa(out), reference(random_ring(), ev, data, idx, ipt, dl, 2), **TOL)

    @pytest.mark.parametrize('backend', BACKENDS)
    @pytest.mark.parametrize('weighted', [False, True])
    def test_ring_keeps_its_unit_and_dtype(self, backend, weighted):
        data, idx, ipt, dl, _ = build()
        lay = CSRDelayLayout.build(data.astype(jnp.float16) * u.mS, idx, ipt, dl, n_slots=L, shape=(N_PRE, N_POST))
        ev = make_events(N_PRE, 'float32')
        ring = random_ring() * u.uA  # mS * mV = uA
        out = jax.jit(lambda r, x: csr_delay_deliver_layout(r, x, lay, 2, weighted=weighted, backend=backend))(
            ring, jnp.asarray(ev) * u.mV)
        assert u.get_unit(out) == u.uA and out.dtype == jnp.float32
        w = np.asarray(data.astype(jnp.float16), np.float64)
        ref = reference(random_ring(), ev, w, idx, ipt, dl, 2, weighted=weighted)
        np.testing.assert_allclose(u.get_mantissa(out), ref, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_scaled_ring_unit(self, backend):
        lay = CSRDelayLayout.build(jnp.array([2.]) * u.mV, np.array([0], np.int32), np.array([0, 1], np.int32),
                                   1, n_slots=3, shape=(1, 1))
        out = csr_delay_deliver_layout(jnp.full((3, 1), .001, jnp.float32) * u.volt, jnp.array([2.]), lay, 0,
                                       weighted=True, backend=backend)
        assert u.get_unit(out) == u.volt and out.dtype == jnp.float32
        np.testing.assert_allclose(u.get_mantissa(out), [[.001], [.005], [.001]], rtol=1e-6)
        with pytest.raises(u.UnitMismatchError):
            csr_delay_deliver_layout(jnp.zeros((3, 1)) * u.ms, jnp.array([2.]), lay, 0, weighted=True)
        with pytest.raises(u.UnitMismatchError):
            csr_delay_deliver_layout(jnp.zeros((3, 1)), jnp.array([2.]), lay, 0, weighted=True)
        # a dimensionless but scaled ring keeps its pytree structure
        ratio = u.mV / u.volt
        lay1 = CSRDelayLayout.build(jnp.array([3.]), np.array([0], np.int32), np.array([0, 1], np.int32), 1,
                                    n_slots=2, shape=(1, 1))
        ring = u.Quantity(jnp.ones((2, 1)), unit=ratio)
        out = jax.jit(lambda r: csr_delay_deliver_layout(r, jnp.array([True]), lay1, 0, backend=backend))(ring)
        assert jax.tree.structure(out) == jax.tree.structure(ring)
        np.testing.assert_allclose(u.get_mantissa(out), [[1.], [3001.]], rtol=1e-6)

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_half_ring_is_not_converted(self, backend):
        # Converting the whole float16 ring from V to mV would overflow (100 V = 1e5 mV).
        lay = CSRDelayLayout.build(jnp.array([2.], jnp.float16) * u.mV, np.array([0], np.int32),
                                   np.array([0, 1], np.int32), 1, n_slots=2, shape=(1, 1))
        ring = jnp.full((2, 1), 100, jnp.float16) * u.volt
        out = csr_delay_deliver_layout(ring, jnp.zeros(1, jnp.float16), lay, 0, backend=backend)
        np.testing.assert_array_equal(u.get_mantissa(out), [[100], [100]])
        # binary activity is tested on the raw event, before any unit scaling
        tiny = jnp.array([1e-5], jnp.float16)
        out = csr_delay_deliver_layout(jnp.zeros((2, 1), jnp.float16) * u.volt, tiny, lay, 0, backend=backend)
        np.testing.assert_allclose(u.get_mantissa(out), [[0], [.002]], rtol=1e-3)


# ---------------------------------------------------------------------------
# Autodiff
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('weighted', [False, True])
def test_grads_match_jax_reference(backend, homo, weighted):
    data, idx, ipt, dl, lay = build(homo=homo)
    ev = jnp.asarray(make_events(N_PRE, 'float32'))
    ring = random_ring()
    g = jnp.asarray(np.random.default_rng(7).normal(size=ring.shape), jnp.float32)
    perm = np.asarray(lay.perm)

    def f_layout(r, w, x):
        lay_w = CSRDelayLayout(w, lay.indices, lay.indptr, lay.delays, lay.delay_ptr, lay.perm, shape=lay.shape)
        return jnp.sum(g * csr_delay_deliver_layout(r, x, lay_w, 5, weighted=weighted, backend=backend))

    def f_ref(r, w, x):
        return jnp.sum(g * jax_reference(r, x, w, idx, ipt, dl, 5, weighted=weighted))

    gl = jax.jit(jax.grad(f_layout, argnums=(0, 1, 2)))(ring, lay.data, ev)
    gr = jax.grad(f_ref, argnums=(0, 1, 2))(ring, data, ev)
    np.testing.assert_allclose(np.asarray(gl[0]), np.asarray(gr[0]), **TOL)
    gr_data = np.asarray(gr[1]) if homo else np.asarray(gr[1])[perm]
    np.testing.assert_allclose(np.asarray(gl[1]), gr_data, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(gl[2]), np.asarray(gr[2]), rtol=1e-4, atol=1e-4)

    _, jl = jax.jvp(f_layout, (ring, lay.data, ev), (jnp.ones_like(ring), jnp.ones_like(lay.data), jnp.ones_like(ev)))
    _, jr = jax.jvp(f_ref, (ring, data, ev), (jnp.ones_like(ring), jnp.ones_like(data), jnp.ones_like(ev)))
    np.testing.assert_allclose(float(jl), float(jr), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('backend', BACKENDS)
def test_grad_through_with_data_is_in_original_order(backend):
    data, idx, ipt, dl, lay = build()
    ev = jnp.asarray(make_events(N_PRE, 'bool'))
    ring = random_ring()

    def f(w):
        return csr_delay_deliver_layout(ring, ev, lay.with_data(w), 1, backend=backend).sum()

    def f_ref(w):
        return jax_reference(ring, ev, w, idx, ipt, dl, 1, weighted=False).sum()

    np.testing.assert_allclose(np.asarray(jax.grad(f)(data)), np.asarray(jax.grad(f_ref)(data)), **TOL)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('weighted', [False, True])
def test_inactive_rows_get_exact_zero_data_cotangent(backend, weighted):
    data, idx, ipt, dl, lay = build()
    ev = make_events(N_PRE, 'float32')
    rows = np.repeat(np.arange(N_PRE), np.diff(ipt))
    ct = np.random.default_rng(1).normal(size=(L, N_POST)).astype(np.float32)
    ct[0, :] = np.nan  # a NaN ring cotangent must not leak into inactive edges
    _, vjp = jax.vjp(lambda w: csr_delay_deliver_layout(random_ring(), jnp.asarray(ev), lay.with_data(w), 0,
                                                        weighted=weighted, backend=backend), data)
    (dw,) = vjp(jnp.asarray(ct))
    inactive = ev[rows] == 0
    np.testing.assert_array_equal(np.asarray(dw)[inactive], 0.)


def _aval(x):
    return jax.core.ShapedArray(jnp.shape(x), jnp.result_type(x))


def test_transpose_rule_structure_and_symbolic_zeros():
    _, _, _, _, lay = build()
    ev = jnp.asarray(make_events(N_PRE, 'float32'))
    ring = random_ring()
    undefined = ad.UndefinedPrimal
    structure = (lay.indices, lay.indptr, lay.delays, lay.delay_ptr, jnp.int32(0))
    kw = dict(shape=lay.shape, weighted=False)
    for k, name in enumerate(('indices', 'indptr', 'delays', 'delay_ptr', 't')):
        args = list(structure)
        args[k] = undefined(_aval(args[k]))
        with pytest.raises(ValueError, match=f'with respect to {name}'):
            _transpose_rule([ring], ring, ev, lay.data, *args, **kw)
    zero = ad.Zero(_aval(ring))
    cts = _transpose_rule([zero], undefined(_aval(ring)), undefined(_aval(ev)),
                          undefined(_aval(lay.data)), *structure, **kw)
    assert all(type(c) is ad.Zero for c in cts[:3]) and cts[3:] == (None,) * 5
    # an integer delay_ptr operand has no tangent space: jax.grad rejects it up front
    with pytest.raises(TypeError):
        jax.grad(lambda dptr: csr_delay_deliver_layout_p_call(
            ring, ev, lay.data, lay.indices, lay.indptr, lay.delays, dptr, 0, **kw, backend='jax_raw')[0].sum())(
            lay.delay_ptr)


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('backend', BACKENDS)
def test_vmap(backend):
    data, idx, ipt, dl, lay = build()
    evs = jnp.asarray(np.stack([make_events(N_PRE, 'bool', seed=s) for s in range(3)]))
    rings = jnp.stack([random_ring(seed=s) for s in range(3)])
    out = jax.vmap(lambda r, e: csr_delay_deliver_layout(r, e, lay, 2, backend=backend))(rings, evs)
    for b in range(3):
        np.testing.assert_allclose(np.asarray(out[b]), reference(rings[b], np.asarray(evs[b]), data, idx, ipt, dl, 2),
                                   **TOL)
    # a batch of weights (original order) and of time steps, on a non-leading axis
    ws = jnp.stack([data * (b + 1) for b in range(3)], axis=1)
    out = jax.vmap(lambda w, t: csr_delay_deliver_layout(rings[0], evs[0], lay.with_data(w), t, backend=backend),
                   in_axes=(1, 0))(ws, jnp.arange(3))
    for b in range(3):
        np.testing.assert_allclose(np.asarray(out[b]),
                                   reference(rings[0], np.asarray(evs[0]), data * (b + 1), idx, ipt, dl, b), **TOL)


@pytest.mark.parametrize('backend', BACKENDS)
def test_scan(backend):
    data, idx, ipt, dl, lay = build()
    train = jnp.asarray(np.random.default_rng(9).random((12, N_PRE)) < 0.3)

    def run(deliver):
        def step(carry, spikes):
            ring, t = carry
            ring = deliver(ring, spikes, t)
            out = ring[t % L]
            ring = ring.at[t % L].set(0.)
            return (ring, t + 1), out

        return jax.lax.scan(step, (jnp.zeros((L, N_POST), jnp.float32), jnp.int32(0)), train)[1]

    a = jax.jit(lambda lay: run(lambda r, s, t: csr_delay_deliver_layout(r, s, lay, t, backend=backend)))(lay)
    b = run(lambda r, s, t: jax_reference(r, s, data, idx, ipt, dl, t, weighted=False))
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), **TOL)


@pytest.mark.parametrize('backend', BACKENDS)
def test_shard_map(backend):
    # one ring per device; the layout is replicated
    data, idx, ipt, dl, lay = build()
    count = len(jax.devices())
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('ring',))
    spec = jax.sharding.PartitionSpec('ring')

    def local(rings, events, layout):
        return csr_delay_deliver_layout(rings[0], events[0], layout, 4, backend=backend)[None]

    f = jax.jit(jax.shard_map(local, mesh=mesh, in_specs=(spec, spec, jax.sharding.PartitionSpec()),
                              out_specs=spec))
    rings = jnp.stack([random_ring(seed=s) for s in range(count)])
    evs = jnp.asarray(np.stack([make_events(N_PRE, 'bool', seed=s) for s in range(count)]))
    out = f(rings, evs, lay)
    for b in range(count):
        np.testing.assert_allclose(np.asarray(out[b]), reference(rings[b], np.asarray(evs[b]), data, idx, ipt, dl, 4),
                                   **TOL)


@pytest.mark.skipif('numba' not in BACKENDS, reason='numba backend unavailable')
def test_numba_layouts_of_different_sizes_share_one_handler():
    # #215: layouts that differ only in size share one registered native handler.
    from brainevent._op.numba_ffi import _NUMBA_CPU_FFI_HANDLES

    sizes = (5, 6, 11)
    layouts = tuple(
        CSRDelayLayout.build(jnp.ones(n, dtype=jnp.float32), jnp.arange(n, dtype=jnp.int32),
                             jnp.array([0, n], dtype=jnp.int32), jnp.ones(n, dtype=jnp.int32),
                             n_slots=2, shape=(1, n))
        for n in sizes
    )
    previous = set(_NUMBA_CPU_FFI_HANDLES)

    @jax.jit
    def deliver():
        return tuple(lay.deliver(jnp.zeros((2, lay.shape[1]), dtype=jnp.float32), jnp.array([True]), 0,
                                 backend='numba')
                     for lay in layouts)

    outs = jax.block_until_ready(deliver())
    for out in outs:
        np.testing.assert_array_equal(np.asarray(out[0]), 0.)
        np.testing.assert_array_equal(np.asarray(out[1]), 1.)
    new = [h for name, h in _NUMBA_CPU_FFI_HANDLES.items()
           if name not in previous and h.kernel.py_func.__name__ == 'deliver']
    assert len(new) <= 1, f'{len(new)} native deliver handlers for {len(sizes)} layout sizes'


@requires_gpu
@pytest.mark.parametrize('dtype', ['float32', 'float16'])
def test_gpu_large_matches_jax_raw(dtype):
    # Enough rows and delays that the persistent grid loops over many items.
    data, idx, ipt, dl, lay = build(n_pre=3000, n_post=2000, max_conn=200, n_slots=60, seed=11)
    lay = lay.with_data(data.astype(dtype))
    ev = jnp.asarray(make_events(3000, 'bool'))
    ring = random_ring(n_slots=60, n_post=2000, dtype=dtype)
    out = csr_delay_deliver_layout(ring, ev, lay, 17, backend='cuda_raw')
    ref = csr_delay_deliver_layout(ring.astype(jnp.float32), ev, lay.with_data(data.astype(dtype).astype(jnp.float32)),
                                   17, backend='jax_raw')
    tol = 1e-4 if dtype == 'float32' else 5e-2
    np.testing.assert_allclose(np.asarray(out, np.float32), np.asarray(ref), rtol=tol, atol=tol)


@requires_gpu
@pytest.mark.parametrize('n_pre,max_conn,n_slots', [(200, 120, 2), (100, 900, 4), (60, 4000, 2), (64, 300, 1)])
@pytest.mark.parametrize('weighted', [False, True])
def test_gpu_long_segments(n_pre, max_conn, n_slots, weighted):
    # mean segment lengths ~30 to ~1000 edges: 32 to 256 lanes per (delay, row) item
    data, idx, ipt, dl, lay = build(n_pre=n_pre, n_post=500, max_conn=max_conn, n_slots=n_slots, seed=3)
    ev = make_events(n_pre, 'float32')
    ring = random_ring(n_slots=n_slots, n_post=500)
    out = csr_delay_deliver_layout(ring, jnp.asarray(ev), lay, 5, weighted=weighted, backend='cuda_raw')
    np.testing.assert_allclose(np.asarray(out), reference(ring, ev, data, idx, ipt, dl, 5, weighted=weighted),
                               rtol=1e-4, atol=1e-3)
