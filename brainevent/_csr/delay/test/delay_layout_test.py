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

"""Tests for ``CSRDelayLayout`` and ``csr_delay_deliver_layout`` (#203, #205)."""

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent import (
    BinaryArray, CSR, CSRDelayLayout, WeightedArray, csr_delay_deliver, csr_delay_deliver_layout,
    csr_delay_deliver_layout_p,
)
from brainevent._csr.delay.delay_layout import _benchmark_data, csr_delay_deliver_layout_p_call
from brainevent._csr.delay.test.delay_test import make_csr, make_events, reference
from brainevent._test_util import jax_x64_enabled, requires_gpu

PLATFORM = jax.default_backend()
BACKENDS = tuple(csr_delay_deliver_layout_p.available_backends(PLATFORM))
# float16/bfloat16 events have no numba type; the numba kernel reads them
# through their mask (binary) or as float32 (weighted).
EVENT_DTYPES = ('bool', 'int8', 'int32', 'float32', 'float16', 'bfloat16')

N_PRE, N_POST, MAX_CONN, L = 40, 30, 9, 7


def build(seed=0, homo=False, indptr_dtype=np.int32, n_pre=N_PRE, n_post=N_POST, max_conn=MAX_CONN, n_slots=L):
    w, idx, ipt, dl = make_csr(n_pre, n_post, max_conn, n_slots, seed=seed, indptr_dtype=indptr_dtype)
    data = jnp.asarray(w[:1] if homo else w)
    return data, idx, ipt, dl, CSRDelayLayout.build(data, idx, ipt, dl, n_slots=n_slots, shape=(n_pre, n_post))


def random_ring(n_slots=L, n_post=N_POST, dtype=jnp.float32, seed=3):
    return jnp.asarray(np.random.default_rng(seed).normal(size=(n_slots, n_post)), dtype)


# ---------------------------------------------------------------------------
# Layout construction
# ---------------------------------------------------------------------------

class TestBuild:
    @pytest.mark.parametrize('indptr_dtype', [np.int32, np.int64])
    def test_order_and_delay_ptr(self, indptr_dtype):
        with jax_x64_enabled():
            self._check_order_and_delay_ptr(indptr_dtype)

    def _check_order_and_delay_ptr(self, indptr_dtype):
        data, idx, ipt, dl, lay = build(indptr_dtype=indptr_dtype)
        perm = np.asarray(lay.perm)
        assert sorted(perm.tolist()) == list(range(idx.size))
        np.testing.assert_array_equal(np.asarray(lay.indices), idx[perm])
        np.testing.assert_array_equal(np.asarray(lay.delays), dl[perm])
        np.testing.assert_array_equal(np.asarray(lay.data), np.asarray(data)[perm])
        np.testing.assert_array_equal(np.asarray(lay.indptr), ipt)
        assert lay.delay_ptr.dtype == lay.indptr.dtype == np.dtype(indptr_dtype)
        assert lay.delay_ptr.shape == (N_PRE, L) and lay.n_slots == L and lay.nse == idx.size
        dptr = np.asarray(lay.delay_ptr)
        lay_dl = np.asarray(lay.delays)
        for i in range(N_PRE):
            s, e = ipt[i], ipt[i + 1]
            # each row is sorted by delay, ties keep the original edge order
            assert np.all(np.diff(lay_dl[s:e]) >= 0)
            for k in range(s, e - 1):
                if lay_dl[k] == lay_dl[k + 1]:
                    assert perm[k] < perm[k + 1]
            for d in range(L):
                assert dptr[i, d] == s + np.count_nonzero(dl[s:e] < d)

    def test_data_round_trip(self):
        data, idx, ipt, dl, lay = build()
        np.testing.assert_array_equal(np.asarray(lay.original_data()), np.asarray(data))
        new = jnp.arange(idx.size, dtype=jnp.float32)
        lay2 = lay.with_data(new)
        np.testing.assert_array_equal(np.asarray(lay2.original_data()), np.asarray(new))
        assert lay2.indices is lay.indices and lay2.shape == lay.shape

    def test_homogeneous_and_scalar_inputs(self):
        _, idx, ipt, dl = make_csr(N_PRE, N_POST, MAX_CONN, L)
        lay = CSRDelayLayout.build(0.5, idx, ipt, 3, n_slots=L, shape=(N_PRE, N_POST))
        assert lay.data.shape == (1,)
        np.testing.assert_array_equal(np.asarray(lay.delays), np.full(idx.size, 3))
        assert lay.with_data(jnp.float32(2.0)).data.shape == (1,)
        assert lay.original_data().shape == (1,)

    def test_units(self):
        data, idx, ipt, dl, _ = build()
        lay = CSRDelayLayout.build(data * u.mV, idx, ipt, dl, n_slots=L, shape=(N_PRE, N_POST))
        assert u.get_unit(lay.data) == u.mV
        restored = lay.original_data()
        assert u.get_unit(restored) == u.mV
        np.testing.assert_allclose(u.get_mantissa(restored), np.asarray(data))

    def test_traced_data_is_allowed(self):
        _, idx, ipt, dl, _ = build()

        @jax.jit
        def f(w):
            return CSRDelayLayout.build(w, idx, ipt, dl, n_slots=L, shape=(N_PRE, N_POST)).data

        w = jnp.arange(idx.size, dtype=jnp.float32)
        np.testing.assert_array_equal(np.asarray(f(w)), np.asarray(w)[np.asarray(build()[4].perm)])

    def test_errors(self):
        data, idx, ipt, dl, _ = build()
        with pytest.raises(ValueError, match='1 <= d < L'):
            CSRDelayLayout.build(data, idx, ipt, np.where(dl == dl[0], 0, dl), n_slots=L, shape=(N_PRE, N_POST))
        with pytest.raises(ValueError, match='1 <= d < L'):
            CSRDelayLayout.build(data, idx, ipt, dl + L, n_slots=L, shape=(N_PRE, N_POST))
        with pytest.raises(TypeError, match='integer'):
            CSRDelayLayout.build(data, idx, ipt, dl.astype(np.float32), n_slots=L, shape=(N_PRE, N_POST))
        with pytest.raises(ValueError, match='n_slots'):
            CSRDelayLayout.build(data, idx, ipt, 1, n_slots=1, shape=(N_PRE, N_POST))
        with pytest.raises(ValueError, match='indptr'):
            CSRDelayLayout.build(data, idx, ipt[:-1], dl, n_slots=L, shape=(N_PRE, N_POST))
        with pytest.raises(ValueError, match='nse'):
            CSRDelayLayout.build(data, idx[:-1], ipt, dl[:-1], n_slots=L, shape=(N_PRE, N_POST))
        with pytest.raises(ValueError, match='delays must have shape'):
            CSRDelayLayout.build(data, idx, ipt, dl[:-1], n_slots=L, shape=(N_PRE, N_POST))
        with pytest.raises(ValueError, match='data must be'):
            CSRDelayLayout.build(data[:-1], idx, ipt, dl, n_slots=L, shape=(N_PRE, N_POST))
        with pytest.raises(ValueError, match='shape'):
            CSRDelayLayout.build(data, idx, ipt, dl, n_slots=L, shape=(N_PRE,))

        @jax.jit
        def traced(d):
            return CSRDelayLayout.build(data, idx, ipt, d, n_slots=L, shape=(N_PRE, N_POST))

        with pytest.raises(TypeError, match='concrete'):
            traced(jnp.asarray(dl))

    def test_pytree_and_jit(self):
        _, _, _, _, lay = build()
        leaves, tree = jax.tree_util.tree_flatten(lay)
        assert len(leaves) == 6
        back = jax.tree_util.tree_unflatten(tree, leaves)
        assert back.shape == lay.shape
        ev = jnp.asarray(make_events(N_PRE, 'bool'))
        f = jax.jit(lambda r, lay: csr_delay_deliver_layout(r, ev, lay, 2, backend='jax_raw'))
        ring = random_ring()
        _, _, _, _, other = build(seed=5)
        # Layouts with different structure go through the same jitted function.
        f(ring, lay)
        f(ring, other)

    def test_csr_method(self):
        data, idx, ipt, dl, lay = build()
        csr = CSR((data, jnp.asarray(idx), jnp.asarray(ipt)), shape=(N_PRE, N_POST))
        lay2 = csr.delay_layout(dl, n_slots=L)
        np.testing.assert_array_equal(np.asarray(lay2.indices), np.asarray(lay.indices))
        np.testing.assert_array_equal(np.asarray(lay2.data), np.asarray(lay.data))


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

class TestForward:
    @pytest.mark.parametrize('backend', BACKENDS)
    @pytest.mark.parametrize('homo', [True, False])
    @pytest.mark.parametrize('weighted', [False, True])
    @pytest.mark.parametrize('ev_dtype', EVENT_DTYPES)
    def test_matches_reference_and_plain_primitive(self, backend, homo, weighted, ev_dtype):
        if weighted and ev_dtype == 'bool':
            pytest.skip('weighted delivery needs numeric events')
        data, idx, ipt, dl, lay = build(homo=homo)
        ev = make_events(N_PRE, ev_dtype)
        ring = random_ring()
        out = csr_delay_deliver_layout(ring, jnp.asarray(ev), lay, 4, weighted=weighted, backend=backend)
        ref = reference(ring, ev, data, idx, ipt, dl, 4, weighted=weighted)
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5, atol=1e-5)
        plain = csr_delay_deliver(ring, jnp.asarray(ev), data, idx, ipt, dl, 4, shape=(N_PRE, N_POST),
                                  weighted=weighted, backend=backend)
        np.testing.assert_allclose(np.asarray(out), np.asarray(plain), rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_int64_indptr(self, backend):
        data, idx, ipt, dl, lay = build(indptr_dtype=np.int64)
        ev = make_events(N_PRE, 'bool')
        ring = random_ring()
        out = csr_delay_deliver_layout(ring, jnp.asarray(ev), lay, 1, backend=backend)
        np.testing.assert_allclose(np.asarray(out), reference(ring, ev, data, idx, ipt, dl, 1), rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize('backend', BACKENDS)
    @pytest.mark.parametrize('t', [0, 6, 7, 123, -5, 2 ** 40 + 3])
    def test_any_time_step(self, backend, t):
        data, idx, ipt, dl, lay = build()
        ev = make_events(N_PRE, 'bool')
        ring = random_ring()
        with jax_x64_enabled():
            out = csr_delay_deliver_layout(ring, jnp.asarray(ev), lay, jnp.asarray(t, jnp.int64), backend=backend)
        np.testing.assert_allclose(np.asarray(out), reference(ring, ev, data, idx, ipt, dl, t), rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_collisions_single_delay_and_empty(self, backend):
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
    def test_event_wrappers(self, backend):
        data, idx, ipt, dl, lay = build()
        ev = make_events(N_PRE, 'float32')
        ring = random_ring()
        binary = csr_delay_deliver_layout(ring, BinaryArray(jnp.asarray(ev)), lay, 2, backend=backend)
        np.testing.assert_allclose(np.asarray(binary), reference(ring, ev, data, idx, ipt, dl, 2),
                                   rtol=1e-5, atol=1e-5)
        weighted = lay.deliver(ring, WeightedArray(jnp.asarray(ev)), 2, backend=backend)
        np.testing.assert_allclose(np.asarray(weighted), reference(ring, ev, data, idx, ipt, dl, 2, weighted=True),
                                   rtol=1e-5, atol=1e-5)
        with pytest.raises(ValueError, match='contradicts'):
            csr_delay_deliver_layout(ring, WeightedArray(jnp.asarray(ev)), lay, 2, weighted=False)

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
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5, atol=1e-5)
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
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_units(self, backend):
        data, idx, ipt, dl, _ = build()
        lay = CSRDelayLayout.build(data * u.mS, idx, ipt, dl, n_slots=L, shape=(N_PRE, N_POST))
        ev = make_events(N_PRE, 'bool')
        out = csr_delay_deliver_layout(random_ring() * u.mS, jnp.asarray(ev), lay, 2, backend=backend)
        assert u.get_unit(out) == u.mS
        np.testing.assert_allclose(u.get_mantissa(out), reference(random_ring(), ev, data, idx, ipt, dl, 2),
                                   rtol=1e-5, atol=1e-5)

    def test_validation(self):
        _, _, _, _, lay = build()
        ev = jnp.asarray(make_events(N_PRE, 'bool'))
        with pytest.raises(ValueError, match='n_post'):
            csr_delay_deliver_layout(random_ring(n_slots=L + 1), ev, lay, 0)
        with pytest.raises(TypeError, match='CSRDelayLayout'):
            csr_delay_deliver_layout(random_ring(), ev, object(), 0)
        with pytest.raises(TypeError, match='integer'):
            csr_delay_deliver_layout(random_ring(), ev, lay, 1.5, backend='jax_raw')
        with pytest.raises(AssertionError, match='delay_ptr'):
            csr_delay_deliver_layout_p_call(
                random_ring(), ev, lay.data, lay.indices, lay.indptr, lay.delays, lay.delay_ptr[:, :-1], 0,
                shape=lay.shape, backend='jax_raw')

    def test_benchmark_data(self):
        configs = _benchmark_data(platform=PLATFORM)
        assert len(configs) == 4
        for cfg in configs:
            out = csr_delay_deliver_layout_p_call(*cfg.args, **cfg.kernel_kwargs, backend='jax_raw')
            assert out[0].shape == cfg.args[0].shape


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('dtype', ['float16', 'bfloat16', 'float64'])
def test_other_float_dtypes(backend, dtype):
    if dtype == 'float64':
        with jax_x64_enabled():
            _check_float_dtype(backend, dtype)
    else:
        _check_float_dtype(backend, dtype)


def _check_float_dtype(backend, dtype):
    if backend == 'numba' and dtype in ('float16', 'bfloat16'):
        pytest.skip('numba supports float32/float64 rings')
    data, idx, ipt, dl, _ = build()
    lay = CSRDelayLayout.build(data.astype(dtype), idx, ipt, dl, n_slots=L, shape=(N_PRE, N_POST))
    ev = make_events(N_PRE, 'float32')
    ring = random_ring(dtype=dtype)
    for weighted in (False, True):
        out = csr_delay_deliver_layout(ring, jnp.asarray(ev).astype(dtype) if weighted else jnp.asarray(ev),
                                       lay, 2, weighted=weighted, backend=backend)
        assert out.dtype == jnp.dtype(dtype)
        ref = reference(np.asarray(ring, np.float64), ev, np.asarray(data.astype(dtype), np.float64), idx, ipt, dl, 2,
                        weighted=weighted)
        tol = 1e-1 if dtype in ('float16', 'bfloat16') else 1e-10
        np.testing.assert_allclose(np.asarray(out, np.float64), ref, rtol=tol, atol=tol)


# ---------------------------------------------------------------------------
# Autodiff and vmap
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('weighted', [False, True])
def test_grads_match_plain_primitive(backend, homo, weighted):
    data, idx, ipt, dl, lay = build(homo=homo)
    ev = jnp.asarray(make_events(N_PRE, 'float32'))
    ring = random_ring()
    g = jnp.asarray(np.random.default_rng(7).normal(size=ring.shape), jnp.float32)
    perm = np.asarray(lay.perm)

    def f_layout(r, w, x):
        lay_w = CSRDelayLayout(w, lay.indices, lay.indptr, lay.delays, lay.delay_ptr, lay.perm, shape=lay.shape)
        return jnp.sum(g * csr_delay_deliver_layout(r, x, lay_w, 5, weighted=weighted, backend=backend))

    def f_plain(r, w, x):
        return jnp.sum(g * csr_delay_deliver(r, x, w, idx, ipt, dl, 5, shape=(N_PRE, N_POST),
                                             weighted=weighted, backend=backend))

    gl = jax.grad(f_layout, argnums=(0, 1, 2))(ring, lay.data, ev)
    gp = jax.grad(f_plain, argnums=(0, 1, 2))(ring, data, ev)
    np.testing.assert_allclose(np.asarray(gl[0]), np.asarray(gp[0]), rtol=1e-5, atol=1e-5)
    gp_data = np.asarray(gp[1]) if homo else np.asarray(gp[1])[perm]
    np.testing.assert_allclose(np.asarray(gl[1]), gp_data, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(gl[2]), np.asarray(gp[2]), rtol=1e-4, atol=1e-4)

    tangents = (jnp.ones_like(ring), jnp.ones_like(lay.data), jnp.ones_like(ev))
    _, jl = jax.jvp(f_layout, (ring, lay.data, ev), tangents)
    _, jp = jax.jvp(f_plain, (ring, data, ev), (jnp.ones_like(ring), jnp.ones_like(data), jnp.ones_like(ev)))
    np.testing.assert_allclose(float(jl), float(jp), rtol=1e-4, atol=1e-4)


def test_non_differentiable_delay_ptr():
    _, _, _, _, lay = build()
    ev = jnp.asarray(make_events(N_PRE, 'bool'))

    def f(dptr):
        return csr_delay_deliver_layout_p_call(
            random_ring(), ev, lay.data, lay.indices, lay.indptr, lay.delays, dptr, 0,
            shape=lay.shape, backend='jax_raw')[0].sum()

    with pytest.raises(Exception):
        jax.grad(f)(lay.delay_ptr.astype(jnp.float32))


@pytest.mark.parametrize('backend', BACKENDS)
def test_vmap(backend):
    data, idx, ipt, dl, lay = build()
    evs = jnp.asarray(np.stack([make_events(N_PRE, 'bool', seed=s) for s in range(3)]))
    rings = jnp.stack([random_ring(seed=s) for s in range(3)])
    out = jax.vmap(lambda r, e: csr_delay_deliver_layout(r, e, lay, 2, backend=backend))(rings, evs)
    for b in range(3):
        np.testing.assert_allclose(np.asarray(out[b]),
                                   reference(rings[b], np.asarray(evs[b]), data, idx, ipt, dl, 2),
                                   rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('backend', BACKENDS)
def test_scan_matches_plain(backend):
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

    a = jax.jit(lambda: run(lambda r, s, t: csr_delay_deliver_layout(r, s, lay, t, backend=backend)))()
    b = jax.jit(lambda: run(lambda r, s, t: csr_delay_deliver(r, s, data, idx, ipt, dl, t, shape=(N_PRE, N_POST),
                                                              backend=backend)))()
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-5)


@requires_gpu
def test_gpu_large_matches_plain():
    # Enough rows and delays that the persistent grid loops over many items.
    data, idx, ipt, dl, lay = build(n_pre=3000, n_post=2000, max_conn=200, n_slots=60, seed=11)
    ev = make_events(3000, 'bool')
    ring = random_ring(n_slots=60, n_post=2000)
    out = csr_delay_deliver_layout(ring, jnp.asarray(ev), lay, 17, backend='cuda_raw')
    plain = csr_delay_deliver(ring, jnp.asarray(ev), data, idx, ipt, dl, 17, shape=(3000, 2000), backend='jax_raw')
    np.testing.assert_allclose(np.asarray(out), np.asarray(plain), rtol=1e-4, atol=1e-4)
