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

"""Tests for ``CSRDelayLayout``: build, sort order, raw constructor and data helpers."""

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent import CSR, CSRDelayLayout, csr_delay_deliver_layout
from brainevent._csr.delay import layout as layout_mod
from brainevent._test_util import jax_x64_enabled
from ._reference import (
    L, MAX_CONN, N_POST, N_PRE, build, make_csr, make_events, random_ring, reference, x64_disabled,
)


def _check_layout(lay, data, idx, ipt, dl, n_slots):
    """``lay`` holds the edges of every row sorted by (delay, column), ties in original order."""
    perm = np.asarray(lay.perm)
    assert sorted(perm.tolist()) == list(range(idx.size))
    np.testing.assert_array_equal(np.asarray(lay.indices), idx[perm])
    np.testing.assert_array_equal(np.asarray(lay.delays), dl[perm])
    if np.size(data) == idx.size and idx.size > 1:
        np.testing.assert_array_equal(np.asarray(lay.data), np.asarray(data)[perm])
    np.testing.assert_array_equal(np.asarray(lay.indptr), ipt)
    dptr = np.asarray(lay.delay_ptr)
    lay_dl, lay_idx = np.asarray(lay.delays), np.asarray(lay.indices)
    for i in range(len(ipt) - 1):
        s, e = int(ipt[i]), int(ipt[i + 1])
        assert np.all((perm[s:e] >= s) & (perm[s:e] < e))  # edges stay in their row
        key = lay_dl[s:e].astype(np.int64) * (int(lay_idx.max(initial=0)) + 1) + lay_idx[s:e]
        assert np.all(np.diff(key) >= 0)
        tie = np.diff(key) == 0
        assert np.all(np.diff(perm[s:e])[tie] > 0)  # stable
        for d in range(n_slots):
            assert dptr[i, d] == s + np.count_nonzero(dl[s:e] < d)


class TestBuild:
    @pytest.mark.parametrize('indptr_dtype', [np.int32, np.int64])
    def test_order_and_delay_ptr(self, indptr_dtype):
        with jax_x64_enabled():
            data, idx, ipt, dl, lay = build(indptr_dtype=indptr_dtype)
            _check_layout(lay, data, idx, ipt, dl, L)
            assert lay.delay_ptr.dtype == lay.indptr.dtype == np.dtype(indptr_dtype)
            assert lay.indices.dtype == lay.delays.dtype == lay.perm.dtype == np.int32
            assert lay.delay_ptr.shape == (N_PRE, L) and lay.n_slots == L and lay.nse == idx.size

    def test_columns_order_equal_delays(self):
        # row 0: columns 2, 0, 1, 0 with delays 1, 1, 0, 1 -> (0, 1), (1, 0), (1, 0), (1, 2)
        lay = CSRDelayLayout.build(jnp.array([1., 2., 3., 4.]), np.array([2, 0, 1, 0], np.int32),
                                   np.array([0, 4], np.int32), np.array([1, 1, 0, 1], np.int32),
                                   n_slots=2, shape=(1, 3))
        np.testing.assert_array_equal(lay.perm, [2, 1, 3, 0])
        np.testing.assert_array_equal(lay.indices, [1, 0, 0, 2])
        np.testing.assert_array_equal(lay.delays, [0, 1, 1, 1])
        np.testing.assert_array_equal(lay.data, [3., 2., 4., 1.])
        np.testing.assert_array_equal(lay.delay_ptr, [[0, 1]])

    def test_sorted_input_has_the_identity_perm(self):
        # brainspike stores its edges by (row, delay, column) and checks this invariant
        w, idx, ipt, dl = make_csr(N_PRE, N_POST, MAX_CONN, L, seed=4)
        for i in range(N_PRE):
            s, e = ipt[i], ipt[i + 1]
            order = np.lexsort((idx[s:e], dl[s:e]))
            idx[s:e], dl[s:e] = idx[s:e][order], dl[s:e][order]
        lay = CSRDelayLayout.build(np.zeros((1,), np.float32), idx, ipt, dl, n_slots=L, shape=(N_PRE, N_POST))
        np.testing.assert_array_equal(lay.perm, np.arange(idx.size))

    def test_data_round_trip(self):
        data, idx, ipt, dl, lay = build()
        np.testing.assert_array_equal(np.asarray(lay.original_data()), np.asarray(data))
        new = jnp.arange(idx.size, dtype=jnp.float32)
        lay2 = lay.with_data(new)
        np.testing.assert_array_equal(np.asarray(lay2.original_data()), np.asarray(new))
        np.testing.assert_array_equal(np.asarray(lay2.data), np.asarray(new)[np.asarray(lay.perm)])
        assert lay2.indices is lay.indices and lay2.delay_ptr is lay.delay_ptr and lay2.shape == lay.shape
        with pytest.raises(ValueError, match='data must be'):
            lay.with_data(jnp.ones(idx.size + 1))

    def test_homogeneous_and_scalar_inputs(self):
        _, idx, ipt, _ = make_csr(N_PRE, N_POST, MAX_CONN, L)
        lay = CSRDelayLayout.build(0.5, idx, ipt, 3, n_slots=L, shape=(N_PRE, N_POST))
        assert lay.data.shape == (1,)
        np.testing.assert_array_equal(np.asarray(lay.delays), np.full(idx.size, 3))
        assert lay.with_data(jnp.float32(2.0)).data.shape == (1,)
        assert lay.original_data().shape == (1,)
        one = CSRDelayLayout.build(0.5, idx, ipt, np.array([2]), n_slots=L, shape=(N_PRE, N_POST))
        np.testing.assert_array_equal(np.asarray(one.delays), np.full(idx.size, 2))

    def test_zero_delays_and_one_slot(self):
        _, idx, ipt, _ = make_csr(N_PRE, N_POST, MAX_CONN, L)
        lay = CSRDelayLayout.build(1.0, idx, ipt, 0, n_slots=1, shape=(N_PRE, N_POST))
        assert lay.n_slots == 1
        np.testing.assert_array_equal(np.asarray(lay.delay_ptr)[:, 0], ipt[:-1])

    @pytest.mark.parametrize('n_slots', [np.int64(L), np.int32(L), jnp.int32(L), np.array(L)])
    def test_integer_like_n_slots(self, n_slots):
        data, idx, ipt, dl, ref = build()
        lay = CSRDelayLayout.build(data, idx, ipt, dl, n_slots=n_slots, shape=(np.int64(N_PRE), N_POST))
        assert lay.n_slots == L and type(lay.shape[0]) is int
        np.testing.assert_array_equal(lay.delay_ptr, ref.delay_ptr)

    def test_units(self):
        data, idx, ipt, dl, _ = build()
        lay = CSRDelayLayout.build(data * u.mV, idx, ipt, dl, n_slots=L, shape=(N_PRE, N_POST))
        assert u.get_unit(lay.data) == u.mV
        restored = lay.original_data()
        assert u.get_unit(restored) == u.mV
        np.testing.assert_allclose(u.get_mantissa(restored), np.asarray(data))
        assert u.get_unit(lay.with_data(data * u.volt).data) == u.volt

    def test_traced_data_is_allowed(self):
        _, idx, ipt, dl, lay = build()

        @jax.jit
        def f(w):
            return CSRDelayLayout.build(w, idx, ipt, dl, n_slots=L, shape=(N_PRE, N_POST)).data

        w = jnp.arange(idx.size, dtype=jnp.float32)
        np.testing.assert_array_equal(np.asarray(f(w)), np.asarray(w)[np.asarray(lay.perm)])

    def test_errors(self):
        data, idx, ipt, dl, _ = build()
        kw = dict(n_slots=L, shape=(N_PRE, N_POST))
        with pytest.raises(ValueError, match='0 <= d < L'):
            CSRDelayLayout.build(data, idx, ipt, np.where(dl == dl[0], -1, dl), **kw)
        with pytest.raises(ValueError, match='0 <= d < L'):
            CSRDelayLayout.build(data, idx, ipt, dl + L, **kw)
        with pytest.raises(ValueError, match='0 <= d < L'):
            CSRDelayLayout.build(data, idx, ipt, L, **kw)
        with pytest.raises(TypeError, match='integer'):
            CSRDelayLayout.build(data, idx, ipt, dl.astype(np.float32), **kw)
        with pytest.raises(TypeError, match='integer'):
            CSRDelayLayout.build(data, idx, ipt, dl.astype(bool), **kw)
        with pytest.raises(TypeError, match='integer'):
            CSRDelayLayout.build(data, idx.astype(np.float32), ipt, dl, **kw)
        with pytest.raises(ValueError, match='n_slots'):
            CSRDelayLayout.build(data, idx, ipt, 0, n_slots=0, shape=(N_PRE, N_POST))
        with pytest.raises(ValueError, match='n_slots'):
            CSRDelayLayout.build(data, idx, ipt, 0, n_slots=2 ** 31, shape=(N_PRE, N_POST))
        with pytest.raises(TypeError, match='n_slots'):
            CSRDelayLayout.build(data, idx, ipt, dl, n_slots=7.0, shape=(N_PRE, N_POST))
        with pytest.raises(ValueError, match='indptr'):
            CSRDelayLayout.build(data, idx, ipt[:-1], dl, **kw)
        with pytest.raises(ValueError, match='nse'):
            CSRDelayLayout.build(data, idx[:-1], ipt, dl[:-1], **kw)
        with pytest.raises(ValueError, match='delays must have shape'):
            CSRDelayLayout.build(data, idx, ipt, dl[:-1], **kw)
        with pytest.raises(ValueError, match='data must be'):
            CSRDelayLayout.build(data[:-1], idx, ipt, dl, **kw)
        with pytest.raises(ValueError, match='shape'):
            CSRDelayLayout.build(data, idx, ipt, dl, n_slots=L, shape=(N_PRE,))
        with pytest.raises(ValueError, match='non-negative'):
            CSRDelayLayout.build(data, idx, ipt, dl, n_slots=L, shape=(N_PRE, -1))
        with pytest.raises(ValueError, match='column indices'):
            CSRDelayLayout.build(data, np.where(idx == idx[0], N_POST, idx), ipt, dl, **kw)
        with pytest.raises(ValueError, match='column indices'):
            CSRDelayLayout.build(data, np.where(idx == idx[0], -1, idx), ipt, dl, **kw)

        @jax.jit
        def traced(d):
            return CSRDelayLayout.build(data, idx, ipt, d, n_slots=L, shape=(N_PRE, N_POST))

        with pytest.raises(TypeError, match='concrete'):
            traced(jnp.asarray(dl))

    @pytest.mark.parametrize('indptr', [[1, 3, 3], [0, 3, 2, 3]])
    def test_malformed_indptr(self, indptr):
        indptr = np.asarray(indptr, np.int32)
        n_pre = indptr.size - 1
        with pytest.raises(ValueError, match='indptr must start at 0 and be non-decreasing'):
            CSRDelayLayout.build(jnp.ones(3), np.zeros(3, np.int32), indptr, 1, n_slots=3, shape=(n_pre, 2))

    @pytest.mark.parametrize('n_pre,n_post', [(0, 0), (0, 5), (4, 0), (4, 5)])
    def test_empty(self, n_pre, n_post):
        lay = CSRDelayLayout.build(jnp.zeros((0,)), np.zeros(0, np.int32), np.zeros(n_pre + 1, np.int32),
                                   np.zeros(0, np.int32), n_slots=3, shape=(n_pre, n_post))
        assert lay.nse == 0 and lay.delay_ptr.shape == (n_pre, 3) and lay.data.shape == (0,)
        np.testing.assert_array_equal(lay.delay_ptr, 0)
        assert lay.original_data().shape == (0,)
        assert CSRDelayLayout.build(1.0, np.zeros(0, np.int32), np.zeros(n_pre + 1, np.int32), 2,
                                    n_slots=3, shape=(n_pre, n_post)).data.shape == (1,)

    def test_pytree_and_jit(self):
        _, _, _, _, lay = build()
        leaves, tree = jax.tree_util.tree_flatten(lay)
        assert len(leaves) == 6
        back = jax.tree_util.tree_unflatten(tree, leaves)
        assert back.shape == lay.shape and back.n_slots == lay.n_slots
        ev = jnp.asarray(make_events(N_PRE, 'bool'))
        f = jax.jit(lambda r, lay: csr_delay_deliver_layout(r, ev, lay, 2, backend='jax_raw'))
        ring = random_ring()
        _, _, _, _, other = build(seed=5)
        # Layouts with different structure go through the same jitted function.
        f(ring, lay)
        f(ring, other)
        assert repr(lay) == f'CSRDelayLayout(shape={lay.shape}, nse={lay.nse}, n_slots={L})'

    def test_raw_constructor_in_a_trace_with_identity_perm(self):
        # brainspike rebuilds the layout inside its step from stored arrays, with perm=None
        data, idx, ipt, dl, lay = build()
        ev = make_events(N_PRE, 'bool')
        ring = random_ring()
        assert len(jax.tree_util.tree_leaves(lay.with_data(lay.data))) == 6

        @jax.jit
        def step(w, r, x):
            raw = CSRDelayLayout(w.astype(r.dtype), lay.indices, lay.indptr, lay.delays, lay.delay_ptr, None,
                                 shape=lay.shape)
            assert raw.perm is None and len(jax.tree_util.tree_leaves(raw)) == 5
            assert raw.with_data(w).data is not None and raw.original_data() is raw.data
            return csr_delay_deliver_layout(r, x, raw, 3, weighted=False)

        out = step(lay.data, ring, jnp.asarray(ev))
        np.testing.assert_allclose(np.asarray(out), reference(ring, ev, data, idx, ipt, dl, 3), rtol=1e-5, atol=1e-5)
        raw = CSRDelayLayout(lay.data, lay.indices, lay.indptr, lay.delays, lay.delay_ptr, shape=lay.shape)
        np.testing.assert_array_equal(raw.with_data(lay.data).data, lay.data)

    def test_csr_method(self):
        data, idx, ipt, dl, lay = build()
        csr = CSR((data, jnp.asarray(idx), jnp.asarray(ipt)), shape=(N_PRE, N_POST))
        lay2 = csr.delay_layout(dl, n_slots=L)
        assert type(lay2) is CSRDelayLayout
        for name in ('data', 'indices', 'indptr', 'delays', 'delay_ptr', 'perm'):
            np.testing.assert_array_equal(np.asarray(getattr(lay2, name)), np.asarray(getattr(lay, name)))
        np.testing.assert_array_equal(np.asarray(csr.indices), idx)  # the CSR is unchanged


# ---------------------------------------------------------------------------
# The chunked build against a one-lexsort reference
# ---------------------------------------------------------------------------

def _reference_build(data, indices, indptr, delays, *, n_slots, shape, int32_perm=2 ** 31):
    """``CSRDelayLayout.build`` as one lexsort over all edges; the six layout arrays in
    :meth:`CSRDelayLayout.tree_flatten` order."""
    n_pre = int(shape[0])
    indices_np = np.asarray(indices)
    indptr_np = np.asarray(indptr)
    nse = indices_np.shape[0]
    delays_np = np.asarray(delays)
    if delays_np.size == 1 and delays_np.shape != (nse,):
        delays_np = np.broadcast_to(delays_np.reshape(()), (nse,))
    row_ids = np.repeat(np.arange(n_pre, dtype=np.int64), np.diff(indptr_np).astype(np.int64))
    perm = np.lexsort((indices_np, delays_np, row_ids))
    keys = row_ids * n_slots + delays_np[perm].astype(np.int64)
    delay_ptr = np.zeros(n_pre * n_slots, dtype=np.int64)
    np.cumsum(np.bincount(keys, minlength=n_pre * n_slots)[:-1], out=delay_ptr[1:])
    ptr_dtype = np.int64 if (indptr_np.dtype == np.int64 or nse >= int32_perm) else np.int32
    perm_dtype = np.int32 if nse < int32_perm else np.int64
    data_arr = data if isinstance(data, u.Quantity) else jnp.asarray(data)
    if np.ndim(data_arr) == 0:
        data_arr = data_arr.reshape(1)
    perm_j = jnp.asarray(perm.astype(perm_dtype))
    if np.shape(data_arr)[0] == nse and nse != 1:
        data_arr = data_arr[perm_j]
    return (
        data_arr,
        jnp.asarray(indices_np[perm].astype(np.int32)),
        jnp.asarray(indptr_np.astype(ptr_dtype)),
        jnp.asarray(delays_np[perm].astype(np.int32)),
        jnp.asarray(delay_ptr.reshape(n_pre, n_slots).astype(ptr_dtype)),
        perm_j,
    )


def _random_csr(rng, n_pre, n_post, max_conn, n_slots, indptr_dtype, delay_dtype, *, long_row=0,
                sorted_columns=False):
    """Random CSR: empty rows, repeated targets in a row (multapses), few distinct delays
    (many ties) and optionally one row of ``long_row`` edges."""
    counts = rng.integers(0, max_conn + 1, n_pre)
    counts[rng.random(n_pre) < 0.2] = 0
    if long_row and n_pre:
        counts[rng.integers(n_pre)] = long_row
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(indptr_dtype)
    nse = int(indptr[-1])
    indices = rng.integers(0, max(1, n_post // 3), nse).astype(np.int32)  # repeats in a row
    if sorted_columns:
        for i in range(n_pre):
            indices[indptr[i]:indptr[i + 1]].sort()
    hi = min(n_slots, 4) if rng.random() < 0.5 else n_slots  # half the cases: many ties
    hi = min(hi, int(np.iinfo(delay_dtype).max) + 1)
    delays = rng.integers(0, hi, nse).astype(delay_dtype)
    weights = rng.normal(size=nse).astype(np.float32)
    return weights, indices, indptr, delays


def _assert_same_layout(lay, ref):
    got = jax.tree_util.tree_leaves(lay, is_leaf=lambda x: isinstance(x, u.Quantity))
    assert len(got) == len(ref) == 6
    for name, a, b in zip(('data', 'indices', 'indptr', 'delays', 'delay_ptr', 'perm'), got, ref):
        assert type(a) is type(b), name
        if isinstance(b, u.Quantity):
            assert u.get_unit(a) == u.get_unit(b), name
        ma, mb = u.get_mantissa(a), u.get_mantissa(b)
        assert ma.dtype == mb.dtype and ma.shape == mb.shape, (name, ma.dtype, mb.dtype, ma.shape, mb.shape)
        np.testing.assert_array_equal(np.asarray(ma), np.asarray(mb), err_msg=name)


class TestBuildMatchesReference:
    """The chunked build gives the layout of one lexsort by (row, delay, column): the
    same perm (the same stable tie order), the same arrays and the same dtypes."""

    @pytest.mark.parametrize('chunk_edges', [1 << 20, 7, 1])
    @pytest.mark.parametrize('indptr_dtype', [np.int32, np.int64])
    @pytest.mark.parametrize('delay_dtype', [np.int32, np.int64, np.uint8])
    def test_random_csrs(self, monkeypatch, chunk_edges, indptr_dtype, delay_dtype):
        # chunk_edges 7 and 1: many chunks, rows longer than a chunk, chunk ends at empty rows
        monkeypatch.setattr(layout_mod, '_BUILD_CHUNK_EDGES', chunk_edges)
        rng = np.random.default_rng(int(chunk_edges) * 7 + np.dtype(indptr_dtype).itemsize
                                    + np.dtype(delay_dtype).itemsize)
        with jax_x64_enabled():
            for case in range(12):
                n_pre = int(rng.integers(0, 60)) if case else 0
                n_post = int(rng.integers(1, 40))
                n_slots = int(rng.choice([1, 2, 3, 7, 50, 300]))
                w, idx, ipt, dl = _random_csr(rng, n_pre, n_post, int(rng.integers(0, 12)), n_slots, indptr_dtype,
                                              delay_dtype, long_row=40 if case % 3 == 0 else 0,
                                              sorted_columns=case % 2 == 0)
                kw = dict(n_slots=n_slots, shape=(n_pre, n_post))
                for data in (jnp.asarray(w), jnp.asarray(w[:1]) if w.size else jnp.float32(0.5),
                             0.25, jnp.asarray(w) * u.mV):
                    _assert_same_layout(CSRDelayLayout.build(data, idx, ipt, dl, **kw),
                                        _reference_build(data, idx, ipt, dl, **kw))
                if idx.size:  # a scalar delay
                    d = int(rng.integers(0, n_slots))
                    _assert_same_layout(CSRDelayLayout.build(jnp.asarray(w), idx, ipt, d, **kw),
                                        _reference_build(jnp.asarray(w), idx, ipt, d, **kw))

    def test_x64_disabled(self):
        # without x64, int64 indptr and delay_ptr become int32 on the device
        rng = np.random.default_rng(11)
        w, idx, ipt, dl = _random_csr(rng, 30, 20, 8, 9, np.int64, np.int64)
        kw = dict(n_slots=9, shape=(30, 20))
        with x64_disabled():
            lay = CSRDelayLayout.build(jnp.asarray(w), idx, ipt, dl, **kw)
            assert lay.indptr.dtype == lay.delay_ptr.dtype == np.int32
            _assert_same_layout(lay, _reference_build(jnp.asarray(w), idx, ipt, dl, **kw))

    @pytest.mark.parametrize('n_slots', [2 ** 16, 2 ** 16 + 1])
    def test_rings_at_the_16_bit_key_limit(self, n_slots):
        # 2**16 slots: one row per chunk, uint16 keys up to 65535; one more: int32 keys
        rng = np.random.default_rng(n_slots)
        w, idx, ipt, dl = _random_csr(rng, 5, 10, 30, n_slots, np.int32, np.int32)
        dl[:3] = n_slots - 1
        kw = dict(n_slots=n_slots, shape=(5, 10))
        _assert_same_layout(CSRDelayLayout.build(jnp.asarray(w), idx, ipt, dl, **kw),
                            _reference_build(jnp.asarray(w), idx, ipt, dl, **kw))

    @pytest.mark.parametrize('keys', ['int32', 'int64'])
    def test_wide_key_fallbacks(self, monkeypatch, keys):
        """The int32 and int64 key paths (rings longer than 2**16 and 2**31 slots), reached
        by lowering their thresholds."""
        monkeypatch.setattr(layout_mod, '_UINT16_KEYS', 1)
        if keys == 'int64':
            monkeypatch.setattr(layout_mod, '_INT32_KEYS', 1)
        seen = []
        argsort = np.argsort

        def spy(a, *args, **kwargs):
            seen.append(a.dtype)
            return argsort(a, *args, **kwargs)

        monkeypatch.setattr(layout_mod.np, 'argsort', spy)
        rng = np.random.default_rng(5)
        with jax_x64_enabled():
            for n_slots in (2, 7, 300):
                for sorted_columns in (True, False):
                    w, idx, ipt, dl = _random_csr(rng, 40, 30, 10, n_slots, np.int64, np.int64, long_row=25,
                                                  sorted_columns=sorted_columns)
                    kw = dict(n_slots=n_slots, shape=(40, 30))
                    _assert_same_layout(CSRDelayLayout.build(jnp.asarray(w), idx, ipt, dl, **kw),
                                        _reference_build(jnp.asarray(w), idx, ipt, dl, **kw))
        assert set(seen) == {np.dtype(keys)}

    def test_wide_columns(self):
        # more than 2**16 columns: the column pass sorts int32 keys
        rng = np.random.default_rng(2)
        n_post = 70_000
        w, idx, ipt, dl = _random_csr(rng, 20, 3 * n_post, 30, 5, np.int32, np.int32)
        assert idx.max() >= 2 ** 16
        kw = dict(n_slots=5, shape=(20, n_post))
        _assert_same_layout(CSRDelayLayout.build(jnp.asarray(w), idx, ipt, dl, **kw),
                            _reference_build(jnp.asarray(w), idx, ipt, dl, **kw))

    def test_int64_perm_fallback(self, monkeypatch):
        """``nse >= 2**31`` gives int64 ``perm``/offsets, reached by lowering the threshold."""
        monkeypatch.setattr(layout_mod, '_INT32_PERM', 10)
        rng = np.random.default_rng(9)
        w, idx, ipt, dl = _random_csr(rng, 20, 15, 6, 5, np.int32, np.int32)
        assert idx.size >= 10
        kw = dict(n_slots=5, shape=(20, 15))
        with x64_disabled(), pytest.raises(ValueError, match='jax_enable_x64'):
            CSRDelayLayout.build(jnp.asarray(w), idx, ipt, dl, **kw)
        with jax_x64_enabled():
            lay = CSRDelayLayout.build(jnp.asarray(w), idx, ipt, dl, **kw)
            assert lay.perm.dtype == lay.indptr.dtype == lay.delay_ptr.dtype == np.int64
            _assert_same_layout(lay, _reference_build(jnp.asarray(w), idx, ipt, dl, int32_perm=10, **kw))

    def test_chunks_cover_the_rows(self, monkeypatch):
        monkeypatch.setattr(layout_mod, '_BUILD_CHUNK_EDGES', 10)
        indptr = np.array([0, 0, 4, 30, 30, 31, 35, 50, 50], np.int64)  # a 26-edge row, empty rows
        for n_slots in (1, 2, 7, 2 ** 15, 2 ** 17):
            chunks = list(layout_mod._row_chunks(indptr, n_slots))
            assert [r0 for r0, _ in chunks] == [0] + [r1 for _, r1 in chunks[:-1]] and chunks[-1][1] == 8
            for r0, r1 in chunks:
                assert r1 - r0 <= max(1, 2 ** 16 // n_slots)
                assert r1 - r0 == 1 or indptr[r1] - indptr[r0] <= 10
        assert list(layout_mod._row_chunks(np.zeros(1, np.int64), 7)) == []


# ---------------------------------------------------------------------------
# Host arrays: no compiles, no aliasing
# ---------------------------------------------------------------------------

def test_build_of_host_arrays_compiles_nothing():
    # #251: ``jnp.asarray(ndarray)`` compiles one identity program per new shape on
    # JAX 0.12; ``jax.device_put`` compiles none.
    def make(n):
        return CSRDelayLayout.build(np.zeros((1,), np.float32), np.arange(n, dtype=np.int32) % 4,
                                    np.array([0, n // 2, n], np.int32), np.arange(n, dtype=np.int32) % 3,
                                    n_slots=3, shape=(2, 4))

    make(10)  # the data scalar's own programs, if any
    seen = []

    def on(event, duration, **kw):
        if event == '/jax/core/compile/backend_compile_duration':
            seen.append(kw.get('fun_name', '?'))

    jax.monitoring.register_event_duration_secs_listener(on)
    try:
        layouts = [make(n) for n in (13, 17, 23)]
    finally:
        jax.monitoring.unregister_event_duration_listener(on)
    assert seen == [], seen
    for n, lay in zip((13, 17, 23), layouts):
        assert lay.nse == n and lay.indices.dtype == np.int32 and lay.delays.dtype == np.int32
        assert lay.indptr.dtype == np.int32 and lay.delay_ptr.shape == (2, 3)


def _aligned(n, dtype):
    # a buffer at a 64-byte boundary: jax 0.12.0.1's CPU device_put zero-copies it (#254)
    dt = np.dtype(dtype)
    raw = np.empty(n * dt.itemsize + 64, np.uint8)
    off = (-raw.ctypes.data) % 64
    return raw[off:off + n * dt.itemsize].view(dt)


def test_build_copies_its_host_arrays():
    data = _aligned(1, np.float32)
    data[:] = 2.0
    idx = _aligned(3, np.int32)
    idx[:] = [1, 0, 2]
    ptr = _aligned(2, np.int32)
    ptr[:] = [0, 3]
    lay = CSRDelayLayout.build(data, idx, ptr, np.ones(3, np.int32), n_slots=2, shape=(1, 3))
    data[:] = 9.0
    idx[:] = 9
    ptr[:] = 7
    np.testing.assert_array_equal(np.asarray(lay.data), [2.0])
    np.testing.assert_array_equal(np.asarray(lay.indices), [0, 1, 2])
    np.testing.assert_array_equal(np.asarray(lay.indptr), [0, 3])


def test_to_device_copies_an_aligned_buffer():
    x = _aligned(64, np.float32)
    x[:] = 1.0
    y = layout_mod._to_device(x)
    x[:] = 5.0
    assert float(y[0]) == 1.0


def test_build_inside_a_trace_keeps_its_arrays_as_constants():
    idx, ptr, dl = np.array([1, 0, 2], np.int32), np.array([0, 3], np.int32), np.ones(3, np.int32)
    jaxpr = jax.make_jaxpr(lambda d: CSRDelayLayout.build(d, idx, ptr, dl, n_slots=2, shape=(1, 3)).indices)(
        jnp.zeros((1,), jnp.float32))
    assert 'device_put' not in str(jaxpr), jaxpr
