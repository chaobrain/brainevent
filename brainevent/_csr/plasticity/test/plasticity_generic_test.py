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

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
from brainevent._csr.plasticity.plasticity_generic import csr_apply_on_post, csr_apply_on_pre

N_PRE, N_POST = 9, 7
SHAPE = (N_PRE, N_POST)
PARAMS = {'lr': 0.05, 'w_max': 1.5, 'tau': 0.8}


def _problem(seed=0):
    rng = np.random.default_rng(seed)
    dense = rng.uniform(0.2, 1.2, SHAPE) * (rng.random(SHAPE) < 0.5)
    csr = brainevent.CSR.fromdense(jnp.asarray(dense, dtype=jnp.float32))
    rows = np.repeat(np.arange(N_PRE), np.diff(np.asarray(csr.indptr)))
    cols = np.asarray(csr.indices)
    state = {
        'w': csr.data,
        'elig': jnp.asarray(rng.uniform(-0.5, 0.5, csr.nse), dtype=jnp.float32),
    }
    pre_spike = jnp.asarray(np.resize([2., 0., -1., 0., 0.5, 1.], N_PRE), dtype=jnp.float32)
    post_spike = jnp.asarray(np.resize([0., 1., -2., 0., 0., 3.], N_POST), dtype=jnp.float32)
    pre_vals = {'x': jnp.asarray(rng.uniform(0, 1, N_PRE), dtype=jnp.float32)}
    post_vals = {'y': jnp.asarray(rng.uniform(0, 1, N_POST), dtype=jnp.float32)}
    return csr, rows, cols, state, pre_spike, post_spike, pre_vals, post_vals


def stdp_rule(edge, pre, post, p):
    """Weight-dependent (multiplicative) STDP with an eligibility trace."""
    elig = p['tau'] * edge['elig'] + post['y'] - pre['x']
    w = edge['w'] + p['lr'] * (p['w_max'] - edge['w']) * elig
    return {'w': w, 'elig': elig}


def _numpy_loop(state, rows, cols, active_rows, active_cols, pre_vals, post_vals, p, on_pre):
    w = np.asarray(state['w'], dtype=np.float64).copy()
    elig = np.asarray(state['elig'], dtype=np.float64).copy()
    x = np.asarray(pre_vals['x'], dtype=np.float64)
    y = np.asarray(post_vals['y'], dtype=np.float64)
    for e in range(w.shape[0]):
        i, j = rows[e], cols[e]
        if (active_rows[i] if on_pre else active_cols[j]):
            new_elig = p['tau'] * elig[e] + y[j] - x[i]
            w[e] = w[e] + p['lr'] * (p['w_max'] - w[e]) * new_elig
            elig[e] = new_elig
    return w, elig


@pytest.mark.parametrize('on_pre', [True, False])
def test_stdp_eligibility_matches_numpy_loop(on_pre):
    csr, rows, cols, state, pre_spike, post_spike, pre_vals, post_vals = _problem(0)
    fn = csr_apply_on_pre if on_pre else csr_apply_on_post
    spike = pre_spike if on_pre else post_spike
    new = fn(stdp_rule, state, csr.indices, csr.indptr, spike,
             pre_vals=pre_vals, post_vals=post_vals, params=PARAMS, shape=SHAPE)
    w, elig = _numpy_loop(state, rows, cols, np.asarray(pre_spike) != 0, np.asarray(post_spike) != 0,
                          pre_vals, post_vals, PARAMS, on_pre)
    np.testing.assert_allclose(np.asarray(new['w']), w, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(new['elig']), elig, rtol=1e-5, atol=1e-6)
    assert new['w'].dtype == state['w'].dtype


@pytest.mark.parametrize('on_pre', [True, False])
def test_inactive_edges_untouched(on_pre):
    csr, rows, cols, state, pre_spike, post_spike, pre_vals, post_vals = _problem(1)
    fn = csr_apply_on_pre if on_pre else csr_apply_on_post
    spike = pre_spike if on_pre else post_spike
    # A rule producing nan everywhere: inactive edges must keep their exact old values.
    rule = lambda e, pre, post, p: jax.tree_util.tree_map(lambda v: v * jnp.nan, e)
    new = fn(rule, state, csr.indices, csr.indptr, spike, shape=SHAPE)
    active = (np.asarray(pre_spike) != 0)[rows] if on_pre else (np.asarray(post_spike) != 0)[cols]
    for k in state:
        assert np.array_equal(np.asarray(new[k])[~active], np.asarray(state[k])[~active])
        assert np.all(np.isnan(np.asarray(new[k])[active]))


def test_bool_spikes_and_all_inactive():
    csr, rows, cols, state, *_ = _problem(2)
    rule = lambda e, pre, post, p: e + 1.
    none = csr_apply_on_pre(rule, state['w'], csr.indices, csr.indptr, jnp.zeros(N_PRE, bool), shape=SHAPE)
    assert np.array_equal(np.asarray(none), np.asarray(state['w']))
    allp = csr_apply_on_post(rule, state['w'], csr.indices, csr.indptr, jnp.ones(N_POST, bool), shape=SHAPE)
    np.testing.assert_allclose(np.asarray(allp), np.asarray(state['w']) + 1.)


def test_trailing_dims_and_tuple_state():
    csr, rows, cols, state, pre_spike, *_ = _problem(3)
    nse = csr.nse
    syn = (jnp.arange(nse * 3, dtype=jnp.float32).reshape(nse, 3), jnp.ones(nse, jnp.int32))
    rule = lambda e, pre, post, p: (e[0] * 2., e[1] + 1)
    new = csr_apply_on_pre(rule, syn, csr.indices, csr.indptr, pre_spike, shape=SHAPE)
    active = (np.asarray(pre_spike) != 0)[rows]
    exp0 = np.where(active[:, None], np.asarray(syn[0]) * 2., np.asarray(syn[0]))
    np.testing.assert_allclose(np.asarray(new[0]), exp0)
    assert new[1].dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(new[1]), np.where(active, 2, 1))


def _dense_reference(state, rows, cols, active, pre_vals, post_vals, p):
    pre_e = jax.tree_util.tree_map(lambda v: v[rows], pre_vals)
    post_e = jax.tree_util.tree_map(lambda v: v[cols], post_vals)
    new = jax.vmap(stdp_rule, in_axes=(0, 0, 0, None))(state, pre_e, post_e, p)
    return jax.tree_util.tree_map(lambda n, o: jnp.where(active, n, o), new, state)


@pytest.mark.parametrize('on_pre', [True, False])
def test_grads_through_params_state_and_neuron_values(on_pre):
    csr, rows, cols, state, pre_spike, post_spike, pre_vals, post_vals = _problem(4)
    fn = csr_apply_on_pre if on_pre else csr_apply_on_post
    spike = pre_spike if on_pre else post_spike
    active = (np.asarray(pre_spike) != 0)[rows] if on_pre else (np.asarray(post_spike) != 0)[cols]
    params = jax.tree_util.tree_map(jnp.float32, PARAMS)

    def loss(st, pv, qv, p):
        new = fn(stdp_rule, st, csr.indices, csr.indptr, spike,
                 pre_vals=pv, post_vals=qv, params=p, shape=SHAPE)
        return jnp.sum(new['w'] ** 2) + jnp.sum(jnp.sin(new['elig']))

    def loss_ref(st, pv, qv, p):
        new = _dense_reference(st, rows, cols, active, pv, qv, p)
        return jnp.sum(new['w'] ** 2) + jnp.sum(jnp.sin(new['elig']))

    args = (state, pre_vals, post_vals, params)
    g = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3)))(*args)
    g_ref = jax.grad(loss_ref, argnums=(0, 1, 2, 3))(*args)
    for a, b in zip(jax.tree_util.tree_leaves(g), jax.tree_util.tree_leaves(g_ref)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-5)
    # Inactive edges pass the cotangent through unchanged: d loss / d w_old = 2 w_old.
    np.testing.assert_allclose(np.asarray(g[0]['w'])[~active], 2 * np.asarray(state['w'])[~active], rtol=1e-6)
    # The learning-rate gradient is non-trivial.
    assert float(jnp.abs(g[3]['lr'])) > 0


def test_vmap_over_state_spikes_and_params():
    csr, rows, cols, state, pre_spike, post_spike, pre_vals, post_vals = _problem(5)
    states = jax.tree_util.tree_map(lambda v: jnp.stack([v, v * 0.5]), state)
    spikes = jnp.stack([pre_spike, jnp.flip(pre_spike)])
    lrs = jnp.array([0.05, 0.2], dtype=jnp.float32)

    def run(st, sp, lr):
        p = dict(PARAMS, lr=lr)
        return csr_apply_on_pre(stdp_rule, st, csr.indices, csr.indptr, sp,
                                pre_vals=pre_vals, post_vals=post_vals, params=p, shape=SHAPE)

    got = jax.jit(jax.vmap(run))(states, spikes, lrs)
    for b in range(2):
        st = jax.tree_util.tree_map(lambda v: v[b], states)
        exp = run(st, spikes[b], lrs[b])
        for k in state:
            np.testing.assert_allclose(np.asarray(got[k][b]), np.asarray(exp[k]), rtol=1e-6, atol=1e-6)


def test_scan_carry_keeps_dtype():
    csr, rows, cols, state, pre_spike, *_ = _problem(6)
    rule = lambda w, pre, post, p: w + p  # p is float64-like python scalar -> weak type

    def body(w, _):
        return csr_apply_on_pre(rule, w, csr.indices, csr.indptr, pre_spike, params=0.1, shape=SHAPE), None

    out, _ = jax.lax.scan(body, state['w'], None, length=3)
    active = (np.asarray(pre_spike) != 0)[rows]
    np.testing.assert_allclose(np.asarray(out), np.asarray(state['w']) + 0.3 * active, rtol=1e-5)
    assert out.dtype == state['w'].dtype


def test_units_preserved():
    csr, rows, cols, state, pre_spike, *_ = _problem(7)
    w = state['w'] * u.mV
    post = jnp.ones(N_POST) * u.volt
    rule = lambda e, pre, q, p: e + 1e-3 * q  # result in volt, stored back in mV
    new = csr_apply_on_pre(rule, w, csr.indices, csr.indptr, pre_spike, post_vals=post, shape=SHAPE)
    assert isinstance(new, u.Quantity) and new.unit == u.mV
    active = (np.asarray(pre_spike) != 0)[rows]
    np.testing.assert_allclose(np.asarray(new.mantissa), np.asarray(state['w']) + active * 1.0, rtol=1e-5)
    bad = lambda e, pre, q, p: e.mantissa
    with pytest.raises(Exception):
        csr_apply_on_pre(bad, w, csr.indices, csr.indptr, pre_spike, shape=SHAPE)


def test_errors():
    csr, rows, cols, state, pre_spike, post_spike, pre_vals, post_vals = _problem(8)
    ok = lambda e, pre, post, p: e
    with pytest.raises(TypeError, match='same structure'):
        csr_apply_on_pre(lambda e, pre, post, p: e['w'], state, csr.indices, csr.indptr, pre_spike, shape=SHAPE)
    with pytest.raises(ValueError, match='edge_state'):
        csr_apply_on_pre(ok, state['w'][:-1], csr.indices, csr.indptr, pre_spike, shape=SHAPE)
    with pytest.raises(ValueError, match='edge_state'):
        csr_apply_on_pre(ok, jnp.float32(1.), csr.indices, csr.indptr, pre_spike, shape=SHAPE)
    with pytest.raises(ValueError, match='pre_vals'):
        csr_apply_on_pre(ok, state, csr.indices, csr.indptr, pre_spike, pre_vals=post_vals, shape=SHAPE)
    with pytest.raises(ValueError, match='post_vals'):
        csr_apply_on_post(ok, state, csr.indices, csr.indptr, post_spike, post_vals=pre_vals, shape=SHAPE)
    with pytest.raises(ValueError, match='pre_spike'):
        csr_apply_on_pre(ok, state, csr.indices, csr.indptr, post_spike, shape=SHAPE)
    with pytest.raises(ValueError, match='post_spike'):
        csr_apply_on_post(ok, state, csr.indices, csr.indptr, pre_spike, shape=SHAPE)
    with pytest.raises(ValueError, match='indptr'):
        csr_apply_on_pre(ok, state, csr.indices, csr.indptr, pre_spike, shape=(N_PRE - 1, N_POST))


def test_csr_methods():
    csr, rows, cols, state, pre_spike, post_spike, pre_vals, post_vals = _problem(9)
    rule = lambda w, pre, post, p: w + p * post
    new_w = csr.apply_on_pre(rule, pre_spike, post_vals=post_vals['y'], params=0.1)
    exp = csr_apply_on_pre(rule, csr.data, csr.indices, csr.indptr, pre_spike,
                           post_vals=post_vals['y'], params=0.1, shape=SHAPE)
    np.testing.assert_allclose(np.asarray(new_w), np.asarray(exp))
    new = csr.apply_on_post(stdp_rule, post_spike, edge_state=state,
                            pre_vals=pre_vals, post_vals=post_vals, params=PARAMS)
    exp = csr_apply_on_post(stdp_rule, state, csr.indices, csr.indptr, post_spike,
                            pre_vals=pre_vals, post_vals=post_vals, params=PARAMS, shape=SHAPE)
    for k in state:
        np.testing.assert_allclose(np.asarray(new[k]), np.asarray(exp[k]))
    # gradient through the method w.r.t. the matrix data
    dep = lambda w, pre, post, p: w - p * pre
    g = jax.grad(lambda d: csr.with_data(d).apply_on_post(dep, post_spike, pre_vals=pre_vals['x'], params=0.1).sum())(csr.data)
    np.testing.assert_allclose(np.asarray(g), np.ones(csr.nse))

    homo = brainevent.CSR((jnp.ones(1), csr.indices, csr.indptr), shape=SHAPE)
    with pytest.raises(ValueError, match='homogeneous'):
        homo.apply_on_pre(rule, pre_spike)
    # explicit edge_state bypasses the homogeneous guard
    out = homo.apply_on_pre(rule, pre_spike, edge_state=state['w'], post_vals=post_vals['y'], params=0.0)
    np.testing.assert_allclose(np.asarray(out), np.asarray(state['w']))


def test_top_level_exports():
    assert brainevent.csr_apply_on_pre is csr_apply_on_pre
    assert brainevent.csr_apply_on_post is csr_apply_on_post
    assert 'csr_apply_on_pre' in brainevent.__all__ and 'csr_apply_on_post' in brainevent.__all__


@pytest.mark.parametrize('on_pre', [True, False])
def test_nan_neuron_values_on_inactive_edges_do_not_leak(on_pre):
    # Inactive edges see NaN neuron values: they stay bit-identical and the
    # output is finite. (Gradients through ``rule`` follow the documented
    # ``jnp.where`` semantics and are not covered.)
    csr, rows, cols, state, pre_spike, post_spike, pre_vals, post_vals = _problem(4)
    fn = csr_apply_on_pre if on_pre else csr_apply_on_post
    spike = pre_spike if on_pre else post_spike
    active_units = np.asarray(spike) != 0
    if on_pre:
        pre_vals = {'x': jnp.where(jnp.asarray(active_units), pre_vals['x'], jnp.nan)}
        active = active_units[rows]
    else:
        post_vals = {'y': jnp.where(jnp.asarray(active_units), post_vals['y'], jnp.nan)}
        active = active_units[cols]

    def run(params):
        return fn(stdp_rule, state, csr.indices, csr.indptr, spike,
                  pre_vals=pre_vals, post_vals=post_vals, params=params, shape=SHAPE)

    new = run(PARAMS)
    for k in state:
        assert np.array_equal(np.asarray(new[k])[~active], np.asarray(state[k])[~active])
        assert np.isfinite(np.asarray(new[k])).all()
