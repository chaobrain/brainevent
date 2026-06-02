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

import brainunit as bu
import brainevent as be
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._fcn.plasticity_binary import (
    fcn_plasticity_row_p,
    fcn_plasticity_row_prim_call,
    update_fixed_post_conn_on_binary_pre,
    update_fixed_pre_conn_on_binary_post,
)

PLATFORM = jax.default_backend()
ROW_BACKENDS = tuple(fcn_plasticity_row_p.available_backends(PLATFORM))

try:
    _CPU_DEVICE = jax.devices('cpu')[0]
except Exception:  # pragma: no cover - cpu backend always present in practice
    _CPU_DEVICE = None


def _ell_ref_row(data, indices, row_spike, col_trace):
    data = np.asarray(data, dtype=np.float64)
    idx = np.asarray(indices)
    active = (np.asarray(row_spike) != 0)
    ct = np.asarray(col_trace, dtype=np.float64)
    out = data.copy()
    for r in range(data.shape[0]):
        if active[r]:
            for k in range(data.shape[1]):
                out[r, k] += ct[idx[r, k]]
    return out


def _ell_ref_col(data, indices, col_spike, row_trace):
    """Per-synapse unfavorable reference: ``data[r, k] += row_trace[r]`` when
    ``col_spike[indices[r, k]]`` is active.  Operates on the stored ELL ``data``
    directly (no ``todense``), so it is valid even with duplicate columns."""
    data = np.asarray(data, dtype=np.float64)
    idx = np.asarray(indices)
    active = (np.asarray(col_spike) != 0)
    rt = np.asarray(row_trace, dtype=np.float64)
    out = data.copy()
    for r in range(data.shape[0]):
        for k in range(data.shape[1]):
            if active[idx[r, k]]:
                out[r, k] += rt[r]
    return out


@pytest.mark.parametrize("backend", ROW_BACKENDS)
@pytest.mark.parametrize("spike_dtype", [jnp.bool_, jnp.float32])
def test_row_prim(backend, spike_dtype):
    rng = np.random.default_rng(0)
    n_row, n_conn, n_col = 5, 3, 7
    data = jnp.asarray(rng.random((n_row, n_conn)), dtype=jnp.float32)
    indices = jnp.asarray(rng.integers(0, n_col, (n_row, n_conn)), dtype=jnp.int32)
    spike = jnp.asarray(rng.random(n_row) > 0.5, dtype=spike_dtype)
    trace = jnp.asarray(rng.random(n_col), dtype=jnp.float32)
    got = fcn_plasticity_row_prim_call(data, indices, spike, trace, backend=backend)[0]
    ref = _ell_ref_row(data, indices, spike, trace)
    assert np.allclose(np.asarray(got), ref, atol=1e-5)


@pytest.mark.slow
@pytest.mark.skipif(_CPU_DEVICE is None, reason="no CPU device for numba backend")
@pytest.mark.parametrize("spike_dtype", [jnp.bool_, jnp.float32])
def test_row_numba_matches_ref(spike_dtype):
    rng = np.random.default_rng(2)
    n_row, n_conn, n_col = 8, 5, 9
    with jax.default_device(_CPU_DEVICE):
        data = jnp.asarray(rng.random((n_row, n_conn)), dtype=jnp.float32)
        indices = jnp.asarray(rng.integers(0, n_col, (n_row, n_conn)), dtype=jnp.int32)
        spike = jnp.asarray(rng.random(n_row) > 0.5, dtype=spike_dtype)
        trace = jnp.asarray(rng.random(n_col), dtype=jnp.float32)
        got = np.asarray(
            fcn_plasticity_row_prim_call(data, indices, spike, trace, backend='numba')[0]
        )
    ref = _ell_ref_row(data, indices, spike, trace)
    assert np.allclose(got, ref, atol=1e-5)


# --------------------------------------------------------------------------- #
# Module / class plasticity: favorable (native row) + unfavorable (CSR-fused)
# --------------------------------------------------------------------------- #

_W_MIN_MAX = [(None, None), (0.0, 1.5), (0.2, None), (None, 0.9)]


def _clip(x, lo, hi):
    return np.clip(x, lo, hi)


@pytest.mark.parametrize("spike_dtype", [jnp.bool_, jnp.float32])
@pytest.mark.parametrize("w_min,w_max", _W_MIN_MAX)
def test_post_conn_on_pre(spike_dtype, w_min, w_max):
    """FixedNumPerPre favorable (row-driven) pre-spike update."""
    rng = np.random.default_rng(10)
    n_pre, n_conn, n_post = 5, 3, 7
    data = jnp.asarray(rng.random((n_pre, n_conn)) + 0.3, dtype=jnp.float32)
    indices = jnp.asarray(rng.integers(0, n_post, (n_pre, n_conn)), dtype=jnp.int32)
    pre_spike = jnp.asarray(rng.random(n_pre) > 0.5, dtype=spike_dtype)
    post_trace = jnp.asarray(rng.random(n_post), dtype=jnp.float32)
    got = update_fixed_post_conn_on_binary_pre(
        data, indices, pre_spike, post_trace, w_min, w_max, shape=(n_pre, n_post))
    ref = _clip(_ell_ref_row(data, indices, pre_spike, post_trace), w_min, w_max)
    assert np.allclose(np.asarray(got), ref, atol=1e-5)


@pytest.mark.parametrize("spike_dtype", [jnp.bool_, jnp.float32])
@pytest.mark.parametrize("w_min,w_max", _W_MIN_MAX)
def test_fixed_per_pre_on_post_unfavorable(spike_dtype, w_min, w_max):
    """FixedNumPerPre.update_on_post is the unfavorable (perm-fused CSR) direction."""
    rng = np.random.default_rng(11)
    n_pre, n_conn, n_post = 6, 4, 8
    data = jnp.asarray(rng.random((n_pre, n_conn)) + 0.3, dtype=jnp.float32)
    indices = jnp.asarray(rng.integers(0, n_post, (n_pre, n_conn)), dtype=jnp.int32)
    pre_trace = jnp.asarray(rng.random(n_pre), dtype=jnp.float32)
    post_spike = jnp.asarray(rng.random(n_post) > 0.5, dtype=spike_dtype)
    m = be.FixedNumPerPre(data, indices, shape=(n_pre, n_post))
    got = m.update_on_post(pre_trace, post_spike, w_min=w_min, w_max=w_max).data
    ref = _clip(_ell_ref_col(data, indices, post_spike, pre_trace), w_min, w_max)
    assert np.allclose(np.asarray(got), ref, atol=1e-5)


@pytest.mark.parametrize("spike_dtype", [jnp.bool_, jnp.float32])
@pytest.mark.parametrize("w_min,w_max", _W_MIN_MAX)
def test_pre_conn_on_post(spike_dtype, w_min, w_max):
    """FixedNumPerPost favorable (row-driven) post-spike update."""
    rng = np.random.default_rng(12)
    n_pre, n_conn, n_post = 7, 3, 6
    data = jnp.asarray(rng.random((n_post, n_conn)) + 0.3, dtype=jnp.float32)
    indices = jnp.asarray(rng.integers(0, n_pre, (n_post, n_conn)), dtype=jnp.int32)
    pre_trace = jnp.asarray(rng.random(n_pre), dtype=jnp.float32)
    post_spike = jnp.asarray(rng.random(n_post) > 0.5, dtype=spike_dtype)
    got = update_fixed_pre_conn_on_binary_post(
        data, indices, pre_trace, post_spike, w_min, w_max, shape=(n_pre, n_post))
    ref = _clip(_ell_ref_row(data, indices, post_spike, pre_trace), w_min, w_max)
    assert np.allclose(np.asarray(got), ref, atol=1e-5)


@pytest.mark.parametrize("spike_dtype", [jnp.bool_, jnp.float32])
@pytest.mark.parametrize("w_min,w_max", _W_MIN_MAX)
def test_fixed_per_post_on_pre_unfavorable(spike_dtype, w_min, w_max):
    """FixedNumPerPost.update_on_pre is the unfavorable (perm-fused CSR) direction."""
    rng = np.random.default_rng(13)
    n_pre, n_conn, n_post = 8, 4, 5
    data = jnp.asarray(rng.random((n_post, n_conn)) + 0.3, dtype=jnp.float32)
    indices = jnp.asarray(rng.integers(0, n_pre, (n_post, n_conn)), dtype=jnp.int32)
    pre_spike = jnp.asarray(rng.random(n_pre) > 0.5, dtype=spike_dtype)
    post_trace = jnp.asarray(rng.random(n_post), dtype=jnp.float32)
    m = be.FixedNumPerPost(data, indices, shape=(n_pre, n_post))
    got = m.update_on_pre(pre_spike, post_trace, w_min=w_min, w_max=w_max).data
    ref = _clip(_ell_ref_col(data, indices, pre_spike, post_trace), w_min, w_max)
    assert np.allclose(np.asarray(got), ref, atol=1e-5)


def test_duplicate_indices_accumulate():
    data = jnp.array([[1.0, 2.0]], dtype=jnp.float32)
    indices = jnp.array([[0, 0]], dtype=jnp.int32)  # duplicate post id 0 in the row
    pre_spike = jnp.array([True])
    post_trace = jnp.array([0.5, 9.9], dtype=jnp.float32)
    got = update_fixed_post_conn_on_binary_pre(data, indices, pre_spike, post_trace, shape=(1, 2))
    # both stored synapses point at post 0 -> each gets +0.5 independently
    assert np.allclose(np.asarray(got), np.array([[1.5, 2.5]]), atol=1e-6)


def test_units_preserved():
    rng = np.random.default_rng(20)
    n_pre, n_conn, n_post = 4, 2, 5
    data = jnp.asarray(rng.random((n_pre, n_conn)) + 0.3, dtype=jnp.float32) * bu.siemens
    indices = jnp.asarray(rng.integers(0, n_post, (n_pre, n_conn)), dtype=jnp.int32)
    pre_spike = jnp.asarray(rng.random(n_pre) > 0.5)
    post_trace = jnp.asarray(rng.random(n_post), dtype=jnp.float32) * bu.siemens
    got = update_fixed_post_conn_on_binary_pre(
        data, indices, pre_spike, post_trace, shape=(n_pre, n_post))
    assert bu.get_unit(got) == bu.siemens
    ref = _ell_ref_row(bu.get_mantissa(data), indices, pre_spike, bu.get_mantissa(post_trace))
    assert np.allclose(np.asarray(bu.get_mantissa(got)), ref, atol=1e-5)


def test_homogeneous_data_raises():
    data = jnp.asarray([1.5], dtype=jnp.float32)  # homogeneous
    indices = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)
    pre_spike = jnp.array([True, False])
    post_trace = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)
    with pytest.raises(ValueError, match="per-synapse"):
        update_fixed_post_conn_on_binary_pre(data, indices, pre_spike, post_trace, shape=(2, 3))


# --------------------------------------------------------------------------- #
# Class methods, transpose duality, jit, exports
# --------------------------------------------------------------------------- #

def test_class_methods_match_reference_and_preserve_structure():
    rng = np.random.default_rng(30)
    n_pre, n_conn, n_post = 5, 3, 7
    data = jnp.asarray(rng.random((n_pre, n_conn)) + 0.3, dtype=jnp.float32)
    indices = jnp.asarray(rng.integers(0, n_post, (n_pre, n_conn)), dtype=jnp.int32)
    pre_spike = jnp.asarray(rng.random(n_pre) > 0.5)
    post_trace = jnp.asarray(rng.random(n_post), dtype=jnp.float32)
    pre_trace = jnp.asarray(rng.random(n_pre), dtype=jnp.float32)
    post_spike = jnp.asarray(rng.random(n_post) > 0.5)

    m = be.FixedNumPerPre(data, indices, shape=(n_pre, n_post))
    # favorable: matches the native row module function
    m2 = m.update_on_pre(pre_spike, post_trace, w_min=0.0, w_max=1.2)
    assert isinstance(m2, be.FixedNumPerPre)
    assert np.array_equal(np.asarray(m2.indices), np.asarray(indices))  # structure preserved
    mod = update_fixed_post_conn_on_binary_pre(
        data, indices, pre_spike, post_trace, 0.0, 1.2, shape=(n_pre, n_post))
    assert np.allclose(np.asarray(m2.data), np.asarray(mod), atol=1e-6)

    # unfavorable: matches the per-synapse column reference + clip
    m3 = m.update_on_post(pre_trace, post_spike, w_min=0.0, w_max=1.2)
    ref3 = _clip(_ell_ref_col(data, indices, post_spike, pre_trace), 0.0, 1.2)
    assert np.allclose(np.asarray(m3.data), ref3, atol=1e-5)


def test_transpose_duality():
    rng = np.random.default_rng(31)
    n_pre, n_conn, n_post = 6, 3, 5
    data = jnp.asarray(rng.random((n_pre, n_conn)) + 0.3, dtype=jnp.float32)
    indices = jnp.asarray(rng.integers(0, n_post, (n_pre, n_conn)), dtype=jnp.int32)
    pre_spike = jnp.asarray(rng.random(n_pre) > 0.5)
    post_trace = jnp.asarray(rng.random(n_post), dtype=jnp.float32)

    m = be.FixedNumPerPre(data, indices, shape=(n_pre, n_post))
    a = m.update_on_pre(pre_spike, post_trace)
    # transpose -> FixedNumPerPost sharing the same arrays; on_post is its favorable dir
    mt = m.transpose()
    b = mt.update_on_post(pre_trace=post_trace, post_spike=pre_spike)
    assert np.allclose(np.asarray(a.data), np.asarray(b.data), atol=1e-6)


def test_jit_class_method():
    rng = np.random.default_rng(32)
    n_pre, n_conn, n_post = 5, 3, 7
    data = jnp.asarray(rng.random((n_pre, n_conn)) + 0.3, dtype=jnp.float32)
    indices = jnp.asarray(rng.integers(0, n_post, (n_pre, n_conn)), dtype=jnp.int32)
    m = be.FixedNumPerPre(data, indices, shape=(n_pre, n_post))
    pre_spike = jnp.asarray(rng.random(n_pre) > 0.5)
    post_trace = jnp.asarray(rng.random(n_post), dtype=jnp.float32)

    @jax.jit
    def f(mat, s, t):
        return mat.update_on_pre(s, t).data

    got = f(m, pre_spike, post_trace)
    ref = m.update_on_pre(pre_spike, post_trace).data
    assert np.allclose(np.asarray(got), np.asarray(ref), atol=1e-6)


def test_top_level_exports():
    for name in [
        'update_fixed_post_conn_on_binary_pre',
        'update_fixed_pre_conn_on_binary_post',
        'fcn_plasticity_row_p',
    ]:
        assert hasattr(be, name), f'missing export: {name}'
    # The redundant native unfavorable operators were removed (breaking change).
    for name in [
        'update_fixed_post_conn_on_binary_post',
        'update_fixed_pre_conn_on_binary_pre',
        'fcn_plasticity_col_p',
        'csc_binary_matvec',
        'csc_binary_matmat',
    ]:
        assert not hasattr(be, name), f'unexpected lingering export: {name}'


# --------------------------------------------------------------------------- #
# Favorable-direction CUDA kernel (registration always; runtime only on GPU)
# --------------------------------------------------------------------------- #

def test_cuda_registered_and_import_ok():
    assert 'cuda_raw' in fcn_plasticity_row_p.available_backends('gpu')


@pytest.mark.skipif(
    not any(d.platform == 'gpu' for d in jax.devices()),
    reason="no GPU device available",
)
@pytest.mark.parametrize("spike_dtype", [jnp.bool_, jnp.float32])
def test_row_cuda_matches_jax(spike_dtype):
    rng = np.random.default_rng(40)
    n_row, n_conn, n_col = 64, 16, 50
    data = jnp.asarray(rng.random((n_row, n_conn)), dtype=jnp.float32)
    indices = jnp.asarray(rng.integers(0, n_col, (n_row, n_conn)), dtype=jnp.int32)
    spike = jnp.asarray(rng.random(n_row) > 0.5, dtype=spike_dtype)
    trace = jnp.asarray(rng.random(n_col), dtype=jnp.float32)
    a = fcn_plasticity_row_prim_call(data, indices, spike, trace, backend='cuda_raw')[0]
    b = fcn_plasticity_row_prim_call(data, indices, spike, trace, backend='jax_raw')[0]
    assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-5)
