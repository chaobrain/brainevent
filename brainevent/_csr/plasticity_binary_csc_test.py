# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

import jax.numpy as jnp
import numpy as np
import pytest

import brainevent as be


def _dense(rng, n_pre, n_post, p=0.5):
    mask = rng.random((n_pre, n_post)) < p
    vals = rng.random((n_pre, n_post)) + 0.5  # in [0.5, 1.5)
    return jnp.asarray(mask * vals, dtype=jnp.float32)


def _ref_pre(W, pre_spike, post_trace, w_min, w_max):
    W = np.asarray(W)
    mask = (W != 0.0)
    active = (np.asarray(pre_spike) != 0).astype(W.dtype)        # (n_pre,)
    delta = mask * active[:, None] * np.asarray(post_trace)[None, :]
    return np.clip(W + delta, w_min, w_max)


def _ref_post(W, pre_trace, post_spike, w_min, w_max):
    W = np.asarray(W)
    mask = (W != 0.0)
    active = (np.asarray(post_spike) != 0).astype(W.dtype)       # (n_post,)
    delta = mask * np.asarray(pre_trace)[:, None] * active[None, :]
    return np.clip(W + delta, w_min, w_max)


@pytest.mark.parametrize("spike_dtype", [jnp.bool_, jnp.float32])
def test_update_csc_on_binary_pre_matches_dense(spike_dtype):
    rng = np.random.default_rng(0)
    n_pre, n_post = 4, 6
    w_min, w_max = 0.0, 1.2
    W = _dense(rng, n_pre, n_post)
    csc = be.CSC.fromdense(W)
    pre_spike = jnp.asarray(rng.random(n_pre) > 0.5, dtype=spike_dtype)
    post_trace = jnp.asarray(rng.random(n_post), dtype=jnp.float32)
    new_w = be.update_csc_on_binary_pre(
        csc.data, csc.indices, csc.indptr, pre_spike, post_trace,
        w_min, w_max, shape=csc.shape,
    )
    got = csc.with_data(new_w).todense()
    ref = _ref_pre(W, pre_spike, post_trace, w_min, w_max)
    assert jnp.allclose(got, jnp.asarray(ref), atol=1e-5)


@pytest.mark.parametrize("spike_dtype", [jnp.bool_, jnp.float32])
def test_update_csc_on_binary_post_matches_dense(spike_dtype):
    rng = np.random.default_rng(1)
    n_pre, n_post = 5, 3
    w_min, w_max = 0.0, 1.2
    W = _dense(rng, n_pre, n_post)
    csc = be.CSC.fromdense(W)
    pre_trace = jnp.asarray(rng.random(n_pre), dtype=jnp.float32)
    post_spike = jnp.asarray(rng.random(n_post) > 0.5, dtype=spike_dtype)
    new_w = be.update_csc_on_binary_post(
        csc.data, csc.indices, csc.indptr, pre_trace, post_spike,
        w_min, w_max, shape=csc.shape,
    )
    got = csc.with_data(new_w).todense()
    ref = _ref_post(W, pre_trace, post_spike, w_min, w_max)
    assert jnp.allclose(got, jnp.asarray(ref), atol=1e-5)


def test_update_csc_on_binary_pre_no_clip():
    # Without bounds the rule is a pure additive update at stored positions.
    rng = np.random.default_rng(2)
    n_pre, n_post = 3, 4
    W = _dense(rng, n_pre, n_post)
    csc = be.CSC.fromdense(W)
    pre_spike = jnp.asarray([True, False, True])
    post_trace = jnp.asarray(rng.random(n_post), dtype=jnp.float32)
    new_w = be.update_csc_on_binary_pre(
        csc.data, csc.indices, csc.indptr, pre_spike, post_trace, shape=csc.shape,
    )
    got = csc.with_data(new_w).todense()
    ref = _ref_pre(W, pre_spike, post_trace, None, None)
    assert jnp.allclose(got, jnp.asarray(ref), atol=1e-5)
