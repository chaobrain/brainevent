# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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


import contextlib

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent import update_dense_on_binary_pre, update_dense_on_binary_post
from brainevent._dense.plasticity_binary import (
    update_dense_on_binary_post_p,
    update_dense_on_binary_pre_p,
)
from brainevent._test_util import jax_x64_enabled

PLATFORM = jax.default_backend()
PRE_BACKENDS = tuple(update_dense_on_binary_pre_p.available_backends(PLATFORM))
POST_BACKENDS = tuple(update_dense_on_binary_post_p.available_backends(PLATFORM))
EVENT_DTYPES = (
    'bool', 'int8', 'uint8', 'int16', 'int32',
    'float16', 'bfloat16', 'float32', 'float64',
)
_EVENT_PATTERN = np.array([2., -1., 0., 0.5, 0., 1., -3., 0.])


def _make_events(n, dtype_name):
    raw = np.resize(_EVENT_PATTERN, n)
    if dtype_name == 'bool':
        arr = raw != 0
    elif dtype_name.startswith(('int', 'uint')):
        arr = raw.astype(np.int64).astype(dtype_name)
    else:
        arr = raw
    return jnp.asarray(arr, dtype=jnp.dtype(dtype_name))


def _x64_if_needed(dtype_name):
    return jax_x64_enabled() if dtype_name == 'float64' else contextlib.nullcontext()


@pytest.mark.parametrize('backend', PRE_BACKENDS)
@pytest.mark.parametrize('dtype_name', EVENT_DTYPES)
def test_dense_on_pre_event_activity_all_dtypes(backend, dtype_name):
    """``pre_spike`` is active iff ``!= 0``; its value is never multiplied in."""
    n_pre, n_post = 8, 5
    rng = np.random.default_rng(0)
    w = rng.standard_normal((n_pre, n_post)).astype(np.float32)
    trace = rng.standard_normal(n_post).astype(np.float32)
    with _x64_if_needed(dtype_name):
        spikes = _make_events(n_pre, dtype_name)
        out = np.asarray(update_dense_on_binary_pre(jnp.asarray(w), spikes, jnp.asarray(trace), backend=backend))
    active = (np.asarray(spikes) != 0).astype(np.float32)
    np.testing.assert_allclose(out, w + np.outer(active, trace), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('backend', POST_BACKENDS)
@pytest.mark.parametrize('dtype_name', EVENT_DTYPES)
def test_dense_on_post_event_activity_all_dtypes(backend, dtype_name):
    n_pre, n_post = 5, 8
    rng = np.random.default_rng(1)
    w = rng.standard_normal((n_pre, n_post)).astype(np.float32)
    trace = rng.standard_normal(n_pre).astype(np.float32)
    with _x64_if_needed(dtype_name):
        spikes = _make_events(n_post, dtype_name)
        out = np.asarray(update_dense_on_binary_post(jnp.asarray(w), jnp.asarray(trace), spikes, backend=backend))
    active = (np.asarray(spikes) != 0).astype(np.float32)
    np.testing.assert_allclose(out, w + np.outer(trace, active), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('backend', PRE_BACKENDS)
def test_dense_plasticity_vmap_over_weights_uses_activity(backend):
    rng = np.random.default_rng(2)
    w = jnp.asarray(rng.standard_normal((3, 4, 4)), jnp.float32)
    trace = jnp.asarray(rng.standard_normal(4), jnp.float32)
    spikes = jnp.asarray([2., -1., 0., 0.5], jnp.float32)
    active = np.asarray([1., 1., 0., 1.], np.float32)
    pre = jax.vmap(lambda m: update_dense_on_binary_pre(m, spikes, trace, backend=backend))(w)
    post = jax.vmap(lambda m: update_dense_on_binary_post(m, trace, spikes, backend=backend))(w)
    np.testing.assert_allclose(np.asarray(pre), np.asarray(w) + np.outer(active, trace)[None], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(post), np.asarray(w) + np.outer(trace, active)[None], rtol=1e-6)


class Test_dense_on_pre:
    def test_dense_on_pre(self):
        n_pre = 100
        n_post = 100
        mat = brainstate.random.random((n_pre, n_post))

        pre_spike = brainstate.random.random((n_pre,)) < 0.1
        post_trace = brainstate.random.random((n_post,))

        mat2 = update_dense_on_binary_pre(mat, pre_spike, post_trace)

        mat = mat + jnp.outer(pre_spike, post_trace)
        assert jnp.allclose(mat2, mat)
        jax.block_until_ready((mat, pre_spike, post_trace, mat2))

    @pytest.mark.parametrize('mat_unit', [u.mV, u.ms])
    @pytest.mark.parametrize('trace_unit', [u.mV, u.ms])
    def test_with_unit(self, mat_unit, trace_unit):
        def run():
            n_pre = 100
            n_post = 100
            mat = brainstate.random.random((n_pre, n_post))
            mask = mat < 0.5
            mat = jnp.where(mask, mat, 0.) * mat_unit
            pre_spike = brainstate.random.random((n_pre,)) < 0.1
            post_trace = brainstate.random.random((n_post,)) * trace_unit

            mat2 = update_dense_on_binary_pre(mat, pre_spike, post_trace)

            mat = mat + u.math.outer(pre_spike, post_trace)
            assert u.math.allclose(mat2, mat)
            jax.block_until_ready((mat, pre_spike, post_trace, mat2))

        if mat_unit.has_same_dim(trace_unit):
            run()
        else:
            with pytest.raises(u.UnitMismatchError):
                run()

    @pytest.mark.parametrize('w_min', [None, 0.1])
    @pytest.mark.parametrize('w_max', [None, 0.5])
    def test_min_max(self, w_min, w_max):
        n_pre = 100
        n_post = 100
        mat = brainstate.random.random((n_pre, n_post))

        pre_spike = brainstate.random.random((n_pre,)) < 0.1
        post_trace = brainstate.random.random((n_post,))

        mat2 = update_dense_on_binary_pre(mat, pre_spike, post_trace, w_min=w_min, w_max=w_max)

        mat = mat + jnp.outer(pre_spike, post_trace)
        if w_min is not None:
            mat = jnp.maximum(mat, w_min)
        if w_max is not None:
            mat = jnp.minimum(mat, w_max)

        assert jnp.allclose(mat2, mat)
        jax.block_until_ready((mat, pre_spike, post_trace, mat2))


class Test_dense_on_post:
    def test_dense_on_post(self):
        n_pre = 100
        n_post = 100
        mat = brainstate.random.random((n_pre, n_post))

        pre_trace = brainstate.random.random((n_pre,))
        post_spike = brainstate.random.random((n_post,)) < 0.1

        mat2 = update_dense_on_binary_post(mat, pre_trace, post_spike)

        mat = mat + jnp.outer(pre_trace, post_spike)
        assert jnp.allclose(mat2, mat)
        jax.block_until_ready((mat, pre_trace, post_spike, mat2))

    @pytest.mark.parametrize('mat_unit', [u.mV, u.ms])
    @pytest.mark.parametrize('trace_unit', [u.mV, u.ms])
    def test_with_unit(self, mat_unit, trace_unit):
        def run():
            n_pre = 100
            n_post = 100
            mat = brainstate.random.random((n_pre, n_post))
            mask = mat < 0.5
            mat = jnp.where(mask, mat, 0.) * mat_unit
            pre_trace = brainstate.random.random((n_pre,)) * trace_unit
            post_spike = brainstate.random.random((n_post,)) < 0.1

            mat2 = update_dense_on_binary_post(mat, pre_trace, post_spike)

            mat = mat + u.math.outer(pre_trace, post_spike)
            assert u.math.allclose(mat2, mat)
            jax.block_until_ready((mat, pre_trace, post_spike, mat2))

        if mat_unit.has_same_dim(trace_unit):
            run()
        else:
            with pytest.raises(u.UnitMismatchError):
                run()

    @pytest.mark.parametrize('w_min', [None, 0.1])
    @pytest.mark.parametrize('w_max', [None, 0.5])
    def test_min_max(self, w_min, w_max):
        n_pre = 100
        n_post = 100
        mat = brainstate.random.random((n_pre, n_post))
        pre_trace = brainstate.random.random((n_pre,))
        post_spike = brainstate.random.random((n_post,)) < 0.1
        mat2 = update_dense_on_binary_post(mat, pre_trace, post_spike, w_min=w_min, w_max=w_max)
        mat = mat + jnp.outer(pre_trace, post_spike)
        if w_min is not None:
            mat = jnp.maximum(mat, w_min)
        if w_max is not None:
            mat = jnp.minimum(mat, w_max)
        assert jnp.allclose(mat2, mat)
        jax.block_until_ready((mat, pre_trace, post_spike, mat2))
