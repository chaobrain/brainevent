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

"""Golden parity tests pinning FixedNumPerPost/FixedNumPerPre matvec behavior.

Captured BEFORE the dispatch simplification (Phase 9) so the refactor can be
checked for behavioral drift. All assertions are against the dense reference
``M.todense()``, making them independent of internal layout machinery.
"""

import jax
import jax.numpy as jnp
import numpy as np

from brainevent import FixedNumPerPost, FixedNumPerPre, BinaryArray


def _cases():
    rng = np.random.default_rng(0)
    n_pre, n_post, n_conn = 6, 5, 3
    for cls in (FixedNumPerPre, FixedNumPerPost):
        rows = n_pre if cls is FixedNumPerPre else n_post
        upper = n_post if cls is FixedNumPerPre else n_pre
        for homo in (True, False):
            indices = rng.integers(0, upper, size=(rows, n_conn)).astype(np.int32)
            if homo:
                data = jnp.asarray(rng.random(1) + 0.5, dtype=jnp.float32)
            else:
                data = jnp.asarray(rng.random((rows, n_conn)) + 0.5, dtype=jnp.float32)
            yield cls, data, jnp.asarray(indices), (n_pre, n_post)


def test_fcn_matvec_golden():
    rng = np.random.default_rng(7)
    for cls, data, indices, shape in _cases():
        M = cls(data, indices, shape=shape)
        dense = jnp.asarray(M.todense(), dtype=jnp.float32)
        n_pre, n_post = shape
        for ev_dtype in (jnp.bool_, jnp.float32):
            left = jnp.asarray(rng.random(n_pre) > 0.5, dtype=ev_dtype)
            right = jnp.asarray(rng.random(n_post) > 0.5, dtype=ev_dtype)

            got_l = BinaryArray(left) @ M
            ref_l = jnp.asarray(left, dtype=jnp.float32) @ dense
            assert jnp.allclose(got_l, ref_l, atol=1e-5), (cls.__name__, str(ev_dtype), 'left')

            got_r = M @ BinaryArray(right)
            ref_r = dense @ jnp.asarray(right, dtype=jnp.float32)
            assert jnp.allclose(got_r, ref_r, atol=1e-5), (cls.__name__, str(ev_dtype), 'right')


def _hetero_cases():
    rng = np.random.default_rng(11)
    n_pre, n_post, n_conn = 6, 5, 3
    for cls in (FixedNumPerPre, FixedNumPerPost):
        rows = n_pre if cls is FixedNumPerPre else n_post
        upper = n_post if cls is FixedNumPerPre else n_pre
        for homo in (True, False):
            indices = rng.integers(0, upper, size=(rows, n_conn)).astype(np.int32)
            data = (jnp.asarray(rng.random(1) + 0.5, dtype=jnp.float32) if homo
                    else jnp.asarray(rng.random((rows, n_conn)) + 0.5, dtype=jnp.float32))
            yield cls, jnp.asarray(data), jnp.asarray(indices), (n_pre, n_post)


def _distinct_indices(rng, rows, upper, n_conn):
    """Indices with distinct columns per row (so todense() has one slot per (i, j))."""
    return jnp.asarray(
        np.stack([rng.choice(upper, size=n_conn, replace=False) for _ in range(rows)]).astype(np.int32)
    )


def test_fcn_unfavorable_matmat_golden():
    rng = np.random.default_rng(3)
    k = 4
    for cls, data, indices, shape in _hetero_cases():
        M = cls(data, indices, shape=shape)
        dense = jnp.asarray(M.todense(), dtype=jnp.float32)
        n_pre, n_post = shape
        for ev_dtype in (jnp.bool_, jnp.float32):
            X = jnp.asarray(rng.random((n_post, k)) > 0.5, dtype=ev_dtype)
            got = M @ BinaryArray(X)
            ref = dense @ jnp.asarray(X, dtype=jnp.float32)
            assert jnp.allclose(got, ref, atol=1e-5), (cls.__name__, 'M@X', str(ev_dtype))

            Xl = jnp.asarray(rng.random((k, n_pre)) > 0.5, dtype=ev_dtype)
            got_l = BinaryArray(Xl) @ M
            ref_l = jnp.asarray(Xl, dtype=jnp.float32) @ dense
            assert jnp.allclose(got_l, ref_l, atol=1e-5), (cls.__name__, 'X@M', str(ev_dtype))


def test_fcn_matvec_grad_golden():
    rng = np.random.default_rng(5)
    for cls, data, indices, shape in _hetero_cases():
        if data.size == 1:
            continue  # gradient wrt per-synapse (heterogeneous) weights
        n_pre, n_post = shape
        ev = jnp.asarray(rng.random(n_post) > 0.5, dtype=jnp.float32)

        def f(d, _indices=indices):
            return (cls(d, _indices, shape=shape) @ BinaryArray(ev)).sum()

        def f_dense(d, _indices=indices):
            M = cls(d, _indices, shape=shape)
            return (jnp.asarray(M.todense(), dtype=jnp.float32)
                    @ jnp.asarray(ev, dtype=jnp.float32)).sum()

        g = jax.grad(f)(data)
        g_ref = jax.grad(f_dense)(data)
        assert jnp.allclose(g, g_ref, atol=1e-4), (cls.__name__, 'grad')


def test_fcn_plasticity_unfavorable_golden():
    rng = np.random.default_rng(7)
    n_pre, n_post, n_conn = 6, 5, 3
    for cls in (FixedNumPerPre, FixedNumPerPost):
        rows = n_pre if cls is FixedNumPerPre else n_post
        upper = n_post if cls is FixedNumPerPre else n_pre
        indices = _distinct_indices(rng, rows, upper, n_conn)
        data = jnp.asarray(rng.random((rows, n_conn)) + 0.5, dtype=jnp.float32)
        M = cls(data, indices, shape=(n_pre, n_post))
        dense = jnp.asarray(M.todense(), dtype=jnp.float32)
        mask = (dense != 0).astype(jnp.float32)
        pre_trace = jnp.asarray(rng.random(n_pre), dtype=jnp.float32)
        post_trace = jnp.asarray(rng.random(n_post), dtype=jnp.float32)
        pre_spike = jnp.asarray(rng.random(n_pre) > 0.5)
        post_spike = jnp.asarray(rng.random(n_post) > 0.5)
        for w_min, w_max in ((None, None), (0.0, 1.0)):
            up = M.update_on_post(pre_trace, post_spike, w_min=w_min, w_max=w_max)
            ref = dense + (pre_trace[:, None] * jnp.asarray(post_spike, jnp.float32)[None, :]) * mask
            if w_min is not None:
                ref = jnp.clip(ref, w_min, w_max) * mask
            assert jnp.allclose(jnp.asarray(up.todense(), jnp.float32), ref, atol=1e-5), (cls.__name__, 'on_post', w_min)

            up2 = M.update_on_pre(pre_spike, post_trace, w_min=w_min, w_max=w_max)
            ref2 = dense + (jnp.asarray(pre_spike, jnp.float32)[:, None] * post_trace[None, :]) * mask
            if w_min is not None:
                ref2 = jnp.clip(ref2, w_min, w_max) * mask
            assert jnp.allclose(jnp.asarray(up2.todense(), jnp.float32), ref2, atol=1e-5), (cls.__name__, 'on_pre', w_min)
