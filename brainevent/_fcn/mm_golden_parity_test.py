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

"""Golden parity pinning FixedPreNumConn/FixedPostNumConn matmat behavior.

Captured BEFORE the unfavorable-direction scatter refactor. Assertions are
against the dense reference ``M.todense()``.
"""

import brainunit as u
import jax.numpy as jnp
import numpy as np

from brainevent import FixedPreNumConn, FixedPostNumConn, BinaryArray


def _cases():
    rng = np.random.default_rng(0)
    n_pre, n_post, n_conn = 6, 5, 3
    for cls in (FixedPostNumConn, FixedPreNumConn):
        rows = n_pre if cls is FixedPostNumConn else n_post
        upper = n_post if cls is FixedPostNumConn else n_pre
        for homo in (True, False):
            indices = rng.integers(0, upper, size=(rows, n_conn)).astype(np.int32)
            if homo:
                data = jnp.asarray(rng.random(1) + 0.5, dtype=jnp.float32)
            else:
                data = jnp.asarray(rng.random((rows, n_conn)) + 0.5, dtype=jnp.float32)
            yield cls, data, jnp.asarray(indices), (n_pre, n_post)


def test_fcn_matmat_golden():
    rng = np.random.default_rng(7)
    for cls, data, indices, shape in _cases():
        M = cls(data, indices, shape=shape)
        dense = jnp.asarray(M.todense(), dtype=jnp.float32)
        n_pre, n_post = shape
        for ev in (jnp.bool_, jnp.float32):
            for n in (1, 4):
                right = jnp.asarray(rng.random((n_post, n)) > 0.5, dtype=ev)   # W @ M
                got_r = M @ BinaryArray(right)
                ref_r = dense @ jnp.asarray(right, dtype=jnp.float32)
                assert jnp.allclose(got_r, ref_r, atol=1e-5), (cls.__name__, str(ev), n, 'right')

                left = jnp.asarray(rng.random((n, n_pre)) > 0.5, dtype=ev)     # M @ W
                got_l = BinaryArray(left) @ M
                ref_l = jnp.asarray(left, dtype=jnp.float32) @ dense
                assert jnp.allclose(got_l, ref_l, atol=1e-5), (cls.__name__, str(ev), n, 'left')


def test_fcn_matmat_unfavorable_builds_weight_indices():
    # Unfavorable matmat must build the cached CSC view (scatter path), exactly
    # as the matvec unfavorable path does. For FixedPostNumConn (axis==0),
    # _ell_plan gives ell_transpose == transpose_W, so __matmul__ (W @ M,
    # transpose_W=False) is the *unfavorable* direction.
    rng = np.random.default_rng(5)
    cls, data, indices, shape = next(_cases())   # FixedPostNumConn, homo
    M = cls(data, indices, shape=shape)
    n_pre, n_post = shape
    assert M._csc is None                          # FCN caches its CSC view in self._csc
    right = jnp.asarray(rng.random((n_post, 3)) > 0.5, dtype=jnp.bool_)
    _ = M @ BinaryArray(right)
    assert M._csc is not None


def test_fcn_matmat_units():
    rng = np.random.default_rng(10)
    cls, data, indices, shape = next(_cases())
    M = cls(data * u.mV, indices, shape=shape)
    dense = jnp.asarray(cls(data, indices, shape=shape).todense(), jnp.float32)
    n_pre, n_post = shape
    right = jnp.asarray(rng.random((n_post, 4)) > 0.5, dtype=jnp.bool_)
    got = M @ BinaryArray(right)
    assert u.get_unit(got) == u.mV
    ref = dense @ jnp.asarray(right, jnp.float32)
    assert jnp.allclose(u.get_mantissa(got), ref, atol=1e-5)
