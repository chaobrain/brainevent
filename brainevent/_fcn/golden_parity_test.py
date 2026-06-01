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

"""Golden parity tests pinning FixedPreNumConn/FixedPostNumConn matvec behavior.

Captured BEFORE the dispatch simplification (Phase 9) so the refactor can be
checked for behavioral drift. All assertions are against the dense reference
``M.todense()``, making them independent of internal layout machinery.
"""

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
