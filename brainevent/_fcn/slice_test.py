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

import os

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._csr.main import CSR
from brainevent._fcn.main import FixedNumPerPre, FixedNumPerPost


def _make_perpre(m, n, num_conn, homo=False):
    rng = np.random.default_rng(0)
    indices = np.stack(
        [rng.choice(n, size=num_conn, replace=False) for _ in range(m)]
    ).astype(np.int32)
    if homo:
        data = jnp.array(1.5, dtype=jnp.float32)
    else:
        data = jnp.asarray(rng.standard_normal((m, num_conn)).astype(np.float32))
    conn = FixedNumPerPre((data, jnp.asarray(indices)), shape=(m, n))
    return conn


def _make_perpost(m, n, num_conn, homo=False):
    rng = np.random.default_rng(1)
    indices = np.stack(
        [rng.choice(m, size=num_conn, replace=False) for _ in range(n)]
    ).astype(np.int32)
    if homo:
        data = jnp.array(2.0, dtype=jnp.float32)
    else:
        data = jnp.asarray(rng.standard_normal((n, num_conn)).astype(np.float32))
    conn = FixedNumPerPost((data, jnp.asarray(indices)), shape=(m, n))
    return conn


class TestFixedNumPerPreGetitem:

    @pytest.mark.parametrize('homo', [False, True])
    def test_single_row(self, homo):
        conn = _make_perpre(8, 12, 3, homo=homo)
        dense = conn.todense()
        r = conn[3]
        assert r.shape == (12,)
        assert jnp.allclose(r, dense[3], atol=1e-5)

    @pytest.mark.parametrize('homo', [False, True])
    def test_multi_and_slice(self, homo):
        conn = _make_perpre(8, 12, 3, homo=homo)
        dense = conn.todense()
        assert jnp.allclose(conn[[1, 4, 6]], np.asarray(dense)[[1, 4, 6]], atol=1e-5)
        assert jnp.allclose(conn[2:7:2], dense[np.arange(2, 7, 2)], atol=1e-5)

    def test_negative_and_oob(self):
        conn = _make_perpre(8, 12, 3)
        dense = conn.todense()
        assert jnp.allclose(conn[-1], dense[7], atol=1e-5)
        with pytest.raises(IndexError):
            _ = conn[8]
