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


class TestFixedNumPerPreSliceRows:

    @pytest.mark.parametrize('homo', [False, True])
    def test_returns_same_type_and_matches_dense(self, homo):
        conn = _make_perpre(8, 12, 3, homo=homo)
        dense = conn.todense()
        sub = conn.slice_rows([1, 3, 5])
        assert isinstance(sub, FixedNumPerPre)
        assert sub.shape == (3, 12)
        assert sub.num_conn == 3
        assert jnp.allclose(sub.todense(), np.asarray(dense)[[1, 3, 5]], atol=1e-5)

    def test_single_int_is_one_row(self):
        conn = _make_perpre(8, 12, 3)
        dense = conn.todense()
        sub = conn.slice_rows(4)
        assert isinstance(sub, FixedNumPerPre)
        assert sub.shape == (1, 12)
        assert jnp.allclose(sub.todense(), dense[4:5], atol=1e-5)

    def test_jit_safe(self):
        conn = _make_perpre(8, 12, 3)
        dense = conn.todense()
        idx = jnp.array([0, 2, 4], dtype=jnp.int32)

        def f(c, i):
            return c.slice_rows(i).todense()
        out = jax.jit(f)(conn, idx)
        assert jnp.allclose(out, np.asarray(dense)[[0, 2, 4]], atol=1e-5)


class TestFixedNumPerPostGetitem:

    @pytest.mark.parametrize('homo', [False, True])
    def test_single_and_multi_row(self, homo):
        conn = _make_perpost(10, 7, 4, homo=homo)  # W is (10, 7)
        dense = conn.todense()
        assert conn[3].shape == (7,)
        assert jnp.allclose(conn[3], dense[3], atol=1e-5)
        assert jnp.allclose(conn[[0, 4, 8]], np.asarray(dense)[[0, 4, 8]], atol=1e-5)

    def test_slice_negative_oob(self):
        conn = _make_perpost(10, 7, 4)
        dense = conn.todense()
        assert jnp.allclose(conn[1:9:3], dense[np.arange(1, 9, 3)], atol=1e-5)
        assert jnp.allclose(conn[-1], dense[9], atol=1e-5)
        with pytest.raises(IndexError):
            _ = conn[10]


class TestFixedNumPerPostSliceRows:

    @pytest.mark.parametrize('homo', [False, True])
    def test_returns_csr_and_matches_dense(self, homo):
        conn = _make_perpost(10, 7, 4, homo=homo)
        dense = conn.todense()
        sub = conn.slice_rows([1, 3, 5])
        assert isinstance(sub, CSR)
        assert sub.shape == (3, 7)
        assert jnp.allclose(sub.todense(), np.asarray(dense)[[1, 3, 5]], atol=1e-5)

    def test_slice_and_single(self):
        conn = _make_perpost(10, 7, 4)
        dense = conn.todense()
        sub = conn.slice_rows(slice(0, 6, 2))
        assert isinstance(sub, CSR)
        assert jnp.allclose(sub.todense(), dense[np.arange(0, 6, 2)], atol=1e-5)
        one = conn.slice_rows(4)
        assert one.shape == (1, 7)
        assert jnp.allclose(one.todense(), dense[4:5], atol=1e-5)
