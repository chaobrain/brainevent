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

# -*- coding: utf-8 -*-

import brainstate
import braintools
import jax.numpy as jnp
import pytest

import brainevent
from brainevent._test_util import allclose, generate_fixed_conn_num_indices

if brainstate.environ.get_platform() == 'cpu':
    shapes = [
        (200, 300),
        (100, 500)
    ]
else:
    shapes = [
        (2000, 3000),
        (1000, 5000)
    ]


class Test_To_Dense:
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_todense(self, shape, replace, homo_w):
        m, n = shape
        x = brainstate.random.rand(m)
        indices = generate_fixed_conn_num_indices(m, n, int(n * 0.1), replace=replace)
        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        csr = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
        csc = csr.T

        out1 = csr.todense()
        out2 = csc.todense().T
        out3 = csr.T.todense().T
        out4 = csc.T.todense()
        assert allclose(out1, out2)
        assert allclose(out1, out3)
        assert allclose(out1, out4)


class Test_To_COO:
    def test_tocoo_round_trip(self):
        post_indices = jnp.array([[0, 1, 2, 2], [1, 3, 3, 1], [2, 0, 3, 1]], dtype=jnp.int32)
        post_data = jnp.array([[1., 9., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]], dtype=jnp.float32)
        post = brainevent.FixedPostNumConn((post_data, post_indices), shape=(3, 4))
        assert allclose(post.tocoo().todense(), post.todense())

        pre_indices = jnp.array([[0, 1, 2, 2], [1, 3, 3, 1], [2, 0, 3, 1]], dtype=jnp.int32)
        pre_data = jnp.array([[1., 9., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]], dtype=jnp.float32)
        pre = brainevent.FixedPreNumConn((pre_data, pre_indices), shape=(4, 3))
        assert allclose(pre.tocoo().todense(), pre.todense())


class Test_Illegal_Slots:
    def test_invalid_indices_rejected_post(self):
        idx = jnp.array([[0, -1, 2, 2], [1, 4, 3, 1], [2, 0, -3, 1]], dtype=jnp.int32)
        data = jnp.array([[1., 9., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]], dtype=jnp.float32)
        with pytest.raises(ValueError, match="invalid indices"):
            brainevent.FixedPostNumConn((data, idx), shape=(3, 4))

    def test_invalid_indices_rejected_pre(self):
        idx = jnp.array([[0, -1, 2, 2], [1, 4, 3, 1], [2, 0, -3, 1]], dtype=jnp.int32)
        data = jnp.array([[1., 9., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]], dtype=jnp.float32)
        with pytest.raises(ValueError, match="invalid indices"):
            brainevent.FixedPreNumConn((data, idx), shape=(4, 3))

    def test_invalid_indices_rejected_homo(self):
        idx = jnp.array([[0, -1, 2], [1, 5, 1]], dtype=jnp.int32)
        with pytest.raises(ValueError, match="invalid indices"):
            brainevent.FixedPostNumConn((jnp.array(1.5, dtype=jnp.float32), idx), shape=(2, 4))

    def test_duplicates_are_supported_post(self):
        idx = jnp.array([[0, 1, 2, 2], [1, 3, 3, 1], [2, 0, 3, 1]], dtype=jnp.int32)
        data = jnp.array([[1., 9., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]], dtype=jnp.float32)
        conn = brainevent.FixedPostNumConn((data, idx), shape=(3, 4))

        dense = conn.todense()
        x = jnp.array([1., 2., 3.], dtype=jnp.float32)
        v = jnp.array([1., 2., 3., 4.], dtype=jnp.float32)
        X = jnp.array([[1., 2., 3.], [4., 5., 6.]], dtype=jnp.float32)
        V = jnp.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]], dtype=jnp.float32)

        assert allclose(x @ conn, x @ dense)
        assert allclose(conn @ v, dense @ v)
        assert allclose(X @ conn, X @ dense)
        assert allclose(conn @ V, dense @ V)

    def test_duplicates_are_supported_pre(self):
        idx = jnp.array([[0, 1, 2, 2], [1, 3, 3, 1], [2, 0, 3, 1]], dtype=jnp.int32)
        data = jnp.array([[1., 9., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]], dtype=jnp.float32)
        conn = brainevent.FixedPreNumConn((data, idx), shape=(4, 3))

        dense = conn.todense()
        x = jnp.array([1., 2., 3., 4.], dtype=jnp.float32)
        v = jnp.array([1., 2., 3.], dtype=jnp.float32)
        X = jnp.array([[1., 2., 3., 4.], [5., 6., 7., 8.]], dtype=jnp.float32)
        V = jnp.array([[1., 2.], [3., 4.], [5., 6.]], dtype=jnp.float32)

        assert allclose(x @ conn, x @ dense)
        assert allclose(conn @ v, dense @ v)
        assert allclose(X @ conn, X @ dense)
        assert allclose(conn @ V, dense @ V)

    def test_homo_weight_with_duplicates(self):
        idx = jnp.array([[0, 1, 2], [1, 3, 1]], dtype=jnp.int32)
        conn = brainevent.FixedPostNumConn((jnp.array(1.5, dtype=jnp.float32), idx), shape=(2, 4))
        dense = conn.todense()
        x = jnp.array([1., 2.], dtype=jnp.float32)
        v = jnp.array([1., 2., 3., 4.], dtype=jnp.float32)

        assert allclose(x @ conn, x @ dense)
        assert allclose(conn @ v, dense @ v)
