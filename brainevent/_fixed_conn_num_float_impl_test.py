# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

import os

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import jax
import brainstate as bst
import jax.numpy as jnp
import pytest

import brainevent
from brainevent._fixed_conn_num_test_util import (
    generate_data,
    vector_csr,
    matrix_csr,
    csr_vector,
    csr_matrix,
)


class TestVector:
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vector_csr(self, replace, homo_w):
        m, n = 20, 40
        x = bst.random.rand(m)
        indices = generate_data(m, n, 8, replace=replace)

        data = 1.5 if homo_w else bst.init.Normal()(indices.shape)
        csr = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
        y1 = x @ csr
        y2 = x @ csr.T

        y_true = vector_csr(x, csr.data, indices, (m, n))
        assert (jnp.allclose(y1, y_true, rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(y2, y_true, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_csr_vector(self, replace, homo_w):
        m, n = 20, 40
        v = bst.random.rand(n)
        indices = generate_data(m, n, 8, replace=replace)

        print(f'replace = {replace}, homo_w = {homo_w}')
        data = 1.5 if homo_w else bst.init.Normal()(indices.shape)
        csr = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
        y = csr @ v
        y2 = csr_vector(v, csr.data, indices, (m, n))
        assert (jnp.allclose(y, y2, rtol=1e-3, atol=1e-3))

    def _test_vjp(self, homo_w, replace, transpose):
        n_in = 20
        n_out = 30
        shape = (n_in, n_out)
        x = bst.random.rand(n_in) if transpose else bst.random.rand(n_out)

        indices = generate_data(n_in, n_out, 8, replace=replace)
        w = 1.5 if homo_w else bst.init.Normal()(indices.shape)
        csr = brainevent.FixedPostNumConn((w, indices), shape=shape)

        def f_brainevent(x, w):
            if transpose:
                r = x @ csr.with_data(w)
            else:
                r = csr.with_data(w) @ x
            return r.sum()

        r1 = jax.jit(lambda: jax.grad(f_brainevent, argnums=(0, 1))(x, w))()

        # -------------------
        # TRUE gradients

        def f_jax(x, w):
            if transpose:
                r = vector_csr(x, w, indices, shape)
            else:
                r = csr_vector(x, w, indices, shape)
            return r.sum()

        r2 = jax.jit(lambda: jax.grad(f_jax, argnums=(0, 1))(x, w))()

        assert (jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vjp(self, replace, transpose, homo_w):
        print(f'replace = {replace}, transpose = {transpose}, homo_w = {homo_w}')
        self._test_vjp(homo_w=homo_w, replace=replace, transpose=transpose)

    def _test_jvp(self, homo_w, replace, transpose):
        n_in = 20
        n_out = 30
        shape = (n_in, n_out)
        x = bst.random.rand(n_in if transpose else n_out)
        indices = generate_data(n_in, n_out, 8, replace=replace)

        indices = generate_data(n_in, n_out, 8, replace=replace)
        w = 1.5 if homo_w else bst.init.Normal()(indices.shape)
        csr = brainevent.FixedPostNumConn((w, indices), shape=shape)

        def f_brainevent(x, w):
            if transpose:
                r = x @ csr.with_data(w)
            else:
                r = csr.with_data(w) @ x
            return r

        o1, r1 = jax.jit(lambda: jax.jvp(f_brainevent, (x, w), (jnp.ones_like(x), jnp.ones_like(w))))()

        # -------------------
        # TRUE gradients

        def f_jax(x, w):
            if transpose:
                r = vector_csr(x, w, indices, shape)
            else:
                r = csr_vector(x, w, indices, shape)
            return r

        o2, r2 = jax.jit(lambda: jax.jvp(f_jax, (x, w), (jnp.ones_like(x), jnp.ones_like(w))))()

        assert (jnp.allclose(r1, r2, rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(o1, o2, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_jvp(self, replace, transpose, homo_w):
        print(f'replace = {replace}, transpose = {transpose}, homo_w = {homo_w}')
        self._test_jvp(homo_w=homo_w, replace=replace, transpose=transpose)


class TestMatrix:
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_matrix_csr(self, replace, homo_w):
        k, m, n = 10, 20, 40
        x = bst.random.rand(k, m)

        indices = generate_data(m, n, 8, replace=replace)

        print(f'replace = {replace}, homo_w = {homo_w}')
        data = 1.5 if homo_w else bst.init.Normal()(indices.shape)
        csr = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
        y = x @ csr
        y2 = matrix_csr(x, csr.data, indices, (m, n))
        assert (jnp.allclose(y, y2, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_csr_matrix(self, replace, homo_w):
        m, n, k = 20, 40, 10
        matrix = bst.random.rand(n, k)

        indices = generate_data(m, n, 8, replace=replace)

        print(f'replace = {replace}, homo_w = {homo_w}')
        data = 1.5 if homo_w else bst.init.Normal()(indices.shape)
        csr = brainevent.FixedPostNumConn((data, indices), shape=(m, n))
        y1 = csr @ matrix
        y2 = csr_matrix(matrix, csr.data, indices, (m, n))
        assert (jnp.allclose(y1, y2, rtol=1e-3, atol=1e-3))
