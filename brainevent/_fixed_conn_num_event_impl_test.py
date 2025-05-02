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
from brainevent._fixed_conn_num_test_util import generate_data, vector_csr, csr_vector


# brainstate.environ.set(platform='cpu')


class TestVector:

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    def test_results(self, homo_w, replace):
        rng = bst.random.RandomState(1)
        m, n = 20, 40
        x = rng.rand(m) < 0.5
        indices = generate_data(m, n, 8, replace=replace, rng=rng)
        print(f'replace = {replace}, homo_w = {homo_w}')
        data = 1.5 if homo_w else rng.randn(*indices.shape)
        csr = brainevent.FixedPostNumConn([data, indices], shape=(m, n))
        y = brainevent.EventArray(x) @ csr
        print(y)

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    def test_vector_csr(self, homo_w, replace):
        m, n = 20, 40
        x = bst.random.rand(m) < 0.5
        indices = generate_data(m, n, 8, replace=replace)
        print(f'replace = {replace}, homo_w = {homo_w}')
        data = 1.5 if homo_w else bst.init.Normal()(indices.shape)
        csr = brainevent.FixedPostNumConn([data, indices], shape=(m, n))
        y = brainevent.EventArray(x) @ csr
        y2 = vector_csr(x, csr.data, indices, shape=[m, n])
        assert (jnp.allclose(y, y2, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    def test_csr_vector(self, homo_w, replace):
        m, n = 20, 40
        v = bst.random.rand(n) < 0.5
        indices = generate_data(m, n, 8, replace=replace)

        print(f'replace = {replace}, homo_w = {homo_w}')
        data = 1.5 if homo_w else bst.init.Normal()(indices.shape)
        csr = brainevent.FixedPostNumConn([data, indices], shape=(m, n))
        y = csr @ brainevent.EventArray(v)
        y2 = csr_vector(v, csr.data, indices, [m, n])
        # print(y)
        # print(y2)
        # print()
        assert (jnp.allclose(y, y2, rtol=1e-3, atol=1e-3))

    # def test_vector_csr_vmap_vector(self):
    #     n_batch, m, n = 10, 20, 40
    #     xs = brainstate.random.rand(n_batch, m)
    #     indices = generate_data(m, n, 8)
    #
    #     for homo_w in [True, False]:
    #         data = 1.5 if homo_w else brainstate.init.Normal()(indices.shape)
    #         csr = brainevent.FixedPostNumConn([data, indices], shape=(m, n))
    #         y = jax.vmap(lambda x: x @ csr)(xs)
    #         y2 = jax.vmap(lambda x: vector_csr(x, csr.data, indices, [m, n]))(xs)
    #
    #         print(y.shape, y2.shape)
    #         assert (jnp.allclose(y, y2, rtol=1e-3, atol=1e-3))

    def _test_vjp(self, homo_w, replace, transpose):
        n_in = 20
        n_out = 30
        shape = [n_in, n_out]
        x = bst.random.rand(n_in) if transpose else bst.random.rand(n_out)
        x = (x < 0.5).astype(float)

        indices = generate_data(n_in, n_out, 8, replace=replace)
        w = 1.5 if homo_w else bst.init.Normal()(indices.shape)
        csr = brainevent.FixedPostNumConn([w, indices], shape=shape)

        def f_brainevent(x, w):
            if transpose:
                r = brainevent.EventArray(x) @ csr.with_data(w)
            else:
                r = csr.with_data(w) @ brainevent.EventArray(x)
            return r.sum()

        r = jax.grad(f_brainevent, argnums=(0, 1))(x, w)

        # -------------------
        # TRUE gradients

        def f_jax(x, w):
            if transpose:
                r = vector_csr(x, w, indices, shape=shape)
            else:
                r = csr_vector(x, w, indices, shape=shape)
            return r.sum()

        r2 = jax.grad(f_jax, argnums=(0, 1))(x, w)
        assert (jnp.allclose(r[0], r2[0], rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(r[1], r2[1], rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_vjp(self, transpose, replace, homo_w):
        print(f'replace = {replace}, transpose = {transpose}, homo_w = {homo_w}')
        self._test_vjp(homo_w=homo_w, replace=replace, transpose=transpose)

    def _test_jvp(self, homo_w, replace, transpose):
        n_in = 20
        n_out = 30
        shape = [n_in, n_out]
        x = bst.random.rand(n_in if transpose else n_out)
        x = (x < 0.5).astype(float)
        indices = generate_data(n_in, n_out, 8, replace=replace)

        indices = generate_data(n_in, n_out, 8, replace=replace)
        w = 1.5 if homo_w else bst.init.Normal()(indices.shape)
        csr = brainevent.FixedPostNumConn([w, indices], shape=shape)

        def f_brainevent(x, w):
            if transpose:
                r = brainevent.EventArray(x) @ csr.with_data(w)
            else:
                r = csr.with_data(w) @ brainevent.EventArray(x)
            return r

        o1, r1 = jax.jvp(f_brainevent, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))

        # -------------------
        # TRUE gradients

        def f_jax(x, w):
            if transpose:
                r = vector_csr(x, w, indices, shape=shape)
            else:
                r = csr_vector(x, w, indices, shape=shape)
            return r

        o2, r2 = jax.jvp(f_jax, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))
        assert (jnp.allclose(r1, r2, rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(o1, o2, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_jvp(self, transpose, replace, homo_w):
        print(f'replace = {replace}, transpose = {transpose}, homo_w = {homo_w}')
        self._test_jvp(homo_w=homo_w, replace=replace, transpose=transpose)
