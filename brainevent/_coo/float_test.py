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


import brainstate
import braintools
import jax
import jax.numpy as jnp
import pytest

import brainevent
from brainevent._coo.test_util import _get_coo, vector_coo, matrix_coo, coo_vector, coo_matrix

pytest.mark.skipif(brainstate.environ.get_platform() != 'cpu', allow_module_level=True)


class TestVectorCOO:
    @pytest.mark.parametrize('parallel', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vector_coo(self, parallel, replace, homo_w):
        with brainevent.config.numba_environ_context(parallel_if_possible=parallel):
            m, n = 20, 40
            x = brainstate.random.rand(m)
            row, col = _get_coo(m, n, 0.1, replace=replace)

            print(f'homo_w = {homo_w}')
            data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(row.shape)
            coo = brainevent.COO((data, row, col), shape=(m, n))
            y = x @ coo
            y2 = vector_coo(x, coo.data, row, col, [m, n])
            assert (jnp.allclose(y, y2, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('parallel', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_coo_vector(self, parallel, replace, homo_w):
        with brainevent.config.numba_environ_context(parallel_if_possible=parallel):
            m, n = 20, 40
            v = brainstate.random.rand(n)
            row, col = _get_coo(m, n, 0.1, replace=replace)

            data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(row.shape)
            coo = brainevent.COO((data, row, col), shape=(m, n))
            y = coo @ v
            y2 = coo_vector(v, coo.data, row, col, [m, n])
            assert (jnp.allclose(y, y2, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('parallel', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vector_coo_vmap_vector(self, parallel, replace, homo_w):
        with brainevent.config.numba_environ_context(parallel_if_possible=parallel):
            n_batch, m, n = 10, 20, 40
            xs = brainstate.random.rand(n_batch, m)
            row, col = _get_coo(m, n, 0.1, replace=replace)

            data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(row.shape)
            coo = brainevent.COO((data, row, col), shape=(m, n))
            y = jax.vmap(lambda x: x @ coo)(xs)
            y2 = jax.vmap(lambda x: vector_coo(x, coo.data, row, col, [m, n]))(xs)

            assert (jnp.allclose(y, y2, rtol=1e-3, atol=1e-3))

    def _test_vjp(self, homo_w, replace, transpose):
        with brainevent.config.numba_environ_context(parallel_if_possible=True):
            n_in = 20
            n_out = 30
            shape = (n_in, n_out)
            x = brainstate.random.rand(n_in) if transpose else brainstate.random.rand(n_out)

            row, col = _get_coo(n_in, n_out, 0.2, replace=replace)
            w = 1.5 if homo_w else braintools.init.Normal(0., 1.)(row.shape)
            coo = brainevent.COO((w, row, col), shape=shape)

            def f_brainevent(x, w):
                if transpose:
                    r = x @ coo.with_data(w)
                else:
                    r = coo.with_data(w) @ x
                return r.sum()

            r = jax.grad(f_brainevent, argnums=(0, 1))(x, w)

            # TRUE gradients
            def f_jax(x, w):
                if transpose:
                    r = vector_coo(x, w, row, col, shape=shape)
                else:
                    r = coo_vector(x, w, row, col, shape=shape)
                return r.sum()

            r2 = jax.grad(f_jax, argnums=(0, 1))(x, w)
            assert (jnp.allclose(r[0], r2[0], rtol=1e-3, atol=1e-3))
            assert (jnp.allclose(r[1], r2[1], rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_vjp(self, transpose, replace, homo_w):
        print(f'replcae = {replace}, transpose = {transpose}, homo_w = {homo_w}')
        self._test_vjp(homo_w=homo_w, replace=replace, transpose=transpose)

    def _test_jvp(self, homo_w, replace, transpose):
        with brainevent.config.numba_environ_context(parallel_if_possible=True):
            n_in = 20
            n_out = 30
            shape = (n_in, n_out)
            x = brainstate.random.rand(n_in if transpose else n_out)
            row, col = _get_coo(n_in, n_out, 0.1, replace=replace)

            w = 1.5 if homo_w else braintools.init.Normal(0., 1.)(row.shape)
            coo = brainevent.COO((w, row, col), shape=shape)

            def f_brainevent(x, w):
                if transpose:
                    r = x @ coo.with_data(w)
                else:
                    r = coo.with_data(w) @ x
                return r

            o1, r1 = jax.jvp(f_brainevent, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))

            # -------------------
            # TRUE gradients

            def f_jax(x, w):
                if transpose:
                    r = vector_coo(x, w, row, col, shape=shape)
                else:
                    r = coo_vector(x, w, row, col, shape=shape)
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


class TestMatrixCOO:
    @pytest.mark.parametrize('parallel', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_matrix_coo(self, parallel, homo_w):
        with brainevent.config.numba_environ_context(parallel_if_possible=parallel):
            k, m, n = 10, 20, 40
            x = brainstate.random.rand(k, m)
            row, col = _get_coo(m, n, 0.1)

            data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(row.shape)
            coo = brainevent.COO((data, row, col), shape=(m, n))
            y = x @ coo
            y2 = matrix_coo(x, coo.data, row, col, [m, n])
            assert (jnp.allclose(y, y2, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('parallel', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_coo_matrix(self, parallel, homo_w):
        with brainevent.config.numba_environ_context(parallel_if_possible=parallel):
            m, n, k = 20, 40, 10
            x = brainstate.random.rand(n, k)
            row, col = _get_coo(m, n, 0.1)

            data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(row.shape)
            coo = brainevent.COO((data, row, col), shape=(m, n))
            y = coo @ x
            y2 = coo_matrix(x, coo.data, row, col, [m, n])
            assert (jnp.allclose(y, y2, rtol=1e-3, atol=1e-3))
