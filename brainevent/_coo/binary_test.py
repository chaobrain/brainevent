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
# -*- coding: utf-8 -*-


import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import pytest

import brainevent
from brainevent._coo.test_util import _get_coo, vector_coo, matrix_coo, coo_vector, coo_matrix

pytest.mark.skipif(brainstate.environ.get_platform() != 'cpu', allow_module_level=True)


class TestCOO:
    def test_event_homo_bool(self):
        for dat in [1., 2., 3.]:
            mask = (brainstate.random.rand(10, 20) < 0.1).astype(float) * dat
            coo = u.sparse.COO.fromdense(mask)
            coo = brainevent.COO((dat, coo.row, coo.col), shape=mask.shape)

            v = brainevent.EventArray(brainstate.random.rand(20) < 0.5)
            assert (
                u.math.allclose(
                    mask.astype(float) @ v.data.astype(float),
                    coo @ v
                )
            )

            v = brainevent.EventArray(brainstate.random.rand(10) < 0.5)
            assert (
                u.math.allclose(
                    v.data.astype(float) @ mask.astype(float),
                    v @ coo
                )
            )

    def test_event_homo_float_as_bool(self):
        mat = brainstate.random.rand(10, 20)
        mask = (mat < 0.1).astype(float) * mat
        coo = u.sparse.COO.fromdense(mask)
        coo = brainevent.COO((coo.data, coo.row, coo.col), shape=mask.shape)

        v = brainevent.EventArray((brainstate.random.rand(20) < 0.5).astype(float))
        assert (
            u.math.allclose(
                mask.astype(float) @ v.data.astype(float),
                coo @ v
            )
        )

        v = brainevent.EventArray((brainstate.random.rand(10) < 0.5).astype(float))
        assert (
            u.math.allclose(
                v.data.astype(float) @ mask.astype(float),
                v @ coo
            )
        )

    def test_event_homo_other(self):
        mat = brainstate.random.rand(10, 20)
        mask = (brainstate.random.rand(10, 20) < 0.1) * mat
        coo = u.sparse.COO.fromdense(mask)
        coo = brainevent.COO((coo.data, coo.row, coo.col), shape=mask.shape)

        v = brainevent.EventArray(brainstate.random.rand(20) < 0.5)
        assert (
            u.math.allclose(
                mask.astype(float) @ v.data.astype(float),
                coo @ v
            )
        )

        v = brainevent.EventArray(brainstate.random.rand(10) < 0.5)
        assert (
            u.math.allclose(
                v.data.astype(float) @ mask.astype(float),
                v @ coo
            )
        )


class TestVectorCOO:
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vector_coo(self, homo_w):
        m, n = 20, 40
        x = brainstate.random.rand(m) < 0.1
        row, col = _get_coo(m, n, 0.1)

        print(f'homo_w = {homo_w}')
        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(row.shape)
        coo = brainevent.COO([data, row, col], shape=(m, n))
        y = brainevent.EventArray(x) @ coo
        y2 = vector_coo(x, coo.data, row, col, [m, n])
        assert (jnp.allclose(y, y2, rtol=1e-5, atol=1e-5))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vector_coo_vmap_vector(self, homo_w):
        n_batch, m, n = 10, 20, 40
        xs = brainstate.random.rand(n_batch, m) < 0.1
        row, col = _get_coo(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(row.shape)
        coo = brainevent.COO([data, row, col], shape=(m, n))
        y = jax.vmap(lambda x: brainevent.EventArray(x) @ coo)(xs)
        y2 = jax.vmap(lambda x: vector_coo(x, coo.data, row, col, [m, n]))(xs)
        assert (jnp.allclose(y, y2, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_coo_vector(self, homo_w):
        m, n = 20, 40
        v = brainstate.random.rand(n) < 0.1
        row, col = _get_coo(m, n, 0.2)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(row.shape)
        coo = brainevent.COO([data, row, col], shape=(m, n))
        y = coo @ brainevent.EventArray(v)
        y2 = coo_vector(v, coo.data, row, col, [m, n])
        assert (jnp.allclose(y, y2, rtol=1e-5, atol=1e-5))

    def _test_vjp(self, homo_w, replace, transpose):
        n_in = 20
        n_out = 30
        shape = [n_in, n_out]
        x = brainstate.random.rand(n_in) if transpose else brainstate.random.rand(n_out)
        x = (x < 0.6).astype(float)

        row, col = _get_coo(n_in, n_out, 0.2, replace=replace)
        w = 1.5 if homo_w else braintools.init.Normal(0., 1.)(row.shape)
        coo = brainevent.COO((w, row, col), shape=shape)

        def f_brainevent(x, w):
            if transpose:
                r = brainevent.EventArray(x) @ coo.with_data(w)
            else:
                r = coo.with_data(w) @ brainevent.EventArray(x)
            return r.sum()

        r = jax.grad(f_brainevent, argnums=(0, 1))(x, w)

        # -------------------
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
        print(f'replace = {replace}, transpose = {transpose}, homo_w = {homo_w}')
        self._test_vjp(homo_w=homo_w, replace=replace, transpose=transpose)

    def _test_jvp(self, homo_w, replace, transpose):
        n_in = 20
        n_out = 30
        shape = [n_in, n_out]
        x = brainstate.random.rand(n_in if transpose else n_out)
        x = (x < 0.6).astype(float)

        row, col = _get_coo(n_in, n_out, 0.1, replace=replace)

        w = 1.5 if homo_w else braintools.init.Normal(0., 1.)(row.shape)
        coo = brainevent.COO((w, row, col), shape=shape)

        def f_brainevent(x, w):
            if transpose:
                r = brainevent.EventArray(x) @ coo.with_data(w)
            else:
                r = coo.with_data(w) @ brainevent.EventArray(x)
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
    def test_matrix_coo(self):
        k, m, n = 10, 20, 40
        x = brainstate.random.rand(k, m) < 0.1
        row, col = _get_coo(m, n, 0.1)

        for homo_w in [True, False]:
            data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(row.shape)
            coo = brainevent.COO([data, row, col], shape=(m, n))
            y = brainevent.EventArray(x) @ coo
            y2 = matrix_coo(x.astype(float), coo.data, row, col, [m, n])
            assert (jnp.allclose(y, y2, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_coo_matrix(self, homo_w):
        m, n, k = 20, 40, 10
        matrix = brainstate.random.rand(n, k) < 0.1
        row, col = _get_coo(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(row.shape)
        coo = brainevent.COO([data, row, col], shape=(m, n))
        y = coo @ brainevent.EventArray(matrix)
        y2 = coo_matrix(matrix.astype(float), coo.data, row, col, [m, n])
        assert (jnp.allclose(y, y2))
