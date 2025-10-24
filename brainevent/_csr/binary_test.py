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
from brainevent._csr.test_util import get_csr, vector_csr, matrix_csr, csr_vector, csr_matrix

pytest.mark.skipif(brainstate.environ.get_platform() != 'cpu', allow_module_level=True)


class TestCSR:
    def test_event_homo_bool(self):
        for dat in [1., 2., 3.]:
            mask = (brainstate.random.rand(10, 20) < 0.1).astype(float) * dat
            csr = u.sparse.CSR.fromdense(mask)
            csr = brainevent.CSR((dat, csr.indices, csr.indptr), shape=mask.shape)

            v = brainevent.EventArray(brainstate.random.rand(20) < 0.5)
            assert (
                u.math.allclose(
                    mask.astype(float) @ v.data.astype(float),
                    csr @ v
                )
            )

            v = brainevent.EventArray(brainstate.random.rand(10) < 0.5)
            assert (
                u.math.allclose(
                    v.data.astype(float) @ mask.astype(float),
                    v @ csr
                )
            )

    def test_event_homo_heter(self):
        mat = brainstate.random.rand(10, 20)
        mask = (brainstate.random.rand(10, 20) < 0.1) * mat
        csr = u.sparse.CSR.fromdense(mask)
        csr = brainevent.CSR((csr.data, csr.indices, csr.indptr), shape=mask.shape)

        v = brainevent.EventArray(brainstate.random.rand(20) < 0.5)
        assert (
            u.math.allclose(
                mask.astype(float) @ v.data.astype(float),
                csr @ v
            )
        )

        v = brainevent.EventArray(brainstate.random.rand(10) < 0.5)
        assert (
            u.math.allclose(
                v.data.astype(float) @ mask.astype(float),
                v @ csr
            )
        )

    def test_event_heter_float_as_bool(self):
        mat = brainstate.random.rand(10, 20)
        mask = (mat < 0.1).astype(float) * mat
        csr = u.sparse.CSR.fromdense(mask)
        csr = brainevent.CSR((csr.data, csr.indices, csr.indptr), shape=mask.shape)

        v = brainevent.EventArray((brainstate.random.rand(20) < 0.5).astype(float))
        assert (
            u.math.allclose(
                mask.astype(float) @ v.data.astype(float),
                csr @ v
            )
        )

        v = brainevent.EventArray((brainstate.random.rand(10) < 0.5).astype(float))
        assert (
            u.math.allclose(
                v.data.astype(float) @ mask.astype(float),
                v @ csr
            )
        )


class TestVectorCSR:
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vector_csr(self, homo_w):
        m, n = 20, 40
        x = brainstate.random.rand(m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        print(f'homo_w = {homo_w}')
        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))
        y = brainevent.EventArray(x) @ csr
        y2 = vector_csr(x, csr.data, indices, indptr, [m, n])
        assert (jnp.allclose(y, y2, rtol=1e-5, atol=1e-5))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vector_csr_vmap_vector(self, homo_w):
        n_batch, m, n = 10, 20, 40
        xs = brainstate.random.rand(n_batch, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))
        y = jax.vmap(lambda x: brainevent.EventArray(x) @ csr)(xs)
        y2 = jax.vmap(lambda x: vector_csr(x, csr.data, indices, indptr, [m, n]))(xs)
        assert (jnp.allclose(y, y2, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_csr_vector(self, homo_w):
        m, n = 20, 40
        v = brainstate.random.rand(n) < 0.1
        indptr, indices = get_csr(m, n, 0.2)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))
        y = csr @ brainevent.EventArray(v)
        y2 = csr_vector(v, csr.data, indices, indptr, [m, n])
        assert (jnp.allclose(y, y2, rtol=1e-5, atol=1e-5))

    def _test_vjp(self, homo_w, replace, transpose):
        n_in = 20
        n_out = 30
        shape = [n_in, n_out]
        x = brainstate.random.rand(n_in) if transpose else brainstate.random.rand(n_out)
        x = (x < 0.6).astype(float)

        indptr, indices = get_csr(n_in, n_out, 0.2, replace=replace)
        w = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        csr = brainevent.CSR((w, indices, indptr), shape=shape)

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
                r = vector_csr(x, w, indices, indptr, shape=shape)
            else:
                r = csr_vector(x, w, indices, indptr, shape=shape)
            return r.sum()

        r2 = jax.grad(f_jax, argnums=(0, 1))(x, w)
        assert (jnp.allclose(r[0], r2[0], rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(r[1], r2[1], rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_vjp(self, homo_w, replace, transpose):
        print(f'replace = {replace}, transpose = {transpose}, homo_w = {homo_w}')
        self._test_vjp(homo_w=homo_w, replace=replace, transpose=transpose)

    def _test_jvp(self, homo_w, replace, transpose):
        n_in = 20
        n_out = 30
        shape = [n_in, n_out]
        x = brainstate.random.rand(n_in if transpose else n_out)
        x = (x < 0.6).astype(float)

        indptr, indices = get_csr(n_in, n_out, 0.1, replace=replace)

        w = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        csr = brainevent.CSR((w, indices, indptr), shape=shape)

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
                r = vector_csr(x, w, indices, indptr, shape=shape)
            else:
                r = csr_vector(x, w, indices, indptr, shape=shape)
            return r

        o2, r2 = jax.jvp(f_jax, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))
        assert (jnp.allclose(r1, r2, rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(o1, o2, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_jvp(self, homo_w, replace, transpose):
        print(f'replace = {replace}, transpose = {transpose}, homo_w = {homo_w}')
        self._test_jvp(homo_w=homo_w, replace=replace, transpose=transpose)


class TestBatchingVectorCSR:
    def _run(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))
        if transpose:
            y1 = brainevent.EventArray(x) @ csr
            y2 = vector_csr(x, csr.data, indices, indptr, [m, n])
        else:
            y1 = csr @ brainevent.EventArray(x)
            y2 = csr_vector(x, csr.data, indices, indptr, [m, n])
        return jnp.allclose(y1, y2)

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_vector(self, homo_w):
        b, m, n = 10, 20, 40
        xs = brainstate.random.rand(b, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        res = jax.vmap(lambda x: self._run(x, data, indices, indptr, m, n))(xs)
        assert jnp.all(res)

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_data(self, homo_w):
        b, m, n = 10, 20, 40
        x = brainstate.random.rand(m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = brainstate.random.rand(b) if homo_w else braintools.init.Normal(0., 1.)((b,) + indices.shape)
        res = jax.vmap(lambda data: self._run(x, data, indices, indptr, m, n))(data)
        assert jnp.all(res)

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_indices(self, homo_w):
        b, m, n, p = 10, 20, 40, 0.1
        x = brainstate.random.rand(m) < 0.1
        indptr, indices = brainstate.transform.for_loop(lambda *a: get_csr(m, n, 0.1), length=b)
        indptr = indptr[0]

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape[1:])
        res = jax.vmap(lambda ind: self._run(x, data, ind, indptr, m, n))(indices)
        assert jnp.all(res)

    def _run_vjp(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        x = x.astype(float)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))

        def f_brainevent(x, w):
            if transpose:
                r = brainevent.EventArray(x) @ csr.with_data(w)
            else:
                r = csr.with_data(w) @ brainevent.EventArray(x)
            return r.sum()

        r1 = jax.grad(f_brainevent, argnums=(0, 1))(x, csr.data)

        def f_jax(x, w):
            if transpose:
                r = vector_csr(x, w, indices, indptr, shape=[m, n])
            else:
                r = csr_vector(x, w, indices, indptr, shape=[m, n])
            return r.sum()

        r2 = jax.grad(f_jax, argnums=(0, 1))(x, csr.data)

        return r1, r2

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_vector_vjp(self, homo_w):
        b, m, n = 10, 20, 40
        xs = brainstate.random.rand(b, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        r1, r2 = jax.vmap(lambda x: self._run_vjp(x, data, indices, indptr, m, n))(xs)

        assert (jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_data_vjp(self, homo_w):
        b, m, n = 10, 20, 40
        x = brainstate.random.rand(m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = brainstate.random.rand(b) if homo_w else braintools.init.Normal(0., 1.)((b,) + indices.shape)
        r1, r2 = jax.vmap(lambda data: self._run_vjp(x, data, indices, indptr, m, n))(data)

        assert (jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3))

    @pytest.mark.skip
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_indices_vjp(self, homo_w):
        b, m, n, p = 10, 20, 40, 0.1
        x = brainstate.random.rand(m) < 0.1
        indptr, indices = brainstate.transform.for_loop(lambda *a: get_csr(m, n, 0.1), length=b)
        indptr = indptr[0]

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape[1:])
        r1, r2 = jax.vmap(lambda ind: self._run_vjp(x, data, ind, indptr, m, n))(indices)

        assert (jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3))

    def _run_jvp(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        x = x.astype(float)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))

        def f_brainevent(x, w):
            if transpose:
                r = brainevent.EventArray(x) @ csr.with_data(w)
            else:
                r = csr.with_data(w) @ brainevent.EventArray(x)
            return r

        r1 = jax.jvp(f_brainevent, (x, data), (jnp.ones_like(x), jnp.ones_like(data)))

        def f_jax(x, w):
            if transpose:
                r = vector_csr(x, w, indices, indptr, shape=(m, n))
            else:
                r = csr_vector(x, w, indices, indptr, shape=(m, n))
            return r

        r2 = jax.jvp(f_jax, (x, data), (jnp.ones_like(x), jnp.ones_like(data)))

        return r1, r2

    @pytest.mark.skip
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_vector_jvp(self, homo_w):
        b, m, n = 10, 20, 40
        xs = brainstate.random.rand(b, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        r1, r2 = jax.vmap(lambda x: self._run_jvp(x, data, indices, indptr, m, n))(xs)

        assert (jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3))

    @pytest.mark.skip
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_data_jvp(self, homo_w):
        b, m, n = 10, 20, 40
        x = brainstate.random.rand(m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = brainstate.random.rand(b) if homo_w else braintools.init.Normal(0., 1.)((b,) + indices.shape)
        r1, r2 = jax.vmap(lambda data: self._run_jvp(x, data, indices, indptr, m, n))(data)

        assert (jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3))

    @pytest.mark.skip
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_indices_jvp(self, homo_w):
        b, m, n, p = 10, 20, 40, 0.1
        x = brainstate.random.rand(m) < 0.1
        indptr, indices = brainstate.transform.for_loop(lambda *a: get_csr(m, n, 0.1), length=b)
        indptr = indptr[0]

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape[1:])
        r1, r2 = jax.vmap(lambda ind: self._run_jvp(x, data, ind, indptr, m, n))(indices)

        assert (jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3))


class TestMatrixCSR:
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_matrix_csr(self, homo_w):
        k, m, n = 10, 20, 40
        x = brainstate.random.rand(k, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))
        y = brainevent.EventArray(x) @ csr
        y2 = matrix_csr(x, csr.data, indices, indptr, [m, n])
        assert (jnp.allclose(y, y2, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_csr_matrix(self, homo_w):
        m, n, k = 20, 40, 10
        matrix = brainstate.random.rand(n, k) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))
        y = csr @ brainevent.EventArray(matrix)
        y2 = csr_matrix(matrix, csr.data, indices, indptr, [m, n])
        assert (jnp.allclose(y, y2))


class TestBatchingMatrixCSR:
    def _run(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))
        if transpose:
            y1 = brainevent.EventArray(x) @ csr
            y2 = matrix_csr(x, csr.data, indices, indptr, [m, n])
        else:
            y1 = csr @ brainevent.EventArray(x)
            y2 = csr_matrix(x, csr.data, indices, indptr, [m, n])
        return jnp.allclose(y1, y2)

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_matrix(self, homo_w):
        b, k, m, n = 10, 15, 20, 40
        xs = brainstate.random.rand(b, k, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        res = jax.vmap(lambda x: self._run(x, data, indices, indptr, m, n))(xs)
        assert jnp.all(res)

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_data(self, homo_w):
        b, k, m, n = 10, 15, 20, 40
        x = brainstate.random.rand(k, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = brainstate.random.rand(b) if homo_w else braintools.init.Normal(0., 1.)((b,) + indices.shape)
        res = jax.vmap(lambda data: self._run(x, data, indices, indptr, m, n))(data)
        assert jnp.all(res)

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_indices(self, homo_w):
        b, k, m, n, p = 10, 15, 20, 40, 0.1
        x = brainstate.random.rand(k, m) < 0.1
        indptr, indices = brainstate.transform.for_loop(lambda *a: get_csr(m, n, 0.1), length=b)
        indptr = indptr[0]

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape[1:])
        res = jax.vmap(lambda ind: self._run(x, data, ind, indptr, m, n))(indices)
        assert jnp.all(res)

    def _run_vjp(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        x = x.astype(float)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))

        def f_brainevent(x, w):
            if transpose:
                r = brainevent.EventArray(x) @ csr.with_data(w)
            else:
                r = csr.with_data(w) @ brainevent.EventArray(x)
            return r.sum()

        r1 = jax.grad(f_brainevent, argnums=(0, 1))(x, csr.data)

        def f_jax(x, w):
            if transpose:
                r = matrix_csr(x, w, indices, indptr, shape=[m, n])
            else:
                r = csr_matrix(x, w, indices, indptr, shape=[m, n])
            return r.sum()

        r2 = jax.grad(f_jax, argnums=(0, 1))(x, csr.data)

        return r1, r2

    @pytest.mark.skip  # TODO: fix bugs
    @pytest.mark.parametrize('transpose', [True, False])
    def test_vmap_matrix_vjp(self, transpose):
        b, k, m, n = 10, 15, 20, 40
        xs = brainstate.random.rand(b, k, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = braintools.init.Normal(0., 1.)(indices.shape)
        r1, r2 = jax.vmap(lambda x: self._run_vjp(x, data, indices, indptr, m, n, transpose=transpose))(xs)

        assert (jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3))

    @pytest.mark.skip  # TODO: fix bugs
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_data_vjp(self, homo_w):
        b, k, m, n = 10, 15, 20, 40
        x = brainstate.random.rand(k, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = brainstate.random.rand(b) if homo_w else braintools.init.Normal(0., 1.)((b,) + indices.shape)
        r1, r2 = jax.vmap(lambda data: self._run_vjp(x, data, indices, indptr, m, n))(data)

        assert (jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3))

    @pytest.mark.skip  # TODO: fix bugs
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_indices_vjp(self, homo_w):
        b, k, m, n, p = 10, 15, 20, 40, 0.1
        x = brainstate.random.rand(k, m) < 0.1
        indptr, indices = brainstate.transform.for_loop(lambda *a: get_csr(m, n, 0.1), length=b)
        indptr = indptr[0]

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape[1:])
        r1, r2 = jax.vmap(lambda ind: self._run_vjp(x, data, ind, indptr, m, n))(indices)

        assert (jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3))

    def _run_jvp(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        x = x.astype(float)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))

        def f_brainevent(x, w):
            if transpose:
                r = brainevent.EventArray(x) @ csr.with_data(w)
            else:
                r = csr.with_data(w) @ brainevent.EventArray(x)
            return r

        r1 = jax.jvp(f_brainevent, (x, data), (jnp.ones_like(x), jnp.ones_like(data)))

        def f_jax(x, w):
            if transpose:
                r = matrix_csr(x, w, indices, indptr, shape=(m, n))
            else:
                r = csr_matrix(x, w, indices, indptr, shape=(m, n))
            return r

        r2 = jax.jvp(f_jax, (x, data), (jnp.ones_like(x), jnp.ones_like(data)))

        return r1, r2

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_vector_jvp(self, homo_w):
        b, k, m, n = 10, 15, 20, 40
        xs = brainstate.random.rand(b, k, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        r1, r2 = jax.vmap(lambda x: self._run_jvp(x, data, indices, indptr, m, n))(xs)

        assert (jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_data_jvp(self, homo_w):
        b, k, m, n = 10, 15, 20, 40
        x = brainstate.random.rand(k, m) < 0.1
        indptr, indices = get_csr(m, n, 0.1)

        data = brainstate.random.rand(b) if homo_w else braintools.init.Normal(0., 1.)((b,) + indices.shape)
        r1, r2 = jax.vmap(lambda data: self._run_jvp(x, data, indices, indptr, m, n))(data)

        assert (jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3))

    @pytest.mark.skip
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vmap_indices_jvp(self, homo_w):
        b, k, m, n, p = 10, 15, 20, 40, 0.1
        x = brainstate.random.rand(k, m) < 0.1
        indptr, indices = brainstate.transform.for_loop(lambda *a: get_csr(m, n, 0.1), length=b)
        indptr = indptr[0]

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape[1:])
        r1, r2 = jax.vmap(lambda ind: self._run_jvp(x, data, ind, indptr, m, n))(indices)

        assert (jnp.allclose(r1[0], r2[0], rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(r1[1], r2[1], rtol=1e-3, atol=1e-3))
