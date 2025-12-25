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
from brainevent._csr.binary_test import TestBatchingVectorCSR, TestBatchingMatrixCSR
from brainevent._csr.float import csrmv_yw2y
from brainevent._csr.test_util import get_csr, vector_csr, matrix_csr, csr_vector, csr_matrix

pytest.mark.skipif(brainstate.environ.get_platform() != 'cpu', allow_module_level=True)


class TestVectorCSR:
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vector_csr(self, homo_w):
        m, n = 20, 40
        x = brainstate.random.rand(m)
        indptr, indices = get_csr(m, n, 0.1)

        print(f'homo_w = {homo_w}')
        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))
        y = x @ csr
        y2 = vector_csr(x, csr.data, indices, indptr, [m, n])
        assert (jnp.allclose(y, y2, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_csr_vector(self, homo_w):
        m, n = 20, 40
        v = brainstate.random.rand(n)
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))
        y = csr @ v
        y2 = csr_vector(v, csr.data, indices, indptr, [m, n])
        assert (jnp.allclose(y, y2, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vector_csr_vmap_vector(self, homo_w):
        n_batch, m, n = 10, 20, 40
        xs = brainstate.random.rand(n_batch, m)
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))
        y = jax.vmap(lambda x: x @ csr)(xs)
        y2 = jax.vmap(lambda x: vector_csr(x, csr.data, indices, indptr, [m, n]))(xs)

        print(y.shape, y2.shape)
        assert (jnp.allclose(y, y2, rtol=1e-3, atol=1e-3))

    def _test_vjp(self, homo_w, replace, transpose):
        n_in = 20
        n_out = 30
        shape = [n_in, n_out]
        x = brainstate.random.rand(n_in) if transpose else brainstate.random.rand(n_out)

        indptr, indices = get_csr(n_in, n_out, 0.2, replace=replace)
        w = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        csr = brainevent.CSR((w, indices, indptr), shape=shape)

        def f_brainevent(x, w):
            if transpose:
                r = x @ csr.with_data(w)
            else:
                r = csr.with_data(w) @ x
            return r.sum()

        r = jax.jit(lambda: jax.grad(f_brainevent, argnums=(0, 1))(x, w))()

        # -------------------
        # TRUE gradients

        def f_jax(x, w):
            if transpose:
                r = vector_csr(x, w, indices, indptr, shape=shape)
            else:
                r = csr_vector(x, w, indices, indptr, shape=shape)
            return r.sum()

        r2 = jax.jit(lambda: jax.grad(f_jax, argnums=(0, 1))(x, w))()
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
        indptr, indices = get_csr(n_in, n_out, 0.1, replace=replace)

        w = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        csr = brainevent.CSR((w, indices, indptr), shape=shape)

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
                r = vector_csr(x, w, indices, indptr, shape=shape)
            else:
                r = csr_vector(x, w, indices, indptr, shape=shape)
            return r

        o2, r2 = jax.jit(lambda: jax.jvp(f_jax, (x, w), (jnp.ones_like(x), jnp.ones_like(w))))()
        assert (jnp.allclose(r1, r2, rtol=1e-3, atol=1e-3))
        assert (jnp.allclose(o1, o2, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_jvp(self, homo_w, replace, transpose):
        print(f'replace = {replace}, transpose = {transpose}, homo_w = {homo_w}')
        self._test_jvp(homo_w=homo_w, replace=replace, transpose=transpose)


class TestBatchingVectorCSRFloat(TestBatchingVectorCSR):
    def _run(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))
        if transpose:
            y1 = x @ csr
            y2 = vector_csr(x, csr.data, indices, indptr, [m, n])
        else:
            y1 = csr @ x
            y2 = csr_vector(x, csr.data, indices, indptr, [m, n])
        return jnp.allclose(y1, y2)

    def _run_vjp(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        x = x.astype(float)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))

        def f_brainevent(x, w):
            if transpose:
                r = x @ csr.with_data(w)
            else:
                r = csr.with_data(w) @ x
            return r.sum()

        r1 = jax.grad(f_brainevent, argnums=(0, 1))(x, csr.data)

        def f_jax(x, w):
            if transpose:
                r = vector_csr(x, w, indices, indptr, shape=[m, n])
            else:
                r = csr_vector(x, w, indices, indptr, shape=[m, n])
            return r.sum()

        r2 = jax.jit(lambda: jax.grad(f_jax, argnums=(0, 1))(x, csr.data))()

        return r1, r2

    def _run_jvp(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        x = x.astype(float)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))

        def f_brainevent(x, w):
            if transpose:
                r = x @ csr.with_data(w)
            else:
                r = csr.with_data(w) @ x
            return r

        r1 = jax.jvp(f_brainevent, (x, data), (jnp.ones_like(x), jnp.ones_like(data)))

        def f_jax(x, w):
            if transpose:
                r = vector_csr(x, w, indices, indptr, shape=(m, n))
            else:
                r = csr_vector(x, w, indices, indptr, shape=(m, n))
            return r

        r2 = jax.jit(lambda: jax.jvp(f_jax, (x, data), (jnp.ones_like(x), jnp.ones_like(data))))()

        return r1, r2


class TestMatrixCSR:
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_matrix_csr(self, homo_w):
        k, m, n = 10, 20, 40
        x = brainstate.random.rand(k, m)
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))
        y = x @ csr
        y2 = matrix_csr(x, csr.data, indices, indptr, [m, n])
        assert (jnp.allclose(y, y2, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_csr_matrix(self, homo_w):
        m, n, k = 20, 40, 10
        matrix = brainstate.random.rand(n, k)
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else braintools.init.Normal(0., 1.)(indices.shape)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))
        y = csr @ matrix
        y2 = csr_matrix(matrix, csr.data, indices, indptr, [m, n])
        assert (jnp.allclose(y, y2, rtol=1e-3, atol=1e-3))


class TestBatchingMatrixCSRFloat(TestBatchingMatrixCSR):
    def _run(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))
        if transpose:
            y1 = x @ csr
            y2 = matrix_csr(x, csr.data, indices, indptr, [m, n])
        else:
            y1 = csr @ x
            y2 = csr_matrix(x, csr.data, indices, indptr, [m, n])
        return jnp.allclose(y1, y2)

    def _run_vjp(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        x = x.astype(float)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))

        def f_brainevent(x, w):
            if transpose:
                r = x @ csr.with_data(w)
            else:
                r = csr.with_data(w) @ x
            return r.sum()

        r1 = jax.grad(f_brainevent, argnums=(0, 1))(x, csr.data)

        def f_jax(x, w):
            if transpose:
                r = matrix_csr(x, w, indices, indptr, shape=[m, n])
            else:
                r = csr_matrix(x, w, indices, indptr, shape=[m, n])
            return r.sum()

        r2 = jax.jit(lambda: jax.grad(f_jax, argnums=(0, 1))(x, csr.data))()

        return r1, r2

    def _run_jvp(self, x, data, indices, indptr, m: int, n: int, transpose: bool = True):
        x = x.astype(float)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))

        def f_brainevent(x, w):
            if transpose:
                r = x @ csr.with_data(w)
            else:
                r = csr.with_data(w) @ x
            return r

        r1 = jax.jvp(f_brainevent, (x, data), (jnp.ones_like(x), jnp.ones_like(data)))

        def f_jax(x, w):
            if transpose:
                r = matrix_csr(x, w, indices, indptr, shape=(m, n))
            else:
                r = csr_matrix(x, w, indices, indptr, shape=(m, n))
            return r

        r2 = jax.jit(lambda: jax.jvp(f_jax, (x, data), (jnp.ones_like(x), jnp.ones_like(data))))()

        return r1, r2


class Test_csrmv_yw2y:
    @pytest.mark.parametrize('shape', [(100, 200), (200, 400)])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_csr(self, shape, transpose):
        m, n = shape
        indptr, indices = get_csr(m, n, 0.5)

        data = braintools.init.Normal(0., 1.)(indices.shape)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))
        dense = csr.todense()

        if transpose:
            y = brainstate.random.rand(n)
        else:
            y = brainstate.random.rand(m)

        res1 = csrmv_yw2y(y, csr.data, indices, indptr, shape=[m, n], transpose=transpose)
        dense_res1 = csr.with_data(res1).todense()
        if transpose:
            print(dense)
            dense_res2 = dense * jnp.expand_dims(y, axis=0)
        else:
            dense_res2 = dense * jnp.expand_dims(y, axis=1)

        assert (jnp.allclose(dense_res1, dense_res2, rtol=1e-2, atol=1e-2))

    def test_csr2(self):
        for shape in [(100, 200), (200, 400)]:
            # for shape in [(200, 400)]:
            m, n = shape
            indptr, indices = get_csr(m, n, 0.5)
            data = braintools.init.Normal(0., 1.)(indices.shape)
            csr = brainevent.CSR((data, indices, indptr), shape=(m, n))
            dense = csr.todense()

            for transpose in [True, False]:
                if transpose:
                    y = brainstate.random.rand(n)
                else:
                    y = brainstate.random.rand(m)

                res1 = csrmv_yw2y(y, csr.data, indices, indptr, shape=[m, n], transpose=transpose)
                dense_res1 = csr.with_data(res1).todense()
                if transpose:
                    dense_res2 = dense * jnp.expand_dims(y, axis=0)
                else:
                    dense_res2 = dense * jnp.expand_dims(y, axis=1)

                print(jnp.abs(dense_res1 - dense_res2).max())
                # assert (jnp.allclose(dense_res1, dense_res2))

    def test_csr_no_transpose(self):
        m, n = 10, 8
        m, n = 1000, 800
        indptr, indices = get_csr(m, n, 0.5)

        data = braintools.init.Normal(0., 1.)(indices.shape)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))
        dense = csr.todense()
        print()
        print('original csr')
        print(dense)

        y = jnp.ones(m)

        res1 = csrmv_yw2y(y, csr.data, indices, indptr, shape=[m, n], transpose=False)
        dense_res1 = csr.with_data(res1).todense()
        dense_res2 = dense * jnp.expand_dims(y, axis=1)
        print('csr')
        print(dense_res1)
        print(csr.indptr)

        print('dense')
        print(dense_res2)

        print('diff')
        print(dense_res1 - dense_res2)
        # print(jnp.abs(dense_res1 - dense_res2).max())
        # assert (jnp.allclose(dense_res1, dense_res2))
