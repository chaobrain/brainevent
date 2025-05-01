# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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


import unittest

import brainstate as bst
import jax
import jax.numpy as jnp
import pytest

import brainevent
from brainevent._csr_event_impl_test import TestBatchingVectorCSR, TestBatchingMatrixCSR
from brainevent._csr_test_util import get_csr, vector_csr, matrix_csr, csr_vector, csr_matrix


# brainstate.environ.set(platform='cpu')


class TestVectorCSR:
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vector_csr(self, homo_w):
        m, n = 20, 40
        x = bst.random.rand(m)
        indptr, indices = get_csr(m, n, 0.1)

        print(f'homo_w = {homo_w}')
        data = 1.5 if homo_w else bst.init.Normal()(indices.shape)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))
        y = x @ csr
        y2 = vector_csr(x, csr.data, indices, indptr, [m, n])
        assert (jnp.allclose(y, y2, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_csr_vector(self, homo_w):
        m, n = 20, 40
        v = bst.random.rand(n)
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else bst.init.Normal()(indices.shape)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))
        y = csr @ v
        y2 = csr_vector(v, csr.data, indices, indptr, [m, n])
        assert (jnp.allclose(y, y2, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vector_csr_vmap_vector(self, homo_w):
        n_batch, m, n = 10, 20, 40
        xs = bst.random.rand(n_batch, m)
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else bst.init.Normal()(indices.shape)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))
        y = jax.vmap(lambda x: x @ csr)(xs)
        y2 = jax.vmap(lambda x: vector_csr(x, csr.data, indices, indptr, [m, n]))(xs)

        print(y.shape, y2.shape)
        assert (jnp.allclose(y, y2, rtol=1e-3, atol=1e-3))

    def _test_vjp(self, homo_w, replace, transpose):
        n_in = 20
        n_out = 30
        shape = [n_in, n_out]
        x = bst.random.rand(n_in) if transpose else bst.random.rand(n_out)

        indptr, indices = get_csr(n_in, n_out, 0.2, replace=replace)
        w = 1.5 if homo_w else bst.init.Normal()(indices.shape)
        csr = brainevent.CSR((w, indices, indptr), shape=shape)

        def f_brainevent(x, w):
            if transpose:
                r = x @ csr.with_data(w)
            else:
                r = csr.with_data(w) @ x
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
        x = bst.random.rand(n_in if transpose else n_out)
        indptr, indices = get_csr(n_in, n_out, 0.1, replace=replace)

        w = 1.5 if homo_w else bst.init.Normal()(indices.shape)
        csr = brainevent.CSR((w, indices, indptr), shape=shape)

        def f_brainevent(x, w):
            if transpose:
                r = x @ csr.with_data(w)
            else:
                r = csr.with_data(w) @ x
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

        r2 = jax.grad(f_jax, argnums=(0, 1))(x, csr.data)

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

        r2 = jax.jvp(f_jax, (x, data), (jnp.ones_like(x), jnp.ones_like(data)))

        return r1, r2


class TestMatrixCSR:
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_matrix_csr(self, homo_w):
        k, m, n = 10, 20, 40
        x = bst.random.rand(k, m)
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else bst.init.Normal()(indices.shape)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))
        y = x @ csr
        y2 = matrix_csr(x, csr.data, indices, indptr, [m, n])
        assert (jnp.allclose(y, y2, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_csr_matrix(self, homo_w):
        m, n, k = 20, 40, 10
        matrix = bst.random.rand(n, k)
        indptr, indices = get_csr(m, n, 0.1)

        data = 1.5 if homo_w else bst.init.Normal()(indices.shape)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))
        y = csr @ matrix
        y2 = csr_matrix(matrix, csr.data, indices, indptr, [m, n])
        assert (jnp.allclose(y, y2, rtol=1e-3, atol=1e-3))

    # @parameterized.product(
    #     bool_x=[True, False],
    #     homo_w=[True, False]
    # )
    # def test_vjp(self, bool_x, homo_w):
    #     n_in = 20
    #     n_out = 30
    #     if bool_x:
    #         x = jax.numpy.asarray(brainstate.random.rand(n_in) < 0.3, dtype=float)
    #     else:
    #         x = brainstate.random.rand(n_in)
    #
    #     indptr, indices = _get_csr(n_in, n_out, 0.1)
    #     fn = brainevent.CSRLinear(n_in, n_out, indptr, indices, 1.5 if homo_w else brainstate.init.Normal())
    #     w = fn.weight.value
    #
    #     def f(x, w):
    #         fn.weight.value = w
    #         return fn(x).sum()
    #
    #     r = jax.grad(f, argnums=(0, 1))(x, w)
    #
    #     # -------------------
    #     # TRUE gradients
    #
    #     def f2(x, w):
    #         return true_fn(x, w, indices, indptr, n_out).sum()
    #
    #     r2 = jax.grad(f2, argnums=(0, 1))(x, w)
    #     assert (jnp.allclose(r[0], r2[0]))
    #     assert (jnp.allclose(r[1], r2[1]))
    #
    # @parameterized.product(
    #     bool_x=[True, False],
    #     homo_w=[True, False]
    # )
    # def test_jvp(self, bool_x, homo_w):
    #     n_in = 20
    #     n_out = 30
    #     if bool_x:
    #         x = jax.numpy.asarray(brainstate.random.rand(n_in) < 0.3, dtype=float)
    #     else:
    #         x = brainstate.random.rand(n_in)
    #
    #     indptr, indices = _get_csr(n_in, n_out, 0.1)
    #     fn = brainevent.CSRLinear(n_in, n_out, indptr, indices,
    #                              1.5 if homo_w else brainstate.init.Normal(), grad_mode='jvp')
    #     w = fn.weight.value
    #
    #     def f(x, w):
    #         fn.weight.value = w
    #         return fn(x)
    #
    #     o1, r1 = jax.jvp(f, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))
    #
    #     # -------------------
    #     # TRUE gradients
    #
    #     def f2(x, w):
    #         return true_fn(x, w, indices, indptr, n_out)
    #
    #     o2, r2 = jax.jvp(f2, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))
    #     assert (jnp.allclose(r1, r2))
    #     assert (jnp.allclose(o1, o2))


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

        r2 = jax.grad(f_jax, argnums=(0, 1))(x, csr.data)

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

        r2 = jax.jvp(f_jax, (x, data), (jnp.ones_like(x), jnp.ones_like(data)))

        return r1, r2
