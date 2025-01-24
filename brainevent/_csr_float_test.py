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

import jax
import jax.numpy as jnp

import brainevent
import brainstate as bst
from ._csr_test_util import _get_csr, vector_csr, matrix_csr, csr_vector, csr_matrix


# bst.environ.set(platform='cpu')

class TestVectorCSR(unittest.TestCase):
    def test_vector_csr(self, ):
        m, n = 20, 40
        x = bst.random.rand(m)
        indptr, indices = _get_csr(m, n, 0.1)

        for homo_w in [True, False]:
            print(f'homo_w = {homo_w}')
            data = 1.5 if homo_w else bst.init.Normal()(indices.shape)
            csr = brainevent.CSR([data, indices, indptr], shape=(m, n))
            y = x @ csr
            y2 = vector_csr(x, csr.data, indices, indptr, [m, n])
            self.assertTrue(jnp.allclose(y, y2))

    def test_vector_csr_vmap_vector(self):
        n_batch, m, n = 10, 20, 40
        xs = bst.random.rand(n_batch, m)
        indptr, indices = _get_csr(m, n, 0.1)

        for homo_w in [True, False]:
            data = 1.5 if homo_w else bst.init.Normal()(indices.shape)
            csr = brainevent.CSR([data, indices, indptr], shape=(m, n))
            y = jax.vmap(lambda x: x @ csr)(xs)
            y2 = jax.vmap(lambda x: vector_csr(x, csr.data, indices, indptr, [m, n]))(xs)

            print(y.shape, y2.shape)
            self.assertTrue(jnp.allclose(y, y2, rtol=1e-3, atol=1e-3))


class TestMatrixCSR(unittest.TestCase):
    def test_matrix_csr(self):
        k, m, n = 10, 20, 40
        x = bst.random.rand(k, m)
        indptr, indices = _get_csr(m, n, 0.1)

        for homo_w in [True, False]:
            data = 1.5 if homo_w else bst.init.Normal()(indices.shape)
            csr = brainevent.CSR([data, indices, indptr], shape=(m, n))
            y = x @ csr
            y2 = matrix_csr(x, csr.data, indices, indptr, [m, n])
            self.assertTrue(jnp.allclose(y, y2))


class TestCSRVector(unittest.TestCase):
    def test_csr_vector(self):
        m, n = 20, 40
        v = bst.random.rand(n)
        indptr, indices = _get_csr(m, n, 0.1)

        for homo_w in [True, False]:
            data = 1.5 if homo_w else bst.init.Normal()(indices.shape)
            csr = brainevent.CSR([data, indices, indptr], shape=(m, n))
            y = csr @ v
            y2 = csr_vector(v, csr.data, indices, indptr, [m, n])
            self.assertTrue(jnp.allclose(y, y2))


class TestCSRMatrix(unittest.TestCase):
    def test_csr_matrix(self):
        m, n, k = 20, 40, 10
        matrix = bst.random.rand(n, k)
        indptr, indices = _get_csr(m, n, 0.1)

        for homo_w in [True, False]:
            data = 1.5 if homo_w else bst.init.Normal()(indices.shape)
            csr = brainevent.CSR([data, indices, indptr], shape=(m, n))
            y = csr @ matrix
            y2 = csr_matrix(matrix, csr.data, indices, indptr, [m, n])
            self.assertTrue(jnp.allclose(y, y2))

    # @parameterized.product(
    #     bool_x=[True, False],
    #     homo_w=[True, False]
    # )
    # def test_vjp(self, bool_x, homo_w):
    #     n_in = 20
    #     n_out = 30
    #     if bool_x:
    #         x = jax.numpy.asarray(bst.random.rand(n_in) < 0.3, dtype=float)
    #     else:
    #         x = bst.random.rand(n_in)
    #
    #     indptr, indices = _get_csr(n_in, n_out, 0.1)
    #     fn = brainevent.CSRLinear(n_in, n_out, indptr, indices, 1.5 if homo_w else bst.init.Normal())
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
    #     self.assertTrue(jnp.allclose(r[0], r2[0]))
    #     self.assertTrue(jnp.allclose(r[1], r2[1]))
    #
    # @parameterized.product(
    #     bool_x=[True, False],
    #     homo_w=[True, False]
    # )
    # def test_jvp(self, bool_x, homo_w):
    #     n_in = 20
    #     n_out = 30
    #     if bool_x:
    #         x = jax.numpy.asarray(bst.random.rand(n_in) < 0.3, dtype=float)
    #     else:
    #         x = bst.random.rand(n_in)
    #
    #     indptr, indices = _get_csr(n_in, n_out, 0.1)
    #     fn = brainevent.CSRLinear(n_in, n_out, indptr, indices,
    #                              1.5 if homo_w else bst.init.Normal(), grad_mode='jvp')
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
    #     self.assertTrue(jnp.allclose(r1, r2))
    #     self.assertTrue(jnp.allclose(o1, o2))
