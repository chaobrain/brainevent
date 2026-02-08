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
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent


def gen_sparse_matrix(shape, prob=0.2):
    matrix = np.random.rand(*shape)
    matrix = np.where(matrix < prob, matrix, 0.)
    return jnp.asarray(matrix, dtype=float)


class TestCOOAddSub:
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_add_scalar(self, shape):
        matrix = gen_sparse_matrix(shape)
        coo = brainevent.COO.fromdense(matrix)
        result = coo + 5.0
        expected = coo.todense() + 5.0
        assert not isinstance(result, brainevent.COO)
        assert jnp.allclose(result, expected)

    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_radd_scalar(self, shape):
        matrix = gen_sparse_matrix(shape)
        coo = brainevent.COO.fromdense(matrix)
        result = 5.0 + coo
        expected = 5.0 + coo.todense()
        assert not isinstance(result, brainevent.COO)
        assert jnp.allclose(result, expected)

    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_sub_dense(self, shape):
        matrix = gen_sparse_matrix(shape)
        coo = brainevent.COO.fromdense(matrix)
        ones = jnp.ones(shape)
        result = coo - ones
        expected = coo.todense() - ones
        assert not isinstance(result, brainevent.COO)
        assert jnp.allclose(result, expected)

    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_rsub_dense(self, shape):
        matrix = gen_sparse_matrix(shape)
        coo = brainevent.COO.fromdense(matrix)
        ones = jnp.ones(shape)
        result = ones - coo
        expected = ones - coo.todense()
        assert not isinstance(result, brainevent.COO)
        assert jnp.allclose(result, expected)

    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_mul_scalar_stays_sparse(self, shape):
        matrix = gen_sparse_matrix(shape)
        coo = brainevent.COO.fromdense(matrix)
        result = coo * 2.0
        assert isinstance(result, brainevent.COO)
        expected = coo.todense() * 2.0
        assert jnp.allclose(result.todense(), expected)


class TestCOOToCsr:
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_tocsr(self, shape):
        matrix = gen_sparse_matrix(shape)
        coo = brainevent.COO.fromdense(matrix)
        csr = coo.tocsr()
        assert isinstance(csr, brainevent.CSR)
        assert jnp.allclose(csr.todense(), coo.todense())


class TestCOOMatmul:
    """Test COO.__matmul__ and COO.__rmatmul__ operator dispatch."""

    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matmul_binary_vector(self, shape):
        matrix = gen_sparse_matrix(shape)
        coo = brainevent.COO.fromdense(matrix)
        v = brainevent.BinaryArray(brainstate.random.rand(shape[1]) < 0.5)
        result = coo @ v
        expected = matrix @ v.value.astype(float)
        assert jnp.allclose(result, expected)

    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_rmatmul_binary_vector(self, shape):
        matrix = gen_sparse_matrix(shape)
        coo = brainevent.COO.fromdense(matrix)
        v = brainevent.BinaryArray(brainstate.random.rand(shape[0]) < 0.5)
        result = v @ coo
        expected = v.value.astype(float) @ matrix
        assert jnp.allclose(result, expected)

    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matmul_binary_matrix(self, shape):
        matrix = gen_sparse_matrix(shape)
        coo = brainevent.COO.fromdense(matrix)
        B = brainevent.BinaryArray(brainstate.random.rand(shape[1], 10) < 0.5)
        result = coo @ B
        expected = matrix @ B.value.astype(float)
        assert jnp.allclose(result, expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_rmatmul_binary_matrix(self, shape):
        matrix = gen_sparse_matrix(shape)
        coo = brainevent.COO.fromdense(matrix)
        B = brainevent.BinaryArray(brainstate.random.rand(10, shape[0]) < 0.5)
        result = B @ coo
        expected = B.value.astype(float) @ matrix
        assert jnp.allclose(result, expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matmul_dense_vector(self, shape):
        matrix = gen_sparse_matrix(shape)
        coo = brainevent.COO.fromdense(matrix)
        v = jnp.asarray(np.random.rand(shape[1]))
        result = coo @ v
        expected = matrix @ v
        assert jnp.allclose(result, expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_rmatmul_dense_vector(self, shape):
        matrix = gen_sparse_matrix(shape)
        coo = brainevent.COO.fromdense(matrix)
        v = jnp.asarray(np.random.rand(shape[0]))
        result = v @ coo
        expected = v @ matrix
        assert jnp.allclose(result, expected, rtol=1e-3, atol=1e-3)
