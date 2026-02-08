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
