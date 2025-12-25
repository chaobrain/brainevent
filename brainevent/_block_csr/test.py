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


import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.sparse import bsr_matrix

import brainevent


def gen_bsr_matrix(data, indices, indptr, shape):
    matrix = bsr_matrix((data, indices, indptr), shape=shape)
    return matrix


class Test_BlockCSR:
    @pytest.mark.parametrize('data', [np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)])
    @pytest.mark.parametrize('indices', [np.array([0, 2, 2, 0, 1, 2])])
    @pytest.mark.parametrize('indptr', [np.array([0, 2, 3, 6])])
    @pytest.mark.parametrize('shape', [(6, 6)])
    def test_to_dense(self, data, indices, indptr, shape):
        dense_base = u.math.asarray(gen_bsr_matrix(data, indices, indptr, shape).toarray())
        blockcsr = brainevent.BlockCSR((data, indices, indptr), shape=shape)
        dense_our = blockcsr.todense()

        assert jnp.allclose(dense_base, dense_our)

    @pytest.mark.parametrize('data', [np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)])
    @pytest.mark.parametrize('indices', [np.array([0, 2, 2, 0, 1, 2])])
    @pytest.mark.parametrize('indptr', [np.array([0, 2, 3, 6])])
    @pytest.mark.parametrize('shape', [(6, 6)])
    def test_todense_new(self, data, indices, indptr, shape):
        dense_base = u.math.asarray(gen_bsr_matrix(data, indices, indptr, shape).toarray())
        blockcsr = brainevent.BlockCSR((data, indices, indptr), shape=shape)
        dense_our = blockcsr.todense_new()

        assert jnp.allclose(dense_base, dense_our)

    @pytest.mark.parametrize('data', [np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)])
    @pytest.mark.parametrize('indices', [np.array([0, 2, 2, 0, 1, 2])])
    @pytest.mark.parametrize('indptr', [np.array([0, 2, 3, 6])])
    @pytest.mark.parametrize('shape', [(6, 6)])
    def test_to_coo(self, data, indices, indptr, shape):
        dense_base = u.math.asarray(gen_bsr_matrix(data, indices, indptr, shape).toarray())
        blockcsr = brainevent.BlockCSR((data, indices, indptr), shape=shape)
        coo = blockcsr.tocoo()
        dense_our = coo.todense()

        assert jnp.allclose(dense_base, dense_our)

    @pytest.mark.parametrize('data', [np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)])
    @pytest.mark.parametrize('indices', [np.array([0, 2, 2, 0, 1, 2])])
    @pytest.mark.parametrize('indptr', [np.array([0, 2, 3, 6])])
    @pytest.mark.parametrize('shape', [(6, 6)])
    def test_fromdense(self, data, indices, indptr, shape):
        dense_base = u.math.asarray(gen_bsr_matrix(data, indices, indptr, shape).toarray())
        blockcsr = brainevent.BlockCSR.fromdense(dense_base, block_size=(2, 2))
        dense_our = blockcsr.todense_new()
        indices_our = blockcsr.indices
        indptr_our = blockcsr.indptr

        assert jnp.allclose(dense_base, dense_our)
        assert jnp.allclose(indices, indices_our)
        assert jnp.allclose(indptr, indptr_our)

    @pytest.mark.parametrize('data', [np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)])
    @pytest.mark.parametrize('indices', [np.array([0, 2, 2, 0, 1, 2])])
    @pytest.mark.parametrize('indptr', [np.array([0, 2, 3, 6])])
    @pytest.mark.parametrize('shape', [(6, 6)])
    def test_fromdense_auto_block_size(self, data, indices, indptr, shape):
        dense_base = u.math.asarray(gen_bsr_matrix(data, indices, indptr, shape).toarray())
        blockcsr = brainevent.BlockCSR.fromdense(dense_base)
        bs = blockcsr.block_size
        assert bs == (2, 2)
        dense_our = blockcsr.todense_new()
        indices_our = blockcsr.indices
        indptr_our = blockcsr.indptr

        assert jnp.allclose(dense_base, dense_our)
        assert jnp.allclose(indices, indices_our)
        assert jnp.allclose(indptr, indptr_our)

    @pytest.mark.parametrize('data1', [np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)])
    @pytest.mark.parametrize('data2', [np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)])
    @pytest.mark.parametrize('indices', [np.array([0, 2, 2, 0, 1, 2])])
    @pytest.mark.parametrize('indptr', [np.array([0, 2, 3, 6])])
    @pytest.mark.parametrize('shape', [(6, 6)])
    def test_with_data(self, data1, data2, indices, indptr, shape):
        dense_base_ori = u.math.asarray(gen_bsr_matrix(data1, indices, indptr, shape).toarray())
        dense_base_new = u.math.asarray(gen_bsr_matrix(data2, indices, indptr, shape).toarray())
        blockcsr = brainevent.BlockCSR.fromdense(dense_base_ori, block_size=(2, 2))
        dense_new = blockcsr.with_data(u.math.asarray(data2)).todense_new()

        assert jnp.allclose(dense_base_new, dense_new)

    @pytest.mark.parametrize('data', [np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)])
    @pytest.mark.parametrize('indices', [np.array([0, 2, 2, 0, 1, 2])])
    @pytest.mark.parametrize('indptr', [np.array([0, 2, 3, 6])])
    @pytest.mark.parametrize('shape', [(6, 6)])
    def test_transpose(self, data, indices, indptr, shape):
        dense_base = u.math.asarray(gen_bsr_matrix(data, indices, indptr, shape).toarray())
        blockcsr = brainevent.BlockCSR((data, indices, indptr), shape=shape)
        blockcsr_transpose = blockcsr.transpose()
        dense_our = blockcsr_transpose.todense_new()

        assert jnp.allclose(dense_base.T, dense_our)

    @pytest.mark.parametrize('data', [np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)])
    @pytest.mark.parametrize('indices', [np.array([0, 2, 2, 0, 1, 2])])
    @pytest.mark.parametrize('indptr', [np.array([0, 2, 3, 6])])
    @pytest.mark.parametrize('shape', [(6, 6)])
    def test_tocsr(self, data, indices, indptr, shape):
        base_matrix = gen_bsr_matrix(data, indices, indptr, shape)
        base_matrix_tocsr = base_matrix.tocsr()
        base_matrix_tocsr_data, base_matrix_tocsr_indices, base_matrix_tocsr_indptr = \
            jnp.array(base_matrix_tocsr.data), jnp.array(base_matrix_tocsr.indices), jnp.array(base_matrix_tocsr.indptr)
        blockcsr = brainevent.BlockCSR((data, indices, indptr), shape=shape)
        csr = blockcsr.tocsr()
        csr_data = csr.data
        csr_indices = csr.indices
        csr_indptr = csr.indptr

        assert jnp.allclose(base_matrix_tocsr_data, csr_data)
        assert jnp.allclose(base_matrix_tocsr_indices, csr_indices)
        assert jnp.allclose(base_matrix_tocsr_indptr, csr_indptr)

    # @pytest.mark.parametrize('data', [np.random.rand(6,4).reshape(6, 2, 2)])
    # @pytest.mark.parametrize('indices', [np.array([0, 2, 2, 0, 1, 2])])
    # @pytest.mark.parametrize('indptr', [np.array([0, 2, 3, 6])])
    # @pytest.mark.parametrize('shape', [(6, 6)])
    # def test_solve(self, data, indices, indptr, shape):
    #     dense = u.math.asarray(gen_bsr_matrix(data, indices, indptr, shape).toarray())
    #     blockcsr = brainevent.BlockCSR.fromdense(dense, block_size=(2, 2))
    #     b = brainstate.random.randn(shape[0])
    #
    #     x = blockcsr.solve(b)
    #     assert jnp.allclose(blockcsr @ x, b, atol=1e0, rtol=1e0)
    #
    #     x2 = jnp.linalg.solve(dense, b)
    #     assert jnp.allclose(x, x2, atol=1e0, rtol=1e0)
