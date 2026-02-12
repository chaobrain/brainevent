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

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO

from brainevent._sddmm import sddmm_indices, sddmm_coo_indices, sddmm_bcoo


def _reference_sddmm(A, B, rows, cols):
    """Compute reference SDDMM values: (A @ B)[rows, cols]."""
    full = A @ B
    return full[rows, cols]


class TestSddmmIndices(unittest.TestCase):
    """Tests for sddmm_indices."""

    def test_basic_correctness(self):
        A = jnp.array([[1., 2.], [3., 4.]])  # (2, 2)
        B = jnp.array([[5., 6.], [7., 8.]])  # (2, 2)
        indices = jnp.array([[0, 0], [1, 1]])  # sample (0,0) and (1,1)

        result = sddmm_indices(A, B, indices)
        expected = _reference_sddmm(A, B, indices[:, 0], indices[:, 1])

        self.assertIsInstance(result, BCOO)
        np.testing.assert_allclose(result.data, expected, rtol=1e-5)

    def test_non_square_matrices(self):
        m, k, n = 4, 3, 5
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        A = jax.random.normal(k1, (m, k))
        B = jax.random.normal(k2, (k, n))
        indices = jnp.array([[0, 1], [2, 3], [3, 4], [1, 0]])

        result = sddmm_indices(A, B, indices)
        expected = _reference_sddmm(A, B, indices[:, 0], indices[:, 1])

        np.testing.assert_allclose(result.data, expected, rtol=1e-5)

    def test_output_shape(self):
        m, k, n = 6, 4, 8
        key = jax.random.PRNGKey(1)
        k1, k2 = jax.random.split(key)
        A = jax.random.normal(k1, (m, k))
        B = jax.random.normal(k2, (k, n))
        indices = jnp.array([[0, 0], [5, 7]])

        result = sddmm_indices(A, B, indices)

        self.assertEqual(result.shape, (m, n))
        self.assertEqual(result.data.shape[0], 2)

    def test_single_element(self):
        A = jnp.ones((3, 4))
        B = jnp.ones((4, 5))
        indices = jnp.array([[1, 2]])

        result = sddmm_indices(A, B, indices)
        expected = _reference_sddmm(A, B, indices[:, 0], indices[:, 1])

        np.testing.assert_allclose(result.data, expected, rtol=1e-5)
        self.assertEqual(result.shape, (3, 5))

    def test_all_elements_sampled(self):
        m, n, k = 2, 3, 2
        key = jax.random.PRNGKey(2)
        k1, k2 = jax.random.split(key)
        A = jax.random.normal(k1, (m, k))
        B = jax.random.normal(k2, (k, n))
        # Sample every element
        rows, cols = jnp.meshgrid(jnp.arange(m), jnp.arange(n), indexing='ij')
        indices = jnp.stack([rows.ravel(), cols.ravel()], axis=1)

        result = sddmm_indices(A, B, indices)
        expected = (A @ B).ravel()

        np.testing.assert_allclose(result.data, expected, rtol=1e-5)

    def test_indices_preserved(self):
        A = jnp.eye(3)
        B = jnp.eye(3)
        indices = jnp.array([[0, 1], [2, 0], [1, 2]])

        result = sddmm_indices(A, B, indices)

        np.testing.assert_array_equal(result.indices, indices)

    def test_assertion_a_not_2d(self):
        A = jnp.ones((3,))
        B = jnp.ones((3, 4))
        indices = jnp.array([[0, 0]])
        with self.assertRaises(AssertionError):
            sddmm_indices(A, B, indices)

    def test_assertion_b_not_2d(self):
        A = jnp.ones((3, 4))
        B = jnp.ones((4,))
        indices = jnp.array([[0, 0]])
        with self.assertRaises(AssertionError):
            sddmm_indices(A, B, indices)

    def test_assertion_inner_dim_mismatch(self):
        A = jnp.ones((3, 4))
        B = jnp.ones((5, 6))  # inner dims 4 != 5
        indices = jnp.array([[0, 0]])
        with self.assertRaises(AssertionError):
            sddmm_indices(A, B, indices)

    def test_assertion_indices_not_2d(self):
        A = jnp.ones((3, 4))
        B = jnp.ones((4, 5))
        indices = jnp.array([0, 1])  # 1-D
        with self.assertRaises(AssertionError):
            sddmm_indices(A, B, indices)

    def test_assertion_indices_wrong_width(self):
        A = jnp.ones((3, 4))
        B = jnp.ones((4, 5))
        indices = jnp.array([[0, 1, 2]])  # width 3, not 2
        with self.assertRaises(AssertionError):
            sddmm_indices(A, B, indices)


class TestSddmmCooIndices(unittest.TestCase):
    """Tests for sddmm_coo_indices."""

    def test_basic_correctness(self):
        A = jnp.array([[1., 2.], [3., 4.]])
        B = jnp.array([[5., 6.], [7., 8.]])
        pre_idx = jnp.array([0, 1])
        post_idx = jnp.array([0, 1])

        result = sddmm_coo_indices(A, B, pre_idx, post_idx)
        expected = _reference_sddmm(A, B, pre_idx, post_idx)

        self.assertIsInstance(result, BCOO)
        np.testing.assert_allclose(result.data, expected, rtol=1e-5)

    def test_matches_sddmm_indices(self):
        m, k, n = 5, 3, 7
        key = jax.random.PRNGKey(3)
        k1, k2 = jax.random.split(key)
        A = jax.random.normal(k1, (m, k))
        B = jax.random.normal(k2, (k, n))
        pre_idx = jnp.array([0, 2, 4, 1])
        post_idx = jnp.array([1, 3, 6, 0])
        indices = jnp.stack([pre_idx, post_idx], axis=1)

        result_coo = sddmm_coo_indices(A, B, pre_idx, post_idx)
        result_idx = sddmm_indices(A, B, indices)

        np.testing.assert_allclose(result_coo.data, result_idx.data, rtol=1e-5)
        np.testing.assert_array_equal(result_coo.indices, result_idx.indices)

    def test_output_shape(self):
        m, k, n = 4, 2, 6
        A = jnp.ones((m, k))
        B = jnp.ones((k, n))
        pre_idx = jnp.array([0, 3])
        post_idx = jnp.array([5, 1])

        result = sddmm_coo_indices(A, B, pre_idx, post_idx)

        self.assertEqual(result.shape, (m, n))

    def test_assertion_pre_idx_not_1d(self):
        A = jnp.ones((3, 4))
        B = jnp.ones((4, 5))
        pre_idx = jnp.array([[0, 1]])  # 2-D
        post_idx = jnp.array([0, 1])
        with self.assertRaises(AssertionError):
            sddmm_coo_indices(A, B, pre_idx, post_idx)

    def test_assertion_post_idx_not_1d(self):
        A = jnp.ones((3, 4))
        B = jnp.ones((4, 5))
        pre_idx = jnp.array([0, 1])
        post_idx = jnp.array([[0, 1]])  # 2-D
        with self.assertRaises(AssertionError):
            sddmm_coo_indices(A, B, pre_idx, post_idx)

    def test_assertion_idx_shape_mismatch(self):
        A = jnp.ones((3, 4))
        B = jnp.ones((4, 5))
        pre_idx = jnp.array([0, 1, 2])
        post_idx = jnp.array([0, 1])  # different length
        with self.assertRaises(AssertionError):
            sddmm_coo_indices(A, B, pre_idx, post_idx)

    def test_assertion_inner_dim_mismatch(self):
        A = jnp.ones((3, 4))
        B = jnp.ones((5, 6))
        pre_idx = jnp.array([0])
        post_idx = jnp.array([0])
        with self.assertRaises(AssertionError):
            sddmm_coo_indices(A, B, pre_idx, post_idx)


class TestSddmmBcoo(unittest.TestCase):
    """Tests for sddmm_bcoo."""

    def test_basic_correctness(self):
        A = jnp.array([[1., 2.], [3., 4.]])
        B = jnp.array([[5., 6.], [7., 8.]])
        indices = jnp.array([[0, 0], [1, 1]])
        pattern = BCOO((jnp.ones(2), indices), shape=(2, 2))

        result = sddmm_bcoo(A, B, pattern)
        expected = _reference_sddmm(A, B, indices[:, 0], indices[:, 1])

        self.assertIsInstance(result, BCOO)
        np.testing.assert_allclose(result.data, expected, rtol=1e-5)

    def test_matches_sddmm_indices(self):
        m, k, n = 5, 3, 7
        key = jax.random.PRNGKey(4)
        k1, k2 = jax.random.split(key)
        A = jax.random.normal(k1, (m, k))
        B = jax.random.normal(k2, (k, n))
        indices = jnp.array([[0, 1], [2, 3], [4, 6]])
        pattern = BCOO((jnp.ones(3), indices), shape=(m, n))

        result_bcoo = sddmm_bcoo(A, B, pattern)
        result_idx = sddmm_indices(A, B, indices)

        np.testing.assert_allclose(result_bcoo.data, result_idx.data, rtol=1e-5)
        np.testing.assert_array_equal(result_bcoo.indices, result_idx.indices)

    def test_sparsity_pattern_values_ignored(self):
        A = jnp.array([[1., 2.], [3., 4.]])
        B = jnp.array([[5., 6.], [7., 8.]])
        indices = jnp.array([[0, 0], [1, 1]])
        pattern_ones = BCOO((jnp.ones(2), indices), shape=(2, 2))
        pattern_rand = BCOO((jnp.array([42., -7.]), indices), shape=(2, 2))

        result_ones = sddmm_bcoo(A, B, pattern_ones)
        result_rand = sddmm_bcoo(A, B, pattern_rand)

        np.testing.assert_allclose(result_ones.data, result_rand.data, rtol=1e-5)

    def test_output_shape(self):
        m, k, n = 8, 4, 10
        A = jnp.ones((m, k))
        B = jnp.ones((k, n))
        indices = jnp.array([[0, 0], [7, 9]])
        pattern = BCOO((jnp.ones(2), indices), shape=(m, n))

        result = sddmm_bcoo(A, B, pattern)

        self.assertEqual(result.shape, (m, n))

    def test_from_dense_sparsity(self):
        m, k, n = 4, 3, 5
        key = jax.random.PRNGKey(5)
        k1, k2, k3 = jax.random.split(key, 3)
        A = jax.random.normal(k1, (m, k))
        B = jax.random.normal(k2, (k, n))
        # Create a sparse mask from a random binary matrix
        mask_dense = (jax.random.uniform(k3, (m, n)) > 0.5).astype(jnp.float32)
        pattern = BCOO.fromdense(mask_dense)

        result = sddmm_bcoo(A, B, pattern)
        rows, cols = pattern.indices[:, 0], pattern.indices[:, 1]
        expected = _reference_sddmm(A, B, rows, cols)

        np.testing.assert_allclose(result.data, expected, rtol=1e-5)


class TestSddmmConsistency(unittest.TestCase):
    """Cross-variant consistency tests."""

    def test_all_three_variants_agree(self):
        m, k, n = 6, 4, 8
        key = jax.random.PRNGKey(6)
        k1, k2 = jax.random.split(key)
        A = jax.random.normal(k1, (m, k))
        B = jax.random.normal(k2, (k, n))
        pre_idx = jnp.array([0, 2, 5, 3, 1])
        post_idx = jnp.array([1, 4, 7, 0, 3])
        indices = jnp.stack([pre_idx, post_idx], axis=1)
        pattern = BCOO((jnp.ones(5), indices), shape=(m, n))

        r1 = sddmm_indices(A, B, indices)
        r2 = sddmm_coo_indices(A, B, pre_idx, post_idx)
        r3 = sddmm_bcoo(A, B, pattern)

        np.testing.assert_allclose(r1.data, r2.data, rtol=1e-5)
        np.testing.assert_allclose(r1.data, r3.data, rtol=1e-5)

    def test_jit_compatibility(self):
        m, k, n = 4, 3, 5
        key = jax.random.PRNGKey(7)
        k1, k2 = jax.random.split(key)
        A = jax.random.normal(k1, (m, k))
        B = jax.random.normal(k2, (k, n))
        indices = jnp.array([[0, 1], [2, 3], [3, 4]])

        result_eager = sddmm_indices(A, B, indices)
        result_jit = jax.jit(sddmm_indices)(A, B, indices)

        np.testing.assert_allclose(result_jit.data, result_eager.data, rtol=1e-5)

    def test_grad_through_sddmm(self):
        m, k, n = 3, 2, 4
        key = jax.random.PRNGKey(8)
        k1, k2 = jax.random.split(key)
        A = jax.random.normal(k1, (m, k))
        B = jax.random.normal(k2, (k, n))
        indices = jnp.array([[0, 0], [1, 2], [2, 3]])

        def loss_fn(A, B):
            result = sddmm_indices(A, B, indices)
            return jnp.sum(result.data)

        grads = jax.grad(loss_fn, argnums=(0, 1))(A, B)
        dA, dB = grads

        self.assertEqual(dA.shape, A.shape)
        self.assertEqual(dB.shape, B.shape)
        # Gradient should not be all zeros for random inputs
        self.assertTrue(jnp.any(dA != 0))
        self.assertTrue(jnp.any(dB != 0))


if __name__ == '__main__':
    unittest.main()
