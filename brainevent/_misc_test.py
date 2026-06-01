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

import numpy as np

from brainevent._misc import generate_block_dim, coo2csr


class TestCoo2Csr(unittest.TestCase):
    def test_basic_conversion(self):
        row_ids = np.array([0, 2, 1, 0, 2])
        col_ids = np.array([0, 3, 1, 2, 0])
        indptr, indices, order = coo2csr(row_ids, col_ids, shape=(3, 4))
        np.testing.assert_array_equal(indptr, [0, 2, 3, 5])
        np.testing.assert_array_equal(indices, [0, 2, 1, 3, 0])
        np.testing.assert_array_equal(order, [0, 3, 2, 1, 4])

    def test_empty_rows(self):
        # row 1 has no stored entries
        row_ids = np.array([0, 0, 2])
        col_ids = np.array([1, 3, 0])
        indptr, indices, order = coo2csr(row_ids, col_ids, shape=(3, 4))
        np.testing.assert_array_equal(indptr, [0, 2, 2, 3])
        np.testing.assert_array_equal(indices, [1, 3, 0])

    def test_data_reorder_matches_dense(self):
        # The `order` permutation must turn COO data into CSR data such that
        # both reconstruct the same dense matrix (duplicates accumulate).
        row_ids = np.array([0, 2, 1, 0, 2])
        col_ids = np.array([0, 3, 1, 2, 0])
        data = np.array([10., 20., 30., 40., 50.])
        indptr, indices, order = coo2csr(row_ids, col_ids, shape=(3, 4))
        csr_data = data[order]

        dense_csr = np.zeros((3, 4))
        for r in range(3):
            for k in range(int(indptr[r]), int(indptr[r + 1])):
                dense_csr[r, int(indices[k])] += csr_data[k]

        dense_coo = np.zeros((3, 4))
        for r, c, v in zip(row_ids, col_ids, data):
            dense_coo[int(r), int(c)] += v

        np.testing.assert_allclose(dense_csr, dense_coo)


class TestGenerateBlockDim(unittest.TestCase):
    def test_small_connections_returns_32(self):
        self.assertEqual(generate_block_dim(10), 32)
        self.assertEqual(generate_block_dim(32), 32)

    def test_medium_connections_returns_64(self):
        self.assertEqual(generate_block_dim(33), 64)
        self.assertEqual(generate_block_dim(64), 64)

    def test_large_connections_returns_128(self):
        self.assertEqual(generate_block_dim(65), 128)
        self.assertEqual(generate_block_dim(128), 128)

    def test_very_large_connections_returns_256(self):
        self.assertEqual(generate_block_dim(129), 256)
        self.assertEqual(generate_block_dim(256), 256)

    def test_connections_above_maximum_returns_maximum(self):
        self.assertEqual(generate_block_dim(257), 256)
        self.assertEqual(generate_block_dim(1000), 256)

    def test_custom_maximum_constrains_block_size(self):
        self.assertEqual(generate_block_dim(100, maximum=64), 64)
        self.assertEqual(generate_block_dim(200, maximum=128), 128)

    def test_small_maximum_returns_maximum(self):
        self.assertEqual(generate_block_dim(50, maximum=16), 16)

    def test_boundary_conditions(self):
        self.assertEqual(generate_block_dim(0), 32)
        self.assertEqual(generate_block_dim(1), 32)

    def test_negative_connections_returns_32(self):
        self.assertEqual(generate_block_dim(-5), 32)

    def test_maximum_zero_returns_zero(self):
        self.assertEqual(generate_block_dim(100, maximum=0), 0)
