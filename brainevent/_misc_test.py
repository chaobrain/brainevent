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

from brainevent._misc import generate_block_dim


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
