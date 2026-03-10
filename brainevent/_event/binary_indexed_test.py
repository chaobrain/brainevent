# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._event.binary_indexed import IndexedBinary1d, IndexedBinary2d


class TestIndexedBinary1d:
    def test_basic_bool(self):
        spikes = jnp.array([True, False, True, False, True], dtype=jnp.bool_)
        ib = IndexedBinary1d(spikes)
        assert ib.n_active[0] == 3
        assert ib.length == 5
        assert jnp.array_equal(ib.active_ids[:ib.n_active[0]], jnp.array([0, 2, 4]))

    def test_basic_float(self):
        spikes = jnp.array([0.0, 1.0, 0.0, 2.0], dtype=jnp.float32)
        ib = IndexedBinary1d(spikes)
        assert ib.n_active[0] == 2
        assert ib.length == 4
        assert jnp.array_equal(ib.active_ids[:ib.n_active[0]], jnp.array([1, 3]))

    def test_all_zero(self):
        spikes = jnp.zeros(10, dtype=jnp.bool_)
        ib = IndexedBinary1d(spikes)
        assert ib.n_active[0] == 0
        assert ib.length == 10

    def test_all_active(self):
        spikes = jnp.ones(5, dtype=jnp.bool_)
        ib = IndexedBinary1d(spikes)
        assert ib.n_active[0] == 5
        assert jnp.array_equal(ib.active_ids[:5], jnp.arange(5))

    def test_backward_compat_aliases(self):
        spikes = jnp.array([True, False, True], dtype=jnp.bool_)
        ib = IndexedBinary1d(spikes)
        assert jnp.array_equal(ib.spike_indices, ib.active_ids)
        assert jnp.array_equal(ib.spike_count, ib.n_active)

    def test_pytree_roundtrip(self):
        spikes = jnp.array([False, True, True, False], dtype=jnp.bool_)
        ib = IndexedBinary1d(spikes)
        leaves, treedef = jax.tree.flatten(ib)
        ib2 = treedef.unflatten(leaves)
        assert jnp.array_equal(ib2.active_ids, ib.active_ids)
        assert jnp.array_equal(ib2.n_active, ib.n_active)
        assert jnp.array_equal(ib2.value, ib.value)


class TestIndexedBinary2d:
    def test_basic_bool(self):
        B = jnp.array([
            [True, False, True],
            [False, False, False],
            [True, True, False],
        ], dtype=jnp.bool_)
        ib = IndexedBinary2d(B)
        assert ib.packed.shape == (3, 1)
        assert ib.n_active[0] == 2
        # Row 0: bits 0,2 -> 5
        assert ib.packed[0, 0] == 5
        # Row 1: no bits -> 0
        assert ib.packed[1, 0] == 0
        # Row 2: bits 0,1 -> 3
        assert ib.packed[2, 0] == 3

    def test_basic_float(self):
        B = jnp.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 2.0],
        ], dtype=jnp.float32)
        ib = IndexedBinary2d(B)
        assert ib.packed.shape == (3, 1)
        assert ib.n_active[0] == 2
        # Row 0: bit 1 -> 2
        assert ib.packed[0, 0] == 2
        # Row 2: bits 0,2 -> 5
        assert ib.packed[2, 0] == 5

    def test_multi_word_packing(self):
        """Test with n_batch > 32, requiring multiple packed words per row."""
        np.random.seed(123)
        n_pre, n_batch = 4, 64
        B_np = np.random.rand(n_pre, n_batch) > 0.5
        B = jnp.asarray(B_np, dtype=jnp.bool_)
        ib = IndexedBinary2d(B)

        assert ib.packed.shape == (4, 2)  # ceil(64/32) = 2

        # Verify every bit
        for row in range(n_pre):
            for w in range(2):
                word = int(ib.packed[row, w])
                for bit in range(32):
                    col = w * 32 + bit
                    if col < n_batch:
                        expected = int(B_np[row, col])
                        actual = (word >> bit) & 1
                        assert expected == actual, f'Mismatch at ({row},{col})'

    def test_all_zero_rows(self):
        B = jnp.zeros((5, 10), dtype=jnp.bool_)
        ib = IndexedBinary2d(B)
        assert ib.n_active[0] == 0
        assert ib.packed.shape == (5, 1)
        assert jnp.all(ib.packed == 0)

    def test_all_active_rows(self):
        B = jnp.ones((3, 8), dtype=jnp.bool_)
        ib = IndexedBinary2d(B)
        assert ib.n_active[0] == 3
        # All 8 bits set in word 0: 0xFF = 255
        assert jnp.all(ib.packed[:, 0] == 255)

    def test_active_ids_correctness(self):
        """Verify active_ids lists exactly the rows with non-zero elements."""
        np.random.seed(456)
        n_pre, n_batch = 100, 32
        B_np = np.random.rand(n_pre, n_batch) > 0.99  # ~1% firing rate
        B = jnp.asarray(B_np, dtype=jnp.bool_)
        ib = IndexedBinary2d(B)

        expected_active = set(i for i in range(n_pre) if B_np[i].any())
        actual_active = set(int(ib.active_ids[i]) for i in range(ib.n_active[0]))
        assert expected_active == actual_active

    def test_backward_compat_aliases(self):
        B = jnp.array([[True, False], [False, True]], dtype=jnp.bool_)
        ib = IndexedBinary2d(B)
        assert jnp.array_equal(ib.spike_indices, ib.active_ids)
        assert jnp.array_equal(ib.spike_count, ib.n_active)

    def test_pytree_roundtrip(self):
        B = jnp.array([
            [True, False, True],
            [False, False, False],
            [True, True, False],
        ], dtype=jnp.bool_)
        ib = IndexedBinary2d(B)
        leaves, treedef = jax.tree.flatten(ib)
        ib2 = treedef.unflatten(leaves)
        assert jnp.array_equal(ib2.packed, ib.packed)
        assert jnp.array_equal(ib2.active_ids, ib.active_ids)
        assert jnp.array_equal(ib2.n_active, ib.n_active)
        assert jnp.array_equal(ib2.value, ib.value)

    def test_large_batch(self):
        """Test with n_batch=128, 4 packed words per row."""
        np.random.seed(789)
        n_pre, n_batch = 50, 128
        B = jnp.asarray(np.random.rand(n_pre, n_batch) > 0.98, dtype=jnp.bool_)
        ib = IndexedBinary2d(B)
        assert ib.packed.shape == (50, 4)  # ceil(128/32) = 4

        # Verify n_active
        n_actually_active = sum(1 for i in range(n_pre) if bool(jnp.any(B[i])))
        assert ib.n_active[0] == n_actually_active


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
