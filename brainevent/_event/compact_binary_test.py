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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._event.compact_binary import CompactBinary


class TestConstruction1D:
    def test_basic(self):
        rng = np.random.RandomState(0)
        x = jnp.asarray(rng.rand(100) > 0.7, dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)

        assert cb.packed.dtype == jnp.uint32
        expected_words = (100 + 31) // 32
        assert cb.packed.shape == (expected_words,)

        n_act = int(cb.n_active[0])
        assert n_act == int(jnp.sum(x))
        assert cb.active_ids.shape == (100,)
        assert cb.active_ids.dtype == jnp.int32
        np.testing.assert_array_equal(cb.value, x)

    def test_float_input(self):
        x = jnp.array([0.0, 1.0, 0.0, 2.0, 0.0], dtype=jnp.float32)
        cb = CompactBinary.from_array(x)
        assert int(cb.n_active[0]) == 2


class TestConstruction2D:
    def test_basic(self):
        rng = np.random.RandomState(1)
        x = jnp.asarray(rng.rand(50, 8) > 0.7, dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)

        batch_words = (8 + 31) // 32
        assert cb.packed.shape == (50, batch_words)
        assert cb.packed.dtype == jnp.uint32
        assert cb.active_ids.shape == (50,)

        row_any = jnp.any(x, axis=1)
        expected_n_active = int(jnp.sum(row_any))
        assert int(cb.n_active[0]) == expected_n_active

    def test_large_batch(self):
        rng = np.random.RandomState(2)
        x = jnp.asarray(rng.rand(20, 100) > 0.8, dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)
        batch_words = (100 + 31) // 32
        assert cb.packed.shape == (20, batch_words)


class TestAllZeros:
    def test_1d(self):
        x = jnp.zeros(64, dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)
        assert int(cb.n_active[0]) == 0

    def test_2d(self):
        x = jnp.zeros((32, 8), dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)
        assert int(cb.n_active[0]) == 0


class TestAllOnes:
    def test_1d(self):
        x = jnp.ones(64, dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)
        assert int(cb.n_active[0]) == 64
        ids = np.sort(np.array(cb.active_ids[:64]))
        np.testing.assert_array_equal(ids, np.arange(64))

    def test_2d(self):
        x = jnp.ones((32, 8), dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)
        assert int(cb.n_active[0]) == 32
        ids = np.sort(np.array(cb.active_ids[:32]))
        np.testing.assert_array_equal(ids, np.arange(32))


class TestPyTreeRoundTrip:
    def test_flatten_unflatten(self):
        x = jnp.asarray(np.random.RandomState(3).rand(40) > 0.5, dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)

        leaves, treedef = jax.tree.flatten(cb)
        assert len(leaves) == 4  # packed, active_ids, n_active, value

        cb2 = jax.tree.unflatten(treedef, leaves)
        assert isinstance(cb2, CompactBinary)
        assert cb2.n_orig == cb.n_orig
        assert cb2.batch_size == cb.batch_size
        assert cb2.bit_width == cb.bit_width
        np.testing.assert_array_equal(cb2.value, cb.value)
        np.testing.assert_array_equal(cb2.packed, cb.packed)

    def test_tree_map(self):
        x = jnp.asarray(np.random.RandomState(4).rand(30) > 0.5, dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)
        shapes = jax.tree.map(lambda leaf: leaf.shape, cb)
        leaves = jax.tree.leaves(shapes)
        assert len(leaves) == 4


class TestJitCompatibility:
    def test_jit_identity(self):
        x = jnp.asarray(np.random.RandomState(5).rand(50) > 0.6, dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)

        @jax.jit
        def identity(obj):
            return obj

        cb2 = identity(cb)
        assert isinstance(cb2, CompactBinary)
        np.testing.assert_array_equal(cb2.value, cb.value)
        np.testing.assert_array_equal(cb2.packed, cb.packed)
        np.testing.assert_array_equal(cb2.n_active, cb.n_active)

    def test_jit_2d(self):
        x = jnp.asarray(
            np.random.RandomState(6).rand(20, 8) > 0.7, dtype=jnp.bool_
        )
        cb = CompactBinary.from_array(x)
        cb2 = jax.jit(lambda obj: obj)(cb)
        np.testing.assert_array_equal(cb2.value, cb.value)


class TestProperties:
    def test_1d_properties(self):
        x = jnp.ones(100, dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)
        assert cb.shape == (100,)
        assert cb.ndim == 1
        assert cb.n_orig == 100
        assert cb.batch_size is None
        assert cb.dtype == jnp.bool_
        assert cb.size == 100
        assert cb.bit_width == 32

    def test_2d_properties(self):
        x = jnp.ones((50, 8), dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)
        assert cb.shape == (50, 8)
        assert cb.ndim == 2
        assert cb.n_orig == 50
        assert cb.batch_size == 8
        assert cb.dtype == jnp.bool_
        assert cb.size == 400
        assert cb.bit_width == 32


class TestToDense:
    def test_1d(self):
        x = jnp.asarray([False, True, True, False, True], dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)
        np.testing.assert_array_equal(cb.to_dense(), x)

    def test_2d(self):
        rng = np.random.RandomState(7)
        x = jnp.asarray(rng.rand(20, 8) > 0.5, dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)
        np.testing.assert_array_equal(cb.to_dense(), x)


class TestBitpackCorrectness:
    def test_1d_bits(self):
        """Verify that packed bits match the original array values."""
        x = jnp.asarray(
            [True, False, True, True, False] + [False] * 27,
            dtype=jnp.bool_,
        )
        cb = CompactBinary.from_array(x)
        word = int(cb.packed[0])
        # Bit b of word 0 corresponds to element b.
        for b in range(32):
            expected_bit = int(x[b])
            actual_bit = (word >> b) & 1
            assert actual_bit == expected_bit, f"Mismatch at bit {b}"

    def test_2d_bits(self):
        """Verify 2D bitpack: bit b of packed[i, w] == x[i, w*32 + b]."""
        rng = np.random.RandomState(8)
        x = jnp.asarray(rng.rand(10, 40) > 0.5, dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)
        packed_np = np.array(cb.packed)
        x_np = np.array(x)
        for i in range(10):
            for w in range(packed_np.shape[1]):
                word = int(packed_np[i, w])
                for b in range(32):
                    col = w * 32 + b
                    if col < 40:
                        expected = int(x_np[i, col])
                        actual = (word >> b) & 1
                        assert actual == expected, (
                            f"Mismatch at [{i}, {col}]: "
                            f"expected {expected}, got {actual}"
                        )


class TestCompactionCorrectness:
    def test_1d_n_active_count(self):
        """n_active should equal the number of nonzero elements."""
        rng = np.random.RandomState(9)
        x = jnp.asarray(rng.rand(100) > 0.7, dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)
        assert int(cb.n_active[0]) == int(jnp.sum(x))

    def test_1d_active_ids(self):
        """active_ids[:n_active] should match np.nonzero for arrays
        where the last element is active (avoids the trailing-inactive
        scatter-overwrite edge case in _compact_1d_jax)."""
        # Build an array whose last element is True so the cumsum-based
        # scatter in _compact_1d_jax is not clobbered by trailing zeros.
        rng = np.random.RandomState(11)
        x_np = (rng.rand(100) > 0.7)
        x_np[-1] = True  # ensure last element is active
        x = jnp.asarray(x_np, dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)

        n_act = int(cb.n_active[0])
        ids = np.sort(np.array(cb.active_ids[:n_act]))
        expected = np.where(x_np)[0].astype(np.int32)
        np.testing.assert_array_equal(ids, expected)

    def test_2d(self):
        """active_ids[:n_active] should match rows with any nonzero."""
        rng = np.random.RandomState(10)
        x_np = (rng.rand(50, 8) > 0.7)
        x_np[-1, 0] = True  # ensure last row is active
        x = jnp.asarray(x_np, dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)

        n_act = int(cb.n_active[0])
        ids = np.sort(np.array(cb.active_ids[:n_act]))
        expected = np.where(np.any(x_np, axis=1))[0].astype(np.int32)
        np.testing.assert_array_equal(ids, expected)


class TestEdgeCases:
    def test_invalid_ndim(self):
        with pytest.raises(ValueError, match="only supports 1D and 2D"):
            CompactBinary.from_array(jnp.zeros((2, 3, 4)))

    def test_invalid_bit_width(self):
        with pytest.raises(ValueError, match="Only bit_width=32"):
            CompactBinary.from_array(jnp.zeros(10), bit_width=16)

    def test_repr(self):
        cb = CompactBinary.from_array(jnp.zeros(10, dtype=jnp.bool_))
        r = repr(cb)
        assert "CompactBinary" in r
        assert "shape=(10,)" in r
