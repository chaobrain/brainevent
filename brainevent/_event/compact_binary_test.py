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

import brainevent._event.compact_binary as compact_binary_mod
from brainevent._event.compact_binary import CompactBinary


class TestConstruction1D:
    def test_basic(self):
        rng = np.random.RandomState(0)
        x = jnp.asarray(rng.rand(100) > 0.7, dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)

        assert cb.packed.dtype == jnp.uint32
        assert cb.packed.shape == ((100 + 31) // 32,)
        assert cb.active_ids.shape == (100,)
        assert cb.active_ids.dtype == jnp.int32
        assert int(cb.n_active[0]) == int(jnp.sum(x))
        np.testing.assert_array_equal(cb.value, x)

    def test_float_input(self):
        x = jnp.array([0.0, 1.0, 0.0, 2.0, 0.0], dtype=jnp.float32)
        cb = CompactBinary.from_array(x)
        assert int(cb.n_active[0]) == 2

    def test_respects_configured_preprocess_backend_for_1d(self, monkeypatch):
        x = jnp.asarray([False, True, False, True], dtype=jnp.bool_)
        recorded = {}
        original = compact_binary_mod.binary_1d_array_index_p_call

        def _spy(spikes, *, backend=None):
            recorded['backend'] = backend
            return original(spikes, backend=backend)

        monkeypatch.setattr(compact_binary_mod, 'binary_1d_array_index_p_call', _spy)
        monkeypatch.setattr(compact_binary_mod, 'COMPACT_BINARY_PREPROCESS_BACKEND', 'jax_raw')
        CompactBinary.from_array(x)
        assert recorded['backend'] == 'jax_raw'


class TestConstruction1DLight:
    def test_skips_compaction_metadata(self):
        x = jnp.asarray([False, True, True, False], dtype=jnp.bool_)
        cb = CompactBinary.from_array_light(x)

        assert cb.packed.shape == (1,)
        np.testing.assert_array_equal(np.asarray(cb.active_ids), np.zeros((4,), dtype=np.int32))
        np.testing.assert_array_equal(np.asarray(cb.n_active), np.zeros((1,), dtype=np.int32))
        np.testing.assert_array_equal(cb.value, x)


class TestCompactOnlyVector:
    def test_basic(self):
        x = jnp.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=jnp.float32)
        cb = CompactBinary.compacy_only_vector(x)

        assert cb.packed.shape == (0,)
        assert cb.packed.dtype == jnp.uint32
        assert int(cb.n_active[0]) == 3
        ids = np.sort(np.asarray(cb.active_ids[:3], dtype=np.int32))
        np.testing.assert_array_equal(ids, np.array([1, 3, 5], dtype=np.int32))
        np.testing.assert_array_equal(cb.value, x)

    def test_invalid_ndim(self):
        with pytest.raises(ValueError, match="only supports 1D arrays"):
            CompactBinary.compacy_only_vector(jnp.zeros((2, 3), dtype=jnp.bool_))


class TestConstruction2D:
    def test_basic(self):
        rng = np.random.RandomState(1)
        x = jnp.asarray(rng.rand(50, 8) > 0.7, dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)

        assert cb.packed.dtype == jnp.uint32
        assert cb.packed.shape == (50, (8 + 31) // 32)
        assert cb.active_ids.shape == (50,)
        assert int(cb.n_active[0]) == int(jnp.sum(jnp.any(x, axis=1)))

    def test_large_batch(self):
        rng = np.random.RandomState(2)
        x = jnp.asarray(rng.rand(20, 100) > 0.8, dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)
        assert cb.packed.shape == (20, (100 + 31) // 32)

    def test_light_matches_from_array_for_2d(self):
        x = jnp.asarray(
            [[True, False, True], [False, False, False], [True, True, False]],
            dtype=jnp.bool_,
        )
        cb = CompactBinary.from_array(x)
        cb_light = CompactBinary.from_array_light(x)
        np.testing.assert_array_equal(cb_light.packed, cb.packed)
        np.testing.assert_array_equal(cb_light.active_ids, cb.active_ids)
        np.testing.assert_array_equal(cb_light.n_active, cb.n_active)


class TestFromPacked:
    def test_roundtrip(self):
        x = jnp.asarray([False, True, False, True], dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)
        rebuilt = CompactBinary.from_packed(
            cb.packed,
            cb.active_ids,
            cb.n_active,
            cb.value,
            n_orig=cb.n_orig,
            batch_size=cb.batch_size,
            bit_width=cb.bit_width,
        )
        np.testing.assert_array_equal(rebuilt.packed, cb.packed)
        np.testing.assert_array_equal(rebuilt.active_ids, cb.active_ids)
        np.testing.assert_array_equal(rebuilt.n_active, cb.n_active)
        np.testing.assert_array_equal(rebuilt.value, cb.value)


class TestAllZeros:
    def test_1d(self):
        cb = CompactBinary.from_array(jnp.zeros(64, dtype=jnp.bool_))
        assert int(cb.n_active[0]) == 0

    def test_2d(self):
        cb = CompactBinary.from_array(jnp.zeros((32, 8), dtype=jnp.bool_))
        assert int(cb.n_active[0]) == 0


class TestAllOnes:
    def test_1d(self):
        cb = CompactBinary.from_array(jnp.ones(64, dtype=jnp.bool_))
        assert int(cb.n_active[0]) == 64
        np.testing.assert_array_equal(
            np.sort(np.asarray(cb.active_ids[:64], dtype=np.int32)),
            np.arange(64),
        )

    def test_2d(self):
        cb = CompactBinary.from_array(jnp.ones((32, 8), dtype=jnp.bool_))
        assert int(cb.n_active[0]) == 32
        np.testing.assert_array_equal(
            np.sort(np.asarray(cb.active_ids[:32], dtype=np.int32)),
            np.arange(32),
        )


class TestPyTreeRoundTrip:
    def test_flatten_unflatten(self):
        x = jnp.asarray(np.random.RandomState(3).rand(40) > 0.5, dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)
        leaves, treedef = jax.tree.flatten(cb)
        assert len(leaves) == 4

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
        assert len(jax.tree.leaves(shapes)) == 4


class TestJitCompatibility:
    def test_jit_identity(self):
        x = jnp.asarray(np.random.RandomState(5).rand(50) > 0.6, dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)
        cb2 = jax.jit(lambda obj: obj)(cb)
        np.testing.assert_array_equal(cb2.value, cb.value)
        np.testing.assert_array_equal(cb2.packed, cb.packed)
        np.testing.assert_array_equal(cb2.n_active, cb.n_active)

    def test_jit_2d(self):
        x = jnp.asarray(np.random.RandomState(6).rand(20, 8) > 0.7, dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)
        cb2 = jax.jit(lambda obj: obj)(cb)
        np.testing.assert_array_equal(cb2.value, cb.value)


class TestProperties:
    def test_1d_properties(self):
        cb = CompactBinary.from_array(jnp.ones(100, dtype=jnp.bool_))
        assert cb.shape == (100,)
        assert cb.ndim == 1
        assert cb.n_orig == 100
        assert cb.batch_size is None
        assert cb.dtype == jnp.bool_
        assert cb.size == 100
        assert cb.bit_width == 32

    def test_2d_properties(self):
        cb = CompactBinary.from_array(jnp.ones((50, 8), dtype=jnp.bool_))
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
        x = jnp.asarray([True, False, True, True, False] + [False] * 27, dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)
        word = int(cb.packed[0])
        for bit in range(32):
            assert ((word >> bit) & 1) == int(x[bit])

    def test_2d_bits(self):
        rng = np.random.RandomState(8)
        x = jnp.asarray(rng.rand(10, 40) > 0.5, dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)
        packed_np = np.asarray(cb.packed)
        x_np = np.asarray(x)
        for i in range(10):
            for w in range(packed_np.shape[1]):
                word = int(packed_np[i, w])
                for bit in range(32):
                    col = w * 32 + bit
                    if col < 40:
                        assert ((word >> bit) & 1) == int(x_np[i, col])


class TestCompactionCorrectness:
    def test_1d_n_active_count(self):
        rng = np.random.RandomState(9)
        x = jnp.asarray(rng.rand(100) > 0.7, dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)
        assert int(cb.n_active[0]) == int(jnp.sum(x))

    def test_1d_active_ids(self):
        rng = np.random.RandomState(11)
        x_np = rng.rand(100) > 0.7
        x_np[-1] = True
        x = jnp.asarray(x_np, dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)

        ids = np.sort(np.asarray(cb.active_ids[: int(cb.n_active[0])], dtype=np.int32))
        np.testing.assert_array_equal(ids, np.where(x_np)[0].astype(np.int32))

    def test_2d_active_rows(self):
        rng = np.random.RandomState(10)
        x_np = rng.rand(50, 8) > 0.7
        x_np[-1, 0] = True
        x = jnp.asarray(x_np, dtype=jnp.bool_)
        cb = CompactBinary.from_array(x)

        ids = np.sort(np.asarray(cb.active_ids[: int(cb.n_active[0])], dtype=np.int32))
        np.testing.assert_array_equal(ids, np.where(np.any(x_np, axis=1))[0].astype(np.int32))


class TestEdgeCases:
    def test_invalid_ndim(self):
        with pytest.raises(ValueError, match="only supports 1D and 2D"):
            CompactBinary.from_array(jnp.zeros((2, 3, 4)))

    def test_invalid_bit_width(self):
        with pytest.raises(ValueError, match="Only bit_width=32"):
            CompactBinary.from_array(jnp.zeros(10), bit_width=16)

    def test_repr(self):
        cb = CompactBinary.from_array(jnp.zeros(10, dtype=jnp.bool_))
        text = repr(cb)
        assert "CompactBinary" in text
        assert "shape=(10,)" in text
