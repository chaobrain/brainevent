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

import numpy as np
import pytest

import brainevent
from brainevent import BinaryArray, EventArray


class TestBinaryArray:
    def test_initialization(self):
        """Test basic initialization of BinaryArray."""
        # Create from numpy array
        data = np.array([0, 1, 0, 1], dtype=np.uint8)
        binary_array = BinaryArray(data)
        assert binary_array.shape == (4,)
        assert np.array_equal(binary_array.value, data)

        # Test if EventArray is an alias for BinaryArray
        event_array = EventArray(data)
        assert type(event_array) == type(binary_array)

        # Test with 2D array
        data_2d = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        binary_array_2d = BinaryArray(data_2d)
        assert binary_array_2d.shape == (2, 2)
        assert np.array_equal(binary_array_2d.value, data_2d)

    def test_matmul_1d_binary_with_2d_dense(self):
        """Test matrix multiplication between 1D binary array and 2D dense array."""
        # 1D binary array (1x3) @ 2D dense array (3x2)
        binary_vec = BinaryArray(np.array([0, 1, 1], dtype=np.uint8))
        dense_mat = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        result = binary_vec @ dense_mat
        expected = np.array([8.0, 10.0])  # [0*1 + 1*3 + 1*5, 0*2 + 1*4 + 1*6]

        assert np.allclose(result, expected)

    def test_matmul_2d_binary_with_2d_dense(self):
        """Test matrix multiplication between 2D binary array and 2D dense array."""
        # 2D binary array (2x3) @ 2D dense array (3x2)
        binary_mat = BinaryArray(np.array([[0, 1, 1], [1, 0, 1]], dtype=np.uint8))
        dense_mat = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        result = binary_mat @ dense_mat
        expected = np.array(
            [[8.0, 10.0], [6.0, 8.0]])  # [[0*1 + 1*3 + 1*5, 0*2 + 1*4 + 1*6], [1*1 + 0*3 + 1*5, 1*2 + 0*4 + 1*6]]

        assert np.allclose(result, expected)

    def test_rmatmul_2d_dense_with_1d_binary(self):
        """Test reverse matrix multiplication between 2D dense array and 1D binary array."""
        # 2D dense array (2x3) @ 1D binary array (3x1)
        dense_mat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        binary_vec = BinaryArray(np.array([0, 1, 1], dtype=np.uint8))

        result = dense_mat @ binary_vec
        expected = np.array([5.0, 11.0])  # [1*0 + 2*1 + 3*1, 4*0 + 5*1 + 6*1]

        assert np.allclose(result, expected)

    def test_rmatmul_2d_dense_with_2d_binary(self):
        """Test reverse matrix multiplication between 2D dense array and 2D binary array."""
        # 2D dense array (2x3) @ 2D binary array (3x2)
        dense_mat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        binary_mat = BinaryArray(np.array([[0, 1], [1, 0], [1, 1]], dtype=np.uint8))

        result = dense_mat @ binary_mat
        expected = np.array(
            [[5.0, 4.0], [11.0, 10.0]])  # [[1*0 + 2*1 + 3*1, 1*1 + 2*0 + 3*1], [4*0 + 5*1 + 6*1, 4*1 + 5*0 + 6*1]]

        assert np.allclose(result, expected)

    def test_imatmul(self):
        """Test in-place matrix multiplication."""
        binary_mat = BinaryArray(np.array([[0, 1, 1], [1, 0, 1]], dtype=np.uint8))
        dense_mat = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        # Store original value for comparison
        original_id = id(binary_mat)

        # Perform in-place operation
        with pytest.raises(brainevent.MathError):
            binary_mat @= dense_mat

            # Check result
            expected = np.array([[8.0, 10.0], [6.0, 8.0]])
            assert np.allclose(binary_mat.value, expected)

            # Ensure it's the same object (in-place)
            assert id(binary_mat) == original_id

    def test_error_conditions(self):
        """Test error conditions for matrix multiplication."""
        # Test with incompatible dimensions
        binary_mat = BinaryArray(np.array([[0, 1, 1], [1, 0, 1]], dtype=np.uint8))
        dense_mat_wrong_dim = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2x2 matrix

        with pytest.raises(AssertionError):
            _ = binary_mat @ dense_mat_wrong_dim

        # Test with scalar array (0D)
        scalar_array = BinaryArray(np.array(1, dtype=np.uint8))
        dense_mat = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(brainevent.MathError):
            _ = scalar_array @ dense_mat

        # Test with 3D array
        array_3d = np.zeros((2, 2, 2), dtype=np.uint8)
        binary_array_3d = BinaryArray(array_3d)

        with pytest.raises(brainevent.MathError):
            _ = binary_array_3d @ dense_mat

        # Test right operand with wrong dimension
        dense_vec = np.array([1.0, 2.0, 3.0])  # 1D array

        with pytest.raises(AssertionError):
            _ = binary_mat @ dense_vec


class TestBinaryArrayIndexed:
    @pytest.mark.xfail(reason="binary_array_index kernel not available on all platforms")
    def test_indexed_construction(self):
        """Test that indexed=True computes spike_indices and spike_count."""
        import jax.numpy as jnp
        data = jnp.array([0, 1, 0, 1, 1], dtype=jnp.float32)
        arr = BinaryArray(data, indexed=True)
        assert arr.indexed is True
        assert arr.spike_indices is not None
        assert arr.spike_count is not None

    def test_non_indexed_construction(self):
        """Test that indexed=False (default) has no spike_indices/spike_count."""
        data = np.array([0, 1, 0, 1], dtype=np.float32)
        arr = BinaryArray(data)
        assert arr.indexed is False
        assert arr.spike_indices is None
        assert arr.spike_count is None

    @pytest.mark.xfail(reason="binary_array_index kernel not available on all platforms")
    def test_indexed_immutability_setitem(self):
        """Test that __setitem__ raises for indexed arrays."""
        import jax.numpy as jnp
        data = jnp.array([0, 1, 0, 1], dtype=jnp.float32)
        arr = BinaryArray(data, indexed=True)
        with pytest.raises(NotImplementedError):
            arr[0] = 1

    @pytest.mark.xfail(reason="binary_array_index kernel not available on all platforms")
    def test_indexed_immutability_update(self):
        """Test that _update raises for indexed arrays."""
        import jax.numpy as jnp
        data = jnp.array([0, 1, 0, 1], dtype=jnp.float32)
        arr = BinaryArray(data, indexed=True)
        with pytest.raises(NotImplementedError):
            arr.value = jnp.array([1, 0, 1, 0], dtype=jnp.float32)

    def test_non_indexed_mutable(self):
        """Test that non-indexed arrays allow mutation."""
        import jax.numpy as jnp
        data = jnp.array([0, 1, 0, 1], dtype=jnp.float32)
        arr = BinaryArray(data)
        arr.value = jnp.array([1, 0, 1, 0], dtype=jnp.float32)
        assert np.array_equal(arr.value, np.array([1, 0, 1, 0]))

    def test_pytree_roundtrip_non_indexed(self):
        """Test JAX pytree flatten/unflatten for non-indexed arrays."""
        import jax
        data = np.array([0, 1, 0, 1], dtype=np.float32)
        arr = BinaryArray(data)
        leaves, treedef = jax.tree.flatten(arr)
        arr2 = jax.tree.unflatten(treedef, leaves)
        assert isinstance(arr2, BinaryArray)
        assert arr2.indexed is False
        assert np.array_equal(arr2.value, data)

    @pytest.mark.xfail(reason="binary_array_index kernel not available on all platforms")
    def test_pytree_roundtrip_indexed(self):
        """Test JAX pytree flatten/unflatten for indexed arrays."""
        import jax
        import jax.numpy as jnp
        data = jnp.array([0, 1, 0, 1, 1], dtype=jnp.float32)
        arr = BinaryArray(data, indexed=True)
        leaves, treedef = jax.tree.flatten(arr)
        arr2 = jax.tree.unflatten(treedef, leaves)
        assert isinstance(arr2, BinaryArray)
        assert arr2.indexed is True
        assert arr2.spike_indices is not None
        assert arr2.spike_count is not None
        assert np.array_equal(arr2.value, data)
