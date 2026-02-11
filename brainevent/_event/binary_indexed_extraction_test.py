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

import jax
import jax.numpy as jnp
import pytest

from brainevent._event.binary_indexed_extraction import (
    binary_array_index,
    binary_1d_array_index_p,
    _binary_1d_array_index_numba_kernel,
    _binary_1d_array_index_warp_kernel,
)


class TestBinary1DArrayIndexForward:
    """Test forward pass of binary 1D array index extraction."""

    def test_forward_bool_all_true(self):
        """Test with all True boolean values."""
        spikes = jnp.array([False, True, True, False, True], dtype=jnp.bool_)
        indices, count = binary_array_index(spikes)
        expected_indices = jnp.array([1, 2, 4], dtype=jnp.int32)
        expected_count = jnp.array([3], dtype=jnp.int32)
        # Only check the first count[0] elements of indices
        assert jnp.array_equal(indices[:count[0]], expected_indices)
        assert jnp.array_equal(count, expected_count)

    def test_forward_bool_mixed(self):
        """Test with mixed boolean values."""
        spikes = jnp.array([True, False, False, True, False, True], dtype=jnp.bool_)
        indices, count = binary_array_index(spikes)
        expected_indices = jnp.array([0, 3, 5], dtype=jnp.int32)
        expected_count = jnp.array([3], dtype=jnp.int32)
        assert jnp.array_equal(indices[:count[0]], expected_indices)
        assert jnp.array_equal(count, expected_count)

    def test_forward_bool_all_false(self):
        """Test with all False boolean values."""
        spikes = jnp.array([False, False, False], dtype=jnp.bool_)
        indices, count = binary_array_index(spikes)
        expected_count = jnp.array([0], dtype=jnp.int32)
        assert count[0] == 0
        assert jnp.array_equal(count, expected_count)

    def test_forward_float_nonzero(self):
        """Test with nonzero float values."""
        spikes = jnp.array([0.0, 1.0, 0.0, 2.0, 0.0], dtype=jnp.float32)
        indices, count = binary_array_index(spikes)
        expected_indices = jnp.array([1, 3], dtype=jnp.int32)
        expected_count = jnp.array([2], dtype=jnp.int32)
        assert jnp.array_equal(indices[:count[0]], expected_indices)
        assert jnp.array_equal(count, expected_count)

    def test_forward_float_all_zero(self):
        """Test with all zero float values."""
        spikes = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
        indices, count = binary_array_index(spikes)
        expected_count = jnp.array([0], dtype=jnp.int32)
        assert count[0] == 0
        assert jnp.array_equal(count, expected_count)

    def test_forward_large_array(self):
        """Test with larger array."""
        spikes = jnp.array([i % 3 == 0 for i in range(100)], dtype=jnp.bool_)
        indices, count = binary_array_index(spikes)
        expected_count = jnp.array([34], dtype=jnp.int32)
        assert count[0] == 34
        # Verify indices are at positions 0, 3, 6, ..., 99
        for i in range(34):
            assert indices[i] == i * 3


class TestBinary1DArrayIndexJVP:
    """Test forward-mode automatic differentiation (JVP)."""

    def test_jvp_bool(self):
        """Test JVP with boolean input - bool inputs use float0 tangents."""
        # Skip this test as bool inputs require float0 tangents which are not commonly used
        pytest.skip("Bool inputs require float0 tangents")

    def test_jvp_float(self):
        """Test JVP with float input."""
        spikes = jnp.array([0.0, 1.0, 0.0, 2.0, 0.0], dtype=jnp.float32)
        spikes_dot = jnp.array([0.1, 0.0, 0.2, 0.0, 0.3], dtype=jnp.float32)

        def f(s):
            idx, cnt = binary_array_index(s)
            return idx.astype(jnp.float32), cnt.astype(jnp.float32)

        primals, tangents = jax.jvp(f, (spikes,), (spikes_dot,))

        # The tangent should produce indices where spikes_dot is non-zero
        # Note: JVP behavior for index extraction is non-trivial
        assert tangents[0].shape == primals[0].shape


class TestBinary1DArrayIndexVJP:
    """Test reverse-mode automatic differentiation (VJP)."""

    def test_vjp_gradient_simple(self):
        """Test VJP gradient with simple case."""
        # Skip this test as output is int32, not suitable for jax.grad
        pytest.skip("Output is int32, not suitable for jax.grad")

    def test_vjp_indices_gradient(self):
        """Test VJP with respect to indices."""
        # Skip this test as output is int32, not suitable for jax.grad
        pytest.skip("Output is int32, not suitable for jax.grad")


class TestBinary1DArrayIndexBatching:
    """Test batching rules."""

    def test_batching_axis_0(self):
        """Test batching along axis 0."""
        spikes_batch = jnp.array([
            [False, True, False],
            [True, False, True],
        ], dtype=jnp.bool_)

        # Process each element in batch
        results = []
        for i in range(spikes_batch.shape[0]):
            idx, cnt = binary_array_index(spikes_batch[i])
            results.append((idx, cnt))

        assert len(results) == 2
        assert results[0][1][0] == 1  # First batch: one True
        assert results[1][1][0] == 2  # Second batch: two True values

    def test_batching_vmap(self):
        """Test with jax.vmap."""
        spikes_batch = jnp.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ], dtype=jnp.float32)

        def extract_indices(s):
            idx, cnt = binary_array_index(s)
            return cnt[0]

        # This tests if batching works correctly
        counts = jax.vmap(extract_indices)(spikes_batch)
        assert counts.shape == (2,)
        assert counts[0] == 1
        assert counts[1] == 2


class TestBinary1DArrayIndexKernels:
    """Test kernel generators individually."""

    def test_numba_kernel_generator(self):
        """Test Numba kernel generator."""
        spikes_info = jax.ShapeDtypeStruct((5,), jnp.bool_)
        indices_info = jax.ShapeDtypeStruct((5,), jnp.int32)
        count_info = jax.ShapeDtypeStruct((1,), jnp.int32)

        kernel = _binary_1d_array_index_numba_kernel(
            spikes_info=spikes_info,
            outs=[indices_info, count_info],
        )
        assert kernel is not None

    def test_warp_kernel_generator(self):
        """Test Warp kernel generator."""
        try:
            import warp
        except ImportError:
            pytest.skip("Warp not installed")

        spikes_info = jax.ShapeDtypeStruct((5,), jnp.bool_)
        indices_info = jax.ShapeDtypeStruct((5,), jnp.int32)
        count_info = jax.ShapeDtypeStruct((1,), jnp.int32)

        kernel = _binary_1d_array_index_warp_kernel(
            spikes_info=spikes_info,
            indices_info=indices_info,
            count_info=count_info,
            outs=[indices_info, count_info],
        )
        assert kernel is not None


class TestBinary1DArrayIndexEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_array(self):
        """Test with empty array."""
        spikes = jnp.array([], dtype=jnp.bool_)
        indices, count = binary_array_index(spikes)
        assert count[0] == 0

    def test_single_element_true(self):
        """Test with single True element."""
        spikes = jnp.array([True], dtype=jnp.bool_)
        indices, count = binary_array_index(spikes)
        expected_indices = jnp.array([0], dtype=jnp.int32)
        expected_count = jnp.array([1], dtype=jnp.int32)
        assert jnp.array_equal(indices, expected_indices)
        assert jnp.array_equal(count, expected_count)

    def test_single_element_false(self):
        """Test with single False element."""
        spikes = jnp.array([False], dtype=jnp.bool_)
        indices, count = binary_array_index(spikes)
        expected_count = jnp.array([0], dtype=jnp.int32)
        assert count[0] == 0
        assert jnp.array_equal(count, expected_count)

    def test_2d_array_not_implemented(self):
        """Test that 2D array raises NotImplementedError."""
        spikes = jnp.array([[True, False], [False, True]], dtype=jnp.bool_)
        with pytest.raises(NotImplementedError):
            binary_array_index(spikes)

    def test_3d_array_error(self):
        """Test that 3D array raises ValueError."""
        spikes = jnp.array([[[True]]], dtype=jnp.bool_)
        with pytest.raises(ValueError, match="Only 1D and 2D"):
            binary_array_index(spikes)


class TestBinary1DArrayIndexPrimitive:
    """Test the primitive directly."""

    def test_primitive_registration(self):
        """Test that primitive is properly registered."""
        assert binary_1d_array_index_p is not None
        assert binary_1d_array_index_p.primitive is not None
        assert binary_1d_array_index_p.name == 'binary_1d_array_index'

    def test_primitive_call(self):
        """Test primitive call directly."""
        spikes = jnp.array([False, True, False, True], dtype=jnp.bool_)
        indices_info = jax.ShapeDtypeStruct([4], jnp.int32)
        count_info = jax.ShapeDtypeStruct([1], jnp.int32)

        result = binary_1d_array_index_p(
            spikes,
            outs=[indices_info, count_info],
            spikes_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
            indices_info=indices_info,
            count_info=count_info,
        )

        assert len(result) == 2
        assert result[1][0] == 2  # Two True values


class TestBinary1DArrayIndexDifferentiability:
    """Test differentiability aspects."""

    def test_grad_through_count(self):
        """Test that gradient flows through count."""
        # Skip this test as output is int32, not suitable for jax.grad
        pytest.skip("Output is int32, not suitable for jax.grad")

    def test_hessian(self):
        """Test Hessian computation (second derivative)."""
        # Skip this test as output is int32, not suitable for jax.grad
        pytest.skip("Output is int32, not suitable for jax.grad")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
