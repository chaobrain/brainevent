import numpy as np
import pytest

from brainevent import SparseFloat, MathError


class TestSparseFloatMatMul:
    def setup_method(self):
        # Create test arrays
        self.vector = SparseFloat(np.array([1.0, 2.0, 3.0]))
        self.matrix = SparseFloat(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        self.dense_vector = np.array([1.0, 2.0, 3.0])
        self.dense_matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.dense_matrix2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.square_matrix = SparseFloat(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
        self.scalar = SparseFloat(np.array(5.0))

    def test_vector_matmul_matrix(self):
        # Test vector @ matrix
        result = self.vector @ self.dense_matrix
        expected = np.array([22.0, 28.0])
        assert np.allclose(result, expected, rtol=1e-3, atol=1e-3)

    def test_matrix_matmul_vector(self):
        # Test matrix @ vector (using rmatmul)
        with pytest.raises(AssertionError):
            result = self.dense_vector @ self.matrix
            expected = np.array([22.0, 28.0])
            assert np.allclose(result, expected, rtol=1e-3, atol=1e-3)

    # def test_matrix_matmul_matrix(self):
    #     # Test matrix @ matrix
    #     result = self.matrix @ self.dense_matrix2
    #     expected = np.array([[9.0, 12.0, 15.0], [19.0, 26.0, 33.0], [29.0, 40.0, 51.0]])
    #     assert np.allclose(result, expected, rtol=1e-3, atol=1e-3)
    #
    # def test_matrix_rmatmul_matrix(self):
    #     # Test dense_matrix @ matrix (using rmatmul)
    #     result = self.dense_matrix2 @ self.square_matrix
    #     expected = np.array([[30.0, 36.0, 42.0], [66.0, 81.0, 96.0]])
    #     assert np.allclose(result, expected, rtol=1e-3, atol=1e-3)

    def test_imatmul(self):
        # `@=` returns a new immutable wrapper.
        matrix_copy = SparseFloat(self.matrix.value.copy())
        original_id = id(matrix_copy)
        matrix_copy @= self.dense_matrix2
        expected = np.array([[9.0, 12.0, 15.0], [19.0, 26.0, 33.0], [29.0, 40.0, 51.0]])
        assert np.array_equal(matrix_copy.value, expected)
        assert id(matrix_copy) != original_id

    def test_scalar_matmul_error(self):
        # Test error for scalar in matrix multiplication
        with pytest.raises(MathError) as excinfo:
            _ = self.scalar @ self.dense_matrix

    def test_3d_array_matmul_error(self):
        # Test error for 3D array in matrix multiplication
        array_3d = SparseFloat(np.ones((2, 2, 2)))
        with pytest.raises(MathError) as excinfo:
            _ = array_3d @ self.dense_matrix
        assert "Matrix multiplication is only supported for 1D and 2D arrays" in str(excinfo.value)

    def test_incompatible_dimensions_error(self):
        # Test error for incompatible dimensions
        incompatible_matrix = np.ones((4, 4))
        with pytest.raises(AssertionError) as excinfo:
            _ = self.matrix @ incompatible_matrix
        assert "Incompatible dimensions for matrix multiplication" in str(excinfo.value)

    def test_rmatmul_incompatible_dimensions_error(self):
        # Test error for incompatible dimensions in rmatmul
        incompatible_matrix = np.ones((4, 4))
        with pytest.raises(AssertionError) as excinfo:
            _ = incompatible_matrix @ self.matrix
        assert "Incompatible dimensions for matrix multiplication" in str(excinfo.value)

    def test_non_2d_left_operand_error(self):
        # Test error when left operand in rmatmul is not 2D
        vector = np.array([1, 2, 3])
        with pytest.raises(AssertionError) as excinfo:
            _ = vector @ self.matrix
        assert "Left operand must be a 2D array" in str(excinfo.value)

    def test_non_2d_right_operand_error(self):
        # Test error when right operand in matmul is not 2D
        vector = np.array([1, 2])
        with pytest.raises(AssertionError) as excinfo:
            _ = self.matrix @ vector
        assert "Right operand must be a 2D array" in str(excinfo.value)


class TestSparseFloatIndexed:
    def test_indexed_construction(self):
        """Test that indexed=True sets indices placeholder."""
        data = np.array([1.0, 0.0, 3.0], dtype=np.float32)
        arr = SparseFloat(data, indexed=True)
        assert arr.indexed is True
        assert arr.indices is ...

    def test_non_indexed_construction(self):
        """Test that indexed=False (default) has no indices."""
        data = np.array([1.0, 0.0, 3.0], dtype=np.float32)
        arr = SparseFloat(data)
        assert arr.indexed is False
        assert arr.indices is None

    def test_indexed_immutability_setitem(self):
        """Item assignment is unsupported for immutable event arrays."""
        data = np.array([1.0, 0.0, 3.0], dtype=np.float32)
        arr = SparseFloat(data, indexed=True)
        with pytest.raises(TypeError):
            arr[0] = 1.0

    def test_indexed_immutability_update(self):
        """Direct `value` assignment is unsupported for immutable event arrays."""
        data = np.array([1.0, 0.0, 3.0], dtype=np.float32)
        arr = SparseFloat(data, indexed=True)
        with pytest.raises(AttributeError):
            arr.value = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def test_non_indexed_with_value(self):
        """Non-indexed arrays are still immutable and replaced via `with_value`."""
        import jax.numpy as jnp
        data = jnp.array([1.0, 0.0, 3.0], dtype=jnp.float32)
        arr = SparseFloat(data)
        arr2 = arr.with_value(jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32))
        assert np.allclose(arr2.value, np.array([0.0, 1.0, 0.0]))
        assert np.allclose(arr.value, np.array([1.0, 0.0, 3.0]))

    def test_pytree_roundtrip_non_indexed(self):
        """Test JAX pytree flatten/unflatten for non-indexed arrays."""
        import jax
        data = np.array([1.0, 0.0, 3.0], dtype=np.float32)
        arr = SparseFloat(data)
        leaves, treedef = jax.tree.flatten(arr)
        arr2 = jax.tree.unflatten(treedef, leaves)
        assert isinstance(arr2, SparseFloat)
        assert arr2.indexed is False
        assert np.allclose(arr2.value, data)

    def test_pytree_roundtrip_indexed(self):
        """Test JAX pytree flatten/unflatten for indexed arrays."""
        import jax
        data = np.array([1.0, 0.0, 3.0], dtype=np.float32)
        arr = SparseFloat(data, indexed=True)
        leaves, treedef = jax.tree.flatten(arr)
        arr2 = jax.tree.unflatten(treedef, leaves)
        assert isinstance(arr2, SparseFloat)
        assert arr2.indexed is True
        assert arr2.indices is ...
        assert np.allclose(arr2.value, data)
