import numpy as np
import pytest

import brainevent
from brainevent import MaskedFloat, MathError


class TestMaskedFloatMatMul:
    def setup_method(self):
        # Create test arrays
        self.vector = MaskedFloat(np.array([1.0, 2.0, 3.0]))
        self.matrix = MaskedFloat(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        self.dense_vector = np.array([1.0, 2.0, 3.0])
        self.dense_matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.dense_matrix2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.square_matrix = MaskedFloat(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
        self.scalar = MaskedFloat(np.array(5.0))

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
        # Test in-place matrix multiplication
        with pytest.raises(brainevent.MathError):
            matrix_copy = MaskedFloat(self.matrix.value.copy())
            matrix_copy @= self.dense_matrix2
            expected = np.array([[9.0, 12.0, 15.0], [19.0, 26.0, 33.0], [29.0, 40.0, 51.0]])
            assert np.array_equal(matrix_copy.value, expected)

    def test_scalar_matmul_error(self):
        # Test error for scalar in matrix multiplication
        with pytest.raises(MathError) as excinfo:
            _ = self.scalar @ self.dense_matrix

    def test_3d_array_matmul_error(self):
        # Test error for 3D array in matrix multiplication
        array_3d = MaskedFloat(np.ones((2, 2, 2)))
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
