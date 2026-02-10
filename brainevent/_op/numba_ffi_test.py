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

"""Tests for Numba CPU FFI integration with JAX."""

import importlib.util
import os
import unittest

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import brainstate

cpu_platform = jax.default_backend() == 'cpu'
if not cpu_platform:
    pytest.skip(allow_module_level=True, reason='Numba CPU FFI tests only run on CPU platform')

from brainevent._op.numba_ffi import (
    _ensure_sequence,
    _normalize_shapes_and_dtypes,
    _numpy_from_buffer,
    _XLA_FFI_DTYPE_TO_NUMPY,
    numba_kernel,
    NumbaCpuFfiHandler,
)

numba_installed = importlib.util.find_spec('numba') is not None


class TestHelperFunctions(unittest.TestCase):
    """Tests for helper functions that don't require Numba."""

    def test_ensure_sequence_with_single_value(self):
        """Test _ensure_sequence with a single ShapeDtypeStruct."""
        single = jax.ShapeDtypeStruct((10,), jnp.float32)
        result = _ensure_sequence(single)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], single)

    def test_ensure_sequence_with_list(self):
        """Test _ensure_sequence with a list of ShapeDtypeStruct."""
        lst = [
            jax.ShapeDtypeStruct((10,), jnp.float32),
            jax.ShapeDtypeStruct((20,), jnp.int32),
        ]
        result = _ensure_sequence(lst)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_ensure_sequence_with_tuple(self):
        """Test _ensure_sequence with a tuple of ShapeDtypeStruct."""
        tpl = (
            jax.ShapeDtypeStruct((10,), jnp.float32),
            jax.ShapeDtypeStruct((20,), jnp.int32),
        )
        result = _ensure_sequence(tpl)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_normalize_shapes_and_dtypes_valid(self):
        """Test _normalize_shapes_and_dtypes with valid inputs."""
        shapes = [(10, 20), (5,)]
        dtypes = [np.float32, np.int64]
        norm_shapes, norm_dtypes = _normalize_shapes_and_dtypes(shapes, dtypes, 'test')

        self.assertEqual(norm_shapes, ((10, 20), (5,)))
        self.assertEqual(norm_dtypes, (np.dtype(np.float32), np.dtype(np.int64)))

    def test_normalize_shapes_and_dtypes_mismatched_length(self):
        """Test _normalize_shapes_and_dtypes with mismatched lengths."""
        shapes = [(10,), (20,)]
        dtypes = [np.float32]

        with self.assertRaises(ValueError) as ctx:
            _normalize_shapes_and_dtypes(shapes, dtypes, 'input')
        self.assertIn('input', str(ctx.exception))

    def test_normalize_shapes_and_dtypes_converts_jax_dtypes(self):
        """Test _normalize_shapes_and_dtypes converts JAX dtypes to numpy."""
        shapes = [(10,)]
        dtypes = [jnp.float32]
        _, norm_dtypes = _normalize_shapes_and_dtypes(shapes, dtypes, 'test')

        self.assertIsInstance(norm_dtypes[0], np.dtype)

    def test_xla_ffi_dtype_mapping(self):
        """Test that XLA FFI dtype mapping contains expected types."""
        self.assertEqual(_XLA_FFI_DTYPE_TO_NUMPY[1], np.dtype(np.bool_))
        self.assertEqual(_XLA_FFI_DTYPE_TO_NUMPY[11], np.dtype(np.float32))
        self.assertEqual(_XLA_FFI_DTYPE_TO_NUMPY[12], np.dtype(np.float64))
        self.assertEqual(_XLA_FFI_DTYPE_TO_NUMPY[4], np.dtype(np.int32))
        self.assertEqual(_XLA_FFI_DTYPE_TO_NUMPY[5], np.dtype(np.int64))

    def test_numpy_from_buffer_1d(self):
        """Test _numpy_from_buffer with 1D array."""
        original = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        data_ptr = original.ctypes.data
        shape = (4,)
        dtype = np.dtype(np.float32)

        result = _numpy_from_buffer(data_ptr, shape, dtype)

        self.assertEqual(result.shape, shape)
        self.assertEqual(result.dtype, dtype)
        np.testing.assert_array_equal(result, original)

    def test_numpy_from_buffer_2d(self):
        """Test _numpy_from_buffer with 2D array."""
        original = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        data_ptr = original.ctypes.data
        shape = (3, 2)
        dtype = np.dtype(np.float64)

        result = _numpy_from_buffer(data_ptr, shape, dtype)

        self.assertEqual(result.shape, shape)
        self.assertEqual(result.dtype, dtype)
        np.testing.assert_array_equal(result, original)

    def test_numpy_from_buffer_empty(self):
        """Test _numpy_from_buffer with empty array."""
        shape = (0,)
        dtype = np.dtype(np.float32)

        result = _numpy_from_buffer(0, shape, dtype)

        self.assertEqual(result.shape, shape)
        self.assertEqual(result.dtype, dtype)


@pytest.mark.skipif(not numba_installed, reason="Numba not installed")
class TestNumbaKernelBasic(unittest.TestCase):
    """Basic functionality tests for numba_kernel."""

    def test_element_wise_addition(self):
        """Test simple element-wise addition kernel."""
        import numba

        @numba.njit
        def add_kernel(x, y, out):
            for i in range(out.size):
                out[i] = x[i] + y[i]

        n = 64
        kernel = numba_kernel(
            add_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
        )

        a = jnp.arange(n, dtype=jnp.float32)
        b = jnp.ones(n, dtype=jnp.float32) * 2
        result = kernel(a, b)
        # numba_kernel always returns tuple
        result = result[0] if isinstance(result, tuple) else result
        expected = a + b

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((a, b, result, expected))

    def test_element_wise_multiplication(self):
        """Test element-wise multiplication kernel."""
        import numba

        @numba.njit
        def mul_kernel(x, y, out):
            for i in range(out.size):
                out[i] = x[i] * y[i]

        n = 128
        kernel = numba_kernel(
            mul_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
        )

        a = jnp.arange(n, dtype=jnp.float32)
        b = jnp.arange(n, dtype=jnp.float32) * 0.5
        result = kernel(a, b)
        result = result[0] if isinstance(result, tuple) else result
        expected = a * b

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((a, b, result, expected))

    def test_single_input_single_output(self):
        """Test kernel with single input and output."""
        import numba

        @numba.njit
        def square_kernel(x, out):
            for i in range(out.size):
                out[i] = x[i] ** 2

        n = 32
        kernel = numba_kernel(
            square_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
        )

        x = jnp.arange(n, dtype=jnp.float32)
        result = kernel(x)
        result = result[0] if isinstance(result, tuple) else result
        expected = x ** 2

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((x, result, expected))

    def test_in_place_style_operation(self):
        """Test kernel that writes result using [...] indexing."""
        import numba

        @numba.njit
        def copy_kernel(x, out):
            out[...] = x + 1.0

        n = 64
        kernel = numba_kernel(
            copy_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
        )

        x = jnp.arange(n, dtype=jnp.float32)
        result = kernel(x)
        result = result[0] if isinstance(result, tuple) else result
        expected = x + 1.0

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((x, result, expected))


@pytest.mark.skipif(not numba_installed, reason="Numba not installed")
class TestNumbaKernelDtypes(unittest.TestCase):
    """Tests for different data types."""

    def test_float32(self):
        """Test kernel with float32."""
        import numba

        @numba.njit
        def copy_kernel(x, out):
            out[...] = x

        n = 64
        kernel = numba_kernel(
            copy_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
        )

        x = jnp.arange(n, dtype=jnp.float32)
        result = kernel(x)
        result = result[0] if isinstance(result, tuple) else result
        self.assertEqual(result.dtype, jnp.float32)
        self.assertTrue(jnp.allclose(result, x))
        jax.block_until_ready((x, result))

    def test_float64(self):
        """Test kernel with float64."""
        import numba

        @numba.njit
        def copy_kernel(x, out):
            out[...] = x

        n = 64
        kernel = numba_kernel(
            copy_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float64),
        )

        with brainstate.environ.context(precision=64):
            x = jnp.arange(n, dtype=jnp.float64)
            result = kernel(x)
            result = result[0] if isinstance(result, tuple) else result
            self.assertEqual(result.dtype, jnp.float64)
            self.assertTrue(jnp.allclose(result, x))
            jax.block_until_ready((x, result))

    def test_int32(self):
        """Test kernel with int32."""
        import numba

        @numba.njit
        def add_one_kernel(x, out):
            for i in range(out.size):
                out[i] = x[i] + 1

        n = 64
        kernel = numba_kernel(
            add_one_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.int32),
        )

        x = jnp.arange(n, dtype=jnp.int32)
        result = kernel(x)
        result = result[0] if isinstance(result, tuple) else result
        self.assertEqual(result.dtype, jnp.int32)
        self.assertTrue(jnp.allclose(result, x + 1))
        jax.block_until_ready((x, result))

    def test_int64(self):
        """Test kernel with int64."""
        import numba

        @numba.njit
        def double_kernel(x, out):
            for i in range(out.size):
                out[i] = x[i] * 2

        n = 64
        kernel = numba_kernel(
            double_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.int64),
        )

        with brainstate.environ.context(precision=64):
            x = jnp.arange(n, dtype=jnp.int64)
            result = kernel(x)
            result = result[0] if isinstance(result, tuple) else result
            self.assertEqual(result.dtype, jnp.int64)
            self.assertTrue(jnp.allclose(result, x * 2))
            jax.block_until_ready((x, result))

    def test_mixed_dtypes_input_output(self):
        """Test kernel with different input and output dtypes."""
        import numba

        @numba.njit
        def cast_kernel(x, out):
            for i in range(out.size):
                out[i] = int(x[i])

        n = 32
        kernel = numba_kernel(
            cast_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.int32),
        )

        x = jnp.arange(n, dtype=jnp.float32)
        result = kernel(x)
        result = result[0] if isinstance(result, tuple) else result
        self.assertEqual(result.dtype, jnp.int32)
        jax.block_until_ready((x, result))


@pytest.mark.skipif(not numba_installed, reason="Numba not installed")
class TestNumbaKernelMultidimensional(unittest.TestCase):
    """Tests for multi-dimensional arrays."""

    def test_2d_array(self):
        """Test kernel with 2D array."""
        import numba

        @numba.njit
        def transpose_kernel(x, out):
            rows, cols = x.shape
            for i in range(rows):
                for j in range(cols):
                    out[j, i] = x[i, j]

        rows, cols = 4, 8
        kernel = numba_kernel(
            transpose_kernel,
            outs=jax.ShapeDtypeStruct((cols, rows), jnp.float32),
        )

        x = jnp.arange(rows * cols, dtype=jnp.float32).reshape(rows, cols)
        result = kernel(x)
        result = result[0] if isinstance(result, tuple) else result
        expected = x.T

        self.assertEqual(result.shape, (cols, rows))
        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((x, result, expected))

    def test_3d_array(self):
        """Test kernel with 3D array."""
        import numba

        @numba.njit
        def sum_along_axis_kernel(x, out):
            d0, d1, d2 = x.shape
            for i in range(d0):
                for j in range(d2):
                    total = 0.0
                    for k in range(d1):
                        total += x[i, k, j]
                    out[i, j] = total

        shape = (2, 3, 4)
        kernel = numba_kernel(
            sum_along_axis_kernel,
            outs=jax.ShapeDtypeStruct((2, 4), jnp.float32),
        )

        x = jnp.arange(24, dtype=jnp.float32).reshape(shape)
        result = kernel(x)
        result = result[0] if isinstance(result, tuple) else result
        expected = jnp.sum(x, axis=1)

        self.assertEqual(result.shape, (2, 4))
        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((x, result, expected))

    def test_matrix_vector_multiply(self):
        """Test matrix-vector multiplication kernel."""
        import numba

        @numba.njit
        def matvec_kernel(A, x, out):
            rows, cols = A.shape
            for i in range(rows):
                total = 0.0
                for j in range(cols):
                    total += A[i, j] * x[j]
                out[i] = total

        m, n = 16, 8
        kernel = numba_kernel(
            matvec_kernel,
            outs=jax.ShapeDtypeStruct((m,), jnp.float32),
        )

        A = jnp.arange(m * n, dtype=jnp.float32).reshape(m, n)
        x = jnp.ones(n, dtype=jnp.float32)
        result = kernel(A, x)
        result = result[0] if isinstance(result, tuple) else result
        expected = A @ x

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((A, x, result, expected))


@pytest.mark.skipif(not numba_installed, reason="Numba not installed")
class TestNumbaKernelMultipleOutputs(unittest.TestCase):
    """Tests for kernels with multiple outputs."""

    def test_two_outputs(self):
        """Test kernel with two output arrays."""
        import numba

        @numba.njit
        def split_kernel(x, out1, out2):
            for i in range(out1.size):
                out1[i] = x[i] * 2
                out2[i] = x[i] * 3

        n = 64
        kernel = numba_kernel(
            split_kernel,
            outs=[
                jax.ShapeDtypeStruct((n,), jnp.float32),
                jax.ShapeDtypeStruct((n,), jnp.float32),
            ],
        )

        x = jnp.arange(n, dtype=jnp.float32)
        out1, out2 = kernel(x)

        self.assertTrue(jnp.allclose(out1, x * 2))
        self.assertTrue(jnp.allclose(out2, x * 3))
        jax.block_until_ready((x, out1, out2))

    def test_three_outputs_different_shapes(self):
        """Test kernel with three outputs of different shapes."""
        import numba

        @numba.njit
        def multi_output_kernel(x, sum_out, mean_out, count_out):
            total = 0.0
            for i in range(x.size):
                total += x[i]
            sum_out[0] = total
            mean_out[0] = total / x.size
            count_out[0] = x.size

        n = 100
        kernel = numba_kernel(
            multi_output_kernel,
            outs=[
                jax.ShapeDtypeStruct((1,), jnp.float32),
                jax.ShapeDtypeStruct((1,), jnp.float32),
                jax.ShapeDtypeStruct((1,), jnp.int32),
            ],
        )

        x = jnp.arange(n, dtype=jnp.float32)
        sum_out, mean_out, count_out = kernel(x)

        self.assertTrue(jnp.allclose(sum_out[0], jnp.sum(x)))
        self.assertTrue(jnp.allclose(mean_out[0], jnp.mean(x)))
        self.assertEqual(count_out[0], n)
        jax.block_until_ready((x, sum_out, mean_out, count_out))

    def test_outputs_different_dtypes(self):
        """Test kernel with outputs of different dtypes."""
        import numba

        @numba.njit
        def mixed_dtype_kernel(x, float_out, int_out):
            for i in range(x.size):
                float_out[i] = x[i] * 1.5
                int_out[i] = int(x[i])

        n = 32
        kernel = numba_kernel(
            mixed_dtype_kernel,
            outs=[
                jax.ShapeDtypeStruct((n,), jnp.float64),
                jax.ShapeDtypeStruct((n,), jnp.int64),
            ],
        )

        with brainstate.environ.context(precision=64):
            x = jnp.arange(n, dtype=jnp.float32)
            float_out, int_out = kernel(x)

            self.assertEqual(float_out.dtype, jnp.float64)
            self.assertEqual(int_out.dtype, jnp.int64)
            jax.block_until_ready((x, float_out, int_out))


@pytest.mark.skipif(not numba_installed, reason="Numba not installed")
class TestNumbaKernelMultipleInputs(unittest.TestCase):
    """Tests for kernels with multiple inputs."""

    def test_three_inputs(self):
        """Test kernel with three input arrays."""
        import numba

        @numba.njit
        def weighted_sum_kernel(a, b, c, out):
            for i in range(out.size):
                out[i] = a[i] + 2 * b[i] + 3 * c[i]

        n = 64
        kernel = numba_kernel(
            weighted_sum_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
        )

        a = jnp.arange(n, dtype=jnp.float32)
        b = jnp.ones(n, dtype=jnp.float32)
        c = jnp.full(n, 2.0, dtype=jnp.float32)
        result = kernel(a, b, c)
        result = result[0] if isinstance(result, tuple) else result
        expected = a + 2 * b + 3 * c

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((a, b, c, result, expected))

    def test_scalar_like_input(self):
        """Test kernel with scalar-like (1-element) array input."""
        import numba

        @numba.njit
        def scale_kernel(x, scale, out):
            s = scale[0]
            for i in range(out.size):
                out[i] = x[i] * s

        n = 64
        kernel = numba_kernel(
            scale_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
        )

        x = jnp.arange(n, dtype=jnp.float32)
        scale = jnp.array([3.0], dtype=jnp.float32)
        result = kernel(x, scale)
        result = result[0] if isinstance(result, tuple) else result
        expected = x * 3.0

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((x, scale, result, expected))


@pytest.mark.skipif(not numba_installed, reason="Numba not installed")
class TestNumbaKernelJaxJit(unittest.TestCase):
    """Tests for usage with jax.jit."""

    def test_inside_jax_jit(self):
        """Test kernel inside @jax.jit decorated function."""
        import numba

        @numba.njit
        def add_kernel(x, y, out):
            out[...] = x + y

        n = 64
        kernel = numba_kernel(
            add_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
        )

        @jax.jit
        def jitted_add(a, b):
            result = kernel(a, b)
            return result[0] if isinstance(result, tuple) else result

        a = jnp.arange(n, dtype=jnp.float32)
        b = jnp.ones(n, dtype=jnp.float32)
        result = jitted_add(a, b)
        expected = a + b

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((a, b, result, expected))

    def test_multiple_calls_in_jit(self):
        """Test multiple kernel calls inside jax.jit."""
        import numba

        @numba.njit
        def add_kernel(x, y, out):
            out[...] = x + y

        @numba.njit
        def mul_kernel(x, y, out):
            for i in range(out.size):
                out[i] = x[i] * y[i]

        n = 64
        add_k = numba_kernel(
            add_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
        )
        mul_k = numba_kernel(
            mul_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
        )

        @jax.jit
        def combined(a, b, c):
            temp = add_k(a, b)
            temp = temp[0] if isinstance(temp, tuple) else temp
            result = mul_k(temp, c)
            return result[0] if isinstance(result, tuple) else result

        a = jnp.arange(n, dtype=jnp.float32)
        b = jnp.ones(n, dtype=jnp.float32)
        c = jnp.full(n, 2.0, dtype=jnp.float32)

        result = combined(a, b, c)
        expected = (a + b) * c

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((a, b, c, result, expected))

    def test_kernel_with_jax_operations(self):
        """Test kernel combined with standard JAX operations."""
        import numba

        @numba.njit
        def custom_op_kernel(x, out):
            for i in range(out.size):
                out[i] = x[i] ** 2 + 1

        n = 64
        kernel = numba_kernel(
            custom_op_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
        )

        @jax.jit
        def mixed_ops(x):
            y = jnp.sin(x)
            z = kernel(y)
            z = z[0] if isinstance(z, tuple) else z
            return jnp.sum(z)

        x = jnp.linspace(0, jnp.pi, n, dtype=jnp.float32)
        result = mixed_ops(x)

        y = jnp.sin(x)
        expected = jnp.sum(y ** 2 + 1)

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((x, result, y, expected))


@pytest.mark.skipif(not numba_installed, reason="Numba not installed")
class TestNumbaKernelErrors(unittest.TestCase):
    """Tests for error handling."""

    def test_non_numba_function_raises(self):
        """Test that non-Numba function raises AssertionError."""

        def regular_function(x, out):
            pass

        with self.assertRaises(AssertionError):
            numba_kernel(
                regular_function,
                outs=jax.ShapeDtypeStruct((64,), jnp.float32),
            )

    def test_lambda_raises(self):
        """Test that lambda function raises AssertionError."""
        with self.assertRaises(AssertionError):
            numba_kernel(
                lambda x, out: None,
                outs=jax.ShapeDtypeStruct((64,), jnp.float32),
            )


@pytest.mark.skipif(not numba_installed, reason="Numba not installed")
class TestNumbaKernelParallel(unittest.TestCase):
    """Tests for parallel Numba kernels."""

    def test_parallel_prange(self):
        """Test kernel using numba.prange for parallelism."""
        import numba

        @numba.njit(parallel=True)
        def parallel_add_kernel(x, y, out):
            for i in numba.prange(out.size):
                out[i] = x[i] + y[i]

        n = 1024
        kernel = numba_kernel(
            parallel_add_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
        )

        a = jnp.arange(n, dtype=jnp.float32)
        b = jnp.ones(n, dtype=jnp.float32)
        result = kernel(a, b)
        result = result[0] if isinstance(result, tuple) else result
        expected = a + b

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((a, b, result, expected))


@pytest.mark.skipif(not numba_installed, reason="Numba not installed")
class TestNumbaKernelVmapMethod(unittest.TestCase):
    """Tests for vmap_method parameter."""

    def test_vmap_method_broadcast_all(self):
        """Test kernel with vmap_method='broadcast_all'."""
        import numba

        @numba.njit
        def add_kernel(x, y, out):
            out[...] = x + y

        n = 32
        kernel = numba_kernel(
            add_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
            vmap_method='broadcast_all',
        )

        a = jnp.arange(n, dtype=jnp.float32)
        b = jnp.ones(n, dtype=jnp.float32)
        result = kernel(a, b)
        result = result[0] if isinstance(result, tuple) else result
        expected = a + b

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((a, b, result, expected))


@pytest.mark.skipif(not numba_installed, reason="Numba not installed")
class TestNumbaCpuFfiHandler(unittest.TestCase):
    """Tests for NumbaCpuFfiHandler class."""

    def test_handler_attributes(self):
        """Test that handler stores correct attributes."""
        import numba

        @numba.njit
        def dummy_kernel(x, out):
            pass

        input_shapes = ((10,),)
        input_dtypes = (np.dtype(np.float32),)
        output_shapes = ((10,),)
        output_dtypes = (np.dtype(np.float32),)

        handler = NumbaCpuFfiHandler(
            name="test_handler",
            kernel=dummy_kernel,
            input_shapes=input_shapes,
            input_dtypes=input_dtypes,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
        )

        self.assertEqual(handler.name, "test_handler")
        self.assertEqual(handler.input_shapes, input_shapes)
        self.assertEqual(handler.input_dtypes, input_dtypes)
        self.assertEqual(handler.output_shapes, output_shapes)
        self.assertEqual(handler.output_dtypes, output_dtypes)


@pytest.mark.skipif(not numba_installed, reason="Numba not installed")
class TestNumbaKernelReduction(unittest.TestCase):
    """Tests for reduction operations."""

    def test_sum_reduction(self):
        """Test sum reduction kernel."""
        import numba

        @numba.njit
        def sum_kernel(x, out):
            total = 0.0
            for i in range(x.size):
                total += x[i]
            out[0] = total

        n = 100
        kernel = numba_kernel(
            sum_kernel,
            outs=jax.ShapeDtypeStruct((1,), jnp.float32),
        )

        x = jnp.arange(n, dtype=jnp.float32)
        result = kernel(x)
        result = result[0] if isinstance(result, tuple) else result
        expected = jnp.sum(x)

        self.assertTrue(jnp.allclose(result[0], expected))
        jax.block_until_ready((x, result, expected))

    def test_max_reduction(self):
        """Test max reduction kernel."""
        import numba

        @numba.njit
        def max_kernel(x, out):
            max_val = x[0]
            for i in range(1, x.size):
                if x[i] > max_val:
                    max_val = x[i]
            out[0] = max_val

        kernel = numba_kernel(
            max_kernel,
            outs=jax.ShapeDtypeStruct((1,), jnp.float32),
        )

        x = jnp.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], dtype=jnp.float32)
        result = kernel(x)
        result = result[0] if isinstance(result, tuple) else result

        self.assertTrue(jnp.allclose(result[0], 9.0))
        jax.block_until_ready((x, result))


@pytest.mark.skipif(not numba_installed, reason="Numba not installed")
class TestNumbaKernelInputOutputAliases(unittest.TestCase):
    """Tests for input_output_aliases parameter."""

    def test_input_output_alias(self):
        """Test kernel with input-output aliasing."""
        import numba

        @numba.njit
        def inplace_add_kernel(x, y, out):
            for i in range(out.size):
                out[i] = x[i] + y[i]

        n = 64
        kernel = numba_kernel(
            inplace_add_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
            input_output_aliases={0: 0},  # alias input 0 to output 0
        )

        a = jnp.arange(n, dtype=jnp.float32)
        b = jnp.ones(n, dtype=jnp.float32)
        result = kernel(a, b)
        result = result[0] if isinstance(result, tuple) else result
        expected = a + b

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((a, b, result, expected))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
