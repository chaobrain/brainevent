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

import ctypes
from ctypes import POINTER
import gc
import importlib.util
import os
import re
import unittest
import warnings
from unittest import mock

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
import brainstate

numba_installed = importlib.util.find_spec('numba') is not None
cpu_platform = jax.default_backend() == 'cpu'
if not cpu_platform or not numba_installed:
    pytest.skip(allow_module_level=True, reason='Numba CPU FFI tests only run on CPU platform with Numba installed')

from brainevent._error import KernelRegistrationError
from brainevent._op import numba_ffi
from brainevent._op.ffi_naming import kernel_content_fingerprint
from brainevent._op.numba_ffi import (
    _ensure_sequence,
    _normalize_shapes_and_dtypes,
    _numpy_from_buffer,
    _detect_xla_ffi_api_version,
    _warn_if_untested_jax,
    _report_unreportable_ffi_error,
    _MAX_VALIDATED_JAX,
    _XLA_FFI_DTYPE_TO_NUMPY,
    XLA_FFI_Api,
    XLA_FFI_Api_Version,
    XLA_FFI_CallFrame,
    XLA_FFI_Args,
    XLA_FFI_Rets,
    XLA_FFI_Attrs,
    XLA_FFI_Extension_Base,
    XLA_FFI_Extension_Type,
    XLA_FFI_Metadata,
    XLA_FFI_Metadata_Extension,
    XLA_FFI_TypeId,
    XLA_FFI_Error_Create_Func,
    XLA_FFI_Error_Destroy_Args,
    XLA_FFI_Error_Destroy_Func,
    XLA_FFI_Stream_Get_Func,
    XLA_FFI_DeviceOrdinal_Get_Args,
    XLA_FFI_DeviceOrdinal_Get_Func,
    get_xla_stream,
    get_device_ordinal,
    resolve_buffer_dtype,
    numba_kernel,
    NumbaCpuFfiHandler,
)


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
        """Test that a non-Numba function raises ``TypeError``."""

        def regular_function(x, out):
            pass

        with self.assertRaises(TypeError):
            numba_kernel(
                regular_function,
                outs=jax.ShapeDtypeStruct((64,), jnp.float32),
            )

    def test_lambda_raises(self):
        """Test that a lambda function raises ``TypeError``."""
        with self.assertRaises(TypeError):
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
    """Tests for NumbaCpuFfiHandler class.

    F8 deviation: ``NumbaCpuFfiHandler`` no longer accepts/stores
    ``input_shapes``/``output_shapes`` -- the callback re-derives shapes at
    run time from ``XLA_FFI_Buffer.dims``, so the stored copies were dead
    weight that (via the old cache key) caused one target to be minted per
    distinct call shape (audit finding 8). This test is updated accordingly;
    see ``TestHandlerSelfPin`` below for the F7 self-pin-on-construction
    behavior this direct-construction pattern now guarantees.
    """

    def test_handler_attributes(self):
        """Test that handler stores correct attributes."""
        import numba

        @numba.njit
        def dummy_kernel(x, out):
            pass

        input_dtypes = (np.dtype(np.float32),)
        output_dtypes = (np.dtype(np.float32),)

        handler = NumbaCpuFfiHandler(
            name="test_handler",
            kernel=dummy_kernel,
            input_dtypes=input_dtypes,
            output_dtypes=output_dtypes,
        )

        self.assertEqual(handler.name, "test_handler")
        self.assertEqual(handler.input_dtypes, input_dtypes)
        self.assertEqual(handler.output_dtypes, output_dtypes)
        self.assertFalse(hasattr(handler, 'input_shapes'))
        self.assertFalse(hasattr(handler, 'output_shapes'))


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


class TestXlaFfiAbiVersionCheck:
    """The hard ``jax<0.11`` install pin was replaced by a runtime ABI check."""

    @pytest.fixture(autouse=True)
    def _restore_state(self):
        # Save and restore the globals these tests mutate so ordering is safe.
        saved_version = jax.__version__
        saved_flag = numba_ffi._jax_ffi_compat_checked
        yield
        jax.__version__ = saved_version
        numba_ffi._jax_ffi_compat_checked = saved_flag

    def test_detect_returns_major_minor_tuple(self):
        major, minor = _detect_xla_ffi_api_version()
        assert isinstance(major, int) and isinstance(minor, int)
        # Floor for every supported jaxlib is (0, 1); detected value is >= that.
        assert (major, minor) >= (0, 1)

    def test_no_warning_on_validated_version(self):
        jax.__version__ = f"{_MAX_VALIDATED_JAX[0]}.{_MAX_VALIDATED_JAX[1]}.99"
        numba_ffi._jax_ffi_compat_checked = False
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes an error
            _warn_if_untested_jax()

    def test_warns_on_newer_version(self):
        newer = f"{_MAX_VALIDATED_JAX[0]}.{_MAX_VALIDATED_JAX[1] + 1}.0"
        jax.__version__ = newer
        numba_ffi._jax_ffi_compat_checked = False
        with pytest.warns(RuntimeWarning, match="XLA FFI ABI"):
            _warn_if_untested_jax()

    def test_warns_at_most_once(self):
        jax.__version__ = f"{_MAX_VALIDATED_JAX[0]}.{_MAX_VALIDATED_JAX[1] + 1}.0"
        numba_ffi._jax_ffi_compat_checked = False
        with pytest.warns(RuntimeWarning):
            _warn_if_untested_jax()
        # Second call must be silent (warn-once guard).
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_if_untested_jax()

    def test_unparseable_version_is_silent(self):
        jax.__version__ = "not-a-version"
        numba_ffi._jax_ffi_compat_checked = False
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_if_untested_jax()  # must not raise or warn


# ---------------------------------------------------------------------------
# Audit regression tests
# ---------------------------------------------------------------------------


# --- C2: byte-accurate buffer views for dtypes ctypes cannot represent ---------

class TestBufferViewDtypes:
    """``_numpy_from_buffer`` must reconstruct every fixed-width dtype exactly."""

    def _roundtrip(self, arr):
        view = _numpy_from_buffer(arr.ctypes.data, arr.shape, arr.dtype)
        np.testing.assert_array_equal(view, arr)

    def test_float16(self):
        self._roundtrip(np.arange(6, dtype=np.float16).reshape(2, 3))

    def test_bfloat16(self):
        import ml_dtypes
        bf16 = np.dtype(ml_dtypes.bfloat16)
        self._roundtrip(np.arange(6, dtype=bf16).reshape(2, 3))

    def test_complex64(self):
        arr = (np.arange(4) + 1j * np.arange(4)).astype(np.complex64)
        self._roundtrip(arr)

    def test_complex128(self):
        arr = (np.arange(4) + 1j * np.arange(4)).astype(np.complex128)
        self._roundtrip(arr)

    def test_float32_still_correct(self):
        self._roundtrip(np.arange(12, dtype=np.float32).reshape(3, 4))


# --- M7: metadata struct must expose state_type_id -----------------------------

def test_metadata_struct_has_state_type_id():
    names = {name for name, _ in XLA_FFI_Metadata._fields_}
    assert 'state_type_id' in names


# --- C1: kernel exceptions must propagate, not be reported as success ----------

class TestErrorPropagation:
    def test_raising_kernel_surfaces_exception(self):
        import numba

        @numba.njit
        def boom(x, out):
            raise ValueError('intentional kernel failure')

        kernel = numba_kernel(boom, outs=jax.ShapeDtypeStruct((4,), jnp.float32))
        with pytest.raises(Exception):
            jax.block_until_ready(kernel(jnp.arange(4, dtype=jnp.float32)))


# --- H1: registration must be cached, not leaked once per call -----------------

class TestRegistrationCaching:
    def test_eager_calls_do_not_leak_targets(self):
        import numba

        @numba.njit
        def add1(x, out):
            for i in range(out.size):
                out[i] = x[i] + 1.0

        kernel = numba_kernel(add1, outs=jax.ShapeDtypeStruct((4,), jnp.float32))
        x = jnp.arange(4, dtype=jnp.float32)
        jax.block_until_ready(kernel(x))  # warm up / first registration
        before = len(numba_ffi._NUMBA_CPU_FFI_HANDLES)
        for _ in range(8):
            jax.block_until_ready(kernel(x))
        after = len(numba_ffi._NUMBA_CPU_FFI_HANDLES)
        assert after == before, f'leaked {after - before} FFI targets across 8 eager calls'


# --- F8: shapes must not be part of the FFI registration key -------------------

class TestShapeExcludedFromRegistrationKey:
    """Finding 8: a single kernel called at different input shapes must not
    mint a new FFI target/handler per shape (the callback re-derives shapes
    from ``buf_ptr.dims`` at run time; ``self.input_shapes`` was dead weight
    that the old cache key nonetheless leaked one entry per)."""

    def test_one_kernel_two_shapes_single_target(self):
        import numba

        @numba.njit
        def double_kernel(x, out):
            for i in range(out.size):
                out[i] = x[i] * 2.0

        # Same underlying numba dispatcher, wrapped at two different output
        # shapes -- exercises the *input* shape varying across calls to the
        # same kernel identity/dtype signature.
        kernel4 = numba_kernel(double_kernel, outs=jax.ShapeDtypeStruct((4,), jnp.float32))
        kernel8 = numba_kernel(double_kernel, outs=jax.ShapeDtypeStruct((8,), jnp.float32))

        before = len(numba_ffi._NUMBA_CPU_FFI_HANDLES)
        x4 = jnp.arange(4, dtype=jnp.float32)
        x8 = jnp.arange(8, dtype=jnp.float32)
        r4 = kernel4(x4)
        r4 = r4[0] if isinstance(r4, tuple) else r4
        r8 = kernel8(x8)
        r8 = r8[0] if isinstance(r8, tuple) else r8
        jax.block_until_ready((r4, r8))
        after = len(numba_ffi._NUMBA_CPU_FFI_HANDLES)

        assert after == before + 1, (
            f'two distinct shapes of the same kernel content minted '
            f'{after - before} targets, expected exactly 1'
        )
        np.testing.assert_allclose(np.asarray(r4), np.asarray(x4) * 2.0)
        np.testing.assert_allclose(np.asarray(r8), np.asarray(x8) * 2.0)


# --- F14: content-derived FFI target naming -------------------------------------

class TestContentDerivedNaming:
    """Finding 14: FFI target names are derived from kernel content
    (``kernel_content_fingerprint``), not a process-order-dependent counter."""

    @staticmethod
    def _target_name_for(kernel, dtype=np.float32):
        dt = (np.dtype(dtype),)
        return numba_ffi._NUMBA_CPU_FFI_TARGETS[(id(kernel), dt, dt)]

    def test_name_format_is_content_hash(self):
        import numba

        @numba.njit
        def add_seven(x, out):
            for i in range(out.size):
                out[i] = x[i] + 7.0

        wrapped = numba_kernel(add_seven, outs=jax.ShapeDtypeStruct((4,), jnp.float32))
        jax.block_until_ready(wrapped(jnp.arange(4, dtype=jnp.float32)))

        name = self._target_name_for(add_seven)
        assert re.match(r'^brainevent_numba_ffi_[0-9a-f]{16}$', name), name

    def test_same_content_two_fresh_functions_share_name(self):
        """Two independently defined (but byte-identical) kernels must
        register once and reuse the same target -- no duplicate-registration
        error -- because they fingerprint identically."""

        def make_kernel():
            import numba

            @numba.njit
            def add_one_fresh(x, out):
                for i in range(out.size):
                    out[i] = x[i] + 1.0

            return add_one_fresh

        kernel_a = make_kernel()
        kernel_b = make_kernel()
        assert kernel_a is not kernel_b

        wrapped_a = numba_kernel(kernel_a, outs=jax.ShapeDtypeStruct((4,), jnp.float32))
        wrapped_b = numba_kernel(kernel_b, outs=jax.ShapeDtypeStruct((4,), jnp.float32))

        x = jnp.arange(4, dtype=jnp.float32)
        ra = wrapped_a(x)
        ra = ra[0] if isinstance(ra, tuple) else ra
        rb = wrapped_b(x)
        rb = rb[0] if isinstance(rb, tuple) else rb
        jax.block_until_ready((ra, rb))

        name_a = self._target_name_for(kernel_a)
        name_b = self._target_name_for(kernel_b)
        assert name_a == name_b, (name_a, name_b)
        np.testing.assert_allclose(np.asarray(ra), np.asarray(x) + 1.0)
        np.testing.assert_allclose(np.asarray(rb), np.asarray(x) + 1.0)

    def test_different_kernels_different_names(self):
        import numba

        @numba.njit
        def add_one_diff(x, out):
            for i in range(out.size):
                out[i] = x[i] + 1.0

        @numba.njit
        def add_two_diff(x, out):
            for i in range(out.size):
                out[i] = x[i] + 2.0

        wrapped1 = numba_kernel(add_one_diff, outs=jax.ShapeDtypeStruct((4,), jnp.float32))
        wrapped2 = numba_kernel(add_two_diff, outs=jax.ShapeDtypeStruct((4,), jnp.float32))
        x = jnp.arange(4, dtype=jnp.float32)
        jax.block_until_ready(wrapped1(x))
        jax.block_until_ready(wrapped2(x))

        name1 = self._target_name_for(add_one_diff)
        name2 = self._target_name_for(add_two_diff)
        assert name1 != name2

    def test_name_collision_different_fingerprint_raises(self):
        """A same-name-different-fingerprint registration must raise rather
        than silently rebind the XLA target to a different kernel body."""
        import numba

        @numba.njit
        def collision_kernel(x, out):
            for i in range(out.size):
                out[i] = x[i] + 9.0

        fake_name = 'brainevent_numba_ffi_deadbeefdeadbeef'
        numba_ffi._NUMBA_CPU_FFI_NAME_FINGERPRINTS[fake_name] = 'not-the-real-fingerprint'
        try:
            with mock.patch.object(
                numba_ffi, 'kernel_content_fingerprint', return_value='deadbeefdeadbeef'
            ):
                wrapped = numba_kernel(collision_kernel, outs=jax.ShapeDtypeStruct((4,), jnp.float32))
                with pytest.raises(KernelRegistrationError, match='different content fingerprint'):
                    wrapped(jnp.arange(4, dtype=jnp.float32))
        finally:
            del numba_ffi._NUMBA_CPU_FFI_NAME_FINGERPRINTS[fake_name]


class TestContentFingerprintFallback:
    """Finding 14: a kernel whose closure/globals cannot be deterministically
    fingerprinted falls back to the pre-existing counter-name scheme for that
    kernel only, and still executes correctly.

    ``kernel_content_fingerprint`` is mocked to return ``None`` rather than
    hand-crafting a numba kernel with a genuinely unserializable closure: numba
    itself restricts what a jitted function may close over, so this isolates
    the test to the fallback *wiring* in ``_register_numba_cpu_ffi_target``
    (the fingerprint helper's own None-returning behavior is covered by
    ``brainevent/_op/ffi_naming_test.py``).
    """

    def test_unserializable_closure_falls_back_to_counter_name_and_works(self):
        import numba

        @numba.njit
        def add_three_fallback(x, out):
            for i in range(out.size):
                out[i] = x[i] + 3.0

        with mock.patch.object(numba_ffi, 'kernel_content_fingerprint', return_value=None):
            wrapped = numba_kernel(add_three_fallback, outs=jax.ShapeDtypeStruct((4,), jnp.float32))
            x = jnp.arange(4, dtype=jnp.float32)
            result = wrapped(x)
            result = result[0] if isinstance(result, tuple) else result
            jax.block_until_ready(result)

        np.testing.assert_allclose(np.asarray(result), np.asarray(x) + 3.0)

        dt = (np.dtype(np.float32),)
        name = numba_ffi._NUMBA_CPU_FFI_TARGETS[(id(add_three_fallback), dt, dt)]
        assert re.match(r'^brainevent_numba_ffi_\d+$', name), name
        assert numba_ffi._NUMBA_CPU_FFI_NAME_FINGERPRINTS[name] is None


# --- HIGH: kernel-pin on fingerprint reuse --------------------------------------

class TestKernelPinOnFingerprintReuse:
    """HIGH fix: a kernel that hits the fingerprint-reuse branch must be pinned
    so its ``id()`` can never be recycled while the memo entry lives.

    Pre-fix, the fingerprint-reuse branch in ``_register_numba_cpu_ffi_target``
    stored ``_NUMBA_CPU_FFI_TARGETS[(id(kernel), ...)] = target_name`` for the
    *second* (content-identical) kernel object without keeping any strong
    reference to that kernel. The shared FFI handler only keeps the *first*
    kernel object alive, so once the second kernel was garbage-collected its
    ``id()`` could be recycled by an unrelated object -- which would then
    wrongly hit the memo (dispatching to a stale/wrong handler) since the memo
    key is ``(id(kernel), input_dtypes, output_dtypes)``. The fix pins every
    reused kernel object into the module-level ``_NUMBA_CPU_FFI_KERNEL_PINS``
    dict, keyed by its own id, for as long as the memo entry exists.
    """

    def test_second_identical_kernel_is_pinned(self):
        import numba

        def make_kernel():
            @numba.njit
            def add_four_pin_test(x, out):
                for i in range(out.size):
                    out[i] = x[i] + 4.0

            return add_four_pin_test

        kernel_a = make_kernel()
        kernel_b = make_kernel()
        assert kernel_a is not kernel_b  # distinct objects, byte-identical content

        x = jnp.arange(4, dtype=jnp.float32)

        wrapped_a = numba_kernel(kernel_a, outs=jax.ShapeDtypeStruct((4,), jnp.float32))
        ra = wrapped_a(x)
        ra = ra[0] if isinstance(ra, tuple) else ra
        jax.block_until_ready(ra)

        # kernel_b is content-identical to kernel_a and hits the fingerprint-reuse
        # branch: it reuses kernel_a's registered target rather than registering
        # a fresh FFI handler.
        wrapped_b = numba_kernel(kernel_b, outs=jax.ShapeDtypeStruct((4,), jnp.float32))
        rb = wrapped_b(x)
        rb = rb[0] if isinstance(rb, tuple) else rb
        jax.block_until_ready(rb)

        dt = (np.dtype(np.float32),)
        name_a = numba_ffi._NUMBA_CPU_FFI_TARGETS[(id(kernel_a), dt, dt)]
        name_b = numba_ffi._NUMBA_CPU_FFI_TARGETS[(id(kernel_b), dt, dt)]
        assert name_a == name_b, 'byte-identical kernels must share one target'

        # The reuse branch must have pinned kernel_b so its id() cannot be
        # recycled while the memo entry above is still alive.
        assert id(kernel_b) in numba_ffi._NUMBA_CPU_FFI_KERNEL_PINS, (
            'reused kernel object was not pinned; its id() could be recycled '
            'by an unrelated object that would then wrongly hit the memo'
        )
        assert numba_ffi._NUMBA_CPU_FFI_KERNEL_PINS[id(kernel_b)] is kernel_b


# --- LOW: dispatcher-aware fingerprinting (globals and closures) ---------------

class _FakeDispatcher:
    """Stand-in for a numba dispatcher: exposes ``.py_func`` like a real
    ``CPUDispatcher``/``CUDADispatcher``, without requiring an actual numba
    compile. ``kernel_content_fingerprint``'s ``_serialize_value``/
    ``_serialize_global`` helpers detect any object with a plain-function
    ``.py_func`` attribute, not just real numba dispatchers, so this is a
    faithful (and much cheaper) stand-in for the fix under test.
    """

    def __init__(self, fn):
        self.py_func = fn


def _make_named_helper(tag, offset):
    """Build a helper function with a FIXED name/qualname but a body that
    differs by *offset* (baked into ``co_consts``).  Using a fixed name proves
    the fingerprint changes because of the recursion into ``py_func`` picking
    up the differing bytecode/constants -- not merely because the wrapped
    function's name changed.
    """

    def _shared_helper_name(x):
        return x + offset

    _shared_helper_name.__qualname__ = f'_shared_helper_name.{tag}'
    return _shared_helper_name


class TestDispatcherAwareFingerprint:
    """LOW fix: ``_serialize_global``/``_serialize_value`` must recurse into
    ``value.py_func`` for a dispatcher-like object, instead of stopping at a
    qualname-only identity (for globals) or aborting to ``None`` (for
    closures).

    Pre-fix behavior:

    - A global bound to a numba dispatcher fell into the generic ``callable``
      qualname-only branch, so editing the *body* of the dispatched helper
      while keeping the same global name did not change the kernel's
      fingerprint -- a stale registration would be silently reused for
      behaviorally different code.
    - A dispatcher captured in a closure cell had no matching branch at all in
      ``_serialize_value``, so fingerprinting aborted to ``None`` entirely,
      losing cross-process/name stability for every such kernel.
    """

    def test_global_dispatcher_body_change_changes_fingerprint(self):
        helper_1 = _make_named_helper('a', 1.0)
        globals()['_fp_test_helper_global'] = _FakeDispatcher(helper_1)
        try:
            def kernel_using_global(x, out):
                out[0] = _fp_test_helper_global(x)

            fp1 = kernel_content_fingerprint(kernel_using_global)
            assert fp1 is not None

            # Same global name, dispatcher wrapping a helper with a DIFFERENT
            # body (same fixed qualname, different baked-in constant).
            helper_2 = _make_named_helper('a', 2.0)
            globals()['_fp_test_helper_global'] = _FakeDispatcher(helper_2)
            fp2 = kernel_content_fingerprint(kernel_using_global)

            assert fp2 is not None
            assert fp1 != fp2, 'fingerprint did not change when the global dispatcher body changed'
        finally:
            del globals()['_fp_test_helper_global']

    def test_closure_dispatcher_fingerprints_and_body_change_changes_fingerprint(self):
        def make_kernel(dispatcher):
            def kernel_using_closure(x, out):
                out[0] = dispatcher(x)

            return kernel_using_closure

        helper_1 = _make_named_helper('b', 10.0)
        helper_2 = _make_named_helper('b', 20.0)

        kernel_1 = make_kernel(_FakeDispatcher(helper_1))
        kernel_2 = make_kernel(_FakeDispatcher(helper_2))

        fp1 = kernel_content_fingerprint(kernel_1)
        fp2 = kernel_content_fingerprint(kernel_2)

        assert fp1 is not None, 'closure over a dispatcher must fingerprint, not abort to None'
        assert fp2 is not None
        assert fp1 != fp2, 'fingerprint did not change when the closed-over dispatcher body changed'


# --- F7: handler self-pins its own lifetime -------------------------------------

class TestHandlerSelfPin:
    """Finding 7: direct ``NumbaCpuFfiHandler`` construction must self-pin so
    registration and lifetime cannot be separated -- a use-after-free of the
    ctypes trampoline is not a catchable Python error, so this proves liveness
    by actually invoking the target through ``jax.ffi.ffi_call`` after the
    only Python reference is dropped and collected."""

    def test_direct_construction_survives_gc_and_stays_callable(self):
        import numba

        @numba.njit
        def add_five_pin(x, out):
            for i in range(out.size):
                out[i] = x[i] + 5.0

        name = 'test_self_pin_handler_b'
        handler = NumbaCpuFfiHandler(
            name=name,
            kernel=add_five_pin,
            input_dtypes=(np.dtype(np.float32),),
            output_dtypes=(np.dtype(np.float32),),
        )
        del handler
        gc.collect()

        assert name in numba_ffi._NUMBA_CPU_FFI_HANDLES, 'handler was not self-pinned'

        out_type = jax.ShapeDtypeStruct((4,), jnp.float32)
        call = jax.ffi.ffi_call(name, out_type)
        x = jnp.arange(4, dtype=jnp.float32)
        result = jax.block_until_ready(call(x))
        np.testing.assert_allclose(np.asarray(result), np.asarray(x) + 5.0)


# --- F18: unknown / packed sub-byte dtype codes must raise ---------------------

class TestResolveBufferDtypeRejections:
    def test_known_dtype_still_resolves(self):
        assert resolve_buffer_dtype(11, np.dtype(np.float32)) == np.dtype(np.float32)

    def test_bf16_still_resolves(self):
        assert resolve_buffer_dtype(16, np.dtype(np.float32)) == np.dtype(ml_dtypes.bfloat16)

    def test_unknown_code_raises_value_error(self):
        with pytest.raises(ValueError, match='999'):
            resolve_buffer_dtype(999, np.dtype(np.float32))

    @pytest.mark.parametrize('code,label', [
        (21, 'S4'), (22, 'U4'), (26, 'S2'), (27, 'U2'),
        (30, 'S1'), (31, 'U1'), (32, 'F4E2M1FN'),
    ])
    def test_packed_subbyte_codes_raise(self, code, label):
        with pytest.raises(ValueError, match=label):
            resolve_buffer_dtype(code, np.dtype(np.float32))

    @pytest.mark.parametrize('code', [19, 20, 23, 24, 25, 28, 29, 33])
    def test_fp8_family_codes_raise(self, code):
        with pytest.raises(ValueError):
            resolve_buffer_dtype(code, np.dtype(np.float32))

    def test_error_names_the_code_and_trace_time_dtype(self):
        with pytest.raises(ValueError) as ctx:
            resolve_buffer_dtype(21, np.dtype(np.float64))
        message = str(ctx.value)
        assert '21' in message
        assert 'float64' in message


# --- F19a: FFI metadata extension chain must be walked, not just inspected -----

class TestExtensionChainWalk:
    """Finding 19: the metadata handshake must follow ``ext.next`` rather than
    only inspecting ``extension_start`` -- a future jaxlib that prepends an
    unrelated extension node ahead of Metadata must not break the handshake."""

    def test_metadata_found_when_not_first_in_chain(self):
        import numba

        @numba.njit
        def dummy_ext(x, out):
            pass

        handler = NumbaCpuFfiHandler(
            name='test_ext_chain_walk',
            kernel=dummy_ext,
            input_dtypes=(np.dtype(np.float32),),
            output_dtypes=(np.dtype(np.float32),),
        )

        metadata = XLA_FFI_Metadata(
            struct_size=ctypes.sizeof(XLA_FFI_Metadata),
            api_version=XLA_FFI_Api_Version(
                struct_size=ctypes.sizeof(XLA_FFI_Api_Version),
                extension_start=None,
                major_version=0,
                minor_version=0,
            ),
            traits=0xFF,  # sentinel: must be overwritten to 0 by the handshake
            state_type_id=XLA_FFI_TypeId(type_id=0),
        )
        metadata_ext = XLA_FFI_Metadata_Extension(
            extension_base=XLA_FFI_Extension_Base(
                struct_size=ctypes.sizeof(XLA_FFI_Extension_Base),
                type=int(XLA_FFI_Extension_Type.Metadata),
                next=None,
            ),
            metadata=ctypes.pointer(metadata),
        )
        # A non-Metadata node placed FIRST, chaining to the real Metadata node
        # via `.next` -- the walk must not stop at this first, unrecognized node.
        unknown_ext = XLA_FFI_Extension_Base(
            struct_size=ctypes.sizeof(XLA_FFI_Extension_Base),
            type=999,
            next=ctypes.cast(ctypes.pointer(metadata_ext), POINTER(XLA_FFI_Extension_Base)),
        )

        call_frame = XLA_FFI_CallFrame(
            struct_size=ctypes.sizeof(XLA_FFI_CallFrame),
            extension_start=ctypes.cast(ctypes.pointer(unknown_ext), POINTER(XLA_FFI_Extension_Base)),
            api=0,
            ctx=0,
            stage=0,
            args=XLA_FFI_Args(struct_size=ctypes.sizeof(XLA_FFI_Args), extension_start=None, size=0, types=None, args=None),
            rets=XLA_FFI_Rets(struct_size=ctypes.sizeof(XLA_FFI_Rets), extension_start=None, size=0, types=None, rets=None),
            attrs=XLA_FFI_Attrs(struct_size=ctypes.sizeof(XLA_FFI_Attrs), extension_start=None, size=0, types=None, names=None, attrs=None),
            future=0,
        )

        result = handler._ffi_callback(ctypes.pointer(call_frame))

        assert result is None  # success: the callback returned before touching args/rets
        assert metadata.api_version.major_version == numba_ffi.XLA_FFI_API_MAJOR
        assert metadata.api_version.minor_version == numba_ffi.XLA_FFI_API_MINOR
        assert metadata.traits == 0


# --- F19b: XLA_FFI_Error objects returned by API calls must be destroyed -------

class TestXlaFfiErrorDestroyed:
    """Finding 19: ``get_xla_stream``/``get_device_ordinal`` must destroy the
    ``XLA_FFI_Error*`` an API call returns on failure via
    ``XLA_FFI_Error_Destroy`` before raising/returning, instead of leaking it."""

    @staticmethod
    def _blank_api_version():
        return XLA_FFI_Api_Version(
            struct_size=ctypes.sizeof(XLA_FFI_Api_Version),
            extension_start=None,
            major_version=0,
            minor_version=0,
        )

    def test_get_xla_stream_destroys_error_before_raising(self):
        destroyed = []
        fake_err = 0x1234

        @XLA_FFI_Stream_Get_Func
        def stream_get_cb(args_ptr):
            return fake_err

        @XLA_FFI_Error_Destroy_Func
        def destroy_cb(args_ptr):
            destroyed.append(args_ptr.contents.error)

        api = XLA_FFI_Api(
            struct_size=ctypes.sizeof(XLA_FFI_Api),
            extension_start=None,
            api_version=self._blank_api_version(),
            internal_api=None,
            XLA_FFI_Error_Create=ctypes.cast(0, XLA_FFI_Error_Create_Func),
            XLA_FFI_Error_GetMessage=None,
            XLA_FFI_Error_Destroy=destroy_cb,
            XLA_FFI_Handler_Register=None,
            XLA_FFI_Stream_Get=stream_get_cb,
            XLA_FFI_Type_Register=None,
            XLA_FFI_ExecutionContext_Get=None,
            XLA_FFI_State_Set=None,
            XLA_FFI_State_Get=None,
            XLA_FFI_DeviceMemory_Allocate=None,
            XLA_FFI_DeviceMemory_Free=None,
            XLA_FFI_ThreadPool_Schedule=None,
            XLA_FFI_ThreadPool_NumThreads=None,
            XLA_FFI_Future_Create=None,
            XLA_FFI_Future_SetAvailable=None,
            XLA_FFI_Future_SetError=None,
            XLA_FFI_RunId_Get=None,
            XLA_FFI_DeviceOrdinal_Get=ctypes.cast(0, XLA_FFI_DeviceOrdinal_Get_Func),
        )

        with pytest.raises(RuntimeError):
            get_xla_stream(ctypes.addressof(api), ctx=0)

        assert destroyed == [fake_err]

    def test_get_device_ordinal_destroys_error_and_returns_none(self):
        destroyed = []
        fake_err = 0x5678

        @XLA_FFI_DeviceOrdinal_Get_Func
        def ordinal_get_cb(args_ptr):
            return fake_err

        @XLA_FFI_Error_Destroy_Func
        def destroy_cb(args_ptr):
            destroyed.append(args_ptr.contents.error)

        api = XLA_FFI_Api(
            struct_size=ctypes.sizeof(XLA_FFI_Api),
            extension_start=None,
            api_version=self._blank_api_version(),
            internal_api=None,
            XLA_FFI_Error_Create=ctypes.cast(0, XLA_FFI_Error_Create_Func),
            XLA_FFI_Error_GetMessage=None,
            XLA_FFI_Error_Destroy=destroy_cb,
            XLA_FFI_Handler_Register=None,
            XLA_FFI_Stream_Get=ctypes.cast(0, XLA_FFI_Stream_Get_Func),
            XLA_FFI_Type_Register=None,
            XLA_FFI_ExecutionContext_Get=None,
            XLA_FFI_State_Set=None,
            XLA_FFI_State_Get=None,
            XLA_FFI_DeviceMemory_Allocate=None,
            XLA_FFI_DeviceMemory_Free=None,
            XLA_FFI_ThreadPool_Schedule=None,
            XLA_FFI_ThreadPool_NumThreads=None,
            XLA_FFI_Future_Create=None,
            XLA_FFI_Future_SetAvailable=None,
            XLA_FFI_Future_SetError=None,
            XLA_FFI_RunId_Get=None,
            XLA_FFI_DeviceOrdinal_Get=ordinal_get_cb,
        )

        result = get_device_ordinal(ctypes.addressof(api), ctx=0)

        assert result is None
        assert destroyed == [fake_err]

    def test_get_xla_stream_tolerates_null_destroy_fn(self):
        """A null ``XLA_FFI_Error_Destroy`` (older jaxlib) must not crash the
        error path -- it is guarded, not assumed present."""
        fake_err = 0x9999

        @XLA_FFI_Stream_Get_Func
        def stream_get_cb(args_ptr):
            return fake_err

        api = XLA_FFI_Api(
            struct_size=ctypes.sizeof(XLA_FFI_Api),
            extension_start=None,
            api_version=self._blank_api_version(),
            internal_api=None,
            XLA_FFI_Error_Create=ctypes.cast(0, XLA_FFI_Error_Create_Func),
            XLA_FFI_Error_GetMessage=None,
            XLA_FFI_Error_Destroy=ctypes.cast(0, XLA_FFI_Error_Destroy_Func),  # null function pointer
            XLA_FFI_Handler_Register=None,
            XLA_FFI_Stream_Get=stream_get_cb,
            XLA_FFI_Type_Register=None,
            XLA_FFI_ExecutionContext_Get=None,
            XLA_FFI_State_Set=None,
            XLA_FFI_State_Get=None,
            XLA_FFI_DeviceMemory_Allocate=None,
            XLA_FFI_DeviceMemory_Free=None,
            XLA_FFI_ThreadPool_Schedule=None,
            XLA_FFI_ThreadPool_NumThreads=None,
            XLA_FFI_Future_Create=None,
            XLA_FFI_Future_SetAvailable=None,
            XLA_FFI_Future_SetError=None,
            XLA_FFI_RunId_Get=None,
            XLA_FFI_DeviceOrdinal_Get=ctypes.cast(0, XLA_FFI_DeviceOrdinal_Get_Func),
        )

        with pytest.raises(RuntimeError):
            get_xla_stream(ctypes.addressof(api), ctx=0)


# --- F19c: a loud stderr message when an error cannot be reported to XLA -------

class TestUnreportableFfiError:
    def test_writes_prominent_message_to_stderr(self, capfd):
        _report_unreportable_ffi_error('my_kernel', ValueError('boom'))
        captured = capfd.readouterr()
        assert 'my_kernel' in captured.err
        assert 'boom' in captured.err
        assert 'FATAL' in captured.err
        assert 'NOT be reported to XLA' in captured.err or 'could NOT be reported' in captured.err
