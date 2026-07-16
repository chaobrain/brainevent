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

"""Tests for Numba CUDA FFI integration with JAX."""

import importlib.util
import os
import unittest

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._op import numba_cuda_ffi
from brainevent._op.numba_cuda_ffi import _compute_launch_config

# Check if Numba CUDA is available
numba_installed = importlib.util.find_spec('numba') is not None
numba_cuda_available = False
if numba_installed:
    try:
        from numba import cuda

        numba_cuda_available = cuda.is_available()
    except ImportError:
        pass

gpu_platform = jax.default_backend() == 'gpu'
numba_cuda_available = numba_cuda_available and gpu_platform

requires_gpu = pytest.mark.skipif(
    not numba_cuda_available, reason="Numba CUDA not available"
)


@pytest.mark.skipif(not numba_cuda_available, reason="Numba CUDA not available")
class TestNumbaCudaKernelBasic(unittest.TestCase):
    """Basic functionality tests for numba_cuda_kernel."""

    def test_element_wise_addition(self):
        """Test simple element-wise addition kernel."""
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def add_kernel(x, y, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i] + y[i]

        n = 1024
        kernel = numba_cuda_kernel(
            add_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
            grid=4,
            block=256,
        )

        a = jnp.arange(n, dtype=jnp.float32)
        b = jnp.ones(n, dtype=jnp.float32) * 2
        result = kernel(a, b)
        expected = a + b

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((a, b, result, expected))

    def test_element_wise_multiplication(self):
        """Test element-wise multiplication kernel."""
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def mul_kernel(x, y, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i] * y[i]

        n = 512
        kernel = numba_cuda_kernel(
            mul_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
            grid=2,
            block=256,
        )

        a = jnp.arange(n, dtype=jnp.float32)
        b = jnp.arange(n, dtype=jnp.float32) * 0.5
        result = kernel(a, b)
        expected = a * b

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((a, b, result, expected))

    def test_different_dtypes(self):
        """Test kernel with different data types."""
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def copy_kernel(x, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i]

        for dtype in [jnp.float32, jnp.float64, jnp.int32, jnp.int64]:
            n = 256
            kernel = numba_cuda_kernel(
                copy_kernel,
                outs=jax.ShapeDtypeStruct((n,), dtype),
                grid=1,
                block=256,
            )

            a = jnp.arange(n, dtype=dtype)
            result = kernel(a)
            self.assertTrue(jnp.allclose(result, a), f"Failed for dtype {dtype}")
            jax.block_until_ready((a, result))


@pytest.mark.skipif(not numba_cuda_available, reason="Numba CUDA not available")
class TestNumbaCudaKernelLaunchConfig(unittest.TestCase):
    """Tests for different launch configurations."""

    def test_launch_dims_1d(self):
        """Test kernel with launch_dims parameter (1D)."""
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def add_kernel(x, y, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i] + y[i]

        n = 1024
        kernel = numba_cuda_kernel(
            add_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
            launch_dims=n,
        )

        a = jnp.arange(n, dtype=jnp.float32)
        b = jnp.ones(n, dtype=jnp.float32)
        result = kernel(a, b)
        expected = a + b

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((a, b, result, expected))

    def test_launch_dims_with_threads_per_block(self):
        """Test kernel with custom threads_per_block."""
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def add_kernel(x, y, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i] + y[i]

        n = 1024
        kernel = numba_cuda_kernel(
            add_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
            launch_dims=n,
            threads_per_block=128,
        )

        a = jnp.arange(n, dtype=jnp.float32)
        b = jnp.ones(n, dtype=jnp.float32)
        result = kernel(a, b)
        expected = a + b

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((a, b, result, expected))

    def test_explicit_grid_block_tuple(self):
        """Test kernel with tuple grid/block dimensions."""
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def add_kernel(x, y, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i] + y[i]

        n = 1024
        kernel = numba_cuda_kernel(
            add_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
            grid=(8,),
            block=(128,),
        )

        a = jnp.arange(n, dtype=jnp.float32)
        b = jnp.ones(n, dtype=jnp.float32)
        result = kernel(a, b)
        expected = a + b

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((a, b, result, expected))


@pytest.mark.skipif(not numba_cuda_available, reason="Numba CUDA not available")
class TestNumbaCudaKernelJaxJit(unittest.TestCase):
    """Tests for usage with jax.jit."""

    def test_inside_jax_jit(self):
        """Test kernel inside @jax.jit decorated function."""
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def add_kernel(x, y, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i] + y[i]

        n = 1024
        kernel = numba_cuda_kernel(
            add_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
            grid=4,
            block=256,
        )

        @jax.jit
        def jitted_add(a, b):
            return kernel(a, b)

        a = jnp.arange(n, dtype=jnp.float32)
        b = jnp.ones(n, dtype=jnp.float32)
        result = jitted_add(a, b)
        expected = a + b

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((a, b, result, expected))

    def test_multiple_calls_in_jit(self):
        """Test multiple kernel calls inside jax.jit."""
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def add_kernel(x, y, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i] + y[i]

        @cuda.jit
        def mul_kernel(x, y, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i] * y[i]

        n = 512
        add_k = numba_cuda_kernel(
            add_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
            launch_dims=n,
        )
        mul_k = numba_cuda_kernel(
            mul_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
            launch_dims=n,
        )

        @jax.jit
        def combined(a, b, c):
            temp = add_k(a, b)
            return mul_k(temp, c)

        a = jnp.arange(n, dtype=jnp.float32)
        b = jnp.ones(n, dtype=jnp.float32)
        c = jnp.full(n, 2.0, dtype=jnp.float32)

        result = combined(a, b, c)
        expected = (a + b) * c

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((a, b, c, result, expected))


@pytest.mark.skipif(not numba_cuda_available, reason="Numba CUDA not available")
class TestNumbaCudaKernelMultipleOutputs(unittest.TestCase):
    """Tests for kernels with multiple outputs."""

    def test_two_outputs(self):
        """Test kernel with two output arrays."""
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def split_kernel(x, out1, out2):
            i = cuda.grid(1)
            if i < out1.size:
                out1[i] = x[i] * 2
                out2[i] = x[i] * 3

        n = 256
        kernel = numba_cuda_kernel(
            split_kernel,
            outs=[
                jax.ShapeDtypeStruct((n,), jnp.float32),
                jax.ShapeDtypeStruct((n,), jnp.float32),
            ],
            launch_dims=n,
        )

        x = jnp.arange(n, dtype=jnp.float32)
        out1, out2 = kernel(x)

        self.assertTrue(jnp.allclose(out1, x * 2))
        self.assertTrue(jnp.allclose(out2, x * 3))
        jax.block_until_ready((x, out1, out2))

    def test_three_outputs(self):
        """Test kernel with three output arrays."""
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def triple_kernel(x, y, sum_out, diff_out, prod_out):
            i = cuda.grid(1)
            if i < sum_out.size:
                sum_out[i] = x[i] + y[i]
                diff_out[i] = x[i] - y[i]
                prod_out[i] = x[i] * y[i]

        n = 128
        kernel = numba_cuda_kernel(
            triple_kernel,
            outs=[
                jax.ShapeDtypeStruct((n,), jnp.float32),
                jax.ShapeDtypeStruct((n,), jnp.float32),
                jax.ShapeDtypeStruct((n,), jnp.float32),
            ],
            launch_dims=n,
        )

        x = jnp.arange(n, dtype=jnp.float32)
        y = jnp.ones(n, dtype=jnp.float32) * 5
        sum_out, diff_out, prod_out = kernel(x, y)

        self.assertTrue(jnp.allclose(sum_out, x + y))
        self.assertTrue(jnp.allclose(diff_out, x - y))
        self.assertTrue(jnp.allclose(prod_out, x * y))
        jax.block_until_ready((x, y, sum_out, diff_out, prod_out))


@pytest.mark.skipif(not numba_cuda_available, reason="Numba CUDA not available")
class TestNumbaCudaKernelSharedMemory(unittest.TestCase):
    """Tests for kernels using shared memory."""

    def test_shared_memory_reduction(self):
        """Test kernel using shared memory for partial reduction."""
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def sum_reduce_kernel(x, out):
            shared = cuda.shared.array(256, dtype=np.float32)

            tid = cuda.threadIdx.x
            bid = cuda.blockIdx.x
            i = cuda.grid(1)

            # Load into shared memory
            if i < x.size:
                shared[tid] = x[i]
            else:
                shared[tid] = 0.0
            cuda.syncthreads()

            # Reduction
            s = 128
            while s > 0:
                if tid < s:
                    shared[tid] += shared[tid + s]
                cuda.syncthreads()
                s //= 2

            # Write block result
            if tid == 0:
                out[bid] = shared[0]

        n = 1024
        num_blocks = 4
        kernel = numba_cuda_kernel(
            sum_reduce_kernel,
            outs=jax.ShapeDtypeStruct((num_blocks,), jnp.float32),
            grid=num_blocks,
            block=256,
            shared_mem=256 * 4,  # 256 floats * 4 bytes
        )

        x = jnp.ones(n, dtype=jnp.float32)
        partial_sums = kernel(x)
        total = jnp.sum(partial_sums)

        self.assertTrue(jnp.allclose(total, float(n)))
        jax.block_until_ready((x, partial_sums, total))


@pytest.mark.skipif(not numba_cuda_available, reason="Numba CUDA not available")
class TestNumbaCudaKernelMultipleInputs(unittest.TestCase):
    """Tests for kernels with various input configurations."""

    def test_scalar_input(self):
        """Test kernel with scalar (single-element array) input."""
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def scale_kernel(x, scale, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i] * scale[0]

        n = 512
        kernel = numba_cuda_kernel(
            scale_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
            launch_dims=n,
        )

        x = jnp.arange(n, dtype=jnp.float32)
        scale = jnp.array([3.0], dtype=jnp.float32)
        result = kernel(x, scale)
        expected = x * 3.0

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((x, scale, result, expected))

    def test_many_inputs(self):
        """Test kernel with multiple input arrays."""
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def weighted_sum_kernel(a, b, c, w1, w2, w3, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = a[i] * w1[0] + b[i] * w2[0] + c[i] * w3[0]

        n = 256
        kernel = numba_cuda_kernel(
            weighted_sum_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
            launch_dims=n,
        )

        a = jnp.arange(n, dtype=jnp.float32)
        b = jnp.ones(n, dtype=jnp.float32)
        c = jnp.full(n, 2.0, dtype=jnp.float32)
        w1 = jnp.array([1.0], dtype=jnp.float32)
        w2 = jnp.array([2.0], dtype=jnp.float32)
        w3 = jnp.array([3.0], dtype=jnp.float32)

        result = kernel(a, b, c, w1, w2, w3)
        expected = a * 1.0 + b * 2.0 + c * 3.0

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((a, b, c, w1, w2, w3, result, expected))


@pytest.mark.skipif(not numba_cuda_available, reason="Numba CUDA not available")
class TestNumbaCudaKernelErrors(unittest.TestCase):
    """Tests for error handling."""

    def test_missing_launch_config_raises(self):
        """Test that missing grid/block and launch_dims raises ValueError."""
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def dummy_kernel(x, out):
            pass

        with self.assertRaises(ValueError):
            numba_cuda_kernel(
                dummy_kernel,
                outs=jax.ShapeDtypeStruct((64,), jnp.float32),
                # No grid, block, or launch_dims
            )

    def test_non_cuda_kernel_raises(self):
        """Test that a non-CUDA kernel raises TypeError.

        The check uses an explicit ``raise TypeError`` (not ``assert``) so it
        survives ``python -O``, which strips assertions (L14).
        """
        from brainevent import numba_cuda_kernel

        def regular_function(x, out):
            pass

        with self.assertRaises(TypeError):
            numba_cuda_kernel(
                regular_function,
                outs=jax.ShapeDtypeStruct((64,), jnp.float32),
                launch_dims=64,
            )


@pytest.mark.skipif(not numba_cuda_available, reason="Numba CUDA not available")
class TestNumbaCudaKernelXLAStream(unittest.TestCase):
    """Tests to verify XLA stream extraction is working."""

    def test_stream_extraction_works(self):
        """Verify that XLA stream is being extracted (not using fallback)."""
        from numba import cuda
        from brainevent import numba_cuda_kernel
        import brainevent._op.numba_cuda_ffi as ffi

        @cuda.jit
        def add_kernel(x, y, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i] + y[i]

        n = 256
        kernel = numba_cuda_kernel(
            add_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
            launch_dims=n,
        )

        # Track if XLA stream was used.  The bridge now resolves the stream via
        # the shared, error-checked ``get_xla_stream(api_ptr, ctx)`` helper
        # imported from ``numba_ffi``; patch that module-global to observe it.
        stream_ptrs = []
        orig_get_stream = ffi.get_xla_stream

        def tracking_get_stream(api_ptr, ctx):
            ptr = orig_get_stream(api_ptr, ctx)
            stream_ptrs.append(ptr)
            return ptr

        ffi.get_xla_stream = tracking_get_stream

        try:
            a = jnp.arange(n, dtype=jnp.float32)
            b = jnp.ones(n, dtype=jnp.float32)
            result = kernel(a, b)

            # Verify result is correct
            self.assertTrue(jnp.allclose(result, a + b))

            # Verify stream was extracted (non-zero pointer)
            self.assertTrue(len(stream_ptrs) > 0, "Stream extraction was not called")
            self.assertTrue(
                any(ptr != 0 for ptr in stream_ptrs),
                "XLA stream was not extracted (all pointers are 0)"
            )
            jax.block_until_ready((a, b, result))
        finally:
            ffi.get_xla_stream = orig_get_stream


# ===========================================================================
# numba_cuda_callable tests
# ===========================================================================


# ===========================================================================
# 1. Validation / Error Handling
# ===========================================================================


class TestNumbaCudaCallableValidation(unittest.TestCase):
    """Validation tests for the callable wrapper."""

    @pytest.mark.skipif(not numba_cuda_available, reason="Numba CUDA not available")
    def test_non_callable_raises_typeerror(self):
        from brainevent import numba_cuda_callable

        with self.assertRaises(TypeError):
            numba_cuda_callable(
                "not a function",
                outs=jax.ShapeDtypeStruct((10,), jnp.float32),
            )

    @pytest.mark.skipif(not numba_cuda_available, reason="Numba CUDA not available")
    def test_returns_callable(self):
        from brainevent import numba_cuda_callable

        def dummy(out, stream):
            pass

        fn = numba_cuda_callable(
            dummy,
            outs=jax.ShapeDtypeStruct((10,), jnp.float32),
        )
        self.assertTrue(callable(fn))


# ===========================================================================
# 2. Basic Single-Kernel Callable
# ===========================================================================


@pytest.mark.skipif(not numba_cuda_available, reason="Numba CUDA not available")
class TestNumbaCudaCallableBasic(unittest.TestCase):
    """Basic functionality tests."""

    def test_single_kernel_add(self):
        """Wrap a single add kernel in a callable."""
        from numba import cuda
        from brainevent import numba_cuda_callable

        @cuda.jit
        def add_kernel(x, y, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i] + y[i]

        def add_fn(x, y, out, stream):
            n = x.size
            threads = 256
            blocks = (n + threads - 1) // threads
            add_kernel[blocks, threads, stream](x, y, out)

        n = 1024
        f = numba_cuda_callable(
            add_fn,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
        )

        a = jnp.arange(n, dtype=jnp.float32)
        b = jnp.ones(n, dtype=jnp.float32) * 2
        result = f(a, b)
        expected = a + b

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((a, b, result, expected))

    def test_single_kernel_scale(self):
        """Scale array using a callable with scalar input."""
        from numba import cuda
        from brainevent import numba_cuda_callable

        @cuda.jit
        def scale_kernel(x, s, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i] * s[0]

        def scale_fn(x, s, out, stream):
            n = x.size
            threads = 256
            blocks = (n + threads - 1) // threads
            scale_kernel[blocks, threads, stream](x, s, out)

        n = 512
        f = numba_cuda_callable(
            scale_fn,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
        )

        x = jnp.arange(n, dtype=jnp.float32)
        s = jnp.array([3.0], dtype=jnp.float32)
        result = f(x, s)
        expected = x * 3.0

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((x, s, result, expected))

    def test_inside_jax_jit(self):
        """Test callable inside @jax.jit."""
        from numba import cuda
        from brainevent import numba_cuda_callable

        @cuda.jit
        def add_kernel(x, y, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i] + y[i]

        def add_fn(x, y, out, stream):
            n = x.size
            threads = 256
            blocks = (n + threads - 1) // threads
            add_kernel[blocks, threads, stream](x, y, out)

        n = 1024
        f = numba_cuda_callable(
            add_fn,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
        )

        @jax.jit
        def jitted_add(a, b):
            return f(a, b)

        a = jnp.arange(n, dtype=jnp.float32)
        b = jnp.ones(n, dtype=jnp.float32)
        result = jitted_add(a, b)
        expected = a + b

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((a, b, result, expected))


# ===========================================================================
# 3. Multi-Kernel Pipeline
# ===========================================================================


@pytest.mark.skipif(not numba_cuda_available, reason="Numba CUDA not available")
class TestNumbaCudaCallableMultiKernel(unittest.TestCase):
    """Test callables that launch multiple kernels sequentially."""

    def test_add_then_scale(self):
        """Two-kernel pipeline: add then scale."""
        from numba import cuda
        from brainevent import numba_cuda_callable

        @cuda.jit
        def add_kernel(x, y, temp):
            i = cuda.grid(1)
            if i < temp.size:
                temp[i] = x[i] + y[i]

        @cuda.jit
        def scale_kernel(temp, out, scale):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = temp[i] * scale

        def add_then_scale(x, y, out, stream):
            n = x.size
            threads = 256
            blocks = (n + threads - 1) // threads
            temp = cuda.device_array(n, dtype=np.float32, stream=stream)
            add_kernel[blocks, threads, stream](x, y, temp)
            scale_kernel[blocks, threads, stream](temp, out, 2.0)

        n = 1024
        f = numba_cuda_callable(
            add_then_scale,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
        )

        a = jnp.arange(n, dtype=jnp.float32)
        b = jnp.ones(n, dtype=jnp.float32)
        result = f(a, b)
        expected = (a + b) * 2.0

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((a, b, result, expected))

    def test_three_kernel_pipeline(self):
        """Three-kernel pipeline: add, square, negate."""
        from numba import cuda
        from brainevent import numba_cuda_callable

        @cuda.jit
        def add_k(a, b, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = a[i] + b[i]

        @cuda.jit
        def square_k(arr):
            i = cuda.grid(1)
            if i < arr.size:
                arr[i] = arr[i] * arr[i]

        @cuda.jit
        def negate_k(arr):
            i = cuda.grid(1)
            if i < arr.size:
                arr[i] = -arr[i]

        def pipeline(a, b, out, stream):
            n = a.size
            threads = 256
            blocks = (n + threads - 1) // threads
            add_k[blocks, threads, stream](a, b, out)
            square_k[blocks, threads, stream](out)
            negate_k[blocks, threads, stream](out)

        n = 512
        f = numba_cuda_callable(
            pipeline,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
        )

        a = jnp.arange(n, dtype=jnp.float32) * 0.01
        b = jnp.ones(n, dtype=jnp.float32)
        result = f(a, b)
        expected = -((a + b) ** 2)

        self.assertTrue(jnp.allclose(result, expected, atol=1e-4))
        jax.block_until_ready((a, b, result, expected))


# ===========================================================================
# 4. Multiple Outputs
# ===========================================================================


@pytest.mark.skipif(not numba_cuda_available, reason="Numba CUDA not available")
class TestNumbaCudaCallableMultipleOutputs(unittest.TestCase):
    """Test callables producing multiple output arrays."""

    def test_two_outputs(self):
        """Callable producing sum and difference."""
        from numba import cuda
        from brainevent import numba_cuda_callable

        @cuda.jit
        def sum_diff_kernel(a, b, sum_out, diff_out):
            i = cuda.grid(1)
            if i < sum_out.size:
                sum_out[i] = a[i] + b[i]
                diff_out[i] = a[i] - b[i]

        def sum_diff_fn(a, b, sum_out, diff_out, stream):
            n = a.size
            threads = 256
            blocks = (n + threads - 1) // threads
            sum_diff_kernel[blocks, threads, stream](a, b, sum_out, diff_out)

        n = 256
        f = numba_cuda_callable(
            sum_diff_fn,
            outs=[
                jax.ShapeDtypeStruct((n,), jnp.float32),
                jax.ShapeDtypeStruct((n,), jnp.float32),
            ],
        )

        a = jnp.arange(n, dtype=jnp.float32)
        b = jnp.ones(n, dtype=jnp.float32) * 3
        s, d = f(a, b)

        self.assertTrue(jnp.allclose(s, a + b))
        self.assertTrue(jnp.allclose(d, a - b))
        jax.block_until_ready((a, b, s, d))

    def test_different_output_shapes(self):
        """Outputs with different shapes."""
        from numba import cuda
        from brainevent import numba_cuda_callable

        @cuda.jit
        def k(x, full_out, scalar_out):
            i = cuda.grid(1)
            if i < full_out.size:
                full_out[i] = x[i] * 2.0
            if i == 0:
                s = 0.0
                for j in range(x.size):
                    s += x[j]
                scalar_out[0] = s

        def fn(x, full_out, scalar_out, stream):
            n = x.size
            threads = 256
            blocks = (n + threads - 1) // threads
            k[blocks, threads, stream](x, full_out, scalar_out)

        n = 128
        f = numba_cuda_callable(
            fn,
            outs=[
                jax.ShapeDtypeStruct((n,), jnp.float32),
                jax.ShapeDtypeStruct((1,), jnp.float32),
            ],
        )

        x = jnp.ones(n, dtype=jnp.float32)
        doubled, total = f(x)

        self.assertTrue(jnp.allclose(doubled, x * 2.0))
        self.assertTrue(jnp.allclose(total[0], float(n)))
        jax.block_until_ready((x, doubled, total))


# ===========================================================================
# 5. Stream Correctness
# ===========================================================================


@pytest.mark.skipif(not numba_cuda_available, reason="Numba CUDA not available")
class TestNumbaCudaCallableStream(unittest.TestCase):
    """Verify that the stream is correctly passed to the callable."""

    def test_stream_is_passed_to_callable(self):
        """Verify the callable receives a valid (non-None) stream object."""
        from numba import cuda
        from brainevent import numba_cuda_callable

        received_streams = []

        @cuda.jit
        def _noop_k(out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = 42.0

        def checking_fn(out, stream):
            received_streams.append(stream)
            _noop_k[1, 256, stream](out)

        f = numba_cuda_callable(
            checking_fn,
            outs=jax.ShapeDtypeStruct((256,), jnp.float32),
        )
        result = f()

        self.assertTrue(len(received_streams) > 0, "Stream was never passed to callable")
        self.assertIsNotNone(received_streams[0], "Stream is None")
        self.assertTrue(jnp.allclose(result, jnp.full(256, 42.0, dtype=jnp.float32)))
        jax.block_until_ready((result,))

    def test_all_kernels_share_same_stream(self):
        """All kernel launches inside the callable should use the same stream."""
        from numba import cuda
        from brainevent import numba_cuda_callable

        stream_ids = []

        @cuda.jit
        def _k1(out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = 1.0

        @cuda.jit
        def _k2(arr):
            i = cuda.grid(1)
            if i < arr.size:
                arr[i] += 1.0

        def fn(out, stream):
            stream_ids.append(id(stream))
            _k1[1, 256, stream](out)
            stream_ids.append(id(stream))
            _k2[1, 256, stream](out)

        f = numba_cuda_callable(
            fn,
            outs=jax.ShapeDtypeStruct((256,), jnp.float32),
        )
        result = f()

        self.assertTrue(len(stream_ids) >= 2)
        self.assertEqual(stream_ids[0], stream_ids[1])
        self.assertTrue(jnp.allclose(result, jnp.full(256, 2.0, dtype=jnp.float32)))
        jax.block_until_ready((result,))


# ===========================================================================
# 6. Error Handling
# ===========================================================================


@pytest.mark.skipif(not numba_cuda_available, reason="Numba CUDA not available")
class TestNumbaCudaCallableErrors(unittest.TestCase):
    """Test error handling and validation."""

    def test_non_callable_raises_typeerror(self):
        """Non-callable func should raise TypeError."""
        from brainevent import numba_cuda_callable

        with self.assertRaises(TypeError):
            numba_cuda_callable(
                42,
                outs=jax.ShapeDtypeStruct((10,), jnp.float32),
            )


# ===========================================================================
# 7. Fused Operations
# ===========================================================================


@pytest.mark.skipif(not numba_cuda_available, reason="Numba CUDA not available")
class TestNumbaCudaCallableFusedOps(unittest.TestCase):
    """Fused multi-kernel operations."""

    def test_softmax_like_pipeline(self):
        """Fused softmax: exp -> sum -> normalize."""
        from numba import cuda
        from brainevent import numba_cuda_callable
        import math

        @cuda.jit
        def _exp_kernel(x, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = math.exp(x[i])

        @cuda.jit
        def _sum_kernel(x, out):
            if cuda.grid(1) == 0:
                s = 0.0
                for i in range(x.size):
                    s += x[i]
                out[0] = s

        @cuda.jit
        def _div_kernel(x, divisor, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i] / divisor[0]

        def softmax_fn(x, out, stream):
            n = x.size
            threads = 256
            blocks = (n + threads - 1) // threads

            exp_buf = cuda.device_array(n, dtype=np.float32, stream=stream)
            sum_buf = cuda.device_array(1, dtype=np.float32, stream=stream)

            _exp_kernel[blocks, threads, stream](x, exp_buf)
            _sum_kernel[1, 1, stream](exp_buf, sum_buf)
            _div_kernel[blocks, threads, stream](exp_buf, sum_buf, out)

        n = 64
        f = numba_cuda_callable(
            softmax_fn,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
        )

        x = jnp.linspace(-2.0, 2.0, n, dtype=jnp.float32)
        result = f(x)

        expected = jax.nn.softmax(x)
        self.assertTrue(jnp.allclose(result, expected, atol=1e-5))
        jax.block_until_ready((x, result, expected))

    def test_norm_then_dot(self):
        """Normalize a vector then compute dot product (two outputs)."""
        from numba import cuda
        from brainevent import numba_cuda_callable
        import math

        @cuda.jit
        def _norm_kernel(x, norm_out):
            if cuda.grid(1) == 0:
                s = 0.0
                for i in range(x.size):
                    s += x[i] * x[i]
                norm_out[0] = math.sqrt(s)

        @cuda.jit
        def _normalize_kernel(x, norm, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i] / norm[0]

        @cuda.jit
        def _dot_kernel(a, b, out):
            if cuda.grid(1) == 0:
                s = 0.0
                for i in range(a.size):
                    s += a[i] * b[i]
                out[0] = s

        def norm_and_dot(x, y, normalized_out, dot_out, stream):
            n = x.size
            threads = 256
            blocks = (n + threads - 1) // threads

            norm_buf = cuda.device_array(1, dtype=np.float32, stream=stream)

            _norm_kernel[1, 1, stream](x, norm_buf)
            _normalize_kernel[blocks, threads, stream](x, norm_buf, normalized_out)
            _dot_kernel[1, 1, stream](normalized_out, y, dot_out)

        n = 64
        f = numba_cuda_callable(
            norm_and_dot,
            outs=[
                jax.ShapeDtypeStruct((n,), jnp.float32),
                jax.ShapeDtypeStruct((1,), jnp.float32),
            ],
        )

        x = jnp.arange(n, dtype=jnp.float32) + 1.0
        y = jnp.ones(n, dtype=jnp.float32)

        normalized, dot = f(x, y)

        x_norm = jnp.linalg.norm(x)
        expected_normalized = x / x_norm
        expected_dot = jnp.dot(expected_normalized, y)

        self.assertTrue(jnp.allclose(normalized, expected_normalized, rtol=1e-5))
        self.assertTrue(jnp.allclose(dot[0], expected_dot, rtol=1e-4))
        jax.block_until_ready((x, y, normalized, dot, x_norm, expected_normalized, expected_dot))


# ===========================================================================
# 8. Neuroscience Domain: Sparse CSR Matrix-Vector Product
# ===========================================================================


@pytest.mark.skipif(not numba_cuda_available, reason="Numba CUDA not available")
class TestNumbaCudaCallableCSRMatvec(unittest.TestCase):
    """CSR sparse matvec followed by ReLU activation."""

    def test_csr_matvec_then_relu(self):
        from numba import cuda
        from brainevent import numba_cuda_callable

        @cuda.jit
        def _csr_matvec_kernel(data, indices, indptr, x, out):
            row = cuda.grid(1)
            if row < out.size:
                start = indptr[row]
                end = indptr[row + 1]
                total = 0.0
                for idx in range(start, end):
                    col = indices[idx]
                    total += data[idx] * x[col]
                out[row] = total

        @cuda.jit
        def _relu_kernel(arr):
            i = cuda.grid(1)
            if i < arr.size:
                if arr[i] < 0.0:
                    arr[i] = 0.0

        def csr_matvec_relu(data, indices, indptr, x, out, stream):
            n_rows = out.size
            threads = 256
            blocks = (n_rows + threads - 1) // threads
            _csr_matvec_kernel[blocks, threads, stream](data, indices, indptr, x, out)
            _relu_kernel[blocks, threads, stream](out)

        # A = [[1, 0, 2], [0, 3, 0], [4, 0, 5]]
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)
        indices = jnp.array([0, 2, 1, 0, 2], dtype=jnp.int32)
        indptr = jnp.array([0, 2, 3, 5], dtype=jnp.int32)
        x = jnp.array([-1.0, 2.0, 3.0], dtype=jnp.float32)

        n_rows = 3
        f = numba_cuda_callable(
            csr_matvec_relu,
            outs=jax.ShapeDtypeStruct((n_rows,), jnp.float32),
        )

        result = f(data, indices, indptr, x)

        # Ax = [1*(-1)+2*3, 3*2, 4*(-1)+5*3] = [5, 6, 11]
        # After ReLU: [5, 6, 11] (all positive)
        expected = jnp.array([5.0, 6.0, 11.0], dtype=jnp.float32)
        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((data, indices, indptr, x, result, expected))

    def test_csr_matvec_with_negative_results(self):
        """Test that ReLU actually clips negative values."""
        from numba import cuda
        from brainevent import numba_cuda_callable

        @cuda.jit
        def _csr_matvec_kernel(data, indices, indptr, x, out):
            row = cuda.grid(1)
            if row < out.size:
                start = indptr[row]
                end = indptr[row + 1]
                total = 0.0
                for idx in range(start, end):
                    col = indices[idx]
                    total += data[idx] * x[col]
                out[row] = total

        @cuda.jit
        def _relu_kernel(arr):
            i = cuda.grid(1)
            if i < arr.size:
                if arr[i] < 0.0:
                    arr[i] = 0.0

        def csr_matvec_relu(data, indices, indptr, x, out, stream):
            n_rows = out.size
            threads = 256
            blocks = (n_rows + threads - 1) // threads
            _csr_matvec_kernel[blocks, threads, stream](data, indices, indptr, x, out)
            _relu_kernel[blocks, threads, stream](out)

        # A = [[1, -2], [-3, 4]]
        data = jnp.array([1.0, -2.0, -3.0, 4.0], dtype=jnp.float32)
        indices = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
        indptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        x = jnp.array([1.0, 1.0], dtype=jnp.float32)

        n_rows = 2
        f = numba_cuda_callable(
            csr_matvec_relu,
            outs=jax.ShapeDtypeStruct((n_rows,), jnp.float32),
        )

        result = f(data, indices, indptr, x)

        # Ax = [1-2, -3+4] = [-1, 1] -> ReLU -> [0, 1]
        expected = jnp.array([0.0, 1.0], dtype=jnp.float32)
        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((data, indices, indptr, x, result, expected))


# ===========================================================================
# 9. Neuroscience Domain: Event-Driven Spike Propagation
# ===========================================================================


@pytest.mark.skipif(not numba_cuda_available, reason="Numba CUDA not available")
class TestNumbaCudaCallableSpikePropagation(unittest.TestCase):
    """Event-driven synaptic integration via multi-kernel callable."""

    def test_event_scatter_add(self):
        from numba import cuda
        from brainevent import numba_cuda_callable

        @cuda.jit
        def _extract_spikes_kernel(spikes, spike_ids, n_spikes):
            if cuda.grid(1) == 0:
                count = 0
                for i in range(spikes.size):
                    if spikes[i] > 0.5:
                        spike_ids[count] = i
                        count += 1
                n_spikes[0] = count

        @cuda.jit
        def _scatter_add_kernel(spike_ids, n_spikes, weights, out):
            j = cuda.grid(1)
            n_post = out.size
            if j < n_post:
                total = 0.0
                ns = n_spikes[0]
                for s in range(ns):
                    i = spike_ids[s]
                    total += weights[i * n_post + j]
                out[j] = total

        def spike_propagate(spikes, weights_flat, out, stream):
            n_pre = spikes.size
            n_post = out.size

            spike_ids = cuda.device_array(n_pre, dtype=np.int32, stream=stream)
            n_spikes = cuda.device_array(1, dtype=np.int32, stream=stream)

            _extract_spikes_kernel[1, 1, stream](spikes, spike_ids, n_spikes)

            threads = 256
            blocks = (n_post + threads - 1) // threads
            _scatter_add_kernel[blocks, threads, stream](
                spike_ids, n_spikes, weights_flat, out
            )

        n_pre, n_post = 8, 16
        f = numba_cuda_callable(
            spike_propagate,
            outs=jax.ShapeDtypeStruct((n_post,), jnp.float32),
        )

        spikes = jnp.array([0, 1, 0, 1, 0, 1, 0, 0], dtype=jnp.float32)
        weights = jnp.array(
            [[(i * 0.1 + j * 0.01) for j in range(n_post)] for i in range(n_pre)],
            dtype=jnp.float32,
        ).flatten()

        result = f(spikes, weights)

        w_matrix = weights.reshape(n_pre, n_post)
        expected = w_matrix[1] + w_matrix[3] + w_matrix[5]
        self.assertTrue(jnp.allclose(result, expected, rtol=1e-5))
        jax.block_until_ready((spikes, weights, result, w_matrix, expected))


# ===========================================================================
# 10. Neuroscience Domain: STDP-like Weight Update
# ===========================================================================


@pytest.mark.skipif(not numba_cuda_available, reason="Numba CUDA not available")
class TestNumbaCudaCallableSTDP(unittest.TestCase):
    """Three-kernel STDP pipeline: compute delta_w, update, clip."""

    def test_stdp_weight_update(self):
        from numba import cuda
        from brainevent import numba_cuda_callable

        @cuda.jit
        def _compute_dw_kernel(pre_trace, post_trace, dw, n_pre, n_post):
            idx = cuda.grid(1)
            if idx < n_pre * n_post:
                i = idx // n_post
                j = idx % n_post
                dw[idx] = pre_trace[i] * post_trace[j]

        @cuda.jit
        def _update_weights_kernel(weights, dw, lr):
            i = cuda.grid(1)
            if i < weights.size:
                weights[i] += lr * dw[i]

        @cuda.jit
        def _clip_kernel(weights, w_max):
            i = cuda.grid(1)
            if i < weights.size:
                val = weights[i]
                if val < 0.0:
                    weights[i] = 0.0
                elif val > w_max:
                    weights[i] = w_max

        def stdp_update(pre_trace, post_trace, weights_in, weights_out, stream):
            n_pre = pre_trace.size
            n_post = post_trace.size
            total = n_pre * n_post

            threads = 256
            blocks = (total + threads - 1) // threads

            dw = cuda.device_array(total, dtype=np.float32, stream=stream)

            _compute_dw_kernel[blocks, threads, stream](
                pre_trace, post_trace, dw, n_pre, n_post
            )
            # weights_in and weights_out are aliased (same buffer)
            _update_weights_kernel[blocks, threads, stream](weights_out, dw, 0.01)
            _clip_kernel[blocks, threads, stream](weights_out, 1.0)

        n_pre, n_post = 4, 8
        f = numba_cuda_callable(
            stdp_update,
            outs=jax.ShapeDtypeStruct((n_pre * n_post,), jnp.float32),
            input_output_aliases={2: 0},
        )

        pre_trace = jnp.array([0.5, 0.3, 0.0, 0.8], dtype=jnp.float32)
        post_trace = jnp.array([0.1, 0.0, 0.4, 0.2, 0.6, 0.0, 0.3, 0.5], dtype=jnp.float32)
        weights = jnp.full(n_pre * n_post, 0.5, dtype=jnp.float32)

        result = f(pre_trace, post_trace, weights)

        dw = jnp.outer(pre_trace, post_trace).flatten()
        expected = jnp.clip(weights + 0.01 * dw, 0.0, 1.0)

        self.assertTrue(jnp.allclose(result, expected, atol=1e-3))
        jax.block_until_ready((pre_trace, post_trace, weights, result, dw, expected))


# ===========================================================================
# 11. Large Array Test
# ===========================================================================


@pytest.mark.skipif(not numba_cuda_available, reason="Numba CUDA not available")
class TestNumbaCudaCallableLargeArrays(unittest.TestCase):
    """Test with larger arrays to catch grid/block config issues."""

    def test_large_multi_kernel_pipeline(self):
        """Pipeline on a large array (1M elements)."""
        from numba import cuda
        from brainevent import numba_cuda_callable

        @cuda.jit
        def _add_k(a, b, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = a[i] + b[i]

        @cuda.jit
        def _mul_k(a, s, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = a[i] * s[0]

        def fn(a, b, s, out, stream):
            n = a.size
            threads = 256
            blocks = (n + threads - 1) // threads
            _add_k[blocks, threads, stream](a, b, out)
            _mul_k[blocks, threads, stream](out, s, out)

        n = 1_000_000
        f = numba_cuda_callable(
            fn,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
        )

        a = jnp.ones(n, dtype=jnp.float32) * 2.0
        b = jnp.ones(n, dtype=jnp.float32) * 3.0
        s = jnp.array([0.5], dtype=jnp.float32)

        result = f(a, b, s)
        expected = (a + b) * 0.5

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((a, b, s, result, expected))


# ===========================================================================
# 12. No-Input Callable (Pure Output Generation)
# ===========================================================================


@pytest.mark.skipif(not numba_cuda_available, reason="Numba CUDA not available")
class TestNumbaCudaCallableNoInputs(unittest.TestCase):
    """Callable with no XLA inputs -- only produces outputs."""

    def test_generate_sequence(self):
        from numba import cuda
        from brainevent import numba_cuda_callable

        @cuda.jit
        def _iota_kernel(out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = float(i)

        def iota_fn(out, stream):
            threads = 256
            blocks = (out.size + threads - 1) // threads
            _iota_kernel[blocks, threads, stream](out)

        n = 128
        f = numba_cuda_callable(
            iota_fn,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
        )

        result = f()
        expected = jnp.arange(n, dtype=jnp.float32)
        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((result, expected))


# ===========================================================================
# 13. Repeated Calls
# ===========================================================================


@pytest.mark.skipif(not numba_cuda_available, reason="Numba CUDA not available")
class TestNumbaCudaCallableRepeatedCalls(unittest.TestCase):
    """Ensure repeated calls don't cause issues (FFI handle leaks, etc.)."""

    def test_call_multiple_times(self):
        from numba import cuda
        from brainevent import numba_cuda_callable

        @cuda.jit
        def _add_k(a, b, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = a[i] + b[i]

        def add_fn(a, b, out, stream):
            _add_k[(a.size + 255) // 256, 256, stream](a, b, out)

        n = 256
        f = numba_cuda_callable(
            add_fn,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
        )

        a = jnp.arange(n, dtype=jnp.float32)
        b = jnp.ones(n, dtype=jnp.float32)
        expected = a + b

        for _ in range(10):
            result = f(a, b)
            self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((a, b, expected, result))


# ===========================================================================
# 14. Different Data Types
# ===========================================================================


@pytest.mark.skipif(not numba_cuda_available, reason="Numba CUDA not available")
class TestNumbaCudaCallableDtypes(unittest.TestCase):
    """Test callables with different data types."""

    def test_float64(self):
        from numba import cuda
        from brainevent import numba_cuda_callable

        @cuda.jit
        def _add_k(a, b, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = a[i] + b[i]

        def add_fn(a, b, out, stream):
            _add_k[(a.size + 255) // 256, 256, stream](a, b, out)

        n = 256
        f = numba_cuda_callable(
            add_fn,
            outs=jax.ShapeDtypeStruct((n,), jnp.float64),
        )

        a = jnp.arange(n, dtype=jnp.float64)
        b = jnp.ones(n, dtype=jnp.float64) * 2
        result = f(a, b)
        expected = a + b

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((a, b, result, expected))

    def test_int32(self):
        from numba import cuda
        from brainevent import numba_cuda_callable

        @cuda.jit
        def _add_k(a, b, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = a[i] + b[i]

        def add_fn(a, b, out, stream):
            _add_k[(a.size + 255) // 256, 256, stream](a, b, out)

        n = 256
        f = numba_cuda_callable(
            add_fn,
            outs=jax.ShapeDtypeStruct((n,), jnp.int32),
        )

        a = jnp.arange(n, dtype=jnp.int32)
        b = jnp.ones(n, dtype=jnp.int32) * 2
        result = f(a, b)
        expected = a + b

        self.assertTrue(jnp.allclose(result, expected))
        jax.block_until_ready((a, b, result, expected))


# ---------------------------------------------------------------------------
# Audit regression tests
# ---------------------------------------------------------------------------


class TestComputeLaunchConfig(unittest.TestCase):
    """``_compute_launch_config`` guards (M3) - pure, no GPU required."""

    def test_basic_1d_2d_3d(self):
        self.assertEqual(_compute_launch_config(1024), ((4,), (256,)))
        self.assertEqual(_compute_launch_config((64, 64)), ((4, 4), (16, 16)))
        grid, block = _compute_launch_config((16, 16, 8))
        self.assertEqual(block, (8, 8, 4))
        self.assertEqual(grid, (2, 2, 2))

    def test_zero_extent_no_zero_division(self):
        # Previously: block=(min(256,0),)=(0,) -> grid = -1 // 0 -> ZeroDivisionError.
        grid, block = _compute_launch_config(0)
        self.assertEqual(grid, (0,))
        self.assertEqual(block, (1,))  # clamped to >= 1
        # A zero extent on one axis of a multi-D launch is also safe.
        grid, block = _compute_launch_config((0, 32))
        self.assertEqual(grid[0], 0)
        self.assertTrue(all(b >= 1 for b in block))

    def test_negative_extent_raises(self):
        with self.assertRaises(ValueError):
            _compute_launch_config(-1)
        with self.assertRaises(ValueError):
            _compute_launch_config((8, -2))

    def test_bad_dimensionality_raises(self):
        with self.assertRaises(ValueError):
            _compute_launch_config(())  # zero dims
        with self.assertRaises(ValueError):
            _compute_launch_config((1, 2, 3, 4))  # > 3 dims

    def test_nonpositive_threads_per_block_raises(self):
        with self.assertRaises(ValueError):
            _compute_launch_config(1024, threads_per_block=0)


@requires_gpu
class TestKernelCorrectness(unittest.TestCase):
    """End-to-end correctness with the device-context binding in place (C3)."""

    def test_elementwise_add(self):
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def add_kernel(x, y, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i] + y[i]

        fn = numba_cuda_kernel(
            add_kernel,
            outs=jax.ShapeDtypeStruct((1024,), jnp.float32),
            launch_dims=1024,
        )
        x = jnp.arange(1024, dtype=jnp.float32)
        y = jnp.ones(1024, dtype=jnp.float32)
        out = np.asarray(jax.jit(fn)(x, y))
        np.testing.assert_allclose(out, np.arange(1024) + 1.0, rtol=1e-6)

    def test_float16_roundtrip(self):
        # float16 is supported by numba CUDA; it must round-trip (C2).
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def copy_kernel(x, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i]

        fn = numba_cuda_kernel(
            copy_kernel,
            outs=jax.ShapeDtypeStruct((256,), jnp.float16),
            launch_dims=256,
        )
        x = jnp.arange(256, dtype=jnp.float16)
        out = np.asarray(jax.jit(fn)(x))
        np.testing.assert_array_equal(out, np.arange(256, dtype=np.float16))


@requires_gpu
class TestErrorPropagation(unittest.TestCase):
    """A raising callback must surface as a JAX error, never silent success (C1)."""

    def test_callable_exception_propagates(self):
        from brainevent import numba_cuda_callable

        def boom(x, out, stream):
            raise RuntimeError('intentional-kernel-failure')

        fn = numba_cuda_callable(
            boom, outs=jax.ShapeDtypeStruct((4,), jnp.float32)
        )
        x = jnp.ones(4, dtype=jnp.float32)
        with self.assertRaises(Exception):
            # Force execution so the FFI error is materialised, not deferred.
            jax.block_until_ready(jax.jit(fn)(x))

    def test_bfloat16_rejected(self):
        # numba CUDA cannot launch bfloat16 kernels; the bridge rejects bf16
        # explicitly with a clear, specific error rather than letting numba fail
        # opaquely deep in the launch (F19/C2).
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def copy_kernel(x, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i]

        # A bf16 *output* is rejected at wrap time with a specific message.
        with self.assertRaisesRegex(ValueError, 'bfloat16'):
            numba_cuda_kernel(
                copy_kernel,
                outs=jax.ShapeDtypeStruct((128,), jnp.bfloat16),
                launch_dims=128,
            )

        # A bf16 *input* is rejected at trace time with a specific message.
        fn = numba_cuda_kernel(
            copy_kernel,
            outs=jax.ShapeDtypeStruct((128,), jnp.float32),
            launch_dims=128,
        )
        with self.assertRaisesRegex(ValueError, 'bfloat16'):
            fn(jnp.arange(128, dtype=jnp.bfloat16))


@requires_gpu
class TestEmptyLaunch(unittest.TestCase):
    """An empty problem must not crash the launch (M3)."""

    def test_zero_size_output(self):
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def add_kernel(x, y, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i] + y[i]

        fn = numba_cuda_kernel(
            add_kernel,
            outs=jax.ShapeDtypeStruct((0,), jnp.float32),
            launch_dims=0,
        )
        x = jnp.zeros((0,), dtype=jnp.float32)
        y = jnp.zeros((0,), dtype=jnp.float32)
        out = np.asarray(jax.jit(fn)(x, y))
        self.assertEqual(out.shape, (0,))


@requires_gpu
class TestRegistrationCaching(unittest.TestCase):
    """Repeated identical calls reuse a single FFI registration (H1)."""

    def test_kernel_registration_cached(self):
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def add_kernel(x, y, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i] + y[i]

        fn = numba_cuda_kernel(
            add_kernel,
            outs=jax.ShapeDtypeStruct((512,), jnp.float32),
            launch_dims=512,
        )
        x = jnp.arange(512, dtype=jnp.float32)
        y = jnp.ones(512, dtype=jnp.float32)

        before = len(numba_cuda_ffi._NUMBA_CUDA_FFI_TARGETS)
        for _ in range(5):
            jax.block_until_ready(fn(x, y))
        after = len(numba_cuda_ffi._NUMBA_CUDA_FFI_TARGETS)
        # Five identical-signature calls add exactly one registration.
        self.assertEqual(after - before, 1)


# ===========================================================================
# CPU-runnable unit tests (no GPU / numba.cuda needed): launch-config
# validation, bf16 rejection, and import-poisoning.  These exercise pure
# argument validation that fails fast *before* numba.cuda is imported, so they
# are deliberately NOT GPU-marked.
# ===========================================================================


def _dummy_kernel(x, out):  # a plain (non-CUDA) callable for validation tests
    pass


class TestNumbaCudaKernelLaunchValidation(unittest.TestCase):
    """Mutual-exclusivity / partial launch-config validation (F19)."""

    def test_no_config_raises(self):
        from brainevent import numba_cuda_kernel
        with self.assertRaises(ValueError):
            numba_cuda_kernel(_dummy_kernel, outs=jax.ShapeDtypeStruct((8,), jnp.float32))

    def test_grid_without_block_raises(self):
        from brainevent import numba_cuda_kernel
        with self.assertRaises(ValueError):
            numba_cuda_kernel(_dummy_kernel, outs=jax.ShapeDtypeStruct((8,), jnp.float32), grid=4)

    def test_block_without_grid_raises(self):
        from brainevent import numba_cuda_kernel
        with self.assertRaises(ValueError):
            numba_cuda_kernel(_dummy_kernel, outs=jax.ShapeDtypeStruct((8,), jnp.float32), block=256)

    def test_grid_block_and_launch_dims_conflict_raises(self):
        from brainevent import numba_cuda_kernel
        with self.assertRaises(ValueError):
            numba_cuda_kernel(
                _dummy_kernel,
                outs=jax.ShapeDtypeStruct((8,), jnp.float32),
                grid=4, block=256, launch_dims=1024,
            )

    def test_explicit_grid_block_with_vmap_method_raises(self):
        """Explicit grid/block + vmap_method is refused at wrap time (F5)."""
        from brainevent import numba_cuda_kernel
        with self.assertRaises(ValueError):
            numba_cuda_kernel(
                _dummy_kernel,
                outs=jax.ShapeDtypeStruct((8,), jnp.float32),
                grid=4, block=256, vmap_method='expand_dims',
            )

    def test_launch_dims_with_vmap_method_is_allowed(self):
        """launch_dims + vmap_method must NOT raise at wrap time (F5)."""
        from brainevent import numba_cuda_kernel
        # Reaches numba import + CUDADispatcher type check; on a non-CUDA build
        # that surfaces as ImportError/TypeError, never a launch-config ValueError.
        try:
            numba_cuda_kernel(
                _dummy_kernel,
                outs=jax.ShapeDtypeStruct((8,), jnp.float32),
                launch_dims=8, vmap_method='expand_dims',
            )
        except ValueError as exc:  # pragma: no cover - would be a regression
            self.fail(f"launch_dims + vmap_method should not raise ValueError: {exc}")
        except (ImportError, TypeError):
            pass  # expected on a machine without numba.cuda / with a plain callable

    def test_bfloat16_output_rejected_at_wrap_time(self):
        from brainevent import numba_cuda_kernel
        with self.assertRaisesRegex(ValueError, 'bfloat16'):
            numba_cuda_kernel(
                _dummy_kernel,
                outs=jax.ShapeDtypeStruct((8,), jnp.bfloat16),
                launch_dims=8,
            )


@pytest.mark.skipif(not numba_installed, reason="numba not installed")
class TestImportNumbaCudaPoisoning(unittest.TestCase):
    """A transient CUDA error must not poison the availability flag (F19)."""

    def test_transient_cuda_error_does_not_poison_flag(self):
        import numba
        m = numba_cuda_ffi

        saved_cuda = m.cuda
        saved_flag = m.numba_cuda_installed
        saved_is_available = numba.cuda.is_available
        try:
            m.cuda = None
            m.numba_cuda_installed = True

            def _boom():
                raise RuntimeError('CUDA_ERROR_NOT_INITIALIZED')

            numba.cuda.is_available = _boom
            with self.assertRaises(ImportError):
                m.import_numba_cuda()
            # The flag reflects "numba importable", so a transient CUDA failure
            # leaves it True (not poisoned).
            self.assertTrue(m.numba_cuda_installed)

            # A retry in a healthy context then succeeds.
            numba.cuda.is_available = lambda: True
            self.assertIs(m.import_numba_cuda(), numba.cuda)
            self.assertTrue(m.numba_cuda_installed)
        finally:
            numba.cuda.is_available = saved_is_available
            m.cuda = saved_cuda
            m.numba_cuda_installed = saved_flag


# ===========================================================================
# GPU tests for the launch-policy / vmap / zero-fill / naming fixes.
# ===========================================================================


@requires_gpu
class TestVmapLaunchDims(unittest.TestCase):
    """vmap over a launch_dims kernel must compute EVERY batched element (F5).

    Per-slice launches (not a single flattened launch) are required so that
    coupled-row kernels — where an output element depends on other elements of
    the same batch row — do not read/write across batch boundaries.
    """

    def test_vmap_elementwise_all_rows_correct(self):
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def add_kernel(x, y, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i] + y[i]

        n, batch = 16, 8
        for vm in ('expand_dims', 'broadcast_all'):
            fn = numba_cuda_kernel(
                add_kernel,
                outs=jax.ShapeDtypeStruct((n,), jnp.float32),
                launch_dims=n,
                vmap_method=vm,
            )
            x = jnp.arange(batch * n, dtype=jnp.float32).reshape(batch, n)
            y = jnp.ones((batch, n), dtype=jnp.float32)
            r = np.asarray(jax.vmap(fn)(x, y))
            jax.block_until_ready(r)
            expected = np.asarray(x) + 1.0
            np.testing.assert_allclose(r, expected, rtol=1e-6,
                                       err_msg=f"vmap_method={vm}")
            self.assertTrue(np.isclose(r, expected).all(),
                            f"some batched rows uninitialised for vmap_method={vm}")

    def test_vmap_coupled_rows_reversal(self):
        """A row-coupling kernel (each out[i] reads a different x element) must
        stay within its batch slice.  This FAILS under a flattened/folded launch
        (rows bleed together) and passes only with per-slice launches."""
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def reverse_kernel(x, out):
            i = cuda.grid(1)
            n = out.size
            if i < n:
                out[i] = x[n - 1 - i]

        n, batch = 4, 4
        for vm in ('expand_dims', 'broadcast_all'):
            fn = numba_cuda_kernel(
                reverse_kernel,
                outs=jax.ShapeDtypeStruct((n,), jnp.float32),
                launch_dims=n,
                vmap_method=vm,
            )
            x = jnp.arange(batch * n, dtype=jnp.float32).reshape(batch, n)
            r = np.asarray(jax.vmap(fn)(x))
            jax.block_until_ready(r)
            expected = np.asarray(x)[:, ::-1]  # per-row reversal
            np.testing.assert_array_equal(r, expected)

    def test_vmap_unmapped_input_in_axes_none(self):
        """vmap in_axes=(0, None): the unmapped input arrives with a leading
        axis of size 1 (expand_dims) or B (broadcast_all); both must broadcast
        correctly across slices, never read out of bounds."""
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def add_kernel(x, y, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i] + y[i]

        n, batch = 8, 4
        for vm in ('expand_dims', 'broadcast_all'):
            fn = numba_cuda_kernel(
                add_kernel,
                outs=jax.ShapeDtypeStruct((n,), jnp.float32),
                launch_dims=n,
                vmap_method=vm,
            )
            x = jnp.arange(batch * n, dtype=jnp.float32).reshape(batch, n)
            y = jnp.arange(n, dtype=jnp.float32)  # unmapped
            r = np.asarray(jax.vmap(fn, in_axes=(0, None))(x, y))
            jax.block_until_ready(r)
            expected = np.asarray(x) + np.asarray(y)[None, :]
            np.testing.assert_allclose(r, expected, rtol=1e-6,
                                       err_msg=f"vmap_method={vm}")

    def test_vmap_batch_size_one(self):
        """Batch-1 vmap adds a leading axis of size 1; rank-based detection must
        treat it as vmapped (a size-ratio check would see B==1 and mis-handle
        the extra rank)."""
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def add_kernel(x, y, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i] + y[i]

        n = 8
        fn = numba_cuda_kernel(
            add_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
            launch_dims=n,
            vmap_method='expand_dims',
        )
        x = jnp.arange(1 * n, dtype=jnp.float32).reshape(1, n)
        y = jnp.ones((1, n), dtype=jnp.float32)
        r = np.asarray(jax.vmap(fn)(x, y))
        jax.block_until_ready(r)
        self.assertEqual(r.shape, (1, n))
        np.testing.assert_allclose(r, np.asarray(x) + 1.0, rtol=1e-6)

    def test_vmap_unbatched_still_correct(self):
        # No vmap (equal ranks) leaves the single-launch path bit-identical.
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def add_kernel(x, y, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i] + y[i]

        n = 32
        fn = numba_cuda_kernel(
            add_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
            launch_dims=n,
            vmap_method='expand_dims',
        )
        x = jnp.arange(n, dtype=jnp.float32)
        y = jnp.ones(n, dtype=jnp.float32)
        r = np.asarray(jax.jit(fn)(x, y))
        jax.block_until_ready(r)
        np.testing.assert_allclose(r, np.asarray(x) + 1.0, rtol=1e-6)


@requires_gpu
class TestZeroGridZeroFill(unittest.TestCase):
    """A skipped (zero) launch with a non-empty output zero-fills it (F9)."""

    def test_launch_dims_zero_nonempty_output_is_zeroed(self):
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def add_one(x, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i] + 1.0

        n = 128
        # Alias input 0 onto output 0 so the output buffer *starts* holding the
        # (non-zero) input values; if the skipped launch failed to zero-fill,
        # the result would be those donated values, not zeros.
        fn = numba_cuda_kernel(
            add_one,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
            launch_dims=0,
            input_output_aliases={0: 0},
        )
        x = jnp.arange(1, n + 1, dtype=jnp.float32)  # all non-zero
        out = np.asarray(jax.jit(fn)(x))
        jax.block_until_ready(out)
        np.testing.assert_array_equal(out, np.zeros(n, dtype=np.float32))


@requires_gpu
class TestTwoShapesOneTarget(unittest.TestCase):
    """Distinct input shapes reuse one registered target (F8)."""

    def test_two_input_shapes_one_target(self):
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def copy_kernel(x, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i]

        fn = numba_cuda_kernel(
            copy_kernel,
            outs=jax.ShapeDtypeStruct((8,), jnp.float32),
            launch_dims=8,
        )
        before = len(numba_cuda_ffi._NUMBA_CUDA_FFI_TARGETS)
        r1 = np.asarray(fn(jnp.arange(8, dtype=jnp.float32)))
        r2 = np.asarray(fn(jnp.arange(16, dtype=jnp.float32)))
        jax.block_until_ready((r1, r2))
        after = len(numba_cuda_ffi._NUMBA_CUDA_FFI_TARGETS)
        # Two distinct input shapes -> exactly one registration.
        self.assertEqual(after - before, 1)
        np.testing.assert_array_equal(r1, np.arange(8, dtype=np.float32))
        np.testing.assert_array_equal(r2, np.arange(8, dtype=np.float32))


@requires_gpu
class TestContentDerivedNaming(unittest.TestCase):
    """FFI target names are content-derived and reused across objects (F14)."""

    @staticmethod
    def _make_kernel():
        # Two separate dispatcher objects with byte-identical content must
        # fingerprint the same and share one target.  (ffi_naming now serialises
        # modules in closure cells, so whether ``cuda`` is a global or a closed-
        # over module, the fingerprint is deterministic and non-None.)
        @cuda.jit
        def scale_kernel(x, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i] * 2.0

        return scale_kernel

    def test_identical_content_shares_target(self):
        m = numba_cuda_ffi
        f32 = np.dtype('float32')
        lm = ('launch_dims', (8,), 256)

        k1 = self._make_kernel()
        k2 = self._make_kernel()  # separate dispatcher, identical content

        name1, _ = m._register_numba_cuda_ffi_target(k1, (f32,), ((8,),), (f32,), lm, 0)
        name2, _ = m._register_numba_cuda_ffi_target(k2, (f32,), ((8,),), (f32,), lm, 0)

        self.assertTrue(name1.startswith('brainevent_numba_cuda_ffi_'))
        # Content-derived: two distinct objects with identical content share one
        # target (and one registration).
        self.assertEqual(name1, name2)

        # Same kernel wrapped with a different `outs` (different abstract size)
        # must map to a DIFFERENT target.
        name3, _ = m._register_numba_cuda_ffi_target(k1, (f32,), ((16,),), (f32,), lm, 0)
        self.assertNotEqual(name1, name3)


@requires_gpu
class TestKernelZeroDimRejection(unittest.TestCase):
    """0-d inputs are rejected at trace time on the kernel path (F19)."""

    def test_scalar_input_rejected(self):
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def copy_kernel(x, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i]

        fn = numba_cuda_kernel(
            copy_kernel,
            outs=jax.ShapeDtypeStruct((1,), jnp.float32),
            launch_dims=1,
        )
        with self.assertRaises(ValueError):
            fn(jnp.asarray(3.0))  # 0-d scalar input


@requires_gpu
class TestNestedVmapRejection(unittest.TestCase):
    """MEDIUM fix: nested vmap (runtime rank == abstract rank + 2) must be
    rejected loudly, not silently treated as unbatched.

    The callback detects vmap by rank: ``extra_axes = runtime_rank -
    abstract_rank`` on output 0. Pre-fix, only ``extra_axes == 1`` (a single
    level of vmap) was handled; a *nested* ``jax.vmap(jax.vmap(f))`` produces
    ``extra_axes == 2``, which fell through to the ``vmapped = (extra_axes ==
    1)`` -> ``False`` branch -- i.e. treated as the unbatched/no-vmap path --
    and silently reconstructed the wrong shape, returning garbage for every
    slice but the first. The callback now checks ``extra_axes not in (0, 1)``
    and returns an ``INVALID_ARGUMENT`` FFI error mentioning "only one level
    of vmap" instead.
    """

    def test_nested_vmap_raises_and_single_vmap_still_works(self):
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def add_kernel(x, y, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i] + y[i]

        n = 8
        fn = numba_cuda_kernel(
            add_kernel,
            outs=jax.ShapeDtypeStruct((n,), jnp.float32),
            launch_dims=n,
            vmap_method='expand_dims',
        )

        # Single vmap (extra_axes == 1) is unaffected by this fix and must
        # still work correctly.
        b1 = 4
        x1 = jnp.arange(b1 * n, dtype=jnp.float32).reshape(b1, n)
        y1 = jnp.ones((b1, n), dtype=jnp.float32)
        r1 = np.asarray(jax.vmap(fn)(x1, y1))
        jax.block_until_ready(r1)
        np.testing.assert_allclose(r1, np.asarray(x1) + 1.0, rtol=1e-6)

        # Nested vmap (extra_axes == 2) must be rejected, not silently
        # mis-batched.
        b_outer, b_inner = 2, 3
        x2 = jnp.arange(b_outer * b_inner * n, dtype=jnp.float32).reshape(b_outer, b_inner, n)
        y2 = jnp.ones((b_outer, b_inner, n), dtype=jnp.float32)
        with pytest.raises(Exception, match='only one level of vmap'):
            jax.block_until_ready(jax.vmap(jax.vmap(fn))(x2, y2))
