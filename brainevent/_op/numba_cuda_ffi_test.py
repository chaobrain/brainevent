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

# Check if Numba CUDA is available
numba_installed = importlib.util.find_spec('numba') is not None
numba_cuda_available = False
if numba_installed:
    try:
        from numba import cuda
        numba_cuda_available = cuda.is_available()
    except ImportError:
        pass


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
        """Test that non-CUDA kernel raises AssertionError."""
        from brainevent import numba_cuda_kernel

        def regular_function(x, out):
            pass

        with self.assertRaises(AssertionError):
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

        # Track if XLA stream was used
        stream_ptrs = []
        orig_get_stream = ffi._get_stream_from_callframe

        def tracking_get_stream(call_frame):
            ptr = orig_get_stream(call_frame)
            stream_ptrs.append(ptr)
            return ptr

        ffi._get_stream_from_callframe = tracking_get_stream

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
        finally:
            ffi._get_stream_from_callframe = orig_get_stream


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
