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

"""Reproduction tests for the numba-CUDA FFI bridge audit fixes.

Covers the GPU-bridge findings from ``dev/2026-06-13-op-issues.md``:

* C1 - a raising kernel/callable surfaces as a JAX error, not silent success.
* C2 - ``float16`` round-trips; ``bfloat16`` (unsupported by numba CUDA) is
  rejected loudly instead of mis-decoded.
* M3 - ``_compute_launch_config`` guards zero/negative extents (no
  ``ZeroDivisionError``); an empty launch is skipped without crashing.
* H1 - repeated calls with an identical signature reuse one FFI registration.

The pure-function ``_compute_launch_config`` tests run on any platform; the
end-to-end tests require a real GPU and are skipped otherwise.
"""

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

# --- backend / numba-cuda detection ----------------------------------------
numba_installed = importlib.util.find_spec('numba') is not None
numba_cuda_available = False
if numba_installed:
    try:
        from numba import cuda

        numba_cuda_available = cuda.is_available()
    except ImportError:
        pass
numba_cuda_available = numba_cuda_available and (jax.default_backend() == 'gpu')

requires_gpu = pytest.mark.skipif(
    not numba_cuda_available, reason='Numba CUDA / GPU backend not available'
)


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
        # numba CUDA cannot launch bfloat16 kernels; the bridge must reject the
        # call (an error) rather than silently mis-decode the bytes (C2).
        from numba import cuda
        from brainevent import numba_cuda_kernel

        @cuda.jit
        def copy_kernel(x, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = x[i]

        fn = numba_cuda_kernel(
            copy_kernel,
            outs=jax.ShapeDtypeStruct((128,), jnp.bfloat16),
            launch_dims=128,
        )
        x = jnp.arange(128, dtype=jnp.bfloat16)
        with self.assertRaises(Exception):
            jax.block_until_ready(jax.jit(fn)(x))


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


if __name__ == '__main__':
    unittest.main()
