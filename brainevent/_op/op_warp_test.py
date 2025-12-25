# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

import importlib.util
import unittest

import brainstate as bst
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent

warp_installed = importlib.util.find_spec('warp') is not None

if warp_installed:
    import warp as wp


@pytest.mark.skipif(
    jax.default_backend() != 'gpu' or not warp_installed,
    reason="No GPU available, or warp not installed",
)
class TestWarpGPU(unittest.TestCase):
    def test_warp1(self):
        # generic kernel definition using Any as a placeholder for concrete types
        @wp.kernel
        def scale(x: wp.array1d(dtype=float), y: wp.array1d(dtype=float), ):
            i = wp.tid()
            y[i] = x[i] * x[i]

        data = jnp.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=jnp.float32)

        op = brainevent.XLACustomKernel(
            name="scale",
            gpu_kernel=brainevent.WarpKernelGenerator(
                lambda **kwargs: scale,
                dim=data.shape,
            ),
        )
        r = op.call(data, outs=jax.ShapeDtypeStruct(data.shape, data.dtype))
        print(r)

        self.assertTrue(jnp.allclose(r, data * data))

    def test_warp_change_with_dtype(self):
        def generate(**kwargs):
            outs = kwargs["outs"][0]
            dtype = brainevent.jaxtype_to_warptype(outs.dtype)

            # generic kernel definition using Any as a placeholder for concrete types
            @wp.kernel
            def scale(x: wp.array1d(dtype=dtype),
                      y: wp.array1d(dtype=dtype)):
                i = wp.tid()
                y[i] = x[i] * x[i]

            return scale

        op = brainevent.XLACustomKernel(
            name="scale",
            gpu_kernel=brainevent.WarpKernelGenerator(
                generate,
                dim=lambda **kwargs: kwargs["outs"][0].shape,
            ),
        )

        @jax.jit
        def f(x):
            return op.call(x, outs=jax.ShapeDtypeStruct(x.shape, x.dtype))

        with bst.environ.context(precision=64):
            print(f(jnp.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=jnp.float32)))
            print(f(bst.random.rand(20, dtype=jnp.float32)))
            print(f(bst.random.rand(20, dtype=jnp.float16)))
            print(f(bst.random.rand(20, dtype=jnp.float64)))

    def test_warp_scalar(self):
        # generic kernel definition using Any as a placeholder for concrete types
        @wp.kernel
        def scale2(
            x: wp.array1d(dtype=float),
            s: wp.array1d(dtype=float),
            y: wp.array1d(dtype=float)
        ):
            i = wp.tid()
            y[i] = s[0] * x[i]

        data = jnp.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=jnp.float32)

        op = brainevent.XLACustomKernel(
            name="scale2",
            gpu_kernel=brainevent.WarpKernelGenerator(
                lambda **kwargs: scale2,
                dim=data.shape,
            ),
        )
        r = op.call(data, jnp.asarray([1.5]), outs=jax.ShapeDtypeStruct(data.shape, data.dtype))
        print(r)
        self.assertTrue(jnp.allclose(r, 1.5 * data))

    def test_warp_two_vectors(self):
        # generic kernel definition using Any as a placeholder for concrete types
        @wp.kernel
        def scale2(
            x: wp.array1d(dtype=float),
            y: wp.array1d(dtype=float),
            z: wp.array1d(dtype=float)
        ):
            i = wp.tid()
            z[i] = x[i] * y[i]

        xs = jnp.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=jnp.float32)
        ys = bst.random.rand_like(xs)

        op = brainevent.XLACustomKernel(
            name="scale2",
            gpu_kernel=brainevent.WarpKernelGenerator(
                lambda **kwargs: scale2,
                dim=xs.shape,
            ),
        )
        r = op.call(xs, ys, outs=jax.ShapeDtypeStruct(xs.shape, xs.dtype))
        print(r)

        self.assertTrue(jnp.allclose(r, xs * ys))

    def test_tile1(self):
        TILE_SIZE = wp.constant(256)
        TILE_THREADS = 64

        @wp.kernel
        def compute(
            a: wp.array2d(dtype=float),
            b: wp.array2d(dtype=float),
        ):
            # obtain our block index
            i = wp.tid()

            # load a row from global memory
            t = wp.tile_load(a[i], 0, TILE_SIZE)

            # cooperatively compute the sum of the tile elements; s is a 1x1 tile
            s = wp.tile_sum(t)

            # store s in global memory
            wp.tile_store(b[0], i, s)

        N = 10
        a_np = np.arange(N).reshape(-1, 1) * np.ones((1, 256), dtype=float)

        op = brainevent.XLACustomKernel(
            name="mm",
            gpu_kernel=brainevent.WarpKernelGenerator(
                lambda **kwargs: compute,
                dim=(a_np.shape[0], TILE_THREADS),
                block_dim=TILE_THREADS,
            ),
        )
        r = op.call(
            jax.numpy.asarray(a_np, dtype=jax.numpy.float32),
            outs=jax.core.ShapedArray([1, N], dtype=jax.numpy.float32)
        )
        r_true = a_np.sum(axis=1)
        print(r)
        print(r_true)
        self.assertTrue(jnp.allclose(r[0], r_true))

    def test_tile_matrix_multiplication(self):
        TILE_M = wp.constant(8)
        TILE_N = wp.constant(4)
        TILE_K = wp.constant(8)
        TILE_THREADS = 64

        @wp.kernel
        def tile_gemm(
            A: wp.array2d(dtype=float),
            B: wp.array2d(dtype=float),
            C: wp.array2d(dtype=float),
        ):
            # output tile index
            i, j = wp.tid()

            sum = wp.tile_zeros(m=TILE_M, n=TILE_N, dtype=wp.float32)

            M = A.shape[0]
            N = B.shape[1]
            K = A.shape[1]

            count = int(K / TILE_K)

            for k in range(0, count):
                a = wp.tile_load(A, i, k, m=TILE_M, n=TILE_K)
                b = wp.tile_load(B, k, j, m=TILE_K, n=TILE_N)

                # sum += a*b
                wp.tile_matmul(a, b, sum)

            wp.tile_store(C, i, j, sum)

        # generate some tile aligned matrix dimensions
        M = TILE_M * 7
        K = TILE_K * 6
        N = TILE_N * 5

        bst.random.seed(42)
        A = bst.random.random((M, K), dtype=np.float32)
        B = bst.random.random((K, N), dtype=np.float32)
        C_true = A @ B

        op = brainevent.XLACustomKernel(
            name="mm",
            gpu_kernel=brainevent.WarpKernelGenerator(
                lambda **kwargs: tile_gemm,
                dim=(int(M / TILE_M), int(N / TILE_N), TILE_THREADS),
                block_dim=TILE_THREADS,
            ),
        )
        r = op.call(A, B, outs=jax.core.ShapedArray([M, N], dtype=jax.numpy.float32))

        self.assertTrue(jnp.allclose(r, C_true, atol=1e-3))
