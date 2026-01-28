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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent

warp_installed = importlib.util.find_spec('warp') is not None
gpu_available = jax.default_backend() == 'gpu'


@pytest.mark.skipif(not warp_installed or not gpu_available, reason="warp not installed or no GPU")
class TestWarpFFI(unittest.TestCase):
    """
    Examples of using _ffi_gpu_lowering via the public brainevent API.

    The FFI path is activated when you pass a warp kernel generator to
    ``XLACustomKernel.def_gpu_kernel(warp=...)``.  Internally this calls
    ``register_warp_gpu_translation`` -> ``_ffi_gpu_lowering``.

    All examples below are compatible with ``jax.jit``.
    """

    # -- Example 1: basic element-wise kernel (static dim) ------------------

    def test_elementwise_square(self):
        """Square every element: y[i] = x[i] * x[i].

        Demonstrates:
        - Defining a warp kernel function (raw, not @wp.kernel decorated)
        - Using brainevent.warp_kernel() with a fixed dim
        - Running both eagerly and under jax.jit
        """
        import warp as wp

        def square_kernel(
            x: wp.array1d(dtype=float),
            y: wp.array1d(dtype=float),
        ):
            i = wp.tid()
            y[i] = x[i] * x[i]

        data = jnp.arange(1, 10, dtype=jnp.float32)

        # The warp= argument is a kernel generator: callable(**kwargs) -> WarpKernel
        op = brainevent.XLACustomKernel(name="square")
        op.def_gpu_kernel(
            warp=lambda **kwargs: brainevent.warp_kernel(
                square_kernel,
                dim=data.shape,
            )
        )

        # Works eagerly
        r = op(data, outs=jax.ShapeDtypeStruct(data.shape, data.dtype))
        np.testing.assert_allclose(np.array(r), np.arange(1, 10) ** 2, rtol=1e-5)

        # Works under jax.jit
        r_jit = jax.jit(lambda x: op(x, outs=jax.ShapeDtypeStruct(x.shape, x.dtype)))(data)
        np.testing.assert_allclose(np.array(r_jit), np.arange(1, 10) ** 2, rtol=1e-5)

    # -- Example 2: two-input kernel under jax.jit -------------------------

    def test_elementwise_add(self):
        """Add two vectors: z[i] = x[i] + y[i].

        Demonstrates:
        - Multiple input arrays
        - Running under jax.jit
        """
        import warp as wp

        def add_kernel(
            x: wp.array1d(dtype=float),
            y: wp.array1d(dtype=float),
            z: wp.array1d(dtype=float),
        ):
            i = wp.tid()
            z[i] = x[i] + y[i]

        n = 64
        a = jnp.ones(n, dtype=jnp.float32) * 3.0
        b = jnp.ones(n, dtype=jnp.float32) * 7.0

        op = brainevent.XLACustomKernel(name="add_vec")
        op.def_gpu_kernel(
            warp=lambda **kwargs: brainevent.warp_kernel(add_kernel, dim=(n,))
        )

        @jax.jit
        def run(x, y):
            return op(x, y, outs=jax.ShapeDtypeStruct((n,), jnp.float32))

        r = run(a, b)
        np.testing.assert_allclose(np.array(r), np.full(n, 10.0), rtol=1e-5)

    # -- Example 3: dynamic dim from kwargs ---------------------------------

    def test_dynamic_dim(self):
        """Kernel whose launch dim is inferred from the ``outs`` kwarg.

        Demonstrates:
        - Using a callable for dim= that reads shape from outs at trace time
        - Works with jax.jit because dim is resolved during tracing
        """
        import warp as wp

        def copy_kernel(
            src: wp.array1d(dtype=float),
            dst: wp.array1d(dtype=float),
        ):
            i = wp.tid()
            dst[i] = src[i]

        op = brainevent.XLACustomKernel(name="copy")
        op.def_gpu_kernel(
            warp=lambda **kwargs: brainevent.warp_kernel(
                copy_kernel,
                dim=lambda **kw: kw["outs"][0].shape,
            )
        )

        @jax.jit
        def run(x):
            return op(x, outs=jax.ShapeDtypeStruct(x.shape, x.dtype))

        data = jnp.arange(32, dtype=jnp.float32)
        r = run(data)
        np.testing.assert_array_equal(np.array(r), np.array(data))

    # -- Example 4: scalar (1-element array) input --------------------------

    def test_scalar_input(self):
        """Scale a vector by a scalar stored in a 1-element array.

        Demonstrates:
        - Passing a scalar array (shape (1,)) as a kernel input
        - Multiple inputs with different shapes
        """
        import warp as wp

        def scale_kernel(
            x: wp.array1d(dtype=float),
            s: wp.array1d(dtype=float),
            y: wp.array1d(dtype=float),
        ):
            i = wp.tid()
            y[i] = s[0] * x[i]

        n = 16
        data = jnp.arange(1, n + 1, dtype=jnp.float32)
        scalar = jnp.array([2.5], dtype=jnp.float32)

        op = brainevent.XLACustomKernel(name="scale")
        op.def_gpu_kernel(
            warp=lambda **kwargs: brainevent.warp_kernel(scale_kernel, dim=(n,))
        )

        @jax.jit
        def run(x, s):
            return op(x, s, outs=jax.ShapeDtypeStruct((n,), jnp.float32))

        r = run(data, scalar)
        np.testing.assert_allclose(np.array(r), np.arange(1, n + 1) * 2.5, rtol=1e-5)

    # -- Example 5: dynamic dtype (generator reads outs at trace time) ------

    def test_dynamic_dtype(self):
        """Kernel dtype determined at trace time from ``outs``.

        Demonstrates:
        - A kernel generator that creates different warp kernels depending
          on the output dtype (resolved at jax.jit trace time)
        - Using brainevent.jaxtype_to_warptype() for dtype conversion
        """
        import warp as wp

        def generate(**kwargs):
            out_info = kwargs["outs"][0]
            dtype = brainevent.jaxtype_to_warptype(out_info.dtype)

            def negate(
                x: wp.array1d(dtype=dtype),
                y: wp.array1d(dtype=dtype),
            ):
                i = wp.tid()
                y[i] = -x[i]

            return brainevent.warp_kernel(
                negate,
                dim=lambda **kw: kw["outs"][0].shape,
            )

        op = brainevent.XLACustomKernel(name="negate")
        op.def_gpu_kernel(warp=generate)

        @jax.jit
        def run(x):
            return op(x, outs=jax.ShapeDtypeStruct(x.shape, x.dtype))

        for dtype in [jnp.float32, jnp.float16]:
            data = jnp.ones(8, dtype=dtype) * 5.0
            r = run(data)
            np.testing.assert_allclose(np.array(r, dtype=np.float32), -5.0 * np.ones(8), atol=1e-2)
