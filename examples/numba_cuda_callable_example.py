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

"""
Demonstrates the numba_cuda_callable interface (multi-kernel XLA FFI).

Unlike numba_cuda_kernel (one fixed kernel), numba_cuda_callable lets you
write a Python function that receives Numba device arrays plus the CUDA stream
from XLA, and then launch any sequence of @cuda.jit kernels on that stream.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
from numba import cuda

from brainevent import numba_cuda_callable


# --- CUDA kernels ------------------------------------------------------------

@cuda.jit
def add_kernel(x, y, out, n):
    i = cuda.grid(1)
    if i < n:
        out[i] = x[i] + y[i]


@cuda.jit
def scale_kernel(out, scale, n):
    i = cuda.grid(1)
    if i < n:
        out[i] = out[i] * scale


@cuda.jit
def axpy_kernel(a, x, y, out, n):
    i = cuda.grid(1)
    if i < n:
        out[i] = a[0] * x[i] + y[i]


# --- Callable wrappers -------------------------------------------------------

def add_then_scale(x, y, out, stream):
    n = x.size
    threads = 256
    blocks = (n + threads - 1) // threads
    add_kernel[blocks, threads, stream](x, y, out, n)
    scale_kernel[blocks, threads, stream](out, 2.0, n)


def fused_axpy(a, x, y, out, stream):
    n = x.size
    threads = 256
    blocks = (n + threads - 1) // threads
    axpy_kernel[blocks, threads, stream](a, x, y, out, n)


# --- Demo --------------------------------------------------------------------

def main():
    if not cuda.is_available():
        print("CUDA is not available; skipping demo.")
        return

    n = 1024
    x = jnp.linspace(0, 1, n, dtype=jnp.float32)
    y = jnp.ones(n, dtype=jnp.float32) * 3

    # numba_cuda_callable infers output shape from outs but keeps dtype;
    # resize outputs with x.shape at call time by passing shape-correct dummy outs.
    out_spec = jax.ShapeDtypeStruct(x.shape, x.dtype)

    add_then_scale_impl = numba_cuda_callable(add_then_scale, outs=out_spec)
    fused_axpy_impl = numba_cuda_callable(fused_axpy, outs=out_spec)

    @jax.jit
    def run_all(a, b):
        r1 = add_then_scale_impl(a, b)
        r2 = fused_axpy_impl(jnp.array([2.0], dtype=a.dtype), a, b)  # 2 * a + b
        return r1, r2

    r_add_scale, r_axpy = run_all(x, y)

    print("add_then_scale first 5:", np.asarray(r_add_scale[:5]))
    print("expected first 5:", np.asarray((x + y) * 2)[:5])
    print("axpy first 5:", np.asarray(r_axpy[:5]))
    print("axpy expected first 5:", np.asarray(2.0 * x + y)[:5])
    print("add_then_scale allclose:", np.allclose(r_add_scale, (x + y) * 2))
    print("axpy allclose:", np.allclose(r_axpy, 2.0 * x + y))


if __name__ == "__main__":
    main()
