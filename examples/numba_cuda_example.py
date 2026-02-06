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
Simple example demonstrating Numba CUDA FFI integration with JAX.

This example shows how to use @cuda.jit kernels from JAX via the
numba_cuda_kernel interface.
"""

import jax
import jax.numpy as jnp
import numpy as np
from numba import cuda

from brainevent import numba_cuda_kernel


# Example 1: Simple element-wise addition
@cuda.jit
def add_kernel(x, y, out):
    """Element-wise addition kernel."""
    i = cuda.grid(1)
    if i < out.size:
        out[i] = x[i] + y[i]


# Example 2: Element-wise multiplication with scaling
@cuda.jit
def scale_mul_kernel(x, y, scale, out):
    """Element-wise multiplication with a scalar scale factor."""
    i = cuda.grid(1)
    if i < out.size:
        out[i] = x[i] * y[i] * scale[0]


# Example 3: Vector norm (uses shared memory for reduction)
@cuda.jit
def squared_sum_kernel(x, out):
    """Compute sum of squared elements (partial reduction per block)."""
    shared = cuda.shared.array(256, dtype=np.float32)

    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    i = cuda.grid(1)

    # Load and square
    if i < x.size:
        shared[tid] = x[i] * x[i]
    else:
        shared[tid] = 0.0

    cuda.syncthreads()

    # Reduction within block
    s = 128
    while s > 0:
        if tid < s:
            shared[tid] += shared[tid + s]
        cuda.syncthreads()
        s //= 2

    # Write block result
    if tid == 0:
        out[bid] = shared[0]


def main():
    print("=" * 60)
    print("Numba CUDA FFI Example")
    print("=" * 60)

    # Check if CUDA is available
    if not cuda.is_available():
        print("CUDA is not available. Skipping example.")
        return

    print(f"CUDA device: {cuda.get_current_device().name}")
    print()

    # --- Example 1: Simple addition ---
    print("Example 1: Element-wise addition")
    print("-" * 40)

    n = 1024
    a = jnp.arange(n, dtype=jnp.float32)
    b = jnp.ones(n, dtype=jnp.float32) * 2

    # Create JAX-callable kernel with explicit grid/block
    add_jax = numba_cuda_kernel(
        add_kernel,
        outs=jax.ShapeDtypeStruct((n,), jnp.float32),
        grid=4,
        block=256,
    )

    result = add_jax(a, b)
    expected = a + b

    print(f"  Input a[:5]: {a[:5]}")
    print(f"  Input b[:5]: {b[:5]}")
    print(f"  Result[:5]:  {result[:5]}")
    print(f"  Expected[:5]: {expected[:5]}")
    print(f"  All close: {jnp.allclose(result, expected)}")
    print()

    # --- Example 2: Using launch_dims (auto grid/block) ---
    print("Example 2: Using launch_dims for auto grid/block")
    print("-" * 40)

    # Alternative: use launch_dims instead of explicit grid/block
    add_jax_auto = numba_cuda_kernel(
        add_kernel,
        outs=jax.ShapeDtypeStruct((n,), jnp.float32),
        launch_dims=n,
        threads_per_block=128,
    )

    result_auto = add_jax_auto(a, b)
    print(f"  Result[:5]:  {result_auto[:5]}")
    print(f"  All close: {jnp.allclose(result_auto, expected)}")
    print()

    # --- Example 3: Inside jax.jit ---
    print("Example 3: Inside @jax.jit")
    print("-" * 40)

    @jax.jit
    def jitted_add(x, y):
        return add_jax(x, y)

    result_jit = jitted_add(a, b)
    print(f"  Result[:5]:  {result_jit[:5]}")
    print(f"  All close: {jnp.allclose(result_jit, expected)}")
    print()

    # --- Example 4: Multiple inputs ---
    print("Example 4: Kernel with scalar parameter")
    print("-" * 40)

    scale = jnp.array([3.0], dtype=jnp.float32)

    scale_mul_jax = numba_cuda_kernel(
        scale_mul_kernel,
        outs=jax.ShapeDtypeStruct((n,), jnp.float32),
        launch_dims=n,
    )

    result_scaled = scale_mul_jax(a, b, scale)
    expected_scaled = a * b * 3.0

    print(f"  a[:5] * b[:5] * 3.0 = {result_scaled[:5]}")
    print(f"  Expected: {expected_scaled[:5]}")
    print(f"  All close: {jnp.allclose(result_scaled, expected_scaled)}")
    print()

    # --- Example 5: Using shared memory ---
    print("Example 5: Kernel with shared memory (partial sum of squares)")
    print("-" * 40)

    x = jnp.arange(1024, dtype=jnp.float32) / 1024.0
    num_blocks = 4

    squared_sum_jax = numba_cuda_kernel(
        squared_sum_kernel,
        outs=jax.ShapeDtypeStruct((num_blocks,), jnp.float32),
        grid=num_blocks,
        block=256,
        shared_mem=256 * 4,  # 256 floats * 4 bytes
    )

    partial_sums = squared_sum_jax(x)
    total = jnp.sum(partial_sums)
    expected_total = jnp.sum(x ** 2)

    print(f"  Partial sums per block: {partial_sums}")
    print(f"  Total sum of squares: {total}")
    print(f"  Expected: {expected_total}")
    print(f"  Close: {jnp.allclose(total, expected_total, rtol=1e-5)}")
    print()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
