# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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


import time

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp


@wp.kernel
def find_nonzeros_kernel(
    array: wp.array(dtype=float),
    indices: wp.array(dtype=int),
    counter: wp.array(dtype=int)
):
    # Find indices of nonzero elements
    tid = wp.tid()
    if array[tid] != 0.0:
        idx = wp.atomic_add(counter, 0, 1)
        indices[idx] = tid


def warp_nonzeros(array):
    """Find nonzero indices using NVIDIA Warp"""
    # Initialize Warp device array from numpy array
    device_array = wp.array(array, dtype=float)

    # Allocate space for indices
    indices = wp.zeros(array.shape, dtype=int)
    counter = wp.zeros((1,), dtype=int)

    # Find nonzero indices
    wp.launch(find_nonzeros_kernel, dim=array.shape, inputs=[device_array, indices, counter])
    return indices


@jax.jit
def jax_nonzeros(array):
    """Find nonzero indices using JAX"""
    return jnp.nonzero(array, size=array.shape, fill_value=-1)[0].block_until_ready()


def benchmark_comparison(size=1000000, sparsity=0.9):
    """Compare performance between Warp and JAX implementations"""
    # Create a sparse array (mostly zeros)
    np.random.seed(42)
    arr = np.random.random(size)
    arr[np.random.random(size) < sparsity] = 0.0  # Set ~90% elements to zero

    # Convert to device arrays
    jax_arr = jnp.array(arr)

    # Warmup
    _ = warp_nonzeros(arr)
    _ = jax_nonzeros(jax_arr)

    # Benchmark Warp
    wp.synchronize()
    warp_start = time.time()
    warp_result = warp_nonzeros(arr)
    wp.synchronize()
    warp_time = time.time() - warp_start

    # Benchmark JAX
    jax_start = time.time()
    jax_result = jax_nonzeros(jax_arr)
    jax_time = time.time() - jax_start

    # Verify correctness
    warp_result = warp_result.numpy()
    np_result = np.nonzero(arr)[0]
    assert np.array_equal(np.sort(warp_result), np.sort(np_result)), "Warp result incorrect"
    assert np.array_equal(np.sort(jax_result), np.sort(np_result)), "JAX result incorrect"

    print(f"Array size: {size}, non-zero elements: {len(np_result)}")
    print(f"Warp time: {warp_time:.6f}s")
    print(f"JAX time: {jax_time:.6f}s")
    print(f"Speedup (Warp/JAX): {jax_time / warp_time:.2f}x" if warp_time < jax_time else
          f"Speedup (JAX/Warp): {warp_time / jax_time:.2f}x")


if __name__ == "__main__":
    # Initialize Warp
    wp.init()

    # Run benchmarks with different array sizes
    print("Small array benchmark:")
    benchmark_comparison(size=10000, sparsity=0.9)

    print("\nMedium array benchmark:")
    benchmark_comparison(size=1000000, sparsity=0.9)

    print("\nLarge array benchmark:")
    benchmark_comparison(size=10000000, sparsity=0.95)
