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

# -*- coding: utf-8 -*-
"""
ELL Sparse Matrix-Vector Multiplication - JAX Benchmark with TVM FFI

This module registers 5 optimized CUDA kernels via TVM FFI and benchmarks
them against each other and a pure JAX reference implementation.

Requirements:
    pip install jax jaxlib jax-tvm-ffi tvm-ffi

Usage:
    python ell_mv_benchmark_jax.py
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import math
import time
from functools import partial
from typing import Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

# Try to import TVM FFI - will fail gracefully if not available
try:
    import jax_tvm_ffi
    import tvm_ffi.cpp

    HAS_TVM_FFI = True
except ImportError:
    HAS_TVM_FFI = False
    print("Warning: jax_tvm_ffi not available. Only JAX reference will be benchmarked.")

# =============================================================================
# CUDA Kernel Source Code
# =============================================================================

CUDA_SOURCE = r"""
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

//=============================================================================
// Kernel 1: Basic - One block per pre-synaptic neuron
//=============================================================================
__global__ void ell_mv_basic_kernel(
    const bool* __restrict__ spikes,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    float* __restrict__ output,
    int n_pre,
    int n_conn
) {
    const int pre_idx = blockIdx.x;
    if (pre_idx >= n_pre) return;
    if (!spikes[pre_idx]) return;

    const int base_offset = pre_idx * n_conn;
    for (int j = threadIdx.x; j < n_conn; j += blockDim.x) {
        atomicAdd(&output[indices[base_offset + j]], weights[base_offset + j]);
    }
}

//=============================================================================
// Kernel 2: Shared Memory - Prefetch to shared memory
//=============================================================================
__global__ void ell_mv_shared_kernel(
    const bool* __restrict__ spikes,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    float* __restrict__ output,
    int n_pre,
    int n_conn
) {
    extern __shared__ char shared_mem[];
    int32_t* s_indices = reinterpret_cast<int32_t*>(shared_mem);
    float* s_weights = reinterpret_cast<float*>(shared_mem + blockDim.x * sizeof(int32_t));

    const int pre_idx = blockIdx.x;
    if (pre_idx >= n_pre) return;
    if (!spikes[pre_idx]) return;

    const int base_offset = pre_idx * n_conn;
    const int tile_size = blockDim.x;
    const int num_tiles = (n_conn + tile_size - 1) / tile_size;

    for (int tile = 0; tile < num_tiles; tile++) {
        const int j = tile * tile_size + threadIdx.x;
        if (j < n_conn) {
            s_indices[threadIdx.x] = indices[base_offset + j];
            s_weights[threadIdx.x] = weights[base_offset + j];
        }
        __syncthreads();

        if (j < n_conn) {
            atomicAdd(&output[s_indices[threadIdx.x]], s_weights[threadIdx.x]);
        }
        __syncthreads();
    }
}

//=============================================================================
// Kernel 3: Grid-Stride - Flattened iteration
//=============================================================================
__global__ void ell_mv_gridstride_kernel(
    const bool* __restrict__ spikes,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    float* __restrict__ output,
    int n_pre,
    int n_conn
) {
    const int total_elements = n_pre * n_conn;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total_elements; idx += stride) {
        const int pre_idx = idx / n_conn;
        if (spikes[pre_idx]) {
            atomicAdd(&output[indices[idx]], weights[idx]);
        }
    }
}

//=============================================================================
// Kernel 4: Warp-Optimized - One warp per neuron
//=============================================================================
__global__ void ell_mv_warp_kernel(
    const bool* __restrict__ spikes,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    float* __restrict__ output,
    int n_pre,
    int n_conn
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int pre_idx = warp_id; pre_idx < n_pre; pre_idx += num_warps) {
        if (!spikes[pre_idx]) continue;

        const int base_offset = pre_idx * n_conn;
        for (int j = lane_id; j < n_conn; j += 32) {
            atomicAdd(&output[indices[base_offset + j]], weights[base_offset + j]);
        }
    }
}

//=============================================================================
// Kernel 5: Vectorized - Uses float4/int4 for coalesced loads
//=============================================================================
__global__ void ell_mv_vectorized_kernel(
    const bool* __restrict__ spikes,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    float* __restrict__ output,
    int n_pre,
    int n_conn
) {
    const int pre_idx = blockIdx.x;
    if (pre_idx >= n_pre) return;
    if (!spikes[pre_idx]) return;

    const int base_offset = pre_idx * n_conn;
    const int n_vec = n_conn / 4;

    const int4* idx_vec = reinterpret_cast<const int4*>(indices + base_offset);
    const float4* wt_vec = reinterpret_cast<const float4*>(weights + base_offset);

    for (int j = threadIdx.x; j < n_vec; j += blockDim.x) {
        int4 idx4 = idx_vec[j];
        float4 w4 = wt_vec[j];

        atomicAdd(&output[idx4.x], w4.x);
        atomicAdd(&output[idx4.y], w4.y);
        atomicAdd(&output[idx4.z], w4.z);
        atomicAdd(&output[idx4.w], w4.w);
    }

    // Handle remainder
    const int remainder_start = n_vec * 4;
    for (int j = remainder_start + threadIdx.x; j < n_conn; j += blockDim.x) {
        atomicAdd(&output[indices[base_offset + j]], weights[base_offset + j]);
    }
}

//=============================================================================
// TVM FFI Entry Points
//=============================================================================

void ell_mv_basic(
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    int n_pre = spikes.size(0);
    int n_conn = indices.size(1);
    int n_post = output.size(0);

    cudaMemsetAsync(output.data_ptr(), 0, n_post * sizeof(float), cuda_stream);

    int threads = min(256, n_conn);
    ell_mv_basic_kernel<<<n_pre, threads, 0, cuda_stream>>>(
        static_cast<const bool*>(spikes.data_ptr()),
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        n_pre, n_conn
    );

    cudaError_t err = cudaGetLastError();
    TVM_FFI_ICHECK(err == cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);
}

void ell_mv_shared(
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    int n_pre = spikes.size(0);
    int n_conn = indices.size(1);
    int n_post = output.size(0);

    cudaMemsetAsync(output.data_ptr(), 0, n_post * sizeof(float), cuda_stream);

    int threads = 256;
    size_t shared_size = threads * (sizeof(int32_t) + sizeof(float));
    ell_mv_shared_kernel<<<n_pre, threads, shared_size, cuda_stream>>>(
        static_cast<const bool*>(spikes.data_ptr()),
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        n_pre, n_conn
    );

    cudaError_t err = cudaGetLastError();
    TVM_FFI_ICHECK(err == cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);
}

void ell_mv_gridstride(
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    int n_pre = spikes.size(0);
    int n_conn = indices.size(1);
    int n_post = output.size(0);

    cudaMemsetAsync(output.data_ptr(), 0, n_post * sizeof(float), cuda_stream);

    int threads = 256;
    int blocks = min(1024, (n_pre * n_conn + threads - 1) / threads);
    ell_mv_gridstride_kernel<<<blocks, threads, 0, cuda_stream>>>(
        static_cast<const bool*>(spikes.data_ptr()),
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        n_pre, n_conn
    );

    cudaError_t err = cudaGetLastError();
    TVM_FFI_ICHECK(err == cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);
}

void ell_mv_warp(
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    int n_pre = spikes.size(0);
    int n_conn = indices.size(1);
    int n_post = output.size(0);

    cudaMemsetAsync(output.data_ptr(), 0, n_post * sizeof(float), cuda_stream);

    int threads = 256;
    int blocks = min(1024, (n_pre + 7) / 8);
    ell_mv_warp_kernel<<<blocks, threads, 0, cuda_stream>>>(
        static_cast<const bool*>(spikes.data_ptr()),
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        n_pre, n_conn
    );

    cudaError_t err = cudaGetLastError();
    TVM_FFI_ICHECK(err == cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);
}

void ell_mv_vectorized(
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    int n_pre = spikes.size(0);
    int n_conn = indices.size(1);
    int n_post = output.size(0);

    cudaMemsetAsync(output.data_ptr(), 0, n_post * sizeof(float), cuda_stream);

    int threads = 256;
    ell_mv_vectorized_kernel<<<n_pre, threads, 0, cuda_stream>>>(
        static_cast<const bool*>(spikes.data_ptr()),
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        n_pre, n_conn
    );

    cudaError_t err = cudaGetLastError();
    TVM_FFI_ICHECK(err == cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);
}
"""

# =============================================================================
# Kernel Registration
# =============================================================================

_cuda_module = None
_kernels_registered = False

KERNEL_NAMES = ["basic", "shared", "gridstride", "warp", "vectorized"]


def _compile_and_register_kernels():
    """Compile CUDA kernels and register with JAX FFI."""
    global _cuda_module, _kernels_registered

    if _kernels_registered:
        return True

    if not HAS_TVM_FFI:
        return False

    try:
        # Compile CUDA module
        _cuda_module = tvm_ffi.cpp.load_inline(
            name="ell_mv_kernels",
            cuda_sources=CUDA_SOURCE,
            functions=[
                "ell_mv_basic",
                "ell_mv_shared",
                "ell_mv_gridstride",
                "ell_mv_warp",
                "ell_mv_vectorized",
            ],
        )

        # Register each kernel with JAX FFI
        for name in KERNEL_NAMES:
            jax_tvm_ffi.register_ffi_target(
                f"ell_mv.{name}",
                getattr(_cuda_module, f"ell_mv_{name}"),
                ["args", "rets", "ctx.stream"],
                platform="gpu",
            )

        _kernels_registered = True
        return True

    except Exception as e:
        print(f"Failed to compile/register CUDA kernels: {e}")
        return False


# =============================================================================
# JAX Wrapper Functions
# =============================================================================

def ell_mv_cuda(
    spikes: Array,
    indices: Array,
    weights: Array,
    *,
    n_post: int,
    kernel: str = "basic"
) -> Array:
    """
    ELL sparse matrix-vector multiplication using CUDA kernel.

    Args:
        spikes: Boolean array [n_pre] indicating active neurons
        indices: Int32 array [n_pre, n_conn] with target indices
        weights: Float32 array [n_pre, n_conn] with connection weights
        n_post: Output size (number of post-synaptic neurons)
        kernel: Kernel variant ("basic", "shared", "gridstride", "warp", "vectorized")

    Returns:
        Float32 array [n_post] with accumulated weights
    """
    if kernel not in KERNEL_NAMES:
        raise ValueError(f"Unknown kernel: {kernel}. Choose from {KERNEL_NAMES}")

    return jax.ffi.ffi_call(
        f"ell_mv.{kernel}",
        jax.ShapeDtypeStruct((n_post,), weights.dtype),
    )(spikes, indices, weights)


def make_ell_mv_jit(n_post: int, kernel: str = "basic") -> Callable:
    """Create a JIT-compiled ELL mv function."""

    @jax.jit
    def _fn(spikes: Array, indices: Array, weights: Array) -> Array:
        return ell_mv_cuda(spikes, indices, weights, n_post=n_post, kernel=kernel)

    return _fn


# =============================================================================
# JAX Reference Implementations
# =============================================================================

def ell_mv_jax_scan(spikes, indices, weights, *, n_post):
    """JAX reference using scan (sequential, for correctness checking)."""

    def accumulate(carry, inputs):
        spike, inds, wts = inputs
        return jax.lax.cond(
            spike,
            lambda c: c.at[inds].add(wts),
            lambda c: c,
            carry
        ), None

    output = jnp.zeros(n_post, dtype=weights.dtype)
    output, _ = jax.lax.scan(accumulate, output, (spikes, indices, weights))
    return output


def ell_mv_jax_segment_sum(spikes, indices, weights, *, n_post):
    """JAX reference using segment_sum (vectorized, faster)."""
    # Mask inactive neurons
    active_mask = spikes[:, None]
    masked_weights = jnp.where(active_mask, weights, 0.0)

    # Flatten and accumulate
    flat_indices = indices.ravel()
    flat_weights = masked_weights.ravel()

    return jax.ops.segment_sum(flat_weights, flat_indices, num_segments=n_post)


def ell_mv_jax_scatter(spikes, indices, weights, *, n_post):
    """JAX reference using scatter_add."""
    # Create output and mask
    output = jnp.zeros(n_post, dtype=weights.dtype)
    active_mask = spikes[:, None]
    masked_weights = jnp.where(active_mask, weights, 0.0)

    # Use at[].add for scatter
    return output.at[indices.ravel()].add(masked_weights.ravel())


# =============================================================================
# Benchmarking Infrastructure
# =============================================================================

def benchmark_function(
    fn: Callable,
    args: Tuple,
    n_warmup: int = 10,
    n_runs: int = 100,
    sync: bool = True
) -> Dict:
    """Benchmark a function and return timing statistics."""
    # Warmup
    for _ in range(n_warmup):
        result = fn(*args)
    if sync:
        jax.block_until_ready(result)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = fn(*args)
        if sync:
            jax.block_until_ready(result)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times = np.array(times)
    return {
        "mean_us": times.mean() * 1e6,
        "std_us": times.std() * 1e6,
        "min_us": times.min() * 1e6,
        "max_us": times.max() * 1e6,
        "median_us": np.median(times) * 1e6,
    }


def check_correctness(
    result: Array,
    reference: Array,
    atol: float = 1e-5,
    rtol: float = 1e-5
) -> Tuple[bool, float]:
    """Check if result matches reference within tolerance."""
    result_np = np.asarray(result)
    reference_np = np.asarray(reference)
    max_diff = np.abs(result_np - reference_np).max()
    correct = np.allclose(result_np, reference_np, atol=atol, rtol=rtol)
    return correct, max_diff


# =============================================================================
# Main Benchmark
# =============================================================================

def run_benchmark():
    """Run comprehensive benchmark of all kernels."""
    print("=" * 75)
    print("ELL Sparse Matrix-Vector Multiplication - JAX Benchmark")
    print("=" * 75)

    # Check for GPU
    try:
        gpu_devices = jax.devices("gpu")
        if not gpu_devices:
            print("\nNo GPU available. Running JAX-only benchmark on CPU.")
            gpu_available = False
        else:
            print(f"\nUsing GPU: {gpu_devices[0]}")
            gpu_available = True
    except RuntimeError:
        print("\nNo GPU backend. Running JAX-only benchmark on CPU.")
        gpu_available = False

    # Try to register CUDA kernels
    cuda_available = False
    if gpu_available:
        cuda_available = _compile_and_register_kernels()
        if cuda_available:
            print("CUDA kernels compiled and registered successfully.")
        else:
            print("Failed to compile CUDA kernels. Running JAX-only benchmark.")

    # Test configurations
    configs = [
        # (n_pre, n_post, n_conn, spike_rate, description)
        (100, 200, 20, 0.5, "Small (100×20, 50% active)"),
        (100, 200, 20, 0.1, "Small (100×20, 10% active)"),
        (1000, 2000, 200, 0.5, "Medium (1K×200, 50% active)"),
        (1000, 2000, 200, 0.1, "Medium (1K×200, 10% active)"),
        (10000, 20000, 200, 0.5, "Large (10K×200, 50% active)"),
        (10000, 20000, 200, 0.1, "Large (10K×200, 10% active)"),
        (10000, 20000, 200, 0.01, "Large (10K×200, 1% active)"),
        (1000, 10000, 1000, 0.5, "Wide (1K×1K, 50% active)"),
        (1000, 10000, 1000, 0.1, "Wide (1K×1K, 10% active)"),
    ]

    n_warmup = 20
    n_runs = 100

    # Store results for summary
    all_results = []

    for n_pre, n_post, n_conn, spike_rate, desc in configs:
        print("\n" + "-" * 75)
        print(f"Config: {desc}")
        print(f"  n_pre={n_pre}, n_post={n_post}, n_conn={n_conn}")
        print(f"  Total connections: {n_pre * n_conn:,}")
        print("-" * 75)

        # Generate test data
        key = jr.PRNGKey(42)
        keys = jr.split(key, 3)

        spikes = jr.uniform(keys[0], (n_pre,)) < spike_rate
        indices = jr.randint(keys[1], (n_pre, n_conn), 0, n_post, dtype=jnp.int32)
        weights = jr.normal(keys[2], (n_pre, n_conn), dtype=jnp.float32)

        n_active = int(spikes.sum())
        print(f"  Active neurons: {n_active} ({100 * n_active / n_pre:.1f}%)")

        # Move to GPU if available
        device = gpu_devices[0] if gpu_available else jax.devices("cpu")[0]
        spikes = jax.device_put(spikes, device)
        indices = jax.device_put(indices, device)
        weights = jax.device_put(weights, device)

        # Compute reference result
        ref_fn = jax.jit(lambda s, i, w: ell_mv_jax_segment_sum(s, i, w, n_post=n_post))
        reference = ref_fn(spikes, indices, weights)
        jax.block_until_ready(reference)

        # Prepare results table
        results = {}

        # Benchmark JAX implementations
        jax_impls = [
            ("JAX-segment", lambda s, i, w: ell_mv_jax_segment_sum(s, i, w, n_post=n_post)),
            ("JAX-scatter", lambda s, i, w: ell_mv_jax_scatter(s, i, w, n_post=n_post)),
        ]

        for name, impl in jax_impls:
            fn = jax.jit(impl)
            result = fn(spikes, indices, weights)
            jax.block_until_ready(result)

            correct, max_diff = check_correctness(result, reference)
            timing = benchmark_function(fn, (spikes, indices, weights), n_warmup, n_runs)

            results[name] = {
                "time_us": timing["mean_us"],
                "std_us": timing["std_us"],
                "correct": correct,
                "max_diff": max_diff,
            }

        # Benchmark CUDA kernels if available
        if cuda_available:
            for kernel_name in KERNEL_NAMES:
                # Skip vectorized if n_conn not divisible by 4
                if kernel_name == "vectorized" and n_conn % 4 != 0:
                    results[f"CUDA-{kernel_name}"] = {
                        "time_us": float('nan'),
                        "std_us": float('nan'),
                        "correct": None,
                        "max_diff": float('nan'),
                        "skipped": True,
                    }
                    continue

                try:
                    fn = make_ell_mv_jit(n_post, kernel_name)
                    result = fn(spikes, indices, weights)
                    jax.block_until_ready(result)

                    correct, max_diff = check_correctness(result, reference)
                    timing = benchmark_function(fn, (spikes, indices, weights), n_warmup, n_runs)

                    results[f"CUDA-{kernel_name}"] = {
                        "time_us": timing["mean_us"],
                        "std_us": timing["std_us"],
                        "correct": correct,
                        "max_diff": max_diff,
                    }
                except Exception as e:
                    results[f"CUDA-{kernel_name}"] = {
                        "time_us": float('nan'),
                        "std_us": float('nan'),
                        "correct": False,
                        "max_diff": float('nan'),
                        "error": str(e),
                    }

        # Print results table
        print(f"\n  {'Kernel':<18} {'Time (µs)':>12} {'Std (µs)':>10} {'Gelem/s':>10} {'Status':<10}")
        print(f"  {'-' * 18} {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10}")

        best_time = float('inf')
        best_kernel = None

        for name, data in results.items():
            if data.get("skipped"):
                print(f"  {name:<18} {'N/A':>12} {'N/A':>10} {'N/A':>10} {'SKIPPED':<10}")
                continue

            if "error" in data:
                print(f"  {name:<18} {'ERROR':>12} {'':>10} {'':>10} {data['error'][:10]:<10}")
                continue

            time_us = data["time_us"]
            std_us = data["std_us"]
            throughput = (n_pre * n_conn) / (time_us * 1000)  # Gelem/s
            status = "OK" if data["correct"] else f"FAIL ({data['max_diff']:.2e})"

            marker = ""
            if time_us < best_time and data["correct"]:
                best_time = time_us
                best_kernel = name

            print(f"  {name:<18} {time_us:>12.2f} {std_us:>10.2f} {throughput:>10.2f} {status:<10}")

        # Mark best
        print(f"\n  Best: {best_kernel} ({best_time:.2f} µs)")

        all_results.append({
            "config": desc,
            "results": results,
            "best_kernel": best_kernel,
            "best_time": best_time,
        })

    # Summary table
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(f"\n{'Configuration':<35} {'Best Kernel':<20} {'Time (µs)':>12}")
    print(f"{'-' * 35} {'-' * 20} {'-' * 12}")

    for r in all_results:
        print(f"{r['config']:<35} {r['best_kernel']:<20} {r['best_time']:>12.2f}")

    print("\n" + "=" * 75)
    print("Benchmark Complete")
    print("=" * 75)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    run_benchmark()