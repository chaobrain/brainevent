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
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

"""
ELL Sparse MV - 6 Kernel Comparison with TVM FFI

Includes the two-pass compact kernel optimized for very sparse firing (<5% active).

Requirements:
    pip install jax jaxlib jax-tvm-ffi tvm-ffi
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import time

# =============================================================================
# CUDA Source with all 6 kernels
# =============================================================================

CUDA_SOURCE = r"""
#include <cuda_runtime.h>
#include <cstdint>

//=============================================================================
// Kernel 1: Basic - One block per pre-synaptic neuron
//=============================================================================
__global__ void basic_kernel(
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
__global__ void shared_kernel(
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
__global__ void gridstride_kernel(
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
__global__ void warp_kernel(
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
__global__ void vectorized_kernel(
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
// Kernel 6: Two-Pass Compact - Best for very sparse firing (<5% active)
//
// Pass 1: Compact active neuron indices into a dense array
// Pass 2: Process only active neurons with perfect load balancing
//=============================================================================

// Sub-kernel: Compact active indices using warp-level primitives
__global__ void compact_active_kernel(
    const bool* __restrict__ spikes,
    int* __restrict__ active_indices,
    int* __restrict__ num_active,
    int n_pre
) {
    __shared__ int warp_counts[32];  // One per warp in block
    __shared__ int block_offset;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int num_warps_in_block = blockDim.x / 32;

    // Check if this thread's neuron is active
    bool is_active = (tid < n_pre) && spikes[tid];

    // Warp-level ballot to find active threads
    unsigned int active_mask = __ballot_sync(0xFFFFFFFF, is_active);
    int warp_active_count = __popc(active_mask);

    // Compute prefix within warp
    unsigned int lower_mask = (1u << lane_id) - 1;
    int lane_offset = __popc(active_mask & lower_mask);

    // First thread in each warp stores warp count
    if (lane_id == 0) {
        warp_counts[warp_id] = warp_active_count;
    }
    __syncthreads();

    // First warp computes block prefix and global offset
    if (warp_id == 0) {
        int val = (lane_id < num_warps_in_block) ? warp_counts[lane_id] : 0;

        // Inclusive scan within first warp
        #pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
            int n = __shfl_up_sync(0xFFFFFFFF, val, offset);
            if (lane_id >= offset) val += n;
        }

        // Store prefix sums back (exclusive)
        if (lane_id < num_warps_in_block) {
            warp_counts[lane_id] = val - ((lane_id < num_warps_in_block) ? 
                ((lane_id > 0) ? (val - warp_counts[lane_id] + __shfl_up_sync(0xFFFFFFFF, val, 1)) : 0) : 0);
        }

        // Last active thread in first warp gets global offset
        int block_total = val;
        if (lane_id == num_warps_in_block - 1 || lane_id == 31) {
            // Actually need the last warp's count
        }
        if (lane_id == 0) {
            // Total for this block
            int total = 0;
            for (int i = 0; i < num_warps_in_block; i++) total += warp_counts[i];
            // Actually we stored prefix, so get total differently
        }
    }
    __syncthreads();

    // Simpler approach: use atomics for global offset
    __shared__ int local_count;
    if (threadIdx.x == 0) local_count = 0;
    __syncthreads();

    int local_offset = -1;
    if (is_active) {
        local_offset = atomicAdd(&local_count, 1);
    }
    __syncthreads();

    // Get global offset for this block
    if (threadIdx.x == 0) {
        block_offset = atomicAdd(num_active, local_count);
    }
    __syncthreads();

    // Write active index
    if (is_active) {
        active_indices[block_offset + local_offset] = tid;
    }
}

// Simpler compact kernel using global atomics (works well for sparse data)
__global__ void compact_simple_kernel(
    const bool* __restrict__ spikes,
    int* __restrict__ active_indices,
    int* __restrict__ counter,
    int n_pre
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n_pre; i += stride) {
        if (spikes[i]) {
            int pos = atomicAdd(counter, 1);
            active_indices[pos] = i;
        }
    }
}

// Process only active neurons (pass 2)
__global__ void process_compact_kernel(
    const int* __restrict__ active_indices,
    int num_active,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    float* __restrict__ output,
    int n_conn
) {
    // Each block handles one active neuron
    const int active_idx = blockIdx.x;
    if (active_idx >= num_active) return;

    const int pre_idx = active_indices[active_idx];
    const int base_offset = pre_idx * n_conn;

    for (int j = threadIdx.x; j < n_conn; j += blockDim.x) {
        atomicAdd(&output[indices[base_offset + j]], weights[base_offset + j]);
    }
}

// Alternative: Grid-stride over active neurons for better load balancing
__global__ void process_compact_gridstride_kernel(
    const int* __restrict__ active_indices,
    int num_active,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    float* __restrict__ output,
    int n_conn
) {
    const int total_work = num_active * n_conn;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total_work; idx += stride) {
        const int active_idx = idx / n_conn;
        const int conn_idx = idx % n_conn;
        const int pre_idx = active_indices[active_idx];
        const int offset = pre_idx * n_conn + conn_idx;

        atomicAdd(&output[indices[offset]], weights[offset]);
    }
}

//=============================================================================
// TVM FFI Entry Points
//=============================================================================

void ell_basic(
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre = spikes.size(0);
    int n_conn = indices.size(1);
    int n_post = output.size(0);

    cudaMemsetAsync(output.data_ptr(), 0, n_post * sizeof(float), s);

    int threads = min(256, n_conn);
    basic_kernel<<<n_pre, threads, 0, s>>>(
        static_cast<const bool*>(spikes.data_ptr()),
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        n_pre, n_conn
    );
}

void ell_shared(
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre = spikes.size(0);
    int n_conn = indices.size(1);
    int n_post = output.size(0);

    cudaMemsetAsync(output.data_ptr(), 0, n_post * sizeof(float), s);

    int threads = 256;
    size_t shared_size = threads * (sizeof(int32_t) + sizeof(float));
    shared_kernel<<<n_pre, threads, shared_size, s>>>(
        static_cast<const bool*>(spikes.data_ptr()),
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        n_pre, n_conn
    );
}

void ell_gridstride(
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre = spikes.size(0);
    int n_conn = indices.size(1);
    int n_post = output.size(0);

    cudaMemsetAsync(output.data_ptr(), 0, n_post * sizeof(float), s);

    int threads = 256;
    int blocks = min(1024, (n_pre * n_conn + threads - 1) / threads);
    gridstride_kernel<<<blocks, threads, 0, s>>>(
        static_cast<const bool*>(spikes.data_ptr()),
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        n_pre, n_conn
    );
}

void ell_warp(
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre = spikes.size(0);
    int n_conn = indices.size(1);
    int n_post = output.size(0);

    cudaMemsetAsync(output.data_ptr(), 0, n_post * sizeof(float), s);

    int threads = 256;
    int blocks = min(1024, (n_pre + 7) / 8);
    warp_kernel<<<blocks, threads, 0, s>>>(
        static_cast<const bool*>(spikes.data_ptr()),
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        n_pre, n_conn
    );
}

void ell_vectorized(
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre = spikes.size(0);
    int n_conn = indices.size(1);
    int n_post = output.size(0);

    cudaMemsetAsync(output.data_ptr(), 0, n_post * sizeof(float), s);

    int threads = 256;
    vectorized_kernel<<<n_pre, threads, 0, s>>>(
        static_cast<const bool*>(spikes.data_ptr()),
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        n_pre, n_conn
    );
}

// Compact kernel needs workspace for active indices and counter
// We allocate these internally using cudaMalloc (could be optimized with memory pool)
void ell_compact(
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre = spikes.size(0);
    int n_conn = indices.size(1);
    int n_post = output.size(0);

    // Allocate workspace
    int* d_active_indices;
    int* d_counter;
    cudaMalloc(&d_active_indices, n_pre * sizeof(int));
    cudaMalloc(&d_counter, sizeof(int));
    cudaMemsetAsync(d_counter, 0, sizeof(int), s);
    cudaMemsetAsync(output.data_ptr(), 0, n_post * sizeof(float), s);

    // Pass 1: Compact active indices
    int threads = 256;
    int blocks = (n_pre + threads - 1) / threads;
    compact_simple_kernel<<<blocks, threads, 0, s>>>(
        static_cast<const bool*>(spikes.data_ptr()),
        d_active_indices,
        d_counter,
        n_pre
    );

    // Get number of active neurons (sync required)
    int h_num_active;
    cudaMemcpyAsync(&h_num_active, d_counter, sizeof(int), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    // Pass 2: Process only active neurons
    if (h_num_active > 0) {
        // Choose between block-per-neuron or grid-stride based on workload
        if (h_num_active * n_conn < 100000) {
            // Small workload: one block per active neuron
            int proc_threads = min(256, n_conn);
            process_compact_kernel<<<h_num_active, proc_threads, 0, s>>>(
                d_active_indices,
                h_num_active,
                static_cast<const int32_t*>(indices.data_ptr()),
                static_cast<const float*>(weights.data_ptr()),
                static_cast<float*>(output.data_ptr()),
                n_conn
            );
        } else {
            // Large workload: grid-stride for load balancing
            int total_work = h_num_active * n_conn;
            int proc_blocks = min(1024, (total_work + 255) / 256);
            process_compact_gridstride_kernel<<<proc_blocks, 256, 0, s>>>(
                d_active_indices,
                h_num_active,
                static_cast<const int32_t*>(indices.data_ptr()),
                static_cast<const float*>(weights.data_ptr()),
                static_cast<float*>(output.data_ptr()),
                n_conn
            );
        }
    }

    // Free workspace
    cudaFree(d_active_indices);
    cudaFree(d_counter);
}

// Version with pre-allocated workspace (avoids malloc in hot path)
void ell_compact_preallocated(
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView workspace,  // [n_pre + 1] int32 for active_indices + counter
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre = spikes.size(0);
    int n_conn = indices.size(1);
    int n_post = output.size(0);

    int* d_active_indices = static_cast<int*>(workspace.data_ptr());
    int* d_counter = d_active_indices + n_pre;

    cudaMemsetAsync(d_counter, 0, sizeof(int), s);
    cudaMemsetAsync(output.data_ptr(), 0, n_post * sizeof(float), s);

    // Pass 1: Compact
    int threads = 256;
    int blocks = (n_pre + threads - 1) / threads;
    compact_simple_kernel<<<blocks, threads, 0, s>>>(
        static_cast<const bool*>(spikes.data_ptr()),
        d_active_indices,
        d_counter,
        n_pre
    );

    // Get count
    int h_num_active;
    cudaMemcpyAsync(&h_num_active, d_counter, sizeof(int), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    // Pass 2: Process
    if (h_num_active > 0) {
        int proc_threads = min(256, n_conn);
        process_compact_kernel<<<h_num_active, proc_threads, 0, s>>>(
            d_active_indices,
            h_num_active,
            static_cast<const int32_t*>(indices.data_ptr()),
            static_cast<const float*>(weights.data_ptr()),
            static_cast<float*>(output.data_ptr()),
            n_conn
        );
    }
}
"""


# =============================================================================
# Setup and Registration
# =============================================================================

def setup_kernels():
    """Compile and register CUDA kernels."""
    import jax_tvm_ffi
    import tvm_ffi.cpp

    mod = tvm_ffi.cpp.load_inline(
        name="ell_kernels_v2",
        cuda_sources=CUDA_SOURCE,
        functions=[
            "ell_basic",
            "ell_shared",
            "ell_gridstride",
            "ell_warp",
            "ell_vectorized",
            "ell_compact",
            "ell_compact_preallocated",
        ],
    )

    # Register standard kernels
    for name in ["basic", "shared", "gridstride", "warp", "vectorized", "compact"]:
        jax_tvm_ffi.register_ffi_target(
            f"ell.{name}",
            getattr(mod, f"ell_{name}"),
            ["args", "rets", "ctx.stream"],
            platform="gpu"
        )

    # Register preallocated version separately (different signature)
    jax_tvm_ffi.register_ffi_target(
        "ell.compact_prealloc",
        mod.ell_compact_preallocated,
        ["args", "rets", "ctx.stream"],
        platform="gpu"
    )

    return True


def make_kernel_fn(kernel_name: str, n_post: int):
    """Create JIT-compiled kernel function."""

    @jax.jit
    def fn(spikes, indices, weights):
        return jax.ffi.ffi_call(
            f"ell.{kernel_name}",
            jax.ShapeDtypeStruct((n_post,), weights.dtype),
        )(spikes, indices, weights)

    return fn


def make_compact_prealloc_fn(n_pre: int, n_post: int):
    """Create JIT-compiled compact kernel with preallocated workspace."""

    @jax.jit
    def fn(spikes, indices, weights, workspace):
        return jax.ffi.ffi_call(
            "ell.compact_prealloc",
            jax.ShapeDtypeStruct((n_post,), weights.dtype),
        )(spikes, indices, weights, workspace)

    return fn


# =============================================================================
# JAX Reference
# =============================================================================

@jax.jit(static_argnums=3)
def ell_jax_ref(spikes, indices, weights, n_post):
    """JAX reference using segment_sum."""
    mask = spikes[:, None]
    masked_wt = jnp.where(mask, weights, 0.0)
    return jax.ops.segment_sum(masked_wt.ravel(), indices.ravel(), num_segments=n_post)


# =============================================================================
# Benchmark
# =============================================================================

def benchmark(fn, args, warmup=20, runs=100):
    """Time a function."""
    for _ in range(warmup):
        r = fn(*args)
    jax.block_until_ready(r)

    t0 = time.perf_counter()
    for _ in range(runs):
        r = fn(*args)
    jax.block_until_ready(r)
    return (time.perf_counter() - t0) / runs * 1e6  # µs


def main():
    print("=" * 80)
    print("ELL Sparse MV - 6 Kernel Comparison (including Two-Pass Compact)")
    print("=" * 80)

    # Check GPU
    try:
        gpu = jax.devices("gpu")[0]
        print(f"GPU: {gpu}")
    except:
        print("ERROR: No GPU available")
        return

    # Setup kernels
    print("Compiling CUDA kernels...")
    try:
        setup_kernels()
        print("OK\n")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test configs: (n_pre, n_post, n_conn, spike_rate, desc)
    configs = [
        # Standard configs
        (100, 200, 20, 0.5, "Small-50%"),
        (1000, 2000, 200, 0.5, "Medium-50%"),
        (1000, 2000, 200, 0.1, "Medium-10%"),
        (10000, 20000, 200, 0.5, "Large-50%"),
        (10000, 20000, 200, 0.1, "Large-10%"),

        # Very sparse - where compact kernel shines
        (10000, 20000, 200, 0.05, "Large-5%"),
        (10000, 20000, 200, 0.01, "Large-1%"),
        (50000, 50000, 100, 0.01, "Huge-1%"),
        (100000, 100000, 50, 0.005, "Massive-0.5%"),

        # Wide connectivity
        (1000, 10000, 1000, 0.5, "Wide-50%"),
        (1000, 10000, 1000, 0.05, "Wide-5%"),
    ]

    kernels = ["basic", "shared", "gridstride", "warp", "vectorized", "compact"]

    # Header
    print(f"{'Config':<14}", end="")
    for k in kernels:
        print(f"{k:>15}", end="")
    print(f"{'JAX-ref':>15} {'Best':>15} {'Speedup':>8}")
    print("-" * (14 + 10 * (len(kernels) + 1) + 10 + 8))

    for n_pre, n_post, n_conn, rate, desc in configs:
        # Generate data
        key = jr.PRNGKey(42)
        k1, k2, k3 = jr.split(key, 3)

        spikes = jax.device_put(jr.uniform(k1, (n_pre,)) < rate, gpu)
        indices = jax.device_put(jr.randint(k2, (n_pre, n_conn), 0, n_post, dtype=jnp.int32), gpu)
        weights = jax.device_put(jr.normal(k3, (n_pre, n_conn), dtype=jnp.float32), gpu)

        n_active = int(spikes.sum())

        # Reference
        ref = ell_jax_ref(spikes, indices, weights, n_post)
        jax.block_until_ready(ref)

        times = {}
        print(f"{desc:<14}", end="")

        # Benchmark each kernel
        for kname in kernels:
            if kname == "vectorized" and n_conn % 4 != 0:
                print(f"{'N/A':>15}", end="")
                times[kname] = float('inf')
                continue

            try:
                fn = make_kernel_fn(kname, n_post)
                result = fn(spikes, indices, weights)
                jax.block_until_ready(result)

                # Check correctness
                diff = float(jnp.abs(result - ref).max())
                if diff > 1e-4:
                    print(f"{'FAIL':>15}", end="")
                    times[kname] = float('inf')
                else:
                    t = benchmark(fn, (spikes, indices, weights))
                    print(f"{t:>15.1f}", end="")
                    times[kname] = t
            except Exception as e:
                print(f"{'ERR':>15}", end="")
                times[kname] = float('inf')

        # JAX reference
        jax_fn = lambda s, i, w: ell_jax_ref(s, i, w, n_post)
        t_jax = benchmark(jax_fn, (spikes, indices, weights))
        print(f"{t_jax:>15.1f}", end="")
        times["JAX"] = t_jax

        # Best CUDA kernel
        cuda_times = {k: v for k, v in times.items() if k != "JAX"}
        best = min(cuda_times, key=cuda_times.get)
        best_time = cuda_times[best]
        speedup = t_jax / best_time if best_time > 0 else 0

        print(f"{best:>15} {speedup:>7.1f}x")

    # Legend
    print("\n" + "=" * 80)
    print("Kernel Descriptions:")
    print("  basic      - One block per pre-neuron, simple but wastes inactive blocks")
    print("  shared     - Shared memory prefetch, good for large n_conn")
    print("  gridstride - Flat iteration, maximum occupancy but branch divergence")
    print("  warp       - One warp per neuron, good for sparse (10-50%)")
    print("  vectorized - float4 loads, requires n_conn % 4 == 0")
    print("  compact    - Two-pass: compact active → process, best for <5% active")
    print("\nTimes in µs. Lower is better. Speedup is vs JAX reference.")
    print("=" * 80)


if __name__ == "__main__":
    main()