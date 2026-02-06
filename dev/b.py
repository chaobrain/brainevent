"""
Optimized CUDA ELL Sparse Matrix-Vector Multiplication Kernel

This module implements an optimized CUDA kernel for ELL (ELLPACK) format
sparse matrix-vector multiplication, commonly used in neural network
simulations for spike-driven weight accumulation.

Operation: For each pre-synaptic neuron i where spikes[i] == True:
    For each connection j: output[indices[i,j]] += weights[i,j]
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import jax
import jax.numpy as jnp
import jax_tvm_ffi
import numpy as np
import tvm_ffi.cpp
from jax import Array
import math

# Compile the optimized CUDA kernels
_cuda_module = tvm_ffi.cpp.load_inline(
    name="ell_mv_cuda",
    cuda_sources=r"""
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

//=============================================================================
// Kernel 1: Basic kernel - one block per pre-synaptic neuron
// Good for small n_conn, simple and readable
//=============================================================================
__global__ void ell_mv_basic_kernel(
    const bool* __restrict__ spikes,      // [n_pre]
    const int32_t* __restrict__ indices,  // [n_pre, n_conn]
    const float* __restrict__ weights,    // [n_pre, n_conn]
    float* __restrict__ output,           // [n_post]
    int n_pre,
    int n_conn
) {
    int pre_idx = blockIdx.x;
    if (pre_idx >= n_pre) return;

    // Early exit if neuron didn't fire
    if (!spikes[pre_idx]) return;

    // Each thread processes multiple connections
    const int32_t* my_indices = indices + pre_idx * n_conn;
    const float* my_weights = weights + pre_idx * n_conn;

    for (int j = threadIdx.x; j < n_conn; j += blockDim.x) {
        int post_idx = my_indices[j];
        float w = my_weights[j];
        atomicAdd(&output[post_idx], w);
    }
}

//=============================================================================
// Kernel 2: Optimized kernel with shared memory prefetching
// Better for larger n_conn values
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

    int pre_idx = blockIdx.x;
    if (pre_idx >= n_pre) return;

    if (!spikes[pre_idx]) return;

    const int32_t* my_indices = indices + pre_idx * n_conn;
    const float* my_weights = weights + pre_idx * n_conn;

    // Process in tiles
    const int TILE_SIZE = blockDim.x;
    int num_tiles = (n_conn + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < num_tiles; tile++) {
        int base_j = tile * TILE_SIZE;
        int j = base_j + threadIdx.x;

        // Cooperative load into shared memory
        if (j < n_conn) {
            s_indices[threadIdx.x] = my_indices[j];
            s_weights[threadIdx.x] = my_weights[j];
        }
        __syncthreads();

        // Process from shared memory
        if (j < n_conn) {
            atomicAdd(&output[s_indices[threadIdx.x]], s_weights[threadIdx.x]);
        }
        __syncthreads();
    }
}

//=============================================================================
// Kernel 3: Warp-optimized kernel with active neuron compaction
// Best for sparse firing patterns (few active neurons)
//=============================================================================
__device__ __forceinline__ int warp_prefix_sum(int val, int lane_id) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        int n = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (lane_id >= offset) val += n;
    }
    return val;
}

__global__ void ell_mv_warp_kernel(
    const bool* __restrict__ spikes,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    float* __restrict__ output,
    int n_pre,
    int n_conn
) {
    // Each warp cooperatively processes connections for active neurons
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    // Each warp handles multiple pre-synaptic neurons
    for (int pre_idx = warp_id; pre_idx < n_pre; pre_idx += num_warps) {
        if (!spikes[pre_idx]) continue;

        const int32_t* my_indices = indices + pre_idx * n_conn;
        const float* my_weights = weights + pre_idx * n_conn;

        // Warp-stride loop over connections
        for (int j = lane_id; j < n_conn; j += 32) {
            int post_idx = my_indices[j];
            float w = my_weights[j];
            atomicAdd(&output[post_idx], w);
        }
    }
}

//=============================================================================
// Kernel 4: Grid-stride kernel for maximum occupancy
// Best general-purpose kernel
//=============================================================================
__global__ void ell_mv_gridstride_kernel(
    const bool* __restrict__ spikes,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    float* __restrict__ output,
    int n_pre,
    int n_conn
) {
    // Total number of elements to process
    int total_elements = n_pre * n_conn;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total_elements; idx += stride) {
        int pre_idx = idx / n_conn;
        int conn_idx = idx % n_conn;

        if (spikes[pre_idx]) {
            int offset = pre_idx * n_conn + conn_idx;
            int post_idx = indices[offset];
            float w = weights[offset];
            atomicAdd(&output[post_idx], w);
        }
    }
}

//=============================================================================
// Kernel 5: Two-pass kernel - first compact active indices, then process
// Best for very sparse firing (<5% active)
//=============================================================================
__global__ void count_active_kernel(
    const bool* __restrict__ spikes,
    int* __restrict__ active_count,
    int n_pre
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory for block reduction
    __shared__ int block_count;
    if (threadIdx.x == 0) block_count = 0;
    __syncthreads();

    int local_count = 0;
    for (int i = tid; i < n_pre; i += blockDim.x * gridDim.x) {
        if (spikes[i]) local_count++;
    }

    atomicAdd(&block_count, local_count);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(active_count, block_count);
    }
}

__global__ void compact_active_kernel(
    const bool* __restrict__ spikes,
    int* __restrict__ active_indices,
    int* __restrict__ counter,
    int n_pre
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tid; i < n_pre; i += blockDim.x * gridDim.x) {
        if (spikes[i]) {
            int pos = atomicAdd(counter, 1);
            active_indices[pos] = i;
        }
    }
}

__global__ void ell_mv_compact_kernel(
    const int* __restrict__ active_indices,
    int num_active,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    float* __restrict__ output,
    int n_conn
) {
    // Each block handles one active neuron
    int active_idx = blockIdx.x;
    if (active_idx >= num_active) return;

    int pre_idx = active_indices[active_idx];
    const int32_t* my_indices = indices + pre_idx * n_conn;
    const float* my_weights = weights + pre_idx * n_conn;

    for (int j = threadIdx.x; j < n_conn; j += blockDim.x) {
        int post_idx = my_indices[j];
        float w = my_weights[j];
        atomicAdd(&output[post_idx], w);
    }
}

//=============================================================================
// Main entry point - selects best kernel based on problem size
//=============================================================================
void ell_mv_cuda(
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    // Validate inputs
    TVM_FFI_ICHECK(spikes.ndim() == 1) << "spikes must be 1D";
    TVM_FFI_ICHECK(indices.ndim() == 2) << "indices must be 2D";
    TVM_FFI_ICHECK(weights.ndim() == 2) << "weights must be 2D";
    TVM_FFI_ICHECK(output.ndim() == 1) << "output must be 1D";

    int n_pre = spikes.size(0);
    int n_conn = indices.size(1);
    int n_post = output.size(0);

    TVM_FFI_ICHECK(indices.size(0) == n_pre) << "indices first dim must match n_pre";
    TVM_FFI_ICHECK(weights.size(0) == n_pre) << "weights first dim must match n_pre";
    TVM_FFI_ICHECK(weights.size(1) == n_conn) << "weights second dim must match n_conn";

    const bool* d_spikes = static_cast<const bool*>(spikes.data_ptr());
    const int32_t* d_indices = static_cast<const int32_t*>(indices.data_ptr());
    const float* d_weights = static_cast<const float*>(weights.data_ptr());
    float* d_output = static_cast<float*>(output.data_ptr());

    // Initialize output to zero
    cudaMemsetAsync(d_output, 0, n_post * sizeof(float), cuda_stream);

    // Select kernel based on problem characteristics
    // Heuristic: use grid-stride for large problems, basic for small
    if (n_pre * n_conn < 10000) {
        // Small problem: use basic kernel
        int threads = min(256, n_conn);
        ell_mv_basic_kernel<<<n_pre, threads, 0, cuda_stream>>>(
            d_spikes, d_indices, d_weights, d_output, n_pre, n_conn
        );
    } else if (n_conn >= 256) {
        // Large n_conn: use shared memory kernel
        int threads = 256;
        size_t shared_size = threads * (sizeof(int32_t) + sizeof(float));
        ell_mv_shared_kernel<<<n_pre, threads, shared_size, cuda_stream>>>(
            d_spikes, d_indices, d_weights, d_output, n_pre, n_conn
        );
    } else {
        // General case: grid-stride kernel
        int threads = 256;
        int blocks = min(1024, (n_pre * n_conn + threads - 1) / threads);
        ell_mv_gridstride_kernel<<<blocks, threads, 0, cuda_stream>>>(
            d_spikes, d_indices, d_weights, d_output, n_pre, n_conn
        );
    }

    // Check for errors
    cudaError_t err = cudaGetLastError();
    TVM_FFI_ICHECK(err == cudaSuccess)
        << "CUDA kernel launch failed: " << cudaGetErrorString(err);
}

//=============================================================================
// Explicit kernel selection entry points for benchmarking
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

    const bool* d_spikes = static_cast<const bool*>(spikes.data_ptr());
    const int32_t* d_indices = static_cast<const int32_t*>(indices.data_ptr());
    const float* d_weights = static_cast<const float*>(weights.data_ptr());
    float* d_output = static_cast<float*>(output.data_ptr());

    cudaMemsetAsync(d_output, 0, n_post * sizeof(float), cuda_stream);

    int threads = min(256, n_conn);
    ell_mv_basic_kernel<<<n_pre, threads, 0, cuda_stream>>>(
        d_spikes, d_indices, d_weights, d_output, n_pre, n_conn
    );
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

    const bool* d_spikes = static_cast<const bool*>(spikes.data_ptr());
    const int32_t* d_indices = static_cast<const int32_t*>(indices.data_ptr());
    const float* d_weights = static_cast<const float*>(weights.data_ptr());
    float* d_output = static_cast<float*>(output.data_ptr());

    cudaMemsetAsync(d_output, 0, n_post * sizeof(float), cuda_stream);

    int threads = 256;
    int blocks = min(1024, (n_pre * n_conn + threads - 1) / threads);
    ell_mv_gridstride_kernel<<<blocks, threads, 0, cuda_stream>>>(
        d_spikes, d_indices, d_weights, d_output, n_pre, n_conn
    );
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

    const bool* d_spikes = static_cast<const bool*>(spikes.data_ptr());
    const int32_t* d_indices = static_cast<const int32_t*>(indices.data_ptr());
    const float* d_weights = static_cast<const float*>(weights.data_ptr());
    float* d_output = static_cast<float*>(output.data_ptr());

    cudaMemsetAsync(d_output, 0, n_post * sizeof(float), cuda_stream);

    int threads = 256;  // 8 warps per block
    int blocks = min(1024, (n_pre + 7) / 8);  // Roughly one warp per pre-neuron
    ell_mv_warp_kernel<<<blocks, threads, 0, cuda_stream>>>(
        d_spikes, d_indices, d_weights, d_output, n_pre, n_conn
    );
}
    """,
    functions=["ell_mv_cuda", "ell_mv_basic", "ell_mv_gridstride", "ell_mv_warp"],
)


def _register_kernels():
    """Register CUDA kernels with JAX FFI."""
    # Main auto-selecting kernel
    jax_tvm_ffi.register_ffi_target(
        "ell_mv.cuda",
        _cuda_module.ell_mv_cuda,
        ["args", "rets", "ctx.stream"],
        platform="gpu",
    )

    # Individual kernel variants for benchmarking
    jax_tvm_ffi.register_ffi_target(
        "ell_mv.basic",
        _cuda_module.ell_mv_basic,
        ["args", "rets", "ctx.stream"],
        platform="gpu",
    )

    jax_tvm_ffi.register_ffi_target(
        "ell_mv.gridstride",
        _cuda_module.ell_mv_gridstride,
        ["args", "rets", "ctx.stream"],
        platform="gpu",
    )

    jax_tvm_ffi.register_ffi_target(
        "ell_mv.warp",
        _cuda_module.ell_mv_warp,
        ["args", "rets", "ctx.stream"],
        platform="gpu",
    )


# Register on module import
_register_kernels()


def ell_mv(
    spikes: Array,
    indices: Array,
    weights: Array,
    *,
    n_post: int,
    kernel: str = "auto"
) -> Array:
    """
    ELL format sparse matrix-vector multiplication for neural networks.

    For each pre-synaptic neuron i where spikes[i] == True:
        For each connection j: output[indices[i,j]] += weights[i,j]

    Args:
        spikes: Boolean array of shape [n_pre] indicating active neurons
        indices: Int32 array of shape [n_pre, n_conn] with post-synaptic indices
        weights: Float32 array of shape [n_pre, n_conn] with connection weights
        n_post: Number of post-synaptic neurons (output size)
        kernel: Kernel variant to use: "auto", "basic", "gridstride", or "warp"

    Returns:
        Float32 array of shape [n_post] with accumulated weights
    """
    kernel_map = {
        "auto": "ell_mv.cuda",
        "basic": "ell_mv.basic",
        "gridstride": "ell_mv.gridstride",
        "warp": "ell_mv.warp",
    }

    if kernel not in kernel_map:
        raise ValueError(f"Unknown kernel: {kernel}. Choose from {list(kernel_map.keys())}")

    return jax.ffi.ffi_call(
        kernel_map[kernel],
        jax.ShapeDtypeStruct((n_post,), weights.dtype),
    )(spikes, indices, weights)


def ell_mv_jit(n_post: int, kernel: str = "auto"):
    """
    Create a JIT-compiled ELL mv function with fixed output size.

    Args:
        n_post: Number of post-synaptic neurons
        kernel: Kernel variant to use

    Returns:
        JIT-compiled function (spikes, indices, weights) -> output
    """

    @jax.jit
    def _ell_mv(spikes: Array, indices: Array, weights: Array) -> Array:
        return ell_mv(spikes, indices, weights, n_post=n_post, kernel=kernel)

    return _ell_mv


# =============================================================================
# Reference JAX implementation for correctness testing
# =============================================================================

def ell_mv_reference(spikes, indices, weights, *, n_post):
    """Pure JAX reference implementation."""

    def accumulate_for_neuron(carry, inputs):
        spike, inds, wts = inputs
        # Use where to conditionally add
        carry = jax.lax.cond(
            spike,
            lambda c: c.at[inds].add(wts),
            lambda c: c,
            carry
        )
        return carry, None

    output = jnp.zeros(n_post, dtype=weights.dtype)
    output, _ = jax.lax.scan(accumulate_for_neuron, output, (spikes, indices, weights))
    return output


def ell_mv_reference_vectorized(spikes, indices, weights, *, n_post):
    """Vectorized JAX reference implementation using segment_sum."""
    # Mask out inactive neurons
    active_mask = spikes[:, None]  # [n_pre, 1]
    masked_weights = jnp.where(active_mask, weights, 0.0)  # [n_pre, n_conn]

    # Flatten and use segment_sum
    flat_indices = indices.ravel()
    flat_weights = masked_weights.ravel()

    output = jax.ops.segment_sum(flat_weights, flat_indices, num_segments=n_post)
    return output


# =============================================================================
# Main demonstration and benchmarking
# =============================================================================

def main():
    import jax.random as jr
    import time

    print("=" * 70)
    print("ELL Sparse Matrix-Vector Multiplication - CUDA Kernel")
    print("=" * 70)

    # Check for GPU
    try:
        gpu_devices = jax.devices("gpu")
        if not gpu_devices:
            print("\nERROR: No GPU available. This example requires a GPU.")
            return
    except RuntimeError:
        print("\nERROR: No GPU backend available. This example requires a GPU.")
        return

    print(f"\nUsing GPU: {gpu_devices[0]}")

    # Test parameters
    n_pre = 50000
    n_post = 50000
    p = 0.1  # Connection probability
    n_conn = int(n_post * p)

    print(f"\nProblem size:")
    print(f"  n_pre  = {n_pre}")
    print(f"  n_post = {n_post}")
    print(f"  n_conn = {n_conn}")
    print(f"  Total connections = {n_pre * n_conn:,}")

    # Generate test data
    keys = jr.split(jr.PRNGKey(42), 3)
    spikes = jr.uniform(keys[0], [n_pre]) < 0.5
    indices = jr.randint(keys[1], [n_pre, n_conn], 0, n_post, dtype=jnp.int32)
    weights = jr.normal(keys[2], [n_pre, n_conn], dtype=jnp.float32)

    n_active = int(spikes.sum())
    print(f"  Active neurons = {n_active} ({100 * n_active / n_pre:.1f}%)")

    # Move to GPU
    spikes_gpu = jax.device_put(spikes, gpu_devices[0])
    indices_gpu = jax.device_put(indices, gpu_devices[0])
    weights_gpu = jax.device_put(weights, gpu_devices[0])

    # Test correctness
    print("\n" + "-" * 40)
    print("Correctness Test")
    print("-" * 40)

    # Reference implementation
    y_ref = ell_mv_reference_vectorized(spikes, indices, weights, n_post=n_post)
    print(f"Reference: mean={float(y_ref.mean()):.6f}, var={float(y_ref.var()):.6f}")

    # Test each kernel variant
    for kernel_name in ["auto", "basic", "gridstride", "warp"]:
        y_cuda = ell_mv(spikes_gpu, indices_gpu, weights_gpu, n_post=n_post, kernel=kernel_name)
        y_cuda = jax.device_get(y_cuda)

        max_diff = float(jnp.abs(y_cuda - y_ref).max())
        print(f"{kernel_name:12s}: mean={float(y_cuda.mean()):.6f}, "
              f"var={float(y_cuda.var()):.6f}, max_diff={max_diff:.2e}")

        if max_diff > 1e-5:
            print(f"  WARNING: Large difference detected!")

    # Benchmark
    print("\n" + "-" * 40)
    print("Performance Benchmark")
    print("-" * 40)

    n_warmup = 10
    n_runs = 100

    for kernel_name in ["auto", "basic", "gridstride", "warp"]:
        fn = ell_mv_jit(n_post, kernel=kernel_name)

        # Warmup
        for _ in range(n_warmup):
            _ = fn(spikes_gpu, indices_gpu, weights_gpu)
        jax.block_until_ready(_)

        # Timed runs
        start = time.perf_counter()
        for _ in range(n_runs):
            result = fn(spikes_gpu, indices_gpu, weights_gpu)
        jax.block_until_ready(result)
        elapsed = time.perf_counter() - start

        avg_time_us = (elapsed / n_runs) * 1e6
        throughput = (n_pre * n_conn) / (elapsed / n_runs) / 1e9  # Billion elements/sec

        print(f"{kernel_name:12s}: {avg_time_us:8.2f} µs/call, {throughput:.2f} Gelem/s")

    # Also benchmark the reference
    fn_ref = jax.jit(lambda s, i, w: ell_mv_reference_vectorized(s, i, w, n_post=n_post))
    for _ in range(n_warmup):
        _ = fn_ref(spikes_gpu, indices_gpu, weights_gpu)
    jax.block_until_ready(_)

    start = time.perf_counter()
    for _ in range(n_runs):
        result = fn_ref(spikes_gpu, indices_gpu, weights_gpu)
    jax.block_until_ready(result)
    elapsed = time.perf_counter() - start

    avg_time_us = (elapsed / n_runs) * 1e6
    throughput = (n_pre * n_conn) / (elapsed / n_runs) / 1e9
    print(f"{'JAX ref':12s}: {avg_time_us:8.2f} µs/call, {throughput:.2f} Gelem/s")

    print("\n" + "=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()