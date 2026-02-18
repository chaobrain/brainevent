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
ELL Sparse Matrix-Vector Multiplication — CUDA Kernels and JAX Benchmarks

Registers six optimized CUDA kernels via TVM FFI and provides Python wrappers
and comprehensive benchmarks for ELL-format spike-driven weight accumulation.

Operation: For each pre-synaptic neuron i where spikes[i] == True:
    For each connection j: output[indices[i,j]] += weights[i,j]

Kernels
-------
auto        : Smart heuristic that selects the best kernel for the problem size
basic       : One CUDA block per pre-synaptic neuron; simple and readable
shared      : Shared-memory prefetching; better for large n_conn
gridstride  : Flattened grid-stride loop; best general-purpose throughput
warp        : One warp per neuron; good for sparse firing patterns
vectorized  : float4/int4 vectorized loads; best when n_conn is a multiple of 4

Requirements
------------
    pip install jax jaxlib jax-tvm-ffi tvm-ffi

Usage
-----
    # Use in code
    from ell_mv_cuda import ell_mv, ell_mv_jit

    y = ell_mv(spikes, indices, weights, n_post=n_post)

    # Run full benchmark suite
    python ell_mv_cuda.py
"""

import time
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

import brainevent

# =============================================================================
# CUDA Kernel Source Code
# =============================================================================

brainevent.register_tvm_cuda_kernels(
    module='ell_mv',
    functions=[
        "ell_mv_cuda",  # auto-selection
        "ell_mv_basic",
        "ell_mv_shared",
        "ell_mv_gridstride",
        "ell_mv_warp",
        "ell_mv_vectorized",
    ],
    source_code=r"""
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

//=============================================================================
// Kernel 1: Basic — one CUDA block per pre-synaptic neuron.
// Good for small n_conn; simple and readable.
//=============================================================================
__global__ void ell_mv_basic_kernel(
    const bool*    __restrict__ spikes,   // [n_pre]
    const int32_t* __restrict__ indices,  // [n_pre, n_conn]
    const float*   __restrict__ weights,  // [n_pre, n_conn]
    float*         __restrict__ output,   // [n_post]
    int n_pre,
    int n_conn
) {
    int pre_idx = blockIdx.x;
    if (pre_idx >= n_pre) return;
    if (!spikes[pre_idx]) return;

    const int32_t* my_indices = indices + pre_idx * n_conn;
    const float*   my_weights = weights + pre_idx * n_conn;

    for (int j = threadIdx.x; j < n_conn; j += blockDim.x) {
        atomicAdd(&output[my_indices[j]], my_weights[j]);
    }
}

//=============================================================================
// Kernel 2: Shared memory — cooperative tile-loading into shared memory.
// Better for larger n_conn values (reduces global memory latency).
//=============================================================================
__global__ void ell_mv_shared_kernel(
    const bool*    __restrict__ spikes,
    const int32_t* __restrict__ indices,
    const float*   __restrict__ weights,
    float*         __restrict__ output,
    int n_pre,
    int n_conn
) {
    extern __shared__ char shared_mem[];
    int32_t* s_indices = reinterpret_cast<int32_t*>(shared_mem);
    float*   s_weights = reinterpret_cast<float*>(shared_mem + blockDim.x * sizeof(int32_t));

    int pre_idx = blockIdx.x;
    if (pre_idx >= n_pre) return;
    if (!spikes[pre_idx]) return;

    const int32_t* my_indices = indices + pre_idx * n_conn;
    const float*   my_weights = weights + pre_idx * n_conn;

    const int TILE_SIZE = blockDim.x;
    int num_tiles = (n_conn + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < num_tiles; tile++) {
        int j = tile * TILE_SIZE + threadIdx.x;

        // Cooperative load into shared memory
        if (j < n_conn) {
            s_indices[threadIdx.x] = my_indices[j];
            s_weights[threadIdx.x] = my_weights[j];
        }
        __syncthreads();

        // Scatter from shared memory
        if (j < n_conn) {
            atomicAdd(&output[s_indices[threadIdx.x]], s_weights[threadIdx.x]);
        }
        __syncthreads();
    }
}

//=============================================================================
// Kernel 3: Warp-optimized — one warp per pre-synaptic neuron.
// Good for sparse firing patterns (few active neurons per step).
//=============================================================================
__global__ void ell_mv_warp_kernel(
    const bool*    __restrict__ spikes,
    const int32_t* __restrict__ indices,
    const float*   __restrict__ weights,
    float*         __restrict__ output,
    int n_pre,
    int n_conn
) {
    int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id   = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int pre_idx = warp_id; pre_idx < n_pre; pre_idx += num_warps) {
        if (!spikes[pre_idx]) continue;

        const int32_t* my_indices = indices + pre_idx * n_conn;
        const float*   my_weights = weights + pre_idx * n_conn;

        for (int j = lane_id; j < n_conn; j += 32) {
            atomicAdd(&output[my_indices[j]], my_weights[j]);
        }
    }
}

//=============================================================================
// Kernel 4: Grid-stride — flattened iteration over all (pre, conn) pairs.
// Best general-purpose kernel; maximizes SM occupancy.
//=============================================================================
__global__ void ell_mv_gridstride_kernel(
    const bool*    __restrict__ spikes,
    const int32_t* __restrict__ indices,
    const float*   __restrict__ weights,
    float*         __restrict__ output,
    int n_pre,
    int n_conn
) {
    int total = n_pre * n_conn;
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total; idx += stride) {
        int pre_idx = idx / n_conn;
        if (spikes[pre_idx]) {
            atomicAdd(&output[indices[idx]], weights[idx]);
        }
    }
}

//=============================================================================
// Kernel 5: Vectorized — float4/int4 coalesced loads.
// Highest throughput when n_conn is a multiple of 4.
//=============================================================================
__global__ void ell_mv_vectorized_kernel(
    const bool*    __restrict__ spikes,
    const int32_t* __restrict__ indices,
    const float*   __restrict__ weights,
    float*         __restrict__ output,
    int n_pre,
    int n_conn
) {
    int pre_idx = blockIdx.x;
    if (pre_idx >= n_pre) return;
    if (!spikes[pre_idx]) return;

    const int base_offset = pre_idx * n_conn;
    const int n_vec = n_conn / 4;

    const int4*   idx_vec = reinterpret_cast<const int4*>(indices + base_offset);
    const float4* wt_vec  = reinterpret_cast<const float4*>(weights + base_offset);

    for (int j = threadIdx.x; j < n_vec; j += blockDim.x) {
        int4   idx4 = idx_vec[j];
        float4 w4   = wt_vec[j];
        atomicAdd(&output[idx4.x], w4.x);
        atomicAdd(&output[idx4.y], w4.y);
        atomicAdd(&output[idx4.z], w4.z);
        atomicAdd(&output[idx4.w], w4.w);
    }

    // Handle remainder elements (n_conn not divisible by 4)
    for (int j = n_vec * 4 + threadIdx.x; j < n_conn; j += blockDim.x) {
        atomicAdd(&output[indices[base_offset + j]], weights[base_offset + j]);
    }
}

//=============================================================================
// Two-pass compact kernels (helpers for the auto-selection entry point).
// Useful for very sparse firing (<5% active neurons).
//=============================================================================
__global__ void count_active_kernel(
    const bool* __restrict__ spikes,
    int* __restrict__ active_count,
    int n_pre
) {
    __shared__ int block_count;
    if (threadIdx.x == 0) block_count = 0;
    __syncthreads();

    int local_count = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_pre; i += blockDim.x * gridDim.x) {
        if (spikes[i]) local_count++;
    }
    atomicAdd(&block_count, local_count);
    __syncthreads();

    if (threadIdx.x == 0) atomicAdd(active_count, block_count);
}

__global__ void compact_active_kernel(
    const bool* __restrict__ spikes,
    int* __restrict__ active_indices,
    int* __restrict__ counter,
    int n_pre
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_pre; i += blockDim.x * gridDim.x) {
        if (spikes[i]) active_indices[atomicAdd(counter, 1)] = i;
    }
}

__global__ void ell_mv_compact_kernel(
    const int*     __restrict__ active_indices,
    int            num_active,
    const int32_t* __restrict__ indices,
    const float*   __restrict__ weights,
    float*         __restrict__ output,
    int n_conn
) {
    int active_idx = blockIdx.x;
    if (active_idx >= num_active) return;

    int pre_idx = active_indices[active_idx];
    const int32_t* my_indices = indices + pre_idx * n_conn;
    const float*   my_weights = weights + pre_idx * n_conn;

    for (int j = threadIdx.x; j < n_conn; j += blockDim.x) {
        atomicAdd(&output[my_indices[j]], my_weights[j]);
    }
}

//=============================================================================
// TVM FFI Entry Points
//=============================================================================

// Auto-selection: picks the best kernel based on problem characteristics.
void ell_mv_cuda(
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    TVM_FFI_ICHECK(spikes.ndim()  == 1) << "spikes must be 1D";
    TVM_FFI_ICHECK(indices.ndim() == 2) << "indices must be 2D";
    TVM_FFI_ICHECK(weights.ndim() == 2) << "weights must be 2D";
    TVM_FFI_ICHECK(output.ndim()  == 1) << "output must be 1D";

    int n_pre  = spikes.size(0);
    int n_conn = indices.size(1);
    int n_post = output.size(0);

    const bool*    d_spikes  = static_cast<const bool*>   (spikes.data_ptr());
    const int32_t* d_indices = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_weights = static_cast<const float*>  (weights.data_ptr());
    float*         d_output  = static_cast<float*>        (output.data_ptr());

    cudaMemsetAsync(d_output, 0, n_post * sizeof(float), cuda_stream);

    if (n_pre * n_conn < 10000) {
        // Small problem: basic kernel
        int threads = min(256, n_conn);
        ell_mv_basic_kernel<<<n_pre, threads, 0, cuda_stream>>>(
            d_spikes, d_indices, d_weights, d_output, n_pre, n_conn);
    } else if (n_conn >= 256) {
        // Large n_conn: shared memory kernel
        int threads = 256;
        size_t shared_size = threads * (sizeof(int32_t) + sizeof(float));
        ell_mv_shared_kernel<<<n_pre, threads, shared_size, cuda_stream>>>(
            d_spikes, d_indices, d_weights, d_output, n_pre, n_conn);
    } else {
        // General case: grid-stride kernel
        int threads = 256;
        int blocks  = min(1024, (n_pre * n_conn + threads - 1) / threads);
        ell_mv_gridstride_kernel<<<blocks, threads, 0, cuda_stream>>>(
            d_spikes, d_indices, d_weights, d_output, n_pre, n_conn);
    }

    cudaError_t err = cudaGetLastError();
    TVM_FFI_ICHECK(err == cudaSuccess)
        << "CUDA kernel launch failed: " << cudaGetErrorString(err);
}

// Explicit entry points for benchmarking individual kernels.

void ell_mv_basic(
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = spikes.size(0);
    int n_conn = indices.size(1);
    int n_post = output.size(0);

    cudaMemsetAsync(output.data_ptr(), 0, n_post * sizeof(float), cuda_stream);
    int threads = min(256, n_conn);
    ell_mv_basic_kernel<<<n_pre, threads, 0, cuda_stream>>>(
        static_cast<const bool*>   (spikes.data_ptr()),
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>  (weights.data_ptr()),
        static_cast<float*>        (output.data_ptr()),
        n_pre, n_conn
    );
    TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "ell_mv_basic failed";
}

void ell_mv_shared(
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = spikes.size(0);
    int n_conn = indices.size(1);
    int n_post = output.size(0);

    cudaMemsetAsync(output.data_ptr(), 0, n_post * sizeof(float), cuda_stream);
    int threads = 256;
    size_t shared_size = threads * (sizeof(int32_t) + sizeof(float));
    ell_mv_shared_kernel<<<n_pre, threads, shared_size, cuda_stream>>>(
        static_cast<const bool*>   (spikes.data_ptr()),
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>  (weights.data_ptr()),
        static_cast<float*>        (output.data_ptr()),
        n_pre, n_conn
    );
    TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "ell_mv_shared failed";
}

void ell_mv_gridstride(
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = spikes.size(0);
    int n_conn = indices.size(1);
    int n_post = output.size(0);

    cudaMemsetAsync(output.data_ptr(), 0, n_post * sizeof(float), cuda_stream);
    int threads = 256;
    int blocks  = min(1024, (n_pre * n_conn + threads - 1) / threads);
    ell_mv_gridstride_kernel<<<blocks, threads, 0, cuda_stream>>>(
        static_cast<const bool*>   (spikes.data_ptr()),
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>  (weights.data_ptr()),
        static_cast<float*>        (output.data_ptr()),
        n_pre, n_conn
    );
    TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "ell_mv_gridstride failed";
}

void ell_mv_warp(
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = spikes.size(0);
    int n_conn = indices.size(1);
    int n_post = output.size(0);

    cudaMemsetAsync(output.data_ptr(), 0, n_post * sizeof(float), cuda_stream);
    int threads = 256;                          // 8 warps per block
    int blocks  = min(1024, (n_pre + 7) / 8);  // roughly one warp per pre-neuron
    ell_mv_warp_kernel<<<blocks, threads, 0, cuda_stream>>>(
        static_cast<const bool*>   (spikes.data_ptr()),
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>  (weights.data_ptr()),
        static_cast<float*>        (output.data_ptr()),
        n_pre, n_conn
    );
    TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "ell_mv_warp failed";
}

void ell_mv_vectorized(
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = spikes.size(0);
    int n_conn = indices.size(1);
    int n_post = output.size(0);

    cudaMemsetAsync(output.data_ptr(), 0, n_post * sizeof(float), cuda_stream);
    int threads = 256;
    ell_mv_vectorized_kernel<<<n_pre, threads, 0, cuda_stream>>>(
        static_cast<const bool*>   (spikes.data_ptr()),
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>  (weights.data_ptr()),
        static_cast<float*>        (output.data_ptr()),
        n_pre, n_conn
    );
    TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "ell_mv_vectorized failed";
}
""",
)

# All available kernel names (used for validation).
KERNEL_NAMES = ["auto", "basic", "shared", "gridstride", "warp", "vectorized"]

# Kernels exposed for per-kernel benchmarking (excludes "auto").
BENCHMARK_KERNELS = ["basic", "shared", "gridstride", "warp", "vectorized"]

_KERNEL_MAP = {
    "auto": "ell_mv.ell_mv_cuda",
    "basic": "ell_mv.ell_mv_basic",
    "shared": "ell_mv.ell_mv_shared",
    "gridstride": "ell_mv.ell_mv_gridstride",
    "warp": "ell_mv.ell_mv_warp",
    "vectorized": "ell_mv.ell_mv_vectorized",
}


# =============================================================================
# Python Wrappers
# =============================================================================

def ell_mv(
    spikes: Array,
    indices: Array,
    weights: Array,
    *,
    n_post: int,
    kernel: str = "auto",
) -> Array:
    """
    ELL format sparse matrix-vector multiplication for neural networks.

    For each active pre-synaptic neuron i (``spikes[i] == True``):
        For each connection j: ``output[indices[i,j]] += weights[i,j]``

    Parameters
    ----------
    spikes : Array, shape [n_pre], dtype bool
        Boolean array indicating which pre-synaptic neurons fired.
    indices : Array, shape [n_pre, n_conn], dtype int32
        Post-synaptic neuron index for each connection.
    weights : Array, shape [n_pre, n_conn], dtype float32
        Synaptic weight for each connection.
    n_post : int
        Number of post-synaptic neurons (output size).
    kernel : str, optional
        Kernel variant: ``"auto"`` (default), ``"basic"``, ``"shared"``,
        ``"gridstride"``, ``"warp"``, or ``"vectorized"``.
        ``"vectorized"`` requires ``n_conn`` to be a multiple of 4.

    Returns
    -------
    Array, shape [n_post], dtype float32
        Accumulated weighted inputs for each post-synaptic neuron.
    """
    if kernel not in _KERNEL_MAP:
        raise ValueError(f"Unknown kernel: {kernel!r}. Choose from {KERNEL_NAMES}")
    out = jax.ShapeDtypeStruct((n_post,), weights.dtype)
    return jax.ffi.ffi_call(_KERNEL_MAP[kernel], out)(spikes, indices, weights)


def ell_mv_jit(n_post: int, kernel: str = "auto") -> Callable:
    """
    Create a JIT-compiled ELL mv function with a fixed output size.

    Parameters
    ----------
    n_post : int
        Number of post-synaptic neurons.
    kernel : str, optional
        Kernel variant to use (see :func:`ell_mv`).

    Returns
    -------
    Callable
        JIT-compiled function ``(spikes, indices, weights) -> output``.
    """

    @jax.jit
    def _fn(spikes: Array, indices: Array, weights: Array) -> Array:
        return ell_mv(spikes, indices, weights, n_post=n_post, kernel=kernel)

    return _fn


# =============================================================================
# JAX Reference Implementations (correctness checking)
# =============================================================================

def ell_mv_reference(spikes, indices, weights, *, n_post: int) -> Array:
    """Pure JAX reference using ``lax.scan`` (sequential, JIT-compatible)."""

    def step(carry, inputs):
        spike, inds, wts = inputs
        carry = jax.lax.cond(
            spike,
            lambda c: c.at[inds].add(wts),
            lambda c: c,
            carry,
        )
        return carry, None

    output = jnp.zeros(n_post, dtype=weights.dtype)
    output, _ = jax.lax.scan(step, output, (spikes, indices, weights))
    return output


def ell_mv_reference_vectorized(spikes, indices, weights, *, n_post: int) -> Array:
    """Pure JAX reference using ``segment_sum`` (vectorized, fast)."""
    masked_weights = jnp.where(spikes[:, None], weights, 0.0)
    return jax.ops.segment_sum(
        masked_weights.ravel(), indices.ravel(), num_segments=n_post
    )


def ell_mv_reference_scatter(spikes, indices, weights, *, n_post: int) -> Array:
    """Pure JAX reference using ``at[].add`` scatter."""
    masked_weights = jnp.where(spikes[:, None], weights, 0.0)
    output = jnp.zeros(n_post, dtype=weights.dtype)
    return output.at[indices.ravel()].add(masked_weights.ravel())


# =============================================================================
# Benchmarking Infrastructure
# =============================================================================

def benchmark_function(
    fn: Callable,
    args: Tuple,
    n_warmup: int = 10,
    n_runs: int = 100,
    sync: bool = True,
) -> Dict:
    """Benchmark a function and return timing statistics (all times in µs)."""
    # Warmup
    result = None
    for _ in range(n_warmup):
        result = fn(*args)
    if sync and result is not None:
        jax.block_until_ready(result)

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn(*args)
        if sync:
            jax.block_until_ready(result)
        times.append(time.perf_counter() - t0)

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
    rtol: float = 1e-5,
) -> Tuple[bool, float]:
    """Return ``(allclose, max_abs_diff)`` comparing *result* to *reference*."""
    r = np.asarray(result)
    ref = np.asarray(reference)
    max_diff = float(np.abs(r - ref).max())
    return bool(np.allclose(r, ref, atol=atol, rtol=rtol)), max_diff


# =============================================================================
# Comprehensive Benchmark Suite
# =============================================================================

def run_benchmark(n_warmup: int = 20, n_runs: int = 100) -> None:
    """Run a comprehensive benchmark across multiple problem configurations."""
    print("=" * 75)
    print("ELL Sparse Matrix-Vector Multiplication — CUDA Kernel Benchmark")
    print("=" * 75)

    # (n_pre, n_post, n_conn, spike_rate, description)
    configs = [
        (100, 200, 20, 0.5, "Small  (100×20,  50% active)"),
        (100, 200, 20, 0.1, "Small  (100×20,  10% active)"),
        (1000, 2000, 200, 0.5, "Medium (1K×200,  50% active)"),
        (1000, 2000, 200, 0.1, "Medium (1K×200,  10% active)"),
        (10000, 20000, 200, 0.5, "Large  (10K×200, 50% active)"),
        (10000, 20000, 200, 0.1, "Large  (10K×200, 10% active)"),
        (10000, 20000, 200, 0.01, "Large  (10K×200,  1% active)"),
        (1000, 10000, 1000, 0.5, "Wide   (1K×1K,   50% active)"),
        (1000, 10000, 1000, 0.1, "Wide   (1K×1K,   10% active)"),
    ]

    # JAX reference implementations (always benchmarked for comparison)
    jax_impls = [
        ("JAX-segment", lambda s, i, w, n: ell_mv_reference_vectorized(s, i, w, n_post=n)),
        ("JAX-scatter", lambda s, i, w, n: ell_mv_reference_scatter(s, i, w, n_post=n)),
    ]

    all_results = []

    for n_pre, n_post, n_conn, spike_rate, desc in configs:
        print(f"\n{'-' * 75}")
        print(f"Config: {desc}")
        print(f"  n_pre={n_pre}, n_post={n_post}, n_conn={n_conn}, "
              f"total_conn={n_pre * n_conn:,}")
        print(f"{'-' * 75}")

        key = jr.PRNGKey(42)
        k1, k2, k3 = jr.split(key, 3)
        spikes = jr.uniform(k1, (n_pre,)) < spike_rate
        indices = jr.randint(k2, (n_pre, n_conn), 0, n_post, dtype=jnp.int32)
        weights = jr.normal(k3, (n_pre, n_conn), dtype=jnp.float32)

        n_active = int(spikes.sum())
        print(f"  Active neurons: {n_active} ({100 * n_active / n_pre:.1f}%)")

        device = jax.devices("gpu")[0]
        spikes = jax.device_put(spikes, device)
        indices = jax.device_put(indices, device)
        weights = jax.device_put(weights, device)

        # Reference output for correctness checks
        ref_fn = jax.jit(lambda s, i, w: ell_mv_reference_vectorized(s, i, w, n_post=n_post))
        reference = ref_fn(spikes, indices, weights)
        jax.block_until_ready(reference)

        results: Dict[str, Dict] = {}

        # Benchmark JAX reference implementations
        for name, impl in jax_impls:
            fn = jax.jit(lambda s, i, w: impl(s, i, w, n_post))
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

        # Benchmark CUDA kernels
        for kname in BENCHMARK_KERNELS:
            if kname == "vectorized" and n_conn % 4 != 0:
                results[f"CUDA-{kname}"] = {
                    "time_us": float("nan"), "std_us": float("nan"),
                    "correct": None, "max_diff": float("nan"), "skipped": True,
                }
                continue
            try:
                fn = ell_mv_jit(n_post, kernel=kname)
                result = fn(spikes, indices, weights)
                jax.block_until_ready(result)
                correct, max_diff = check_correctness(result, reference)
                timing = benchmark_function(fn, (spikes, indices, weights), n_warmup, n_runs)
                results[f"CUDA-{kname}"] = {
                    "time_us": timing["mean_us"],
                    "std_us": timing["std_us"],
                    "correct": correct,
                    "max_diff": max_diff,
                }
            except Exception as exc:
                results[f"CUDA-{kname}"] = {
                    "time_us": float("nan"), "std_us": float("nan"),
                    "correct": False, "max_diff": float("nan"), "error": str(exc),
                }

        # Print results table
        print(f"\n  {'Kernel':<20} {'Time (µs)':>12} {'Std (µs)':>10} "
              f"{'Gelem/s':>10} {'Status'}")
        print(f"  {'-' * 20} {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10}")

        best_time, best_name = float("inf"), None
        for name, data in results.items():
            if data.get("skipped"):
                print(f"  {name:<20} {'N/A':>12} {'N/A':>10} {'N/A':>10} SKIPPED")
                continue
            if "error" in data:
                print(f"  {name:<20} {'ERROR':>12} {'' * 10} {'' * 10} {data['error'][:20]}")
                continue
            t_us = data["time_us"]
            s_us = data["std_us"]
            gelem = (n_pre * n_conn) / (t_us * 1e3)  # Gelem/s
            status = "OK" if data["correct"] else f"FAIL ({data['max_diff']:.2e})"
            print(f"  {name:<20} {t_us:>12.2f} {s_us:>10.2f} {gelem:>10.2f} {status}")
            if data["correct"] and t_us < best_time:
                best_time, best_name = t_us, name

        if best_name:
            print(f"\n  Best: {best_name} ({best_time:.2f} µs)")

        all_results.append({"config": desc, "results": results,
                            "best": best_name, "best_time": best_time})

    # Summary
    print(f"\n{'=' * 75}")
    print("SUMMARY")
    print(f"{'=' * 75}")
    print(f"\n  {'Configuration':<38} {'Best Kernel':<22} {'Time (µs)':>12}")
    print(f"  {'-' * 38} {'-' * 22} {'-' * 12}")
    for r in all_results:
        best = r["best"] or "N/A"
        t = r["best_time"] if r["best_time"] < float("inf") else float("nan")
        print(f"  {r['config']:<38} {best:<22} {t:>12.2f}")

    print(f"\n{'=' * 75}")
    print("Benchmark complete.")
    print(f"{'=' * 75}")


# =============================================================================
# Quick Demo (single config)
# =============================================================================

def main() -> None:
    """Quick correctness + performance demo for a single problem size."""
    print("=" * 70)
    print("ELL Sparse Matrix-Vector Multiplication — Quick Demo")
    print("=" * 70)

    try:
        gpu_devices = jax.devices("gpu")
        if not gpu_devices:
            print("\nERROR: No GPU available.")
            return
    except RuntimeError:
        print("\nERROR: No GPU backend available.")
        return

    print(f"\nUsing GPU: {gpu_devices[0]}")

    n_pre, n_post, n_conn = 50000, 50000, 5000
    spike_rate = 0.1

    keys = jr.split(jr.PRNGKey(42), 3)
    spikes = jr.uniform(keys[0], [n_pre]) < spike_rate
    indices = jr.randint(keys[1], [n_pre, n_conn], 0, n_post, dtype=jnp.int32)
    weights = jr.normal(keys[2], [n_pre, n_conn], dtype=jnp.float32)

    print(f"\nProblem: n_pre={n_pre}, n_post={n_post}, n_conn={n_conn}, "
          f"active={int(spikes.sum())} ({100 * float(spikes.mean()):.1f}%)")

    spikes_gpu = jax.device_put(spikes, gpu_devices[0])
    indices_gpu = jax.device_put(indices, gpu_devices[0])
    weights_gpu = jax.device_put(weights, gpu_devices[0])

    reference = ell_mv_reference_vectorized(spikes, indices, weights, n_post=n_post)

    print("\nCorrectness check:")
    for kname in KERNEL_NAMES:
        if kname == "vectorized" and n_conn % 4 != 0:
            print(f"  {kname:<12}: SKIPPED (n_conn not divisible by 4)")
            continue
        try:
            y = ell_mv(spikes_gpu, indices_gpu, weights_gpu, n_post=n_post, kernel=kname)
            y = jax.device_get(y)
            ok, diff = check_correctness(y, reference)
            print(f"  {kname:<12}: {'OK' if ok else 'FAIL'}  max_diff={diff:.2e}")
        except Exception as exc:
            print(f"  {kname:<12}: ERROR — {exc}")

    print("\nPerformance (µs per call):")
    n_warmup, n_runs = 10, 100
    for kname in KERNEL_NAMES:
        if kname == "vectorized" and n_conn % 4 != 0:
            continue
        try:
            fn = ell_mv_jit(n_post, kernel=kname)
            timing = benchmark_function(fn, (spikes_gpu, indices_gpu, weights_gpu),
                                        n_warmup, n_runs)
            gelem = (n_pre * n_conn) / (timing["mean_us"] * 1e3)
            print(f"  {kname:<12}: {timing['mean_us']:8.2f} µs  ({gelem:.2f} Gelem/s)")
        except Exception as exc:
            print(f"  {kname:<12}: ERROR — {exc}")

    print("\n" + "=" * 70)
    print("Done!")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    run_benchmark()
