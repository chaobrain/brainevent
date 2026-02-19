// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

/*
 * binary_densemv.cu -- Event-Driven Binary Dense Matrix-Vector CUDA Kernels
 * ==========================================================================
 *
 * Python API: brainevent.binary_densemv(weights, spikes, *, transpose, backend)
 *
 * Event-driven dense matrix-vector product where spikes are boolean (0/1)
 * or float (>0 = active). Only columns/rows corresponding to active spikes
 * contribute, allowing the kernel to skip work for inactive entries.
 *
 * Parameters
 * ----------
 * weights : dense float32 matrix.
 *     transpose=False (gather): shape (m, k)  =>  out[m]
 *     transpose=True  (scatter): shape (k, n)  =>  out[n]
 * spikes : 1-D vector of shape (k,).
 *     bool  (int8): active when != 0.
 *     float (f32):  active when  > 0.
 *
 * Gather mode (transpose=False):
 *   out[i] = sum_{j where spikes[j] active} weights[i, j]
 *
 * Scatter mode (transpose=True):
 *   out[j] = sum_{i where spikes[i] active} weights[i, j]
 *
 * Kernel variants
 * ---------------
 * Gather:
 *   _gather_warp:  one warp  (32 threads) per output row. Best for k <= 1024.
 *   _gather_block: one block (256 threads) per output row. Best for k > 1024.
 *   _gather_auto:  auto-selects warp vs block based on k.
 *
 * Scatter:
 *   _scatter:      each thread handles one output element j, iterates over all
 *                  k spikes. All threads in a warp evaluate the SAME spike,
 *                  so there is zero warp divergence. Coalesced weight reads.
 *
 * Each variant has _bool and _float suffixes for the two spike types.
 *
 * IMPORTANT: weights.data_ptr() is a GPU device pointer -- NEVER dereference
 * on host. Pass it to kernels unchanged. GPU threads read from device memory.
 */

#include <cuda_runtime.h>
#include <cstdint>

// =========================================================================
// Warp-level reduction helper
// =========================================================================

__device__ __inline__ float warp_reduce_sum_f32(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// =========================================================================
// Gather device kernels (transpose=False)
//
// out[i] = sum_{j where spikes[j] active} weights[i, j]
// weights: [m, k] row-major.  spikes: [k].  output: [m].
//
// Strategy: one block/warp per output row.  Threads within the block
// stride over columns j, check spikes[j], conditionally load weights[i,j].
// Consecutive threads read consecutive columns => coalesced loads.
// Block/warp reduction produces the final scalar output[row].
// =========================================================================

// ---- Macro-templated gather_warp (one warp per row, k <= 1024) ----------
#define DEFINE_GATHER_WARP(SUFFIX, SPIKE_T, IS_ACTIVE)                     \
__global__ void _gather_warp_kern##SUFFIX(                                 \
    const float*    __restrict__ weights,                                   \
    const SPIKE_T*  __restrict__ spikes,                                    \
    float*          __restrict__ output,                                    \
    int m, int k                                                            \
) {                                                                         \
    int row = blockIdx.x;                                                   \
    if (row >= m) return;                                                    \
    const float* w_row = weights + (size_t)row * k;                        \
    float acc = 0.0f;                                                       \
    for (int j = threadIdx.x; j < k; j += 32) {                           \
        if (IS_ACTIVE(spikes[j])) {                                        \
            acc += w_row[j];                                                \
        }                                                                   \
    }                                                                       \
    acc = warp_reduce_sum_f32(acc);                                        \
    if (threadIdx.x == 0) output[row] = acc;                               \
}

// ---- Macro-templated gather_block (one block per row, large k) ----------
#define DEFINE_GATHER_BLOCK(SUFFIX, SPIKE_T, IS_ACTIVE)                    \
__global__ void _gather_block_kern##SUFFIX(                                \
    const float*    __restrict__ weights,                                   \
    const SPIKE_T*  __restrict__ spikes,                                    \
    float*          __restrict__ output,                                    \
    int m, int k                                                            \
) {                                                                         \
    extern __shared__ float smem_red[];                                    \
    int row = blockIdx.x;                                                   \
    if (row >= m) return;                                                    \
    const float* w_row = weights + (size_t)row * k;                        \
    float acc = 0.0f;                                                       \
    for (int j = threadIdx.x; j < k; j += blockDim.x) {                   \
        if (IS_ACTIVE(spikes[j])) {                                        \
            acc += w_row[j];                                                \
        }                                                                   \
    }                                                                       \
    /* Block reduction: warp shuffle + shared memory */                    \
    int lane   = threadIdx.x & 31;                                         \
    int warpid = threadIdx.x >> 5;                                         \
    acc = warp_reduce_sum_f32(acc);                                        \
    if (lane == 0) smem_red[warpid] = acc;                                 \
    __syncthreads();                                                        \
    int n_warps = (blockDim.x + 31) >> 5;                                  \
    acc = (threadIdx.x < n_warps) ? smem_red[lane] : 0.0f;                \
    if (warpid == 0) acc = warp_reduce_sum_f32(acc);                       \
    if (threadIdx.x == 0) output[row] = acc;                               \
}

// =========================================================================
// Scatter device kernels (transpose=True)
//
// out[j] = sum_{i where spikes[i] active} weights[i, j]
// weights: [k, n] row-major.  spikes: [k].  output: [n].
//
// Strategy: each thread handles one output element j.  All threads in a
// warp evaluate the SAME spike[i] => zero warp divergence.  Weight reads
// weights[i*n + j] for consecutive threads j => coalesced.  No atomics.
// =========================================================================

#define DEFINE_SCATTER(SUFFIX, SPIKE_T, IS_ACTIVE)                         \
__global__ void _scatter_kern##SUFFIX(                                     \
    const float*    __restrict__ weights,                                   \
    const SPIKE_T*  __restrict__ spikes,                                    \
    float*          __restrict__ output,                                    \
    int k, int n                                                            \
) {                                                                         \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (j >= n) return;                                                     \
    float acc = 0.0f;                                                       \
    for (int i = 0; i < k; i++) {                                          \
        if (IS_ACTIVE(spikes[i])) {                                        \
            acc += weights[(size_t)i * n + j];                             \
        }                                                                   \
    }                                                                       \
    output[j] = acc;                                                        \
}

// =========================================================================
// Instantiate kernels for bool (int8) and float spike types
// =========================================================================

// Active-check predicates
#define IS_ACTIVE_BOOL(s)  ((s) != 0)
#define IS_ACTIVE_FLOAT(s) ((s) > 0.0f)

// Gather warp variants
DEFINE_GATHER_WARP(_f32_bool,  int8_t, IS_ACTIVE_BOOL)
DEFINE_GATHER_WARP(_f32_float, float,  IS_ACTIVE_FLOAT)

// Gather block variants
DEFINE_GATHER_BLOCK(_f32_bool,  int8_t, IS_ACTIVE_BOOL)
DEFINE_GATHER_BLOCK(_f32_float, float,  IS_ACTIVE_FLOAT)

// Scatter variants
DEFINE_SCATTER(_f32_bool,  int8_t, IS_ACTIVE_BOOL)
DEFINE_SCATTER(_f32_float, float,  IS_ACTIVE_FLOAT)


// =========================================================================
// TVM FFI Entry Points
// =========================================================================
//
// Convention: args = (weights, spikes, output, stream)
//   Gather: weights [m, k], spikes [k], output [m]
//   Scatter: weights [k, n], spikes [k], output [n]
//
// IMPORTANT: data_ptr() returns GPU device pointers.
// NEVER dereference on the host. Pass to kernels unchanged.
// =========================================================================

// --- Gather entry points (transpose=False) ---

void binary_densemv_gather_warp_f32_bool(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int m = static_cast<int>(weights.size(0));
    int k = static_cast<int>(weights.size(1));
    const float*  d_w   = static_cast<const float*>(weights.data_ptr());
    const int8_t* d_spk = static_cast<const int8_t*>(spikes.data_ptr());
    float*        d_out = static_cast<float*>(output.data_ptr());
    _gather_warp_kern_f32_bool<<<m, 32, 0, s>>>(d_w, d_spk, d_out, m, k);
}

void binary_densemv_gather_warp_f32_float(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int m = static_cast<int>(weights.size(0));
    int k = static_cast<int>(weights.size(1));
    const float* d_w   = static_cast<const float*>(weights.data_ptr());
    const float* d_spk = static_cast<const float*>(spikes.data_ptr());
    float*       d_out = static_cast<float*>(output.data_ptr());
    _gather_warp_kern_f32_float<<<m, 32, 0, s>>>(d_w, d_spk, d_out, m, k);
}

void binary_densemv_gather_block_f32_bool(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int m = static_cast<int>(weights.size(0));
    int k = static_cast<int>(weights.size(1));
    const float*  d_w   = static_cast<const float*>(weights.data_ptr());
    const int8_t* d_spk = static_cast<const int8_t*>(spikes.data_ptr());
    float*        d_out = static_cast<float*>(output.data_ptr());
    size_t shm = 32 * sizeof(float);  // block reduction scratchpad
    _gather_block_kern_f32_bool<<<m, 256, shm, s>>>(d_w, d_spk, d_out, m, k);
}

void binary_densemv_gather_block_f32_float(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int m = static_cast<int>(weights.size(0));
    int k = static_cast<int>(weights.size(1));
    const float* d_w   = static_cast<const float*>(weights.data_ptr());
    const float* d_spk = static_cast<const float*>(spikes.data_ptr());
    float*       d_out = static_cast<float*>(output.data_ptr());
    size_t shm = 32 * sizeof(float);
    _gather_block_kern_f32_float<<<m, 256, shm, s>>>(d_w, d_spk, d_out, m, k);
}

// Auto-dispatch: selects warp or block kernel based on k.
void binary_densemv_gather_auto_f32_bool(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int m = static_cast<int>(weights.size(0));
    int k = static_cast<int>(weights.size(1));
    const float*  d_w   = static_cast<const float*>(weights.data_ptr());
    const int8_t* d_spk = static_cast<const int8_t*>(spikes.data_ptr());
    float*        d_out = static_cast<float*>(output.data_ptr());
    if (k <= 1024) {
        _gather_warp_kern_f32_bool<<<m, 32, 0, s>>>(d_w, d_spk, d_out, m, k);
    } else {
        size_t shm = 32 * sizeof(float);
        _gather_block_kern_f32_bool<<<m, 256, shm, s>>>(d_w, d_spk, d_out, m, k);
    }
}

void binary_densemv_gather_auto_f32_float(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int m = static_cast<int>(weights.size(0));
    int k = static_cast<int>(weights.size(1));
    const float* d_w   = static_cast<const float*>(weights.data_ptr());
    const float* d_spk = static_cast<const float*>(spikes.data_ptr());
    float*       d_out = static_cast<float*>(output.data_ptr());
    if (k <= 1024) {
        _gather_warp_kern_f32_float<<<m, 32, 0, s>>>(d_w, d_spk, d_out, m, k);
    } else {
        size_t shm = 32 * sizeof(float);
        _gather_block_kern_f32_float<<<m, 256, shm, s>>>(d_w, d_spk, d_out, m, k);
    }
}

// --- Scatter entry points (transpose=True) ---

void binary_densemv_scatter_f32_bool(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int k = static_cast<int>(weights.size(0));
    int n = static_cast<int>(weights.size(1));
    const float*  d_w   = static_cast<const float*>(weights.data_ptr());
    const int8_t* d_spk = static_cast<const int8_t*>(spikes.data_ptr());
    float*        d_out = static_cast<float*>(output.data_ptr());
    int blocks = (n + 255) / 256;
    _scatter_kern_f32_bool<<<blocks, 256, 0, s>>>(d_w, d_spk, d_out, k, n);
}

void binary_densemv_scatter_f32_float(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int k = static_cast<int>(weights.size(0));
    int n = static_cast<int>(weights.size(1));
    const float* d_w   = static_cast<const float*>(weights.data_ptr());
    const float* d_spk = static_cast<const float*>(spikes.data_ptr());
    float*       d_out = static_cast<float*>(output.data_ptr());
    int blocks = (n + 255) / 256;
    _scatter_kern_f32_float<<<blocks, 256, 0, s>>>(d_w, d_spk, d_out, k, n);
}
