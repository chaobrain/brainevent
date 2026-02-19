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
 * weights : dense matrix of float16, bfloat16, float32, or float64.
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
 * Each variant has _bool and _float suffixes for the two spike types,
 * and _f32, _f64, _f16, _bf16 suffixes for the weight dtype.
 *
 * Float16 and bfloat16 kernels accumulate in float32 for numerical stability.
 * Bfloat16 requires CUDA 11.0+.
 *
 * IMPORTANT: weights.data_ptr() is a GPU device pointer -- NEVER dereference
 * on host. Pass it to kernels unchanged. GPU threads read from device memory.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// =========================================================================
// Warp-level reduction helpers
// =========================================================================

__device__ __inline__ float warp_reduce_sum_f32(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __inline__ double warp_reduce_sum_f64(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// =========================================================================
// Active-check predicates
// =========================================================================

#define IS_ACTIVE_BOOL(s)  ((s) != 0)
#define IS_ACTIVE_FLOAT(s) ((s) > 0.0f)

// =========================================================================
// Per-dtype conversion macros: READ converts WEIGHT_T -> ACC_T,
//                              WRITE converts ACC_T -> WEIGHT_T.
// =========================================================================

#define READ_F32(x)   (x)
#define WRITE_F32(x)  (x)

#define READ_F64(x)   (x)
#define WRITE_F64(x)  (x)

#define READ_F16(x)   __half2float(x)
#define WRITE_F16(x)  __float2half(x)

#define READ_BF16(x)  __bfloat162float(x)
#define WRITE_BF16(x) __float2bfloat16(x)

// =========================================================================
// Gather warp kernel macro (one warp per output row, k <= 1024)
//
// out[i] = sum_{j where spikes[j] active} weights[i, j]
// weights: [m, k] row-major.  spikes: [k].  output: [m].
// =========================================================================

#define DEFINE_GATHER_WARP(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T,    \
                           READ_W, WRITE_W, WARP_RED, ACC_ZERO)             \
__global__ void _gather_warp_kern##SUFFIX(                                  \
    const WEIGHT_T* __restrict__ weights,                                   \
    const SPIKE_T*  __restrict__ spikes,                                    \
    WEIGHT_T*       __restrict__ output,                                    \
    int m, int k                                                            \
) {                                                                         \
    int row = blockIdx.x;                                                   \
    if (row >= m) return;                                                   \
    const WEIGHT_T* w_row = weights + (size_t)row * k;                     \
    ACC_T acc = ACC_ZERO;                                                   \
    for (int j = threadIdx.x; j < k; j += 32) {                           \
        if (IS_ACTIVE(spikes[j])) {                                        \
            acc += READ_W(w_row[j]);                                       \
        }                                                                   \
    }                                                                       \
    acc = WARP_RED(acc);                                                   \
    if (threadIdx.x == 0) output[row] = WRITE_W(acc);                     \
}

// =========================================================================
// Gather block kernel macro (one block per output row, large k)
// =========================================================================

#define DEFINE_GATHER_BLOCK(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T,   \
                            READ_W, WRITE_W, WARP_RED, ACC_ZERO)            \
__global__ void _gather_block_kern##SUFFIX(                                 \
    const WEIGHT_T* __restrict__ weights,                                   \
    const SPIKE_T*  __restrict__ spikes,                                    \
    WEIGHT_T*       __restrict__ output,                                    \
    int m, int k                                                            \
) {                                                                         \
    extern __shared__ char _smem_bytes[];                                   \
    ACC_T* smem_red = reinterpret_cast<ACC_T*>(_smem_bytes);               \
    int row = blockIdx.x;                                                   \
    if (row >= m) return;                                                   \
    const WEIGHT_T* w_row = weights + (size_t)row * k;                     \
    ACC_T acc = ACC_ZERO;                                                   \
    for (int j = threadIdx.x; j < k; j += blockDim.x) {                   \
        if (IS_ACTIVE(spikes[j])) {                                        \
            acc += READ_W(w_row[j]);                                       \
        }                                                                   \
    }                                                                       \
    /* Block reduction: warp shuffle + shared memory */                    \
    int lane   = threadIdx.x & 31;                                         \
    int warpid = threadIdx.x >> 5;                                         \
    acc = WARP_RED(acc);                                                   \
    if (lane == 0) smem_red[warpid] = acc;                                 \
    __syncthreads();                                                        \
    int n_warps = (blockDim.x + 31) >> 5;                                  \
    acc = (threadIdx.x < n_warps) ? smem_red[lane] : ACC_ZERO;            \
    if (warpid == 0) acc = WARP_RED(acc);                                  \
    if (threadIdx.x == 0) output[row] = WRITE_W(acc);                     \
}

// =========================================================================
// Scatter kernel macro (transpose=True)
//
// out[j] = sum_{i where spikes[i] active} weights[i, j]
// weights: [k, n] row-major.  spikes: [k].  output: [n].
// =========================================================================

#define DEFINE_SCATTER(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T,        \
                       READ_W, WRITE_W, ACC_ZERO)                           \
__global__ void _scatter_kern##SUFFIX(                                      \
    const WEIGHT_T* __restrict__ weights,                                   \
    const SPIKE_T*  __restrict__ spikes,                                    \
    WEIGHT_T*       __restrict__ output,                                    \
    int k, int n                                                            \
) {                                                                         \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (j >= n) return;                                                    \
    ACC_T acc = ACC_ZERO;                                                   \
    for (int i = 0; i < k; i++) {                                          \
        if (IS_ACTIVE(spikes[i])) {                                        \
            acc += READ_W(weights[(size_t)i * n + j]);                     \
        }                                                                   \
    }                                                                       \
    output[j] = WRITE_W(acc);                                              \
}

// =========================================================================
// Instantiate device kernels: 4 weight dtypes x 2 spike types = 8 combos,
//                             3 kernel shapes each = 24 kernels
// =========================================================================

// ---- Float32 (identity conversion, accumulate in float32) ----
DEFINE_GATHER_WARP(_f32_bool,   int8_t, IS_ACTIVE_BOOL,  float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_GATHER_WARP(_f32_float,  float,  IS_ACTIVE_FLOAT, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_GATHER_BLOCK(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_GATHER_BLOCK(_f32_float, float,  IS_ACTIVE_FLOAT, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_SCATTER(_f32_bool,       int8_t, IS_ACTIVE_BOOL,  float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_SCATTER(_f32_float,      float,  IS_ACTIVE_FLOAT, float, float, READ_F32, WRITE_F32, 0.0f)

// ---- Float64 (identity conversion, accumulate in float64) ----
DEFINE_GATHER_WARP(_f64_bool,   int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_GATHER_WARP(_f64_float,  float,  IS_ACTIVE_FLOAT, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_GATHER_BLOCK(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_GATHER_BLOCK(_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_SCATTER(_f64_bool,       int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_SCATTER(_f64_float,      float,  IS_ACTIVE_FLOAT, double, double, READ_F64, WRITE_F64, 0.0)

// ---- Float16 (accumulate in float32 for stability) ----
DEFINE_GATHER_WARP(_f16_bool,   int8_t, IS_ACTIVE_BOOL,  __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_GATHER_WARP(_f16_float,  float,  IS_ACTIVE_FLOAT, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_GATHER_BLOCK(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_GATHER_BLOCK(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_SCATTER(_f16_bool,       int8_t, IS_ACTIVE_BOOL,  __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_SCATTER(_f16_float,      float,  IS_ACTIVE_FLOAT, __half, float, READ_F16, WRITE_F16, 0.0f)

// ---- BFloat16 (accumulate in float32 for stability; requires CUDA 11.0+) ----
DEFINE_GATHER_WARP(_bf16_bool,   int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_GATHER_WARP(_bf16_float,  float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_GATHER_BLOCK(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_GATHER_BLOCK(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_SCATTER(_bf16_bool,       int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_SCATTER(_bf16_float,      float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)


// =========================================================================
// TVM FFI Entry Point Macros
// =========================================================================
//
// Convention: args = (weights, spikes, output, stream)
//   Gather: weights [m, k], spikes [k], output [m]
//   Scatter: weights [k, n], spikes [k], output [n]
//
// IMPORTANT: data_ptr() returns GPU device pointers.
// NEVER dereference on the host. Pass to kernels unchanged.
//
// These macros generate the TVM FFI entry functions. The Python-side
// parser discovers them via  // @tvm_ffi  annotation comments placed
// before each macro invocation.
// =========================================================================

// ---- FFI macro: gather_warp ----
#define FFI_GATHER_WARP(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                    \
void binary_densemv_gather_warp##SUFFIX(                                   \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView spikes,            \
    tvm::ffi::TensorView output, int64_t stream                            \
) {                                                                         \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);               \
    int m = static_cast<int>(weights.size(0));                             \
    int k = static_cast<int>(weights.size(1));                             \
    _gather_warp_kern##SUFFIX<<<m, 32, 0, s>>>(                            \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                \
        static_cast<const SPIKE_C_T*>(spikes.data_ptr()),                  \
        static_cast<WEIGHT_C_T*>(output.data_ptr()), m, k);               \
}

// ---- FFI macro: gather_block ----
#define FFI_GATHER_BLOCK(SUFFIX, WEIGHT_C_T, SPIKE_C_T, SHM_SIZE)         \
void binary_densemv_gather_block##SUFFIX(                                  \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView spikes,            \
    tvm::ffi::TensorView output, int64_t stream                            \
) {                                                                         \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);               \
    int m = static_cast<int>(weights.size(0));                             \
    int k = static_cast<int>(weights.size(1));                             \
    _gather_block_kern##SUFFIX<<<m, 256, SHM_SIZE, s>>>(                   \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                \
        static_cast<const SPIKE_C_T*>(spikes.data_ptr()),                  \
        static_cast<WEIGHT_C_T*>(output.data_ptr()), m, k);               \
}

// ---- FFI macro: gather_auto (warp for k<=1024, block otherwise) ----
#define FFI_GATHER_AUTO(SUFFIX, WEIGHT_C_T, SPIKE_C_T, SHM_SIZE)          \
void binary_densemv_gather_auto##SUFFIX(                                   \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView spikes,            \
    tvm::ffi::TensorView output, int64_t stream                            \
) {                                                                         \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);               \
    int m = static_cast<int>(weights.size(0));                             \
    int k = static_cast<int>(weights.size(1));                             \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const SPIKE_C_T*  d_spk = static_cast<const SPIKE_C_T*>(spikes.data_ptr());   \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr()); \
    if (k <= 1024) {                                                        \
        _gather_warp_kern##SUFFIX<<<m, 32, 0, s>>>(d_w, d_spk, d_out, m, k); \
    } else {                                                                \
        _gather_block_kern##SUFFIX<<<m, 256, SHM_SIZE, s>>>(d_w, d_spk, d_out, m, k); \
    }                                                                       \
}

// ---- FFI macro: scatter ----
#define FFI_SCATTER(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                         \
void binary_densemv_scatter##SUFFIX(                                       \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView spikes,            \
    tvm::ffi::TensorView output, int64_t stream                            \
) {                                                                         \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);               \
    int k = static_cast<int>(weights.size(0));                             \
    int n = static_cast<int>(weights.size(1));                             \
    int blocks = (n + 255) / 256;                                          \
    _scatter_kern##SUFFIX<<<blocks, 256, 0, s>>>(                          \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                \
        static_cast<const SPIKE_C_T*>(spikes.data_ptr()),                  \
        static_cast<WEIGHT_C_T*>(output.data_ptr()), k, n);               \
}

// =========================================================================
// Instantiate TVM FFI entry points via macros + @tvm_ffi annotations
// =========================================================================

// ---- Float32 (shm: 32 * sizeof(float) = 128 bytes) ----
// @tvm_ffi binary_densemv_gather_warp_f32_bool
FFI_GATHER_WARP(_f32_bool,    float,   int8_t)
// @tvm_ffi binary_densemv_gather_warp_f32_float
FFI_GATHER_WARP(_f32_float,   float,   float)
// @tvm_ffi binary_densemv_gather_block_f32_bool
FFI_GATHER_BLOCK(_f32_bool,   float,   int8_t, 32 * sizeof(float))
// @tvm_ffi binary_densemv_gather_block_f32_float
FFI_GATHER_BLOCK(_f32_float,  float,   float,  32 * sizeof(float))
// @tvm_ffi binary_densemv_gather_auto_f32_bool
FFI_GATHER_AUTO(_f32_bool,    float,   int8_t, 32 * sizeof(float))
// @tvm_ffi binary_densemv_gather_auto_f32_float
FFI_GATHER_AUTO(_f32_float,   float,   float,  32 * sizeof(float))
// @tvm_ffi binary_densemv_scatter_f32_bool
FFI_SCATTER(_f32_bool,        float,   int8_t)
// @tvm_ffi binary_densemv_scatter_f32_float
FFI_SCATTER(_f32_float,       float,   float)

// ---- Float64 (shm: 32 * sizeof(double) = 256 bytes) ----
// @tvm_ffi binary_densemv_gather_auto_f64_bool
FFI_GATHER_AUTO(_f64_bool,    double,  int8_t, 32 * sizeof(double))
// @tvm_ffi binary_densemv_gather_auto_f64_float
FFI_GATHER_AUTO(_f64_float,   double,  float,  32 * sizeof(double))
// @tvm_ffi binary_densemv_scatter_f64_bool
FFI_SCATTER(_f64_bool,        double,  int8_t)
// @tvm_ffi binary_densemv_scatter_f64_float
FFI_SCATTER(_f64_float,       double,  float)

// ---- Float16 (shm: 32 * sizeof(float) = 128 bytes; accumulates in f32) ----
// @tvm_ffi binary_densemv_gather_auto_f16_bool
FFI_GATHER_AUTO(_f16_bool,    __half,  int8_t, 32 * sizeof(float))
// @tvm_ffi binary_densemv_gather_auto_f16_float
FFI_GATHER_AUTO(_f16_float,   __half,  float,  32 * sizeof(float))
// @tvm_ffi binary_densemv_scatter_f16_bool
FFI_SCATTER(_f16_bool,        __half,  int8_t)
// @tvm_ffi binary_densemv_scatter_f16_float
FFI_SCATTER(_f16_float,       __half,  float)

// ---- BFloat16 (shm: 32 * sizeof(float) = 128 bytes; accumulates in f32) ----
// @tvm_ffi binary_densemv_gather_auto_bf16_bool
FFI_GATHER_AUTO(_bf16_bool,   __nv_bfloat16, int8_t, 32 * sizeof(float))
// @tvm_ffi binary_densemv_gather_auto_bf16_float
FFI_GATHER_AUTO(_bf16_float,  __nv_bfloat16, float,  32 * sizeof(float))
// @tvm_ffi binary_densemv_scatter_bf16_bool
FFI_SCATTER(_bf16_bool,       __nv_bfloat16, int8_t)
// @tvm_ffi binary_densemv_scatter_bf16_float
FFI_SCATTER(_bf16_float,      __nv_bfloat16, float)
