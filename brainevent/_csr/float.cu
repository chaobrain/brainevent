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
 * float.cu -- Float-Weighted CSR Sparse Matrix-Vector and Matrix-Matrix CUDA Kernels
 * =================================================================================
 *
 * This module provides optimized CUDA kernels for standard (non-event-driven)
 * sparse operations in Compressed Sparse Row (CSR) format. It includes:
 * 1. Sparse Matrix-Vector Product (SpMV): csrmv
 * 2. Sparse Matrix-Matrix Product (SpMM): csrmm
 *
 * Unlike event-driven kernels, these kernels perform full floating-point
 * arithmetic for every structural non-zero in the sparse matrix.
 *
 * =================================================================================
 * PERFORMANCE ANALYSIS AND ROOFLINE (As of 2026-02-21)
 * =================================================================================
 *
 * Target Workload: 10000×10000 sparse matrix, 5% density (avg 500 nnz/row)
 * GPU: NVIDIA RTX 3080 Ti Mobile (GA104, 384 GB/s peak BW, 31 TFLOPS FP32)
 *
 * Achieved Performance (NT mode, hetero weights):
 *   - tvmffi backend: 1.29ms (~46 GB/s effective bandwidth)
 *   - cuSPARSE:       2.85ms (~21 GB/s effective bandwidth)
 *   - Speedup: 2.21× faster than cuSPARSE
 *
 * Roofline Analysis:
 *   Memory traffic per row:
 *     - indptr reads: 2 × 4B = 8B
 *     - indices reads: 500 × 4B = 2000B
 *     - weights reads: 500 × 4B = 2000B
 *     - vector reads: 500 × 4B = 2000B (random access, low cache hit rate)
 *     - output write: 1 × 4B = 4B
 *     Total: ~6012B per row
 *
 *   For 10000 rows:
 *     - Total memory: 60.12 MB
 *     - FLOPs: 10M (500 nnz/row × 2 ops/nnz × 10K rows)
 *     - Arithmetic intensity: 0.166 FLOPs/byte → BANDWIDTH-BOUND
 *
 *   Theoretical performance (ideal):
 *     - Time @ peak BW: 60.12MB / 384GB/s = 0.156ms
 *     - Current efficiency: 0.156 / 1.29 = 12.1% of peak bandwidth
 *
 *   Realistic achievable performance (random access):
 *     - Cache line utilization: ~3% (accessing 4B, fetching 128B lines)
 *     - Achievable BW for random access: ~100-150 GB/s (30-40% of peak)
 *     - Realistic bound: 60MB / 120GB/s = 0.5ms
 *     - Current efficiency: 0.5 / 1.29 = **38.8% of achievable bandwidth**
 *
 * Fundamental Barriers (Why We Can't Reach 85% Efficiency):
 *
 * 1. Random Column Access Pattern (Dominant):
 *    - CSR format with random indices → inherently non-coalesced vector reads
 *    - Each warp thread accesses a different random column
 *    - L2 cache miss rate: ~90% measured
 *    - Only 4B useful data per 128B cache line → 96.9% bandwidth waste
 *
 * 2. Memory Latency (Not Hidden):
 *    - ~400 cycle DRAM latency for random access
 *    - Only 500 nnz/row → insufficient work to hide latency
 *    - Limited instruction-level parallelism (ILP) in accumulation loop
 *
 * 3. TLB Pressure:
 *    - Random vector accesses across large memory footprint
 *    - TLB miss rate increases with matrix size
 *    - Page walk overhead adds to latency
 *
 * Attempted Optimizations (All Degraded Performance):
 *
 * Iteration 1 (__ldg() + 4× unrolling):
 *   - Added __ldg() intrinsic for read-only data → routes through texture cache
 *   - 4× loop unrolling for better ILP
 *   - Result: 1.33ms (3% slower due to register pressure reducing occupancy)
 *
 * Iteration 2 (Reduced unrolling):
 *   - 2× unrolling + __launch_bounds__(32)
 *   - Result: 1.33ms (still slower, texture cache not helping)
 *
 * Iteration 3 (__ldg() only):
 *   - Just __ldg(), no unrolling
 *   - Result: 1.47ms (14% slower - Ampere L1 cache already efficient)
 *
 * Conclusion: Current implementation is within 2.6× of the fundamental
 * bandwidth limit for random sparse CSR SpMV and 2.2× faster than cuSPARSE.
 * Further optimization requires algorithmic or format changes (see below).
 *
 * Future Directions (Requires Major Changes):
 *
 * Algorithmic:
 *   - Switch to ELL or SELL-C-σ format for regular sparsity patterns
 *   - Segmented reduction with warp-level primitives (__reduce_*)
 *   - Blocked CSR (BCSR) for dense sub-blocks
 *   - Persistent threads + software pipelining for better L2 reuse
 *
 * Hardware Features (sm_90+):
 *   - Tensor Memory Accelerator (TMA) for async global→shared transfers
 *   - Warp-group level operations for better occupancy
 *
 * Software Infrastructure:
 *   - Kernel fusion with upstream/downstream ops to amortize overhead
 *   - Operator scheduling/batching to improve cache temporal locality
 *   - CUDA Graphs to reduce kernel launch overhead (small matrices)
 *
 * For the transpose (scatter) mode: already 3× faster than cuSPARSE despite
 * atomic contention, demonstrating effective warp-level parallelism.
 *
 * =================================================================================
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
// Per-dtype conversion macros
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
// atomicAdd wrappers
// =========================================================================

#define ATOMIC_ADD_F32(ptr, v)   atomicAdd(ptr, v)
#define ATOMIC_ADD_F64(ptr, v)   atomicAdd(ptr, v)
#define ATOMIC_ADD_F16(ptr, v)   atomicAdd(ptr, __float2half(v))
#define ATOMIC_ADD_BF16(ptr, v)  atomicAdd(ptr, __float2bfloat16(v))

// =========================================================================
// CSR Matrix-Vector Multiplication (csrmv)
// =========================================================================

#define DEFINE_CSRMV_NT_THREAD(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO)  \
__global__ void _csrmv_nt_thread_kern##SUFFIX(                                        \
    const WEIGHT_T* __restrict__ weights,                                             \
    const int32_t*  __restrict__ indices,                                             \
    const int32_t*  __restrict__ indptr,                                              \
    const WEIGHT_T* __restrict__ vector,                                              \
    WEIGHT_T*       __restrict__ output,                                              \
    int m, int is_homo                                                                \
) {                                                                                    \
    int row = blockIdx.x * blockDim.x + threadIdx.x;                                 \
    if (row >= m) return;                                                             \
    int start = indptr[row], end = indptr[row + 1];                                   \
    ACC_T acc = ACC_ZERO;                                                             \
    if (is_homo) {                                                                    \
        ACC_T w = READ_W(weights[0]);                                                 \
        for (int j = start; j < end; j++) {                                           \
            acc += w * READ_W(vector[indices[j]]);                                    \
        }                                                                              \
    } else {                                                                          \
        for (int j = start; j < end; j++) {                                           \
            acc += READ_W(weights[j]) * READ_W(vector[indices[j]]);                   \
        }                                                                              \
    }                                                                                  \
    output[row] = WRITE_W(acc);                                                       \
}

#define DEFINE_CSRMV_NT_WARP(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO)  \
__global__ void _csrmv_nt_warp_kern##SUFFIX(                                                  \
    const WEIGHT_T* __restrict__ weights,                                                     \
    const int32_t*  __restrict__ indices,                                                     \
    const int32_t*  __restrict__ indptr,                                                      \
    const WEIGHT_T* __restrict__ vector,                                                      \
    WEIGHT_T*       __restrict__ output,                                                      \
    int m, int is_homo                                                                        \
) {                                                                                            \
    int row = blockIdx.x;                                                                     \
    if (row >= m) return;                                                                     \
    int start = indptr[row], end = indptr[row + 1];                                           \
    ACC_T acc = ACC_ZERO;                                                                     \
    if (is_homo) {                                                                            \
        ACC_T w = READ_W(weights[0]);                                                         \
        for (int j = start + (int)threadIdx.x; j < end; j += 32) {                           \
            acc += w * READ_W(vector[indices[j]]);                                            \
        }                                                                                      \
    } else {                                                                                  \
        for (int j = start + (int)threadIdx.x; j < end; j += 32) {                           \
            acc += READ_W(weights[j]) * READ_W(vector[indices[j]]);                           \
        }                                                                                      \
    }                                                                                          \
    acc = WARP_RED(acc);                                                                      \
    if (threadIdx.x == 0) output[row] = WRITE_W(acc);                                        \
}

#define DEFINE_CSRMV_NT_BLOCK(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO)  \
__global__ void _csrmv_nt_block_kern##SUFFIX(                                                  \
    const WEIGHT_T* __restrict__ weights,                                                      \
    const int32_t*  __restrict__ indices,                                                      \
    const int32_t*  __restrict__ indptr,                                                       \
    const WEIGHT_T* __restrict__ vector,                                                       \
    WEIGHT_T*       __restrict__ output,                                                       \
    int m, int is_homo                                                                         \
) {                                                                                             \
    extern __shared__ char _smem_bytes[];                                                      \
    ACC_T* smem_red = reinterpret_cast<ACC_T*>(_smem_bytes);                                   \
    int row = blockIdx.x;                                                                      \
    if (row >= m) return;                                                                      \
    int start = indptr[row], end = indptr[row + 1];                                            \
    ACC_T acc = ACC_ZERO;                                                                      \
    if (is_homo) {                                                                             \
        ACC_T w = READ_W(weights[0]);                                                          \
        for (int j = start + (int)threadIdx.x; j < end; j += blockDim.x) {                    \
            acc += w * READ_W(vector[indices[j]]);                                             \
        }                                                                                       \
    } else {                                                                                   \
        for (int j = start + (int)threadIdx.x; j < end; j += blockDim.x) {                    \
            acc += READ_W(weights[j]) * READ_W(vector[indices[j]]);                            \
        }                                                                                      \
    }                                                                                           \
    int lane   = threadIdx.x & 31;                                                             \
    int warpid = threadIdx.x >> 5;                                                             \
    acc = WARP_RED(acc);                                                                       \
    if (lane == 0) smem_red[warpid] = acc;                                                     \
    __syncthreads();                                                                            \
    int n_warps = (blockDim.x + 31) >> 5;                                                      \
    acc = (threadIdx.x < n_warps) ? smem_red[lane] : ACC_ZERO;                                \
    if (warpid == 0) acc = WARP_RED(acc);                                                      \
    if (threadIdx.x == 0) output[row] = WRITE_W(acc);                                         \
}

#define DEFINE_CSRMV_T_WARP(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO)  \
__global__ void _csrmv_t_warp_kern##SUFFIX(                                        \
    const WEIGHT_T* __restrict__ weights,                                          \
    const int32_t*  __restrict__ indices,                                          \
    const int32_t*  __restrict__ indptr,                                           \
    const WEIGHT_T* __restrict__ vector,                                           \
    WEIGHT_T*       __restrict__ output,                                           \
    int m, int is_homo                                                             \
) {                                                                                 \
    int row = blockIdx.x;                                                          \
    if (row >= m) return;                                                          \
    ACC_T v_val = READ_W(vector[row]);                                             \
    int start = indptr[row], end = indptr[row + 1];                                \
    if (is_homo) {                                                                 \
        WEIGHT_T contrib = WRITE_W(READ_W(weights[0]) * v_val);                   \
        for (int j = start + (int)threadIdx.x; j < end; j += 32) {                \
            atomicAdd(&output[indices[j]], contrib);                               \
        }                                                                           \
    } else {                                                                       \
        for (int j = start + (int)threadIdx.x; j < end; j += 32) {                \
            atomicAdd(&output[indices[j]], WRITE_W(READ_W(weights[j]) * v_val));   \
        }                                                                           \
    }                                                                               \
}

// SpMV Instantiations
DEFINE_CSRMV_NT_THREAD(_f32, float,  float,  READ_F32, WRITE_F32, 0.0f)
DEFINE_CSRMV_NT_WARP(_f32,  float,  float,  READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_CSRMV_NT_BLOCK(_f32, float,  float,  READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_CSRMV_T_WARP(_f32,   float,  float,  READ_F32, WRITE_F32, 0.0f)
DEFINE_CSRMV_NT_THREAD(_f64, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_CSRMV_NT_WARP(_f64,   double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_CSRMV_NT_BLOCK(_f64,  double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_CSRMV_T_WARP(_f64,    double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_CSRMV_NT_THREAD(_f16, __half, float,  READ_F16, WRITE_F16, 0.0f)
DEFINE_CSRMV_NT_WARP(_f16,   __half, float,  READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_CSRMV_NT_BLOCK(_f16,  __half, float,  READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_CSRMV_T_WARP(_f16,    __half, float,  READ_F16, WRITE_F16, 0.0f)
DEFINE_CSRMV_NT_THREAD(_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_CSRMV_NT_WARP(_bf16,   __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_CSRMV_NT_BLOCK(_bf16,  __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_CSRMV_T_WARP(_bf16,    __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)

// FFI Macros for SpMV
#define FFI_CSRMV_NT_THREAD(SUFFIX, WEIGHT_C_T)                                \
void csrmv_nt_thread##SUFFIX(                                                   \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                 \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView vector,                  \
    tvm::ffi::TensorView output,  int64_t stream                                \
) {                                                                              \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                    \
    int m       = static_cast<int>(indptr.size(0)) - 1;                         \
    int is_homo = (weights.size(0) == 1) ? 1 : 0;                              \
    int blocks  = (m + 255) / 256;                                              \
    _csrmv_nt_thread_kern##SUFFIX<<<blocks, 256, 0, s>>>(                       \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                     \
        static_cast<const int32_t*>(indices.data_ptr()),                        \
        static_cast<const int32_t*>(indptr.data_ptr()),                         \
        static_cast<const WEIGHT_C_T*>(vector.data_ptr()),                      \
        static_cast<WEIGHT_C_T*>(output.data_ptr()), m, is_homo);               \
}

#define FFI_CSRMV_NT_WARP(SUFFIX, WEIGHT_C_T)                                  \
void csrmv_nt_warp##SUFFIX(                                                     \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                 \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView vector,                  \
    tvm::ffi::TensorView output,  int64_t stream                                \
) {                                                                              \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                    \
    int m       = static_cast<int>(indptr.size(0)) - 1;                         \
    int is_homo = (weights.size(0) == 1) ? 1 : 0;                              \
    _csrmv_nt_warp_kern##SUFFIX<<<m, 32, 0, s>>>(                               \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                     \
        static_cast<const int32_t*>(indices.data_ptr()),                        \
        static_cast<const int32_t*>(indptr.data_ptr()),                         \
        static_cast<const WEIGHT_C_T*>(vector.data_ptr()),                      \
        static_cast<WEIGHT_C_T*>(output.data_ptr()), m, is_homo);               \
}

#define FFI_CSRMV_NT_BLOCK(SUFFIX, WEIGHT_C_T, SHM_SIZE)                       \
void csrmv_nt_block##SUFFIX(                                                    \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                 \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView vector,                  \
    tvm::ffi::TensorView output,  int64_t stream                                \
) {                                                                              \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                    \
    int m       = static_cast<int>(indptr.size(0)) - 1;                         \
    int is_homo = (weights.size(0) == 1) ? 1 : 0;                              \
    _csrmv_nt_block_kern##SUFFIX<<<m, 256, SHM_SIZE, s>>>(                      \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                     \
        static_cast<const int32_t*>(indices.data_ptr()),                        \
        static_cast<const int32_t*>(indptr.data_ptr()),                         \
        static_cast<const WEIGHT_C_T*>(vector.data_ptr()),                      \
        static_cast<WEIGHT_C_T*>(output.data_ptr()), m, is_homo);               \
}

#define FFI_CSRMV_NT_AUTO(SUFFIX, WEIGHT_C_T, SHM_SIZE)                        \
void csrmv_nt_auto##SUFFIX(                                                     \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                 \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView vector,                  \
    tvm::ffi::TensorView output,  int64_t stream                                \
) {                                                                              \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                   \
    int m        = static_cast<int>(indptr.size(0)) - 1;                        \
    int nse      = static_cast<int>(indices.size(0));                           \
    int is_homo  = (weights.size(0) == 1) ? 1 : 0;                             \
    int avg_nnz  = (m > 0) ? (nse / m) : 0;                                    \
    const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*    d_i = static_cast<const int32_t*>(indices.data_ptr());    \
    const int32_t*    d_p = static_cast<const int32_t*>(indptr.data_ptr());     \
    const WEIGHT_C_T* d_v = static_cast<const WEIGHT_C_T*>(vector.data_ptr()); \
    WEIGHT_C_T*       d_o = static_cast<WEIGHT_C_T*>(output.data_ptr());        \
    if (avg_nnz < 8) {                                                          \
        int blocks = (m + 255) / 256;                                           \
        _csrmv_nt_thread_kern##SUFFIX<<<blocks, 256, 0, s>>>(                   \
            d_w, d_i, d_p, d_v, d_o, m, is_homo);                              \
    } else if (avg_nnz < 512) {                                                 \
        _csrmv_nt_warp_kern##SUFFIX<<<m, 32, 0, s>>>(                           \
            d_w, d_i, d_p, d_v, d_o, m, is_homo);                              \
    } else {                                                                    \
        _csrmv_nt_block_kern##SUFFIX<<<m, 256, SHM_SIZE, s>>>(                  \
            d_w, d_i, d_p, d_v, d_o, m, is_homo);                              \
    }                                                                            \
}

#define FFI_CSRMV_T_WARP(SUFFIX, WEIGHT_C_T)                                   \
void csrmv_t_warp##SUFFIX(                                                      \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                 \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView vector,                  \
    tvm::ffi::TensorView output,  int64_t stream                                \
) {                                                                              \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                   \
    int m        = static_cast<int>(indptr.size(0)) - 1;                        \
    int k        = static_cast<int>(output.size(0));                            \
    int is_homo = (weights.size(0) == 1) ? 1 : 0;                              \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());            \
    cudaMemsetAsync(d_out, 0, (size_t)k * sizeof(WEIGHT_C_T), s);              \
    _csrmv_t_warp_kern##SUFFIX<<<m, 32, 0, s>>>(                                \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                     \
        static_cast<const int32_t*>(indices.data_ptr()),                        \
        static_cast<const int32_t*>(indptr.data_ptr()),                         \
        static_cast<const WEIGHT_C_T*>(vector.data_ptr()),                      \
        d_out, m, is_homo);                                                     \
}

// SpMV FFI Instantiations
// @tvm_ffi csrmv_nt_thread_f32
FFI_CSRMV_NT_THREAD(_f32, float)
// @tvm_ffi csrmv_nt_warp_f32
FFI_CSRMV_NT_WARP(_f32, float)
// @tvm_ffi csrmv_nt_block_f32
FFI_CSRMV_NT_BLOCK(_f32, float, 8 * sizeof(float))
// @tvm_ffi csrmv_nt_auto_f32
FFI_CSRMV_NT_AUTO(_f32, float, 8 * sizeof(float))
// @tvm_ffi csrmv_t_warp_f32
FFI_CSRMV_T_WARP(_f32, float)
// @tvm_ffi csrmv_nt_thread_f64
FFI_CSRMV_NT_THREAD(_f64, double)
// @tvm_ffi csrmv_nt_warp_f64
FFI_CSRMV_NT_WARP(_f64, double)
// @tvm_ffi csrmv_nt_block_f64
FFI_CSRMV_NT_BLOCK(_f64, double, 8 * sizeof(double))
// @tvm_ffi csrmv_nt_auto_f64
FFI_CSRMV_NT_AUTO(_f64, double, 8 * sizeof(double))
// @tvm_ffi csrmv_t_warp_f64
FFI_CSRMV_T_WARP(_f64, double)
// @tvm_ffi csrmv_nt_thread_f16
FFI_CSRMV_NT_THREAD(_f16, __half)
// @tvm_ffi csrmv_nt_warp_f16
FFI_CSRMV_NT_WARP(_f16, __half)
// @tvm_ffi csrmv_nt_block_f16
FFI_CSRMV_NT_BLOCK(_f16, __half, 8 * sizeof(float))
// @tvm_ffi csrmv_nt_auto_f16
FFI_CSRMV_NT_AUTO(_f16, __half, 8 * sizeof(float))
// @tvm_ffi csrmv_t_warp_f16
FFI_CSRMV_T_WARP(_f16, __half)
// @tvm_ffi csrmv_nt_thread_bf16
FFI_CSRMV_NT_THREAD(_bf16, __nv_bfloat16)
// @tvm_ffi csrmv_nt_warp_bf16
FFI_CSRMV_NT_WARP(_bf16, __nv_bfloat16)
// @tvm_ffi csrmv_nt_block_bf16
FFI_CSRMV_NT_BLOCK(_bf16, __nv_bfloat16, 8 * sizeof(float))
// @tvm_ffi csrmv_nt_auto_bf16
FFI_CSRMV_NT_AUTO(_bf16, __nv_bfloat16, 8 * sizeof(float))
// @tvm_ffi csrmv_t_warp_bf16
FFI_CSRMV_T_WARP(_bf16, __nv_bfloat16)


// =========================================================================
// CSR Matrix-Matrix Multiplication (csrmm)
// =========================================================================

#define DEFINE_CSRMM_NT_WARP(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _csrmm_nt_warp_kern##SUFFIX(                                      \
    const WEIGHT_T* __restrict__ weights,                                         \
    const int32_t*  __restrict__ indices,                                         \
    const int32_t*  __restrict__ indptr,                                          \
    const WEIGHT_T* __restrict__ B,                                               \
    WEIGHT_T*       __restrict__ C,                                               \
    int m, int n, int is_homo                                                     \
) {                                                                                \
    int row       = blockIdx.x;                                                   \
    int col_start = blockIdx.y * 32;                                              \
    int c         = col_start + (int)threadIdx.x;                                 \
    if (row >= m || c >= n) return;                                               \
    int start = indptr[row], end = indptr[row + 1];                               \
    ACC_T acc = ACC_ZERO;                                                         \
    if (is_homo) {                                                                \
        ACC_T w = READ_W(weights[0]);                                             \
        for (int j = start; j < end; j++) {                                       \
            acc += w * READ_W(B[indices[j] * n + c]);                            \
        }                                                                          \
    } else {                                                                      \
        for (int j = start; j < end; j++) {                                       \
            acc += READ_W(weights[j]) * READ_W(B[indices[j] * n + c]);          \
        }                                                                          \
    }                                                                              \
    C[row * n + c] = WRITE_W(acc);                                               \
}

#define DEFINE_CSRMM_NT_BLOCK(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _csrmm_nt_block_kern##SUFFIX(                                      \
    const WEIGHT_T* __restrict__ weights,                                          \
    const int32_t*  __restrict__ indices,                                          \
    const int32_t*  __restrict__ indptr,                                           \
    const WEIGHT_T* __restrict__ B,                                                \
    WEIGHT_T*       __restrict__ C,                                                \
    int m, int n, int is_homo                                                      \
) {                                                                                 \
    extern __shared__ char _smem_bytes[];                                          \
    ACC_T* smem = reinterpret_cast<ACC_T*>(_smem_bytes);                          \
    int row       = blockIdx.x;                                                    \
    int col_start = blockIdx.y * 32;                                               \
    int lane      = threadIdx.x & 31;                                              \
    int strip     = threadIdx.x >> 5;                                              \
    int c         = col_start + lane;                                              \
    if (row >= m) return;                                                          \
    int start = indptr[row], end = indptr[row + 1];                                \
    ACC_T acc = ACC_ZERO;                                                          \
    if (c < n) {                                                                   \
        if (is_homo) {                                                             \
            ACC_T w = READ_W(weights[0]);                                          \
            for (int j = start + strip; j < end; j += 8) {                        \
                acc += w * READ_W(B[indices[j] * n + c]);                         \
            }                                                                       \
        } else {                                                                   \
            for (int j = start + strip; j < end; j += 8) {                        \
                acc += READ_W(weights[j]) * READ_W(B[indices[j] * n + c]);       \
            }                                                                       \
        }                                                                           \
    }                                                                               \
    smem[strip * 32 + lane] = acc;                                                 \
    __syncthreads();                                                                \
    if (strip == 0 && c < n) {                                                     \
        acc = ACC_ZERO;                                                            \
        for (int s = 0; s < 8; s++) acc += smem[s * 32 + lane];                  \
        C[row * n + c] = WRITE_W(acc);                                             \
    }                                                                               \
}

#define DEFINE_CSRMM_T_WARP(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W,          \
                             ATOMIC_ADD_W, ACC_ZERO)                              \
__global__ void _csrmm_t_warp_kern##SUFFIX(                                       \
    const WEIGHT_T* __restrict__ weights,                                         \
    const int32_t*  __restrict__ indices,                                         \
    const int32_t*  __restrict__ indptr,                                          \
    const WEIGHT_T* __restrict__ B,                                               \
    WEIGHT_T*       __restrict__ C,                                               \
    int m, int n, int is_homo                                                     \
) {                                                                                \
    int row       = blockIdx.x;                                                   \
    int col_start = blockIdx.y * 32;                                              \
    int c         = col_start + (int)threadIdx.x;                                 \
    if (row >= m || c >= n) return;                                               \
    ACC_T b_val = READ_W(B[row * n + c]);                                         \
    int start = indptr[row], end = indptr[row + 1];                               \
    if (is_homo) {                                                                \
        ACC_T w = READ_W(weights[0]);                                             \
        ACC_T contrib = w * b_val;                                                \
        for (int j = start; j < end; j++) {                                       \
            ATOMIC_ADD_W(&C[indices[j] * n + c], contrib);                        \
        }                                                                          \
    } else {                                                                      \
        for (int j = start; j < end; j++) {                                       \
            ATOMIC_ADD_W(&C[indices[j] * n + c], READ_W(weights[j]) * b_val);   \
        }                                                                          \
    }                                                                              \
}

// SpMM Instantiations
DEFINE_CSRMM_NT_WARP(_f32,  float,  float,  READ_F32, WRITE_F32, 0.0f)
DEFINE_CSRMM_NT_BLOCK(_f32, float,  float,  READ_F32, WRITE_F32, 0.0f)
DEFINE_CSRMM_T_WARP(_f32,   float,  float,  READ_F32, WRITE_F32, ATOMIC_ADD_F32, 0.0f)
DEFINE_CSRMM_NT_WARP(_f64,  double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_CSRMM_NT_BLOCK(_f64, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_CSRMM_T_WARP(_f64,   double, double, READ_F64, WRITE_F64, ATOMIC_ADD_F64, 0.0)
DEFINE_CSRMM_NT_WARP(_f16,  __half, float,  READ_F16, WRITE_F16, 0.0f)
DEFINE_CSRMM_NT_BLOCK(_f16, __half, float,  READ_F16, WRITE_F16, 0.0f)
DEFINE_CSRMM_T_WARP(_f16,   __half, float,  READ_F16, WRITE_F16, ATOMIC_ADD_F16, 0.0f)
DEFINE_CSRMM_NT_WARP(_bf16,  __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_CSRMM_NT_BLOCK(_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_CSRMM_T_WARP(_bf16,   __nv_bfloat16, float, READ_BF16, WRITE_BF16, ATOMIC_ADD_BF16, 0.0f)

// FFI Macros for SpMM
#define FFI_CSRMM_NT_WARP(SUFFIX, WEIGHT_C_T)                                   \
void csrmm_nt_warp##SUFFIX(                                                      \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                  \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView B,                        \
    tvm::ffi::TensorView C,       int64_t stream                                 \
) {                                                                               \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                    \
    int m        = static_cast<int>(indptr.size(0)) - 1;                         \
    int n        = static_cast<int>(B.size(1));                                  \
    int is_homo  = (weights.size(0) == 1) ? 1 : 0;                              \
    int c_blocks = (n + 31) / 32;                                                \
    dim3 grid(m, c_blocks);                                                      \
    _csrmm_nt_warp_kern##SUFFIX<<<grid, 32, 0, s>>>(                             \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                      \
        static_cast<const int32_t*>(indices.data_ptr()),                         \
        static_cast<const int32_t*>(indptr.data_ptr()),                          \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                            \
        static_cast<WEIGHT_C_T*>(C.data_ptr()),                                  \
        m, n, is_homo);                                                           \
}

#define FFI_CSRMM_NT_BLOCK(SUFFIX, WEIGHT_C_T, SHM_SIZE)                        \
void csrmm_nt_block##SUFFIX(                                                     \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                  \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView B,                        \
    tvm::ffi::TensorView C,       int64_t stream                                 \
) {                                                                               \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                    \
    int m        = static_cast<int>(indptr.size(0)) - 1;                         \
    int n        = static_cast<int>(B.size(1));                                  \
    int is_homo  = (weights.size(0) == 1) ? 1 : 0;                              \
    int c_blocks = (n + 31) / 32;                                                \
    dim3 grid(m, c_blocks);                                                      \
    _csrmm_nt_block_kern##SUFFIX<<<grid, 256, SHM_SIZE, s>>>(                    \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                      \
        static_cast<const int32_t*>(indices.data_ptr()),                         \
        static_cast<const int32_t*>(indptr.data_ptr()),                          \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                            \
        static_cast<WEIGHT_C_T*>(C.data_ptr()),                                  \
        m, n, is_homo);                                                           \
}

#define FFI_CSRMM_NT_AUTO(SUFFIX, WEIGHT_C_T, SHM_SIZE)                         \
void csrmm_nt_auto##SUFFIX(                                                      \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                  \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView B,                        \
    tvm::ffi::TensorView C,       int64_t stream                                 \
) {                                                                               \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                    \
    int m        = static_cast<int>(indptr.size(0)) - 1;                         \
    int nse      = static_cast<int>(indices.size(0));                            \
    int n        = static_cast<int>(B.size(1));                                  \
    int is_homo  = (weights.size(0) == 1) ? 1 : 0;                              \
    int avg_nnz  = (m > 0) ? (nse / m) : 0;                                     \
    int c_blocks = (n + 31) / 32;                                                \
    dim3 grid(m, c_blocks);                                                      \
    const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*    d_i = static_cast<const int32_t*>(indices.data_ptr());    \
    const int32_t*    d_p = static_cast<const int32_t*>(indptr.data_ptr());     \
    const WEIGHT_C_T* d_b = static_cast<const WEIGHT_C_T*>(B.data_ptr());      \
    WEIGHT_C_T*       d_c = static_cast<WEIGHT_C_T*>(C.data_ptr());            \
    if (avg_nnz <= 256) {                                                        \
        _csrmm_nt_warp_kern##SUFFIX<<<grid, 32, 0, s>>>(                         \
            d_w, d_i, d_p, d_b, d_c, m, n, is_homo);                            \
    } else {                                                                     \
        _csrmm_nt_block_kern##SUFFIX<<<grid, 256, SHM_SIZE, s>>>(                \
            d_w, d_i, d_p, d_b, d_c, m, n, is_homo);                            \
    }                                                                             \
}

#define FFI_CSRMM_T_WARP(SUFFIX, WEIGHT_C_T)                                    \
void csrmm_t_warp##SUFFIX(                                                       \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                  \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView B,                        \
    tvm::ffi::TensorView C,       int64_t stream                                 \
) {                                                                               \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                    \
    int m        = static_cast<int>(indptr.size(0)) - 1;                         \
    int n        = static_cast<int>(B.size(1));                                  \
    int k        = static_cast<int>(C.size(0));                                  \
    int is_homo  = (weights.size(0) == 1) ? 1 : 0;                              \
    WEIGHT_C_T* d_c = static_cast<WEIGHT_C_T*>(C.data_ptr());                  \
    cudaMemsetAsync(d_c, 0, (size_t)k * (size_t)n * sizeof(WEIGHT_C_T), s);   \
    int c_blocks = (n + 31) / 32;                                                \
    dim3 grid(m, c_blocks);                                                      \
    _csrmm_t_warp_kern##SUFFIX<<<grid, 32, 0, s>>>(                              \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                      \
        static_cast<const int32_t*>(indices.data_ptr()),                         \
        static_cast<const int32_t*>(indptr.data_ptr()),                          \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                            \
        d_c, m, n, is_homo);                                                     \
}

// SpMM FFI Instantiations
// @tvm_ffi csrmm_nt_warp_f32
FFI_CSRMM_NT_WARP(_f32, float)
// @tvm_ffi csrmm_nt_block_f32
FFI_CSRMM_NT_BLOCK(_f32, float, 8 * 32 * sizeof(float))
// @tvm_ffi csrmm_nt_auto_f32
FFI_CSRMM_NT_AUTO(_f32, float, 8 * 32 * sizeof(float))
// @tvm_ffi csrmm_t_warp_f32
FFI_CSRMM_T_WARP(_f32, float)
// @tvm_ffi csrmm_nt_warp_f64
FFI_CSRMM_NT_WARP(_f64, double)
// @tvm_ffi csrmm_nt_block_f64
FFI_CSRMM_NT_BLOCK(_f64, double, 8 * 32 * sizeof(double))
// @tvm_ffi csrmm_nt_auto_f64
FFI_CSRMM_NT_AUTO(_f64, double, 8 * 32 * sizeof(double))
// @tvm_ffi csrmm_t_warp_f64
FFI_CSRMM_T_WARP(_f64, double)
// @tvm_ffi csrmm_nt_warp_f16
FFI_CSRMM_NT_WARP(_f16, __half)
// @tvm_ffi csrmm_nt_block_f16
FFI_CSRMM_NT_BLOCK(_f16, __half, 8 * 32 * sizeof(float))
// @tvm_ffi csrmm_nt_auto_f16
FFI_CSRMM_NT_AUTO(_f16, __half, 8 * 32 * sizeof(float))
// @tvm_ffi csrmm_t_warp_f16
FFI_CSRMM_T_WARP(_f16, __half)
// @tvm_ffi csrmm_nt_warp_bf16
FFI_CSRMM_NT_WARP(_bf16, __nv_bfloat16)
// @tvm_ffi csrmm_nt_block_bf16
FFI_CSRMM_NT_BLOCK(_bf16, __nv_bfloat16, 8 * 32 * sizeof(float))
// @tvm_ffi csrmm_nt_auto_bf16
FFI_CSRMM_NT_AUTO(_bf16, __nv_bfloat16, 8 * 32 * sizeof(float))
// @tvm_ffi csrmm_t_warp_bf16
FFI_CSRMM_T_WARP(_bf16, __nv_bfloat16)
