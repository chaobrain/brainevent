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
 * binary.cu -- Event-Driven Binary COO Sparse Matrix-Vector and Matrix-Matrix CUDA Kernels
 * =======================================================================================
 *
 * This module provides high-performance, event-driven CUDA kernels for sparse
 * matrix-vector (SpMV) and sparse matrix-matrix (SpMM) multiplications where the
 * sparse matrix is in Coordinate (COO) format and the dense operand (vector or
 * matrix) contains binary events (spikes).
 *
 * Event-Driven Optimization:
 * -------------------------
 * In SNN simulations, dense vectors/matrices are often very sparse in time
 * (most elements are zero/inactive). These kernels exploit this by checking
 * the activity of the dense operand before performing expensive atomic
 * accumulations. This "event-driven" approach significantly reduces memory
 * traffic and contention on the output buffer.
 *
 * Supported Operations:
 * --------------------
 * 1. binary_coomv (SpMV): out = A @ v  or  out = A.T @ v
 *    - Uses a grid-stride loop with atomic additions.
 *    - Optimized for various data types (f32, f64, f16, bf16).
 *
 * 2. binary_coomm (SpMM): out = A @ B  or  out = A.T @ B
 *    - Column-Tiled (CT) Variant: Optimized for small number of columns (n <= 64).
 *      Uses warp-level voting (__ballot_sync) to skip entire tiles of inactive events.
 *    - Warp-Per-Entry (WPE) Variant: Optimized for large number of columns (n > 64).
 *      Assigns one warp per NNZ entry to maximize parallelism.
 *
 * Data Types and Numerical Stability:
 * ----------------------------------
 * - Supports float32, float64, float16 (sm_70+), and bfloat16 (sm_80+).
 * - For reduced-precision types (f16, bf16), accumulation is performed in
 *   float32 to maintain numerical precision, with results written back
 *   atomically.
 *
 * Performance Analysis (RTX 3080 Ti Laptop, 616 GB/s peak):
 * ----------------------------------------------------------
 * Benchmark: 5000×5000 matrix, nnz=1.25M (5% density), 1% spike rate
 *  - Memory traffic: 16.35 MB (all NNZ processed, only 1% active)
 *  - Measured latency: ~0.4 ms (typical), 0.1 ms (best case with boost)
 *  - Achieved bandwidth: ~41 GB/s typical, ~163 GB/s best
 *  - Efficiency vs peak: 6.7% typical, 26% best
 *
 * Fundamental Bottleneck:
 *  COO format has inherently random row/col access patterns. Each thread
 *  accesses a different cache line (no coalescing). This is the dominant
 *  bottleneck - standard SpMV optimization techniques (shared memory tiling,
 *  warp cooperation, __ldg()) provide negligible benefit because the random
 *  access pattern prevents effective cache utilization.
 *
 * Optimization Barriers:
 *  1. Random column/row indexing prevents memory coalescing (fundamental
 *     to COO format; would require CSR/CSC conversion for gather/scatter).
 *  2. Event-driven check eliminates only the atomic write (8B), but all
 *     index loads (13B) must still occur for every NNZ entry.
 *  3. Atomic contention at high spike rates (>10%) serializes writes,
 *     but cannot be avoided without algorithm change (e.g., segmented
 *     reduction or two-pass histogram).
 *
 * Recommendations for Higher Performance:
 *  - For gather (A @ v, transpose=False): Convert to CSR format for
 *    coalesced row access and shared-memory vector tiling.
 *  - For scatter (A.T @ v, transpose=True): Convert to CSC format or
 *    use a two-pass algorithm (1: gather active spikes; 2: process).
 *  - For very high spike rates (>50%): Use dense matrix multiplication
 *    as the random-access overhead dominates sparse savings.
 *
 * TVM FFI Integration:
 * -------------------
 * All kernels are exposed via TVM FFI with @tvm_ffi annotations for seamless
 * integration with JAX.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// ============================================================================
// Warp-level reduction helpers
// ============================================================================

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

// ============================================================================
// Active-check predicates
// ============================================================================

#define IS_ACTIVE_BOOL(s)  ((s) != 0)
#define IS_ACTIVE_FLOAT(s) ((s) > 0.0f)

// ============================================================================
// Per-dtype conversion macros: READ converts WEIGHT_T -> ACC_T
// ============================================================================

#define READ_F32(x)   (x)
#define READ_F64(x)   (x)
#define READ_F16(x)   __half2float(x)
#define READ_BF16(x)  __bfloat162float(x)

// ============================================================================
// Per-dtype atomic-add helpers (accumulator value -> weight memory)
// ============================================================================

__device__ __inline__ void atomic_add_f32(float* addr, float val) {
    atomicAdd(addr, val);
}

__device__ __inline__ void atomic_add_f64(double* addr, double val) {
    atomicAdd(addr, val);
}

__device__ __inline__ void atomic_add_f16(__half* addr, float val) {
#if __CUDA_ARCH__ >= 700
    atomicAdd(addr, __float2half(val));
#else
    unsigned int* base = reinterpret_cast<unsigned int*>(
        reinterpret_cast<size_t>(addr) & ~(size_t)2
    );
    int shift = ((reinterpret_cast<size_t>(addr) & 2) != 0) ? 16 : 0;
    unsigned int assumed, old_val = *base, updated;
    do {
        assumed = old_val;
        unsigned short h = static_cast<unsigned short>((assumed >> shift) & 0xFFFF);
        float cur = __half2float(*reinterpret_cast<__half*>(&h));
        __half new_h = __float2half(cur + val);
        unsigned short new_us = *reinterpret_cast<unsigned short*>(&new_h);
        updated = (assumed & ~(0xFFFFu << shift)) | (static_cast<unsigned int>(new_us) << shift);
        old_val = atomicCAS(base, assumed, updated);
    } while (assumed != old_val);
#endif
}

__device__ __inline__ void atomic_add_bf16(__nv_bfloat16* addr, float val) {
#if __CUDA_ARCH__ >= 800
    atomicAdd(addr, __float2bfloat16(val));
#else
    unsigned int* base = reinterpret_cast<unsigned int*>(
        reinterpret_cast<size_t>(addr) & ~(size_t)2
    );
    int shift = ((reinterpret_cast<size_t>(addr) & 2) != 0) ? 16 : 0;
    unsigned int assumed, old_val = *base, updated;
    do {
        assumed = old_val;
        unsigned short h = static_cast<unsigned short>((assumed >> shift) & 0xFFFF);
        float cur = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&h));
        __nv_bfloat16 new_h = __float2bfloat16(cur + val);
        unsigned short new_us = *reinterpret_cast<unsigned short*>(&new_h);
        updated = (assumed & ~(0xFFFFu << shift)) | (static_cast<unsigned int>(new_us) << shift);
        old_val = atomicCAS(base, assumed, updated);
    } while (assumed != old_val);
#endif
}

// ============================================================================
// COO Matrix-Vector Multiplication (coomv)
// ============================================================================

#define DEFINE_COOMV_ATOMIC_NT(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _coomv_atomic_nt_kern##SUFFIX(                                                      \
    const WEIGHT_T* __restrict__ data,                                                              \
    const int32_t*  __restrict__ row,                                                               \
    const int32_t*  __restrict__ col,                                                               \
    const SPIKE_T*  __restrict__ v,                                                                 \
    WEIGHT_T*                    out,                                                               \
    int nnz, int is_homo                                                                            \
) {                                                                                                 \
    ACC_T homo_w = READ_W(data[0]);                                                                 \
    int k = blockIdx.x * blockDim.x + threadIdx.x;                                                 \
    const int stride = gridDim.x * blockDim.x;                                                     \
    while (k < nnz) {                                                                               \
        if (IS_ACTIVE(v[col[k]])) {                                                                 \
            ACC_T w = is_homo ? homo_w : READ_W(data[k]);                                           \
            ATOMIC_ADD_W(out + row[k], w);                                                          \
        }                                                                                           \
        k += stride;                                                                                \
    }                                                                                               \
}

#define DEFINE_COOMV_ATOMIC_T(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _coomv_atomic_t_kern##SUFFIX(                                                      \
    const WEIGHT_T* __restrict__ data,                                                             \
    const int32_t*  __restrict__ row,                                                              \
    const int32_t*  __restrict__ col,                                                              \
    const SPIKE_T*  __restrict__ v,                                                                \
    WEIGHT_T*                    out,                                                              \
    int nnz, int is_homo                                                                           \
) {                                                                                                \
    ACC_T homo_w = READ_W(data[0]);                                                                \
    int k = blockIdx.x * blockDim.x + threadIdx.x;                                                \
    const int stride = gridDim.x * blockDim.x;                                                    \
    while (k < nnz) {                                                                              \
        if (IS_ACTIVE(v[row[k]])) {                                                                \
            ACC_T w = is_homo ? homo_w : READ_W(data[k]);                                          \
            ATOMIC_ADD_W(out + col[k], w);                                                         \
        }                                                                                          \
        k += stride;                                                                               \
    }                                                                                              \
}

// Instantiations
DEFINE_COOMV_ATOMIC_NT(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMV_ATOMIC_NT(_f32_float, float,  IS_ACTIVE_FLOAT, float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMV_ATOMIC_T (_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMV_ATOMIC_T (_f32_float, float,  IS_ACTIVE_FLOAT, float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMV_ATOMIC_NT(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMV_ATOMIC_NT(_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMV_ATOMIC_T (_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMV_ATOMIC_T (_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMV_ATOMIC_NT(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMV_ATOMIC_NT(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMV_ATOMIC_T (_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMV_ATOMIC_T (_f16_float, float,  IS_ACTIVE_FLOAT, __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMV_ATOMIC_NT(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMV_ATOMIC_NT(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMV_ATOMIC_T (_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMV_ATOMIC_T (_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)

// FFI Macros for SpMV
#define FFI_COOMV_ATOMIC_NT(SUFFIX, WEIGHT_C_T, SPIKE_C_T, OUT_BYTES_PER_ELEM)              \
void binary_coomv_atomic_nt##SUFFIX(                                                          \
    tvm::ffi::TensorView data,                                                                \
    tvm::ffi::TensorView row_idx,                                                             \
    tvm::ffi::TensorView col_idx,                                                             \
    tvm::ffi::TensorView v,                                                                   \
    tvm::ffi::TensorView output,                                                              \
    int64_t stream                                                                            \
) {                                                                                           \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                 \
    int nnz = static_cast<int>(row_idx.size(0));                                             \
    int m   = static_cast<int>(output.size(0));                                              \
    int is_homo = (data.size(0) == 1) ? 1 : 0;                                              \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                        \
    cudaMemsetAsync(d_out, 0, (size_t)m * OUT_BYTES_PER_ELEM, s);                           \
    if (nnz == 0) return;                                                                     \
    int block = 256;                                                                          \
    int grid  = (nnz + block - 1) / block;                                                  \
    _coomv_atomic_nt_kern##SUFFIX<<<grid, block, 0, s>>>(                                    \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                                     \
        static_cast<const int32_t*>(row_idx.data_ptr()),                                     \
        static_cast<const int32_t*>(col_idx.data_ptr()),                                     \
        static_cast<const SPIKE_C_T*>(v.data_ptr()),                                         \
        d_out, nnz, is_homo                                                                  \
    );                                                                                        \
}

#define FFI_COOMV_ATOMIC_T(SUFFIX, WEIGHT_C_T, SPIKE_C_T, OUT_BYTES_PER_ELEM)               \
void binary_coomv_atomic_t##SUFFIX(                                                           \
    tvm::ffi::TensorView data,                                                                \
    tvm::ffi::TensorView row_idx,                                                             \
    tvm::ffi::TensorView col_idx,                                                             \
    tvm::ffi::TensorView v,                                                                   \
    tvm::ffi::TensorView output,                                                              \
    int64_t stream                                                                            \
) {                                                                                           \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                 \
    int nnz = static_cast<int>(row_idx.size(0));                                             \
    int k   = static_cast<int>(output.size(0));                                              \
    int is_homo = (data.size(0) == 1) ? 1 : 0;                                              \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                        \
    cudaMemsetAsync(d_out, 0, (size_t)k * OUT_BYTES_PER_ELEM, s);                           \
    if (nnz == 0) return;                                                                     \
    int block = 256;                                                                          \
    int grid  = (nnz + block - 1) / block;                                                  \
    _coomv_atomic_t_kern##SUFFIX<<<grid, block, 0, s>>>(                                     \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                                     \
        static_cast<const int32_t*>(row_idx.data_ptr()),                                     \
        static_cast<const int32_t*>(col_idx.data_ptr()),                                     \
        static_cast<const SPIKE_C_T*>(v.data_ptr()),                                         \
        d_out, nnz, is_homo                                                                  \
    );                                                                                        \
}

// @tvm_ffi binary_coomv_atomic_nt_f32_bool
FFI_COOMV_ATOMIC_NT(_f32_bool,  float,  int8_t, sizeof(float))
// @tvm_ffi binary_coomv_atomic_nt_f32_float
FFI_COOMV_ATOMIC_NT(_f32_float, float,  float,  sizeof(float))
// @tvm_ffi binary_coomv_atomic_t_f32_bool
FFI_COOMV_ATOMIC_T (_f32_bool,  float,  int8_t, sizeof(float))
// @tvm_ffi binary_coomv_atomic_t_f32_float
FFI_COOMV_ATOMIC_T (_f32_float, float,  float,  sizeof(float))
// @tvm_ffi binary_coomv_atomic_nt_f64_bool
FFI_COOMV_ATOMIC_NT(_f64_bool,  double, int8_t, sizeof(double))
// @tvm_ffi binary_coomv_atomic_nt_f64_float
FFI_COOMV_ATOMIC_NT(_f64_float, double, float,  sizeof(double))
// @tvm_ffi binary_coomv_atomic_t_f64_bool
FFI_COOMV_ATOMIC_T (_f64_bool,  double, int8_t, sizeof(double))
// @tvm_ffi binary_coomv_atomic_t_f64_float
FFI_COOMV_ATOMIC_T (_f64_float, double, float,  sizeof(double))
// @tvm_ffi binary_coomv_atomic_nt_f16_bool
FFI_COOMV_ATOMIC_NT(_f16_bool,  __half, int8_t, sizeof(__half))
// @tvm_ffi binary_coomv_atomic_nt_f16_float
FFI_COOMV_ATOMIC_NT(_f16_float, __half, float,  sizeof(__half))
// @tvm_ffi binary_coomv_atomic_t_f16_bool
FFI_COOMV_ATOMIC_T (_f16_bool,  __half, int8_t, sizeof(__half))
// @tvm_ffi binary_coomv_atomic_t_f16_float
FFI_COOMV_ATOMIC_T (_f16_float, __half, float,  sizeof(__half))
// @tvm_ffi binary_coomv_atomic_nt_bf16_bool
FFI_COOMV_ATOMIC_NT(_bf16_bool,  __nv_bfloat16, int8_t, sizeof(__nv_bfloat16))
// @tvm_ffi binary_coomv_atomic_nt_bf16_float
FFI_COOMV_ATOMIC_NT(_bf16_float, __nv_bfloat16, float,  sizeof(__nv_bfloat16))
// @tvm_ffi binary_coomv_atomic_t_bf16_bool
FFI_COOMV_ATOMIC_T (_bf16_bool,  __nv_bfloat16, int8_t, sizeof(__nv_bfloat16))
// @tvm_ffi binary_coomv_atomic_t_bf16_float
FFI_COOMV_ATOMIC_T (_bf16_float, __nv_bfloat16, float,  sizeof(__nv_bfloat16))


// ============================================================================
// COO Matrix-Matrix Multiplication (coomm)
// ============================================================================

#define COOMM_CT_BLOCK_K   32
#define COOMM_CT_BLOCK_N   32
#define COOMM_WPE_WARPS    8
#define COOMM_WPE_COLS     32

#define DEFINE_COOMM_CT_NT(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _coomm_ct_nt_kern##SUFFIX(                                                      \
    const WEIGHT_T* __restrict__ data,                                                          \
    const int32_t*  __restrict__ row,                                                           \
    const int32_t*  __restrict__ col,                                                           \
    const SPIKE_T*  __restrict__ B,                                                             \
    WEIGHT_T*                    out,                                                           \
    int nnz, int n, int is_homo                                                                 \
) {                                                                                             \
    int nnz_start = blockIdx.x * COOMM_CT_BLOCK_K;                                             \
    int col_start = blockIdx.y * COOMM_CT_BLOCK_N;                                             \
    int t         = threadIdx.x;                                                                \
    int my_col    = col_start + t;                                                              \
    bool col_valid = (my_col < n);                                                              \
    ACC_T homo_w = READ_W(data[0]);                                                             \
    int nnz_end = nnz_start + COOMM_CT_BLOCK_K;                                                \
    if (nnz_end > nnz) nnz_end = nnz;                                                          \
    for (int s = nnz_start; s < nnz_end; s++) {                                                \
        int src = col[s];                                                                      \
        int dst = row[s];                                                                      \
        SPIKE_T spike = col_valid ? B[(int64_t)src * n + my_col] : (SPIKE_T)0;                \
        bool active = IS_ACTIVE(spike) && col_valid;                                            \
        uint32_t ballot = __ballot_sync(0xffffffff, active);                                    \
        if (ballot == 0u) continue;                                                             \
        if (active) {                                                                           \
            ACC_T w = is_homo ? homo_w : READ_W(data[s]);                                      \
            ATOMIC_ADD_W(out + (int64_t)dst * n + my_col, w);                                  \
        }                                                                                       \
    }                                                                                           \
}

#define DEFINE_COOMM_CT_T(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _coomm_ct_t_kern##SUFFIX(                                                      \
    const WEIGHT_T* __restrict__ data,                                                         \
    const int32_t*  __restrict__ row,                                                          \
    const int32_t*  __restrict__ col,                                                          \
    const SPIKE_T*  __restrict__ B,                                                            \
    WEIGHT_T*                    out,                                                          \
    int nnz, int n, int is_homo                                                                \
) {                                                                                            \
    int nnz_start = blockIdx.x * COOMM_CT_BLOCK_K;                                            \
    int col_start = blockIdx.y * COOMM_CT_BLOCK_N;                                            \
    int t         = threadIdx.x;                                                               \
    int my_col    = col_start + t;                                                             \
    bool col_valid = (my_col < n);                                                             \
    ACC_T homo_w = READ_W(data[0]);                                                            \
    int nnz_end = nnz_start + COOMM_CT_BLOCK_K;                                               \
    if (nnz_end > nnz) nnz_end = nnz;                                                         \
    for (int s = nnz_start; s < nnz_end; s++) {                                               \
        int src = row[s];                                                                     \
        int dst = col[s];                                                                     \
        SPIKE_T spike = col_valid ? B[(int64_t)src * n + my_col] : (SPIKE_T)0;               \
        bool active = IS_ACTIVE(spike) && col_valid;                                           \
        uint32_t ballot = __ballot_sync(0xffffffff, active);                                   \
        if (ballot == 0u) continue;                                                            \
        if (active) {                                                                          \
            ACC_T w = is_homo ? homo_w : READ_W(data[s]);                                     \
            ATOMIC_ADD_W(out + (int64_t)dst * n + my_col, w);                                 \
        }                                                                                      \
    }                                                                                          \
}

#define DEFINE_COOMM_WPE_NT(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _coomm_wpe_nt_kern##SUFFIX(                                                      \
    const WEIGHT_T* __restrict__ data,                                                           \
    const int32_t*  __restrict__ row,                                                            \
    const int32_t*  __restrict__ col,                                                            \
    const SPIKE_T*  __restrict__ B,                                                              \
    WEIGHT_T*                    out,                                                           \
    int nnz, int n, int is_homo                                                                  \
) {                                                                                              \
    int warp_id  = (int)(blockIdx.x * COOMM_WPE_WARPS) + (int)(threadIdx.x >> 5);              \
    int lane     = threadIdx.x & 31;                                                             \
    int col_start = blockIdx.y * COOMM_WPE_COLS;                                                \
    int my_col    = col_start + lane;                                                            \
    if (warp_id >= nnz) return;                                                                  \
    bool col_valid = (my_col < n);                                                               \
    int s   = warp_id;                                                                           \
    int src = col[s];                                                                           \
    int dst = row[s];                                                                           \
    SPIKE_T spike = col_valid ? B[(int64_t)src * n + my_col] : (SPIKE_T)0;                     \
    bool active = IS_ACTIVE(spike) && col_valid;                                                 \
    uint32_t ballot = __ballot_sync(0xffffffff, active);                                         \
    if (ballot == 0u) return;                                                                    \
    if (active) {                                                                                 \
        ACC_T w = is_homo ? READ_W(data[0]) : READ_W(data[s]);                                  \
        ATOMIC_ADD_W(out + (int64_t)dst * n + my_col, w);                                       \
    }                                                                                            \
}

#define DEFINE_COOMM_WPE_T(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _coomm_wpe_t_kern##SUFFIX(                                                      \
    const WEIGHT_T* __restrict__ data,                                                          \
    const int32_t*  __restrict__ row,                                                           \
    const int32_t*  __restrict__ col,                                                           \
    const SPIKE_T*  __restrict__ B,                                                             \
    WEIGHT_T*                    out,                                                           \
    int nnz, int n, int is_homo                                                                 \
) {                                                                                             \
    int warp_id   = (int)(blockIdx.x * COOMM_WPE_WARPS) + (int)(threadIdx.x >> 5);            \
    int lane      = threadIdx.x & 31;                                                           \
    int col_start = blockIdx.y * COOMM_WPE_COLS;                                               \
    int my_col    = col_start + lane;                                                           \
    if (warp_id >= nnz) return;                                                                 \
    bool col_valid = (my_col < n);                                                              \
    int s   = warp_id;                                                                          \
    int src = row[s];                                                                          \
    int dst = col[s];                                                                          \
    SPIKE_T spike = col_valid ? B[(int64_t)src * n + my_col] : (SPIKE_T)0;                    \
    bool active = IS_ACTIVE(spike) && col_valid;                                                \
    uint32_t ballot = __ballot_sync(0xffffffff, active);                                        \
    if (ballot == 0u) return;                                                                   \
    if (active) {                                                                               \
        ACC_T w = is_homo ? READ_W(data[0]) : READ_W(data[s]);                                 \
        ATOMIC_ADD_W(out + (int64_t)dst * n + my_col, w);                                      \
    }                                                                                           \
}

// Instantiations
DEFINE_COOMM_CT_NT(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_CT_NT(_f32_float, float,  IS_ACTIVE_FLOAT, float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_CT_T (_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_CT_T (_f32_float, float,  IS_ACTIVE_FLOAT, float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_WPE_NT(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_WPE_NT(_f32_float, float,  IS_ACTIVE_FLOAT, float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_WPE_T (_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_WPE_T (_f32_float, float,  IS_ACTIVE_FLOAT, float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_CT_NT(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_CT_NT(_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_CT_T (_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_CT_T (_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_WPE_NT(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_WPE_NT(_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_WPE_T (_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_WPE_T (_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_CT_NT(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_CT_NT(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_CT_T (_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_CT_T (_f16_float, float,  IS_ACTIVE_FLOAT, __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_WPE_NT(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_WPE_NT(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_WPE_T (_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_WPE_T (_f16_float, float,  IS_ACTIVE_FLOAT, __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_CT_NT(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMM_CT_NT(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMM_CT_T (_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMM_CT_T (_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMM_WPE_NT(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMM_WPE_NT(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMM_WPE_T (_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMM_WPE_T (_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)

// FFI Macros for SpMM
#define FFI_COOMM_CT_NT(SUFFIX, WEIGHT_C_T, SPIKE_C_T, OUT_BYTES_PER_ELEM)               \
void binary_coomm_ct_nt##SUFFIX(                                                           \
    tvm::ffi::TensorView data,                                                             \
    tvm::ffi::TensorView row_idx,                                                          \
    tvm::ffi::TensorView col_idx,                                                          \
    tvm::ffi::TensorView B,                                                                \
    tvm::ffi::TensorView output,                                                           \
    int64_t stream                                                                         \
) {                                                                                        \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                              \
    int nnz  = static_cast<int>(row_idx.size(0));                                         \
    int n    = static_cast<int>(B.size(1));                                               \
    int m    = static_cast<int>(output.size(0));                                          \
    int is_homo = (data.size(0) == 1) ? 1 : 0;                                           \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                     \
    cudaMemsetAsync(d_out, 0, (size_t)m * n * OUT_BYTES_PER_ELEM, s);                    \
    if (nnz == 0) return;                                                                  \
    dim3 block(COOMM_CT_BLOCK_N, 1, 1);                                                   \
    dim3 grid(                                                                             \
        (nnz + COOMM_CT_BLOCK_K - 1) / COOMM_CT_BLOCK_K,                                 \
        (n   + COOMM_CT_BLOCK_N - 1) / COOMM_CT_BLOCK_N,                                 \
        1                                                                                  \
    );                                                                                     \
    _coomm_ct_nt_kern##SUFFIX<<<grid, block, 0, s>>>(                                     \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                                  \
        static_cast<const int32_t*>(row_idx.data_ptr()),                                  \
        static_cast<const int32_t*>(col_idx.data_ptr()),                                  \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                                      \
        d_out, nnz, n, is_homo                                                            \
    );                                                                                     \
}

#define FFI_COOMM_CT_T(SUFFIX, WEIGHT_C_T, SPIKE_C_T, OUT_BYTES_PER_ELEM)                \
void binary_coomm_ct_t##SUFFIX(                                                            \
    tvm::ffi::TensorView data,                                                             \
    tvm::ffi::TensorView row_idx,                                                          \
    tvm::ffi::TensorView col_idx,                                                          \
    tvm::ffi::TensorView B,                                                                \
    tvm::ffi::TensorView output,                                                           \
    int64_t stream                                                                         \
) {                                                                                        \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                              \
    int nnz  = static_cast<int>(row_idx.size(0));                                         \
    int n    = static_cast<int>(B.size(1));                                               \
    int k_out = static_cast<int>(output.size(0));                                         \
    int is_homo = (data.size(0) == 1) ? 1 : 0;                                           \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                     \
    cudaMemsetAsync(d_out, 0, (size_t)k_out * n * OUT_BYTES_PER_ELEM, s);                \
    if (nnz == 0) return;                                                                  \
    dim3 block(COOMM_CT_BLOCK_N, 1, 1);                                                   \
    dim3 grid(                                                                             \
        (nnz + COOMM_CT_BLOCK_K - 1) / COOMM_CT_BLOCK_K,                                 \
        (n   + COOMM_CT_BLOCK_N - 1) / COOMM_CT_BLOCK_N,                                 \
        1                                                                                  \
    );                                                                                     \
    _coomm_ct_t_kern##SUFFIX<<<grid, block, 0, s>>>(                                      \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                                  \
        static_cast<const int32_t*>(row_idx.data_ptr()),                                  \
        static_cast<const int32_t*>(col_idx.data_ptr()),                                  \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                                      \
        d_out, nnz, n, is_homo                                                            \
    );                                                                                     \
}

#define FFI_COOMM_WPE_NT(SUFFIX, WEIGHT_C_T, SPIKE_C_T, OUT_BYTES_PER_ELEM)              \
void binary_coomm_wpe_nt##SUFFIX(                                                          \
    tvm::ffi::TensorView data,                                                             \
    tvm::ffi::TensorView row_idx,                                                          \
    tvm::ffi::TensorView col_idx,                                                          \
    tvm::ffi::TensorView B,                                                                \
    tvm::ffi::TensorView output,                                                           \
    int64_t stream                                                                         \
) {                                                                                        \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                              \
    int nnz   = static_cast<int>(row_idx.size(0));                                        \
    int n     = static_cast<int>(B.size(1));                                              \
    int m     = static_cast<int>(output.size(0));                                         \
    int is_homo = (data.size(0) == 1) ? 1 : 0;                                           \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                     \
    cudaMemsetAsync(d_out, 0, (size_t)m * n * OUT_BYTES_PER_ELEM, s);                    \
    if (nnz == 0) return;                                                                  \
    dim3 block(COOMM_WPE_WARPS * 32, 1, 1);                                               \
    dim3 grid(                                                                             \
        (nnz + COOMM_WPE_WARPS - 1) / COOMM_WPE_WARPS,                                   \
        (n   + COOMM_WPE_COLS  - 1) / COOMM_WPE_COLS,                                    \
        1                                                                                  \
    );                                                                                     \
    _coomm_wpe_nt_kern##SUFFIX<<<grid, block, 0, s>>>(                                    \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                                  \
        static_cast<const int32_t*>(row_idx.data_ptr()),                                  \
        static_cast<const int32_t*>(col_idx.data_ptr()),                                  \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                                      \
        d_out, nnz, n, is_homo                                                            \
    );                                                                                     \
}

#define FFI_COOMM_WPE_T(SUFFIX, WEIGHT_C_T, SPIKE_C_T, OUT_BYTES_PER_ELEM)               \
void binary_coomm_wpe_t##SUFFIX(                                                           \
    tvm::ffi::TensorView data,                                                             \
    tvm::ffi::TensorView row_idx,                                                          \
    tvm::ffi::TensorView col_idx,                                                          \
    tvm::ffi::TensorView B,                                                                \
    tvm::ffi::TensorView output,                                                           \
    int64_t stream                                                                         \
) {                                                                                        \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                              \
    int nnz   = static_cast<int>(row_idx.size(0));                                        \
    int n     = static_cast<int>(B.size(1));                                              \
    int k_out = static_cast<int>(output.size(0));                                         \
    int is_homo = (data.size(0) == 1) ? 1 : 0;                                           \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                     \
    cudaMemsetAsync(d_out, 0, (size_t)k_out * n * OUT_BYTES_PER_ELEM, s);                \
    if (nnz == 0) return;                                                                  \
    dim3 block(COOMM_WPE_WARPS * 32, 1, 1);                                               \
    dim3 grid(                                                                             \
        (nnz + COOMM_WPE_WARPS - 1) / COOMM_WPE_WARPS,                                   \
        (n   + COOMM_WPE_COLS  - 1) / COOMM_WPE_COLS,                                    \
        1                                                                                  \
    );                                                                                     \
    _coomm_wpe_t_kern##SUFFIX<<<grid, block, 0, s>>>(                                     \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                                  \
        static_cast<const int32_t*>(row_idx.data_ptr()),                                  \
        static_cast<const int32_t*>(col_idx.data_ptr()),                                  \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                                      \
        d_out, nnz, n, is_homo                                                            \
    );                                                                                     \
}

// SpMM Instantiations
// CT-NT
// @tvm_ffi binary_coomm_ct_nt_f32_bool
FFI_COOMM_CT_NT(_f32_bool,  float,  int8_t, sizeof(float))
// @tvm_ffi binary_coomm_ct_nt_f32_float
FFI_COOMM_CT_NT(_f32_float, float,  float,  sizeof(float))
// @tvm_ffi binary_coomm_ct_nt_f64_bool
FFI_COOMM_CT_NT(_f64_bool,  double, int8_t, sizeof(double))
// @tvm_ffi binary_coomm_ct_nt_f64_float
FFI_COOMM_CT_NT(_f64_float, double, float,  sizeof(double))
// @tvm_ffi binary_coomm_ct_nt_f16_bool
FFI_COOMM_CT_NT(_f16_bool,  __half, int8_t, sizeof(__half))
// @tvm_ffi binary_coomm_ct_nt_f16_float
FFI_COOMM_CT_NT(_f16_float, __half, float,  sizeof(__half))
// @tvm_ffi binary_coomm_ct_nt_bf16_bool
FFI_COOMM_CT_NT(_bf16_bool,  __nv_bfloat16, int8_t, sizeof(__nv_bfloat16))
// @tvm_ffi binary_coomm_ct_nt_bf16_float
FFI_COOMM_CT_NT(_bf16_float, __nv_bfloat16, float,  sizeof(__nv_bfloat16))

// CT-T
// @tvm_ffi binary_coomm_ct_t_f32_bool
FFI_COOMM_CT_T(_f32_bool,  float,  int8_t, sizeof(float))
// @tvm_ffi binary_coomm_ct_t_f32_float
FFI_COOMM_CT_T(_f32_float, float,  float,  sizeof(float))
// @tvm_ffi binary_coomm_ct_t_f64_bool
FFI_COOMM_CT_T(_f64_bool,  double, int8_t, sizeof(double))
// @tvm_ffi binary_coomm_ct_t_f64_float
FFI_COOMM_CT_T(_f64_float, double, float,  sizeof(double))
// @tvm_ffi binary_coomm_ct_t_f16_bool
FFI_COOMM_CT_T(_f16_bool,  __half, int8_t, sizeof(__half))
// @tvm_ffi binary_coomm_ct_t_f16_float
FFI_COOMM_CT_T(_f16_float, __half, float,  sizeof(__half))
// @tvm_ffi binary_coomm_ct_t_bf16_bool
FFI_COOMM_CT_T(_bf16_bool,  __nv_bfloat16, int8_t, sizeof(__nv_bfloat16))
// @tvm_ffi binary_coomm_ct_t_bf16_float
FFI_COOMM_CT_T(_bf16_float, __nv_bfloat16, float,  sizeof(__nv_bfloat16))

// WPE-NT
// @tvm_ffi binary_coomm_wpe_nt_f32_bool
FFI_COOMM_WPE_NT(_f32_bool,  float,  int8_t, sizeof(float))
// @tvm_ffi binary_coomm_wpe_nt_f32_float
FFI_COOMM_WPE_NT(_f32_float, float,  float,  sizeof(float))
// @tvm_ffi binary_coomm_wpe_nt_f64_bool
FFI_COOMM_WPE_NT(_f64_bool,  double, int8_t, sizeof(double))
// @tvm_ffi binary_coomm_wpe_nt_f64_float
FFI_COOMM_WPE_NT(_f64_float, double, float,  sizeof(double))
// @tvm_ffi binary_coomm_wpe_nt_f16_bool
FFI_COOMM_WPE_NT(_f16_bool,  __half, int8_t, sizeof(__half))
// @tvm_ffi binary_coomm_wpe_nt_f16_float
FFI_COOMM_WPE_NT(_f16_float, __half, float,  sizeof(__half))
// @tvm_ffi binary_coomm_wpe_nt_bf16_bool
FFI_COOMM_WPE_NT(_bf16_bool,  __nv_bfloat16, int8_t, sizeof(__nv_bfloat16))
// @tvm_ffi binary_coomm_wpe_nt_bf16_float
FFI_COOMM_WPE_NT(_bf16_float, __nv_bfloat16, float,  sizeof(__nv_bfloat16))

// WPE-T
// @tvm_ffi binary_coomm_wpe_t_f32_bool
FFI_COOMM_WPE_T(_f32_bool,  float,  int8_t, sizeof(float))
// @tvm_ffi binary_coomm_wpe_t_f32_float
FFI_COOMM_WPE_T(_f32_float, float,  float,  sizeof(float))
// @tvm_ffi binary_coomm_wpe_t_f64_bool
FFI_COOMM_WPE_T(_f64_bool,  double, int8_t, sizeof(double))
// @tvm_ffi binary_coomm_wpe_t_f64_float
FFI_COOMM_WPE_T(_f64_float, double, float,  sizeof(double))
// @tvm_ffi binary_coomm_wpe_t_f16_bool
FFI_COOMM_WPE_T(_f16_bool,  __half, int8_t, sizeof(__half))
// @tvm_ffi binary_coomm_wpe_t_f16_float
FFI_COOMM_WPE_T(_f16_float, __half, float,  sizeof(__half))
// @tvm_ffi binary_coomm_wpe_t_bf16_bool
FFI_COOMM_WPE_T(_bf16_bool,  __nv_bfloat16, int8_t, sizeof(__nv_bfloat16))
// @tvm_ffi binary_coomm_wpe_t_bf16_float
FFI_COOMM_WPE_T(_bf16_float, __nv_bfloat16, float,  sizeof(__nv_bfloat16))
