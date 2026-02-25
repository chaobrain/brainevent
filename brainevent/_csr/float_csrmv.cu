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
 * float_csrmv.cu -- Float-Weighted CSR Sparse Matrix-Vector Product (SpMV) CUDA Kernels
 * ======================================================================================
 *
 * This module provides optimized CUDA kernels for standard (non-event-driven)
 * sparse matrix-vector multiplication in Compressed Sparse Row (CSR) format.
 *
 * Operator: csrmv
 *   y = A * x          (non-transpose, NT)
 *   y = A^T * x        (transpose, T)
 *
 * Where A is an m×k CSR sparse matrix and x is a dense vector.
 *
 * Kernel Variants:
 *   - csrmv_nt_thread_{f32,f64,f16,bf16} : one thread per row (avg_nnz < 8)
 *   - csrmv_nt_warp_{f32,f64,f16,bf16}   : one warp per row  (8 <= avg_nnz < 512)
 *   - csrmv_nt_block_{f32,f64,f16,bf16}  : one block per row (avg_nnz >= 512)
 *   - csrmv_nt_auto_{f32,f64,f16,bf16}   : auto-selects thread/warp/block
 *   - csrmv_t_warp_{f32,f64,f16,bf16}    : transpose scatter, one warp per row
 *
 * Parameters (CUDA entry points):
 *   weights  : [nnz] or [1] float array  (hetero or homo weights)
 *   indices  : [nnz] int32 column indices
 *   indptr   : [m+1] int32 row pointers
 *   vector   : [k] float input vector
 *   output   : [m] (NT) or [k] (T) float output vector
 *   stream   : int64 CUDA stream handle
 * ======================================================================================
 */

#include "cuda_common.h"
#include "brainevent/common.h"

// =========================================================================
// CSR Matrix-Vector Multiplication (csrmv)
// =========================================================================

#define DEFINE_CSRMV_NT_THREAD(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _csrmv_nt_thread_kern##SUFFIX(                                     \
    const WEIGHT_T* __restrict__ weights,                                          \
    const int32_t*  __restrict__ indices,                                          \
    const int32_t*  __restrict__ indptr,                                           \
    const WEIGHT_T* __restrict__ vector,                                           \
    WEIGHT_T*       __restrict__ output,                                           \
    int m, int is_homo                                                             \
) {                                                                                \
    int row = blockIdx.x * blockDim.x + threadIdx.x;                               \
    if (row >= m) return;                                                          \
    int start = indptr[row], end = indptr[row + 1];                                \
    ACC_T acc = ACC_ZERO;                                                          \
    if (is_homo) {                                                                 \
        ACC_T w = READ_W(weights[0]);                                              \
        for (int j = start; j < end; j++) {                                        \
            acc += w * READ_W(vector[indices[j]]);                                 \
        }                                                                          \
    } else {                                                                       \
        for (int j = start; j < end; j++) {                                        \
            acc += READ_W(weights[j]) * READ_W(vector[indices[j]]);                \
        }                                                                          \
    }                                                                              \
    output[row] = WRITE_W(acc);                                                    \
}

#define DEFINE_CSRMV_NT_WARP(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _csrmv_nt_warp_kern##SUFFIX(                                               \
    const WEIGHT_T* __restrict__ weights,                                                  \
    const int32_t*  __restrict__ indices,                                                  \
    const int32_t*  __restrict__ indptr,                                                   \
    const WEIGHT_T* __restrict__ vector,                                                   \
    WEIGHT_T*       __restrict__ output,                                                   \
    int m, int is_homo                                                                     \
) {                                                                                        \
    int row = blockIdx.x;                                                                  \
    if (row >= m) return;                                                                  \
    int start = indptr[row], end = indptr[row + 1];                                        \
    ACC_T acc = ACC_ZERO;                                                                  \
    if (is_homo) {                                                                         \
        ACC_T w = READ_W(weights[0]);                                                      \
        for (int j = start + (int)threadIdx.x; j < end; j += 32) {                         \
            acc += w * READ_W(vector[indices[j]]);                                         \
        }                                                                                  \
    } else {                                                                               \
        for (int j = start + (int)threadIdx.x; j < end; j += 32) {                         \
            acc += READ_W(weights[j]) * READ_W(vector[indices[j]]);                        \
        }                                                                                  \
    }                                                                                      \
    acc = WARP_RED(acc);                                                                   \
    if (threadIdx.x == 0) output[row] = WRITE_W(acc);                                      \
}

#define DEFINE_CSRMV_NT_BLOCK(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _csrmv_nt_block_kern##SUFFIX(                                               \
    const WEIGHT_T* __restrict__ weights,                                                   \
    const int32_t*  __restrict__ indices,                                                   \
    const int32_t*  __restrict__ indptr,                                                    \
    const WEIGHT_T* __restrict__ vector,                                                    \
    WEIGHT_T*       __restrict__ output,                                                    \
    int m, int is_homo                                                                      \
) {                                                                                         \
    extern __shared__ char _smem_bytes[];                                                   \
    ACC_T* smem_red = reinterpret_cast<ACC_T*>(_smem_bytes);                                \
    int row = blockIdx.x;                                                                   \
    if (row >= m) return;                                                                   \
    int start = indptr[row], end = indptr[row + 1];                                         \
    ACC_T acc = ACC_ZERO;                                                                   \
    if (is_homo) {                                                                          \
        ACC_T w = READ_W(weights[0]);                                                       \
        for (int j = start + (int)threadIdx.x; j < end; j += blockDim.x) {                  \
            acc += w * READ_W(vector[indices[j]]);                                          \
        }                                                                                   \
    } else {                                                                                \
        for (int j = start + (int)threadIdx.x; j < end; j += blockDim.x) {                  \
            acc += READ_W(weights[j]) * READ_W(vector[indices[j]]);                         \
        }                                                                                   \
    }                                                                                       \
    int lane   = threadIdx.x & 31;                                                          \
    int warpid = threadIdx.x >> 5;                                                          \
    acc = WARP_RED(acc);                                                                    \
    if (lane == 0) smem_red[warpid] = acc;                                                  \
    __syncthreads();                                                                        \
    int n_warps = (blockDim.x + 31) >> 5;                                                   \
    acc = (threadIdx.x < n_warps) ? smem_red[lane] : ACC_ZERO;                              \
    if (warpid == 0) acc = WARP_RED(acc);                                                   \
    if (threadIdx.x == 0) output[row] = WRITE_W(acc);                                       \
}

#define DEFINE_CSRMV_T_WARP(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO)  \
__global__ void _csrmv_t_warp_kern##SUFFIX(                                      \
    const WEIGHT_T* __restrict__ weights,                                        \
    const int32_t*  __restrict__ indices,                                        \
    const int32_t*  __restrict__ indptr,                                         \
    const WEIGHT_T* __restrict__ vector,                                         \
    WEIGHT_T*       __restrict__ output,                                         \
    int m, int is_homo                                                           \
) {                                                                              \
    int row = blockIdx.x;                                                        \
    if (row >= m) return;                                                        \
    ACC_T v_val = READ_W(vector[row]);                                           \
    int start = indptr[row], end = indptr[row + 1];                              \
    if (is_homo) {                                                               \
        WEIGHT_T contrib = WRITE_W(READ_W(weights[0]) * v_val);                  \
        for (int j = start + (int)threadIdx.x; j < end; j += 32) {               \
            atomicAdd(&output[indices[j]], contrib);                             \
        }                                                                        \
    } else {                                                                     \
        for (int j = start + (int)threadIdx.x; j < end; j += 32) {               \
            atomicAdd(&output[indices[j]], WRITE_W(READ_W(weights[j]) * v_val)); \
        }                                                                        \
    }                                                                            \
}

// SpMV Instantiations
DEFINE_CSRMV_NT_THREAD(_f32,  float,          float,  READ_F32,  WRITE_F32,  0.0f)
DEFINE_CSRMV_NT_WARP(_f32,    float,          float,  READ_F32,  WRITE_F32,  warp_reduce_sum_f32, 0.0f)
DEFINE_CSRMV_NT_BLOCK(_f32,   float,          float,  READ_F32,  WRITE_F32,  warp_reduce_sum_f32, 0.0f)
DEFINE_CSRMV_T_WARP(_f32,     float,          float,  READ_F32,  WRITE_F32,  0.0f)
DEFINE_CSRMV_NT_THREAD(_f64,  double,         double, READ_F64,  WRITE_F64,  0.0)
DEFINE_CSRMV_NT_WARP(_f64,    double,         double, READ_F64,  WRITE_F64,  warp_reduce_sum_f64, 0.0)
DEFINE_CSRMV_NT_BLOCK(_f64,   double,         double, READ_F64,  WRITE_F64,  warp_reduce_sum_f64, 0.0)
DEFINE_CSRMV_T_WARP(_f64,     double,         double, READ_F64,  WRITE_F64,  0.0)
DEFINE_CSRMV_NT_THREAD(_f16,  __half,         float,  READ_F16,  WRITE_F16,  0.0f)
DEFINE_CSRMV_NT_WARP(_f16,    __half,         float,  READ_F16,  WRITE_F16,  warp_reduce_sum_f32, 0.0f)
DEFINE_CSRMV_NT_BLOCK(_f16,   __half,         float,  READ_F16,  WRITE_F16,  warp_reduce_sum_f32, 0.0f)
DEFINE_CSRMV_T_WARP(_f16,     __half,         float,  READ_F16,  WRITE_F16,  0.0f)
DEFINE_CSRMV_NT_THREAD(_bf16, __nv_bfloat16,  float,  READ_BF16, WRITE_BF16, 0.0f)
DEFINE_CSRMV_NT_WARP(_bf16,   __nv_bfloat16,  float,  READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_CSRMV_NT_BLOCK(_bf16,  __nv_bfloat16,  float,  READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_CSRMV_T_WARP(_bf16,    __nv_bfloat16,  float,  READ_BF16, WRITE_BF16, 0.0f)

// FFI Macros for SpMV
#define FFI_CSRMV_NT_THREAD(SUFFIX, WEIGHT_C_T)                   \
void csrmv_nt_thread##SUFFIX(                                     \
    const BE::Tensor weights, const BE::Tensor indices,   \
    const BE::Tensor indptr,  const BE::Tensor vector,    \
    BE::Tensor output,  int64_t stream                  \
) {                                                               \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);     \
    int m           = static_cast<int>(indptr.size(0)) - 1;       \
    int is_homo     = (weights.size(0) == 1) ? 1 : 0;             \
    int blocks      = (m + 255) / 256;                            \
    _csrmv_nt_thread_kern##SUFFIX<<<blocks, 256, 0, s>>>(         \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),       \
        static_cast<const int32_t*>(indices.data_ptr()),          \
        static_cast<const int32_t*>(indptr.data_ptr()),           \
        static_cast<const WEIGHT_C_T*>(vector.data_ptr()),        \
        static_cast<WEIGHT_C_T*>(output.data_ptr()), m, is_homo); \
}

#define FFI_CSRMV_NT_WARP(SUFFIX, WEIGHT_C_T)                     \
void csrmv_nt_warp##SUFFIX(                                       \
    const BE::Tensor weights, const BE::Tensor indices,   \
    const BE::Tensor indptr,  const BE::Tensor vector,    \
    BE::Tensor output,  int64_t stream                  \
) {                                                               \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);     \
    int m           = static_cast<int>(indptr.size(0)) - 1;       \
    int is_homo     = (weights.size(0) == 1) ? 1 : 0;             \
    _csrmv_nt_warp_kern##SUFFIX<<<m, 32, 0, s>>>(                 \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),       \
        static_cast<const int32_t*>(indices.data_ptr()),          \
        static_cast<const int32_t*>(indptr.data_ptr()),           \
        static_cast<const WEIGHT_C_T*>(vector.data_ptr()),        \
        static_cast<WEIGHT_C_T*>(output.data_ptr()), m, is_homo); \
}

#define FFI_CSRMV_NT_BLOCK(SUFFIX, WEIGHT_C_T, SHM_SIZE)          \
void csrmv_nt_block##SUFFIX(                                      \
    const BE::Tensor weights, const BE::Tensor indices,   \
    const BE::Tensor indptr,  const BE::Tensor vector,    \
    BE::Tensor output,  int64_t stream                  \
) {                                                               \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);     \
    int m           = static_cast<int>(indptr.size(0)) - 1;       \
    int is_homo     = (weights.size(0) == 1) ? 1 : 0;             \
    _csrmv_nt_block_kern##SUFFIX<<<m, 256, SHM_SIZE, s>>>(        \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),       \
        static_cast<const int32_t*>(indices.data_ptr()),          \
        static_cast<const int32_t*>(indptr.data_ptr()),           \
        static_cast<const WEIGHT_C_T*>(vector.data_ptr()),        \
        static_cast<WEIGHT_C_T*>(output.data_ptr()), m, is_homo); \
}

#define FFI_CSRMV_NT_AUTO(SUFFIX, WEIGHT_C_T, SHM_SIZE)                         \
void csrmv_nt_auto##SUFFIX(                                                     \
    const BE::Tensor weights, const BE::Tensor indices,                 \
    const BE::Tensor indptr,  const BE::Tensor vector,                  \
    BE::Tensor output,  int64_t stream                                \
) {                                                                             \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                   \
    int m           = static_cast<int>(indptr.size(0)) - 1;                     \
    int nse         = static_cast<int>(indices.size(0));                        \
    int is_homo     = (weights.size(0) == 1) ? 1 : 0;                           \
    int avg_nnz     = (m > 0) ? (nse / m) : 0;                                  \
    const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*    d_i = static_cast<const int32_t*>(indices.data_ptr());    \
    const int32_t*    d_p = static_cast<const int32_t*>(indptr.data_ptr());     \
    const WEIGHT_C_T* d_v = static_cast<const WEIGHT_C_T*>(vector.data_ptr());  \
    WEIGHT_C_T*       d_o = static_cast<WEIGHT_C_T*>(output.data_ptr());        \
    if (avg_nnz < 8) {                                                          \
        int blocks = (m + 255) / 256;                                           \
        _csrmv_nt_thread_kern##SUFFIX<<<blocks, 256, 0, s>>>(                   \
            d_w, d_i, d_p, d_v, d_o, m, is_homo);                               \
    } else if (avg_nnz < 512) {                                                 \
        _csrmv_nt_warp_kern##SUFFIX<<<m, 32, 0, s>>>(                           \
            d_w, d_i, d_p, d_v, d_o, m, is_homo);                               \
    } else {                                                                    \
        _csrmv_nt_block_kern##SUFFIX<<<m, 256, SHM_SIZE, s>>>(                  \
            d_w, d_i, d_p, d_v, d_o, m, is_homo);                               \
    }                                                                           \
}

#define FFI_CSRMV_T_WARP(SUFFIX, WEIGHT_C_T)                         \
void csrmv_t_warp##SUFFIX(                                           \
    const BE::Tensor weights, const BE::Tensor indices,      \
    const BE::Tensor indptr,  const BE::Tensor vector,       \
    BE::Tensor output,  int64_t stream                     \
) {                                                                  \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);        \
    int m           = static_cast<int>(indptr.size(0)) - 1;          \
    int k           = static_cast<int>(output.size(0));              \
    int is_homo     = (weights.size(0) == 1) ? 1 : 0;                \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr()); \
    cudaMemsetAsync(d_out, 0, (size_t)k * sizeof(WEIGHT_C_T), s);    \
    _csrmv_t_warp_kern##SUFFIX<<<m, 32, 0, s>>>(                     \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),          \
        static_cast<const int32_t*>(indices.data_ptr()),             \
        static_cast<const int32_t*>(indptr.data_ptr()),              \
        static_cast<const WEIGHT_C_T*>(vector.data_ptr()),           \
        d_out, m, is_homo);                                          \
}

// SpMV FFI Instantiations
// @BE csrmv_nt_thread_f32
FFI_CSRMV_NT_THREAD(_f32, float)
// @BE csrmv_nt_warp_f32
FFI_CSRMV_NT_WARP(_f32, float)
// @BE csrmv_nt_block_f32
FFI_CSRMV_NT_BLOCK(_f32, float, 8 * sizeof(float))
// @BE csrmv_nt_auto_f32
FFI_CSRMV_NT_AUTO(_f32, float, 8 * sizeof(float))
// @BE csrmv_t_warp_f32
FFI_CSRMV_T_WARP(_f32, float)
// @BE csrmv_nt_thread_f64
FFI_CSRMV_NT_THREAD(_f64, double)
// @BE csrmv_nt_warp_f64
FFI_CSRMV_NT_WARP(_f64, double)
// @BE csrmv_nt_block_f64
FFI_CSRMV_NT_BLOCK(_f64, double, 8 * sizeof(double))
// @BE csrmv_nt_auto_f64
FFI_CSRMV_NT_AUTO(_f64, double, 8 * sizeof(double))
// @BE csrmv_t_warp_f64
FFI_CSRMV_T_WARP(_f64, double)
// @BE csrmv_nt_thread_f16
FFI_CSRMV_NT_THREAD(_f16, __half)
// @BE csrmv_nt_warp_f16
FFI_CSRMV_NT_WARP(_f16, __half)
// @BE csrmv_nt_block_f16
FFI_CSRMV_NT_BLOCK(_f16, __half, 8 * sizeof(float))
// @BE csrmv_nt_auto_f16
FFI_CSRMV_NT_AUTO(_f16, __half, 8 * sizeof(float))
// @BE csrmv_t_warp_f16
FFI_CSRMV_T_WARP(_f16, __half)
// @BE csrmv_nt_thread_bf16
FFI_CSRMV_NT_THREAD(_bf16, __nv_bfloat16)
// @BE csrmv_nt_warp_bf16
FFI_CSRMV_NT_WARP(_bf16, __nv_bfloat16)
// @BE csrmv_nt_block_bf16
FFI_CSRMV_NT_BLOCK(_bf16, __nv_bfloat16, 8 * sizeof(float))
// @BE csrmv_nt_auto_bf16
FFI_CSRMV_NT_AUTO(_bf16, __nv_bfloat16, 8 * sizeof(float))
// @BE csrmv_t_warp_bf16
FFI_CSRMV_T_WARP(_bf16, __nv_bfloat16)
