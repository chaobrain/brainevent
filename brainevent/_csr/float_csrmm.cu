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
 * float_csrmm.cu -- Float-Weighted CSR Sparse Matrix-Matrix Product (SpMM) CUDA Kernels
 * ======================================================================================
 *
 * This module provides optimized CUDA kernels for standard (non-event-driven)
 * sparse matrix-matrix multiplication in Compressed Sparse Row (CSR) format.
 *
 * Operator: csrmm
 *   C = A * B          (non-transpose, NT)
 *   C = A^T * B        (transpose, T)
 *
 * Where A is an m×k CSR sparse matrix and B is a dense k×n (NT) or m×n (T) matrix.
 *
 * Kernel Variants:
 *   - csrmm_nt_warp_{homo,hetero}_{f32,f64,f16,bf16}  : one warp per (row, col_strip)
 *   - csrmm_nt_block_{homo,hetero}_{f32,f64,f16,bf16} : one block per (row, col_strip)
 *   - csrmm_nt_auto_{homo,hetero}_{f32,f64,f16,bf16}  : auto-selects warp/block
 *   - csrmm_t_warp_{homo,hetero}_{f32,f64,f16,bf16}   : transpose scatter
 *
 * Performance (NT mode, hetero weights, 10K×10K @ 2% density, n=128):
 *   - tvmffi backend: 1.19-1.34ms (8.4-9.5× faster than cuSPARSE)
 *   - Threshold warp→block: avg_nnz > 64
 *
 * Parameters (TVM FFI entry points):
 *   weights  : [nnz] or [1] float array  (hetero or homo weights)
 *   indices  : [nnz] int32 column indices
 *   indptr   : [m+1] int32 row pointers
 *   B        : [k, n] (NT) or [m, n] (T) float input matrix
 *   C        : [m, n] (NT) or [k, n] (T) float output matrix
 *   stream   : int64 CUDA stream handle
 * ======================================================================================
 */

#include "cuda_common.h"

// =========================================================================
// atomicAdd wrappers
// =========================================================================

#define ATOMIC_ADD_F32(ptr, v)   atomicAdd(ptr, v)
#define ATOMIC_ADD_F64(ptr, v)   atomicAdd(ptr, v)
#define ATOMIC_ADD_F16(ptr, v)   atomicAdd(ptr, __float2half(v))
#define ATOMIC_ADD_BF16(ptr, v)  atomicAdd(ptr, __float2bfloat16(v))

// =========================================================================
// CSR Matrix-Matrix Multiplication (csrmm) - Homogeneous Weights
// =========================================================================

#define DEFINE_CSRMM_NT_WARP_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, \
                                   ACC_ZERO)                                \
__global__ void _csrmm_nt_warp_homo_kern##SUFFIX(                           \
    const WEIGHT_T* __restrict__ weights,                                   \
    const int32_t*  __restrict__ indices,                                   \
    const int32_t*  __restrict__ indptr,                                    \
    const WEIGHT_T* __restrict__ B,                                         \
    WEIGHT_T*       __restrict__ C,                                         \
    int m, int n                                                            \
) {                                                                         \
    int row       = blockIdx.x;                                             \
    int col_start = blockIdx.y * 32;                                        \
    int c         = col_start + (int)threadIdx.x;                           \
    if (row >= m || c >= n) return;                                         \
    int start = indptr[row], end = indptr[row + 1];                         \
    ACC_T w   = READ_W(weights[0]);                                         \
    ACC_T acc = ACC_ZERO;                                                   \
    for (int j = start; j < end; j++) {                                     \
        acc += w * READ_W(B[indices[j] * n + c]);                           \
    }                                                                       \
    C[row * n + c] = WRITE_W(acc);                                          \
}

#define DEFINE_CSRMM_NT_BLOCK_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, \
                                    ACC_ZERO)                                \
__global__ void _csrmm_nt_block_homo_kern##SUFFIX(                           \
    const WEIGHT_T* __restrict__ weights,                                    \
    const int32_t*  __restrict__ indices,                                    \
    const int32_t*  __restrict__ indptr,                                     \
    const WEIGHT_T* __restrict__ B,                                          \
    WEIGHT_T*       __restrict__ C,                                          \
    int m, int n                                                             \
) {                                                                          \
    extern __shared__ char _smem_bytes[];                                    \
    ACC_T* smem = reinterpret_cast<ACC_T*>(_smem_bytes);                     \
    int row       = blockIdx.x;                                              \
    int col_start = blockIdx.y * 32;                                         \
    int lane      = threadIdx.x & 31;                                        \
    int strip     = threadIdx.x >> 5;                                        \
    int c         = col_start + lane;                                        \
    if (row >= m) return;                                                    \
    int start = indptr[row], end = indptr[row + 1];                          \
    ACC_T w   = READ_W(weights[0]);                                          \
    ACC_T acc = ACC_ZERO;                                                    \
    if (c < n) {                                                             \
        for (int j = start + strip; j < end; j += 8) {                       \
            acc += w * READ_W(B[indices[j] * n + c]);                        \
        }                                                                    \
    }                                                                        \
    smem[strip * 32 + lane] = acc;                                           \
    __syncthreads();                                                         \
    if (strip == 0 && c < n) {                                               \
        acc = ACC_ZERO;                                                      \
        for (int s = 0; s < 8; s++) acc += smem[s * 32 + lane];              \
        C[row * n + c] = WRITE_W(acc);                                       \
    }                                                                        \
}

#define DEFINE_CSRMM_T_WARP_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, \
                                  ATOMIC_ADD_W, ACC_ZERO)                  \
__global__ void _csrmm_t_warp_homo_kern##SUFFIX(                           \
    const WEIGHT_T* __restrict__ weights,                                  \
    const int32_t*  __restrict__ indices,                                  \
    const int32_t*  __restrict__ indptr,                                   \
    const WEIGHT_T* __restrict__ B,                                        \
    WEIGHT_T*       __restrict__ C,                                        \
    int m, int n                                                           \
) {                                                                        \
    int row       = blockIdx.x;                                            \
    int col_start = blockIdx.y * 32;                                       \
    int c         = col_start + (int)threadIdx.x;                          \
    if (row >= m || c >= n) return;                                        \
    ACC_T b_val   = READ_W(B[row * n + c]);                                \
    ACC_T w       = READ_W(weights[0]);                                    \
    ACC_T contrib = w * b_val;                                             \
    int start = indptr[row], end = indptr[row + 1];                        \
    for (int j = start; j < end; j++) {                                    \
        ATOMIC_ADD_W(&C[indices[j] * n + c], contrib);                     \
    }                                                                      \
}

// =========================================================================
// CSR Matrix-Matrix Multiplication (csrmm) - Heterogeneous Weights
// =========================================================================

#define DEFINE_CSRMM_NT_WARP_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, \
                                     ACC_ZERO)                                \
__global__ void _csrmm_nt_warp_hetero_kern##SUFFIX(                           \
    const WEIGHT_T* __restrict__ weights,                                     \
    const int32_t*  __restrict__ indices,                                     \
    const int32_t*  __restrict__ indptr,                                      \
    const WEIGHT_T* __restrict__ B,                                           \
    WEIGHT_T*       __restrict__ C,                                           \
    int m, int n                                                              \
) {                                                                           \
    int row       = blockIdx.x;                                               \
    int col_start = blockIdx.y * 32;                                          \
    int c         = col_start + (int)threadIdx.x;                             \
    if (row >= m || c >= n) return;                                           \
    int start = indptr[row], end = indptr[row + 1];                           \
    ACC_T acc = ACC_ZERO;                                                     \
    for (int j = start; j < end; j++) {                                       \
        acc += READ_W(weights[j]) * READ_W(B[indices[j] * n + c]);            \
    }                                                                         \
    C[row * n + c] = WRITE_W(acc);                                            \
}

#define DEFINE_CSRMM_NT_BLOCK_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, \
                                      ACC_ZERO)                                \
__global__ void _csrmm_nt_block_hetero_kern##SUFFIX(                           \
    const WEIGHT_T* __restrict__ weights,                                      \
    const int32_t*  __restrict__ indices,                                      \
    const int32_t*  __restrict__ indptr,                                       \
    const WEIGHT_T* __restrict__ B,                                            \
    WEIGHT_T*       __restrict__ C,                                            \
    int m, int n                                                               \
) {                                                                            \
    extern __shared__ char _smem_bytes[];                                      \
    ACC_T* smem = reinterpret_cast<ACC_T*>(_smem_bytes);                       \
    int row       = blockIdx.x;                                                \
    int col_start = blockIdx.y * 32;                                           \
    int lane      = threadIdx.x & 31;                                          \
    int strip     = threadIdx.x >> 5;                                          \
    int c         = col_start + lane;                                          \
    if (row >= m) return;                                                      \
    int start = indptr[row], end = indptr[row + 1];                            \
    ACC_T acc = ACC_ZERO;                                                      \
    if (c < n) {                                                               \
        for (int j = start + strip; j < end; j += 8) {                         \
            acc += READ_W(weights[j]) * READ_W(B[indices[j] * n + c]);         \
        }                                                                      \
    }                                                                          \
    smem[strip * 32 + lane] = acc;                                             \
    __syncthreads();                                                           \
    if (strip == 0 && c < n) {                                                 \
        acc = ACC_ZERO;                                                        \
        for (int s = 0; s < 8; s++) acc += smem[s * 32 + lane];                \
        C[row * n + c] = WRITE_W(acc);                                         \
    }                                                                          \
}

#define DEFINE_CSRMM_T_WARP_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, \
                                    ATOMIC_ADD_W, ACC_ZERO)                  \
__global__ void _csrmm_t_warp_hetero_kern##SUFFIX(                           \
    const WEIGHT_T* __restrict__ weights,                                    \
    const int32_t*  __restrict__ indices,                                    \
    const int32_t*  __restrict__ indptr,                                     \
    const WEIGHT_T* __restrict__ B,                                          \
    WEIGHT_T*       __restrict__ C,                                          \
    int m, int n                                                             \
) {                                                                          \
    int row       = blockIdx.x;                                              \
    int col_start = blockIdx.y * 32;                                         \
    int c         = col_start + (int)threadIdx.x;                            \
    if (row >= m || c >= n) return;                                          \
    ACC_T b_val = READ_W(B[row * n + c]);                                    \
    int start = indptr[row], end = indptr[row + 1];                          \
    for (int j = start; j < end; j++) {                                      \
        ATOMIC_ADD_W(&C[indices[j] * n + c], READ_W(weights[j]) * b_val);    \
    }                                                                        \
}

// =========================================================================
// Kernel Instantiations - Homogeneous
// =========================================================================

DEFINE_CSRMM_NT_WARP_HOMO(_f32,  float,         float,  READ_F32,  WRITE_F32,  \
                           ZERO_F32)
DEFINE_CSRMM_NT_BLOCK_HOMO(_f32, float,         float,  READ_F32,  WRITE_F32,  \
                            ZERO_F32)
DEFINE_CSRMM_T_WARP_HOMO(_f32,   float,         float,  READ_F32,  WRITE_F32,  \
                          ATOMIC_ADD_F32, ZERO_F32)

DEFINE_CSRMM_NT_WARP_HOMO(_f64,  double,        double, READ_F64,  WRITE_F64,  \
                           ZERO_F64)
DEFINE_CSRMM_NT_BLOCK_HOMO(_f64, double,        double, READ_F64,  WRITE_F64,  \
                            ZERO_F64)
DEFINE_CSRMM_T_WARP_HOMO(_f64,   double,        double, READ_F64,  WRITE_F64,  \
                          ATOMIC_ADD_F64, ZERO_F64)

DEFINE_CSRMM_NT_WARP_HOMO(_f16,  __half,        float,  READ_F16,  WRITE_F16,  \
                           ZERO_F16)
DEFINE_CSRMM_NT_BLOCK_HOMO(_f16, __half,        float,  READ_F16,  WRITE_F16,  \
                            ZERO_F16)
DEFINE_CSRMM_T_WARP_HOMO(_f16,   __half,        float,  READ_F16,  WRITE_F16,  \
                          ATOMIC_ADD_F16, ZERO_F16)

DEFINE_CSRMM_NT_WARP_HOMO(_bf16,  __nv_bfloat16, float, READ_BF16, WRITE_BF16, \
                           ZERO_BF16)
DEFINE_CSRMM_NT_BLOCK_HOMO(_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, \
                            ZERO_BF16)
DEFINE_CSRMM_T_WARP_HOMO(_bf16,   __nv_bfloat16, float, READ_BF16, WRITE_BF16, \
                          ATOMIC_ADD_BF16, ZERO_BF16)

// =========================================================================
// Kernel Instantiations - Heterogeneous
// =========================================================================

DEFINE_CSRMM_NT_WARP_HETERO(_f32,  float,         float,  READ_F32,  WRITE_F32,  \
                             ZERO_F32)
DEFINE_CSRMM_NT_BLOCK_HETERO(_f32, float,         float,  READ_F32,  WRITE_F32,  \
                              ZERO_F32)
DEFINE_CSRMM_T_WARP_HETERO(_f32,   float,         float,  READ_F32,  WRITE_F32,  \
                            ATOMIC_ADD_F32, ZERO_F32)

DEFINE_CSRMM_NT_WARP_HETERO(_f64,  double,        double, READ_F64,  WRITE_F64,  \
                             ZERO_F64)
DEFINE_CSRMM_NT_BLOCK_HETERO(_f64, double,        double, READ_F64,  WRITE_F64,  \
                              ZERO_F64)
DEFINE_CSRMM_T_WARP_HETERO(_f64,   double,        double, READ_F64,  WRITE_F64,  \
                            ATOMIC_ADD_F64, ZERO_F64)

DEFINE_CSRMM_NT_WARP_HETERO(_f16,  __half,        float,  READ_F16,  WRITE_F16,  \
                             ZERO_F16)
DEFINE_CSRMM_NT_BLOCK_HETERO(_f16, __half,        float,  READ_F16,  WRITE_F16,  \
                              ZERO_F16)
DEFINE_CSRMM_T_WARP_HETERO(_f16,   __half,        float,  READ_F16,  WRITE_F16,  \
                            ATOMIC_ADD_F16, ZERO_F16)

DEFINE_CSRMM_NT_WARP_HETERO(_bf16,  __nv_bfloat16, float, READ_BF16, WRITE_BF16, \
                             ZERO_BF16)
DEFINE_CSRMM_NT_BLOCK_HETERO(_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, \
                              ZERO_BF16)
DEFINE_CSRMM_T_WARP_HETERO(_bf16,   __nv_bfloat16, float, READ_BF16, WRITE_BF16, \
                            ATOMIC_ADD_BF16, ZERO_BF16)

// =========================================================================
// FFI Entry Point Macros - Homogeneous
// =========================================================================

#define FFI_CSRMM_NT_WARP_HOMO(SUFFIX, WEIGHT_C_T)              \
void csrmm_nt_warp_homo##SUFFIX(                                \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices, \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView B,       \
    tvm::ffi::TensorView C,       int64_t stream                \
) {                                                             \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);   \
    int m           = static_cast<int>(indptr.size(0)) - 1;     \
    int n           = static_cast<int>(B.size(1));              \
    int c_blocks    = (n + 31) / 32;                            \
    dim3 grid(m, c_blocks);                                     \
    _csrmm_nt_warp_homo_kern##SUFFIX<<<grid, 32, 0, s>>>(       \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),     \
        static_cast<const int32_t*>(indices.data_ptr()),        \
        static_cast<const int32_t*>(indptr.data_ptr()),         \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),           \
        static_cast<WEIGHT_C_T*>(C.data_ptr()),                 \
        m, n);                                                  \
}

#define FFI_CSRMM_NT_BLOCK_HOMO(SUFFIX, WEIGHT_C_T, SHM_SIZE)      \
void csrmm_nt_block_homo##SUFFIX(                                  \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,    \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView B,          \
    tvm::ffi::TensorView C,       int64_t stream                   \
) {                                                                \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);      \
    int m           = static_cast<int>(indptr.size(0)) - 1;        \
    int n           = static_cast<int>(B.size(1));                 \
    int c_blocks    = (n + 31) / 32;                               \
    dim3 grid(m, c_blocks);                                        \
    _csrmm_nt_block_homo_kern##SUFFIX<<<grid, 256, SHM_SIZE, s>>>( \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),        \
        static_cast<const int32_t*>(indices.data_ptr()),           \
        static_cast<const int32_t*>(indptr.data_ptr()),            \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),              \
        static_cast<WEIGHT_C_T*>(C.data_ptr()),                    \
        m, n);                                                     \
}

#define FFI_CSRMM_NT_AUTO_HOMO(SUFFIX, WEIGHT_C_T, SHM_SIZE)                    \
void csrmm_nt_auto_homo##SUFFIX(                                                \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                 \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView B,                       \
    tvm::ffi::TensorView C,       int64_t stream                                \
) {                                                                             \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                   \
    int m           = static_cast<int>(indptr.size(0)) - 1;                     \
    int nse         = static_cast<int>(indices.size(0));                        \
    int n           = static_cast<int>(B.size(1));                              \
    int avg_nnz     = (m > 0) ? (nse / m) : 0;                                  \
    int c_blocks    = (n + 31) / 32;                                            \
    dim3 grid(m, c_blocks);                                                     \
    const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*    d_i = static_cast<const int32_t*>(indices.data_ptr());    \
    const int32_t*    d_p = static_cast<const int32_t*>(indptr.data_ptr());     \
    const WEIGHT_C_T* d_b = static_cast<const WEIGHT_C_T*>(B.data_ptr());       \
    WEIGHT_C_T*       d_c = static_cast<WEIGHT_C_T*>(C.data_ptr());             \
    if (avg_nnz <= 64) {                                                        \
        _csrmm_nt_warp_homo_kern##SUFFIX<<<grid, 32, 0, s>>>(                   \
            d_w, d_i, d_p, d_b, d_c, m, n);                                     \
    } else {                                                                    \
        _csrmm_nt_block_homo_kern##SUFFIX<<<grid, 256, SHM_SIZE, s>>>(          \
            d_w, d_i, d_p, d_b, d_c, m, n);                                     \
    }                                                                           \
}

#define FFI_CSRMM_T_WARP_HOMO(SUFFIX, WEIGHT_C_T)                           \
void csrmm_t_warp_homo##SUFFIX(                                             \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,             \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView B,                   \
    tvm::ffi::TensorView C,       int64_t stream                            \
) {                                                                         \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);               \
    int m           = static_cast<int>(indptr.size(0)) - 1;                 \
    int n           = static_cast<int>(B.size(1));                          \
    int k           = static_cast<int>(C.size(0));                          \
    WEIGHT_C_T* d_c = static_cast<WEIGHT_C_T*>(C.data_ptr());               \
    cudaMemsetAsync(d_c, 0, (size_t)k * (size_t)n * sizeof(WEIGHT_C_T), s); \
    int c_blocks    = (n + 31) / 32;                                        \
    dim3 grid(m, c_blocks);                                                 \
    _csrmm_t_warp_homo_kern##SUFFIX<<<grid, 32, 0, s>>>(                    \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                 \
        static_cast<const int32_t*>(indices.data_ptr()),                    \
        static_cast<const int32_t*>(indptr.data_ptr()),                     \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                       \
        d_c, m, n);                                                         \
}

// =========================================================================
// FFI Entry Point Macros - Heterogeneous
// =========================================================================

#define FFI_CSRMM_NT_WARP_HETERO(SUFFIX, WEIGHT_C_T)            \
void csrmm_nt_warp_hetero##SUFFIX(                              \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices, \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView B,       \
    tvm::ffi::TensorView C,       int64_t stream                \
) {                                                             \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);   \
    int m           = static_cast<int>(indptr.size(0)) - 1;     \
    int n           = static_cast<int>(B.size(1));              \
    int c_blocks    = (n + 31) / 32;                            \
    dim3 grid(m, c_blocks);                                     \
    _csrmm_nt_warp_hetero_kern##SUFFIX<<<grid, 32, 0, s>>>(     \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),     \
        static_cast<const int32_t*>(indices.data_ptr()),        \
        static_cast<const int32_t*>(indptr.data_ptr()),         \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),           \
        static_cast<WEIGHT_C_T*>(C.data_ptr()),                 \
        m, n);                                                  \
}

#define FFI_CSRMM_NT_BLOCK_HETERO(SUFFIX, WEIGHT_C_T, SHM_SIZE)      \
void csrmm_nt_block_hetero##SUFFIX(                                  \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,      \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView B,            \
    tvm::ffi::TensorView C,       int64_t stream                     \
) {                                                                  \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);        \
    int m           = static_cast<int>(indptr.size(0)) - 1;          \
    int n           = static_cast<int>(B.size(1));                   \
    int c_blocks    = (n + 31) / 32;                                 \
    dim3 grid(m, c_blocks);                                          \
    _csrmm_nt_block_hetero_kern##SUFFIX<<<grid, 256, SHM_SIZE, s>>>( \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),          \
        static_cast<const int32_t*>(indices.data_ptr()),             \
        static_cast<const int32_t*>(indptr.data_ptr()),              \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                \
        static_cast<WEIGHT_C_T*>(C.data_ptr()),                      \
        m, n);                                                       \
}

#define FFI_CSRMM_NT_AUTO_HETERO(SUFFIX, WEIGHT_C_T, SHM_SIZE)                  \
void csrmm_nt_auto_hetero##SUFFIX(                                              \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                 \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView B,                       \
    tvm::ffi::TensorView C,       int64_t stream                                \
) {                                                                             \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                   \
    int m           = static_cast<int>(indptr.size(0)) - 1;                     \
    int nse         = static_cast<int>(indices.size(0));                        \
    int n           = static_cast<int>(B.size(1));                              \
    int avg_nnz     = (m > 0) ? (nse / m) : 0;                                  \
    int c_blocks    = (n + 31) / 32;                                            \
    dim3 grid(m, c_blocks);                                                     \
    const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*    d_i = static_cast<const int32_t*>(indices.data_ptr());    \
    const int32_t*    d_p = static_cast<const int32_t*>(indptr.data_ptr());     \
    const WEIGHT_C_T* d_b = static_cast<const WEIGHT_C_T*>(B.data_ptr());       \
    WEIGHT_C_T*       d_c = static_cast<WEIGHT_C_T*>(C.data_ptr());             \
    if (avg_nnz <= 64) {                                                        \
        _csrmm_nt_warp_hetero_kern##SUFFIX<<<grid, 32, 0, s>>>(                 \
            d_w, d_i, d_p, d_b, d_c, m, n);                                     \
    } else {                                                                    \
        _csrmm_nt_block_hetero_kern##SUFFIX<<<grid, 256, SHM_SIZE, s>>>(        \
            d_w, d_i, d_p, d_b, d_c, m, n);                                     \
    }                                                                           \
}

#define FFI_CSRMM_T_WARP_HETERO(SUFFIX, WEIGHT_C_T)                         \
void csrmm_t_warp_hetero##SUFFIX(                                           \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,             \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView B,                   \
    tvm::ffi::TensorView C,       int64_t stream                            \
) {                                                                         \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);               \
    int m           = static_cast<int>(indptr.size(0)) - 1;                 \
    int n           = static_cast<int>(B.size(1));                          \
    int k           = static_cast<int>(C.size(0));                          \
    WEIGHT_C_T* d_c = static_cast<WEIGHT_C_T*>(C.data_ptr());               \
    cudaMemsetAsync(d_c, 0, (size_t)k * (size_t)n * sizeof(WEIGHT_C_T), s); \
    int c_blocks    = (n + 31) / 32;                                        \
    dim3 grid(m, c_blocks);                                                 \
    _csrmm_t_warp_hetero_kern##SUFFIX<<<grid, 32, 0, s>>>(                  \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                 \
        static_cast<const int32_t*>(indices.data_ptr()),                    \
        static_cast<const int32_t*>(indptr.data_ptr()),                     \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                       \
        d_c, m, n);                                                         \
}

// =========================================================================
// FFI Instantiations - Homogeneous
// =========================================================================

// @tvm_ffi csrmm_nt_warp_homo_f32
FFI_CSRMM_NT_WARP_HOMO(_f32, float)
// @tvm_ffi csrmm_nt_block_homo_f32
FFI_CSRMM_NT_BLOCK_HOMO(_f32, float, 8 * 32 * sizeof(float))
// @tvm_ffi csrmm_nt_auto_homo_f32
FFI_CSRMM_NT_AUTO_HOMO(_f32, float, 8 * 32 * sizeof(float))
// @tvm_ffi csrmm_t_warp_homo_f32
FFI_CSRMM_T_WARP_HOMO(_f32, float)

// @tvm_ffi csrmm_nt_warp_homo_f64
FFI_CSRMM_NT_WARP_HOMO(_f64, double)
// @tvm_ffi csrmm_nt_block_homo_f64
FFI_CSRMM_NT_BLOCK_HOMO(_f64, double, 8 * 32 * sizeof(double))
// @tvm_ffi csrmm_nt_auto_homo_f64
FFI_CSRMM_NT_AUTO_HOMO(_f64, double, 8 * 32 * sizeof(double))
// @tvm_ffi csrmm_t_warp_homo_f64
FFI_CSRMM_T_WARP_HOMO(_f64, double)

// @tvm_ffi csrmm_nt_warp_homo_f16
FFI_CSRMM_NT_WARP_HOMO(_f16, __half)
// @tvm_ffi csrmm_nt_block_homo_f16
FFI_CSRMM_NT_BLOCK_HOMO(_f16, __half, 8 * 32 * sizeof(float))
// @tvm_ffi csrmm_nt_auto_homo_f16
FFI_CSRMM_NT_AUTO_HOMO(_f16, __half, 8 * 32 * sizeof(float))
// @tvm_ffi csrmm_t_warp_homo_f16
FFI_CSRMM_T_WARP_HOMO(_f16, __half)

// @tvm_ffi csrmm_nt_warp_homo_bf16
FFI_CSRMM_NT_WARP_HOMO(_bf16, __nv_bfloat16)
// @tvm_ffi csrmm_nt_block_homo_bf16
FFI_CSRMM_NT_BLOCK_HOMO(_bf16, __nv_bfloat16, 8 * 32 * sizeof(float))
// @tvm_ffi csrmm_nt_auto_homo_bf16
FFI_CSRMM_NT_AUTO_HOMO(_bf16, __nv_bfloat16, 8 * 32 * sizeof(float))
// @tvm_ffi csrmm_t_warp_homo_bf16
FFI_CSRMM_T_WARP_HOMO(_bf16, __nv_bfloat16)

// =========================================================================
// FFI Instantiations - Heterogeneous
// =========================================================================

// @tvm_ffi csrmm_nt_warp_hetero_f32
FFI_CSRMM_NT_WARP_HETERO(_f32, float)
// @tvm_ffi csrmm_nt_block_hetero_f32
FFI_CSRMM_NT_BLOCK_HETERO(_f32, float, 8 * 32 * sizeof(float))
// @tvm_ffi csrmm_nt_auto_hetero_f32
FFI_CSRMM_NT_AUTO_HETERO(_f32, float, 8 * 32 * sizeof(float))
// @tvm_ffi csrmm_t_warp_hetero_f32
FFI_CSRMM_T_WARP_HETERO(_f32, float)

// @tvm_ffi csrmm_nt_warp_hetero_f64
FFI_CSRMM_NT_WARP_HETERO(_f64, double)
// @tvm_ffi csrmm_nt_block_hetero_f64
FFI_CSRMM_NT_BLOCK_HETERO(_f64, double, 8 * 32 * sizeof(double))
// @tvm_ffi csrmm_nt_auto_hetero_f64
FFI_CSRMM_NT_AUTO_HETERO(_f64, double, 8 * 32 * sizeof(double))
// @tvm_ffi csrmm_t_warp_hetero_f64
FFI_CSRMM_T_WARP_HETERO(_f64, double)

// @tvm_ffi csrmm_nt_warp_hetero_f16
FFI_CSRMM_NT_WARP_HETERO(_f16, __half)
// @tvm_ffi csrmm_nt_block_hetero_f16
FFI_CSRMM_NT_BLOCK_HETERO(_f16, __half, 8 * 32 * sizeof(float))
// @tvm_ffi csrmm_nt_auto_hetero_f16
FFI_CSRMM_NT_AUTO_HETERO(_f16, __half, 8 * 32 * sizeof(float))
// @tvm_ffi csrmm_t_warp_hetero_f16
FFI_CSRMM_T_WARP_HETERO(_f16, __half)

// @tvm_ffi csrmm_nt_warp_hetero_bf16
FFI_CSRMM_NT_WARP_HETERO(_bf16, __nv_bfloat16)
// @tvm_ffi csrmm_nt_block_hetero_bf16
FFI_CSRMM_NT_BLOCK_HETERO(_bf16, __nv_bfloat16, 8 * 32 * sizeof(float))
// @tvm_ffi csrmm_nt_auto_hetero_bf16
FFI_CSRMM_NT_AUTO_HETERO(_bf16, __nv_bfloat16, 8 * 32 * sizeof(float))
// @tvm_ffi csrmm_t_warp_hetero_bf16
FFI_CSRMM_T_WARP_HETERO(_bf16, __nv_bfloat16)

