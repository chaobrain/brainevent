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
 * sparse_float_csrmm.cu -- Sparse-Float CSR Sparse Matrix-Matrix CUDA Kernels
 * ============================================================================
 *
 * This module provides optimized CUDA kernels for sparse-float SpMM operations
 * in Compressed Sparse Row (CSR) format.
 *
 * Operator: spfloat_csrmm
 *   Computes C = A @ B  (non-transpose) or C = A.T @ B  (transpose), where A
 *   is a CSR sparse matrix and B is a dense matrix.  Only non-zero entries in
 *   B contribute to the result (sparse-float semantics).
 */

#include "cuda_common.h"

// =========================================================================
// CSR Matrix-Matrix Multiplication (csrmm)
// =========================================================================

#define DEFINE_SPFLOAT_CSRMM_NT_WARP_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _spfloat_csrmm_nt_warp_homo_kern##SUFFIX(                                     \
    const WEIGHT_T* __restrict__ weights,                                                     \
    const int32_t*  __restrict__ indices,                                                     \
    const int32_t*  __restrict__ indptr,                                                      \
    const WEIGHT_T* __restrict__ B,                                                           \
    WEIGHT_T*       __restrict__ C,                                                           \
    int m, int n                                                                              \
) {                                                                                           \
    int row       = blockIdx.x;                                                               \
    int col_start = blockIdx.y * 32;                                                          \
    int c         = col_start + (int)threadIdx.x;                                             \
    if (row >= m || c >= n) return;                                                           \
    int start = indptr[row], end = indptr[row + 1];                                           \
    ACC_T acc = ACC_ZERO;                                                                     \
    ACC_T w = READ_W(__ldg(&weights[0]));                                                     \
    int j = start;                                                                            \
    for (; j + 3 < end; j += 4) {                                                             \
        int col0 = __ldg(&indices[j]);                                                        \
        int col1 = __ldg(&indices[j+1]);                                                      \
        int col2 = __ldg(&indices[j+2]);                                                      \
        int col3 = __ldg(&indices[j+3]);                                                      \
        ACC_T b0 = READ_W(__ldg(&B[col0 * n + c]));                                           \
        ACC_T b1 = READ_W(__ldg(&B[col1 * n + c]));                                           \
        ACC_T b2 = READ_W(__ldg(&B[col2 * n + c]));                                           \
        ACC_T b3 = READ_W(__ldg(&B[col3 * n + c]));                                           \
        acc += w * b0;                                                                        \
        acc += w * b1;                                                                        \
        acc += w * b2;                                                                        \
        acc += w * b3;                                                                        \
    }                                                                                         \
    for (; j < end; j++) {                                                                    \
        int col = __ldg(&indices[j]);                                                         \
        ACC_T b_val = READ_W(__ldg(&B[col * n + c]));                                         \
        acc += w * b_val;                                                                     \
    }                                                                                         \
    C[row * n + c] = WRITE_W(acc);                                                            \
}

#define DEFINE_SPFLOAT_CSRMM_NT_WARP_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _spfloat_csrmm_nt_warp_hetero_kern##SUFFIX(                                     \
    const WEIGHT_T* __restrict__ weights,                                                       \
    const int32_t*  __restrict__ indices,                                                       \
    const int32_t*  __restrict__ indptr,                                                        \
    const WEIGHT_T* __restrict__ B,                                                             \
    WEIGHT_T*       __restrict__ C,                                                             \
    int m, int n                                                                                \
) {                                                                                             \
    int row       = blockIdx.x;                                                                 \
    int col_start = blockIdx.y * 32;                                                            \
    int c         = col_start + (int)threadIdx.x;                                               \
    if (row >= m || c >= n) return;                                                             \
    int start = indptr[row], end = indptr[row + 1];                                             \
    ACC_T acc = ACC_ZERO;                                                                       \
    int j = start;                                                                              \
    for (; j + 3 < end; j += 4) {                                                               \
        int col0 = __ldg(&indices[j]);                                                          \
        int col1 = __ldg(&indices[j+1]);                                                        \
        int col2 = __ldg(&indices[j+2]);                                                        \
        int col3 = __ldg(&indices[j+3]);                                                        \
        ACC_T w0 = READ_W(__ldg(&weights[j]));                                                  \
        ACC_T w1 = READ_W(__ldg(&weights[j+1]));                                                \
        ACC_T w2 = READ_W(__ldg(&weights[j+2]));                                                \
        ACC_T w3 = READ_W(__ldg(&weights[j+3]));                                                \
        ACC_T b0 = READ_W(__ldg(&B[col0 * n + c]));                                             \
        ACC_T b1 = READ_W(__ldg(&B[col1 * n + c]));                                             \
        ACC_T b2 = READ_W(__ldg(&B[col2 * n + c]));                                             \
        ACC_T b3 = READ_W(__ldg(&B[col3 * n + c]));                                             \
        acc += w0 * b0 + w1 * b1 + w2 * b2 + w3 * b3;                                           \
    }                                                                                           \
    for (; j < end; j++) {                                                                      \
        int col = __ldg(&indices[j]);                                                           \
        ACC_T b_val = READ_W(__ldg(&B[col * n + c]));                                           \
        acc += READ_W(__ldg(&weights[j])) * b_val;                                              \
    }                                                                                           \
    C[row * n + c] = WRITE_W(acc);                                                              \
}

#define DEFINE_SPFLOAT_CSRMM_NT_BLOCK_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _spfloat_csrmm_nt_block_homo_kern##SUFFIX(                                     \
    const WEIGHT_T* __restrict__ weights,                                                      \
    const int32_t*  __restrict__ indices,                                                      \
    const int32_t*  __restrict__ indptr,                                                       \
    const WEIGHT_T* __restrict__ B,                                                            \
    WEIGHT_T*       __restrict__ C,                                                            \
    int m, int n                                                                               \
) {                                                                                            \
    extern __shared__ char _smem_bytes[];                                                      \
    ACC_T* smem = reinterpret_cast<ACC_T*>(_smem_bytes);                                       \
    int row       = blockIdx.x;                                                                \
    int col_start = blockIdx.y * 32;                                                           \
    int lane      = threadIdx.x & 31;                                                          \
    int strip     = threadIdx.x >> 5;                                                          \
    int c         = col_start + lane;                                                          \
    if (row >= m) return;                                                                      \
    int start = indptr[row], end = indptr[row + 1];                                            \
    ACC_T acc = ACC_ZERO;                                                                      \
    if (c < n) {                                                                               \
        ACC_T w = READ_W(__ldg(&weights[0]));                                                  \
        int j = start + strip;                                                                 \
        for (; j + 31 < end; j += 32) {                                                        \
            int col0 = __ldg(&indices[j]);                                                     \
            int col1 = __ldg(&indices[j+8]);                                                   \
            int col2 = __ldg(&indices[j+16]);                                                  \
            int col3 = __ldg(&indices[j+24]);                                                  \
            ACC_T b0 = READ_W(__ldg(&B[col0 * n + c]));                                        \
            ACC_T b1 = READ_W(__ldg(&B[col1 * n + c]));                                        \
            ACC_T b2 = READ_W(__ldg(&B[col2 * n + c]));                                        \
            ACC_T b3 = READ_W(__ldg(&B[col3 * n + c]));                                        \
            acc += w * b0;                                                                     \
            acc += w * b1;                                                                     \
            acc += w * b2;                                                                     \
            acc += w * b3;                                                                     \
        }                                                                                      \
        for (; j < end; j += 8) {                                                              \
            int col = __ldg(&indices[j]);                                                      \
            ACC_T b_val = READ_W(__ldg(&B[col * n + c]));                                      \
            acc += w * b_val;                                                                  \
        }                                                                                      \
    }                                                                                          \
    smem[strip * 32 + lane] = acc;                                                             \
    __syncthreads();                                                                           \
    if (strip == 0 && c < n) {                                                                 \
        acc = smem[lane] + smem[32 + lane] + smem[64 + lane] + smem[96 + lane]                 \
            + smem[128 + lane] + smem[160 + lane] + smem[192 + lane] + smem[224 + lane];       \
        C[row * n + c] = WRITE_W(acc);                                                         \
    }                                                                                          \
}

#define DEFINE_SPFLOAT_CSRMM_NT_BLOCK_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _spfloat_csrmm_nt_block_hetero_kern##SUFFIX(                                     \
    const WEIGHT_T* __restrict__ weights,                                                        \
    const int32_t*  __restrict__ indices,                                                        \
    const int32_t*  __restrict__ indptr,                                                         \
    const WEIGHT_T* __restrict__ B,                                                              \
    WEIGHT_T*       __restrict__ C,                                                              \
    int m, int n                                                                                 \
) {                                                                                              \
    extern __shared__ char _smem_bytes[];                                                        \
    ACC_T* smem = reinterpret_cast<ACC_T*>(_smem_bytes);                                         \
    int row       = blockIdx.x;                                                                  \
    int col_start = blockIdx.y * 32;                                                             \
    int lane      = threadIdx.x & 31;                                                            \
    int strip     = threadIdx.x >> 5;                                                            \
    int c         = col_start + lane;                                                            \
    if (row >= m) return;                                                                        \
    int start = indptr[row], end = indptr[row + 1];                                              \
    ACC_T acc = ACC_ZERO;                                                                        \
    if (c < n) {                                                                                 \
        int j = start + strip;                                                                   \
        for (; j + 31 < end; j += 32) {                                                          \
            int col0 = __ldg(&indices[j]);                                                       \
            int col1 = __ldg(&indices[j+8]);                                                     \
            int col2 = __ldg(&indices[j+16]);                                                    \
            int col3 = __ldg(&indices[j+24]);                                                    \
            ACC_T w0 = READ_W(__ldg(&weights[j]));                                               \
            ACC_T w1 = READ_W(__ldg(&weights[j+8]));                                             \
            ACC_T w2 = READ_W(__ldg(&weights[j+16]));                                            \
            ACC_T w3 = READ_W(__ldg(&weights[j+24]));                                            \
            ACC_T b0 = READ_W(__ldg(&B[col0 * n + c]));                                          \
            ACC_T b1 = READ_W(__ldg(&B[col1 * n + c]));                                          \
            ACC_T b2 = READ_W(__ldg(&B[col2 * n + c]));                                          \
            ACC_T b3 = READ_W(__ldg(&B[col3 * n + c]));                                          \
            acc += w0 * b0 + w1 * b1 + w2 * b2 + w3 * b3;                                        \
        }                                                                                        \
        for (; j < end; j += 8) {                                                                \
            int col = __ldg(&indices[j]);                                                        \
            ACC_T b_val = READ_W(__ldg(&B[col * n + c]));                                        \
            acc += READ_W(__ldg(&weights[j])) * b_val;                                           \
        }                                                                                        \
    }                                                                                            \
    smem[strip * 32 + lane] = acc;                                                               \
    __syncthreads();                                                                             \
    if (strip == 0 && c < n) {                                                                   \
        acc = smem[lane] + smem[32 + lane] + smem[64 + lane] + smem[96 + lane]                   \
            + smem[128 + lane] + smem[160 + lane] + smem[192 + lane] + smem[224 + lane];         \
        C[row * n + c] = WRITE_W(acc);                                                           \
    }                                                                                            \
}

#define DEFINE_SPFLOAT_CSRMM_T_WARP_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, \
                                         ATOMIC_ADD_W, ACC_ZERO)                   \
__global__ void _spfloat_csrmm_t_warp_homo_kern##SUFFIX(                           \
    const WEIGHT_T* __restrict__ weights,                                          \
    const int32_t*  __restrict__ indices,                                          \
    const int32_t*  __restrict__ indptr,                                           \
    const WEIGHT_T* __restrict__ B,                                                \
    WEIGHT_T*       __restrict__ C,                                                \
    int m, int n                                                                   \
) {                                                                                \
    int row       = blockIdx.x;                                                    \
    int col_start = blockIdx.y * 32;                                               \
    int c         = col_start + (int)threadIdx.x;                                  \
    if (row >= m || c >= n) return;                                                \
    ACC_T b_val = READ_W(__ldg(&B[row * n + c]));                                  \
    if (b_val == ACC_ZERO) return;                                                 \
    int start = indptr[row], end = indptr[row + 1];                                \
    ACC_T contrib = READ_W(__ldg(&weights[0])) * b_val;                            \
    for (int j = start; j < end; j++) {                                            \
        ATOMIC_ADD_W(&C[__ldg(&indices[j]) * n + c], contrib);                     \
    }                                                                              \
}

#define DEFINE_SPFLOAT_CSRMM_T_WARP_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W,      \
                                           ATOMIC_ADD_W, ACC_ZERO)                        \
__global__ void _spfloat_csrmm_t_warp_hetero_kern##SUFFIX(                                \
    const WEIGHT_T* __restrict__ weights,                                                 \
    const int32_t*  __restrict__ indices,                                                 \
    const int32_t*  __restrict__ indptr,                                                  \
    const WEIGHT_T* __restrict__ B,                                                       \
    WEIGHT_T*       __restrict__ C,                                                       \
    int m, int n                                                                          \
) {                                                                                       \
    int row       = blockIdx.x;                                                           \
    int col_start = blockIdx.y * 32;                                                      \
    int c         = col_start + (int)threadIdx.x;                                         \
    if (row >= m || c >= n) return;                                                       \
    ACC_T b_val = READ_W(__ldg(&B[row * n + c]));                                         \
    if (b_val == ACC_ZERO) return;                                                        \
    int start = indptr[row], end = indptr[row + 1];                                       \
    for (int j = start; j < end; j++) {                                                   \
        ATOMIC_ADD_W(&C[__ldg(&indices[j]) * n + c], READ_W(__ldg(&weights[j])) * b_val); \
    }                                                                                     \
}

// SpMM Instantiations
// ---- float32 ----
DEFINE_SPFLOAT_CSRMM_NT_WARP_HOMO  (_f32, float, float, READ_F32, WRITE_F32, ZERO_F32)
DEFINE_SPFLOAT_CSRMM_NT_WARP_HETERO(_f32, float, float, READ_F32, WRITE_F32, ZERO_F32)
DEFINE_SPFLOAT_CSRMM_NT_BLOCK_HOMO (_f32, float, float, READ_F32, WRITE_F32, ZERO_F32)
DEFINE_SPFLOAT_CSRMM_NT_BLOCK_HETERO(_f32, float, float, READ_F32, WRITE_F32, ZERO_F32)
DEFINE_SPFLOAT_CSRMM_T_WARP_HOMO   (_f32, float, float, READ_F32, WRITE_F32, atomic_add_f32, ZERO_F32)
DEFINE_SPFLOAT_CSRMM_T_WARP_HETERO (_f32, float, float, READ_F32, WRITE_F32, atomic_add_f32, ZERO_F32)

// ---- float64 ----
DEFINE_SPFLOAT_CSRMM_NT_WARP_HOMO  (_f64, double, double, READ_F64, WRITE_F64, ZERO_F64)
DEFINE_SPFLOAT_CSRMM_NT_WARP_HETERO(_f64, double, double, READ_F64, WRITE_F64, ZERO_F64)
DEFINE_SPFLOAT_CSRMM_NT_BLOCK_HOMO (_f64, double, double, READ_F64, WRITE_F64, ZERO_F64)
DEFINE_SPFLOAT_CSRMM_NT_BLOCK_HETERO(_f64, double, double, READ_F64, WRITE_F64, ZERO_F64)
DEFINE_SPFLOAT_CSRMM_T_WARP_HOMO   (_f64, double, double, READ_F64, WRITE_F64, atomic_add_f64, ZERO_F64)
DEFINE_SPFLOAT_CSRMM_T_WARP_HETERO (_f64, double, double, READ_F64, WRITE_F64, atomic_add_f64, ZERO_F64)

// ---- float16 ----
DEFINE_SPFLOAT_CSRMM_NT_WARP_HOMO  (_f16, __half, float, READ_F16, WRITE_F16, ZERO_F16)
DEFINE_SPFLOAT_CSRMM_NT_WARP_HETERO(_f16, __half, float, READ_F16, WRITE_F16, ZERO_F16)
DEFINE_SPFLOAT_CSRMM_NT_BLOCK_HOMO (_f16, __half, float, READ_F16, WRITE_F16, ZERO_F16)
DEFINE_SPFLOAT_CSRMM_NT_BLOCK_HETERO(_f16, __half, float, READ_F16, WRITE_F16, ZERO_F16)
DEFINE_SPFLOAT_CSRMM_T_WARP_HOMO   (_f16, __half, float, READ_F16, WRITE_F16, atomic_add_f16, ZERO_F16)
DEFINE_SPFLOAT_CSRMM_T_WARP_HETERO (_f16, __half, float, READ_F16, WRITE_F16, atomic_add_f16, ZERO_F16)

// ---- bfloat16 ----
DEFINE_SPFLOAT_CSRMM_NT_WARP_HOMO  (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, ZERO_BF16)
DEFINE_SPFLOAT_CSRMM_NT_WARP_HETERO(_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, ZERO_BF16)
DEFINE_SPFLOAT_CSRMM_NT_BLOCK_HOMO (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, ZERO_BF16)
DEFINE_SPFLOAT_CSRMM_NT_BLOCK_HETERO(_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, ZERO_BF16)
DEFINE_SPFLOAT_CSRMM_T_WARP_HOMO   (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, atomic_add_bf16, ZERO_BF16)
DEFINE_SPFLOAT_CSRMM_T_WARP_HETERO (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, atomic_add_bf16, ZERO_BF16)

// FFI Macros for SpMM
// ---- FFI macro: forward homo warp ----
#define FFI_SPFLOAT_CSRMM_NT_HOMO_WARP(SUFFIX, WEIGHT_C_T)        \
void spfloat_csrmm_nt_homo_warp##SUFFIX(                          \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,   \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView B,         \
    tvm::ffi::TensorView C,       int64_t stream                  \
) {                                                               \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);     \
    int m        = static_cast<int>(indptr.size(0)) - 1;          \
    int n        = static_cast<int>(B.size(1));                   \
    int c_blocks = (n + 31) / 32;                                 \
    dim3 grid(m, c_blocks);                                       \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(C.data_ptr());             \
    cudaMemsetAsync(d_out, 0, (size_t)m * n * sizeof(WEIGHT_C_T), s);       \
    _spfloat_csrmm_nt_warp_homo_kern##SUFFIX<<<grid, 32, 0, s>>>( \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),       \
        static_cast<const int32_t*>(indices.data_ptr()),          \
        static_cast<const int32_t*>(indptr.data_ptr()),           \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),             \
        static_cast<WEIGHT_C_T*>(C.data_ptr()),                   \
        m, n);                                                    \
}

// ---- FFI macro: forward hetero warp ----
#define FFI_SPFLOAT_CSRMM_NT_HETERO_WARP(SUFFIX, WEIGHT_C_T)        \
void spfloat_csrmm_nt_hetero_warp##SUFFIX(                          \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,     \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView B,           \
    tvm::ffi::TensorView C,       int64_t stream                    \
) {                                                                 \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);       \
    int m        = static_cast<int>(indptr.size(0)) - 1;            \
    int n        = static_cast<int>(B.size(1));                     \
    int c_blocks = (n + 31) / 32;                                   \
    dim3 grid(m, c_blocks);                                         \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(C.data_ptr());             \
    cudaMemsetAsync(d_out, 0, (size_t)m * n * sizeof(WEIGHT_C_T), s);       \
    _spfloat_csrmm_nt_warp_hetero_kern##SUFFIX<<<grid, 32, 0, s>>>( \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),         \
        static_cast<const int32_t*>(indices.data_ptr()),            \
        static_cast<const int32_t*>(indptr.data_ptr()),             \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),               \
        static_cast<WEIGHT_C_T*>(C.data_ptr()),                     \
        m, n);                                                      \
}

// ---- FFI macro: forward homo block ----
#define FFI_SPFLOAT_CSRMM_NT_HOMO_BLOCK(SUFFIX, WEIGHT_C_T, SHM_SIZE)      \
void spfloat_csrmm_nt_homo_block##SUFFIX(                                  \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,            \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView B,                  \
    tvm::ffi::TensorView C,       int64_t stream                           \
) {                                                                        \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);              \
    int m        = static_cast<int>(indptr.size(0)) - 1;                   \
    int n        = static_cast<int>(B.size(1));                            \
    int c_blocks = (n + 31) / 32;                                          \
    dim3 grid(m, c_blocks);                                                \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(C.data_ptr());             \
    cudaMemsetAsync(d_out, 0, (size_t)m * n * sizeof(WEIGHT_C_T), s);       \
    _spfloat_csrmm_nt_block_homo_kern##SUFFIX<<<grid, 256, SHM_SIZE, s>>>( \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                \
        static_cast<const int32_t*>(indices.data_ptr()),                   \
        static_cast<const int32_t*>(indptr.data_ptr()),                    \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                      \
        static_cast<WEIGHT_C_T*>(C.data_ptr()),                            \
        m, n);                                                             \
}

// ---- FFI macro: forward hetero block ----
#define FFI_SPFLOAT_CSRMM_NT_HETERO_BLOCK(SUFFIX, WEIGHT_C_T, SHM_SIZE)      \
void spfloat_csrmm_nt_hetero_block##SUFFIX(                                  \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,              \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView B,                    \
    tvm::ffi::TensorView C,       int64_t stream                             \
) {                                                                          \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                \
    int m        = static_cast<int>(indptr.size(0)) - 1;                     \
    int n        = static_cast<int>(B.size(1));                              \
    int c_blocks = (n + 31) / 32;                                            \
    dim3 grid(m, c_blocks);                                                  \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(C.data_ptr());             \
    cudaMemsetAsync(d_out, 0, (size_t)m * n * sizeof(WEIGHT_C_T), s);       \
    _spfloat_csrmm_nt_block_hetero_kern##SUFFIX<<<grid, 256, SHM_SIZE, s>>>( \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                  \
        static_cast<const int32_t*>(indices.data_ptr()),                     \
        static_cast<const int32_t*>(indptr.data_ptr()),                      \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                        \
        static_cast<WEIGHT_C_T*>(C.data_ptr()),                              \
        m, n);                                                               \
}

// ---- FFI macro: forward homo auto ----
#define FFI_SPFLOAT_CSRMM_NT_HOMO_AUTO(SUFFIX, WEIGHT_C_T, SHM_SIZE)            \
void spfloat_csrmm_nt_homo_auto##SUFFIX(                                        \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                 \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView B,                       \
    tvm::ffi::TensorView C,       int64_t stream                                \
) {                                                                             \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                   \
    int m        = static_cast<int>(indptr.size(0)) - 1;                        \
    int nse      = static_cast<int>(indices.size(0));                           \
    int n        = static_cast<int>(B.size(1));                                 \
    int avg_nnz  = (m > 0) ? (nse / m) : 0;                                     \
    int c_blocks = (n + 31) / 32;                                               \
    dim3 grid(m, c_blocks);                                                     \
    const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*    d_i = static_cast<const int32_t*>(indices.data_ptr());    \
    const int32_t*    d_p = static_cast<const int32_t*>(indptr.data_ptr());     \
    const WEIGHT_C_T* d_b = static_cast<const WEIGHT_C_T*>(B.data_ptr());       \
    WEIGHT_C_T*       d_c = static_cast<WEIGHT_C_T*>(C.data_ptr());             \
    cudaMemsetAsync(d_c, 0, (size_t)m * n * sizeof(WEIGHT_C_T), s);             \
    if (avg_nnz <= 256) {                                                       \
        _spfloat_csrmm_nt_warp_homo_kern##SUFFIX<<<grid, 32, 0, s>>>(           \
            d_w, d_i, d_p, d_b, d_c, m, n);                                     \
    } else {                                                                    \
        _spfloat_csrmm_nt_block_homo_kern##SUFFIX<<<grid, 256, SHM_SIZE, s>>>(  \
            d_w, d_i, d_p, d_b, d_c, m, n);                                     \
    }                                                                           \
}

// ---- FFI macro: forward hetero auto ----
#define FFI_SPFLOAT_CSRMM_NT_HETERO_AUTO(SUFFIX, WEIGHT_C_T, SHM_SIZE)           \
void spfloat_csrmm_nt_hetero_auto##SUFFIX(                                       \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                  \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView B,                        \
    tvm::ffi::TensorView C,       int64_t stream                                 \
) {                                                                              \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                    \
    int m        = static_cast<int>(indptr.size(0)) - 1;                         \
    int nse      = static_cast<int>(indices.size(0));                            \
    int n        = static_cast<int>(B.size(1));                                  \
    int avg_nnz  = (m > 0) ? (nse / m) : 0;                                      \
    int c_blocks = (n + 31) / 32;                                                \
    dim3 grid(m, c_blocks);                                                      \
    const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr());  \
    const int32_t*    d_i = static_cast<const int32_t*>(indices.data_ptr());     \
    const int32_t*    d_p = static_cast<const int32_t*>(indptr.data_ptr());      \
    const WEIGHT_C_T* d_b = static_cast<const WEIGHT_C_T*>(B.data_ptr());        \
    WEIGHT_C_T*       d_c = static_cast<WEIGHT_C_T*>(C.data_ptr());              \
    cudaMemsetAsync(d_c, 0, (size_t)m * n * sizeof(WEIGHT_C_T), s);             \
    if (avg_nnz <= 256) {                                                        \
        _spfloat_csrmm_nt_warp_hetero_kern##SUFFIX<<<grid, 32, 0, s>>>(          \
            d_w, d_i, d_p, d_b, d_c, m, n);                                      \
    } else {                                                                     \
        _spfloat_csrmm_nt_block_hetero_kern##SUFFIX<<<grid, 256, SHM_SIZE, s>>>( \
            d_w, d_i, d_p, d_b, d_c, m, n);                                      \
    }                                                                            \
}

// ---- FFI macro: transpose homo warp ----
#define FFI_SPFLOAT_CSRMM_T_HOMO_WARP(SUFFIX, WEIGHT_C_T)                   \
void spfloat_csrmm_t_homo_warp##SUFFIX(                                     \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,             \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView B,                   \
    tvm::ffi::TensorView C,       int64_t stream                            \
) {                                                                         \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);               \
    int m        = static_cast<int>(indptr.size(0)) - 1;                    \
    int n        = static_cast<int>(B.size(1));                             \
    int k        = static_cast<int>(C.size(0));                             \
    WEIGHT_C_T* d_c = static_cast<WEIGHT_C_T*>(C.data_ptr());               \
    cudaMemsetAsync(d_c, 0, (size_t)k * (size_t)n * sizeof(WEIGHT_C_T), s); \
    int c_blocks = (n + 31) / 32;                                           \
    dim3 grid(m, c_blocks);                                                 \
    _spfloat_csrmm_t_warp_homo_kern##SUFFIX<<<grid, 32, 0, s>>>(            \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                 \
        static_cast<const int32_t*>(indices.data_ptr()),                    \
        static_cast<const int32_t*>(indptr.data_ptr()),                     \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                       \
        d_c, m, n);                                                         \
}

// ---- FFI macro: transpose hetero warp ----
#define FFI_SPFLOAT_CSRMM_T_HETERO_WARP(SUFFIX, WEIGHT_C_T)                 \
void spfloat_csrmm_t_hetero_warp##SUFFIX(                                   \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,             \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView B,                   \
    tvm::ffi::TensorView C,       int64_t stream                            \
) {                                                                         \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);               \
    int m        = static_cast<int>(indptr.size(0)) - 1;                    \
    int n        = static_cast<int>(B.size(1));                             \
    int k        = static_cast<int>(C.size(0));                             \
    WEIGHT_C_T* d_c = static_cast<WEIGHT_C_T*>(C.data_ptr());               \
    cudaMemsetAsync(d_c, 0, (size_t)k * (size_t)n * sizeof(WEIGHT_C_T), s); \
    int c_blocks = (n + 31) / 32;                                           \
    dim3 grid(m, c_blocks);                                                 \
    _spfloat_csrmm_t_warp_hetero_kern##SUFFIX<<<grid, 32, 0, s>>>(          \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                 \
        static_cast<const int32_t*>(indices.data_ptr()),                    \
        static_cast<const int32_t*>(indptr.data_ptr()),                     \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                       \
        d_c, m, n);                                                         \
}

// SpMM FFI Instantiations
// ---- float32 ----
// @tvm_ffi spfloat_csrmm_nt_homo_warp_f32
FFI_SPFLOAT_CSRMM_NT_HOMO_WARP  (_f32, float)
// @tvm_ffi spfloat_csrmm_nt_hetero_warp_f32
FFI_SPFLOAT_CSRMM_NT_HETERO_WARP(_f32, float)
// @tvm_ffi spfloat_csrmm_nt_homo_block_f32
FFI_SPFLOAT_CSRMM_NT_HOMO_BLOCK (_f32, float, 8 * 32 * sizeof(float))
// @tvm_ffi spfloat_csrmm_nt_hetero_block_f32
FFI_SPFLOAT_CSRMM_NT_HETERO_BLOCK(_f32, float, 8 * 32 * sizeof(float))
// @tvm_ffi spfloat_csrmm_nt_homo_auto_f32
FFI_SPFLOAT_CSRMM_NT_HOMO_AUTO  (_f32, float, 8 * 32 * sizeof(float))
// @tvm_ffi spfloat_csrmm_nt_hetero_auto_f32
FFI_SPFLOAT_CSRMM_NT_HETERO_AUTO(_f32, float, 8 * 32 * sizeof(float))
// @tvm_ffi spfloat_csrmm_t_homo_warp_f32
FFI_SPFLOAT_CSRMM_T_HOMO_WARP   (_f32, float)
// @tvm_ffi spfloat_csrmm_t_hetero_warp_f32
FFI_SPFLOAT_CSRMM_T_HETERO_WARP (_f32, float)

// ---- float64 ----
// @tvm_ffi spfloat_csrmm_nt_homo_warp_f64
FFI_SPFLOAT_CSRMM_NT_HOMO_WARP  (_f64, double)
// @tvm_ffi spfloat_csrmm_nt_hetero_warp_f64
FFI_SPFLOAT_CSRMM_NT_HETERO_WARP(_f64, double)
// @tvm_ffi spfloat_csrmm_nt_homo_block_f64
FFI_SPFLOAT_CSRMM_NT_HOMO_BLOCK (_f64, double, 8 * 32 * sizeof(double))
// @tvm_ffi spfloat_csrmm_nt_hetero_block_f64
FFI_SPFLOAT_CSRMM_NT_HETERO_BLOCK(_f64, double, 8 * 32 * sizeof(double))
// @tvm_ffi spfloat_csrmm_nt_homo_auto_f64
FFI_SPFLOAT_CSRMM_NT_HOMO_AUTO  (_f64, double, 8 * 32 * sizeof(double))
// @tvm_ffi spfloat_csrmm_nt_hetero_auto_f64
FFI_SPFLOAT_CSRMM_NT_HETERO_AUTO(_f64, double, 8 * 32 * sizeof(double))
// @tvm_ffi spfloat_csrmm_t_homo_warp_f64
FFI_SPFLOAT_CSRMM_T_HOMO_WARP   (_f64, double)
// @tvm_ffi spfloat_csrmm_t_hetero_warp_f64
FFI_SPFLOAT_CSRMM_T_HETERO_WARP (_f64, double)

// ---- float16 ----
// @tvm_ffi spfloat_csrmm_nt_homo_warp_f16
FFI_SPFLOAT_CSRMM_NT_HOMO_WARP  (_f16, __half)
// @tvm_ffi spfloat_csrmm_nt_hetero_warp_f16
FFI_SPFLOAT_CSRMM_NT_HETERO_WARP(_f16, __half)
// @tvm_ffi spfloat_csrmm_nt_homo_block_f16
FFI_SPFLOAT_CSRMM_NT_HOMO_BLOCK (_f16, __half, 8 * 32 * sizeof(float))
// @tvm_ffi spfloat_csrmm_nt_hetero_block_f16
FFI_SPFLOAT_CSRMM_NT_HETERO_BLOCK(_f16, __half, 8 * 32 * sizeof(float))
// @tvm_ffi spfloat_csrmm_nt_homo_auto_f16
FFI_SPFLOAT_CSRMM_NT_HOMO_AUTO  (_f16, __half, 8 * 32 * sizeof(float))
// @tvm_ffi spfloat_csrmm_nt_hetero_auto_f16
FFI_SPFLOAT_CSRMM_NT_HETERO_AUTO(_f16, __half, 8 * 32 * sizeof(float))
// @tvm_ffi spfloat_csrmm_t_homo_warp_f16
FFI_SPFLOAT_CSRMM_T_HOMO_WARP   (_f16, __half)
// @tvm_ffi spfloat_csrmm_t_hetero_warp_f16
FFI_SPFLOAT_CSRMM_T_HETERO_WARP (_f16, __half)

// ---- bfloat16 ----
// @tvm_ffi spfloat_csrmm_nt_homo_warp_bf16
FFI_SPFLOAT_CSRMM_NT_HOMO_WARP  (_bf16, __nv_bfloat16)
// @tvm_ffi spfloat_csrmm_nt_hetero_warp_bf16
FFI_SPFLOAT_CSRMM_NT_HETERO_WARP(_bf16, __nv_bfloat16)
// @tvm_ffi spfloat_csrmm_nt_homo_block_bf16
FFI_SPFLOAT_CSRMM_NT_HOMO_BLOCK (_bf16, __nv_bfloat16, 8 * 32 * sizeof(float))
// @tvm_ffi spfloat_csrmm_nt_hetero_block_bf16
FFI_SPFLOAT_CSRMM_NT_HETERO_BLOCK(_bf16, __nv_bfloat16, 8 * 32 * sizeof(float))
// @tvm_ffi spfloat_csrmm_nt_homo_auto_bf16
FFI_SPFLOAT_CSRMM_NT_HOMO_AUTO  (_bf16, __nv_bfloat16, 8 * 32 * sizeof(float))
// @tvm_ffi spfloat_csrmm_nt_hetero_auto_bf16
FFI_SPFLOAT_CSRMM_NT_HETERO_AUTO(_bf16, __nv_bfloat16, 8 * 32 * sizeof(float))
// @tvm_ffi spfloat_csrmm_t_homo_warp_bf16
FFI_SPFLOAT_CSRMM_T_HOMO_WARP   (_bf16, __nv_bfloat16)
// @tvm_ffi spfloat_csrmm_t_hetero_warp_bf16
FFI_SPFLOAT_CSRMM_T_HETERO_WARP (_bf16, __nv_bfloat16)
