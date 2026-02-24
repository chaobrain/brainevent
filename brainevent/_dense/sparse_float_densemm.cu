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
 * sparse_float_densemm.cu -- Sparse-Float Dense Matrix-Matrix CUDA Kernels
 * =========================================================================
 *
 * This module provides optimized CUDA kernels for dense matrix-matrix
 * operations with sparse-float inputs (spfloat_densemm):
 *
 * 1. spfloat_densemm_nt  -- weights[m,k] @ spikes[k,n] -> out[m,n]
 *    (transpose=False): warp-per-row or thread-per-element based on n.
 *
 * 2. spfloat_densemm_t  -- spikes[m,k] @ weights[k,n] -> out[m,n]
 *    (transpose=True): event-driven scatter over active spike rows.
 *
 * Python API (brainevent._dense.sparse_float):
 *   spfloat_densemm(weights, spikes, transpose=False)
 *     weights : float16/float32/float64/bfloat16 matrix
 *     spikes  : float16/float32/float64/bfloat16 sparse-float matrix
 *     returns : output matrix
 *
 * TVM FFI entry points:
 *   spfloat_densemm_nt_{dtype}       (warp-per-row NT)
 *   spfloat_densemm_nt_tpe_{dtype}   (thread-per-element NT)
 *   spfloat_densemm_t_{dtype}        (T mode)
 */

#include "cuda_common.h"

// =========================================================================
// Dense Matrix-Matrix Multiplication (spfloat_densemm) - NT MODE
// =========================================================================

#define MM_BLOCK_SIZE 256

#define DEFINE_SPFLOAT_MM_NT_WPR(SUFFIX, WEIGHT_T, ACC_T, CHUNK_N,  \
                                  READ_W, WRITE_W, READ_S,          \
                                  WARP_RED, ACC_ZERO)               \
__global__ void _spfloat_mm_nt_wpr_kern##SUFFIX(                    \
    const WEIGHT_T* __restrict__ weights,                           \
    const WEIGHT_T* __restrict__ spikes,                            \
    WEIGHT_T*       __restrict__ output,                            \
    int m, int k, int n                                             \
) {                                                                 \
    int warp_id = threadIdx.x >> 5;                                 \
    int lane    = threadIdx.x & 31;                                 \
    int warps_per_block = blockDim.x >> 5;                          \
    int row = blockIdx.x * warps_per_block + warp_id;               \
    if (row >= m) return;                                           \
    int col_start = blockIdx.y * CHUNK_N;                           \
    int chunk_n = min(CHUNK_N, n - col_start);                      \
    const WEIGHT_T* w_row = weights + (size_t)row * k;              \
    ACC_T acc[CHUNK_N];                                             \
    _Pragma("unroll")                                               \
    for (int j = 0; j < CHUNK_N; j++) acc[j] = ACC_ZERO;            \
    for (int l = lane; l < k; l += 32) {                            \
        ACC_T w_val = READ_W(w_row[l]);                             \
        const WEIGHT_T* spk_l = spikes + (size_t)l * n + col_start; \
        _Pragma("unroll")                                           \
        for (int j = 0; j < CHUNK_N; j++)                           \
            if (j < chunk_n)                                        \
                acc[j] += w_val * READ_S(spk_l[j]);                 \
    }                                                               \
    WEIGHT_T* out_row = output + (size_t)row * n + col_start;       \
    _Pragma("unroll")                                               \
    for (int j = 0; j < CHUNK_N; j++) {                             \
        ACC_T val = WARP_RED(acc[j]);                               \
        if (lane == 0 && j < chunk_n) out_row[j] = WRITE_W(val);    \
    }                                                               \
}

#define DEFINE_SPFLOAT_MM_NT_TPE(SUFFIX, WEIGHT_T, ACC_T,                  \
                                  READ_W, WRITE_W, READ_S, ACC_ZERO)       \
__global__ void _spfloat_mm_nt_tpe_kern##SUFFIX(                           \
    const WEIGHT_T* __restrict__ weights,                                  \
    const WEIGHT_T* __restrict__ spikes,                                   \
    WEIGHT_T*       __restrict__ output,                                   \
    int m, int k, int n                                                    \
) {                                                                        \
    int warp_id = threadIdx.x >> 5;                                        \
    int lane    = threadIdx.x & 31;                                        \
    int warps_per_block = blockDim.x >> 5;                                 \
    int row = blockIdx.x * warps_per_block + warp_id;                      \
    int col = blockIdx.y * 32 + lane;                                      \
    if (row >= m || col >= n) return;                                      \
    const WEIGHT_T* w_row = weights + (size_t)row * k;                     \
    ACC_T acc = ACC_ZERO;                                                  \
    int l = 0;                                                             \
    for (; l <= k - 4; l += 4) {                                           \
        ACC_T sv0 = READ_S(spikes[(size_t)(l+0) * n + col]);               \
        ACC_T sv1 = READ_S(spikes[(size_t)(l+1) * n + col]);               \
        ACC_T sv2 = READ_S(spikes[(size_t)(l+2) * n + col]);               \
        ACC_T sv3 = READ_S(spikes[(size_t)(l+3) * n + col]);               \
        bool any_nz = (sv0 != ACC_ZERO) | (sv1 != ACC_ZERO) |              \
                      (sv2 != ACC_ZERO) | (sv3 != ACC_ZERO);               \
        if (__ballot_sync(__activemask(), any_nz) == 0u) continue;         \
        acc += READ_W(w_row[l+0]) * sv0;                                   \
        acc += READ_W(w_row[l+1]) * sv1;                                   \
        acc += READ_W(w_row[l+2]) * sv2;                                   \
        acc += READ_W(w_row[l+3]) * sv3;                                   \
    }                                                                      \
    for (; l < k; l++) {                                                   \
        ACC_T sv = READ_S(spikes[(size_t)l * n + col]);                    \
        if (__ballot_sync(__activemask(), sv != ACC_ZERO) == 0u) continue; \
        acc += READ_W(w_row[l]) * sv;                                      \
    }                                                                      \
    output[(size_t)row * n + col] = WRITE_W(acc);                          \
}

// =========================================================================
// Dense Matrix-Matrix Multiplication (spfloat_densemm) - T MODE
// =========================================================================

#define DEFINE_SPFLOAT_MM_T(SUFFIX, WEIGHT_T, ACC_T, CHUNK_N,          \
                             READ_W, WRITE_W, READ_S,                  \
                             WARP_RED, ACC_ZERO)                       \
__global__ void _spfloat_mm_t_kern##SUFFIX(                            \
    const WEIGHT_T* __restrict__ weights,                              \
    const WEIGHT_T* __restrict__ spikes,                               \
    WEIGHT_T*       __restrict__ output,                               \
    int m, int k, int n                                                \
) {                                                                    \
    int warp_id = threadIdx.x >> 5;                                    \
    int lane    = threadIdx.x & 31;                                    \
    int warps_per_block = blockDim.x >> 5;                             \
    int row = blockIdx.x * warps_per_block + warp_id;                  \
    if (row >= m) return;                                              \
    int col_start = blockIdx.y * CHUNK_N;                              \
    int chunk_n = min(CHUNK_N, n - col_start);                         \
    const WEIGHT_T* s_row = spikes + (size_t)row * k;                  \
    ACC_T acc[CHUNK_N];                                                \
    _Pragma("unroll")                                                  \
    for (int j = 0; j < CHUNK_N; j++) acc[j] = ACC_ZERO;               \
    for (int l = lane; l < k; l += 32) {                               \
        ACC_T spk_val = READ_S(s_row[l]);                              \
        if (__ballot_sync(__activemask(), spk_val != ACC_ZERO) == 0u)  \
            continue;                                                  \
        if (spk_val != ACC_ZERO) {                                     \
            const WEIGHT_T* w_l = weights + (size_t)l * n + col_start; \
            _Pragma("unroll")                                          \
            for (int j = 0; j < CHUNK_N; j++)                          \
                if (j < chunk_n)                                       \
                    acc[j] += spk_val * READ_W(w_l[j]);                \
        }                                                              \
    }                                                                  \
    WEIGHT_T* out_row = output + (size_t)row * n + col_start;          \
    _Pragma("unroll")                                                  \
    for (int j = 0; j < CHUNK_N; j++) {                                \
        ACC_T val = WARP_RED(acc[j]);                                  \
        if (lane == 0 && j < chunk_n) out_row[j] = WRITE_W(val);       \
    }                                                                  \
}

// SpMM Instantiations
DEFINE_SPFLOAT_MM_NT_WPR(_f32, float, float, 32, READ_F32, WRITE_F32, READ_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_MM_NT_TPE(_f32, float, float, READ_F32, WRITE_F32, READ_F32, 0.0f)
DEFINE_SPFLOAT_MM_T(_f32, float, float, 32, READ_F32, WRITE_F32, READ_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_MM_NT_WPR(_f64, double, double, 16, READ_F64, WRITE_F64, READ_F64, warp_reduce_sum_f64, 0.0)
DEFINE_SPFLOAT_MM_NT_TPE(_f64, double, double, READ_F64, WRITE_F64, READ_F64, 0.0)
DEFINE_SPFLOAT_MM_T(_f64, double, double, 16, READ_F64, WRITE_F64, READ_F64, warp_reduce_sum_f64, 0.0)
DEFINE_SPFLOAT_MM_NT_WPR(_f16, __half, float, 32, READ_F16, WRITE_F16, READ_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_MM_NT_TPE(_f16, __half, float, READ_F16, WRITE_F16, READ_F16, 0.0f)
DEFINE_SPFLOAT_MM_T(_f16, __half, float, 32, READ_F16, WRITE_F16, READ_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_MM_NT_WPR(_bf16, __nv_bfloat16, float, 32, READ_BF16, WRITE_BF16, READ_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_MM_NT_TPE(_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, READ_BF16, 0.0f)
DEFINE_SPFLOAT_MM_T(_bf16, __nv_bfloat16, float, 32, READ_BF16, WRITE_BF16, READ_BF16, warp_reduce_sum_f32, 0.0f)

// FFI Macros for SpMM
#define FFI_SPFLOAT_MM_NT_WPR(SUFFIX, WEIGHT_C_T, CHUNK_N_VAL)                  \
void spfloat_densemm_nt##SUFFIX(                                                \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView spikes,                  \
    tvm::ffi::TensorView output, int64_t stream                                 \
) {                                                                             \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                    \
    int m = static_cast<int>(weights.size(0));                                  \
    int k = static_cast<int>(weights.size(1));                                  \
    int n = static_cast<int>(spikes.size(1));                                   \
    int warps_per_block = MM_BLOCK_SIZE / 32;                                   \
    int m_blocks  = (m + warps_per_block - 1) / warps_per_block;                \
    int n_chunks  = (n + CHUNK_N_VAL - 1) / CHUNK_N_VAL;                        \
    dim3 grid(m_blocks, n_chunks);                                              \
    const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const WEIGHT_C_T* d_s = static_cast<const WEIGHT_C_T*>(spikes.data_ptr());  \
    WEIGHT_C_T*       d_o = static_cast<WEIGHT_C_T*>(output.data_ptr());        \
    _spfloat_mm_nt_wpr_kern##SUFFIX<<<grid, MM_BLOCK_SIZE, 0, s>>>(             \
        d_w, d_s, d_o, m, k, n);                                                \
}

#define FFI_SPFLOAT_MM_NT_TPE(SUFFIX, WEIGHT_C_T)                               \
void spfloat_densemm_nt_tpe##SUFFIX(                                            \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView spikes,                  \
    tvm::ffi::TensorView output, int64_t stream                                 \
) {                                                                             \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                    \
    int m = static_cast<int>(weights.size(0));                                  \
    int k = static_cast<int>(weights.size(1));                                  \
    int n = static_cast<int>(spikes.size(1));                                   \
    int warps_per_block = MM_BLOCK_SIZE / 32;                                   \
    int m_blocks = (m + warps_per_block - 1) / warps_per_block;                 \
    int n_blocks = (n + 31) / 32;                                               \
    dim3 grid(m_blocks, n_blocks);                                              \
    const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const WEIGHT_C_T* d_s = static_cast<const WEIGHT_C_T*>(spikes.data_ptr());  \
    WEIGHT_C_T*       d_o = static_cast<WEIGHT_C_T*>(output.data_ptr());        \
    _spfloat_mm_nt_tpe_kern##SUFFIX<<<grid, MM_BLOCK_SIZE, 0, s>>>(             \
        d_w, d_s, d_o, m, k, n);                                                \
}

#define FFI_SPFLOAT_MM_T(SUFFIX, WEIGHT_C_T, CHUNK_N_VAL)                       \
void spfloat_densemm_t##SUFFIX(                                                 \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView spikes,                  \
    tvm::ffi::TensorView output, int64_t stream                                 \
) {                                                                             \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                    \
    int k = static_cast<int>(weights.size(0));                                  \
    int n = static_cast<int>(weights.size(1));                                  \
    int m = static_cast<int>(spikes.size(0));                                   \
    int warps_per_block = MM_BLOCK_SIZE / 32;                                   \
    int m_blocks  = (m + warps_per_block - 1) / warps_per_block;                \
    int n_chunks  = (n + CHUNK_N_VAL - 1) / CHUNK_N_VAL;                        \
    dim3 grid(m_blocks, n_chunks);                                              \
    const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const WEIGHT_C_T* d_s = static_cast<const WEIGHT_C_T*>(spikes.data_ptr());  \
    WEIGHT_C_T*       d_o = static_cast<WEIGHT_C_T*>(output.data_ptr());        \
    _spfloat_mm_t_kern##SUFFIX<<<grid, MM_BLOCK_SIZE, 0, s>>>(                  \
        d_w, d_s, d_o, m, k, n);                                                \
}

// SpMM FFI Instantiations
// @tvm_ffi spfloat_densemm_nt_f32
FFI_SPFLOAT_MM_NT_WPR(_f32, float, 32)
// @tvm_ffi spfloat_densemm_nt_tpe_f32
FFI_SPFLOAT_MM_NT_TPE(_f32, float)
// @tvm_ffi spfloat_densemm_t_f32
FFI_SPFLOAT_MM_T(_f32, float, 32)
// @tvm_ffi spfloat_densemm_nt_f64
FFI_SPFLOAT_MM_NT_WPR(_f64, double, 16)
// @tvm_ffi spfloat_densemm_nt_tpe_f64
FFI_SPFLOAT_MM_NT_TPE(_f64, double)
// @tvm_ffi spfloat_densemm_t_f64
FFI_SPFLOAT_MM_T(_f64, double, 16)
// @tvm_ffi spfloat_densemm_nt_f16
FFI_SPFLOAT_MM_NT_WPR(_f16, __half, 32)
// @tvm_ffi spfloat_densemm_nt_tpe_f16
FFI_SPFLOAT_MM_NT_TPE(_f16, __half)
// @tvm_ffi spfloat_densemm_t_f16
FFI_SPFLOAT_MM_T(_f16, __half, 32)
// @tvm_ffi spfloat_densemm_nt_bf16
FFI_SPFLOAT_MM_NT_WPR(_bf16, __nv_bfloat16, 32)
// @tvm_ffi spfloat_densemm_nt_tpe_bf16
FFI_SPFLOAT_MM_NT_TPE(_bf16, __nv_bfloat16)
// @tvm_ffi spfloat_densemm_t_bf16
FFI_SPFLOAT_MM_T(_bf16, __nv_bfloat16, 32)
