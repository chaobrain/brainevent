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
 * sparse_float_fcnmm.cu -- Sparse-Float FCN Sparse Matrix-Matrix CUDA Kernels
 * =============================================================================
 *
 * This module provides optimized CUDA kernels for sparse operations with
 * fixed connection number (FCN) and sparse-float inputs. It includes:
 *   spfloat_fcnmm -- Sparse Matrix-Matrix Product (SpMM)
 *
 * These kernels exploit "sparse-float" sparsity: only connections to non-zero
 * floating-point entries contribute to the output, skipping unnecessary work.
 *
 * Supports weight dtypes: float32, float64, float16, bfloat16
 * Supports spike dtypes:  float32, float64, float16, bfloat16
 * Supports homo (scalar weight) and hetero (per-connection weight array) modes.
 */

#include "cuda_common.h"
#include "brainevent/common.h"

// ============================================================================
// FCN Matrix-Matrix Multiplication (spfloat_fcnmm)
// ============================================================================

#define FCN_MM_GATHER_TILE_K 128

// ---------------------------------------------------------------------------
// Gather tiled: shared-memory tiling of indices (and weights for hetero).
// ---------------------------------------------------------------------------
#define DEFINE_SPFLOAT_FCN_MM_GATHER_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _spfloat_fcnmm_gather_homo_kern##SUFFIX(                                      \
    const int32_t* __restrict__ indices,                                                      \
    const WEIGHT_T* __restrict__ matrix,                                                      \
    WEIGHT_T*       __restrict__ output,                                                      \
    const WEIGHT_T* __restrict__ weights,                                                     \
    int n_pre, int n_conn, int n_col                                                          \
) {                                                                                           \
    extern __shared__ char _smem_raw[];                                                       \
    int32_t* s_idx = reinterpret_cast<int32_t*>(_smem_raw);                                   \
                                                                                              \
    int row = blockIdx.x;                                                                     \
    int j   = blockIdx.y * blockDim.x + threadIdx.x;                                          \
    if (row >= n_pre) return;                                                                 \
                                                                                              \
    const int32_t*  idx_row = indices + (size_t)row * n_conn;                                 \
    ACC_T homo_w = READ_W(__ldg(&weights[0]));                                                \
    ACC_T acc = ACC_ZERO;                                                                     \
                                                                                              \
    for (int base = 0; base < n_conn; base += FCN_MM_GATHER_TILE_K) {                         \
        int tile_size = n_conn - base;                                                        \
        if (tile_size > FCN_MM_GATHER_TILE_K) tile_size = FCN_MM_GATHER_TILE_K;               \
        for (int t = threadIdx.x; t < tile_size; t += blockDim.x)                             \
            s_idx[t] = __ldg(&idx_row[base + t]);                                             \
        __syncthreads();                                                                      \
                                                                                              \
        if (j < n_col) {                                                                      \
            for (int k = 0; k < tile_size; k++) {                                             \
                ACC_T m_val = READ_W(__ldg(&matrix[(size_t)s_idx[k] * n_col + j]));           \
                if (m_val != ACC_ZERO) acc += m_val;                                          \
            }                                                                                 \
        }                                                                                     \
        __syncthreads();                                                                      \
    }                                                                                         \
                                                                                              \
    if (j < n_col)                                                                            \
        output[(size_t)row * n_col + j] = WRITE_W(homo_w * acc);                              \
}

#define DEFINE_SPFLOAT_FCN_MM_GATHER_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _spfloat_fcnmm_gather_hetero_kern##SUFFIX(                                      \
    const int32_t* __restrict__ indices,                                                        \
    const WEIGHT_T* __restrict__ matrix,                                                        \
    WEIGHT_T*       __restrict__ output,                                                        \
    const WEIGHT_T* __restrict__ weights,                                                       \
    int n_pre, int n_conn, int n_col                                                            \
) {                                                                                             \
    extern __shared__ char _smem_raw[];                                                         \
    int32_t* s_idx = reinterpret_cast<int32_t*>(_smem_raw);                                     \
    ACC_T*   s_wt  = reinterpret_cast<ACC_T*>(                                                  \
        _smem_raw + FCN_MM_GATHER_TILE_K * sizeof(int32_t));                                    \
                                                                                                \
    int row = blockIdx.x;                                                                       \
    int j   = blockIdx.y * blockDim.x + threadIdx.x;                                            \
    if (row >= n_pre) return;                                                                   \
                                                                                                \
    const int32_t*  idx_row = indices + (size_t)row * n_conn;                                   \
    const WEIGHT_T* w_row   = weights + (size_t)row * n_conn;                                   \
    ACC_T acc = ACC_ZERO;                                                                       \
                                                                                                \
    for (int base = 0; base < n_conn; base += FCN_MM_GATHER_TILE_K) {                           \
        int tile_size = n_conn - base;                                                          \
        if (tile_size > FCN_MM_GATHER_TILE_K) tile_size = FCN_MM_GATHER_TILE_K;                 \
        for (int t = threadIdx.x; t < tile_size; t += blockDim.x) {                             \
            s_idx[t] = __ldg(&idx_row[base + t]);                                               \
            s_wt[t]  = READ_W(__ldg(&w_row[base + t]));                                         \
        }                                                                                       \
        __syncthreads();                                                                        \
                                                                                                \
        if (j < n_col) {                                                                        \
            for (int k = 0; k < tile_size; k++) {                                               \
                ACC_T m_val = READ_W(__ldg(&matrix[(size_t)s_idx[k] * n_col + j]));             \
                if (m_val != ACC_ZERO) acc += s_wt[k] * m_val;                                  \
            }                                                                                   \
        }                                                                                       \
        __syncthreads();                                                                        \
    }                                                                                           \
                                                                                                \
    if (j < n_col) output[(size_t)row * n_col + j] = WRITE_W(acc);                              \
}

// ---------------------------------------------------------------------------
// Scatter with smem caching
// ---------------------------------------------------------------------------
#define DEFINE_SPFLOAT_FCN_MM_SCATTER_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ATOMIC_ADD_W, ACC_ZERO)            \
__global__ void _spfloat_fcnmm_scatter_homo_kern##SUFFIX(                                                               \
    const int32_t* __restrict__ indices,                                                                                \
    const WEIGHT_T* __restrict__ matrix,                                                                                \
    WEIGHT_T*       __restrict__ output,                                                                                \
    const WEIGHT_T* __restrict__ weights,                                                                               \
    int n_pre, int n_conn, int n_col                                                                                    \
) {                                                                                                                     \
    extern __shared__ char _smem_raw[];                                                                                 \
    WEIGHT_T* s_mrow = reinterpret_cast<WEIGHT_T*>(_smem_raw);                                                          \
    int i = blockIdx.x;                                                                                                 \
    if (i >= n_pre) return;                                                                                             \
    const WEIGHT_T* m_row = matrix + (size_t)i * n_col;                                                                 \
    for (int j = threadIdx.x; j < n_col; j += blockDim.x) s_mrow[j] = __ldg(&m_row[j]);                                 \
    __syncthreads();                                                                                                    \
    int has_nz = 0;                                                                                                     \
    for (int j = threadIdx.x; j < n_col; j += blockDim.x) { if (READ_W(s_mrow[j]) != ACC_ZERO) { has_nz = 1; break; } } \
    if (__syncthreads_count(has_nz) == 0) return;                                                                       \
    const int32_t*  idx_row = indices + (size_t)i * n_conn;                                                             \
    ACC_T w0 = READ_W(__ldg(&weights[0]));                                                                              \
    for (int k = 0; k < n_conn; k++) {                                                                                  \
        int tgt = __ldg(&idx_row[k]);                                                                                   \
        WEIGHT_T* out_row = output + (size_t)tgt * n_col;                                                               \
        for (int j = threadIdx.x; j < n_col; j += blockDim.x) {                                                         \
            ACC_T m_val = READ_W(s_mrow[j]);                                                                            \
            if (m_val != ACC_ZERO) ATOMIC_ADD_W(&out_row[j], w0 * m_val);                                               \
        }                                                                                                               \
    }                                                                                                                   \
}

#define DEFINE_SPFLOAT_FCN_MM_SCATTER_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ATOMIC_ADD_W, ACC_ZERO)          \
__global__ void _spfloat_fcnmm_scatter_hetero_kern##SUFFIX(                                                             \
    const int32_t* __restrict__ indices,                                                                                \
    const WEIGHT_T* __restrict__ matrix,                                                                                \
    WEIGHT_T*       __restrict__ output,                                                                                \
    const WEIGHT_T* __restrict__ weights,                                                                               \
    int n_pre, int n_conn, int n_col                                                                                    \
) {                                                                                                                     \
    extern __shared__ char _smem_raw[];                                                                                 \
    WEIGHT_T* s_mrow = reinterpret_cast<WEIGHT_T*>(_smem_raw);                                                          \
    int i = blockIdx.x;                                                                                                 \
    if (i >= n_pre) return;                                                                                             \
    const WEIGHT_T* m_row = matrix + (size_t)i * n_col;                                                                 \
    for (int j = threadIdx.x; j < n_col; j += blockDim.x) s_mrow[j] = __ldg(&m_row[j]);                                 \
    __syncthreads();                                                                                                    \
    int has_nz = 0;                                                                                                     \
    for (int j = threadIdx.x; j < n_col; j += blockDim.x) { if (READ_W(s_mrow[j]) != ACC_ZERO) { has_nz = 1; break; } } \
    if (__syncthreads_count(has_nz) == 0) return;                                                                       \
    const int32_t*  idx_row = indices + (size_t)i * n_conn;                                                             \
    const WEIGHT_T* w_row   = weights + (size_t)i * n_conn;                                                             \
    for (int k = 0; k < n_conn; k++) {                                                                                  \
        int tgt = __ldg(&idx_row[k]);                                                                                   \
        ACC_T wk = READ_W(__ldg(&w_row[k]));                                                                            \
        WEIGHT_T* out_row = output + (size_t)tgt * n_col;                                                               \
        for (int j = threadIdx.x; j < n_col; j += blockDim.x) {                                                         \
            ACC_T m_val = READ_W(s_mrow[j]);                                                                            \
            if (m_val != ACC_ZERO) ATOMIC_ADD_W(&out_row[j], wk * m_val);                                               \
        }                                                                                                               \
    }                                                                                                                   \
}

// FCN SpMM Instantiations
// ---- float32 ----
DEFINE_SPFLOAT_FCN_MM_GATHER_HOMO  (_f32, float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_SPFLOAT_FCN_MM_GATHER_HETERO(_f32, float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_SPFLOAT_FCN_MM_SCATTER_HOMO (_f32, float, float, READ_F32, WRITE_F32, atomic_add_f32, 0.0f)
DEFINE_SPFLOAT_FCN_MM_SCATTER_HETERO(_f32, float, float, READ_F32, WRITE_F32, atomic_add_f32, 0.0f)

// ---- float64 ----
DEFINE_SPFLOAT_FCN_MM_GATHER_HOMO  (_f64, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_SPFLOAT_FCN_MM_GATHER_HETERO(_f64, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_SPFLOAT_FCN_MM_SCATTER_HOMO (_f64, double, double, READ_F64, WRITE_F64, atomic_add_f64, 0.0)
DEFINE_SPFLOAT_FCN_MM_SCATTER_HETERO(_f64, double, double, READ_F64, WRITE_F64, atomic_add_f64, 0.0)

// ---- float16 ----
DEFINE_SPFLOAT_FCN_MM_GATHER_HOMO  (_f16, __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_SPFLOAT_FCN_MM_GATHER_HETERO(_f16, __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_SPFLOAT_FCN_MM_SCATTER_HOMO (_f16, __half, float, READ_F16, WRITE_F16, atomic_add_f16, 0.0f)
DEFINE_SPFLOAT_FCN_MM_SCATTER_HETERO(_f16, __half, float, READ_F16, WRITE_F16, atomic_add_f16, 0.0f)

// FCN SpMM FFI Entry Macros
// ---- FFI macro: gather homo auto ----
#define FFI_SPFLOAT_FCN_MM_GATHER_HOMO(SUFFIX, WEIGHT_C_T)                                \
void spfloat_fcnmm_gather_homo_auto##SUFFIX(                                              \
    const BE::Tensor weights, const BE::Tensor indices,                           \
    const BE::Tensor matrix,  BE::Tensor output, int64_t stream             \
) {                                                                                       \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                             \
    int n_pre       = static_cast<int>(indices.size(0));                                  \
    int n_conn      = static_cast<int>(indices.size(1));                                  \
    int n_col       = static_cast<int>(matrix.size(1));                                   \
    int block_x = ((n_col + 31) >> 5) << 5;                                               \
    block_x = block_x < 32 ? 32 : block_x > 256 ? 256 : block_x;                          \
    int y_blocks = (n_col + block_x - 1) / block_x;                                       \
    size_t smem = FCN_MM_GATHER_TILE_K * sizeof(int32_t);                                 \
    _spfloat_fcnmm_gather_homo_kern##SUFFIX<<<dim3(n_pre, y_blocks), block_x, smem, s>>>( \
        static_cast<const int32_t*>(indices.data_ptr()),                                  \
        static_cast<const WEIGHT_C_T*>(matrix.data_ptr()),                                \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                                      \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                               \
        n_pre, n_conn, n_col);                                                            \
}

// ---- FFI macro: gather hetero auto ----
#define FFI_SPFLOAT_FCN_MM_GATHER_HETERO(SUFFIX, WEIGHT_C_T)                                \
void spfloat_fcnmm_gather_hetero_auto##SUFFIX(                                              \
    const BE::Tensor weights, const BE::Tensor indices,                             \
    const BE::Tensor matrix,  BE::Tensor output, int64_t stream               \
) {                                                                                         \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                               \
    int n_pre       = static_cast<int>(indices.size(0));                                    \
    int n_conn      = static_cast<int>(indices.size(1));                                    \
    int n_col       = static_cast<int>(matrix.size(1));                                     \
    int block_x = ((n_col + 31) >> 5) << 5;                                                 \
    block_x = block_x < 32 ? 32 : block_x > 256 ? 256 : block_x;                            \
    int y_blocks = (n_col + block_x - 1) / block_x;                                         \
    int acc_sz = (sizeof(WEIGHT_C_T) < 4) ? 4 : static_cast<int>(sizeof(WEIGHT_C_T));       \
    size_t smem = FCN_MM_GATHER_TILE_K * (sizeof(int32_t) + acc_sz);                        \
    _spfloat_fcnmm_gather_hetero_kern##SUFFIX<<<dim3(n_pre, y_blocks), block_x, smem, s>>>( \
        static_cast<const int32_t*>(indices.data_ptr()),                                    \
        static_cast<const WEIGHT_C_T*>(matrix.data_ptr()),                                  \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                                        \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                                 \
        n_pre, n_conn, n_col);                                                              \
}

// ---- FFI macro: scatter homo auto ----
#define FFI_SPFLOAT_FCN_MM_SCATTER_HOMO(SUFFIX, WEIGHT_C_T)                                \
void spfloat_fcnmm_scatter_homo_auto##SUFFIX(                                              \
    const BE::Tensor weights, const BE::Tensor indices,                            \
    const BE::Tensor matrix,  BE::Tensor output, int64_t stream              \
) {                                                                                        \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                              \
    int n_pre       = static_cast<int>(indices.size(0));                                   \
    int n_conn      = static_cast<int>(indices.size(1));                                   \
    int n_post      = static_cast<int>(output.size(0));                                    \
    int n_col       = static_cast<int>(matrix.size(1));                                    \
    cudaMemsetAsync(output.data_ptr(), 0, (size_t)n_post * n_col * sizeof(WEIGHT_C_T), s); \
    int block_x = ((n_col + 31) >> 5) << 5;                                                \
    block_x = block_x < 32 ? 32 : block_x > 256 ? 256 : block_x;                           \
    size_t smem = static_cast<size_t>(n_col) * sizeof(WEIGHT_C_T);                         \
    _spfloat_fcnmm_scatter_homo_kern##SUFFIX<<<n_pre, block_x, smem, s>>>(                 \
        static_cast<const int32_t*>(indices.data_ptr()),                                   \
        static_cast<const WEIGHT_C_T*>(matrix.data_ptr()),                                 \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                                       \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                                \
        n_pre, n_conn, n_col);                                                             \
}

// ---- FFI macro: scatter hetero auto ----
#define FFI_SPFLOAT_FCN_MM_SCATTER_HETERO(SUFFIX, WEIGHT_C_T)                              \
void spfloat_fcnmm_scatter_hetero_auto##SUFFIX(                                            \
    const BE::Tensor weights, const BE::Tensor indices,                            \
    const BE::Tensor matrix,  BE::Tensor output, int64_t stream              \
) {                                                                                        \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                              \
    int n_pre       = static_cast<int>(indices.size(0));                                   \
    int n_conn      = static_cast<int>(indices.size(1));                                   \
    int n_post      = static_cast<int>(output.size(0));                                    \
    int n_col       = static_cast<int>(matrix.size(1));                                    \
    cudaMemsetAsync(output.data_ptr(), 0, (size_t)n_post * n_col * sizeof(WEIGHT_C_T), s); \
    int block_x = ((n_col + 31) >> 5) << 5;                                                \
    block_x = block_x < 32 ? 32 : block_x > 256 ? 256 : block_x;                           \
    size_t smem = static_cast<size_t>(n_col) * sizeof(WEIGHT_C_T);                         \
    _spfloat_fcnmm_scatter_hetero_kern##SUFFIX<<<n_pre, block_x, smem, s>>>(               \
        static_cast<const int32_t*>(indices.data_ptr()),                                   \
        static_cast<const WEIGHT_C_T*>(matrix.data_ptr()),                                 \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                                       \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                                \
        n_pre, n_conn, n_col);                                                             \
}

// SpMM FFI Instantiations
// ---- float32 ----
// @BE spfloat_fcnmm_gather_homo_auto_f32
FFI_SPFLOAT_FCN_MM_GATHER_HOMO  (_f32, float)
// @BE spfloat_fcnmm_gather_hetero_auto_f32
FFI_SPFLOAT_FCN_MM_GATHER_HETERO(_f32, float)
// @BE spfloat_fcnmm_scatter_homo_auto_f32
FFI_SPFLOAT_FCN_MM_SCATTER_HOMO (_f32, float)
// @BE spfloat_fcnmm_scatter_hetero_auto_f32
FFI_SPFLOAT_FCN_MM_SCATTER_HETERO(_f32, float)

// ---- float64 ----
// @BE spfloat_fcnmm_gather_homo_auto_f64
FFI_SPFLOAT_FCN_MM_GATHER_HOMO  (_f64, double)
// @BE spfloat_fcnmm_gather_hetero_auto_f64
FFI_SPFLOAT_FCN_MM_GATHER_HETERO(_f64, double)
// @BE spfloat_fcnmm_scatter_homo_auto_f64
FFI_SPFLOAT_FCN_MM_SCATTER_HOMO (_f64, double)
// @BE spfloat_fcnmm_scatter_hetero_auto_f64
FFI_SPFLOAT_FCN_MM_SCATTER_HETERO(_f64, double)

// ---- float16 ----
// @BE spfloat_fcnmm_gather_homo_auto_f16
FFI_SPFLOAT_FCN_MM_GATHER_HOMO  (_f16, __half)
// @BE spfloat_fcnmm_gather_hetero_auto_f16
FFI_SPFLOAT_FCN_MM_GATHER_HETERO(_f16, __half)
// @BE spfloat_fcnmm_scatter_homo_auto_f16
FFI_SPFLOAT_FCN_MM_SCATTER_HOMO (_f16, __half)
// @BE spfloat_fcnmm_scatter_hetero_auto_f16
FFI_SPFLOAT_FCN_MM_SCATTER_HETERO(_f16, __half)
