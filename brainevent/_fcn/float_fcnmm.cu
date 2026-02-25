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
 * float_fcnmm.cu -- Float-Weighted FCN Sparse Matrix-Matrix CUDA Kernels
 * ========================================================================
 *
 * This module provides optimized CUDA kernels for sparse operations with
 * fixed connection number (FCN) and floating-point weights:
 * 1. Sparse Matrix-Matrix Product (SpMM): fcnmm
 */
#include "cuda_common.h"

// ============================================================================
// FCN Matrix-Matrix Multiplication (fcnmm)
// ============================================================================

#define DEFINE_MM_GATHER_BASIC_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W)  \
__global__ void _mm_gather_basic_homo_kern##SUFFIX(                            \
    const int32_t* __restrict__ indices,                                       \
    const WEIGHT_T* __restrict__ matrix,                                       \
    WEIGHT_T*       __restrict__ output,                                       \
    const WEIGHT_T* __restrict__ weights,                                      \
    int n_pre, int n_conn, int n_col                                           \
) {                                                                            \
    int i = blockIdx.x;                                                        \
    int j = blockIdx.y * blockDim.x + threadIdx.x;                             \
    if (i >= n_pre || j >= n_col) return;                                      \
    const int32_t*  idx_row = indices + (size_t)i * n_conn;                    \
    ACC_T acc = (ACC_T)0;                                                      \
    for (int k = 0; k < n_conn; k++) {                                         \
        int32_t col = __ldg(&idx_row[k]);                                      \
        acc += READ_W(__ldg(&matrix[(size_t)col * n_col + j]));                \
    }                                                                          \
    output[(size_t)i * n_col + j] = WRITE_W(READ_W(__ldg(&weights[0])) * acc); \
}

#define DEFINE_MM_GATHER_BASIC_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W)            \
__global__ void _mm_gather_basic_hetero_kern##SUFFIX(                                      \
    const int32_t* __restrict__ indices,                                                   \
    const WEIGHT_T* __restrict__ matrix,                                                   \
    WEIGHT_T*       __restrict__ output,                                                   \
    const WEIGHT_T* __restrict__ weights,                                                  \
    int n_pre, int n_conn, int n_col                                                       \
) {                                                                                        \
    int i = blockIdx.x;                                                                    \
    int j = blockIdx.y * blockDim.x + threadIdx.x;                                         \
    if (i >= n_pre || j >= n_col) return;                                                  \
    const int32_t*  idx_row = indices + (size_t)i * n_conn;                                \
    const WEIGHT_T* w_row   = weights + (size_t)i * n_conn;                                \
    ACC_T acc = (ACC_T)0;                                                                  \
    for (int k = 0; k < n_conn; k++) {                                                     \
        int32_t col = __ldg(&idx_row[k]);                                                  \
        acc += READ_W(__ldg(&w_row[k])) * READ_W(__ldg(&matrix[(size_t)col * n_col + j])); \
    }                                                                                      \
    output[(size_t)i * n_col + j] = WRITE_W(acc);                                          \
}

#define DEFINE_MM_SCATTER_BLOCK_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _mm_scatter_block_homo_kern##SUFFIX(                               \
    const int32_t* __restrict__ indices,                                           \
    const WEIGHT_T* __restrict__ matrix,                                           \
    WEIGHT_T*       __restrict__ output,                                           \
    const WEIGHT_T* __restrict__ weights,                                          \
    int n_pre, int n_conn, int n_col                                               \
) {                                                                                \
    int i = blockIdx.x;                                                            \
    if (i >= n_pre) return;                                                        \
    const int32_t*  idx_row = indices + (size_t)i * n_conn;                        \
    const WEIGHT_T* m_row   = matrix + (size_t)i * n_col;                          \
    ACC_T w0 = READ_W(__ldg(&weights[0]));                                         \
    for (int k = 0; k < n_conn; k++) {                                             \
        int tgt = __ldg(&idx_row[k]);                                              \
        WEIGHT_T* out_row = output + (size_t)tgt * n_col;                          \
        for (int j = threadIdx.x; j < n_col; j += blockDim.x)                      \
            ATOMIC_ADD_W(&out_row[j], w0 * READ_W(__ldg(&m_row[j])));              \
    }                                                                              \
}

#define DEFINE_MM_SCATTER_BLOCK_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _mm_scatter_block_hetero_kern##SUFFIX(                               \
    const int32_t* __restrict__ indices,                                             \
    const WEIGHT_T* __restrict__ matrix,                                             \
    WEIGHT_T*       __restrict__ output,                                             \
    const WEIGHT_T* __restrict__ weights,                                            \
    int n_pre, int n_conn, int n_col                                                 \
) {                                                                                  \
    int i = blockIdx.x;                                                              \
    if (i >= n_pre) return;                                                          \
    const int32_t*  idx_row = indices + (size_t)i * n_conn;                          \
    const WEIGHT_T* w_row   = weights + (size_t)i * n_conn;                          \
    const WEIGHT_T* m_row   = matrix + (size_t)i * n_col;                            \
    for (int k = 0; k < n_conn; k++) {                                               \
        int tgt = __ldg(&idx_row[k]);                                                \
        ACC_T wk = READ_W(__ldg(&w_row[k]));                                         \
        WEIGHT_T* out_row = output + (size_t)tgt * n_col;                            \
        for (int j = threadIdx.x; j < n_col; j += blockDim.x)                        \
            ATOMIC_ADD_W(&out_row[j], wk * READ_W(__ldg(&m_row[j])));               \
    }                                                                                \
}

#define DEFINE_MM_SCATTER_WARP_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _mm_scatter_warp_homo_kern##SUFFIX(                               \
    const int32_t* __restrict__ indices,                                          \
    const WEIGHT_T* __restrict__ matrix,                                          \
    WEIGHT_T*       __restrict__ output,                                          \
    const WEIGHT_T* __restrict__ weights,                                         \
    int n_pre, int n_conn, int n_col                                              \
) {                                                                               \
    int wid     = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;                   \
    int lane    = threadIdx.x & 31;                                               \
    int n_warps = (gridDim.x * blockDim.x) >> 5;                                  \
    int n_pairs = n_pre * n_conn;                                                 \
    ACC_T w0 = READ_W(__ldg(&weights[0]));                                        \
    for (int pair = wid; pair < n_pairs; pair += n_warps) {                       \
        int i = pair / n_conn;                                                    \
        int k = pair % n_conn;                                                    \
        int tgt = __ldg(&indices[(size_t)i * n_conn + k]);                        \
        const WEIGHT_T* m_row   = matrix + (size_t)i * n_col;                     \
        WEIGHT_T*       out_row = output + (size_t)tgt * n_col;                   \
        for (int j = lane; j < n_col; j += 32)                                    \
            ATOMIC_ADD_W(&out_row[j], w0 * READ_W(__ldg(&m_row[j])));             \
    }                                                                             \
}

#define DEFINE_MM_SCATTER_WARP_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _mm_scatter_warp_hetero_kern##SUFFIX(                               \
    const int32_t* __restrict__ indices,                                            \
    const WEIGHT_T* __restrict__ matrix,                                            \
    WEIGHT_T*       __restrict__ output,                                            \
    const WEIGHT_T* __restrict__ weights,                                           \
    int n_pre, int n_conn, int n_col                                                \
) {                                                                                 \
    int wid     = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;                     \
    int lane    = threadIdx.x & 31;                                                 \
    int n_warps = (gridDim.x * blockDim.x) >> 5;                                    \
    int n_pairs = n_pre * n_conn;                                                   \
    for (int pair = wid; pair < n_pairs; pair += n_warps) {                         \
        int i = pair / n_conn;                                                      \
        int k = pair % n_conn;                                                      \
        int tgt = __ldg(&indices[(size_t)i * n_conn + k]);                          \
        ACC_T wk = READ_W(__ldg(&weights[(size_t)i * n_conn + k]));                 \
        const WEIGHT_T* m_row   = matrix + (size_t)i * n_col;                       \
        WEIGHT_T*       out_row = output + (size_t)tgt * n_col;                     \
        for (int j = lane; j < n_col; j += 32)                                      \
            ATOMIC_ADD_W(&out_row[j], wk * READ_W(__ldg(&m_row[j])));               \
    }                                                                               \
}

// SpMM Instantiations
// ---- float32 ----
DEFINE_MM_GATHER_BASIC_HOMO  (_f32, float, float, READ_F32, WRITE_F32)
DEFINE_MM_GATHER_BASIC_HETERO(_f32, float, float, READ_F32, WRITE_F32)
DEFINE_MM_SCATTER_BLOCK_HOMO (_f32, float, float, READ_F32, atomic_add_f32)
DEFINE_MM_SCATTER_BLOCK_HETERO(_f32, float, float, READ_F32, atomic_add_f32)
DEFINE_MM_SCATTER_WARP_HOMO  (_f32, float, float, READ_F32, atomic_add_f32)
DEFINE_MM_SCATTER_WARP_HETERO(_f32, float, float, READ_F32, atomic_add_f32)

// ---- float64 ----
DEFINE_MM_GATHER_BASIC_HOMO  (_f64, double, double, READ_F64, WRITE_F64)
DEFINE_MM_GATHER_BASIC_HETERO(_f64, double, double, READ_F64, WRITE_F64)
DEFINE_MM_SCATTER_BLOCK_HOMO (_f64, double, double, READ_F64, atomic_add_f64)
DEFINE_MM_SCATTER_BLOCK_HETERO(_f64, double, double, READ_F64, atomic_add_f64)
DEFINE_MM_SCATTER_WARP_HOMO  (_f64, double, double, READ_F64, atomic_add_f64)
DEFINE_MM_SCATTER_WARP_HETERO(_f64, double, double, READ_F64, atomic_add_f64)

// ---- float16 ----
DEFINE_MM_GATHER_BASIC_HOMO  (_f16, __half, float, READ_F16, WRITE_F16)
DEFINE_MM_GATHER_BASIC_HETERO(_f16, __half, float, READ_F16, WRITE_F16)
DEFINE_MM_SCATTER_BLOCK_HOMO (_f16, __half, float, READ_F16, atomic_add_f16)
DEFINE_MM_SCATTER_BLOCK_HETERO(_f16, __half, float, READ_F16, atomic_add_f16)
DEFINE_MM_SCATTER_WARP_HOMO  (_f16, __half, float, READ_F16, atomic_add_f16)
DEFINE_MM_SCATTER_WARP_HETERO(_f16, __half, float, READ_F16, atomic_add_f16)

// ---- bfloat16 ----
DEFINE_MM_GATHER_BASIC_HOMO  (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16)
DEFINE_MM_GATHER_BASIC_HETERO(_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16)
DEFINE_MM_SCATTER_BLOCK_HOMO (_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_MM_SCATTER_BLOCK_HETERO(_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_MM_SCATTER_WARP_HOMO  (_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_MM_SCATTER_WARP_HETERO(_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)

// SpMM Specializations (f32 only)
#define MMTK 128
__global__ void _mm_gather_shared_homo_kern(const int32_t* __restrict__ indices, const float* __restrict__ matrix, float* __restrict__ output, const float* __restrict__ weights, int n_pre, int n_conn, int n_col) {
    extern __shared__ char smem_mm[];
    int32_t* s_idx = reinterpret_cast<int32_t*>(smem_mm);
    int i = blockIdx.x, j = blockIdx.y * blockDim.x + threadIdx.x; if (i >= n_pre) return;
    const int32_t* idx_row = indices + (size_t)i * n_conn;
    float acc = 0.0f;
    for (int k0 = 0; k0 < n_conn; k0 += MMTK) {
        int tile = (k0 + MMTK < n_conn) ? MMTK : (n_conn - k0);
        for (int t = threadIdx.x; t < tile; t += blockDim.x) { s_idx[t] = __ldg(&idx_row[k0 + t]); }
        __syncthreads();
        if (j < n_col) for (int t = 0; t < tile; t++) acc += __ldg(&matrix[(size_t)s_idx[t] * n_col + j]);
        __syncthreads();
    }
    if (j < n_col) output[(size_t)i * n_col + j] = __ldg(&weights[0]) * acc;
}

__global__ void _mm_gather_shared_hetero_kern(const int32_t* __restrict__ indices, const float* __restrict__ matrix, float* __restrict__ output, const float* __restrict__ weights, int n_pre, int n_conn, int n_col) {
    extern __shared__ char smem_mm[];
    int32_t* s_idx = reinterpret_cast<int32_t*>(smem_mm);
    float*   s_w   = reinterpret_cast<float*>(smem_mm + MMTK * sizeof(int32_t));
    int i = blockIdx.x, j = blockIdx.y * blockDim.x + threadIdx.x; if (i >= n_pre) return;
    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const float*   w_row   = weights + (size_t)i * n_conn;
    float acc = 0.0f;
    for (int k0 = 0; k0 < n_conn; k0 += MMTK) {
        int tile = (k0 + MMTK < n_conn) ? MMTK : (n_conn - k0);
        for (int t = threadIdx.x; t < tile; t += blockDim.x) { s_idx[t] = __ldg(&idx_row[k0 + t]); s_w[t] = __ldg(&w_row[k0 + t]); }
        __syncthreads();
        if (j < n_col) for (int t = 0; t < tile; t++) acc += s_w[t] * __ldg(&matrix[(size_t)s_idx[t] * n_col + j]);
        __syncthreads();
    }
    if (j < n_col) output[(size_t)i * n_col + j] = acc;
}

__global__ void _mm_gather_vec4_homo_kern(const int32_t* __restrict__ indices, const float* __restrict__ matrix, float* __restrict__ output, const float* __restrict__ weights, int n_pre, int n_conn, int n_col) {
    int i = blockIdx.x, j4 = blockIdx.y * blockDim.x + threadIdx.x, nc4 = n_col >> 2; if (i >= n_pre || j4 >= nc4) return;
    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const float4* mat4 = (const float4*)matrix; float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
    float w = __ldg(&weights[0]);
    for (int k = 0; k < n_conn; k++) {
        float4 m = __ldg(&mat4[(size_t)__ldg(&idx_row[k]) * nc4 + j4]);
        acc.x += w * m.x; acc.y += w * m.y; acc.z += w * m.z; acc.w += w * m.w;
    }
    ((float4*)output)[(size_t)i * nc4 + j4] = acc;
}

__global__ void _mm_gather_vec4_hetero_kern(const int32_t* __restrict__ indices, const float* __restrict__ vector, float* __restrict__ output, const float* __restrict__ weights, int n_pre, int n_conn, int n_col) {
    int i = blockIdx.x, j4 = blockIdx.y * blockDim.x + threadIdx.x, nc4 = n_col >> 2; if (i >= n_pre || j4 >= nc4) return;
    const int32_t* idx_row = indices + (size_t)i * n_conn; const float* w_row = weights + (size_t)i * n_conn;
    const float4* mat4 = (const float4*)vector; float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = 0; k < n_conn; k++) {
        float wk = __ldg(&w_row[k]);
        float4 m = __ldg(&mat4[(size_t)__ldg(&idx_row[k]) * nc4 + j4]);
        acc.x += wk * m.x; acc.y += wk * m.y; acc.z += wk * m.z; acc.w += wk * m.w;
    }
    ((float4*)output)[(size_t)i * nc4 + j4] = acc;
}

__global__ void _mm_gather_shm_vec4_homo_kern(const int32_t* __restrict__ indices, const float* __restrict__ matrix, float* __restrict__ output, const float* __restrict__ weights, int n_pre, int n_conn, int n_col) {
    extern __shared__ char smem_sv[];
    int32_t* s_idx = reinterpret_cast<int32_t*>(smem_sv);
    int i = blockIdx.x, j4 = blockIdx.y * blockDim.x + threadIdx.x, nc4 = n_col >> 2;
    if (i >= n_pre || j4 >= nc4) return;
    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const float4*  mat4    = (const float4*)matrix;
    float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k0 = 0; k0 < n_conn; k0 += MMTK) {
        int tile = (k0 + MMTK <= n_conn) ? MMTK : (n_conn - k0);
        for (int t = threadIdx.x; t < tile; t += blockDim.x) {
            s_idx[t] = __ldg(&idx_row[k0 + t]);
        }
        __syncthreads();
        for (int t = 0; t < tile; t++) {
            float4 m = __ldg(&mat4[(size_t)s_idx[t] * nc4 + j4]);
            acc.x += m.x; acc.y += m.y; acc.z += m.z; acc.w += m.w;
        }
        __syncthreads();
    }
    float hw = __ldg(&weights[0]); acc.x *= hw; acc.y *= hw; acc.z *= hw; acc.w *= hw;
    ((float4*)output)[(size_t)i * nc4 + j4] = acc;
}

__global__ void _mm_gather_shm_vec4_hetero_kern(const int32_t* __restrict__ indices, const float* __restrict__ matrix, float* __restrict__ output, const float* __restrict__ weights, int n_pre, int n_conn, int n_col) {
    extern __shared__ char smem_sv[];
    int32_t* s_idx = reinterpret_cast<int32_t*>(smem_sv);
    float*   s_w   = reinterpret_cast<float*>(smem_sv + MMTK * sizeof(int32_t));
    int i = blockIdx.x, j4 = blockIdx.y * blockDim.x + threadIdx.x, nc4 = n_col >> 2;
    if (i >= n_pre || j4 >= nc4) return;
    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const float*   w_row   = weights + (size_t)i * n_conn;
    const float4*  mat4    = (const float4*)matrix;
    float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k0 = 0; k0 < n_conn; k0 += MMTK) {
        int tile = (k0 + MMTK <= n_conn) ? MMTK : (n_conn - k0);
        for (int t = threadIdx.x; t < tile; t += blockDim.x) {
            s_idx[t] = __ldg(&idx_row[k0 + t]);
            s_w[t] = __ldg(&w_row[k0 + t]);
        }
        __syncthreads();
        for (int t = 0; t < tile; t++) {
            float w = s_w[t];
            float4 m = __ldg(&mat4[(size_t)s_idx[t] * nc4 + j4]);
            acc.x += w * m.x; acc.y += w * m.y; acc.z += w * m.z; acc.w += w * m.w;
        }
        __syncthreads();
    }
    ((float4*)output)[(size_t)i * nc4 + j4] = acc;
}

#define MM_SCATTER_BJ 128
#define SCATTER_TK 64
__global__ void _mm_scatter_cached_homo_kern(const int32_t* __restrict__ indices, const float* __restrict__ matrix, float* __restrict__ output, const float* __restrict__ weights, int n_pre, int n_conn, int n_col) {
    extern __shared__ char smem_sc[];
    float*   s_m   = reinterpret_cast<float*>(smem_sc);
    int32_t* s_idx = reinterpret_cast<int32_t*>(smem_sc + blockDim.x * sizeof(float));
    int i = blockIdx.x, j = blockIdx.y * blockDim.x + threadIdx.x; if (i >= n_pre) return;
    s_m[threadIdx.x] = (j < n_col) ? __ldg(&matrix[(size_t)i * n_col + j]) : 0.0f; __syncthreads();
    const int32_t* idx_row = indices + (size_t)i * n_conn;
    float w0 = __ldg(&weights[0]);
    if (j < n_col) {
        float mw = s_m[threadIdx.x] * w0;
        for (int k0 = 0; k0 < n_conn; k0 += SCATTER_TK) {
            int tile = (k0 + SCATTER_TK <= n_conn) ? SCATTER_TK : (n_conn - k0);
            for (int t = threadIdx.x; t < tile; t += blockDim.x) { s_idx[t] = __ldg(&idx_row[k0 + t]); }
            __syncthreads();
            for (int t = 0; t < tile; t++) { atomic_add_f32(&output[(size_t)s_idx[t] * n_col + j], mw); }
            __syncthreads();
        }
    }
}

__global__ void _mm_scatter_cached_hetero_kern(const int32_t* __restrict__ indices, const float* __restrict__ matrix, float* __restrict__ output, const float* __restrict__ weights, int n_pre, int n_conn, int n_col) {
    extern __shared__ char smem_sc[];
    float*   s_m   = reinterpret_cast<float*>(smem_sc);
    int32_t* s_idx = reinterpret_cast<int32_t*>(smem_sc + blockDim.x * sizeof(float));
    float*   s_w   = reinterpret_cast<float*>(smem_sc + blockDim.x * sizeof(float) + SCATTER_TK * sizeof(int32_t));
    int i = blockIdx.x, j = blockIdx.y * blockDim.x + threadIdx.x; if (i >= n_pre) return;
    s_m[threadIdx.x] = (j < n_col) ? __ldg(&matrix[(size_t)i * n_col + j]) : 0.0f; __syncthreads();
    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const float* w_row = weights + (size_t)i * n_conn;
    if (j < n_col) {
        float m_val = s_m[threadIdx.x];
        for (int k0 = 0; k0 < n_conn; k0 += SCATTER_TK) {
            int tile = (k0 + SCATTER_TK <= n_conn) ? SCATTER_TK : (n_conn - k0);
            for (int t = threadIdx.x; t < tile; t += blockDim.x) {
                s_idx[t] = __ldg(&idx_row[k0 + t]);
                s_w[t] = __ldg(&w_row[k0 + t]);
            }
            __syncthreads();
            for (int t = 0; t < tile; t++) { atomic_add_f32(&output[(size_t)s_idx[t] * n_col + j], s_w[t] * m_val); }
            __syncthreads();
        }
    }
}

// SpMM FFI Entries
// ---- FFI macro: gather homo auto ----
#define FFI_MM_GATHER_HOMO_AUTO(SUFFIX, WEIGHT_C_T)                                       \
void fcnmm_gather_homo_auto##SUFFIX(                                                      \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                           \
    tvm::ffi::TensorView matrix,  tvm::ffi::TensorView output, int64_t stream             \
) {                                                                                       \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                             \
    int n_pre       = static_cast<int>(indices.size(0));                                  \
    int n_conn      = static_cast<int>(indices.size(1));                                  \
    int n_col       = static_cast<int>(matrix.size(1));                                   \
    int bk = ((n_col + 31) / 32) * 32;                                                    \
    if (bk < 32) bk = 32; else if (bk > 256) bk = 256;                                    \
    _mm_gather_basic_homo_kern##SUFFIX<<<dim3(n_pre, (n_col + bk - 1) / bk), bk, 0, s>>>( \
        static_cast<const int32_t*>(indices.data_ptr()),                                  \
        static_cast<const WEIGHT_C_T*>(matrix.data_ptr()),                                \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                                      \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                               \
        n_pre, n_conn, n_col);                                                            \
}

// ---- FFI macro: gather hetero auto ----
#define FFI_MM_GATHER_HETERO_AUTO(SUFFIX, WEIGHT_C_T)                                       \
void fcnmm_gather_hetero_auto##SUFFIX(                                                      \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                             \
    tvm::ffi::TensorView matrix,  tvm::ffi::TensorView output, int64_t stream               \
) {                                                                                         \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                               \
    int n_pre       = static_cast<int>(indices.size(0));                                    \
    int n_conn      = static_cast<int>(indices.size(1));                                    \
    int n_col       = static_cast<int>(matrix.size(1));                                     \
    int bk = ((n_col + 31) / 32) * 32;                                                      \
    if (bk < 32) bk = 32; else if (bk > 256) bk = 256;                                      \
    _mm_gather_basic_hetero_kern##SUFFIX<<<dim3(n_pre, (n_col + bk - 1) / bk), bk, 0, s>>>( \
        static_cast<const int32_t*>(indices.data_ptr()),                                    \
        static_cast<const WEIGHT_C_T*>(matrix.data_ptr()),                                  \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                                        \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                                 \
        n_pre, n_conn, n_col);                                                              \
}

// ---- FFI macro: scatter homo auto ----
#define FFI_MM_SCATTER_HOMO_AUTO(SUFFIX, WEIGHT_C_T)                                       \
void fcnmm_scatter_homo_auto##SUFFIX(                                                      \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                            \
    tvm::ffi::TensorView matrix,  tvm::ffi::TensorView output, int64_t stream              \
) {                                                                                        \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                              \
    int n_pre       = static_cast<int>(indices.size(0));                                   \
    int n_conn      = static_cast<int>(indices.size(1));                                   \
    int n_post      = static_cast<int>(output.size(0));                                    \
    int n_col       = static_cast<int>(matrix.size(1));                                    \
    cudaMemsetAsync(output.data_ptr(), 0, (size_t)n_post * n_col * sizeof(WEIGHT_C_T), s); \
    if (n_col <= 64) {                                                                     \
        int n_pairs = n_pre * n_conn;                                                      \
        int blocks  = min(4096, (n_pairs + 7) / 8);                                        \
        _mm_scatter_warp_homo_kern##SUFFIX<<<blocks, 256, 0, s>>>(                         \
            static_cast<const int32_t*>(indices.data_ptr()),                               \
            static_cast<const WEIGHT_C_T*>(matrix.data_ptr()),                             \
            static_cast<WEIGHT_C_T*>(output.data_ptr()),                                   \
            static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                            \
            n_pre, n_conn, n_col);                                                         \
    } else {                                                                               \
        _mm_scatter_block_homo_kern##SUFFIX<<<n_pre, 256, 0, s>>>(                         \
            static_cast<const int32_t*>(indices.data_ptr()),                               \
            static_cast<const WEIGHT_C_T*>(matrix.data_ptr()),                             \
            static_cast<WEIGHT_C_T*>(output.data_ptr()),                                   \
            static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                            \
            n_pre, n_conn, n_col);                                                         \
    }                                                                                      \
}

// ---- FFI macro: scatter hetero auto ----
#define FFI_MM_SCATTER_HETERO_AUTO(SUFFIX, WEIGHT_C_T)                                     \
void fcnmm_scatter_hetero_auto##SUFFIX(                                                    \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                            \
    tvm::ffi::TensorView matrix,  tvm::ffi::TensorView output, int64_t stream              \
) {                                                                                        \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                              \
    int n_pre       = static_cast<int>(indices.size(0));                                   \
    int n_conn      = static_cast<int>(indices.size(1));                                   \
    int n_post      = static_cast<int>(output.size(0));                                    \
    int n_col       = static_cast<int>(matrix.size(1));                                    \
    cudaMemsetAsync(output.data_ptr(), 0, (size_t)n_post * n_col * sizeof(WEIGHT_C_T), s); \
    if (n_col <= 64) {                                                                     \
        int n_pairs = n_pre * n_conn;                                                      \
        int blocks  = min(4096, (n_pairs + 7) / 8);                                        \
        _mm_scatter_warp_hetero_kern##SUFFIX<<<blocks, 256, 0, s>>>(                       \
            static_cast<const int32_t*>(indices.data_ptr()),                               \
            static_cast<const WEIGHT_C_T*>(matrix.data_ptr()),                             \
            static_cast<WEIGHT_C_T*>(output.data_ptr()),                                   \
            static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                            \
            n_pre, n_conn, n_col);                                                         \
    } else {                                                                               \
        _mm_scatter_block_hetero_kern##SUFFIX<<<n_pre, 256, 0, s>>>(                       \
            static_cast<const int32_t*>(indices.data_ptr()),                               \
            static_cast<const WEIGHT_C_T*>(matrix.data_ptr()),                             \
            static_cast<WEIGHT_C_T*>(output.data_ptr()),                                   \
            static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                            \
            n_pre, n_conn, n_col);                                                         \
    }                                                                                      \
}

// SpMM FFI Instantiations
// ---- float32 ----
// @tvm_ffi fcnmm_gather_homo_auto_f32
FFI_MM_GATHER_HOMO_AUTO  (_f32, float)
// @tvm_ffi fcnmm_gather_hetero_auto_f32
FFI_MM_GATHER_HETERO_AUTO(_f32, float)
// @tvm_ffi fcnmm_scatter_homo_auto_f32
FFI_MM_SCATTER_HOMO_AUTO (_f32, float)
// @tvm_ffi fcnmm_scatter_hetero_auto_f32
FFI_MM_SCATTER_HETERO_AUTO(_f32, float)

// ---- float64 ----
// @tvm_ffi fcnmm_gather_homo_auto_f64
FFI_MM_GATHER_HOMO_AUTO  (_f64, double)
// @tvm_ffi fcnmm_gather_hetero_auto_f64
FFI_MM_GATHER_HETERO_AUTO(_f64, double)
// @tvm_ffi fcnmm_scatter_homo_auto_f64
FFI_MM_SCATTER_HOMO_AUTO (_f64, double)
// @tvm_ffi fcnmm_scatter_hetero_auto_f64
FFI_MM_SCATTER_HETERO_AUTO(_f64, double)

// ---- float16 ----
// @tvm_ffi fcnmm_gather_homo_auto_f16
FFI_MM_GATHER_HOMO_AUTO  (_f16, __half)
// @tvm_ffi fcnmm_gather_hetero_auto_f16
FFI_MM_GATHER_HETERO_AUTO(_f16, __half)
// @tvm_ffi fcnmm_scatter_homo_auto_f16
FFI_MM_SCATTER_HOMO_AUTO (_f16, __half)
// @tvm_ffi fcnmm_scatter_hetero_auto_f16
FFI_MM_SCATTER_HETERO_AUTO(_f16, __half)

// ---- bfloat16 ----
// @tvm_ffi fcnmm_gather_homo_auto_bf16
FFI_MM_GATHER_HOMO_AUTO  (_bf16, __nv_bfloat16)
// @tvm_ffi fcnmm_gather_hetero_auto_bf16
FFI_MM_GATHER_HETERO_AUTO(_bf16, __nv_bfloat16)
// @tvm_ffi fcnmm_scatter_homo_auto_bf16
FFI_MM_SCATTER_HOMO_AUTO (_bf16, __nv_bfloat16)
// @tvm_ffi fcnmm_scatter_hetero_auto_bf16
FFI_MM_SCATTER_HETERO_AUTO(_bf16, __nv_bfloat16)

// SpMM f32-specific specializations
// @tvm_ffi fcnmm_gather_vec4_homo_f32
void fcnmm_gather_vec4_homo_f32(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_col  = static_cast<int>(matrix.size(1));
    dim3 grid(n_pre, (n_col / 4 + 63) / 64);
    _mm_gather_vec4_homo_kern<<<grid, 64, 0, s>>>(
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>(matrix.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        n_pre, n_conn, n_col);
}

// @tvm_ffi fcnmm_gather_vec4_hetero_f32
void fcnmm_gather_vec4_hetero_f32(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_col  = static_cast<int>(matrix.size(1));
    dim3 grid(n_pre, (n_col / 4 + 63) / 64);
    _mm_gather_vec4_hetero_kern<<<grid, 64, 0, s>>>(
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>(matrix.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        n_pre, n_conn, n_col);
}

// @tvm_ffi fcnmm_gather_shared_homo_f32
void fcnmm_gather_shared_homo_f32(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_col  = static_cast<int>(matrix.size(1));
    dim3 grid(n_pre, (n_col + 63) / 64);
    size_t shm = MMTK * sizeof(int32_t);
    _mm_gather_shared_homo_kern<<<grid, 64, shm, s>>>(
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>(matrix.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        n_pre, n_conn, n_col);
}

// @tvm_ffi fcnmm_gather_shared_hetero_f32
void fcnmm_gather_shared_hetero_f32(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_col  = static_cast<int>(matrix.size(1));
    dim3 grid(n_pre, (n_col + 63) / 64);
    size_t shm = MMTK * (sizeof(int32_t) + sizeof(float));
    _mm_gather_shared_hetero_kern<<<grid, 64, shm, s>>>(
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>(matrix.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        n_pre, n_conn, n_col);
}

/*
 * fcnmm gather auto (f32) — dispatch strategy:
 *
 * Three kernel tiers, selected by n_col and n_conn:
 *
 *   1. Vec4 + shared-memory tiling (_mm_gather_shm_vec4_kern):
 *      Used when n_col >= 128 and n_col % 4 == 0.  float4 loads give 4x fewer
 *      load instructions; shared-memory tiles indices/weights for reuse.
 *      blockDim = min(128, nc4) rounded to warp multiple.
 *
 *   2. Scalar + shared-memory tiling (_mm_gather_shared_kern):
 *      Used when n_col < 128 (or n_col not aligned to 4) and n_conn > 64.
 *      blockDim adapts to n_col to avoid thread waste.
 *
 *   3. Scalar basic (_mm_gather_basic_kern_f32):
 *      Fallback for small n_conn (<= 64) where tiling overhead isn't worthwhile.
 */
// @tvm_ffi fcnmm_gather_auto_f32
void fcnmm_gather_auto_f32(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_col  = static_cast<int>(matrix.size(1));
    bool homo = (weights.ndim() == 1);
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_mat = static_cast<const float*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    int nc4 = n_col / 4;
    if (n_col % 4 == 0 && nc4 >= 32) {
        int bk = nc4; if (bk > 128) bk = 128; bk = ((bk + 31) / 32) * 32;
        dim3 grid(n_pre, (nc4 + bk - 1) / bk);
        size_t shm = MMTK * (sizeof(int32_t) + (homo ? 0 : sizeof(float)));
        if (homo) _mm_gather_shm_vec4_homo_kern<<<grid, bk, shm, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col);
        else      _mm_gather_shm_vec4_hetero_kern<<<grid, bk, shm, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col);
    } else {
        int bk = ((n_col + 31) / 32) * 32; if (bk < 64) bk = 64; if (bk > 256) bk = 256;
        dim3 grid(n_pre, (n_col + bk - 1) / bk);
        if (n_conn > 64) {
            size_t shm = MMTK * (sizeof(int32_t) + (homo ? 0 : sizeof(float)));
            if (homo) _mm_gather_shared_homo_kern<<<grid, bk, shm, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col);
            else      _mm_gather_shared_hetero_kern<<<grid, bk, shm, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col);
        } else {
            if (homo) _mm_gather_basic_homo_kern_f32<<<grid, bk, 0, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col);
            else      _mm_gather_basic_hetero_kern_f32<<<grid, bk, 0, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col);
        }
    }
}

// @tvm_ffi fcnmm_scatter_auto_f32
void fcnmm_scatter_auto_f32(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int n_col  = static_cast<int>(matrix.size(1));
    bool homo = (weights.ndim() == 1);
    cudaMemsetAsync(output.data_ptr(), 0, (size_t)n_post * n_col * sizeof(float), s);
    if (n_col <= 64) {
        int n_pairs = n_pre * n_conn;
        int blocks  = min(4096, (n_pairs + 7) / 8);
        if (homo) _mm_scatter_warp_homo_kern_f32<<<blocks, 256, 0, s>>>(static_cast<const int32_t*>(indices.data_ptr()), static_cast<const float*>(matrix.data_ptr()), static_cast<float*>(output.data_ptr()), static_cast<const float*>(weights.data_ptr()), n_pre, n_conn, n_col);
        else      _mm_scatter_warp_hetero_kern_f32<<<blocks, 256, 0, s>>>(static_cast<const int32_t*>(indices.data_ptr()), static_cast<const float*>(matrix.data_ptr()), static_cast<float*>(output.data_ptr()), static_cast<const float*>(weights.data_ptr()), n_pre, n_conn, n_col);
    } else if (n_conn > 32) {
        int BJ = 128; dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
        size_t shm = BJ * sizeof(float) + SCATTER_TK * (sizeof(int32_t) + (homo ? 0 : sizeof(float)));
        if (homo) _mm_scatter_cached_homo_kern<<<grid, BJ, shm, s>>>(static_cast<const int32_t*>(indices.data_ptr()), static_cast<const float*>(matrix.data_ptr()), static_cast<float*>(output.data_ptr()), static_cast<const float*>(weights.data_ptr()), n_pre, n_conn, n_col);
        else      _mm_scatter_cached_hetero_kern<<<grid, BJ, shm, s>>>(static_cast<const int32_t*>(indices.data_ptr()), static_cast<const float*>(matrix.data_ptr()), static_cast<float*>(output.data_ptr()), static_cast<const float*>(weights.data_ptr()), n_pre, n_conn, n_col);
    } else {
        if (homo) _mm_scatter_block_homo_kern_f32<<<n_pre, 256, 0, s>>>(static_cast<const int32_t*>(indices.data_ptr()), static_cast<const float*>(matrix.data_ptr()), static_cast<float*>(output.data_ptr()), static_cast<const float*>(weights.data_ptr()), n_pre, n_conn, n_col);
        else      _mm_scatter_block_hetero_kern_f32<<<n_pre, 256, 0, s>>>(static_cast<const int32_t*>(indices.data_ptr()), static_cast<const float*>(matrix.data_ptr()), static_cast<float*>(output.data_ptr()), static_cast<const float*>(weights.data_ptr()), n_pre, n_conn, n_col);
    }
}
