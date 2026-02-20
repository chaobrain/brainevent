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
 * sparse_float.cu -- Sparse-Float Dense Matrix-Vector and Matrix-Matrix CUDA Kernels
 * ==================================================================================
 *
 * This module provides optimized CUDA kernels for dense operations with sparse-float
 * inputs. It includes:
 * 1. Sparse Matrix-Vector Product (SpMV): spfloat_densemv
 * 2. Sparse Matrix-Matrix Product (SpMM): spfloat_densemm
 *
 * These kernels exploit "sparse-float" sparsity: only non-zero entries in the
 * dense input (vector or matrix) contribute to the output, similar to binary
 * event-driven kernels but using the actual floating-point values for multiplication.
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
// Dense Matrix-Vector Multiplication (spfloat_densemv)
// =========================================================================

#define DEFINE_SPFLOAT_GATHER_WARP(SUFFIX, WEIGHT_T, ACC_T,                \
                                    READ_W, WRITE_W, READ_S,                \
                                    WARP_RED, ACC_ZERO)                     \
__global__ void _spfloat_gather_warp_kern##SUFFIX(                          \
    const WEIGHT_T* __restrict__ weights,                                   \
    const WEIGHT_T* __restrict__ spikes,                                    \
    WEIGHT_T*       __restrict__ output,                                    \
    int m, int k                                                            \
) {                                                                         \
    int row = blockIdx.x;                                                   \
    if (row >= m) return;                                                   \
    const WEIGHT_T* w_row = weights + (size_t)row * k;                     \
    ACC_T acc = ACC_ZERO;                                                   \
    for (int j = threadIdx.x; j < k; j += 32) {                           \
        ACC_T spk_val = READ_S(spikes[j]);                                 \
        if (spk_val != ACC_ZERO) {                                         \
            acc += READ_W(w_row[j]) * spk_val;                            \
        }                                                                   \
    }                                                                       \
    acc = WARP_RED(acc);                                                   \
    if (threadIdx.x == 0) output[row] = WRITE_W(acc);                     \
}

#define DEFINE_SPFLOAT_GATHER_BLOCK(SUFFIX, WEIGHT_T, ACC_T,               \
                                     READ_W, WRITE_W, READ_S,               \
                                     WARP_RED, ACC_ZERO)                    \
__global__ void _spfloat_gather_block_kern##SUFFIX(                         \
    const WEIGHT_T* __restrict__ weights,                                   \
    const WEIGHT_T* __restrict__ spikes,                                    \
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
        ACC_T spk_val = READ_S(spikes[j]);                                 \
        if (spk_val != ACC_ZERO) {                                         \
            acc += READ_W(w_row[j]) * spk_val;                            \
        }                                                                   \
    }                                                                       \
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

#define DEFINE_SPFLOAT_SCATTER(SUFFIX, WEIGHT_T, ACC_T,                    \
                                READ_W, WRITE_W, READ_S, ACC_ZERO)          \
__global__ void _spfloat_scatter_kern##SUFFIX(                              \
    const WEIGHT_T* __restrict__ weights,                                   \
    const WEIGHT_T* __restrict__ spikes,                                    \
    WEIGHT_T*       __restrict__ output,                                    \
    int k, int n                                                            \
) {                                                                         \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (j >= n) return;                                                    \
    ACC_T acc = ACC_ZERO;                                                   \
    for (int i = 0; i < k; i++) {                                          \
        ACC_T spk_val = READ_S(spikes[i]);                                 \
        if (spk_val != ACC_ZERO) {                                         \
            acc += READ_W(weights[(size_t)i * n + j]) * spk_val;          \
        }                                                                   \
    }                                                                       \
    output[j] = WRITE_W(acc);                                              \
}

// SpMV Instantiations
DEFINE_SPFLOAT_GATHER_WARP(_f32,  float, float, READ_F32, WRITE_F32, READ_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_GATHER_BLOCK(_f32, float, float, READ_F32, WRITE_F32, READ_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_SCATTER(_f32,      float, float, READ_F32, WRITE_F32, READ_F32, 0.0f)
DEFINE_SPFLOAT_GATHER_WARP(_f64,  double, double, READ_F64, WRITE_F64, READ_F64, warp_reduce_sum_f64, 0.0)
DEFINE_SPFLOAT_GATHER_BLOCK(_f64, double, double, READ_F64, WRITE_F64, READ_F64, warp_reduce_sum_f64, 0.0)
DEFINE_SPFLOAT_SCATTER(_f64,      double, double, READ_F64, WRITE_F64, READ_F64, 0.0)
DEFINE_SPFLOAT_GATHER_WARP(_f16,  __half, float, READ_F16, WRITE_F16, READ_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_GATHER_BLOCK(_f16, __half, float, READ_F16, WRITE_F16, READ_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_SCATTER(_f16,      __half, float, READ_F16, WRITE_F16, READ_F16, 0.0f)
DEFINE_SPFLOAT_GATHER_WARP(_bf16,  __nv_bfloat16, float, READ_BF16, WRITE_BF16, READ_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_GATHER_BLOCK(_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, READ_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_SCATTER(_bf16,      __nv_bfloat16, float, READ_BF16, WRITE_BF16, READ_BF16, 0.0f)

// FFI Macros for SpMV
#define FFI_SPFLOAT_GATHER_WARP(SUFFIX, WEIGHT_C_T)                        \
void spfloat_densemv_gather_warp##SUFFIX(                                   \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView spikes,            \
    tvm::ffi::TensorView output, int64_t stream                            \
) {                                                                         \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);               \
    int m = static_cast<int>(weights.size(0));                             \
    int k = static_cast<int>(weights.size(1));                             \
    _spfloat_gather_warp_kern##SUFFIX<<<m, 32, 0, s>>>(                    \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                \
        static_cast<const WEIGHT_C_T*>(spikes.data_ptr()),                 \
        static_cast<WEIGHT_C_T*>(output.data_ptr()), m, k);               \
}

#define FFI_SPFLOAT_GATHER_BLOCK(SUFFIX, WEIGHT_C_T, SHM_SIZE)            \
void spfloat_densemv_gather_block##SUFFIX(                                  \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView spikes,            \
    tvm::ffi::TensorView output, int64_t stream                            \
) {                                                                         \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);               \
    int m = static_cast<int>(weights.size(0));                             \
    int k = static_cast<int>(weights.size(1));                             \
    _spfloat_gather_block_kern##SUFFIX<<<m, 256, SHM_SIZE, s>>>(          \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                \
        static_cast<const WEIGHT_C_T*>(spikes.data_ptr()),                 \
        static_cast<WEIGHT_C_T*>(output.data_ptr()), m, k);               \
}

#define FFI_SPFLOAT_GATHER_AUTO(SUFFIX, WEIGHT_C_T, SHM_SIZE)             \
void spfloat_densemv_gather_auto##SUFFIX(                                   \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView spikes,            \
    tvm::ffi::TensorView output, int64_t stream                            \
) {                                                                         \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);               \
    int m = static_cast<int>(weights.size(0));                             \
    int k = static_cast<int>(weights.size(1));                             \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const WEIGHT_C_T* d_spk = static_cast<const WEIGHT_C_T*>(spikes.data_ptr()); \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr()); \
    if (k <= 1024) {                                                        \
        _spfloat_gather_warp_kern##SUFFIX<<<m, 32, 0, s>>>(d_w, d_spk, d_out, m, k); \
    } else {                                                                \
        _spfloat_gather_block_kern##SUFFIX<<<m, 256, SHM_SIZE, s>>>(d_w, d_spk, d_out, m, k); \
    }                                                                       \
}

#define FFI_SPFLOAT_SCATTER(SUFFIX, WEIGHT_C_T)                            \
void spfloat_densemv_scatter##SUFFIX(                                       \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView spikes,            \
    tvm::ffi::TensorView output, int64_t stream                            \
) {                                                                         \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);               \
    int k = static_cast<int>(weights.size(0));                             \
    int n = static_cast<int>(weights.size(1));                             \
    int blocks = (n + 255) / 256;                                          \
    _spfloat_scatter_kern##SUFFIX<<<blocks, 256, 0, s>>>(                  \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                \
        static_cast<const WEIGHT_C_T*>(spikes.data_ptr()),                 \
        static_cast<WEIGHT_C_T*>(output.data_ptr()), k, n);               \
}

// @tvm_ffi spfloat_densemv_gather_warp_f32
FFI_SPFLOAT_GATHER_WARP(_f32,    float)
// @tvm_ffi spfloat_densemv_gather_block_f32
FFI_SPFLOAT_GATHER_BLOCK(_f32,   float,  32 * sizeof(float))
// @tvm_ffi spfloat_densemv_gather_auto_f32
FFI_SPFLOAT_GATHER_AUTO(_f32,    float,  32 * sizeof(float))
// @tvm_ffi spfloat_densemv_scatter_f32
FFI_SPFLOAT_SCATTER(_f32,        float)
// @tvm_ffi spfloat_densemv_gather_auto_f64
FFI_SPFLOAT_GATHER_AUTO(_f64,    double, 32 * sizeof(double))
// @tvm_ffi spfloat_densemv_scatter_f64
FFI_SPFLOAT_SCATTER(_f64,        double)
// @tvm_ffi spfloat_densemv_gather_auto_f16
FFI_SPFLOAT_GATHER_AUTO(_f16,    __half, 32 * sizeof(float))
// @tvm_ffi spfloat_densemv_scatter_f16
FFI_SPFLOAT_SCATTER(_f16,        __half)
// @tvm_ffi spfloat_densemv_gather_auto_bf16
FFI_SPFLOAT_GATHER_AUTO(_bf16,   __nv_bfloat16, 32 * sizeof(float))
// @tvm_ffi spfloat_densemv_scatter_bf16
FFI_SPFLOAT_SCATTER(_bf16,       __nv_bfloat16)


// =========================================================================
// Dense Matrix-Matrix Multiplication (spfloat_densemm)
// =========================================================================

#define MM_BLOCK_SIZE 256

#define DEFINE_SPFLOAT_MM_NT_WPR(SUFFIX, WEIGHT_T, ACC_T, CHUNK_N,          \
                                  READ_W, WRITE_W, READ_S,                   \
                                  WARP_RED, ACC_ZERO)                        \
__global__ void _spfloat_mm_nt_wpr_kern##SUFFIX(                             \
    const WEIGHT_T* __restrict__ weights,                                    \
    const WEIGHT_T* __restrict__ spikes,                                     \
    WEIGHT_T*       __restrict__ output,                                     \
    int m, int k, int n                                                      \
) {                                                                          \
    int warp_id = threadIdx.x >> 5;                                         \
    int lane    = threadIdx.x & 31;                                         \
    int warps_per_block = blockDim.x >> 5;                                  \
    int row = blockIdx.x * warps_per_block + warp_id;                       \
    if (row >= m) return;                                                    \
    int col_start = blockIdx.y * CHUNK_N;                                   \
    int chunk_n = min(CHUNK_N, n - col_start);                              \
    const WEIGHT_T* w_row = weights + (size_t)row * k;                      \
    ACC_T acc[CHUNK_N];                                                      \
    _Pragma("unroll")                                                        \
    for (int j = 0; j < CHUNK_N; j++) acc[j] = ACC_ZERO;                   \
    for (int l = lane; l < k; l += 32) {                                    \
        ACC_T w_val = READ_W(w_row[l]);                                     \
        const WEIGHT_T* spk_l = spikes + (size_t)l * n + col_start;        \
        _Pragma("unroll")                                                    \
        for (int j = 0; j < CHUNK_N; j++)                                  \
            if (j < chunk_n)                                                \
                acc[j] += w_val * READ_S(spk_l[j]);                        \
    }                                                                        \
    WEIGHT_T* out_row = output + (size_t)row * n + col_start;              \
    _Pragma("unroll")                                                        \
    for (int j = 0; j < CHUNK_N; j++) {                                    \
        ACC_T val = WARP_RED(acc[j]);                                       \
        if (lane == 0 && j < chunk_n) out_row[j] = WRITE_W(val);           \
    }                                                                        \
}

#define DEFINE_SPFLOAT_MM_NT_TPE(SUFFIX, WEIGHT_T, ACC_T,                   \
                                  READ_W, WRITE_W, READ_S, ACC_ZERO)         \
__global__ void _spfloat_mm_nt_tpe_kern##SUFFIX(                             \
    const WEIGHT_T* __restrict__ weights,                                    \
    const WEIGHT_T* __restrict__ spikes,                                     \
    WEIGHT_T*       __restrict__ output,                                     \
    int m, int k, int n                                                      \
) {                                                                          \
    int warp_id = threadIdx.x >> 5;                                         \
    int lane    = threadIdx.x & 31;                                         \
    int warps_per_block = blockDim.x >> 5;                                  \
    int row = blockIdx.x * warps_per_block + warp_id;                       \
    int col = blockIdx.y * 32 + lane;                                       \
    if (row >= m || col >= n) return;                                        \
    const WEIGHT_T* w_row = weights + (size_t)row * k;                      \
    ACC_T acc = ACC_ZERO;                                                    \
    int l = 0;                                                               \
    for (; l <= k - 4; l += 4) {                                            \
        ACC_T sv0 = READ_S(spikes[(size_t)(l+0) * n + col]);               \
        ACC_T sv1 = READ_S(spikes[(size_t)(l+1) * n + col]);               \
        ACC_T sv2 = READ_S(spikes[(size_t)(l+2) * n + col]);               \
        ACC_T sv3 = READ_S(spikes[(size_t)(l+3) * n + col]);               \
        bool any_nz = (sv0 != ACC_ZERO) | (sv1 != ACC_ZERO) |             \
                      (sv2 != ACC_ZERO) | (sv3 != ACC_ZERO);              \
        if (__ballot_sync(__activemask(), any_nz) == 0u) continue;          \
        acc += READ_W(w_row[l+0]) * sv0;                                    \
        acc += READ_W(w_row[l+1]) * sv1;                                    \
        acc += READ_W(w_row[l+2]) * sv2;                                    \
        acc += READ_W(w_row[l+3]) * sv3;                                    \
    }                                                                        \
    for (; l < k; l++) {                                                    \
        ACC_T sv = READ_S(spikes[(size_t)l * n + col]);                     \
        if (__ballot_sync(__activemask(), sv != ACC_ZERO) == 0u) continue;  \
        acc += READ_W(w_row[l]) * sv;                                       \
    }                                                                        \
    output[(size_t)row * n + col] = WRITE_W(acc);                           \
}

#define DEFINE_SPFLOAT_MM_T(SUFFIX, WEIGHT_T, ACC_T, CHUNK_N,               \
                             READ_W, WRITE_W, READ_S,                        \
                             WARP_RED, ACC_ZERO)                             \
__global__ void _spfloat_mm_t_kern##SUFFIX(                                  \
    const WEIGHT_T* __restrict__ weights,                                    \
    const WEIGHT_T* __restrict__ spikes,                                     \
    WEIGHT_T*       __restrict__ output,                                     \
    int m, int k, int n                                                      \
) {                                                                          \
    int warp_id = threadIdx.x >> 5;                                         \
    int lane    = threadIdx.x & 31;                                         \
    int warps_per_block = blockDim.x >> 5;                                  \
    int row = blockIdx.x * warps_per_block + warp_id;                       \
    if (row >= m) return;                                                    \
    int col_start = blockIdx.y * CHUNK_N;                                   \
    int chunk_n = min(CHUNK_N, n - col_start);                              \
    const WEIGHT_T* s_row = spikes + (size_t)row * k;                       \
    ACC_T acc[CHUNK_N];                                                      \
    _Pragma("unroll")                                                        \
    for (int j = 0; j < CHUNK_N; j++) acc[j] = ACC_ZERO;                   \
    for (int l = lane; l < k; l += 32) {                                    \
        ACC_T spk_val = READ_S(s_row[l]);                                   \
        if (__ballot_sync(__activemask(), spk_val != ACC_ZERO) == 0u)       \
            continue;                                                        \
        if (spk_val != ACC_ZERO) {                                          \
            const WEIGHT_T* w_l = weights + (size_t)l * n + col_start;     \
            _Pragma("unroll")                                                \
            for (int j = 0; j < CHUNK_N; j++)                              \
                if (j < chunk_n)                                            \
                    acc[j] += spk_val * READ_W(w_l[j]);                    \
        }                                                                    \
    }                                                                        \
    WEIGHT_T* out_row = output + (size_t)row * n + col_start;              \
    _Pragma("unroll")                                                        \
    for (int j = 0; j < CHUNK_N; j++) {                                    \
        ACC_T val = WARP_RED(acc[j]);                                       \
        if (lane == 0 && j < chunk_n) out_row[j] = WRITE_W(val);           \
    }                                                                        \
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
#define FFI_SPFLOAT_MM_NT_WPR(SUFFIX, WEIGHT_C_T, CHUNK_N_VAL)              \
void spfloat_densemm_nt##SUFFIX(                                             \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView spikes,              \
    tvm::ffi::TensorView output, int64_t stream                              \
) {                                                                          \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                 \
    int m = static_cast<int>(weights.size(0));                               \
    int k = static_cast<int>(weights.size(1));                               \
    int n = static_cast<int>(spikes.size(1));                                \
    int warps_per_block = MM_BLOCK_SIZE / 32;                                \
    int m_blocks  = (m + warps_per_block - 1) / warps_per_block;             \
    int n_chunks  = (n + CHUNK_N_VAL - 1) / CHUNK_N_VAL;                    \
    dim3 grid(m_blocks, n_chunks);                                           \
    const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const WEIGHT_C_T* d_s = static_cast<const WEIGHT_C_T*>(spikes.data_ptr()); \
    WEIGHT_C_T*       d_o = static_cast<WEIGHT_C_T*>(output.data_ptr());     \
    _spfloat_mm_nt_wpr_kern##SUFFIX<<<grid, MM_BLOCK_SIZE, 0, s>>>(         \
        d_w, d_s, d_o, m, k, n);                                            \
}

#define FFI_SPFLOAT_MM_NT_TPE(SUFFIX, WEIGHT_C_T)                           \
void spfloat_densemm_nt_tpe##SUFFIX(                                         \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView spikes,              \
    tvm::ffi::TensorView output, int64_t stream                              \
) {                                                                          \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                 \
    int m = static_cast<int>(weights.size(0));                               \
    int k = static_cast<int>(weights.size(1));                               \
    int n = static_cast<int>(spikes.size(1));                                \
    int warps_per_block = MM_BLOCK_SIZE / 32;                                \
    int m_blocks = (m + warps_per_block - 1) / warps_per_block;             \
    int n_blocks = (n + 31) / 32;                                            \
    dim3 grid(m_blocks, n_blocks);                                           \
    const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const WEIGHT_C_T* d_s = static_cast<const WEIGHT_C_T*>(spikes.data_ptr()); \
    WEIGHT_C_T*       d_o = static_cast<WEIGHT_C_T*>(output.data_ptr());     \
    _spfloat_mm_nt_tpe_kern##SUFFIX<<<grid, MM_BLOCK_SIZE, 0, s>>>(         \
        d_w, d_s, d_o, m, k, n);                                            \
}

#define FFI_SPFLOAT_MM_T(SUFFIX, WEIGHT_C_T, CHUNK_N_VAL)                   \
void spfloat_densemm_t##SUFFIX(                                              \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView spikes,              \
    tvm::ffi::TensorView output, int64_t stream                              \
) {                                                                          \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                 \
    int k = static_cast<int>(weights.size(0));                               \
    int n = static_cast<int>(weights.size(1));                               \
    int m = static_cast<int>(spikes.size(0));                                \
    int warps_per_block = MM_BLOCK_SIZE / 32;                                \
    int m_blocks  = (m + warps_per_block - 1) / warps_per_block;             \
    int n_chunks  = (n + CHUNK_N_VAL - 1) / CHUNK_N_VAL;                    \
    dim3 grid(m_blocks, n_chunks);                                           \
    const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const WEIGHT_C_T* d_s = static_cast<const WEIGHT_C_T*>(spikes.data_ptr()); \
    WEIGHT_C_T*       d_o = static_cast<WEIGHT_C_T*>(output.data_ptr());     \
    _spfloat_mm_t_kern##SUFFIX<<<grid, MM_BLOCK_SIZE, 0, s>>>(              \
        d_w, d_s, d_o, m, k, n);                                            \
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
