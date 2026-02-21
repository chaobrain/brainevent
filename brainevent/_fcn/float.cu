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
 * float.cu -- Float-Weighted FCN Sparse Matrix-Vector and Matrix-Matrix CUDA Kernels
 * ==================================================================================
 *
 * This module provides optimized CUDA kernels for sparse operations with
 * fixed connection number (FCN) and floating-point weights. It includes:
 * 1. Sparse Matrix-Vector Product (SpMV): fcnmv
 * 2. Sparse Matrix-Matrix Product (SpMM): fcnmm
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
// Per-dtype conversion macros
// ============================================================================

#define READ_F32(x)   (x)
#define WRITE_F32(x)  (x)
#define READ_F64(x)   (x)
#define WRITE_F64(x)  (x)
#define READ_F16(x)   __half2float(x)
#define WRITE_F16(x)  __float2half(x)
#define READ_BF16(x)  __bfloat162float(x)
#define WRITE_BF16(x) __float2bfloat16(x)

// ============================================================================
// Per-dtype atomic-add helpers (ACC_T value -> WEIGHT_T memory)
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
// FCN Matrix-Vector Multiplication (fcnmv)
// ============================================================================

#define DEFINE_GATHER_WARP(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _gather_warp_kern##SUFFIX( \
    const int32_t* __restrict__ indices, \
    const WEIGHT_T* __restrict__ vector, \
    WEIGHT_T* __restrict__ output, \
    const WEIGHT_T* __restrict__ weights, \
    int n_pre, int n_conn, int is_homo \
) { \
    int row = blockIdx.x; \
    if (row >= n_pre) return; \
    int lane = threadIdx.x; \
    const int32_t* i_row = indices + (size_t)row * n_conn; \
    const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)row * n_conn; \
    ACC_T val = ACC_ZERO; \
    for (int k = threadIdx.x; k < n_conn; k += 32) \
        val += is_homo ? READ_W(vector[i_row[k]]) \
                       : (READ_W(w_row[k]) * READ_W(vector[i_row[k]])); \
    val = WARP_RED(val); \
    if (threadIdx.x == 0) \
        output[row] = WRITE_W(is_homo ? (READ_W(weights[0]) * val) : val); \
}

#define DEFINE_GATHER_BASIC(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _gather_basic_kern##SUFFIX( \
    const int32_t* __restrict__ indices, \
    const WEIGHT_T* __restrict__ vector, \
    WEIGHT_T* __restrict__ output, \
    const WEIGHT_T* __restrict__ weights, \
    int n_pre, int n_conn, int is_homo \
) { \
    extern __shared__ char _smem_bytes[]; \
    ACC_T* smem_red = reinterpret_cast<ACC_T*>(_smem_bytes); \
    int row = blockIdx.x; \
    if (row >= n_pre) return; \
    const int32_t* i_row = indices + (size_t)row * n_conn; \
    const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)row * n_conn; \
    int lane   = threadIdx.x & 31; \
    int warpid = threadIdx.x >> 5; \
    int nwarps = blockDim.x >> 5; \
    ACC_T val = ACC_ZERO; \
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x) \
        val += is_homo ? READ_W(vector[i_row[k]]) \
                       : (READ_W(w_row[k]) * READ_W(vector[i_row[k]])); \
    val = WARP_RED(val); \
    if (lane == 0) smem_red[warpid] = val; \
    __syncthreads(); \
    int n_warps_in_block = blockDim.x >> 5; \
    val = (threadIdx.x < n_warps_in_block) ? smem_red[lane] : ACC_ZERO; \
    if (warpid == 0) val = WARP_RED(val); \
    if (threadIdx.x == 0) \
        output[row] = WRITE_W(is_homo ? (READ_W(weights[0]) * val) : val); \
}

#define DEFINE_SCATTER_BASIC(SUFFIX, WEIGHT_T, READ_W, ATOMIC_ADD_W) \
__global__ void _scatter_basic_kern##SUFFIX( \
    const int32_t* __restrict__ indices, \
    const WEIGHT_T* __restrict__ vector, \
    WEIGHT_T*       __restrict__ output, \
    const WEIGHT_T* __restrict__ weights, \
    int n_pre, int n_conn, int is_homo \
) { \
    int row = blockIdx.x; \
    if (row >= n_pre) return; \
    float v = READ_W(vector[row]); \
    const int32_t* i_row = indices + (size_t)row * n_conn; \
    const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)row * n_conn; \
    float w0 = is_homo ? READ_W(weights[0]) : 0.0f; \
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x) \
        ATOMIC_ADD_W(&output[i_row[k]], (is_homo ? w0 : READ_W(w_row[k])) * v); \
}

#define DEFINE_SCATTER_WARP(SUFFIX, WEIGHT_T, READ_W, ATOMIC_ADD_W) \
__global__ void _scatter_warp_kern##SUFFIX( \
    const int32_t* __restrict__ indices, \
    const WEIGHT_T* __restrict__ vector, \
    WEIGHT_T*       __restrict__ output, \
    const WEIGHT_T* __restrict__ weights, \
    int n_pre, int n_conn, int is_homo \
) { \
    int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; \
    int lane_id   = threadIdx.x & 31; \
    int num_warps = (gridDim.x * blockDim.x) >> 5; \
    for (int row = warp_id; row < n_pre; row += num_warps) { \
        float v = READ_W(vector[row]); \
        const int32_t* i_row = indices + (size_t)row * n_conn; \
        const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)row * n_conn; \
        float w0 = is_homo ? READ_W(weights[0]) : 0.0f; \
        for (int k = lane_id; k < n_conn; k += 32) \
            ATOMIC_ADD_W(&output[i_row[k]], (is_homo ? w0 : READ_W(w_row[k])) * v); \
    } \
}

#define DEFINE_SCATTER_GS(SUFFIX, WEIGHT_T, READ_W, ATOMIC_ADD_W) \
__global__ void _scatter_gs_kern##SUFFIX( \
    const int32_t* __restrict__ indices, \
    const WEIGHT_T* __restrict__ vector, \
    WEIGHT_T*       __restrict__ output, \
    const WEIGHT_T* __restrict__ weights, \
    int n_pre, int n_conn, int is_homo \
) { \
    int total  = n_pre * n_conn; \
    int tid    = blockIdx.x * blockDim.x + threadIdx.x; \
    int stride = blockDim.x * gridDim.x; \
    for (int idx = tid; idx < total; idx += stride) { \
        int row = idx / n_conn; \
        float w = is_homo ? READ_W(weights[0]) : READ_W(weights[idx]); \
        ATOMIC_ADD_W(&output[indices[idx]], w * READ_W(vector[row])); \
    } \
}

// Instantiations
DEFINE_GATHER_WARP(_f32,  float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_GATHER_BASIC(_f32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_SCATTER_BASIC(_f32, float,          READ_F32,  atomic_add_f32)
DEFINE_SCATTER_WARP(_f32,  float,          READ_F32,  atomic_add_f32)
DEFINE_SCATTER_GS(_f32,    float,          READ_F32,  atomic_add_f32)
DEFINE_GATHER_WARP(_f64,  double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_GATHER_BASIC(_f64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_SCATTER_BASIC(_f64, double,         READ_F64,  atomic_add_f64)
DEFINE_SCATTER_WARP(_f64,  double,         READ_F64,  atomic_add_f64)
DEFINE_SCATTER_GS(_f64,    double,         READ_F64,  atomic_add_f64)
DEFINE_GATHER_WARP(_f16,  __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_GATHER_BASIC(_f16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_SCATTER_BASIC(_f16, __half,         READ_F16,  atomic_add_f16)
DEFINE_SCATTER_WARP(_f16,  __half,         READ_F16,  atomic_add_f16)
DEFINE_SCATTER_GS(_f16,    __half,         READ_F16,  atomic_add_f16)
DEFINE_GATHER_WARP(_bf16,  __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_GATHER_BASIC(_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_SCATTER_BASIC(_bf16, __nv_bfloat16, READ_BF16, atomic_add_bf16)
DEFINE_SCATTER_WARP(_bf16,  __nv_bfloat16, READ_BF16, atomic_add_bf16)
DEFINE_SCATTER_GS(_bf16,    __nv_bfloat16, READ_BF16, atomic_add_bf16)

// SpMV Specializations
__global__ void _gather_shared_kern(const int32_t* __restrict__ indices, const float* __restrict__ vector, float* __restrict__ output, const float* __restrict__ weights, int n_pre, int n_conn, int is_homo) {
    extern __shared__ char smem_raw[];
    int32_t* s_idx = reinterpret_cast<int32_t*>(smem_raw);
    float*   s_wt  = reinterpret_cast<float*>(smem_raw + blockDim.x * sizeof(int32_t));
    float*   s_red = reinterpret_cast<float*>(smem_raw + blockDim.x * (sizeof(int32_t) + sizeof(float)));
    int row = blockIdx.x; if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float val = 0.0f;
    for (int base = 0; base < n_conn; base += blockDim.x) {
        int k = base + threadIdx.x;
        if (k < n_conn) { s_idx[threadIdx.x] = i_row[k]; s_wt[threadIdx.x] = is_homo ? 1.0f : w_row[k]; }
        __syncthreads();
        int tile = min((int)blockDim.x, n_conn - base);
        if (threadIdx.x < tile) val += s_wt[threadIdx.x] * vector[s_idx[threadIdx.x]];
        __syncthreads();
    }
    int lane = threadIdx.x & 31, warpid = threadIdx.x >> 5; val = warp_reduce_sum_f32(val);
    if (lane == 0) s_red[warpid] = val; __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5; val = (threadIdx.x < n_warps) ? s_red[lane] : 0.0f;
    if (warpid == 0) val = warp_reduce_sum_f32(val);
    if (threadIdx.x == 0) output[row] = is_homo ? (weights[0] * val) : val;
}
__global__ void _gather_vec4_kern(const int32_t* __restrict__ indices, const float* __restrict__ vector, float* __restrict__ output, const float* __restrict__ weights, int n_pre, int n_conn, int is_homo) {
    extern __shared__ float smem_red[];
    int row = blockIdx.x; if (row >= n_pre) return;
    size_t base = (size_t)row * n_conn;
    const int4* i4 = (const int4*)(indices + base); const float4* w4 = is_homo ? nullptr : (const float4*)(weights + base);
    int n4 = n_conn >> 2; float val = 0.0f;
    for (int k = threadIdx.x; k < n4; k += blockDim.x) {
        int4 idx = i4[k]; if (!is_homo) { float4 ww = w4[k]; val += ww.x * vector[idx.x] + ww.y * vector[idx.y] + ww.z * vector[idx.z] + ww.w * vector[idx.w]; }
        else { val += vector[idx.x] + vector[idx.y] + vector[idx.z] + vector[idx.w]; }
    }
    for (int k = (n4 << 2) + threadIdx.x; k < n_conn; k += blockDim.x) { float v = vector[indices[base + k]]; val += is_homo ? v : (weights[base + k] * v); }
    int lane = threadIdx.x & 31, warpid = threadIdx.x >> 5; val = warp_reduce_sum_f32(val);
    if (lane == 0) smem_red[warpid] = val; __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5; val = (threadIdx.x < n_warps) ? smem_red[lane] : 0.0f;
    if (warpid == 0) val = warp_reduce_sum_f32(val);
    if (threadIdx.x == 0) output[row] = is_homo ? (weights[0] * val) : val;
}

// SpMV FFI Entries
#define FFI_GATHER_AUTO(SUFFIX, WEIGHT_C_T, SHM_SIZE)                                      \
void fcnmv_gather_auto##SUFFIX(                                                             \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                             \
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream               \
) {                                                                                         \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                               \
    int n_pre       = static_cast<int>(indices.size(0));                                    \
    int n_conn      = static_cast<int>(indices.size(1));                                    \
    int is_homo     = (weights.ndim() == 1) ? 1 : 0;                                       \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());          \
    const WEIGHT_C_T* d_vec = static_cast<const WEIGHT_C_T*>(vector.data_ptr());           \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());             \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                 \
    if (n_conn <= 32)                                                                       \
        _gather_warp_kern##SUFFIX<<<n_pre, 32, 0, s>>>(                                    \
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);                             \
    else                                                                                    \
        _gather_basic_kern##SUFFIX<<<n_pre, 256, SHM_SIZE, s>>>(                           \
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);                             \
}

#define FFI_SCATTER_AUTO(SUFFIX, WEIGHT_C_T)                                                \
void fcnmv_scatter_auto##SUFFIX(                                                            \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                             \
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream               \
) {                                                                                         \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                               \
    int n_pre       = static_cast<int>(indices.size(0));                                    \
    int n_conn      = static_cast<int>(indices.size(1));                                    \
    int n_post      = static_cast<int>(output.size(0));                                     \
    int is_homo     = (weights.ndim() == 1) ? 1 : 0;                                       \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());          \
    const WEIGHT_C_T* d_vec = static_cast<const WEIGHT_C_T*>(vector.data_ptr());           \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());             \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                 \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(WEIGHT_C_T), s);                     \
    if (n_conn <= 32) {                                                                     \
        int blocks = (n_pre + 7) / 8;                                                       \
        _scatter_warp_kern##SUFFIX<<<blocks, 256, 0, s>>>(                                  \
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);                             \
    } else if ((long long)n_pre * n_conn > 262144LL) {                                      \
        int blocks = min(1024, (int)((n_pre * n_conn + 255) / 256));                        \
        _scatter_gs_kern##SUFFIX<<<blocks, 256, 0, s>>>(                                    \
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);                             \
    } else {                                                                                \
        _scatter_basic_kern##SUFFIX<<<n_pre, 256, 0, s>>>(                                  \
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);                             \
    }                                                                                       \
}

// @tvm_ffi fcnmv_gather_warp_f32
void fcnmv_gather_warp_f32(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    _gather_warp_kern_f32<<<n_pre, 32, 0, s>>>(
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>(vector.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        n_pre, n_conn, is_homo);
}
// @tvm_ffi fcnmv_gather_basic_f32
void fcnmv_gather_basic_f32(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    _gather_basic_kern_f32<<<n_pre, 256, 32 * sizeof(float), s>>>(
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>(vector.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        n_pre, n_conn, is_homo);
}
// @tvm_ffi fcnmv_scatter_warp_f32
void fcnmv_scatter_warp_f32(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    cudaMemsetAsync(output.data_ptr(), 0, (size_t)n_post * sizeof(float), s);
    int blocks = (n_pre + 7) / 8;
    _scatter_warp_kern_f32<<<blocks, 256, 0, s>>>(
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>(vector.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        n_pre, n_conn, is_homo);
}
// @tvm_ffi fcnmv_scatter_auto_f32
FFI_SCATTER_AUTO(_f32, float)
// @tvm_ffi fcnmv_gather_warp_f64
void fcnmv_gather_warp_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    _gather_warp_kern_f64<<<n_pre, 32, 0, s>>>(
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const double*>(vector.data_ptr()),
        static_cast<double*>(output.data_ptr()),
        static_cast<const double*>(weights.data_ptr()),
        n_pre, n_conn, is_homo);
}
// @tvm_ffi fcnmv_gather_auto_f64
FFI_GATHER_AUTO(_f64, double, 32 * sizeof(double))
// @tvm_ffi fcnmv_scatter_warp_f64
void fcnmv_scatter_warp_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    cudaMemsetAsync(output.data_ptr(), 0, (size_t)n_post * sizeof(double), s);
    int blocks = (n_pre + 7) / 8;
    _scatter_warp_kern_f64<<<blocks, 256, 0, s>>>(
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const double*>(vector.data_ptr()),
        static_cast<double*>(output.data_ptr()),
        static_cast<const double*>(weights.data_ptr()),
        n_pre, n_conn, is_homo);
}
// @tvm_ffi fcnmv_scatter_auto_f64
FFI_SCATTER_AUTO(_f64, double)
// @tvm_ffi fcnmv_gather_warp_f16
void fcnmv_gather_warp_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    _gather_warp_kern_f16<<<n_pre, 32, 0, s>>>(
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const __half*>(vector.data_ptr()),
        static_cast<__half*>(output.data_ptr()),
        static_cast<const __half*>(weights.data_ptr()),
        n_pre, n_conn, is_homo);
}
// @tvm_ffi fcnmv_gather_auto_f16
FFI_GATHER_AUTO(_f16, __half, 32 * sizeof(float))
// @tvm_ffi fcnmv_scatter_warp_f16
void fcnmv_scatter_warp_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    cudaMemsetAsync(output.data_ptr(), 0, (size_t)n_post * sizeof(__half), s);
    int blocks = (n_pre + 7) / 8;
    _scatter_warp_kern_f16<<<blocks, 256, 0, s>>>(
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const __half*>(vector.data_ptr()),
        static_cast<__half*>(output.data_ptr()),
        static_cast<const __half*>(weights.data_ptr()),
        n_pre, n_conn, is_homo);
}
// @tvm_ffi fcnmv_scatter_auto_f16
FFI_SCATTER_AUTO(_f16, __half)
// @tvm_ffi fcnmv_gather_warp_bf16
void fcnmv_gather_warp_bf16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    _gather_warp_kern_bf16<<<n_pre, 32, 0, s>>>(
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const __nv_bfloat16*>(vector.data_ptr()),
        static_cast<__nv_bfloat16*>(output.data_ptr()),
        static_cast<const __nv_bfloat16*>(weights.data_ptr()),
        n_pre, n_conn, is_homo);
}
// @tvm_ffi fcnmv_gather_auto_bf16
FFI_GATHER_AUTO(_bf16, __nv_bfloat16, 32 * sizeof(float))
// @tvm_ffi fcnmv_scatter_warp_bf16
void fcnmv_scatter_warp_bf16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    cudaMemsetAsync(output.data_ptr(), 0, (size_t)n_post * sizeof(__nv_bfloat16), s);
    int blocks = (n_pre + 7) / 8;
    _scatter_warp_kern_bf16<<<blocks, 256, 0, s>>>(
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const __nv_bfloat16*>(vector.data_ptr()),
        static_cast<__nv_bfloat16*>(output.data_ptr()),
        static_cast<const __nv_bfloat16*>(weights.data_ptr()),
        n_pre, n_conn, is_homo);
}
// @tvm_ffi fcnmv_scatter_auto_bf16
FFI_SCATTER_AUTO(_bf16, __nv_bfloat16)
void fcnmv_gather_vec4_f32(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    _gather_vec4_kern<<<n_pre, 256, 32 * sizeof(float), s>>>(
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>(vector.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        n_pre, n_conn, is_homo);
}
void fcnmv_gather_auto_f32(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    if (n_conn <= 32)
        _gather_warp_kern_f32<<<n_pre, 32, 0, s>>>(
            static_cast<const int32_t*>(indices.data_ptr()),
            static_cast<const float*>(vector.data_ptr()),
            static_cast<float*>(output.data_ptr()),
            static_cast<const float*>(weights.data_ptr()),
            n_pre, n_conn, is_homo);
    else if (n_conn % 4 == 0 && n_conn >= 128)
        _gather_vec4_kern<<<n_pre, 256, 32 * sizeof(float), s>>>(
            static_cast<const int32_t*>(indices.data_ptr()),
            static_cast<const float*>(vector.data_ptr()),
            static_cast<float*>(output.data_ptr()),
            static_cast<const float*>(weights.data_ptr()),
            n_pre, n_conn, is_homo);
    else if (n_conn > 512) {
        int threads = 256;
        size_t shm = (size_t)threads * (sizeof(int32_t) + sizeof(float)) + 32 * sizeof(float);
        _gather_shared_kern<<<n_pre, threads, shm, s>>>(
            static_cast<const int32_t*>(indices.data_ptr()),
            static_cast<const float*>(vector.data_ptr()),
            static_cast<float*>(output.data_ptr()),
            static_cast<const float*>(weights.data_ptr()),
            n_pre, n_conn, is_homo);
    } else
        _gather_basic_kern_f32<<<n_pre, 256, 32 * sizeof(float), s>>>(
            static_cast<const int32_t*>(indices.data_ptr()),
            static_cast<const float*>(vector.data_ptr()),
            static_cast<float*>(output.data_ptr()),
            static_cast<const float*>(weights.data_ptr()),
            n_pre, n_conn, is_homo);
}

// ============================================================================
// FCN Matrix-Matrix Multiplication (fcnmm)
// ============================================================================

#define DEFINE_MM_GATHER_BASIC(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W) \
__global__ void _mm_gather_basic_kern##SUFFIX( \
    const int32_t* __restrict__ indices, \
    const WEIGHT_T* __restrict__ matrix, \
    WEIGHT_T*       __restrict__ output, \
    const WEIGHT_T* __restrict__ weights, \
    int n_pre, int n_conn, int n_col, int is_homo \
) { \
    int i = blockIdx.x; \
    int j = blockIdx.y * blockDim.x + threadIdx.x; \
    if (i >= n_pre || j >= n_col) return; \
    const int32_t*  idx_row = indices + (size_t)i * n_conn; \
    const WEIGHT_T* w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn; \
    ACC_T acc = (ACC_T)0; \
    for (int k = 0; k < n_conn; k++) { \
        ACC_T w = is_homo ? (ACC_T)1 : READ_W(w_row[k]); \
        acc += w * READ_W(matrix[(size_t)idx_row[k] * n_col + j]); \
    } \
    output[(size_t)i * n_col + j] = WRITE_W(is_homo ? (READ_W(weights[0]) * acc) : acc); \
}

#define DEFINE_MM_SCATTER_BLOCK(SUFFIX, WEIGHT_T, READ_W, ATOMIC_ADD_W) \
__global__ void _mm_scatter_block_kern##SUFFIX( \
    const int32_t* __restrict__ indices, \
    const WEIGHT_T* __restrict__ matrix, \
    WEIGHT_T*       __restrict__ output, \
    const WEIGHT_T* __restrict__ weights, \
    int n_pre, int n_conn, int n_col, int is_homo \
) { \
    int i = blockIdx.x; \
    if (i >= n_pre) return; \
    const int32_t*  idx_row = indices + (size_t)i * n_conn; \
    const WEIGHT_T* w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn; \
    const WEIGHT_T* m_row   = matrix + (size_t)i * n_col; \
    for (int k = 0; k < n_conn; k++) { \
        int tgt = idx_row[k]; \
        float w = is_homo ? READ_W(weights[0]) : READ_W(w_row[k]); \
        WEIGHT_T* out_row = output + (size_t)tgt * n_col; \
        for (int j = threadIdx.x; j < n_col; j += blockDim.x) \
            ATOMIC_ADD_W(&out_row[j], w * READ_W(m_row[j])); \
    } \
}

#define DEFINE_MM_SCATTER_WARP(SUFFIX, WEIGHT_T, READ_W, ATOMIC_ADD_W) \
__global__ void _mm_scatter_warp_kern##SUFFIX( \
    const int32_t* __restrict__ indices, \
    const WEIGHT_T* __restrict__ matrix, \
    WEIGHT_T*       __restrict__ output, \
    const WEIGHT_T* __restrict__ weights, \
    int n_pre, int n_conn, int n_col, int is_homo \
) { \
    int wid     = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; \
    int lane    = threadIdx.x & 31; \
    int n_warps = (gridDim.x * blockDim.x) >> 5; \
    int n_pairs = n_pre * n_conn; \
    for (int pair = wid; pair < n_pairs; pair += n_warps) { \
        int i = pair / n_conn; \
        int k = pair % n_conn; \
        int   tgt = indices[(size_t)i * n_conn + k]; \
        float w   = is_homo ? READ_W(weights[0]) \
                            : READ_W(weights[(size_t)i * n_conn + k]); \
        const WEIGHT_T* m_row   = matrix + (size_t)i * n_col; \
        WEIGHT_T*       out_row = output + (size_t)tgt * n_col; \
        for (int j = lane; j < n_col; j += 32) \
            ATOMIC_ADD_W(&out_row[j], w * READ_W(m_row[j])); \
    } \
}

// SpMM Instantiations
DEFINE_MM_GATHER_BASIC(_f32,  float,          float,  READ_F32,  WRITE_F32)
DEFINE_MM_GATHER_BASIC(_f64,  double,         double, READ_F64,  WRITE_F64)
DEFINE_MM_GATHER_BASIC(_f16,  __half,         float,  READ_F16,  WRITE_F16)
DEFINE_MM_GATHER_BASIC(_bf16, __nv_bfloat16,  float,  READ_BF16, WRITE_BF16)
DEFINE_MM_SCATTER_BLOCK(_f32,  float,          READ_F32,  atomic_add_f32)
DEFINE_MM_SCATTER_BLOCK(_f64,  double,         READ_F64,  atomic_add_f64)
DEFINE_MM_SCATTER_BLOCK(_f16,  __half,         READ_F16,  atomic_add_f16)
DEFINE_MM_SCATTER_BLOCK(_bf16, __nv_bfloat16,  READ_BF16, atomic_add_bf16)
DEFINE_MM_SCATTER_WARP(_f32,  float,          READ_F32,  atomic_add_f32)
DEFINE_MM_SCATTER_WARP(_f64,  double,         READ_F64,  atomic_add_f64)
DEFINE_MM_SCATTER_WARP(_f16,  __half,         READ_F16,  atomic_add_f16)
DEFINE_MM_SCATTER_WARP(_bf16, __nv_bfloat16,  READ_BF16, atomic_add_bf16)

// SpMM Specializations
#define MMTK 128
__global__ void _mm_gather_shared_kern(const int32_t* __restrict__ indices, const float* __restrict__ matrix, float* __restrict__ output, const float* __restrict__ weights, int n_pre, int n_conn, int n_col, int is_homo) {
    extern __shared__ char smem_mm[];
    int32_t* s_idx = reinterpret_cast<int32_t*>(smem_mm);
    float*   s_w   = reinterpret_cast<float*>(smem_mm + MMTK * sizeof(int32_t));
    int i = blockIdx.x, j = blockIdx.y * blockDim.x + threadIdx.x; if (i >= n_pre) return;
    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const float*   w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;
    float acc = 0.0f;
    for (int k0 = 0; k0 < n_conn; k0 += MMTK) {
        int tile = (k0 + MMTK < n_conn) ? MMTK : (n_conn - k0);
        for (int t = threadIdx.x; t < tile; t += blockDim.x) { s_idx[t] = idx_row[k0 + t]; s_w[t] = is_homo ? 1.0f : w_row[k0 + t]; }
        __syncthreads();
        if (j < n_col) for (int t = 0; t < tile; t++) acc += s_w[t] * matrix[(size_t)s_idx[t] * n_col + j];
        __syncthreads();
    }
    if (j < n_col) output[(size_t)i * n_col + j] = is_homo ? (weights[0] * acc) : acc;
}
__global__ void _mm_gather_vec4_kern(const int32_t* __restrict__ indices, const float* __restrict__ matrix, float* __restrict__ output, const float* __restrict__ weights, int n_pre, int n_conn, int n_col, int is_homo) {
    int i = blockIdx.x, j4 = blockIdx.y * blockDim.x + threadIdx.x, nc4 = n_col >> 2; if (i >= n_pre || j4 >= nc4) return;
    const int32_t* idx_row = indices + (size_t)i * n_conn; const float* w_row = is_homo ? nullptr : weights + (size_t)i * n_conn;
    const float4* mat4 = (const float4*)matrix; float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = 0; k < n_conn; k++) { float w = is_homo ? weights[0] : w_row[k]; float4 m = mat4[(size_t)idx_row[k] * nc4 + j4]; acc.x += w * m.x; acc.y += w * m.y; acc.z += w * m.z; acc.w += w * m.w; }
    ((float4*)output)[(size_t)i * nc4 + j4] = acc;
}
#define MM_SCATTER_BJ 128
__global__ void _mm_scatter_cached_kern(const int32_t* __restrict__ indices, const float* __restrict__ matrix, float* __restrict__ output, const float* __restrict__ weights, int n_pre, int n_conn, int n_col, int is_homo) {
    extern __shared__ float s_m[];
    int i = blockIdx.x, j = blockIdx.y * blockDim.x + threadIdx.x; if (i >= n_pre) return;
    s_m[threadIdx.x] = (j < n_col) ? matrix[(size_t)i * n_col + j] : 0.0f; __syncthreads();
    const int32_t* idx_row = indices + (size_t)i * n_conn; const float* w_row = is_homo ? nullptr : weights + (size_t)i * n_conn;
    if (j < n_col) { float m_val = s_m[threadIdx.x]; for (int k = 0; k < n_conn; k++) { int tgt = idx_row[k]; float w = is_homo ? weights[0] : w_row[k]; atomic_add_f32(&output[(size_t)tgt * n_col + j], w * m_val); } }
}

// SpMM FFI Entries
#define FFI_MM_GATHER_AUTO(SUFFIX, WEIGHT_C_T)                                              \
void fcnmm_gather_auto##SUFFIX(                                                             \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                             \
    tvm::ffi::TensorView matrix,  tvm::ffi::TensorView output, int64_t stream               \
) {                                                                                         \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                               \
    int n_pre       = static_cast<int>(indices.size(0));                                    \
    int n_conn      = static_cast<int>(indices.size(1));                                    \
    int n_col       = static_cast<int>(matrix.size(1));                                     \
    int is_homo     = (weights.ndim() == 1) ? 1 : 0;                                       \
    _mm_gather_basic_kern##SUFFIX<<<dim3(n_pre, (n_col + 63) / 64), 64, 0, s>>>(           \
        static_cast<const int32_t*>(indices.data_ptr()),                                    \
        static_cast<const WEIGHT_C_T*>(matrix.data_ptr()),                                  \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                                        \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                                 \
        n_pre, n_conn, n_col, is_homo);                                                     \
}

#define FFI_MM_SCATTER_AUTO(SUFFIX, WEIGHT_C_T)                                             \
void fcnmm_scatter_auto##SUFFIX(                                                            \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                             \
    tvm::ffi::TensorView matrix,  tvm::ffi::TensorView output, int64_t stream               \
) {                                                                                         \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                               \
    int n_pre       = static_cast<int>(indices.size(0));                                    \
    int n_conn      = static_cast<int>(indices.size(1));                                    \
    int n_post      = static_cast<int>(output.size(0));                                     \
    int n_col       = static_cast<int>(matrix.size(1));                                     \
    int is_homo     = (weights.ndim() == 1) ? 1 : 0;                                       \
    cudaMemsetAsync(output.data_ptr(), 0, (size_t)n_post * n_col * sizeof(WEIGHT_C_T), s); \
    if (n_col <= 64) {                                                                      \
        int n_pairs = n_pre * n_conn;                                                       \
        int blocks  = min(4096, (n_pairs + 7) / 8);                                        \
        _mm_scatter_warp_kern##SUFFIX<<<blocks, 256, 0, s>>>(                               \
            static_cast<const int32_t*>(indices.data_ptr()),                                \
            static_cast<const WEIGHT_C_T*>(matrix.data_ptr()),                              \
            static_cast<WEIGHT_C_T*>(output.data_ptr()),                                    \
            static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                             \
            n_pre, n_conn, n_col, is_homo);                                                 \
    } else {                                                                                \
        _mm_scatter_block_kern##SUFFIX<<<n_pre, 256, 0, s>>>(                               \
            static_cast<const int32_t*>(indices.data_ptr()),                                \
            static_cast<const WEIGHT_C_T*>(matrix.data_ptr()),                              \
            static_cast<WEIGHT_C_T*>(output.data_ptr()),                                    \
            static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                             \
            n_pre, n_conn, n_col, is_homo);                                                 \
    }                                                                                       \
}

// @tvm_ffi fcnmm_gather_auto_f64
FFI_MM_GATHER_AUTO(_f64, double)
// @tvm_ffi fcnmm_scatter_auto_f64
FFI_MM_SCATTER_AUTO(_f64, double)
// @tvm_ffi fcnmm_gather_auto_f16
FFI_MM_GATHER_AUTO(_f16, __half)
// @tvm_ffi fcnmm_scatter_auto_f16
FFI_MM_SCATTER_AUTO(_f16, __half)
// @tvm_ffi fcnmm_gather_auto_bf16
FFI_MM_GATHER_AUTO(_bf16, __nv_bfloat16)
// @tvm_ffi fcnmm_scatter_auto_bf16
FFI_MM_SCATTER_AUTO(_bf16, __nv_bfloat16)
void fcnmm_gather_auto_f32(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    if (n_col % 4 == 0 && n_col >= 64) {
        dim3 grid(n_pre, (n_col / 4 + 63) / 64);
        _mm_gather_vec4_kern<<<grid, 64, 0, s>>>(
            static_cast<const int32_t*>(indices.data_ptr()),
            static_cast<const float*>(matrix.data_ptr()),
            static_cast<float*>(output.data_ptr()),
            static_cast<const float*>(weights.data_ptr()),
            n_pre, n_conn, n_col, is_homo);
    } else if (n_conn > 128) {
        dim3 grid(n_pre, (n_col + 63) / 64);
        size_t shm = MMTK * (sizeof(int32_t) + sizeof(float));
        _mm_gather_shared_kern<<<grid, 64, shm, s>>>(
            static_cast<const int32_t*>(indices.data_ptr()),
            static_cast<const float*>(matrix.data_ptr()),
            static_cast<float*>(output.data_ptr()),
            static_cast<const float*>(weights.data_ptr()),
            n_pre, n_conn, n_col, is_homo);
    } else {
        dim3 grid(n_pre, (n_col + 63) / 64);
        _mm_gather_basic_kern_f32<<<grid, 64, 0, s>>>(
            static_cast<const int32_t*>(indices.data_ptr()),
            static_cast<const float*>(matrix.data_ptr()),
            static_cast<float*>(output.data_ptr()),
            static_cast<const float*>(weights.data_ptr()),
            n_pre, n_conn, n_col, is_homo);
    }
}
void fcnmm_scatter_auto_f32(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    cudaMemsetAsync(output.data_ptr(), 0, (size_t)n_post * n_col * sizeof(float), s);
    if (n_col <= 64) {
        int n_pairs = n_pre * n_conn;
        int blocks  = min(4096, (n_pairs + 7) / 8);
        _mm_scatter_warp_kern_f32<<<blocks, 256, 0, s>>>(
            static_cast<const int32_t*>(indices.data_ptr()),
            static_cast<const float*>(matrix.data_ptr()),
            static_cast<float*>(output.data_ptr()),
            static_cast<const float*>(weights.data_ptr()),
            n_pre, n_conn, n_col, is_homo);
    } else if (n_conn > 32) {
        int BJ = 128;
        dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
        size_t shm = BJ * sizeof(float);
        _mm_scatter_cached_kern<<<grid, BJ, shm, s>>>(
            static_cast<const int32_t*>(indices.data_ptr()),
            static_cast<const float*>(matrix.data_ptr()),
            static_cast<float*>(output.data_ptr()),
            static_cast<const float*>(weights.data_ptr()),
            n_pre, n_conn, n_col, is_homo);
    } else {
        _mm_scatter_block_kern_f32<<<n_pre, 256, 0, s>>>(
            static_cast<const int32_t*>(indices.data_ptr()),
            static_cast<const float*>(matrix.data_ptr()),
            static_cast<float*>(output.data_ptr()),
            static_cast<const float*>(weights.data_ptr()),
            n_pre, n_conn, n_col, is_homo);
    }
}
