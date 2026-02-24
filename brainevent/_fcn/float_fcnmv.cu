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
 * float_fcnmv.cu -- Float-Weighted FCN Sparse Matrix-Vector CUDA Kernels
 * ========================================================================
 *
 * This module provides optimized CUDA kernels for sparse operations with
 * fixed connection number (FCN) and floating-point weights:
 * 1. Sparse Matrix-Vector Product (SpMV): fcnmv
 */
#include "../cuda_common.h"

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
    const int32_t* i_row = indices + (size_t)row * n_conn; \
    const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)row * n_conn; \
    ACC_T val = ACC_ZERO; \
    for (int k = threadIdx.x; k < n_conn; k += 32) { \
        int32_t idx = __ldg(&i_row[k]); \
        ACC_T v = READ_W(__ldg(&vector[idx])); \
        if (is_homo) val += v; \
        else val += READ_W(__ldg(&w_row[k])) * v; \
    } \
    val = WARP_RED(val); \
    if (threadIdx.x == 0) \
        output[row] = WRITE_W(is_homo ? (READ_W(__ldg(&weights[0])) * val) : val); \
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
    ACC_T val = ACC_ZERO; \
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x) { \
        int32_t idx = __ldg(&i_row[k]); \
        ACC_T v = READ_W(__ldg(&vector[idx])); \
        if (is_homo) val += v; \
        else val += READ_W(__ldg(&w_row[k])) * v; \
    } \
    val = WARP_RED(val); \
    if (lane == 0) smem_red[warpid] = val; \
    __syncthreads(); \
    int n_warps_in_block = blockDim.x >> 5; \
    val = (threadIdx.x < n_warps_in_block) ? smem_red[lane] : ACC_ZERO; \
    if (warpid == 0) val = WARP_RED(val); \
    if (threadIdx.x == 0) \
        output[row] = WRITE_W(is_homo ? (READ_W(__ldg(&weights[0])) * val) : val); \
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
    float v = READ_W(__ldg(&vector[row])); \
    const int32_t* i_row = indices + (size_t)row * n_conn; \
    const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)row * n_conn; \
    float w0 = is_homo ? READ_W(__ldg(&weights[0])) : 0.0f; \
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x) { \
        int32_t idx = __ldg(&i_row[k]); \
        float wk = is_homo ? w0 : READ_W(__ldg(&w_row[k])); \
        ATOMIC_ADD_W(&output[idx], wk * v); \
    } \
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
        float v = READ_W(__ldg(&vector[row])); \
        const int32_t* i_row = indices + (size_t)row * n_conn; \
        const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)row * n_conn; \
        float w0 = is_homo ? READ_W(__ldg(&weights[0])) : 0.0f; \
        for (int k = lane_id; k < n_conn; k += 32) { \
            int32_t idx = __ldg(&i_row[k]); \
            float wk = is_homo ? w0 : READ_W(__ldg(&w_row[k])); \
            ATOMIC_ADD_W(&output[idx], wk * v); \
        } \
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
        float w = is_homo ? READ_W(__ldg(&weights[0])) : READ_W(__ldg(&weights[idx])); \
        ATOMIC_ADD_W(&output[__ldg(&indices[idx])], w * READ_W(__ldg(&vector[row]))); \
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
        if (k < n_conn) { s_idx[threadIdx.x] = __ldg(&i_row[k]); s_wt[threadIdx.x] = is_homo ? 1.0f : __ldg(&w_row[k]); }
        __syncthreads();
        int tile = min((int)blockDim.x, n_conn - base);
        if (threadIdx.x < tile) val += s_wt[threadIdx.x] * __ldg(&vector[s_idx[threadIdx.x]]);
        __syncthreads();
    }
    int lane = threadIdx.x & 31, warpid = threadIdx.x >> 5; val = warp_reduce_sum_f32(val);
    if (lane == 0) s_red[warpid] = val; __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5; val = (threadIdx.x < n_warps) ? s_red[lane] : 0.0f;
    if (warpid == 0) val = warp_reduce_sum_f32(val);
    if (threadIdx.x == 0) output[row] = is_homo ? (__ldg(&weights[0]) * val) : val;
}
__global__ void _gather_vec4_kern(const int32_t* __restrict__ indices, const float* __restrict__ vector, float* __restrict__ output, const float* __restrict__ weights, int n_pre, int n_conn, int is_homo) {
    extern __shared__ float smem_red[];
    int row = blockIdx.x; if (row >= n_pre) return;
    size_t base = (size_t)row * n_conn;
    const int4* i4 = (const int4*)(indices + base); const float4* w4 = is_homo ? nullptr : (const float4*)(weights + base);
    int n4 = n_conn >> 2; float val = 0.0f;
    for (int k = threadIdx.x; k < n4; k += blockDim.x) {
        int4 idx = __ldg(&i4[k]);
        if (!is_homo) {
            float4 ww = __ldg(&w4[k]);
            val += ww.x * __ldg(&vector[idx.x]) + ww.y * __ldg(&vector[idx.y])
                 + ww.z * __ldg(&vector[idx.z]) + ww.w * __ldg(&vector[idx.w]);
        } else {
            val += __ldg(&vector[idx.x]) + __ldg(&vector[idx.y])
                 + __ldg(&vector[idx.z]) + __ldg(&vector[idx.w]);
        }
    }
    for (int k = (n4 << 2) + threadIdx.x; k < n_conn; k += blockDim.x) {
        float v = __ldg(&vector[__ldg(&indices[base + k])]);
        val += is_homo ? v : (__ldg(&weights[base + k]) * v);
    }
    int lane = threadIdx.x & 31, warpid = threadIdx.x >> 5; val = warp_reduce_sum_f32(val);
    if (lane == 0) smem_red[warpid] = val; __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5; val = (threadIdx.x < n_warps) ? smem_red[lane] : 0.0f;
    if (warpid == 0) val = warp_reduce_sum_f32(val);
    if (threadIdx.x == 0) output[row] = is_homo ? (__ldg(&weights[0]) * val) : val;
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
/*
 * fcnmv gather auto (f32) — dispatch strategy:
 *
 * Achieved throughput (amortized, RTX 3080 Ti, 512 GB/s peak DRAM BW):
 *   10Kx10Kx1000 hetero: 0.124 ms → ~647 GB/s (126% of peak, L2-assisted)
 *   5Kx5Kx500  hetero: 0.026 ms → ~772 GB/s (L2-cached regime)
 *   1Kx1Kx100  hetero: 0.009 ms → ~90 GB/s  (launch-overhead-dominated)
 *
 * Fundamental barriers:
 *   - Random column access for vector[indices[i,k]] prevents global memory
 *     coalescing; performance relies on the vector fitting in L2 cache.
 *     For n_post > ~1M elements (4 MB in f32), L2 thrashing degrades BW.
 *   - TVM FFI per-call dispatch overhead (~1.4 ms) dominates for small
 *     matrices (n_pre * n_conn < 100K); irreducible without kernel fusion
 *     or persistent-kernel approaches at a higher level.
 *   - At ~10% density the format is equivalent to dense; format overhead
 *     (index loads) adds ~50% more traffic than a dense matmul would need.
 *
 * Future directions:
 *   - Shared-memory vector caching for n_post <= 12K (48 KB / 4B).
 *   - Persistent kernel with CUDA Graphs to amortize launch overhead.
 *   - Two-pass approach: sort indices, then segmented reduction for better
 *     coalescing (requires preprocessing).
 */
void fcnmv_gather_auto_f32(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    if (n_conn <= 32)
        _gather_warp_kern_f32<<<n_pre, 32, 0, s>>>(
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    else if (n_conn % 4 == 0 && n_conn >= 1024)
        // vec4: use only when n_conn/4 >= blockDim to avoid idle threads
        _gather_vec4_kern<<<n_pre, 256, 32 * sizeof(float), s>>>(
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    else
        _gather_basic_kern_f32<<<n_pre, 256, 32 * sizeof(float), s>>>(
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
}

