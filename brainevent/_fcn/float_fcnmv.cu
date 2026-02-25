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
#include "cuda_common.h"
#include "brainevent/common.h"

// ============================================================================
// FCN Matrix-Vector Multiplication (fcnmv)
// ============================================================================

#define DEFINE_GATHER_WARP_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _gather_warp_homo_kern##SUFFIX(                                               \
    const int32_t* __restrict__ indices,                                                      \
    const WEIGHT_T* __restrict__ vector,                                                      \
    WEIGHT_T* __restrict__ output,                                                            \
    const WEIGHT_T* __restrict__ weights,                                                     \
    int n_pre, int n_conn                                                                     \
) {                                                                                           \
    int row = blockIdx.x;                                                                     \
    if (row >= n_pre) return;                                                                 \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                    \
    ACC_T val = ACC_ZERO;                                                                     \
    for (int k = threadIdx.x; k < n_conn; k += 32) {                                          \
        int32_t idx = __ldg(&i_row[k]);                                                       \
        val += READ_W(__ldg(&vector[idx]));                                                   \
    }                                                                                         \
    val = WARP_RED(val);                                                                      \
    if (threadIdx.x == 0)                                                                     \
        output[row] = WRITE_W(READ_W(__ldg(&weights[0])) * val);                              \
}

#define DEFINE_GATHER_WARP_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _gather_warp_hetero_kern##SUFFIX(                                               \
    const int32_t* __restrict__ indices,                                                        \
    const WEIGHT_T* __restrict__ vector,                                                        \
    WEIGHT_T* __restrict__ output,                                                              \
    const WEIGHT_T* __restrict__ weights,                                                       \
    int n_pre, int n_conn                                                                       \
) {                                                                                             \
    int row = blockIdx.x;                                                                       \
    if (row >= n_pre) return;                                                                   \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                      \
    const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                     \
    ACC_T val = ACC_ZERO;                                                                       \
    for (int k = threadIdx.x; k < n_conn; k += 32) {                                            \
        int32_t idx = __ldg(&i_row[k]);                                                         \
        val += READ_W(__ldg(&w_row[k])) * READ_W(__ldg(&vector[idx]));                          \
    }                                                                                           \
    val = WARP_RED(val);                                                                        \
    if (threadIdx.x == 0)                                                                       \
        output[row] = WRITE_W(val);                                                             \
}

#define DEFINE_GATHER_BASIC_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _gather_basic_homo_kern##SUFFIX(                                               \
    const int32_t* __restrict__ indices,                                                       \
    const WEIGHT_T* __restrict__ vector,                                                       \
    WEIGHT_T* __restrict__ output,                                                             \
    const WEIGHT_T* __restrict__ weights,                                                      \
    int n_pre, int n_conn                                                                      \
) {                                                                                            \
    extern __shared__ char _smem_bytes[];                                                      \
    ACC_T* smem_red = reinterpret_cast<ACC_T*>(_smem_bytes);                                   \
    int row = blockIdx.x;                                                                      \
    if (row >= n_pre) return;                                                                  \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                     \
    int lane   = threadIdx.x & 31;                                                             \
    int warpid = threadIdx.x >> 5;                                                             \
    ACC_T val = ACC_ZERO;                                                                      \
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x) {                                   \
        int32_t idx = __ldg(&i_row[k]);                                                        \
        val += READ_W(__ldg(&vector[idx]));                                                    \
    }                                                                                          \
    val = WARP_RED(val);                                                                       \
    if (lane == 0) smem_red[warpid] = val;                                                     \
    __syncthreads();                                                                           \
    int n_warps_in_block = blockDim.x >> 5;                                                    \
    val = (threadIdx.x < n_warps_in_block) ? smem_red[lane] : ACC_ZERO;                        \
    if (warpid == 0) val = WARP_RED(val);                                                      \
    if (threadIdx.x == 0)                                                                      \
        output[row] = WRITE_W(READ_W(__ldg(&weights[0])) * val);                               \
}

#define DEFINE_GATHER_BASIC_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _gather_basic_hetero_kern##SUFFIX(                                               \
    const int32_t* __restrict__ indices,                                                         \
    const WEIGHT_T* __restrict__ vector,                                                         \
    WEIGHT_T* __restrict__ output,                                                               \
    const WEIGHT_T* __restrict__ weights,                                                        \
    int n_pre, int n_conn                                                                        \
) {                                                                                              \
    extern __shared__ char _smem_bytes[];                                                        \
    ACC_T* smem_red = reinterpret_cast<ACC_T*>(_smem_bytes);                                     \
    int row = blockIdx.x;                                                                        \
    if (row >= n_pre) return;                                                                    \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                       \
    const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                      \
    int lane   = threadIdx.x & 31;                                                               \
    int warpid = threadIdx.x >> 5;                                                               \
    ACC_T val = ACC_ZERO;                                                                        \
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x) {                                     \
        int32_t idx = __ldg(&i_row[k]);                                                          \
        val += READ_W(__ldg(&w_row[k])) * READ_W(__ldg(&vector[idx]));                           \
    }                                                                                            \
    val = WARP_RED(val);                                                                         \
    if (lane == 0) smem_red[warpid] = val;                                                       \
    __syncthreads();                                                                             \
    int n_warps_in_block = blockDim.x >> 5;                                                      \
    val = (threadIdx.x < n_warps_in_block) ? smem_red[lane] : ACC_ZERO;                          \
    if (warpid == 0) val = WARP_RED(val);                                                        \
    if (threadIdx.x == 0)                                                                        \
        output[row] = WRITE_W(val);                                                              \
}

#define DEFINE_SCATTER_BASIC_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _scatter_basic_homo_kern##SUFFIX(                               \
    const int32_t* __restrict__ indices,                                        \
    const WEIGHT_T* __restrict__ vector,                                        \
    WEIGHT_T*       __restrict__ output,                                        \
    const WEIGHT_T* __restrict__ weights,                                       \
    int n_pre, int n_conn                                                       \
) {                                                                             \
    int row = blockIdx.x;                                                       \
    if (row >= n_pre) return;                                                   \
    ACC_T v = READ_W(__ldg(&vector[row]));                                      \
    const int32_t* i_row = indices + (size_t)row * n_conn;                      \
    ACC_T w0 = READ_W(__ldg(&weights[0]));                                      \
    ACC_T wv = w0 * v;                                                          \
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x) {                    \
        int32_t idx = __ldg(&i_row[k]);                                         \
        ATOMIC_ADD_W(&output[idx], wv);                                         \
    }                                                                           \
}

#define DEFINE_SCATTER_BASIC_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _scatter_basic_hetero_kern##SUFFIX(                               \
    const int32_t* __restrict__ indices,                                          \
    const WEIGHT_T* __restrict__ vector,                                          \
    WEIGHT_T*       __restrict__ output,                                          \
    const WEIGHT_T* __restrict__ weights,                                         \
    int n_pre, int n_conn                                                         \
) {                                                                               \
    int row = blockIdx.x;                                                         \
    if (row >= n_pre) return;                                                     \
    ACC_T v = READ_W(__ldg(&vector[row]));                                        \
    const int32_t* i_row = indices + (size_t)row * n_conn;                        \
    const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                       \
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x) {                      \
        int32_t idx = __ldg(&i_row[k]);                                           \
        ACC_T wk = READ_W(__ldg(&w_row[k]));                                      \
        ATOMIC_ADD_W(&output[idx], wk * v);                                       \
    }                                                                             \
}

#define DEFINE_SCATTER_WARP_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _scatter_warp_homo_kern##SUFFIX(                               \
    const int32_t* __restrict__ indices,                                       \
    const WEIGHT_T* __restrict__ vector,                                       \
    WEIGHT_T*       __restrict__ output,                                       \
    const WEIGHT_T* __restrict__ weights,                                      \
    int n_pre, int n_conn                                                      \
) {                                                                            \
    int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;              \
    int lane_id   = threadIdx.x & 31;                                          \
    int num_warps = (gridDim.x * blockDim.x) >> 5;                             \
    ACC_T w0 = READ_W(__ldg(&weights[0]));                                     \
    for (int row = warp_id; row < n_pre; row += num_warps) {                   \
        ACC_T v = READ_W(__ldg(&vector[row]));                                 \
        const int32_t* i_row = indices + (size_t)row * n_conn;                 \
        ACC_T wv = w0 * v;                                                     \
        for (int k = lane_id; k < n_conn; k += 32) {                           \
            int32_t idx = __ldg(&i_row[k]);                                    \
            ATOMIC_ADD_W(&output[idx], wv);                                    \
        }                                                                      \
    }                                                                          \
}

#define DEFINE_SCATTER_WARP_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _scatter_warp_hetero_kern##SUFFIX(                               \
    const int32_t* __restrict__ indices,                                         \
    const WEIGHT_T* __restrict__ vector,                                         \
    WEIGHT_T*       __restrict__ output,                                         \
    const WEIGHT_T* __restrict__ weights,                                        \
    int n_pre, int n_conn                                                        \
) {                                                                              \
    int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;                \
    int lane_id   = threadIdx.x & 31;                                            \
    int num_warps = (gridDim.x * blockDim.x) >> 5;                               \
    for (int row = warp_id; row < n_pre; row += num_warps) {                     \
        ACC_T v = READ_W(__ldg(&vector[row]));                                   \
        const int32_t* i_row = indices + (size_t)row * n_conn;                   \
        const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                  \
        for (int k = lane_id; k < n_conn; k += 32) {                             \
            int32_t idx = __ldg(&i_row[k]);                                      \
            ACC_T wk = READ_W(__ldg(&w_row[k]));                                 \
            ATOMIC_ADD_W(&output[idx], wk * v);                                  \
        }                                                                        \
    }                                                                            \
}

#define DEFINE_SCATTER_GS_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)          \
__global__ void _scatter_gs_homo_kern##SUFFIX(                                        \
    const int32_t* __restrict__ indices,                                              \
    const WEIGHT_T* __restrict__ vector,                                              \
    WEIGHT_T*       __restrict__ output,                                              \
    const WEIGHT_T* __restrict__ weights,                                             \
    int n_pre, int n_conn                                                             \
) {                                                                                   \
    int total  = n_pre * n_conn;                                                      \
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;                               \
    int stride = blockDim.x * gridDim.x;                                              \
    ACC_T w = READ_W(__ldg(&weights[0]));                                             \
    for (int idx = tid; idx < total; idx += stride) {                                 \
        int row = idx / n_conn;                                                       \
        ATOMIC_ADD_W(&output[__ldg(&indices[idx])], w * READ_W(__ldg(&vector[row]))); \
    }                                                                                 \
}

#define DEFINE_SCATTER_GS_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)        \
__global__ void _scatter_gs_hetero_kern##SUFFIX(                                      \
    const int32_t* __restrict__ indices,                                              \
    const WEIGHT_T* __restrict__ vector,                                              \
    WEIGHT_T*       __restrict__ output,                                              \
    const WEIGHT_T* __restrict__ weights,                                             \
    int n_pre, int n_conn                                                             \
) {                                                                                   \
    int total  = n_pre * n_conn;                                                      \
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;                               \
    int stride = blockDim.x * gridDim.x;                                              \
    for (int idx = tid; idx < total; idx += stride) {                                 \
        int row = idx / n_conn;                                                       \
        ACC_T w = READ_W(__ldg(&weights[idx]));                                       \
        ATOMIC_ADD_W(&output[__ldg(&indices[idx])], w * READ_W(__ldg(&vector[row]))); \
    }                                                                                 \
}

// Instantiations
// ---- float32 ----
DEFINE_GATHER_WARP_HOMO   (_f32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_GATHER_WARP_HETERO (_f32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_GATHER_BASIC_HOMO  (_f32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_GATHER_BASIC_HETERO(_f32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_SCATTER_BASIC_HOMO (_f32, float, float, READ_F32, atomic_add_f32)
DEFINE_SCATTER_BASIC_HETERO(_f32, float, float, READ_F32, atomic_add_f32)
DEFINE_SCATTER_WARP_HOMO  (_f32, float, float, READ_F32, atomic_add_f32)
DEFINE_SCATTER_WARP_HETERO(_f32, float, float, READ_F32, atomic_add_f32)
DEFINE_SCATTER_GS_HOMO     (_f32, float, float, READ_F32, atomic_add_f32)
DEFINE_SCATTER_GS_HETERO   (_f32, float, float, READ_F32, atomic_add_f32)

// ---- float64 ----
DEFINE_GATHER_WARP_HOMO   (_f64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_GATHER_WARP_HETERO (_f64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_GATHER_BASIC_HOMO  (_f64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_GATHER_BASIC_HETERO(_f64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_SCATTER_BASIC_HOMO (_f64, double, double, READ_F64, atomic_add_f64)
DEFINE_SCATTER_BASIC_HETERO(_f64, double, double, READ_F64, atomic_add_f64)
DEFINE_SCATTER_WARP_HOMO  (_f64, double, double, READ_F64, atomic_add_f64)
DEFINE_SCATTER_WARP_HETERO(_f64, double, double, READ_F64, atomic_add_f64)
DEFINE_SCATTER_GS_HOMO     (_f64, double, double, READ_F64, atomic_add_f64)
DEFINE_SCATTER_GS_HETERO   (_f64, double, double, READ_F64, atomic_add_f64)

// ---- float16 ----
DEFINE_GATHER_WARP_HOMO   (_f16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_GATHER_WARP_HETERO (_f16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_GATHER_BASIC_HOMO  (_f16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_GATHER_BASIC_HETERO(_f16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_SCATTER_BASIC_HOMO (_f16, __half, float, READ_F16, atomic_add_f16)
DEFINE_SCATTER_BASIC_HETERO(_f16, __half, float, READ_F16, atomic_add_f16)
DEFINE_SCATTER_WARP_HOMO  (_f16, __half, float, READ_F16, atomic_add_f16)
DEFINE_SCATTER_WARP_HETERO(_f16, __half, float, READ_F16, atomic_add_f16)
DEFINE_SCATTER_GS_HOMO     (_f16, __half, float, READ_F16, atomic_add_f16)
DEFINE_SCATTER_GS_HETERO   (_f16, __half, float, READ_F16, atomic_add_f16)

// ---- bfloat16 ----
DEFINE_GATHER_WARP_HOMO   (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_GATHER_WARP_HETERO (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_GATHER_BASIC_HOMO  (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_GATHER_BASIC_HETERO(_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_SCATTER_BASIC_HOMO (_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_SCATTER_BASIC_HETERO(_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_SCATTER_WARP_HOMO  (_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_SCATTER_WARP_HETERO(_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_SCATTER_GS_HOMO     (_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_SCATTER_GS_HETERO   (_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)

// SpMV Specializations (f32 only)
__global__ void _gather_shared_homo_kern(const int32_t* __restrict__ indices, const float* __restrict__ vector, float* __restrict__ output, const float* __restrict__ weights, int n_pre, int n_conn) {
    extern __shared__ char smem_raw[];
    int32_t* s_idx = reinterpret_cast<int32_t*>(smem_raw);
    float*   s_red = reinterpret_cast<float*>(smem_raw + blockDim.x * sizeof(int32_t));
    int row = blockIdx.x; if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    float val = 0.0f;
    for (int base = 0; base < n_conn; base += blockDim.x) {
        int k = base + threadIdx.x;
        if (k < n_conn) { s_idx[threadIdx.x] = __ldg(&i_row[k]); }
        __syncthreads();
        int tile = min((int)blockDim.x, n_conn - base);
        if (threadIdx.x < tile) val += __ldg(&vector[s_idx[threadIdx.x]]);
        __syncthreads();
    }
    int lane = threadIdx.x & 31, warpid = threadIdx.x >> 5; val = warp_reduce_sum_f32(val);
    if (lane == 0) s_red[warpid] = val; __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5; val = (threadIdx.x < n_warps) ? s_red[lane] : 0.0f;
    if (warpid == 0) val = warp_reduce_sum_f32(val);
    if (threadIdx.x == 0) output[row] = __ldg(&weights[0]) * val;
}

__global__ void _gather_shared_hetero_kern(const int32_t* __restrict__ indices, const float* __restrict__ vector, float* __restrict__ output, const float* __restrict__ weights, int n_pre, int n_conn) {
    extern __shared__ char smem_raw[];
    int32_t* s_idx = reinterpret_cast<int32_t*>(smem_raw);
    float*   s_wt  = reinterpret_cast<float*>(smem_raw + blockDim.x * sizeof(int32_t));
    float*   s_red = reinterpret_cast<float*>(smem_raw + blockDim.x * (sizeof(int32_t) + sizeof(float)));
    int row = blockIdx.x; if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = weights + (size_t)row * n_conn;
    float val = 0.0f;
    for (int base = 0; base < n_conn; base += blockDim.x) {
        int k = base + threadIdx.x;
        if (k < n_conn) { s_idx[threadIdx.x] = __ldg(&i_row[k]); s_wt[threadIdx.x] = __ldg(&w_row[k]); }
        __syncthreads();
        int tile = min((int)blockDim.x, n_conn - base);
        if (threadIdx.x < tile) val += s_wt[threadIdx.x] * __ldg(&vector[s_idx[threadIdx.x]]);
        __syncthreads();
    }
    int lane = threadIdx.x & 31, warpid = threadIdx.x >> 5; val = warp_reduce_sum_f32(val);
    if (lane == 0) s_red[warpid] = val; __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5; val = (threadIdx.x < n_warps) ? s_red[lane] : 0.0f;
    if (warpid == 0) val = warp_reduce_sum_f32(val);
    if (threadIdx.x == 0) output[row] = val;
}

__global__ void _gather_vec4_homo_kern(const int32_t* __restrict__ indices, const float* __restrict__ vector, float* __restrict__ output, const float* __restrict__ weights, int n_pre, int n_conn) {
    extern __shared__ float smem_red[];
    int row = blockIdx.x; if (row >= n_pre) return;
    size_t base = (size_t)row * n_conn;
    const int4* i4 = (const int4*)(indices + base);
    int n4 = n_conn >> 2; float val = 0.0f;
    for (int k = threadIdx.x; k < n4; k += blockDim.x) {
        int4 idx = __ldg(&i4[k]);
        val += __ldg(&vector[idx.x]) + __ldg(&vector[idx.y])
             + __ldg(&vector[idx.z]) + __ldg(&vector[idx.w]);
    }
    for (int k = (n4 << 2) + threadIdx.x; k < n_conn; k += blockDim.x) {
        val += __ldg(&vector[__ldg(&indices[base + k])]);
    }
    int lane = threadIdx.x & 31, warpid = threadIdx.x >> 5; val = warp_reduce_sum_f32(val);
    if (lane == 0) smem_red[warpid] = val; __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5; val = (threadIdx.x < n_warps) ? smem_red[lane] : 0.0f;
    if (warpid == 0) val = warp_reduce_sum_f32(val);
    if (threadIdx.x == 0) output[row] = __ldg(&weights[0]) * val;
}

__global__ void _gather_vec4_hetero_kern(const int32_t* __restrict__ indices, const float* __restrict__ vector, float* __restrict__ output, const float* __restrict__ weights, int n_pre, int n_conn) {
    extern __shared__ float smem_red[];
    int row = blockIdx.x; if (row >= n_pre) return;
    size_t base = (size_t)row * n_conn;
    const int4* i4 = (const int4*)(indices + base);
    const float4* w4 = (const float4*)(weights + base);
    int n4 = n_conn >> 2; float val = 0.0f;
    for (int k = threadIdx.x; k < n4; k += blockDim.x) {
        int4 idx = __ldg(&i4[k]);
        float4 ww = __ldg(&w4[k]);
        val += ww.x * __ldg(&vector[idx.x]) + ww.y * __ldg(&vector[idx.y])
             + ww.z * __ldg(&vector[idx.z]) + ww.w * __ldg(&vector[idx.w]);
    }
    for (int k = (n4 << 2) + threadIdx.x; k < n_conn; k += blockDim.x) {
        val += __ldg(&weights[base + k]) * __ldg(&vector[__ldg(&indices[base + k])]);
    }
    int lane = threadIdx.x & 31, warpid = threadIdx.x >> 5; val = warp_reduce_sum_f32(val);
    if (lane == 0) smem_red[warpid] = val; __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5; val = (threadIdx.x < n_warps) ? smem_red[lane] : 0.0f;
    if (warpid == 0) val = warp_reduce_sum_f32(val);
    if (threadIdx.x == 0) output[row] = val;
}

// SpMV FFI Entries
// ---- FFI macro: gather homo auto ----
#define FFI_GATHER_HOMO_AUTO(SUFFIX, WEIGHT_C_T, SHM_SIZE)                        \
void fcnmv_gather_homo_auto##SUFFIX(                                              \
    const BE::Tensor weights, const BE::Tensor indices,                   \
    const BE::Tensor vector,  BE::Tensor output, int64_t stream     \
) {                                                                               \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                     \
    int n_pre       = static_cast<int>(indices.size(0));                          \
    int n_conn      = static_cast<int>(indices.size(1));                          \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const WEIGHT_C_T* d_vec = static_cast<const WEIGHT_C_T*>(vector.data_ptr());  \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());    \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());        \
    if (n_conn <= 32)                                                             \
        _gather_warp_homo_kern##SUFFIX<<<n_pre, 32, 0, s>>>(                      \
            d_idx, d_vec, d_out, d_w, n_pre, n_conn);                             \
    else                                                                          \
        _gather_basic_homo_kern##SUFFIX<<<n_pre, 256, SHM_SIZE, s>>>(             \
            d_idx, d_vec, d_out, d_w, n_pre, n_conn);                             \
}

// ---- FFI macro: gather hetero auto ----
#define FFI_GATHER_HETERO_AUTO(SUFFIX, WEIGHT_C_T, SHM_SIZE)                      \
void fcnmv_gather_hetero_auto##SUFFIX(                                            \
    const BE::Tensor weights, const BE::Tensor indices,                   \
    const BE::Tensor vector,  BE::Tensor output, int64_t stream     \
) {                                                                               \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                     \
    int n_pre       = static_cast<int>(indices.size(0));                          \
    int n_conn      = static_cast<int>(indices.size(1));                          \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const WEIGHT_C_T* d_vec = static_cast<const WEIGHT_C_T*>(vector.data_ptr());  \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());    \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());        \
    if (n_conn <= 32)                                                             \
        _gather_warp_hetero_kern##SUFFIX<<<n_pre, 32, 0, s>>>(                    \
            d_idx, d_vec, d_out, d_w, n_pre, n_conn);                             \
    else                                                                          \
        _gather_basic_hetero_kern##SUFFIX<<<n_pre, 256, SHM_SIZE, s>>>(           \
            d_idx, d_vec, d_out, d_w, n_pre, n_conn);                             \
}

// ---- FFI macro: scatter homo auto ----
#define FFI_SCATTER_HOMO_AUTO(SUFFIX, WEIGHT_C_T)                                 \
void fcnmv_scatter_homo_auto##SUFFIX(                                             \
    const BE::Tensor weights, const BE::Tensor indices,                   \
    const BE::Tensor vector,  BE::Tensor output, int64_t stream     \
) {                                                                               \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                     \
    int n_pre       = static_cast<int>(indices.size(0));                          \
    int n_conn      = static_cast<int>(indices.size(1));                          \
    int n_post      = static_cast<int>(output.size(0));                           \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const WEIGHT_C_T* d_vec = static_cast<const WEIGHT_C_T*>(vector.data_ptr());  \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());    \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());        \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(WEIGHT_C_T), s);            \
    if (n_conn <= 32) {                                                           \
        int blocks = (n_pre + 7) / 8;                                             \
        _scatter_warp_homo_kern##SUFFIX<<<blocks, 256, 0, s>>>(                   \
            d_idx, d_vec, d_out, d_w, n_pre, n_conn);                             \
    } else if ((long long)n_pre * n_conn > 262144LL) {                            \
        int blocks = min(1024, (int)((n_pre * n_conn + 255) / 256));              \
        _scatter_gs_homo_kern##SUFFIX<<<blocks, 256, 0, s>>>(                     \
            d_idx, d_vec, d_out, d_w, n_pre, n_conn);                             \
    } else {                                                                      \
        _scatter_basic_homo_kern##SUFFIX<<<n_pre, 256, 0, s>>>(                   \
            d_idx, d_vec, d_out, d_w, n_pre, n_conn);                             \
    }                                                                             \
}

// ---- FFI macro: scatter hetero auto ----
#define FFI_SCATTER_HETERO_AUTO(SUFFIX, WEIGHT_C_T)                               \
void fcnmv_scatter_hetero_auto##SUFFIX(                                           \
    const BE::Tensor weights, const BE::Tensor indices,                   \
    const BE::Tensor vector,  BE::Tensor output, int64_t stream     \
) {                                                                               \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                     \
    int n_pre       = static_cast<int>(indices.size(0));                          \
    int n_conn      = static_cast<int>(indices.size(1));                          \
    int n_post      = static_cast<int>(output.size(0));                           \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const WEIGHT_C_T* d_vec = static_cast<const WEIGHT_C_T*>(vector.data_ptr());  \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());    \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());        \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(WEIGHT_C_T), s);            \
    if (n_conn <= 32) {                                                           \
        int blocks = (n_pre + 7) / 8;                                             \
        _scatter_warp_hetero_kern##SUFFIX<<<blocks, 256, 0, s>>>(                 \
            d_idx, d_vec, d_out, d_w, n_pre, n_conn);                             \
    } else if ((long long)n_pre * n_conn > 262144LL) {                            \
        int blocks = min(1024, (int)((n_pre * n_conn + 255) / 256));              \
        _scatter_gs_hetero_kern##SUFFIX<<<blocks, 256, 0, s>>>(                   \
            d_idx, d_vec, d_out, d_w, n_pre, n_conn);                             \
    } else {                                                                      \
        _scatter_basic_hetero_kern##SUFFIX<<<n_pre, 256, 0, s>>>(                 \
            d_idx, d_vec, d_out, d_w, n_pre, n_conn);                             \
    }                                                                             \
}

// SpMV FFI Instantiations
// ---- float32 ----
// @BE fcnmv_gather_homo_auto_f32
FFI_GATHER_HOMO_AUTO  (_f32, float, 32 * sizeof(float))
// @BE fcnmv_gather_hetero_auto_f32
FFI_GATHER_HETERO_AUTO(_f32, float, 32 * sizeof(float))
// @BE fcnmv_scatter_homo_auto_f32
FFI_SCATTER_HOMO_AUTO (_f32, float)
// @BE fcnmv_scatter_hetero_auto_f32
FFI_SCATTER_HETERO_AUTO(_f32, float)

// ---- float64 ----
// @BE fcnmv_gather_homo_auto_f64
FFI_GATHER_HOMO_AUTO  (_f64, double, 32 * sizeof(double))
// @BE fcnmv_gather_hetero_auto_f64
FFI_GATHER_HETERO_AUTO(_f64, double, 32 * sizeof(double))
// @BE fcnmv_scatter_homo_auto_f64
FFI_SCATTER_HOMO_AUTO (_f64, double)
// @BE fcnmv_scatter_hetero_auto_f64
FFI_SCATTER_HETERO_AUTO(_f64, double)

// ---- float16 ----
// @BE fcnmv_gather_homo_auto_f16
FFI_GATHER_HOMO_AUTO  (_f16, __half, 32 * sizeof(float))
// @BE fcnmv_gather_hetero_auto_f16
FFI_GATHER_HETERO_AUTO(_f16, __half, 32 * sizeof(float))
// @BE fcnmv_scatter_homo_auto_f16
FFI_SCATTER_HOMO_AUTO (_f16, __half)
// @BE fcnmv_scatter_hetero_auto_f16
FFI_SCATTER_HETERO_AUTO(_f16, __half)

// ---- bfloat16 ----
// @BE fcnmv_gather_homo_auto_bf16
FFI_GATHER_HOMO_AUTO  (_bf16, __nv_bfloat16, 32 * sizeof(float))
// @BE fcnmv_gather_hetero_auto_bf16
FFI_GATHER_HETERO_AUTO(_bf16, __nv_bfloat16, 32 * sizeof(float))
// @BE fcnmv_scatter_homo_auto_bf16
FFI_SCATTER_HOMO_AUTO (_bf16, __nv_bfloat16)
// @BE fcnmv_scatter_hetero_auto_bf16
FFI_SCATTER_HETERO_AUTO(_bf16, __nv_bfloat16)

// SpMV f32-specific specializations
// @BE fcnmv_gather_shared_homo_f32
void fcnmv_gather_shared_homo_f32(
    const BE::Tensor weights, const BE::Tensor indices,
    const BE::Tensor vector,  BE::Tensor output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    _gather_shared_homo_kern<<<n_pre, 256, 256 * (sizeof(int32_t) + sizeof(float)), s>>>(
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>(vector.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        n_pre, n_conn);
}

// @BE fcnmv_gather_shared_hetero_f32
void fcnmv_gather_shared_hetero_f32(
    const BE::Tensor weights, const BE::Tensor indices,
    const BE::Tensor vector,  BE::Tensor output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    _gather_shared_hetero_kern<<<n_pre, 256, 256 * (sizeof(int32_t) + 2 * sizeof(float)), s>>>(
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>(vector.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        n_pre, n_conn);
}

// @BE fcnmv_gather_vec4_homo_f32
void fcnmv_gather_vec4_homo_f32(
    const BE::Tensor weights, const BE::Tensor indices,
    const BE::Tensor vector,  BE::Tensor output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    _gather_vec4_homo_kern<<<n_pre, 256, 32 * sizeof(float), s>>>(
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>(vector.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        n_pre, n_conn);
}

// @BE fcnmv_gather_vec4_hetero_f32
void fcnmv_gather_vec4_hetero_f32(
    const BE::Tensor weights, const BE::Tensor indices,
    const BE::Tensor vector,  BE::Tensor output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    _gather_vec4_hetero_kern<<<n_pre, 256, 32 * sizeof(float), s>>>(
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>(vector.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        n_pre, n_conn);
}

/*
 * fcnmv gather auto (f32) — dispatch strategy:
 *
 * Achieved throughput (amortized, RTX 3080 Ti, 512 GB/s peak DRAM BW):
 *   10Kx10Kx1000 hetero: 0.124 ms → ~647 GB/s (126% of peak, L2-assisted)
 *   5Kx5Kx500  hetero: 0.026 ms → ~772 GB/s (L2-cached regime)
 *   1Kx1Kx100  hetero: 0.009 ms → ~90 GB/s  (launch-overhead-dominated)
 */
// @BE fcnmv_gather_auto_f32
void fcnmv_gather_auto_f32(
    const BE::Tensor weights, const BE::Tensor indices,
    const BE::Tensor vector,  BE::Tensor output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    bool homo = (weights.ndim() == 1);
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    if (homo) {
        if (n_conn <= 32)
            _gather_warp_homo_kern_f32<<<n_pre, 32, 0, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn);
        else if (n_conn % 4 == 0 && n_conn >= 1024)
            _gather_vec4_homo_kern<<<n_pre, 256, 32 * sizeof(float), s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn);
        else
            _gather_basic_homo_kern_f32<<<n_pre, 256, 32 * sizeof(float), s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn);
    } else {
        if (n_conn <= 32)
            _gather_warp_hetero_kern_f32<<<n_pre, 32, 0, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn);
        else if (n_conn % 4 == 0 && n_conn >= 1024)
            _gather_vec4_hetero_kern<<<n_pre, 256, 32 * sizeof(float), s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn);
        else
            _gather_basic_hetero_kern_f32<<<n_pre, 256, 32 * sizeof(float), s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn);
    }
}
