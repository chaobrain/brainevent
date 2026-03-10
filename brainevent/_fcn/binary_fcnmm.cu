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
 * binary_fcnmm.cu -- Event-Driven Binary FCN Sparse Matrix-Matrix CUDA Kernels
 * ==============================================================================
 *
 * This module provides optimized CUDA kernels for event-driven sparse
 * matrix-matrix multiplication with fixed connection number (FCN).
 *
 * Operator: binary_fcnmm
 *   - Gather mode (transpose=False): output[i,j] = sum_k weights[i,k] * is_active(matrix[indices[i,k], j])
 *   - Scatter mode (transpose=True): output[indices[i,k], j] += weights[i,k] * is_active(matrix[i,j])
 *
 * Supports weight dtypes: float32, float64, float16, bfloat16
 * Supports spike dtypes:  bool (uint8), float32, float64, float16, bfloat16
 * Supports homo (scalar weight) and hetero (per-connection weight array) modes.
 */

#include "cuda_common.h"
#include "brainevent/common.h"

// ============================================================================
// FCN Matrix-Matrix Multiplication (fcnmm) — Optimized CUDA Kernels
// ============================================================================

// Gather warp kernel: 1 block per row, 32 threads per block (for n_conn <= 32)
#define DEFINE_BGM_WARP_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W)    \
__global__ void _bgm_warp_homo_kern##SUFFIX(                                                  \
    const int32_t* __restrict__ indices,                                                      \
    const SPIKE_T* __restrict__ matrix,                                                       \
    WEIGHT_T*      __restrict__ output,                                                       \
    const WEIGHT_T* __restrict__ weights,                                                     \
    int n_pre, int n_conn, int n_batch                                                        \
) {                                                                                           \
    int row = blockIdx.x;                                                                     \
    int t   = threadIdx.x;                                                                    \
    int j   = (int)blockIdx.y * 32 + t;                                                       \
    if (row >= n_pre) return;                                                                 \
    bool col_valid = (j < n_batch);                                                           \
    int  safe_j    = col_valid ? j : 0;                                                       \
    const int32_t*  i_row = indices + (size_t)row * n_conn;                                   \
    ACC_T accum = (ACC_T)0;                                                                   \
    for (int k = 0; k < n_conn; k++) {                                                        \
        int  src    = __ldg(&i_row[k]);                                                       \
        bool active = col_valid && IS_ACTIVE(__ldg(&matrix[(size_t)src * n_batch + safe_j])); \
        if (active) accum += (ACC_T)1;                                                        \
    }                                                                                         \
    if (col_valid)                                                                            \
        output[(size_t)row * n_batch + j] = WRITE_W(READ_W(__ldg(&weights[0])) * accum);      \
}

#define DEFINE_BGM_WARP_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W)  \
__global__ void _bgm_warp_hetero_kern##SUFFIX(                                                \
    const int32_t* __restrict__ indices,                                                      \
    const SPIKE_T* __restrict__ matrix,                                                       \
    WEIGHT_T*      __restrict__ output,                                                       \
    const WEIGHT_T* __restrict__ weights,                                                     \
    int n_pre, int n_conn, int n_batch                                                        \
) {                                                                                           \
    int row = blockIdx.x;                                                                     \
    int t   = threadIdx.x;                                                                    \
    int j   = (int)blockIdx.y * 32 + t;                                                       \
    if (row >= n_pre) return;                                                                 \
    bool col_valid = (j < n_batch);                                                           \
    int  safe_j    = col_valid ? j : 0;                                                       \
    const int32_t*  i_row = indices + (size_t)row * n_conn;                                   \
    const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                   \
    ACC_T accum = (ACC_T)0;                                                                   \
    for (int k = 0; k < n_conn; k++) {                                                        \
        int  src    = __ldg(&i_row[k]);                                                       \
        bool active = col_valid && IS_ACTIVE(__ldg(&matrix[(size_t)src * n_batch + safe_j])); \
        if (active) accum += READ_W(__ldg(&w_row[k]));                                        \
    }                                                                                         \
    if (col_valid) output[(size_t)row * n_batch + j] = WRITE_W(accum);                        \
}

#define DEFINE_BGM_BASIC_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO)  \
__global__ void _bgm_basic_homo_kern##SUFFIX(                                                          \
    const int32_t* __restrict__ indices,                                                               \
    const SPIKE_T* __restrict__ matrix,                                                                \
    WEIGHT_T*      __restrict__ output,                                                                \
    const WEIGHT_T* __restrict__ weights,                                                              \
    int n_pre, int n_conn, int n_batch                                                                 \
) {                                                                                                    \
    extern __shared__ char _smem_bytes[];                                                              \
    int32_t* s_idx = reinterpret_cast<int32_t*>(_smem_bytes);                                          \
    int row = blockIdx.x;                                                                              \
    if (row >= n_pre) return;                                                                          \
    int lane   = threadIdx.x & 31;                                                                     \
    int warpid = threadIdx.x >> 5;                                                                     \
    int nwarps = blockDim.x >> 5;                                                                      \
    int j = (int)blockIdx.y * 32 + lane;                                                               \
    bool col_valid = (j < n_batch);                                                                    \
    int  safe_j    = col_valid ? j : 0;                                                                \
    const int32_t*  i_row = indices + (size_t)row * n_conn;                                            \
    for (int i = threadIdx.x; i < n_conn; i += blockDim.x) s_idx[i] = __ldg(&i_row[i]);                \
    __syncthreads();                                                                                   \
    ACC_T accum = ACC_ZERO;                                                                            \
    int k = warpid;                                                                                    \
    for (; k < n_conn; k += nwarps) {                                                                  \
        int src = s_idx[k];                                                                            \
        if (col_valid && IS_ACTIVE(__ldg(&matrix[(size_t)src * n_batch + safe_j]))) accum += (ACC_T)1; \
    }                                                                                                  \
    __syncthreads();                                                                                   \
    ACC_T* smem_red = reinterpret_cast<ACC_T*>(_smem_bytes);                                           \
    smem_red[warpid * 32 + lane] = accum;                                                              \
    __syncthreads();                                                                                   \
    if (warpid == 0) {                                                                                 \
        ACC_T sum = ACC_ZERO;                                                                          \
        for (int w = 0; w < nwarps; w++) sum += smem_red[w * 32 + lane];                               \
        if (col_valid) output[(size_t)row * n_batch + j] = WRITE_W(READ_W(__ldg(&weights[0])) * sum);  \
    }                                                                                                  \
}

#define DEFINE_BGM_BASIC_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _bgm_basic_hetero_kern##SUFFIX(                                                         \
    const int32_t* __restrict__ indices,                                                                \
    const SPIKE_T* __restrict__ matrix,                                                                 \
    WEIGHT_T*      __restrict__ output,                                                                 \
    const WEIGHT_T* __restrict__ weights,                                                               \
    int n_pre, int n_conn, int n_batch                                                                  \
) {                                                                                                     \
    extern __shared__ char _smem_bytes[];                                                               \
    int32_t* s_idx = reinterpret_cast<int32_t*>(_smem_bytes);                                           \
    int row = blockIdx.x;                                                                               \
    if (row >= n_pre) return;                                                                           \
    int lane   = threadIdx.x & 31;                                                                      \
    int warpid = threadIdx.x >> 5;                                                                      \
    int nwarps = blockDim.x >> 5;                                                                       \
    int j = (int)blockIdx.y * 32 + lane;                                                                \
    bool col_valid = (j < n_batch);                                                                     \
    int  safe_j    = col_valid ? j : 0;                                                                 \
    const int32_t*  i_row = indices + (size_t)row * n_conn;                                             \
    const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                             \
    for (int i = threadIdx.x; i < n_conn; i += blockDim.x) s_idx[i] = __ldg(&i_row[i]);                 \
    __syncthreads();                                                                                    \
    ACC_T accum = ACC_ZERO;                                                                             \
    int k = warpid;                                                                                     \
    for (; k < n_conn; k += nwarps) {                                                                   \
        int src = s_idx[k];                                                                             \
        if (col_valid && IS_ACTIVE(__ldg(&matrix[(size_t)src * n_batch + safe_j])))                     \
            accum += READ_W(__ldg(&w_row[k]));                                                          \
    }                                                                                                   \
    __syncthreads();                                                                                    \
    ACC_T* smem_red = reinterpret_cast<ACC_T*>(_smem_bytes);                                            \
    smem_red[warpid * 32 + lane] = accum;                                                               \
    __syncthreads();                                                                                    \
    if (warpid == 0) {                                                                                  \
        ACC_T sum = ACC_ZERO;                                                                           \
        for (int w = 0; w < nwarps; w++) sum += smem_red[w * 32 + lane];                                \
        if (col_valid) output[(size_t)row * n_batch + j] = WRITE_W(sum);                                \
    }                                                                                                   \
}

#define DEFINE_BSM_WARP_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _bsm_warp_homo_kern##SUFFIX(                                                    \
    const int32_t* __restrict__ indices,                                                        \
    const SPIKE_T* __restrict__ matrix,                                                         \
    WEIGHT_T*      __restrict__ output,                                                         \
    const WEIGHT_T* __restrict__ weights,                                                       \
    int n_pre, int n_conn, int n_batch                                                          \
) {                                                                                             \
    int row = blockIdx.x;                                                                       \
    int t   = threadIdx.x;                                                                      \
    int j   = (int)blockIdx.y * 32 + t;                                                         \
    if (row >= n_pre) return;                                                                   \
    bool col_valid = (j < n_batch);                                                             \
    int  safe_j    = col_valid ? j : 0;                                                         \
    bool active    = col_valid && IS_ACTIVE(__ldg(&matrix[(size_t)row * n_batch + safe_j]));    \
    if (__ballot_sync(0xffffffff, active) == 0) return;                                         \
    if (!active) return;                                                                        \
    const int32_t*  i_row = indices + (size_t)row * n_conn;                                     \
    ACC_T w0 = READ_W(__ldg(&weights[0]));                                                      \
    for (int k = 0; k < n_conn; k++)                                                            \
        ATOMIC_ADD_W(&output[(size_t)__ldg(&i_row[k]) * n_batch + j], w0);                      \
}

#define DEFINE_BSM_WARP_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _bsm_warp_hetero_kern##SUFFIX(                                                    \
    const int32_t* __restrict__ indices,                                                          \
    const SPIKE_T* __restrict__ matrix,                                                           \
    WEIGHT_T*      __restrict__ output,                                                           \
    const WEIGHT_T* __restrict__ weights,                                                         \
    int n_pre, int n_conn, int n_batch                                                            \
) {                                                                                               \
    int row = blockIdx.x;                                                                         \
    int t   = threadIdx.x;                                                                        \
    int j   = (int)blockIdx.y * 32 + t;                                                           \
    if (row >= n_pre) return;                                                                     \
    bool col_valid = (j < n_batch);                                                               \
    int  safe_j    = col_valid ? j : 0;                                                           \
    bool active    = col_valid && IS_ACTIVE(__ldg(&matrix[(size_t)row * n_batch + safe_j]));      \
    if (__ballot_sync(0xffffffff, active) == 0) return;                                           \
    if (!active) return;                                                                          \
    const int32_t*  i_row = indices + (size_t)row * n_conn;                                       \
    const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                       \
    for (int k = 0; k < n_conn; k++)                                                              \
        ATOMIC_ADD_W(&output[(size_t)__ldg(&i_row[k]) * n_batch + j], READ_W(__ldg(&w_row[k])));  \
}

#define DEFINE_BSM_BASIC_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _bsm_basic_homo_kern##SUFFIX(                                                    \
    const int32_t* __restrict__ indices,                                                         \
    const SPIKE_T* __restrict__ matrix,                                                          \
    WEIGHT_T*      __restrict__ output,                                                          \
    const WEIGHT_T* __restrict__ weights,                                                        \
    int n_pre, int n_conn, int n_batch                                                           \
) {                                                                                              \
    extern __shared__ int _smem_flag[];                                                          \
    int row = blockIdx.x;                                                                        \
    if (row >= n_pre) return;                                                                    \
    if (threadIdx.x == 0) _smem_flag[0] = 0;                                                     \
    __syncthreads();                                                                             \
    for (int j = threadIdx.x; j < n_batch; j += blockDim.x)                                      \
        if (IS_ACTIVE(__ldg(&matrix[(size_t)row * n_batch + j]))) {                              \
            atomicOr(_smem_flag, 1); break;                                                      \
        }                                                                                        \
    __syncthreads();                                                                             \
    if (_smem_flag[0] == 0) return;                                                              \
    const int32_t*  i_row = indices + (size_t)row * n_conn;                                      \
    ACC_T w0 = READ_W(__ldg(&weights[0]));                                                       \
    for (int j = 0; j < n_batch; j++) {                                                          \
        if (!IS_ACTIVE(__ldg(&matrix[(size_t)row * n_batch + j]))) continue;                     \
        for (int k = threadIdx.x; k < n_conn; k += blockDim.x)                                   \
            ATOMIC_ADD_W(&output[(size_t)__ldg(&i_row[k]) * n_batch + j], w0);                   \
    }                                                                                            \
}

#define DEFINE_BSM_BASIC_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)   \
__global__ void _bsm_basic_hetero_kern##SUFFIX(                                                      \
    const int32_t* __restrict__ indices,                                                             \
    const SPIKE_T* __restrict__ matrix,                                                              \
    WEIGHT_T*      __restrict__ output,                                                              \
    const WEIGHT_T* __restrict__ weights,                                                            \
    int n_pre, int n_conn, int n_batch                                                               \
) {                                                                                                  \
    extern __shared__ int _smem_flag[];                                                              \
    int row = blockIdx.x;                                                                            \
    if (row >= n_pre) return;                                                                        \
    if (threadIdx.x == 0) _smem_flag[0] = 0;                                                         \
    __syncthreads();                                                                                 \
    for (int j = threadIdx.x; j < n_batch; j += blockDim.x)                                          \
        if (IS_ACTIVE(__ldg(&matrix[(size_t)row * n_batch + j]))) {                                  \
            atomicOr(_smem_flag, 1); break;                                                          \
        }                                                                                            \
    __syncthreads();                                                                                 \
    if (_smem_flag[0] == 0) return;                                                                  \
    const int32_t*  i_row = indices + (size_t)row * n_conn;                                          \
    const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                          \
    for (int j = 0; j < n_batch; j++) {                                                              \
        if (!IS_ACTIVE(__ldg(&matrix[(size_t)row * n_batch + j]))) continue;                         \
        for (int k = threadIdx.x; k < n_conn; k += blockDim.x)                                       \
            ATOMIC_ADD_W(&output[(size_t)__ldg(&i_row[k]) * n_batch + j], READ_W(__ldg(&w_row[k]))); \
    }                                                                                                \
}

// Instantiations
// ---- float32 ----
DEFINE_BGM_WARP_HOMO   (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32)
DEFINE_BGM_WARP_HETERO (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32)
DEFINE_BGM_WARP_HOMO   (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32)
DEFINE_BGM_WARP_HETERO (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32)
DEFINE_BGM_BASIC_HOMO  (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_BGM_BASIC_HETERO(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_BGM_BASIC_HOMO  (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_BGM_BASIC_HETERO(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_BSM_WARP_HOMO   (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, atomicAdd)
DEFINE_BSM_WARP_HETERO (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, atomicAdd)
DEFINE_BSM_WARP_HOMO   (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, atomicAdd)
DEFINE_BSM_WARP_HETERO (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, atomicAdd)
DEFINE_BSM_BASIC_HOMO  (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, atomicAdd)
DEFINE_BSM_BASIC_HETERO(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, atomicAdd)
DEFINE_BSM_BASIC_HOMO  (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, atomicAdd)
DEFINE_BSM_BASIC_HETERO(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, atomicAdd)

// ---- float64 ----
DEFINE_BGM_WARP_HOMO   (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64)
DEFINE_BGM_WARP_HETERO (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64)
DEFINE_BGM_WARP_HOMO   (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64)
DEFINE_BGM_WARP_HETERO (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64)
DEFINE_BGM_BASIC_HOMO  (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_BGM_BASIC_HETERO(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_BGM_BASIC_HOMO  (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_BGM_BASIC_HETERO(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_BSM_WARP_HOMO   (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, atomic_add_f64)
DEFINE_BSM_WARP_HETERO (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, atomic_add_f64)
DEFINE_BSM_WARP_HOMO   (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, atomic_add_f64)
DEFINE_BSM_WARP_HETERO (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, atomic_add_f64)
DEFINE_BSM_BASIC_HOMO  (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, atomic_add_f64)
DEFINE_BSM_BASIC_HETERO(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, atomic_add_f64)
DEFINE_BSM_BASIC_HOMO  (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, atomic_add_f64)
DEFINE_BSM_BASIC_HETERO(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, atomic_add_f64)

// ---- float16 ----
DEFINE_BGM_WARP_HOMO   (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16)
DEFINE_BGM_WARP_HETERO (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16)
DEFINE_BGM_WARP_HOMO   (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16)
DEFINE_BGM_WARP_HETERO (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16)
DEFINE_BGM_BASIC_HOMO  (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_BGM_BASIC_HETERO(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_BGM_BASIC_HOMO  (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_BGM_BASIC_HETERO(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_BSM_WARP_HOMO   (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, atomic_add_f16)
DEFINE_BSM_WARP_HETERO (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, atomic_add_f16)
DEFINE_BSM_WARP_HOMO   (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, atomic_add_f16)
DEFINE_BSM_WARP_HETERO (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, atomic_add_f16)
DEFINE_BSM_BASIC_HOMO  (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, atomic_add_f16)
DEFINE_BSM_BASIC_HETERO(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, atomic_add_f16)
DEFINE_BSM_BASIC_HOMO  (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, atomic_add_f16)
DEFINE_BSM_BASIC_HETERO(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, atomic_add_f16)

// ---- bfloat16 ----
DEFINE_BGM_WARP_HOMO   (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16)
DEFINE_BGM_WARP_HETERO (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16)
DEFINE_BGM_WARP_HOMO   (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16)
DEFINE_BGM_WARP_HETERO (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16)
DEFINE_BGM_BASIC_HOMO  (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_BGM_BASIC_HETERO(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_BGM_BASIC_HOMO  (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_BGM_BASIC_HETERO(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_BSM_WARP_HOMO   (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BSM_WARP_HETERO (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BSM_WARP_HOMO   (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BSM_WARP_HETERO (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BSM_BASIC_HOMO  (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BSM_BASIC_HETERO(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BSM_BASIC_HOMO  (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BSM_BASIC_HETERO(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)

// FFI Macros for SpMM
// ---- FFI macro: gather homo warp ----
#define FFI_BGM_HOMO_WARP(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                               \
void binary_fcnmm_gather_homo_warp##SUFFIX(                                                            \
    const BE::Tensor weights, const BE::Tensor indices,                                                \
    const BE::Tensor matrix,  BE::Tensor output, int64_t stream                                        \
) {                                                                                                    \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                           \
    int n_pre   = static_cast<int>(indices.size(0));                                                   \
    int n_conn  = static_cast<int>(indices.size(1));                                                   \
    int n_batch = static_cast<int>(matrix.size(1));                                                    \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                      \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                         \
    const SPIKE_C_T*  d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr());                        \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                             \
    int batch_tiles = (n_batch + 31) / 32;                                                             \
    dim3 grid(n_pre, batch_tiles);                                                                     \
    _bgm_warp_homo_kern##SUFFIX<<<grid, 32, 0, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch); \
}

// ---- FFI macro: gather hetero warp ----
#define FFI_BGM_HETERO_WARP(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                               \
void binary_fcnmm_gather_hetero_warp##SUFFIX(                                                            \
    const BE::Tensor weights, const BE::Tensor indices,                                                  \
    const BE::Tensor matrix,  BE::Tensor output, int64_t stream                                          \
) {                                                                                                      \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                             \
    int n_pre   = static_cast<int>(indices.size(0));                                                     \
    int n_conn  = static_cast<int>(indices.size(1));                                                     \
    int n_batch = static_cast<int>(matrix.size(1));                                                      \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                        \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                           \
    const SPIKE_C_T*  d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr());                          \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                               \
    int batch_tiles = (n_batch + 31) / 32;                                                               \
    dim3 grid(n_pre, batch_tiles);                                                                       \
    _bgm_warp_hetero_kern##SUFFIX<<<grid, 32, 0, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch); \
}

// ---- FFI macro: gather homo basic ----
#define FFI_BGM_HOMO_BASIC(SUFFIX, WEIGHT_C_T, SPIKE_C_T, ACC_SIZE)                                        \
void binary_fcnmm_gather_homo_basic##SUFFIX(                                                               \
    const BE::Tensor weights, const BE::Tensor indices,                                                    \
    const BE::Tensor matrix,  BE::Tensor output, int64_t stream                                            \
) {                                                                                                        \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                               \
    int n_pre   = static_cast<int>(indices.size(0));                                                       \
    int n_conn  = static_cast<int>(indices.size(1));                                                       \
    int n_batch = static_cast<int>(matrix.size(1));                                                        \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                          \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                             \
    const SPIKE_C_T*  d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr());                            \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                                 \
    int bsz = 256; int nwarps = bsz >> 5;                                                                  \
    size_t idx_bytes = (size_t)n_conn * sizeof(int32_t);                                                   \
    size_t red_bytes = (size_t)nwarps * 32 * ACC_SIZE;                                                     \
    size_t shm = (idx_bytes > red_bytes) ? idx_bytes : red_bytes;                                          \
    int batch_tiles = (n_batch + 31) / 32; dim3 grid(n_pre, batch_tiles);                                  \
    _bgm_basic_homo_kern##SUFFIX<<<grid, bsz, shm, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch); \
}

// ---- FFI macro: gather hetero basic ----
#define FFI_BGM_HETERO_BASIC(SUFFIX, WEIGHT_C_T, SPIKE_C_T, ACC_SIZE)                                        \
void binary_fcnmm_gather_hetero_basic##SUFFIX(                                                               \
    const BE::Tensor weights, const BE::Tensor indices,                                                      \
    const BE::Tensor matrix,  BE::Tensor output, int64_t stream                                              \
) {                                                                                                          \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                                 \
    int n_pre   = static_cast<int>(indices.size(0));                                                         \
    int n_conn  = static_cast<int>(indices.size(1));                                                         \
    int n_batch = static_cast<int>(matrix.size(1));                                                          \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                            \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                               \
    const SPIKE_C_T*  d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr());                              \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                                   \
    int bsz = 256; int nwarps = bsz >> 5;                                                                    \
    size_t idx_bytes = (size_t)n_conn * sizeof(int32_t);                                                     \
    size_t red_bytes = (size_t)nwarps * 32 * ACC_SIZE;                                                       \
    size_t shm = (idx_bytes > red_bytes) ? idx_bytes : red_bytes;                                            \
    int batch_tiles = (n_batch + 31) / 32; dim3 grid(n_pre, batch_tiles);                                    \
    _bgm_basic_hetero_kern##SUFFIX<<<grid, bsz, shm, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch); \
}

// ---- FFI macro: scatter homo warp ----
#define FFI_BSM_HOMO_WARP(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                               \
void binary_fcnmm_scatter_homo_warp##SUFFIX(                                                           \
    const BE::Tensor weights, const BE::Tensor indices,                                                \
    const BE::Tensor matrix,  BE::Tensor output, int64_t stream                                        \
) {                                                                                                    \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                           \
    int n_pre   = static_cast<int>(indices.size(0));                                                   \
    int n_conn  = static_cast<int>(indices.size(1));                                                   \
    int n_post  = static_cast<int>(output.size(0));                                                    \
    int n_batch = static_cast<int>(matrix.size(1));                                                    \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                      \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                         \
    const SPIKE_C_T*  d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr());                        \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                             \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_batch * sizeof(WEIGHT_C_T), s);                       \
    int batch_tiles = (n_batch + 31) / 32; dim3 grid(n_pre, batch_tiles);                              \
    _bsm_warp_homo_kern##SUFFIX<<<grid, 32, 0, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch); \
}

// ---- FFI macro: scatter hetero warp ----
#define FFI_BSM_HETERO_WARP(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                               \
void binary_fcnmm_scatter_hetero_warp##SUFFIX(                                                           \
    const BE::Tensor weights, const BE::Tensor indices,                                                  \
    const BE::Tensor matrix,  BE::Tensor output, int64_t stream                                          \
) {                                                                                                      \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                             \
    int n_pre   = static_cast<int>(indices.size(0));                                                     \
    int n_conn  = static_cast<int>(indices.size(1));                                                     \
    int n_post  = static_cast<int>(output.size(0));                                                      \
    int n_batch = static_cast<int>(matrix.size(1));                                                      \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                        \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                           \
    const SPIKE_C_T*  d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr());                          \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                               \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_batch * sizeof(WEIGHT_C_T), s);                         \
    int batch_tiles = (n_batch + 31) / 32; dim3 grid(n_pre, batch_tiles);                                \
    _bsm_warp_hetero_kern##SUFFIX<<<grid, 32, 0, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch); \
}

// ---- FFI macro: scatter homo basic ----
#define FFI_BSM_HOMO_BASIC(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                                           \
void binary_fcnmm_scatter_homo_basic##SUFFIX(                                                                       \
    const BE::Tensor weights, const BE::Tensor indices,                                                             \
    const BE::Tensor matrix,  BE::Tensor output, int64_t stream                                                     \
) {                                                                                                                 \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                                        \
    int n_pre   = static_cast<int>(indices.size(0));                                                                \
    int n_conn  = static_cast<int>(indices.size(1));                                                                \
    int n_post  = static_cast<int>(output.size(0));                                                                 \
    int n_batch = static_cast<int>(matrix.size(1));                                                                 \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                                   \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                                      \
    const SPIKE_C_T*  d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr());                                     \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                                          \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_batch * sizeof(WEIGHT_C_T), s);                                    \
    _bsm_basic_homo_kern##SUFFIX<<<n_pre, 256, sizeof(int), s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch); \
}

// ---- FFI macro: scatter hetero basic ----
#define FFI_BSM_HETERO_BASIC(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                                           \
void binary_fcnmm_scatter_hetero_basic##SUFFIX(                                                                       \
    const BE::Tensor weights, const BE::Tensor indices,                                                               \
    const BE::Tensor matrix,  BE::Tensor output, int64_t stream                                                       \
) {                                                                                                                   \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                                          \
    int n_pre   = static_cast<int>(indices.size(0));                                                                  \
    int n_conn  = static_cast<int>(indices.size(1));                                                                  \
    int n_post  = static_cast<int>(output.size(0));                                                                   \
    int n_batch = static_cast<int>(matrix.size(1));                                                                   \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                                     \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                                        \
    const SPIKE_C_T*  d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr());                                       \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                                            \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_batch * sizeof(WEIGHT_C_T), s);                                      \
    _bsm_basic_hetero_kern##SUFFIX<<<n_pre, 256, sizeof(int), s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch); \
}

// SpMM FFI Instantiations
// ---- float32 ----
// @BE binary_fcnmm_gather_homo_warp_bool_f32
FFI_BGM_HOMO_WARP  (_bool_f32, float, uint8_t)
// @BE binary_fcnmm_gather_hetero_warp_bool_f32
FFI_BGM_HETERO_WARP(_bool_f32, float, uint8_t)
// @BE binary_fcnmm_gather_homo_warp_float_f32
FFI_BGM_HOMO_WARP  (_float_f32, float, float)
// @BE binary_fcnmm_gather_hetero_warp_float_f32
FFI_BGM_HETERO_WARP(_float_f32, float, float)
// @BE binary_fcnmm_gather_homo_basic_bool_f32
FFI_BGM_HOMO_BASIC (_bool_f32, float, uint8_t, sizeof(float))
// @BE binary_fcnmm_gather_hetero_basic_bool_f32
FFI_BGM_HETERO_BASIC(_bool_f32, float, uint8_t, sizeof(float))
// @BE binary_fcnmm_gather_homo_basic_float_f32
FFI_BGM_HOMO_BASIC (_float_f32, float, float, sizeof(float))
// @BE binary_fcnmm_gather_hetero_basic_float_f32
FFI_BGM_HETERO_BASIC(_float_f32, float, float, sizeof(float))
// @BE binary_fcnmm_scatter_homo_warp_bool_f32
FFI_BSM_HOMO_WARP  (_bool_f32, float, uint8_t)
// @BE binary_fcnmm_scatter_hetero_warp_bool_f32
FFI_BSM_HETERO_WARP(_bool_f32, float, uint8_t)
// @BE binary_fcnmm_scatter_homo_warp_float_f32
FFI_BSM_HOMO_WARP  (_float_f32, float, float)
// @BE binary_fcnmm_scatter_hetero_warp_float_f32
FFI_BSM_HETERO_WARP(_float_f32, float, float)
// @BE binary_fcnmm_scatter_homo_basic_bool_f32
FFI_BSM_HOMO_BASIC (_bool_f32, float, uint8_t)
// @BE binary_fcnmm_scatter_hetero_basic_bool_f32
FFI_BSM_HETERO_BASIC(_bool_f32, float, uint8_t)
// @BE binary_fcnmm_scatter_homo_basic_float_f32
FFI_BSM_HOMO_BASIC (_float_f32, float, float)
// @BE binary_fcnmm_scatter_hetero_basic_float_f32
FFI_BSM_HETERO_BASIC(_float_f32, float, float)

// ---- float64 ----
// @BE binary_fcnmm_gather_homo_warp_bool_f64
FFI_BGM_HOMO_WARP  (_bool_f64, double, uint8_t)
// @BE binary_fcnmm_gather_hetero_warp_bool_f64
FFI_BGM_HETERO_WARP(_bool_f64, double, uint8_t)
// @BE binary_fcnmm_gather_homo_warp_float_f64
FFI_BGM_HOMO_WARP  (_float_f64, double, double)
// @BE binary_fcnmm_gather_hetero_warp_float_f64
FFI_BGM_HETERO_WARP(_float_f64, double, double)
// @BE binary_fcnmm_gather_homo_basic_bool_f64
FFI_BGM_HOMO_BASIC (_bool_f64, double, uint8_t, sizeof(double))
// @BE binary_fcnmm_gather_hetero_basic_bool_f64
FFI_BGM_HETERO_BASIC(_bool_f64, double, uint8_t, sizeof(double))
// @BE binary_fcnmm_gather_homo_basic_float_f64
FFI_BGM_HOMO_BASIC (_float_f64, double, double, sizeof(double))
// @BE binary_fcnmm_gather_hetero_basic_float_f64
FFI_BGM_HETERO_BASIC(_float_f64, double, double, sizeof(double))
// @BE binary_fcnmm_scatter_homo_warp_bool_f64
FFI_BSM_HOMO_WARP  (_bool_f64, double, uint8_t)
// @BE binary_fcnmm_scatter_hetero_warp_bool_f64
FFI_BSM_HETERO_WARP(_bool_f64, double, uint8_t)
// @BE binary_fcnmm_scatter_homo_warp_float_f64
FFI_BSM_HOMO_WARP  (_float_f64, double, double)
// @BE binary_fcnmm_scatter_hetero_warp_float_f64
FFI_BSM_HETERO_WARP(_float_f64, double, double)
// @BE binary_fcnmm_scatter_homo_basic_bool_f64
FFI_BSM_HOMO_BASIC (_bool_f64, double, uint8_t)
// @BE binary_fcnmm_scatter_hetero_basic_bool_f64
FFI_BSM_HETERO_BASIC(_bool_f64, double, uint8_t)
// @BE binary_fcnmm_scatter_homo_basic_float_f64
FFI_BSM_HOMO_BASIC (_float_f64, double, double)
// @BE binary_fcnmm_scatter_hetero_basic_float_f64
FFI_BSM_HETERO_BASIC(_float_f64, double, double)

// ---- float16 ----
// @BE binary_fcnmm_gather_homo_warp_bool_f16
FFI_BGM_HOMO_WARP  (_bool_f16, __half, uint8_t)
// @BE binary_fcnmm_gather_hetero_warp_bool_f16
FFI_BGM_HETERO_WARP(_bool_f16, __half, uint8_t)
// @BE binary_fcnmm_gather_homo_warp_float_f16
FFI_BGM_HOMO_WARP  (_float_f16, __half, __half)
// @BE binary_fcnmm_gather_hetero_warp_float_f16
FFI_BGM_HETERO_WARP(_float_f16, __half, __half)
// @BE binary_fcnmm_gather_homo_basic_bool_f16
FFI_BGM_HOMO_BASIC (_bool_f16, __half, uint8_t, sizeof(float))
// @BE binary_fcnmm_gather_hetero_basic_bool_f16
FFI_BGM_HETERO_BASIC(_bool_f16, __half, uint8_t, sizeof(float))
// @BE binary_fcnmm_gather_homo_basic_float_f16
FFI_BGM_HOMO_BASIC (_float_f16, __half, __half, sizeof(float))
// @BE binary_fcnmm_gather_hetero_basic_float_f16
FFI_BGM_HETERO_BASIC(_float_f16, __half, __half, sizeof(float))
// @BE binary_fcnmm_scatter_homo_warp_bool_f16
FFI_BSM_HOMO_WARP  (_bool_f16, __half, uint8_t)
// @BE binary_fcnmm_scatter_hetero_warp_bool_f16
FFI_BSM_HETERO_WARP(_bool_f16, __half, uint8_t)
// @BE binary_fcnmm_scatter_homo_warp_float_f16
FFI_BSM_HOMO_WARP  (_float_f16, __half, __half)
// @BE binary_fcnmm_scatter_hetero_warp_float_f16
FFI_BSM_HETERO_WARP(_float_f16, __half, __half)
// @BE binary_fcnmm_scatter_homo_basic_bool_f16
FFI_BSM_HOMO_BASIC (_bool_f16, __half, uint8_t)
// @BE binary_fcnmm_scatter_hetero_basic_bool_f16
FFI_BSM_HETERO_BASIC(_bool_f16, __half, uint8_t)
// @BE binary_fcnmm_scatter_homo_basic_float_f16
FFI_BSM_HOMO_BASIC (_float_f16, __half, __half)
// @BE binary_fcnmm_scatter_hetero_basic_float_f16
FFI_BSM_HETERO_BASIC(_float_f16, __half, __half)

// ---- bfloat16 ----
// @BE binary_fcnmm_gather_homo_warp_bool_bf16
FFI_BGM_HOMO_WARP  (_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmm_gather_hetero_warp_bool_bf16
FFI_BGM_HETERO_WARP(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmm_gather_homo_warp_float_bf16
FFI_BGM_HOMO_WARP  (_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmm_gather_hetero_warp_float_bf16
FFI_BGM_HETERO_WARP(_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmm_gather_homo_basic_bool_bf16
FFI_BGM_HOMO_BASIC (_bool_bf16, __nv_bfloat16, uint8_t, sizeof(float))
// @BE binary_fcnmm_gather_hetero_basic_bool_bf16
FFI_BGM_HETERO_BASIC(_bool_bf16, __nv_bfloat16, uint8_t, sizeof(float))
// @BE binary_fcnmm_gather_homo_basic_float_bf16
FFI_BGM_HOMO_BASIC (_float_bf16, __nv_bfloat16, __nv_bfloat16, sizeof(float))
// @BE binary_fcnmm_gather_hetero_basic_float_bf16
FFI_BGM_HETERO_BASIC(_float_bf16, __nv_bfloat16, __nv_bfloat16, sizeof(float))
// @BE binary_fcnmm_scatter_homo_warp_bool_bf16
FFI_BSM_HOMO_WARP  (_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmm_scatter_hetero_warp_bool_bf16
FFI_BSM_HETERO_WARP(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmm_scatter_homo_warp_float_bf16
FFI_BSM_HOMO_WARP  (_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmm_scatter_hetero_warp_float_bf16
FFI_BSM_HETERO_WARP(_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmm_scatter_homo_basic_bool_bf16
FFI_BSM_HOMO_BASIC (_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmm_scatter_hetero_basic_bool_bf16
FFI_BSM_HETERO_BASIC(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmm_scatter_homo_basic_float_bf16
FFI_BSM_HOMO_BASIC (_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmm_scatter_hetero_basic_float_bf16
FFI_BSM_HETERO_BASIC(_float_bf16, __nv_bfloat16, __nv_bfloat16)


// --------------------------------------------------------------------------------------------------------
//UNBRANCH

#define DEFINE_BGM_BASIC_HOMO_UNBRANCH(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _bgm_basic_homo_kern_unbranch##SUFFIX(                                                 \
    const int32_t* __restrict__ indices,                                                               \
    const SPIKE_T* __restrict__ matrix,                                                                \
    WEIGHT_T* __restrict__ output,                                                                     \
    const WEIGHT_T* __restrict__ weights,                                                              \
    int n_pre, int n_conn, int n_batch                                                                 \
) {                                                                                                    \
    extern __shared__ char _smem_bytes[];                                                              \
    int32_t* s_idx = reinterpret_cast<int32_t*>(_smem_bytes);                                          \
    int row = blockIdx.x;                                                                              \
    if (row >= n_pre) return;                                                                          \
                                                                                                       \
    int lane   = threadIdx.x & 31;                                                                     \
    int warpid = threadIdx.x >> 5;                                                                     \
    int nwarps = blockDim.x >> 5;                                                                      \
    int j = (int)blockIdx.y * 32 + lane;                                                               \
    bool col_valid = (j < n_batch);                                                                    \
    int  safe_j    = col_valid ? j : 0;                                                                \
                                                                                                       \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                            \
    for (int i = threadIdx.x; i < n_conn; i += blockDim.x) s_idx[i] = __ldg(&i_row[i]);                \
    __syncthreads();                                                                                   \
                                                                                                       \
    ACC_T accum = ACC_ZERO;                                                                            \
                                                                                                       \
    int main_iters = n_conn / nwarps;                                                                  \
    int tail = n_conn % nwarps;                                                                        \
    int limit = main_iters & ~1;                                                                       \
    int k = warpid;                                                                                    \
                                                                                                       \
    for (int i = 0; i < limit; i += 2) {                                                               \
        int src0 = s_idx[k];                                                                           \
        int src1 = s_idx[k + nwarps];                                                                  \
        if (col_valid && IS_ACTIVE(__ldg(&matrix[(size_t)src0 * n_batch + safe_j]))) accum += (ACC_T)1;\
        if (col_valid && IS_ACTIVE(__ldg(&matrix[(size_t)src1 * n_batch + safe_j]))) accum += (ACC_T)1;\
        k += (nwarps << 1);                                                                            \
    }                                                                                                  \
    if (main_iters & 1) {                                                                              \
        int src = s_idx[k];                                                                            \
        if (col_valid && IS_ACTIVE(__ldg(&matrix[(size_t)src * n_batch + safe_j]))) accum += (ACC_T)1; \
        k += nwarps;                                                                                   \
    }                                                                                                  \
    if (warpid < tail) {                                                                               \
        int src = s_idx[k];                                                                            \
        if (col_valid && IS_ACTIVE(__ldg(&matrix[(size_t)src * n_batch + safe_j]))) accum += (ACC_T)1; \
    }                                                                                                  \
                                                                                                       \
    __syncthreads();                                                                                   \
    ACC_T* smem_red = reinterpret_cast<ACC_T*>(_smem_bytes);                                           \
    smem_red[warpid * 32 + lane] = accum;                                                              \
    __syncthreads();                                                                                   \
                                                                                                       \
    if (warpid == 0) {                                                                                 \
        ACC_T sum = ACC_ZERO;                                                                          \
        for (int w = 0; w < nwarps; w++) sum += smem_red[w * 32 + lane];                               \
        if (col_valid) output[(size_t)row * n_batch + j] = WRITE_W(READ_W(__ldg(&weights[0])) * sum);  \
    }                                                                                                  \
}

#define DEFINE_BGM_BASIC_HETERO_UNBRANCH(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _bgm_basic_hetero_kern_unbranch##SUFFIX(                                               \
    const int32_t* __restrict__ indices,                                                               \
    const SPIKE_T* __restrict__ matrix,                                                                \
    WEIGHT_T* __restrict__ output,                                                                     \
    const WEIGHT_T* __restrict__ weights,                                                              \
    int n_pre, int n_conn, int n_batch                                                                 \
) {                                                                                                    \
    extern __shared__ char _smem_bytes[];                                                              \
    int32_t* s_idx = reinterpret_cast<int32_t*>(_smem_bytes);                                          \
    int row = blockIdx.x;                                                                              \
    if (row >= n_pre) return;                                                                          \
                                                                                                       \
    int lane   = threadIdx.x & 31;                                                                     \
    int warpid = threadIdx.x >> 5;                                                                     \
    int nwarps = blockDim.x >> 5;                                                                      \
    int j = (int)blockIdx.y * 32 + lane;                                                               \
    bool col_valid = (j < n_batch);                                                                    \
    int  safe_j    = col_valid ? j : 0;                                                                \
                                                                                                       \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                            \
    const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                            \
    for (int i = threadIdx.x; i < n_conn; i += blockDim.x) s_idx[i] = __ldg(&i_row[i]);                \
    __syncthreads();                                                                                   \
                                                                                                       \
    ACC_T accum = ACC_ZERO;                                                                            \
                                                                                                       \
    int main_iters = n_conn / nwarps;                                                                  \
    int tail = n_conn % nwarps;                                                                        \
    int limit = main_iters & ~1;                                                                       \
    int k = warpid;                                                                                    \
                                                                                                       \
    for (int i = 0; i < limit; i += 2) {                                                               \
        int src0 = s_idx[k];                                                                           \
        int src1 = s_idx[k + nwarps];                                                                  \
        if (col_valid && IS_ACTIVE(__ldg(&matrix[(size_t)src0 * n_batch + safe_j])))                   \
            accum += READ_W(__ldg(&w_row[k]));                                                         \
        if (col_valid && IS_ACTIVE(__ldg(&matrix[(size_t)src1 * n_batch + safe_j])))                   \
            accum += READ_W(__ldg(&w_row[k + nwarps]));                                                \
        k += (nwarps << 1);                                                                            \
    }                                                                                                  \
    if (main_iters & 1) {                                                                              \
        int src = s_idx[k];                                                                            \
        if (col_valid && IS_ACTIVE(__ldg(&matrix[(size_t)src * n_batch + safe_j])))                   \
            accum += READ_W(__ldg(&w_row[k]));                                                         \
        k += nwarps;                                                                                   \
    }                                                                                                  \
    if (warpid < tail) {                                                                               \
        int src = s_idx[k];                                                                            \
        if (col_valid && IS_ACTIVE(__ldg(&matrix[(size_t)src * n_batch + safe_j])))                   \
            accum += READ_W(__ldg(&w_row[k]));                                                         \
    }                                                                                                  \
                                                                                                       \
    __syncthreads();                                                                                   \
    ACC_T* smem_red = reinterpret_cast<ACC_T*>(_smem_bytes);                                           \
    smem_red[warpid * 32 + lane] = accum;                                                              \
    __syncthreads();                                                                                   \
                                                                                                       \
    if (warpid == 0) {                                                                                 \
        ACC_T sum = ACC_ZERO;                                                                          \
        for (int w = 0; w < nwarps; w++) sum += smem_red[w * 32 + lane];                               \
        if (col_valid) output[(size_t)row * n_batch + j] = WRITE_W(sum);                               \
    }                                                                                                  \
}

// ---- FFI宏: gather homo unbranch ----
#define FFI_BGM_HOMO_UNBRANCH(SUFFIX, WEIGHT_C_T, SPIKE_C_T, ACC_SIZE) \
void binary_fcnmm_gather_homo_unbranch##SUFFIX( \
    const BE::Tensor weights, const BE::Tensor indices, \
    const BE::Tensor matrix,  BE::Tensor output, int64_t stream \
) { \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream); \
    int n_pre   = static_cast<int>(indices.size(0)); \
    int n_conn  = static_cast<int>(indices.size(1)); \
    int n_batch = static_cast<int>(matrix.size(1)); \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr()); \
    const SPIKE_C_T*  d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr()); \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr()); \
    int bsz = 256; int nwarps = bsz >> 5; \
    size_t idx_bytes = (size_t)n_conn * sizeof(int32_t); \
    size_t red_bytes = (size_t)nwarps * 32 * ACC_SIZE; \
    size_t shm = (idx_bytes > red_bytes) ? idx_bytes : red_bytes; \
    int batch_tiles = (n_batch + 31) / 32; dim3 grid(n_pre, batch_tiles); \
    _bgm_basic_homo_kern_unbranch##SUFFIX<<<grid, bsz, shm, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch); \
}

// ---- FFI宏: gather hetero unbranch ----
#define FFI_BGM_HETERO_UNBRANCH(SUFFIX, WEIGHT_C_T, SPIKE_C_T, ACC_SIZE) \
void binary_fcnmm_gather_hetero_unbranch##SUFFIX( \
    const BE::Tensor weights, const BE::Tensor indices, \
    const BE::Tensor matrix,  BE::Tensor output, int64_t stream \
) { \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream); \
    int n_pre   = static_cast<int>(indices.size(0)); \
    int n_conn  = static_cast<int>(indices.size(1)); \
    int n_batch = static_cast<int>(matrix.size(1)); \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr()); \
    const SPIKE_C_T*  d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr()); \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr()); \
    int bsz = 256; int nwarps = bsz >> 5; \
    size_t idx_bytes = (size_t)n_conn * sizeof(int32_t); \
    size_t red_bytes = (size_t)nwarps * 32 * ACC_SIZE; \
    size_t shm = (idx_bytes > red_bytes) ? idx_bytes : red_bytes; \
    int batch_tiles = (n_batch + 31) / 32; dim3 grid(n_pre, batch_tiles); \
    _bgm_basic_hetero_kern_unbranch##SUFFIX<<<grid, bsz, shm, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch); \
}

DEFINE_BGM_BASIC_HOMO_UNBRANCH  (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_BGM_BASIC_HETERO_UNBRANCH(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_BGM_BASIC_HOMO_UNBRANCH  (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_BGM_BASIC_HETERO_UNBRANCH(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, 0.0f)

// ---- float64 ----
DEFINE_BGM_BASIC_HOMO_UNBRANCH  (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_BGM_BASIC_HETERO_UNBRANCH(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_BGM_BASIC_HOMO_UNBRANCH  (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_BGM_BASIC_HETERO_UNBRANCH(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, 0.0)

// ---- float16 ----
DEFINE_BGM_BASIC_HOMO_UNBRANCH  (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_BGM_BASIC_HETERO_UNBRANCH(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_BGM_BASIC_HOMO_UNBRANCH  (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_BGM_BASIC_HETERO_UNBRANCH(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, 0.0f)

// ---- bfloat16 ----
DEFINE_BGM_BASIC_HOMO_UNBRANCH  (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_BGM_BASIC_HETERO_UNBRANCH(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_BGM_BASIC_HOMO_UNBRANCH  (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_BGM_BASIC_HETERO_UNBRANCH(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)

// ---- Unbranch FFI实例化 ----
// ---- float32 ----
// @BE binary_fcnmm_gather_homo_unbranch_bool_f32
FFI_BGM_HOMO_UNBRANCH  (_bool_f32, float, uint8_t, sizeof(float))
// @BE binary_fcnmm_gather_hetero_unbranch_bool_f32
FFI_BGM_HETERO_UNBRANCH(_bool_f32, float, uint8_t, sizeof(float))
// @BE binary_fcnmm_gather_homo_unbranch_float_f32
FFI_BGM_HOMO_UNBRANCH  (_float_f32, float, float, sizeof(float))
// @BE binary_fcnmm_gather_hetero_unbranch_float_f32
FFI_BGM_HETERO_UNBRANCH(_float_f32, float, float, sizeof(float))
// ---- float64 ----
// @BE binary_fcnmm_gather_homo_unbranch_bool_f64
FFI_BGM_HOMO_UNBRANCH  (_bool_f64, double, uint8_t, sizeof(double))
// @BE binary_fcnmm_gather_hetero_unbranch_bool_f64
FFI_BGM_HETERO_UNBRANCH(_bool_f64, double, uint8_t, sizeof(double))
// @BE binary_fcnmm_gather_homo_unbranch_float_f64
FFI_BGM_HOMO_UNBRANCH  (_float_f64, double, double, sizeof(double))
// @BE binary_fcnmm_gather_hetero_unbranch_float_f64
FFI_BGM_HETERO_UNBRANCH(_float_f64, double, double, sizeof(double))
// ---- float16 ----
// @BE binary_fcnmm_gather_homo_unbranch_bool_f16
FFI_BGM_HOMO_UNBRANCH  (_bool_f16, __half, uint8_t, sizeof(float))
// @BE binary_fcnmm_gather_hetero_unbranch_bool_f16
FFI_BGM_HETERO_UNBRANCH(_bool_f16, __half, uint8_t, sizeof(float))
// @BE binary_fcnmm_gather_homo_unbranch_float_f16
FFI_BGM_HOMO_UNBRANCH  (_float_f16, __half, __half, sizeof(float))
// @BE binary_fcnmm_gather_hetero_unbranch_float_f16
FFI_BGM_HETERO_UNBRANCH(_float_f16, __half, __half, sizeof(float))
// ---- bfloat16 ----
// @BE binary_fcnmm_gather_homo_unbranch_bool_bf16
FFI_BGM_HOMO_UNBRANCH  (_bool_bf16, __nv_bfloat16, uint8_t, sizeof(float))
// @BE binary_fcnmm_gather_hetero_unbranch_bool_bf16
FFI_BGM_HETERO_UNBRANCH(_bool_bf16, __nv_bfloat16, uint8_t, sizeof(float))
// @BE binary_fcnmm_gather_homo_unbranch_float_bf16
FFI_BGM_HOMO_UNBRANCH  (_float_bf16, __nv_bfloat16, __nv_bfloat16, sizeof(float))
// @BE binary_fcnmm_gather_hetero_unbranch_float_bf16
FFI_BGM_HETERO_UNBRANCH(_float_bf16, __nv_bfloat16, __nv_bfloat16, sizeof(float))


// --------------------------------------------------------------------------------------------------------
//SPFLOAT_STYLE

#define FCN_MM_GATHER_TILE_K 128 
#define DEFINE_BGM_BASIC_HETERO_SPFLOAT_STYLE(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _bgm_basic_hetero_kern_spfloat_style##SUFFIX(                                 \
    const int32_t* __restrict__ indices,                                                      \
    const SPIKE_T* __restrict__ matrix,                                                       \
    WEIGHT_T* __restrict__ output,                                                       \
    const WEIGHT_T* __restrict__ weights,                                                     \
    int n_pre, int n_conn, int n_batch                                                        \
) {                                                                                           \
    /* 1. 物理共享内存的划分状态 (Tile 容量分配) */                                           \
    extern __shared__ char _smem_raw[];                                                       \
    int32_t* s_idx = reinterpret_cast<int32_t*>(_smem_raw);                                   \
    ACC_T* s_wt  = reinterpret_cast<ACC_T*>(_smem_raw + FCN_MM_GATHER_TILE_K * sizeof(int32_t)); \
                                                                                              \
    /* 2. 线程到输出矩阵的 1D 平铺映射状态 */                                                 \
    int row = blockIdx.x;                                                                     \
    int j   = blockIdx.y * blockDim.x + threadIdx.x;                                          \
    if (row >= n_pre) return;                                                                 \
                                                                                              \
    /* 3. 越界保护与物理寻址基址 */                                                           \
    bool col_valid = (j < n_batch);                                                           \
    int  safe_j    = col_valid ? j : 0;                                                       \
    const int32_t* idx_row = indices + (size_t)row * n_conn;                                 \
    const WEIGHT_T* w_row   = weights + (size_t)row * n_conn;                                 \
                                                                                              \
    /* 4. 线程私有累加器状态初始化 */                                                         \
    ACC_T accum = ACC_ZERO;                                                                   \
                                                                                              \
    /* 5. 步长为 TILE_K 的时间维度切片循环 */                                                 \
    for (int base = 0; base < n_conn; base += FCN_MM_GATHER_TILE_K) {                         \
        int tile_size = n_conn - base;                                                        \
        if (tile_size > FCN_MM_GATHER_TILE_K) tile_size = FCN_MM_GATHER_TILE_K;               \
                                                                                              \
        /* [状态 A] 协作读取：从 Global Memory 提取拓扑与权重至 Shared Memory */              \
        for (int t = threadIdx.x; t < tile_size; t += blockDim.x) {                           \
            s_idx[t] = __ldg(&idx_row[base + t]);                                             \
            s_wt[t]  = READ_W(__ldg(&w_row[base + t]));                                       \
        }                                                                                     \
        __syncthreads(); /* 绝对同步屏障 */                                                   \
                                                                                              \
        /* [状态 B] 线程私有计算：基于 Shared Memory 的物理广播与条件累加 */                  \
        if (col_valid) {                                                                      \
            for (int k = 0; k < tile_size; k++) {                                             \
                /* 布尔求值剪枝 */                                                            \
                bool active = IS_ACTIVE(__ldg(&matrix[(size_t)s_idx[k] * n_batch + safe_j])); \
                if (active) accum += s_wt[k];                                                 \
            }                                                                                 \
        }                                                                                     \
        __syncthreads(); /* 防止下一个 Tile 提前覆写 Shared Memory */                         \
    }                                                                                         \
                                                                                              \
    /* 6. 最终状态直接写回 Global Memory (无需跨线程通信) */                                  \
    if (col_valid) output[(size_t)row * n_batch + j] = WRITE_W(accum);                        \
}

// ---- FFI macro: gather hetero spfloat_style basic ----
#define FFI_BGM_HETERO_SPFLOAT_STYLE(SUFFIX, WEIGHT_C_T, SPIKE_C_T, ACC_SIZE) \
void binary_fcnmm_gather_hetero_spfloat_style##SUFFIX( \
    const BE::Tensor weights, const BE::Tensor indices, \
    const BE::Tensor matrix,  BE::Tensor output, int64_t stream \
) { \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream); \
    int n_pre   = static_cast<int>(indices.size(0)); \
    int n_conn  = static_cast<int>(indices.size(1)); \
    int n_batch = static_cast<int>(matrix.size(1)); \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr()); \
    const SPIKE_C_T*  d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr()); \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr()); \
    int bsz = 256; int nwarps = bsz >> 5; \
    size_t idx_bytes = (size_t)n_conn * sizeof(int32_t); \
    size_t red_bytes = (size_t)nwarps * 32 * ACC_SIZE; \
    size_t shm = (idx_bytes > red_bytes) ? idx_bytes : red_bytes; \
    int batch_tiles = (n_batch + 31) / 32; dim3 grid(n_pre, batch_tiles); \
    _bgm_basic_hetero_kern_spfloat_style##SUFFIX<<<grid, bsz, shm, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch); \
}



DEFINE_BGM_BASIC_HETERO_SPFLOAT_STYLE(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_BGM_BASIC_HETERO_SPFLOAT_STYLE(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, 0.0f)

// ---- float64 ----
DEFINE_BGM_BASIC_HETERO_SPFLOAT_STYLE(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_BGM_BASIC_HETERO_SPFLOAT_STYLE(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, 0.0)

// ---- float16 ----
DEFINE_BGM_BASIC_HETERO_SPFLOAT_STYLE(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_BGM_BASIC_HETERO_SPFLOAT_STYLE(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, 0.0f)

// ---- bfloat16 ----
DEFINE_BGM_BASIC_HETERO_SPFLOAT_STYLE(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_BGM_BASIC_HETERO_SPFLOAT_STYLE(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)


// ---- FFI instantiations: gather hetero spfloat_style ----
// ---- float32 ----
// @BE binary_fcnmm_gather_hetero_spfloat_style_bool_f32
FFI_BGM_HETERO_SPFLOAT_STYLE(_bool_f32, float, uint8_t, sizeof(float))
// @BE binary_fcnmm_gather_hetero_spfloat_style_float_f32
FFI_BGM_HETERO_SPFLOAT_STYLE(_float_f32, float, float, sizeof(float))
// ---- float64 ----
// @BE binary_fcnmm_gather_hetero_spfloat_style_bool_f64
FFI_BGM_HETERO_SPFLOAT_STYLE(_bool_f64, double, uint8_t, sizeof(double))
// @BE binary_fcnmm_gather_hetero_spfloat_style_float_f64
FFI_BGM_HETERO_SPFLOAT_STYLE(_float_f64, double, double, sizeof(double))
// ---- float16 ----
// @BE binary_fcnmm_gather_hetero_spfloat_style_bool_f16
FFI_BGM_HETERO_SPFLOAT_STYLE(_bool_f16, __half, uint8_t, sizeof(float))
// @BE binary_fcnmm_gather_hetero_spfloat_style_float_f16
FFI_BGM_HETERO_SPFLOAT_STYLE(_float_f16, __half, __half, sizeof(float))
// ---- bfloat16 ----
// @BE binary_fcnmm_gather_hetero_spfloat_style_bool_bf16
FFI_BGM_HETERO_SPFLOAT_STYLE(_bool_bf16, __nv_bfloat16, uint8_t, sizeof(float))
// @BE binary_fcnmm_gather_hetero_spfloat_style_float_bf16
FFI_BGM_HETERO_SPFLOAT_STYLE(_float_bf16, __nv_bfloat16, __nv_bfloat16, sizeof(float))

