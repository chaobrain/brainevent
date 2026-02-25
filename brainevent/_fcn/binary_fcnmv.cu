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
 * binary_fcnmv.cu -- Event-Driven Binary FCN Sparse Matrix-Vector CUDA Kernels
 * ==============================================================================
 *
 * This module provides optimized CUDA kernels for event-driven sparse
 * matrix-vector multiplication with fixed connection number (FCN).
 *
 * Operator: binary_fcnmv
 *   - Gather mode (transpose=False): y[i] = sum_k weights[i,k] * is_active(spikes[indices[i,k]])
 *   - Scatter mode (transpose=True): output[indices[i,k]] += weights[i,k] * is_active(spikes[i])
 *
 * Supports weight dtypes: float32, float64, float16, bfloat16
 * Supports spike dtypes:  bool (uint8), float32, float64, float16, bfloat16
 * Supports homo (scalar weight) and hetero (per-connection weight array) modes.
 */

#include "cuda_common.h"
#include "brainevent/common.h"

// ============================================================================
// FCN Matrix-Vector Multiplication (fcnmv) — Optimized CUDA Kernels
// ============================================================================

#define DEFINE_BG_WARP_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _bg_warp_homo_kern##SUFFIX(                                                                   \
    const int32_t* __restrict__ indices,                                                                      \
    const SPIKE_T* __restrict__ spikes,                                                                       \
    WEIGHT_T*      __restrict__ output,                                                                       \
    const WEIGHT_T* __restrict__ weights,                                                                     \
    int n_pre, int n_conn                                                                                     \
) {                                                                                                           \
    int row = blockIdx.x;                                                                                     \
    if (row >= n_pre) return;                                                                                 \
    int lane = threadIdx.x;                                                                                   \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                                    \
    bool in_range = (lane < n_conn);                                                                          \
    int safe_lane = in_range ? lane : (n_conn - 1);                                                           \
    int idx = __ldg(&i_row[safe_lane]);                                                                       \
    bool active = in_range && IS_ACTIVE(__ldg(&spikes[idx]));                                                 \
    ACC_T val = active ? (ACC_T)1 : ACC_ZERO;                                                                 \
    val = WARP_RED(val);                                                                                      \
    if (lane == 0)                                                                                            \
        output[row] = WRITE_W(READ_W(weights[0]) * val);                                                      \
}

#define DEFINE_BG_WARP_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _bg_warp_hetero_kern##SUFFIX(                                                                   \
    const int32_t* __restrict__ indices,                                                                        \
    const SPIKE_T* __restrict__ spikes,                                                                         \
    WEIGHT_T*      __restrict__ output,                                                                         \
    const WEIGHT_T* __restrict__ weights,                                                                       \
    int n_pre, int n_conn                                                                                       \
) {                                                                                                             \
    int row = blockIdx.x;                                                                                       \
    if (row >= n_pre) return;                                                                                   \
    int lane = threadIdx.x;                                                                                     \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                                      \
    const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                                     \
    bool in_range = (lane < n_conn);                                                                            \
    int safe_lane = in_range ? lane : (n_conn - 1);                                                             \
    int idx = __ldg(&i_row[safe_lane]);                                                                         \
    bool active = in_range && IS_ACTIVE(__ldg(&spikes[idx]));                                                   \
    ACC_T val = active ? READ_W(__ldg(&w_row[safe_lane])) : ACC_ZERO;                                           \
    val = WARP_RED(val);                                                                                        \
    if (lane == 0)                                                                                              \
        output[row] = WRITE_W(val);                                                                             \
}

#define DEFINE_BG_MR_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _bg_mr_homo_kern##SUFFIX(                                                                   \
    const int32_t* __restrict__ indices,                                                                    \
    const SPIKE_T* __restrict__ spikes,                                                                     \
    WEIGHT_T*      __restrict__ output,                                                                     \
    const WEIGHT_T* __restrict__ weights,                                                                   \
    int n_pre, int n_conn                                                                                   \
) {                                                                                                         \
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);                                          \
    if (row >= n_pre) return;                                                                               \
    int lane = threadIdx.x & 31;                                                                            \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                                  \
    ACC_T val = ACC_ZERO;                                                                                   \
    for (int k = lane; k < n_conn; k += 32) {                                                               \
        int idx = __ldg(&i_row[k]);                                                                         \
        if (IS_ACTIVE(__ldg(&spikes[idx])))                                                                 \
            val += (ACC_T)1;                                                                                \
    }                                                                                                       \
    val = WARP_RED(val);                                                                                    \
    if (lane == 0)                                                                                          \
        output[row] = WRITE_W(READ_W(weights[0]) * val);                                                    \
}

#define DEFINE_BG_MR_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _bg_mr_hetero_kern##SUFFIX(                                                                   \
    const int32_t* __restrict__ indices,                                                                      \
    const SPIKE_T* __restrict__ spikes,                                                                       \
    WEIGHT_T*      __restrict__ output,                                                                       \
    const WEIGHT_T* __restrict__ weights,                                                                     \
    int n_pre, int n_conn                                                                                     \
) {                                                                                                           \
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);                                            \
    if (row >= n_pre) return;                                                                                 \
    int lane = threadIdx.x & 31;                                                                              \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                                    \
    const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                                   \
    ACC_T val = ACC_ZERO;                                                                                     \
    for (int k = lane; k < n_conn; k += 32) {                                                                 \
        int idx = __ldg(&i_row[k]);                                                                           \
        if (IS_ACTIVE(__ldg(&spikes[idx])))                                                                   \
            val += READ_W(__ldg(&w_row[k]));                                                                  \
    }                                                                                                         \
    val = WARP_RED(val);                                                                                      \
    if (lane == 0)                                                                                            \
        output[row] = WRITE_W(val);                                                                           \
}

#define DEFINE_BS_WARP_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _bs_warp_homo_kern##SUFFIX(                                                   \
    const int32_t* __restrict__ indices,                                                      \
    const SPIKE_T* __restrict__ spikes,                                                       \
    WEIGHT_T*      __restrict__ output,                                                       \
    const WEIGHT_T* __restrict__ weights,                                                     \
    int n_pre, int n_conn                                                                     \
) {                                                                                           \
    int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;                             \
    int lane_id   = threadIdx.x & 31;                                                         \
    int num_warps = (gridDim.x * blockDim.x) >> 5;                                            \
    ACC_T w0 = READ_W(weights[0]);                                                            \
    for (int row = warp_id; row < n_pre; row += num_warps) {                            \
        if (!IS_ACTIVE(__ldg(&spikes[row]))) continue;                                  \
        const int32_t* i_row = indices + (size_t)row * n_conn;                          \
        for (int k = lane_id; k < n_conn; k += 32) {                                    \
            int idx = __ldg(&i_row[k]);                                                 \
            ATOMIC_ADD_W(&output[idx], w0);                                             \
        }                                                                               \
    }                                                                                   \
}

#define DEFINE_BS_WARP_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _bs_warp_hetero_kern##SUFFIX(                                                  \
    const int32_t* __restrict__ indices,                                                       \
    const SPIKE_T* __restrict__ spikes,                                                        \
    WEIGHT_T*      __restrict__ output,                                                        \
    const WEIGHT_T* __restrict__ weights,                                                      \
    int n_pre, int n_conn                                                                      \
) {                                                                                            \
    int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;                              \
    int lane_id   = threadIdx.x & 31;                                                          \
    int num_warps = (gridDim.x * blockDim.x) >> 5;                                             \
    for (int row = warp_id; row < n_pre; row += num_warps) {                                   \
        if (!IS_ACTIVE(__ldg(&spikes[row]))) continue;                                         \
        const int32_t* i_row = indices + (size_t)row * n_conn;                                 \
        const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                \
        for (int k = lane_id; k < n_conn; k += 32) {                                           \
            int idx = __ldg(&i_row[k]);                                                        \
            ACC_T wk = READ_W(__ldg(&w_row[k]));                                               \
            ATOMIC_ADD_W(&output[idx], wk);                                               \
        }                                                                                 \
    }                                                                                     \
}

#define DEFINE_BS_BASIC_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _bs_basic_homo_kern##SUFFIX(                                                   \
    const int32_t* __restrict__ indices,                                                       \
    const SPIKE_T* __restrict__ spikes,                                                        \
    WEIGHT_T*      __restrict__ output,                                                        \
    const WEIGHT_T* __restrict__ weights,                                                      \
    int n_pre, int n_conn                                                                      \
) {                                                                                            \
    int row = blockIdx.x;                                                                      \
    if (row >= n_pre) return;                                                                  \
    if (!IS_ACTIVE(__ldg(&spikes[row]))) return;                                               \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                     \
    ACC_T w0 = READ_W(weights[0]);                                                             \
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x) {                             \
        int idx = __ldg(&i_row[k]);                                                      \
        ATOMIC_ADD_W(&output[idx], w0);                                                  \
    }                                                                                    \
}

#define DEFINE_BS_BASIC_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _bs_basic_hetero_kern##SUFFIX(                                                  \
    const int32_t* __restrict__ indices,                                                        \
    const SPIKE_T* __restrict__ spikes,                                                         \
    WEIGHT_T*      __restrict__ output,                                                         \
    const WEIGHT_T* __restrict__ weights,                                                       \
    int n_pre, int n_conn                                                                       \
) {                                                                                             \
    int row = blockIdx.x;                                                                       \
    if (row >= n_pre) return;                                                                   \
    if (!IS_ACTIVE(__ldg(&spikes[row]))) return;                                                \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                      \
    const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                     \
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x) {                                    \
        int idx = __ldg(&i_row[k]);                                                             \
        ACC_T wk = READ_W(__ldg(&w_row[k]));                                                    \
        ATOMIC_ADD_W(&output[idx], wk);                                                    \
    }                                                                                      \
}

// Instantiations
// ---- float32 ----
DEFINE_BG_WARP_HOMO   (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_WARP_HETERO (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_WARP_HOMO   (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_WARP_HETERO (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO     (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO   (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO     (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO   (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BS_WARP_HOMO   (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, atomicAdd)
DEFINE_BS_WARP_HETERO (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, atomicAdd)
DEFINE_BS_WARP_HOMO   (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, atomicAdd)
DEFINE_BS_WARP_HETERO (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, atomicAdd)
DEFINE_BS_BASIC_HOMO  (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, atomicAdd)
DEFINE_BS_BASIC_HETERO(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, atomicAdd)
DEFINE_BS_BASIC_HOMO  (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, atomicAdd)
DEFINE_BS_BASIC_HETERO(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, atomicAdd)

// ---- float64 ----
DEFINE_BG_WARP_HOMO   (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_WARP_HETERO (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_WARP_HOMO   (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_WARP_HETERO (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HOMO     (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HETERO   (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HOMO     (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HETERO   (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BS_WARP_HOMO   (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, atomic_add_f64)
DEFINE_BS_WARP_HETERO (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, atomic_add_f64)
DEFINE_BS_WARP_HOMO   (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, atomic_add_f64)
DEFINE_BS_WARP_HETERO (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, atomic_add_f64)
DEFINE_BS_BASIC_HOMO  (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, atomic_add_f64)
DEFINE_BS_BASIC_HETERO(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, atomic_add_f64)
DEFINE_BS_BASIC_HOMO  (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, atomic_add_f64)
DEFINE_BS_BASIC_HETERO(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, atomic_add_f64)

// ---- float16 ----
DEFINE_BG_WARP_HOMO   (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_WARP_HETERO (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_WARP_HOMO   (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_WARP_HETERO (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO     (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO   (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO     (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO   (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BS_WARP_HOMO   (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, atomic_add_f16)
DEFINE_BS_WARP_HETERO (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, atomic_add_f16)
DEFINE_BS_WARP_HOMO   (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, atomic_add_f16)
DEFINE_BS_WARP_HETERO (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, atomic_add_f16)
DEFINE_BS_BASIC_HOMO  (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, atomic_add_f16)
DEFINE_BS_BASIC_HETERO(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, atomic_add_f16)
DEFINE_BS_BASIC_HOMO  (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, atomic_add_f16)
DEFINE_BS_BASIC_HETERO(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, atomic_add_f16)

// ---- bfloat16 ----
DEFINE_BG_WARP_HOMO   (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_WARP_HETERO (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_WARP_HOMO   (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_WARP_HETERO (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO     (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO   (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO     (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO   (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BS_WARP_HOMO   (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BS_WARP_HETERO (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BS_WARP_HOMO   (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BS_WARP_HETERO (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BS_BASIC_HOMO  (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BS_BASIC_HETERO(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BS_BASIC_HOMO  (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BS_BASIC_HETERO(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)

// FFI Macros for SpMV
// ---- FFI macro: gather homo warp ----
#define FFI_BG_HOMO_WARP(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                       \
void binary_fcnmv_gather_homo_warp##SUFFIX(                                                   \
    const BE::Tensor weights,                                                                 \
    const BE::Tensor indices,                                                                 \
    const BE::Tensor spikes,                                                                  \
    BE::Tensor output, int64_t stream                                                         \
) {                                                                                           \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                  \
    int n_pre  = static_cast<int>(indices.size(0));                                           \
    int n_conn = static_cast<int>(indices.size(1));                                           \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());             \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                \
    const SPIKE_C_T*  d_spk = static_cast<const SPIKE_C_T*>(spikes.data_ptr());               \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                    \
    _bg_warp_homo_kern##SUFFIX<<<n_pre, 32, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn); \
}

// ---- FFI macro: gather hetero warp ----
#define FFI_BG_HETERO_WARP(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                       \
void binary_fcnmv_gather_hetero_warp##SUFFIX(                                                   \
    const BE::Tensor weights, const BE::Tensor indices,                                         \
    const BE::Tensor spikes,  BE::Tensor output, int64_t stream                                 \
) {                                                                                             \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                    \
    int n_pre  = static_cast<int>(indices.size(0));                                             \
    int n_conn = static_cast<int>(indices.size(1));                                             \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());               \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                  \
    const SPIKE_C_T*  d_spk = static_cast<const SPIKE_C_T*>(spikes.data_ptr());                 \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                      \
    _bg_warp_hetero_kern##SUFFIX<<<n_pre, 32, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn); \
}

// ---- FFI macro: gather homo basic (multi-row) ----
#define FFI_BG_HOMO_BASIC(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                        \
void binary_fcnmv_gather_homo_basic##SUFFIX(                                                    \
    const BE::Tensor weights, const BE::Tensor indices,                                         \
    const BE::Tensor spikes,  BE::Tensor output, int64_t stream                                 \
) {                                                                                             \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                    \
    int n_pre  = static_cast<int>(indices.size(0));                                             \
    int n_conn = static_cast<int>(indices.size(1));                                             \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());               \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                  \
    const SPIKE_C_T*  d_spk = static_cast<const SPIKE_C_T*>(spikes.data_ptr());                 \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                      \
    int bsz = 256; int rpb = bsz >> 5; int n_blocks = (n_pre + rpb - 1) / rpb;                  \
    _bg_mr_homo_kern##SUFFIX<<<n_blocks, bsz, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn); \
}

// ---- FFI macro: gather hetero basic (multi-row) ----
#define FFI_BG_HETERO_BASIC(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                        \
void binary_fcnmv_gather_hetero_basic##SUFFIX(                                                    \
    const BE::Tensor weights, const BE::Tensor indices,                                           \
    const BE::Tensor spikes,  BE::Tensor output, int64_t stream                                   \
) {                                                                                               \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                      \
    int n_pre  = static_cast<int>(indices.size(0));                                               \
    int n_conn = static_cast<int>(indices.size(1));                                               \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                 \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                    \
    const SPIKE_C_T*  d_spk = static_cast<const SPIKE_C_T*>(spikes.data_ptr());                   \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                        \
    int bsz = 256; int rpb = bsz >> 5; int n_blocks = (n_pre + rpb - 1) / rpb;                    \
    _bg_mr_hetero_kern##SUFFIX<<<n_blocks, bsz, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn); \
}

// ---- FFI macro: scatter homo warp ----
#define FFI_BS_HOMO_WARP(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                         \
void binary_fcnmv_scatter_homo_warp##SUFFIX(                                                    \
    const BE::Tensor weights, const BE::Tensor indices,                                         \
    const BE::Tensor spikes,  BE::Tensor output, int64_t stream                                 \
) {                                                                                             \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                    \
    int n_pre  = static_cast<int>(indices.size(0));                                             \
    int n_conn = static_cast<int>(indices.size(1));                                             \
    int n_post = static_cast<int>(output.size(0));                                              \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());               \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                  \
    const SPIKE_C_T*  d_spk = static_cast<const SPIKE_C_T*>(spikes.data_ptr());                 \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                      \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(WEIGHT_C_T), s);                          \
    int blocks = (n_pre + 7) / 8;                                                               \
    _bs_warp_homo_kern##SUFFIX<<<blocks, 256, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn); \
}

// ---- FFI macro: scatter hetero warp ----
#define FFI_BS_HETERO_WARP(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                         \
void binary_fcnmv_scatter_hetero_warp##SUFFIX(                                                    \
    const BE::Tensor weights, const BE::Tensor indices,                                           \
    const BE::Tensor spikes,  BE::Tensor output, int64_t stream                                   \
) {                                                                                               \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                      \
    int n_pre  = static_cast<int>(indices.size(0));                                               \
    int n_conn = static_cast<int>(indices.size(1));                                               \
    int n_post = static_cast<int>(output.size(0));                                                \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                 \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                    \
    const SPIKE_C_T*  d_spk = static_cast<const SPIKE_C_T*>(spikes.data_ptr());                   \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                        \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(WEIGHT_C_T), s);                            \
    int blocks = (n_pre + 7) / 8;                                                                 \
    _bs_warp_hetero_kern##SUFFIX<<<blocks, 256, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn); \
}

// ---- FFI macro: scatter homo basic ----
#define FFI_BS_HOMO_BASIC(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                        \
void binary_fcnmv_scatter_homo_basic##SUFFIX(                                                   \
    const BE::Tensor weights, const BE::Tensor indices,                                 \
    const BE::Tensor spikes,  BE::Tensor output, int64_t stream                   \
) {                                                                                             \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                    \
    int n_pre  = static_cast<int>(indices.size(0));                                             \
    int n_conn = static_cast<int>(indices.size(1));                                             \
    int n_post = static_cast<int>(output.size(0));                                              \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());               \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                  \
    const SPIKE_C_T*  d_spk = static_cast<const SPIKE_C_T*>(spikes.data_ptr());                 \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                      \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(WEIGHT_C_T), s);                          \
    _bs_basic_homo_kern##SUFFIX<<<n_pre, 256, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn); \
}

// ---- FFI macro: scatter hetero basic ----
#define FFI_BS_HETERO_BASIC(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                        \
void binary_fcnmv_scatter_hetero_basic##SUFFIX(                                                   \
    const BE::Tensor weights, const BE::Tensor indices,                                   \
    const BE::Tensor spikes,  BE::Tensor output, int64_t stream                     \
) {                                                                                               \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                      \
    int n_pre  = static_cast<int>(indices.size(0));                                               \
    int n_conn = static_cast<int>(indices.size(1));                                               \
    int n_post = static_cast<int>(output.size(0));                                                \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                 \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                    \
    const SPIKE_C_T*  d_spk = static_cast<const SPIKE_C_T*>(spikes.data_ptr());                   \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                        \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(WEIGHT_C_T), s);                            \
    _bs_basic_hetero_kern##SUFFIX<<<n_pre, 256, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn); \
}

// SpMV FFI Instantiations
// ---- float32 ----
// @BE binary_fcnmv_gather_homo_warp_bool_f32
FFI_BG_HOMO_WARP  (_bool_f32, float, uint8_t)
// @BE binary_fcnmv_gather_hetero_warp_bool_f32
FFI_BG_HETERO_WARP(_bool_f32, float, uint8_t)
// @BE binary_fcnmv_gather_homo_warp_float_f32
FFI_BG_HOMO_WARP  (_float_f32, float, float)
// @BE binary_fcnmv_gather_hetero_warp_float_f32
FFI_BG_HETERO_WARP(_float_f32, float, float)
// @BE binary_fcnmv_gather_homo_basic_bool_f32
FFI_BG_HOMO_BASIC (_bool_f32, float, uint8_t)
// @BE binary_fcnmv_gather_hetero_basic_bool_f32
FFI_BG_HETERO_BASIC(_bool_f32, float, uint8_t)
// @BE binary_fcnmv_gather_homo_basic_float_f32
FFI_BG_HOMO_BASIC (_float_f32, float, float)
// @BE binary_fcnmv_gather_hetero_basic_float_f32
FFI_BG_HETERO_BASIC(_float_f32, float, float)
// @BE binary_fcnmv_scatter_homo_warp_bool_f32
FFI_BS_HOMO_WARP  (_bool_f32, float, uint8_t)
// @BE binary_fcnmv_scatter_hetero_warp_bool_f32
FFI_BS_HETERO_WARP(_bool_f32, float, uint8_t)
// @BE binary_fcnmv_scatter_homo_warp_float_f32
FFI_BS_HOMO_WARP  (_float_f32, float, float)
// @BE binary_fcnmv_scatter_hetero_warp_float_f32
FFI_BS_HETERO_WARP(_float_f32, float, float)
// @BE binary_fcnmv_scatter_homo_basic_bool_f32
FFI_BS_HOMO_BASIC (_bool_f32, float, uint8_t)
// @BE binary_fcnmv_scatter_hetero_basic_bool_f32
FFI_BS_HETERO_BASIC(_bool_f32, float, uint8_t)
// @BE binary_fcnmv_scatter_homo_basic_float_f32
FFI_BS_HOMO_BASIC (_float_f32, float, float)
// @BE binary_fcnmv_scatter_hetero_basic_float_f32
FFI_BS_HETERO_BASIC(_float_f32, float, float)

// ---- float64 ----
// @BE binary_fcnmv_gather_homo_warp_bool_f64
FFI_BG_HOMO_WARP  (_bool_f64, double, uint8_t)
// @BE binary_fcnmv_gather_hetero_warp_bool_f64
FFI_BG_HETERO_WARP(_bool_f64, double, uint8_t)
// @BE binary_fcnmv_gather_homo_warp_float_f64
FFI_BG_HOMO_WARP  (_float_f64, double, double)
// @BE binary_fcnmv_gather_hetero_warp_float_f64
FFI_BG_HETERO_WARP(_float_f64, double, double)
// @BE binary_fcnmv_gather_homo_basic_bool_f64
FFI_BG_HOMO_BASIC (_bool_f64, double, uint8_t)
// @BE binary_fcnmv_gather_hetero_basic_bool_f64
FFI_BG_HETERO_BASIC(_bool_f64, double, uint8_t)
// @BE binary_fcnmv_gather_homo_basic_float_f64
FFI_BG_HOMO_BASIC (_float_f64, double, double)
// @BE binary_fcnmv_gather_hetero_basic_float_f64
FFI_BG_HETERO_BASIC(_float_f64, double, double)
// @BE binary_fcnmv_scatter_homo_warp_bool_f64
FFI_BS_HOMO_WARP  (_bool_f64, double, uint8_t)
// @BE binary_fcnmv_scatter_hetero_warp_bool_f64
FFI_BS_HETERO_WARP(_bool_f64, double, uint8_t)
// @BE binary_fcnmv_scatter_homo_warp_float_f64
FFI_BS_HOMO_WARP  (_float_f64, double, double)
// @BE binary_fcnmv_scatter_hetero_warp_float_f64
FFI_BS_HETERO_WARP(_float_f64, double, double)
// @BE binary_fcnmv_scatter_homo_basic_bool_f64
FFI_BS_HOMO_BASIC (_bool_f64, double, uint8_t)
// @BE binary_fcnmv_scatter_hetero_basic_bool_f64
FFI_BS_HETERO_BASIC(_bool_f64, double, uint8_t)
// @BE binary_fcnmv_scatter_homo_basic_float_f64
FFI_BS_HOMO_BASIC (_float_f64, double, double)
// @BE binary_fcnmv_scatter_hetero_basic_float_f64
FFI_BS_HETERO_BASIC(_float_f64, double, double)

// ---- float16 ----
// @BE binary_fcnmv_gather_homo_warp_bool_f16
FFI_BG_HOMO_WARP  (_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_gather_hetero_warp_bool_f16
FFI_BG_HETERO_WARP(_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_gather_homo_warp_float_f16
FFI_BG_HOMO_WARP  (_float_f16, __half, __half)
// @BE binary_fcnmv_gather_hetero_warp_float_f16
FFI_BG_HETERO_WARP(_float_f16, __half, __half)
// @BE binary_fcnmv_gather_homo_basic_bool_f16
FFI_BG_HOMO_BASIC (_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_gather_hetero_basic_bool_f16
FFI_BG_HETERO_BASIC(_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_gather_homo_basic_float_f16
FFI_BG_HOMO_BASIC (_float_f16, __half, __half)
// @BE binary_fcnmv_gather_hetero_basic_float_f16
FFI_BG_HETERO_BASIC(_float_f16, __half, __half)
// @BE binary_fcnmv_scatter_homo_warp_bool_f16
FFI_BS_HOMO_WARP  (_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_scatter_hetero_warp_bool_f16
FFI_BS_HETERO_WARP(_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_scatter_homo_warp_float_f16
FFI_BS_HOMO_WARP  (_float_f16, __half, __half)
// @BE binary_fcnmv_scatter_hetero_warp_float_f16
FFI_BS_HETERO_WARP(_float_f16, __half, __half)
// @BE binary_fcnmv_scatter_homo_basic_bool_f16
FFI_BS_HOMO_BASIC (_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_scatter_hetero_basic_bool_f16
FFI_BS_HETERO_BASIC(_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_scatter_homo_basic_float_f16
FFI_BS_HOMO_BASIC (_float_f16, __half, __half)
// @BE binary_fcnmv_scatter_hetero_basic_float_f16
FFI_BS_HETERO_BASIC(_float_f16, __half, __half)

// ---- bfloat16 ----
// @BE binary_fcnmv_gather_homo_warp_bool_bf16
FFI_BG_HOMO_WARP  (_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_gather_hetero_warp_bool_bf16
FFI_BG_HETERO_WARP(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_gather_homo_warp_float_bf16
FFI_BG_HOMO_WARP  (_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmv_gather_hetero_warp_float_bf16
FFI_BG_HETERO_WARP(_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmv_gather_homo_basic_bool_bf16
FFI_BG_HOMO_BASIC (_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_gather_hetero_basic_bool_bf16
FFI_BG_HETERO_BASIC(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_gather_homo_basic_float_bf16
FFI_BG_HOMO_BASIC (_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmv_gather_hetero_basic_float_bf16
FFI_BG_HETERO_BASIC(_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmv_scatter_homo_warp_bool_bf16
FFI_BS_HOMO_WARP  (_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_scatter_hetero_warp_bool_bf16
FFI_BS_HETERO_WARP(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_scatter_homo_warp_float_bf16
FFI_BS_HOMO_WARP  (_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmv_scatter_hetero_warp_float_bf16
FFI_BS_HETERO_WARP(_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmv_scatter_homo_basic_bool_bf16
FFI_BS_HOMO_BASIC (_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_scatter_hetero_basic_bool_bf16
FFI_BS_HETERO_BASIC(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_scatter_homo_basic_float_bf16
FFI_BS_HOMO_BASIC (_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmv_scatter_hetero_basic_float_bf16
FFI_BS_HETERO_BASIC(_float_bf16, __nv_bfloat16, __nv_bfloat16)
