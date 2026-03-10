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



// ----------------------------------------------------------------------------------------------------
///cuda_raw 
///
/*
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

DEFINE_BG_MR_HOMO     (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO   (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO     (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO   (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)

DEFINE_BG_MR_HOMO     (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HETERO   (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HOMO     (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HETERO   (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)


DEFINE_BG_MR_HOMO     (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO   (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO     (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO   (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)

DEFINE_BG_MR_HOMO     (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO   (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO     (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO   (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)


// @BE_DISABLED binary_fcnmv_gather_homo_basic_bool_f32
FFI_BG_HOMO_BASIC (_bool_f32, float, uint8_t)
// @BE_DISABLED binary_fcnmv_gather_hetero_basic_bool_f32
FFI_BG_HETERO_BASIC(_bool_f32, float, uint8_t)
// @BE_DISABLED binary_fcnmv_gather_homo_basic_float_f32
FFI_BG_HOMO_BASIC (_float_f32, float, float)
// @BE_DISABLED binary_fcnmv_gather_hetero_basic_float_f32
FFI_BG_HETERO_BASIC(_float_f32, float, float)

// @BE_DISABLED binary_fcnmv_gather_homo_basic_bool_f64
FFI_BG_HOMO_BASIC (_bool_f64, double, uint8_t)
// @BE_DISABLED binary_fcnmv_gather_hetero_basic_bool_f64
FFI_BG_HETERO_BASIC(_bool_f64, double, uint8_t)
// @BE_DISABLED binary_fcnmv_gather_homo_basic_float_f64
FFI_BG_HOMO_BASIC (_float_f64, double, double)
// @BE_DISABLED binary_fcnmv_gather_hetero_basic_float_f64
FFI_BG_HETERO_BASIC(_float_f64, double, double)

// @BE_DISABLED binary_fcnmv_gather_homo_basic_bool_f16
FFI_BG_HOMO_BASIC (_bool_f16, __half, uint8_t)
// @BE_DISABLED binary_fcnmv_gather_hetero_basic_bool_f16
FFI_BG_HETERO_BASIC(_bool_f16, __half, uint8_t)
// @BE_DISABLED binary_fcnmv_gather_homo_basic_float_f16
FFI_BG_HOMO_BASIC (_float_f16, __half, __half)
// @BE_DISABLED binary_fcnmv_gather_hetero_basic_float_f16
FFI_BG_HETERO_BASIC(_float_f16, __half, __half)

// @BE_DISABLED binary_fcnmv_gather_homo_basic_bool_bf16
FFI_BG_HOMO_BASIC (_bool_bf16, __nv_bfloat16, uint8_t)
// @BE_DISABLED binary_fcnmv_gather_hetero_basic_bool_bf16
FFI_BG_HETERO_BASIC(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE_DISABLED binary_fcnmv_gather_homo_basic_float_bf16
FFI_BG_HOMO_BASIC (_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE_DISABLED binary_fcnmv_gather_hetero_basic_float_bf16
FFI_BG_HETERO_BASIC(_float_bf16, __nv_bfloat16, __nv_bfloat16)
*/

// ----------------------------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------------------------
/// cuda_raw_unbranch
///

#define DEFINE_BG_MR_HOMO_RAW_UNBRANCH(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
    __global__ void _bg_mr_homo_kern_raw_unbranch##SUFFIX(                                                               \
        const int32_t *__restrict__ indices,                                                                             \
        const SPIKE_T *__restrict__ spikes,                                                                              \
        WEIGHT_T *__restrict__ output,                                                                                   \
        const WEIGHT_T *__restrict__ weights,                                                                            \
        int n_pre, int n_conn) {                                                                                         \
        int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);                                                   \
        if (row >= n_pre)                                                                                                \
            return;                                                                                                      \
        int lane = threadIdx.x & 31;                                                                                     \
        const int32_t *i_row = indices + (size_t)row * n_conn;                                                           \
        ACC_T val = ACC_ZERO;                                                                                            \
        int main_iters = n_conn >> 5;                                                                                    \
        int tail = n_conn & 31;                                                                                          \
        int k = lane;                                                                                                    \
        int limit = main_iters & ~1;                                                                                     \
        for (int i = 0; i < limit; i += 2)                                                                               \
        {                                                                                                                \
            int idx0 = __ldg(&i_row[k]);                                                                                 \
            int idx1 = __ldg(&i_row[k + 32]);                                                                            \
            SPIKE_T spk0 = __ldg(&spikes[idx0]);                                                                         \
            SPIKE_T spk1 = __ldg(&spikes[idx1]);                                                                         \
            if (IS_ACTIVE(spk0))                                                                                         \
                val += (ACC_T)1;                                                                                         \
            if (IS_ACTIVE(spk1))                                                                                         \
                val += (ACC_T)1;                                                                                         \
            k += 64;                                                                                                     \
        }                                                                                                                \
        if (main_iters & 1)                                                                                              \
        {                                                                                                                \
            int idx = __ldg(&i_row[k]);                                                                                  \
            if (IS_ACTIVE(__ldg(&spikes[idx])))                                                                          \
                val += (ACC_T)1;                                                                                         \
            k += 32;                                                                                                     \
        }                                                                                                                \
        if (lane < tail)                                                                                                 \
        {                                                                                                                \
            int idx = __ldg(&i_row[k]);                                                                                  \
            if (IS_ACTIVE(__ldg(&spikes[idx])))                                                                          \
                val += (ACC_T)1;                                                                                         \
        }                                                                                                                \
        val = WARP_RED(val);                                                                                             \
        if (lane == 0)                                                                                                   \
            output[row] = WRITE_W(READ_W(weights[0]) * val);                                                             \
    }
    
#define DEFINE_BG_MR_HETERO_RAW_UNBRANCH(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
    __global__ void _bg_mr_hetero_kern_raw_unbranch##SUFFIX(                                                       \
        const int32_t *__restrict__ indices,                                                                       \
        const SPIKE_T *__restrict__ spikes,                                                                        \
        WEIGHT_T *__restrict__ output,                                                                             \
        const WEIGHT_T *__restrict__ weights,                                                                      \
        int n_pre, int n_conn                                                                                      \
    ) {                                                                                                            \
        int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);                                             \
        if (row >= n_pre) return;                                                                                  \
                                                                                                                   \
        int lane = threadIdx.x & 31;                                                                               \
        const int32_t* i_row = indices + (size_t)row * n_conn;                                                     \
        const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                                    \
        ACC_T val = ACC_ZERO;                                                                                      \
                                                                                                                   \
        int main_iters = n_conn >> 5;                                                                              \
        int tail = n_conn & 31;                                                                                    \
        int k = lane;                                                                                              \
        int limit = main_iters & ~1;                                                                               \
                                                                                                                   \
        for (int i = 0; i < limit; i += 2) {                                                                       \
            int idx0 = __ldg(&i_row[k]);                                                                           \
            int idx1 = __ldg(&i_row[k + 32]);                                                                      \
            SPIKE_T spk0 = __ldg(&spikes[idx0]);                                                                   \
            SPIKE_T spk1 = __ldg(&spikes[idx1]);                                                                   \
                                                                                                                   \
            if (IS_ACTIVE(spk0)) val += READ_W(__ldg(&w_row[k]));                                                  \
            if (IS_ACTIVE(spk1)) val += READ_W(__ldg(&w_row[k + 32]));                                             \
            k += 64;                                                                                               \
        }                                                                                                          \
                                                                                                                   \
        if (main_iters & 1) {                                                                                      \
            int idx = __ldg(&i_row[k]);                                                                            \
            if (IS_ACTIVE(__ldg(&spikes[idx]))) val += READ_W(__ldg(&w_row[k]));                                   \
            k += 32;                                                                                               \
        }                                                                                                          \
                                                                                                                   \
        if (lane < tail) {                                                                                         \
            int idx = __ldg(&i_row[k]);                                                                            \
            if (IS_ACTIVE(__ldg(&spikes[idx]))) val += READ_W(__ldg(&w_row[k]));                                   \
        }                                                                                                          \
                                                                                                                   \
        val = WARP_RED(val);                                                                                       \
        if (lane == 0) output[row] = WRITE_W(val);                                                                 \
    }

#define FFI_BG_HOMO_BASIC_RAW_UNBRANCH(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                            \
    void binary_fcnmv_gather_homo_basic_raw_unbranch##SUFFIX(                                                    \
        const BE::Tensor weights, const BE::Tensor indices,                                                      \
        const BE::Tensor spikes, BE::Tensor output, int64_t stream)                                              \
    {                                                                                                            \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                                 \
        int n_pre = static_cast<int>(indices.size(0));                                                           \
        int n_conn = static_cast<int>(indices.size(1));                                                          \
        const WEIGHT_C_T *d_w = static_cast<const WEIGHT_C_T *>(weights.data_ptr());                             \
        const int32_t *d_idx = static_cast<const int32_t *>(indices.data_ptr());                                 \
        const SPIKE_C_T *d_spk = static_cast<const SPIKE_C_T *>(spikes.data_ptr());                              \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr());                                        \
        int bsz = 256;                                                                                           \
        int rpb = bsz >> 5;                                                                                      \
        int n_blocks = (n_pre + rpb - 1) / rpb;                                                                  \
        _bg_mr_homo_kern_raw_unbranch##SUFFIX<<<n_blocks, bsz, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn); \
    }

#define FFI_BG_HETERO_BASIC_RAW_UNBRANCH(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                            \
    void binary_fcnmv_gather_hetero_basic_raw_unbranch##SUFFIX(                                                    \
        const BE::Tensor weights, const BE::Tensor indices,                                                        \
        const BE::Tensor spikes, BE::Tensor output, int64_t stream)                                                \
    {                                                                                                              \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                                   \
        int n_pre = static_cast<int>(indices.size(0));                                                             \
        int n_conn = static_cast<int>(indices.size(1));                                                            \
        const WEIGHT_C_T *d_w = static_cast<const WEIGHT_C_T *>(weights.data_ptr());                               \
        const int32_t *d_idx = static_cast<const int32_t *>(indices.data_ptr());                                   \
        const SPIKE_C_T *d_spk = static_cast<const SPIKE_C_T *>(spikes.data_ptr());                                \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr());                                          \
        int bsz = 256;                                                                                             \
        int rpb = bsz >> 5;                                                                                        \
        int n_blocks = (n_pre + rpb - 1) / rpb;                                                                    \
        _bg_mr_hetero_kern_raw_unbranch##SUFFIX<<<n_blocks, bsz, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn); \
    }

DEFINE_BG_MR_HOMO_RAW_UNBRANCH(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_RAW_UNBRANCH(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO_RAW_UNBRANCH(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_RAW_UNBRANCH(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)

DEFINE_BG_MR_HOMO_RAW_UNBRANCH(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HETERO_RAW_UNBRANCH(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HOMO_RAW_UNBRANCH(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HETERO_RAW_UNBRANCH(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)

DEFINE_BG_MR_HOMO_RAW_UNBRANCH(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_RAW_UNBRANCH(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO_RAW_UNBRANCH(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_RAW_UNBRANCH(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)

DEFINE_BG_MR_HOMO_RAW_UNBRANCH(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_RAW_UNBRANCH(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO_RAW_UNBRANCH(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_RAW_UNBRANCH(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)

// @BE binary_fcnmv_gather_homo_basic_raw_unbranch_bool_f32
FFI_BG_HOMO_BASIC_RAW_UNBRANCH(_bool_f32, float, uint8_t)
// @BE binary_fcnmv_gather_hetero_basic_raw_unbranch_bool_f32
FFI_BG_HETERO_BASIC_RAW_UNBRANCH(_bool_f32, float, uint8_t)
// @BE binary_fcnmv_gather_homo_basic_raw_unbranch_float_f32
FFI_BG_HOMO_BASIC_RAW_UNBRANCH(_float_f32, float, float)
// @BE binary_fcnmv_gather_hetero_basic_raw_unbranch_float_f32
FFI_BG_HETERO_BASIC_RAW_UNBRANCH(_float_f32, float, float)

// @BE binary_fcnmv_gather_homo_basic_raw_unbranch_bool_f64
FFI_BG_HOMO_BASIC_RAW_UNBRANCH(_bool_f64, double, uint8_t)
// @BE binary_fcnmv_gather_hetero_basic_raw_unbranch_bool_f64
FFI_BG_HETERO_BASIC_RAW_UNBRANCH(_bool_f64, double, uint8_t)
// @BE binary_fcnmv_gather_homo_basic_raw_unbranch_float_f64
FFI_BG_HOMO_BASIC_RAW_UNBRANCH(_float_f64, double, double)
// @BE binary_fcnmv_gather_hetero_basic_raw_unbranch_float_f64
FFI_BG_HETERO_BASIC_RAW_UNBRANCH(_float_f64, double, double)

// @BE binary_fcnmv_gather_homo_basic_raw_unbranch_bool_f16
FFI_BG_HOMO_BASIC_RAW_UNBRANCH(_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_gather_hetero_basic_raw_unbranch_bool_f16
FFI_BG_HETERO_BASIC_RAW_UNBRANCH(_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_gather_homo_basic_raw_unbranch_float_f16
FFI_BG_HOMO_BASIC_RAW_UNBRANCH(_float_f16, __half, __half)
// @BE binary_fcnmv_gather_hetero_basic_raw_unbranch_float_f16
FFI_BG_HETERO_BASIC_RAW_UNBRANCH(_float_f16, __half, __half)

// @BE binary_fcnmv_gather_homo_basic_raw_unbranch_bool_bf16
FFI_BG_HOMO_BASIC_RAW_UNBRANCH(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_gather_hetero_basic_raw_unbranch_bool_bf16
FFI_BG_HETERO_BASIC_RAW_UNBRANCH(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_gather_homo_basic_raw_unbranch_float_bf16
FFI_BG_HOMO_BASIC_RAW_UNBRANCH(_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmv_gather_hetero_basic_raw_unbranch_float_bf16
FFI_BG_HETERO_BASIC_RAW_UNBRANCH(_float_bf16, __nv_bfloat16, __nv_bfloat16)


// ----------------------------------------------------------------------------------------------------
///cuda_tempalte
///

template <typename T>
struct BgMrSpikeActive;

template <>
struct BgMrSpikeActive<uint8_t>
{
    __device__ __forceinline__ static bool eval(uint8_t v) { return IS_ACTIVE_BOOL(v); }
};

template <>
struct BgMrSpikeActive<float>
{
    __device__ __forceinline__ static bool eval(float v) { return IS_ACTIVE_F32(v); }
};

template <>
struct BgMrSpikeActive<double>
{
    __device__ __forceinline__ static bool eval(double v) { return IS_ACTIVE_F64(v); }
};

template <>
struct BgMrSpikeActive<__half>
{
    __device__ __forceinline__ static bool eval(__half v) { return IS_ACTIVE_F16(v); }
};

template <>
struct BgMrSpikeActive<__nv_bfloat16>
{
    __device__ __forceinline__ static bool eval(__nv_bfloat16 v) { return IS_ACTIVE_BF16(v); }
};

template <typename WeightT, typename AccT>
struct BgMrWeightIO;

template <>
struct BgMrWeightIO<float, float>
{
    __device__ __forceinline__ static float load(float v) { return READ_F32(v); }
    __device__ __forceinline__ static float store(float v) { return WRITE_F32(v); }
};

template <>
struct BgMrWeightIO<double, double>
{
    __device__ __forceinline__ static double load(double v) { return READ_F64(v); }
    __device__ __forceinline__ static double store(double v) { return WRITE_F64(v); }
};

template <>
struct BgMrWeightIO<__half, float>
{
    __device__ __forceinline__ static float load(__half v) { return READ_F16(v); }
    __device__ __forceinline__ static __half store(float v) { return WRITE_F16(v); }
};

template <>
struct BgMrWeightIO<__nv_bfloat16, float>
{
    __device__ __forceinline__ static float load(__nv_bfloat16 v) { return READ_BF16(v); }
    __device__ __forceinline__ static __nv_bfloat16 store(float v) { return WRITE_BF16(v); }
};

template <typename AccT>
__device__ __forceinline__ AccT bg_mr_fma(AccT a, AccT b, AccT c)
{
    return a * b + c;
}

template <>
__device__ __forceinline__ float bg_mr_fma<float>(float a, float b, float c)
{
    return __fmaf_rn(a, b, c);
}

template <>
__device__ __forceinline__ double bg_mr_fma<double>(double a, double b, double c)
{
    return fma(a, b, c);
}

template <typename AccT>
__device__ __forceinline__ AccT bg_mr_warp_reduce(AccT v);

template <>
__device__ __forceinline__ float bg_mr_warp_reduce<float>(float v)
{
    return warp_reduce_sum_f32(v);
}

template <>
__device__ __forceinline__ double bg_mr_warp_reduce<double>(double v)
{
    return warp_reduce_sum_f64(v);
}

template <int kUnroll, typename SpikeT, typename WeightT, typename AccT, typename SpikeActive, typename WeightIO>
__global__ void _bg_mr_kern_template_homo(
    const int32_t *__restrict__ indices,
    const SpikeT *__restrict__ spikes,
    WeightT *__restrict__ output,
    const WeightT *__restrict__ weights,
    int n_pre, int n_conn)
{
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= n_pre)
        return;

    int lane = threadIdx.x & 31;
    const int32_t *__restrict__ i_row = indices + (size_t)row * n_conn;
    AccT val = (AccT)0;

#pragma unroll kUnroll
    for (int k = lane; k < n_conn; k += 32)
    {
        int idx = __ldg(&i_row[k]);
        if (SpikeActive::eval(__ldg(&spikes[idx])))
            val += (AccT)1;
    }

    val = bg_mr_warp_reduce<AccT>(val);
    if (lane == 0)
    {
        AccT w0 = WeightIO::load(__ldg(&weights[0]));
        output[row] = WeightIO::store(w0 * val);
    }
}

template <int kUnroll, typename SpikeT, typename WeightT, typename AccT, typename SpikeActive, typename WeightIO>
__global__ void _bg_mr_kern_template_hetero(
    const int32_t *__restrict__ indices,
    const SpikeT *__restrict__ spikes,
    WeightT *__restrict__ output,
    const WeightT *__restrict__ weights,
    int n_pre, int n_conn)
{
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= n_pre)
        return;

    int lane = threadIdx.x & 31;
    const int32_t *__restrict__ i_row = indices + (size_t)row * n_conn;
    const WeightT *__restrict__ w_row = weights + (size_t)row * n_conn;
    AccT val = (AccT)0;

#pragma unroll kUnroll
    for (int k = lane; k < n_conn; k += 32)
    {
        int idx = __ldg(&i_row[k]);
        AccT w = WeightIO::load(__ldg(&w_row[k]));
        AccT mask = SpikeActive::eval(__ldg(&spikes[idx])) ? (AccT)1 : (AccT)0;
        val = bg_mr_fma<AccT>(w, mask, val);
    }

    val = bg_mr_warp_reduce<AccT>(val);
    if (lane == 0)
        output[row] = WeightIO::store(val);
}

template <typename WeightT, typename SpikeT, typename AccT>
static inline void launch_bg_mr_kern_template_homo(
    const int32_t *d_idx,
    const SpikeT *d_spk,
    WeightT *d_out,
    const WeightT *d_w,
    int n_pre,
    int n_conn,
    cudaStream_t s)
{
    const int bsz = 256;
    const int rpb = bsz >> 5;
    const int n_blocks = (n_pre + rpb - 1) / rpb;
    using Active = BgMrSpikeActive<SpikeT>;
    using WIO = BgMrWeightIO<WeightT, AccT>;
    _bg_mr_kern_template_homo<4, SpikeT, WeightT, AccT, Active, WIO><<<n_blocks, bsz, 0, s>>>(
        d_idx, d_spk, d_out, d_w, n_pre, n_conn);
}

template <typename WeightT, typename SpikeT, typename AccT>
static inline void launch_bg_mr_kern_template_hetero(
    const int32_t *d_idx,
    const SpikeT *d_spk,
    WeightT *d_out,
    const WeightT *d_w,
    int n_pre,
    int n_conn,
    cudaStream_t s)
{
    const int bsz = 256;
    const int rpb = bsz >> 5;
    const int n_blocks = (n_pre + rpb - 1) / rpb;
    using Active = BgMrSpikeActive<SpikeT>;
    using WIO = BgMrWeightIO<WeightT, AccT>;
    _bg_mr_kern_template_hetero<4, SpikeT, WeightT, AccT, Active, WIO><<<n_blocks, bsz, 0, s>>>(
        d_idx, d_spk, d_out, d_w, n_pre, n_conn);
}

#define FFI_BG_HOMO_BASIC_TEMPLATE(SUFFIX, WEIGHT_C_T, SPIKE_C_T, ACC_T)                          \
    void binary_fcnmv_gather_homo_basic_bg_mr_kern_template##SUFFIX(                               \
        const BE::Tensor weights, const BE::Tensor indices,                                         \
        const BE::Tensor spikes, BE::Tensor output, int64_t stream)                                 \
    {                                                                                                \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                    \
        int n_pre = static_cast<int>(indices.size(0));                                              \
        int n_conn = static_cast<int>(indices.size(1));                                             \
        const WEIGHT_C_T *d_w = static_cast<const WEIGHT_C_T *>(weights.data_ptr());               \
        const int32_t *d_idx = static_cast<const int32_t *>(indices.data_ptr());                   \
        const SPIKE_C_T *d_spk = static_cast<const SPIKE_C_T *>(spikes.data_ptr());                \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr());                          \
        launch_bg_mr_kern_template_homo<WEIGHT_C_T, SPIKE_C_T, ACC_T>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, s); \
    }

#define FFI_BG_HETERO_BASIC_TEMPLATE(SUFFIX, WEIGHT_C_T, SPIKE_C_T, ACC_T)                        \
    void binary_fcnmv_gather_hetero_basic_bg_mr_kern_template##SUFFIX(                             \
        const BE::Tensor weights, const BE::Tensor indices,                                         \
        const BE::Tensor spikes, BE::Tensor output, int64_t stream)                                 \
    {                                                                                                \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                    \
        int n_pre = static_cast<int>(indices.size(0));                                              \
        int n_conn = static_cast<int>(indices.size(1));                                             \
        const WEIGHT_C_T *d_w = static_cast<const WEIGHT_C_T *>(weights.data_ptr());               \
        const int32_t *d_idx = static_cast<const int32_t *>(indices.data_ptr());                   \
        const SPIKE_C_T *d_spk = static_cast<const SPIKE_C_T *>(spikes.data_ptr());                \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr());                          \
        launch_bg_mr_kern_template_hetero<WEIGHT_C_T, SPIKE_C_T, ACC_T>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, s); \
    }

// @BE binary_fcnmv_gather_homo_basic_bg_mr_kern_template_bool_f32
FFI_BG_HOMO_BASIC_TEMPLATE(_bool_f32, float, uint8_t, float)
// @BE binary_fcnmv_gather_hetero_basic_bg_mr_kern_template_bool_f32
FFI_BG_HETERO_BASIC_TEMPLATE(_bool_f32, float, uint8_t, float)
// @BE binary_fcnmv_gather_homo_basic_bg_mr_kern_template_float_f32
FFI_BG_HOMO_BASIC_TEMPLATE(_float_f32, float, float, float)
// @BE binary_fcnmv_gather_hetero_basic_bg_mr_kern_template_float_f32
FFI_BG_HETERO_BASIC_TEMPLATE(_float_f32, float, float, float)

// @BE binary_fcnmv_gather_homo_basic_bg_mr_kern_template_bool_f64
FFI_BG_HOMO_BASIC_TEMPLATE(_bool_f64, double, uint8_t, double)
// @BE binary_fcnmv_gather_hetero_basic_bg_mr_kern_template_bool_f64
FFI_BG_HETERO_BASIC_TEMPLATE(_bool_f64, double, uint8_t, double)
// @BE binary_fcnmv_gather_homo_basic_bg_mr_kern_template_float_f64
FFI_BG_HOMO_BASIC_TEMPLATE(_float_f64, double, double, double)
// @BE binary_fcnmv_gather_hetero_basic_bg_mr_kern_template_float_f64
FFI_BG_HETERO_BASIC_TEMPLATE(_float_f64, double, double, double)

// @BE binary_fcnmv_gather_homo_basic_bg_mr_kern_template_bool_f16
FFI_BG_HOMO_BASIC_TEMPLATE(_bool_f16, __half, uint8_t, float)
// @BE binary_fcnmv_gather_hetero_basic_bg_mr_kern_template_bool_f16
FFI_BG_HETERO_BASIC_TEMPLATE(_bool_f16, __half, uint8_t, float)
// @BE binary_fcnmv_gather_homo_basic_bg_mr_kern_template_float_f16
FFI_BG_HOMO_BASIC_TEMPLATE(_float_f16, __half, __half, float)
// @BE binary_fcnmv_gather_hetero_basic_bg_mr_kern_template_float_f16
FFI_BG_HETERO_BASIC_TEMPLATE(_float_f16, __half, __half, float)

// @BE binary_fcnmv_gather_homo_basic_bg_mr_kern_template_bool_bf16
FFI_BG_HOMO_BASIC_TEMPLATE(_bool_bf16, __nv_bfloat16, uint8_t, float)
// @BE binary_fcnmv_gather_hetero_basic_bg_mr_kern_template_bool_bf16
FFI_BG_HETERO_BASIC_TEMPLATE(_bool_bf16, __nv_bfloat16, uint8_t, float)
// @BE binary_fcnmv_gather_homo_basic_bg_mr_kern_template_float_bf16
FFI_BG_HOMO_BASIC_TEMPLATE(_float_bf16, __nv_bfloat16, __nv_bfloat16, float)
// @BE binary_fcnmv_gather_hetero_basic_bg_mr_kern_template_float_bf16
FFI_BG_HETERO_BASIC_TEMPLATE(_float_bf16, __nv_bfloat16, __nv_bfloat16, float)

// ----------------------------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------------------------
/// cuda_untail
///

#define DEFINE_BG_MR_UNTAIL_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
    __global__ void _bg_mr_homo_untail_kern##SUFFIX(                                                               \
        const int32_t *__restrict__ indices,                                                                \
        const SPIKE_T *__restrict__ spikes,                                                                 \
        WEIGHT_T *__restrict__ output,                                                                      \
        const WEIGHT_T *__restrict__ weights,                                                               \
        int n_pre, int n_conn)                                                                              \
    {                                                                                                       \
        int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);                                      \
        if (row >= n_pre)                                                                                   \
            return;                                                                                         \
        int lane = threadIdx.x & 31;                                                                        \
        const int32_t *i_row = indices + (size_t)row * n_conn;                                              \
        ACC_T val = ACC_ZERO;                                                                               \
        int n_conn_aligned = n_conn & ~31;                                                                  \
                                                                                                            \
        for (int k = lane; k < n_conn_aligned; k += 32)                                                     \
        {                                                                                                   \
            int idx = __ldg(&i_row[k]);                                                                     \
            if (IS_ACTIVE(__ldg(&spikes[idx])))                                                             \
                val += (ACC_T)1;                                                                            \
        }                                                                                                   \
                                                                                                            \
        int tail_idx = n_conn_aligned + lane;                                                               \
        if (tail_idx < n_conn)                                                                              \
        {                                                                                                   \
            int idx = __ldg(&i_row[tail_idx]);                                                              \
            if (IS_ACTIVE(__ldg(&spikes[idx])))                                                             \
                val += (ACC_T)1;                                                                            \
        }                                                                                                   \
                                                                                                            \
        val = WARP_RED(val);                                                                                \
        if (lane == 0)                                                                                      \
            output[row] = WRITE_W(READ_W(weights[0]) * val);                                                \
    }

#define DEFINE_BG_MR_UNTAIL_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
    __global__ void _bg_mr_hetero_untail_kern##SUFFIX(                                                               \
        const int32_t *__restrict__ indices,                                                                  \
        const SPIKE_T *__restrict__ spikes,                                                                   \
        WEIGHT_T *__restrict__ output,                                                                        \
        const WEIGHT_T *__restrict__ weights,                                                                 \
        int n_pre, int n_conn)                                                                                \
    {                                                                                                         \
        int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);                                        \
        if (row >= n_pre)                                                                                     \
            return;                                                                                           \
        int lane = threadIdx.x & 31;                                                                          \
        const int32_t *i_row = indices + (size_t)row * n_conn;                                                \
        const WEIGHT_T *w_row = weights + (size_t)row * n_conn;                                               \
        ACC_T val = ACC_ZERO;                                                                                 \
        int n_conn_aligned = n_conn & ~31;                                                                    \
                                                                                                              \
        for (int k = lane; k < n_conn_aligned; k += 32)                                                       \
        {                                                                                                     \
            int idx = __ldg(&i_row[k]);                                                                       \
            if (IS_ACTIVE(__ldg(&spikes[idx])))                                                               \
                val += READ_W(__ldg(&w_row[k]));                                                              \
        }                                                                                                     \
                                                                                                              \
        int tail_idx = n_conn_aligned + lane;                                                                 \
        if (tail_idx < n_conn)                                                                                \
        {                                                                                                     \
            int idx = __ldg(&i_row[tail_idx]);                                                                \
            if (IS_ACTIVE(__ldg(&spikes[idx])))                                                               \
                val += READ_W(__ldg(&w_row[tail_idx]));                                                       \
        }                                                                                                     \
                                                                                                              \
        val = WARP_RED(val);                                                                                  \
        if (lane == 0)                                                                                        \
            output[row] = WRITE_W(val);                                                                       \
    }

// ---- FFI macro: gather homo basic (multi-row) ----
#define FFI_BG_HOMO_UNTAIL_BASIC(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                            \
    void binary_fcnmv_gather_homo_untail_basic##SUFFIX(                                                    \
        const BE::Tensor weights, const BE::Tensor indices,                                         \
        const BE::Tensor spikes, BE::Tensor output, int64_t stream)                                 \
    {                                                                                               \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                    \
        int n_pre = static_cast<int>(indices.size(0));                                              \
        int n_conn = static_cast<int>(indices.size(1));                                             \
        const WEIGHT_C_T *d_w = static_cast<const WEIGHT_C_T *>(weights.data_ptr());                \
        const int32_t *d_idx = static_cast<const int32_t *>(indices.data_ptr());                    \
        const SPIKE_C_T *d_spk = static_cast<const SPIKE_C_T *>(spikes.data_ptr());                 \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr());                           \
        int bsz = 256;                                                                              \
        int rpb = bsz >> 5;                                                                         \
        int n_blocks = (n_pre + rpb - 1) / rpb;                                                     \
        _bg_mr_homo_untail_kern##SUFFIX<<<n_blocks, bsz, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn); \
    }

// ---- FFI macro: gather hetero basic (multi-row) ----
#define FFI_BG_HETERO_UNTAIL_BASIC(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                            \
    void binary_fcnmv_gather_hetero_untail_basic##SUFFIX(                                                    \
        const BE::Tensor weights, const BE::Tensor indices,                                           \
        const BE::Tensor spikes, BE::Tensor output, int64_t stream)                                   \
    {                                                                                                 \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                      \
        int n_pre = static_cast<int>(indices.size(0));                                                \
        int n_conn = static_cast<int>(indices.size(1));                                               \
        const WEIGHT_C_T *d_w = static_cast<const WEIGHT_C_T *>(weights.data_ptr());                  \
        const int32_t *d_idx = static_cast<const int32_t *>(indices.data_ptr());                      \
        const SPIKE_C_T *d_spk = static_cast<const SPIKE_C_T *>(spikes.data_ptr());                   \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr());                             \
        int bsz = 256;                                                                                \
        int rpb = bsz >> 5;                                                                           \
        int n_blocks = (n_pre + rpb - 1) / rpb;                                                       \
        _bg_mr_hetero_untail_kern##SUFFIX<<<n_blocks, bsz, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn); \
    }

DEFINE_BG_MR_UNTAIL_HOMO(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_UNTAIL_HETERO(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_UNTAIL_HOMO(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_UNTAIL_HETERO(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)

DEFINE_BG_MR_UNTAIL_HOMO(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_UNTAIL_HETERO(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_UNTAIL_HOMO(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_UNTAIL_HETERO(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)

DEFINE_BG_MR_UNTAIL_HOMO(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_UNTAIL_HETERO(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_UNTAIL_HOMO(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_UNTAIL_HETERO(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)

DEFINE_BG_MR_UNTAIL_HOMO(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_UNTAIL_HETERO(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_UNTAIL_HOMO(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_UNTAIL_HETERO(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)

// @BE binary_fcnmv_gather_homo_untail_basic_bool_f32
FFI_BG_HOMO_UNTAIL_BASIC(_bool_f32, float, uint8_t)
// @BE binary_fcnmv_gather_hetero_untail_basic_bool_f32
FFI_BG_HETERO_UNTAIL_BASIC(_bool_f32, float, uint8_t)
// @BE binary_fcnmv_gather_homo_untail_basic_float_f32
FFI_BG_HOMO_UNTAIL_BASIC(_float_f32, float, float)
// @BE binary_fcnmv_gather_hetero_untail_basic_float_f32
FFI_BG_HETERO_UNTAIL_BASIC(_float_f32, float, float)

// @BE binary_fcnmv_gather_homo_untail_basic_bool_f64
FFI_BG_HOMO_UNTAIL_BASIC(_bool_f64, double, uint8_t)
// @BE binary_fcnmv_gather_hetero_untail_basic_bool_f64
FFI_BG_HETERO_UNTAIL_BASIC(_bool_f64, double, uint8_t)
// @BE binary_fcnmv_gather_homo_untail_basic_float_f64
FFI_BG_HOMO_UNTAIL_BASIC(_float_f64, double, double)
// @BE binary_fcnmv_gather_hetero_untail_basic_float_f64
FFI_BG_HETERO_UNTAIL_BASIC(_float_f64, double, double)

// @BE binary_fcnmv_gather_homo_untail_basic_bool_f16
FFI_BG_HOMO_UNTAIL_BASIC(_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_gather_hetero_untail_basic_bool_f16
FFI_BG_HETERO_UNTAIL_BASIC(_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_gather_homo_untail_basic_float_f16
FFI_BG_HOMO_UNTAIL_BASIC(_float_f16, __half, __half)
// @BE binary_fcnmv_gather_hetero_untail_basic_float_f16
FFI_BG_HETERO_UNTAIL_BASIC(_float_f16, __half, __half)

// @BE binary_fcnmv_gather_homo_untail_basic_bool_bf16
FFI_BG_HOMO_UNTAIL_BASIC(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_gather_hetero_untail_basic_bool_bf16
FFI_BG_HETERO_UNTAIL_BASIC(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_gather_homo_untail_basic_float_bf16
FFI_BG_HOMO_UNTAIL_BASIC(_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmv_gather_hetero_untail_basic_float_bf16
FFI_BG_HETERO_UNTAIL_BASIC(_float_bf16, __nv_bfloat16, __nv_bfloat16)

// ----------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------
/// BASIC -> 128 / 32
///

#define DEFINE_BG_MR_HOMO_128_4(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
    __global__ void _bg_mr_homo_kern_128_4##SUFFIX(                                                               \
        const int32_t *__restrict__ indices,                                                                      \
        const SPIKE_T *__restrict__ spikes,                                                                       \
        WEIGHT_T *__restrict__ output,                                                                            \
        const WEIGHT_T *__restrict__ weights,                                                                     \
        int n_pre, int n_conn)                                                                                    \
    {                                                                                                             \
        int warp_id = threadIdx.x >> 5;                                                                           \
        int lane = threadIdx.x & 31;                                                                              \
        int row = (blockIdx.x * 4) + warp_id;                                                                     \
        if (row >= n_pre)                                                                                         \
            return;                                                                                               \
                                                                                                                  \
        const int32_t *i_row = indices + (size_t)row * n_conn;                                                    \
        ACC_T val = ACC_ZERO;                                                                                     \
                                                                                                                  \
        for (int k = lane; k < n_conn; k += 32)                                                                   \
        {                                                                                                         \
            int idx = __ldg(&i_row[k]);                                                                           \
            val += (ACC_T)(IS_ACTIVE(__ldg(&spikes[idx])));                                                       \
        }                                                                                                         \
                                                                                                                  \
        val = WARP_RED(val);                                                                                      \
        if (lane == 0)                                                                                            \
        {                                                                                                         \
            output[row] = WRITE_W(READ_W(weights[0]) * val);                                                      \
        }                                                                                                         \
    }

#define DEFINE_BG_MR_HETERO_128_4(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
    __global__ void _bg_mr_hetero_kern_128_4##SUFFIX(                                                               \
        const int32_t *__restrict__ indices,                                                                        \
        const SPIKE_T *__restrict__ spikes,                                                                         \
        WEIGHT_T *__restrict__ output,                                                                              \
        const WEIGHT_T *__restrict__ weights,                                                                       \
        int n_pre, int n_conn)                                                                                      \
    {                                                                                                               \
        int warp_id = threadIdx.x >> 5;                                                                             \
        int lane = threadIdx.x & 31;                                                                                \
        int row = (blockIdx.x * 4) + warp_id;                                                                       \
        if (row >= n_pre)                                                                                           \
            return;                                                                                                 \
                                                                                                                    \
        const int32_t *i_row = indices + (size_t)row * n_conn;                                                      \
        const WEIGHT_T *w_row = weights + (size_t)row * n_conn;                                                     \
        ACC_T val = ACC_ZERO;                                                                                       \
                                                                                                                    \
        for (int k = lane; k < n_conn; k += 32)                                                                     \
        {                                                                                                           \
            int idx = __ldg(&i_row[k]);                                                                             \
            val += (ACC_T)(IS_ACTIVE(__ldg(&spikes[idx]))) * READ_W(__ldg(&w_row[k]));                              \
        }                                                                                                           \
                                                                                                                    \
        val = WARP_RED(val);                                                                                        \
        if (lane == 0)                                                                                              \
        {                                                                                                           \
            output[row] = WRITE_W(val);                                                                             \
        }                                                                                                           \
    }

#define FFI_BG_HOMO_BASIC_128_4(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                            \
    void binary_fcnmv_gather_homo_128_4##SUFFIX(                                                          \
        const BE::Tensor weights, const BE::Tensor indices,                                               \
        const BE::Tensor spikes, BE::Tensor output, int64_t stream)                                       \
    {                                                                                                     \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                          \
        int n_pre = static_cast<int>(indices.size(0));                                                    \
        int n_conn = static_cast<int>(indices.size(1));                                                   \
        const WEIGHT_C_T *d_w = static_cast<const WEIGHT_C_T *>(weights.data_ptr());                      \
        const int32_t *d_idx = static_cast<const int32_t *>(indices.data_ptr());                          \
        const SPIKE_C_T *d_spk = static_cast<const SPIKE_C_T *>(spikes.data_ptr());                       \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr());                                 \
                                                                                                          \
        int bsz = 128;                                                                                    \
        int rows_per_block = 4;                                                                           \
        int n_blocks = (n_pre + rows_per_block - 1) / rows_per_block;                                     \
        _bg_mr_homo_kern_128_4##SUFFIX<<<n_blocks, bsz, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn); \
    }

#define FFI_BG_HETERO_BASIC_128_4(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                            \
    void binary_fcnmv_gather_hetero_128_4##SUFFIX(                                                          \
        const BE::Tensor weights, const BE::Tensor indices,                                                 \
        const BE::Tensor spikes, BE::Tensor output, int64_t stream)                                         \
    {                                                                                                       \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                            \
        int n_pre = static_cast<int>(indices.size(0));                                                      \
        int n_conn = static_cast<int>(indices.size(1));                                                     \
        const WEIGHT_C_T *d_w = static_cast<const WEIGHT_C_T *>(weights.data_ptr());                        \
        const int32_t *d_idx = static_cast<const int32_t *>(indices.data_ptr());                            \
        const SPIKE_C_T *d_spk = static_cast<const SPIKE_C_T *>(spikes.data_ptr());                         \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr());                                   \
                                                                                                            \
        int bsz = 128;                                                                                      \
        int rows_per_block = 4;                                                                             \
        int n_blocks = (n_pre + rows_per_block - 1) / rows_per_block;                                       \
        _bg_mr_hetero_kern_128_4##SUFFIX<<<n_blocks, bsz, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn); \
    }

DEFINE_BG_MR_HOMO_128_4(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_128_4(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO_128_4(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_128_4(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)

DEFINE_BG_MR_HOMO_128_4(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HETERO_128_4(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HOMO_128_4(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HETERO_128_4(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)

DEFINE_BG_MR_HOMO_128_4(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_128_4(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO_128_4(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_128_4(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)

DEFINE_BG_MR_HOMO_128_4(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_128_4(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO_128_4(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_128_4(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)

// @BE binary_fcnmv_gather_homo_128_4_bool_f32
FFI_BG_HOMO_BASIC_128_4(_bool_f32, float, uint8_t)
// @BE binary_fcnmv_gather_hetero_128_4_bool_f32
FFI_BG_HETERO_BASIC_128_4(_bool_f32, float, uint8_t)
// @BE binary_fcnmv_gather_homo_128_4_float_f32
FFI_BG_HOMO_BASIC_128_4(_float_f32, float, float)
// @BE binary_fcnmv_gather_hetero_128_4_float_f32
FFI_BG_HETERO_BASIC_128_4(_float_f32, float, float)

// @BE binary_fcnmv_gather_homo_128_4_bool_f64
FFI_BG_HOMO_BASIC_128_4(_bool_f64, double, uint8_t)
// @BE binary_fcnmv_gather_hetero_128_4_bool_f64
FFI_BG_HETERO_BASIC_128_4(_bool_f64, double, uint8_t)
// @BE binary_fcnmv_gather_homo_128_4_float_f64
FFI_BG_HOMO_BASIC_128_4(_float_f64, double, double)
// @BE binary_fcnmv_gather_hetero_128_4_float_f64
FFI_BG_HETERO_BASIC_128_4(_float_f64, double, double)

// @BE binary_fcnmv_gather_homo_128_4_bool_f16
FFI_BG_HOMO_BASIC_128_4(_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_gather_hetero_128_4_bool_f16
FFI_BG_HETERO_BASIC_128_4(_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_gather_homo_128_4_float_f16
FFI_BG_HOMO_BASIC_128_4(_float_f16, __half, __half)
// @BE binary_fcnmv_gather_hetero_128_4_float_f16
FFI_BG_HETERO_BASIC_128_4(_float_f16, __half, __half)

// @BE binary_fcnmv_gather_homo_128_4_bool_bf16
FFI_BG_HOMO_BASIC_128_4(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_gather_hetero_128_4_bool_bf16
FFI_BG_HETERO_BASIC_128_4(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_gather_homo_128_4_float_bf16
FFI_BG_HOMO_BASIC_128_4(_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmv_gather_hetero_128_4_float_bf16
FFI_BG_HETERO_BASIC_128_4(_float_bf16, __nv_bfloat16, __nv_bfloat16)

// ----------------------------------------------------------------------------------------------------
/// BASIC -> 256 / 8 (256 threads per block, 32 threads per row -> 8 rows per block)
///

#define DEFINE_BG_MR_HOMO_256_8(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
    __global__ void _bg_mr_homo_kern_256_8##SUFFIX(                                                               \
        const int32_t *__restrict__ indices,                                                                      \
        const SPIKE_T *__restrict__ spikes,                                                                       \
        WEIGHT_T *__restrict__ output,                                                                            \
        const WEIGHT_T *__restrict__ weights,                                                                     \
        int n_pre, int n_conn)                                                                                    \
    {                                                                                                             \
        int warp_id = threadIdx.x >> 5;                                                                           \
        int lane = threadIdx.x & 31;                                                                              \
        int row = (blockIdx.x * 8) + warp_id;                                                                     \
        if (row >= n_pre)                                                                                         \
            return;                                                                                               \
                                                                                                                  \
        const int32_t *i_row = indices + (size_t)row * n_conn;                                                    \
        ACC_T val = ACC_ZERO;                                                                                     \
                                                                                                                  \
        for (int k = lane; k < n_conn; k += 32)                                                                   \
        {                                                                                                         \
            int idx = __ldg(&i_row[k]);                                                                           \
            SPIKE_T spk_val = __ldg(&spikes[idx]);                                                                \
            if (IS_ACTIVE(spk_val))                                                                               \
            {                                                                                                     \
                val += (ACC_T)(IS_ACTIVE(spk_val));                                                               \
            }                                                                                                     \
        }                                                                                                         \
                                                                                                                  \
        val = WARP_RED(val);                                                                                      \
        if (lane == 0)                                                                                            \
        {                                                                                                         \
            output[row] = WRITE_W(READ_W(weights[0]) * val);                                                      \
        }                                                                                                         \
    }

#define DEFINE_BG_MR_HETERO_256_8(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
    __global__ void _bg_mr_hetero_kern_256_8##SUFFIX(                                                               \
        const int32_t *__restrict__ indices,                                                                        \
        const SPIKE_T *__restrict__ spikes,                                                                         \
        WEIGHT_T *__restrict__ output,                                                                              \
        const WEIGHT_T *__restrict__ weights,                                                                       \
        int n_pre, int n_conn)                                                                                      \
    {                                                                                                               \
        int warp_id = threadIdx.x >> 5;                                                                             \
        int lane = threadIdx.x & 31;                                                                                \
        int row = (blockIdx.x * 8) + warp_id;                                                                       \
        if (row >= n_pre)                                                                                           \
            return;                                                                                                 \
                                                                                                                    \
        const int32_t *i_row = indices + (size_t)row * n_conn;                                                      \
        const WEIGHT_T *w_row = weights + (size_t)row * n_conn;                                                     \
        ACC_T val = ACC_ZERO;                                                                                       \
                                                                                                                    \
        for (int k = lane; k < n_conn; k += 32)                                                                     \
        {                                                                                                           \
            int idx = __ldg(&i_row[k]);                                                                             \
            SPIKE_T spk_val = __ldg(&spikes[idx]);                                                                  \
            if (IS_ACTIVE(spk_val))                                                                                 \
            {                                                                                                       \
                val += (ACC_T)(IS_ACTIVE(spk_val)) * READ_W(__ldg(&w_row[k]));                                      \
            }                                                                                                       \
        }                                                                                                           \
                                                                                                                    \
        val = WARP_RED(val);                                                                                        \
        if (lane == 0)                                                                                              \
        {                                                                                                           \
            output[row] = WRITE_W(val);                                                                             \
        }                                                                                                           \
    }

#define FFI_BG_HOMO_BASIC_256_8(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                            \
    void binary_fcnmv_gather_homo_256_8##SUFFIX(                                                          \
        const BE::Tensor weights, const BE::Tensor indices,                                               \
        const BE::Tensor spikes, BE::Tensor output, int64_t stream)                                       \
    {                                                                                                     \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                          \
        int n_pre = static_cast<int>(indices.size(0));                                                    \
        int n_conn = static_cast<int>(indices.size(1));                                                   \
        const WEIGHT_C_T *d_w = static_cast<const WEIGHT_C_T *>(weights.data_ptr());                      \
        const int32_t *d_idx = static_cast<const int32_t *>(indices.data_ptr());                          \
        const SPIKE_C_T *d_spk = static_cast<const SPIKE_C_T *>(spikes.data_ptr());                       \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr());                                 \
                                                                                                          \
        int bsz = 256;                                                                                    \
        int rows_per_block = 8;                                                                           \
        int n_blocks = (n_pre + rows_per_block - 1) / rows_per_block;                                     \
        _bg_mr_homo_kern_256_8##SUFFIX<<<n_blocks, bsz, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn); \
    }

#define FFI_BG_HETERO_BASIC_256_8(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                            \
    void binary_fcnmv_gather_hetero_256_8##SUFFIX(                                                          \
        const BE::Tensor weights, const BE::Tensor indices,                                                 \
        const BE::Tensor spikes, BE::Tensor output, int64_t stream)                                         \
    {                                                                                                       \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                            \
        int n_pre = static_cast<int>(indices.size(0));                                                      \
        int n_conn = static_cast<int>(indices.size(1));                                                     \
        const WEIGHT_C_T *d_w = static_cast<const WEIGHT_C_T *>(weights.data_ptr());                        \
        const int32_t *d_idx = static_cast<const int32_t *>(indices.data_ptr());                            \
        const SPIKE_C_T *d_spk = static_cast<const SPIKE_C_T *>(spikes.data_ptr());                         \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr());                                   \
                                                                                                            \
        int bsz = 256;                                                                                      \
        int rows_per_block = 8;                                                                             \
        int n_blocks = (n_pre + rows_per_block - 1) / rows_per_block;                                       \
        _bg_mr_hetero_kern_256_8##SUFFIX<<<n_blocks, bsz, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn); \
    }

DEFINE_BG_MR_HOMO_256_8(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_256_8(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO_256_8(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_256_8(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)

DEFINE_BG_MR_HOMO_256_8(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HETERO_256_8(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HOMO_256_8(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HETERO_256_8(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)

DEFINE_BG_MR_HOMO_256_8(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_256_8(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO_256_8(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_256_8(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)

DEFINE_BG_MR_HOMO_256_8(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_256_8(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO_256_8(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_256_8(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)

// @BE binary_fcnmv_gather_homo_256_8_bool_f32
FFI_BG_HOMO_BASIC_256_8(_bool_f32, float, uint8_t)
// @BE binary_fcnmv_gather_hetero_256_8_bool_f32
FFI_BG_HETERO_BASIC_256_8(_bool_f32, float, uint8_t)
// @BE binary_fcnmv_gather_homo_256_8_float_f32
FFI_BG_HOMO_BASIC_256_8(_float_f32, float, float)
// @BE binary_fcnmv_gather_hetero_256_8_float_f32
FFI_BG_HETERO_BASIC_256_8(_float_f32, float, float)

// @BE binary_fcnmv_gather_homo_256_8_bool_f64
FFI_BG_HOMO_BASIC_256_8(_bool_f64, double, uint8_t)
// @BE binary_fcnmv_gather_hetero_256_8_bool_f64
FFI_BG_HETERO_BASIC_256_8(_bool_f64, double, uint8_t)
// @BE binary_fcnmv_gather_homo_256_8_float_f64
FFI_BG_HOMO_BASIC_256_8(_float_f64, double, double)
// @BE binary_fcnmv_gather_hetero_256_8_float_f64
FFI_BG_HETERO_BASIC_256_8(_float_f64, double, double)

// @BE binary_fcnmv_gather_homo_256_8_bool_f16
FFI_BG_HOMO_BASIC_256_8(_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_gather_hetero_256_8_bool_f16
FFI_BG_HETERO_BASIC_256_8(_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_gather_homo_256_8_float_f16
FFI_BG_HOMO_BASIC_256_8(_float_f16, __half, __half)
// @BE binary_fcnmv_gather_hetero_256_8_float_f16
FFI_BG_HETERO_BASIC_256_8(_float_f16, __half, __half)

// @BE binary_fcnmv_gather_homo_256_8_bool_bf16
FFI_BG_HOMO_BASIC_256_8(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_gather_hetero_256_8_bool_bf16
FFI_BG_HETERO_BASIC_256_8(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_gather_homo_256_8_float_bf16
FFI_BG_HOMO_BASIC_256_8(_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmv_gather_hetero_256_8_float_bf16
FFI_BG_HETERO_BASIC_256_8(_float_bf16, __nv_bfloat16, __nv_bfloat16)

// ----------------------------------------------------------------------------------------------------
/// BASIC -> 256 / 64
///

#define DEFINE_BG_MR_HOMO_256_4(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
    __global__ void _bg_mr_homo_kern_256_4##SUFFIX(                                                               \
        const int32_t *__restrict__ indices,                                                                      \
        const SPIKE_T *__restrict__ spikes,                                                                       \
        WEIGHT_T *__restrict__ output,                                                                            \
        const WEIGHT_T *__restrict__ weights,                                                                     \
        int n_pre, int n_conn)                                                                                    \
    {                                                                                                             \
        /* 状态变更: 将 Block 划分为 4 个 Group，每个 Group (64 线程 = 2 Warp) 负责 1 行 */                       \
        int group_id = threadIdx.x >> 6;                                                                          \
        int local_id = threadIdx.x & 63;                                                                          \
        int warp_in_group = local_id >> 5;                                                                        \
        int lane = local_id & 31;                                                                                 \
        /* 状态流转: 全局行索引 = Block基础偏移 (每个Block 4行) + Group局部偏移 */                                \
        int row = (blockIdx.x * 4) + group_id;                                                                    \
        if (row >= n_pre)                                                                                         \
            return;                                                                                               \
                                                                                                                  \
        const int32_t *i_row = indices + (size_t)row * n_conn;                                                    \
        ACC_T val = ACC_ZERO;                                                                                     \
                                                                                                                  \
        /* 数据流: 64 线程步长遍历，双 Warp 并行覆盖列 */                                                         \
        for (int k = local_id; k < n_conn; k += 64)                                                               \
        {                                                                                                         \
            int idx = __ldg(&i_row[k]);                                                                           \
            val += (ACC_T)(IS_ACTIVE(__ldg(&spikes[idx])));                                                       \
        }                                                                                                         \
                                                                                                                  \
        /* 状态聚合: 先 Warp 内规约，再通过共享内存跨 Warp 合并 */                                                \
        val = WARP_RED(val);                                                                                      \
        __shared__ ACC_T smem_256_4[8];                                                                           \
        if (lane == 0)                                                                                            \
        {                                                                                                         \
            smem_256_4[group_id * 2 + warp_in_group] = val;                                                       \
        }                                                                                                         \
        __syncthreads();                                                                                          \
        if (local_id == 0)                                                                                        \
        {                                                                                                         \
            ACC_T sum = smem_256_4[group_id * 2] + smem_256_4[group_id * 2 + 1];                                  \
            output[row] = WRITE_W(READ_W(weights[0]) * sum);                                                      \
        }                                                                                                         \
    }

#define DEFINE_BG_MR_HETERO_256_4(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
    __global__ void _bg_mr_hetero_kern_256_4##SUFFIX(                                                               \
        const int32_t *__restrict__ indices,                                                                        \
        const SPIKE_T *__restrict__ spikes,                                                                         \
        WEIGHT_T *__restrict__ output,                                                                              \
        const WEIGHT_T *__restrict__ weights,                                                                       \
        int n_pre, int n_conn)                                                                                      \
    {                                                                                                               \
        /* 状态变更: 将 Block 划分为 4 个 Group，每个 Group (64 线程 = 2 Warp) 负责 1 行 */                         \
        int group_id = threadIdx.x >> 6;                                                                            \
        int local_id = threadIdx.x & 63;                                                                            \
        int warp_in_group = local_id >> 5;                                                                          \
        int lane = local_id & 31;                                                                                   \
        /* 状态流转: 全局行索引 = Block基础偏移 (每个Block 4行) + Group局部偏移 */                                  \
        int row = (blockIdx.x * 4) + group_id;                                                                      \
        if (row >= n_pre)                                                                                           \
            return;                                                                                                 \
                                                                                                                    \
        const int32_t *i_row = indices + (size_t)row * n_conn;                                                      \
        const WEIGHT_T *w_row = weights + (size_t)row * n_conn;                                                     \
        ACC_T val = ACC_ZERO;                                                                                       \
                                                                                                                    \
        /* 数据流: 64 线程步长遍历，乘法掩码合并权重提取 */                                                         \
        for (int k = local_id; k < n_conn; k += 64)                                                                 \
        {                                                                                                           \
            int idx = __ldg(&i_row[k]);                                                                             \
            val += (ACC_T)(IS_ACTIVE(__ldg(&spikes[idx]))) * READ_W(__ldg(&w_row[k]));                              \
        }                                                                                                           \
                                                                                                                    \
        /* 状态聚合: 先 Warp 内规约，再通过共享内存跨 Warp 合并 */                                                  \
        val = WARP_RED(val);                                                                                        \
        __shared__ ACC_T smem_256_4[8];                                                                             \
        if (lane == 0)                                                                                              \
        {                                                                                                           \
            smem_256_4[group_id * 2 + warp_in_group] = val;                                                         \
        }                                                                                                           \
        __syncthreads();                                                                                            \
        if (local_id == 0)                                                                                          \
        {                                                                                                           \
            ACC_T sum = smem_256_4[group_id * 2] + smem_256_4[group_id * 2 + 1];                                    \
            output[row] = WRITE_W(sum);                                                                             \
        }                                                                                                           \
    }

#define FFI_BG_HOMO_BASIC_256_4(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                            \
    void binary_fcnmv_gather_homo_256_4##SUFFIX(                                                          \
        const BE::Tensor weights, const BE::Tensor indices,                                               \
        const BE::Tensor spikes, BE::Tensor output, int64_t stream)                                       \
    {                                                                                                     \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                          \
        int n_pre = static_cast<int>(indices.size(0));                                                    \
        int n_conn = static_cast<int>(indices.size(1));                                                   \
        const WEIGHT_C_T *d_w = static_cast<const WEIGHT_C_T *>(weights.data_ptr());                      \
        const int32_t *d_idx = static_cast<const int32_t *>(indices.data_ptr());                          \
        const SPIKE_C_T *d_spk = static_cast<const SPIKE_C_T *>(spikes.data_ptr());                       \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr());                                 \
                                                                                                          \
        /* 状态控制流: 强制单 Block = 256 线程，行批处理量 = 4 (每行64线程) */                            \
        int bsz = 256;                                                                                    \
        int rows_per_block = 4;                                                                           \
        int n_blocks = (n_pre + rows_per_block - 1) / rows_per_block;                                     \
        _bg_mr_homo_kern_256_4##SUFFIX<<<n_blocks, bsz, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn); \
    }

#define FFI_BG_HETERO_BASIC_256_4(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                            \
    void binary_fcnmv_gather_hetero_256_4##SUFFIX(                                                          \
        const BE::Tensor weights, const BE::Tensor indices,                                                 \
        const BE::Tensor spikes, BE::Tensor output, int64_t stream)                                         \
    {                                                                                                       \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                            \
        int n_pre = static_cast<int>(indices.size(0));                                                      \
        int n_conn = static_cast<int>(indices.size(1));                                                     \
        const WEIGHT_C_T *d_w = static_cast<const WEIGHT_C_T *>(weights.data_ptr());                        \
        const int32_t *d_idx = static_cast<const int32_t *>(indices.data_ptr());                            \
        const SPIKE_C_T *d_spk = static_cast<const SPIKE_C_T *>(spikes.data_ptr());                         \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr());                                   \
                                                                                                            \
        /* 状态控制流: 强制单 Block = 256 线程，行批处理量 = 4 (每行64线程) */                              \
        int bsz = 256;                                                                                      \
        int rows_per_block = 4;                                                                             \
        int n_blocks = (n_pre + rows_per_block - 1) / rows_per_block;                                       \
        _bg_mr_hetero_kern_256_4##SUFFIX<<<n_blocks, bsz, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn); \
    }

DEFINE_BG_MR_HOMO_256_4(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_256_4(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO_256_4(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_256_4(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)

DEFINE_BG_MR_HOMO_256_4(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HETERO_256_4(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HOMO_256_4(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HETERO_256_4(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)

DEFINE_BG_MR_HOMO_256_4(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_256_4(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO_256_4(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_256_4(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)

DEFINE_BG_MR_HOMO_256_4(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_256_4(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO_256_4(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_256_4(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)

// @BE binary_fcnmv_gather_homo_256_4_bool_f32
FFI_BG_HOMO_BASIC_256_4(_bool_f32, float, uint8_t)
// @BE binary_fcnmv_gather_hetero_256_4_bool_f32
FFI_BG_HETERO_BASIC_256_4(_bool_f32, float, uint8_t)
// @BE binary_fcnmv_gather_homo_256_4_float_f32
FFI_BG_HOMO_BASIC_256_4(_float_f32, float, float)
// @BE binary_fcnmv_gather_hetero_256_4_float_f32
FFI_BG_HETERO_BASIC_256_4(_float_f32, float, float)

// @BE binary_fcnmv_gather_homo_256_4_bool_f64
FFI_BG_HOMO_BASIC_256_4(_bool_f64, double, uint8_t)
// @BE binary_fcnmv_gather_hetero_256_4_bool_f64
FFI_BG_HETERO_BASIC_256_4(_bool_f64, double, uint8_t)
// @BE binary_fcnmv_gather_homo_256_4_float_f64
FFI_BG_HOMO_BASIC_256_4(_float_f64, double, double)
// @BE binary_fcnmv_gather_hetero_256_4_float_f64
FFI_BG_HETERO_BASIC_256_4(_float_f64, double, double)

// @BE binary_fcnmv_gather_homo_256_4_bool_f16
FFI_BG_HOMO_BASIC_256_4(_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_gather_hetero_256_4_bool_f16
FFI_BG_HETERO_BASIC_256_4(_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_gather_homo_256_4_float_f16
FFI_BG_HOMO_BASIC_256_4(_float_f16, __half, __half)
// @BE binary_fcnmv_gather_hetero_256_4_float_f16
FFI_BG_HETERO_BASIC_256_4(_float_f16, __half, __half)

// @BE binary_fcnmv_gather_homo_256_4_bool_bf16
FFI_BG_HOMO_BASIC_256_4(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_gather_hetero_256_4_bool_bf16
FFI_BG_HETERO_BASIC_256_4(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_gather_homo_256_4_float_bf16
FFI_BG_HOMO_BASIC_256_4(_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmv_gather_hetero_256_4_float_bf16
FFI_BG_HETERO_BASIC_256_4(_float_bf16, __nv_bfloat16, __nv_bfloat16)

// ----------------------------------------------------------------------------------------------------
/// BASIC -> 128 / 64
///

#define DEFINE_BG_MR_HOMO_128_2(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
    __global__ void _bg_mr_homo_kern_128_2##SUFFIX(                                                               \
        const int32_t *__restrict__ indices,                                                                      \
        const SPIKE_T *__restrict__ spikes,                                                                       \
        WEIGHT_T *__restrict__ output,                                                                            \
        const WEIGHT_T *__restrict__ weights,                                                                     \
        int n_pre, int n_conn)                                                                                    \
    {                                                                                                             \
        /* 状态变更: 将 Block 划分为 2 个 Group，每个 Group (64 线程 = 2 Warp) 负责 1 行 */                       \
        int group_id = threadIdx.x >> 6;                                                                          \
        int local_id = threadIdx.x & 63;                                                                          \
        int warp_in_group = local_id >> 5;                                                                        \
        int lane = local_id & 31;                                                                                 \
        /* 状态流转: 全局行索引 = Block基础偏移 (每个Block 2行) + Group局部偏移 */                                \
        int row = (blockIdx.x * 2) + group_id;                                                                    \
        if (row >= n_pre)                                                                                         \
            return;                                                                                               \
                                                                                                                  \
        const int32_t *i_row = indices + (size_t)row * n_conn;                                                    \
        ACC_T val = ACC_ZERO;                                                                                     \
                                                                                                                  \
        /* 数据流: 64 线程步长遍历，双 Warp 并行覆盖列 */                                                         \
        for (int k = local_id; k < n_conn; k += 64)                                                               \
        {                                                                                                         \
            int idx = __ldg(&i_row[k]);                                                                           \
            val += (ACC_T)(IS_ACTIVE(__ldg(&spikes[idx])));                                                       \
        }                                                                                                         \
                                                                                                                  \
        /* 状态聚合: 先 Warp 内规约，再通过共享内存跨 Warp 合并 */                                                \
        val = WARP_RED(val);                                                                                      \
        __shared__ ACC_T smem_128_2[4];                                                                           \
        if (lane == 0)                                                                                            \
        {                                                                                                         \
            smem_128_2[group_id * 2 + warp_in_group] = val;                                                       \
        }                                                                                                         \
        __syncthreads();                                                                                          \
        if (local_id == 0)                                                                                        \
        {                                                                                                         \
            ACC_T sum = smem_128_2[group_id * 2] + smem_128_2[group_id * 2 + 1];                                  \
            output[row] = WRITE_W(READ_W(weights[0]) * sum);                                                      \
        }                                                                                                         \
    }

#define DEFINE_BG_MR_HETERO_128_2(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
    __global__ void _bg_mr_hetero_kern_128_2##SUFFIX(                                                               \
        const int32_t *__restrict__ indices,                                                                        \
        const SPIKE_T *__restrict__ spikes,                                                                         \
        WEIGHT_T *__restrict__ output,                                                                              \
        const WEIGHT_T *__restrict__ weights,                                                                       \
        int n_pre, int n_conn)                                                                                      \
    {                                                                                                               \
        /* 状态变更: 将 Block 划分为 2 个 Group，每个 Group (64 线程 = 2 Warp) 负责 1 行 */                         \
        int group_id = threadIdx.x >> 6;                                                                            \
        int local_id = threadIdx.x & 63;                                                                            \
        int warp_in_group = local_id >> 5;                                                                          \
        int lane = local_id & 31;                                                                                   \
        /* 状态流转: 全局行索引 = Block基础偏移 (每个Block 2行) + Group局部偏移 */                                  \
        int row = (blockIdx.x * 2) + group_id;                                                                      \
        if (row >= n_pre)                                                                                           \
            return;                                                                                                 \
                                                                                                                    \
        const int32_t *i_row = indices + (size_t)row * n_conn;                                                      \
        const WEIGHT_T *w_row = weights + (size_t)row * n_conn;                                                     \
        ACC_T val = ACC_ZERO;                                                                                       \
                                                                                                                    \
        /* 数据流: 64 线程步长遍历，乘法掩码合并权重提取 */                                                         \
        for (int k = local_id; k < n_conn; k += 64)                                                                 \
        {                                                                                                           \
            int idx = __ldg(&i_row[k]);                                                                             \
            val += (ACC_T)(IS_ACTIVE(__ldg(&spikes[idx]))) * READ_W(__ldg(&w_row[k]));                              \
        }                                                                                                           \
                                                                                                                    \
        /* 状态聚合: 先 Warp 内规约，再通过共享内存跨 Warp 合并 */                                                  \
        val = WARP_RED(val);                                                                                        \
        __shared__ ACC_T smem_128_2[4];                                                                             \
        if (lane == 0)                                                                                              \
        {                                                                                                           \
            smem_128_2[group_id * 2 + warp_in_group] = val;                                                         \
        }                                                                                                           \
        __syncthreads();                                                                                            \
        if (local_id == 0)                                                                                          \
        {                                                                                                           \
            ACC_T sum = smem_128_2[group_id * 2] + smem_128_2[group_id * 2 + 1];                                    \
            output[row] = WRITE_W(sum);                                                                             \
        }                                                                                                           \
    }

#define FFI_BG_HOMO_BASIC_128_2(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                            \
    void binary_fcnmv_gather_homo_128_2##SUFFIX(                                                          \
        const BE::Tensor weights, const BE::Tensor indices,                                               \
        const BE::Tensor spikes, BE::Tensor output, int64_t stream)                                       \
    {                                                                                                     \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                          \
        int n_pre = static_cast<int>(indices.size(0));                                                    \
        int n_conn = static_cast<int>(indices.size(1));                                                   \
        const WEIGHT_C_T *d_w = static_cast<const WEIGHT_C_T *>(weights.data_ptr());                      \
        const int32_t *d_idx = static_cast<const int32_t *>(indices.data_ptr());                          \
        const SPIKE_C_T *d_spk = static_cast<const SPIKE_C_T *>(spikes.data_ptr());                       \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr());                                 \
                                                                                                          \
        /* 状态控制流: 强制单 Block = 128 线程，行批处理量 = 2 (每行64线程) */                            \
        int bsz = 128;                                                                                    \
        int rows_per_block = 2;                                                                           \
        int n_blocks = (n_pre + rows_per_block - 1) / rows_per_block;                                     \
        _bg_mr_homo_kern_128_2##SUFFIX<<<n_blocks, bsz, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn); \
    }

#define FFI_BG_HETERO_BASIC_128_2(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                            \
    void binary_fcnmv_gather_hetero_128_2##SUFFIX(                                                          \
        const BE::Tensor weights, const BE::Tensor indices,                                                 \
        const BE::Tensor spikes, BE::Tensor output, int64_t stream)                                         \
    {                                                                                                       \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                            \
        int n_pre = static_cast<int>(indices.size(0));                                                      \
        int n_conn = static_cast<int>(indices.size(1));                                                     \
        const WEIGHT_C_T *d_w = static_cast<const WEIGHT_C_T *>(weights.data_ptr());                        \
        const int32_t *d_idx = static_cast<const int32_t *>(indices.data_ptr());                            \
        const SPIKE_C_T *d_spk = static_cast<const SPIKE_C_T *>(spikes.data_ptr());                         \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr());                                   \
                                                                                                            \
        /* 状态控制流: 强制单 Block = 128 线程，行批处理量 = 2 (每行64线程) */                              \
        int bsz = 128;                                                                                      \
        int rows_per_block = 2;                                                                             \
        int n_blocks = (n_pre + rows_per_block - 1) / rows_per_block;                                       \
        _bg_mr_hetero_kern_128_2##SUFFIX<<<n_blocks, bsz, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn); \
    }

DEFINE_BG_MR_HOMO_128_2(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_128_2(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO_128_2(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_128_2(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)

DEFINE_BG_MR_HOMO_128_2(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HETERO_128_2(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HOMO_128_2(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HETERO_128_2(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)

DEFINE_BG_MR_HOMO_128_2(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_128_2(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO_128_2(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_128_2(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)

DEFINE_BG_MR_HOMO_128_2(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_128_2(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO_128_2(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_128_2(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)

// @BE binary_fcnmv_gather_homo_128_2_bool_f32
FFI_BG_HOMO_BASIC_128_2(_bool_f32, float, uint8_t)
// @BE binary_fcnmv_gather_hetero_128_2_bool_f32
FFI_BG_HETERO_BASIC_128_2(_bool_f32, float, uint8_t)
// @BE binary_fcnmv_gather_homo_128_2_float_f32
FFI_BG_HOMO_BASIC_128_2(_float_f32, float, float)
// @BE binary_fcnmv_gather_hetero_128_2_float_f32
FFI_BG_HETERO_BASIC_128_2(_float_f32, float, float)

// @BE binary_fcnmv_gather_homo_128_2_bool_f64
FFI_BG_HOMO_BASIC_128_2(_bool_f64, double, uint8_t)
// @BE binary_fcnmv_gather_hetero_128_2_bool_f64
FFI_BG_HETERO_BASIC_128_2(_bool_f64, double, uint8_t)
// @BE binary_fcnmv_gather_homo_128_2_float_f64
FFI_BG_HOMO_BASIC_128_2(_float_f64, double, double)
// @BE binary_fcnmv_gather_hetero_128_2_float_f64
FFI_BG_HETERO_BASIC_128_2(_float_f64, double, double)

// @BE binary_fcnmv_gather_homo_128_2_bool_f16
FFI_BG_HOMO_BASIC_128_2(_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_gather_hetero_128_2_bool_f16
FFI_BG_HETERO_BASIC_128_2(_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_gather_homo_128_2_float_f16
FFI_BG_HOMO_BASIC_128_2(_float_f16, __half, __half)
// @BE binary_fcnmv_gather_hetero_128_2_float_f16
FFI_BG_HETERO_BASIC_128_2(_float_f16, __half, __half)

// @BE binary_fcnmv_gather_homo_128_2_bool_bf16
FFI_BG_HOMO_BASIC_128_2(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_gather_hetero_128_2_bool_bf16
FFI_BG_HETERO_BASIC_128_2(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_gather_homo_128_2_float_bf16
FFI_BG_HOMO_BASIC_128_2(_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmv_gather_hetero_128_2_float_bf16
FFI_BG_HETERO_BASIC_128_2(_float_bf16, __nv_bfloat16, __nv_bfloat16)

//------------------------------------------------------------------------------------------

#define DEFINE_BG_MR_L2_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _bg_mr_L2_homo_kern##SUFFIX(                                                                 \
    const int32_t* __restrict__ indices,                                                                  \
    const SPIKE_T* __restrict__ spikes,                                                                   \
    WEIGHT_T* __restrict__ output,                                                                   \
    const WEIGHT_T* __restrict__ weights,                                                                 \
    int n_pre, int n_conn                                                                                 \
) {                                                                                                       \
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);                                        \
    if (row >= n_pre) return;                                                                             \
    int lane = threadIdx.x & 31;                                                                          \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                                \
    ACC_T val = ACC_ZERO;                                                                                 \
    for (int k = lane; k < n_conn; k += 32) {                                                             \
        int idx = __ldg(&i_row[k]);                                                                       \
        SPIKE_T spk_val = __ldcg(&spikes[idx]);                                                           \
        if (IS_ACTIVE(spk_val)) {                                                                         \
            val += (ACC_T)1;                                                                              \
        }                                                                                                 \
    }                                                                                                     \
    val = WARP_RED(val);                                                                                  \
    if (lane == 0)                                                                                        \
        output[row] = WRITE_W(READ_W(weights[0]) * val);                                                  \
}

#define DEFINE_BG_MR_L2_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _bg_mr_L2_hetero_kern##SUFFIX(                                                                 \
    const int32_t* __restrict__ indices,                                                                    \
    const SPIKE_T* __restrict__ spikes,                                                                     \
    WEIGHT_T* __restrict__ output,                                                                     \
    const WEIGHT_T* __restrict__ weights,                                                                   \
    int n_pre, int n_conn                                                                                   \
) {                                                                                                         \
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);                                          \
    if (row >= n_pre) return;                                                                               \
    int lane = threadIdx.x & 31;                                                                            \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                                  \
    const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                                 \
    ACC_T val = ACC_ZERO;                                                                                   \
    for (int k = lane; k < n_conn; k += 32) {                                                               \
        int idx = __ldg(&i_row[k]);                                                                         \
        /* 状态变更: 强制旁路 L1 缓存以获取离散 spikes */                                                 \
        SPIKE_T spk_val = __ldcg(&spikes[idx]);                                                             \
        if (IS_ACTIVE(spk_val)) {                                                                           \
            /* 只有脉冲处于激活状态时，才去触发权重访存及后续计算 */                                      \
            val += READ_W(__ldg(&w_row[k]));                                                                \
        }                                                                                                   \
    }                                                                                                       \
    val = WARP_RED(val);                                                                                    \
    if (lane == 0)                                                                                          \
        output[row] = WRITE_W(val);                                                                         \
}

// ---- FFI macro: gather homo basic (multi-row) ----
#define FFI_BG_HOMO_L2_BASIC(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                        \
void binary_fcnmv_gather_homo_L2_basic##SUFFIX(                                                    \
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
    _bg_mr_L2_homo_kern##SUFFIX<<<n_blocks, bsz, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn); \
}

// ---- FFI macro: gather hetero basic (multi-row) ----
#define FFI_BG_HETERO_L2_BASIC(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                        \
void binary_fcnmv_gather_hetero_L2_basic##SUFFIX(                                                    \
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
    _bg_mr_L2_hetero_kern##SUFFIX<<<n_blocks, bsz, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn); \
}


DEFINE_BG_MR_L2_HOMO     (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_L2_HETERO   (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_L2_HOMO     (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_L2_HETERO   (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)

DEFINE_BG_MR_L2_HOMO     (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_L2_HETERO   (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_L2_HOMO     (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_L2_HETERO   (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)


DEFINE_BG_MR_L2_HOMO     (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_L2_HETERO   (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_L2_HOMO     (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_L2_HETERO   (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)

DEFINE_BG_MR_L2_HOMO     (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_L2_HETERO   (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_L2_HOMO     (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_L2_HETERO   (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)


// @BE binary_fcnmv_gather_homo_L2_basic_bool_f32
FFI_BG_HOMO_L2_BASIC (_bool_f32, float, uint8_t)
// @BE binary_fcnmv_gather_hetero_L2_basic_bool_f32
FFI_BG_HETERO_L2_BASIC(_bool_f32, float, uint8_t)
// @BE binary_fcnmv_gather_homo_L2_basic_float_f32
FFI_BG_HOMO_L2_BASIC (_float_f32, float, float)
// @BE binary_fcnmv_gather_hetero_L2_basic_float_f32
FFI_BG_HETERO_L2_BASIC(_float_f32, float, float)

// @BE binary_fcnmv_gather_homo_L2_basic_bool_f64
FFI_BG_HOMO_L2_BASIC (_bool_f64, double, uint8_t)
// @BE binary_fcnmv_gather_hetero_L2_basic_bool_f64
FFI_BG_HETERO_L2_BASIC(_bool_f64, double, uint8_t)
// @BE binary_fcnmv_gather_homo_L2_basic_float_f64
FFI_BG_HOMO_L2_BASIC (_float_f64, double, double)
// @BE binary_fcnmv_gather_hetero_L2_basic_float_f64
FFI_BG_HETERO_L2_BASIC(_float_f64, double, double)

// @BE binary_fcnmv_gather_homo_L2_basic_bool_f16
FFI_BG_HOMO_L2_BASIC (_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_gather_hetero_L2_basic_bool_f16
FFI_BG_HETERO_L2_BASIC(_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_gather_homo_L2_basic_float_f16
FFI_BG_HOMO_L2_BASIC (_float_f16, __half, __half)
// @BE binary_fcnmv_gather_hetero_L2_basic_float_f16
FFI_BG_HETERO_L2_BASIC(_float_f16, __half, __half)

// @BE binary_fcnmv_gather_homo_L2_basic_bool_bf16
FFI_BG_HOMO_L2_BASIC (_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_gather_hetero_L2_basic_bool_bf16
FFI_BG_HETERO_L2_BASIC(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_gather_homo_L2_basic_float_bf16
FFI_BG_HOMO_L2_BASIC (_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmv_gather_hetero_L2_basic_float_bf16
FFI_BG_HETERO_L2_BASIC(_float_bf16, __nv_bfloat16, __nv_bfloat16)

// ----------------------------------------------------------------------------------------------------
/// cuda_BITPACK

__global__ void _pack_bool_to_bits_kern(
    const uint8_t *__restrict__ spikes,
    uint32_t *__restrict__ binary_spikes,
    int n_spikes)
{
    int word_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base = word_idx << 5;
    if (base >= n_spikes)
        return;
    uint32_t packed = 0u;
    int limit = min(32, n_spikes - base);
    for (int b = 0; b < limit; ++b)
    {
        if (__ldg(&spikes[base + b]))
            packed |= (1u << b);
    }
    binary_spikes[word_idx] = packed;
}

// ---- Bitpack gather kernel macros (bool spikes bit-compressed) ----
#define DEFINE_BG_MR_HOMO_BITPACK(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
    __global__ void _bg_mr_homo_bitpack_kern##SUFFIX(                                           \
        const int32_t *__restrict__ indices,                                                    \
        const uint32_t *__restrict__ binary_spikes,                                             \
        WEIGHT_T *__restrict__ output,                                                          \
        const WEIGHT_T *__restrict__ weights,                                                   \
        int n_pre, int n_conn)                                                                  \
    {                                                                                           \
        int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);                          \
        if (row >= n_pre)                                                                       \
            return;                                                                             \
        int lane = threadIdx.x & 31;                                                            \
        const int32_t *i_row = indices + (size_t)row * n_conn;                                  \
        ACC_T val = ACC_ZERO;                                                                   \
        for (int k = lane; k < n_conn; k += 32)                                                 \
        {                                                                                       \
            int idx = __ldg(&i_row[k]);                                                         \
            if ((__ldg(&binary_spikes[idx >> 5]) >> (idx & 31)) & 1u)                           \
                val += (ACC_T)1;                                                                \
        }                                                                                       \
        val = WARP_RED(val);                                                                    \
        if (lane == 0)                                                                          \
            output[row] = WRITE_W(READ_W(weights[0]) * val);                                    \
    }

#define DEFINE_BG_MR_HETERO_BITPACK(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
    __global__ void _bg_mr_hetero_bitpack_kern##SUFFIX(                                           \
        const int32_t *__restrict__ indices,                                                      \
        const uint32_t *__restrict__ binary_spikes,                                               \
        WEIGHT_T *__restrict__ output,                                                            \
        const WEIGHT_T *__restrict__ weights,                                                     \
        int n_pre, int n_conn)                                                                    \
    {                                                                                             \
        int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);                            \
        if (row >= n_pre)                                                                         \
            return;                                                                               \
        int lane = threadIdx.x & 31;                                                              \
        const int32_t *i_row = indices + (size_t)row * n_conn;                                    \
        const WEIGHT_T *w_row = weights + (size_t)row * n_conn;                                   \
        ACC_T val = ACC_ZERO;                                                                     \
        for (int k = lane; k < n_conn; k += 32)                                                   \
        {                                                                                         \
            int idx = __ldg(&i_row[k]);                                                           \
            if ((__ldg(&binary_spikes[idx >> 5]) >> (idx & 31)) & 1u)                             \
                val += READ_W(__ldg(&w_row[k]));                                                  \
        }                                                                                         \
        val = WARP_RED(val);                                                                      \
        if (lane == 0)                                                                            \
            output[row] = WRITE_W(val);                                                           \
    }

// ---- FFI macro: gather homo basic BITPACK (bool spikes, multi-row) ----
#define FFI_BG_HOMO_BASIC_BITPACK(SUFFIX, WEIGHT_C_T)                                                        \
    void binary_fcnmv_gather_homo_basic_bit##SUFFIX(                                                         \
        const BE::Tensor weights, const BE::Tensor indices,                                                  \
        const BE::Tensor spikes, BE::Tensor output, int64_t stream)                                          \
    {                                                                                                        \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                             \
        int n_pre = static_cast<int>(indices.size(0));                                                       \
        int n_conn = static_cast<int>(indices.size(1));                                                      \
        int n_spikes = static_cast<int>(spikes.size(0));                                                     \
        int n_words = (n_spikes + 31) >> 5;                                                                  \
        const WEIGHT_C_T *d_w = static_cast<const WEIGHT_C_T *>(weights.data_ptr());                         \
        const int32_t *d_idx = static_cast<const int32_t *>(indices.data_ptr());                             \
        const uint8_t *d_spk = static_cast<const uint8_t *>(spikes.data_ptr());                              \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr());                                    \
        uint32_t *d_bspk = nullptr;                                                                          \
        cudaMallocAsync(&d_bspk, (size_t)n_words * sizeof(uint32_t), s);                                     \
        {                                                                                                    \
            int pb = 256;                                                                                    \
            _pack_bool_to_bits_kern<<<(n_words + pb - 1) / pb, pb, 0, s>>>(d_spk, d_bspk, n_spikes);         \
        }                                                                                                    \
        int bsz = 256;                                                                                       \
        int rpb = bsz >> 5;                                                                                  \
        int n_blocks = (n_pre + rpb - 1) / rpb;                                                              \
        _bg_mr_homo_bitpack_kern##SUFFIX<<<n_blocks, bsz, 0, s>>>(d_idx, d_bspk, d_out, d_w, n_pre, n_conn); \
        cudaFreeAsync(d_bspk, s);                                                                            \
    }

// ---- FFI macro: gather hetero basic BITPACK (bool spikes, multi-row) ----
#define FFI_BG_HETERO_BASIC_BITPACK(SUFFIX, WEIGHT_C_T)                                                        \
    void binary_fcnmv_gather_hetero_basic_bit##SUFFIX(                                                         \
        const BE::Tensor weights, const BE::Tensor indices,                                                    \
        const BE::Tensor spikes, BE::Tensor output, int64_t stream)                                            \
    {                                                                                                          \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                               \
        int n_pre = static_cast<int>(indices.size(0));                                                         \
        int n_conn = static_cast<int>(indices.size(1));                                                        \
        int n_spikes = static_cast<int>(spikes.size(0));                                                       \
        int n_words = (n_spikes + 31) >> 5;                                                                    \
        const WEIGHT_C_T *d_w = static_cast<const WEIGHT_C_T *>(weights.data_ptr());                           \
        const int32_t *d_idx = static_cast<const int32_t *>(indices.data_ptr());                               \
        const uint8_t *d_spk = static_cast<const uint8_t *>(spikes.data_ptr());                                \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr());                                      \
        uint32_t *d_bspk = nullptr;                                                                            \
        cudaMallocAsync(&d_bspk, (size_t)n_words * sizeof(uint32_t), s);                                       \
        {                                                                                                      \
            int pb = 256;                                                                                      \
            _pack_bool_to_bits_kern<<<(n_words + pb - 1) / pb, pb, 0, s>>>(d_spk, d_bspk, n_spikes);           \
        }                                                                                                      \
        int bsz = 256;                                                                                         \
        int rpb = bsz >> 5;                                                                                    \
        int n_blocks = (n_pre + rpb - 1) / rpb;                                                                \
        _bg_mr_hetero_bitpack_kern##SUFFIX<<<n_blocks, bsz, 0, s>>>(d_idx, d_bspk, d_out, d_w, n_pre, n_conn); \
        cudaFreeAsync(d_bspk, s);                                                                              \
    }

DEFINE_BG_MR_HOMO_BITPACK(_bool_f32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_BITPACK(_bool_f32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO_BITPACK(_bool_f64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HETERO_BITPACK(_bool_f64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HOMO_BITPACK(_bool_f16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_BITPACK(_bool_f16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO_BITPACK(_bool_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_BITPACK(_bool_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)

// @BE binary_fcnmv_gather_homo_basic_bit_bool_f32
FFI_BG_HOMO_BASIC_BITPACK(_bool_f32, float)
// @BE binary_fcnmv_gather_hetero_basic_bit_bool_f32
FFI_BG_HETERO_BASIC_BITPACK(_bool_f32, float)

// @BE binary_fcnmv_gather_homo_basic_bit_bool_f64
FFI_BG_HOMO_BASIC_BITPACK(_bool_f64, double)
// @BE binary_fcnmv_gather_hetero_basic_bit_bool_f64
FFI_BG_HETERO_BASIC_BITPACK(_bool_f64, double)

// @BE binary_fcnmv_gather_homo_basic_bit_bool_f16
FFI_BG_HOMO_BASIC_BITPACK(_bool_f16, __half)
// @BE binary_fcnmv_gather_hetero_basic_bit_bool_f16
FFI_BG_HETERO_BASIC_BITPACK(_bool_f16, __half)

// @BE binary_fcnmv_gather_homo_basic_bit_bool_bf16
FFI_BG_HOMO_BASIC_BITPACK(_bool_bf16, __nv_bfloat16)
// @BE binary_fcnmv_gather_hetero_basic_bit_bool_bf16
FFI_BG_HETERO_BASIC_BITPACK(_bool_bf16, __nv_bfloat16)

// ----------------------------------------------------------------------------------------------------