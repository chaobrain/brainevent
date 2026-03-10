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
/// NT WARP -> the 1 t 1 r
///

#define DEFINE_BG_THREAD_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
    __global__ void _bg_thread_homo_kern##SUFFIX(                                                               \
        const int32_t *__restrict__ indices,                                                                    \
        const SPIKE_T *__restrict__ spikes,                                                                     \
        WEIGHT_T *__restrict__ output,                                                                          \
        const WEIGHT_T *__restrict__ weights,                                                                   \
        int n_pre, int n_conn)                                                                                  \
    {                                                                                                           \
        /* 状态变更: 线程索引映射从 blockIdx.x 更改为全局线程 ID */                                             \
        int row = blockIdx.x * blockDim.x + threadIdx.x;                                                        \
        if (row >= n_pre)                                                                                       \
            return;                                                                                             \
                                                                                                                \
        const int32_t *i_row = indices + (size_t)row * n_conn;                                                  \
        ACC_T val = ACC_ZERO;                                                                                   \
                                                                                                                \
        /* 状态变更: 消除 Warp 规约，转为单线程内部的寄存器状态累加 */                                          \
        for (int c = 0; c < n_conn; ++c)                                                                        \
        {                                                                                                       \
            int idx = __ldg(&i_row[c]);                                                                         \
            if (IS_ACTIVE(__ldg(&spikes[idx])))                                                                 \
            {                                                                                                   \
                val += (ACC_T)1;                                                                                \
            }                                                                                                   \
        }                                                                                                       \
        /* 数据流: 计算完成后直接写入全局内存 */                                                                \
        output[row] = WRITE_W(READ_W(weights[0]) * val);                                                        \
    }

#define DEFINE_BG_THREAD_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
    __global__ void _bg_thread_hetero_kern##SUFFIX(                                                               \
        const int32_t *__restrict__ indices,                                                                      \
        const SPIKE_T *__restrict__ spikes,                                                                       \
        WEIGHT_T *__restrict__ output,                                                                            \
        const WEIGHT_T *__restrict__ weights,                                                                     \
        int n_pre, int n_conn)                                                                                    \
    {                                                                                                             \
        /* 状态变更: 线程索引映射从 blockIdx.x 更改为全局线程 ID */                                               \
        int row = blockIdx.x * blockDim.x + threadIdx.x;                                                          \
        if (row >= n_pre)                                                                                         \
            return;                                                                                               \
                                                                                                                  \
        const int32_t *i_row = indices + (size_t)row * n_conn;                                                    \
        const WEIGHT_T *w_row = weights + (size_t)row * n_conn;                                                   \
        ACC_T val = ACC_ZERO;                                                                                     \
                                                                                                                  \
        /* 状态变更: 消除 Warp 规约，转为单线程内部的寄存器状态累加 */                                            \
        for (int c = 0; c < n_conn; ++c)                                                                          \
        {                                                                                                         \
            int idx = __ldg(&i_row[c]);                                                                           \
            if (IS_ACTIVE(__ldg(&spikes[idx])))                                                                   \
            {                                                                                                     \
                val += READ_W(__ldg(&w_row[c]));                                                                  \
            }                                                                                                     \
        }                                                                                                         \
        /* 数据流: 计算完成后直接写入全局内存 */                                                                  \
        output[row] = WRITE_W(val);                                                                               \
    }

DEFINE_BG_THREAD_HOMO(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_THREAD_HETERO(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_THREAD_HOMO(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_THREAD_HETERO(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)

DEFINE_BG_THREAD_HOMO(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_THREAD_HETERO(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_THREAD_HOMO(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_THREAD_HETERO(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)

DEFINE_BG_THREAD_HOMO(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_THREAD_HETERO(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_THREAD_HOMO(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_THREAD_HETERO(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)

DEFINE_BG_THREAD_HOMO(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_THREAD_HETERO(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_THREAD_HOMO(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_THREAD_HETERO(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)

#define FFI_BG_HOMO_THREAD(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                                 \
    void binary_fcnmv_gather_homo_thread##SUFFIX(                                                         \
        const BE::Tensor weights,                                                                         \
        const BE::Tensor indices,                                                                         \
        const BE::Tensor spikes,                                                                          \
        BE::Tensor output, int64_t stream)                                                                \
    {                                                                                                     \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                          \
        int n_pre = static_cast<int>(indices.size(0));                                                    \
        int n_conn = static_cast<int>(indices.size(1));                                                   \
        const WEIGHT_C_T *d_w = static_cast<const WEIGHT_C_T *>(weights.data_ptr());                      \
        const int32_t *d_idx = static_cast<const int32_t *>(indices.data_ptr());                          \
        const SPIKE_C_T *d_spk = static_cast<const SPIKE_C_T *>(spikes.data_ptr());                       \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr());                                 \
        int threads = 32;                                                                                 \
        int blocks = (n_pre + threads - 1) / threads;                                                     \
        _bg_thread_homo_kern##SUFFIX<<<blocks, threads, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn); \
    }

// ---- FFI macro: gather hetero warp ----
#define FFI_BG_HETERO_THREAD(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                                 \
    void binary_fcnmv_gather_hetero_thread##SUFFIX(                                                         \
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
        int threads = 32;                                                                                   \
        int blocks = (n_pre + threads - 1) / threads;                                                       \
        _bg_thread_hetero_kern##SUFFIX<<<blocks, threads, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn); \
    }

// @BE binary_fcnmv_gather_homo_thread_bool_f32
FFI_BG_HOMO_THREAD(_bool_f32, float, uint8_t)
// @BE binary_fcnmv_gather_hetero_thread_bool_f32
FFI_BG_HETERO_THREAD(_bool_f32, float, uint8_t)
// @BE binary_fcnmv_gather_homo_thread_float_f32
FFI_BG_HOMO_THREAD(_float_f32, float, float)
// @BE binary_fcnmv_gather_hetero_thread_float_f32
FFI_BG_HETERO_THREAD(_float_f32, float, float)

// @BE binary_fcnmv_gather_homo_thread_bool_f64
FFI_BG_HOMO_THREAD(_bool_f64, double, uint8_t)
// @BE binary_fcnmv_gather_hetero_thread_bool_f64
FFI_BG_HETERO_THREAD(_bool_f64, double, uint8_t)
// @BE binary_fcnmv_gather_homo_thread_float_f64
FFI_BG_HOMO_THREAD(_float_f64, double, double)
// @BE binary_fcnmv_gather_hetero_thread_float_f64
FFI_BG_HETERO_THREAD(_float_f64, double, double)

// @BE binary_fcnmv_gather_homo_thread_bool_f16
FFI_BG_HOMO_THREAD(_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_gather_hetero_thread_bool_f16
FFI_BG_HETERO_THREAD(_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_gather_homo_thread_float_f16
FFI_BG_HOMO_THREAD(_float_f16, __half, __half)
// @BE binary_fcnmv_gather_hetero_thread_float_f16
FFI_BG_HETERO_THREAD(_float_f16, __half, __half)

// @BE binary_fcnmv_gather_homo_thread_bool_bf16
FFI_BG_HOMO_THREAD(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_gather_hetero_thread_bool_bf16
FFI_BG_HETERO_THREAD(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_gather_homo_thread_float_bf16
FFI_BG_HOMO_THREAD(_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmv_gather_hetero_thread_float_bf16
FFI_BG_HETERO_THREAD(_float_bf16, __nv_bfloat16, __nv_bfloat16)

// ----------------------------------------------------------------------------------------------------
/// NT WARP -> the 1 t 1 r (Unroll 4)
///

#define DEFINE_BG_THREAD_UNROLL4_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
    __global__ void _bg_thread_unroll4_hetero_kern##SUFFIX(                                                               \
        const int32_t *__restrict__ indices,                                                                              \
        const SPIKE_T *__restrict__ spikes,                                                                               \
        WEIGHT_T *__restrict__ output,                                                                                    \
        const WEIGHT_T *__restrict__ weights,                                                                             \
        int n_pre, int n_conn)                                                                                            \
    {                                                                                                                     \
        int row = blockIdx.x * blockDim.x + threadIdx.x;                                                                  \
        if (row >= n_pre)                                                                                                 \
            return;                                                                                                       \
                                                                                                                          \
        const int32_t *i_row = indices + (size_t)row * n_conn;                                                            \
        const WEIGHT_T *w_row = weights + (size_t)row * n_conn;                                                           \
        ACC_T val = ACC_ZERO;                                                                                             \
                                                                                                                          \
        int c = 0;                                                                                                        \
        int n_conn_aligned = n_conn & ~3;                                                                                 \
                                                                                                                          \
        for (; c < n_conn_aligned; c += 4)                                                                                \
        {                                                                                                                 \
            /* 状态变更: 连续发起 4 个索引读取指令 */                                                                     \
            int idx0 = __ldg(&i_row[c]);                                                                                  \
            int idx1 = __ldg(&i_row[c + 1]);                                                                              \
            int idx2 = __ldg(&i_row[c + 2]);                                                                              \
            int idx3 = __ldg(&i_row[c + 3]);                                                                              \
                                                                                                                          \
            /* 状态变更: 无条件发起 4 个权重读取指令，数据流向寄存器 */                                                   \
            WEIGHT_T w0 = __ldg(&w_row[c]);                                                                               \
            WEIGHT_T w1 = __ldg(&w_row[c + 1]);                                                                           \
            WEIGHT_T w2 = __ldg(&w_row[c + 2]);                                                                           \
            WEIGHT_T w3 = __ldg(&w_row[c + 3]);                                                                           \
                                                                                                                          \
            /* 状态变更: 读取 spike 状态，基于谓词执行累加指令，消耗寄存器中的权重数据 */                                 \
            if (IS_ACTIVE(__ldg(&spikes[idx0])))                                                                          \
                val += READ_W(w0);                                                                                        \
            if (IS_ACTIVE(__ldg(&spikes[idx1])))                                                                          \
                val += READ_W(w1);                                                                                        \
            if (IS_ACTIVE(__ldg(&spikes[idx2])))                                                                          \
                val += READ_W(w2);                                                                                        \
            if (IS_ACTIVE(__ldg(&spikes[idx3])))                                                                          \
                val += READ_W(w3);                                                                                        \
        }                                                                                                                 \
                                                                                                                          \
        /* 状态流转: 处理尾部边界 */                                                                                      \
        for (; c < n_conn; ++c)                                                                                           \
        {                                                                                                                 \
            int idx = __ldg(&i_row[c]);                                                                                   \
            WEIGHT_T w = __ldg(&w_row[c]); /* 同样先将数据加载至寄存器 */                                                 \
            if (IS_ACTIVE(__ldg(&spikes[idx])))                                                                           \
            {                                                                                                             \
                val += READ_W(w);                                                                                         \
            }                                                                                                             \
        }                                                                                                                 \
        output[row] = WRITE_W(val);                                                                                       \
    }

#define DEFINE_BG_THREAD_UNROLL4_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
    __global__ void _bg_thread_unroll4_homo_kern##SUFFIX(                                                               \
        const int32_t *__restrict__ indices,                                                                            \
        const SPIKE_T *__restrict__ spikes,                                                                             \
        WEIGHT_T *__restrict__ output,                                                                                  \
        const WEIGHT_T *__restrict__ weights,                                                                           \
        int n_pre, int n_conn)                                                                                          \
    {                                                                                                                   \
        int row = blockIdx.x * blockDim.x + threadIdx.x;                                                                \
        if (row >= n_pre)                                                                                               \
            return;                                                                                                     \
                                                                                                                        \
        const int32_t *i_row = indices + (size_t)row * n_conn;                                                          \
        ACC_T val = ACC_ZERO;                                                                                           \
                                                                                                                        \
        /* 状态流转: 主循环，步长为 4 进行批量处理 */                                                                   \
        int c = 0;                                                                                                      \
        int n_conn_aligned = n_conn & ~3; /* 计算向下取整到 4 的倍数 */                                                 \
                                                                                                                        \
        for (; c < n_conn_aligned; c += 4)                                                                              \
        {                                                                                                               \
            /* 状态变更: 连续发起 4 个内存读取指令，目标状态暂存至寄存器 */                                             \
            int idx0 = __ldg(&i_row[c]);                                                                                \
            int idx1 = __ldg(&i_row[c + 1]);                                                                            \
            int idx2 = __ldg(&i_row[c + 2]);                                                                            \
            int idx3 = __ldg(&i_row[c + 3]);                                                                            \
                                                                                                                        \
            /* 状态变更: 独立的读后写累加操作，解除指令间依赖 */                                                        \
            if (IS_ACTIVE(__ldg(&spikes[idx0])))                                                                        \
                val += (ACC_T)1;                                                                                        \
            if (IS_ACTIVE(__ldg(&spikes[idx1])))                                                                        \
                val += (ACC_T)1;                                                                                        \
            if (IS_ACTIVE(__ldg(&spikes[idx2])))                                                                        \
                val += (ACC_T)1;                                                                                        \
            if (IS_ACTIVE(__ldg(&spikes[idx3])))                                                                        \
                val += (ACC_T)1;                                                                                        \
        }                                                                                                               \
                                                                                                                        \
        /* 状态流转: 处理尾部剩余的列（0-3 列） */                                                                      \
        for (; c < n_conn; ++c)                                                                                         \
        {                                                                                                               \
            int idx = __ldg(&i_row[c]);                                                                                 \
            if (IS_ACTIVE(__ldg(&spikes[idx])))                                                                         \
            {                                                                                                           \
                val += (ACC_T)1;                                                                                        \
            }                                                                                                           \
        }                                                                                                               \
        output[row] = WRITE_W(READ_W(weights[0]) * val);                                                                \
    }



DEFINE_BG_THREAD_UNROLL4_HOMO(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_THREAD_UNROLL4_HETERO(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_THREAD_UNROLL4_HOMO(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_THREAD_UNROLL4_HETERO(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)

DEFINE_BG_THREAD_UNROLL4_HOMO(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_THREAD_UNROLL4_HETERO(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_THREAD_UNROLL4_HOMO(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_THREAD_UNROLL4_HETERO(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)

DEFINE_BG_THREAD_UNROLL4_HOMO(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_THREAD_UNROLL4_HETERO(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_THREAD_UNROLL4_HOMO(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_THREAD_UNROLL4_HETERO(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)

DEFINE_BG_THREAD_UNROLL4_HOMO(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_THREAD_UNROLL4_HETERO(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_THREAD_UNROLL4_HOMO(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_THREAD_UNROLL4_HETERO(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)

#define FFI_BG_HOMO_THREAD_UNROLL4(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                                 \
    void binary_fcnmv_gather_homo_thread_unroll4##SUFFIX(                                                         \
        const BE::Tensor weights,                                                                                 \
        const BE::Tensor indices,                                                                                 \
        const BE::Tensor spikes,                                                                                  \
        BE::Tensor output, int64_t stream)                                                                        \
    {                                                                                                             \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                                  \
        int n_pre = static_cast<int>(indices.size(0));                                                            \
        int n_conn = static_cast<int>(indices.size(1));                                                           \
        const WEIGHT_C_T *d_w = static_cast<const WEIGHT_C_T *>(weights.data_ptr());                              \
        const int32_t *d_idx = static_cast<const int32_t *>(indices.data_ptr());                                  \
        const SPIKE_C_T *d_spk = static_cast<const SPIKE_C_T *>(spikes.data_ptr());                               \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr());                                         \
        int threads = 256; /* 状态变更: 提升 Block 内线程数以提高 Occupancy */                                    \
        int blocks = (n_pre + threads - 1) / threads;                                                             \
        _bg_thread_unroll4_homo_kern##SUFFIX<<<blocks, threads, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn); \
    }

// ---- FFI macro: gather hetero warp ----
#define FFI_BG_HETERO_THREAD_UNROLL4(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                                 \
    void binary_fcnmv_gather_hetero_thread_unroll4##SUFFIX(                                                         \
        const BE::Tensor weights, const BE::Tensor indices,                                                         \
        const BE::Tensor spikes, BE::Tensor output, int64_t stream)                                                 \
    {                                                                                                               \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                                    \
        int n_pre = static_cast<int>(indices.size(0));                                                              \
        int n_conn = static_cast<int>(indices.size(1));                                                             \
        const WEIGHT_C_T *d_w = static_cast<const WEIGHT_C_T *>(weights.data_ptr());                                \
        const int32_t *d_idx = static_cast<const int32_t *>(indices.data_ptr());                                    \
        const SPIKE_C_T *d_spk = static_cast<const SPIKE_C_T *>(spikes.data_ptr());                                 \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr());                                           \
        int threads = 256; /* 状态变更: 提升 Block 内线程数以提高 Occupancy */                                      \
        int blocks = (n_pre + threads - 1) / threads;                                                               \
        _bg_thread_unroll4_hetero_kern##SUFFIX<<<blocks, threads, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn); \
    }

// @BE binary_fcnmv_gather_homo_thread_unroll4_bool_f32
FFI_BG_HOMO_THREAD_UNROLL4(_bool_f32, float, uint8_t)
// @BE binary_fcnmv_gather_hetero_thread_unroll4_bool_f32
FFI_BG_HETERO_THREAD_UNROLL4(_bool_f32, float, uint8_t)
// @BE binary_fcnmv_gather_homo_thread_unroll4_float_f32
FFI_BG_HOMO_THREAD_UNROLL4(_float_f32, float, float)
// @BE binary_fcnmv_gather_hetero_thread_unroll4_float_f32
FFI_BG_HETERO_THREAD_UNROLL4(_float_f32, float, float)

// @BE binary_fcnmv_gather_homo_thread_unroll4_bool_f64
FFI_BG_HOMO_THREAD_UNROLL4(_bool_f64, double, uint8_t)
// @BE binary_fcnmv_gather_hetero_thread_unroll4_bool_f64
FFI_BG_HETERO_THREAD_UNROLL4(_bool_f64, double, uint8_t)
// @BE binary_fcnmv_gather_homo_thread_unroll4_float_f64
FFI_BG_HOMO_THREAD_UNROLL4(_float_f64, double, double)
// @BE binary_fcnmv_gather_hetero_thread_unroll4_float_f64
FFI_BG_HETERO_THREAD_UNROLL4(_float_f64, double, double)

// @BE binary_fcnmv_gather_homo_thread_unroll4_bool_f16
FFI_BG_HOMO_THREAD_UNROLL4(_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_gather_hetero_thread_unroll4_bool_f16
FFI_BG_HETERO_THREAD_UNROLL4(_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_gather_homo_thread_unroll4_float_f16
FFI_BG_HOMO_THREAD_UNROLL4(_float_f16, __half, __half)
// @BE binary_fcnmv_gather_hetero_thread_unroll4_float_f16
FFI_BG_HETERO_THREAD_UNROLL4(_float_f16, __half, __half)

// @BE binary_fcnmv_gather_homo_thread_unroll4_bool_bf16
FFI_BG_HOMO_THREAD_UNROLL4(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_gather_hetero_thread_unroll4_bool_bf16
FFI_BG_HETERO_THREAD_UNROLL4(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_gather_homo_thread_unroll4_float_bf16
FFI_BG_HOMO_THREAD_UNROLL4(_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmv_gather_hetero_thread_unroll4_float_bf16
FFI_BG_HETERO_THREAD_UNROLL4(_float_bf16, __nv_bfloat16, __nv_bfloat16)

// ----------------------------------------------------------------------------------------------------
/// NT WARP -> the 1 t 1 r Vec and Software Pipelining
///

template <typename T>
struct IsActiveFunc;

template <>
struct IsActiveFunc<uint8_t>
{
    __device__ __forceinline__ bool operator()(uint8_t s) const { return s > 0; }
};
template <>
struct IsActiveFunc<float>
{
    __device__ __forceinline__ bool operator()(float s) const { return s > 0.0f; }
};
template <>
struct IsActiveFunc<double>
{
    __device__ __forceinline__ bool operator()(double s) const { return s > 0.0; }
};
template <>
struct IsActiveFunc<__half>
{
    __device__ __forceinline__ bool operator()(__half s) const { return __half2float(s) > 0.0f; }
};
template <>
struct IsActiveFunc<__nv_bfloat16>
{
    __device__ __forceinline__ bool operator()(__nv_bfloat16 s) const { return __bfloat162float(s) > 0.0f; }
};

// ---------------------------------------------------------------------------------
// 2. 内存指令发射抽象：四元素向量化加载
// ---------------------------------------------------------------------------------
// 默认标量回退路径 (Fallback)
template <typename W_T, typename ACC_T>
__device__ __forceinline__ void load4_weights_state(const W_T *addr, ACC_T &w0, ACC_T &w1, ACC_T &w2, ACC_T &w3)
{
    w0 = static_cast<ACC_T>(__ldg(&addr[0]));
    w1 = static_cast<ACC_T>(__ldg(&addr[1]));
    w2 = static_cast<ACC_T>(__ldg(&addr[2]));
    w3 = static_cast<ACC_T>(__ldg(&addr[3]));
}

// 针对 FP32 的物理特化 (单条 128-bit 指令)
template <>
__device__ __forceinline__ void load4_weights_state<float, float>(const float *addr, float &w0, float &w1, float &w2, float &w3)
{
    float4 vec = __ldg((const float4 *)addr);
    w0 = vec.x;
    w1 = vec.y;
    w2 = vec.z;
    w3 = vec.w;
}

// 针对 FP64 的物理特化 (两条 128-bit 指令，规避 32 字节溢出)
template <>
__device__ __forceinline__ void load4_weights_state<double, double>(const double *addr, double &w0, double &w1, double &w2, double &w3)
{
    double2 v0 = __ldg((const double2 *)&addr[0]);
    double2 v1 = __ldg((const double2 *)&addr[2]);
    w0 = v0.x;
    w1 = v0.y;
    w2 = v1.x;
    w3 = v1.y;
}

// 针对 INT32 的物理特化 (单条 128-bit 指令加载 4 个索引)
template <>
__device__ __forceinline__ void load4_weights_state<int32_t, int>(const int32_t *addr, int &w0, int &w1, int &w2, int &w3)
{
    int4 vec = __ldg((const int4 *)addr);
    w0 = vec.x;
    w1 = vec.y;
    w2 = vec.z;
    w3 = vec.w;
}

template <typename SPIKE_T, typename WEIGHT_T, typename ACC_T, typename IS_ACTIVE_FUNC>
__global__ void _bg_thread_pipeline_hetero_kern_template(
    const int32_t *__restrict__ indices,
    const SPIKE_T *__restrict__ spikes,
    WEIGHT_T *__restrict__ output,
    const WEIGHT_T *__restrict__ weights,
    int n_pre, int n_conn,
    IS_ACTIVE_FUNC is_active)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_pre)
        return;

    const int32_t *i_row = indices + (size_t)row * n_conn;
    const WEIGHT_T *w_row = weights + (size_t)row * n_conn;
    ACC_T val = ACC_T{}; // 状态初始化：类型安全的零初始化

    int c = 0;
    int n_conn_aligned = n_conn & ~3;

    if (n_conn_aligned > 0)
    {
        // 状态变更: 利用抽象层发起预取访存 (Prolog)
        int idx0, idx1, idx2, idx3;
        ACC_T w0, w1, w2, w3;

        load4_weights_state<int32_t, int>(i_row, idx0, idx1, idx2, idx3); // 需要你补充对 int32 映射 int4 的特化
        load4_weights_state<WEIGHT_T, ACC_T>(w_row, w0, w1, w2, w3);

        SPIKE_T s0 = __ldg(&spikes[idx0]);
        SPIKE_T s1 = __ldg(&spikes[idx1]);
        SPIKE_T s2 = __ldg(&spikes[idx2]);
        SPIKE_T s3 = __ldg(&spikes[idx3]);

        for (c = 0; c < n_conn_aligned - 4; c += 4)
        {
            // [操作 A] 预取下一轮状态
            int next_idx0, next_idx1, next_idx2, next_idx3;
            ACC_T next_w0, next_w1, next_w2, next_w3;
            load4_weights_state<int32_t, int>(&i_row[c + 4], next_idx0, next_idx1, next_idx2, next_idx3);
            load4_weights_state<WEIGHT_T, ACC_T>(&w_row[c + 4], next_w0, next_w1, next_w2, next_w3);

            // [操作 B] 消化当前状态 (谓词执行转换为数据流)
            ACC_T m0 = is_active(s0) ? (ACC_T)1 : (ACC_T)0;
            ACC_T m1 = is_active(s1) ? (ACC_T)1 : (ACC_T)0;
            ACC_T m2 = is_active(s2) ? (ACC_T)1 : (ACC_T)0;
            ACC_T m3 = is_active(s3) ? (ACC_T)1 : (ACC_T)0;

            // [操作 C] 发起下一次 Spike 内存请求
            SPIKE_T next_s0 = __ldg(&spikes[next_idx0]);
            SPIKE_T next_s1 = __ldg(&spikes[next_idx1]);
            SPIKE_T next_s2 = __ldg(&spikes[next_idx2]);
            SPIKE_T next_s3 = __ldg(&spikes[next_idx3]);

            // [操作 D] FMA 累加当前状态
            val += m0 * w0;
            val += m1 * w1;
            val += m2 * w2;
            val += m3 * w3;

            // 状态轮转
            idx0 = next_idx0;
            idx1 = next_idx1;
            idx2 = next_idx2;
            idx3 = next_idx3;
            w0 = next_w0;
            w1 = next_w1;
            w2 = next_w2;
            w3 = next_w3;
            s0 = next_s0;
            s1 = next_s1;
            s2 = next_s2;
            s3 = next_s3;
        }

        /* 尾部逻辑: 消化最后一轮，与上文一致 */
        ACC_T m0 = is_active(s0) ? static_cast<ACC_T>(1) : static_cast<ACC_T>(0);
        ACC_T m1 = is_active(s1) ? static_cast<ACC_T>(1) : static_cast<ACC_T>(0);
        ACC_T m2 = is_active(s2) ? static_cast<ACC_T>(1) : static_cast<ACC_T>(0);
        ACC_T m3 = is_active(s3) ? static_cast<ACC_T>(1) : static_cast<ACC_T>(0);

        val += m0 * w0;
        val += m1 * w1;
        val += m2 * w2;
        val += m3 * w3;
        c += 4;
    }

    // 边界处理
    for (; c < n_conn; ++c)
    {
        int idx = __ldg(&i_row[c]);
        if (is_active(__ldg(&spikes[idx])))
        {
            val += static_cast<ACC_T>(__ldg(&w_row[c]));
        }
    }

    output[row] = static_cast<WEIGHT_T>(val); // 精度截断写入
}

template <typename SPIKE_T, typename WEIGHT_T, typename ACC_T, typename IS_ACTIVE_FUNC>
__global__ void _bg_thread_pipeline_homo_kern_template(
    const int32_t *__restrict__ indices,
    const SPIKE_T *__restrict__ spikes,
    WEIGHT_T *__restrict__ output,
    const WEIGHT_T *__restrict__ weights,
    int n_pre, int n_conn,
    IS_ACTIVE_FUNC is_active)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_pre)
        return;

    const int32_t *i_row = indices + (size_t)row * n_conn;
    ACC_T val = ACC_T{}; // 状态初始化：类型安全的零初始化

    int c = 0;
    int n_conn_aligned = n_conn & ~3;

    if (n_conn_aligned > 0)
    {
        // 状态变更: 利用抽象层发起预取访存 (Prolog)
        int idx0, idx1, idx2, idx3;

        load4_weights_state<int32_t, int>(i_row, idx0, idx1, idx2, idx3);

        SPIKE_T s0 = __ldg(&spikes[idx0]);
        SPIKE_T s1 = __ldg(&spikes[idx1]);
        SPIKE_T s2 = __ldg(&spikes[idx2]);
        SPIKE_T s3 = __ldg(&spikes[idx3]);

        for (c = 0; c < n_conn_aligned - 4; c += 4)
        {
            // [操作 A] 预取下一轮状态
            int next_idx0, next_idx1, next_idx2, next_idx3;
            load4_weights_state<int32_t, int>(&i_row[c + 4], next_idx0, next_idx1, next_idx2, next_idx3);

            // [操作 B] 消化当前状态 (谓词执行转换为数据流)
            ACC_T m0 = is_active(s0) ? (ACC_T)1 : (ACC_T)0;
            ACC_T m1 = is_active(s1) ? (ACC_T)1 : (ACC_T)0;
            ACC_T m2 = is_active(s2) ? (ACC_T)1 : (ACC_T)0;
            ACC_T m3 = is_active(s3) ? (ACC_T)1 : (ACC_T)0;

            // [操作 C] 发起下一次 Spike 内存请求
            SPIKE_T next_s0 = __ldg(&spikes[next_idx0]);
            SPIKE_T next_s1 = __ldg(&spikes[next_idx1]);
            SPIKE_T next_s2 = __ldg(&spikes[next_idx2]);
            SPIKE_T next_s3 = __ldg(&spikes[next_idx3]);

            // [操作 D] 累加当前状态
            val += m0 + m1 + m2 + m3;

            // 状态轮转
            idx0 = next_idx0;
            idx1 = next_idx1;
            idx2 = next_idx2;
            idx3 = next_idx3;
            s0 = next_s0;
            s1 = next_s1;
            s2 = next_s2;
            s3 = next_s3;
        }

        /* 尾部逻辑: 消化最后一轮 */
        ACC_T m0 = is_active(s0) ? static_cast<ACC_T>(1) : static_cast<ACC_T>(0);
        ACC_T m1 = is_active(s1) ? static_cast<ACC_T>(1) : static_cast<ACC_T>(0);
        ACC_T m2 = is_active(s2) ? static_cast<ACC_T>(1) : static_cast<ACC_T>(0);
        ACC_T m3 = is_active(s3) ? static_cast<ACC_T>(1) : static_cast<ACC_T>(0);

        val += m0 + m1 + m2 + m3;
        c += 4;
    }

    // 边界处理
    for (; c < n_conn; ++c)
    {
        int idx = __ldg(&i_row[c]);
        if (is_active(__ldg(&spikes[idx])))
        {
            val += static_cast<ACC_T>(1);
        }
    }

    // 读取标量权重，与累加计数相乘后写入
    ACC_T w = static_cast<ACC_T>(weights[0]);
    output[row] = static_cast<WEIGHT_T>(w * val); // 精度截断写入
}

#define FFI_BG_HOMO_THREAD_PIPELINE(SUFFIX, WEIGHT_C_T, SPIKE_C_T, ACC_C_T)                            \
    void binary_fcnmv_gather_homo_thread_pipeline##SUFFIX(                                             \
        const BE::Tensor weights,                                                                     \
        const BE::Tensor indices,                                                                     \
        const BE::Tensor spikes,                                                                      \
        BE::Tensor output, int64_t stream)                                                            \
    {                                                                                                 \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                      \
        int n_pre = static_cast<int>(indices.size(0));                                                \
        int n_conn = static_cast<int>(indices.size(1));                                               \
        const WEIGHT_C_T *d_w = static_cast<const WEIGHT_C_T *>(weights.data_ptr());                  \
        const int32_t *d_idx = static_cast<const int32_t *>(indices.data_ptr());                      \
        const SPIKE_C_T *d_spk = static_cast<const SPIKE_C_T *>(spikes.data_ptr());                   \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr());                             \
                                                                                                      \
        int threads = 256;                                                                            \
        int blocks = (n_pre + threads - 1) / threads;                                                 \
                                                                                                      \
        IsActiveFunc<SPIKE_C_T> is_active_op;                                                         \
                                                                                                      \
        _bg_thread_pipeline_homo_kern_template<SPIKE_C_T, WEIGHT_C_T, ACC_C_T, decltype(is_active_op)> \
            <<<blocks, threads, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_active_op);       \
    }

#define FFI_BG_HETERO_THREAD_PIPELINE(SUFFIX, WEIGHT_C_T, SPIKE_C_T, ACC_C_T)                            \
    void binary_fcnmv_gather_hetero_thread_pipeline##SUFFIX(                                             \
        const BE::Tensor weights, const BE::Tensor indices,                                             \
        const BE::Tensor spikes, BE::Tensor output, int64_t stream)                                     \
    {                                                                                                   \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                        \
        int n_pre = static_cast<int>(indices.size(0));                                                  \
        int n_conn = static_cast<int>(indices.size(1));                                                 \
        const WEIGHT_C_T *d_w = static_cast<const WEIGHT_C_T *>(weights.data_ptr());                    \
        const int32_t *d_idx = static_cast<const int32_t *>(indices.data_ptr());                        \
        const SPIKE_C_T *d_spk = static_cast<const SPIKE_C_T *>(spikes.data_ptr());                     \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr());                               \
                                                                                                        \
        int threads = 256;                                                                              \
        int blocks = (n_pre + threads - 1) / threads;                                                   \
                                                                                                        \
        /* 状态变更: 实例化对应类型的仿函数，并将其作为参数传入模板 */                                  \
        IsActiveFunc<SPIKE_C_T> is_active_op;                                                           \
                                                                                                        \
        /* 状态派遣: 启动 C++ 模板核函数 */                                                             \
        _bg_thread_pipeline_hetero_kern_template<SPIKE_C_T, WEIGHT_C_T, ACC_C_T, decltype(is_active_op)> \
            <<<blocks, threads, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_active_op);         \
    }

// 使用范例：
// 注意：最后一个参数指明了物理寄存器内部的累加器类型
// @BE binary_fcnmv_gather_homo_thread_pipeline_bool_f32
FFI_BG_HOMO_THREAD_PIPELINE(_bool_f32, float, uint8_t, float)
// @BE binary_fcnmv_gather_hetero_thread_pipeline_bool_f32
FFI_BG_HETERO_THREAD_PIPELINE(_bool_f32, float, uint8_t, float)
// @BE binary_fcnmv_gather_homo_thread_pipeline_float_f32
FFI_BG_HOMO_THREAD_PIPELINE(_float_f32, float, float, float)
// @BE binary_fcnmv_gather_hetero_thread_pipeline_float_f32
FFI_BG_HETERO_THREAD_PIPELINE(_float_f32, float, float, float)

// @BE binary_fcnmv_gather_homo_thread_pipeline_bool_f64
FFI_BG_HOMO_THREAD_PIPELINE(_bool_f64, double, uint8_t, double)
// @BE binary_fcnmv_gather_hetero_thread_pipeline_bool_f64
FFI_BG_HETERO_THREAD_PIPELINE(_bool_f64, double, uint8_t, double)
// @BE binary_fcnmv_gather_homo_thread_pipeline_float_f64
FFI_BG_HOMO_THREAD_PIPELINE(_float_f64, double, double, double)
// @BE binary_fcnmv_gather_hetero_thread_pipeline_float_f64
FFI_BG_HETERO_THREAD_PIPELINE(_float_f64, double, double, double)

// 对于半精度，外部存储类型为 __half，但内部计算状态 (ACC) 强制维持 float 以防止精度崩塌
// @BE binary_fcnmv_gather_homo_thread_pipeline_bool_f16
FFI_BG_HOMO_THREAD_PIPELINE(_bool_f16, __half, uint8_t, float)
// @BE binary_fcnmv_gather_hetero_thread_pipeline_bool_f16
FFI_BG_HETERO_THREAD_PIPELINE(_bool_f16, __half, uint8_t, float)
// @BE binary_fcnmv_gather_homo_thread_pipeline_float_f16
FFI_BG_HOMO_THREAD_PIPELINE(_float_f16, __half, __half, float)
// @BE binary_fcnmv_gather_hetero_thread_pipeline_float_f16
FFI_BG_HETERO_THREAD_PIPELINE(_float_f16, __half, __half, float)

// 对于 bfloat16，同理使用 float 作为内部累加器
// @BE binary_fcnmv_gather_homo_thread_pipeline_bool_bf16
FFI_BG_HOMO_THREAD_PIPELINE(_bool_bf16, __nv_bfloat16, uint8_t, float)
// @BE binary_fcnmv_gather_hetero_thread_pipeline_bool_bf16
FFI_BG_HETERO_THREAD_PIPELINE(_bool_bf16, __nv_bfloat16, uint8_t, float)
// @BE binary_fcnmv_gather_homo_thread_pipeline_float_bf16
FFI_BG_HOMO_THREAD_PIPELINE(_float_bf16, __nv_bfloat16, __nv_bfloat16, float)
// @BE binary_fcnmv_gather_hetero_thread_pipeline_float_bf16
FFI_BG_HETERO_THREAD_PIPELINE(_float_bf16, __nv_bfloat16, __nv_bfloat16, float)

// ----------------------------------------------------------------------------------------------------
/// NT WARP -> the 1 t 1 r (Unroll 2)
///

#define DEFINE_BG_THREAD_UNROLL2_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
    __global__ void _bg_thread_unroll2_hetero_kern##SUFFIX(                                                       \
        const int32_t *__restrict__ indices,                                                                      \
        const SPIKE_T *__restrict__ spikes,                                                                       \
        WEIGHT_T *__restrict__ output,                                                                            \
        const WEIGHT_T *__restrict__ weights,                                                                     \
        int n_pre, int n_conn)                                                                                    \
    {                                                                                                             \
        int row = blockIdx.x * blockDim.x + threadIdx.x;                                                          \
        if (row >= n_pre)                                                                                         \
            return;                                                                                               \
                                                                                                                  \
        const int32_t *i_row = indices + (size_t)row * n_conn;                                                    \
        const WEIGHT_T *w_row = weights + (size_t)row * n_conn;                                                   \
        ACC_T val = ACC_ZERO;                                                                                     \
                                                                                                                  \
        int c = 0;                                                                                                \
        int n_conn_aligned = n_conn & ~1; /* 状态变更: 对齐到 2 的倍数 */                                           \
                                                                                                                  \
        for (; c < n_conn_aligned; c += 2)                                                                        \
        {                                                                                                         \
            /* 状态变更: 发起 2 个索引读取指令 */                                                                 \
            int idx0 = __ldg(&i_row[c]);                                                                          \
            int idx1 = __ldg(&i_row[c + 1]);                                                                      \
                                                                                                                  \
            /* 状态变更: 显式将 Spike 状态读取分离，确立条件分支的数据基础 */                                           \
            bool act0 = IS_ACTIVE(__ldg(&spikes[idx0]));                                                          \
            bool act1 = IS_ACTIVE(__ldg(&spikes[idx1]));                                                          \
                                                                                                                  \
            /* 状态流转: 寄存器分配延后，仅在谓词为真时发起权重全局内存读取并累加 */                                    \
            if (act0)                                                                                             \
            {                                                                                                     \
                WEIGHT_T w0 = __ldg(&w_row[c]);                                                                   \
                val += READ_W(w0);                                                                                \
            }                                                                                                     \
            if (act1)                                                                                             \
            {                                                                                                     \
                WEIGHT_T w1 = __ldg(&w_row[c + 1]);                                                               \
                val += READ_W(w1);                                                                                \
            }                                                                                                     \
        }                                                                                                         \
                                                                                                                  \
        /* 状态流转: 处理尾部边界 (最多 1 个元素) */                                                                \
        for (; c < n_conn; ++c)                                                                                   \
        {                                                                                                         \
            int idx = __ldg(&i_row[c]);                                                                           \
            if (IS_ACTIVE(__ldg(&spikes[idx])))                                                                   \
            {                                                                                                     \
                /* 状态变更: 权重读取移入条件分支内 */                                                              \
                WEIGHT_T w = __ldg(&w_row[c]);                                                                    \
                val += READ_W(w);                                                                                 \
            }                                                                                                     \
        }                                                                                                         \
        output[row] = WRITE_W(val);                                                                               \
    }

#define DEFINE_BG_THREAD_UNROLL2_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
    __global__ void _bg_thread_unroll2_homo_kern##SUFFIX(                                                         \
        const int32_t *__restrict__ indices,                                                                      \
        const SPIKE_T *__restrict__ spikes,                                                                       \
        WEIGHT_T *__restrict__ output,                                                                            \
        const WEIGHT_T *__restrict__ weights,                                                                     \
        int n_pre, int n_conn)                                                                                    \
    {                                                                                                             \
        int row = blockIdx.x * blockDim.x + threadIdx.x;                                                          \
        if (row >= n_pre)                                                                                         \
            return;                                                                                               \
                                                                                                                  \
        const int32_t *i_row = indices + (size_t)row * n_conn;                                                    \
        ACC_T val = ACC_ZERO;                                                                                     \
                                                                                                                  \
        /* 状态流转: 主循环，步长为 2 进行批量处理 */                                                               \
        int c = 0;                                                                                                \
        int n_conn_aligned = n_conn & ~1; /* 计算向下取整到 2 的倍数 */                                             \
                                                                                                                  \
        for (; c < n_conn_aligned; c += 2)                                                                        \
        {                                                                                                         \
            /* 状态变更: 连续发起 2 个内存读取指令，目标状态暂存至寄存器 */                                           \
            int idx0 = __ldg(&i_row[c]);                                                                          \
            int idx1 = __ldg(&i_row[c + 1]);                                                                      \
                                                                                                                  \
            /* 状态变更: 独立的读后写累加操作，解除指令间依赖 */                                                      \
            if (IS_ACTIVE(__ldg(&spikes[idx0])))                                                                  \
                val += (ACC_T)1;                                                                                  \
            if (IS_ACTIVE(__ldg(&spikes[idx1])))                                                                  \
                val += (ACC_T)1;                                                                                  \
        }                                                                                                         \
                                                                                                                  \
        /* 状态流转: 处理尾部剩余的列（最多 1 列） */                                                               \
        for (; c < n_conn; ++c)                                                                                   \
        {                                                                                                         \
            int idx = __ldg(&i_row[c]);                                                                           \
            if (IS_ACTIVE(__ldg(&spikes[idx])))                                                                   \
            {                                                                                                     \
                val += (ACC_T)1;                                                                                  \
            }                                                                                                     \
        }                                                                                                         \
        output[row] = WRITE_W(READ_W(weights[0]) * val);                                                          \
    }


DEFINE_BG_THREAD_UNROLL2_HOMO(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_THREAD_UNROLL2_HETERO(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_THREAD_UNROLL2_HOMO(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_THREAD_UNROLL2_HETERO(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)

DEFINE_BG_THREAD_UNROLL2_HOMO(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_THREAD_UNROLL2_HETERO(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_THREAD_UNROLL2_HOMO(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_THREAD_UNROLL2_HETERO(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)

DEFINE_BG_THREAD_UNROLL2_HOMO(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_THREAD_UNROLL2_HETERO(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_THREAD_UNROLL2_HOMO(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_THREAD_UNROLL2_HETERO(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)

DEFINE_BG_THREAD_UNROLL2_HOMO(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_THREAD_UNROLL2_HETERO(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_THREAD_UNROLL2_HOMO(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_THREAD_UNROLL2_HETERO(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)

#define FFI_BG_HOMO_THREAD_UNROLL2(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                             \
    void binary_fcnmv_gather_homo_thread_unroll2##SUFFIX(                                                     \
        const BE::Tensor weights,                                                                             \
        const BE::Tensor indices,                                                                             \
        const BE::Tensor spikes,                                                                              \
        BE::Tensor output, int64_t stream)                                                                    \
    {                                                                                                         \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                              \
        int n_pre = static_cast<int>(indices.size(0));                                                        \
        int n_conn = static_cast<int>(indices.size(1));                                                       \
        const WEIGHT_C_T *d_w = static_cast<const WEIGHT_C_T *>(weights.data_ptr());                          \
        const int32_t *d_idx = static_cast<const int32_t *>(indices.data_ptr());                              \
        const SPIKE_C_T *d_spk = static_cast<const SPIKE_C_T *>(spikes.data_ptr());                           \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr());                                     \
        int threads = 512; /* 状态变更: 寄存器压力减小，提升 Block 线程数至 512 以填满 SM 并行度 */               \
        int blocks = (n_pre + threads - 1) / threads;                                                         \
        _bg_thread_unroll2_homo_kern##SUFFIX<<<blocks, threads, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn); \
    }

// ---- FFI macro: gather hetero warp ----
#define FFI_BG_HETERO_THREAD_UNROLL2(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                           \
    void binary_fcnmv_gather_hetero_thread_unroll2##SUFFIX(                                                   \
        const BE::Tensor weights, const BE::Tensor indices,                                                   \
        const BE::Tensor spikes, BE::Tensor output, int64_t stream)                                           \
    {                                                                                                         \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                              \
        int n_pre = static_cast<int>(indices.size(0));                                                        \
        int n_conn = static_cast<int>(indices.size(1));                                                       \
        const WEIGHT_C_T *d_w = static_cast<const WEIGHT_C_T *>(weights.data_ptr());                          \
        const int32_t *d_idx = static_cast<const int32_t *>(indices.data_ptr());                              \
        const SPIKE_C_T *d_spk = static_cast<const SPIKE_C_T *>(spikes.data_ptr());                           \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr());                                     \
        int threads = 512; /* 状态变更: 寄存器压力减小，提升 Block 线程数至 512 以填满 SM 并行度 */               \
        int blocks = (n_pre + threads - 1) / threads;                                                         \
        _bg_thread_unroll2_hetero_kern##SUFFIX<<<blocks, threads, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn); \
    }

// @BE binary_fcnmv_gather_homo_thread_unroll2_bool_f32
FFI_BG_HOMO_THREAD_UNROLL2(_bool_f32, float, uint8_t)
// @BE binary_fcnmv_gather_hetero_thread_unroll2_bool_f32
FFI_BG_HETERO_THREAD_UNROLL2(_bool_f32, float, uint8_t)
// @BE binary_fcnmv_gather_homo_thread_unroll2_float_f32
FFI_BG_HOMO_THREAD_UNROLL2(_float_f32, float, float)
// @BE binary_fcnmv_gather_hetero_thread_unroll2_float_f32
FFI_BG_HETERO_THREAD_UNROLL2(_float_f32, float, float)

// @BE binary_fcnmv_gather_homo_thread_unroll2_bool_f64
FFI_BG_HOMO_THREAD_UNROLL2(_bool_f64, double, uint8_t)
// @BE binary_fcnmv_gather_hetero_thread_unroll2_bool_f64
FFI_BG_HETERO_THREAD_UNROLL2(_bool_f64, double, uint8_t)
// @BE binary_fcnmv_gather_homo_thread_unroll2_float_f64
FFI_BG_HOMO_THREAD_UNROLL2(_float_f64, double, double)
// @BE binary_fcnmv_gather_hetero_thread_unroll2_float_f64
FFI_BG_HETERO_THREAD_UNROLL2(_float_f64, double, double)

// @BE binary_fcnmv_gather_homo_thread_unroll2_bool_f16
FFI_BG_HOMO_THREAD_UNROLL2(_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_gather_hetero_thread_unroll2_bool_f16
FFI_BG_HETERO_THREAD_UNROLL2(_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_gather_homo_thread_unroll2_float_f16
FFI_BG_HOMO_THREAD_UNROLL2(_float_f16, __half, __half)
// @BE binary_fcnmv_gather_hetero_thread_unroll2_float_f16
FFI_BG_HETERO_THREAD_UNROLL2(_float_f16, __half, __half)

// @BE binary_fcnmv_gather_homo_thread_unroll2_bool_bf16
FFI_BG_HOMO_THREAD_UNROLL2(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_gather_hetero_thread_unroll2_bool_bf16
FFI_BG_HETERO_THREAD_UNROLL2(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_gather_homo_thread_unroll2_float_bf16
FFI_BG_HOMO_THREAD_UNROLL2(_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmv_gather_hetero_thread_unroll2_float_bf16
FFI_BG_HETERO_THREAD_UNROLL2(_float_bf16, __nv_bfloat16, __nv_bfloat16)