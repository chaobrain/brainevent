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
 * binary_densemv.cu -- Event-Driven Binary Dense Matrix-Vector CUDA Kernels
 * ==========================================================================
 *
 * This module provides optimized CUDA kernels for event-driven dense
 * matrix-vector operations (SpMV):
 *
 * 1. binary_densemv_no_transpose  -- weights[m,k] @ spikes[k] -> out[m]
 *    (transpose=False): event-driven scatter over active spike columns.
 *
 * 2. binary_densemv_transpose  -- spikes[k] @ weights[k,n] -> out[n]
 *    (transpose=True): event-driven scatter over active spike rows.
 *
 * Python API (brainevent._dense.binary):
 *   binary_densemv(weights, spikes, transpose=False)
 *     weights : float16/float32/float64/bfloat16 matrix
 *     spikes  : bool (int8) or float32 spike vector
 *     returns : output vector
 *
 * CUDA entry points:
 *   binary_densemv_no_transpose_{dtype}_{spike_dtype}
 *   binary_densemv_transpose_{dtype}_{spike_dtype}
 */

#include "cuda_common.h"
#include "brainevent/common.h"

// =========================================================================
// Dense Matrix-Vector Multiplication (densemv)
// =========================================================================

/*
 * Event-driven transpose path:
 *   spikes[k] @ weights[k,n] -> output[n]
 */

#define DEFINE_TRANSPOSE(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, READ_W, ATOMIC_ADD_W) \
    __global__ void _transpose_kern##SUFFIX(                                        \
        const WEIGHT_T *__restrict__ weights,                                      \
        const SPIKE_T *__restrict__ spikes,                                        \
        WEIGHT_T *__restrict__ output,                                             \
        int k, int n)                                                              \
    {                                                                              \
        int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);             \
        int lane = threadIdx.x & 31;                                               \
        if (row >= k)                                                              \
            return;                                                                \
        if (!IS_ACTIVE(__ldg(&spikes[row]))) return;                               \
        for (int j = lane; j < n; j += 32)                                         \
        {                                                                          \
            ATOMIC_ADD_W(&output[j], READ_W(__ldg(&weights[(size_t)row * n + j])));\
        }                                                                          \
    }

/*
 * Event-driven no-transpose path:
 *   weights[m,k] @ spikes[k] -> output[m]
 */

#define DEFINE_NO_TRANSPOSE(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, READ_W, ATOMIC_ADD_W) \
    __global__ void _no_transpose_kern##SUFFIX(                                     \
        const WEIGHT_T *__restrict__ weights,                                      \
        const SPIKE_T *__restrict__ spikes,                                        \
        WEIGHT_T *__restrict__ output,                                             \
        int m, int n)                                                              \
    {                                                                              \
        int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);             \
        int lane = threadIdx.x & 31;                                               \
        if (row >= n)                                                              \
            return;                                                                \
        if (!IS_ACTIVE(__ldg(&spikes[row]))) return;                               \
        for (int j = lane; j < m; j += 32)                                         \
        {                                                                          \
            ATOMIC_ADD_W(&output[j], READ_W(__ldg(&weights[(size_t)j * n + row])));\
        }                                                                          \
    }

// Transpose Instantiations
DEFINE_TRANSPOSE(_f32_bool, int8_t, IS_ACTIVE_BOOL, float, READ_F32, atomic_add_f32)
DEFINE_TRANSPOSE(_f32_float, float, IS_ACTIVE_FLOAT, float, READ_F32, atomic_add_f32)
DEFINE_TRANSPOSE(_f64_bool, int8_t, IS_ACTIVE_BOOL, double, READ_F64, atomic_add_f64)
DEFINE_TRANSPOSE(_f64_float, float, IS_ACTIVE_FLOAT, double, READ_F64, atomic_add_f64)
DEFINE_TRANSPOSE(_f16_bool, int8_t, IS_ACTIVE_BOOL, __half, READ_F16, atomic_add_f16)
DEFINE_TRANSPOSE(_f16_float, float, IS_ACTIVE_FLOAT, __half, READ_F16, atomic_add_f16)
DEFINE_TRANSPOSE(_bf16_bool, int8_t, IS_ACTIVE_BOOL, __nv_bfloat16, READ_BF16, atomic_add_bf16)
DEFINE_TRANSPOSE(_bf16_float, float, IS_ACTIVE_FLOAT, __nv_bfloat16, READ_BF16, atomic_add_bf16)

// No-transpose Instantiations
DEFINE_NO_TRANSPOSE(_f32_bool, int8_t, IS_ACTIVE_BOOL, float, READ_F32, atomic_add_f32)
DEFINE_NO_TRANSPOSE(_f32_float, float, IS_ACTIVE_FLOAT, float, READ_F32, atomic_add_f32)
DEFINE_NO_TRANSPOSE(_f64_bool, int8_t, IS_ACTIVE_BOOL, double, READ_F64, atomic_add_f64)
DEFINE_NO_TRANSPOSE(_f64_float, float, IS_ACTIVE_FLOAT, double, READ_F64, atomic_add_f64)
DEFINE_NO_TRANSPOSE(_f16_bool, int8_t, IS_ACTIVE_BOOL, __half, READ_F16, atomic_add_f16)
DEFINE_NO_TRANSPOSE(_f16_float, float, IS_ACTIVE_FLOAT, __half, READ_F16, atomic_add_f16)
DEFINE_NO_TRANSPOSE(_bf16_bool, int8_t, IS_ACTIVE_BOOL, __nv_bfloat16, READ_BF16, atomic_add_bf16)
DEFINE_NO_TRANSPOSE(_bf16_float, float, IS_ACTIVE_FLOAT, __nv_bfloat16, READ_BF16, atomic_add_bf16)

// FFI Macros for SpMV
#define FFI_TRANSPOSE(SUFFIX, WEIGHT_C_T, SPIKE_C_T)             \
    void binary_densemv_transpose##SUFFIX(                       \
        const BE::Tensor weights, const BE::Tensor spikes,       \
        BE::Tensor output, int64_t stream)                       \
    {                                                            \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream); \
        int k = static_cast<int>(weights.size(0));               \
        int n = static_cast<int>(weights.size(1));               \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr()); \
        cudaMemsetAsync(d_out, 0, (size_t)n * sizeof(WEIGHT_C_T), s); \
        int bsz = 256;                                           \
        int warps_per_block = bsz / 32;                          \
        int blocks = (k + warps_per_block - 1) / warps_per_block; \
        _transpose_kern##SUFFIX<<<blocks, bsz, 0, s>>>(          \
            static_cast<const WEIGHT_C_T *>(weights.data_ptr()), \
            static_cast<const SPIKE_C_T *>(spikes.data_ptr()),   \
            d_out, k, n);                                        \
    }

#define FFI_NO_TRANSPOSE(SUFFIX, WEIGHT_C_T, SPIKE_C_T)          \
    void binary_densemv_no_transpose##SUFFIX(                    \
        const BE::Tensor weights, const BE::Tensor spikes,       \
        BE::Tensor output, int64_t stream)                       \
    {                                                            \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream); \
        int m = static_cast<int>(weights.size(0));               \
        int n = static_cast<int>(weights.size(1));               \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr()); \
        cudaMemsetAsync(d_out, 0, (size_t)m * sizeof(WEIGHT_C_T), s); \
        int bsz = 256;                                           \
        int warps_per_block = bsz / 32;                          \
        int blocks = (n + warps_per_block - 1) / warps_per_block; \
        _no_transpose_kern##SUFFIX<<<blocks, bsz, 0, s>>>(       \
            static_cast<const WEIGHT_C_T *>(weights.data_ptr()), \
            static_cast<const SPIKE_C_T *>(spikes.data_ptr()),   \
            d_out, m, n);                                        \
    }

// Transpose FFI Instantiations
// @BE binary_densemv_transpose_f32_bool
FFI_TRANSPOSE(_f32_bool, float, int8_t)
// @BE binary_densemv_transpose_f32_float
FFI_TRANSPOSE(_f32_float, float, float)
// @BE binary_densemv_transpose_f64_bool
FFI_TRANSPOSE(_f64_bool, double, int8_t)
// @BE binary_densemv_transpose_f64_float
FFI_TRANSPOSE(_f64_float, double, float)
// @BE binary_densemv_transpose_f16_bool
FFI_TRANSPOSE(_f16_bool, __half, int8_t)
// @BE binary_densemv_transpose_f16_float
FFI_TRANSPOSE(_f16_float, __half, float)
// @BE binary_densemv_transpose_bf16_bool
FFI_TRANSPOSE(_bf16_bool, __nv_bfloat16, int8_t)
// @BE binary_densemv_transpose_bf16_float
FFI_TRANSPOSE(_bf16_float, __nv_bfloat16, float)

// No-transpose FFI Instantiations
// @BE binary_densemv_no_transpose_f32_bool
FFI_NO_TRANSPOSE(_f32_bool, float, int8_t)
// @BE binary_densemv_no_transpose_f32_float
FFI_NO_TRANSPOSE(_f32_float, float, float)
// @BE binary_densemv_no_transpose_f64_bool
FFI_NO_TRANSPOSE(_f64_bool, double, int8_t)
// @BE binary_densemv_no_transpose_f64_float
FFI_NO_TRANSPOSE(_f64_float, double, float)
// @BE binary_densemv_no_transpose_f16_bool
FFI_NO_TRANSPOSE(_f16_bool, __half, int8_t)
// @BE binary_densemv_no_transpose_f16_float
FFI_NO_TRANSPOSE(_f16_float, __half, float)
// @BE binary_densemv_no_transpose_bf16_bool
FFI_NO_TRANSPOSE(_bf16_bool, __nv_bfloat16, int8_t)
// @BE binary_densemv_no_transpose_bf16_float
FFI_NO_TRANSPOSE(_bf16_float, __nv_bfloat16, float)
