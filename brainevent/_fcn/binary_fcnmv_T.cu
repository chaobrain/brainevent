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
 * binary_fcnmv_T.cu -- Compact CSC scatter path for binary_fcnmv
 * ==============================================================================
 *
 * This file provides the CUDA scatter implementation used to compute
 *
 *   y = A x
 *
 * for a matrix A represented in compact CSC layout:
 *
 *   - `indices` : int32, shape (nnz,)
 *       Row ids grouped by logical input column.
 *   - `indptr` : int32, shape (n_col + 1,)
 *       Column boundaries, where entries for column `col` live in
 *       `indices[indptr[col]:indptr[col + 1]]`.
 *   - `weights` : scalar homo weight or hetero weights aligned with `indices`.
 */

#include "cuda_common.h"
#include "brainevent/common.h"

#define DEFINE_BS_CSC_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
    __global__ void _bs_csc_homo_kern##SUFFIX(                                                 \
        const int32_t *__restrict__ indices,                                                   \
        const int32_t *__restrict__ indptr,                                                    \
        const SPIKE_T *__restrict__ spikes,                                                    \
        WEIGHT_T *__restrict__ output,                                                         \
        const WEIGHT_T *__restrict__ weights,                                                  \
        int n_col)                                                                             \
    {                                                                                          \
        int col = blockIdx.x * blockDim.x + threadIdx.x;                                       \
        if (col >= n_col)                                                                      \
            return;                                                                            \
        if (!IS_ACTIVE(__ldg(&spikes[col])))                                                   \
            return;                                                                            \
        int start = __ldg(&indptr[col]);                                                       \
        int end = __ldg(&indptr[col + 1]);                                                     \
        ACC_T w0 = READ_W(weights[0]);                                                         \
        for (int pos = start; pos < end; pos++)                                                \
        {                                                                                      \
            int row = __ldg(&indices[pos]);                                                    \
            if (row < 0)                                                                       \
                continue;                                                                      \
            ATOMIC_ADD_W(&output[row], w0);                                                    \
        }                                                                                      \
    }

#define DEFINE_BS_CSC_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
    __global__ void _bs_csc_hetero_kern##SUFFIX(                                                \
        const int32_t *__restrict__ indices,                                                    \
        const int32_t *__restrict__ indptr,                                                     \
        const SPIKE_T *__restrict__ spikes,                                                     \
        WEIGHT_T *__restrict__ output,                                                          \
        const WEIGHT_T *__restrict__ weights,                                                   \
        int n_col)                                                                              \
    {                                                                                           \
        int col = blockIdx.x * blockDim.x + threadIdx.x;                                        \
        if (col >= n_col)                                                                       \
            return;                                                                             \
        if (!IS_ACTIVE(__ldg(&spikes[col])))                                                    \
            return;                                                                             \
        int start = __ldg(&indptr[col]);                                                        \
        int end = __ldg(&indptr[col + 1]);                                                      \
        for (int pos = start; pos < end; pos++)                                                 \
        {                                                                                       \
            int row = __ldg(&indices[pos]);                                                     \
            if (row < 0)                                                                        \
                continue;                                                                       \
            ACC_T wk = READ_W(__ldg(&weights[pos]));                                            \
            ATOMIC_ADD_W(&output[row], wk);                                                     \
        }                                                                                       \
    }

// ---- float32 ----
DEFINE_BS_CSC_HOMO(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, atomicAdd)
DEFINE_BS_CSC_HETERO(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, atomicAdd)
DEFINE_BS_CSC_HOMO(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, atomicAdd)
DEFINE_BS_CSC_HETERO(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, atomicAdd)

// ---- float64 ----
DEFINE_BS_CSC_HOMO(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, atomic_add_f64)
DEFINE_BS_CSC_HETERO(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, atomic_add_f64)
DEFINE_BS_CSC_HOMO(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, atomic_add_f64)
DEFINE_BS_CSC_HETERO(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, atomic_add_f64)

// ---- float16 ----
DEFINE_BS_CSC_HOMO(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, atomic_add_f16)
DEFINE_BS_CSC_HETERO(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, atomic_add_f16)
DEFINE_BS_CSC_HOMO(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, atomic_add_f16)
DEFINE_BS_CSC_HETERO(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, atomic_add_f16)

// ---- bfloat16 ----
DEFINE_BS_CSC_HOMO(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BS_CSC_HETERO(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BS_CSC_HOMO(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BS_CSC_HETERO(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)

#define FFI_BS_HOMO(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                      \
    void binary_fcnmv_scatter_homo##SUFFIX(                                             \
        const BE::Tensor weights, const BE::Tensor indices, const BE::Tensor indptr,    \
        const BE::Tensor spikes, BE::Tensor output, int64_t stream)                     \
    {                                                                                   \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                        \
        int n_col = static_cast<int>(indptr.size(0)) - 1;                               \
        int n_row = static_cast<int>(output.size(0));                                   \
        int nnz = static_cast<int>(indices.size(0));                                    \
        const WEIGHT_C_T *d_w = static_cast<const WEIGHT_C_T *>(weights.data_ptr());    \
        const int32_t *d_idx = static_cast<const int32_t *>(indices.data_ptr());        \
        const int32_t *d_ptr = static_cast<const int32_t *>(indptr.data_ptr());         \
        const SPIKE_C_T *d_spk = static_cast<const SPIKE_C_T *>(spikes.data_ptr());     \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr());               \
        cudaMemsetAsync(d_out, 0, (size_t)n_row * sizeof(WEIGHT_C_T), s);               \
        if (n_col <= 0 || nnz <= 0)                                                     \
            return;                                                                     \
        int bsz = 256;                                                                  \
        int n_blocks = (n_col + bsz - 1) / bsz;                                         \
        _bs_csc_homo_kern##SUFFIX<<<n_blocks, bsz, 0, s>>>(                             \
            d_idx, d_ptr, d_spk, d_out, d_w, n_col);                                    \
        BE_CHECK_KERNEL_LAUNCH();                                                       \
    }

#define FFI_BS_HETERO(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                    \
    void binary_fcnmv_scatter_hetero##SUFFIX(                                           \
        const BE::Tensor weights, const BE::Tensor indices, const BE::Tensor indptr,    \
        const BE::Tensor spikes, BE::Tensor output, int64_t stream)                     \
    {                                                                                   \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                        \
        int n_col = static_cast<int>(indptr.size(0)) - 1;                               \
        int n_row = static_cast<int>(output.size(0));                                   \
        int nnz = static_cast<int>(indices.size(0));                                    \
        const WEIGHT_C_T *d_w = static_cast<const WEIGHT_C_T *>(weights.data_ptr());    \
        const int32_t *d_idx = static_cast<const int32_t *>(indices.data_ptr());        \
        const int32_t *d_ptr = static_cast<const int32_t *>(indptr.data_ptr());         \
        const SPIKE_C_T *d_spk = static_cast<const SPIKE_C_T *>(spikes.data_ptr());     \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr());               \
        cudaMemsetAsync(d_out, 0, (size_t)n_row * sizeof(WEIGHT_C_T), s);               \
        if (n_col <= 0 || nnz <= 0)                                                     \
            return;                                                                     \
        int bsz = 256;                                                                  \
        int n_blocks = (n_col + bsz - 1) / bsz;                                         \
        _bs_csc_hetero_kern##SUFFIX<<<n_blocks, bsz, 0, s>>>(                           \
            d_idx, d_ptr, d_spk, d_out, d_w, n_col);                                    \
        BE_CHECK_KERNEL_LAUNCH();                                                       \
    }

// ---- float32 ----
// @BE binary_fcnmv_scatter_homo_bool_f32
FFI_BS_HOMO(_bool_f32, float, uint8_t)
// @BE binary_fcnmv_scatter_hetero_bool_f32
FFI_BS_HETERO(_bool_f32, float, uint8_t)
// @BE binary_fcnmv_scatter_homo_float_f32
FFI_BS_HOMO(_float_f32, float, float)
// @BE binary_fcnmv_scatter_hetero_float_f32
FFI_BS_HETERO(_float_f32, float, float)

// ---- float64 ----
// @BE binary_fcnmv_scatter_homo_bool_f64
FFI_BS_HOMO(_bool_f64, double, uint8_t)
// @BE binary_fcnmv_scatter_hetero_bool_f64
FFI_BS_HETERO(_bool_f64, double, uint8_t)
// @BE binary_fcnmv_scatter_homo_float_f64
FFI_BS_HOMO(_float_f64, double, double)
// @BE binary_fcnmv_scatter_hetero_float_f64
FFI_BS_HETERO(_float_f64, double, double)

// ---- float16 ----
// @BE binary_fcnmv_scatter_homo_bool_f16
FFI_BS_HOMO(_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_scatter_hetero_bool_f16
FFI_BS_HETERO(_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_scatter_homo_float_f16
FFI_BS_HOMO(_float_f16, __half, __half)
// @BE binary_fcnmv_scatter_hetero_float_f16
FFI_BS_HETERO(_float_f16, __half, __half)

// ---- bfloat16 ----
// @BE binary_fcnmv_scatter_homo_bool_bf16
FFI_BS_HOMO(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_scatter_hetero_bool_bf16
FFI_BS_HETERO(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_scatter_homo_float_bf16
FFI_BS_HOMO(_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmv_scatter_hetero_float_bf16
FFI_BS_HETERO(_float_bf16, __nv_bfloat16, __nv_bfloat16)
