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
 * compact_binary_fcnmv_T.cu -- Compact CSC scatter path for compact_binary_fcnmv
 * ==============================================================================
 *
 * This file provides CUDA scatter kernels that compute
 *
 *   y = A x
 *
 * for a matrix A represented in compact CSC layout together with CompactBinary
 * event metadata (`active_ids` + `n_active`).
 */

#include "cuda_common.h"
#include "brainevent/common.h"

#define DEFINE_CS_CSC_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)      \
    __global__ void _cs_csc_homo_kern##SUFFIX(                                  \
        const int32_t *__restrict__ indices,                                    \
        const int32_t *__restrict__ indptr,                                     \
        const int32_t *__restrict__ active_ids,                                 \
        const int32_t *__restrict__ n_active_ptr,                               \
        WEIGHT_T *__restrict__ output,                                          \
        const WEIGHT_T *__restrict__ weights)                                   \
    {                                                                           \
        int tid = blockIdx.x * blockDim.x + threadIdx.x;                        \
        int na = __ldg(n_active_ptr);                                           \
        if (tid >= na)                                                          \
            return;                                                             \
        int col = __ldg(&active_ids[tid]);                                      \
        int start = __ldg(&indptr[col]);                                        \
        int end = __ldg(&indptr[col + 1]);                                      \
        ACC_T w0 = READ_W(weights[0]);                                          \
        for (int pos = start; pos < end; pos++)                                 \
        {                                                                       \
            int row = __ldg(&indices[pos]);                                     \
            if (row < 0)                                                        \
                continue;                                                       \
            ATOMIC_ADD_W(&output[row], w0);                                     \
        }                                                                       \
    }

#define DEFINE_CS_CSC_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)    \
    __global__ void _cs_csc_hetero_kern##SUFFIX(                                \
        const int32_t *__restrict__ indices,                                    \
        const int32_t *__restrict__ indptr,                                     \
        const int32_t *__restrict__ active_ids,                                 \
        const int32_t *__restrict__ n_active_ptr,                               \
        WEIGHT_T *__restrict__ output,                                          \
        const WEIGHT_T *__restrict__ weights)                                   \
    {                                                                           \
        int tid = blockIdx.x * blockDim.x + threadIdx.x;                        \
        int na = __ldg(n_active_ptr);                                           \
        if (tid >= na)                                                          \
            return;                                                             \
        int col = __ldg(&active_ids[tid]);                                      \
        int start = __ldg(&indptr[col]);                                        \
        int end = __ldg(&indptr[col + 1]);                                      \
        for (int pos = start; pos < end; pos++)                                 \
        {                                                                       \
            int row = __ldg(&indices[pos]);                                     \
            if (row < 0)                                                        \
                continue;                                                       \
            ACC_T wk = READ_W(__ldg(&weights[pos]));                            \
            ATOMIC_ADD_W(&output[row], wk);                                     \
        }                                                                       \
    }

// ---- float32 ----
DEFINE_CS_CSC_HOMO(_f32, float, float, READ_F32, atomicAdd)
DEFINE_CS_CSC_HETERO(_f32, float, float, READ_F32, atomicAdd)

// ---- float64 ----
DEFINE_CS_CSC_HOMO(_f64, double, double, READ_F64, atomic_add_f64)
DEFINE_CS_CSC_HETERO(_f64, double, double, READ_F64, atomic_add_f64)

// ---- float16 ----
DEFINE_CS_CSC_HOMO(_f16, __half, float, READ_F16, atomic_add_f16)
DEFINE_CS_CSC_HETERO(_f16, __half, float, READ_F16, atomic_add_f16)

// ---- bfloat16 ----
DEFINE_CS_CSC_HOMO(_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_CS_CSC_HETERO(_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)

#define FFI_CS_HOMO(SUFFIX, WEIGHT_C_T)                                         \
    void compact_binary_fcnmv_scatter_homo##SUFFIX(                             \
        const BE::Tensor weights, const BE::Tensor indices, const BE::Tensor indptr, \
        const BE::Tensor packed, const BE::Tensor active_ids,                   \
        const BE::Tensor n_active, BE::Tensor output, int64_t stream)           \
    {                                                                           \
        (void)packed;                                                           \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                \
        int n_cols = static_cast<int>(active_ids.size(0));                      \
        int n_post = static_cast<int>(output.size(0));                          \
        int nnz = static_cast<int>(indices.size(0));                            \
        const WEIGHT_C_T *d_w = static_cast<const WEIGHT_C_T *>(weights.data_ptr()); \
        const int32_t *d_idx = static_cast<const int32_t *>(indices.data_ptr()); \
        const int32_t *d_ptr = static_cast<const int32_t *>(indptr.data_ptr()); \
        const int32_t *d_aids = static_cast<const int32_t *>(active_ids.data_ptr()); \
        const int32_t *d_na = static_cast<const int32_t *>(n_active.data_ptr()); \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr());       \
        cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(WEIGHT_C_T), s);      \
        if (n_cols <= 0 || nnz <= 0)                                            \
            return;                                                             \
        int bsz = 256;                                                          \
        int n_blocks = (n_cols + bsz - 1) / bsz;                                \
        _cs_csc_homo_kern##SUFFIX<<<n_blocks, bsz, 0, s>>>(                     \
            d_idx, d_ptr, d_aids, d_na, d_out, d_w);                            \
        BE_CHECK_KERNEL_LAUNCH();                                               \
    }

#define FFI_CS_HETERO(SUFFIX, WEIGHT_C_T)                                       \
    void compact_binary_fcnmv_scatter_hetero##SUFFIX(                           \
        const BE::Tensor weights, const BE::Tensor indices, const BE::Tensor indptr, \
        const BE::Tensor packed, const BE::Tensor active_ids,                   \
        const BE::Tensor n_active, BE::Tensor output, int64_t stream)           \
    {                                                                           \
        (void)packed;                                                           \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                \
        int n_cols = static_cast<int>(active_ids.size(0));                      \
        int n_post = static_cast<int>(output.size(0));                          \
        int nnz = static_cast<int>(indices.size(0));                            \
        const WEIGHT_C_T *d_w = static_cast<const WEIGHT_C_T *>(weights.data_ptr()); \
        const int32_t *d_idx = static_cast<const int32_t *>(indices.data_ptr()); \
        const int32_t *d_ptr = static_cast<const int32_t *>(indptr.data_ptr()); \
        const int32_t *d_aids = static_cast<const int32_t *>(active_ids.data_ptr()); \
        const int32_t *d_na = static_cast<const int32_t *>(n_active.data_ptr()); \
        WEIGHT_C_T *d_out = static_cast<WEIGHT_C_T *>(output.data_ptr());       \
        cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(WEIGHT_C_T), s);      \
        if (n_cols <= 0 || nnz <= 0)                                            \
            return;                                                             \
        int bsz = 256;                                                          \
        int n_blocks = (n_cols + bsz - 1) / bsz;                                \
        _cs_csc_hetero_kern##SUFFIX<<<n_blocks, bsz, 0, s>>>(                   \
            d_idx, d_ptr, d_aids, d_na, d_out, d_w);                            \
        BE_CHECK_KERNEL_LAUNCH();                                               \
    }

// ---- float32 ----
// @BE compact_binary_fcnmv_scatter_homo_f32
FFI_CS_HOMO(_f32, float)
// @BE compact_binary_fcnmv_scatter_hetero_f32
FFI_CS_HETERO(_f32, float)

// ---- float64 ----
// @BE compact_binary_fcnmv_scatter_homo_f64
FFI_CS_HOMO(_f64, double)
// @BE compact_binary_fcnmv_scatter_hetero_f64
FFI_CS_HETERO(_f64, double)

// ---- float16 ----
// @BE compact_binary_fcnmv_scatter_homo_f16
FFI_CS_HOMO(_f16, __half)
// @BE compact_binary_fcnmv_scatter_hetero_f16
FFI_CS_HETERO(_f16, __half)

// ---- bfloat16 ----
// @BE compact_binary_fcnmv_scatter_homo_bf16
FFI_CS_HOMO(_bf16, __nv_bfloat16)
// @BE compact_binary_fcnmv_scatter_hetero_bf16
FFI_CS_HETERO(_bf16, __nv_bfloat16)
