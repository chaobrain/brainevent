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
 * binary_fcnmm_col_scatter.cu -- Column-scatter path for binary_fcnmm
 * ==============================================================================
 *
 * This file provides a warp-per-W-column column-scatter implementation for
 * transpose=True binary_fcnmm that consumes:
 *
 *   - `weights` : scalar homo weight or hetero column-major weights aligned with `indices`
 *   - `indices` : int32 column-major row-index array, shape (nnz,)
 *   - `indptr`  : int32 column pointer array, shape (n_pre + 1,)
 *   - `matrix`  : dense column-major spike view `matrix_t[n_batch, n_pre]`
 *
 * The output is written in the same physical layout as fcnmm_testing_op.cu:
 *
 *   output[j, row]  ->  output[(size_t)j * n_post + row]
 */

#include "cuda_common.h"
#include "brainevent/common.h"

#define BFCNMM_COL_WARP_THREADS 32
#define BFCNMM_COL_WARPS_PER_BLOCK 8

#define DEFINE_BFCNMM_COL_WARP_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
    __global__ void _bfcnmm_col_warp_homo_kern##SUFFIX(                                               \
        const int32_t* __restrict__ indices,                                                                \
        const int32_t* __restrict__ indptr,                                                                 \
        const SPIKE_T* __restrict__ matrix_t,                                                               \
        WEIGHT_T* __restrict__ output,                                                                      \
        const WEIGHT_T* __restrict__ weights,                                                               \
        int n_pre, int n_post, int n_batch)                                                                 \
    {                                                                                                       \
        int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x); \
        int col = tid >> 5;                                                                                 \
        int lane = tid & 31;                                                                                \
        int batch = static_cast<int>(blockIdx.y);                                                           \
        if (col >= n_pre || batch >= n_batch) return;                                                       \
        if (!IS_ACTIVE(__ldg(&matrix_t[static_cast<size_t>(batch) * n_pre + col]))) return;                \
        int start = __ldg(&indptr[col]);                                                                    \
        int end = __ldg(&indptr[col + 1]);                                                                  \
        ACC_T w0 = READ_W(__ldg(&weights[0]));                                                              \
        for (int pos = start + lane; pos < end; pos += BFCNMM_COL_WARP_THREADS) {                          \
            int row = __ldg(&indices[pos]);                                                                 \
            if (row < 0 || row >= n_post) continue;                                                         \
            ATOMIC_ADD_W(&output[static_cast<size_t>(batch) * n_post + row], w0);                          \
        }                                                                                                   \
    }

#define DEFINE_BFCNMM_COL_WARP_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
    __global__ void _bfcnmm_col_warp_hetero_kern##SUFFIX(                                               \
        const int32_t* __restrict__ indices,                                                                  \
        const int32_t* __restrict__ indptr,                                                                   \
        const SPIKE_T* __restrict__ matrix_t,                                                                 \
        WEIGHT_T* __restrict__ output,                                                                        \
        const WEIGHT_T* __restrict__ weights,                                                                 \
        int n_pre, int n_post, int n_batch)                                                                   \
    {                                                                                                         \
        int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x); \
        int col = tid >> 5;                                                                                   \
        int lane = tid & 31;                                                                                  \
        int batch = static_cast<int>(blockIdx.y);                                                             \
        if (col >= n_pre || batch >= n_batch) return;                                                         \
        if (!IS_ACTIVE(__ldg(&matrix_t[static_cast<size_t>(batch) * n_pre + col]))) return;                  \
        int start = __ldg(&indptr[col]);                                                                      \
        int end = __ldg(&indptr[col + 1]);                                                                    \
        for (int pos = start + lane; pos < end; pos += BFCNMM_COL_WARP_THREADS) {                            \
            int row = __ldg(&indices[pos]);                                                                   \
            if (row < 0 || row >= n_post) continue;                                                           \
            ACC_T wk = READ_W(__ldg(&weights[pos]));                                                          \
            ATOMIC_ADD_W(&output[static_cast<size_t>(batch) * n_post + row], wk);                            \
        }                                                                                                     \
    }

// ---- float32 ----
DEFINE_BFCNMM_COL_WARP_HOMO(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, atomicAdd)
DEFINE_BFCNMM_COL_WARP_HETERO(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, atomicAdd)
DEFINE_BFCNMM_COL_WARP_HOMO(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, atomicAdd)
DEFINE_BFCNMM_COL_WARP_HETERO(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, atomicAdd)

// ---- float64 ----
DEFINE_BFCNMM_COL_WARP_HOMO(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, atomic_add_f64)
DEFINE_BFCNMM_COL_WARP_HETERO(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, atomic_add_f64)
DEFINE_BFCNMM_COL_WARP_HOMO(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, atomic_add_f64)
DEFINE_BFCNMM_COL_WARP_HETERO(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, atomic_add_f64)

// ---- float16 ----
DEFINE_BFCNMM_COL_WARP_HOMO(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, atomic_add_f16)
DEFINE_BFCNMM_COL_WARP_HETERO(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, atomic_add_f16)
DEFINE_BFCNMM_COL_WARP_HOMO(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, atomic_add_f16)
DEFINE_BFCNMM_COL_WARP_HETERO(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, atomic_add_f16)

// ---- bfloat16 ----
DEFINE_BFCNMM_COL_WARP_HOMO(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BFCNMM_COL_WARP_HETERO(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BFCNMM_COL_WARP_HOMO(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BFCNMM_COL_WARP_HETERO(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)

#define FFI_BFCNMM_COL_HOMO(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                     \
    void binary_fcnmm_col_scatter_homo##SUFFIX(                                                 \
        const BE::Tensor weights, const BE::Tensor indices, const BE::Tensor indptr,            \
        const BE::Tensor matrix, BE::Tensor output, int64_t stream)                              \
    {                                                                                            \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                 \
        int n_pre = static_cast<int>(indptr.size(0)) - 1;                                        \
        int n_post = static_cast<int>(output.size(1));                                           \
        int n_batch = static_cast<int>(matrix.size(0));                                          \
        int nnz = static_cast<int>(indices.size(0));                                             \
        const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr());              \
        const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());                  \
        const int32_t* d_ptr = static_cast<const int32_t*>(indptr.data_ptr());                   \
        const SPIKE_C_T* d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr());               \
        WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                         \
        cudaMemsetAsync(d_out, 0, (size_t)n_batch * n_post * sizeof(WEIGHT_C_T), s);            \
        if (n_pre <= 0 || n_post <= 0 || n_batch <= 0 || nnz <= 0) return;                       \
        int warps_per_block = BFCNMM_COL_WARPS_PER_BLOCK;                                        \
        dim3 block(warps_per_block * BFCNMM_COL_WARP_THREADS);                                   \
        int grid_x = (n_pre + warps_per_block - 1) / warps_per_block;                            \
        dim3 grid(grid_x, n_batch);                                                             \
        _bfcnmm_col_warp_homo_kern##SUFFIX<<<grid, block, 0, s>>>(                               \
            d_idx, d_ptr, d_mat, d_out, d_w, n_pre, n_post, n_batch);                            \
        BE_CHECK_KERNEL_LAUNCH();                                                                 \
    }

#define FFI_BFCNMM_COL_HETERO(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                   \
    void binary_fcnmm_col_scatter_hetero##SUFFIX(                                               \
        const BE::Tensor weights, const BE::Tensor indices, const BE::Tensor indptr,            \
        const BE::Tensor matrix, BE::Tensor output, int64_t stream)                              \
    {                                                                                            \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                 \
        int n_pre = static_cast<int>(indptr.size(0)) - 1;                                        \
        int n_post = static_cast<int>(output.size(1));                                           \
        int n_batch = static_cast<int>(matrix.size(0));                                          \
        int nnz = static_cast<int>(indices.size(0));                                             \
        const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr());              \
        const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());                  \
        const int32_t* d_ptr = static_cast<const int32_t*>(indptr.data_ptr());                   \
        const SPIKE_C_T* d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr());               \
        WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                         \
        cudaMemsetAsync(d_out, 0, (size_t)n_batch * n_post * sizeof(WEIGHT_C_T), s);            \
        if (n_pre <= 0 || n_post <= 0 || n_batch <= 0 || nnz <= 0) return;                       \
        int warps_per_block = BFCNMM_COL_WARPS_PER_BLOCK;                                        \
        dim3 block(warps_per_block * BFCNMM_COL_WARP_THREADS);                                   \
        int grid_x = (n_pre + warps_per_block - 1) / warps_per_block;                            \
        dim3 grid(grid_x, n_batch);                                                             \
        _bfcnmm_col_warp_hetero_kern##SUFFIX<<<grid, block, 0, s>>>(                             \
            d_idx, d_ptr, d_mat, d_out, d_w, n_pre, n_post, n_batch);                            \
        BE_CHECK_KERNEL_LAUNCH();                                                                 \
    }

// ---- float32 ----
// @BE binary_fcnmm_col_scatter_homo_bool_f32
FFI_BFCNMM_COL_HOMO(_bool_f32, float, uint8_t)
// @BE binary_fcnmm_col_scatter_hetero_bool_f32
FFI_BFCNMM_COL_HETERO(_bool_f32, float, uint8_t)
// @BE binary_fcnmm_col_scatter_homo_float_f32
FFI_BFCNMM_COL_HOMO(_float_f32, float, float)
// @BE binary_fcnmm_col_scatter_hetero_float_f32
FFI_BFCNMM_COL_HETERO(_float_f32, float, float)

// ---- float64 ----
// @BE binary_fcnmm_col_scatter_homo_bool_f64
FFI_BFCNMM_COL_HOMO(_bool_f64, double, uint8_t)
// @BE binary_fcnmm_col_scatter_hetero_bool_f64
FFI_BFCNMM_COL_HETERO(_bool_f64, double, uint8_t)
// @BE binary_fcnmm_col_scatter_homo_float_f64
FFI_BFCNMM_COL_HOMO(_float_f64, double, double)
// @BE binary_fcnmm_col_scatter_hetero_float_f64
FFI_BFCNMM_COL_HETERO(_float_f64, double, double)

// ---- float16 ----
// @BE binary_fcnmm_col_scatter_homo_bool_f16
FFI_BFCNMM_COL_HOMO(_bool_f16, __half, uint8_t)
// @BE binary_fcnmm_col_scatter_hetero_bool_f16
FFI_BFCNMM_COL_HETERO(_bool_f16, __half, uint8_t)
// @BE binary_fcnmm_col_scatter_homo_float_f16
FFI_BFCNMM_COL_HOMO(_float_f16, __half, __half)
// @BE binary_fcnmm_col_scatter_hetero_float_f16
FFI_BFCNMM_COL_HETERO(_float_f16, __half, __half)

// ---- bfloat16 ----
// @BE binary_fcnmm_col_scatter_homo_bool_bf16
FFI_BFCNMM_COL_HOMO(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmm_col_scatter_hetero_bool_bf16
FFI_BFCNMM_COL_HETERO(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmm_col_scatter_homo_float_bf16
FFI_BFCNMM_COL_HOMO(_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmm_col_scatter_hetero_float_bf16
FFI_BFCNMM_COL_HETERO(_float_bf16, __nv_bfloat16, __nv_bfloat16)
