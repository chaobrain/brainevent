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
 * fcnmm_SRAW.cu -- SRAW Event-Driven Binary FCN Matrix-Matrix CUDA Kernels
 * ==============================================================================
 *
 * This module provides the scatter-read / aggregate-write (SRAW) FCN binary
 * matrix-matrix operator.  The kernel reads an event matrix laid out as
 * [stored_rows, n_batch] and writes the physical CUDA output as
 * [n_batch, target_rows].  The Python backend adapts that physical layout back
 * to the public binary_fcnmm transpose=True contract [target_rows, n_batch].
 *
 * Operator: binary_fcnmm SRAW backend
 *   - SRAW scatter mode: output[j, indices[i,k]] += weights[i,k] * is_active(matrix[i,j])
 *
 * Supports weight dtypes: float32, float64, float16, bfloat16
 * Supports spike dtypes:  bool (uint8), float32, float64, float16, bfloat16
 * Supports homo (scalar weight) and hetero (per-connection weight array) modes.
 *
 * SRAW is intended to become the general FCN binary matrix-matrix operator and
 * should replace the current MM operator family after the indexed SRAW variant
 * is implemented and verified.
 *
 * Optimizations:
 *   - Full-warp row processing: each warp handles one stored row and one batch
 *     column, then lanes stride across the fixed connection list.
 *   - Scatter-read / aggregate-write: active events write directly to target
 *     rows with atomics, avoiding the gather path's random target-row reads.
 *   - Column-major physical output: [n_batch, target_rows] improves write
 *     locality for the SRAW kernel; Python transposes it for API compatibility.
 */

#include "cuda_common.h"
#include "brainevent/common.h"

#define TEST_FCNMM_WARP_THREADS 32
#define TEST_FCNMM_WARPS_PER_BLOCK 8
#define TEST_FCNMM_BLOCK_THREADS (TEST_FCNMM_WARPS_PER_BLOCK * TEST_FCNMM_WARP_THREADS)

#define DEFINE_TEST_FCNMM_COLMAJOR_FULLWARP_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
    __global__ void _test_fcnmm_colmajor_fullwarp_homo_kern##SUFFIX(                                                 \
        const int32_t* __restrict__ indices,                                                                          \
        const SPIKE_T* __restrict__ matrix,                                                                           \
        WEIGHT_T*      __restrict__ output,                                                                           \
        const WEIGHT_T* __restrict__ weights,                                                                         \
        int n_pre, int n_conn, int n_post, int n_batch)                                                               \
    {                                                                                                                 \
        int warp_id = threadIdx.x >> 5;                                                                               \
        int lane = threadIdx.x & 31;                                                                                  \
        int row = static_cast<int>(blockIdx.x) * TEST_FCNMM_WARPS_PER_BLOCK + warp_id;                               \
        int j = static_cast<int>(blockIdx.y);                                                                         \
        if (row >= n_pre || j >= n_batch) return;                                                                     \
        ACC_T w0 = READ_W(__ldg(&weights[0]));                                                                        \
        if (!IS_ACTIVE(__ldg(&matrix[static_cast<size_t>(row) * n_batch + j]))) return;                              \
        const int32_t* i_row = indices + static_cast<size_t>(row) * n_conn;                                          \
        for (int k = lane; k < n_conn; k += TEST_FCNMM_WARP_THREADS) {                                               \
            int target = __ldg(&i_row[k]);                                                                            \
            ATOMIC_ADD_W(&output[static_cast<size_t>(j) * n_post + target], w0);                                     \
        }                                                                                                             \
    }

#define DEFINE_TEST_FCNMM_COLMAJOR_FULLWARP_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
    __global__ void _test_fcnmm_colmajor_fullwarp_hetero_kern##SUFFIX(                                                 \
        const int32_t* __restrict__ indices,                                                                           \
        const SPIKE_T* __restrict__ matrix,                                                                            \
        WEIGHT_T*      __restrict__ output,                                                                            \
        const WEIGHT_T* __restrict__ weights,                                                                          \
        int n_pre, int n_conn, int n_post, int n_batch)                                                                \
    {                                                                                                                  \
        int warp_id = threadIdx.x >> 5;                                                                                \
        int lane = threadIdx.x & 31;                                                                                   \
        int row = static_cast<int>(blockIdx.x) * TEST_FCNMM_WARPS_PER_BLOCK + warp_id;                                \
        int j = static_cast<int>(blockIdx.y);                                                                          \
        if (row >= n_pre || j >= n_batch) return;                                                                      \
        if (!IS_ACTIVE(__ldg(&matrix[static_cast<size_t>(row) * n_batch + j]))) return;                               \
        const int32_t* i_row = indices + static_cast<size_t>(row) * n_conn;                                           \
        const WEIGHT_T* w_row = weights + static_cast<size_t>(row) * n_conn;                                          \
        for (int k = lane; k < n_conn; k += TEST_FCNMM_WARP_THREADS) {                                                \
            int target = __ldg(&i_row[k]);                                                                             \
            ACC_T wk = READ_W(__ldg(&w_row[k]));                                                                       \
            ATOMIC_ADD_W(&output[static_cast<size_t>(j) * n_post + target], wk);                                      \
        }                                                                                                              \
    }


DEFINE_TEST_FCNMM_COLMAJOR_FULLWARP_HOMO(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, atomicAdd)
DEFINE_TEST_FCNMM_COLMAJOR_FULLWARP_HOMO(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, atomicAdd)
DEFINE_TEST_FCNMM_COLMAJOR_FULLWARP_HOMO(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, atomic_add_f64)
DEFINE_TEST_FCNMM_COLMAJOR_FULLWARP_HOMO(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, atomic_add_f64)
DEFINE_TEST_FCNMM_COLMAJOR_FULLWARP_HOMO(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, atomic_add_f16)
DEFINE_TEST_FCNMM_COLMAJOR_FULLWARP_HOMO(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, atomic_add_f16)
DEFINE_TEST_FCNMM_COLMAJOR_FULLWARP_HOMO(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_TEST_FCNMM_COLMAJOR_FULLWARP_HOMO(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_TEST_FCNMM_COLMAJOR_FULLWARP_HETERO(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, atomicAdd)
DEFINE_TEST_FCNMM_COLMAJOR_FULLWARP_HETERO(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, atomicAdd)
DEFINE_TEST_FCNMM_COLMAJOR_FULLWARP_HETERO(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, atomic_add_f64)
DEFINE_TEST_FCNMM_COLMAJOR_FULLWARP_HETERO(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, atomic_add_f64)
DEFINE_TEST_FCNMM_COLMAJOR_FULLWARP_HETERO(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, atomic_add_f16)
DEFINE_TEST_FCNMM_COLMAJOR_FULLWARP_HETERO(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, atomic_add_f16)
DEFINE_TEST_FCNMM_COLMAJOR_FULLWARP_HETERO(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_TEST_FCNMM_COLMAJOR_FULLWARP_HETERO(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)

#define FFI_TEST_FCNMM_COLMAJOR_FULLWARP_NOCAP_HOMO(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                        \
    void binary_fcnmm_test_colmajor_fullwarp_nocap_homo##SUFFIX(                                           \
        const BE::Tensor weights, const BE::Tensor indices, const BE::Tensor matrix,                      \
        BE::Tensor output, int64_t stream)                                                                 \
    {                                                                                                      \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                           \
        int n_pre = static_cast<int>(indices.size(0));                                                     \
        int n_conn = static_cast<int>(indices.size(1));                                                    \
        int n_post = static_cast<int>(output.size(1));                                                     \
        int n_batch = static_cast<int>(matrix.size(1));                                                    \
        const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                        \
        const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());                            \
        const SPIKE_C_T* d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr());                         \
        WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                                   \
        BE_CUDA_CHECK(cudaMemsetAsync(                                                                     \
            d_out,                                                                                         \
            0,                                                                                             \
            static_cast<size_t>(output.size(0)) * output.size(1) * sizeof(WEIGHT_C_T),                    \
            s));                                                                                           \
        if (n_pre == 0 || n_conn == 0 || n_batch == 0) return;                                            \
        int warps_per_block = 8;                                                                           \
        int grid_x = (n_pre + warps_per_block - 1) / warps_per_block;                                      \
        dim3 grid(grid_x, n_batch);                                                                        \
        _test_fcnmm_colmajor_fullwarp_homo_kern##SUFFIX<<<grid, 256, 0, s>>>(                            \
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_post, n_batch);                                     \
        BE_CHECK_KERNEL_LAUNCH();                                                                          \
    }

#define FFI_TEST_FCNMM_COLMAJOR_FULLWARP_NOCAP_HETERO(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                       \
    void binary_fcnmm_test_colmajor_fullwarp_nocap_hetero##SUFFIX(                                         \
        const BE::Tensor weights, const BE::Tensor indices, const BE::Tensor matrix,                       \
        BE::Tensor output, int64_t stream)                                                                  \
    {                                                                                                       \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                            \
        int n_pre = static_cast<int>(indices.size(0));                                                      \
        int n_conn = static_cast<int>(indices.size(1));                                                     \
        int n_post = static_cast<int>(output.size(1));                                                      \
        int n_batch = static_cast<int>(matrix.size(1));                                                     \
        const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                         \
        const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());                             \
        const SPIKE_C_T* d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr());                          \
        WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                                    \
        BE_CUDA_CHECK(cudaMemsetAsync(                                                                      \
            d_out,                                                                                          \
            0,                                                                                              \
            static_cast<size_t>(output.size(0)) * output.size(1) * sizeof(WEIGHT_C_T),                     \
            s));                                                                                            \
        if (n_pre == 0 || n_conn == 0 || n_batch == 0) return;                                             \
        int warps_per_block = 8;                                                                            \
        int grid_x = (n_pre + warps_per_block - 1) / warps_per_block;                                       \
        dim3 grid(grid_x, n_batch);                                                                         \
        _test_fcnmm_colmajor_fullwarp_hetero_kern##SUFFIX<<<grid, 256, 0, s>>>(                            \
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_post, n_batch);                                      \
        BE_CHECK_KERNEL_LAUNCH();                                                                           \
    }
    
// @BE binary_fcnmm_test_colmajor_fullwarp_nocap_homo_bool_f32
FFI_TEST_FCNMM_COLMAJOR_FULLWARP_NOCAP_HOMO(_bool_f32, float, uint8_t)
// @BE binary_fcnmm_test_colmajor_fullwarp_nocap_homo_float_f32
FFI_TEST_FCNMM_COLMAJOR_FULLWARP_NOCAP_HOMO(_float_f32, float, float)
// @BE binary_fcnmm_test_colmajor_fullwarp_nocap_homo_bool_f64
FFI_TEST_FCNMM_COLMAJOR_FULLWARP_NOCAP_HOMO(_bool_f64, double, uint8_t)
// @BE binary_fcnmm_test_colmajor_fullwarp_nocap_homo_float_f64
FFI_TEST_FCNMM_COLMAJOR_FULLWARP_NOCAP_HOMO(_float_f64, double, double)
// @BE binary_fcnmm_test_colmajor_fullwarp_nocap_homo_bool_f16
FFI_TEST_FCNMM_COLMAJOR_FULLWARP_NOCAP_HOMO(_bool_f16, __half, uint8_t)
// @BE binary_fcnmm_test_colmajor_fullwarp_nocap_homo_float_f16
FFI_TEST_FCNMM_COLMAJOR_FULLWARP_NOCAP_HOMO(_float_f16, __half, __half)
// @BE binary_fcnmm_test_colmajor_fullwarp_nocap_homo_bool_bf16
FFI_TEST_FCNMM_COLMAJOR_FULLWARP_NOCAP_HOMO(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmm_test_colmajor_fullwarp_nocap_homo_float_bf16
FFI_TEST_FCNMM_COLMAJOR_FULLWARP_NOCAP_HOMO(_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmm_test_colmajor_fullwarp_nocap_hetero_bool_f32
FFI_TEST_FCNMM_COLMAJOR_FULLWARP_NOCAP_HETERO(_bool_f32, float, uint8_t)
// @BE binary_fcnmm_test_colmajor_fullwarp_nocap_hetero_float_f32
FFI_TEST_FCNMM_COLMAJOR_FULLWARP_NOCAP_HETERO(_float_f32, float, float)
// @BE binary_fcnmm_test_colmajor_fullwarp_nocap_hetero_bool_f64
FFI_TEST_FCNMM_COLMAJOR_FULLWARP_NOCAP_HETERO(_bool_f64, double, uint8_t)
// @BE binary_fcnmm_test_colmajor_fullwarp_nocap_hetero_float_f64
FFI_TEST_FCNMM_COLMAJOR_FULLWARP_NOCAP_HETERO(_float_f64, double, double)
// @BE binary_fcnmm_test_colmajor_fullwarp_nocap_hetero_bool_f16
FFI_TEST_FCNMM_COLMAJOR_FULLWARP_NOCAP_HETERO(_bool_f16, __half, uint8_t)
// @BE binary_fcnmm_test_colmajor_fullwarp_nocap_hetero_float_f16
FFI_TEST_FCNMM_COLMAJOR_FULLWARP_NOCAP_HETERO(_float_f16, __half, __half)
// @BE binary_fcnmm_test_colmajor_fullwarp_nocap_hetero_bool_bf16
FFI_TEST_FCNMM_COLMAJOR_FULLWARP_NOCAP_HETERO(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmm_test_colmajor_fullwarp_nocap_hetero_float_bf16
FFI_TEST_FCNMM_COLMAJOR_FULLWARP_NOCAP_HETERO(_float_bf16, __nv_bfloat16, __nv_bfloat16)
