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
 * binary_densemm wpr.cu -- Event-driven dense matrix-matrix WPR kernels
 * ==========================================================================
 *
 * This dev backend keeps only two compute kernels:
 *
 * 1. binary_densemm_no_transpose
 *      weights[m, k] @ spikes[k, batch] -> output[m, batch]
 *
 * 2. binary_densemm_transpose
 *      weights[k, m].T @ spikes[k, batch] -> output[m, batch]
 *
 * The CUDA kernels write a physical batch-major buffer:
 *
 *   output[batch, post] -> output[batch * n_post + post]
 *
 * The Python wrapper transposes that physical buffer back to the logical
 * [post, batch] result.  This mirrors the CSR AW path and keeps writes
 * contiguous within each batch row.
 */

#include "cuda_common.h"
#include "brainevent/common.h"

#define DENSEMM_WPR_WARP_THREADS 32
#define DENSEMM_WPR_WARPS_PER_BLOCK 8
#define DENSEMM_WPR_BLOCK_THREADS \
    (DENSEMM_WPR_WARPS_PER_BLOCK * DENSEMM_WPR_WARP_THREADS)

/*
 * Event-driven transpose path:
 *   weights[k,m].T @ spikes[k,batch] -> output[m,batch]
 */

#define DEFINE_TRANSPOSE(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _transpose_kern##SUFFIX(                                                    \
    const WEIGHT_T* __restrict__ weights,                                                   \
    const SPIKE_T*  __restrict__ spikes,                                                    \
    WEIGHT_T*       __restrict__ output,                                                    \
    int k, int m, int n_batch, int is_homo                                                  \
) {                                                                                         \
    int warp_id = static_cast<int>(threadIdx.x) >> 5;                                       \
    int lane = static_cast<int>(threadIdx.x) & 31;                                          \
    int row = static_cast<int>(blockIdx.x) * DENSEMM_WPR_WARPS_PER_BLOCK + warp_id;         \
    int batch = static_cast<int>(blockIdx.y);                                                \
    if (row >= k || batch >= n_batch) return;                                                \
    if (!IS_ACTIVE(__ldg(&spikes[static_cast<size_t>(row) * n_batch + batch]))) return;      \
    ACC_T homo_w = is_homo ? READ_W(__ldg(&weights[0])) : ACC_T(0);                         \
    for (int post = lane; post < m; post += DENSEMM_WPR_WARP_THREADS) {                     \
        ACC_T wk = is_homo ? homo_w : READ_W(__ldg(&weights[static_cast<size_t>(row) * m + post])); \
        ATOMIC_ADD_W(&output[static_cast<size_t>(batch) * m + post], wk);                   \
    }                                                                                       \
}

/*
 * Event-driven no-transpose path:
 *   weights[m,k] @ spikes[k,batch] -> output[m,batch]
 */

#define DEFINE_NO_TRANSPOSE(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _no_transpose_kern##SUFFIX(                                                    \
    const WEIGHT_T* __restrict__ weights,                                                       \
    const SPIKE_T*  __restrict__ spikes,                                                        \
    WEIGHT_T*       __restrict__ output,                                                        \
    int m, int k, int n_batch, int is_homo                                                      \
) {                                                                                             \
    int warp_id = static_cast<int>(threadIdx.x) >> 5;                                           \
    int lane = static_cast<int>(threadIdx.x) & 31;                                              \
    int row = static_cast<int>(blockIdx.x) * DENSEMM_WPR_WARPS_PER_BLOCK + warp_id;             \
    int batch = static_cast<int>(blockIdx.y);                                                    \
    if (row >= k || batch >= n_batch) return;                                                    \
    if (!IS_ACTIVE(__ldg(&spikes[static_cast<size_t>(row) * n_batch + batch]))) return;          \
    ACC_T homo_w = is_homo ? READ_W(__ldg(&weights[0])) : ACC_T(0);                             \
    for (int post = lane; post < m; post += DENSEMM_WPR_WARP_THREADS) {                         \
        ACC_T wk = is_homo ? homo_w : READ_W(__ldg(&weights[static_cast<size_t>(post) * k + row])); \
        ATOMIC_ADD_W(&output[static_cast<size_t>(batch) * m + post], wk);                       \
    }                                                                                           \
}

// Transpose Instantiations
DEFINE_TRANSPOSE(_f32_bool, int8_t, IS_ACTIVE_BOOL, float, float, READ_F32, atomic_add_f32)
DEFINE_TRANSPOSE(_f32_float, float, IS_ACTIVE_FLOAT, float, float, READ_F32, atomic_add_f32)
DEFINE_TRANSPOSE(_f64_bool, int8_t, IS_ACTIVE_BOOL, double, double, READ_F64, atomic_add_f64)
DEFINE_TRANSPOSE(_f64_float, float, IS_ACTIVE_FLOAT, double, double, READ_F64, atomic_add_f64)
DEFINE_TRANSPOSE(_f16_bool, int8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, atomic_add_f16)
DEFINE_TRANSPOSE(_f16_float, float, IS_ACTIVE_FLOAT, __half, float, READ_F16, atomic_add_f16)
DEFINE_TRANSPOSE(_bf16_bool, int8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_TRANSPOSE(_bf16_float, float, IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)

// No-transpose Instantiations
DEFINE_NO_TRANSPOSE(_f32_bool, int8_t, IS_ACTIVE_BOOL, float, float, READ_F32, atomic_add_f32)
DEFINE_NO_TRANSPOSE(_f32_float, float, IS_ACTIVE_FLOAT, float, float, READ_F32, atomic_add_f32)
DEFINE_NO_TRANSPOSE(_f64_bool, int8_t, IS_ACTIVE_BOOL, double, double, READ_F64, atomic_add_f64)
DEFINE_NO_TRANSPOSE(_f64_float, float, IS_ACTIVE_FLOAT, double, double, READ_F64, atomic_add_f64)
DEFINE_NO_TRANSPOSE(_f16_bool, int8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, atomic_add_f16)
DEFINE_NO_TRANSPOSE(_f16_float, float, IS_ACTIVE_FLOAT, __half, float, READ_F16, atomic_add_f16)
DEFINE_NO_TRANSPOSE(_bf16_bool, int8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_NO_TRANSPOSE(_bf16_float, float, IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)

#define DENSEMM_WPR_IS_HOMO(weights) \
    (((weights).ndim() == 0 || (weights).numel() == 1) ? 1 : 0)

// FFI Macros
#define FFI_TRANSPOSE(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                         \
void binary_densemm_transpose##SUFFIX(                                        \
    const BE::Tensor weights, const BE::Tensor spikes,                        \
    BE::Tensor output, int64_t stream                                         \
) {                                                                          \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                  \
    int k = static_cast<int>(spikes.size(0));                                 \
    int n_batch = static_cast<int>(spikes.size(1));                           \
    int m = static_cast<int>(output.size(1));                                 \
    int is_homo = DENSEMM_WPR_IS_HOMO(weights);                               \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());          \
    BE_CUDA_CHECK(cudaMemsetAsync(                                            \
        d_out, 0, static_cast<size_t>(output.size(0)) * output.size(1) * sizeof(WEIGHT_C_T), s)); \
    if (m <= 0 || k <= 0 || n_batch <= 0) return;                             \
    int grid_x = (k + DENSEMM_WPR_WARPS_PER_BLOCK - 1) / DENSEMM_WPR_WARPS_PER_BLOCK; \
    dim3 grid(grid_x, n_batch);                                               \
    _transpose_kern##SUFFIX<<<grid, DENSEMM_WPR_BLOCK_THREADS, 0, s>>>(       \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                   \
        static_cast<const SPIKE_C_T*>(spikes.data_ptr()),                     \
        d_out, k, m, n_batch, is_homo);                                       \
    BE_CHECK_KERNEL_LAUNCH();                                                 \
}

#define FFI_NO_TRANSPOSE(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                      \
void binary_densemm_no_transpose##SUFFIX(                                    \
    const BE::Tensor weights, const BE::Tensor spikes,                       \
    BE::Tensor output, int64_t stream                                        \
) {                                                                         \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                 \
    int k = static_cast<int>(spikes.size(0));                                \
    int n_batch = static_cast<int>(spikes.size(1));                          \
    int m = static_cast<int>(output.size(1));                                \
    int is_homo = DENSEMM_WPR_IS_HOMO(weights);                              \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());         \
    BE_CUDA_CHECK(cudaMemsetAsync(                                           \
        d_out, 0, static_cast<size_t>(output.size(0)) * output.size(1) * sizeof(WEIGHT_C_T), s)); \
    if (m <= 0 || k <= 0 || n_batch <= 0) return;                            \
    int grid_x = (k + DENSEMM_WPR_WARPS_PER_BLOCK - 1) / DENSEMM_WPR_WARPS_PER_BLOCK; \
    dim3 grid(grid_x, n_batch);                                              \
    _no_transpose_kern##SUFFIX<<<grid, DENSEMM_WPR_BLOCK_THREADS, 0, s>>>(   \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                  \
        static_cast<const SPIKE_C_T*>(spikes.data_ptr()),                    \
        d_out, m, k, n_batch, is_homo);                                      \
    BE_CHECK_KERNEL_LAUNCH();                                                \
}

// Transpose FFI Instantiations
// @BE binary_densemm_transpose_f32_bool
FFI_TRANSPOSE(_f32_bool, float, int8_t)
// @BE binary_densemm_transpose_f32_float
FFI_TRANSPOSE(_f32_float, float, float)
// @BE binary_densemm_transpose_f64_bool
FFI_TRANSPOSE(_f64_bool, double, int8_t)
// @BE binary_densemm_transpose_f64_float
FFI_TRANSPOSE(_f64_float, double, float)
// @BE binary_densemm_transpose_f16_bool
FFI_TRANSPOSE(_f16_bool, __half, int8_t)
// @BE binary_densemm_transpose_f16_float
FFI_TRANSPOSE(_f16_float, __half, float)
// @BE binary_densemm_transpose_bf16_bool
FFI_TRANSPOSE(_bf16_bool, __nv_bfloat16, int8_t)
// @BE binary_densemm_transpose_bf16_float
FFI_TRANSPOSE(_bf16_float, __nv_bfloat16, float)

// No-transpose FFI Instantiations
// @BE binary_densemm_no_transpose_f32_bool
FFI_NO_TRANSPOSE(_f32_bool, float, int8_t)
// @BE binary_densemm_no_transpose_f32_float
FFI_NO_TRANSPOSE(_f32_float, float, float)
// @BE binary_densemm_no_transpose_f64_bool
FFI_NO_TRANSPOSE(_f64_bool, double, int8_t)
// @BE binary_densemm_no_transpose_f64_float
FFI_NO_TRANSPOSE(_f64_float, double, float)
// @BE binary_densemm_no_transpose_f16_bool
FFI_NO_TRANSPOSE(_f16_bool, __half, int8_t)
// @BE binary_densemm_no_transpose_f16_float
FFI_NO_TRANSPOSE(_f16_float, __half, float)
// @BE binary_densemm_no_transpose_bf16_bool
FFI_NO_TRANSPOSE(_bf16_bool, __nv_bfloat16, int8_t)
// @BE binary_densemm_no_transpose_bf16_float
FFI_NO_TRANSPOSE(_bf16_float, __nv_bfloat16, float)
