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
 * bitpack_binary_fcnmv.cu -- Bit-Packed Binary FCN Sparse Matrix-Vector CUDA Kernels
 * ====================================================================================
 *
 * Optimised CUDA kernels for event-driven sparse matrix-vector multiplication
 * using bit-packed binary spike vectors.  Each uint32 word stores 32 spikes,
 * reducing spike array bandwidth by 8x compared to bool (uint8) representation.
 *
 * Operator: bitpack_binary_fcnmv
 *   - Gather mode (transpose=False): y[i] = sum_k weights[i,k] * is_active_packed(packed, indices[i,k])
 *   - Scatter mode (transpose=True): output[indices[i,k]] += weights[i,k] * is_active_packed(packed, i)
 *
 * Gather kernels include shared-memory caching of the packed spike array
 * when it fits in 48 KB (up to ~384K neurons).
 *
 * Supports weight dtypes: float32, float64, float16, bfloat16
 * Supports homo (scalar weight) and hetero (per-connection weight array) modes.
 *
 * Kernel strategy mirrors binary_fcnmv.cu:
 *   Gather: TPR for n_conn <= 512, MR for n_conn > 512 (with optional smem).
 *   Scatter: TPR always.
 */

#include "cuda_common.h"
#include "brainevent/common.h"

// ============================================================================
// Bit extraction macros
// ============================================================================

// Extract bit `idx` from packed uint32 array in global memory (via __ldg)
#define IS_ACTIVE_PACKED(packed, idx) \
    ((__ldg(&(packed)[(idx) >> 5]) >> ((idx) & 31)) & 1)

// Extract bit `idx` from packed uint32 array in shared memory
#define IS_ACTIVE_PACKED_SMEM(smem, idx) \
    (((smem)[(idx) >> 5] >> ((idx) & 31)) & 1)

// ============================================================================
// GATHER kernels — bit-packed spikes with optional shared memory
// ============================================================================

// --- Gather TPR homo: shared memory path ---
#define DEFINE_BG_TPR_HOMO_PACKED_SMEM(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _bg_tpr_homo_packed_smem_kern##SUFFIX(                                      \
    const int32_t*  __restrict__ indices,                                                   \
    const uint32_t* __restrict__ packed,                                                    \
    WEIGHT_T*       __restrict__ output,                                                    \
    const WEIGHT_T* __restrict__ weights,                                                   \
    int n_pre, int n_conn, int n_words                                                      \
) {                                                                                         \
    extern __shared__ uint32_t smem_packed[];                                               \
    for (int i = threadIdx.x; i < n_words; i += blockDim.x)                                 \
        smem_packed[i] = packed[i];                                                         \
    __syncthreads();                                                                        \
    int row = blockIdx.x * blockDim.x + threadIdx.x;                                        \
    if (row >= n_pre) return;                                                               \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                  \
    ACC_T val = ACC_ZERO;                                                                   \
    for (int k = 0; k < n_conn; k++) {                                                      \
        int idx = __ldg(&i_row[k]);                                                         \
        if (IS_ACTIVE_PACKED_SMEM(smem_packed, idx))                                        \
            val += (ACC_T)1;                                                                \
    }                                                                                       \
    output[row] = WRITE_W(READ_W(weights[0]) * val);                                        \
}

// --- Gather TPR homo: global memory path ---
#define DEFINE_BG_TPR_HOMO_PACKED(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO)      \
__global__ void _bg_tpr_homo_packed_kern##SUFFIX(                                           \
    const int32_t*  __restrict__ indices,                                                   \
    const uint32_t* __restrict__ packed,                                                    \
    WEIGHT_T*       __restrict__ output,                                                    \
    const WEIGHT_T* __restrict__ weights,                                                   \
    int n_pre, int n_conn                                                                   \
) {                                                                                         \
    int row = blockIdx.x * blockDim.x + threadIdx.x;                                        \
    if (row >= n_pre) return;                                                               \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                  \
    ACC_T val = ACC_ZERO;                                                                   \
    for (int k = 0; k < n_conn; k++) {                                                      \
        int idx = __ldg(&i_row[k]);                                                         \
        if (IS_ACTIVE_PACKED(packed, idx))                                                  \
            val += (ACC_T)1;                                                                \
    }                                                                                       \
    output[row] = WRITE_W(READ_W(weights[0]) * val);                                        \
}

// --- Gather TPR hetero: shared memory path ---
#define DEFINE_BG_TPR_HETERO_PACKED_SMEM(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _bg_tpr_hetero_packed_smem_kern##SUFFIX(                                      \
    const int32_t*  __restrict__ indices,                                                     \
    const uint32_t* __restrict__ packed,                                                      \
    WEIGHT_T*       __restrict__ output,                                                      \
    const WEIGHT_T* __restrict__ weights,                                                     \
    int n_pre, int n_conn, int n_words                                                        \
) {                                                                                           \
    extern __shared__ uint32_t smem_packed[];                                                 \
    for (int i = threadIdx.x; i < n_words; i += blockDim.x)                                   \
        smem_packed[i] = packed[i];                                                           \
    __syncthreads();                                                                          \
    int row = blockIdx.x * blockDim.x + threadIdx.x;                                          \
    if (row >= n_pre) return;                                                                 \
    const int32_t*  i_row = indices + (size_t)row * n_conn;                                   \
    const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                   \
    ACC_T val = ACC_ZERO;                                                                     \
    _Pragma("unroll 4")                                                                       \
    for (int k = 0; k < n_conn; k++) {                                                        \
        int idx = __ldg(&i_row[k]);                                                           \
        if (IS_ACTIVE_PACKED_SMEM(smem_packed, idx))                                          \
            val += READ_W(__ldg(&w_row[k]));                                                  \
    }                                                                                         \
    output[row] = WRITE_W(val);                                                               \
}

// --- Gather TPR hetero: global memory path ---
#define DEFINE_BG_TPR_HETERO_PACKED(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO)     \
__global__ void _bg_tpr_hetero_packed_kern##SUFFIX(                                          \
    const int32_t*  __restrict__ indices,                                                    \
    const uint32_t* __restrict__ packed,                                                     \
    WEIGHT_T*       __restrict__ output,                                                     \
    const WEIGHT_T* __restrict__ weights,                                                    \
    int n_pre, int n_conn                                                                    \
) {                                                                                          \
    int row = blockIdx.x * blockDim.x + threadIdx.x;                                         \
    if (row >= n_pre) return;                                                                \
    const int32_t*  i_row = indices + (size_t)row * n_conn;                                  \
    const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                  \
    ACC_T val = ACC_ZERO;                                                                    \
    _Pragma("unroll 4")                                                                      \
    for (int k = 0; k < n_conn; k++) {                                                       \
        int idx = __ldg(&i_row[k]);                                                          \
        if (IS_ACTIVE_PACKED(packed, idx))                                                   \
            val += READ_W(__ldg(&w_row[k]));                                                 \
    }                                                                                        \
    output[row] = WRITE_W(val);                                                              \
}

// --- Gather MR homo: shared memory path ---
#define DEFINE_BG_MR_HOMO_PACKED_SMEM(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _bg_mr_homo_packed_smem_kern##SUFFIX(                                                \
    const int32_t*  __restrict__ indices,                                                            \
    const uint32_t* __restrict__ packed,                                                             \
    WEIGHT_T*       __restrict__ output,                                                             \
    const WEIGHT_T* __restrict__ weights,                                                            \
    int n_pre, int n_conn, int n_words                                                               \
) {                                                                                                  \
    extern __shared__ uint32_t smem_packed[];                                                        \
    for (int i = threadIdx.x; i < n_words; i += blockDim.x)                                          \
        smem_packed[i] = packed[i];                                                                  \
    __syncthreads();                                                                                 \
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);                                   \
    if (row >= n_pre) return;                                                                        \
    int lane = threadIdx.x & 31;                                                                     \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                           \
    ACC_T val = ACC_ZERO;                                                                            \
    for (int k = lane; k < n_conn; k += 32) {                                                        \
        int idx = __ldg(&i_row[k]);                                                                  \
        if (IS_ACTIVE_PACKED_SMEM(smem_packed, idx))                                                 \
            val += (ACC_T)1;                                                                         \
    }                                                                                                \
    val = WARP_RED(val);                                                                             \
    if (lane == 0)                                                                                   \
        output[row] = WRITE_W(READ_W(weights[0]) * val);                                             \
}

// --- Gather MR homo: global memory path ---
#define DEFINE_BG_MR_HOMO_PACKED(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _bg_mr_homo_packed_kern##SUFFIX(                                                \
    const int32_t*  __restrict__ indices,                                                       \
    const uint32_t* __restrict__ packed,                                                        \
    WEIGHT_T*       __restrict__ output,                                                        \
    const WEIGHT_T* __restrict__ weights,                                                       \
    int n_pre, int n_conn                                                                       \
) {                                                                                             \
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);                              \
    if (row >= n_pre) return;                                                                   \
    int lane = threadIdx.x & 31;                                                                \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                      \
    ACC_T val = ACC_ZERO;                                                                       \
    for (int k = lane; k < n_conn; k += 32) {                                                   \
        int idx = __ldg(&i_row[k]);                                                             \
        if (IS_ACTIVE_PACKED(packed, idx))                                                      \
            val += (ACC_T)1;                                                                    \
    }                                                                                           \
    val = WARP_RED(val);                                                                        \
    if (lane == 0)                                                                              \
        output[row] = WRITE_W(READ_W(weights[0]) * val);                                        \
}

// --- Gather MR hetero: shared memory path ---
#define DEFINE_BG_MR_HETERO_PACKED_SMEM(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _bg_mr_hetero_packed_smem_kern##SUFFIX(                                                \
    const int32_t*  __restrict__ indices,                                                              \
    const uint32_t* __restrict__ packed,                                                               \
    WEIGHT_T*       __restrict__ output,                                                               \
    const WEIGHT_T* __restrict__ weights,                                                              \
    int n_pre, int n_conn, int n_words                                                                 \
) {                                                                                                    \
    extern __shared__ uint32_t smem_packed[];                                                          \
    for (int i = threadIdx.x; i < n_words; i += blockDim.x)                                            \
        smem_packed[i] = packed[i];                                                                    \
    __syncthreads();                                                                                   \
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);                                     \
    if (row >= n_pre) return;                                                                          \
    int lane = threadIdx.x & 31;                                                                       \
    const int32_t*  i_row = indices + (size_t)row * n_conn;                                            \
    const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                            \
    ACC_T val = ACC_ZERO;                                                                              \
    _Pragma("unroll 4")                                                                                \
    for (int k = lane; k < n_conn; k += 32) {                                                          \
        int idx = __ldg(&i_row[k]);                                                                    \
        if (IS_ACTIVE_PACKED_SMEM(smem_packed, idx))                                                   \
            val += READ_W(__ldg(&w_row[k]));                                                           \
    }                                                                                                  \
    val = WARP_RED(val);                                                                               \
    if (lane == 0)                                                                                     \
        output[row] = WRITE_W(val);                                                                    \
}

// --- Gather MR hetero: global memory path ---
#define DEFINE_BG_MR_HETERO_PACKED(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _bg_mr_hetero_packed_kern##SUFFIX(                                                \
    const int32_t*  __restrict__ indices,                                                         \
    const uint32_t* __restrict__ packed,                                                          \
    WEIGHT_T*       __restrict__ output,                                                          \
    const WEIGHT_T* __restrict__ weights,                                                         \
    int n_pre, int n_conn                                                                         \
) {                                                                                               \
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);                                \
    if (row >= n_pre) return;                                                                     \
    int lane = threadIdx.x & 31;                                                                  \
    const int32_t*  i_row = indices + (size_t)row * n_conn;                                       \
    const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                       \
    ACC_T val = ACC_ZERO;                                                                         \
    _Pragma("unroll 4")                                                                            \
    for (int k = lane; k < n_conn; k += 32) {                                                     \
        int idx = __ldg(&i_row[k]);                                                               \
        if (IS_ACTIVE_PACKED(packed, idx))                                                        \
            val += READ_W(__ldg(&w_row[k]));                                                      \
    }                                                                                             \
    val = WARP_RED(val);                                                                          \
    if (lane == 0)                                                                                \
        output[row] = WRITE_W(val);                                                               \
}

// ============================================================================
// SCATTER kernels — bit-packed spikes (no shared memory needed)
// ============================================================================

// --- Scatter TPR homo ---
#define DEFINE_BS_TPR_HOMO_PACKED(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)             \
__global__ void _bs_tpr_homo_packed_kern##SUFFIX(                                             \
    const int32_t*  __restrict__ indices,                                                     \
    const uint32_t* __restrict__ packed,                                                      \
    WEIGHT_T*       __restrict__ output,                                                      \
    const WEIGHT_T* __restrict__ weights,                                                     \
    int n_pre, int n_conn                                                                     \
) {                                                                                           \
    int row = blockIdx.x * blockDim.x + threadIdx.x;                                          \
    if (row >= n_pre) return;                                                                 \
    if (!IS_ACTIVE_PACKED(packed, row)) return;                                               \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                    \
    ACC_T w0 = READ_W(weights[0]);                                                            \
    for (int k = 0; k < n_conn; k++) {                                                        \
        int idx = __ldg(&i_row[k]);                                                           \
        ATOMIC_ADD_W(&output[idx], w0);                                                       \
    }                                                                                         \
}

// --- Scatter TPR hetero ---
#define DEFINE_BS_TPR_HETERO_PACKED(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)           \
__global__ void _bs_tpr_hetero_packed_kern##SUFFIX(                                           \
    const int32_t*  __restrict__ indices,                                                     \
    const uint32_t* __restrict__ packed,                                                      \
    WEIGHT_T*       __restrict__ output,                                                      \
    const WEIGHT_T* __restrict__ weights,                                                     \
    int n_pre, int n_conn                                                                     \
) {                                                                                           \
    int row = blockIdx.x * blockDim.x + threadIdx.x;                                          \
    if (row >= n_pre) return;                                                                 \
    if (!IS_ACTIVE_PACKED(packed, row)) return;                                               \
    const int32_t*  i_row = indices + (size_t)row * n_conn;                                   \
    const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                   \
    for (int k = 0; k < n_conn; k++) {                                                        \
        int idx = __ldg(&i_row[k]);                                                           \
        ACC_T wk = READ_W(__ldg(&w_row[k]));                                                  \
        ATOMIC_ADD_W(&output[idx], wk);                                                       \
    }                                                                                         \
}

// ============================================================================
// Kernel Instantiations
// ============================================================================

// ---- float32 ----
DEFINE_BG_TPR_HOMO_PACKED_SMEM    (_f32, float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_BG_TPR_HOMO_PACKED         (_f32, float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_BG_TPR_HETERO_PACKED_SMEM  (_f32, float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_BG_TPR_HETERO_PACKED       (_f32, float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_BG_MR_HOMO_PACKED_SMEM     (_f32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO_PACKED          (_f32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_PACKED_SMEM   (_f32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_PACKED        (_f32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BS_TPR_HOMO_PACKED         (_f32, float, float, READ_F32, atomicAdd)
DEFINE_BS_TPR_HETERO_PACKED       (_f32, float, float, READ_F32, atomicAdd)

// ---- float64 ----
DEFINE_BG_TPR_HOMO_PACKED_SMEM    (_f64, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_BG_TPR_HOMO_PACKED         (_f64, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_BG_TPR_HETERO_PACKED_SMEM  (_f64, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_BG_TPR_HETERO_PACKED       (_f64, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_BG_MR_HOMO_PACKED_SMEM     (_f64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HOMO_PACKED          (_f64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HETERO_PACKED_SMEM   (_f64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HETERO_PACKED        (_f64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BS_TPR_HOMO_PACKED         (_f64, double, double, READ_F64, atomic_add_f64)
DEFINE_BS_TPR_HETERO_PACKED       (_f64, double, double, READ_F64, atomic_add_f64)

// ---- float16 ----
DEFINE_BG_TPR_HOMO_PACKED_SMEM    (_f16, __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_BG_TPR_HOMO_PACKED         (_f16, __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_BG_TPR_HETERO_PACKED_SMEM  (_f16, __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_BG_TPR_HETERO_PACKED       (_f16, __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_BG_MR_HOMO_PACKED_SMEM     (_f16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO_PACKED          (_f16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_PACKED_SMEM   (_f16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_PACKED        (_f16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BS_TPR_HOMO_PACKED         (_f16, __half, float, READ_F16, atomic_add_f16)
DEFINE_BS_TPR_HETERO_PACKED       (_f16, __half, float, READ_F16, atomic_add_f16)

// ---- bfloat16 ----
DEFINE_BG_TPR_HOMO_PACKED_SMEM    (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_BG_TPR_HOMO_PACKED         (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_BG_TPR_HETERO_PACKED_SMEM  (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_BG_TPR_HETERO_PACKED       (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_BG_MR_HOMO_PACKED_SMEM     (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO_PACKED          (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_PACKED_SMEM   (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO_PACKED        (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BS_TPR_HOMO_PACKED         (_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BS_TPR_HETERO_PACKED       (_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)


// ============================================================================
// FFI Entry Points
// ============================================================================
//
// All entry points accept an int64_t pack_axis attribute indicating which
// axis of the original spike array was packed.  For fcnmv (1-D spikes),
// pack_axis must be 0.  The parameter is reserved for future 2-D packed
// support (fcnmm) where pack_axis selects different bit-extraction logic.

// ---- FFI macro: gather homo packed (auto-select TPR/MR, auto-select smem/global) ----
#define FFI_BG_HOMO_PACKED(SUFFIX, WEIGHT_C_T)                                                    \
void bitpack_binary_fcnmv_gather_homo##SUFFIX(                                                    \
    const BE::Tensor weights, const BE::Tensor indices,                                           \
    const BE::Tensor packed,  BE::Tensor output,                                                  \
    int64_t pack_axis, int64_t stream                                                             \
) {                                                                                               \
    (void)pack_axis;                                                                              \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                      \
    int n_pre   = static_cast<int>(indices.size(0));                                              \
    int n_conn  = static_cast<int>(indices.size(1));                                              \
    int n_words = static_cast<int>(packed.size(0));                                               \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                 \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                    \
    const uint32_t*   d_pk  = static_cast<const uint32_t*>(packed.data_ptr());                    \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                        \
    size_t smem_bytes = (size_t)n_words * sizeof(uint32_t);                                       \
    if (n_conn <= 512) {                                                                          \
        int bsz = 256;                                                                            \
        int n_blocks = (n_pre + bsz - 1) / bsz;                                                  \
        if (smem_bytes <= 48u * 1024u) {                                                          \
            _bg_tpr_homo_packed_smem_kern##SUFFIX<<<n_blocks, bsz, smem_bytes, s>>>(              \
                d_idx, d_pk, d_out, d_w, n_pre, n_conn, n_words);                                \
        } else {                                                                                  \
            _bg_tpr_homo_packed_kern##SUFFIX<<<n_blocks, bsz, 0, s>>>(                            \
                d_idx, d_pk, d_out, d_w, n_pre, n_conn);                                         \
        }                                                                                         \
    } else {                                                                                      \
        int bsz = 256; int rpb = bsz >> 5; int n_blocks = (n_pre + rpb - 1) / rpb;               \
        if (smem_bytes <= 48u * 1024u) {                                                          \
            _bg_mr_homo_packed_smem_kern##SUFFIX<<<n_blocks, bsz, smem_bytes, s>>>(               \
                d_idx, d_pk, d_out, d_w, n_pre, n_conn, n_words);                                \
        } else {                                                                                  \
            _bg_mr_homo_packed_kern##SUFFIX<<<n_blocks, bsz, 0, s>>>(                             \
                d_idx, d_pk, d_out, d_w, n_pre, n_conn);                                         \
        }                                                                                         \
    }                                                                                             \
}

// ---- FFI macro: gather hetero packed ----
#define FFI_BG_HETERO_PACKED(SUFFIX, WEIGHT_C_T)                                                  \
void bitpack_binary_fcnmv_gather_hetero##SUFFIX(                                                  \
    const BE::Tensor weights, const BE::Tensor indices,                                           \
    const BE::Tensor packed,  BE::Tensor output,                                                  \
    int64_t pack_axis, int64_t stream                                                             \
) {                                                                                               \
    (void)pack_axis;                                                                              \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                      \
    int n_pre   = static_cast<int>(indices.size(0));                                              \
    int n_conn  = static_cast<int>(indices.size(1));                                              \
    int n_words = static_cast<int>(packed.size(0));                                               \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                 \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                    \
    const uint32_t*   d_pk  = static_cast<const uint32_t*>(packed.data_ptr());                    \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                        \
    size_t smem_bytes = (size_t)n_words * sizeof(uint32_t);                                       \
    if (n_conn <= 512) {                                                                          \
        int bsz = 256;                                                                            \
        int n_blocks = (n_pre + bsz - 1) / bsz;                                                  \
        if (smem_bytes <= 48u * 1024u) {                                                          \
            _bg_tpr_hetero_packed_smem_kern##SUFFIX<<<n_blocks, bsz, smem_bytes, s>>>(            \
                d_idx, d_pk, d_out, d_w, n_pre, n_conn, n_words);                                \
        } else {                                                                                  \
            _bg_tpr_hetero_packed_kern##SUFFIX<<<n_blocks, bsz, 0, s>>>(                          \
                d_idx, d_pk, d_out, d_w, n_pre, n_conn);                                         \
        }                                                                                         \
    } else {                                                                                      \
        int bsz = 256; int rpb = bsz >> 5; int n_blocks = (n_pre + rpb - 1) / rpb;               \
        if (smem_bytes <= 48u * 1024u) {                                                          \
            _bg_mr_hetero_packed_smem_kern##SUFFIX<<<n_blocks, bsz, smem_bytes, s>>>(             \
                d_idx, d_pk, d_out, d_w, n_pre, n_conn, n_words);                                \
        } else {                                                                                  \
            _bg_mr_hetero_packed_kern##SUFFIX<<<n_blocks, bsz, 0, s>>>(                           \
                d_idx, d_pk, d_out, d_w, n_pre, n_conn);                                         \
        }                                                                                         \
    }                                                                                             \
}

// ---- FFI macro: scatter homo packed (always TPR) ----
#define FFI_BS_HOMO_PACKED(SUFFIX, WEIGHT_C_T)                                                    \
void bitpack_binary_fcnmv_scatter_homo##SUFFIX(                                                   \
    const BE::Tensor weights, const BE::Tensor indices,                                           \
    const BE::Tensor packed,  BE::Tensor output,                                                  \
    int64_t pack_axis, int64_t stream                                                             \
) {                                                                                               \
    (void)pack_axis;                                                                              \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                      \
    int n_pre  = static_cast<int>(indices.size(0));                                               \
    int n_conn = static_cast<int>(indices.size(1));                                               \
    int n_post = static_cast<int>(output.size(0));                                                \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                 \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                    \
    const uint32_t*   d_pk  = static_cast<const uint32_t*>(packed.data_ptr());                    \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                        \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(WEIGHT_C_T), s);                            \
    int bsz = 256;                                                                                \
    int n_blocks = (n_pre + bsz - 1) / bsz;                                                      \
    _bs_tpr_homo_packed_kern##SUFFIX<<<n_blocks, bsz, 0, s>>>(                                    \
        d_idx, d_pk, d_out, d_w, n_pre, n_conn);                                                 \
}

// ---- FFI macro: scatter hetero packed (always TPR) ----
#define FFI_BS_HETERO_PACKED(SUFFIX, WEIGHT_C_T)                                                  \
void bitpack_binary_fcnmv_scatter_hetero##SUFFIX(                                                 \
    const BE::Tensor weights, const BE::Tensor indices,                                           \
    const BE::Tensor packed,  BE::Tensor output,                                                  \
    int64_t pack_axis, int64_t stream                                                             \
) {                                                                                               \
    (void)pack_axis;                                                                              \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                      \
    int n_pre  = static_cast<int>(indices.size(0));                                               \
    int n_conn = static_cast<int>(indices.size(1));                                               \
    int n_post = static_cast<int>(output.size(0));                                                \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                 \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                    \
    const uint32_t*   d_pk  = static_cast<const uint32_t*>(packed.data_ptr());                    \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                        \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(WEIGHT_C_T), s);                            \
    int bsz = 256;                                                                                \
    int n_blocks = (n_pre + bsz - 1) / bsz;                                                      \
    _bs_tpr_hetero_packed_kern##SUFFIX<<<n_blocks, bsz, 0, s>>>(                                  \
        d_idx, d_pk, d_out, d_w, n_pre, n_conn);                                                 \
}

// ============================================================================
// FFI Instantiations
// ============================================================================

// ---- float32 ----
// @BE bitpack_binary_fcnmv_gather_homo_f32
FFI_BG_HOMO_PACKED   (_f32, float)
// @BE bitpack_binary_fcnmv_gather_hetero_f32
FFI_BG_HETERO_PACKED (_f32, float)
// @BE bitpack_binary_fcnmv_scatter_homo_f32
FFI_BS_HOMO_PACKED   (_f32, float)
// @BE bitpack_binary_fcnmv_scatter_hetero_f32
FFI_BS_HETERO_PACKED (_f32, float)

// ---- float64 ----
// @BE bitpack_binary_fcnmv_gather_homo_f64
FFI_BG_HOMO_PACKED   (_f64, double)
// @BE bitpack_binary_fcnmv_gather_hetero_f64
FFI_BG_HETERO_PACKED (_f64, double)
// @BE bitpack_binary_fcnmv_scatter_homo_f64
FFI_BS_HOMO_PACKED   (_f64, double)
// @BE bitpack_binary_fcnmv_scatter_hetero_f64
FFI_BS_HETERO_PACKED (_f64, double)

// ---- float16 ----
// @BE bitpack_binary_fcnmv_gather_homo_f16
FFI_BG_HOMO_PACKED   (_f16, __half)
// @BE bitpack_binary_fcnmv_gather_hetero_f16
FFI_BG_HETERO_PACKED (_f16, __half)
// @BE bitpack_binary_fcnmv_scatter_homo_f16
FFI_BS_HOMO_PACKED   (_f16, __half)
// @BE bitpack_binary_fcnmv_scatter_hetero_f16
FFI_BS_HETERO_PACKED (_f16, __half)

// ---- bfloat16 ----
// @BE bitpack_binary_fcnmv_gather_homo_bf16
FFI_BG_HOMO_PACKED   (_bf16, __nv_bfloat16)
// @BE bitpack_binary_fcnmv_gather_hetero_bf16
FFI_BG_HETERO_PACKED (_bf16, __nv_bfloat16)
// @BE bitpack_binary_fcnmv_scatter_homo_bf16
FFI_BS_HOMO_PACKED   (_bf16, __nv_bfloat16)
// @BE bitpack_binary_fcnmv_scatter_hetero_bf16
FFI_BS_HETERO_PACKED (_bf16, __nv_bfloat16)
