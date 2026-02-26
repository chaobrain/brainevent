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
 * sparse_float_fcnmv.cu -- Sparse-Float FCN Sparse Matrix-Vector CUDA Kernels
 * =============================================================================
 *
 * This module provides optimized CUDA kernels for sparse operations with
 * fixed connection number (FCN) and sparse-float inputs. It includes:
 *   spfloat_fcnmv -- Sparse Matrix-Vector Product (SpMV)
 *
 * These kernels exploit "sparse-float" sparsity: only connections to non-zero
 * floating-point entries contribute to the output, skipping unnecessary work.
 *
 * Supports weight dtypes: float32, float64, float16, bfloat16
 * Supports spike dtypes:  float32, float64, float16, bfloat16
 * Supports homo (scalar weight) and hetero (per-connection weight array) modes.
 */

#include "cuda_common.h"
#include "brainevent/common.h"

// ============================================================================
// FCN Matrix-Vector Multiplication (spfloat_fcnmv)
// ============================================================================

// ---------------------------------------------------------------------------
// Gather warp: 1 warp (32 threads) per row, n_conn <= 64
// Uses __ldg() for read-only cache and __ballot_sync for zero-skip.
// ---------------------------------------------------------------------------
#define DEFINE_SPFLOAT_GATHER_WARP_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _spfloat_gather_warp_homo_kern##SUFFIX(                                               \
    const int32_t* __restrict__ indices,                                                              \
    const WEIGHT_T* __restrict__ vector,                                                              \
    WEIGHT_T* __restrict__ output,                                                                    \
    const WEIGHT_T* __restrict__ weights,                                                             \
    int n_pre, int n_conn                                                                             \
) {                                                                                                   \
    int row = blockIdx.x;                                                                             \
    if (row >= n_pre) return;                                                                         \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                            \
    ACC_T w = READ_W(__ldg(&weights[0]));                                                             \
    ACC_T val = ACC_ZERO;                                                                             \
    for (int base = 0; base < n_conn; base += 32) {                                                   \
        int k = base + threadIdx.x;                                                                   \
        int32_t idx = (k < n_conn) ? __ldg(&i_row[k]) : 0;                                            \
        ACC_T sp = (k < n_conn) ? READ_W(__ldg(&vector[idx])) : ACC_ZERO;                             \
        unsigned ballot = __ballot_sync(0xffffffff, sp != ACC_ZERO);                                  \
        if (ballot && k < n_conn && sp != ACC_ZERO)                                                   \
            val += w * sp;                                                                            \
    }                                                                                                 \
    val = WARP_RED(val);                                                                              \
    if (threadIdx.x == 0)                                                                             \
        output[row] = WRITE_W(val);                                                                   \
}

#define DEFINE_SPFLOAT_GATHER_WARP_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _spfloat_gather_warp_hetero_kern##SUFFIX(                                               \
    const int32_t* __restrict__ indices,                                                                \
    const WEIGHT_T* __restrict__ vector,                                                                \
    WEIGHT_T* __restrict__ output,                                                                      \
    const WEIGHT_T* __restrict__ weights,                                                               \
    int n_pre, int n_conn                                                                               \
) {                                                                                                     \
    int row = blockIdx.x;                                                                               \
    if (row >= n_pre) return;                                                                           \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                              \
    const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                             \
    ACC_T val = ACC_ZERO;                                                                               \
    for (int base = 0; base < n_conn; base += 32) {                                                     \
        int k = base + threadIdx.x;                                                                     \
        int32_t idx = (k < n_conn) ? __ldg(&i_row[k]) : 0;                                              \
        ACC_T sp = (k < n_conn) ? READ_W(__ldg(&vector[idx])) : ACC_ZERO;                               \
        unsigned ballot = __ballot_sync(0xffffffff, sp != ACC_ZERO);                                    \
        if (ballot && k < n_conn && sp != ACC_ZERO)                                                     \
            val += READ_W(__ldg(&w_row[k])) * sp;                                                       \
    }                                                                                                   \
    val = WARP_RED(val);                                                                                \
    if (threadIdx.x == 0)                                                                               \
        output[row] = WRITE_W(val);                                                                     \
}

// ---------------------------------------------------------------------------
// Gather basic: 256 threads (8 warps) per row, for medium n_conn (65..512)
// Uses __ldg() and block-level reduction via shared memory.
// ---------------------------------------------------------------------------
#define DEFINE_SPFLOAT_GATHER_BASIC_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _spfloat_gather_basic_homo_kern##SUFFIX(                                               \
    const int32_t* __restrict__ indices,                                                               \
    const WEIGHT_T* __restrict__ vector,                                                               \
    WEIGHT_T* __restrict__ output,                                                                     \
    const WEIGHT_T* __restrict__ weights,                                                              \
    int n_pre, int n_conn                                                                              \
) {                                                                                                    \
    extern __shared__ char _smem_bytes[];                                                              \
    ACC_T* smem_red = reinterpret_cast<ACC_T*>(_smem_bytes);                                           \
    int row = blockIdx.x;                                                                              \
    if (row >= n_pre) return;                                                                          \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                             \
    ACC_T w = READ_W(__ldg(&weights[0]));                                                              \
    int warp_id = threadIdx.x >> 5;                                                                    \
    int lane    = threadIdx.x & 31;                                                                    \
    ACC_T val = ACC_ZERO;                                                                              \
    for (int base = warp_id * 32; base < n_conn; base += blockDim.x) {                                 \
        int k = base + lane;                                                                           \
        int32_t idx = (k < n_conn) ? __ldg(&i_row[k]) : 0;                                             \
        ACC_T sp = (k < n_conn) ? READ_W(__ldg(&vector[idx])) : ACC_ZERO;                              \
        unsigned ballot = __ballot_sync(0xffffffff, sp != ACC_ZERO);                                   \
        if (ballot && k < n_conn && sp != ACC_ZERO)                                                    \
            val += w * sp;                                                                             \
    }                                                                                                  \
    val = WARP_RED(val);                                                                               \
    if (lane == 0) smem_red[warp_id] = val;                                                            \
    __syncthreads();                                                                                   \
    int n_warps_in_block = blockDim.x >> 5;                                                            \
    val = (threadIdx.x < n_warps_in_block) ? smem_red[lane] : ACC_ZERO;                                \
    if (warp_id == 0) val = WARP_RED(val);                                                             \
    if (threadIdx.x == 0) output[row] = WRITE_W(val);                                                  \
}

#define DEFINE_SPFLOAT_GATHER_BASIC_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _spfloat_gather_basic_hetero_kern##SUFFIX(                                               \
    const int32_t* __restrict__ indices,                                                                 \
    const WEIGHT_T* __restrict__ vector,                                                                 \
    WEIGHT_T* __restrict__ output,                                                                       \
    const WEIGHT_T* __restrict__ weights,                                                                \
    int n_pre, int n_conn                                                                                \
) {                                                                                                      \
    extern __shared__ char _smem_bytes[];                                                                \
    ACC_T* smem_red = reinterpret_cast<ACC_T*>(_smem_bytes);                                             \
    int row = blockIdx.x;                                                                                \
    if (row >= n_pre) return;                                                                            \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                               \
    const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                              \
    int warp_id = threadIdx.x >> 5;                                                                      \
    int lane    = threadIdx.x & 31;                                                                      \
    ACC_T val = ACC_ZERO;                                                                                \
    for (int base = warp_id * 32; base < n_conn; base += blockDim.x) {                                   \
        int k = base + lane;                                                                             \
        int32_t idx = (k < n_conn) ? __ldg(&i_row[k]) : 0;                                               \
        ACC_T sp = (k < n_conn) ? READ_W(__ldg(&vector[idx])) : ACC_ZERO;                                \
        unsigned ballot = __ballot_sync(0xffffffff, sp != ACC_ZERO);                                     \
        if (ballot && k < n_conn && sp != ACC_ZERO)                                                      \
            val += READ_W(__ldg(&w_row[k])) * sp;                                                        \
    }                                                                                                    \
    val = WARP_RED(val);                                                                                 \
    if (lane == 0) smem_red[warp_id] = val;                                                              \
    __syncthreads();                                                                                     \
    int n_warps_in_block = blockDim.x >> 5;                                                              \
    val = (threadIdx.x < n_warps_in_block) ? smem_red[lane] : ACC_ZERO;                                  \
    if (warp_id == 0) val = WARP_RED(val);                                                               \
    if (threadIdx.x == 0) output[row] = WRITE_W(val);                                                    \
}

// ---------------------------------------------------------------------------
// Scatter basic: 1 block (256 threads) per row, uses atomicAdd.
// Entire block skips if vector[row] == 0 (sparse-float early exit).
// ---------------------------------------------------------------------------
#define DEFINE_SPFLOAT_SCATTER_BASIC_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ATOMIC_ADD_W, ACC_ZERO) \
__global__ void _spfloat_scatter_basic_homo_kern##SUFFIX(                                                   \
    const int32_t* __restrict__ indices,                                                                    \
    const WEIGHT_T* __restrict__ vector,                                                                    \
    WEIGHT_T*       __restrict__ output,                                                                    \
    const WEIGHT_T* __restrict__ weights,                                                                   \
    int n_pre, int n_conn                                                                                   \
) {                                                                                                         \
    int row = blockIdx.x;                                                                                   \
    if (row >= n_pre) return;                                                                               \
    ACC_T sp = READ_W(__ldg(&vector[row]));                                                                 \
    if (sp == ACC_ZERO) return;                                                                             \
    ACC_T w0 = READ_W(__ldg(&weights[0]));                                                                  \
    ACC_T homo_wsp = w0 * sp;                                                                               \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                                  \
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x) {                                                \
        ATOMIC_ADD_W(&output[__ldg(&i_row[k])], homo_wsp);                                                  \
    }                                                                                                       \
}

#define DEFINE_SPFLOAT_SCATTER_BASIC_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ATOMIC_ADD_W, ACC_ZERO) \
__global__ void _spfloat_scatter_basic_hetero_kern##SUFFIX(                                                   \
    const int32_t* __restrict__ indices,                                                                      \
    const WEIGHT_T* __restrict__ vector,                                                                      \
    WEIGHT_T*       __restrict__ output,                                                                      \
    const WEIGHT_T* __restrict__ weights,                                                                     \
    int n_pre, int n_conn                                                                                     \
) {                                                                                                           \
    int row = blockIdx.x;                                                                                     \
    if (row >= n_pre) return;                                                                                 \
    ACC_T sp = READ_W(__ldg(&vector[row]));                                                                   \
    if (sp == ACC_ZERO) return;                                                                               \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                                    \
    const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                                   \
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x) {                                                  \
        ATOMIC_ADD_W(&output[__ldg(&i_row[k])], READ_W(__ldg(&w_row[k])) * sp);                               \
    }                                                                                                         \
}

// ---------------------------------------------------------------------------
// Scatter warp: multiple rows per block, 1 warp per row, n_conn <= 32.
// Uses __shfl_sync to broadcast spike value across the warp.
// ---------------------------------------------------------------------------
#define DEFINE_SPFLOAT_SCATTER_WARP_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ATOMIC_ADD_W, ACC_ZERO) \
__global__ void _spfloat_scatter_warp_homo_kern##SUFFIX(                                                   \
    const int32_t* __restrict__ indices,                                                                   \
    const WEIGHT_T* __restrict__ vector,                                                                   \
    WEIGHT_T*       __restrict__ output,                                                                   \
    const WEIGHT_T* __restrict__ weights,                                                                  \
    int n_pre, int n_conn                                                                                  \
) {                                                                                                        \
    int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;                                          \
    int lane_id   = threadIdx.x & 31;                                                                      \
    int num_warps = (gridDim.x * blockDim.x) >> 5;                                                         \
    ACC_T w0 = READ_W(__ldg(&weights[0]));                                                                 \
    for (int row = warp_id; row < n_pre; row += num_warps) {                                               \
        ACC_T sp = (lane_id == 0) ? READ_W(__ldg(&vector[row])) : ACC_ZERO;                                \
        sp = __shfl_sync(0xffffffff, sp, 0);                                                               \
        if (sp == ACC_ZERO) continue;                                                                      \
        ACC_T homo_wsp = w0 * sp;                                                                          \
        const int32_t* i_row = indices + (size_t)row * n_conn;                                             \
        for (int k = lane_id; k < n_conn; k += 32) {                                                       \
            ATOMIC_ADD_W(&output[__ldg(&i_row[k])], homo_wsp);                                             \
        }                                                                                                  \
    }                                                                                                      \
}

#define DEFINE_SPFLOAT_SCATTER_WARP_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ATOMIC_ADD_W, ACC_ZERO) \
__global__ void _spfloat_scatter_warp_hetero_kern##SUFFIX(                                                   \
    const int32_t* __restrict__ indices,                                                                     \
    const WEIGHT_T* __restrict__ vector,                                                                     \
    WEIGHT_T*       __restrict__ output,                                                                     \
    const WEIGHT_T* __restrict__ weights,                                                                    \
    int n_pre, int n_conn                                                                                    \
) {                                                                                                          \
    int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;                                            \
    int lane_id   = threadIdx.x & 31;                                                                        \
    int num_warps = (gridDim.x * blockDim.x) >> 5;                                                           \
    for (int row = warp_id; row < n_pre; row += num_warps) {                                                 \
        ACC_T sp = (lane_id == 0) ? READ_W(__ldg(&vector[row])) : ACC_ZERO;                                  \
        sp = __shfl_sync(0xffffffff, sp, 0);                                                                 \
        if (sp == ACC_ZERO) continue;                                                                        \
        const int32_t* i_row = indices + (size_t)row * n_conn;                                               \
        const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                              \
        for (int k = lane_id; k < n_conn; k += 32) {                                                         \
            ATOMIC_ADD_W(&output[__ldg(&i_row[k])], READ_W(__ldg(&w_row[k])) * sp);                          \
        }                                                                                                    \
    }                                                                                                        \
}

// Instantiations
// ---- float32 ----
DEFINE_SPFLOAT_GATHER_WARP_HOMO   (_f32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_GATHER_WARP_HETERO (_f32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_GATHER_BASIC_HOMO  (_f32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_GATHER_BASIC_HETERO(_f32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_SCATTER_BASIC_HOMO (_f32, float, float, READ_F32, WRITE_F32, atomic_add_f32, 0.0f)
DEFINE_SPFLOAT_SCATTER_BASIC_HETERO(_f32, float, float, READ_F32, WRITE_F32, atomic_add_f32, 0.0f)
DEFINE_SPFLOAT_SCATTER_WARP_HOMO  (_f32, float, float, READ_F32, WRITE_F32, atomic_add_f32, 0.0f)
DEFINE_SPFLOAT_SCATTER_WARP_HETERO(_f32, float, float, READ_F32, WRITE_F32, atomic_add_f32, 0.0f)

// ---- float64 ----
DEFINE_SPFLOAT_GATHER_WARP_HOMO   (_f64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_SPFLOAT_GATHER_WARP_HETERO (_f64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_SPFLOAT_GATHER_BASIC_HOMO  (_f64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_SPFLOAT_GATHER_BASIC_HETERO(_f64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_SPFLOAT_SCATTER_BASIC_HOMO (_f64, double, double, READ_F64, WRITE_F64, atomic_add_f64, 0.0)
DEFINE_SPFLOAT_SCATTER_BASIC_HETERO(_f64, double, double, READ_F64, WRITE_F64, atomic_add_f64, 0.0)
DEFINE_SPFLOAT_SCATTER_WARP_HOMO  (_f64, double, double, READ_F64, WRITE_F64, atomic_add_f64, 0.0)
DEFINE_SPFLOAT_SCATTER_WARP_HETERO(_f64, double, double, READ_F64, WRITE_F64, atomic_add_f64, 0.0)

// ---- float16 ----
DEFINE_SPFLOAT_GATHER_WARP_HOMO   (_f16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_GATHER_WARP_HETERO (_f16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_GATHER_BASIC_HOMO  (_f16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_GATHER_BASIC_HETERO(_f16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_SCATTER_BASIC_HOMO (_f16, __half, float, READ_F16, WRITE_F16, atomic_add_f16, 0.0f)
DEFINE_SPFLOAT_SCATTER_BASIC_HETERO(_f16, __half, float, READ_F16, WRITE_F16, atomic_add_f16, 0.0f)
DEFINE_SPFLOAT_SCATTER_WARP_HOMO  (_f16, __half, float, READ_F16, WRITE_F16, atomic_add_f16, 0.0f)
DEFINE_SPFLOAT_SCATTER_WARP_HETERO(_f16, __half, float, READ_F16, WRITE_F16, atomic_add_f16, 0.0f)

// ---- bfloat16 ----
DEFINE_SPFLOAT_GATHER_WARP_HOMO   (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_GATHER_WARP_HETERO (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_GATHER_BASIC_HOMO  (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_GATHER_BASIC_HETERO(_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_SCATTER_BASIC_HOMO (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, atomic_add_bf16, 0.0f)
DEFINE_SPFLOAT_SCATTER_BASIC_HETERO(_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, atomic_add_bf16, 0.0f)
DEFINE_SPFLOAT_SCATTER_WARP_HOMO  (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, atomic_add_bf16, 0.0f)
DEFINE_SPFLOAT_SCATTER_WARP_HETERO(_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, atomic_add_bf16, 0.0f)

// ---------------------------------------------------------------------------
// Gather shared (float32 only): shared-memory tiling of indices and weights.
// ---------------------------------------------------------------------------
__global__ void _spfloat_gather_shared_homo_kern(const int32_t* __restrict__ indices, const float* __restrict__ vector, float* __restrict__ output, const float* __restrict__ weights, int n_pre, int n_conn) {
    extern __shared__ char smem_raw[];
    int32_t* s_idx = reinterpret_cast<int32_t*>(smem_raw);
    float*   s_red = reinterpret_cast<float*>(smem_raw + blockDim.x * sizeof(int32_t));
    int row = blockIdx.x; if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    float w0 = __ldg(&weights[0]); float val = 0.0f;
    for (int base = 0; base < n_conn; base += blockDim.x) {
        int k = base + threadIdx.x;
        if (k < n_conn) s_idx[threadIdx.x] = __ldg(&i_row[k]);
        __syncthreads();
        int tile = min((int)blockDim.x, n_conn - base);
        if (threadIdx.x < tile) { float sp = __ldg(&vector[s_idx[threadIdx.x]]); if (sp != 0.0f) val += sp; }
        __syncthreads();
    }
    int lane = threadIdx.x & 31, warpid = threadIdx.x >> 5; val = warp_reduce_sum_f32(val);
    if (lane == 0) s_red[warpid] = val; __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5; val = (threadIdx.x < n_warps) ? s_red[lane] : 0.0f;
    if (warpid == 0) val = warp_reduce_sum_f32(val);
    if (threadIdx.x == 0) output[row] = w0 * val;
}

__global__ void _spfloat_gather_shared_hetero_kern(const int32_t* __restrict__ indices, const float* __restrict__ vector, float* __restrict__ output, const float* __restrict__ weights, int n_pre, int n_conn) {
    extern __shared__ char smem_raw[];
    int32_t* s_idx = reinterpret_cast<int32_t*>(smem_raw);
    float*   s_wt  = reinterpret_cast<float*>(smem_raw + blockDim.x * sizeof(int32_t));
    float*   s_red = reinterpret_cast<float*>(smem_raw + blockDim.x * (sizeof(int32_t) + sizeof(float)));
    int row = blockIdx.x; if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = weights + (size_t)row * n_conn;
    float val = 0.0f;
    for (int base = 0; base < n_conn; base += blockDim.x) {
        int k = base + threadIdx.x;
        if (k < n_conn) { s_idx[threadIdx.x] = __ldg(&i_row[k]); s_wt[threadIdx.x] = __ldg(&w_row[k]); }
        __syncthreads();
        int tile = min((int)blockDim.x, n_conn - base);
        if (threadIdx.x < tile) { float sp = __ldg(&vector[s_idx[threadIdx.x]]); if (sp != 0.0f) val += s_wt[threadIdx.x] * sp; }
        __syncthreads();
    }
    int lane = threadIdx.x & 31, warpid = threadIdx.x >> 5; val = warp_reduce_sum_f32(val);
    if (lane == 0) s_red[warpid] = val; __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5; val = (threadIdx.x < n_warps) ? s_red[lane] : 0.0f;
    if (warpid == 0) val = warp_reduce_sum_f32(val);
    if (threadIdx.x == 0) output[row] = val;
}

// SpMV FFI Entry Macros
// ---- FFI macro: gather homo auto ----
#define FFI_SPFLOAT_GATHER_HOMO_AUTO(SUFFIX, WEIGHT_C_T, SHM_SIZE)                                                     \
void spfloat_fcnmv_gather_homo_auto##SUFFIX(                                                                           \
    const BE::Tensor weights, const BE::Tensor indices,                                                                \
    const BE::Tensor vector,  BE::Tensor output, int64_t stream                                                        \
) {                                                                                                                    \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                                                          \
    int n_pre       = static_cast<int>(indices.size(0));                                                               \
    int n_conn      = static_cast<int>(indices.size(1));                                                               \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                                         \
    const WEIGHT_C_T* d_vec = static_cast<const WEIGHT_C_T*>(vector.data_ptr());                                       \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                                             \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                                      \
    if (n_conn <= 64)                                                                                                  \
        _spfloat_gather_warp_homo_kern##SUFFIX<<<n_pre, 32, 0, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn);          \
    else                                                                                                               \
        _spfloat_gather_basic_homo_kern##SUFFIX<<<n_pre, 256, SHM_SIZE, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn); \
}

// ---- FFI macro: gather hetero auto ----
#define FFI_SPFLOAT_GATHER_HETERO_AUTO(SUFFIX, WEIGHT_C_T, SHM_SIZE)                                                     \
void spfloat_fcnmv_gather_hetero_auto##SUFFIX(                                                                           \
    const BE::Tensor weights, const BE::Tensor indices,                                                                  \
    const BE::Tensor vector,  BE::Tensor output, int64_t stream                                                          \
) {                                                                                                                      \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                                                            \
    int n_pre       = static_cast<int>(indices.size(0));                                                                 \
    int n_conn      = static_cast<int>(indices.size(1));                                                                 \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                                           \
    const WEIGHT_C_T* d_vec = static_cast<const WEIGHT_C_T*>(vector.data_ptr());                                         \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                                               \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                                        \
    if (n_conn <= 64)                                                                                                    \
        _spfloat_gather_warp_hetero_kern##SUFFIX<<<n_pre, 32, 0, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn);          \
    else                                                                                                                 \
        _spfloat_gather_basic_hetero_kern##SUFFIX<<<n_pre, 256, SHM_SIZE, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn); \
}

// ---- FFI macro: scatter homo auto ----
#define FFI_SPFLOAT_SCATTER_HOMO_AUTO(SUFFIX, WEIGHT_C_T)                                                        \
void spfloat_fcnmv_scatter_homo_auto##SUFFIX(                                                                    \
    const BE::Tensor weights, const BE::Tensor indices,                                                          \
    const BE::Tensor vector,  BE::Tensor output, int64_t stream                                                  \
) {                                                                                                              \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                                                    \
    int n_pre       = static_cast<int>(indices.size(0));                                                         \
    int n_conn      = static_cast<int>(indices.size(1));                                                         \
    int n_post      = static_cast<int>(output.size(0));                                                          \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                                   \
    const WEIGHT_C_T* d_vec = static_cast<const WEIGHT_C_T*>(vector.data_ptr());                                 \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                                       \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                                \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(WEIGHT_C_T), s);                                           \
    if (n_conn <= 32) {                                                                                          \
        int blocks = (n_pre + 7) / 8;                                                                            \
        _spfloat_scatter_warp_homo_kern##SUFFIX<<<blocks, 256, 0, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn); \
    } else                                                                                                       \
        _spfloat_scatter_basic_homo_kern##SUFFIX<<<n_pre, 256, 0, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn); \
}

// ---- FFI macro: scatter hetero auto ----
#define FFI_SPFLOAT_SCATTER_HETERO_AUTO(SUFFIX, WEIGHT_C_T)                                                        \
void spfloat_fcnmv_scatter_hetero_auto##SUFFIX(                                                                    \
    const BE::Tensor weights, const BE::Tensor indices,                                                            \
    const BE::Tensor vector,  BE::Tensor output, int64_t stream                                                    \
) {                                                                                                                \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                                                      \
    int n_pre       = static_cast<int>(indices.size(0));                                                           \
    int n_conn      = static_cast<int>(indices.size(1));                                                           \
    int n_post      = static_cast<int>(output.size(0));                                                            \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                                     \
    const WEIGHT_C_T* d_vec = static_cast<const WEIGHT_C_T*>(vector.data_ptr());                                   \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                                         \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                                  \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(WEIGHT_C_T), s);                                             \
    if (n_conn <= 32) {                                                                                            \
        int blocks = (n_pre + 7) / 8;                                                                              \
        _spfloat_scatter_warp_hetero_kern##SUFFIX<<<blocks, 256, 0, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn); \
    } else                                                                                                         \
        _spfloat_scatter_basic_hetero_kern##SUFFIX<<<n_pre, 256, 0, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn); \
}

// SpMV FFI Instantiations
// ---- float32 ----
// @BE spfloat_fcnmv_gather_homo_auto_f32
FFI_SPFLOAT_GATHER_HOMO_AUTO  (_f32, float, 32 * sizeof(float))
// @BE spfloat_fcnmv_gather_hetero_auto_f32
FFI_SPFLOAT_GATHER_HETERO_AUTO(_f32, float, 32 * sizeof(float))
// @BE spfloat_fcnmv_scatter_homo_auto_f32
FFI_SPFLOAT_SCATTER_HOMO_AUTO (_f32, float)
// @BE spfloat_fcnmv_scatter_hetero_auto_f32
FFI_SPFLOAT_SCATTER_HETERO_AUTO(_f32, float)

// ---- float64 ----
// @BE spfloat_fcnmv_gather_homo_auto_f64
FFI_SPFLOAT_GATHER_HOMO_AUTO  (_f64, double, 32 * sizeof(double))
// @BE spfloat_fcnmv_gather_hetero_auto_f64
FFI_SPFLOAT_GATHER_HETERO_AUTO(_f64, double, 32 * sizeof(double))
// @BE spfloat_fcnmv_scatter_homo_auto_f64
FFI_SPFLOAT_SCATTER_HOMO_AUTO (_f64, double)
// @BE spfloat_fcnmv_scatter_hetero_auto_f64
FFI_SPFLOAT_SCATTER_HETERO_AUTO(_f64, double)

// ---- float16 ----
// @BE spfloat_fcnmv_gather_homo_auto_f16
FFI_SPFLOAT_GATHER_HOMO_AUTO  (_f16, __half, 32 * sizeof(float))
// @BE spfloat_fcnmv_gather_hetero_auto_f16
FFI_SPFLOAT_GATHER_HETERO_AUTO(_f16, __half, 32 * sizeof(float))
// @BE spfloat_fcnmv_scatter_homo_auto_f16
FFI_SPFLOAT_SCATTER_HOMO_AUTO (_f16, __half)
// @BE spfloat_fcnmv_scatter_hetero_auto_f16
FFI_SPFLOAT_SCATTER_HETERO_AUTO(_f16, __half)

// ---- bfloat16 ----
// @BE spfloat_fcnmv_gather_homo_auto_bf16
FFI_SPFLOAT_GATHER_HOMO_AUTO  (_bf16, __nv_bfloat16, 32 * sizeof(float))
// @BE spfloat_fcnmv_gather_hetero_auto_bf16
FFI_SPFLOAT_GATHER_HETERO_AUTO(_bf16, __nv_bfloat16, 32 * sizeof(float))
// @BE spfloat_fcnmv_scatter_homo_auto_bf16
FFI_SPFLOAT_SCATTER_HOMO_AUTO (_bf16, __nv_bfloat16)
// @BE spfloat_fcnmv_scatter_hetero_auto_bf16
FFI_SPFLOAT_SCATTER_HETERO_AUTO(_bf16, __nv_bfloat16)

// SpMV f32-specific specializations
// @BE spfloat_fcnmv_gather_shared_homo_f32
void spfloat_fcnmv_gather_shared_homo_f32(
    const BE::Tensor weights, const BE::Tensor indices,
    const BE::Tensor vector,  BE::Tensor output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int bk = 256; size_t shm = bk * sizeof(int32_t) + 32 * sizeof(float);
    _spfloat_gather_shared_homo_kern<<<n_pre, bk, shm, s>>>(
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>(vector.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        n_pre, n_conn);
}

// @BE spfloat_fcnmv_gather_shared_hetero_f32
void spfloat_fcnmv_gather_shared_hetero_f32(
    const BE::Tensor weights, const BE::Tensor indices,
    const BE::Tensor vector,  BE::Tensor output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int bk = 256; size_t shm = bk * (sizeof(int32_t) + sizeof(float)) + 32 * sizeof(float);
    _spfloat_gather_shared_hetero_kern<<<n_pre, bk, shm, s>>>(
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>(vector.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        n_pre, n_conn);
}
