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
 * binary_fcnmm.cu -- Event-Driven Binary FCN Sparse Matrix-Matrix CUDA Kernels
 * ==============================================================================
 *
 * This module provides optimized CUDA kernels for event-driven sparse
 * matrix-matrix multiplication with fixed connection number (FCN).
 *
 * Operator: binary_fcnmm
 *   - Gather mode (transpose=False): output[i,j] = sum_k weights[i,k] * is_active(matrix[indices[i,k], j])
 *   - Scatter mode (transpose=True): output[indices[i,k], j] += weights[i,k] * is_active(matrix[i,j])
 *
 * Supports weight dtypes: float32, float64, float16, bfloat16
 * Supports spike dtypes:  bool (uint8), float32, float64, float16, bfloat16
 * Supports homo (scalar weight) and hetero (per-connection weight array) modes.
 *
 * Optimizations:
 *   - BGM_WARP: Multi-row per block (4 warps), shuffle-based index/weight
 *               loading, branchless accumulation
 *   - BGM_BASIC: Tree reduction, weight caching in shared memory (hetero),
 *                branchless accumulation
 *   - BSM_WARP: Multi-row per block, shuffle-based index/weight loading,
 *               cooperative warp execution (no early thread exit)
 *   - BSM_BASIC: Batch tiling via grid.y, single-pass active column detection,
 *                index/weight caching in shared memory
 */

#include "cuda_common.h"
#include "brainevent/common.h"

// ============================================================================
// FCN Matrix-Matrix Multiplication (fcnmm) — Optimized CUDA Kernels
// ============================================================================

// Rows per block for warp kernels (multi-row for better occupancy)
#define WARP_ROWS_PER_BLOCK 4

// Tile size for batch dimension in BSM_BASIC kernels
#define BSM_BASIC_TILE_J 32

// Maximum grid.x for scatter kernels (grid-stride loop).
// Caps block count to reduce scheduling overhead at large n_pre.
// 4096 blocks × 256 threads ≈ 1M threads → saturates all SMs with margin.
#define BSM_MAX_GRID_X 4096

// Output tiling: scatter to L2-sized output tiles to improve cache hit rate.
// Tile target = 4 × L2_cache_size (empirically optimal on Ampere).
// Queried once from cudaDevAttrL2CacheSize; minimum 4MB.
#define BSM_MIN_TILE 1024

static inline int _bsm_l2_target() {
    static int target = -1;
    if (target < 0) {
        int l2_size = 0;
        cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, 0);
        target = l2_size * 4;
        if (target < 4 * 1024 * 1024) target = 4 * 1024 * 1024;
    }
    return target;
}

// -----------------------------------------------------------------------
// BGM_WARP: Gather warp kernel (n_conn <= 32)
//   - Multi-row per block: 4 warps, each handling one row
//   - Shuffle-based index loading: indices read once into registers,
//     then broadcast via __shfl_sync (zero global memory in inner loop)
//   - Branchless accumulation: accum += (ACC_T)active
// -----------------------------------------------------------------------
#define DEFINE_BGM_WARP_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W)     \
__global__ void _bgm_warp_homo_kern##SUFFIX(                                                   \
    const int32_t* __restrict__ indices,                                                       \
    const SPIKE_T* __restrict__ matrix,                                                        \
    WEIGHT_T*      __restrict__ output,                                                        \
    const WEIGHT_T* __restrict__ weights,                                                      \
    int n_pre, int n_conn, int n_batch                                                         \
) {                                                                                            \
    int warp_id = threadIdx.x >> 5;                                                            \
    int lane    = threadIdx.x & 31;                                                            \
    int rpb     = blockDim.x >> 5;                                                             \
    int row     = blockIdx.x * rpb + warp_id;                                                  \
    int j       = (int)blockIdx.y * 32 + lane;                                                 \
    if (row >= n_pre) return;                                                                  \
    bool col_valid = (j < n_batch);                                                            \
    int  safe_j    = col_valid ? j : 0;                                                        \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                     \
    /* Each thread preloads one index into register */                                         \
    int my_idx = (lane < n_conn) ? __ldg(&i_row[lane]) : 0;                                   \
    ACC_T accum = (ACC_T)0;                                                                    \
    for (int k = 0; k < n_conn; k++) {                                                        \
        /* Broadcast index from lane k — zero global memory traffic */                         \
        int  src    = __shfl_sync(0xffffffff, my_idx, k);                                      \
        bool active = col_valid &&                                                             \
                      IS_ACTIVE(__ldg(&matrix[(size_t)src * n_batch + safe_j]));                \
        accum += (ACC_T)active;  /* branchless */                                              \
    }                                                                                          \
    if (col_valid)                                                                             \
        output[(size_t)row * n_batch + j] = WRITE_W(READ_W(__ldg(&weights[0])) * accum);       \
}

#define DEFINE_BGM_WARP_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W)   \
__global__ void _bgm_warp_hetero_kern##SUFFIX(                                                 \
    const int32_t* __restrict__ indices,                                                       \
    const SPIKE_T* __restrict__ matrix,                                                        \
    WEIGHT_T*      __restrict__ output,                                                        \
    const WEIGHT_T* __restrict__ weights,                                                      \
    int n_pre, int n_conn, int n_batch                                                         \
) {                                                                                            \
    int warp_id = threadIdx.x >> 5;                                                            \
    int lane    = threadIdx.x & 31;                                                            \
    int rpb     = blockDim.x >> 5;                                                             \
    int row     = blockIdx.x * rpb + warp_id;                                                  \
    int j       = (int)blockIdx.y * 32 + lane;                                                 \
    if (row >= n_pre) return;                                                                  \
    bool col_valid = (j < n_batch);                                                            \
    int  safe_j    = col_valid ? j : 0;                                                        \
    const int32_t*  i_row = indices + (size_t)row * n_conn;                                    \
    const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                    \
    /* Each thread preloads one index and one weight */                                        \
    int   my_idx = (lane < n_conn) ? __ldg(&i_row[lane]) : 0;                                 \
    ACC_T my_w   = (lane < n_conn) ? READ_W(__ldg(&w_row[lane])) : (ACC_T)0;                  \
    ACC_T accum  = (ACC_T)0;                                                                   \
    for (int k = 0; k < n_conn; k++) {                                                        \
        int   src = __shfl_sync(0xffffffff, my_idx, k);                                        \
        ACC_T wk  = __shfl_sync(0xffffffff, my_w, k);                                         \
        bool active = col_valid &&                                                             \
                      IS_ACTIVE(__ldg(&matrix[(size_t)src * n_batch + safe_j]));                \
        accum += wk * (ACC_T)active;  /* branchless */                                        \
    }                                                                                          \
    if (col_valid) output[(size_t)row * n_batch + j] = WRITE_W(accum);                         \
}

// -----------------------------------------------------------------------
// BGM_BASIC: Gather basic kernel (n_conn > 32)
//   - Indices cached in shared memory (same as before)
//   - Tree reduction: O(log nwarps) instead of O(nwarps) serial loop
//   - Branchless accumulation
//   - Hetero: weights also cached in shared memory
// -----------------------------------------------------------------------
#define DEFINE_BGM_BASIC_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO)  \
__global__ void _bgm_basic_homo_kern##SUFFIX(                                                          \
    const int32_t* __restrict__ indices,                                                               \
    const SPIKE_T* __restrict__ matrix,                                                                \
    WEIGHT_T*      __restrict__ output,                                                                \
    const WEIGHT_T* __restrict__ weights,                                                              \
    int n_pre, int n_conn, int n_batch                                                                 \
) {                                                                                                    \
    extern __shared__ char _smem_bytes[];                                                              \
    int32_t* s_idx = reinterpret_cast<int32_t*>(_smem_bytes);                                          \
    int row = blockIdx.x;                                                                              \
    if (row >= n_pre) return;                                                                          \
    int lane   = threadIdx.x & 31;                                                                     \
    int warpid = threadIdx.x >> 5;                                                                     \
    int nwarps = blockDim.x >> 5;                                                                      \
    int j = (int)blockIdx.y * 32 + lane;                                                               \
    bool col_valid = (j < n_batch);                                                                    \
    int  safe_j    = col_valid ? j : 0;                                                                \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                             \
    for (int i = threadIdx.x; i < n_conn; i += blockDim.x) s_idx[i] = __ldg(&i_row[i]);               \
    __syncthreads();                                                                                   \
    ACC_T accum = ACC_ZERO;                                                                            \
    for (int k = warpid; k < n_conn; k += nwarps) {                                                   \
        int  src    = s_idx[k];                                                                        \
        bool active = col_valid &&                                                                     \
                      IS_ACTIVE(__ldg(&matrix[(size_t)src * n_batch + safe_j]));                        \
        accum += (ACC_T)active;  /* branchless */                                                      \
    }                                                                                                  \
    /* Tree reduction across warps */                                                                  \
    __syncthreads();                                                                                   \
    ACC_T* smem_red = reinterpret_cast<ACC_T*>(_smem_bytes);                                           \
    smem_red[warpid * 32 + lane] = accum;                                                              \
    __syncthreads();                                                                                   \
    for (int stride = nwarps >> 1; stride > 0; stride >>= 1) {                                        \
        if (warpid < stride)                                                                           \
            smem_red[warpid * 32 + lane] += smem_red[(warpid + stride) * 32 + lane];                   \
        __syncthreads();                                                                               \
    }                                                                                                  \
    if (warpid == 0 && col_valid)                                                                      \
        output[(size_t)row * n_batch + j] = WRITE_W(READ_W(__ldg(&weights[0])) * smem_red[lane]);      \
}

#define DEFINE_BGM_BASIC_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _bgm_basic_hetero_kern##SUFFIX(                                                         \
    const int32_t* __restrict__ indices,                                                                \
    const SPIKE_T* __restrict__ matrix,                                                                 \
    WEIGHT_T*      __restrict__ output,                                                                 \
    const WEIGHT_T* __restrict__ weights,                                                               \
    int n_pre, int n_conn, int n_batch                                                                  \
) {                                                                                                     \
    extern __shared__ char _smem_bytes[];                                                               \
    int32_t*  s_idx = reinterpret_cast<int32_t*>(_smem_bytes);                                          \
    /* Align weight array to 8-byte boundary for all types */                                           \
    size_t wt_off = ((size_t)n_conn * sizeof(int32_t) + 7) & ~(size_t)7;                               \
    WEIGHT_T* s_wt = reinterpret_cast<WEIGHT_T*>(_smem_bytes + wt_off);                                \
    int row = blockIdx.x;                                                                               \
    if (row >= n_pre) return;                                                                           \
    int lane   = threadIdx.x & 31;                                                                      \
    int warpid = threadIdx.x >> 5;                                                                      \
    int nwarps = blockDim.x >> 5;                                                                       \
    int j = (int)blockIdx.y * 32 + lane;                                                                \
    bool col_valid = (j < n_batch);                                                                     \
    int  safe_j    = col_valid ? j : 0;                                                                 \
    /* Load indices AND weights into shared memory */                                                   \
    const int32_t*  i_row = indices + (size_t)row * n_conn;                                             \
    const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                             \
    for (int i = threadIdx.x; i < n_conn; i += blockDim.x) {                                           \
        s_idx[i] = __ldg(&i_row[i]);                                                                   \
        s_wt[i]  = __ldg(&w_row[i]);                                                                   \
    }                                                                                                   \
    __syncthreads();                                                                                    \
    ACC_T accum = ACC_ZERO;                                                                             \
    for (int k = warpid; k < n_conn; k += nwarps) {                                                    \
        int  src    = s_idx[k];                                                                         \
        bool active = col_valid &&                                                                      \
                      IS_ACTIVE(__ldg(&matrix[(size_t)src * n_batch + safe_j]));                         \
        accum += READ_W(s_wt[k]) * (ACC_T)active;  /* branchless, shared mem */                        \
    }                                                                                                   \
    /* Tree reduction across warps */                                                                   \
    __syncthreads();                                                                                    \
    ACC_T* smem_red = reinterpret_cast<ACC_T*>(_smem_bytes);                                            \
    smem_red[warpid * 32 + lane] = accum;                                                               \
    __syncthreads();                                                                                    \
    for (int stride = nwarps >> 1; stride > 0; stride >>= 1) {                                         \
        if (warpid < stride)                                                                            \
            smem_red[warpid * 32 + lane] += smem_red[(warpid + stride) * 32 + lane];                    \
        __syncthreads();                                                                                \
    }                                                                                                   \
    if (warpid == 0 && col_valid) output[(size_t)row * n_batch + j] = WRITE_W(smem_red[lane]);          \
}

// -----------------------------------------------------------------------
// BSM_WARP: Scatter warp kernel (n_conn <= 32)
//   - Multi-row per block (4 warps)
//   - Shuffle-based index/weight loading
//   - All threads stay alive for shuffle; only active threads do atomicAdd
// -----------------------------------------------------------------------
#define DEFINE_BSM_WARP_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _bsm_warp_homo_kern##SUFFIX(                                                    \
    const int32_t* __restrict__ indices,                                                        \
    const SPIKE_T* __restrict__ matrix,                                                         \
    WEIGHT_T*      __restrict__ output,                                                         \
    const WEIGHT_T* __restrict__ weights,                                                       \
    int n_pre, int n_conn, int n_batch, int tile_start, int tile_size                            \
) {                                                                                             \
    int warp_id = threadIdx.x >> 5;                                                             \
    int lane    = threadIdx.x & 31;                                                             \
    int rpb     = blockDim.x >> 5;                                                              \
    int j       = (int)blockIdx.y * 32 + lane;                                                  \
    bool col_valid = (j < n_batch);                                                             \
    int  safe_j    = col_valid ? j : 0;                                                         \
    ACC_T w0 = READ_W(__ldg(&weights[0]));                                                      \
    for (int base = blockIdx.x * rpb; base < n_pre; base += gridDim.x * rpb) {                  \
        int row = base + warp_id;                                                                \
        if (row >= n_pre) continue;                                                              \
        bool active = col_valid &&                                                               \
                      IS_ACTIVE(__ldg(&matrix[(size_t)row * n_batch + safe_j]));                  \
        uint32_t active_mask = __ballot_sync(0xffffffff, active);                                \
        if (active_mask == 0) continue;                                                          \
        const int32_t* i_row = indices + (size_t)row * n_conn;                                   \
        /* Preload indices into registers via shuffle */                                         \
        int my_idx = (lane < n_conn) ? __ldg(&i_row[lane]) : 0;                                 \
        for (int k = 0; k < n_conn; k++) {                                                      \
            int target = __shfl_sync(0xffffffff, my_idx, k);                                     \
            if (active && (unsigned)(target - tile_start) < (unsigned)tile_size)                  \
                ATOMIC_ADD_W(&output[(size_t)target * n_batch + j], w0);                         \
        }                                                                                        \
    }                                                                                            \
}

#define DEFINE_BSM_WARP_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _bsm_warp_hetero_kern##SUFFIX(                                                    \
    const int32_t* __restrict__ indices,                                                          \
    const SPIKE_T* __restrict__ matrix,                                                           \
    WEIGHT_T*      __restrict__ output,                                                           \
    const WEIGHT_T* __restrict__ weights,                                                         \
    int n_pre, int n_conn, int n_batch, int tile_start, int tile_size                              \
) {                                                                                               \
    int warp_id = threadIdx.x >> 5;                                                               \
    int lane    = threadIdx.x & 31;                                                               \
    int rpb     = blockDim.x >> 5;                                                                \
    int j       = (int)blockIdx.y * 32 + lane;                                                    \
    bool col_valid = (j < n_batch);                                                               \
    int  safe_j    = col_valid ? j : 0;                                                           \
    for (int base = blockIdx.x * rpb; base < n_pre; base += gridDim.x * rpb) {                    \
        int row = base + warp_id;                                                                  \
        if (row >= n_pre) continue;                                                                \
        bool active = col_valid &&                                                                 \
                      IS_ACTIVE(__ldg(&matrix[(size_t)row * n_batch + safe_j]));                    \
        uint32_t active_mask = __ballot_sync(0xffffffff, active);                                  \
        if (active_mask == 0) continue;                                                            \
        const int32_t*  i_row = indices + (size_t)row * n_conn;                                    \
        const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                    \
        /* Preload indices and weights into registers via shuffle */                               \
        int   my_idx = (lane < n_conn) ? __ldg(&i_row[lane]) : 0;                                 \
        ACC_T my_w   = (lane < n_conn) ? READ_W(__ldg(&w_row[lane])) : (ACC_T)0;                  \
        for (int k = 0; k < n_conn; k++) {                                                        \
            int   target = __shfl_sync(0xffffffff, my_idx, k);                                     \
            ACC_T wk     = __shfl_sync(0xffffffff, my_w, k);                                       \
            if (active && (unsigned)(target - tile_start) < (unsigned)tile_size)                    \
                ATOMIC_ADD_W(&output[(size_t)target * n_batch + j], wk);                           \
        }                                                                                          \
    }                                                                                              \
}

// -----------------------------------------------------------------------
// BSM_BASIC: Scatter basic kernel (n_conn > 32)
//   - Batch tiling via grid.y: eliminates serial batch loop
//   - Single-pass active column detection (matrix read once, not twice)
//   - Index caching in shared memory (hetero: weights too)
// -----------------------------------------------------------------------
#define DEFINE_BSM_BASIC_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _bsm_basic_homo_kern##SUFFIX(                                                    \
    const int32_t* __restrict__ indices,                                                         \
    const SPIKE_T* __restrict__ matrix,                                                          \
    WEIGHT_T*      __restrict__ output,                                                          \
    const WEIGHT_T* __restrict__ weights,                                                        \
    int n_pre, int n_conn, int n_batch, int tile_start, int tile_size                            \
) {                                                                                              \
    extern __shared__ char _smem_bytes[];                                                        \
    int32_t* s_idx      = reinterpret_cast<int32_t*>(_smem_bytes);                               \
    int*     s_active_j = reinterpret_cast<int*>(s_idx + n_conn);                                \
    int*     s_n_active = s_active_j + BSM_BASIC_TILE_J;                                         \
    int tile_base = (int)blockIdx.y * BSM_BASIC_TILE_J;                                          \
    ACC_T w0 = READ_W(__ldg(&weights[0]));                                                       \
    for (int row = blockIdx.x; row < n_pre; row += gridDim.x) {                                  \
        /* Phase 1: Check activity FIRST (lightweight — only reads matrix) */                    \
        if (threadIdx.x == 0) *s_n_active = 0;                                                  \
        __syncthreads();                                                                         \
        if (threadIdx.x < BSM_BASIC_TILE_J) {                                                   \
            int j_local = tile_base + threadIdx.x;                                               \
            if (j_local < n_batch &&                                                             \
                IS_ACTIVE(__ldg(&matrix[(size_t)row * n_batch + j_local]))) {                    \
                int pos = atomicAdd(s_n_active, 1);                                              \
                s_active_j[pos] = j_local;                                                       \
            }                                                                                    \
        }                                                                                        \
        __syncthreads();                                                                         \
        int n_active = *s_n_active;                                                              \
        if (n_active == 0) continue;                                                             \
        /* Phase 2: Load indices only for active rows */                                         \
        const int32_t* i_row = indices + (size_t)row * n_conn;                                   \
        for (int i = threadIdx.x; i < n_conn; i += blockDim.x)                                   \
            s_idx[i] = __ldg(&i_row[i]);                                                         \
        __syncthreads();                                                                         \
        /* Phase 3: Scatter (filtered to output tile) */                                         \
        for (int a = 0; a < n_active; a++) {                                                     \
            int j = s_active_j[a];                                                               \
            for (int k = threadIdx.x; k < n_conn; k += blockDim.x) {                             \
                int tgt = s_idx[k];                                                              \
                if ((unsigned)(tgt - tile_start) < (unsigned)tile_size)                           \
                    ATOMIC_ADD_W(&output[(size_t)tgt * n_batch + j], w0);                        \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
}

#define DEFINE_BSM_BASIC_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)   \
__global__ void _bsm_basic_hetero_kern##SUFFIX(                                                      \
    const int32_t* __restrict__ indices,                                                             \
    const SPIKE_T* __restrict__ matrix,                                                              \
    WEIGHT_T*      __restrict__ output,                                                              \
    const WEIGHT_T* __restrict__ weights,                                                            \
    int n_pre, int n_conn, int n_batch, int tile_start, int tile_size                                \
) {                                                                                                  \
    extern __shared__ char _smem_bytes[];                                                            \
    int32_t* s_idx = reinterpret_cast<int32_t*>(_smem_bytes);                                        \
    /* Align weight array to 8-byte boundary */                                                      \
    size_t wt_off = ((size_t)n_conn * sizeof(int32_t) + 7) & ~(size_t)7;                            \
    WEIGHT_T* s_wt = reinterpret_cast<WEIGHT_T*>(_smem_bytes + wt_off);                             \
    size_t active_off = wt_off + (size_t)n_conn * sizeof(WEIGHT_T);                                 \
    active_off = (active_off + 3) & ~(size_t)3;  /* align to int */                                 \
    int* s_active_j = reinterpret_cast<int*>(_smem_bytes + active_off);                              \
    int* s_n_active = s_active_j + BSM_BASIC_TILE_J;                                                 \
    int tile_base = (int)blockIdx.y * BSM_BASIC_TILE_J;                                              \
    for (int row = blockIdx.x; row < n_pre; row += gridDim.x) {                                      \
        /* Phase 1: Check activity FIRST (lightweight — only reads matrix) */                        \
        if (threadIdx.x == 0) *s_n_active = 0;                                                       \
        __syncthreads();                                                                             \
        if (threadIdx.x < BSM_BASIC_TILE_J) {                                                        \
            int j_local = tile_base + threadIdx.x;                                                   \
            if (j_local < n_batch &&                                                                 \
                IS_ACTIVE(__ldg(&matrix[(size_t)row * n_batch + j_local]))) {                        \
                int pos = atomicAdd(s_n_active, 1);                                                  \
                s_active_j[pos] = j_local;                                                           \
            }                                                                                        \
        }                                                                                            \
        __syncthreads();                                                                             \
        int n_active = *s_n_active;                                                                  \
        if (n_active == 0) continue;                                                                 \
        /* Phase 2: Load indices and weights only for active rows */                                 \
        const int32_t*  i_row = indices + (size_t)row * n_conn;                                      \
        const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                      \
        for (int i = threadIdx.x; i < n_conn; i += blockDim.x) {                                     \
            s_idx[i] = __ldg(&i_row[i]);                                                             \
            s_wt[i]  = __ldg(&w_row[i]);                                                             \
        }                                                                                            \
        __syncthreads();                                                                             \
        /* Phase 3: Scatter (filtered to output tile) */                                             \
        for (int a = 0; a < n_active; a++) {                                                         \
            int j = s_active_j[a];                                                                   \
            for (int k = threadIdx.x; k < n_conn; k += blockDim.x) {                                 \
                int tgt = s_idx[k];                                                                  \
                if ((unsigned)(tgt - tile_start) < (unsigned)tile_size)                               \
                    ATOMIC_ADD_W(&output[(size_t)tgt * n_batch + j], READ_W(s_wt[k]));               \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
}

// Instantiations
// ---- float32 ----
DEFINE_BGM_WARP_HOMO   (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32)
DEFINE_BGM_WARP_HETERO (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32)
DEFINE_BGM_WARP_HOMO   (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32)
DEFINE_BGM_WARP_HETERO (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32)
DEFINE_BGM_BASIC_HOMO  (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_BGM_BASIC_HETERO(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_BGM_BASIC_HOMO  (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_BGM_BASIC_HETERO(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_BSM_WARP_HOMO   (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, atomicAdd)
DEFINE_BSM_WARP_HETERO (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, atomicAdd)
DEFINE_BSM_WARP_HOMO   (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, atomicAdd)
DEFINE_BSM_WARP_HETERO (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, atomicAdd)
DEFINE_BSM_BASIC_HOMO  (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, atomicAdd)
DEFINE_BSM_BASIC_HETERO(_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, atomicAdd)
DEFINE_BSM_BASIC_HOMO  (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, atomicAdd)
DEFINE_BSM_BASIC_HETERO(_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, atomicAdd)

// ---- float64 ----
DEFINE_BGM_WARP_HOMO   (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64)
DEFINE_BGM_WARP_HETERO (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64)
DEFINE_BGM_WARP_HOMO   (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64)
DEFINE_BGM_WARP_HETERO (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64)
DEFINE_BGM_BASIC_HOMO  (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_BGM_BASIC_HETERO(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_BGM_BASIC_HOMO  (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_BGM_BASIC_HETERO(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_BSM_WARP_HOMO   (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, atomic_add_f64)
DEFINE_BSM_WARP_HETERO (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, atomic_add_f64)
DEFINE_BSM_WARP_HOMO   (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, atomic_add_f64)
DEFINE_BSM_WARP_HETERO (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, atomic_add_f64)
DEFINE_BSM_BASIC_HOMO  (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, atomic_add_f64)
DEFINE_BSM_BASIC_HETERO(_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, atomic_add_f64)
DEFINE_BSM_BASIC_HOMO  (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, atomic_add_f64)
DEFINE_BSM_BASIC_HETERO(_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, atomic_add_f64)

// ---- float16 ----
DEFINE_BGM_WARP_HOMO   (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16)
DEFINE_BGM_WARP_HETERO (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16)
DEFINE_BGM_WARP_HOMO   (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16)
DEFINE_BGM_WARP_HETERO (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16)
DEFINE_BGM_BASIC_HOMO  (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_BGM_BASIC_HETERO(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_BGM_BASIC_HOMO  (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_BGM_BASIC_HETERO(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_BSM_WARP_HOMO   (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, atomic_add_f16)
DEFINE_BSM_WARP_HETERO (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, atomic_add_f16)
DEFINE_BSM_WARP_HOMO   (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, atomic_add_f16)
DEFINE_BSM_WARP_HETERO (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, atomic_add_f16)
DEFINE_BSM_BASIC_HOMO  (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, atomic_add_f16)
DEFINE_BSM_BASIC_HETERO(_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, atomic_add_f16)
DEFINE_BSM_BASIC_HOMO  (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, atomic_add_f16)
DEFINE_BSM_BASIC_HETERO(_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, atomic_add_f16)

// ---- bfloat16 ----
DEFINE_BGM_WARP_HOMO   (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16)
DEFINE_BGM_WARP_HETERO (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16)
DEFINE_BGM_WARP_HOMO   (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16)
DEFINE_BGM_WARP_HETERO (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16)
DEFINE_BGM_BASIC_HOMO  (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_BGM_BASIC_HETERO(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_BGM_BASIC_HOMO  (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_BGM_BASIC_HETERO(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_BSM_WARP_HOMO   (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BSM_WARP_HETERO (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BSM_WARP_HOMO   (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BSM_WARP_HETERO (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BSM_BASIC_HOMO  (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BSM_BASIC_HETERO(_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BSM_BASIC_HOMO  (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BSM_BASIC_HETERO(_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)

// ============================================================================
// FFI Macros for SpMM
// ============================================================================

// ---- FFI macro: gather homo warp (multi-row block) ----
#define FFI_BGM_HOMO_WARP(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                                    \
void binary_fcnmm_gather_homo_warp##SUFFIX(                                                                 \
    const BE::Tensor weights, const BE::Tensor indices,                                                      \
    const BE::Tensor matrix,  BE::Tensor output, int64_t stream                                              \
) {                                                                                                          \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                                 \
    int n_pre   = static_cast<int>(indices.size(0));                                                         \
    int n_conn  = static_cast<int>(indices.size(1));                                                         \
    int n_batch = static_cast<int>(matrix.size(1));                                                          \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                            \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                               \
    const SPIKE_C_T*  d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr());                              \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                                   \
    int rpb = WARP_ROWS_PER_BLOCK;                                                                           \
    int batch_tiles = (n_batch + 31) / 32;                                                                   \
    dim3 grid((n_pre + rpb - 1) / rpb, batch_tiles);                                                         \
    _bgm_warp_homo_kern##SUFFIX<<<grid, rpb * 32, 0, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch); \
}

// ---- FFI macro: gather hetero warp (multi-row block) ----
#define FFI_BGM_HETERO_WARP(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                                    \
void binary_fcnmm_gather_hetero_warp##SUFFIX(                                                                  \
    const BE::Tensor weights, const BE::Tensor indices,                                                        \
    const BE::Tensor matrix,  BE::Tensor output, int64_t stream                                                \
) {                                                                                                            \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                                   \
    int n_pre   = static_cast<int>(indices.size(0));                                                           \
    int n_conn  = static_cast<int>(indices.size(1));                                                           \
    int n_batch = static_cast<int>(matrix.size(1));                                                            \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                              \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                                 \
    const SPIKE_C_T*  d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr());                                \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                                     \
    int rpb = WARP_ROWS_PER_BLOCK;                                                                             \
    int batch_tiles = (n_batch + 31) / 32;                                                                     \
    dim3 grid((n_pre + rpb - 1) / rpb, batch_tiles);                                                           \
    _bgm_warp_hetero_kern##SUFFIX<<<grid, rpb * 32, 0, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch); \
}

// ---- FFI macro: gather homo basic (tree reduction) ----
#define FFI_BGM_HOMO_BASIC(SUFFIX, WEIGHT_C_T, SPIKE_C_T, ACC_SIZE)                                        \
void binary_fcnmm_gather_homo_basic##SUFFIX(                                                               \
    const BE::Tensor weights, const BE::Tensor indices,                                                    \
    const BE::Tensor matrix,  BE::Tensor output, int64_t stream                                            \
) {                                                                                                        \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                               \
    int n_pre   = static_cast<int>(indices.size(0));                                                       \
    int n_conn  = static_cast<int>(indices.size(1));                                                       \
    int n_batch = static_cast<int>(matrix.size(1));                                                        \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                          \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                             \
    const SPIKE_C_T*  d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr());                            \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                                 \
    int bsz = 256; int nwarps = bsz >> 5;                                                                  \
    size_t idx_bytes = (size_t)n_conn * sizeof(int32_t);                                                   \
    size_t red_bytes = (size_t)nwarps * 32 * ACC_SIZE;                                                     \
    size_t shm = (idx_bytes > red_bytes) ? idx_bytes : red_bytes;                                          \
    int batch_tiles = (n_batch + 31) / 32; dim3 grid(n_pre, batch_tiles);                                  \
    _bgm_basic_homo_kern##SUFFIX<<<grid, bsz, shm, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch); \
}

// ---- FFI macro: gather hetero basic (tree reduction + weight caching) ----
#define FFI_BGM_HETERO_BASIC(SUFFIX, WEIGHT_C_T, SPIKE_C_T, ACC_SIZE)                                          \
void binary_fcnmm_gather_hetero_basic##SUFFIX(                                                                 \
    const BE::Tensor weights, const BE::Tensor indices,                                                        \
    const BE::Tensor matrix,  BE::Tensor output, int64_t stream                                                \
) {                                                                                                            \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                                   \
    int n_pre   = static_cast<int>(indices.size(0));                                                           \
    int n_conn  = static_cast<int>(indices.size(1));                                                           \
    int n_batch = static_cast<int>(matrix.size(1));                                                            \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                              \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                                 \
    const SPIKE_C_T*  d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr());                                \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                                     \
    int bsz = 256; int nwarps = bsz >> 5;                                                                      \
    size_t wt_off = ((size_t)n_conn * sizeof(int32_t) + 7) & ~(size_t)7;                                      \
    size_t idx_wt_bytes = wt_off + (size_t)n_conn * sizeof(WEIGHT_C_T);                                        \
    size_t red_bytes = (size_t)nwarps * 32 * ACC_SIZE;                                                          \
    size_t shm = (idx_wt_bytes > red_bytes) ? idx_wt_bytes : red_bytes;                                        \
    int batch_tiles = (n_batch + 31) / 32; dim3 grid(n_pre, batch_tiles);                                       \
    _bgm_basic_hetero_kern##SUFFIX<<<grid, bsz, shm, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch);   \
}

// ---- FFI macro: scatter homo warp (multi-row block) ----
#define FFI_BSM_HOMO_WARP(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                                    \
void binary_fcnmm_scatter_homo_warp##SUFFIX(                                                                 \
    const BE::Tensor weights, const BE::Tensor indices,                                                      \
    const BE::Tensor matrix,  BE::Tensor output, int64_t stream                                              \
) {                                                                                                          \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                                 \
    int n_pre   = static_cast<int>(indices.size(0));                                                         \
    int n_conn  = static_cast<int>(indices.size(1));                                                         \
    int n_post  = static_cast<int>(output.size(0));                                                          \
    int n_batch = static_cast<int>(matrix.size(1));                                                          \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                            \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                               \
    const SPIKE_C_T*  d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr());                              \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                                   \
    int rpb = WARP_ROWS_PER_BLOCK;                                                                           \
    int batch_tiles = (n_batch + 31) / 32;                                                                   \
    int raw_gx = (n_pre + rpb - 1) / rpb;                                                                   \
    int grid_x = (raw_gx < BSM_MAX_GRID_X) ? raw_gx : BSM_MAX_GRID_X;                                      \
    dim3 grid(grid_x, batch_tiles);                                                                          \
    size_t out_bytes = (size_t)n_post * n_batch * sizeof(WEIGHT_C_T);                                        \
    int tile_post = n_post;                                                                                  \
    if (out_bytes > _bsm_l2_target()) {                                                                         \
        tile_post = _bsm_l2_target() / (int)(n_batch * sizeof(WEIGHT_C_T));                                     \
        if (tile_post < BSM_MIN_TILE) tile_post = BSM_MIN_TILE;                                              \
    }                                                                                                        \
    for (int ts = 0; ts < n_post; ts += tile_post) {                                                         \
        int tsz = ((ts + tile_post) < n_post) ? tile_post : (n_post - ts);                                   \
        cudaMemsetAsync(d_out + (size_t)ts * n_batch, 0,                                                     \
                        (size_t)tsz * n_batch * sizeof(WEIGHT_C_T), s);                                      \
        _bsm_warp_homo_kern##SUFFIX<<<grid, rpb * 32, 0, s>>>(                                               \
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, ts, tsz);                                     \
    }                                                                                                        \
}

// ---- FFI macro: scatter hetero warp (multi-row block) ----
#define FFI_BSM_HETERO_WARP(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                                    \
void binary_fcnmm_scatter_hetero_warp##SUFFIX(                                                                 \
    const BE::Tensor weights, const BE::Tensor indices,                                                        \
    const BE::Tensor matrix,  BE::Tensor output, int64_t stream                                                \
) {                                                                                                            \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                                   \
    int n_pre   = static_cast<int>(indices.size(0));                                                           \
    int n_conn  = static_cast<int>(indices.size(1));                                                           \
    int n_post  = static_cast<int>(output.size(0));                                                            \
    int n_batch = static_cast<int>(matrix.size(1));                                                            \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                              \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                                 \
    const SPIKE_C_T*  d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr());                                \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                                     \
    int rpb = WARP_ROWS_PER_BLOCK;                                                                             \
    int batch_tiles = (n_batch + 31) / 32;                                                                     \
    int raw_gx = (n_pre + rpb - 1) / rpb;                                                                     \
    int grid_x = (raw_gx < BSM_MAX_GRID_X) ? raw_gx : BSM_MAX_GRID_X;                                        \
    dim3 grid(grid_x, batch_tiles);                                                                            \
    size_t out_bytes = (size_t)n_post * n_batch * sizeof(WEIGHT_C_T);                                          \
    int tile_post = n_post;                                                                                    \
    if (out_bytes > _bsm_l2_target()) {                                                                           \
        tile_post = _bsm_l2_target() / (int)(n_batch * sizeof(WEIGHT_C_T));                                       \
        if (tile_post < BSM_MIN_TILE) tile_post = BSM_MIN_TILE;                                                \
    }                                                                                                          \
    for (int ts = 0; ts < n_post; ts += tile_post) {                                                           \
        int tsz = ((ts + tile_post) < n_post) ? tile_post : (n_post - ts);                                     \
        cudaMemsetAsync(d_out + (size_t)ts * n_batch, 0,                                                       \
                        (size_t)tsz * n_batch * sizeof(WEIGHT_C_T), s);                                        \
        _bsm_warp_hetero_kern##SUFFIX<<<grid, rpb * 32, 0, s>>>(                                               \
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, ts, tsz);                                       \
    }                                                                                                          \
}

// ---- FFI macro: scatter homo basic (batch tiling + index caching) ----
#define FFI_BSM_HOMO_BASIC(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                                       \
void binary_fcnmm_scatter_homo_basic##SUFFIX(                                                                    \
    const BE::Tensor weights, const BE::Tensor indices,                                                          \
    const BE::Tensor matrix,  BE::Tensor output, int64_t stream                                                  \
) {                                                                                                              \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                                     \
    int n_pre   = static_cast<int>(indices.size(0));                                                             \
    int n_conn  = static_cast<int>(indices.size(1));                                                             \
    int n_post  = static_cast<int>(output.size(0));                                                              \
    int n_batch = static_cast<int>(matrix.size(1));                                                              \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                                \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                                   \
    const SPIKE_C_T*  d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr());                                  \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                                       \
    int bsz = 256;                                                                                               \
    size_t idx_bytes = (size_t)n_conn * sizeof(int32_t);                                                         \
    size_t active_bytes = (BSM_BASIC_TILE_J + 1) * sizeof(int);                                                  \
    size_t shm = idx_bytes + active_bytes;                                                                       \
    int batch_tiles = (n_batch + BSM_BASIC_TILE_J - 1) / BSM_BASIC_TILE_J;                                      \
    int grid_x = (n_pre < BSM_MAX_GRID_X) ? n_pre : BSM_MAX_GRID_X;                                            \
    dim3 grid(grid_x, batch_tiles);                                                                              \
    size_t out_bytes = (size_t)n_post * n_batch * sizeof(WEIGHT_C_T);                                            \
    int tile_post = n_post;                                                                                      \
    if (out_bytes > _bsm_l2_target()) {                                                                             \
        tile_post = _bsm_l2_target() / (int)(n_batch * sizeof(WEIGHT_C_T));                                         \
        if (tile_post < BSM_MIN_TILE) tile_post = BSM_MIN_TILE;                                                  \
    }                                                                                                            \
    for (int ts = 0; ts < n_post; ts += tile_post) {                                                             \
        int tsz = ((ts + tile_post) < n_post) ? tile_post : (n_post - ts);                                       \
        cudaMemsetAsync(d_out + (size_t)ts * n_batch, 0,                                                         \
                        (size_t)tsz * n_batch * sizeof(WEIGHT_C_T), s);                                          \
        _bsm_basic_homo_kern##SUFFIX<<<grid, bsz, shm, s>>>(                                                    \
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, ts, tsz);                                         \
    }                                                                                                            \
}

// ---- FFI macro: scatter hetero basic (batch tiling + index/weight caching) ----
#define FFI_BSM_HETERO_BASIC(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                                     \
void binary_fcnmm_scatter_hetero_basic##SUFFIX(                                                                  \
    const BE::Tensor weights, const BE::Tensor indices,                                                          \
    const BE::Tensor matrix,  BE::Tensor output, int64_t stream                                                  \
) {                                                                                                              \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                                     \
    int n_pre   = static_cast<int>(indices.size(0));                                                             \
    int n_conn  = static_cast<int>(indices.size(1));                                                             \
    int n_post  = static_cast<int>(output.size(0));                                                              \
    int n_batch = static_cast<int>(matrix.size(1));                                                              \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                                \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                                   \
    const SPIKE_C_T*  d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr());                                  \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                                       \
    int bsz = 256;                                                                                               \
    size_t wt_off = ((size_t)n_conn * sizeof(int32_t) + 7) & ~(size_t)7;                                        \
    size_t idx_wt_bytes = wt_off + (size_t)n_conn * sizeof(WEIGHT_C_T);                                          \
    size_t active_off = (idx_wt_bytes + 3) & ~(size_t)3;                                                        \
    size_t shm = active_off + (BSM_BASIC_TILE_J + 1) * sizeof(int);                                             \
    int batch_tiles = (n_batch + BSM_BASIC_TILE_J - 1) / BSM_BASIC_TILE_J;                                      \
    int grid_x = (n_pre < BSM_MAX_GRID_X) ? n_pre : BSM_MAX_GRID_X;                                            \
    dim3 grid(grid_x, batch_tiles);                                                                              \
    size_t out_bytes = (size_t)n_post * n_batch * sizeof(WEIGHT_C_T);                                            \
    int tile_post = n_post;                                                                                      \
    if (out_bytes > _bsm_l2_target()) {                                                                             \
        tile_post = _bsm_l2_target() / (int)(n_batch * sizeof(WEIGHT_C_T));                                         \
        if (tile_post < BSM_MIN_TILE) tile_post = BSM_MIN_TILE;                                                  \
    }                                                                                                            \
    for (int ts = 0; ts < n_post; ts += tile_post) {                                                             \
        int tsz = ((ts + tile_post) < n_post) ? tile_post : (n_post - ts);                                       \
        cudaMemsetAsync(d_out + (size_t)ts * n_batch, 0,                                                         \
                        (size_t)tsz * n_batch * sizeof(WEIGHT_C_T), s);                                          \
        _bsm_basic_hetero_kern##SUFFIX<<<grid, bsz, shm, s>>>(                                                  \
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, ts, tsz);                                         \
    }                                                                                                            \
}

// ============================================================================
// SpMM FFI Instantiations
// ============================================================================

// ---- float32 ----
// @BE binary_fcnmm_gather_homo_warp_bool_f32
FFI_BGM_HOMO_WARP  (_bool_f32, float, uint8_t)
// @BE binary_fcnmm_gather_hetero_warp_bool_f32
FFI_BGM_HETERO_WARP(_bool_f32, float, uint8_t)
// @BE binary_fcnmm_gather_homo_warp_float_f32
FFI_BGM_HOMO_WARP  (_float_f32, float, float)
// @BE binary_fcnmm_gather_hetero_warp_float_f32
FFI_BGM_HETERO_WARP(_float_f32, float, float)
// @BE binary_fcnmm_gather_homo_basic_bool_f32
FFI_BGM_HOMO_BASIC (_bool_f32, float, uint8_t, sizeof(float))
// @BE binary_fcnmm_gather_hetero_basic_bool_f32
FFI_BGM_HETERO_BASIC(_bool_f32, float, uint8_t, sizeof(float))
// @BE binary_fcnmm_gather_homo_basic_float_f32
FFI_BGM_HOMO_BASIC (_float_f32, float, float, sizeof(float))
// @BE binary_fcnmm_gather_hetero_basic_float_f32
FFI_BGM_HETERO_BASIC(_float_f32, float, float, sizeof(float))
// @BE binary_fcnmm_scatter_homo_warp_bool_f32
FFI_BSM_HOMO_WARP  (_bool_f32, float, uint8_t)
// @BE binary_fcnmm_scatter_hetero_warp_bool_f32
FFI_BSM_HETERO_WARP(_bool_f32, float, uint8_t)
// @BE binary_fcnmm_scatter_homo_warp_float_f32
FFI_BSM_HOMO_WARP  (_float_f32, float, float)
// @BE binary_fcnmm_scatter_hetero_warp_float_f32
FFI_BSM_HETERO_WARP(_float_f32, float, float)
// @BE binary_fcnmm_scatter_homo_basic_bool_f32
FFI_BSM_HOMO_BASIC (_bool_f32, float, uint8_t)
// @BE binary_fcnmm_scatter_hetero_basic_bool_f32
FFI_BSM_HETERO_BASIC(_bool_f32, float, uint8_t)
// @BE binary_fcnmm_scatter_homo_basic_float_f32
FFI_BSM_HOMO_BASIC (_float_f32, float, float)
// @BE binary_fcnmm_scatter_hetero_basic_float_f32
FFI_BSM_HETERO_BASIC(_float_f32, float, float)

// ---- float64 ----
// @BE binary_fcnmm_gather_homo_warp_bool_f64
FFI_BGM_HOMO_WARP  (_bool_f64, double, uint8_t)
// @BE binary_fcnmm_gather_hetero_warp_bool_f64
FFI_BGM_HETERO_WARP(_bool_f64, double, uint8_t)
// @BE binary_fcnmm_gather_homo_warp_float_f64
FFI_BGM_HOMO_WARP  (_float_f64, double, double)
// @BE binary_fcnmm_gather_hetero_warp_float_f64
FFI_BGM_HETERO_WARP(_float_f64, double, double)
// @BE binary_fcnmm_gather_homo_basic_bool_f64
FFI_BGM_HOMO_BASIC (_bool_f64, double, uint8_t, sizeof(double))
// @BE binary_fcnmm_gather_hetero_basic_bool_f64
FFI_BGM_HETERO_BASIC(_bool_f64, double, uint8_t, sizeof(double))
// @BE binary_fcnmm_gather_homo_basic_float_f64
FFI_BGM_HOMO_BASIC (_float_f64, double, double, sizeof(double))
// @BE binary_fcnmm_gather_hetero_basic_float_f64
FFI_BGM_HETERO_BASIC(_float_f64, double, double, sizeof(double))
// @BE binary_fcnmm_scatter_homo_warp_bool_f64
FFI_BSM_HOMO_WARP  (_bool_f64, double, uint8_t)
// @BE binary_fcnmm_scatter_hetero_warp_bool_f64
FFI_BSM_HETERO_WARP(_bool_f64, double, uint8_t)
// @BE binary_fcnmm_scatter_homo_warp_float_f64
FFI_BSM_HOMO_WARP  (_float_f64, double, double)
// @BE binary_fcnmm_scatter_hetero_warp_float_f64
FFI_BSM_HETERO_WARP(_float_f64, double, double)
// @BE binary_fcnmm_scatter_homo_basic_bool_f64
FFI_BSM_HOMO_BASIC (_bool_f64, double, uint8_t)
// @BE binary_fcnmm_scatter_hetero_basic_bool_f64
FFI_BSM_HETERO_BASIC(_bool_f64, double, uint8_t)
// @BE binary_fcnmm_scatter_homo_basic_float_f64
FFI_BSM_HOMO_BASIC (_float_f64, double, double)
// @BE binary_fcnmm_scatter_hetero_basic_float_f64
FFI_BSM_HETERO_BASIC(_float_f64, double, double)

// ---- float16 ----
// @BE binary_fcnmm_gather_homo_warp_bool_f16
FFI_BGM_HOMO_WARP  (_bool_f16, __half, uint8_t)
// @BE binary_fcnmm_gather_hetero_warp_bool_f16
FFI_BGM_HETERO_WARP(_bool_f16, __half, uint8_t)
// @BE binary_fcnmm_gather_homo_warp_float_f16
FFI_BGM_HOMO_WARP  (_float_f16, __half, __half)
// @BE binary_fcnmm_gather_hetero_warp_float_f16
FFI_BGM_HETERO_WARP(_float_f16, __half, __half)
// @BE binary_fcnmm_gather_homo_basic_bool_f16
FFI_BGM_HOMO_BASIC (_bool_f16, __half, uint8_t, sizeof(float))
// @BE binary_fcnmm_gather_hetero_basic_bool_f16
FFI_BGM_HETERO_BASIC(_bool_f16, __half, uint8_t, sizeof(float))
// @BE binary_fcnmm_gather_homo_basic_float_f16
FFI_BGM_HOMO_BASIC (_float_f16, __half, __half, sizeof(float))
// @BE binary_fcnmm_gather_hetero_basic_float_f16
FFI_BGM_HETERO_BASIC(_float_f16, __half, __half, sizeof(float))
// @BE binary_fcnmm_scatter_homo_warp_bool_f16
FFI_BSM_HOMO_WARP  (_bool_f16, __half, uint8_t)
// @BE binary_fcnmm_scatter_hetero_warp_bool_f16
FFI_BSM_HETERO_WARP(_bool_f16, __half, uint8_t)
// @BE binary_fcnmm_scatter_homo_warp_float_f16
FFI_BSM_HOMO_WARP  (_float_f16, __half, __half)
// @BE binary_fcnmm_scatter_hetero_warp_float_f16
FFI_BSM_HETERO_WARP(_float_f16, __half, __half)
// @BE binary_fcnmm_scatter_homo_basic_bool_f16
FFI_BSM_HOMO_BASIC (_bool_f16, __half, uint8_t)
// @BE binary_fcnmm_scatter_hetero_basic_bool_f16
FFI_BSM_HETERO_BASIC(_bool_f16, __half, uint8_t)
// @BE binary_fcnmm_scatter_homo_basic_float_f16
FFI_BSM_HOMO_BASIC (_float_f16, __half, __half)
// @BE binary_fcnmm_scatter_hetero_basic_float_f16
FFI_BSM_HETERO_BASIC(_float_f16, __half, __half)

// ---- bfloat16 ----
// @BE binary_fcnmm_gather_homo_warp_bool_bf16
FFI_BGM_HOMO_WARP  (_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmm_gather_hetero_warp_bool_bf16
FFI_BGM_HETERO_WARP(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmm_gather_homo_warp_float_bf16
FFI_BGM_HOMO_WARP  (_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmm_gather_hetero_warp_float_bf16
FFI_BGM_HETERO_WARP(_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmm_gather_homo_basic_bool_bf16
FFI_BGM_HOMO_BASIC (_bool_bf16, __nv_bfloat16, uint8_t, sizeof(float))
// @BE binary_fcnmm_gather_hetero_basic_bool_bf16
FFI_BGM_HETERO_BASIC(_bool_bf16, __nv_bfloat16, uint8_t, sizeof(float))
// @BE binary_fcnmm_gather_homo_basic_float_bf16
FFI_BGM_HOMO_BASIC (_float_bf16, __nv_bfloat16, __nv_bfloat16, sizeof(float))
// @BE binary_fcnmm_gather_hetero_basic_float_bf16
FFI_BGM_HETERO_BASIC(_float_bf16, __nv_bfloat16, __nv_bfloat16, sizeof(float))
// @BE binary_fcnmm_scatter_homo_warp_bool_bf16
FFI_BSM_HOMO_WARP  (_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmm_scatter_hetero_warp_bool_bf16
FFI_BSM_HETERO_WARP(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmm_scatter_homo_warp_float_bf16
FFI_BSM_HOMO_WARP  (_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmm_scatter_hetero_warp_float_bf16
FFI_BSM_HETERO_WARP(_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmm_scatter_homo_basic_bool_bf16
FFI_BSM_HOMO_BASIC (_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmm_scatter_hetero_basic_bool_bf16
FFI_BSM_HETERO_BASIC(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmm_scatter_homo_basic_float_bf16
FFI_BSM_HOMO_BASIC (_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmm_scatter_hetero_basic_float_bf16
FFI_BSM_HETERO_BASIC(_float_bf16, __nv_bfloat16, __nv_bfloat16)
