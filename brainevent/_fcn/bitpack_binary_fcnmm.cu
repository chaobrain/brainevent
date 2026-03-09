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
 * bitpack_binary_fcnmm.cu -- Bit-Packed Binary FCN Sparse Matrix-Matrix CUDA Kernels
 * ====================================================================================
 *
 * Optimised CUDA kernels for event-driven sparse matrix-matrix multiplication
 * using pre-packed binary spike matrices.  Each uint32 word stores 32 spikes,
 * reducing spike matrix bandwidth by 8x compared to bool (uint8) representation.
 *
 * Operator: bitpack_binary_fcnmm
 *   Gather mode (transpose=False):
 *     output[i,j] = sum_k weights[i,k] * is_active_packed(packed, indices[i,k], j)
 *   Scatter mode (transpose=True):
 *     output[indices[i,k],j] += weights[i,k] * is_active_packed(packed, i, j)
 *
 * Two packing axes are supported:
 *   pack_axis=1 (batch-packed): packed[n_source, n_batch_words]
 *     - Bit b of packed[row, w] ≡ matrix[row, w*32+b]
 *     - Warp-friendly: all 32 lanes read the SAME word (broadcast)
 *   pack_axis=0 (row-packed):   packed[n_source_words, n_batch]
 *     - Bit b of packed[w, col] ≡ matrix[w*32+b, col]
 *     - Coalesced: 32 lanes read 32 consecutive words
 *
 * Supports weight dtypes: float32, float64, float16, bfloat16
 * Supports homo (scalar weight) and hetero (per-connection weight array) modes.
 *
 * Kernel strategy:
 *   Gather:  Warp (n_conn <= 32) or Basic (n_conn > 32)
 *   Scatter: Warp (n_conn <= 32) or Basic (n_conn > 32)
 */

#include "cuda_common.h"
#include "brainevent/common.h"

// ============================================================================
// Constants
// ============================================================================
#define WARP_ROWS_PER_BLOCK 4
#define BGM_MAX_GRID_X 4096
#define BSM_MAX_GRID_X 4096
#define BSM_BASIC_TILE_J 32

// ============================================================================
// GATHER — pack_axis=1 (batch-packed)
//   packed: [n_source, n_batch_words], bit b of word w = matrix[row, w*32+b]
//   Warp reads ONE word per connection (broadcast to all 32 lanes).
// ============================================================================

// --- Gather warp homo, pack_axis=1 ---
#define DEFINE_BGM_A1_WARP_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W)           \
__global__ void _bgm_a1_warp_homo_kern##SUFFIX(                                     \
    const int32_t*  __restrict__ indices,                                            \
    const uint32_t* __restrict__ packed,                                             \
    WEIGHT_T*       __restrict__ output,                                             \
    const WEIGHT_T* __restrict__ weights,                                            \
    int n_pre, int n_conn, int n_batch, int n_batch_words                            \
) {                                                                                 \
    int warp_id = threadIdx.x >> 5;                                                 \
    int lane    = threadIdx.x & 31;                                                 \
    int rpb     = blockDim.x >> 5;                                                  \
    int bw      = (int)blockIdx.y;                                                  \
    int j       = bw * 32 + lane;                                                   \
    bool col_valid = (j < n_batch);                                                 \
    ACC_T w0 = READ_W(__ldg(&weights[0]));                                          \
    for (int base = blockIdx.x * rpb; base < n_pre; base += gridDim.x * rpb) {     \
        int row = base + warp_id;                                                   \
        if (row >= n_pre) continue;                                                 \
        const int32_t* i_row = indices + (size_t)row * n_conn;                      \
        int my_idx = (lane < n_conn) ? __ldg(&i_row[lane]) : 0;                    \
        ACC_T accum = (ACC_T)0;                                                     \
        for (int k = 0; k < n_conn; k++) {                                          \
            int src = __shfl_sync(0xffffffff, my_idx, k);                           \
            uint32_t word = __ldg(&packed[(size_t)src * n_batch_words + bw]);        \
            accum += (ACC_T)(col_valid & ((word >> lane) & 1u));                    \
        }                                                                           \
        if (col_valid)                                                              \
            output[(size_t)row * n_batch + j] = WRITE_W(w0 * accum);               \
    }                                                                               \
}

// --- Gather warp hetero, pack_axis=1 ---
#define DEFINE_BGM_A1_WARP_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W)         \
__global__ void _bgm_a1_warp_hetero_kern##SUFFIX(                                   \
    const int32_t*  __restrict__ indices,                                            \
    const uint32_t* __restrict__ packed,                                             \
    WEIGHT_T*       __restrict__ output,                                             \
    const WEIGHT_T* __restrict__ weights,                                            \
    int n_pre, int n_conn, int n_batch, int n_batch_words                            \
) {                                                                                 \
    int warp_id = threadIdx.x >> 5;                                                 \
    int lane    = threadIdx.x & 31;                                                 \
    int rpb     = blockDim.x >> 5;                                                  \
    int bw      = (int)blockIdx.y;                                                  \
    int j       = bw * 32 + lane;                                                   \
    bool col_valid = (j < n_batch);                                                 \
    for (int base = blockIdx.x * rpb; base < n_pre; base += gridDim.x * rpb) {     \
        int row = base + warp_id;                                                   \
        if (row >= n_pre) continue;                                                 \
        const int32_t*  i_row = indices + (size_t)row * n_conn;                     \
        const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                     \
        int   my_idx = (lane < n_conn) ? __ldg(&i_row[lane]) : 0;                  \
        ACC_T my_w   = (lane < n_conn) ? READ_W(__ldg(&w_row[lane])) : (ACC_T)0;   \
        ACC_T accum  = (ACC_T)0;                                                    \
        for (int k = 0; k < n_conn; k++) {                                          \
            int   src = __shfl_sync(0xffffffff, my_idx, k);                         \
            ACC_T wk  = __shfl_sync(0xffffffff, my_w, k);                           \
            uint32_t word = __ldg(&packed[(size_t)src * n_batch_words + bw]);        \
            accum += wk * (ACC_T)(col_valid & ((word >> lane) & 1u));               \
        }                                                                           \
        if (col_valid) output[(size_t)row * n_batch + j] = WRITE_W(accum);          \
    }                                                                               \
}

// --- Gather basic homo, pack_axis=1 ---
#define DEFINE_BGM_A1_BASIC_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _bgm_a1_basic_homo_kern##SUFFIX(                                      \
    const int32_t*  __restrict__ indices,                                              \
    const uint32_t* __restrict__ packed,                                               \
    WEIGHT_T*       __restrict__ output,                                               \
    const WEIGHT_T* __restrict__ weights,                                              \
    int n_pre, int n_conn, int n_batch, int n_batch_words                              \
) {                                                                                   \
    extern __shared__ char _smem_bytes[];                                              \
    int32_t* s_idx    = reinterpret_cast<int32_t*>(_smem_bytes);                       \
    size_t   red_off  = ((size_t)n_conn * sizeof(int32_t) + 7) & ~(size_t)7;           \
    ACC_T*   smem_red = reinterpret_cast<ACC_T*>(_smem_bytes + red_off);               \
    int lane   = threadIdx.x & 31;                                                     \
    int warpid = threadIdx.x >> 5;                                                     \
    int nwarps = blockDim.x >> 5;                                                      \
    int bw     = (int)blockIdx.y;                                                      \
    int j      = bw * 32 + lane;                                                       \
    bool col_valid = (j < n_batch);                                                    \
    ACC_T w0 = READ_W(__ldg(&weights[0]));                                             \
    for (int row = blockIdx.x; row < n_pre; row += gridDim.x) {                        \
        const int32_t* i_row = indices + (size_t)row * n_conn;                         \
        for (int i = threadIdx.x; i < n_conn; i += blockDim.x)                         \
            s_idx[i] = __ldg(&i_row[i]);                                               \
        __syncthreads();                                                               \
        ACC_T accum = ACC_ZERO;                                                        \
        for (int k = warpid; k < n_conn; k += nwarps) {                                \
            int src = s_idx[k];                                                        \
            uint32_t word = __ldg(&packed[(size_t)src * n_batch_words + bw]);           \
            accum += (ACC_T)(col_valid & ((word >> lane) & 1u));                       \
        }                                                                              \
        __syncthreads();                                                               \
        smem_red[warpid * 32 + lane] = accum;                                          \
        __syncthreads();                                                               \
        for (int stride = nwarps >> 1; stride > 0; stride >>= 1) {                    \
            if (warpid < stride)                                                       \
                smem_red[warpid * 32 + lane] += smem_red[(warpid + stride) * 32 + lane]; \
            __syncthreads();                                                           \
        }                                                                              \
        if (warpid == 0 && col_valid)                                                  \
            output[(size_t)row * n_batch + j] = WRITE_W(w0 * smem_red[lane]);          \
        __syncthreads();                                                               \
    }                                                                                  \
}

// --- Gather basic hetero, pack_axis=1 ---
#define DEFINE_BGM_A1_BASIC_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _bgm_a1_basic_hetero_kern##SUFFIX(                                      \
    const int32_t*  __restrict__ indices,                                                \
    const uint32_t* __restrict__ packed,                                                 \
    WEIGHT_T*       __restrict__ output,                                                 \
    const WEIGHT_T* __restrict__ weights,                                                \
    int n_pre, int n_conn, int n_batch, int n_batch_words                                \
) {                                                                                     \
    extern __shared__ char _smem_bytes[];                                                \
    int32_t*  s_idx = reinterpret_cast<int32_t*>(_smem_bytes);                           \
    size_t wt_off = ((size_t)n_conn * sizeof(int32_t) + 7) & ~(size_t)7;                \
    WEIGHT_T* s_wt = reinterpret_cast<WEIGHT_T*>(_smem_bytes + wt_off);                 \
    size_t red_off = wt_off + (size_t)n_conn * sizeof(WEIGHT_T);                         \
    red_off = (red_off + 7) & ~(size_t)7;                                               \
    ACC_T* smem_red = reinterpret_cast<ACC_T*>(_smem_bytes + red_off);                   \
    int lane   = threadIdx.x & 31;                                                       \
    int warpid = threadIdx.x >> 5;                                                       \
    int nwarps = blockDim.x >> 5;                                                        \
    int bw     = (int)blockIdx.y;                                                        \
    int j      = bw * 32 + lane;                                                         \
    bool col_valid = (j < n_batch);                                                      \
    for (int row = blockIdx.x; row < n_pre; row += gridDim.x) {                          \
        const int32_t*  i_row = indices + (size_t)row * n_conn;                          \
        const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                          \
        for (int i = threadIdx.x; i < n_conn; i += blockDim.x) {                         \
            s_idx[i] = __ldg(&i_row[i]);                                                 \
            s_wt[i]  = __ldg(&w_row[i]);                                                 \
        }                                                                                \
        __syncthreads();                                                                 \
        ACC_T accum = ACC_ZERO;                                                          \
        for (int k = warpid; k < n_conn; k += nwarps) {                                  \
            int src = s_idx[k];                                                          \
            uint32_t word = __ldg(&packed[(size_t)src * n_batch_words + bw]);             \
            accum += READ_W(s_wt[k]) * (ACC_T)(col_valid & ((word >> lane) & 1u));       \
        }                                                                                \
        __syncthreads();                                                                 \
        smem_red[warpid * 32 + lane] = accum;                                            \
        __syncthreads();                                                                 \
        for (int stride = nwarps >> 1; stride > 0; stride >>= 1) {                      \
            if (warpid < stride)                                                         \
                smem_red[warpid * 32 + lane] += smem_red[(warpid + stride) * 32 + lane]; \
            __syncthreads();                                                             \
        }                                                                                \
        if (warpid == 0 && col_valid)                                                    \
            output[(size_t)row * n_batch + j] = WRITE_W(smem_red[lane]);                 \
        __syncthreads();                                                                 \
    }                                                                                    \
}

// ============================================================================
// GATHER — pack_axis=0 (row-packed)
//   packed: [n_source_words, n_batch], bit b of packed[w, j] = matrix[w*32+b, j]
//   Each lane reads a DIFFERENT word (coalesced access).
// ============================================================================

// --- Gather warp homo, pack_axis=0 ---
#define DEFINE_BGM_A0_WARP_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W)           \
__global__ void _bgm_a0_warp_homo_kern##SUFFIX(                                     \
    const int32_t*  __restrict__ indices,                                            \
    const uint32_t* __restrict__ packed,                                             \
    WEIGHT_T*       __restrict__ output,                                             \
    const WEIGHT_T* __restrict__ weights,                                            \
    int n_pre, int n_conn, int n_batch                                               \
) {                                                                                 \
    int warp_id = threadIdx.x >> 5;                                                 \
    int lane    = threadIdx.x & 31;                                                 \
    int rpb     = blockDim.x >> 5;                                                  \
    int bt      = (int)blockIdx.y;                                                  \
    int j       = bt * 32 + lane;                                                   \
    bool col_valid = (j < n_batch);                                                 \
    ACC_T w0 = READ_W(__ldg(&weights[0]));                                          \
    for (int base = blockIdx.x * rpb; base < n_pre; base += gridDim.x * rpb) {     \
        int row = base + warp_id;                                                   \
        if (row >= n_pre) continue;                                                 \
        const int32_t* i_row = indices + (size_t)row * n_conn;                      \
        int my_idx = (lane < n_conn) ? __ldg(&i_row[lane]) : 0;                    \
        ACC_T accum = (ACC_T)0;                                                     \
        for (int k = 0; k < n_conn; k++) {                                          \
            int src = __shfl_sync(0xffffffff, my_idx, k);                           \
            int word_row = src >> 5;                                                \
            int bit_pos  = src & 31;                                                \
            uint32_t word = col_valid ?                                             \
                __ldg(&packed[(size_t)word_row * n_batch + j]) : 0u;                \
            accum += (ACC_T)((word >> bit_pos) & 1u);                               \
        }                                                                           \
        if (col_valid)                                                              \
            output[(size_t)row * n_batch + j] = WRITE_W(w0 * accum);               \
    }                                                                               \
}

// --- Gather warp hetero, pack_axis=0 ---
#define DEFINE_BGM_A0_WARP_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W)         \
__global__ void _bgm_a0_warp_hetero_kern##SUFFIX(                                   \
    const int32_t*  __restrict__ indices,                                            \
    const uint32_t* __restrict__ packed,                                             \
    WEIGHT_T*       __restrict__ output,                                             \
    const WEIGHT_T* __restrict__ weights,                                            \
    int n_pre, int n_conn, int n_batch                                               \
) {                                                                                 \
    int warp_id = threadIdx.x >> 5;                                                 \
    int lane    = threadIdx.x & 31;                                                 \
    int rpb     = blockDim.x >> 5;                                                  \
    int bt      = (int)blockIdx.y;                                                  \
    int j       = bt * 32 + lane;                                                   \
    bool col_valid = (j < n_batch);                                                 \
    for (int base = blockIdx.x * rpb; base < n_pre; base += gridDim.x * rpb) {     \
        int row = base + warp_id;                                                   \
        if (row >= n_pre) continue;                                                 \
        const int32_t*  i_row = indices + (size_t)row * n_conn;                     \
        const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                     \
        int   my_idx = (lane < n_conn) ? __ldg(&i_row[lane]) : 0;                  \
        ACC_T my_w   = (lane < n_conn) ? READ_W(__ldg(&w_row[lane])) : (ACC_T)0;   \
        ACC_T accum  = (ACC_T)0;                                                    \
        for (int k = 0; k < n_conn; k++) {                                          \
            int   src = __shfl_sync(0xffffffff, my_idx, k);                         \
            ACC_T wk  = __shfl_sync(0xffffffff, my_w, k);                           \
            int word_row = src >> 5;                                                \
            int bit_pos  = src & 31;                                                \
            uint32_t word = col_valid ?                                             \
                __ldg(&packed[(size_t)word_row * n_batch + j]) : 0u;                \
            accum += wk * (ACC_T)((word >> bit_pos) & 1u);                          \
        }                                                                           \
        if (col_valid) output[(size_t)row * n_batch + j] = WRITE_W(accum);          \
    }                                                                               \
}

// --- Gather basic homo, pack_axis=0 ---
#define DEFINE_BGM_A0_BASIC_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _bgm_a0_basic_homo_kern##SUFFIX(                                      \
    const int32_t*  __restrict__ indices,                                              \
    const uint32_t* __restrict__ packed,                                               \
    WEIGHT_T*       __restrict__ output,                                               \
    const WEIGHT_T* __restrict__ weights,                                              \
    int n_pre, int n_conn, int n_batch                                                 \
) {                                                                                   \
    extern __shared__ char _smem_bytes[];                                              \
    int32_t* s_idx    = reinterpret_cast<int32_t*>(_smem_bytes);                       \
    size_t   red_off  = ((size_t)n_conn * sizeof(int32_t) + 7) & ~(size_t)7;           \
    ACC_T*   smem_red = reinterpret_cast<ACC_T*>(_smem_bytes + red_off);               \
    int lane   = threadIdx.x & 31;                                                     \
    int warpid = threadIdx.x >> 5;                                                     \
    int nwarps = blockDim.x >> 5;                                                      \
    int bt     = (int)blockIdx.y;                                                      \
    int j      = bt * 32 + lane;                                                       \
    bool col_valid = (j < n_batch);                                                    \
    ACC_T w0 = READ_W(__ldg(&weights[0]));                                             \
    for (int row = blockIdx.x; row < n_pre; row += gridDim.x) {                        \
        const int32_t* i_row = indices + (size_t)row * n_conn;                         \
        for (int i = threadIdx.x; i < n_conn; i += blockDim.x)                         \
            s_idx[i] = __ldg(&i_row[i]);                                               \
        __syncthreads();                                                               \
        ACC_T accum = ACC_ZERO;                                                        \
        for (int k = warpid; k < n_conn; k += nwarps) {                                \
            int src      = s_idx[k];                                                   \
            int word_row = src >> 5;                                                   \
            int bit_pos  = src & 31;                                                   \
            uint32_t word = col_valid ?                                                \
                __ldg(&packed[(size_t)word_row * n_batch + j]) : 0u;                   \
            accum += (ACC_T)((word >> bit_pos) & 1u);                                  \
        }                                                                              \
        __syncthreads();                                                               \
        smem_red[warpid * 32 + lane] = accum;                                          \
        __syncthreads();                                                               \
        for (int stride = nwarps >> 1; stride > 0; stride >>= 1) {                    \
            if (warpid < stride)                                                       \
                smem_red[warpid * 32 + lane] += smem_red[(warpid + stride) * 32 + lane]; \
            __syncthreads();                                                           \
        }                                                                              \
        if (warpid == 0 && col_valid)                                                  \
            output[(size_t)row * n_batch + j] = WRITE_W(w0 * smem_red[lane]);          \
        __syncthreads();                                                               \
    }                                                                                  \
}

// --- Gather basic hetero, pack_axis=0 ---
#define DEFINE_BGM_A0_BASIC_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _bgm_a0_basic_hetero_kern##SUFFIX(                                      \
    const int32_t*  __restrict__ indices,                                                \
    const uint32_t* __restrict__ packed,                                                 \
    WEIGHT_T*       __restrict__ output,                                                 \
    const WEIGHT_T* __restrict__ weights,                                                \
    int n_pre, int n_conn, int n_batch                                                   \
) {                                                                                     \
    extern __shared__ char _smem_bytes[];                                                \
    int32_t*  s_idx = reinterpret_cast<int32_t*>(_smem_bytes);                           \
    size_t wt_off = ((size_t)n_conn * sizeof(int32_t) + 7) & ~(size_t)7;                \
    WEIGHT_T* s_wt = reinterpret_cast<WEIGHT_T*>(_smem_bytes + wt_off);                 \
    size_t red_off = wt_off + (size_t)n_conn * sizeof(WEIGHT_T);                         \
    red_off = (red_off + 7) & ~(size_t)7;                                               \
    ACC_T* smem_red = reinterpret_cast<ACC_T*>(_smem_bytes + red_off);                   \
    int lane   = threadIdx.x & 31;                                                       \
    int warpid = threadIdx.x >> 5;                                                       \
    int nwarps = blockDim.x >> 5;                                                        \
    int bt     = (int)blockIdx.y;                                                        \
    int j      = bt * 32 + lane;                                                         \
    bool col_valid = (j < n_batch);                                                      \
    for (int row = blockIdx.x; row < n_pre; row += gridDim.x) {                          \
        const int32_t*  i_row = indices + (size_t)row * n_conn;                          \
        const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                          \
        for (int i = threadIdx.x; i < n_conn; i += blockDim.x) {                         \
            s_idx[i] = __ldg(&i_row[i]);                                                 \
            s_wt[i]  = __ldg(&w_row[i]);                                                 \
        }                                                                                \
        __syncthreads();                                                                 \
        ACC_T accum = ACC_ZERO;                                                          \
        for (int k = warpid; k < n_conn; k += nwarps) {                                  \
            int src      = s_idx[k];                                                     \
            int word_row = src >> 5;                                                     \
            int bit_pos  = src & 31;                                                     \
            uint32_t word = col_valid ?                                                  \
                __ldg(&packed[(size_t)word_row * n_batch + j]) : 0u;                     \
            accum += READ_W(s_wt[k]) * (ACC_T)((word >> bit_pos) & 1u);                  \
        }                                                                                \
        __syncthreads();                                                                 \
        smem_red[warpid * 32 + lane] = accum;                                            \
        __syncthreads();                                                                 \
        for (int stride = nwarps >> 1; stride > 0; stride >>= 1) {                      \
            if (warpid < stride)                                                         \
                smem_red[warpid * 32 + lane] += smem_red[(warpid + stride) * 32 + lane]; \
            __syncthreads();                                                             \
        }                                                                                \
        if (warpid == 0 && col_valid)                                                    \
            output[(size_t)row * n_batch + j] = WRITE_W(smem_red[lane]);                 \
        __syncthreads();                                                                 \
    }                                                                                    \
}

// ============================================================================
// SCATTER — pack_axis=1 (batch-packed)
//   packed: [n_pre, n_batch_words], bit b of packed[row, w] = matrix[row, w*32+b]
//   One word broadcast per source row.
// ============================================================================

// --- Scatter warp homo, pack_axis=1 ---
#define DEFINE_BSM_A1_WARP_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)      \
__global__ void _bsm_a1_warp_homo_kern##SUFFIX(                                     \
    const int32_t*  __restrict__ indices,                                            \
    const uint32_t* __restrict__ packed,                                             \
    WEIGHT_T*       __restrict__ output,                                             \
    const WEIGHT_T* __restrict__ weights,                                            \
    int n_pre, int n_conn, int n_batch, int n_batch_words                            \
) {                                                                                 \
    int warp_id = threadIdx.x >> 5;                                                 \
    int lane    = threadIdx.x & 31;                                                 \
    int rpb     = blockDim.x >> 5;                                                  \
    int bw      = (int)blockIdx.y;                                                  \
    int j       = bw * 32 + lane;                                                   \
    bool col_valid = (j < n_batch);                                                 \
    ACC_T w0 = READ_W(__ldg(&weights[0]));                                          \
    for (int base = blockIdx.x * rpb; base < n_pre; base += gridDim.x * rpb) {     \
        int row = base + warp_id;                                                   \
        if (row >= n_pre) continue;                                                 \
        uint32_t word = __ldg(&packed[(size_t)row * n_batch_words + bw]);            \
        bool active = col_valid && ((word >> lane) & 1u);                           \
        uint32_t active_mask = __ballot_sync(0xffffffff, active);                   \
        if (active_mask == 0) continue;                                             \
        const int32_t* i_row = indices + (size_t)row * n_conn;                      \
        int my_idx = (lane < n_conn) ? __ldg(&i_row[lane]) : 0;                    \
        for (int k = 0; k < n_conn; k++) {                                          \
            int target = __shfl_sync(0xffffffff, my_idx, k);                        \
            if (active)                                                             \
                ATOMIC_ADD_W(&output[(size_t)target * n_batch + j], w0);            \
        }                                                                           \
    }                                                                               \
}

// --- Scatter warp hetero, pack_axis=1 ---
#define DEFINE_BSM_A1_WARP_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)    \
__global__ void _bsm_a1_warp_hetero_kern##SUFFIX(                                   \
    const int32_t*  __restrict__ indices,                                            \
    const uint32_t* __restrict__ packed,                                             \
    WEIGHT_T*       __restrict__ output,                                             \
    const WEIGHT_T* __restrict__ weights,                                            \
    int n_pre, int n_conn, int n_batch, int n_batch_words                            \
) {                                                                                 \
    int warp_id = threadIdx.x >> 5;                                                 \
    int lane    = threadIdx.x & 31;                                                 \
    int rpb     = blockDim.x >> 5;                                                  \
    int bw      = (int)blockIdx.y;                                                  \
    int j       = bw * 32 + lane;                                                   \
    bool col_valid = (j < n_batch);                                                 \
    for (int base = blockIdx.x * rpb; base < n_pre; base += gridDim.x * rpb) {     \
        int row = base + warp_id;                                                   \
        if (row >= n_pre) continue;                                                 \
        uint32_t word = __ldg(&packed[(size_t)row * n_batch_words + bw]);            \
        bool active = col_valid && ((word >> lane) & 1u);                           \
        uint32_t active_mask = __ballot_sync(0xffffffff, active);                   \
        if (active_mask == 0) continue;                                             \
        const int32_t*  i_row = indices + (size_t)row * n_conn;                     \
        const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                     \
        int   my_idx = (lane < n_conn) ? __ldg(&i_row[lane]) : 0;                  \
        ACC_T my_w   = (lane < n_conn) ? READ_W(__ldg(&w_row[lane])) : (ACC_T)0;   \
        for (int k = 0; k < n_conn; k++) {                                          \
            int   target = __shfl_sync(0xffffffff, my_idx, k);                      \
            ACC_T wk     = __shfl_sync(0xffffffff, my_w, k);                        \
            if (active)                                                             \
                ATOMIC_ADD_W(&output[(size_t)target * n_batch + j], wk);            \
        }                                                                           \
    }                                                                               \
}

// --- Scatter basic homo, pack_axis=1 ---
#define DEFINE_BSM_A1_BASIC_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)     \
__global__ void _bsm_a1_basic_homo_kern##SUFFIX(                                    \
    const int32_t*  __restrict__ indices,                                            \
    const uint32_t* __restrict__ packed,                                             \
    WEIGHT_T*       __restrict__ output,                                             \
    const WEIGHT_T* __restrict__ weights,                                            \
    int n_pre, int n_conn, int n_batch, int n_batch_words                            \
) {                                                                                 \
    extern __shared__ char _smem_bytes[];                                            \
    int32_t* s_idx      = reinterpret_cast<int32_t*>(_smem_bytes);                   \
    int*     s_active_j = reinterpret_cast<int*>(s_idx + n_conn);                    \
    int*     s_n_active = s_active_j + BSM_BASIC_TILE_J;                             \
    int tile_base = (int)blockIdx.y * BSM_BASIC_TILE_J;                              \
    ACC_T w0 = READ_W(__ldg(&weights[0]));                                           \
    int bw_start = tile_base >> 5;                                                   \
    for (int row = blockIdx.x; row < n_pre; row += gridDim.x) {                      \
        if (threadIdx.x == 0) *s_n_active = 0;                                      \
        __syncthreads();                                                             \
        if (threadIdx.x < BSM_BASIC_TILE_J) {                                       \
            int j_local = tile_base + threadIdx.x;                                   \
            if (j_local < n_batch) {                                                 \
                int bw = j_local >> 5;                                               \
                int bit = j_local & 31;                                              \
                uint32_t word = __ldg(&packed[(size_t)row * n_batch_words + bw]);     \
                if ((word >> bit) & 1u) {                                            \
                    int pos = atomicAdd(s_n_active, 1);                              \
                    s_active_j[pos] = j_local;                                       \
                }                                                                    \
            }                                                                        \
        }                                                                            \
        __syncthreads();                                                             \
        int n_active = *s_n_active;                                                  \
        if (n_active == 0) continue;                                                 \
        const int32_t* i_row = indices + (size_t)row * n_conn;                       \
        for (int i = threadIdx.x; i < n_conn; i += blockDim.x)                       \
            s_idx[i] = __ldg(&i_row[i]);                                             \
        __syncthreads();                                                             \
        for (int a = 0; a < n_active; a++) {                                         \
            int j = s_active_j[a];                                                   \
            for (int k = threadIdx.x; k < n_conn; k += blockDim.x) {                 \
                int tgt = s_idx[k];                                                  \
                ATOMIC_ADD_W(&output[(size_t)tgt * n_batch + j], w0);                \
            }                                                                        \
        }                                                                            \
        __syncthreads();                                                             \
    }                                                                                \
}

// --- Scatter basic hetero, pack_axis=1 ---
#define DEFINE_BSM_A1_BASIC_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)   \
__global__ void _bsm_a1_basic_hetero_kern##SUFFIX(                                  \
    const int32_t*  __restrict__ indices,                                            \
    const uint32_t* __restrict__ packed,                                             \
    WEIGHT_T*       __restrict__ output,                                             \
    const WEIGHT_T* __restrict__ weights,                                            \
    int n_pre, int n_conn, int n_batch, int n_batch_words                            \
) {                                                                                 \
    extern __shared__ char _smem_bytes[];                                            \
    int32_t*  s_idx = reinterpret_cast<int32_t*>(_smem_bytes);                       \
    size_t wt_off = ((size_t)n_conn * sizeof(int32_t) + 7) & ~(size_t)7;            \
    WEIGHT_T* s_wt = reinterpret_cast<WEIGHT_T*>(_smem_bytes + wt_off);             \
    size_t active_off = wt_off + (size_t)n_conn * sizeof(WEIGHT_T);                  \
    active_off = (active_off + 3) & ~(size_t)3;                                     \
    int* s_active_j = reinterpret_cast<int*>(_smem_bytes + active_off);              \
    int* s_n_active = s_active_j + BSM_BASIC_TILE_J;                                 \
    int tile_base = (int)blockIdx.y * BSM_BASIC_TILE_J;                              \
    for (int row = blockIdx.x; row < n_pre; row += gridDim.x) {                      \
        if (threadIdx.x == 0) *s_n_active = 0;                                      \
        __syncthreads();                                                             \
        if (threadIdx.x < BSM_BASIC_TILE_J) {                                       \
            int j_local = tile_base + threadIdx.x;                                   \
            if (j_local < n_batch) {                                                 \
                int bw = j_local >> 5;                                               \
                int bit = j_local & 31;                                              \
                uint32_t word = __ldg(&packed[(size_t)row * n_batch_words + bw]);     \
                if ((word >> bit) & 1u) {                                            \
                    int pos = atomicAdd(s_n_active, 1);                              \
                    s_active_j[pos] = j_local;                                       \
                }                                                                    \
            }                                                                        \
        }                                                                            \
        __syncthreads();                                                             \
        int n_active = *s_n_active;                                                  \
        if (n_active == 0) continue;                                                 \
        const int32_t*  i_row = indices + (size_t)row * n_conn;                      \
        const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                      \
        for (int i = threadIdx.x; i < n_conn; i += blockDim.x) {                     \
            s_idx[i] = __ldg(&i_row[i]);                                             \
            s_wt[i]  = __ldg(&w_row[i]);                                             \
        }                                                                            \
        __syncthreads();                                                             \
        for (int a = 0; a < n_active; a++) {                                         \
            int j = s_active_j[a];                                                   \
            for (int k = threadIdx.x; k < n_conn; k += blockDim.x) {                 \
                int tgt = s_idx[k];                                                  \
                ACC_T wk = READ_W(s_wt[k]);                                          \
                ATOMIC_ADD_W(&output[(size_t)tgt * n_batch + j], wk);                \
            }                                                                        \
        }                                                                            \
        __syncthreads();                                                             \
    }                                                                                \
}

// ============================================================================
// SCATTER — pack_axis=0 (row-packed)
//   packed: [n_pre_words, n_batch], bit b of packed[w, j] = matrix[w*32+b, j]
//   Each lane reads a different word (coalesced).
// ============================================================================

// --- Scatter warp homo, pack_axis=0 ---
#define DEFINE_BSM_A0_WARP_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)      \
__global__ void _bsm_a0_warp_homo_kern##SUFFIX(                                     \
    const int32_t*  __restrict__ indices,                                            \
    const uint32_t* __restrict__ packed,                                             \
    WEIGHT_T*       __restrict__ output,                                             \
    const WEIGHT_T* __restrict__ weights,                                            \
    int n_pre, int n_conn, int n_batch                                               \
) {                                                                                 \
    int warp_id = threadIdx.x >> 5;                                                 \
    int lane    = threadIdx.x & 31;                                                 \
    int rpb     = blockDim.x >> 5;                                                  \
    int bt      = (int)blockIdx.y;                                                  \
    int j       = bt * 32 + lane;                                                   \
    bool col_valid = (j < n_batch);                                                 \
    ACC_T w0 = READ_W(__ldg(&weights[0]));                                          \
    for (int base = blockIdx.x * rpb; base < n_pre; base += gridDim.x * rpb) {     \
        int row = base + warp_id;                                                   \
        if (row >= n_pre) continue;                                                 \
        int word_row = row >> 5;                                                    \
        int bit_pos  = row & 31;                                                    \
        uint32_t word = col_valid ?                                                 \
            __ldg(&packed[(size_t)word_row * n_batch + j]) : 0u;                    \
        bool active = (word >> bit_pos) & 1u;                                       \
        uint32_t active_mask = __ballot_sync(0xffffffff, active);                   \
        if (active_mask == 0) continue;                                             \
        const int32_t* i_row = indices + (size_t)row * n_conn;                      \
        int my_idx = (lane < n_conn) ? __ldg(&i_row[lane]) : 0;                    \
        for (int k = 0; k < n_conn; k++) {                                          \
            int target = __shfl_sync(0xffffffff, my_idx, k);                        \
            if (active)                                                             \
                ATOMIC_ADD_W(&output[(size_t)target * n_batch + j], w0);            \
        }                                                                           \
    }                                                                               \
}

// --- Scatter warp hetero, pack_axis=0 ---
#define DEFINE_BSM_A0_WARP_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)    \
__global__ void _bsm_a0_warp_hetero_kern##SUFFIX(                                   \
    const int32_t*  __restrict__ indices,                                            \
    const uint32_t* __restrict__ packed,                                             \
    WEIGHT_T*       __restrict__ output,                                             \
    const WEIGHT_T* __restrict__ weights,                                            \
    int n_pre, int n_conn, int n_batch                                               \
) {                                                                                 \
    int warp_id = threadIdx.x >> 5;                                                 \
    int lane    = threadIdx.x & 31;                                                 \
    int rpb     = blockDim.x >> 5;                                                  \
    int bt      = (int)blockIdx.y;                                                  \
    int j       = bt * 32 + lane;                                                   \
    bool col_valid = (j < n_batch);                                                 \
    for (int base = blockIdx.x * rpb; base < n_pre; base += gridDim.x * rpb) {     \
        int row = base + warp_id;                                                   \
        if (row >= n_pre) continue;                                                 \
        int word_row = row >> 5;                                                    \
        int bit_pos  = row & 31;                                                    \
        uint32_t word = col_valid ?                                                 \
            __ldg(&packed[(size_t)word_row * n_batch + j]) : 0u;                    \
        bool active = (word >> bit_pos) & 1u;                                       \
        uint32_t active_mask = __ballot_sync(0xffffffff, active);                   \
        if (active_mask == 0) continue;                                             \
        const int32_t*  i_row = indices + (size_t)row * n_conn;                     \
        const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                     \
        int   my_idx = (lane < n_conn) ? __ldg(&i_row[lane]) : 0;                  \
        ACC_T my_w   = (lane < n_conn) ? READ_W(__ldg(&w_row[lane])) : (ACC_T)0;   \
        for (int k = 0; k < n_conn; k++) {                                          \
            int   target = __shfl_sync(0xffffffff, my_idx, k);                      \
            ACC_T wk     = __shfl_sync(0xffffffff, my_w, k);                        \
            if (active)                                                             \
                ATOMIC_ADD_W(&output[(size_t)target * n_batch + j], wk);            \
        }                                                                           \
    }                                                                               \
}

// --- Scatter basic homo, pack_axis=0 ---
#define DEFINE_BSM_A0_BASIC_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)     \
__global__ void _bsm_a0_basic_homo_kern##SUFFIX(                                    \
    const int32_t*  __restrict__ indices,                                            \
    const uint32_t* __restrict__ packed,                                             \
    WEIGHT_T*       __restrict__ output,                                             \
    const WEIGHT_T* __restrict__ weights,                                            \
    int n_pre, int n_conn, int n_batch                                               \
) {                                                                                 \
    extern __shared__ char _smem_bytes[];                                            \
    int32_t* s_idx      = reinterpret_cast<int32_t*>(_smem_bytes);                   \
    int*     s_active_j = reinterpret_cast<int*>(s_idx + n_conn);                    \
    int*     s_n_active = s_active_j + BSM_BASIC_TILE_J;                             \
    int tile_base = (int)blockIdx.y * BSM_BASIC_TILE_J;                              \
    ACC_T w0 = READ_W(__ldg(&weights[0]));                                           \
    for (int row = blockIdx.x; row < n_pre; row += gridDim.x) {                      \
        if (threadIdx.x == 0) *s_n_active = 0;                                      \
        __syncthreads();                                                             \
        int word_row = row >> 5;                                                     \
        int bit_pos  = row & 31;                                                     \
        if (threadIdx.x < BSM_BASIC_TILE_J) {                                       \
            int j_local = tile_base + threadIdx.x;                                   \
            if (j_local < n_batch) {                                                 \
                uint32_t word = __ldg(&packed[(size_t)word_row * n_batch + j_local]); \
                if ((word >> bit_pos) & 1u) {                                        \
                    int pos = atomicAdd(s_n_active, 1);                              \
                    s_active_j[pos] = j_local;                                       \
                }                                                                    \
            }                                                                        \
        }                                                                            \
        __syncthreads();                                                             \
        int n_active = *s_n_active;                                                  \
        if (n_active == 0) continue;                                                 \
        const int32_t* i_row = indices + (size_t)row * n_conn;                       \
        for (int i = threadIdx.x; i < n_conn; i += blockDim.x)                       \
            s_idx[i] = __ldg(&i_row[i]);                                             \
        __syncthreads();                                                             \
        for (int a = 0; a < n_active; a++) {                                         \
            int j = s_active_j[a];                                                   \
            for (int k = threadIdx.x; k < n_conn; k += blockDim.x) {                 \
                int tgt = s_idx[k];                                                  \
                ATOMIC_ADD_W(&output[(size_t)tgt * n_batch + j], w0);                \
            }                                                                        \
        }                                                                            \
        __syncthreads();                                                             \
    }                                                                                \
}

// --- Scatter basic hetero, pack_axis=0 ---
#define DEFINE_BSM_A0_BASIC_HETERO(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)   \
__global__ void _bsm_a0_basic_hetero_kern##SUFFIX(                                  \
    const int32_t*  __restrict__ indices,                                            \
    const uint32_t* __restrict__ packed,                                             \
    WEIGHT_T*       __restrict__ output,                                             \
    const WEIGHT_T* __restrict__ weights,                                            \
    int n_pre, int n_conn, int n_batch                                               \
) {                                                                                 \
    extern __shared__ char _smem_bytes[];                                            \
    int32_t*  s_idx = reinterpret_cast<int32_t*>(_smem_bytes);                       \
    size_t wt_off = ((size_t)n_conn * sizeof(int32_t) + 7) & ~(size_t)7;            \
    WEIGHT_T* s_wt = reinterpret_cast<WEIGHT_T*>(_smem_bytes + wt_off);             \
    size_t active_off = wt_off + (size_t)n_conn * sizeof(WEIGHT_T);                  \
    active_off = (active_off + 3) & ~(size_t)3;                                     \
    int* s_active_j = reinterpret_cast<int*>(_smem_bytes + active_off);              \
    int* s_n_active = s_active_j + BSM_BASIC_TILE_J;                                 \
    int tile_base = (int)blockIdx.y * BSM_BASIC_TILE_J;                              \
    for (int row = blockIdx.x; row < n_pre; row += gridDim.x) {                      \
        if (threadIdx.x == 0) *s_n_active = 0;                                      \
        __syncthreads();                                                             \
        int word_row = row >> 5;                                                     \
        int bit_pos  = row & 31;                                                     \
        if (threadIdx.x < BSM_BASIC_TILE_J) {                                       \
            int j_local = tile_base + threadIdx.x;                                   \
            if (j_local < n_batch) {                                                 \
                uint32_t word = __ldg(&packed[(size_t)word_row * n_batch + j_local]); \
                if ((word >> bit_pos) & 1u) {                                        \
                    int pos = atomicAdd(s_n_active, 1);                              \
                    s_active_j[pos] = j_local;                                       \
                }                                                                    \
            }                                                                        \
        }                                                                            \
        __syncthreads();                                                             \
        int n_active = *s_n_active;                                                  \
        if (n_active == 0) continue;                                                 \
        const int32_t*  i_row = indices + (size_t)row * n_conn;                      \
        const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                      \
        for (int i = threadIdx.x; i < n_conn; i += blockDim.x) {                     \
            s_idx[i] = __ldg(&i_row[i]);                                             \
            s_wt[i]  = __ldg(&w_row[i]);                                             \
        }                                                                            \
        __syncthreads();                                                             \
        for (int a = 0; a < n_active; a++) {                                         \
            int j = s_active_j[a];                                                   \
            for (int k = threadIdx.x; k < n_conn; k += blockDim.x) {                 \
                int tgt = s_idx[k];                                                  \
                ACC_T wk = READ_W(s_wt[k]);                                          \
                ATOMIC_ADD_W(&output[(size_t)tgt * n_batch + j], wk);                \
            }                                                                        \
        }                                                                            \
        __syncthreads();                                                             \
    }                                                                                \
}


// ============================================================================
// Kernel Instantiations
// ============================================================================

// ---- float32 ----
DEFINE_BGM_A1_WARP_HOMO    (_f32, float, float, READ_F32, WRITE_F32)
DEFINE_BGM_A1_WARP_HETERO  (_f32, float, float, READ_F32, WRITE_F32)
DEFINE_BGM_A1_BASIC_HOMO   (_f32, float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_BGM_A1_BASIC_HETERO (_f32, float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_BGM_A0_WARP_HOMO    (_f32, float, float, READ_F32, WRITE_F32)
DEFINE_BGM_A0_WARP_HETERO  (_f32, float, float, READ_F32, WRITE_F32)
DEFINE_BGM_A0_BASIC_HOMO   (_f32, float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_BGM_A0_BASIC_HETERO (_f32, float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_BSM_A1_WARP_HOMO    (_f32, float, float, READ_F32, atomicAdd)
DEFINE_BSM_A1_WARP_HETERO  (_f32, float, float, READ_F32, atomicAdd)
DEFINE_BSM_A1_BASIC_HOMO   (_f32, float, float, READ_F32, atomicAdd)
DEFINE_BSM_A1_BASIC_HETERO (_f32, float, float, READ_F32, atomicAdd)
DEFINE_BSM_A0_WARP_HOMO    (_f32, float, float, READ_F32, atomicAdd)
DEFINE_BSM_A0_WARP_HETERO  (_f32, float, float, READ_F32, atomicAdd)
DEFINE_BSM_A0_BASIC_HOMO   (_f32, float, float, READ_F32, atomicAdd)
DEFINE_BSM_A0_BASIC_HETERO (_f32, float, float, READ_F32, atomicAdd)

// ---- float64 ----
DEFINE_BGM_A1_WARP_HOMO    (_f64, double, double, READ_F64, WRITE_F64)
DEFINE_BGM_A1_WARP_HETERO  (_f64, double, double, READ_F64, WRITE_F64)
DEFINE_BGM_A1_BASIC_HOMO   (_f64, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_BGM_A1_BASIC_HETERO (_f64, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_BGM_A0_WARP_HOMO    (_f64, double, double, READ_F64, WRITE_F64)
DEFINE_BGM_A0_WARP_HETERO  (_f64, double, double, READ_F64, WRITE_F64)
DEFINE_BGM_A0_BASIC_HOMO   (_f64, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_BGM_A0_BASIC_HETERO (_f64, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_BSM_A1_WARP_HOMO    (_f64, double, double, READ_F64, atomic_add_f64)
DEFINE_BSM_A1_WARP_HETERO  (_f64, double, double, READ_F64, atomic_add_f64)
DEFINE_BSM_A1_BASIC_HOMO   (_f64, double, double, READ_F64, atomic_add_f64)
DEFINE_BSM_A1_BASIC_HETERO (_f64, double, double, READ_F64, atomic_add_f64)
DEFINE_BSM_A0_WARP_HOMO    (_f64, double, double, READ_F64, atomic_add_f64)
DEFINE_BSM_A0_WARP_HETERO  (_f64, double, double, READ_F64, atomic_add_f64)
DEFINE_BSM_A0_BASIC_HOMO   (_f64, double, double, READ_F64, atomic_add_f64)
DEFINE_BSM_A0_BASIC_HETERO (_f64, double, double, READ_F64, atomic_add_f64)

// ---- float16 ----
DEFINE_BGM_A1_WARP_HOMO    (_f16, __half, float, READ_F16, WRITE_F16)
DEFINE_BGM_A1_WARP_HETERO  (_f16, __half, float, READ_F16, WRITE_F16)
DEFINE_BGM_A1_BASIC_HOMO   (_f16, __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_BGM_A1_BASIC_HETERO (_f16, __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_BGM_A0_WARP_HOMO    (_f16, __half, float, READ_F16, WRITE_F16)
DEFINE_BGM_A0_WARP_HETERO  (_f16, __half, float, READ_F16, WRITE_F16)
DEFINE_BGM_A0_BASIC_HOMO   (_f16, __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_BGM_A0_BASIC_HETERO (_f16, __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_BSM_A1_WARP_HOMO    (_f16, __half, float, READ_F16, atomic_add_f16)
DEFINE_BSM_A1_WARP_HETERO  (_f16, __half, float, READ_F16, atomic_add_f16)
DEFINE_BSM_A1_BASIC_HOMO   (_f16, __half, float, READ_F16, atomic_add_f16)
DEFINE_BSM_A1_BASIC_HETERO (_f16, __half, float, READ_F16, atomic_add_f16)
DEFINE_BSM_A0_WARP_HOMO    (_f16, __half, float, READ_F16, atomic_add_f16)
DEFINE_BSM_A0_WARP_HETERO  (_f16, __half, float, READ_F16, atomic_add_f16)
DEFINE_BSM_A0_BASIC_HOMO   (_f16, __half, float, READ_F16, atomic_add_f16)
DEFINE_BSM_A0_BASIC_HETERO (_f16, __half, float, READ_F16, atomic_add_f16)

// ---- bfloat16 ----
DEFINE_BGM_A1_WARP_HOMO    (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16)
DEFINE_BGM_A1_WARP_HETERO  (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16)
DEFINE_BGM_A1_BASIC_HOMO   (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_BGM_A1_BASIC_HETERO (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_BGM_A0_WARP_HOMO    (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16)
DEFINE_BGM_A0_WARP_HETERO  (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16)
DEFINE_BGM_A0_BASIC_HOMO   (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_BGM_A0_BASIC_HETERO (_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_BSM_A1_WARP_HOMO    (_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BSM_A1_WARP_HETERO  (_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BSM_A1_BASIC_HOMO   (_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BSM_A1_BASIC_HETERO (_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BSM_A0_WARP_HOMO    (_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BSM_A0_WARP_HETERO  (_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BSM_A0_BASIC_HOMO   (_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BSM_A0_BASIC_HETERO (_bf16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)


// ============================================================================
// FFI Entry Points — Gather, pack_axis=1
// ============================================================================

#define FFI_BGM_A1_HOMO(SUFFIX, WEIGHT_C_T, ACC_SIZE)                                \
void bitpack_binary_fcnmm_gather_homo_a1##SUFFIX(                                    \
    const BE::Tensor weights, const BE::Tensor indices,                               \
    const BE::Tensor packed, BE::Tensor output, int64_t stream                        \
) {                                                                                   \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                          \
    int n_pre  = static_cast<int>(indices.size(0));                                   \
    int n_conn = static_cast<int>(indices.size(1));                                   \
    int n_batch = static_cast<int>(output.size(1));                                   \
    int nbw    = static_cast<int>(packed.size(1));                                    \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());     \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());         \
    const uint32_t*   d_pk  = static_cast<const uint32_t*>(packed.data_ptr());         \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());             \
    int batch_tiles = (n_batch + 31) / 32;                                            \
    if (n_conn <= 32) {                                                               \
        int rpb = WARP_ROWS_PER_BLOCK;                                                \
        int raw_gx = (n_pre + rpb - 1) / rpb;                                        \
        int grid_x = (raw_gx < BGM_MAX_GRID_X) ? raw_gx : BGM_MAX_GRID_X;           \
        dim3 grid(grid_x, batch_tiles);                                               \
        _bgm_a1_warp_homo_kern##SUFFIX<<<grid, rpb * 32, 0, s>>>(                    \
            d_idx, d_pk, d_out, d_w, n_pre, n_conn, n_batch, nbw);                   \
    } else {                                                                          \
        int bsz = 256; int nwarps = bsz >> 5;                                        \
        size_t idx_bytes = (size_t)n_conn * sizeof(int32_t);                          \
        size_t red_off = (idx_bytes + 7) & ~(size_t)7;                                \
        size_t shm = red_off + (size_t)nwarps * 32 * ACC_SIZE;                        \
        int grid_x = (n_pre < BGM_MAX_GRID_X) ? n_pre : BGM_MAX_GRID_X;              \
        dim3 grid(grid_x, batch_tiles);                                               \
        _bgm_a1_basic_homo_kern##SUFFIX<<<grid, bsz, shm, s>>>(                      \
            d_idx, d_pk, d_out, d_w, n_pre, n_conn, n_batch, nbw);                   \
    }                                                                                 \
}

#define FFI_BGM_A1_HETERO(SUFFIX, WEIGHT_C_T, ACC_SIZE)                               \
void bitpack_binary_fcnmm_gather_hetero_a1##SUFFIX(                                   \
    const BE::Tensor weights, const BE::Tensor indices,                               \
    const BE::Tensor packed, BE::Tensor output, int64_t stream                        \
) {                                                                                   \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                          \
    int n_pre  = static_cast<int>(indices.size(0));                                   \
    int n_conn = static_cast<int>(indices.size(1));                                   \
    int n_batch = static_cast<int>(output.size(1));                                   \
    int nbw    = static_cast<int>(packed.size(1));                                    \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());     \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());         \
    const uint32_t*   d_pk  = static_cast<const uint32_t*>(packed.data_ptr());         \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());             \
    int batch_tiles = (n_batch + 31) / 32;                                            \
    if (n_conn <= 32) {                                                               \
        int rpb = WARP_ROWS_PER_BLOCK;                                                \
        int raw_gx = (n_pre + rpb - 1) / rpb;                                        \
        int grid_x = (raw_gx < BGM_MAX_GRID_X) ? raw_gx : BGM_MAX_GRID_X;           \
        dim3 grid(grid_x, batch_tiles);                                               \
        _bgm_a1_warp_hetero_kern##SUFFIX<<<grid, rpb * 32, 0, s>>>(                  \
            d_idx, d_pk, d_out, d_w, n_pre, n_conn, n_batch, nbw);                   \
    } else {                                                                          \
        int bsz = 256; int nwarps = bsz >> 5;                                        \
        size_t wt_off = ((size_t)n_conn * sizeof(int32_t) + 7) & ~(size_t)7;         \
        size_t idx_wt = wt_off + (size_t)n_conn * sizeof(WEIGHT_C_T);                \
        size_t red_off = (idx_wt + 7) & ~(size_t)7;                                  \
        size_t shm = red_off + (size_t)nwarps * 32 * ACC_SIZE;                        \
        int grid_x = (n_pre < BGM_MAX_GRID_X) ? n_pre : BGM_MAX_GRID_X;              \
        dim3 grid(grid_x, batch_tiles);                                               \
        _bgm_a1_basic_hetero_kern##SUFFIX<<<grid, bsz, shm, s>>>(                    \
            d_idx, d_pk, d_out, d_w, n_pre, n_conn, n_batch, nbw);                   \
    }                                                                                 \
}

// ============================================================================
// FFI Entry Points — Gather, pack_axis=0
// ============================================================================

#define FFI_BGM_A0_HOMO(SUFFIX, WEIGHT_C_T, ACC_SIZE)                                \
void bitpack_binary_fcnmm_gather_homo_a0##SUFFIX(                                    \
    const BE::Tensor weights, const BE::Tensor indices,                               \
    const BE::Tensor packed, BE::Tensor output, int64_t stream                        \
) {                                                                                   \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                          \
    int n_pre  = static_cast<int>(indices.size(0));                                   \
    int n_conn = static_cast<int>(indices.size(1));                                   \
    int n_batch = static_cast<int>(output.size(1));                                   \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());     \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());         \
    const uint32_t*   d_pk  = static_cast<const uint32_t*>(packed.data_ptr());         \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());             \
    int batch_tiles = (n_batch + 31) / 32;                                            \
    if (n_conn <= 32) {                                                               \
        int rpb = WARP_ROWS_PER_BLOCK;                                                \
        int raw_gx = (n_pre + rpb - 1) / rpb;                                        \
        int grid_x = (raw_gx < BGM_MAX_GRID_X) ? raw_gx : BGM_MAX_GRID_X;           \
        dim3 grid(grid_x, batch_tiles);                                               \
        _bgm_a0_warp_homo_kern##SUFFIX<<<grid, rpb * 32, 0, s>>>(                    \
            d_idx, d_pk, d_out, d_w, n_pre, n_conn, n_batch);                        \
    } else {                                                                          \
        int bsz = 256; int nwarps = bsz >> 5;                                        \
        size_t idx_bytes = (size_t)n_conn * sizeof(int32_t);                          \
        size_t red_off = (idx_bytes + 7) & ~(size_t)7;                                \
        size_t shm = red_off + (size_t)nwarps * 32 * ACC_SIZE;                        \
        int grid_x = (n_pre < BGM_MAX_GRID_X) ? n_pre : BGM_MAX_GRID_X;              \
        dim3 grid(grid_x, batch_tiles);                                               \
        _bgm_a0_basic_homo_kern##SUFFIX<<<grid, bsz, shm, s>>>(                      \
            d_idx, d_pk, d_out, d_w, n_pre, n_conn, n_batch);                        \
    }                                                                                 \
}

#define FFI_BGM_A0_HETERO(SUFFIX, WEIGHT_C_T, ACC_SIZE)                               \
void bitpack_binary_fcnmm_gather_hetero_a0##SUFFIX(                                   \
    const BE::Tensor weights, const BE::Tensor indices,                               \
    const BE::Tensor packed, BE::Tensor output, int64_t stream                        \
) {                                                                                   \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                          \
    int n_pre  = static_cast<int>(indices.size(0));                                   \
    int n_conn = static_cast<int>(indices.size(1));                                   \
    int n_batch = static_cast<int>(output.size(1));                                   \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());     \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());         \
    const uint32_t*   d_pk  = static_cast<const uint32_t*>(packed.data_ptr());         \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());             \
    int batch_tiles = (n_batch + 31) / 32;                                            \
    if (n_conn <= 32) {                                                               \
        int rpb = WARP_ROWS_PER_BLOCK;                                                \
        int raw_gx = (n_pre + rpb - 1) / rpb;                                        \
        int grid_x = (raw_gx < BGM_MAX_GRID_X) ? raw_gx : BGM_MAX_GRID_X;           \
        dim3 grid(grid_x, batch_tiles);                                               \
        _bgm_a0_warp_hetero_kern##SUFFIX<<<grid, rpb * 32, 0, s>>>(                  \
            d_idx, d_pk, d_out, d_w, n_pre, n_conn, n_batch);                        \
    } else {                                                                          \
        int bsz = 256; int nwarps = bsz >> 5;                                        \
        size_t wt_off = ((size_t)n_conn * sizeof(int32_t) + 7) & ~(size_t)7;         \
        size_t idx_wt = wt_off + (size_t)n_conn * sizeof(WEIGHT_C_T);                \
        size_t red_off = (idx_wt + 7) & ~(size_t)7;                                  \
        size_t shm = red_off + (size_t)nwarps * 32 * ACC_SIZE;                        \
        int grid_x = (n_pre < BGM_MAX_GRID_X) ? n_pre : BGM_MAX_GRID_X;              \
        dim3 grid(grid_x, batch_tiles);                                               \
        _bgm_a0_basic_hetero_kern##SUFFIX<<<grid, bsz, shm, s>>>(                    \
            d_idx, d_pk, d_out, d_w, n_pre, n_conn, n_batch);                        \
    }                                                                                 \
}

// ============================================================================
// FFI Entry Points — Scatter, pack_axis=1
// ============================================================================

#define FFI_BSM_A1_HOMO(SUFFIX, WEIGHT_C_T)                                           \
void bitpack_binary_fcnmm_scatter_homo_a1##SUFFIX(                                    \
    const BE::Tensor weights, const BE::Tensor indices,                               \
    const BE::Tensor packed, BE::Tensor output, int64_t stream                        \
) {                                                                                   \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                          \
    int n_pre  = static_cast<int>(indices.size(0));                                   \
    int n_conn = static_cast<int>(indices.size(1));                                   \
    int n_post = static_cast<int>(output.size(0));                                    \
    int n_batch = static_cast<int>(output.size(1));                                   \
    int nbw    = static_cast<int>(packed.size(1));                                    \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());     \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());         \
    const uint32_t*   d_pk  = static_cast<const uint32_t*>(packed.data_ptr());         \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());             \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_batch * sizeof(WEIGHT_C_T), s);      \
    int batch_tiles = (n_batch + 31) / 32;                                            \
    if (n_conn <= 32) {                                                               \
        int rpb = WARP_ROWS_PER_BLOCK;                                                \
        int raw_gx = (n_pre + rpb - 1) / rpb;                                        \
        int grid_x = (raw_gx < BSM_MAX_GRID_X) ? raw_gx : BSM_MAX_GRID_X;           \
        dim3 grid(grid_x, batch_tiles);                                               \
        _bsm_a1_warp_homo_kern##SUFFIX<<<grid, rpb * 32, 0, s>>>(                    \
            d_idx, d_pk, d_out, d_w, n_pre, n_conn, n_batch, nbw);                   \
    } else {                                                                          \
        int bsz = 256;                                                                \
        size_t idx_bytes = (size_t)n_conn * sizeof(int32_t);                          \
        size_t active_bytes = (BSM_BASIC_TILE_J + 1) * sizeof(int);                   \
        size_t shm = idx_bytes + active_bytes;                                        \
        int bt = (n_batch + BSM_BASIC_TILE_J - 1) / BSM_BASIC_TILE_J;                \
        int grid_x = (n_pre < BSM_MAX_GRID_X) ? n_pre : BSM_MAX_GRID_X;              \
        dim3 grid(grid_x, bt);                                                        \
        _bsm_a1_basic_homo_kern##SUFFIX<<<grid, bsz, shm, s>>>(                      \
            d_idx, d_pk, d_out, d_w, n_pre, n_conn, n_batch, nbw);                   \
    }                                                                                 \
}

#define FFI_BSM_A1_HETERO(SUFFIX, WEIGHT_C_T)                                         \
void bitpack_binary_fcnmm_scatter_hetero_a1##SUFFIX(                                  \
    const BE::Tensor weights, const BE::Tensor indices,                               \
    const BE::Tensor packed, BE::Tensor output, int64_t stream                        \
) {                                                                                   \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                          \
    int n_pre  = static_cast<int>(indices.size(0));                                   \
    int n_conn = static_cast<int>(indices.size(1));                                   \
    int n_post = static_cast<int>(output.size(0));                                    \
    int n_batch = static_cast<int>(output.size(1));                                   \
    int nbw    = static_cast<int>(packed.size(1));                                    \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());     \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());         \
    const uint32_t*   d_pk  = static_cast<const uint32_t*>(packed.data_ptr());         \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());             \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_batch * sizeof(WEIGHT_C_T), s);      \
    int batch_tiles = (n_batch + 31) / 32;                                            \
    if (n_conn <= 32) {                                                               \
        int rpb = WARP_ROWS_PER_BLOCK;                                                \
        int raw_gx = (n_pre + rpb - 1) / rpb;                                        \
        int grid_x = (raw_gx < BSM_MAX_GRID_X) ? raw_gx : BSM_MAX_GRID_X;           \
        dim3 grid(grid_x, batch_tiles);                                               \
        _bsm_a1_warp_hetero_kern##SUFFIX<<<grid, rpb * 32, 0, s>>>(                  \
            d_idx, d_pk, d_out, d_w, n_pre, n_conn, n_batch, nbw);                   \
    } else {                                                                          \
        int bsz = 256;                                                                \
        size_t wt_off = ((size_t)n_conn * sizeof(int32_t) + 7) & ~(size_t)7;         \
        size_t idx_wt = wt_off + (size_t)n_conn * sizeof(WEIGHT_C_T);                \
        size_t active_off = (idx_wt + 3) & ~(size_t)3;                               \
        size_t shm = active_off + (BSM_BASIC_TILE_J + 1) * sizeof(int);              \
        int bt = (n_batch + BSM_BASIC_TILE_J - 1) / BSM_BASIC_TILE_J;                \
        int grid_x = (n_pre < BSM_MAX_GRID_X) ? n_pre : BSM_MAX_GRID_X;              \
        dim3 grid(grid_x, bt);                                                        \
        _bsm_a1_basic_hetero_kern##SUFFIX<<<grid, bsz, shm, s>>>(                    \
            d_idx, d_pk, d_out, d_w, n_pre, n_conn, n_batch, nbw);                   \
    }                                                                                 \
}

// ============================================================================
// FFI Entry Points — Scatter, pack_axis=0
// ============================================================================

#define FFI_BSM_A0_HOMO(SUFFIX, WEIGHT_C_T)                                           \
void bitpack_binary_fcnmm_scatter_homo_a0##SUFFIX(                                    \
    const BE::Tensor weights, const BE::Tensor indices,                               \
    const BE::Tensor packed, BE::Tensor output, int64_t stream                        \
) {                                                                                   \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                          \
    int n_pre  = static_cast<int>(indices.size(0));                                   \
    int n_conn = static_cast<int>(indices.size(1));                                   \
    int n_post = static_cast<int>(output.size(0));                                    \
    int n_batch = static_cast<int>(output.size(1));                                   \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());     \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());         \
    const uint32_t*   d_pk  = static_cast<const uint32_t*>(packed.data_ptr());         \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());             \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_batch * sizeof(WEIGHT_C_T), s);      \
    int batch_tiles = (n_batch + 31) / 32;                                            \
    if (n_conn <= 32) {                                                               \
        int rpb = WARP_ROWS_PER_BLOCK;                                                \
        int raw_gx = (n_pre + rpb - 1) / rpb;                                        \
        int grid_x = (raw_gx < BSM_MAX_GRID_X) ? raw_gx : BSM_MAX_GRID_X;           \
        dim3 grid(grid_x, batch_tiles);                                               \
        _bsm_a0_warp_homo_kern##SUFFIX<<<grid, rpb * 32, 0, s>>>(                    \
            d_idx, d_pk, d_out, d_w, n_pre, n_conn, n_batch);                        \
    } else {                                                                          \
        int bsz = 256;                                                                \
        size_t idx_bytes = (size_t)n_conn * sizeof(int32_t);                          \
        size_t active_bytes = (BSM_BASIC_TILE_J + 1) * sizeof(int);                   \
        size_t shm = idx_bytes + active_bytes;                                        \
        int bt = (n_batch + BSM_BASIC_TILE_J - 1) / BSM_BASIC_TILE_J;                \
        int grid_x = (n_pre < BSM_MAX_GRID_X) ? n_pre : BSM_MAX_GRID_X;              \
        dim3 grid(grid_x, bt);                                                        \
        _bsm_a0_basic_homo_kern##SUFFIX<<<grid, bsz, shm, s>>>(                      \
            d_idx, d_pk, d_out, d_w, n_pre, n_conn, n_batch);                        \
    }                                                                                 \
}

#define FFI_BSM_A0_HETERO(SUFFIX, WEIGHT_C_T)                                         \
void bitpack_binary_fcnmm_scatter_hetero_a0##SUFFIX(                                  \
    const BE::Tensor weights, const BE::Tensor indices,                               \
    const BE::Tensor packed, BE::Tensor output, int64_t stream                        \
) {                                                                                   \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                          \
    int n_pre  = static_cast<int>(indices.size(0));                                   \
    int n_conn = static_cast<int>(indices.size(1));                                   \
    int n_post = static_cast<int>(output.size(0));                                    \
    int n_batch = static_cast<int>(output.size(1));                                   \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());     \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());         \
    const uint32_t*   d_pk  = static_cast<const uint32_t*>(packed.data_ptr());         \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());             \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_batch * sizeof(WEIGHT_C_T), s);      \
    int batch_tiles = (n_batch + 31) / 32;                                            \
    if (n_conn <= 32) {                                                               \
        int rpb = WARP_ROWS_PER_BLOCK;                                                \
        int raw_gx = (n_pre + rpb - 1) / rpb;                                        \
        int grid_x = (raw_gx < BSM_MAX_GRID_X) ? raw_gx : BSM_MAX_GRID_X;           \
        dim3 grid(grid_x, batch_tiles);                                               \
        _bsm_a0_warp_hetero_kern##SUFFIX<<<grid, rpb * 32, 0, s>>>(                  \
            d_idx, d_pk, d_out, d_w, n_pre, n_conn, n_batch);                        \
    } else {                                                                          \
        int bsz = 256;                                                                \
        size_t wt_off = ((size_t)n_conn * sizeof(int32_t) + 7) & ~(size_t)7;         \
        size_t idx_wt = wt_off + (size_t)n_conn * sizeof(WEIGHT_C_T);                \
        size_t active_off = (idx_wt + 3) & ~(size_t)3;                               \
        size_t shm = active_off + (BSM_BASIC_TILE_J + 1) * sizeof(int);              \
        int bt = (n_batch + BSM_BASIC_TILE_J - 1) / BSM_BASIC_TILE_J;                \
        int grid_x = (n_pre < BSM_MAX_GRID_X) ? n_pre : BSM_MAX_GRID_X;              \
        dim3 grid(grid_x, bt);                                                        \
        _bsm_a0_basic_hetero_kern##SUFFIX<<<grid, bsz, shm, s>>>(                    \
            d_idx, d_pk, d_out, d_w, n_pre, n_conn, n_batch);                        \
    }                                                                                 \
}

// ============================================================================
// FFI Instantiations
// ============================================================================

// ---- float32 ----
// @BE bitpack_binary_fcnmm_gather_homo_a1_f32
FFI_BGM_A1_HOMO   (_f32, float, sizeof(float))
// @BE bitpack_binary_fcnmm_gather_hetero_a1_f32
FFI_BGM_A1_HETERO (_f32, float, sizeof(float))
// @BE bitpack_binary_fcnmm_gather_homo_a0_f32
FFI_BGM_A0_HOMO   (_f32, float, sizeof(float))
// @BE bitpack_binary_fcnmm_gather_hetero_a0_f32
FFI_BGM_A0_HETERO (_f32, float, sizeof(float))
// @BE bitpack_binary_fcnmm_scatter_homo_a1_f32
FFI_BSM_A1_HOMO   (_f32, float)
// @BE bitpack_binary_fcnmm_scatter_hetero_a1_f32
FFI_BSM_A1_HETERO (_f32, float)
// @BE bitpack_binary_fcnmm_scatter_homo_a0_f32
FFI_BSM_A0_HOMO   (_f32, float)
// @BE bitpack_binary_fcnmm_scatter_hetero_a0_f32
FFI_BSM_A0_HETERO (_f32, float)

// ---- float64 ----
// @BE bitpack_binary_fcnmm_gather_homo_a1_f64
FFI_BGM_A1_HOMO   (_f64, double, sizeof(double))
// @BE bitpack_binary_fcnmm_gather_hetero_a1_f64
FFI_BGM_A1_HETERO (_f64, double, sizeof(double))
// @BE bitpack_binary_fcnmm_gather_homo_a0_f64
FFI_BGM_A0_HOMO   (_f64, double, sizeof(double))
// @BE bitpack_binary_fcnmm_gather_hetero_a0_f64
FFI_BGM_A0_HETERO (_f64, double, sizeof(double))
// @BE bitpack_binary_fcnmm_scatter_homo_a1_f64
FFI_BSM_A1_HOMO   (_f64, double)
// @BE bitpack_binary_fcnmm_scatter_hetero_a1_f64
FFI_BSM_A1_HETERO (_f64, double)
// @BE bitpack_binary_fcnmm_scatter_homo_a0_f64
FFI_BSM_A0_HOMO   (_f64, double)
// @BE bitpack_binary_fcnmm_scatter_hetero_a0_f64
FFI_BSM_A0_HETERO (_f64, double)

// ---- float16 ----
// @BE bitpack_binary_fcnmm_gather_homo_a1_f16
FFI_BGM_A1_HOMO   (_f16, __half, sizeof(float))
// @BE bitpack_binary_fcnmm_gather_hetero_a1_f16
FFI_BGM_A1_HETERO (_f16, __half, sizeof(float))
// @BE bitpack_binary_fcnmm_gather_homo_a0_f16
FFI_BGM_A0_HOMO   (_f16, __half, sizeof(float))
// @BE bitpack_binary_fcnmm_gather_hetero_a0_f16
FFI_BGM_A0_HETERO (_f16, __half, sizeof(float))
// @BE bitpack_binary_fcnmm_scatter_homo_a1_f16
FFI_BSM_A1_HOMO   (_f16, __half)
// @BE bitpack_binary_fcnmm_scatter_hetero_a1_f16
FFI_BSM_A1_HETERO (_f16, __half)
// @BE bitpack_binary_fcnmm_scatter_homo_a0_f16
FFI_BSM_A0_HOMO   (_f16, __half)
// @BE bitpack_binary_fcnmm_scatter_hetero_a0_f16
FFI_BSM_A0_HETERO (_f16, __half)

// ---- bfloat16 ----
// @BE bitpack_binary_fcnmm_gather_homo_a1_bf16
FFI_BGM_A1_HOMO   (_bf16, __nv_bfloat16, sizeof(float))
// @BE bitpack_binary_fcnmm_gather_hetero_a1_bf16
FFI_BGM_A1_HETERO (_bf16, __nv_bfloat16, sizeof(float))
// @BE bitpack_binary_fcnmm_gather_homo_a0_bf16
FFI_BGM_A0_HOMO   (_bf16, __nv_bfloat16, sizeof(float))
// @BE bitpack_binary_fcnmm_gather_hetero_a0_bf16
FFI_BGM_A0_HETERO (_bf16, __nv_bfloat16, sizeof(float))
// @BE bitpack_binary_fcnmm_scatter_homo_a1_bf16
FFI_BSM_A1_HOMO   (_bf16, __nv_bfloat16)
// @BE bitpack_binary_fcnmm_scatter_hetero_a1_bf16
FFI_BSM_A1_HETERO (_bf16, __nv_bfloat16)
// @BE bitpack_binary_fcnmm_scatter_homo_a0_bf16
FFI_BSM_A0_HOMO   (_bf16, __nv_bfloat16)
// @BE bitpack_binary_fcnmm_scatter_hetero_a0_bf16
FFI_BSM_A0_HETERO (_bf16, __nv_bfloat16)
