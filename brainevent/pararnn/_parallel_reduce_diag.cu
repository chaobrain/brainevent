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
 * _parallel_reduce_diag.cu -- Parallel Reduction for Diagonal Bidiagonal Systems
 * ===============================================================================
 *
 * Solves systems of the form:
 *     h[t] = jac[t] * h[t-1] + rhs[t],  h[0] = rhs[0]
 *
 * where jac and rhs are scalars per (batch, hidden_dim) element.
 *
 * Uses a 5-step hierarchical parallel prefix scan:
 *   1. Sequential prefix scan within register chunks
 *   2. Warp-level prefix scan via __shfl_up_sync
 *   3. Block-level prefix scan via shared memory
 *   4. Warp-level forward substitution
 *   5. Chunk-level forward substitution
 *
 * For sequences exceeding single-block capacity, a 3-phase multi-block
 * scheme is used:
 *   Phase 1 (_local): Intra-block prefix scan, write partial results
 *           and per-element accumulated Jacobians, extract block boundaries.
 *   Phase 2 (_boundary): Prefix scan over block boundary elements.
 *   Phase 3 (_final): Correct partial results using solved boundaries:
 *           output[t] += accum_jac[t] * boundary_solution[prev_block].
 *
 * Monoid: (J_b, r_b) @ (J_a, r_a) = (J_b * J_a, J_b * r_a + r_b)
 * Identity: (1, 0)
 *
 * Data layout: jac and rhs are (batch_size, seq_len, hidden_dim) contiguous.
 * Grid: (hidden_dim, batch_size), Block: (threads_per_block).
 * Each thread handles chunk_size consecutive timesteps.
 *
 * CUDA entry points:
 *   pararnn_reduce_diag_f32(jac, rhs, output, stream)
 *   pararnn_reduce_diag_f64(jac, rhs, output, stream)
 */

#include <cuda_runtime.h>
#include <cstdint>
#include "brainevent/common.h"

// ============================================================================
// Constants
// ============================================================================

#define THREADS_PER_WARP 32
#define THREADS_PER_BLOCK 1024

// ============================================================================
// Diagonal Reduction Kernel (parameterized by SCALAR_T, CHUNK_SIZE)
// ============================================================================

#define DEFINE_DIAG_REDUCE(SUFFIX, SCALAR_T, CHUNK_SIZE)                       \
                                                                               \
/* Single-block kernel: handles the entire sequence in one block */            \
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 1)                        \
_pararnn_reduce_diag_single##SUFFIX(                                           \
    const SCALAR_T* __restrict__ jac_in,                                       \
    const SCALAR_T* __restrict__ rhs_in,                                       \
    SCALAR_T* __restrict__ output,                                             \
    int seq_len, int hidden_dim, int batch_size                                \
) {                                                                            \
    const int h_idx = blockIdx.x;                                              \
    const int b_idx = blockIdx.y;                                              \
    if (h_idx >= hidden_dim || b_idx >= batch_size) return;                    \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid & (THREADS_PER_WARP - 1);                             \
    const int warp_id = tid >> 5;                                              \
    const int warps_per_block = (THREADS_PER_BLOCK + THREADS_PER_WARP - 1)     \
                                / THREADS_PER_WARP;                            \
                                                                               \
    const int base = b_idx * seq_len * hidden_dim + h_idx;                     \
    const int stride = hidden_dim;                                             \
                                                                               \
    /* Load chunk into registers */                                            \
    SCALAR_T reg_jac[CHUNK_SIZE];                                              \
    SCALAR_T reg_rhs[CHUNK_SIZE];                                              \
    const int t_start = tid * CHUNK_SIZE;                                      \
    int num_valid = 0;                                                         \
    for (int c = 0; c < CHUNK_SIZE; c++) {                                     \
        int t = t_start + c;                                                   \
        if (t < seq_len) {                                                     \
            reg_jac[c] = jac_in[base + t * stride];                            \
            reg_rhs[c] = rhs_in[base + t * stride];                            \
            num_valid++;                                                       \
        } else {                                                               \
            reg_jac[c] = (SCALAR_T)(1.0);                                      \
            reg_rhs[c] = (SCALAR_T)(0.0);                                      \
        }                                                                      \
    }                                                                          \
    if (tid == 0) {                                                            \
        reg_jac[0] = (SCALAR_T)(0.0);                                          \
    }                                                                          \
                                                                               \
    /* Step 1: Sequential prefix scan within chunk */                          \
    for (int c = 1; c < CHUNK_SIZE; c++) {                                     \
        reg_rhs[c] += reg_jac[c] * reg_rhs[c - 1];                             \
        reg_jac[c] *= reg_jac[c - 1];                                          \
    }                                                                          \
                                                                               \
    /* Step 2: Warp-level prefix scan */                                       \
    SCALAR_T jac_last = reg_jac[CHUNK_SIZE - 1];                               \
    SCALAR_T rhs_last = reg_rhs[CHUNK_SIZE - 1];                               \
    for (int d = 1; d < THREADS_PER_WARP; d <<= 1) {                           \
        SCALAR_T jac_prev = __shfl_up_sync(0xffffffff, jac_last, d);           \
        SCALAR_T rhs_prev = __shfl_up_sync(0xffffffff, rhs_last, d);           \
        if (lane >= d) {                                                       \
            rhs_last += jac_last * rhs_prev;                                   \
            jac_last *= jac_prev;                                              \
        }                                                                      \
    }                                                                          \
    reg_jac[CHUNK_SIZE - 1] = jac_last;                                        \
    reg_rhs[CHUNK_SIZE - 1] = rhs_last;                                        \
                                                                               \
    /* Step 3: Block-level prefix scan via shared memory */                    \
    extern __shared__ char _smem_bytes[];                                      \
    SCALAR_T* smem_jac = reinterpret_cast<SCALAR_T*>(_smem_bytes);             \
    SCALAR_T* smem_rhs = smem_jac + warps_per_block;                           \
    if (lane == (THREADS_PER_WARP - 1) && num_valid > 0) {                     \
        smem_jac[warp_id] = reg_jac[CHUNK_SIZE - 1];                           \
        smem_rhs[warp_id] = reg_rhs[CHUNK_SIZE - 1];                           \
    }                                                                          \
    __syncthreads();                                                           \
    for (int d = 1; d < warps_per_block; d <<= 1) {                            \
        int prev_warp = warp_id - d;                                           \
        if (lane == (THREADS_PER_WARP - 1) && prev_warp >= 0                   \
            && num_valid > 0) {                                                \
            SCALAR_T jp = smem_jac[prev_warp];                                 \
            SCALAR_T rp = smem_rhs[prev_warp];                                 \
            reg_rhs[CHUNK_SIZE - 1] += reg_jac[CHUNK_SIZE - 1] * rp;           \
            reg_jac[CHUNK_SIZE - 1] *= jp;                                     \
        }                                                                      \
        __syncthreads();                                                       \
        if (lane == (THREADS_PER_WARP - 1) && num_valid > 0) {                 \
            smem_jac[warp_id] = reg_jac[CHUNK_SIZE - 1];                       \
            smem_rhs[warp_id] = reg_rhs[CHUNK_SIZE - 1];                       \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
                                                                               \
    /* Step 4: Warp-level forward substitution */                              \
    SCALAR_T sol_prev;                                                         \
    if (warp_id > 0 && lane != (THREADS_PER_WARP - 1)) {                       \
        sol_prev = smem_rhs[warp_id - 1];                                      \
    } else {                                                                   \
        sol_prev = (SCALAR_T)(0.0);                                            \
    }                                                                          \
    reg_rhs[CHUNK_SIZE - 1] += reg_jac[CHUNK_SIZE - 1] * sol_prev;             \
    {                                                                          \
        SCALAR_T prev_rhs = __shfl_up_sync(                                    \
            0xffffffff, reg_rhs[CHUNK_SIZE - 1], 1);                           \
        if (lane > 0) {                                                        \
            sol_prev = prev_rhs;                                               \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* Step 5: Chunk-level forward substitution */                             \
    for (int c = 0; c < CHUNK_SIZE - 1; c++) {                                 \
        reg_rhs[c] += reg_jac[c] * sol_prev;                                   \
    }                                                                          \
                                                                               \
    /* Write results to output */                                              \
    for (int c = 0; c < CHUNK_SIZE; c++) {                                     \
        int t = t_start + c;                                                   \
        if (t < seq_len) {                                                     \
            output[base + t * stride] = reg_rhs[c];                            \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
/* ================================================================== */       \
/* Multi-block Phase 1: Local reduction within each block              */      \
/* Writes partial rhs to output, accumulated jac to accum_jac buffer,  */      \
/* and block boundary (jac, rhs) to bnd_jac/bnd_rhs.                   */      \
/* ================================================================== */       \
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 1)                        \
_pararnn_reduce_diag_local##SUFFIX(                                            \
    const SCALAR_T* __restrict__ jac_in,                                       \
    const SCALAR_T* __restrict__ rhs_in,                                       \
    SCALAR_T* __restrict__ output,                                             \
    SCALAR_T* __restrict__ accum_jac,                                          \
    SCALAR_T* __restrict__ bnd_jac,                                            \
    SCALAR_T* __restrict__ bnd_rhs,                                            \
    int seq_len, int hidden_dim, int batch_size, int block_stride              \
) {                                                                            \
    const int blk_seq = blockIdx.x;                                            \
    const int h_idx = blockIdx.y;                                              \
    const int b_idx = blockIdx.z;                                              \
    if (h_idx >= hidden_dim || b_idx >= batch_size) return;                    \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid & (THREADS_PER_WARP - 1);                             \
    const int warp_id = tid >> 5;                                              \
    const int warps_per_block = (THREADS_PER_BLOCK + THREADS_PER_WARP - 1)     \
                                / THREADS_PER_WARP;                            \
                                                                               \
    const int base = b_idx * seq_len * hidden_dim + h_idx;                     \
    const int stride = hidden_dim;                                             \
    const int t_offset = blk_seq * block_stride;                               \
                                                                               \
    /* Load chunk */                                                           \
    SCALAR_T reg_jac[CHUNK_SIZE];                                              \
    SCALAR_T reg_rhs[CHUNK_SIZE];                                              \
    const int t_start = t_offset + tid * CHUNK_SIZE;                           \
    int num_valid = 0;                                                         \
    for (int c = 0; c < CHUNK_SIZE; c++) {                                     \
        int t = t_start + c;                                                   \
        if (t < seq_len) {                                                     \
            reg_jac[c] = jac_in[base + t * stride];                            \
            reg_rhs[c] = rhs_in[base + t * stride];                            \
            num_valid++;                                                       \
        } else {                                                               \
            reg_jac[c] = (SCALAR_T)(1.0);                                      \
            reg_rhs[c] = (SCALAR_T)(0.0);                                      \
        }                                                                      \
    }                                                                          \
    if (tid == 0 && blk_seq == 0) {                                            \
        reg_jac[0] = (SCALAR_T)(0.0);                                          \
    }                                                                          \
                                                                               \
    /* Step 1: Sequential prefix scan within chunk */                          \
    for (int c = 1; c < CHUNK_SIZE; c++) {                                     \
        reg_rhs[c] += reg_jac[c] * reg_rhs[c - 1];                             \
        reg_jac[c] *= reg_jac[c - 1];                                          \
    }                                                                          \
                                                                               \
    /* Step 2: Warp prefix scan */                                             \
    SCALAR_T jac_last = reg_jac[CHUNK_SIZE - 1];                               \
    SCALAR_T rhs_last = reg_rhs[CHUNK_SIZE - 1];                               \
    for (int d = 1; d < THREADS_PER_WARP; d <<= 1) {                           \
        SCALAR_T jp = __shfl_up_sync(0xffffffff, jac_last, d);                 \
        SCALAR_T rp = __shfl_up_sync(0xffffffff, rhs_last, d);                 \
        if (lane >= d) {                                                       \
            rhs_last += jac_last * rp;                                         \
            jac_last *= jp;                                                    \
        }                                                                      \
    }                                                                          \
    reg_jac[CHUNK_SIZE - 1] = jac_last;                                        \
    reg_rhs[CHUNK_SIZE - 1] = rhs_last;                                        \
                                                                               \
    /* Step 3: Block prefix scan */                                            \
    extern __shared__ char _smem_bytes[];                                      \
    SCALAR_T* smem_jac = reinterpret_cast<SCALAR_T*>(_smem_bytes);             \
    SCALAR_T* smem_rhs = smem_jac + warps_per_block;                           \
    if (lane == (THREADS_PER_WARP - 1) && num_valid > 0) {                     \
        smem_jac[warp_id] = reg_jac[CHUNK_SIZE - 1];                           \
        smem_rhs[warp_id] = reg_rhs[CHUNK_SIZE - 1];                           \
    }                                                                          \
    __syncthreads();                                                           \
    for (int d = 1; d < warps_per_block; d <<= 1) {                            \
        int pw = warp_id - d;                                                  \
        if (lane == (THREADS_PER_WARP - 1) && pw >= 0 && num_valid > 0) {      \
            SCALAR_T jp = smem_jac[pw];                                        \
            SCALAR_T rp = smem_rhs[pw];                                        \
            reg_rhs[CHUNK_SIZE - 1] += reg_jac[CHUNK_SIZE - 1] * rp;           \
            reg_jac[CHUNK_SIZE - 1] *= jp;                                     \
        }                                                                      \
        __syncthreads();                                                       \
        if (lane == (THREADS_PER_WARP - 1) && num_valid > 0) {                 \
            smem_jac[warp_id] = reg_jac[CHUNK_SIZE - 1];                       \
            smem_rhs[warp_id] = reg_rhs[CHUNK_SIZE - 1];                       \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
                                                                               \
    /* Step 4: Warp forward substitution (tracks both jac and rhs) */          \
    SCALAR_T sol_prev, jac_prev_v;                                             \
    if (warp_id > 0 && lane != (THREADS_PER_WARP - 1)) {                       \
        jac_prev_v = smem_jac[warp_id - 1];                                    \
        sol_prev = smem_rhs[warp_id - 1];                                      \
    } else {                                                                   \
        jac_prev_v = (SCALAR_T)(1.0);                                          \
        sol_prev = (SCALAR_T)(0.0);                                            \
    }                                                                          \
    reg_rhs[CHUNK_SIZE - 1] += reg_jac[CHUNK_SIZE - 1] * sol_prev;             \
    reg_jac[CHUNK_SIZE - 1] *= jac_prev_v;                                     \
    {                                                                          \
        SCALAR_T pr = __shfl_up_sync(                                          \
            0xffffffff, reg_rhs[CHUNK_SIZE - 1], 1);                           \
        SCALAR_T pj = __shfl_up_sync(                                          \
            0xffffffff, reg_jac[CHUNK_SIZE - 1], 1);                           \
        if (lane > 0) {                                                        \
            sol_prev = pr;                                                     \
            jac_prev_v = pj;                                                   \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* Step 5: Chunk forward substitution (both jac and rhs) */                \
    for (int c = 0; c < CHUNK_SIZE - 1; c++) {                                 \
        reg_rhs[c] += reg_jac[c] * sol_prev;                                   \
        reg_jac[c] *= jac_prev_v;                                              \
    }                                                                          \
                                                                               \
    /* Write partial rhs to output, accumulated jac to accum_jac */            \
    for (int c = 0; c < CHUNK_SIZE; c++) {                                     \
        int t = t_start + c;                                                   \
        if (t < seq_len) {                                                     \
            output[base + t * stride] = reg_rhs[c];                            \
            accum_jac[base + t * stride] = reg_jac[c];                         \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* Store block boundary from last thread */                                \
    if (tid == THREADS_PER_BLOCK - 1) {                                        \
        int bnd_idx = blk_seq * batch_size * hidden_dim                        \
                      + b_idx * hidden_dim + h_idx;                            \
        bnd_jac[bnd_idx] = reg_jac[CHUNK_SIZE - 1];                            \
        bnd_rhs[bnd_idx] = reg_rhs[CHUNK_SIZE - 1];                            \
    }                                                                          \
}                                                                              \
                                                                               \
/* ================================================================== */       \
/* Multi-block Phase 2: Boundary prefix scan                           */      \
/* Solves the boundary elements via prefix scan, giving the true       */      \
/* solution at the end of each block.                                  */      \
/* ================================================================== */       \
__global__ void __launch_bounds__(1024, 1)                                     \
_pararnn_reduce_diag_boundary##SUFFIX(                                         \
    SCALAR_T* __restrict__ bnd_jac,                                            \
    SCALAR_T* __restrict__ bnd_rhs,                                            \
    int num_blocks, int hidden_dim, int batch_size                             \
) {                                                                            \
    const int h_idx = blockIdx.x;                                              \
    const int b_idx = blockIdx.y;                                              \
    if (h_idx >= hidden_dim || b_idx >= batch_size) return;                    \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid & (THREADS_PER_WARP - 1);                             \
    const int warp_id = tid >> 5;                                              \
    const int hb_stride = batch_size * hidden_dim;                             \
    const int hb_off = b_idx * hidden_dim + h_idx;                             \
                                                                               \
    /* Load boundary elements */                                               \
    SCALAR_T my_jac, my_rhs;                                                   \
    if (tid < num_blocks) {                                                    \
        my_jac = bnd_jac[tid * hb_stride + hb_off];                            \
        my_rhs = bnd_rhs[tid * hb_stride + hb_off];                            \
    } else {                                                                   \
        my_jac = (SCALAR_T)(1.0);                                              \
        my_rhs = (SCALAR_T)(0.0);                                              \
    }                                                                          \
    /* Block 0 boundary: jac[0] was zeroed in _local, so bnd_jac[0]=0 */       \
    /* Explicitly zero to be safe */                                           \
    if (tid == 0) my_jac = (SCALAR_T)(0.0);                                    \
                                                                               \
    /* Warp-level prefix scan */                                               \
    for (int d = 1; d < THREADS_PER_WARP; d <<= 1) {                           \
        SCALAR_T jp = __shfl_up_sync(0xffffffff, my_jac, d);                   \
        SCALAR_T rp = __shfl_up_sync(0xffffffff, my_rhs, d);                   \
        if (lane >= d) {                                                       \
            my_rhs += my_jac * rp;                                             \
            my_jac *= jp;                                                      \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* Block-level prefix scan (if > 1 warp needed) */                         \
    int warps_needed = (num_blocks + THREADS_PER_WARP - 1) / THREADS_PER_WARP; \
    if (warps_needed > 1) {                                                    \
        extern __shared__ char _smem_bytes[];                                  \
        SCALAR_T* smem_jac = reinterpret_cast<SCALAR_T*>(_smem_bytes);         \
        SCALAR_T* smem_rhs = smem_jac + warps_needed;                          \
        if (lane == (THREADS_PER_WARP - 1) && tid < num_blocks) {              \
            smem_jac[warp_id] = my_jac;                                        \
            smem_rhs[warp_id] = my_rhs;                                        \
        }                                                                      \
        __syncthreads();                                                       \
        for (int d = 1; d < warps_needed; d <<= 1) {                           \
            int pw = warp_id - d;                                              \
            if (lane == (THREADS_PER_WARP - 1) && pw >= 0                      \
                && tid < num_blocks) {                                         \
                SCALAR_T jp = smem_jac[pw];                                    \
                SCALAR_T rp = smem_rhs[pw];                                    \
                my_rhs += my_jac * rp;                                         \
                my_jac *= jp;                                                  \
            }                                                                  \
            __syncthreads();                                                   \
            if (lane == (THREADS_PER_WARP - 1) && tid < num_blocks) {          \
                smem_jac[warp_id] = my_jac;                                    \
                smem_rhs[warp_id] = my_rhs;                                    \
            }                                                                  \
            __syncthreads();                                                   \
        }                                                                      \
        /* Forward substitution across warps */                                \
        SCALAR_T sp;                                                           \
        if (warp_id > 0 && lane != (THREADS_PER_WARP - 1)) {                   \
            sp = smem_rhs[warp_id - 1];                                        \
        } else {                                                               \
            sp = (SCALAR_T)(0.0);                                              \
        }                                                                      \
        my_rhs += my_jac * sp;                                                 \
        {                                                                      \
            SCALAR_T pr = __shfl_up_sync(0xffffffff, my_rhs, 1);               \
            if (lane > 0) { sp = pr; }                                         \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* Write back solved boundary values */                                    \
    if (tid < num_blocks) {                                                    \
        bnd_rhs[tid * hb_stride + hb_off] = my_rhs;                            \
    }                                                                          \
}                                                                              \
                                                                               \
/* ================================================================== */       \
/* Multi-block Phase 3: Final correction using solved boundaries       */      \
/* For each block b > 0: output[t] += accum_jac[t] * bnd_rhs[b-1]     */       \
/* ================================================================== */       \
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 1)                        \
_pararnn_reduce_diag_final##SUFFIX(                                            \
    SCALAR_T* __restrict__ output,                                             \
    const SCALAR_T* __restrict__ accum_jac,                                    \
    const SCALAR_T* __restrict__ bnd_rhs,                                      \
    int seq_len, int hidden_dim, int batch_size, int block_stride              \
) {                                                                            \
    const int blk_seq = blockIdx.x + 1;  /* skip first block */                \
    const int h_idx = blockIdx.y;                                              \
    const int b_idx = blockIdx.z;                                              \
    if (h_idx >= hidden_dim || b_idx >= batch_size) return;                    \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int base = b_idx * seq_len * hidden_dim + h_idx;                     \
    const int stride = hidden_dim;                                             \
    const int t_offset = blk_seq * block_stride;                               \
    const int hb_stride = batch_size * hidden_dim;                             \
    const int hb_off = b_idx * hidden_dim + h_idx;                             \
                                                                               \
    /* Read previous block's solved boundary solution */                       \
    SCALAR_T prev_sol = bnd_rhs[(blk_seq - 1) * hb_stride + hb_off];           \
                                                                               \
    /* Correct each element: h[t] = partial_rhs[t] + accum_jac[t]*prev_sol */  \
    for (int c = 0; c < CHUNK_SIZE; c++) {                                     \
        int t = t_offset + tid * CHUNK_SIZE + c;                               \
        if (t < seq_len) {                                                     \
            int idx = base + t * stride;                                       \
            output[idx] += accum_jac[idx] * prev_sol;                          \
        }                                                                      \
    }                                                                          \
}


// ============================================================================
// Instantiate kernels for float32 (chunk_size=2) and float64 (chunk_size=4)
// ============================================================================

DEFINE_DIAG_REDUCE(_f32, float, 2)
DEFINE_DIAG_REDUCE(_f64, double, 4)

// ============================================================================
// CUDA entry points
// ============================================================================

#define DEFINE_FFI_DIAG_REDUCE(SUFFIX, SCALAR_T, CHUNK_SIZE)                     \
void pararnn_reduce_diag##SUFFIX(                                                \
    const BE::Tensor jac_tv,                                                 \
    const BE::Tensor rhs_tv,                                                 \
    BE::Tensor output_tv,                                              \
    int64_t stream                                                               \
) {                                                                              \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                     \
    int batch_size = static_cast<int>(jac_tv.size(0));                           \
    int seq_len    = static_cast<int>(jac_tv.size(1));                           \
    int hidden_dim = static_cast<int>(jac_tv.size(2));                           \
                                                                                 \
    const SCALAR_T* d_jac = static_cast<const SCALAR_T*>(jac_tv.data_ptr());     \
    const SCALAR_T* d_rhs = static_cast<const SCALAR_T*>(rhs_tv.data_ptr());     \
    SCALAR_T* d_out = static_cast<SCALAR_T*>(output_tv.data_ptr());              \
                                                                                 \
    int block_stride = THREADS_PER_BLOCK * CHUNK_SIZE;                           \
    int num_seq_blocks = (seq_len + block_stride - 1) / block_stride;            \
                                                                                 \
    int warps_per_block = (THREADS_PER_BLOCK + THREADS_PER_WARP - 1)             \
                          / THREADS_PER_WARP;                                    \
    unsigned int smem_size = 2 * warps_per_block * sizeof(SCALAR_T);             \
                                                                                 \
    if (num_seq_blocks <= 1) {                                                   \
        /* Single-block path */                                                  \
        dim3 grid(hidden_dim, batch_size);                                       \
        dim3 block(THREADS_PER_BLOCK);                                           \
        _pararnn_reduce_diag_single##SUFFIX<<<grid, block, smem_size, s>>>(      \
            d_jac, d_rhs, d_out, seq_len, hidden_dim, batch_size);               \
    } else {                                                                     \
        /* Multi-block path */                                                   \
        size_t hb_elems = (size_t)batch_size * hidden_dim;                       \
        size_t bnd_bytes = (size_t)num_seq_blocks * hb_elems * sizeof(SCALAR_T); \
        size_t accum_bytes = (size_t)batch_size * seq_len * hidden_dim           \
                             * sizeof(SCALAR_T);                                 \
        size_t total_bytes = 2 * bnd_bytes + accum_bytes;                        \
                                                                                 \
        char* temp = nullptr;                                                    \
        cudaMalloc((void**)&temp, total_bytes);                                  \
        SCALAR_T* bnd_jac_d   = reinterpret_cast<SCALAR_T*>(temp);               \
        SCALAR_T* bnd_rhs_d   = reinterpret_cast<SCALAR_T*>(                     \
                                    temp + bnd_bytes);                           \
        SCALAR_T* accum_jac_d = reinterpret_cast<SCALAR_T*>(                     \
                                    temp + 2 * bnd_bytes);                       \
                                                                                 \
        /* Phase 1: Local reduction per block */                                 \
        dim3 grid1(num_seq_blocks, hidden_dim, batch_size);                      \
        dim3 block1(THREADS_PER_BLOCK);                                          \
        _pararnn_reduce_diag_local##SUFFIX<<<grid1, block1, smem_size, s>>>(     \
            d_jac, d_rhs, d_out, accum_jac_d, bnd_jac_d, bnd_rhs_d,              \
            seq_len, hidden_dim, batch_size, block_stride);                      \
                                                                                 \
        /* Phase 2: Boundary prefix scan */                                      \
        int bnd_threads = num_seq_blocks;                                        \
        if (bnd_threads > 1024) bnd_threads = 1024;                              \
        int bnd_warps = (bnd_threads + THREADS_PER_WARP - 1)                     \
                        / THREADS_PER_WARP;                                      \
        unsigned int bnd_smem = 2 * bnd_warps * sizeof(SCALAR_T);                \
        dim3 grid2(hidden_dim, batch_size);                                      \
        dim3 block2(bnd_threads);                                                \
        _pararnn_reduce_diag_boundary##SUFFIX<<<grid2, block2,                   \
                                                bnd_smem, s>>>(                  \
            bnd_jac_d, bnd_rhs_d, num_seq_blocks, hidden_dim, batch_size);       \
                                                                                 \
        /* Phase 3: Final correction (blocks > 0) */                             \
        dim3 grid3(num_seq_blocks - 1, hidden_dim, batch_size);                  \
        dim3 block3(THREADS_PER_BLOCK);                                          \
        _pararnn_reduce_diag_final##SUFFIX<<<grid3, block3, 0, s>>>(             \
            d_out, accum_jac_d, bnd_rhs_d,                                       \
            seq_len, hidden_dim, batch_size, block_stride);                      \
                                                                                 \
        /* Free temp memory (stream-ordered) */                                  \
        cudaFreeAsync(temp, s);                                                  \
    }                                                                            \
}

// @BE pararnn_reduce_diag_f32
DEFINE_FFI_DIAG_REDUCE(_f32, float, 2)

// @BE pararnn_reduce_diag_f64
DEFINE_FFI_DIAG_REDUCE(_f64, double, 4)
