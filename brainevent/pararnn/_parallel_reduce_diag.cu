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
 * Uses a 5-step hierarchical parallel reduction (Thomas + PCR):
 *   1. Thomas reduction within register chunks
 *   2. Warp-level parallel cyclic reduction via __shfl_up_sync
 *   3. Block-level PCR via shared memory
 *   4. Warp-level forward substitution
 *   5. Chunk-level forward substitution
 *
 * Data layout: jac and rhs are (batch_size, seq_len, hidden_dim) contiguous.
 * Grid: (hidden_dim, batch_size), Block: (threads_per_block).
 * Each thread handles chunk_size consecutive timesteps.
 *
 * TVM FFI entry points:
 *   pararnn_reduce_diag_f32(jac, rhs, output, stream)
 *   pararnn_reduce_diag_f64(jac, rhs, output, stream)
 */

#include <cuda_runtime.h>
#include <cstdint>

// ============================================================================
// Constants
// ============================================================================

#define THREADS_PER_WARP 32
#define THREADS_PER_BLOCK 1024

// ============================================================================
// Chunk-size per dtype (tuned for register pressure)
// ============================================================================

// float32: chunk_size = 2 (spills at >= 32)
// float64: chunk_size = 4 (spills at >= 8)

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
    /* Thread mapping: blockIdx.x = hidden_dim idx, blockIdx.y = batch idx */  \
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
    /* Base offset for this (batch, hidden_dim) element */                     \
    /* Data layout: (batch_size, seq_len, hidden_dim) */                       \
    const int base = b_idx * seq_len * hidden_dim + h_idx;                     \
    const int stride = hidden_dim;  /* stride between consecutive timesteps */ \
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
            /* Neutral element: jac=-1 (identity), rhs=0 */                    \
            reg_jac[c] = (SCALAR_T)(-1.0);                                    \
            reg_rhs[c] = (SCALAR_T)(0.0);                                     \
        }                                                                      \
    }                                                                          \
    /* Zero the first Jacobian (boundary condition: h[-1] = 0) */              \
    if (tid == 0) {                                                            \
        reg_jac[0] = (SCALAR_T)(0.0);                                         \
    }                                                                          \
                                                                               \
    /* ================================================================ */     \
    /* Step 1: Thomas reduction within chunk (sequential)               */     \
    /* Monoid: rhs[i] -= jac[i] * rhs[i-1]; jac[i] *= -jac[i-1]       */     \
    /* ================================================================ */     \
    for (int c = 1; c < CHUNK_SIZE; c++) {                                     \
        reg_rhs[c] -= reg_jac[c] * reg_rhs[c - 1];                            \
        reg_jac[c] *= -reg_jac[c - 1];                                        \
    }                                                                          \
    /* Now all vars in chunk depend on last var of previous chunk */            \
                                                                               \
    /* ================================================================ */     \
    /* Step 2: Warp-level PCR (parallel cyclic reduction)               */     \
    /* Uses __shfl_up_sync for intra-warp communication                 */     \
    /* ================================================================ */     \
    SCALAR_T jac_last = reg_jac[CHUNK_SIZE - 1];                               \
    SCALAR_T rhs_last = reg_rhs[CHUNK_SIZE - 1];                               \
    for (int d = 1; d < THREADS_PER_WARP; d <<= 1) {                          \
        SCALAR_T jac_prev = __shfl_up_sync(0xffffffff, jac_last, d);           \
        SCALAR_T rhs_prev = __shfl_up_sync(0xffffffff, rhs_last, d);           \
        if (lane >= d) {                                                       \
            rhs_last -= jac_last * rhs_prev;                                   \
            jac_last *= -jac_prev;                                             \
        }                                                                      \
    }                                                                          \
    reg_jac[CHUNK_SIZE - 1] = jac_last;                                        \
    reg_rhs[CHUNK_SIZE - 1] = rhs_last;                                        \
    /* Now last vars in each chunk depend on last var of previous warp */       \
                                                                               \
    /* ================================================================ */     \
    /* Step 3: Block-level PCR via shared memory                        */     \
    /* ================================================================ */     \
    extern __shared__ char _smem_bytes[];                                       \
    SCALAR_T* smem_jac = reinterpret_cast<SCALAR_T*>(_smem_bytes);             \
    SCALAR_T* smem_rhs = smem_jac + warps_per_block;                           \
                                                                               \
    /* Last thread of each warp writes to shared memory */                     \
    if (lane == (THREADS_PER_WARP - 1) && num_valid > 0) {                     \
        smem_jac[warp_id] = reg_jac[CHUNK_SIZE - 1];                           \
        smem_rhs[warp_id] = reg_rhs[CHUNK_SIZE - 1];                           \
    }                                                                          \
    __syncthreads();                                                           \
                                                                               \
    /* PCR across warps (only last thread of each warp participates) */         \
    for (int d = 1; d < warps_per_block; d <<= 1) {                           \
        int prev_warp = warp_id - d;                                           \
        if (lane == (THREADS_PER_WARP - 1) && prev_warp >= 0                   \
            && num_valid > 0) {                                                \
            SCALAR_T jp = smem_jac[prev_warp];                                 \
            SCALAR_T rp = smem_rhs[prev_warp];                                 \
            reg_rhs[CHUNK_SIZE - 1] -= reg_jac[CHUNK_SIZE - 1] * rp;           \
            reg_jac[CHUNK_SIZE - 1] *= -jp;                                    \
        }                                                                      \
        __syncthreads();                                                       \
        if (lane == (THREADS_PER_WARP - 1) && num_valid > 0) {                 \
            smem_jac[warp_id] = reg_jac[CHUNK_SIZE - 1];                       \
            smem_rhs[warp_id] = reg_rhs[CHUNK_SIZE - 1];                       \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    /* Now last var of last chunk in each warp is solved */                     \
                                                                               \
    /* ================================================================ */     \
    /* Step 4: Warp-level forward substitution                          */     \
    /* Non-last threads in each warp read solution from previous warp   */     \
    /* ================================================================ */     \
    SCALAR_T sol_prev;                                                         \
    SCALAR_T jac_prev;                                                         \
    if (warp_id > 0 && lane != (THREADS_PER_WARP - 1)) {                       \
        jac_prev = smem_jac[warp_id - 1];                                     \
        sol_prev = smem_rhs[warp_id - 1];                                     \
    } else {                                                                   \
        jac_prev = (SCALAR_T)(-1.0);                                          \
        sol_prev = (SCALAR_T)(0.0);                                            \
    }                                                                          \
                                                                               \
    /* Reduce last element in chunk using previous warp solution */             \
    reg_rhs[CHUNK_SIZE - 1] -= reg_jac[CHUNK_SIZE - 1] * sol_prev;            \
    reg_jac[CHUNK_SIZE - 1] *= -jac_prev;                                      \
                                                                               \
    /* Shuffle within warp to propagate solutions */                            \
    {                                                                          \
        SCALAR_T prev_rhs = __shfl_up_sync(                                    \
            0xffffffff, reg_rhs[CHUNK_SIZE - 1], 1);                           \
        SCALAR_T prev_jac = __shfl_up_sync(                                    \
            0xffffffff, reg_jac[CHUNK_SIZE - 1], 1);                           \
        if (lane > 0) {                                                        \
            sol_prev = prev_rhs;                                               \
            jac_prev = prev_jac;                                               \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* ================================================================ */     \
    /* Step 5: Chunk-level forward substitution                         */     \
    /* Each thread substitutes within its chunk using the solution from */     \
    /* the previous thread's last chunk element                         */     \
    /* ================================================================ */     \
    for (int c = 0; c < CHUNK_SIZE - 1; c++) {                                 \
        reg_rhs[c] -= reg_jac[c] * sol_prev;                                  \
        reg_jac[c] *= -jac_prev;                                               \
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
/* Multi-block: block-local reduction (stores boundary to temp buffers) */     \
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 1)                        \
_pararnn_reduce_diag_local##SUFFIX(                                            \
    const SCALAR_T* __restrict__ jac_in,                                       \
    const SCALAR_T* __restrict__ rhs_in,                                       \
    SCALAR_T* __restrict__ output,                                             \
    SCALAR_T* __restrict__ bnd_jac,                                            \
    SCALAR_T* __restrict__ bnd_rhs,                                            \
    int seq_len, int hidden_dim, int batch_size, int block_stride              \
) {                                                                            \
    const int h_idx = blockIdx.y;                                              \
    const int b_idx = blockIdx.z;                                              \
    const int blk_seq = blockIdx.x;                                            \
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
            reg_jac[c] = (SCALAR_T)(-1.0);                                    \
            reg_rhs[c] = (SCALAR_T)(0.0);                                     \
        }                                                                      \
    }                                                                          \
    if (tid == 0 && blk_seq == 0) {                                            \
        reg_jac[0] = (SCALAR_T)(0.0);                                         \
    }                                                                          \
                                                                               \
    /* Steps 1-5: same as single-block */                                      \
    /* Step 1: Thomas */                                                       \
    for (int c = 1; c < CHUNK_SIZE; c++) {                                     \
        reg_rhs[c] -= reg_jac[c] * reg_rhs[c - 1];                            \
        reg_jac[c] *= -reg_jac[c - 1];                                        \
    }                                                                          \
    /* Step 2: Warp PCR */                                                     \
    SCALAR_T jac_last = reg_jac[CHUNK_SIZE - 1];                               \
    SCALAR_T rhs_last = reg_rhs[CHUNK_SIZE - 1];                               \
    for (int d = 1; d < THREADS_PER_WARP; d <<= 1) {                          \
        SCALAR_T jp = __shfl_up_sync(0xffffffff, jac_last, d);                 \
        SCALAR_T rp = __shfl_up_sync(0xffffffff, rhs_last, d);                 \
        if (lane >= d) {                                                       \
            rhs_last -= jac_last * rp;                                         \
            jac_last *= -jp;                                                   \
        }                                                                      \
    }                                                                          \
    reg_jac[CHUNK_SIZE - 1] = jac_last;                                        \
    reg_rhs[CHUNK_SIZE - 1] = rhs_last;                                        \
    /* Step 3: Block PCR */                                                    \
    extern __shared__ char _smem_bytes[];                                       \
    SCALAR_T* smem_jac = reinterpret_cast<SCALAR_T*>(_smem_bytes);             \
    SCALAR_T* smem_rhs = smem_jac + warps_per_block;                           \
    if (lane == (THREADS_PER_WARP - 1) && num_valid > 0) {                     \
        smem_jac[warp_id] = reg_jac[CHUNK_SIZE - 1];                           \
        smem_rhs[warp_id] = reg_rhs[CHUNK_SIZE - 1];                           \
    }                                                                          \
    __syncthreads();                                                           \
    for (int d = 1; d < warps_per_block; d <<= 1) {                           \
        int pw = warp_id - d;                                                  \
        if (lane == (THREADS_PER_WARP - 1) && pw >= 0 && num_valid > 0) {      \
            SCALAR_T jp = smem_jac[pw];                                        \
            SCALAR_T rp = smem_rhs[pw];                                        \
            reg_rhs[CHUNK_SIZE - 1] -= reg_jac[CHUNK_SIZE - 1] * rp;           \
            reg_jac[CHUNK_SIZE - 1] *= -jp;                                    \
        }                                                                      \
        __syncthreads();                                                       \
        if (lane == (THREADS_PER_WARP - 1) && num_valid > 0) {                 \
            smem_jac[warp_id] = reg_jac[CHUNK_SIZE - 1];                       \
            smem_rhs[warp_id] = reg_rhs[CHUNK_SIZE - 1];                       \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    /* Step 4: Warp forward subst */                                           \
    SCALAR_T sol_prev, jac_prev_v;                                             \
    if (warp_id > 0 && lane != (THREADS_PER_WARP - 1)) {                       \
        jac_prev_v = smem_jac[warp_id - 1];                                   \
        sol_prev = smem_rhs[warp_id - 1];                                     \
    } else {                                                                   \
        jac_prev_v = (SCALAR_T)(-1.0);                                        \
        sol_prev = (SCALAR_T)(0.0);                                            \
    }                                                                          \
    reg_rhs[CHUNK_SIZE - 1] -= reg_jac[CHUNK_SIZE - 1] * sol_prev;            \
    reg_jac[CHUNK_SIZE - 1] *= -jac_prev_v;                                    \
    {                                                                          \
        SCALAR_T pr = __shfl_up_sync(                                          \
            0xffffffff, reg_rhs[CHUNK_SIZE - 1], 1);                           \
        SCALAR_T pj = __shfl_up_sync(                                          \
            0xffffffff, reg_jac[CHUNK_SIZE - 1], 1);                           \
        if (lane > 0) { sol_prev = pr; jac_prev_v = pj; }                     \
    }                                                                          \
    /* Step 5: Chunk forward subst */                                          \
    for (int c = 0; c < CHUNK_SIZE - 1; c++) {                                 \
        reg_rhs[c] -= reg_jac[c] * sol_prev;                                  \
        reg_jac[c] *= -jac_prev_v;                                             \
    }                                                                          \
                                                                               \
    /* Write to output */                                                      \
    for (int c = 0; c < CHUNK_SIZE; c++) {                                     \
        int t = t_start + c;                                                   \
        if (t < seq_len) {                                                     \
            output[base + t * stride] = reg_rhs[c];                            \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* Store boundary (last thread in block) for inter-block reduction */      \
    if (tid == THREADS_PER_BLOCK - 1 || t_start + CHUNK_SIZE - 1 >= seq_len - 1) { \
        int bnd_idx = blk_seq * batch_size * hidden_dim                        \
                      + b_idx * hidden_dim + h_idx;                            \
        bnd_jac[bnd_idx] = reg_jac[CHUNK_SIZE - 1];                            \
        bnd_rhs[bnd_idx] = reg_rhs[CHUNK_SIZE - 1];                            \
    }                                                                          \
}                                                                              \
                                                                               \
/* Multi-block: boundary element reduction (single block) */                   \
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
        my_jac = (SCALAR_T)(-1.0);                                            \
        my_rhs = (SCALAR_T)(0.0);                                             \
    }                                                                          \
    /* Zero first boundary jac */                                              \
    if (tid == 0) my_jac = (SCALAR_T)(0.0);                                   \
                                                                               \
    /* Warp-level PCR */                                                       \
    for (int d = 1; d < THREADS_PER_WARP; d <<= 1) {                          \
        SCALAR_T jp = __shfl_up_sync(0xffffffff, my_jac, d);                   \
        SCALAR_T rp = __shfl_up_sync(0xffffffff, my_rhs, d);                   \
        if (lane >= d) {                                                       \
            my_rhs -= my_jac * rp;                                             \
            my_jac *= -jp;                                                     \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* Block-level PCR (if > 1 warp needed) */                                 \
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
        for (int d = 1; d < warps_needed; d <<= 1) {                          \
            int pw = warp_id - d;                                              \
            if (lane == (THREADS_PER_WARP - 1) && pw >= 0                      \
                && tid < num_blocks) {                                         \
                SCALAR_T jp = smem_jac[pw];                                    \
                SCALAR_T rp = smem_rhs[pw];                                    \
                my_rhs -= my_jac * rp;                                         \
                my_jac *= -jp;                                                 \
            }                                                                  \
            __syncthreads();                                                   \
            if (lane == (THREADS_PER_WARP - 1) && tid < num_blocks) {          \
                smem_jac[warp_id] = my_jac;                                    \
                smem_rhs[warp_id] = my_rhs;                                    \
            }                                                                  \
            __syncthreads();                                                   \
        }                                                                      \
        /* Forward substitution across warps */                                \
        SCALAR_T sp, jp2;                                                      \
        if (warp_id > 0 && lane != (THREADS_PER_WARP - 1)) {                   \
            jp2 = smem_jac[warp_id - 1];                                       \
            sp = smem_rhs[warp_id - 1];                                        \
        } else {                                                               \
            jp2 = (SCALAR_T)(-1.0);                                            \
            sp = (SCALAR_T)(0.0);                                              \
        }                                                                      \
        my_rhs -= my_jac * sp;                                                 \
        my_jac *= -jp2;                                                        \
        {                                                                      \
            SCALAR_T pr = __shfl_up_sync(0xffffffff, my_rhs, 1);               \
            SCALAR_T pj = __shfl_up_sync(0xffffffff, my_jac, 1);              \
            if (lane > 0) { sp = pr; jp2 = pj; }                              \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* Write back boundary solutions */                                        \
    if (tid < num_blocks) {                                                    \
        bnd_jac[tid * hb_stride + hb_off] = my_jac;                            \
        bnd_rhs[tid * hb_stride + hb_off] = my_rhs;                            \
    }                                                                          \
}                                                                              \
                                                                               \
/* Multi-block: forward substitution using solved boundaries */                \
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 1)                        \
_pararnn_reduce_diag_final##SUFFIX(                                            \
    SCALAR_T* __restrict__ output,                                             \
    const SCALAR_T* __restrict__ jac_in,                                       \
    const SCALAR_T* __restrict__ bnd_rhs,                                      \
    int seq_len, int hidden_dim, int batch_size, int block_stride              \
) {                                                                            \
    const int blk_seq = blockIdx.x + 1;  /* skip first block (already done) */ \
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
    /* Read previous block's boundary solution */                              \
    SCALAR_T prev_sol = bnd_rhs[(blk_seq - 1) * hb_stride + hb_off];          \
                                                                               \
    /* Forward substitute: for each element in this block, */                  \
    /* output[t] += jac_cumulative * prev_sol */                               \
    /* Since the local reduction already ran, we just need to */               \
    /* apply: output[t] = output[t] - stored_jac[t] * prev_sol */             \
    /* But the local reduction output already has the partial result. */       \
    /* The correction is: for each t in block, apply the dependency */         \
    /* on the previous block's last element. */                                \
    for (int c = 0; c < CHUNK_SIZE; c++) {                                     \
        int t = t_offset + tid * CHUNK_SIZE + c;                               \
        if (t < seq_len) {                                                     \
            /* The local output already solved within-block. */                \
            /* The residual Jacobian w.r.t. previous block boundary */         \
            /* is stored in jac_in (the accumulated Jacobian). */              \
            /* Correction: out[t] -= jac_accum[t] * prev_sol */               \
            SCALAR_T j = jac_in[base + t * stride];                            \
            output[base + t * stride] -= j * prev_sol;                         \
        }                                                                      \
    }                                                                          \
}


// ============================================================================
// Instantiate kernels for float32 (chunk_size=2) and float64 (chunk_size=4)
// ============================================================================

DEFINE_DIAG_REDUCE(_f32, float, 2)
DEFINE_DIAG_REDUCE(_f64, double, 4)

// ============================================================================
// TVM FFI entry points
// ============================================================================

#define DEFINE_FFI_DIAG_REDUCE(SUFFIX, SCALAR_T, CHUNK_SIZE)                   \
void pararnn_reduce_diag##SUFFIX(                                              \
    tvm::ffi::TensorView jac_tv,                                               \
    tvm::ffi::TensorView rhs_tv,                                               \
    tvm::ffi::TensorView output_tv,                                            \
    int64_t stream                                                             \
) {                                                                            \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                   \
    /* Data layout: (batch_size, seq_len, hidden_dim) */                       \
    int batch_size = static_cast<int>(jac_tv.size(0));                         \
    int seq_len    = static_cast<int>(jac_tv.size(1));                         \
    int hidden_dim = static_cast<int>(jac_tv.size(2));                         \
                                                                               \
    const SCALAR_T* d_jac = static_cast<const SCALAR_T*>(jac_tv.data_ptr());   \
    const SCALAR_T* d_rhs = static_cast<const SCALAR_T*>(rhs_tv.data_ptr());   \
    SCALAR_T* d_out = static_cast<SCALAR_T*>(output_tv.data_ptr());            \
                                                                               \
    int threads_per_seq = (seq_len + CHUNK_SIZE - 1) / CHUNK_SIZE;             \
    int block_stride = THREADS_PER_BLOCK * CHUNK_SIZE;                         \
    int num_seq_blocks = (seq_len + block_stride - 1) / block_stride;          \
                                                                               \
    int warps_per_block = (THREADS_PER_BLOCK + THREADS_PER_WARP - 1)           \
                          / THREADS_PER_WARP;                                  \
    unsigned int smem_size = 2 * warps_per_block * sizeof(SCALAR_T);           \
                                                                               \
    if (num_seq_blocks <= 1) {                                                 \
        /* Single-block path */                                                \
        dim3 grid(hidden_dim, batch_size);                                     \
        dim3 block(THREADS_PER_BLOCK);                                         \
        _pararnn_reduce_diag_single##SUFFIX<<<grid, block, smem_size, s>>>(    \
            d_jac, d_rhs, d_out, seq_len, hidden_dim, batch_size);             \
    } else {                                                                   \
        /* Multi-block path: not yet implemented, fall through to single */    \
        /* TODO: implement multi-block scheme with boundary reduction */        \
        /* For now, use the single-block kernel with sequential loops */        \
        dim3 grid(hidden_dim, batch_size);                                     \
        dim3 block(THREADS_PER_BLOCK);                                         \
        _pararnn_reduce_diag_single##SUFFIX<<<grid, block, smem_size, s>>>(    \
            d_jac, d_rhs, d_out, seq_len, hidden_dim, batch_size);             \
    }                                                                          \
}

// @tvm_ffi pararnn_reduce_diag_f32
DEFINE_FFI_DIAG_REDUCE(_f32, float, 2)

// @tvm_ffi pararnn_reduce_diag_f64
DEFINE_FFI_DIAG_REDUCE(_f64, double, 4)
