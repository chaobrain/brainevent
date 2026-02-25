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
 * _parallel_reduce_block2.cu -- Parallel Reduction for 2x2 Block-Diagonal Systems
 * ================================================================================
 *
 * Solves systems of the form:
 *     h[t] = J[t] @ h[t-1] + rhs[t],  h[0] = rhs[0]
 *
 * where J[t] is a 2x2 matrix and h[t], rhs[t] are 2-vectors, per (batch, block_dim).
 * Used for LSTM-CIFG which has block-diagonal [c, h] Jacobians.
 *
 * Monoid: (J_b, r_b) @ (J_a, r_a) = (J_b @ J_a, J_b @ r_a + r_b)
 * Identity: (I, 0) where I is the 2x2 identity matrix
 *
 * Multi-block scheme (same 3-phase approach as diagonal):
 *   Phase 1 (_local): Intra-block prefix scan, write partial results,
 *           per-element accumulated 2x2 Jacobians, and block boundaries.
 *   Phase 2 (_boundary): Prefix scan over block boundary elements (2x2 monoid).
 *   Phase 3 (_final): Correct partial results using solved boundaries:
 *           output[t] += accum_jac[t] @ boundary_solution[prev_block].
 *
 * Data layout:
 *   jac: (batch_size, seq_len, block_dim, 2, 2) contiguous
 *   rhs: (batch_size, seq_len, block_dim, 2)     contiguous
 *
 * Grid: (block_dim, batch_size), Block: (threads_per_block).
 * Each thread handles chunk_size consecutive timesteps.
 *
 * TVM FFI entry points:
 *   pararnn_reduce_block2_f32(jac, rhs, output, stream)
 *   pararnn_reduce_block2_f64(jac, rhs, output, stream)
 */

#include <cuda_runtime.h>
#include <cstdint>
#include "brainevent/common.h"

#define THREADS_PER_WARP 32
#define THREADS_PER_BLOCK_B2 512

// ============================================================================
// 2x2 matrix/vector operations (register-level)
// ============================================================================

// 2x2 matmul: C = A @ B
#define MAT2_MUL(a00,a01,a10,a11, b00,b01,b10,b11, c00,c01,c10,c11) \
    c00 = a00*b00 + a01*b10;                                        \
    c01 = a00*b01 + a01*b11;                                        \
    c10 = a10*b00 + a11*b10;                                        \
    c11 = a10*b01 + a11*b11;

// 2x2 matvec: c = A @ b
#define MAT2_VEC(a00,a01,a10,a11, b0,b1, c0,c1) \
    c0 = a00*b0 + a01*b1;                       \
    c1 = a10*b0 + a11*b1;

// ============================================================================
// Block-diagonal 2x2 reduction kernel
// ============================================================================

#define DEFINE_BLOCK2_REDUCE(SUFFIX, SCALAR_T, CHUNK_SIZE)                     \
                                                                               \
/* Single-block kernel: handles the entire sequence in one block */            \
__global__ void __launch_bounds__(THREADS_PER_BLOCK_B2, 1)                     \
_pararnn_reduce_block2_single##SUFFIX(                                         \
    const SCALAR_T* __restrict__ jac_in,                                       \
    const SCALAR_T* __restrict__ rhs_in,                                       \
    SCALAR_T* __restrict__ output,                                             \
    int seq_len, int block_dim, int batch_size                                 \
) {                                                                            \
    const int n_idx = blockIdx.x;                                              \
    const int b_idx = blockIdx.y;                                              \
    if (n_idx >= block_dim || b_idx >= batch_size) return;                     \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid & (THREADS_PER_WARP - 1);                             \
    const int warp_id = tid >> 5;                                              \
    const int warps_per_block = (THREADS_PER_BLOCK_B2 + THREADS_PER_WARP - 1)  \
                                / THREADS_PER_WARP;                            \
                                                                               \
    const int jac_base = b_idx * seq_len * block_dim * 4 + n_idx * 4;          \
    const int rhs_base = b_idx * seq_len * block_dim * 2 + n_idx * 2;          \
    const int jac_stride = block_dim * 4;                                      \
    const int rhs_stride = block_dim * 2;                                      \
                                                                               \
    /* Load chunks */                                                          \
    SCALAR_T rj00[CHUNK_SIZE], rj01[CHUNK_SIZE];                               \
    SCALAR_T rj10[CHUNK_SIZE], rj11[CHUNK_SIZE];                               \
    SCALAR_T rr0[CHUNK_SIZE], rr1[CHUNK_SIZE];                                 \
    const int t_start = tid * CHUNK_SIZE;                                      \
    int num_valid = 0;                                                         \
    for (int c = 0; c < CHUNK_SIZE; c++) {                                     \
        int t = t_start + c;                                                   \
        if (t < seq_len) {                                                     \
            int jo = jac_base + t * jac_stride;                                \
            rj00[c] = jac_in[jo+0]; rj01[c] = jac_in[jo+1];                    \
            rj10[c] = jac_in[jo+2]; rj11[c] = jac_in[jo+3];                    \
            int ro = rhs_base + t * rhs_stride;                                \
            rr0[c] = rhs_in[ro+0]; rr1[c] = rhs_in[ro+1];                      \
            num_valid++;                                                       \
        } else {                                                               \
            rj00[c] = (SCALAR_T)(1.0); rj01[c] = (SCALAR_T)(0.0);              \
            rj10[c] = (SCALAR_T)(0.0); rj11[c] = (SCALAR_T)(1.0);              \
            rr0[c] = (SCALAR_T)(0.0);  rr1[c] = (SCALAR_T)(0.0);               \
        }                                                                      \
    }                                                                          \
    if (tid == 0) {                                                            \
        rj00[0] = (SCALAR_T)(0.0); rj01[0] = (SCALAR_T)(0.0);                  \
        rj10[0] = (SCALAR_T)(0.0); rj11[0] = (SCALAR_T)(0.0);                  \
    }                                                                          \
                                                                               \
    /* Step 1: Sequential prefix scan within chunk */                          \
    for (int c = 1; c < CHUNK_SIZE; c++) {                                     \
        SCALAR_T mv0, mv1;                                                     \
        MAT2_VEC(rj00[c],rj01[c],rj10[c],rj11[c],                              \
                 rr0[c-1],rr1[c-1], mv0,mv1);                                  \
        rr0[c] += mv0; rr1[c] += mv1;                                          \
        SCALAR_T m00,m01,m10,m11;                                              \
        MAT2_MUL(rj00[c],rj01[c],rj10[c],rj11[c],                              \
                 rj00[c-1],rj01[c-1],rj10[c-1],rj11[c-1],                      \
                 m00,m01,m10,m11);                                             \
        rj00[c] = m00; rj01[c] = m01;                                          \
        rj10[c] = m10; rj11[c] = m11;                                          \
    }                                                                          \
                                                                               \
    /* Step 2: Warp-level prefix scan */                                       \
    int lc = CHUNK_SIZE - 1;                                                   \
    for (int d = 1; d < THREADS_PER_WARP; d <<= 1) {                           \
        SCALAR_T pj00 = __shfl_up_sync(0xffffffff, rj00[lc], d);               \
        SCALAR_T pj01 = __shfl_up_sync(0xffffffff, rj01[lc], d);               \
        SCALAR_T pj10 = __shfl_up_sync(0xffffffff, rj10[lc], d);               \
        SCALAR_T pj11 = __shfl_up_sync(0xffffffff, rj11[lc], d);               \
        SCALAR_T pr0  = __shfl_up_sync(0xffffffff, rr0[lc], d);                \
        SCALAR_T pr1  = __shfl_up_sync(0xffffffff, rr1[lc], d);                \
        if (lane >= d) {                                                       \
            SCALAR_T mv0, mv1;                                                 \
            MAT2_VEC(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                      \
                     pr0,pr1, mv0,mv1);                                        \
            rr0[lc] += mv0; rr1[lc] += mv1;                                    \
            SCALAR_T m00,m01,m10,m11;                                          \
            MAT2_MUL(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                      \
                     pj00,pj01,pj10,pj11, m00,m01,m10,m11);                    \
            rj00[lc] = m00; rj01[lc] = m01;                                    \
            rj10[lc] = m10; rj11[lc] = m11;                                    \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* Step 3: Block-level prefix scan via shared memory */                    \
    extern __shared__ char _smem_bytes[];                                      \
    SCALAR_T* smem = reinterpret_cast<SCALAR_T*>(_smem_bytes);                 \
    if (lane == (THREADS_PER_WARP - 1) && num_valid > 0) {                     \
        int off = warp_id * 6;                                                 \
        smem[off+0] = rj00[lc]; smem[off+1] = rj01[lc];                        \
        smem[off+2] = rj10[lc]; smem[off+3] = rj11[lc];                        \
        smem[off+4] = rr0[lc];  smem[off+5] = rr1[lc];                         \
    }                                                                          \
    __syncthreads();                                                           \
    for (int d = 1; d < warps_per_block; d <<= 1) {                            \
        int pw = warp_id - d;                                                  \
        if (lane == (THREADS_PER_WARP - 1) && pw >= 0 && num_valid > 0) {      \
            int poff = pw * 6;                                                 \
            SCALAR_T pj00_ = smem[poff+0], pj01_ = smem[poff+1];               \
            SCALAR_T pj10_ = smem[poff+2], pj11_ = smem[poff+3];               \
            SCALAR_T pr0_ = smem[poff+4], pr1_ = smem[poff+5];                 \
            SCALAR_T mv0, mv1;                                                 \
            MAT2_VEC(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                      \
                     pr0_,pr1_, mv0,mv1);                                      \
            rr0[lc] += mv0; rr1[lc] += mv1;                                    \
            SCALAR_T m00,m01,m10,m11;                                          \
            MAT2_MUL(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                      \
                     pj00_,pj01_,pj10_,pj11_, m00,m01,m10,m11);                \
            rj00[lc] = m00; rj01[lc] = m01;                                    \
            rj10[lc] = m10; rj11[lc] = m11;                                    \
        }                                                                      \
        __syncthreads();                                                       \
        if (lane == (THREADS_PER_WARP - 1) && num_valid > 0) {                 \
            int off = warp_id * 6;                                             \
            smem[off+0] = rj00[lc]; smem[off+1] = rj01[lc];                    \
            smem[off+2] = rj10[lc]; smem[off+3] = rj11[lc];                    \
            smem[off+4] = rr0[lc];  smem[off+5] = rr1[lc];                     \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
                                                                               \
    /* Step 4: Warp-level forward substitution */                              \
    SCALAR_T sp_r0, sp_r1;                                                     \
    if (warp_id > 0 && lane != (THREADS_PER_WARP - 1)) {                       \
        int poff = (warp_id - 1) * 6;                                          \
        sp_r0 = smem[poff+4]; sp_r1 = smem[poff+5];                            \
    } else {                                                                   \
        sp_r0 = (SCALAR_T)(0.0); sp_r1 = (SCALAR_T)(0.0);                      \
    }                                                                          \
    {                                                                          \
        SCALAR_T mv0, mv1;                                                     \
        MAT2_VEC(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                          \
                 sp_r0,sp_r1, mv0,mv1);                                        \
        rr0[lc] += mv0; rr1[lc] += mv1;                                        \
    }                                                                          \
    {                                                                          \
        SCALAR_T pr0 = __shfl_up_sync(0xffffffff, rr0[lc], 1);                 \
        SCALAR_T pr1 = __shfl_up_sync(0xffffffff, rr1[lc], 1);                 \
        if (lane > 0) {                                                        \
            sp_r0 = pr0; sp_r1 = pr1;                                          \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* Step 5: Chunk-level forward substitution */                             \
    for (int c = 0; c < CHUNK_SIZE - 1; c++) {                                 \
        SCALAR_T mv0, mv1;                                                     \
        MAT2_VEC(rj00[c],rj01[c],rj10[c],rj11[c],                              \
                 sp_r0,sp_r1, mv0,mv1);                                        \
        rr0[c] += mv0; rr1[c] += mv1;                                          \
    }                                                                          \
                                                                               \
    /* Write output */                                                         \
    for (int c = 0; c < CHUNK_SIZE; c++) {                                     \
        int t = t_start + c;                                                   \
        if (t < seq_len) {                                                     \
            int ro = rhs_base + t * rhs_stride;                                \
            output[ro+0] = rr0[c]; output[ro+1] = rr1[c];                      \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
/* ================================================================== */       \
/* Multi-block Phase 1: Local reduction within each block              */      \
/* ================================================================== */       \
__global__ void __launch_bounds__(THREADS_PER_BLOCK_B2, 1)                     \
_pararnn_reduce_block2_local##SUFFIX(                                          \
    const SCALAR_T* __restrict__ jac_in,                                       \
    const SCALAR_T* __restrict__ rhs_in,                                       \
    SCALAR_T* __restrict__ output,                                             \
    SCALAR_T* __restrict__ accum_jac,                                          \
    SCALAR_T* __restrict__ bnd_jac,                                            \
    SCALAR_T* __restrict__ bnd_rhs,                                            \
    int seq_len, int block_dim, int batch_size, int block_stride_t             \
) {                                                                            \
    const int blk_seq = blockIdx.x;                                            \
    const int n_idx = blockIdx.y;                                              \
    const int b_idx = blockIdx.z;                                              \
    if (n_idx >= block_dim || b_idx >= batch_size) return;                     \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid & (THREADS_PER_WARP - 1);                             \
    const int warp_id = tid >> 5;                                              \
    const int warps_per_block = (THREADS_PER_BLOCK_B2 + THREADS_PER_WARP - 1)  \
                                / THREADS_PER_WARP;                            \
                                                                               \
    const int jac_base = b_idx * seq_len * block_dim * 4 + n_idx * 4;          \
    const int rhs_base = b_idx * seq_len * block_dim * 2 + n_idx * 2;          \
    const int jac_stride = block_dim * 4;                                      \
    const int rhs_stride = block_dim * 2;                                      \
    const int t_offset = blk_seq * block_stride_t;                             \
                                                                               \
    /* Load chunks */                                                          \
    SCALAR_T rj00[CHUNK_SIZE], rj01[CHUNK_SIZE];                               \
    SCALAR_T rj10[CHUNK_SIZE], rj11[CHUNK_SIZE];                               \
    SCALAR_T rr0[CHUNK_SIZE], rr1[CHUNK_SIZE];                                 \
    const int t_start = t_offset + tid * CHUNK_SIZE;                           \
    int num_valid = 0;                                                         \
    for (int c = 0; c < CHUNK_SIZE; c++) {                                     \
        int t = t_start + c;                                                   \
        if (t < seq_len) {                                                     \
            int jo = jac_base + t * jac_stride;                                \
            rj00[c] = jac_in[jo+0]; rj01[c] = jac_in[jo+1];                    \
            rj10[c] = jac_in[jo+2]; rj11[c] = jac_in[jo+3];                    \
            int ro = rhs_base + t * rhs_stride;                                \
            rr0[c] = rhs_in[ro+0]; rr1[c] = rhs_in[ro+1];                      \
            num_valid++;                                                       \
        } else {                                                               \
            rj00[c] = (SCALAR_T)(1.0); rj01[c] = (SCALAR_T)(0.0);              \
            rj10[c] = (SCALAR_T)(0.0); rj11[c] = (SCALAR_T)(1.0);              \
            rr0[c] = (SCALAR_T)(0.0);  rr1[c] = (SCALAR_T)(0.0);               \
        }                                                                      \
    }                                                                          \
    if (tid == 0 && blk_seq == 0) {                                            \
        rj00[0] = (SCALAR_T)(0.0); rj01[0] = (SCALAR_T)(0.0);                  \
        rj10[0] = (SCALAR_T)(0.0); rj11[0] = (SCALAR_T)(0.0);                  \
    }                                                                          \
                                                                               \
    /* Step 1: Sequential prefix scan within chunk */                          \
    for (int c = 1; c < CHUNK_SIZE; c++) {                                     \
        SCALAR_T mv0, mv1;                                                     \
        MAT2_VEC(rj00[c],rj01[c],rj10[c],rj11[c],                              \
                 rr0[c-1],rr1[c-1], mv0,mv1);                                  \
        rr0[c] += mv0; rr1[c] += mv1;                                          \
        SCALAR_T m00,m01,m10,m11;                                              \
        MAT2_MUL(rj00[c],rj01[c],rj10[c],rj11[c],                              \
                 rj00[c-1],rj01[c-1],rj10[c-1],rj11[c-1],                      \
                 m00,m01,m10,m11);                                             \
        rj00[c] = m00; rj01[c] = m01;                                          \
        rj10[c] = m10; rj11[c] = m11;                                          \
    }                                                                          \
                                                                               \
    /* Step 2: Warp-level prefix scan */                                       \
    int lc = CHUNK_SIZE - 1;                                                   \
    for (int d = 1; d < THREADS_PER_WARP; d <<= 1) {                           \
        SCALAR_T pj00 = __shfl_up_sync(0xffffffff, rj00[lc], d);               \
        SCALAR_T pj01 = __shfl_up_sync(0xffffffff, rj01[lc], d);               \
        SCALAR_T pj10 = __shfl_up_sync(0xffffffff, rj10[lc], d);               \
        SCALAR_T pj11 = __shfl_up_sync(0xffffffff, rj11[lc], d);               \
        SCALAR_T pr0  = __shfl_up_sync(0xffffffff, rr0[lc], d);                \
        SCALAR_T pr1  = __shfl_up_sync(0xffffffff, rr1[lc], d);                \
        if (lane >= d) {                                                       \
            SCALAR_T mv0, mv1;                                                 \
            MAT2_VEC(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                      \
                     pr0,pr1, mv0,mv1);                                        \
            rr0[lc] += mv0; rr1[lc] += mv1;                                    \
            SCALAR_T m00,m01,m10,m11;                                          \
            MAT2_MUL(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                      \
                     pj00,pj01,pj10,pj11, m00,m01,m10,m11);                    \
            rj00[lc] = m00; rj01[lc] = m01;                                    \
            rj10[lc] = m10; rj11[lc] = m11;                                    \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* Step 3: Block-level prefix scan via shared memory */                    \
    extern __shared__ char _smem_bytes[];                                      \
    SCALAR_T* smem = reinterpret_cast<SCALAR_T*>(_smem_bytes);                 \
    if (lane == (THREADS_PER_WARP - 1) && num_valid > 0) {                     \
        int off = warp_id * 6;                                                 \
        smem[off+0] = rj00[lc]; smem[off+1] = rj01[lc];                        \
        smem[off+2] = rj10[lc]; smem[off+3] = rj11[lc];                        \
        smem[off+4] = rr0[lc];  smem[off+5] = rr1[lc];                         \
    }                                                                          \
    __syncthreads();                                                           \
    for (int d = 1; d < warps_per_block; d <<= 1) {                            \
        int pw = warp_id - d;                                                  \
        if (lane == (THREADS_PER_WARP - 1) && pw >= 0 && num_valid > 0) {      \
            int poff = pw * 6;                                                 \
            SCALAR_T pj00_ = smem[poff+0], pj01_ = smem[poff+1];               \
            SCALAR_T pj10_ = smem[poff+2], pj11_ = smem[poff+3];               \
            SCALAR_T pr0_ = smem[poff+4], pr1_ = smem[poff+5];                 \
            SCALAR_T mv0, mv1;                                                 \
            MAT2_VEC(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                      \
                     pr0_,pr1_, mv0,mv1);                                      \
            rr0[lc] += mv0; rr1[lc] += mv1;                                    \
            SCALAR_T m00,m01,m10,m11;                                          \
            MAT2_MUL(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                      \
                     pj00_,pj01_,pj10_,pj11_, m00,m01,m10,m11);                \
            rj00[lc] = m00; rj01[lc] = m01;                                    \
            rj10[lc] = m10; rj11[lc] = m11;                                    \
        }                                                                      \
        __syncthreads();                                                       \
        if (lane == (THREADS_PER_WARP - 1) && num_valid > 0) {                 \
            int off = warp_id * 6;                                             \
            smem[off+0] = rj00[lc]; smem[off+1] = rj01[lc];                    \
            smem[off+2] = rj10[lc]; smem[off+3] = rj11[lc];                    \
            smem[off+4] = rr0[lc];  smem[off+5] = rr1[lc];                     \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
                                                                               \
    /* Step 4: Warp-level forward substitution (tracks both jac and rhs) */    \
    SCALAR_T sp_j00, sp_j01, sp_j10, sp_j11, sp_r0, sp_r1;                     \
    if (warp_id > 0 && lane != (THREADS_PER_WARP - 1)) {                       \
        int poff = (warp_id - 1) * 6;                                          \
        sp_j00 = smem[poff+0]; sp_j01 = smem[poff+1];                          \
        sp_j10 = smem[poff+2]; sp_j11 = smem[poff+3];                          \
        sp_r0  = smem[poff+4]; sp_r1  = smem[poff+5];                          \
    } else {                                                                   \
        sp_j00 = (SCALAR_T)(1.0); sp_j01 = (SCALAR_T)(0.0);                    \
        sp_j10 = (SCALAR_T)(0.0); sp_j11 = (SCALAR_T)(1.0);                    \
        sp_r0  = (SCALAR_T)(0.0); sp_r1  = (SCALAR_T)(0.0);                    \
    }                                                                          \
    /* Compose last chunk element with warp boundary */                        \
    {                                                                          \
        SCALAR_T mv0, mv1;                                                     \
        MAT2_VEC(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                          \
                 sp_r0,sp_r1, mv0,mv1);                                        \
        rr0[lc] += mv0; rr1[lc] += mv1;                                        \
        SCALAR_T m00,m01,m10,m11;                                              \
        MAT2_MUL(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                          \
                 sp_j00,sp_j01,sp_j10,sp_j11, m00,m01,m10,m11);                \
        rj00[lc] = m00; rj01[lc] = m01;                                        \
        rj10[lc] = m10; rj11[lc] = m11;                                        \
    }                                                                          \
    /* Shuffle to get previous thread's solved values */                       \
    {                                                                          \
        SCALAR_T pr0 = __shfl_up_sync(0xffffffff, rr0[lc], 1);                 \
        SCALAR_T pr1 = __shfl_up_sync(0xffffffff, rr1[lc], 1);                 \
        SCALAR_T pj00_ = __shfl_up_sync(0xffffffff, rj00[lc], 1);              \
        SCALAR_T pj01_ = __shfl_up_sync(0xffffffff, rj01[lc], 1);              \
        SCALAR_T pj10_ = __shfl_up_sync(0xffffffff, rj10[lc], 1);              \
        SCALAR_T pj11_ = __shfl_up_sync(0xffffffff, rj11[lc], 1);              \
        if (lane > 0) {                                                        \
            sp_r0 = pr0; sp_r1 = pr1;                                          \
            sp_j00 = pj00_; sp_j01 = pj01_;                                    \
            sp_j10 = pj10_; sp_j11 = pj11_;                                    \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* Step 5: Chunk-level forward substitution (both jac and rhs) */          \
    for (int c = 0; c < CHUNK_SIZE - 1; c++) {                                 \
        SCALAR_T mv0, mv1;                                                     \
        MAT2_VEC(rj00[c],rj01[c],rj10[c],rj11[c],                              \
                 sp_r0,sp_r1, mv0,mv1);                                        \
        rr0[c] += mv0; rr1[c] += mv1;                                          \
        SCALAR_T m00,m01,m10,m11;                                              \
        MAT2_MUL(rj00[c],rj01[c],rj10[c],rj11[c],                              \
                 sp_j00,sp_j01,sp_j10,sp_j11, m00,m01,m10,m11);                \
        rj00[c] = m00; rj01[c] = m01;                                          \
        rj10[c] = m10; rj11[c] = m11;                                          \
    }                                                                          \
                                                                               \
    /* Write partial rhs to output, accumulated jac to accum_jac */            \
    for (int c = 0; c < CHUNK_SIZE; c++) {                                     \
        int t = t_start + c;                                                   \
        if (t < seq_len) {                                                     \
            int ro = rhs_base + t * rhs_stride;                                \
            output[ro+0] = rr0[c]; output[ro+1] = rr1[c];                      \
            int jo = jac_base + t * jac_stride;                                \
            accum_jac[jo+0] = rj00[c]; accum_jac[jo+1] = rj01[c];              \
            accum_jac[jo+2] = rj10[c]; accum_jac[jo+3] = rj11[c];              \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* Store block boundary from last thread */                                \
    if (tid == THREADS_PER_BLOCK_B2 - 1) {                                     \
        int bn = blk_seq * batch_size * block_dim + b_idx * block_dim + n_idx; \
        bnd_jac[bn*4+0] = rj00[lc]; bnd_jac[bn*4+1] = rj01[lc];                \
        bnd_jac[bn*4+2] = rj10[lc]; bnd_jac[bn*4+3] = rj11[lc];                \
        bnd_rhs[bn*2+0] = rr0[lc];  bnd_rhs[bn*2+1] = rr1[lc];                 \
    }                                                                          \
}                                                                              \
                                                                               \
/* ================================================================== */       \
/* Multi-block Phase 2: Boundary prefix scan (2x2 monoid)              */      \
/* ================================================================== */       \
__global__ void __launch_bounds__(512, 1)                                      \
_pararnn_reduce_block2_boundary##SUFFIX(                                       \
    SCALAR_T* __restrict__ bnd_jac,                                            \
    SCALAR_T* __restrict__ bnd_rhs,                                            \
    int num_blocks, int block_dim, int batch_size                              \
) {                                                                            \
    const int n_idx = blockIdx.x;                                              \
    const int b_idx = blockIdx.y;                                              \
    if (n_idx >= block_dim || b_idx >= batch_size) return;                     \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid & (THREADS_PER_WARP - 1);                             \
    const int warp_id = tid >> 5;                                              \
    const int nb_stride = batch_size * block_dim;                              \
    const int nb_off = b_idx * block_dim + n_idx;                              \
                                                                               \
    /* Load boundary elements */                                               \
    SCALAR_T j00, j01, j10, j11, r0, r1;                                       \
    if (tid < num_blocks) {                                                    \
        int idx = tid * nb_stride + nb_off;                                    \
        j00 = bnd_jac[idx*4+0]; j01 = bnd_jac[idx*4+1];                        \
        j10 = bnd_jac[idx*4+2]; j11 = bnd_jac[idx*4+3];                        \
        r0  = bnd_rhs[idx*2+0]; r1  = bnd_rhs[idx*2+1];                        \
    } else {                                                                   \
        j00 = (SCALAR_T)(1.0); j01 = (SCALAR_T)(0.0);                          \
        j10 = (SCALAR_T)(0.0); j11 = (SCALAR_T)(1.0);                          \
        r0  = (SCALAR_T)(0.0); r1  = (SCALAR_T)(0.0);                          \
    }                                                                          \
    if (tid == 0) {                                                            \
        j00 = (SCALAR_T)(0.0); j01 = (SCALAR_T)(0.0);                          \
        j10 = (SCALAR_T)(0.0); j11 = (SCALAR_T)(0.0);                          \
    }                                                                          \
                                                                               \
    /* Warp-level prefix scan */                                               \
    for (int d = 1; d < THREADS_PER_WARP; d <<= 1) {                           \
        SCALAR_T pj00 = __shfl_up_sync(0xffffffff, j00, d);                    \
        SCALAR_T pj01 = __shfl_up_sync(0xffffffff, j01, d);                    \
        SCALAR_T pj10 = __shfl_up_sync(0xffffffff, j10, d);                    \
        SCALAR_T pj11 = __shfl_up_sync(0xffffffff, j11, d);                    \
        SCALAR_T pr0  = __shfl_up_sync(0xffffffff, r0, d);                     \
        SCALAR_T pr1  = __shfl_up_sync(0xffffffff, r1, d);                     \
        if (lane >= d) {                                                       \
            SCALAR_T mv0, mv1;                                                 \
            MAT2_VEC(j00,j01,j10,j11, pr0,pr1, mv0,mv1);                       \
            r0 += mv0; r1 += mv1;                                              \
            SCALAR_T m00,m01,m10,m11;                                          \
            MAT2_MUL(j00,j01,j10,j11, pj00,pj01,pj10,pj11,                     \
                     m00,m01,m10,m11);                                         \
            j00 = m00; j01 = m01; j10 = m10; j11 = m11;                        \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* Block-level prefix scan (if > 1 warp needed) */                         \
    int warps_needed = (num_blocks + THREADS_PER_WARP - 1) / THREADS_PER_WARP; \
    if (warps_needed > 1) {                                                    \
        extern __shared__ char _smem_bytes[];                                  \
        SCALAR_T* smem = reinterpret_cast<SCALAR_T*>(_smem_bytes);             \
        if (lane == (THREADS_PER_WARP - 1) && tid < num_blocks) {              \
            int off = warp_id * 6;                                             \
            smem[off+0] = j00; smem[off+1] = j01;                              \
            smem[off+2] = j10; smem[off+3] = j11;                              \
            smem[off+4] = r0;  smem[off+5] = r1;                               \
        }                                                                      \
        __syncthreads();                                                       \
        for (int d = 1; d < warps_needed; d <<= 1) {                           \
            int pw = warp_id - d;                                              \
            if (lane == (THREADS_PER_WARP - 1) && pw >= 0                      \
                && tid < num_blocks) {                                         \
                int poff = pw * 6;                                             \
                SCALAR_T pj00_ = smem[poff+0], pj01_ = smem[poff+1];           \
                SCALAR_T pj10_ = smem[poff+2], pj11_ = smem[poff+3];           \
                SCALAR_T pr0_ = smem[poff+4], pr1_ = smem[poff+5];             \
                SCALAR_T mv0, mv1;                                             \
                MAT2_VEC(j00,j01,j10,j11, pr0_,pr1_, mv0,mv1);                 \
                r0 += mv0; r1 += mv1;                                          \
                SCALAR_T m00,m01,m10,m11;                                      \
                MAT2_MUL(j00,j01,j10,j11,                                      \
                         pj00_,pj01_,pj10_,pj11_, m00,m01,m10,m11);            \
                j00 = m00; j01 = m01; j10 = m10; j11 = m11;                    \
            }                                                                  \
            __syncthreads();                                                   \
            if (lane == (THREADS_PER_WARP - 1) && tid < num_blocks) {          \
                int off = warp_id * 6;                                         \
                smem[off+0] = j00; smem[off+1] = j01;                          \
                smem[off+2] = j10; smem[off+3] = j11;                          \
                smem[off+4] = r0;  smem[off+5] = r1;                           \
            }                                                                  \
            __syncthreads();                                                   \
        }                                                                      \
        /* Forward substitution across warps */                                \
        SCALAR_T sp0, sp1;                                                     \
        if (warp_id > 0 && lane != (THREADS_PER_WARP - 1)) {                   \
            int poff = (warp_id - 1) * 6;                                      \
            sp0 = smem[poff+4]; sp1 = smem[poff+5];                            \
        } else {                                                               \
            sp0 = (SCALAR_T)(0.0); sp1 = (SCALAR_T)(0.0);                      \
        }                                                                      \
        {                                                                      \
            SCALAR_T mv0, mv1;                                                 \
            MAT2_VEC(j00,j01,j10,j11, sp0,sp1, mv0,mv1);                       \
            r0 += mv0; r1 += mv1;                                              \
        }                                                                      \
        {                                                                      \
            SCALAR_T pr0_ = __shfl_up_sync(0xffffffff, r0, 1);                 \
            SCALAR_T pr1_ = __shfl_up_sync(0xffffffff, r1, 1);                 \
            if (lane > 0) { sp0 = pr0_; sp1 = pr1_; }                          \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* Write back solved boundary rhs */                                       \
    if (tid < num_blocks) {                                                    \
        int idx = tid * nb_stride + nb_off;                                    \
        bnd_rhs[idx*2+0] = r0; bnd_rhs[idx*2+1] = r1;                          \
    }                                                                          \
}                                                                              \
                                                                               \
/* ================================================================== */       \
/* Multi-block Phase 3: Final correction                               */      \
/* output[t] += accum_jac[t] @ bnd_rhs[prev_block]                    */       \
/* ================================================================== */       \
__global__ void __launch_bounds__(THREADS_PER_BLOCK_B2, 1)                     \
_pararnn_reduce_block2_final##SUFFIX(                                          \
    SCALAR_T* __restrict__ output,                                             \
    const SCALAR_T* __restrict__ accum_jac,                                    \
    const SCALAR_T* __restrict__ bnd_rhs,                                      \
    int seq_len, int block_dim, int batch_size, int block_stride_t             \
) {                                                                            \
    const int blk_seq = blockIdx.x + 1;  /* skip first block */                \
    const int n_idx = blockIdx.y;                                              \
    const int b_idx = blockIdx.z;                                              \
    if (n_idx >= block_dim || b_idx >= batch_size) return;                     \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int jac_base = b_idx * seq_len * block_dim * 4 + n_idx * 4;          \
    const int rhs_base = b_idx * seq_len * block_dim * 2 + n_idx * 2;          \
    const int jac_stride = block_dim * 4;                                      \
    const int rhs_stride = block_dim * 2;                                      \
    const int t_offset = blk_seq * block_stride_t;                             \
    const int nb_stride = batch_size * block_dim;                              \
    const int nb_off = b_idx * block_dim + n_idx;                              \
                                                                               \
    /* Read previous block's solved boundary solution */                       \
    int prev_idx = (blk_seq - 1) * nb_stride + nb_off;                         \
    SCALAR_T prev_r0 = bnd_rhs[prev_idx*2+0];                                  \
    SCALAR_T prev_r1 = bnd_rhs[prev_idx*2+1];                                  \
                                                                               \
    /* Correct each element: output += accum_jac @ prev_sol */                 \
    for (int c = 0; c < CHUNK_SIZE; c++) {                                     \
        int t = t_offset + tid * CHUNK_SIZE + c;                               \
        if (t < seq_len) {                                                     \
            int jo = jac_base + t * jac_stride;                                \
            SCALAR_T aj00 = accum_jac[jo+0], aj01 = accum_jac[jo+1];           \
            SCALAR_T aj10 = accum_jac[jo+2], aj11 = accum_jac[jo+3];           \
            SCALAR_T mv0, mv1;                                                 \
            MAT2_VEC(aj00,aj01,aj10,aj11, prev_r0,prev_r1, mv0,mv1);           \
            int ro = rhs_base + t * rhs_stride;                                \
            output[ro+0] += mv0; output[ro+1] += mv1;                          \
        }                                                                      \
    }                                                                          \
}


// ============================================================================
// Instantiate kernels for float32 and float64 (chunk_size=2 for both)
// ============================================================================

DEFINE_BLOCK2_REDUCE(_f32, float, 2)
DEFINE_BLOCK2_REDUCE(_f64, double, 2)

// ============================================================================
// TVM FFI entry points
// ============================================================================

#define DEFINE_FFI_BLOCK2_REDUCE(SUFFIX, SCALAR_T, CHUNK_SIZE)                \
void pararnn_reduce_block2##SUFFIX(                                           \
    const BE::Tensor jac_tv,                                              \
    const BE::Tensor rhs_tv,                                              \
    BE::Tensor output_tv,                                           \
    int64_t stream                                                            \
) {                                                                           \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                  \
    int batch_size = static_cast<int>(jac_tv.size(0));                        \
    int seq_len    = static_cast<int>(jac_tv.size(1));                        \
    int block_dim  = static_cast<int>(jac_tv.size(2));                        \
                                                                              \
    const SCALAR_T* d_jac = static_cast<const SCALAR_T*>(jac_tv.data_ptr());  \
    const SCALAR_T* d_rhs = static_cast<const SCALAR_T*>(rhs_tv.data_ptr());  \
    SCALAR_T* d_out = static_cast<SCALAR_T*>(output_tv.data_ptr());           \
                                                                              \
    int block_stride_t = THREADS_PER_BLOCK_B2 * CHUNK_SIZE;                   \
    int num_seq_blocks = (seq_len + block_stride_t - 1) / block_stride_t;     \
                                                                              \
    int warps_per_block = (THREADS_PER_BLOCK_B2 + THREADS_PER_WARP - 1)       \
                          / THREADS_PER_WARP;                                 \
    unsigned int smem_size = warps_per_block * 6 * sizeof(SCALAR_T);          \
                                                                              \
    if (num_seq_blocks <= 1) {                                                \
        /* Single-block path */                                               \
        dim3 grid(block_dim, batch_size);                                     \
        dim3 block(THREADS_PER_BLOCK_B2);                                     \
        _pararnn_reduce_block2_single##SUFFIX<<<grid, block,                  \
                                                smem_size, s>>>(              \
            d_jac, d_rhs, d_out, seq_len, block_dim, batch_size);             \
    } else {                                                                  \
        /* Multi-block path */                                                \
        size_t nb_elems = (size_t)batch_size * block_dim;                     \
        size_t bnd_jac_bytes = (size_t)num_seq_blocks * nb_elems              \
                               * 4 * sizeof(SCALAR_T);                        \
        size_t bnd_rhs_bytes = (size_t)num_seq_blocks * nb_elems              \
                               * 2 * sizeof(SCALAR_T);                        \
        size_t accum_jac_bytes = (size_t)batch_size * seq_len * block_dim     \
                                 * 4 * sizeof(SCALAR_T);                      \
        size_t total_bytes = bnd_jac_bytes + bnd_rhs_bytes + accum_jac_bytes; \
                                                                              \
        char* temp = nullptr;                                                 \
        cudaMalloc((void**)&temp, total_bytes);                               \
        SCALAR_T* bnd_jac_d   = reinterpret_cast<SCALAR_T*>(temp);            \
        SCALAR_T* bnd_rhs_d   = reinterpret_cast<SCALAR_T*>(                  \
                                    temp + bnd_jac_bytes);                    \
        SCALAR_T* accum_jac_d = reinterpret_cast<SCALAR_T*>(                  \
                                    temp + bnd_jac_bytes + bnd_rhs_bytes);    \
                                                                              \
        /* Phase 1: Local reduction per block */                              \
        dim3 grid1(num_seq_blocks, block_dim, batch_size);                    \
        dim3 block1(THREADS_PER_BLOCK_B2);                                    \
        _pararnn_reduce_block2_local##SUFFIX<<<grid1, block1,                 \
                                               smem_size, s>>>(               \
            d_jac, d_rhs, d_out, accum_jac_d, bnd_jac_d, bnd_rhs_d,           \
            seq_len, block_dim, batch_size, block_stride_t);                  \
                                                                              \
        /* Phase 2: Boundary prefix scan */                                   \
        int bnd_threads = num_seq_blocks;                                     \
        if (bnd_threads > 512) bnd_threads = 512;                             \
        int bnd_warps = (bnd_threads + THREADS_PER_WARP - 1)                  \
                        / THREADS_PER_WARP;                                   \
        unsigned int bnd_smem = bnd_warps * 6 * sizeof(SCALAR_T);             \
        dim3 grid2(block_dim, batch_size);                                    \
        dim3 block2(bnd_threads);                                             \
        _pararnn_reduce_block2_boundary##SUFFIX<<<grid2, block2,              \
                                                  bnd_smem, s>>>(             \
            bnd_jac_d, bnd_rhs_d, num_seq_blocks, block_dim, batch_size);     \
                                                                              \
        /* Phase 3: Final correction (blocks > 0) */                          \
        dim3 grid3(num_seq_blocks - 1, block_dim, batch_size);                \
        dim3 block3(THREADS_PER_BLOCK_B2);                                    \
        _pararnn_reduce_block2_final##SUFFIX<<<grid3, block3, 0, s>>>(        \
            d_out, accum_jac_d, bnd_rhs_d,                                    \
            seq_len, block_dim, batch_size, block_stride_t);                  \
                                                                              \
        /* Free temp memory (stream-ordered) */                               \
        cudaFreeAsync(temp, s);                                               \
    }                                                                         \
}

// @BE pararnn_reduce_block2_f32
DEFINE_FFI_BLOCK2_REDUCE(_f32, float, 2)

// @BE pararnn_reduce_block2_f64
DEFINE_FFI_BLOCK2_REDUCE(_f64, double, 2)
