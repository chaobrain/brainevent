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
 * Monoid: (J_b, r_b) ⊗ (J_a, r_a) = (J_b @ J_a, J_b @ r_a + r_b)
 *   where @ is 2x2 matrix multiplication.
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

#define THREADS_PER_WARP 32
#define THREADS_PER_BLOCK_B2 512

// ============================================================================
// 2x2 matrix/vector operations (register-level)
// ============================================================================

// mat2: stored as [a00, a01, a10, a11] in registers
// vec2: stored as [v0, v1] in registers

// 2x2 matmul: C = A @ B
#define MAT2_MUL(a00,a01,a10,a11, b00,b01,b10,b11, c00,c01,c10,c11) \
    c00 = a00*b00 + a01*b10;                                         \
    c01 = a00*b01 + a01*b11;                                         \
    c10 = a10*b00 + a11*b10;                                         \
    c11 = a10*b01 + a11*b11;

// 2x2 matvec: c = A @ b
#define MAT2_VEC(a00,a01,a10,a11, b0,b1, c0,c1)                     \
    c0 = a00*b0 + a01*b1;                                            \
    c1 = a10*b0 + a11*b1;

// ============================================================================
// Block-diagonal 2x2 reduction kernel
// ============================================================================

#define DEFINE_BLOCK2_REDUCE(SUFFIX, SCALAR_T, CHUNK_SIZE)                     \
                                                                               \
__global__ void __launch_bounds__(THREADS_PER_BLOCK_B2, 1)                     \
_pararnn_reduce_block2_single##SUFFIX(                                         \
    const SCALAR_T* __restrict__ jac_in,                                       \
    const SCALAR_T* __restrict__ rhs_in,                                       \
    SCALAR_T* __restrict__ output,                                             \
    int seq_len, int block_dim, int batch_size                                 \
) {                                                                            \
    /* blockIdx.x = block_dim idx, blockIdx.y = batch idx */                   \
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
    /* jac layout: (B, T, N, 2, 2) -> stride per T step is N*4 scalars */      \
    /* rhs layout: (B, T, N, 2) -> stride per T step is N*2 scalars */         \
    const int jac_base = b_idx * seq_len * block_dim * 4 + n_idx * 4;          \
    const int rhs_base = b_idx * seq_len * block_dim * 2 + n_idx * 2;          \
    const int jac_stride = block_dim * 4;                                      \
    const int rhs_stride = block_dim * 2;                                      \
                                                                               \
    /* Load chunks: each chunk has a 2x2 jac and 2-vec rhs */                  \
    SCALAR_T rj00[CHUNK_SIZE], rj01[CHUNK_SIZE];                               \
    SCALAR_T rj10[CHUNK_SIZE], rj11[CHUNK_SIZE];                               \
    SCALAR_T rr0[CHUNK_SIZE], rr1[CHUNK_SIZE];                                 \
    const int t_start = tid * CHUNK_SIZE;                                      \
    int num_valid = 0;                                                         \
    for (int c = 0; c < CHUNK_SIZE; c++) {                                     \
        int t = t_start + c;                                                   \
        if (t < seq_len) {                                                     \
            int jo = jac_base + t * jac_stride;                                \
            rj00[c] = jac_in[jo + 0]; rj01[c] = jac_in[jo + 1];              \
            rj10[c] = jac_in[jo + 2]; rj11[c] = jac_in[jo + 3];              \
            int ro = rhs_base + t * rhs_stride;                                \
            rr0[c] = rhs_in[ro + 0]; rr1[c] = rhs_in[ro + 1];                \
            num_valid++;                                                       \
        } else {                                                               \
            /* Neutral: J = -I, rhs = 0 */                                     \
            rj00[c] = (SCALAR_T)(-1.0); rj01[c] = (SCALAR_T)(0.0);           \
            rj10[c] = (SCALAR_T)(0.0);  rj11[c] = (SCALAR_T)(-1.0);          \
            rr0[c] = (SCALAR_T)(0.0);   rr1[c] = (SCALAR_T)(0.0);            \
        }                                                                      \
    }                                                                          \
    /* Zero first Jacobian */                                                  \
    if (tid == 0) {                                                            \
        rj00[0] = (SCALAR_T)(0.0); rj01[0] = (SCALAR_T)(0.0);                \
        rj10[0] = (SCALAR_T)(0.0); rj11[0] = (SCALAR_T)(0.0);                \
    }                                                                          \
                                                                               \
    /* ============================================================== */       \
    /* Step 1: Thomas reduction within chunk                          */       \
    /* reduceEqs: rhs -= J @ rhsPrev; J = -(J @ Jprev)               */       \
    /* ============================================================== */       \
    for (int c = 1; c < CHUNK_SIZE; c++) {                                     \
        /* rhs[c] -= J[c] @ rhs[c-1] */                                       \
        SCALAR_T mv0, mv1;                                                     \
        MAT2_VEC(rj00[c],rj01[c],rj10[c],rj11[c],                            \
                 rr0[c-1],rr1[c-1], mv0,mv1);                                 \
        rr0[c] -= mv0; rr1[c] -= mv1;                                         \
        /* J[c] = -(J[c] @ J[c-1]) */                                         \
        SCALAR_T m00,m01,m10,m11;                                              \
        MAT2_MUL(rj00[c],rj01[c],rj10[c],rj11[c],                            \
                 rj00[c-1],rj01[c-1],rj10[c-1],rj11[c-1],                    \
                 m00,m01,m10,m11);                                             \
        rj00[c] = -m00; rj01[c] = -m01;                                       \
        rj10[c] = -m10; rj11[c] = -m11;                                       \
    }                                                                          \
                                                                               \
    /* ============================================================== */       \
    /* Step 2: Warp-level PCR for last element of each chunk          */       \
    /* ============================================================== */       \
    int lc = CHUNK_SIZE - 1;                                                   \
    for (int d = 1; d < THREADS_PER_WARP; d <<= 1) {                          \
        /* Shuffle previous thread's last element */                           \
        SCALAR_T pj00 = __shfl_up_sync(0xffffffff, rj00[lc], d);              \
        SCALAR_T pj01 = __shfl_up_sync(0xffffffff, rj01[lc], d);              \
        SCALAR_T pj10 = __shfl_up_sync(0xffffffff, rj10[lc], d);              \
        SCALAR_T pj11 = __shfl_up_sync(0xffffffff, rj11[lc], d);              \
        SCALAR_T pr0  = __shfl_up_sync(0xffffffff, rr0[lc], d);               \
        SCALAR_T pr1  = __shfl_up_sync(0xffffffff, rr1[lc], d);               \
        if (lane >= d) {                                                       \
            /* rhs -= J @ prev_rhs */                                          \
            SCALAR_T mv0, mv1;                                                 \
            MAT2_VEC(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                    \
                     pr0,pr1, mv0,mv1);                                        \
            rr0[lc] -= mv0; rr1[lc] -= mv1;                                   \
            /* J = -(J @ prev_J) */                                            \
            SCALAR_T m00,m01,m10,m11;                                          \
            MAT2_MUL(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                    \
                     pj00,pj01,pj10,pj11, m00,m01,m10,m11);                   \
            rj00[lc] = -m00; rj01[lc] = -m01;                                 \
            rj10[lc] = -m10; rj11[lc] = -m11;                                 \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* ============================================================== */       \
    /* Step 3: Block-level PCR via shared memory                      */       \
    /* Each warp's last thread writes to shared, then PCR across      */       \
    /* ============================================================== */       \
    extern __shared__ char _smem_bytes[];                                       \
    /* shared: warps_per_block * (4 jac scalars + 2 rhs scalars) */            \
    SCALAR_T* smem = reinterpret_cast<SCALAR_T*>(_smem_bytes);                 \
    /* Layout: [jac00, jac01, jac10, jac11, rhs0, rhs1] per warp */            \
    /* smem[warp_id * 6 + 0..5] */                                             \
    if (lane == (THREADS_PER_WARP - 1) && num_valid > 0) {                     \
        int off = warp_id * 6;                                                 \
        smem[off+0] = rj00[lc]; smem[off+1] = rj01[lc];                       \
        smem[off+2] = rj10[lc]; smem[off+3] = rj11[lc];                       \
        smem[off+4] = rr0[lc];  smem[off+5] = rr1[lc];                        \
    }                                                                          \
    __syncthreads();                                                           \
                                                                               \
    for (int d = 1; d < warps_per_block; d <<= 1) {                           \
        int pw = warp_id - d;                                                  \
        if (lane == (THREADS_PER_WARP - 1) && pw >= 0 && num_valid > 0) {      \
            int poff = pw * 6;                                                 \
            SCALAR_T pj00_ = smem[poff+0], pj01_ = smem[poff+1];              \
            SCALAR_T pj10_ = smem[poff+2], pj11_ = smem[poff+3];              \
            SCALAR_T pr0_ = smem[poff+4], pr1_ = smem[poff+5];                \
            /* rhs -= J @ prev_rhs */                                          \
            SCALAR_T mv0, mv1;                                                 \
            MAT2_VEC(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                    \
                     pr0_,pr1_, mv0,mv1);                                      \
            rr0[lc] -= mv0; rr1[lc] -= mv1;                                   \
            /* J = -(J @ prev_J) */                                            \
            SCALAR_T m00,m01,m10,m11;                                          \
            MAT2_MUL(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                    \
                     pj00_,pj01_,pj10_,pj11_, m00,m01,m10,m11);               \
            rj00[lc] = -m00; rj01[lc] = -m01;                                 \
            rj10[lc] = -m10; rj11[lc] = -m11;                                 \
        }                                                                      \
        __syncthreads();                                                       \
        if (lane == (THREADS_PER_WARP - 1) && num_valid > 0) {                 \
            int off = warp_id * 6;                                             \
            smem[off+0] = rj00[lc]; smem[off+1] = rj01[lc];                   \
            smem[off+2] = rj10[lc]; smem[off+3] = rj11[lc];                   \
            smem[off+4] = rr0[lc];  smem[off+5] = rr1[lc];                    \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
                                                                               \
    /* ============================================================== */       \
    /* Step 4: Warp-level forward substitution                        */       \
    /* ============================================================== */       \
    SCALAR_T sp_j00, sp_j01, sp_j10, sp_j11, sp_r0, sp_r1;                    \
    if (warp_id > 0 && lane != (THREADS_PER_WARP - 1)) {                       \
        int poff = (warp_id - 1) * 6;                                         \
        sp_j00 = smem[poff+0]; sp_j01 = smem[poff+1];                         \
        sp_j10 = smem[poff+2]; sp_j11 = smem[poff+3];                         \
        sp_r0  = smem[poff+4]; sp_r1  = smem[poff+5];                         \
    } else {                                                                   \
        sp_j00 = (SCALAR_T)(-1.0); sp_j01 = (SCALAR_T)(0.0);                 \
        sp_j10 = (SCALAR_T)(0.0);  sp_j11 = (SCALAR_T)(-1.0);                \
        sp_r0  = (SCALAR_T)(0.0);  sp_r1  = (SCALAR_T)(0.0);                 \
    }                                                                          \
    /* Reduce last chunk element using previous warp solution */                \
    {                                                                          \
        SCALAR_T mv0, mv1;                                                     \
        MAT2_VEC(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                        \
                 sp_r0,sp_r1, mv0,mv1);                                        \
        rr0[lc] -= mv0; rr1[lc] -= mv1;                                       \
        SCALAR_T m00,m01,m10,m11;                                              \
        MAT2_MUL(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                        \
                 sp_j00,sp_j01,sp_j10,sp_j11, m00,m01,m10,m11);               \
        rj00[lc] = -m00; rj01[lc] = -m01;                                     \
        rj10[lc] = -m10; rj11[lc] = -m11;                                     \
    }                                                                          \
    /* Shuffle to get previous thread's last solution */                        \
    {                                                                          \
        SCALAR_T pr0 = __shfl_up_sync(0xffffffff, rr0[lc], 1);                \
        SCALAR_T pr1 = __shfl_up_sync(0xffffffff, rr1[lc], 1);                \
        SCALAR_T pj00_ = __shfl_up_sync(0xffffffff, rj00[lc], 1);             \
        SCALAR_T pj01_ = __shfl_up_sync(0xffffffff, rj01[lc], 1);             \
        SCALAR_T pj10_ = __shfl_up_sync(0xffffffff, rj10[lc], 1);             \
        SCALAR_T pj11_ = __shfl_up_sync(0xffffffff, rj11[lc], 1);             \
        if (lane > 0) {                                                        \
            sp_r0 = pr0; sp_r1 = pr1;                                          \
            sp_j00 = pj00_; sp_j01 = pj01_;                                   \
            sp_j10 = pj10_; sp_j11 = pj11_;                                   \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* ============================================================== */       \
    /* Step 5: Chunk-level forward substitution                       */       \
    /* ============================================================== */       \
    for (int c = 0; c < CHUNK_SIZE - 1; c++) {                                 \
        SCALAR_T mv0, mv1;                                                     \
        MAT2_VEC(rj00[c],rj01[c],rj10[c],rj11[c],                            \
                 sp_r0,sp_r1, mv0,mv1);                                        \
        rr0[c] -= mv0; rr1[c] -= mv1;                                         \
        SCALAR_T m00,m01,m10,m11;                                              \
        MAT2_MUL(rj00[c],rj01[c],rj10[c],rj11[c],                            \
                 sp_j00,sp_j01,sp_j10,sp_j11, m00,m01,m10,m11);               \
        rj00[c] = -m00; rj01[c] = -m01;                                       \
        rj10[c] = -m10; rj11[c] = -m11;                                       \
    }                                                                          \
                                                                               \
    /* Write output */                                                         \
    for (int c = 0; c < CHUNK_SIZE; c++) {                                     \
        int t = t_start + c;                                                   \
        if (t < seq_len) {                                                     \
            int ro = rhs_base + t * rhs_stride;                                \
            output[ro + 0] = rr0[c]; output[ro + 1] = rr1[c];                 \
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

#define DEFINE_FFI_BLOCK2_REDUCE(SUFFIX, SCALAR_T, CHUNK_SIZE)                 \
void pararnn_reduce_block2##SUFFIX(                                            \
    tvm::ffi::TensorView jac_tv,                                               \
    tvm::ffi::TensorView rhs_tv,                                               \
    tvm::ffi::TensorView output_tv,                                            \
    int64_t stream                                                             \
) {                                                                            \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                   \
    /* jac: (B, T, N, 2, 2) -> B=dim0, T=dim1, N=dim2 */                      \
    int batch_size = static_cast<int>(jac_tv.size(0));                         \
    int seq_len    = static_cast<int>(jac_tv.size(1));                         \
    int block_dim  = static_cast<int>(jac_tv.size(2));                         \
                                                                               \
    const SCALAR_T* d_jac = static_cast<const SCALAR_T*>(jac_tv.data_ptr());   \
    const SCALAR_T* d_rhs = static_cast<const SCALAR_T*>(rhs_tv.data_ptr());   \
    SCALAR_T* d_out = static_cast<SCALAR_T*>(output_tv.data_ptr());            \
                                                                               \
    int warps_per_block = (THREADS_PER_BLOCK_B2 + THREADS_PER_WARP - 1)        \
                          / THREADS_PER_WARP;                                  \
    unsigned int smem_size = warps_per_block * 6 * sizeof(SCALAR_T);           \
                                                                               \
    dim3 grid(block_dim, batch_size);                                          \
    dim3 block(THREADS_PER_BLOCK_B2);                                          \
    _pararnn_reduce_block2_single##SUFFIX<<<grid, block, smem_size, s>>>(      \
        d_jac, d_rhs, d_out, seq_len, block_dim, batch_size);                  \
}

// @tvm_ffi pararnn_reduce_block2_f32
DEFINE_FFI_BLOCK2_REDUCE(_f32, float, 2)

// @tvm_ffi pararnn_reduce_block2_f64
DEFINE_FFI_BLOCK2_REDUCE(_f64, double, 2)
