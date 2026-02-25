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
 * _fused_lstm_cifg_diag.cu -- Fused LSTM-CIFG Forward/Backward CUDA Kernels
 * ==========================================================================
 *
 * Fuses the entire Newton-parallel-reduction solve for LSTM-CIFG with
 * diagonal recurrence and peephole connections:
 *   - LSTM-CIFG cell evaluation
 *   - 2x2 block-diagonal Jacobian computation
 *   - Newton iteration (configurable max_its, omega_sor)
 *   - 5-step hierarchical parallel reduction with 2x2 monoid
 *
 * LSTM-CIFG equations (diagonal form):
 *   f     = sigma_f(A_f * h + Bxpb_f + C_f * c)
 *   c_new = sigma_c(A_c * h + Bxpb_c)
 *   c     = f * c + (1 - f) * c_new
 *   o     = sigma_o(A_o * h + Bxpb_o + C_o * c)
 *   h     = o * sigma_h(c)
 *
 * where Bxpb = B @ x + b is precomputed by Python.
 *
 * State is [c, h] concatenated. The Jacobian -d[c_new, h_new]/d[c_prev, h_prev]
 * is a 2x2 block-diagonal matrix per hidden dimension.
 *
 * Data layout:
 *   A:          (3, state_dim) contiguous  -- [A_f, A_o, A_c]
 *   Bxpb:       (batch_size, seq_len, 3, state_dim) contiguous -- [f, o, c]
 *   C:          (2, state_dim) contiguous  -- [C_f, C_o]
 *   full_state: (batch_size, seq_len, 2, state_dim) contiguous -- [c, h]
 *
 * Grid: (state_dim, batch_size), Block: (threads_per_block).
 * Each thread handles CHUNK_SIZE consecutive timesteps.
 *
 * TVM FFI entry points:
 *   fused_fwd_lstm_cifg_diag_f32(A, Bxpb, C, full_state, stream)
 *   fused_fwd_lstm_cifg_diag_f64(A, Bxpb, C, full_state, stream)
 *   fused_bwd_lstm_cifg_diag_f32(grad, full_state, A, Bxpb, C, dl_dh, stream)
 *   fused_bwd_lstm_cifg_diag_f64(grad, full_state, A, Bxpb, C, dl_dh, stream)
 */

#include <cuda_runtime.h>
#include <cstdint>
#include "brainevent/common.h"

#define THREADS_PER_WARP 32
#define THREADS_PER_BLOCK_LSTM 512

// ============================================================================
// Activation functions
// ============================================================================

template<typename T>
__device__ __forceinline__ T cuda_sigmoid(T x) {
    return (T)(1.0) / ((T)(1.0) + exp(-x));
}

template<typename T>
__device__ __forceinline__ T cuda_sigmoid_deriv(T x) {
    T s = cuda_sigmoid(x);
    return s * ((T)(1.0) - s);
}

template<typename T>
__device__ __forceinline__ T cuda_tanh_fn(T x) {
    return tanh(x);
}

template<typename T>
__device__ __forceinline__ T cuda_tanh_deriv(T x) {
    T t = tanh(x);
    return (T)(1.0) - t * t;
}

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
// Fused LSTM-CIFG Forward Kernel
// ============================================================================

#define DEFINE_FUSED_FWD_LSTM_CIFG(SUFFIX, SCALAR_T, CHUNK_SIZE)                \
                                                                                \
__global__ void __launch_bounds__(THREADS_PER_BLOCK_LSTM, 1)                    \
_fused_fwd_lstm_cifg_diag##SUFFIX(                                              \
    const SCALAR_T* __restrict__ A,                                             \
    const SCALAR_T* __restrict__ Bxpb,                                          \
    const SCALAR_T* __restrict__ C,                                             \
    SCALAR_T* __restrict__ full_state,                                          \
    int seq_len, int state_dim, int batch_size,                                 \
    int max_its, SCALAR_T omega                                                 \
) {                                                                             \
    const int h_idx = blockIdx.x;                                               \
    const int b_idx = blockIdx.y;                                               \
    if (h_idx >= state_dim || b_idx >= batch_size) return;                      \
                                                                                \
    const int tid = threadIdx.x;                                                \
    const int lane = tid & (THREADS_PER_WARP - 1);                              \
    const int warp_id = tid >> 5;                                               \
    const int warps_per_block = (THREADS_PER_BLOCK_LSTM + THREADS_PER_WARP - 1) \
                                / THREADS_PER_WARP;                             \
                                                                                \
    /* Load diagonal weights (shared across time) */                            \
    const SCALAR_T A_f = A[0 * state_dim + h_idx];                              \
    const SCALAR_T A_o = A[1 * state_dim + h_idx];                              \
    const SCALAR_T A_c = A[2 * state_dim + h_idx];                              \
    const SCALAR_T C_f = C[0 * state_dim + h_idx];                              \
    const SCALAR_T C_o = C[1 * state_dim + h_idx];                              \
                                                                                \
    /* Bxpb layout: (B, T, 3, state_dim) */                                     \
    const int bxpb_base = b_idx * seq_len * 3 * state_dim + h_idx;              \
    const int bxpb_stride = 3 * state_dim;                                      \
                                                                                \
    /* Load Bxpb values for this thread's chunk */                              \
    SCALAR_T bxpb_f[CHUNK_SIZE], bxpb_o[CHUNK_SIZE], bxpb_c[CHUNK_SIZE];        \
    const int t_start = tid * CHUNK_SIZE;                                       \
    int num_valid = 0;                                                          \
    for (int ci = 0; ci < CHUNK_SIZE; ci++) {                                   \
        int t = t_start + ci;                                                   \
        if (t < seq_len) {                                                      \
            int off = bxpb_base + t * bxpb_stride;                              \
            bxpb_f[ci] = Bxpb[off + 0 * state_dim];                             \
            bxpb_o[ci] = Bxpb[off + 1 * state_dim];                             \
            bxpb_c[ci] = Bxpb[off + 2 * state_dim];                             \
            num_valid++;                                                        \
        } else {                                                                \
            bxpb_f[ci] = (SCALAR_T)(0.0);                                       \
            bxpb_o[ci] = (SCALAR_T)(0.0);                                       \
            bxpb_c[ci] = (SCALAR_T)(0.0);                                       \
        }                                                                       \
    }                                                                           \
                                                                                \
    /* Initialize solution: LSTM step from zero state */                        \
    SCALAR_T sol_c[CHUNK_SIZE], sol_h[CHUNK_SIZE];                              \
    for (int ci = 0; ci < CHUNK_SIZE; ci++) {                                   \
        /* c_prev=0, h_prev=0 */                                                \
        SCALAR_T f = cuda_sigmoid<SCALAR_T>(bxpb_f[ci]);                        \
        SCALAR_T cn = cuda_tanh_fn<SCALAR_T>(bxpb_c[ci]);                       \
        SCALAR_T cc = ((SCALAR_T)(1.0) - f) * cn;                               \
        SCALAR_T o = cuda_sigmoid<SCALAR_T>(bxpb_o[ci] + C_o * cc);             \
        sol_c[ci] = cc;                                                         \
        sol_h[ci] = o * cuda_tanh_fn<SCALAR_T>(cc);                             \
    }                                                                           \
                                                                                \
    /* Shared memory: 6 scalars per warp (4 jac + 2 rhs during reduction;       \
       reused as 2 vals per warp for h_prev between Newton iterations) */       \
    extern __shared__ char _smem_bytes[];                                       \
    SCALAR_T* smem = reinterpret_cast<SCALAR_T*>(_smem_bytes);                  \
                                                                                \
    /* Newton iterations */                                                     \
    for (int nit = 0; nit < max_its; nit++) {                                   \
                                                                                \
        /* ---- Get (c_prev, h_prev) from shifted solution ---- */              \
        SCALAR_T cp_arr[CHUNK_SIZE], hp_arr[CHUNK_SIZE];                        \
        /* Shuffle to get previous thread's last (c, h) */                      \
        SCALAR_T prev_c = __shfl_up_sync(                                       \
            0xffffffff, sol_c[CHUNK_SIZE - 1], 1);                              \
        SCALAR_T prev_h = __shfl_up_sync(                                       \
            0xffffffff, sol_h[CHUNK_SIZE - 1], 1);                              \
        if (lane == 0) {                                                        \
            if (warp_id > 0) {                                                  \
                prev_c = smem[(warp_id - 1) * 6 + 0];                           \
                prev_h = smem[(warp_id - 1) * 6 + 1];                           \
            } else {                                                            \
                prev_c = (SCALAR_T)(0.0);                                       \
                prev_h = (SCALAR_T)(0.0);                                       \
            }                                                                   \
        }                                                                       \
        cp_arr[0] = prev_c;                                                     \
        hp_arr[0] = prev_h;                                                     \
        for (int ci = 1; ci < CHUNK_SIZE; ci++) {                               \
            cp_arr[ci] = sol_c[ci - 1];                                         \
            hp_arr[ci] = sol_h[ci - 1];                                         \
        }                                                                       \
        if (tid == 0) {                                                         \
            cp_arr[0] = (SCALAR_T)(0.0);                                        \
            hp_arr[0] = (SCALAR_T)(0.0);                                        \
        }                                                                       \
                                                                                \
        /* ---- Compute residuals and 2x2 Jacobians ---- */                     \
        SCALAR_T rj00[CHUNK_SIZE], rj01[CHUNK_SIZE];                            \
        SCALAR_T rj10[CHUNK_SIZE], rj11[CHUNK_SIZE];                            \
        SCALAR_T rr0[CHUNK_SIZE], rr1[CHUNK_SIZE];                              \
        for (int ci = 0; ci < CHUNK_SIZE; ci++) {                               \
            int t = t_start + ci;                                               \
            if (t < seq_len) {                                                  \
                SCALAR_T hp = hp_arr[ci], cp = cp_arr[ci];                      \
                SCALAR_T pre_f = A_f * hp + bxpb_f[ci] + C_f * cp;              \
                SCALAR_T pre_cc = A_c * hp + bxpb_c[ci];                        \
                SCALAR_T f = cuda_sigmoid<SCALAR_T>(pre_f);                     \
                SCALAR_T cn = cuda_tanh_fn<SCALAR_T>(pre_cc);                   \
                SCALAR_T cc = f * cp + ((SCALAR_T)(1.0) - f) * cn;              \
                SCALAR_T pre_o = A_o * hp + bxpb_o[ci] + C_o * cc;              \
                SCALAR_T o = cuda_sigmoid<SCALAR_T>(pre_o);                     \
                SCALAR_T scc = cuda_tanh_fn<SCALAR_T>(cc);                      \
                SCALAR_T hh = o * scc;                                          \
                                                                                \
                /* Negative residuals */                                        \
                rr0[ci] = -(sol_c[ci] - cc);                                    \
                rr1[ci] = -(sol_h[ci] - hh);                                    \
                                                                                \
                /* Derivatives */                                               \
                SCALAR_T df = cuda_sigmoid_deriv<SCALAR_T>(pre_f);              \
                SCALAR_T dc_val = cuda_tanh_deriv<SCALAR_T>(pre_cc);            \
                SCALAR_T do_val = cuda_sigmoid_deriv<SCALAR_T>(pre_o);          \
                SCALAR_T ds = cuda_tanh_deriv<SCALAR_T>(cc);                    \
                                                                                \
                SCALAR_T Jh_f = A_f * df;                                       \
                SCALAR_T Jh_o = A_o * do_val;                                   \
                SCALAR_T Jh_c = A_c * dc_val;                                   \
                SCALAR_T Jc_f = C_f * df;                                       \
                SCALAR_T Jc_o = C_o * do_val;                                   \
                SCALAR_T o_sdercc = o * ds;                                     \
                SCALAR_T diff_cp_cn = cp - cn;                                  \
                                                                                \
                /* 2x2 Jacobian: -d[c_new,h_new]/d[c_prev,h_prev] */            \
                SCALAR_T Jcc = -(Jc_f * diff_cp_cn + f);                        \
                SCALAR_T Jch = -(Jh_f * diff_cp_cn                              \
                    + ((SCALAR_T)(1.0) - f) * Jh_c);                            \
                SCALAR_T Jhc = Jcc * (Jc_o * scc + o_sdercc);                   \
                SCALAR_T Jhh = -(scc * (Jh_o - Jc_o * Jch)                      \
                    - o_sdercc * Jch);                                          \
                                                                                \
                rj00[ci] = Jcc;  rj01[ci] = Jch;                                \
                rj10[ci] = Jhc;  rj11[ci] = Jhh;                                \
            } else {                                                            \
                /* Neutral element: J = -I, rhs = 0 */                          \
                rj00[ci] = (SCALAR_T)(-1.0); rj01[ci] = (SCALAR_T)(0.0);        \
                rj10[ci] = (SCALAR_T)(0.0);  rj11[ci] = (SCALAR_T)(-1.0);       \
                rr0[ci] = (SCALAR_T)(0.0);   rr1[ci] = (SCALAR_T)(0.0);         \
            }                                                                   \
        }                                                                       \
        if (tid == 0) {                                                         \
            rj00[0] = (SCALAR_T)(0.0); rj01[0] = (SCALAR_T)(0.0);               \
            rj10[0] = (SCALAR_T)(0.0); rj11[0] = (SCALAR_T)(0.0);               \
        }                                                                       \
                                                                                \
        /* ============ 2x2 Parallel reduction (5-step) ============ */         \
                                                                                \
        /* Step 1: Thomas reduction within chunk */                             \
        for (int ci = 1; ci < CHUNK_SIZE; ci++) {                               \
            /* rhs[ci] -= J[ci] @ rhs[ci-1] */                                  \
            SCALAR_T mv0, mv1;                                                  \
            MAT2_VEC(rj00[ci],rj01[ci],rj10[ci],rj11[ci],                       \
                     rr0[ci-1],rr1[ci-1], mv0,mv1);                             \
            rr0[ci] -= mv0; rr1[ci] -= mv1;                                     \
            /* J[ci] = -(J[ci] @ J[ci-1]) */                                    \
            SCALAR_T m00,m01,m10,m11;                                           \
            MAT2_MUL(rj00[ci],rj01[ci],rj10[ci],rj11[ci],                       \
                     rj00[ci-1],rj01[ci-1],rj10[ci-1],rj11[ci-1],               \
                     m00,m01,m10,m11);                                          \
            rj00[ci] = -m00; rj01[ci] = -m01;                                   \
            rj10[ci] = -m10; rj11[ci] = -m11;                                   \
        }                                                                       \
                                                                                \
        /* Step 2: Warp-level PCR for last chunk element */                     \
        int lc = CHUNK_SIZE - 1;                                                \
        for (int d = 1; d < THREADS_PER_WARP; d <<= 1) {                        \
            SCALAR_T pj00 = __shfl_up_sync(0xffffffff, rj00[lc], d);            \
            SCALAR_T pj01 = __shfl_up_sync(0xffffffff, rj01[lc], d);            \
            SCALAR_T pj10 = __shfl_up_sync(0xffffffff, rj10[lc], d);            \
            SCALAR_T pj11 = __shfl_up_sync(0xffffffff, rj11[lc], d);            \
            SCALAR_T pr0  = __shfl_up_sync(0xffffffff, rr0[lc], d);             \
            SCALAR_T pr1  = __shfl_up_sync(0xffffffff, rr1[lc], d);             \
            if (lane >= d) {                                                    \
                SCALAR_T mv0, mv1;                                              \
                MAT2_VEC(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                   \
                         pr0,pr1, mv0,mv1);                                     \
                rr0[lc] -= mv0; rr1[lc] -= mv1;                                 \
                SCALAR_T m00,m01,m10,m11;                                       \
                MAT2_MUL(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                   \
                         pj00,pj01,pj10,pj11, m00,m01,m10,m11);                 \
                rj00[lc] = -m00; rj01[lc] = -m01;                               \
                rj10[lc] = -m10; rj11[lc] = -m11;                               \
            }                                                                   \
        }                                                                       \
                                                                                \
        /* Step 3: Block-level PCR via shared memory */                         \
        if (lane == (THREADS_PER_WARP - 1) && num_valid > 0) {                  \
            int off = warp_id * 6;                                              \
            smem[off+0] = rj00[lc]; smem[off+1] = rj01[lc];                     \
            smem[off+2] = rj10[lc]; smem[off+3] = rj11[lc];                     \
            smem[off+4] = rr0[lc];  smem[off+5] = rr1[lc];                      \
        }                                                                       \
        __syncthreads();                                                        \
        for (int d = 1; d < warps_per_block; d <<= 1) {                         \
            int pw = warp_id - d;                                               \
            if (lane == (THREADS_PER_WARP-1) && pw >= 0 && num_valid > 0) {     \
                int poff = pw * 6;                                              \
                SCALAR_T pj00_=smem[poff+0], pj01_=smem[poff+1];                \
                SCALAR_T pj10_=smem[poff+2], pj11_=smem[poff+3];                \
                SCALAR_T pr0_=smem[poff+4], pr1_=smem[poff+5];                  \
                SCALAR_T mv0, mv1;                                              \
                MAT2_VEC(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                   \
                         pr0_,pr1_, mv0,mv1);                                   \
                rr0[lc] -= mv0; rr1[lc] -= mv1;                                 \
                SCALAR_T m00,m01,m10,m11;                                       \
                MAT2_MUL(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                   \
                         pj00_,pj01_,pj10_,pj11_, m00,m01,m10,m11);             \
                rj00[lc] = -m00; rj01[lc] = -m01;                               \
                rj10[lc] = -m10; rj11[lc] = -m11;                               \
            }                                                                   \
            __syncthreads();                                                    \
            if (lane == (THREADS_PER_WARP - 1) && num_valid > 0) {              \
                int off = warp_id * 6;                                          \
                smem[off+0] = rj00[lc]; smem[off+1] = rj01[lc];                 \
                smem[off+2] = rj10[lc]; smem[off+3] = rj11[lc];                 \
                smem[off+4] = rr0[lc];  smem[off+5] = rr1[lc];                  \
            }                                                                   \
            __syncthreads();                                                    \
        }                                                                       \
                                                                                \
        /* Step 4: Warp-level forward substitution */                           \
        SCALAR_T sp_j00, sp_j01, sp_j10, sp_j11, sp_r0, sp_r1;                  \
        if (warp_id > 0 && lane != (THREADS_PER_WARP - 1)) {                    \
            int poff = (warp_id - 1) * 6;                                       \
            sp_j00 = smem[poff+0]; sp_j01 = smem[poff+1];                       \
            sp_j10 = smem[poff+2]; sp_j11 = smem[poff+3];                       \
            sp_r0  = smem[poff+4]; sp_r1  = smem[poff+5];                       \
        } else {                                                                \
            sp_j00 = (SCALAR_T)(-1.0); sp_j01 = (SCALAR_T)(0.0);                \
            sp_j10 = (SCALAR_T)(0.0);  sp_j11 = (SCALAR_T)(-1.0);               \
            sp_r0  = (SCALAR_T)(0.0);  sp_r1  = (SCALAR_T)(0.0);                \
        }                                                                       \
        {                                                                       \
            SCALAR_T mv0, mv1;                                                  \
            MAT2_VEC(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                       \
                     sp_r0,sp_r1, mv0,mv1);                                     \
            rr0[lc] -= mv0; rr1[lc] -= mv1;                                     \
            SCALAR_T m00,m01,m10,m11;                                           \
            MAT2_MUL(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                       \
                     sp_j00,sp_j01,sp_j10,sp_j11, m00,m01,m10,m11);             \
            rj00[lc] = -m00; rj01[lc] = -m01;                                   \
            rj10[lc] = -m10; rj11[lc] = -m11;                                   \
        }                                                                       \
        {                                                                       \
            SCALAR_T pr0 = __shfl_up_sync(0xffffffff, rr0[lc], 1);              \
            SCALAR_T pr1 = __shfl_up_sync(0xffffffff, rr1[lc], 1);              \
            SCALAR_T pj00_ = __shfl_up_sync(0xffffffff, rj00[lc], 1);           \
            SCALAR_T pj01_ = __shfl_up_sync(0xffffffff, rj01[lc], 1);           \
            SCALAR_T pj10_ = __shfl_up_sync(0xffffffff, rj10[lc], 1);           \
            SCALAR_T pj11_ = __shfl_up_sync(0xffffffff, rj11[lc], 1);           \
            if (lane > 0) {                                                     \
                sp_r0 = pr0; sp_r1 = pr1;                                       \
                sp_j00 = pj00_; sp_j01 = pj01_;                                 \
                sp_j10 = pj10_; sp_j11 = pj11_;                                 \
            }                                                                   \
        }                                                                       \
                                                                                \
        /* Step 5: Chunk-level forward substitution */                          \
        for (int ci = 0; ci < CHUNK_SIZE - 1; ci++) {                           \
            SCALAR_T mv0, mv1;                                                  \
            MAT2_VEC(rj00[ci],rj01[ci],rj10[ci],rj11[ci],                       \
                     sp_r0,sp_r1, mv0,mv1);                                     \
            rr0[ci] -= mv0; rr1[ci] -= mv1;                                     \
            SCALAR_T m00,m01,m10,m11;                                           \
            MAT2_MUL(rj00[ci],rj01[ci],rj10[ci],rj11[ci],                       \
                     sp_j00,sp_j01,sp_j10,sp_j11, m00,m01,m10,m11);             \
            rj00[ci] = -m00; rj01[ci] = -m01;                                   \
            rj10[ci] = -m10; rj11[ci] = -m11;                                   \
        }                                                                       \
                                                                                \
        /* Update solution: sol += omega * dh */                                \
        for (int ci = 0; ci < CHUNK_SIZE; ci++) {                               \
            sol_c[ci] += omega * rr0[ci];                                       \
            sol_h[ci] += omega * rr1[ci];                                       \
        }                                                                       \
                                                                                \
        /* Store last (c, h) for next iteration's h_prev lookup */              \
        if (lane == (THREADS_PER_WARP - 1)) {                                   \
            smem[warp_id * 6 + 0] = sol_c[CHUNK_SIZE - 1];                      \
            smem[warp_id * 6 + 1] = sol_h[CHUNK_SIZE - 1];                      \
        }                                                                       \
        __syncthreads();                                                        \
    }                                                                           \
                                                                                \
    /* Write full state [c, h] to output: (B, T, 2, state_dim) */               \
    const int fs_base = b_idx * seq_len * 2 * state_dim + h_idx;                \
    const int fs_stride = 2 * state_dim;                                        \
    for (int ci = 0; ci < CHUNK_SIZE; ci++) {                                   \
        int t = t_start + ci;                                                   \
        if (t < seq_len) {                                                      \
            int off = fs_base + t * fs_stride;                                  \
            full_state[off + 0 * state_dim] = sol_c[ci];                        \
            full_state[off + 1 * state_dim] = sol_h[ci];                        \
        }                                                                       \
    }                                                                           \
}

// Instantiate forward kernels
DEFINE_FUSED_FWD_LSTM_CIFG(_f32, float, 2)
DEFINE_FUSED_FWD_LSTM_CIFG(_f64, double, 1)

// ============================================================================
// Fused LSTM-CIFG Backward Kernel
// ============================================================================

#define DEFINE_FUSED_BWD_LSTM_CIFG(SUFFIX, SCALAR_T, CHUNK_SIZE)                \
                                                                                \
__global__ void __launch_bounds__(THREADS_PER_BLOCK_LSTM, 1)                    \
_fused_bwd_lstm_cifg_diag##SUFFIX(                                              \
    const SCALAR_T* __restrict__ grad_y,                                        \
    const SCALAR_T* __restrict__ full_state,                                    \
    const SCALAR_T* __restrict__ A,                                             \
    const SCALAR_T* __restrict__ Bxpb,                                          \
    const SCALAR_T* __restrict__ C,                                             \
    SCALAR_T* __restrict__ dl_dh_out,                                           \
    int seq_len, int state_dim, int batch_size                                  \
) {                                                                             \
    const int h_idx = blockIdx.x;                                               \
    const int b_idx = blockIdx.y;                                               \
    if (h_idx >= state_dim || b_idx >= batch_size) return;                      \
                                                                                \
    const int tid = threadIdx.x;                                                \
    const int lane = tid & (THREADS_PER_WARP - 1);                              \
    const int warp_id = tid >> 5;                                               \
    const int warps_per_block = (THREADS_PER_BLOCK_LSTM + THREADS_PER_WARP - 1) \
                                / THREADS_PER_WARP;                             \
                                                                                \
    const SCALAR_T A_f = A[0 * state_dim + h_idx];                              \
    const SCALAR_T A_o = A[1 * state_dim + h_idx];                              \
    const SCALAR_T A_c = A[2 * state_dim + h_idx];                              \
    const SCALAR_T C_f = C[0 * state_dim + h_idx];                              \
    const SCALAR_T C_o = C[1 * state_dim + h_idx];                              \
                                                                                \
    /* Base addresses for global memory */                                      \
    const int grad_base = b_idx * seq_len * state_dim + h_idx;                  \
    const int fs_base = b_idx * seq_len * 2 * state_dim + h_idx;                \
    const int bxpb_base = b_idx * seq_len * 3 * state_dim + h_idx;              \
    const int fs_stride = 2 * state_dim;                                        \
    const int bxpb_stride = 3 * state_dim;                                      \
                                                                                \
    const int t_start = tid * CHUNK_SIZE;                                       \
    int num_valid = 0;                                                          \
                                                                                \
    /* Load grad_y in reversed time; RHS = [0, grad_y[t_fwd]] */                \
    SCALAR_T rr0[CHUNK_SIZE], rr1[CHUNK_SIZE];                                  \
    for (int ci = 0; ci < CHUNK_SIZE; ci++) {                                   \
        int t_fwd = seq_len - 1 - (t_start + ci);                               \
        if (t_fwd >= 0 && t_fwd < seq_len) {                                    \
            rr0[ci] = (SCALAR_T)(0.0);                                          \
            rr1[ci] = grad_y[grad_base + t_fwd * state_dim];                    \
            num_valid++;                                                        \
        } else {                                                                \
            rr0[ci] = (SCALAR_T)(0.0);                                          \
            rr1[ci] = (SCALAR_T)(0.0);                                          \
        }                                                                       \
    }                                                                           \
                                                                                \
    /* Compute backward (transposed) 2x2 Jacobians.                             \
       At reversed time t_rev, use forward Jacobian from t_fwd = T-t_rev,       \
       then transpose (swap off-diagonal). */                                   \
    SCALAR_T rj00[CHUNK_SIZE], rj01[CHUNK_SIZE];                                \
    SCALAR_T rj10[CHUNK_SIZE], rj11[CHUNK_SIZE];                                \
    for (int ci = 0; ci < CHUNK_SIZE; ci++) {                                   \
        int t_rev = t_start + ci;                                               \
        int t_jac = seq_len - t_rev;  /* forward timestep for Jacobian */       \
        if (t_jac > 0 && t_jac < seq_len) {                                     \
            /* Load h_prev = state[t_jac-1] and cc = c[t_jac] */                \
            SCALAR_T cp_prev, hp_prev, cc;                                      \
            if (t_jac > 0) {                                                    \
                int fs_off = fs_base + (t_jac - 1) * fs_stride;                 \
                cp_prev = full_state[fs_off + 0 * state_dim];                   \
                hp_prev = full_state[fs_off + 1 * state_dim];                   \
            } else {                                                            \
                cp_prev = (SCALAR_T)(0.0);                                      \
                hp_prev = (SCALAR_T)(0.0);                                      \
            }                                                                   \
            {                                                                   \
                int fs_off = fs_base + t_jac * fs_stride;                       \
                cc = full_state[fs_off + 0 * state_dim];                        \
            }                                                                   \
                                                                                \
            /* Load Bxpb at t_jac */                                            \
            int boff = bxpb_base + t_jac * bxpb_stride;                         \
            SCALAR_T bxpb_fv = Bxpb[boff + 0 * state_dim];                      \
            SCALAR_T bxpb_ov = Bxpb[boff + 1 * state_dim];                      \
            SCALAR_T bxpb_cv = Bxpb[boff + 2 * state_dim];                      \
                                                                                \
            /* Recompute gate pre-activations */                                \
            SCALAR_T pre_f = A_f * hp_prev + bxpb_fv + C_f * cp_prev;           \
            SCALAR_T pre_cc = A_c * hp_prev + bxpb_cv;                          \
            SCALAR_T f = cuda_sigmoid<SCALAR_T>(pre_f);                         \
            SCALAR_T cn = cuda_tanh_fn<SCALAR_T>(pre_cc);                       \
            SCALAR_T pre_o = A_o * hp_prev + bxpb_ov + C_o * cc;                \
            SCALAR_T o = cuda_sigmoid<SCALAR_T>(pre_o);                         \
            SCALAR_T scc = cuda_tanh_fn<SCALAR_T>(cc);                          \
                                                                                \
            /* Derivatives */                                                   \
            SCALAR_T df = cuda_sigmoid_deriv<SCALAR_T>(pre_f);                  \
            SCALAR_T dc_val = cuda_tanh_deriv<SCALAR_T>(pre_cc);                \
            SCALAR_T do_val = cuda_sigmoid_deriv<SCALAR_T>(pre_o);              \
            SCALAR_T ds = cuda_tanh_deriv<SCALAR_T>(cc);                        \
                                                                                \
            SCALAR_T Jh_f = A_f * df;                                           \
            SCALAR_T Jh_o = A_o * do_val;                                       \
            SCALAR_T Jh_c = A_c * dc_val;                                       \
            SCALAR_T Jc_f = C_f * df;                                           \
            SCALAR_T Jc_o = C_o * do_val;                                       \
            SCALAR_T o_sdercc = o * ds;                                         \
            SCALAR_T diff_cp_cn = cp_prev - cn;                                 \
                                                                                \
            /* Forward Jacobian */                                              \
            SCALAR_T Jcc = -(Jc_f * diff_cp_cn + f);                            \
            SCALAR_T Jch = -(Jh_f * diff_cp_cn                                  \
                + ((SCALAR_T)(1.0) - f) * Jh_c);                                \
            SCALAR_T Jhc = Jcc * (Jc_o * scc + o_sdercc);                       \
            SCALAR_T Jhh = -(scc * (Jh_o - Jc_o * Jch)                          \
                - o_sdercc * Jch);                                              \
                                                                                \
            /* Transpose: swap off-diagonal */                                  \
            rj00[ci] = Jcc;  rj01[ci] = Jhc;                                    \
            rj10[ci] = Jch;  rj11[ci] = Jhh;                                    \
        } else {                                                                \
            rj00[ci] = (SCALAR_T)(0.0); rj01[ci] = (SCALAR_T)(0.0);             \
            rj10[ci] = (SCALAR_T)(0.0); rj11[ci] = (SCALAR_T)(0.0);             \
        }                                                                       \
    }                                                                           \
    if (tid == 0) {                                                             \
        rj00[0] = (SCALAR_T)(0.0); rj01[0] = (SCALAR_T)(0.0);                   \
        rj10[0] = (SCALAR_T)(0.0); rj11[0] = (SCALAR_T)(0.0);                   \
    }                                                                           \
                                                                                \
    /* ============ 2x2 Parallel reduction (5-step) ============ */             \
    extern __shared__ char _smem_bytes[];                                       \
    SCALAR_T* smem = reinterpret_cast<SCALAR_T*>(_smem_bytes);                  \
                                                                                \
    /* Step 1: Thomas */                                                        \
    for (int ci = 1; ci < CHUNK_SIZE; ci++) {                                   \
        SCALAR_T mv0, mv1;                                                      \
        MAT2_VEC(rj00[ci],rj01[ci],rj10[ci],rj11[ci],                           \
                 rr0[ci-1],rr1[ci-1], mv0,mv1);                                 \
        rr0[ci] -= mv0; rr1[ci] -= mv1;                                         \
        SCALAR_T m00,m01,m10,m11;                                               \
        MAT2_MUL(rj00[ci],rj01[ci],rj10[ci],rj11[ci],                           \
                 rj00[ci-1],rj01[ci-1],rj10[ci-1],rj11[ci-1],                   \
                 m00,m01,m10,m11);                                              \
        rj00[ci] = -m00; rj01[ci] = -m01;                                       \
        rj10[ci] = -m10; rj11[ci] = -m11;                                       \
    }                                                                           \
                                                                                \
    /* Step 2: Warp PCR */                                                      \
    int lc = CHUNK_SIZE - 1;                                                    \
    for (int d = 1; d < THREADS_PER_WARP; d <<= 1) {                            \
        SCALAR_T pj00 = __shfl_up_sync(0xffffffff, rj00[lc], d);                \
        SCALAR_T pj01 = __shfl_up_sync(0xffffffff, rj01[lc], d);                \
        SCALAR_T pj10 = __shfl_up_sync(0xffffffff, rj10[lc], d);                \
        SCALAR_T pj11 = __shfl_up_sync(0xffffffff, rj11[lc], d);                \
        SCALAR_T pr0  = __shfl_up_sync(0xffffffff, rr0[lc], d);                 \
        SCALAR_T pr1  = __shfl_up_sync(0xffffffff, rr1[lc], d);                 \
        if (lane >= d) {                                                        \
            SCALAR_T mv0, mv1;                                                  \
            MAT2_VEC(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                       \
                     pr0,pr1, mv0,mv1);                                         \
            rr0[lc] -= mv0; rr1[lc] -= mv1;                                     \
            SCALAR_T m00,m01,m10,m11;                                           \
            MAT2_MUL(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                       \
                     pj00,pj01,pj10,pj11, m00,m01,m10,m11);                     \
            rj00[lc] = -m00; rj01[lc] = -m01;                                   \
            rj10[lc] = -m10; rj11[lc] = -m11;                                   \
        }                                                                       \
    }                                                                           \
                                                                                \
    /* Step 3: Block PCR */                                                     \
    if (lane == (THREADS_PER_WARP - 1) && num_valid > 0) {                      \
        int off = warp_id * 6;                                                  \
        smem[off+0] = rj00[lc]; smem[off+1] = rj01[lc];                         \
        smem[off+2] = rj10[lc]; smem[off+3] = rj11[lc];                         \
        smem[off+4] = rr0[lc];  smem[off+5] = rr1[lc];                          \
    }                                                                           \
    __syncthreads();                                                            \
    for (int d = 1; d < warps_per_block; d <<= 1) {                             \
        int pw = warp_id - d;                                                   \
        if (lane == (THREADS_PER_WARP-1) && pw >= 0 && num_valid > 0) {         \
            int poff = pw * 6;                                                  \
            SCALAR_T pj00_=smem[poff+0], pj01_=smem[poff+1];                    \
            SCALAR_T pj10_=smem[poff+2], pj11_=smem[poff+3];                    \
            SCALAR_T pr0_=smem[poff+4], pr1_=smem[poff+5];                      \
            SCALAR_T mv0, mv1;                                                  \
            MAT2_VEC(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                       \
                     pr0_,pr1_, mv0,mv1);                                       \
            rr0[lc] -= mv0; rr1[lc] -= mv1;                                     \
            SCALAR_T m00,m01,m10,m11;                                           \
            MAT2_MUL(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                       \
                     pj00_,pj01_,pj10_,pj11_, m00,m01,m10,m11);                 \
            rj00[lc] = -m00; rj01[lc] = -m01;                                   \
            rj10[lc] = -m10; rj11[lc] = -m11;                                   \
        }                                                                       \
        __syncthreads();                                                        \
        if (lane == (THREADS_PER_WARP - 1) && num_valid > 0) {                  \
            int off = warp_id * 6;                                              \
            smem[off+0] = rj00[lc]; smem[off+1] = rj01[lc];                     \
            smem[off+2] = rj10[lc]; smem[off+3] = rj11[lc];                     \
            smem[off+4] = rr0[lc];  smem[off+5] = rr1[lc];                      \
        }                                                                       \
        __syncthreads();                                                        \
    }                                                                           \
                                                                                \
    /* Step 4: Warp fwd subst */                                                \
    SCALAR_T sp_j00, sp_j01, sp_j10, sp_j11, sp_r0, sp_r1;                      \
    if (warp_id > 0 && lane != (THREADS_PER_WARP - 1)) {                        \
        int poff = (warp_id - 1) * 6;                                           \
        sp_j00 = smem[poff+0]; sp_j01 = smem[poff+1];                           \
        sp_j10 = smem[poff+2]; sp_j11 = smem[poff+3];                           \
        sp_r0  = smem[poff+4]; sp_r1  = smem[poff+5];                           \
    } else {                                                                    \
        sp_j00 = (SCALAR_T)(-1.0); sp_j01 = (SCALAR_T)(0.0);                    \
        sp_j10 = (SCALAR_T)(0.0);  sp_j11 = (SCALAR_T)(-1.0);                   \
        sp_r0  = (SCALAR_T)(0.0);  sp_r1  = (SCALAR_T)(0.0);                    \
    }                                                                           \
    {                                                                           \
        SCALAR_T mv0, mv1;                                                      \
        MAT2_VEC(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                           \
                 sp_r0,sp_r1, mv0,mv1);                                         \
        rr0[lc] -= mv0; rr1[lc] -= mv1;                                         \
        SCALAR_T m00,m01,m10,m11;                                               \
        MAT2_MUL(rj00[lc],rj01[lc],rj10[lc],rj11[lc],                           \
                 sp_j00,sp_j01,sp_j10,sp_j11, m00,m01,m10,m11);                 \
        rj00[lc] = -m00; rj01[lc] = -m01;                                       \
        rj10[lc] = -m10; rj11[lc] = -m11;                                       \
    }                                                                           \
    {                                                                           \
        SCALAR_T pr0 = __shfl_up_sync(0xffffffff, rr0[lc], 1);                  \
        SCALAR_T pr1 = __shfl_up_sync(0xffffffff, rr1[lc], 1);                  \
        SCALAR_T pj00_ = __shfl_up_sync(0xffffffff, rj00[lc], 1);               \
        SCALAR_T pj01_ = __shfl_up_sync(0xffffffff, rj01[lc], 1);               \
        SCALAR_T pj10_ = __shfl_up_sync(0xffffffff, rj10[lc], 1);               \
        SCALAR_T pj11_ = __shfl_up_sync(0xffffffff, rj11[lc], 1);               \
        if (lane > 0) {                                                         \
            sp_r0 = pr0; sp_r1 = pr1;                                           \
            sp_j00 = pj00_; sp_j01 = pj01_;                                     \
            sp_j10 = pj10_; sp_j11 = pj11_;                                     \
        }                                                                       \
    }                                                                           \
                                                                                \
    /* Step 5: Chunk fwd subst */                                               \
    for (int ci = 0; ci < CHUNK_SIZE - 1; ci++) {                               \
        SCALAR_T mv0, mv1;                                                      \
        MAT2_VEC(rj00[ci],rj01[ci],rj10[ci],rj11[ci],                           \
                 sp_r0,sp_r1, mv0,mv1);                                         \
        rr0[ci] -= mv0; rr1[ci] -= mv1;                                         \
        SCALAR_T m00,m01,m10,m11;                                               \
        MAT2_MUL(rj00[ci],rj01[ci],rj10[ci],rj11[ci],                           \
                 sp_j00,sp_j01,sp_j10,sp_j11, m00,m01,m10,m11);                 \
        rj00[ci] = -m00; rj01[ci] = -m01;                                       \
        rj10[ci] = -m10; rj11[ci] = -m11;                                       \
    }                                                                           \
                                                                                \
    /* Write dl_dh in forward order: (B, T, 2, state_dim) */                    \
    const int out_base = b_idx * seq_len * 2 * state_dim + h_idx;               \
    const int out_stride = 2 * state_dim;                                       \
    for (int ci = 0; ci < CHUNK_SIZE; ci++) {                                   \
        int t_fwd = seq_len - 1 - (t_start + ci);                               \
        if (t_fwd >= 0 && t_fwd < seq_len) {                                    \
            int off = out_base + t_fwd * out_stride;                            \
            dl_dh_out[off + 0 * state_dim] = rr0[ci];                           \
            dl_dh_out[off + 1 * state_dim] = rr1[ci];                           \
        }                                                                       \
    }                                                                           \
}

// Instantiate backward kernels
DEFINE_FUSED_BWD_LSTM_CIFG(_f32, float, 2)
DEFINE_FUSED_BWD_LSTM_CIFG(_f64, double, 1)

// ============================================================================
// TVM FFI entry points
// ============================================================================

// @BE fused_fwd_lstm_cifg_diag_f32
void fused_fwd_lstm_cifg_diag_f32(
    const BE::Tensor A_tv,
    const BE::Tensor Bxpb_tv,
    const BE::Tensor C_tv,
    const BE::Tensor full_state_tv,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int state_dim  = static_cast<int>(A_tv.size(1));
    int batch_size = static_cast<int>(Bxpb_tv.size(0));
    int seq_len    = static_cast<int>(Bxpb_tv.size(1));

    int max_its = 3;
    float omega = 1.0f;

    const float* d_A    = static_cast<const float*>(A_tv.data_ptr());
    const float* d_Bxpb = static_cast<const float*>(Bxpb_tv.data_ptr());
    const float* d_C    = static_cast<const float*>(C_tv.data_ptr());
    float* d_fs         = static_cast<float*>(full_state_tv.data_ptr());

    int warps_per_block = (THREADS_PER_BLOCK_LSTM + THREADS_PER_WARP - 1)
                          / THREADS_PER_WARP;
    unsigned int smem_size = warps_per_block * 6 * sizeof(float);

    dim3 grid(state_dim, batch_size);
    dim3 block(THREADS_PER_BLOCK_LSTM);
    _fused_fwd_lstm_cifg_diag_f32<<<grid, block, smem_size, s>>>(
        d_A, d_Bxpb, d_C, d_fs, seq_len, state_dim, batch_size,
        max_its, omega);
}

// @BE fused_fwd_lstm_cifg_diag_f64
void fused_fwd_lstm_cifg_diag_f64(
    const BE::Tensor A_tv,
    const BE::Tensor Bxpb_tv,
    const BE::Tensor C_tv,
    const BE::Tensor full_state_tv,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int state_dim  = static_cast<int>(A_tv.size(1));
    int batch_size = static_cast<int>(Bxpb_tv.size(0));
    int seq_len    = static_cast<int>(Bxpb_tv.size(1));

    int max_its = 3;
    double omega = 1.0;

    const double* d_A    = static_cast<const double*>(A_tv.data_ptr());
    const double* d_Bxpb = static_cast<const double*>(Bxpb_tv.data_ptr());
    const double* d_C    = static_cast<const double*>(C_tv.data_ptr());
    double* d_fs         = static_cast<double*>(full_state_tv.data_ptr());

    int warps_per_block = (THREADS_PER_BLOCK_LSTM + THREADS_PER_WARP - 1)
                          / THREADS_PER_WARP;
    unsigned int smem_size = warps_per_block * 6 * sizeof(double);

    dim3 grid(state_dim, batch_size);
    dim3 block(THREADS_PER_BLOCK_LSTM);
    _fused_fwd_lstm_cifg_diag_f64<<<grid, block, smem_size, s>>>(
        d_A, d_Bxpb, d_C, d_fs, seq_len, state_dim, batch_size,
        max_its, omega);
}

// @BE fused_bwd_lstm_cifg_diag_f32
void fused_bwd_lstm_cifg_diag_f32(
    const BE::Tensor grad_tv,
    const BE::Tensor full_state_tv,
    const BE::Tensor A_tv,
    const BE::Tensor Bxpb_tv,
    const BE::Tensor C_tv,
    const BE::Tensor dl_dh_tv,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int state_dim  = static_cast<int>(A_tv.size(1));
    int batch_size = static_cast<int>(grad_tv.size(0));
    int seq_len    = static_cast<int>(grad_tv.size(1));

    int warps_per_block = (THREADS_PER_BLOCK_LSTM + THREADS_PER_WARP - 1)
                          / THREADS_PER_WARP;
    unsigned int smem_size = warps_per_block * 6 * sizeof(float);

    dim3 grid(state_dim, batch_size);
    dim3 block(THREADS_PER_BLOCK_LSTM);
    _fused_bwd_lstm_cifg_diag_f32<<<grid, block, smem_size, s>>>(
        static_cast<const float*>(grad_tv.data_ptr()),
        static_cast<const float*>(full_state_tv.data_ptr()),
        static_cast<const float*>(A_tv.data_ptr()),
        static_cast<const float*>(Bxpb_tv.data_ptr()),
        static_cast<const float*>(C_tv.data_ptr()),
        static_cast<float*>(dl_dh_tv.data_ptr()),
        seq_len, state_dim, batch_size);
}

// @BE fused_bwd_lstm_cifg_diag_f64
void fused_bwd_lstm_cifg_diag_f64(
    const BE::Tensor grad_tv,
    const BE::Tensor full_state_tv,
    const BE::Tensor A_tv,
    const BE::Tensor Bxpb_tv,
    const BE::Tensor C_tv,
    const BE::Tensor dl_dh_tv,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int state_dim  = static_cast<int>(A_tv.size(1));
    int batch_size = static_cast<int>(grad_tv.size(0));
    int seq_len    = static_cast<int>(grad_tv.size(1));

    int warps_per_block = (THREADS_PER_BLOCK_LSTM + THREADS_PER_WARP - 1)
                          / THREADS_PER_WARP;
    unsigned int smem_size = warps_per_block * 6 * sizeof(double);

    dim3 grid(state_dim, batch_size);
    dim3 block(THREADS_PER_BLOCK_LSTM);
    _fused_bwd_lstm_cifg_diag_f64<<<grid, block, smem_size, s>>>(
        static_cast<const double*>(grad_tv.data_ptr()),
        static_cast<const double*>(full_state_tv.data_ptr()),
        static_cast<const double*>(A_tv.data_ptr()),
        static_cast<const double*>(Bxpb_tv.data_ptr()),
        static_cast<const double*>(C_tv.data_ptr()),
        static_cast<double*>(dl_dh_tv.data_ptr()),
        seq_len, state_dim, batch_size);
}
