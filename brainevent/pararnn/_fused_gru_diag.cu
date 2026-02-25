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
 * _fused_gru_diag.cu -- Fused GRU Forward/Backward CUDA Kernels
 * ==============================================================
 *
 * Fuses the entire Newton-parallel-reduction solve for a diagonal GRU:
 *   - GRU cell evaluation (gates z, r, candidate h_new)
 *   - Jacobian computation
 *   - Newton iteration (configurable max_its, omega_sor)
 *   - 5-step hierarchical parallel reduction
 *
 * All within a single kernel launch, avoiding intermediate global memory
 * round-trips between Newton iterations.
 *
 * GRU equations (diagonal form):
 *   z     = sigmoid(A[0]*h + Bxpb[0])
 *   r     = sigmoid(A[1]*h + Bxpb[1])
 *   h_new = tanh(A[2]*h*r + Bxpb[2])
 *   h     = (1-z)*h + z*h_new
 *
 * where Bxpb = B @ x + b is precomputed by Python.
 *
 * Data layout:
 *   A:     (3, hidden_dim) contiguous
 *   Bxpb:  (batch_size, seq_len, 3, hidden_dim) contiguous
 *   output: (batch_size, seq_len, hidden_dim) contiguous
 *
 * Grid: (hidden_dim, batch_size), Block: (threads_per_block).
 * Each thread handles CHUNK_SIZE consecutive timesteps.
 *
 * CUDA entry points:
 *   fused_fwd_gru_diag_f32(A, Bxpb, output, stream)
 *   fused_fwd_gru_diag_f64(A, Bxpb, output, stream)
 *   fused_bwd_gru_diag_f32(grad, h, A, Bxpb, dl_dh, stream)
 *   fused_bwd_gru_diag_f64(grad, h, A, Bxpb, dl_dh, stream)
 */

#include <cuda_runtime.h>
#include <cstdint>
#include "brainevent/common.h"

#define THREADS_PER_WARP 32
#define THREADS_PER_BLOCK_GRU 1024

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
__device__ __forceinline__ T cuda_tanh(T x) {
    return tanh(x);
}

template<typename T>
__device__ __forceinline__ T cuda_tanh_deriv(T x) {
    T t = tanh(x);
    return (T)(1.0) - t * t;
}

// ============================================================================
// Fused GRU Forward Kernel
// ============================================================================

#define DEFINE_FUSED_FWD_GRU(SUFFIX, SCALAR_T, CHUNK_SIZE)                     \
                                                                               \
__global__ void __launch_bounds__(THREADS_PER_BLOCK_GRU, 1)                    \
_fused_fwd_gru_diag##SUFFIX(                                                   \
    const SCALAR_T* __restrict__ A,                                            \
    const SCALAR_T* __restrict__ Bxpb,                                         \
    SCALAR_T* __restrict__ output,                                             \
    int seq_len, int hidden_dim, int batch_size,                               \
    int max_its, SCALAR_T omega                                                \
) {                                                                            \
    const int h_idx = blockIdx.x;                                              \
    const int b_idx = blockIdx.y;                                              \
    if (h_idx >= hidden_dim || b_idx >= batch_size) return;                    \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid & (THREADS_PER_WARP - 1);                             \
    const int warp_id = tid >> 5;                                              \
    const int warps_per_block = (THREADS_PER_BLOCK_GRU + THREADS_PER_WARP - 1) \
                                / THREADS_PER_WARP;                            \
                                                                               \
    /* Load diagonal recurrence weights A (shared across time) */              \
    const SCALAR_T A_z = A[0 * hidden_dim + h_idx];                            \
    const SCALAR_T A_r = A[1 * hidden_dim + h_idx];                            \
    const SCALAR_T A_h = A[2 * hidden_dim + h_idx];                            \
                                                                               \
    /* Bxpb layout: (B, T, 3, hidden_dim) */                                   \
    const int bxpb_base = b_idx * seq_len * 3 * hidden_dim + h_idx;            \
    const int bxpb_stride = 3 * hidden_dim;                                    \
                                                                               \
    /* Load Bxpb values for this thread's chunk */                             \
    SCALAR_T bxpb_z[CHUNK_SIZE], bxpb_r[CHUNK_SIZE], bxpb_h[CHUNK_SIZE];       \
    const int t_start = tid * CHUNK_SIZE;                                      \
    int num_valid = 0;                                                         \
    for (int c = 0; c < CHUNK_SIZE; c++) {                                     \
        int t = t_start + c;                                                   \
        if (t < seq_len) {                                                     \
            int off = bxpb_base + t * bxpb_stride;                             \
            bxpb_z[c] = Bxpb[off + 0 * hidden_dim];                            \
            bxpb_r[c] = Bxpb[off + 1 * hidden_dim];                            \
            bxpb_h[c] = Bxpb[off + 2 * hidden_dim];                            \
            num_valid++;                                                       \
        } else {                                                               \
            bxpb_z[c] = (SCALAR_T)(0.0);                                       \
            bxpb_r[c] = (SCALAR_T)(0.0);                                       \
            bxpb_h[c] = (SCALAR_T)(0.0);                                       \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* Initialize solution: h = GRU(0, x) (one-step from zero) */              \
    SCALAR_T sol[CHUNK_SIZE];                                                  \
    for (int c = 0; c < CHUNK_SIZE; c++) {                                     \
        SCALAR_T z = cuda_sigmoid<SCALAR_T>(bxpb_z[c]);                        \
        SCALAR_T h_new = cuda_tanh<SCALAR_T>(bxpb_h[c]);                       \
        sol[c] = z * h_new;                                                    \
    }                                                                          \
                                                                               \
    /* Shared memory for inter-warp communication */                           \
    extern __shared__ char _smem_bytes[];                                      \
    SCALAR_T* smem_jac = reinterpret_cast<SCALAR_T*>(_smem_bytes);             \
    SCALAR_T* smem_rhs = smem_jac + warps_per_block;                           \
                                                                               \
    /* Newton iterations */                                                    \
    for (int newton_it = 0; newton_it < max_its; newton_it++) {                \
                                                                               \
        /* Compute h_prev (shifted): h_prev[t] = sol[t-1], h_prev[0] = 0 */    \
        SCALAR_T h_prev[CHUNK_SIZE];                                           \
        /* Get previous thread's last solution via shuffle */                  \
        SCALAR_T prev_thread_sol = __shfl_up_sync(                             \
            0xffffffff, sol[CHUNK_SIZE - 1], 1);                               \
        /* For first thread in warp, need from shared memory */                \
        if (lane == 0) {                                                       \
            if (warp_id > 0) {                                                 \
                prev_thread_sol = smem_rhs[warp_id - 1];                       \
            } else {                                                           \
                prev_thread_sol = (SCALAR_T)(0.0);                             \
            }                                                                  \
        }                                                                      \
        h_prev[0] = prev_thread_sol;                                           \
        for (int c = 1; c < CHUNK_SIZE; c++) {                                 \
            h_prev[c] = sol[c - 1];                                            \
        }                                                                      \
        /* Boundary: first element of first thread */                          \
        if (tid == 0) h_prev[0] = (SCALAR_T)(0.0);                             \
                                                                               \
        /* Compute negative residuals and Jacobians */                         \
        SCALAR_T reg_jac[CHUNK_SIZE];                                          \
        SCALAR_T reg_rhs[CHUNK_SIZE];                                          \
        for (int c = 0; c < CHUNK_SIZE; c++) {                                 \
            int t = t_start + c;                                               \
            if (t < seq_len) {                                                 \
                SCALAR_T hp = h_prev[c];                                       \
                SCALAR_T pre_z = A_z * hp + bxpb_z[c];                         \
                SCALAR_T pre_r = A_r * hp + bxpb_r[c];                         \
                SCALAR_T z = cuda_sigmoid<SCALAR_T>(pre_z);                    \
                SCALAR_T r = cuda_sigmoid<SCALAR_T>(pre_r);                    \
                SCALAR_T pre_h = A_h * hp * r + bxpb_h[c];                     \
                SCALAR_T h_new = cuda_tanh<SCALAR_T>(pre_h);                   \
                SCALAR_T f_val = z * h_new + ((SCALAR_T)(1.0) - z) * hp;       \
                                                                               \
                /* Negative residual */                                        \
                reg_rhs[c] = -(sol[c] - f_val);                                \
                                                                               \
                /* Jacobian: -df/dh_prev (diagonal) */                         \
                SCALAR_T dz = cuda_sigmoid_deriv<SCALAR_T>(pre_z);             \
                SCALAR_T dr = cuda_sigmoid_deriv<SCALAR_T>(pre_r);             \
                SCALAR_T dh = cuda_tanh_deriv<SCALAR_T>(pre_h);                \
                SCALAR_T J_z = A_z * dz;                                       \
                SCALAR_T J_r = A_r * dr;                                       \
                SCALAR_T J_h = A_h * dh;                                       \
                J_h = J_h * (r + hp * J_r);                                    \
                SCALAR_T jac_val = ((SCALAR_T)(1.0) - z)                       \
                    + (h_new - hp) * J_z + z * J_h;                            \
                reg_jac[c] = -jac_val;                                         \
            } else {                                                           \
                reg_jac[c] = (SCALAR_T)(-1.0);                                 \
                reg_rhs[c] = (SCALAR_T)(0.0);                                  \
            }                                                                  \
        }                                                                      \
        if (tid == 0) reg_jac[0] = (SCALAR_T)(0.0);                            \
                                                                               \
        /* ============== Parallel reduction (5-step) ============== */        \
        /* Step 1: Thomas */                                                   \
        for (int c = 1; c < CHUNK_SIZE; c++) {                                 \
            reg_rhs[c] -= reg_jac[c] * reg_rhs[c - 1];                         \
            reg_jac[c] *= -reg_jac[c - 1];                                     \
        }                                                                      \
        /* Step 2: Warp PCR */                                                 \
        SCALAR_T jl = reg_jac[CHUNK_SIZE - 1];                                 \
        SCALAR_T rl = reg_rhs[CHUNK_SIZE - 1];                                 \
        for (int d = 1; d < THREADS_PER_WARP; d <<= 1) {                       \
            SCALAR_T jp = __shfl_up_sync(0xffffffff, jl, d);                   \
            SCALAR_T rp = __shfl_up_sync(0xffffffff, rl, d);                   \
            if (lane >= d) { rl -= jl * rp; jl *= -jp; }                       \
        }                                                                      \
        reg_jac[CHUNK_SIZE - 1] = jl;                                          \
        reg_rhs[CHUNK_SIZE - 1] = rl;                                          \
        /* Step 3: Block PCR */                                                \
        if (lane == (THREADS_PER_WARP - 1) && num_valid > 0) {                 \
            smem_jac[warp_id] = reg_jac[CHUNK_SIZE - 1];                       \
            smem_rhs[warp_id] = reg_rhs[CHUNK_SIZE - 1];                       \
        }                                                                      \
        __syncthreads();                                                       \
        for (int d = 1; d < warps_per_block; d <<= 1) {                        \
            int pw = warp_id - d;                                              \
            if (lane == (THREADS_PER_WARP - 1) && pw >= 0                      \
                && num_valid > 0) {                                            \
                SCALAR_T jpw = smem_jac[pw];                                   \
                SCALAR_T rpw = smem_rhs[pw];                                   \
                reg_rhs[CHUNK_SIZE-1] -= reg_jac[CHUNK_SIZE-1] * rpw;          \
                reg_jac[CHUNK_SIZE-1] *= -jpw;                                 \
            }                                                                  \
            __syncthreads();                                                   \
            if (lane == (THREADS_PER_WARP - 1) && num_valid > 0) {             \
                smem_jac[warp_id] = reg_jac[CHUNK_SIZE - 1];                   \
                smem_rhs[warp_id] = reg_rhs[CHUNK_SIZE - 1];                   \
            }                                                                  \
            __syncthreads();                                                   \
        }                                                                      \
        /* Step 4: Warp fwd subst */                                           \
        SCALAR_T sp, jpv;                                                      \
        if (warp_id > 0 && lane != (THREADS_PER_WARP - 1)) {                   \
            jpv = smem_jac[warp_id - 1];                                       \
            sp = smem_rhs[warp_id - 1];                                        \
        } else {                                                               \
            jpv = (SCALAR_T)(-1.0);                                            \
            sp = (SCALAR_T)(0.0);                                              \
        }                                                                      \
        reg_rhs[CHUNK_SIZE-1] -= reg_jac[CHUNK_SIZE-1] * sp;                   \
        reg_jac[CHUNK_SIZE-1] *= -jpv;                                         \
        {                                                                      \
            SCALAR_T pr = __shfl_up_sync(                                      \
                0xffffffff, reg_rhs[CHUNK_SIZE-1], 1);                         \
            SCALAR_T pj = __shfl_up_sync(                                      \
                0xffffffff, reg_jac[CHUNK_SIZE-1], 1);                         \
            if (lane > 0) { sp = pr; jpv = pj; }                               \
        }                                                                      \
        /* Step 5: Chunk fwd subst */                                          \
        for (int c = 0; c < CHUNK_SIZE - 1; c++) {                             \
            reg_rhs[c] -= reg_jac[c] * sp;                                     \
            reg_jac[c] *= -jpv;                                                \
        }                                                                      \
                                                                               \
        /* Update solution: sol += omega * dh */                               \
        for (int c = 0; c < CHUNK_SIZE; c++) {                                 \
            sol[c] += omega * reg_rhs[c];                                      \
        }                                                                      \
                                                                               \
        /* Store last thread's sol for next iteration's h_prev lookup */       \
        if (lane == (THREADS_PER_WARP - 1)) {                                  \
            smem_rhs[warp_id] = sol[CHUNK_SIZE - 1];                           \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
                                                                               \
    /* Write final solution to output */                                       \
    const int out_base = b_idx * seq_len * hidden_dim + h_idx;                 \
    for (int c = 0; c < CHUNK_SIZE; c++) {                                     \
        int t = t_start + c;                                                   \
        if (t < seq_len) {                                                     \
            output[out_base + t * hidden_dim] = sol[c];                        \
        }                                                                      \
    }                                                                          \
}

// Instantiate
DEFINE_FUSED_FWD_GRU(_f32, float, 4)
DEFINE_FUSED_FWD_GRU(_f64, double, 1)

// ============================================================================
// Fused GRU Backward Kernel
// ============================================================================

#define DEFINE_FUSED_BWD_GRU(SUFFIX, SCALAR_T, CHUNK_SIZE)                     \
                                                                               \
__global__ void __launch_bounds__(THREADS_PER_BLOCK_GRU, 1)                    \
_fused_bwd_gru_diag##SUFFIX(                                                   \
    const SCALAR_T* __restrict__ grad_y,                                       \
    const SCALAR_T* __restrict__ h,                                            \
    const SCALAR_T* __restrict__ A,                                            \
    const SCALAR_T* __restrict__ Bxpb,                                         \
    SCALAR_T* __restrict__ dl_dh_out,                                          \
    int seq_len, int hidden_dim, int batch_size,                               \
    int max_its, SCALAR_T omega                                                \
) {                                                                            \
    const int h_idx = blockIdx.x;                                              \
    const int b_idx = blockIdx.y;                                              \
    if (h_idx >= hidden_dim || b_idx >= batch_size) return;                    \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid & (THREADS_PER_WARP - 1);                             \
    const int warp_id = tid >> 5;                                              \
    const int warps_per_block = (THREADS_PER_BLOCK_GRU + THREADS_PER_WARP - 1) \
                                / THREADS_PER_WARP;                            \
                                                                               \
    const SCALAR_T A_z = A[0 * hidden_dim + h_idx];                            \
    const SCALAR_T A_r = A[1 * hidden_dim + h_idx];                            \
    const SCALAR_T A_h = A[2 * hidden_dim + h_idx];                            \
                                                                               \
    const int h_base = b_idx * seq_len * hidden_dim + h_idx;                   \
    const int bxpb_base = b_idx * seq_len * 3 * hidden_dim + h_idx;            \
    const int bxpb_stride = 3 * hidden_dim;                                    \
                                                                               \
    /* Load grad_y in reverse time order for backward solve */                 \
    const int t_start = tid * CHUNK_SIZE;                                      \
    SCALAR_T grad_rev[CHUNK_SIZE];                                             \
    int num_valid = 0;                                                         \
    for (int c = 0; c < CHUNK_SIZE; c++) {                                     \
        int t_fwd = seq_len - 1 - (t_start + c);                               \
        if (t_fwd >= 0 && t_fwd < seq_len) {                                   \
            grad_rev[c] = grad_y[h_base + t_fwd * hidden_dim];                 \
            num_valid++;                                                       \
        } else {                                                               \
            grad_rev[c] = (SCALAR_T)(0.0);                                     \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* Compute backward Jacobians (reversed, shifted) */                       \
    SCALAR_T reg_jac[CHUNK_SIZE];                                              \
    for (int c = 0; c < CHUNK_SIZE; c++) {                                     \
        int t_fwd = seq_len - 1 - (t_start + c);                               \
        /* For backward: use Jacobian from t_fwd+1 (shifted) */                \
        int t_jac = t_fwd + 1;                                                 \
        if (t_jac >= 0 && t_jac < seq_len) {                                   \
            /* Get h[t_jac-1] (h_prev for t_jac) */                            \
            SCALAR_T hp;                                                       \
            if (t_jac > 0) {                                                   \
                hp = h[h_base + (t_jac - 1) * hidden_dim];                     \
            } else {                                                           \
                hp = (SCALAR_T)(0.0);                                          \
            }                                                                  \
            int boff = bxpb_base + t_jac * bxpb_stride;                        \
            SCALAR_T pre_z = A_z * hp + Bxpb[boff + 0 * hidden_dim];           \
            SCALAR_T pre_r = A_r * hp + Bxpb[boff + 1 * hidden_dim];           \
            SCALAR_T z = cuda_sigmoid<SCALAR_T>(pre_z);                        \
            SCALAR_T r = cuda_sigmoid<SCALAR_T>(pre_r);                        \
            SCALAR_T pre_hh = A_h * hp * r + Bxpb[boff + 2 * hidden_dim];      \
            SCALAR_T h_new = cuda_tanh<SCALAR_T>(pre_hh);                      \
            SCALAR_T dz = cuda_sigmoid_deriv<SCALAR_T>(pre_z);                 \
            SCALAR_T dr = cuda_sigmoid_deriv<SCALAR_T>(pre_r);                 \
            SCALAR_T dh_val = cuda_tanh_deriv<SCALAR_T>(pre_hh);               \
            SCALAR_T J_z = A_z * dz;                                           \
            SCALAR_T J_r = A_r * dr;                                           \
            SCALAR_T J_h = A_h * dh_val * (r + hp * J_r);                      \
            SCALAR_T jac_val = ((SCALAR_T)(1.0) - z)                           \
                + (h_new - hp) * J_z + z * J_h;                                \
            /* For diagonal, transpose is identity */                          \
            reg_jac[c] = -jac_val;                                             \
        } else {                                                               \
            reg_jac[c] = (SCALAR_T)(0.0);                                      \
        }                                                                      \
    }                                                                          \
    if (tid == 0) reg_jac[0] = (SCALAR_T)(0.0);                                \
                                                                               \
    /* Solve backward system via parallel reduction */                         \
    extern __shared__ char _smem_bytes[];                                      \
    SCALAR_T* smem_jac = reinterpret_cast<SCALAR_T*>(_smem_bytes);             \
    SCALAR_T* smem_rhs = smem_jac + warps_per_block;                           \
                                                                               \
    SCALAR_T reg_rhs[CHUNK_SIZE];                                              \
    for (int c = 0; c < CHUNK_SIZE; c++) reg_rhs[c] = grad_rev[c];             \
                                                                               \
    /* Step 1: Thomas */                                                       \
    for (int c = 1; c < CHUNK_SIZE; c++) {                                     \
        reg_rhs[c] -= reg_jac[c] * reg_rhs[c - 1];                             \
        reg_jac[c] *= -reg_jac[c - 1];                                         \
    }                                                                          \
    /* Step 2: Warp PCR */                                                     \
    SCALAR_T jl = reg_jac[CHUNK_SIZE - 1];                                     \
    SCALAR_T rl = reg_rhs[CHUNK_SIZE - 1];                                     \
    for (int d = 1; d < THREADS_PER_WARP; d <<= 1) {                           \
        SCALAR_T jp = __shfl_up_sync(0xffffffff, jl, d);                       \
        SCALAR_T rp = __shfl_up_sync(0xffffffff, rl, d);                       \
        if (lane >= d) { rl -= jl * rp; jl *= -jp; }                           \
    }                                                                          \
    reg_jac[CHUNK_SIZE - 1] = jl;                                              \
    reg_rhs[CHUNK_SIZE - 1] = rl;                                              \
    /* Step 3: Block PCR */                                                    \
    if (lane == (THREADS_PER_WARP - 1) && num_valid > 0) {                     \
        smem_jac[warp_id] = reg_jac[CHUNK_SIZE - 1];                           \
        smem_rhs[warp_id] = reg_rhs[CHUNK_SIZE - 1];                           \
    }                                                                          \
    __syncthreads();                                                           \
    for (int d = 1; d < warps_per_block; d <<= 1) {                            \
        int pw = warp_id - d;                                                  \
        if (lane == (THREADS_PER_WARP - 1) && pw >= 0 && num_valid > 0) {      \
            SCALAR_T jpw = smem_jac[pw];                                       \
            SCALAR_T rpw = smem_rhs[pw];                                       \
            reg_rhs[CHUNK_SIZE-1] -= reg_jac[CHUNK_SIZE-1] * rpw;              \
            reg_jac[CHUNK_SIZE-1] *= -jpw;                                     \
        }                                                                      \
        __syncthreads();                                                       \
        if (lane == (THREADS_PER_WARP - 1) && num_valid > 0) {                 \
            smem_jac[warp_id] = reg_jac[CHUNK_SIZE - 1];                       \
            smem_rhs[warp_id] = reg_rhs[CHUNK_SIZE - 1];                       \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    /* Step 4: Warp fwd subst */                                               \
    SCALAR_T sp, jpv;                                                          \
    if (warp_id > 0 && lane != (THREADS_PER_WARP - 1)) {                       \
        jpv = smem_jac[warp_id - 1];                                           \
        sp = smem_rhs[warp_id - 1];                                            \
    } else {                                                                   \
        jpv = (SCALAR_T)(-1.0);                                                \
        sp = (SCALAR_T)(0.0);                                                  \
    }                                                                          \
    reg_rhs[CHUNK_SIZE-1] -= reg_jac[CHUNK_SIZE-1] * sp;                       \
    reg_jac[CHUNK_SIZE-1] *= -jpv;                                             \
    {                                                                          \
        SCALAR_T pr = __shfl_up_sync(                                          \
            0xffffffff, reg_rhs[CHUNK_SIZE-1], 1);                             \
        SCALAR_T pj = __shfl_up_sync(                                          \
            0xffffffff, reg_jac[CHUNK_SIZE-1], 1);                             \
        if (lane > 0) { sp = pr; jpv = pj; }                                   \
    }                                                                          \
    /* Step 5: Chunk fwd subst */                                              \
    for (int c = 0; c < CHUNK_SIZE - 1; c++) {                                 \
        reg_rhs[c] -= reg_jac[c] * sp;                                         \
        reg_jac[c] *= -jpv;                                                    \
    }                                                                          \
                                                                               \
    /* Write dl/dh in forward order (reverse back from reversed) */            \
    for (int c = 0; c < CHUNK_SIZE; c++) {                                     \
        int t_fwd = seq_len - 1 - (t_start + c);                               \
        if (t_fwd >= 0 && t_fwd < seq_len) {                                   \
            dl_dh_out[h_base + t_fwd * hidden_dim] = reg_rhs[c];               \
        }                                                                      \
    }                                                                          \
}

// Instantiate
DEFINE_FUSED_BWD_GRU(_f32, float, 4)
DEFINE_FUSED_BWD_GRU(_f64, double, 1)

// ============================================================================
// CUDA entry points
// ============================================================================

// @BE fused_fwd_gru_diag_f32
void fused_fwd_gru_diag_f32(
    const BE::Tensor A_tv,
    const BE::Tensor Bxpb_tv,
    BE::Tensor output_tv,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int hidden_dim = static_cast<int>(A_tv.size(1));
    int batch_size = static_cast<int>(Bxpb_tv.size(0));
    int seq_len    = static_cast<int>(Bxpb_tv.size(1));

    int max_its = 3;
    float omega = 1.0f;

    const float* d_A = static_cast<const float*>(A_tv.data_ptr());
    const float* d_Bxpb = static_cast<const float*>(Bxpb_tv.data_ptr());
    float* d_out = static_cast<float*>(output_tv.data_ptr());

    int warps_per_block = (THREADS_PER_BLOCK_GRU + THREADS_PER_WARP - 1)
                          / THREADS_PER_WARP;
    unsigned int smem_size = 2 * warps_per_block * sizeof(float);

    dim3 grid(hidden_dim, batch_size);
    dim3 block(THREADS_PER_BLOCK_GRU);
    _fused_fwd_gru_diag_f32<<<grid, block, smem_size, s>>>(
        d_A, d_Bxpb, d_out, seq_len, hidden_dim, batch_size,
        max_its, omega);
}

// @BE fused_fwd_gru_diag_f64
void fused_fwd_gru_diag_f64(
    const BE::Tensor A_tv,
    const BE::Tensor Bxpb_tv,
    BE::Tensor output_tv,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int hidden_dim = static_cast<int>(A_tv.size(1));
    int batch_size = static_cast<int>(Bxpb_tv.size(0));
    int seq_len    = static_cast<int>(Bxpb_tv.size(1));

    int max_its = 3;
    double omega = 1.0;

    const double* d_A = static_cast<const double*>(A_tv.data_ptr());
    const double* d_Bxpb = static_cast<const double*>(Bxpb_tv.data_ptr());
    double* d_out = static_cast<double*>(output_tv.data_ptr());

    int warps_per_block = (THREADS_PER_BLOCK_GRU + THREADS_PER_WARP - 1)
                          / THREADS_PER_WARP;
    unsigned int smem_size = 2 * warps_per_block * sizeof(double);

    dim3 grid(hidden_dim, batch_size);
    dim3 block(THREADS_PER_BLOCK_GRU);
    _fused_fwd_gru_diag_f64<<<grid, block, smem_size, s>>>(
        d_A, d_Bxpb, d_out, seq_len, hidden_dim, batch_size,
        max_its, omega);
}

// @BE fused_bwd_gru_diag_f32
void fused_bwd_gru_diag_f32(
    const BE::Tensor grad_tv,
    const BE::Tensor h_tv,
    const BE::Tensor A_tv,
    const BE::Tensor Bxpb_tv,
    const BE::Tensor dl_dh_tv,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int hidden_dim = static_cast<int>(A_tv.size(1));
    int batch_size = static_cast<int>(grad_tv.size(0));
    int seq_len    = static_cast<int>(grad_tv.size(1));
    int max_its = 3;
    float omega = 1.0f;

    int warps_per_block = (THREADS_PER_BLOCK_GRU + THREADS_PER_WARP - 1)
                          / THREADS_PER_WARP;
    unsigned int smem_size = 2 * warps_per_block * sizeof(float);

    dim3 grid(hidden_dim, batch_size);
    dim3 block(THREADS_PER_BLOCK_GRU);
    _fused_bwd_gru_diag_f32<<<grid, block, smem_size, s>>>(
        static_cast<const float*>(grad_tv.data_ptr()),
        static_cast<const float*>(h_tv.data_ptr()),
        static_cast<const float*>(A_tv.data_ptr()),
        static_cast<const float*>(Bxpb_tv.data_ptr()),
        static_cast<float*>(dl_dh_tv.data_ptr()),
        seq_len, hidden_dim, batch_size, max_its, omega);
}

// @BE fused_bwd_gru_diag_f64
void fused_bwd_gru_diag_f64(
    const BE::Tensor grad_tv,
    const BE::Tensor h_tv,
    const BE::Tensor A_tv,
    const BE::Tensor Bxpb_tv,
    const BE::Tensor dl_dh_tv,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int hidden_dim = static_cast<int>(A_tv.size(1));
    int batch_size = static_cast<int>(grad_tv.size(0));
    int seq_len    = static_cast<int>(grad_tv.size(1));
    int max_its = 3;
    double omega = 1.0;

    int warps_per_block = (THREADS_PER_BLOCK_GRU + THREADS_PER_WARP - 1)
                          / THREADS_PER_WARP;
    unsigned int smem_size = 2 * warps_per_block * sizeof(double);

    dim3 grid(hidden_dim, batch_size);
    dim3 block(THREADS_PER_BLOCK_GRU);
    _fused_bwd_gru_diag_f64<<<grid, block, smem_size, s>>>(
        static_cast<const double*>(grad_tv.data_ptr()),
        static_cast<const double*>(h_tv.data_ptr()),
        static_cast<const double*>(A_tv.data_ptr()),
        static_cast<const double*>(Bxpb_tv.data_ptr()),
        static_cast<double*>(dl_dh_tv.data_ptr()),
        seq_len, hidden_dim, batch_size, max_its, omega);
}
