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
 * binary_fcnmv.cu -- Event-Driven Binary FCN Sparse Matrix-Vector CUDA Kernels
 * ==============================================================================
 *
 * This module provides optimized CUDA kernels for event-driven sparse
 * matrix-vector multiplication with fixed connection number (FCN).
 *
 * Operator: binary_fcnmv
 *   - Gather mode (transpose=False): y[i] = sum_k weights[i,k] * is_active(spikes[indices[i,k]])
 *   - Scatter mode (transpose=True): output[indices[i,k]] += weights[i,k] * is_active(spikes[i])
 *
 * Supports weight dtypes: float32, float64, float16, bfloat16
 * Supports spike dtypes:  bool (uint8), float32, float64, float16, bfloat16
 * Supports homo (scalar weight) and hetero (per-connection weight array) modes.
 *
 * Kernel strategy (universal dispatch based on n_conn):
 *
 *   Gather:
 *     n_conn <= 512  -> TPR (thread-per-row): one thread per row, serial accumulation.
 *                       High occupancy (256 rows/block), no warp-reduction overhead.
 *     n_conn > 512   -> MR (warp-per-row, multi-round): one warp (32 threads) per row,
 *                       lanes stride over connections, warp-shuffle reduction.
 *                       Coalesced memory access dominates at large n_conn.
 *
 *   Scatter:
 *     n_conn <= threshold -> TPR (thread-per-row): one thread per row, serial atomicAdd loop.
 *                            Wins when n_conn is small; low fan-out keeps atomicAdd contention low.
 *     n_conn > threshold  -> WPR (warp-per-row): one warp per row, 32 lanes stride n_conn,
 *                            each lane issues its own atomicAdd — amortizes warp overhead at
 *                            large n_conn.  Launch: <<<ceil(n_pre/8), 256>>>
 *                            threshold = cubic polynomial of n_pre, fit from benchmark data
 *                            (see boundary_dis.py).
 *
 * Benchmark crossover (n_pre=n_post=100K, 50% spike rate):
 *   Gather TPR vs MR: TPR wins for n_conn <= 512, MR wins 2x at 1024, 4.3x at 4096.
 *   Scatter TPR vs WPR: crossover n_conn is a cubic function of n_pre (see boundary_dis.py).
 */

#include "cuda_common.h"
#include "brainevent/common.h"

// ============================================================================
// GATHER kernels  (transpose=False)
// y[i] = sum_k weights[i,k] * is_active(spikes[indices[i,k]])
// ============================================================================

// --- Gather, thread-per-row (TPR): one thread per row, serial loop ---
// Optimal for n_conn <= 512.  Each thread accumulates in a register, no sync needed.
// Launch: <<<ceil(n_pre/256), 256>>>

#define DEFINE_BG_TPR_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _bg_tpr_homo_kern##SUFFIX(                                                         \
    const int32_t* __restrict__ indices,                                                            \
    const SPIKE_T* __restrict__ spikes,                                                             \
    WEIGHT_T*      __restrict__ output,                                                             \
    const WEIGHT_T* __restrict__ weights,                                                           \
    int n_pre, int n_conn                                                                           \
) {                                                                                                 \
    int row = blockIdx.x * blockDim.x + threadIdx.x;                                                \
    if (row >= n_pre) return;                                                                       \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                          \
    ACC_T val = ACC_ZERO;                                                                           \
    for (int k = 0; k < n_conn; k++) {                                                              \
        int idx = __ldg(&i_row[k]);                                                                 \
        if (IS_ACTIVE(__ldg(&spikes[idx])))                                                         \
            val += (ACC_T)1;                                                                        \
    }                                                                                               \
    output[row] = WRITE_W(READ_W(weights[0]) * val);                                                \
}

#define DEFINE_BG_TPR_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _bg_tpr_hetero_kern##SUFFIX(                                                         \
    const int32_t* __restrict__ indices,                                                              \
    const SPIKE_T* __restrict__ spikes,                                                               \
    WEIGHT_T*      __restrict__ output,                                                               \
    const WEIGHT_T* __restrict__ weights,                                                             \
    int n_pre, int n_conn                                                                             \
) {                                                                                                   \
    int row = blockIdx.x * blockDim.x + threadIdx.x;                                                  \
    if (row >= n_pre) return;                                                                         \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                            \
    const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                           \
    ACC_T val = ACC_ZERO;                                                                             \
    for (int k = 0; k < n_conn; k++) {                                                                \
        int idx = __ldg(&i_row[k]);                                                                   \
        if (IS_ACTIVE(__ldg(&spikes[idx])))                                                           \
            val += READ_W(__ldg(&w_row[k]));                                                          \
    }                                                                                                 \
    output[row] = WRITE_W(val);                                                                       \
}

// --- Gather, warp-per-row multi-round (MR): 32 threads per row, warp reduction ---
// Optimal for n_conn > 512.  32 lanes read consecutive indices (coalesced), then
// warp-shuffle reduction.  Launch: <<<ceil(n_pre/8), 256>>> (8 warps/block = 8 rows/block).

#define DEFINE_BG_MR_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _bg_mr_homo_kern##SUFFIX(                                                                   \
    const int32_t* __restrict__ indices,                                                                    \
    const SPIKE_T* __restrict__ spikes,                                                                     \
    WEIGHT_T*      __restrict__ output,                                                                     \
    const WEIGHT_T* __restrict__ weights,                                                                   \
    int n_pre, int n_conn                                                                                   \
) {                                                                                                         \
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);                                          \
    if (row >= n_pre) return;                                                                               \
    int lane = threadIdx.x & 31;                                                                            \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                                  \
    ACC_T val = ACC_ZERO;                                                                                   \
    for (int k = lane; k < n_conn; k += 32) {                                                               \
        int idx = __ldg(&i_row[k]);                                                                         \
        if (IS_ACTIVE(__ldg(&spikes[idx])))                                                                 \
            val += (ACC_T)1;                                                                                \
    }                                                                                                       \
    val = WARP_RED(val);                                                                                    \
    if (lane == 0)                                                                                          \
        output[row] = WRITE_W(READ_W(weights[0]) * val);                                                    \
}

#define DEFINE_BG_MR_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _bg_mr_hetero_kern##SUFFIX(                                                                   \
    const int32_t* __restrict__ indices,                                                                      \
    const SPIKE_T* __restrict__ spikes,                                                                       \
    WEIGHT_T*      __restrict__ output,                                                                       \
    const WEIGHT_T* __restrict__ weights,                                                                     \
    int n_pre, int n_conn                                                                                     \
) {                                                                                                           \
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);                                            \
    if (row >= n_pre) return;                                                                                 \
    int lane = threadIdx.x & 31;                                                                              \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                                    \
    const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                                   \
    ACC_T val = ACC_ZERO;                                                                                     \
    for (int k = lane; k < n_conn; k += 32) {                                                                 \
        int idx = __ldg(&i_row[k]);                                                                           \
        if (IS_ACTIVE(__ldg(&spikes[idx])))                                                                   \
            val += READ_W(__ldg(&w_row[k]));                                                                  \
    }                                                                                                         \
    val = WARP_RED(val);                                                                                      \
    if (lane == 0)                                                                                            \
        output[row] = WRITE_W(val);                                                                           \
}

// ============================================================================
// SCATTER kernels  (transpose=True)
// output[indices[i,k]] += weights[i,k] * is_active(spikes[i])
// ============================================================================

// --- Scatter: WPR (warp-per-row) and TPR (thread-per-row) ---
// WPR: one warp (32 threads) per row; each lane strides n_conn with step 32 and issues its own
//      atomicAdd.  Wins when n_conn > threshold(n_pre).  Launch: <<<ceil(n_pre/8), 256>>>
// TPR: one thread per row, serial atomicAdd loop.  Wins for small n_conn.
//      Launch: <<<ceil(n_pre/256), 256>>>


#define DEFINE_BS_WPR_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _bs_wpr_homo_kern##SUFFIX(                                                    \
    const int32_t* __restrict__ indices,                                                      \
    const SPIKE_T* __restrict__ spikes,                                                       \
    WEIGHT_T* __restrict__ output,                                                       \
    const WEIGHT_T* __restrict__ weights,                                                     \
    int n_pre, int n_conn                                                                     \
) {                                                                                           \
    int tid = blockIdx.x * blockDim.x + threadIdx.x;                                          \
    int row = tid / 32;                                                                       \
    int lane = tid % 32;                                                                      \
    if (row >= n_pre) return;                                                                 \
    if (!IS_ACTIVE(__ldg(&spikes[row]))) return;                                              \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                    \
    ACC_T w0 = READ_W(weights[0]);                                                            \
    for (int k = lane; k < n_conn; k += 32) {                                                 \
        int idx = __ldg(&i_row[k]);                                                           \
        ATOMIC_ADD_W(&output[idx], w0);                                                       \
    }                                                                                         \
}

#define DEFINE_BS_WPR_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _bs_wpr_hetero_kern##SUFFIX(                                                    \
    const int32_t*  __restrict__ indices,                                                       \
    const SPIKE_T*  __restrict__ spikes,                                                        \
    WEIGHT_T*       __restrict__ output,                                                        \
    const WEIGHT_T* __restrict__ weights,                                                       \
    int n_pre, int n_conn                                                                       \
) {                                                                                             \
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;                                           \
    int row  = tid / 32;                                                                        \
    int lane = tid % 32;                                                                        \
    if (row >= n_pre) return;                                                                   \
    if (!IS_ACTIVE(__ldg(&spikes[row]))) return;                                                \
    const int32_t*  i_row = indices + (size_t)row * n_conn;                                     \
    const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                     \
    for (int k = lane; k < n_conn; k += 32) {                                                   \
        int   idx = __ldg(&i_row[k]);                                                           \
        ACC_T  wk = READ_W(__ldg(&w_row[k]));                                                   \
        ATOMIC_ADD_W(&output[idx], wk);                                                         \
    }                                                                                           \
}

/**/
#define DEFINE_BS_TPR_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _bs_tpr_homo_kern##SUFFIX(                                                    \
    const int32_t* __restrict__ indices,                                                      \
    const SPIKE_T* __restrict__ spikes,                                                       \
    WEIGHT_T*      __restrict__ output,                                                       \
    const WEIGHT_T* __restrict__ weights,                                                     \
    int n_pre, int n_conn                                                                     \
) {                                                                                           \
    int row = blockIdx.x * blockDim.x + threadIdx.x;                                          \
    if (row >= n_pre) return;                                                                 \
    if (!IS_ACTIVE(__ldg(&spikes[row]))) return;                                              \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                    \
    ACC_T w0 = READ_W(weights[0]);                                                            \
    for (int k = 0; k < n_conn; k++) {                                                        \
        int idx = __ldg(&i_row[k]);                                                           \
        ATOMIC_ADD_W(&output[idx], w0);                                                       \
    }                                                                                         \
}


#define DEFINE_BS_TPR_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _bs_tpr_hetero_kern##SUFFIX(                                                    \
    const int32_t* __restrict__ indices,                                                        \
    const SPIKE_T* __restrict__ spikes,                                                         \
    WEIGHT_T*      __restrict__ output,                                                         \
    const WEIGHT_T* __restrict__ weights,                                                       \
    int n_pre, int n_conn                                                                       \
) {                                                                                             \
    int row = blockIdx.x * blockDim.x + threadIdx.x;                                            \
    if (row >= n_pre) return;                                                                   \
    if (!IS_ACTIVE(__ldg(&spikes[row]))) return;                                                \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                      \
    const WEIGHT_T* w_row = weights + (size_t)row * n_conn;                                     \
    for (int k = 0; k < n_conn; k++) {                                                          \
        int idx = __ldg(&i_row[k]);                                                             \
        ACC_T wk = READ_W(__ldg(&w_row[k]));                                                    \
        ATOMIC_ADD_W(&output[idx], wk);                                                         \
    }                                                                                           \
}

// ============================================================================
// Kernel Instantiations
// ============================================================================

// ---- float32 ----
DEFINE_BG_TPR_HOMO     (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_BG_TPR_HETERO   (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_BG_TPR_HOMO     (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_BG_TPR_HETERO   (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_BG_MR_HOMO      (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO    (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO      (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO    (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_BS_TPR_HOMO     (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, atomicAdd)
DEFINE_BS_TPR_HETERO   (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, atomicAdd)
DEFINE_BS_WPR_HOMO     (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, atomicAdd)
DEFINE_BS_WPR_HETERO   (_bool_f32, uint8_t, IS_ACTIVE_BOOL, float, float, READ_F32, atomicAdd)
DEFINE_BS_TPR_HOMO     (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, atomicAdd)
DEFINE_BS_TPR_HETERO   (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, atomicAdd)
DEFINE_BS_WPR_HOMO     (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, atomicAdd)
DEFINE_BS_WPR_HETERO   (_float_f32, float, IS_ACTIVE_F32, float, float, READ_F32, atomicAdd)

// ---- float64 ----
DEFINE_BG_TPR_HOMO     (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_BG_TPR_HETERO   (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_BG_TPR_HOMO     (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_BG_TPR_HETERO   (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_BG_MR_HOMO      (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HETERO    (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HOMO      (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BG_MR_HETERO    (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_BS_TPR_HOMO     (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, atomic_add_f64)
DEFINE_BS_TPR_HETERO   (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, atomic_add_f64)
DEFINE_BS_WPR_HOMO     (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, atomic_add_f64)
DEFINE_BS_WPR_HETERO   (_bool_f64, uint8_t, IS_ACTIVE_BOOL, double, double, READ_F64, atomic_add_f64)
DEFINE_BS_TPR_HOMO     (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, atomic_add_f64)
DEFINE_BS_TPR_HETERO   (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, atomic_add_f64)
DEFINE_BS_WPR_HOMO     (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, atomic_add_f64)
DEFINE_BS_WPR_HETERO   (_float_f64, double, IS_ACTIVE_F64, double, double, READ_F64, atomic_add_f64)

// ---- float16 ----
DEFINE_BG_TPR_HOMO     (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_BG_TPR_HETERO   (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_BG_TPR_HOMO     (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_BG_TPR_HETERO   (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_BG_MR_HOMO      (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO    (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO      (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO    (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_BS_TPR_HOMO     (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, atomic_add_f16)
DEFINE_BS_TPR_HETERO   (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, atomic_add_f16)
DEFINE_BS_WPR_HOMO     (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, atomic_add_f16)
DEFINE_BS_WPR_HETERO   (_bool_f16, uint8_t, IS_ACTIVE_BOOL, __half, float, READ_F16, atomic_add_f16)
DEFINE_BS_TPR_HOMO     (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, atomic_add_f16)
DEFINE_BS_TPR_HETERO   (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, atomic_add_f16)
DEFINE_BS_WPR_HOMO     (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, atomic_add_f16)
DEFINE_BS_WPR_HETERO   (_float_f16, __half, IS_ACTIVE_F16, __half, float, READ_F16, atomic_add_f16)

// ---- bfloat16 ----
DEFINE_BG_TPR_HOMO     (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_BG_TPR_HETERO   (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_BG_TPR_HOMO     (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_BG_TPR_HETERO   (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_BG_MR_HOMO      (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO    (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HOMO      (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_MR_HETERO    (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BS_TPR_HOMO     (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BS_TPR_HETERO   (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BS_WPR_HOMO     (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BS_WPR_HETERO   (_bool_bf16, uint8_t, IS_ACTIVE_BOOL, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BS_TPR_HOMO     (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BS_TPR_HETERO   (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BS_WPR_HOMO     (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_BS_WPR_HETERO   (_float_bf16, __nv_bfloat16, IS_ACTIVE_BF16, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)


// ============================================================================
// FFI Entry Points — unified dispatch (no warp/basic split)
// ============================================================================

// ---- FFI macro: gather homo (auto-select TPR or MR based on n_conn) ----
#define FFI_BG_HOMO(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                               \
void binary_fcnmv_gather_homo##SUFFIX(                                                            \
    const BE::Tensor weights, const BE::Tensor indices,                                           \
    const BE::Tensor spikes,  BE::Tensor output, int64_t stream                                   \
) {                                                                                               \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                      \
    int n_pre  = static_cast<int>(indices.size(0));                                               \
    int n_conn = static_cast<int>(indices.size(1));                                               \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                 \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                    \
    const SPIKE_C_T*  d_spk = static_cast<const SPIKE_C_T*>(spikes.data_ptr());                   \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                        \
    if (n_conn <= 512) {                                                                          \
        int bsz = 256;                                                                            \
        int n_blocks = (n_pre + bsz - 1) / bsz;                                                  \
        _bg_tpr_homo_kern##SUFFIX<<<n_blocks, bsz, 0, s>>>(                                       \
            d_idx, d_spk, d_out, d_w, n_pre, n_conn);                                            \
    } else {                                                                                      \
        int bsz = 256; int rpb = bsz >> 5; int n_blocks = (n_pre + rpb - 1) / rpb;                \
        _bg_mr_homo_kern##SUFFIX<<<n_blocks, bsz, 0, s>>>(                                        \
            d_idx, d_spk, d_out, d_w, n_pre, n_conn);                                            \
    }                                                                                             \
}

// ---- FFI macro: gather hetero (auto-select TPR or MR based on n_conn) ----
#define FFI_BG_HETERO(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                             \
void binary_fcnmv_gather_hetero##SUFFIX(                                                          \
    const BE::Tensor weights, const BE::Tensor indices,                                           \
    const BE::Tensor spikes,  BE::Tensor output, int64_t stream                                   \
) {                                                                                               \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                      \
    int n_pre  = static_cast<int>(indices.size(0));                                               \
    int n_conn = static_cast<int>(indices.size(1));                                               \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());                 \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());                    \
    const SPIKE_C_T*  d_spk = static_cast<const SPIKE_C_T*>(spikes.data_ptr());                   \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                        \
    if (n_conn <= 512) {                                                                          \
        int bsz = 256;                                                                            \
        int n_blocks = (n_pre + bsz - 1) / bsz;                                                  \
        _bg_tpr_hetero_kern##SUFFIX<<<n_blocks, bsz, 0, s>>>(                                     \
            d_idx, d_spk, d_out, d_w, n_pre, n_conn);                                            \
    } else {                                                                                      \
        int bsz = 256; int rpb = bsz >> 5; int n_blocks = (n_pre + rpb - 1) / rpb;                \
        _bg_mr_hetero_kern##SUFFIX<<<n_blocks, bsz, 0, s>>>(                                      \
            d_idx, d_spk, d_out, d_w, n_pre, n_conn);                                            \
    }                                                                                             \
}

#define FFI_BS_HOMO(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                              \
void binary_fcnmv_scatter_homo##SUFFIX(                                                         \
    const BE::Tensor weights, const BE::Tensor indices,                                         \
    const BE::Tensor spikes,  BE::Tensor output, int64_t stream                                 \
) {                                                                                             \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                    \
    int n_pre  = static_cast<int>(indices.size(0));                                             \
    int n_conn = static_cast<int>(indices.size(1));                                             \
    int n_post = static_cast<int>(output.size(0));                                              \
                                                                                                \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());               \
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());                     \
    const SPIKE_C_T* d_spk = static_cast<const SPIKE_C_T*>(spikes.data_ptr());                  \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                            \
                                                                                                \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(WEIGHT_C_T), s);                          \
                                                                                                \
    if (n_pre == 0) return;                                                                     \
                                                                                                \
    int bsz = 256;                                                                              \
    /* 一次函数边界: c = (1539/521) * s, 其中 s = n_pre / 4000 */                                  \
    /* 整理得: n_conn * 521 * 4000 > n_pre * 1539 */                                              \
    if ((int64_t)n_conn * 2084000 > (int64_t)n_pre * 1539) {                                    \
        int warps_per_block = bsz / 32;                                                         \
        int n_blocks_wpr = (n_pre + warps_per_block - 1) / warps_per_block;                     \
        _bs_wpr_homo_kern##SUFFIX<<<n_blocks_wpr, bsz, 0, s>>>(                                 \
            d_idx, d_spk, d_out, d_w, n_pre, n_conn);                                           \
    } else {                                                                                    \
        int n_blocks_tpr = (n_pre + bsz - 1) / bsz;                                             \
        _bs_tpr_homo_kern##SUFFIX<<<n_blocks_tpr, bsz, 0, s>>>(                                 \
            d_idx, d_spk, d_out, d_w, n_pre, n_conn);                                           \
    }                                                                                           \
}
//(int64_t)n_conn * 2084000 > (int64_t)n_pre * 1539

#define FFI_BS_HETERO(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                            \
void binary_fcnmv_scatter_hetero##SUFFIX(                                                       \
    const BE::Tensor weights, const BE::Tensor indices,                                         \
    const BE::Tensor spikes,  BE::Tensor output, int64_t stream                                 \
) {                                                                                             \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                    \
    int n_pre  = static_cast<int>(indices.size(0));                                             \
    int n_conn = static_cast<int>(indices.size(1));                                             \
    int n_post = static_cast<int>(output.size(0));                                              \
                                                                                                \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());               \
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());                     \
    const SPIKE_C_T* d_spk = static_cast<const SPIKE_C_T*>(spikes.data_ptr());                  \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                            \
                                                                                                \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(WEIGHT_C_T), s);                          \
                                                                                                \
    if (n_pre == 0) return;                                                                     \
                                                                                                \
    int bsz = 256;                                                                              \
    if ((int64_t)n_conn * 2084000 > (int64_t)n_pre * 1539) {                                    \
        int warps_per_block = bsz / 32;                                                         \
        int n_blocks_wpr = (n_pre + warps_per_block - 1) / warps_per_block;                     \
        _bs_wpr_hetero_kern##SUFFIX<<<n_blocks_wpr, bsz, 0, s>>>(                               \
            d_idx, d_spk, d_out, d_w, n_pre, n_conn);                                           \
    } else {                                                                                    \
        int n_blocks_tpr = (n_pre + bsz - 1) / bsz;                                             \
        _bs_tpr_hetero_kern##SUFFIX<<<n_blocks_tpr, bsz, 0, s>>>(                               \
            d_idx, d_spk, d_out, d_w, n_pre, n_conn);                                           \
    }                                                                                           \
}

// ============================================================================
// FFI Instantiations
// ============================================================================

// ---- float32 ----
// @BE binary_fcnmv_gather_homo_bool_f32
FFI_BG_HOMO  (_bool_f32, float, uint8_t)
// @BE binary_fcnmv_gather_hetero_bool_f32
FFI_BG_HETERO(_bool_f32, float, uint8_t)
// @BE binary_fcnmv_gather_homo_float_f32
FFI_BG_HOMO  (_float_f32, float, float)
// @BE binary_fcnmv_gather_hetero_float_f32
FFI_BG_HETERO(_float_f32, float, float)
// @BE binary_fcnmv_scatter_homo_bool_f32
FFI_BS_HOMO  (_bool_f32, float, uint8_t)
// @BE binary_fcnmv_scatter_hetero_bool_f32
FFI_BS_HETERO(_bool_f32, float, uint8_t)
// @BE binary_fcnmv_scatter_homo_float_f32
FFI_BS_HOMO  (_float_f32, float, float)
// @BE binary_fcnmv_scatter_hetero_float_f32
FFI_BS_HETERO(_float_f32, float, float)

// ---- float64 ----
// @BE binary_fcnmv_gather_homo_bool_f64
FFI_BG_HOMO  (_bool_f64, double, uint8_t)
// @BE binary_fcnmv_gather_hetero_bool_f64
FFI_BG_HETERO(_bool_f64, double, uint8_t)
// @BE binary_fcnmv_gather_homo_float_f64
FFI_BG_HOMO  (_float_f64, double, double)
// @BE binary_fcnmv_gather_hetero_float_f64
FFI_BG_HETERO(_float_f64, double, double)
// @BE binary_fcnmv_scatter_homo_bool_f64
FFI_BS_HOMO  (_bool_f64, double, uint8_t)
// @BE binary_fcnmv_scatter_hetero_bool_f64
FFI_BS_HETERO(_bool_f64, double, uint8_t)
// @BE binary_fcnmv_scatter_homo_float_f64
FFI_BS_HOMO  (_float_f64, double, double)
// @BE binary_fcnmv_scatter_hetero_float_f64
FFI_BS_HETERO(_float_f64, double, double)

// ---- float16 ----
// @BE binary_fcnmv_gather_homo_bool_f16
FFI_BG_HOMO  (_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_gather_hetero_bool_f16
FFI_BG_HETERO(_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_gather_homo_float_f16
FFI_BG_HOMO  (_float_f16, __half, __half)
// @BE binary_fcnmv_gather_hetero_float_f16
FFI_BG_HETERO(_float_f16, __half, __half)
// @BE binary_fcnmv_scatter_homo_bool_f16
FFI_BS_HOMO  (_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_scatter_hetero_bool_f16
FFI_BS_HETERO(_bool_f16, __half, uint8_t)
// @BE binary_fcnmv_scatter_homo_float_f16
FFI_BS_HOMO  (_float_f16, __half, __half)
// @BE binary_fcnmv_scatter_hetero_float_f16
FFI_BS_HETERO(_float_f16, __half, __half)

// ---- bfloat16 ----
// @BE binary_fcnmv_gather_homo_bool_bf16
FFI_BG_HOMO  (_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_gather_hetero_bool_bf16
FFI_BG_HETERO(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_gather_homo_float_bf16
FFI_BG_HOMO  (_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmv_gather_hetero_float_bf16
FFI_BG_HETERO(_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmv_scatter_homo_bool_bf16
FFI_BS_HOMO  (_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_scatter_hetero_bool_bf16
FFI_BS_HETERO(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE binary_fcnmv_scatter_homo_float_bf16
FFI_BS_HOMO  (_float_bf16, __nv_bfloat16, __nv_bfloat16)
// @BE binary_fcnmv_scatter_hetero_float_bf16
FFI_BS_HETERO(_float_bf16, __nv_bfloat16, __nv_bfloat16)

