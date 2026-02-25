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
 * binary_coomm.cu -- Event-Driven Binary COO Sparse Matrix-Matrix CUDA Kernels
 * =============================================================================
 *
 * This module provides high-performance, event-driven CUDA kernels for sparse
 * matrix-matrix (SpMM) multiplication where the sparse matrix is in Coordinate
 * (COO) format and the dense matrix contains binary events (spikes).
 *
 * Event-Driven Optimization:
 * -------------------------
 * In SNN simulations, dense matrices are often very sparse in time
 * (most elements are zero/inactive). These kernels exploit this by checking
 * the activity of the dense matrix before performing expensive atomic
 * accumulations. This "event-driven" approach significantly reduces memory
 * traffic and contention on the output buffer.
 *
 * Kernel Variants:
 * ---------------
 * Each variant is provided in two weight modes:
 *   - Homogeneous (homo): a single scalar weight data[0] is broadcast to all
 *     connections, eliminating per-NNZ weight loads.
 *   - Heterogeneous (hetero): per-connection weights data[s] are loaded for
 *     each NNZ entry.
 *
 * Supported Operations:
 * --------------------
 * binary_coomm (SpMM): out = A @ B  or  out = A.T @ B
 *   - Column-Tiled (CT) Variant: Optimized for small number of columns (n <= 64).
 *     Uses warp-level voting (__ballot_sync) to skip entire tiles of inactive events.
 *   - Warp-Per-Entry (WPE) Variant: Optimized for large number of columns (n > 64).
 *     Assigns one warp per NNZ entry to maximize parallelism.
 *
 * Data Types and Numerical Stability:
 * ----------------------------------
 * - Supports float32, float64, float16 (sm_70+), and bfloat16 (sm_80+).
 * - For reduced-precision types (f16, bf16), accumulation is performed in
 *   float32 to maintain numerical precision, with results written back
 *   atomically.
 *
 * TVM FFI Integration:
 * -------------------
 * All kernels are exposed via TVM FFI with @tvm_ffi annotations for seamless
 * integration with JAX.  Homo vs. hetero dispatch is resolved at compile time
 * on the Python side (based on weight_info.size), so there is no runtime
 * is_homo branch in the kernels.
 */

#include "cuda_common.h"
#include "brainevent/common.h"

// ============================================================================
// COO Matrix-Matrix Multiplication (coomm) — block size constants
// ============================================================================

#define COOMM_CT_BLOCK_K   16
#define COOMM_CT_BLOCK_N   32
#define COOMM_WPE_WARPS    8
#define COOMM_WPE_COLS     32

// ============================================================================
// Homogeneous kernels — scalar weight data[0] broadcast to all connections
// ============================================================================

#define DEFINE_COOMM_HOMO_CT_NT(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _coomm_homo_ct_nt_kern##SUFFIX(                                                    \
    const WEIGHT_T* __restrict__ data,                                                             \
    const int32_t*  __restrict__ row,                                                              \
    const int32_t*  __restrict__ col,                                                              \
    const SPIKE_T*  __restrict__ B,                                                                \
    WEIGHT_T*                    out,                                                              \
    int nnz, int n                                                                                 \
) {                                                                                                \
    int nnz_start = blockIdx.x * COOMM_CT_BLOCK_K;                                                 \
    int col_start = blockIdx.y * COOMM_CT_BLOCK_N;                                                 \
    int t         = threadIdx.x;                                                                   \
    int my_col    = col_start + t;                                                                 \
    bool col_valid = (my_col < n);                                                                 \
    ACC_T homo_w = READ_W(data[0]);                                                                \
    int nnz_end = nnz_start + COOMM_CT_BLOCK_K;                                                    \
    if (nnz_end > nnz) nnz_end = nnz;                                                              \
    for (int s = nnz_start; s < nnz_end; s++) {                                                    \
        int src = col[s];                                                                          \
        int dst = row[s];                                                                          \
        SPIKE_T spike = col_valid ? B[(int64_t)src * n + my_col] : (SPIKE_T)0;                     \
        bool active = IS_ACTIVE(spike) && col_valid;                                               \
        uint32_t ballot = __ballot_sync(0xffffffff, active);                                       \
        if (ballot == 0u) continue;                                                                \
        if (active) {                                                                              \
            ATOMIC_ADD_W(out + (int64_t)dst * n + my_col, homo_w);                                 \
        }                                                                                          \
    }                                                                                              \
}

#define DEFINE_COOMM_HOMO_CT_T(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _coomm_homo_ct_t_kern##SUFFIX(                                                    \
    const WEIGHT_T* __restrict__ data,                                                            \
    const int32_t*  __restrict__ row,                                                             \
    const int32_t*  __restrict__ col,                                                             \
    const SPIKE_T*  __restrict__ B,                                                               \
    WEIGHT_T*                    out,                                                             \
    int nnz, int n                                                                                \
) {                                                                                               \
    int nnz_start = blockIdx.x * COOMM_CT_BLOCK_K;                                                \
    int col_start = blockIdx.y * COOMM_CT_BLOCK_N;                                                \
    int t         = threadIdx.x;                                                                  \
    int my_col    = col_start + t;                                                                \
    bool col_valid = (my_col < n);                                                                \
    ACC_T homo_w = READ_W(data[0]);                                                               \
    int nnz_end = nnz_start + COOMM_CT_BLOCK_K;                                                   \
    if (nnz_end > nnz) nnz_end = nnz;                                                             \
    for (int s = nnz_start; s < nnz_end; s++) {                                                   \
        int src = row[s];                                                                         \
        int dst = col[s];                                                                         \
        SPIKE_T spike = col_valid ? B[(int64_t)src * n + my_col] : (SPIKE_T)0;                    \
        bool active = IS_ACTIVE(spike) && col_valid;                                              \
        uint32_t ballot = __ballot_sync(0xffffffff, active);                                      \
        if (ballot == 0u) continue;                                                               \
        if (active) {                                                                             \
            ATOMIC_ADD_W(out + (int64_t)dst * n + my_col, homo_w);                                \
        }                                                                                         \
    }                                                                                             \
}

#define DEFINE_COOMM_HOMO_WPE_NT(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _coomm_homo_wpe_nt_kern##SUFFIX(                                                    \
    const WEIGHT_T* __restrict__ data,                                                              \
    const int32_t*  __restrict__ row,                                                               \
    const int32_t*  __restrict__ col,                                                               \
    const SPIKE_T*  __restrict__ B,                                                                 \
    WEIGHT_T*                    out,                                                               \
    int nnz, int n                                                                                  \
) {                                                                                                 \
    int warp_id  = (int)(blockIdx.x * COOMM_WPE_WARPS) + (int)(threadIdx.x >> 5);                   \
    int lane     = threadIdx.x & 31;                                                                \
    int col_start = blockIdx.y * COOMM_WPE_COLS;                                                    \
    int my_col    = col_start + lane;                                                               \
    if (warp_id >= nnz) return;                                                                     \
    bool col_valid = (my_col < n);                                                                  \
    int s   = warp_id;                                                                              \
    int src = col[s];                                                                               \
    int dst = row[s];                                                                               \
    SPIKE_T spike = col_valid ? B[(int64_t)src * n + my_col] : (SPIKE_T)0;                          \
    bool active = IS_ACTIVE(spike) && col_valid;                                                    \
    uint32_t ballot = __ballot_sync(0xffffffff, active);                                            \
    if (ballot == 0u) return;                                                                       \
    if (active) {                                                                                   \
        ATOMIC_ADD_W(out + (int64_t)dst * n + my_col, READ_W(data[0]));                             \
    }                                                                                               \
}

#define DEFINE_COOMM_HOMO_WPE_T(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _coomm_homo_wpe_t_kern##SUFFIX(                                                    \
    const WEIGHT_T* __restrict__ data,                                                             \
    const int32_t*  __restrict__ row,                                                              \
    const int32_t*  __restrict__ col,                                                              \
    const SPIKE_T*  __restrict__ B,                                                                \
    WEIGHT_T*                    out,                                                              \
    int nnz, int n                                                                                 \
) {                                                                                                \
    int warp_id   = (int)(blockIdx.x * COOMM_WPE_WARPS) + (int)(threadIdx.x >> 5);                 \
    int lane      = threadIdx.x & 31;                                                              \
    int col_start = blockIdx.y * COOMM_WPE_COLS;                                                   \
    int my_col    = col_start + lane;                                                              \
    if (warp_id >= nnz) return;                                                                    \
    bool col_valid = (my_col < n);                                                                 \
    int s   = warp_id;                                                                             \
    int src = row[s];                                                                              \
    int dst = col[s];                                                                              \
    SPIKE_T spike = col_valid ? B[(int64_t)src * n + my_col] : (SPIKE_T)0;                         \
    bool active = IS_ACTIVE(spike) && col_valid;                                                   \
    uint32_t ballot = __ballot_sync(0xffffffff, active);                                           \
    if (ballot == 0u) return;                                                                      \
    if (active) {                                                                                  \
        ATOMIC_ADD_W(out + (int64_t)dst * n + my_col, READ_W(data[0]));                            \
    }                                                                                              \
}

// ============================================================================
// Heterogeneous kernels — per-connection weight data[s]
// ============================================================================

#define DEFINE_COOMM_HETERO_CT_NT(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _coomm_hetero_ct_nt_kern##SUFFIX(                                                    \
    const WEIGHT_T* __restrict__ data,                                                               \
    const int32_t*  __restrict__ row,                                                                \
    const int32_t*  __restrict__ col,                                                                \
    const SPIKE_T*  __restrict__ B,                                                                  \
    WEIGHT_T*                    out,                                                                \
    int nnz, int n                                                                                   \
) {                                                                                                  \
    int nnz_start = blockIdx.x * COOMM_CT_BLOCK_K;                                                   \
    int col_start = blockIdx.y * COOMM_CT_BLOCK_N;                                                   \
    int t         = threadIdx.x;                                                                     \
    int my_col    = col_start + t;                                                                   \
    bool col_valid = (my_col < n);                                                                   \
    int nnz_end = nnz_start + COOMM_CT_BLOCK_K;                                                      \
    if (nnz_end > nnz) nnz_end = nnz;                                                                \
    for (int s = nnz_start; s < nnz_end; s++) {                                                      \
        int src = col[s];                                                                            \
        int dst = row[s];                                                                            \
        SPIKE_T spike = col_valid ? B[(int64_t)src * n + my_col] : (SPIKE_T)0;                       \
        bool active = IS_ACTIVE(spike) && col_valid;                                                 \
        uint32_t ballot = __ballot_sync(0xffffffff, active);                                         \
        if (ballot == 0u) continue;                                                                  \
        if (active) {                                                                                \
            ATOMIC_ADD_W(out + (int64_t)dst * n + my_col, READ_W(data[s]));                          \
        }                                                                                            \
    }                                                                                                \
}

#define DEFINE_COOMM_HETERO_CT_T(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _coomm_hetero_ct_t_kern##SUFFIX(                                                    \
    const WEIGHT_T* __restrict__ data,                                                              \
    const int32_t*  __restrict__ row,                                                               \
    const int32_t*  __restrict__ col,                                                               \
    const SPIKE_T*  __restrict__ B,                                                                 \
    WEIGHT_T*                    out,                                                               \
    int nnz, int n                                                                                  \
) {                                                                                                 \
    int nnz_start = blockIdx.x * COOMM_CT_BLOCK_K;                                                  \
    int col_start = blockIdx.y * COOMM_CT_BLOCK_N;                                                  \
    int t         = threadIdx.x;                                                                    \
    int my_col    = col_start + t;                                                                  \
    bool col_valid = (my_col < n);                                                                  \
    int nnz_end = nnz_start + COOMM_CT_BLOCK_K;                                                     \
    if (nnz_end > nnz) nnz_end = nnz;                                                               \
    for (int s = nnz_start; s < nnz_end; s++) {                                                     \
        int src = row[s];                                                                           \
        int dst = col[s];                                                                           \
        SPIKE_T spike = col_valid ? B[(int64_t)src * n + my_col] : (SPIKE_T)0;                      \
        bool active = IS_ACTIVE(spike) && col_valid;                                                \
        uint32_t ballot = __ballot_sync(0xffffffff, active);                                        \
        if (ballot == 0u) continue;                                                                 \
        if (active) {                                                                               \
            ATOMIC_ADD_W(out + (int64_t)dst * n + my_col, READ_W(data[s]));                         \
        }                                                                                           \
    }                                                                                               \
}

#define DEFINE_COOMM_HETERO_WPE_NT(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _coomm_hetero_wpe_nt_kern##SUFFIX(                                                    \
    const WEIGHT_T* __restrict__ data,                                                                \
    const int32_t*  __restrict__ row,                                                                 \
    const int32_t*  __restrict__ col,                                                                 \
    const SPIKE_T*  __restrict__ B,                                                                   \
    WEIGHT_T*                    out,                                                                 \
    int nnz, int n                                                                                    \
) {                                                                                                   \
    int warp_id  = (int)(blockIdx.x * COOMM_WPE_WARPS) + (int)(threadIdx.x >> 5);                     \
    int lane     = threadIdx.x & 31;                                                                  \
    int col_start = blockIdx.y * COOMM_WPE_COLS;                                                      \
    int my_col    = col_start + lane;                                                                 \
    if (warp_id >= nnz) return;                                                                       \
    bool col_valid = (my_col < n);                                                                    \
    int s   = warp_id;                                                                                \
    int src = col[s];                                                                                 \
    int dst = row[s];                                                                                 \
    SPIKE_T spike = col_valid ? B[(int64_t)src * n + my_col] : (SPIKE_T)0;                            \
    bool active = IS_ACTIVE(spike) && col_valid;                                                      \
    uint32_t ballot = __ballot_sync(0xffffffff, active);                                              \
    if (ballot == 0u) return;                                                                         \
    if (active) {                                                                                     \
        ATOMIC_ADD_W(out + (int64_t)dst * n + my_col, READ_W(data[s]));                               \
    }                                                                                                 \
}

#define DEFINE_COOMM_HETERO_WPE_T(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _coomm_hetero_wpe_t_kern##SUFFIX(                                                    \
    const WEIGHT_T* __restrict__ data,                                                               \
    const int32_t*  __restrict__ row,                                                                \
    const int32_t*  __restrict__ col,                                                                \
    const SPIKE_T*  __restrict__ B,                                                                  \
    WEIGHT_T*                    out,                                                                \
    int nnz, int n                                                                                   \
) {                                                                                                  \
    int warp_id   = (int)(blockIdx.x * COOMM_WPE_WARPS) + (int)(threadIdx.x >> 5);                   \
    int lane      = threadIdx.x & 31;                                                                \
    int col_start = blockIdx.y * COOMM_WPE_COLS;                                                     \
    int my_col    = col_start + lane;                                                                \
    if (warp_id >= nnz) return;                                                                      \
    bool col_valid = (my_col < n);                                                                   \
    int s   = warp_id;                                                                               \
    int src = row[s];                                                                                \
    int dst = col[s];                                                                                \
    SPIKE_T spike = col_valid ? B[(int64_t)src * n + my_col] : (SPIKE_T)0;                           \
    bool active = IS_ACTIVE(spike) && col_valid;                                                     \
    uint32_t ballot = __ballot_sync(0xffffffff, active);                                             \
    if (ballot == 0u) return;                                                                        \
    if (active) {                                                                                    \
        ATOMIC_ADD_W(out + (int64_t)dst * n + my_col, READ_W(data[s]));                              \
    }                                                                                                \
}

// ============================================================================
// Kernel instantiations — homogeneous
// ============================================================================

DEFINE_COOMM_HOMO_CT_NT(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_HOMO_CT_NT(_f32_float, float,  IS_ACTIVE_FLOAT, float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_HOMO_CT_T (_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_HOMO_CT_T (_f32_float, float,  IS_ACTIVE_FLOAT, float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_HOMO_WPE_NT(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_HOMO_WPE_NT(_f32_float, float,  IS_ACTIVE_FLOAT, float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_HOMO_WPE_T (_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_HOMO_WPE_T (_f32_float, float,  IS_ACTIVE_FLOAT, float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_HOMO_CT_NT(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_HOMO_CT_NT(_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_HOMO_CT_T (_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_HOMO_CT_T (_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_HOMO_WPE_NT(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_HOMO_WPE_NT(_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_HOMO_WPE_T (_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_HOMO_WPE_T (_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_HOMO_CT_NT(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_HOMO_CT_NT(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_HOMO_CT_T (_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_HOMO_CT_T (_f16_float, float,  IS_ACTIVE_FLOAT, __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_HOMO_WPE_NT(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_HOMO_WPE_NT(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_HOMO_WPE_T (_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_HOMO_WPE_T (_f16_float, float,  IS_ACTIVE_FLOAT, __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_HOMO_CT_NT(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMM_HOMO_CT_NT(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMM_HOMO_CT_T (_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMM_HOMO_CT_T (_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMM_HOMO_WPE_NT(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMM_HOMO_WPE_NT(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMM_HOMO_WPE_T (_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMM_HOMO_WPE_T (_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)

// ============================================================================
// Kernel instantiations — heterogeneous
// ============================================================================

DEFINE_COOMM_HETERO_CT_NT(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_HETERO_CT_NT(_f32_float, float,  IS_ACTIVE_FLOAT, float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_HETERO_CT_T (_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_HETERO_CT_T (_f32_float, float,  IS_ACTIVE_FLOAT, float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_HETERO_WPE_NT(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_HETERO_WPE_NT(_f32_float, float,  IS_ACTIVE_FLOAT, float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_HETERO_WPE_T (_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_HETERO_WPE_T (_f32_float, float,  IS_ACTIVE_FLOAT, float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_HETERO_CT_NT(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_HETERO_CT_NT(_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_HETERO_CT_T (_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_HETERO_CT_T (_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_HETERO_WPE_NT(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_HETERO_WPE_NT(_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_HETERO_WPE_T (_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_HETERO_WPE_T (_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_HETERO_CT_NT(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_HETERO_CT_NT(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_HETERO_CT_T (_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_HETERO_CT_T (_f16_float, float,  IS_ACTIVE_FLOAT, __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_HETERO_WPE_NT(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_HETERO_WPE_NT(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_HETERO_WPE_T (_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_HETERO_WPE_T (_f16_float, float,  IS_ACTIVE_FLOAT, __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_HETERO_CT_NT(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMM_HETERO_CT_NT(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMM_HETERO_CT_T (_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMM_HETERO_CT_T (_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMM_HETERO_WPE_NT(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMM_HETERO_WPE_NT(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMM_HETERO_WPE_T (_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMM_HETERO_WPE_T (_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)

// ============================================================================
// FFI entry points — homogeneous
// ============================================================================

#define FFI_COOMM_HOMO_CT_NT(SUFFIX, WEIGHT_C_T, SPIKE_C_T, OUT_BYTES_PER_ELEM) \
void binary_coomm_homo_ct_nt##SUFFIX(                                           \
    const BE::Tensor data,                                                  \
    const BE::Tensor row_idx,                                               \
    const BE::Tensor col_idx,                                               \
    const BE::Tensor B,                                                     \
    BE::Tensor output,                                                \
    int64_t stream                                                              \
) {                                                                             \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                    \
    int nnz  = static_cast<int>(row_idx.size(0));                               \
    int n    = static_cast<int>(B.size(1));                                     \
    int m    = static_cast<int>(output.size(0));                                \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());            \
    cudaMemsetAsync(d_out, 0, (size_t)m * n * OUT_BYTES_PER_ELEM, s);           \
    if (nnz == 0) return;                                                       \
    dim3 block(COOMM_CT_BLOCK_N, 1, 1);                                         \
    dim3 grid(                                                                  \
        (nnz + COOMM_CT_BLOCK_K - 1) / COOMM_CT_BLOCK_K,                        \
        (n   + COOMM_CT_BLOCK_N - 1) / COOMM_CT_BLOCK_N,                        \
        1                                                                       \
    );                                                                          \
    _coomm_homo_ct_nt_kern##SUFFIX<<<grid, block, 0, s>>>(                      \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                        \
        static_cast<const int32_t*>(row_idx.data_ptr()),                        \
        static_cast<const int32_t*>(col_idx.data_ptr()),                        \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                            \
        d_out, nnz, n                                                           \
    );                                                                          \
}

#define FFI_COOMM_HOMO_CT_T(SUFFIX, WEIGHT_C_T, SPIKE_C_T, OUT_BYTES_PER_ELEM) \
void binary_coomm_homo_ct_t##SUFFIX(                                           \
    const BE::Tensor data,                                                 \
    const BE::Tensor row_idx,                                              \
    const BE::Tensor col_idx,                                              \
    const BE::Tensor B,                                                    \
    BE::Tensor output,                                               \
    int64_t stream                                                             \
) {                                                                            \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                   \
    int nnz  = static_cast<int>(row_idx.size(0));                              \
    int n    = static_cast<int>(B.size(1));                                    \
    int k_out = static_cast<int>(output.size(0));                              \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());           \
    cudaMemsetAsync(d_out, 0, (size_t)k_out * n * OUT_BYTES_PER_ELEM, s);      \
    if (nnz == 0) return;                                                      \
    dim3 block(COOMM_CT_BLOCK_N, 1, 1);                                        \
    dim3 grid(                                                                 \
        (nnz + COOMM_CT_BLOCK_K - 1) / COOMM_CT_BLOCK_K,                       \
        (n   + COOMM_CT_BLOCK_N - 1) / COOMM_CT_BLOCK_N,                       \
        1                                                                      \
    );                                                                         \
    _coomm_homo_ct_t_kern##SUFFIX<<<grid, block, 0, s>>>(                      \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                       \
        static_cast<const int32_t*>(row_idx.data_ptr()),                       \
        static_cast<const int32_t*>(col_idx.data_ptr()),                       \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                           \
        d_out, nnz, n                                                          \
    );                                                                         \
}

#define FFI_COOMM_HOMO_WPE_NT(SUFFIX, WEIGHT_C_T, SPIKE_C_T, OUT_BYTES_PER_ELEM) \
void binary_coomm_homo_wpe_nt##SUFFIX(                                           \
    const BE::Tensor data,                                                   \
    const BE::Tensor row_idx,                                                \
    const BE::Tensor col_idx,                                                \
    const BE::Tensor B,                                                      \
    BE::Tensor output,                                                 \
    int64_t stream                                                               \
) {                                                                              \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                     \
    int nnz   = static_cast<int>(row_idx.size(0));                               \
    int n     = static_cast<int>(B.size(1));                                     \
    int m     = static_cast<int>(output.size(0));                                \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());             \
    cudaMemsetAsync(d_out, 0, (size_t)m * n * OUT_BYTES_PER_ELEM, s);            \
    if (nnz == 0) return;                                                        \
    dim3 block(COOMM_WPE_WARPS * 32, 1, 1);                                      \
    dim3 grid(                                                                   \
        (nnz + COOMM_WPE_WARPS - 1) / COOMM_WPE_WARPS,                           \
        (n   + COOMM_WPE_COLS  - 1) / COOMM_WPE_COLS,                            \
        1                                                                        \
    );                                                                           \
    _coomm_homo_wpe_nt_kern##SUFFIX<<<grid, block, 0, s>>>(                      \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                         \
        static_cast<const int32_t*>(row_idx.data_ptr()),                         \
        static_cast<const int32_t*>(col_idx.data_ptr()),                         \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                             \
        d_out, nnz, n                                                            \
    );                                                                           \
}

#define FFI_COOMM_HOMO_WPE_T(SUFFIX, WEIGHT_C_T, SPIKE_C_T, OUT_BYTES_PER_ELEM) \
void binary_coomm_homo_wpe_t##SUFFIX(                                           \
    const BE::Tensor data,                                                  \
    const BE::Tensor row_idx,                                               \
    const BE::Tensor col_idx,                                               \
    const BE::Tensor B,                                                     \
    BE::Tensor output,                                                \
    int64_t stream                                                              \
) {                                                                             \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                    \
    int nnz   = static_cast<int>(row_idx.size(0));                              \
    int n     = static_cast<int>(B.size(1));                                    \
    int k_out = static_cast<int>(output.size(0));                               \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());            \
    cudaMemsetAsync(d_out, 0, (size_t)k_out * n * OUT_BYTES_PER_ELEM, s);       \
    if (nnz == 0) return;                                                       \
    dim3 block(COOMM_WPE_WARPS * 32, 1, 1);                                     \
    dim3 grid(                                                                  \
        (nnz + COOMM_WPE_WARPS - 1) / COOMM_WPE_WARPS,                          \
        (n   + COOMM_WPE_COLS  - 1) / COOMM_WPE_COLS,                           \
        1                                                                       \
    );                                                                          \
    _coomm_homo_wpe_t_kern##SUFFIX<<<grid, block, 0, s>>>(                      \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                        \
        static_cast<const int32_t*>(row_idx.data_ptr()),                        \
        static_cast<const int32_t*>(col_idx.data_ptr()),                        \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                            \
        d_out, nnz, n                                                           \
    );                                                                          \
}

// ============================================================================
// FFI entry points — heterogeneous
// ============================================================================

#define FFI_COOMM_HETERO_CT_NT(SUFFIX, WEIGHT_C_T, SPIKE_C_T, OUT_BYTES_PER_ELEM) \
void binary_coomm_hetero_ct_nt##SUFFIX(                                           \
    const BE::Tensor data,                                                    \
    const BE::Tensor row_idx,                                                 \
    const BE::Tensor col_idx,                                                 \
    const BE::Tensor B,                                                       \
    BE::Tensor output,                                                  \
    int64_t stream                                                                \
) {                                                                               \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                      \
    int nnz  = static_cast<int>(row_idx.size(0));                                 \
    int n    = static_cast<int>(B.size(1));                                       \
    int m    = static_cast<int>(output.size(0));                                  \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());              \
    cudaMemsetAsync(d_out, 0, (size_t)m * n * OUT_BYTES_PER_ELEM, s);             \
    if (nnz == 0) return;                                                         \
    dim3 block(COOMM_CT_BLOCK_N, 1, 1);                                           \
    dim3 grid(                                                                    \
        (nnz + COOMM_CT_BLOCK_K - 1) / COOMM_CT_BLOCK_K,                          \
        (n   + COOMM_CT_BLOCK_N - 1) / COOMM_CT_BLOCK_N,                          \
        1                                                                         \
    );                                                                            \
    _coomm_hetero_ct_nt_kern##SUFFIX<<<grid, block, 0, s>>>(                      \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                          \
        static_cast<const int32_t*>(row_idx.data_ptr()),                          \
        static_cast<const int32_t*>(col_idx.data_ptr()),                          \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                              \
        d_out, nnz, n                                                             \
    );                                                                            \
}

#define FFI_COOMM_HETERO_CT_T(SUFFIX, WEIGHT_C_T, SPIKE_C_T, OUT_BYTES_PER_ELEM) \
void binary_coomm_hetero_ct_t##SUFFIX(                                           \
    const BE::Tensor data,                                                   \
    const BE::Tensor row_idx,                                                \
    const BE::Tensor col_idx,                                                \
    const BE::Tensor B,                                                      \
    BE::Tensor output,                                                 \
    int64_t stream                                                               \
) {                                                                              \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                     \
    int nnz  = static_cast<int>(row_idx.size(0));                                \
    int n    = static_cast<int>(B.size(1));                                      \
    int k_out = static_cast<int>(output.size(0));                                \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());             \
    cudaMemsetAsync(d_out, 0, (size_t)k_out * n * OUT_BYTES_PER_ELEM, s);        \
    if (nnz == 0) return;                                                        \
    dim3 block(COOMM_CT_BLOCK_N, 1, 1);                                          \
    dim3 grid(                                                                   \
        (nnz + COOMM_CT_BLOCK_K - 1) / COOMM_CT_BLOCK_K,                         \
        (n   + COOMM_CT_BLOCK_N - 1) / COOMM_CT_BLOCK_N,                         \
        1                                                                        \
    );                                                                           \
    _coomm_hetero_ct_t_kern##SUFFIX<<<grid, block, 0, s>>>(                      \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                         \
        static_cast<const int32_t*>(row_idx.data_ptr()),                         \
        static_cast<const int32_t*>(col_idx.data_ptr()),                         \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                             \
        d_out, nnz, n                                                            \
    );                                                                           \
}

#define FFI_COOMM_HETERO_WPE_NT(SUFFIX, WEIGHT_C_T, SPIKE_C_T, OUT_BYTES_PER_ELEM) \
void binary_coomm_hetero_wpe_nt##SUFFIX(                                           \
    const BE::Tensor data,                                                     \
    const BE::Tensor row_idx,                                                  \
    const BE::Tensor col_idx,                                                  \
    const BE::Tensor B,                                                        \
    BE::Tensor output,                                                   \
    int64_t stream                                                                 \
) {                                                                                \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                       \
    int nnz   = static_cast<int>(row_idx.size(0));                                 \
    int n     = static_cast<int>(B.size(1));                                       \
    int m     = static_cast<int>(output.size(0));                                  \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());               \
    cudaMemsetAsync(d_out, 0, (size_t)m * n * OUT_BYTES_PER_ELEM, s);              \
    if (nnz == 0) return;                                                          \
    dim3 block(COOMM_WPE_WARPS * 32, 1, 1);                                        \
    dim3 grid(                                                                     \
        (nnz + COOMM_WPE_WARPS - 1) / COOMM_WPE_WARPS,                             \
        (n   + COOMM_WPE_COLS  - 1) / COOMM_WPE_COLS,                              \
        1                                                                          \
    );                                                                             \
    _coomm_hetero_wpe_nt_kern##SUFFIX<<<grid, block, 0, s>>>(                      \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                           \
        static_cast<const int32_t*>(row_idx.data_ptr()),                           \
        static_cast<const int32_t*>(col_idx.data_ptr()),                           \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                               \
        d_out, nnz, n                                                              \
    );                                                                             \
}

#define FFI_COOMM_HETERO_WPE_T(SUFFIX, WEIGHT_C_T, SPIKE_C_T, OUT_BYTES_PER_ELEM) \
void binary_coomm_hetero_wpe_t##SUFFIX(                                           \
    const BE::Tensor data,                                                    \
    const BE::Tensor row_idx,                                                 \
    const BE::Tensor col_idx,                                                 \
    const BE::Tensor B,                                                       \
    BE::Tensor output,                                                  \
    int64_t stream                                                                \
) {                                                                               \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                      \
    int nnz   = static_cast<int>(row_idx.size(0));                                \
    int n     = static_cast<int>(B.size(1));                                      \
    int k_out = static_cast<int>(output.size(0));                                 \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());              \
    cudaMemsetAsync(d_out, 0, (size_t)k_out * n * OUT_BYTES_PER_ELEM, s);         \
    if (nnz == 0) return;                                                         \
    dim3 block(COOMM_WPE_WARPS * 32, 1, 1);                                       \
    dim3 grid(                                                                    \
        (nnz + COOMM_WPE_WARPS - 1) / COOMM_WPE_WARPS,                            \
        (n   + COOMM_WPE_COLS  - 1) / COOMM_WPE_COLS,                             \
        1                                                                         \
    );                                                                            \
    _coomm_hetero_wpe_t_kern##SUFFIX<<<grid, block, 0, s>>>(                      \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                          \
        static_cast<const int32_t*>(row_idx.data_ptr()),                          \
        static_cast<const int32_t*>(col_idx.data_ptr()),                          \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                              \
        d_out, nnz, n                                                             \
    );                                                                            \
}

// ============================================================================
// FFI instantiations — homogeneous
// ============================================================================

// CT-NT homo
// @BE binary_coomm_homo_ct_nt_f32_bool
FFI_COOMM_HOMO_CT_NT(_f32_bool,  float,  int8_t, sizeof(float))
// @BE binary_coomm_homo_ct_nt_f32_float
FFI_COOMM_HOMO_CT_NT(_f32_float, float,  float,  sizeof(float))
// @BE binary_coomm_homo_ct_nt_f64_bool
FFI_COOMM_HOMO_CT_NT(_f64_bool,  double, int8_t, sizeof(double))
// @BE binary_coomm_homo_ct_nt_f64_float
FFI_COOMM_HOMO_CT_NT(_f64_float, double, float,  sizeof(double))
// @BE binary_coomm_homo_ct_nt_f16_bool
FFI_COOMM_HOMO_CT_NT(_f16_bool,  __half, int8_t, sizeof(__half))
// @BE binary_coomm_homo_ct_nt_f16_float
FFI_COOMM_HOMO_CT_NT(_f16_float, __half, float,  sizeof(__half))
// @BE binary_coomm_homo_ct_nt_bf16_bool
FFI_COOMM_HOMO_CT_NT(_bf16_bool,  __nv_bfloat16, int8_t, sizeof(__nv_bfloat16))
// @BE binary_coomm_homo_ct_nt_bf16_float
FFI_COOMM_HOMO_CT_NT(_bf16_float, __nv_bfloat16, float,  sizeof(__nv_bfloat16))

// CT-T homo
// @BE binary_coomm_homo_ct_t_f32_bool
FFI_COOMM_HOMO_CT_T(_f32_bool,  float,  int8_t, sizeof(float))
// @BE binary_coomm_homo_ct_t_f32_float
FFI_COOMM_HOMO_CT_T(_f32_float, float,  float,  sizeof(float))
// @BE binary_coomm_homo_ct_t_f64_bool
FFI_COOMM_HOMO_CT_T(_f64_bool,  double, int8_t, sizeof(double))
// @BE binary_coomm_homo_ct_t_f64_float
FFI_COOMM_HOMO_CT_T(_f64_float, double, float,  sizeof(double))
// @BE binary_coomm_homo_ct_t_f16_bool
FFI_COOMM_HOMO_CT_T(_f16_bool,  __half, int8_t, sizeof(__half))
// @BE binary_coomm_homo_ct_t_f16_float
FFI_COOMM_HOMO_CT_T(_f16_float, __half, float,  sizeof(__half))
// @BE binary_coomm_homo_ct_t_bf16_bool
FFI_COOMM_HOMO_CT_T(_bf16_bool,  __nv_bfloat16, int8_t, sizeof(__nv_bfloat16))
// @BE binary_coomm_homo_ct_t_bf16_float
FFI_COOMM_HOMO_CT_T(_bf16_float, __nv_bfloat16, float,  sizeof(__nv_bfloat16))

// WPE-NT homo
// @BE binary_coomm_homo_wpe_nt_f32_bool
FFI_COOMM_HOMO_WPE_NT(_f32_bool,  float,  int8_t, sizeof(float))
// @BE binary_coomm_homo_wpe_nt_f32_float
FFI_COOMM_HOMO_WPE_NT(_f32_float, float,  float,  sizeof(float))
// @BE binary_coomm_homo_wpe_nt_f64_bool
FFI_COOMM_HOMO_WPE_NT(_f64_bool,  double, int8_t, sizeof(double))
// @BE binary_coomm_homo_wpe_nt_f64_float
FFI_COOMM_HOMO_WPE_NT(_f64_float, double, float,  sizeof(double))
// @BE binary_coomm_homo_wpe_nt_f16_bool
FFI_COOMM_HOMO_WPE_NT(_f16_bool,  __half, int8_t, sizeof(__half))
// @BE binary_coomm_homo_wpe_nt_f16_float
FFI_COOMM_HOMO_WPE_NT(_f16_float, __half, float,  sizeof(__half))
// @BE binary_coomm_homo_wpe_nt_bf16_bool
FFI_COOMM_HOMO_WPE_NT(_bf16_bool,  __nv_bfloat16, int8_t, sizeof(__nv_bfloat16))
// @BE binary_coomm_homo_wpe_nt_bf16_float
FFI_COOMM_HOMO_WPE_NT(_bf16_float, __nv_bfloat16, float,  sizeof(__nv_bfloat16))

// WPE-T homo
// @BE binary_coomm_homo_wpe_t_f32_bool
FFI_COOMM_HOMO_WPE_T(_f32_bool,  float,  int8_t, sizeof(float))
// @BE binary_coomm_homo_wpe_t_f32_float
FFI_COOMM_HOMO_WPE_T(_f32_float, float,  float,  sizeof(float))
// @BE binary_coomm_homo_wpe_t_f64_bool
FFI_COOMM_HOMO_WPE_T(_f64_bool,  double, int8_t, sizeof(double))
// @BE binary_coomm_homo_wpe_t_f64_float
FFI_COOMM_HOMO_WPE_T(_f64_float, double, float,  sizeof(double))
// @BE binary_coomm_homo_wpe_t_f16_bool
FFI_COOMM_HOMO_WPE_T(_f16_bool,  __half, int8_t, sizeof(__half))
// @BE binary_coomm_homo_wpe_t_f16_float
FFI_COOMM_HOMO_WPE_T(_f16_float, __half, float,  sizeof(__half))
// @BE binary_coomm_homo_wpe_t_bf16_bool
FFI_COOMM_HOMO_WPE_T(_bf16_bool,  __nv_bfloat16, int8_t, sizeof(__nv_bfloat16))
// @BE binary_coomm_homo_wpe_t_bf16_float
FFI_COOMM_HOMO_WPE_T(_bf16_float, __nv_bfloat16, float,  sizeof(__nv_bfloat16))

// ============================================================================
// FFI instantiations — heterogeneous
// ============================================================================

// CT-NT hetero
// @BE binary_coomm_hetero_ct_nt_f32_bool
FFI_COOMM_HETERO_CT_NT(_f32_bool,  float,  int8_t, sizeof(float))
// @BE binary_coomm_hetero_ct_nt_f32_float
FFI_COOMM_HETERO_CT_NT(_f32_float, float,  float,  sizeof(float))
// @BE binary_coomm_hetero_ct_nt_f64_bool
FFI_COOMM_HETERO_CT_NT(_f64_bool,  double, int8_t, sizeof(double))
// @BE binary_coomm_hetero_ct_nt_f64_float
FFI_COOMM_HETERO_CT_NT(_f64_float, double, float,  sizeof(double))
// @BE binary_coomm_hetero_ct_nt_f16_bool
FFI_COOMM_HETERO_CT_NT(_f16_bool,  __half, int8_t, sizeof(__half))
// @BE binary_coomm_hetero_ct_nt_f16_float
FFI_COOMM_HETERO_CT_NT(_f16_float, __half, float,  sizeof(__half))
// @BE binary_coomm_hetero_ct_nt_bf16_bool
FFI_COOMM_HETERO_CT_NT(_bf16_bool,  __nv_bfloat16, int8_t, sizeof(__nv_bfloat16))
// @BE binary_coomm_hetero_ct_nt_bf16_float
FFI_COOMM_HETERO_CT_NT(_bf16_float, __nv_bfloat16, float,  sizeof(__nv_bfloat16))

// CT-T hetero
// @BE binary_coomm_hetero_ct_t_f32_bool
FFI_COOMM_HETERO_CT_T(_f32_bool,  float,  int8_t, sizeof(float))
// @BE binary_coomm_hetero_ct_t_f32_float
FFI_COOMM_HETERO_CT_T(_f32_float, float,  float,  sizeof(float))
// @BE binary_coomm_hetero_ct_t_f64_bool
FFI_COOMM_HETERO_CT_T(_f64_bool,  double, int8_t, sizeof(double))
// @BE binary_coomm_hetero_ct_t_f64_float
FFI_COOMM_HETERO_CT_T(_f64_float, double, float,  sizeof(double))
// @BE binary_coomm_hetero_ct_t_f16_bool
FFI_COOMM_HETERO_CT_T(_f16_bool,  __half, int8_t, sizeof(__half))
// @BE binary_coomm_hetero_ct_t_f16_float
FFI_COOMM_HETERO_CT_T(_f16_float, __half, float,  sizeof(__half))
// @BE binary_coomm_hetero_ct_t_bf16_bool
FFI_COOMM_HETERO_CT_T(_bf16_bool,  __nv_bfloat16, int8_t, sizeof(__nv_bfloat16))
// @BE binary_coomm_hetero_ct_t_bf16_float
FFI_COOMM_HETERO_CT_T(_bf16_float, __nv_bfloat16, float,  sizeof(__nv_bfloat16))

// WPE-NT hetero
// @BE binary_coomm_hetero_wpe_nt_f32_bool
FFI_COOMM_HETERO_WPE_NT(_f32_bool,  float,  int8_t, sizeof(float))
// @BE binary_coomm_hetero_wpe_nt_f32_float
FFI_COOMM_HETERO_WPE_NT(_f32_float, float,  float,  sizeof(float))
// @BE binary_coomm_hetero_wpe_nt_f64_bool
FFI_COOMM_HETERO_WPE_NT(_f64_bool,  double, int8_t, sizeof(double))
// @BE binary_coomm_hetero_wpe_nt_f64_float
FFI_COOMM_HETERO_WPE_NT(_f64_float, double, float,  sizeof(double))
// @BE binary_coomm_hetero_wpe_nt_f16_bool
FFI_COOMM_HETERO_WPE_NT(_f16_bool,  __half, int8_t, sizeof(__half))
// @BE binary_coomm_hetero_wpe_nt_f16_float
FFI_COOMM_HETERO_WPE_NT(_f16_float, __half, float,  sizeof(__half))
// @BE binary_coomm_hetero_wpe_nt_bf16_bool
FFI_COOMM_HETERO_WPE_NT(_bf16_bool,  __nv_bfloat16, int8_t, sizeof(__nv_bfloat16))
// @BE binary_coomm_hetero_wpe_nt_bf16_float
FFI_COOMM_HETERO_WPE_NT(_bf16_float, __nv_bfloat16, float,  sizeof(__nv_bfloat16))

// WPE-T hetero
// @BE binary_coomm_hetero_wpe_t_f32_bool
FFI_COOMM_HETERO_WPE_T(_f32_bool,  float,  int8_t, sizeof(float))
// @BE binary_coomm_hetero_wpe_t_f32_float
FFI_COOMM_HETERO_WPE_T(_f32_float, float,  float,  sizeof(float))
// @BE binary_coomm_hetero_wpe_t_f64_bool
FFI_COOMM_HETERO_WPE_T(_f64_bool,  double, int8_t, sizeof(double))
// @BE binary_coomm_hetero_wpe_t_f64_float
FFI_COOMM_HETERO_WPE_T(_f64_float, double, float,  sizeof(double))
// @BE binary_coomm_hetero_wpe_t_f16_bool
FFI_COOMM_HETERO_WPE_T(_f16_bool,  __half, int8_t, sizeof(__half))
// @BE binary_coomm_hetero_wpe_t_f16_float
FFI_COOMM_HETERO_WPE_T(_f16_float, __half, float,  sizeof(__half))
// @BE binary_coomm_hetero_wpe_t_bf16_bool
FFI_COOMM_HETERO_WPE_T(_bf16_bool,  __nv_bfloat16, int8_t, sizeof(__nv_bfloat16))
// @BE binary_coomm_hetero_wpe_t_bf16_float
FFI_COOMM_HETERO_WPE_T(_bf16_float, __nv_bfloat16, float,  sizeof(__nv_bfloat16))
