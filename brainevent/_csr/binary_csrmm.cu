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
 * binary_csrmm.cu -- Event-Driven Binary CSR Sparse Matrix-Matrix CUDA Kernels
 * =============================================================================
 *
 * This module provides optimized CUDA kernels for event-driven sparse
 * matrix-matrix operations in Compressed Sparse Row (CSR) format.
 *
 * Operation: binary_csrmm
 *   Sparse Matrix-Matrix Product (SpMM): computes C = A @ B (NT) or
 *   C = A^T @ B (T), where entries corresponding to active (nonzero) elements
 *   in the dense input matrix B contribute to the output matrix C.
 *
 * Python API parameters:
 *   weights  -- 1-D (homo) or 1-D (hetero, length == nnz) weight array
 *   indices  -- column indices of CSR non-zeros (int32, length == nnz)
 *   indptr   -- row pointer array (int32, length == m+1)
 *   B        -- dense input matrix (bool/int8 or float, shape [m, n] or [k, n])
 *   C        -- dense output matrix (same dtype as weights, shape [m, n] or [k, n])
 *   stream   -- CUDA stream handle (int64)
 *
 * Optimizations:
 *   - Multi-row per block: 4 warps x 32 lanes = 128 threads per block
 *   - Grid-stride loop: caps grid.x at CSRMM_MAX_GRID_X to reduce scheduling
 *   - Unrolled inner loops for instruction-level parallelism
 */

#include "cuda_common.h"
#include "brainevent/common.h"

// Maximum grid.x dimension — caps block count for reduced scheduling overhead.
// 4096 blocks x 128 threads = 524K threads, saturates all SMs.
#define CSRMM_MAX_GRID_X 4096

// Rows per block for warp kernels (4 warps x 32 lanes = 128 threads)
#define CSRMM_WARP_RPB 4

// =========================================================================
// Homogeneous Weight Kernels (weights.size == 1)
// =========================================================================

#define DEFINE_CSRMM_NT_WARP_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, \
                                   READ_W, WRITE_W, ACC_ZERO)                  \
__global__ void _csrmm_nt_warp_homo_kern##SUFFIX(                              \
    const WEIGHT_T* __restrict__ weights,                                      \
    const int32_t*  __restrict__ indices,                                      \
    const int32_t*  __restrict__ indptr,                                       \
    const SPIKE_T*  __restrict__ B,                                            \
    WEIGHT_T*       __restrict__ C,                                            \
    int m, int n                                                               \
) {                                                                            \
    int warp_id   = threadIdx.x >> 5;                                          \
    int lane      = threadIdx.x & 31;                                          \
    int rpb       = blockDim.x >> 5;                                           \
    int col_start = blockIdx.y * 32;                                           \
    int c         = col_start + lane;                                          \
    if (c >= n) return;                                                        \
    ACC_T w = READ_W(weights[0]);                                              \
    for (int base = blockIdx.x * rpb; base < m; base += gridDim.x * rpb) {    \
        int row = base + warp_id;                                              \
        if (row >= m) continue;                                                \
        int start = indptr[row], end = indptr[row + 1];                        \
        ACC_T acc0 = ACC_ZERO, acc1 = ACC_ZERO;                                \
        int j = start;                                                         \
        _Pragma("unroll 4")                                                    \
        for (; j + 1 < end; j += 2) {                                          \
            ACC_T mask0 = (ACC_T)IS_ACTIVE(B[indices[j]   * n + c]);           \
            ACC_T mask1 = (ACC_T)IS_ACTIVE(B[indices[j+1] * n + c]);          \
            acc0 += w * mask0;                                                 \
            acc1 += w * mask1;                                                 \
        }                                                                      \
        for (; j < end; j++) {                                                 \
            ACC_T mask = (ACC_T)IS_ACTIVE(B[indices[j] * n + c]);              \
            acc0 += w * mask;                                                  \
        }                                                                      \
        C[row * n + c] = WRITE_W(acc0 + acc1);                                 \
    }                                                                          \
}

#define DEFINE_CSRMM_NT_BLOCK_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, \
                                    READ_W, WRITE_W, ACC_ZERO)                  \
__global__ void _csrmm_nt_block_homo_kern##SUFFIX(                              \
    const WEIGHT_T* __restrict__ weights,                                       \
    const int32_t*  __restrict__ indices,                                       \
    const int32_t*  __restrict__ indptr,                                        \
    const SPIKE_T*  __restrict__ B,                                             \
    WEIGHT_T*       __restrict__ C,                                             \
    int m, int n                                                                \
) {                                                                             \
    extern __shared__ char _smem_bytes[];                                       \
    ACC_T* smem = reinterpret_cast<ACC_T*>(_smem_bytes);                        \
    int col_start = blockIdx.y * 32;                                            \
    int lane      = threadIdx.x & 31;                                           \
    int strip     = threadIdx.x >> 5;                                           \
    int c         = col_start + lane;                                           \
    for (int row = blockIdx.x; row < m; row += gridDim.x) {                    \
        int start = indptr[row], end = indptr[row + 1];                        \
        ACC_T acc0 = ACC_ZERO, acc1 = ACC_ZERO;                                \
        if (c < n) {                                                            \
            ACC_T w = READ_W(weights[0]);                                      \
            int j = start + strip;                                              \
            _Pragma("unroll 4")                                                \
            for (; j + 8 < end; j += 16) {                                     \
                ACC_T mask0 = (ACC_T)IS_ACTIVE(B[indices[j]   * n + c]);       \
                ACC_T mask1 = (ACC_T)IS_ACTIVE(B[indices[j+8] * n + c]);      \
                acc0 += w * mask0;                                              \
                acc1 += w * mask1;                                              \
            }                                                                  \
            for (; j < end; j += 8) {                                           \
                ACC_T mask = (ACC_T)IS_ACTIVE(B[indices[j] * n + c]);          \
                acc0 += w * mask;                                               \
            }                                                                  \
        }                                                                       \
        smem[strip * 32 + lane] = acc0 + acc1;                                 \
        __syncthreads();                                                        \
        if (strip == 0 && c < n) {                                              \
            ACC_T sum = ACC_ZERO;                                               \
            _Pragma("unroll 8")                                                \
            for (int s = 0; s < 8; s++) sum += smem[s * 32 + lane];            \
            C[row * n + c] = WRITE_W(sum);                                     \
        }                                                                       \
        __syncthreads();                                                        \
    }                                                                           \
}

#define DEFINE_CSRMM_T_WARP_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, \
                                  READ_W, WRITE_W, ACC_ZERO)                  \
__global__ void _csrmm_t_warp_homo_kern##SUFFIX(                              \
    const WEIGHT_T* __restrict__ weights,                                     \
    const int32_t*  __restrict__ indices,                                     \
    const int32_t*  __restrict__ indptr,                                      \
    const SPIKE_T*  __restrict__ B,                                           \
    WEIGHT_T*       __restrict__ C,                                           \
    int m, int n                                                              \
) {                                                                           \
    int row       = blockIdx.x;                                               \
    int col_start = blockIdx.y * 32;                                          \
    int c         = col_start + (int)threadIdx.x;                             \
    if (row >= m || c >= n) return;                                           \
    if (!IS_ACTIVE(B[row * n + c])) return;                                  \
    int start = indptr[row], end = indptr[row + 1];                          \
    WEIGHT_T w_out = weights[0];                                              \
    for (int j = start; j < end; j++) {                                       \
        atomicAdd(&C[indices[j] * n + c], w_out);                            \
    }                                                                         \
}

// =========================================================================
// Heterogeneous Weight Kernels (weights.size == nnz)
// =========================================================================

#define DEFINE_CSRMM_NT_WARP_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, \
                                     READ_W, WRITE_W, ACC_ZERO)                  \
__global__ void _csrmm_nt_warp_hetero_kern##SUFFIX(                              \
    const WEIGHT_T* __restrict__ weights,                                        \
    const int32_t*  __restrict__ indices,                                        \
    const int32_t*  __restrict__ indptr,                                         \
    const SPIKE_T*  __restrict__ B,                                              \
    WEIGHT_T*       __restrict__ C,                                              \
    int m, int n                                                                 \
) {                                                                              \
    int warp_id   = threadIdx.x >> 5;                                            \
    int lane      = threadIdx.x & 31;                                            \
    int rpb       = blockDim.x >> 5;                                             \
    int col_start = blockIdx.y * 32;                                             \
    int c         = col_start + lane;                                            \
    if (c >= n) return;                                                          \
    for (int base = blockIdx.x * rpb; base < m; base += gridDim.x * rpb) {      \
        int row = base + warp_id;                                                \
        if (row >= m) continue;                                                  \
        int start = indptr[row], end = indptr[row + 1];                          \
        ACC_T acc0 = ACC_ZERO, acc1 = ACC_ZERO;                                  \
        int j = start;                                                           \
        _Pragma("unroll 4")                                                      \
        for (; j + 1 < end; j += 2) {                                            \
            ACC_T mask0 = (ACC_T)IS_ACTIVE(B[indices[j]   * n + c]);             \
            ACC_T mask1 = (ACC_T)IS_ACTIVE(B[indices[j+1] * n + c]);            \
            acc0 += READ_W(weights[j])   * mask0;                                \
            acc1 += READ_W(weights[j+1]) * mask1;                                \
        }                                                                        \
        for (; j < end; j++) {                                                   \
            ACC_T mask = (ACC_T)IS_ACTIVE(B[indices[j] * n + c]);                \
            acc0 += READ_W(weights[j]) * mask;                                   \
        }                                                                        \
        C[row * n + c] = WRITE_W(acc0 + acc1);                                   \
    }                                                                            \
}

#define DEFINE_CSRMM_NT_BLOCK_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, \
                                      READ_W, WRITE_W, ACC_ZERO)                  \
__global__ void _csrmm_nt_block_hetero_kern##SUFFIX(                              \
    const WEIGHT_T* __restrict__ weights,                                         \
    const int32_t*  __restrict__ indices,                                         \
    const int32_t*  __restrict__ indptr,                                          \
    const SPIKE_T*  __restrict__ B,                                               \
    WEIGHT_T*       __restrict__ C,                                               \
    int m, int n                                                                  \
) {                                                                               \
    extern __shared__ char _smem_bytes[];                                         \
    ACC_T* smem = reinterpret_cast<ACC_T*>(_smem_bytes);                          \
    int col_start = blockIdx.y * 32;                                              \
    int lane      = threadIdx.x & 31;                                             \
    int strip     = threadIdx.x >> 5;                                             \
    int c         = col_start + lane;                                             \
    for (int row = blockIdx.x; row < m; row += gridDim.x) {                      \
        int start = indptr[row], end = indptr[row + 1];                          \
        ACC_T acc0 = ACC_ZERO, acc1 = ACC_ZERO;                                  \
        if (c < n) {                                                              \
            int j = start + strip;                                                \
            _Pragma("unroll 4")                                                  \
            for (; j + 8 < end; j += 16) {                                       \
                ACC_T mask0 = (ACC_T)IS_ACTIVE(B[indices[j]   * n + c]);         \
                ACC_T mask1 = (ACC_T)IS_ACTIVE(B[indices[j+8] * n + c]);        \
                acc0 += READ_W(weights[j])   * mask0;                            \
                acc1 += READ_W(weights[j+8]) * mask1;                            \
            }                                                                    \
            for (; j < end; j += 8) {                                             \
                ACC_T mask = (ACC_T)IS_ACTIVE(B[indices[j] * n + c]);            \
                acc0 += READ_W(weights[j]) * mask;                               \
            }                                                                    \
        }                                                                         \
        smem[strip * 32 + lane] = acc0 + acc1;                                   \
        __syncthreads();                                                          \
        if (strip == 0 && c < n) {                                                \
            ACC_T sum = ACC_ZERO;                                                 \
            _Pragma("unroll 8")                                                  \
            for (int s = 0; s < 8; s++) sum += smem[s * 32 + lane];              \
            C[row * n + c] = WRITE_W(sum);                                       \
        }                                                                         \
        __syncthreads();                                                          \
    }                                                                             \
}

#define DEFINE_CSRMM_T_WARP_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, \
                                    READ_W, WRITE_W, ACC_ZERO)                  \
__global__ void _csrmm_t_warp_hetero_kern##SUFFIX(                              \
    const WEIGHT_T* __restrict__ weights,                                       \
    const int32_t*  __restrict__ indices,                                       \
    const int32_t*  __restrict__ indptr,                                        \
    const SPIKE_T*  __restrict__ B,                                             \
    WEIGHT_T*       __restrict__ C,                                             \
    int m, int n                                                                \
) {                                                                             \
    int row       = blockIdx.x;                                                 \
    int col_start = blockIdx.y * 32;                                            \
    int c         = col_start + (int)threadIdx.x;                               \
    if (row >= m || c >= n) return;                                             \
    if (!IS_ACTIVE(B[row * n + c])) return;                                    \
    int start = indptr[row], end = indptr[row + 1];                            \
    for (int j = start; j < end; j++) {                                         \
        atomicAdd(&C[indices[j] * n + c], weights[j]);                         \
    }                                                                           \
}

// =========================================================================
// Kernel Instantiations — Homogeneous Weights
// =========================================================================

// float32 homogeneous
DEFINE_CSRMM_NT_WARP_HOMO(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float, float, \
                           READ_F32, WRITE_F32, 0.0f)
DEFINE_CSRMM_NT_WARP_HOMO(_f32_float, float,  IS_ACTIVE_FLOAT, float, float, \
                           READ_F32, WRITE_F32, 0.0f)
DEFINE_CSRMM_NT_BLOCK_HOMO(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float, float, \
                            READ_F32, WRITE_F32, 0.0f)
DEFINE_CSRMM_NT_BLOCK_HOMO(_f32_float, float,  IS_ACTIVE_FLOAT, float, float, \
                            READ_F32, WRITE_F32, 0.0f)
DEFINE_CSRMM_T_WARP_HOMO(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float, float,   \
                          READ_F32, WRITE_F32, 0.0f)
DEFINE_CSRMM_T_WARP_HOMO(_f32_float, float,  IS_ACTIVE_FLOAT, float, float,   \
                          READ_F32, WRITE_F32, 0.0f)

// float64 homogeneous
DEFINE_CSRMM_NT_WARP_HOMO(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, \
                           READ_F64, WRITE_F64, 0.0)
DEFINE_CSRMM_NT_WARP_HOMO(_f64_float, float,  IS_ACTIVE_FLOAT, double, double, \
                           READ_F64, WRITE_F64, 0.0)
DEFINE_CSRMM_NT_BLOCK_HOMO(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, \
                            READ_F64, WRITE_F64, 0.0)
DEFINE_CSRMM_NT_BLOCK_HOMO(_f64_float, float,  IS_ACTIVE_FLOAT, double, double, \
                            READ_F64, WRITE_F64, 0.0)
DEFINE_CSRMM_T_WARP_HOMO(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double,   \
                          READ_F64, WRITE_F64, 0.0)
DEFINE_CSRMM_T_WARP_HOMO(_f64_float, float,  IS_ACTIVE_FLOAT, double, double,   \
                          READ_F64, WRITE_F64, 0.0)

// float16 homogeneous
DEFINE_CSRMM_NT_WARP_HOMO(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float, \
                           READ_F16, WRITE_F16, 0.0f)
DEFINE_CSRMM_NT_WARP_HOMO(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float, \
                           READ_F16, WRITE_F16, 0.0f)
DEFINE_CSRMM_NT_BLOCK_HOMO(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float, \
                            READ_F16, WRITE_F16, 0.0f)
DEFINE_CSRMM_NT_BLOCK_HOMO(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float, \
                            READ_F16, WRITE_F16, 0.0f)
DEFINE_CSRMM_T_WARP_HOMO(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float,   \
                          READ_F16, WRITE_F16, 0.0f)
DEFINE_CSRMM_T_WARP_HOMO(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float,   \
                          READ_F16, WRITE_F16, 0.0f)

// bfloat16 homogeneous
DEFINE_CSRMM_NT_WARP_HOMO(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, \
                           READ_BF16, WRITE_BF16, 0.0f)
DEFINE_CSRMM_NT_WARP_HOMO(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, \
                           READ_BF16, WRITE_BF16, 0.0f)
DEFINE_CSRMM_NT_BLOCK_HOMO(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, \
                            READ_BF16, WRITE_BF16, 0.0f)
DEFINE_CSRMM_NT_BLOCK_HOMO(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, \
                            READ_BF16, WRITE_BF16, 0.0f)
DEFINE_CSRMM_T_WARP_HOMO(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float,   \
                          READ_BF16, WRITE_BF16, 0.0f)
DEFINE_CSRMM_T_WARP_HOMO(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float,   \
                          READ_BF16, WRITE_BF16, 0.0f)

// =========================================================================
// Kernel Instantiations — Heterogeneous Weights
// =========================================================================

// float32 heterogeneous
DEFINE_CSRMM_NT_WARP_HETERO(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float, float, \
                             READ_F32, WRITE_F32, 0.0f)
DEFINE_CSRMM_NT_WARP_HETERO(_f32_float, float,  IS_ACTIVE_FLOAT, float, float, \
                             READ_F32, WRITE_F32, 0.0f)
DEFINE_CSRMM_NT_BLOCK_HETERO(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float, float, \
                              READ_F32, WRITE_F32, 0.0f)
DEFINE_CSRMM_NT_BLOCK_HETERO(_f32_float, float,  IS_ACTIVE_FLOAT, float, float, \
                              READ_F32, WRITE_F32, 0.0f)
DEFINE_CSRMM_T_WARP_HETERO(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float, float,   \
                            READ_F32, WRITE_F32, 0.0f)
DEFINE_CSRMM_T_WARP_HETERO(_f32_float, float,  IS_ACTIVE_FLOAT, float, float,   \
                            READ_F32, WRITE_F32, 0.0f)

// float64 heterogeneous
DEFINE_CSRMM_NT_WARP_HETERO(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, \
                             READ_F64, WRITE_F64, 0.0)
DEFINE_CSRMM_NT_WARP_HETERO(_f64_float, float,  IS_ACTIVE_FLOAT, double, double, \
                             READ_F64, WRITE_F64, 0.0)
DEFINE_CSRMM_NT_BLOCK_HETERO(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, \
                              READ_F64, WRITE_F64, 0.0)
DEFINE_CSRMM_NT_BLOCK_HETERO(_f64_float, float,  IS_ACTIVE_FLOAT, double, double, \
                              READ_F64, WRITE_F64, 0.0)
DEFINE_CSRMM_T_WARP_HETERO(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double,   \
                            READ_F64, WRITE_F64, 0.0)
DEFINE_CSRMM_T_WARP_HETERO(_f64_float, float,  IS_ACTIVE_FLOAT, double, double,   \
                            READ_F64, WRITE_F64, 0.0)

// float16 heterogeneous
DEFINE_CSRMM_NT_WARP_HETERO(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float, \
                             READ_F16, WRITE_F16, 0.0f)
DEFINE_CSRMM_NT_WARP_HETERO(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float, \
                             READ_F16, WRITE_F16, 0.0f)
DEFINE_CSRMM_NT_BLOCK_HETERO(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float, \
                              READ_F16, WRITE_F16, 0.0f)
DEFINE_CSRMM_NT_BLOCK_HETERO(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float, \
                              READ_F16, WRITE_F16, 0.0f)
DEFINE_CSRMM_T_WARP_HETERO(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float,   \
                            READ_F16, WRITE_F16, 0.0f)
DEFINE_CSRMM_T_WARP_HETERO(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float,   \
                            READ_F16, WRITE_F16, 0.0f)

// bfloat16 heterogeneous
DEFINE_CSRMM_NT_WARP_HETERO(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, \
                             READ_BF16, WRITE_BF16, 0.0f)
DEFINE_CSRMM_NT_WARP_HETERO(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, \
                             READ_BF16, WRITE_BF16, 0.0f)
DEFINE_CSRMM_NT_BLOCK_HETERO(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, \
                              READ_BF16, WRITE_BF16, 0.0f)
DEFINE_CSRMM_NT_BLOCK_HETERO(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, \
                              READ_BF16, WRITE_BF16, 0.0f)
DEFINE_CSRMM_T_WARP_HETERO(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float,   \
                            READ_BF16, WRITE_BF16, 0.0f)
DEFINE_CSRMM_T_WARP_HETERO(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float,   \
                            READ_BF16, WRITE_BF16, 0.0f)


// =========================================================================
// FFI Entry Points — Homogeneous Weights
// =========================================================================

#define FFI_CSRMM_NT_WARP_HOMO(SUFFIX, WEIGHT_C_T, SPIKE_C_T)             \
void binary_csrmm_nt_warp_homo##SUFFIX(                                    \
    const BE::Tensor weights, const BE::Tensor indices,                    \
    const BE::Tensor indptr,  const BE::Tensor B,                          \
    BE::Tensor C,       int64_t stream                                     \
) {                                                                        \
    cudaStream_t s   = reinterpret_cast<cudaStream_t>(stream);             \
    int m        = static_cast<int>(indptr.size(0)) - 1;                   \
    int n        = static_cast<int>(B.size(1));                            \
    int c_blocks = (n + 31) / 32;                                          \
    int rpb      = CSRMM_WARP_RPB;                                         \
    int gx       = (m + rpb - 1) / rpb;                                    \
    if (gx > CSRMM_MAX_GRID_X) gx = CSRMM_MAX_GRID_X;                    \
    if (gx < 1) gx = 1;                                                   \
    dim3 grid(gx, c_blocks);                                               \
    _csrmm_nt_warp_homo_kern##SUFFIX<<<grid, rpb * 32, 0, s>>>(           \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                \
        static_cast<const int32_t*>(indices.data_ptr()),                   \
        static_cast<const int32_t*>(indptr.data_ptr()),                    \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                       \
        static_cast<WEIGHT_C_T*>(C.data_ptr()),                            \
        m, n);                                                             \
}

#define FFI_CSRMM_NT_BLOCK_HOMO(SUFFIX, WEIGHT_C_T, SPIKE_C_T, SHM_SIZE)  \
void binary_csrmm_nt_block_homo##SUFFIX(                                   \
    const BE::Tensor weights, const BE::Tensor indices,                    \
    const BE::Tensor indptr,  const BE::Tensor B,                          \
    BE::Tensor C,       int64_t stream                                     \
) {                                                                        \
    cudaStream_t s   = reinterpret_cast<cudaStream_t>(stream);             \
    int m        = static_cast<int>(indptr.size(0)) - 1;                   \
    int n        = static_cast<int>(B.size(1));                            \
    int c_blocks = (n + 31) / 32;                                          \
    int gx       = m;                                                      \
    if (gx > CSRMM_MAX_GRID_X) gx = CSRMM_MAX_GRID_X;                    \
    if (gx < 1) gx = 1;                                                   \
    dim3 grid(gx, c_blocks);                                               \
    _csrmm_nt_block_homo_kern##SUFFIX<<<grid, 256, SHM_SIZE, s>>>(        \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                \
        static_cast<const int32_t*>(indices.data_ptr()),                   \
        static_cast<const int32_t*>(indptr.data_ptr()),                    \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                       \
        static_cast<WEIGHT_C_T*>(C.data_ptr()),                            \
        m, n);                                                             \
}

#define FFI_CSRMM_NT_AUTO_HOMO(SUFFIX, WEIGHT_C_T, SPIKE_C_T, SHM_SIZE)         \
void binary_csrmm_nt_auto_homo##SUFFIX(                                         \
    const BE::Tensor weights, const BE::Tensor indices,                         \
    const BE::Tensor indptr,  const BE::Tensor B,                               \
    BE::Tensor C,       int64_t stream                                          \
) {                                                                             \
    cudaStream_t s   = reinterpret_cast<cudaStream_t>(stream);                  \
    int m        = static_cast<int>(indptr.size(0)) - 1;                        \
    int n        = static_cast<int>(B.size(1));                                 \
    int nse      = static_cast<int>(indices.size(0));                           \
    int avg_nnz  = (m > 0) ? (nse / m) : 0;                                     \
    int c_blocks = (n + 31) / 32;                                               \
    const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*    d_i = static_cast<const int32_t*>(indices.data_ptr());    \
    const int32_t*    d_p = static_cast<const int32_t*>(indptr.data_ptr());     \
    const SPIKE_C_T*  d_b = static_cast<const SPIKE_C_T*>(B.data_ptr());        \
    WEIGHT_C_T*       d_c = static_cast<WEIGHT_C_T*>(C.data_ptr());             \
    if (avg_nnz <= 512) {                                                       \
        int rpb = CSRMM_WARP_RPB;                                               \
        int gx  = (m + rpb - 1) / rpb;                                          \
        if (gx > CSRMM_MAX_GRID_X) gx = CSRMM_MAX_GRID_X;                      \
        if (gx < 1) gx = 1;                                                     \
        dim3 grid(gx, c_blocks);                                                \
        _csrmm_nt_warp_homo_kern##SUFFIX<<<grid, rpb * 32, 0, s>>>(             \
            d_w, d_i, d_p, d_b, d_c, m, n);                                     \
    } else {                                                                    \
        int gx = m;                                                             \
        if (gx > CSRMM_MAX_GRID_X) gx = CSRMM_MAX_GRID_X;                      \
        if (gx < 1) gx = 1;                                                     \
        dim3 grid(gx, c_blocks);                                                \
        _csrmm_nt_block_homo_kern##SUFFIX<<<grid, 256, SHM_SIZE, s>>>(          \
            d_w, d_i, d_p, d_b, d_c, m, n);                                     \
    }                                                                           \
}

#define FFI_CSRMM_T_WARP_HOMO(SUFFIX, WEIGHT_C_T, SPIKE_C_T)              \
void binary_csrmm_t_warp_homo##SUFFIX(                                     \
    const BE::Tensor weights, const BE::Tensor indices,                    \
    const BE::Tensor indptr,  const BE::Tensor B,                          \
    BE::Tensor C,       int64_t stream                                     \
) {                                                                        \
    cudaStream_t s   = reinterpret_cast<cudaStream_t>(stream);             \
    int m        = static_cast<int>(indptr.size(0)) - 1;                   \
    int n        = static_cast<int>(B.size(1));                            \
    int k        = static_cast<int>(C.size(0));                            \
    int c_blocks = (n + 31) / 32;                                          \
    dim3 grid(m, c_blocks);                                                \
    WEIGHT_C_T* d_c = static_cast<WEIGHT_C_T*>(C.data_ptr());             \
    cudaMemsetAsync(d_c, 0, (size_t)k * n * sizeof(WEIGHT_C_T), s);       \
    _csrmm_t_warp_homo_kern##SUFFIX<<<grid, 32, 0, s>>>(                  \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                \
        static_cast<const int32_t*>(indices.data_ptr()),                   \
        static_cast<const int32_t*>(indptr.data_ptr()),                    \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                       \
        d_c, m, n);                                                        \
}

// =========================================================================
// FFI Entry Points — Heterogeneous Weights
// =========================================================================

#define FFI_CSRMM_NT_WARP_HETERO(SUFFIX, WEIGHT_C_T, SPIKE_C_T)           \
void binary_csrmm_nt_warp_hetero##SUFFIX(                                  \
    const BE::Tensor weights, const BE::Tensor indices,                    \
    const BE::Tensor indptr,  const BE::Tensor B,                          \
    BE::Tensor C,       int64_t stream                                     \
) {                                                                        \
    cudaStream_t s   = reinterpret_cast<cudaStream_t>(stream);             \
    int m        = static_cast<int>(indptr.size(0)) - 1;                   \
    int n        = static_cast<int>(B.size(1));                            \
    int c_blocks = (n + 31) / 32;                                          \
    int rpb      = CSRMM_WARP_RPB;                                         \
    int gx       = (m + rpb - 1) / rpb;                                    \
    if (gx > CSRMM_MAX_GRID_X) gx = CSRMM_MAX_GRID_X;                    \
    if (gx < 1) gx = 1;                                                   \
    dim3 grid(gx, c_blocks);                                               \
    _csrmm_nt_warp_hetero_kern##SUFFIX<<<grid, rpb * 32, 0, s>>>(         \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                \
        static_cast<const int32_t*>(indices.data_ptr()),                   \
        static_cast<const int32_t*>(indptr.data_ptr()),                    \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                       \
        static_cast<WEIGHT_C_T*>(C.data_ptr()),                            \
        m, n);                                                             \
}

#define FFI_CSRMM_NT_BLOCK_HETERO(SUFFIX, WEIGHT_C_T, SPIKE_C_T, SHM_SIZE) \
void binary_csrmm_nt_block_hetero##SUFFIX(                                  \
    const BE::Tensor weights, const BE::Tensor indices,                     \
    const BE::Tensor indptr,  const BE::Tensor B,                           \
    BE::Tensor C,       int64_t stream                                      \
) {                                                                         \
    cudaStream_t s   = reinterpret_cast<cudaStream_t>(stream);              \
    int m        = static_cast<int>(indptr.size(0)) - 1;                    \
    int n        = static_cast<int>(B.size(1));                             \
    int c_blocks = (n + 31) / 32;                                           \
    int gx       = m;                                                       \
    if (gx > CSRMM_MAX_GRID_X) gx = CSRMM_MAX_GRID_X;                     \
    if (gx < 1) gx = 1;                                                    \
    dim3 grid(gx, c_blocks);                                                \
    _csrmm_nt_block_hetero_kern##SUFFIX<<<grid, 256, SHM_SIZE, s>>>(       \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                 \
        static_cast<const int32_t*>(indices.data_ptr()),                    \
        static_cast<const int32_t*>(indptr.data_ptr()),                     \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                        \
        static_cast<WEIGHT_C_T*>(C.data_ptr()),                             \
        m, n);                                                              \
}

#define FFI_CSRMM_NT_AUTO_HETERO(SUFFIX, WEIGHT_C_T, SPIKE_C_T, SHM_SIZE)       \
void binary_csrmm_nt_auto_hetero##SUFFIX(                                       \
    const BE::Tensor weights, const BE::Tensor indices,                         \
    const BE::Tensor indptr,  const BE::Tensor B,                               \
    BE::Tensor C,       int64_t stream                                          \
) {                                                                             \
    cudaStream_t s   = reinterpret_cast<cudaStream_t>(stream);                  \
    int m        = static_cast<int>(indptr.size(0)) - 1;                        \
    int n        = static_cast<int>(B.size(1));                                 \
    int nse      = static_cast<int>(indices.size(0));                           \
    int avg_nnz  = (m > 0) ? (nse / m) : 0;                                     \
    int c_blocks = (n + 31) / 32;                                               \
    const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*    d_i = static_cast<const int32_t*>(indices.data_ptr());    \
    const int32_t*    d_p = static_cast<const int32_t*>(indptr.data_ptr());     \
    const SPIKE_C_T*  d_b = static_cast<const SPIKE_C_T*>(B.data_ptr());        \
    WEIGHT_C_T*       d_c = static_cast<WEIGHT_C_T*>(C.data_ptr());             \
    if (avg_nnz <= 512) {                                                       \
        int rpb = CSRMM_WARP_RPB;                                               \
        int gx  = (m + rpb - 1) / rpb;                                          \
        if (gx > CSRMM_MAX_GRID_X) gx = CSRMM_MAX_GRID_X;                      \
        if (gx < 1) gx = 1;                                                     \
        dim3 grid(gx, c_blocks);                                                \
        _csrmm_nt_warp_hetero_kern##SUFFIX<<<grid, rpb * 32, 0, s>>>(           \
            d_w, d_i, d_p, d_b, d_c, m, n);                                     \
    } else {                                                                    \
        int gx = m;                                                             \
        if (gx > CSRMM_MAX_GRID_X) gx = CSRMM_MAX_GRID_X;                      \
        if (gx < 1) gx = 1;                                                     \
        dim3 grid(gx, c_blocks);                                                \
        _csrmm_nt_block_hetero_kern##SUFFIX<<<grid, 256, SHM_SIZE, s>>>(        \
            d_w, d_i, d_p, d_b, d_c, m, n);                                     \
    }                                                                           \
}

#define FFI_CSRMM_T_WARP_HETERO(SUFFIX, WEIGHT_C_T, SPIKE_C_T)            \
void binary_csrmm_t_warp_hetero##SUFFIX(                                   \
    const BE::Tensor weights, const BE::Tensor indices,                    \
    const BE::Tensor indptr,  const BE::Tensor B,                          \
    BE::Tensor C,       int64_t stream                                     \
) {                                                                        \
    cudaStream_t s   = reinterpret_cast<cudaStream_t>(stream);             \
    int m        = static_cast<int>(indptr.size(0)) - 1;                   \
    int n        = static_cast<int>(B.size(1));                            \
    int k        = static_cast<int>(C.size(0));                            \
    int c_blocks = (n + 31) / 32;                                          \
    dim3 grid(m, c_blocks);                                                \
    WEIGHT_C_T* d_c = static_cast<WEIGHT_C_T*>(C.data_ptr());             \
    cudaMemsetAsync(d_c, 0, (size_t)k * n * sizeof(WEIGHT_C_T), s);       \
    _csrmm_t_warp_hetero_kern##SUFFIX<<<grid, 32, 0, s>>>(                \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                \
        static_cast<const int32_t*>(indices.data_ptr()),                   \
        static_cast<const int32_t*>(indptr.data_ptr()),                    \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                       \
        d_c, m, n);                                                        \
}

// =========================================================================
// FFI Instantiations — Homogeneous Weights
// =========================================================================

// float32 homogeneous
// @BE binary_csrmm_nt_warp_homo_f32_bool
FFI_CSRMM_NT_WARP_HOMO(_f32_bool,  float,  int8_t)
// @BE binary_csrmm_nt_warp_homo_f32_float
FFI_CSRMM_NT_WARP_HOMO(_f32_float, float,  float)
// @BE binary_csrmm_nt_block_homo_f32_bool
FFI_CSRMM_NT_BLOCK_HOMO(_f32_bool,  float,  int8_t, 8 * sizeof(float))
// @BE binary_csrmm_nt_block_homo_f32_float
FFI_CSRMM_NT_BLOCK_HOMO(_f32_float, float,  float,  8 * sizeof(float))
// @BE binary_csrmm_nt_auto_homo_f32_bool
FFI_CSRMM_NT_AUTO_HOMO(_f32_bool,  float,  int8_t, 8 * sizeof(float))
// @BE binary_csrmm_nt_auto_homo_f32_float
FFI_CSRMM_NT_AUTO_HOMO(_f32_float, float,  float,  8 * sizeof(float))
// @BE binary_csrmm_t_warp_homo_f32_bool
FFI_CSRMM_T_WARP_HOMO(_f32_bool,  float,  int8_t)
// @BE binary_csrmm_t_warp_homo_f32_float
FFI_CSRMM_T_WARP_HOMO(_f32_float, float,  float)

// float64 homogeneous
// @BE binary_csrmm_nt_auto_homo_f64_bool
FFI_CSRMM_NT_AUTO_HOMO(_f64_bool,  double, int8_t, 8 * sizeof(double))
// @BE binary_csrmm_nt_auto_homo_f64_float
FFI_CSRMM_NT_AUTO_HOMO(_f64_float, double, float,  8 * sizeof(double))
// @BE binary_csrmm_t_warp_homo_f64_bool
FFI_CSRMM_T_WARP_HOMO(_f64_bool,  double, int8_t)
// @BE binary_csrmm_t_warp_homo_f64_float
FFI_CSRMM_T_WARP_HOMO(_f64_float, double, float)

// float16 homogeneous
// @BE binary_csrmm_nt_auto_homo_f16_bool
FFI_CSRMM_NT_AUTO_HOMO(_f16_bool,  __half, int8_t, 8 * sizeof(float))
// @BE binary_csrmm_nt_auto_homo_f16_float
FFI_CSRMM_NT_AUTO_HOMO(_f16_float, __half, float,  8 * sizeof(float))
// @BE binary_csrmm_t_warp_homo_f16_bool
FFI_CSRMM_T_WARP_HOMO(_f16_bool,  __half, int8_t)
// @BE binary_csrmm_t_warp_homo_f16_float
FFI_CSRMM_T_WARP_HOMO(_f16_float, __half, float)

// bfloat16 homogeneous
// @BE binary_csrmm_nt_auto_homo_bf16_bool
FFI_CSRMM_NT_AUTO_HOMO(_bf16_bool,  __nv_bfloat16, int8_t, 8 * sizeof(float))
// @BE binary_csrmm_nt_auto_homo_bf16_float
FFI_CSRMM_NT_AUTO_HOMO(_bf16_float, __nv_bfloat16, float,  8 * sizeof(float))
// @BE binary_csrmm_t_warp_homo_bf16_bool
FFI_CSRMM_T_WARP_HOMO(_bf16_bool,  __nv_bfloat16, int8_t)
// @BE binary_csrmm_t_warp_homo_bf16_float
FFI_CSRMM_T_WARP_HOMO(_bf16_float, __nv_bfloat16, float)

// =========================================================================
// FFI Instantiations — Heterogeneous Weights
// =========================================================================

// float32 heterogeneous
// @BE binary_csrmm_nt_warp_hetero_f32_bool
FFI_CSRMM_NT_WARP_HETERO(_f32_bool,  float,  int8_t)
// @BE binary_csrmm_nt_warp_hetero_f32_float
FFI_CSRMM_NT_WARP_HETERO(_f32_float, float,  float)
// @BE binary_csrmm_nt_block_hetero_f32_bool
FFI_CSRMM_NT_BLOCK_HETERO(_f32_bool,  float,  int8_t, 8 * sizeof(float))
// @BE binary_csrmm_nt_block_hetero_f32_float
FFI_CSRMM_NT_BLOCK_HETERO(_f32_float, float,  float,  8 * sizeof(float))
// @BE binary_csrmm_nt_auto_hetero_f32_bool
FFI_CSRMM_NT_AUTO_HETERO(_f32_bool,  float,  int8_t, 8 * sizeof(float))
// @BE binary_csrmm_nt_auto_hetero_f32_float
FFI_CSRMM_NT_AUTO_HETERO(_f32_float, float,  float,  8 * sizeof(float))
// @BE binary_csrmm_t_warp_hetero_f32_bool
FFI_CSRMM_T_WARP_HETERO(_f32_bool,  float,  int8_t)
// @BE binary_csrmm_t_warp_hetero_f32_float
FFI_CSRMM_T_WARP_HETERO(_f32_float, float,  float)

// float64 heterogeneous
// @BE binary_csrmm_nt_auto_hetero_f64_bool
FFI_CSRMM_NT_AUTO_HETERO(_f64_bool,  double, int8_t, 8 * sizeof(double))
// @BE binary_csrmm_nt_auto_hetero_f64_float
FFI_CSRMM_NT_AUTO_HETERO(_f64_float, double, float,  8 * sizeof(double))
// @BE binary_csrmm_t_warp_hetero_f64_bool
FFI_CSRMM_T_WARP_HETERO(_f64_bool,  double, int8_t)
// @BE binary_csrmm_t_warp_hetero_f64_float
FFI_CSRMM_T_WARP_HETERO(_f64_float, double, float)

// float16 heterogeneous
// @BE binary_csrmm_nt_auto_hetero_f16_bool
FFI_CSRMM_NT_AUTO_HETERO(_f16_bool,  __half, int8_t, 8 * sizeof(float))
// @BE binary_csrmm_nt_auto_hetero_f16_float
FFI_CSRMM_NT_AUTO_HETERO(_f16_float, __half, float,  8 * sizeof(float))
// @BE binary_csrmm_t_warp_hetero_f16_bool
FFI_CSRMM_T_WARP_HETERO(_f16_bool,  __half, int8_t)
// @BE binary_csrmm_t_warp_hetero_f16_float
FFI_CSRMM_T_WARP_HETERO(_f16_float, __half, float)

// bfloat16 heterogeneous
// @BE binary_csrmm_nt_auto_hetero_bf16_bool
FFI_CSRMM_NT_AUTO_HETERO(_bf16_bool,  __nv_bfloat16, int8_t, 8 * sizeof(float))
// @BE binary_csrmm_nt_auto_hetero_bf16_float
FFI_CSRMM_NT_AUTO_HETERO(_bf16_float, __nv_bfloat16, float,  8 * sizeof(float))
// @BE binary_csrmm_t_warp_hetero_bf16_bool
FFI_CSRMM_T_WARP_HETERO(_bf16_bool,  __nv_bfloat16, int8_t)
// @BE binary_csrmm_t_warp_hetero_bf16_float
FFI_CSRMM_T_WARP_HETERO(_bf16_float, __nv_bfloat16, float)
