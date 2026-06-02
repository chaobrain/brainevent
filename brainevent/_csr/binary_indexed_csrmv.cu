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
 * binary_indexed_csrmv.cu -- Fused Indexed Event-Driven Binary CSR Mat-Vec Kernels
 * ===============================================================================
 *
 * CUDA backing kernels for ``binary_csrmv_indexed`` (see
 * ``brainevent/_csr/binary_indexed.py``).  These are the *permuted*
 * heterogeneous kernels used in the *unfavorable* direction
 * (``CSR @ event`` / ``event @ CSC``).  The structure (indices/indptr) is the
 * transposed (CSC-like) view, while the weight for structural slot ``j`` lives
 * at the canonical position ``perm[j]``; the gather ``weights[perm[j]]`` is
 * fused into the scatter so only active rows/columns are read.
 *
 * Homogeneous weights need no perm variant (every slot reads ``weights[0]``);
 * the Python dispatcher reuses the plain homogeneous kernels in
 * ``binary_csrmv.cu`` for that case.
 *
 * Python API parameters:
 *   weights  -- 1-D heterogeneous weight array (length == nnz), canonical order
 *   indices  -- column indices of CSR non-zeros (int32, length == nnz)
 *   indptr   -- row pointer array (int32, length == m+1)
 *   perm     -- permutation mapping slot j -> canonical weight index (int32)
 *   vector   -- dense input vector (bool/int8 or float)
 *   output   -- dense output vector (same dtype as weights)
 *   stream   -- CUDA stream handle (int64)
 */

#include "cuda_common.h"
#include "brainevent/common.h"

// =========================================================================
// Indexed (perm) Heterogeneous Kernels — weights read via weights[perm[j]]
// =========================================================================

#define DEFINE_CSRMV_NT_THREAD_PERM_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, \
                                            READ_W, WRITE_W, ACC_ZERO)                   \
__global__ void _csrmv_nt_thread_perm_hetero_kern##SUFFIX(                               \
    const WEIGHT_T* __restrict__ weights,                                               \
    const int32_t*  __restrict__ indices,                                               \
    const int32_t*  __restrict__ indptr,                                                \
    const int32_t*  __restrict__ perm,                                                  \
    const SPIKE_T*  __restrict__ vector,                                                \
    WEIGHT_T*       __restrict__ output,                                                \
    int m                                                                               \
) {                                                                                     \
    int row = blockIdx.x * blockDim.x + threadIdx.x;                                    \
    if (row >= m) return;                                                               \
    int start = indptr[row], end = indptr[row + 1];                                     \
    if (start == end) { output[row] = WRITE_W(ACC_ZERO); return; }                      \
    ACC_T acc = ACC_ZERO;                                                               \
    _Pragma("unroll 4")                                                                 \
    for (int j = start; j < end; j++) {                                                 \
        ACC_T mask = (ACC_T)IS_ACTIVE(vector[indices[j]]);                              \
        acc += READ_W(weights[perm[j]]) * mask;                                         \
    }                                                                                   \
    output[row] = WRITE_W(acc);                                                         \
}

#define DEFINE_CSRMV_NT_BLOCK_PERM_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, \
                                           READ_W, WRITE_W, WARP_RED, ACC_ZERO)         \
__global__ void _csrmv_nt_block_perm_hetero_kern##SUFFIX(                               \
    const WEIGHT_T* __restrict__ weights,                                              \
    const int32_t*  __restrict__ indices,                                              \
    const int32_t*  __restrict__ indptr,                                               \
    const int32_t*  __restrict__ perm,                                                 \
    const SPIKE_T*  __restrict__ vector,                                               \
    WEIGHT_T*       __restrict__ output,                                               \
    int m                                                                              \
) {                                                                                    \
    extern __shared__ char _smem_bytes[];                                              \
    ACC_T* smem_red = reinterpret_cast<ACC_T*>(_smem_bytes);                           \
    int row = blockIdx.x;                                                              \
    if (row >= m) return;                                                              \
    int start = indptr[row], end = indptr[row + 1];                                    \
    if (start == end) {                                                                \
        if (threadIdx.x == 0) output[row] = WRITE_W(ACC_ZERO);                         \
        return;                                                                        \
    }                                                                                  \
    ACC_T acc = ACC_ZERO;                                                              \
    _Pragma("unroll 2")                                                                \
    for (int j = start + (int)threadIdx.x; j < end; j += blockDim.x) {                 \
        ACC_T mask = (ACC_T)IS_ACTIVE(vector[indices[j]]);                             \
        acc += READ_W(weights[perm[j]]) * mask;                                        \
    }                                                                                  \
    int lane   = threadIdx.x & 31;                                                     \
    int warpid = threadIdx.x >> 5;                                                     \
    acc = WARP_RED(acc);                                                               \
    if (lane == 0) smem_red[warpid] = acc;                                             \
    __syncthreads();                                                                   \
    int n_warps = (blockDim.x + 31) >> 5;                                              \
    acc = (threadIdx.x < n_warps) ? smem_red[lane] : ACC_ZERO;                         \
    if (warpid == 0) acc = WARP_RED(acc);                                              \
    if (threadIdx.x == 0) output[row] = WRITE_W(acc);                                  \
}

#define DEFINE_CSRMV_T_WARP_PERM_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, \
                                         READ_W, WRITE_W, ACC_ZERO)                   \
__global__ void _csrmv_t_warp_perm_hetero_kern##SUFFIX(                               \
    const WEIGHT_T* __restrict__ weights,                                            \
    const int32_t*  __restrict__ indices,                                            \
    const int32_t*  __restrict__ indptr,                                             \
    const int32_t*  __restrict__ perm,                                               \
    const SPIKE_T*  __restrict__ vector,                                             \
    WEIGHT_T*       __restrict__ output,                                             \
    int m                                                                            \
) {                                                                                  \
    int row = blockIdx.x * blockDim.x + threadIdx.x;                                 \
    if (row >= m) return;                                                            \
    if (!IS_ACTIVE(vector[row])) return;                                             \
    int start = indptr[row], end = indptr[row + 1];                                  \
    if (start == end) return;                                                        \
    for (int j = start; j < end; j++) {                                              \
        atomicAdd(&output[indices[j]], WRITE_W(READ_W(weights[perm[j]])));           \
    }                                                                                \
}

// float32 perm heterogeneous
DEFINE_CSRMV_NT_THREAD_PERM_HETERO(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float, float, \
                                    READ_F32, WRITE_F32, 0.0f)
DEFINE_CSRMV_NT_THREAD_PERM_HETERO(_f32_float, float,  IS_ACTIVE_FLOAT, float, float, \
                                    READ_F32, WRITE_F32, 0.0f)
DEFINE_CSRMV_NT_BLOCK_PERM_HETERO(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float, float,  \
                                   READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_CSRMV_NT_BLOCK_PERM_HETERO(_f32_float, float,  IS_ACTIVE_FLOAT, float, float,  \
                                   READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_CSRMV_T_WARP_PERM_HETERO(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float, float,    \
                                 READ_F32, WRITE_F32, 0.0f)
DEFINE_CSRMV_T_WARP_PERM_HETERO(_f32_float, float,  IS_ACTIVE_FLOAT, float, float,    \
                                 READ_F32, WRITE_F32, 0.0f)

// float64 perm heterogeneous
DEFINE_CSRMV_NT_THREAD_PERM_HETERO(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, \
                                    READ_F64, WRITE_F64, 0.0)
DEFINE_CSRMV_NT_THREAD_PERM_HETERO(_f64_float, float,  IS_ACTIVE_FLOAT, double, double, \
                                    READ_F64, WRITE_F64, 0.0)
DEFINE_CSRMV_NT_BLOCK_PERM_HETERO(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double,  \
                                   READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_CSRMV_NT_BLOCK_PERM_HETERO(_f64_float, float,  IS_ACTIVE_FLOAT, double, double,  \
                                   READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_CSRMV_T_WARP_PERM_HETERO(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double,    \
                                 READ_F64, WRITE_F64, 0.0)
DEFINE_CSRMV_T_WARP_PERM_HETERO(_f64_float, float,  IS_ACTIVE_FLOAT, double, double,    \
                                 READ_F64, WRITE_F64, 0.0)

// float16 perm heterogeneous
DEFINE_CSRMV_NT_THREAD_PERM_HETERO(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float, \
                                    READ_F16, WRITE_F16, 0.0f)
DEFINE_CSRMV_NT_THREAD_PERM_HETERO(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float, \
                                    READ_F16, WRITE_F16, 0.0f)
DEFINE_CSRMV_NT_BLOCK_PERM_HETERO(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float,  \
                                   READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_CSRMV_NT_BLOCK_PERM_HETERO(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float,  \
                                   READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_CSRMV_T_WARP_PERM_HETERO(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float,    \
                                 READ_F16, WRITE_F16, 0.0f)
DEFINE_CSRMV_T_WARP_PERM_HETERO(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float,    \
                                 READ_F16, WRITE_F16, 0.0f)

// bfloat16 perm heterogeneous
DEFINE_CSRMV_NT_THREAD_PERM_HETERO(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, \
                                    READ_BF16, WRITE_BF16, 0.0f)
DEFINE_CSRMV_NT_THREAD_PERM_HETERO(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, \
                                    READ_BF16, WRITE_BF16, 0.0f)
DEFINE_CSRMV_NT_BLOCK_PERM_HETERO(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float,  \
                                   READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_CSRMV_NT_BLOCK_PERM_HETERO(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float,  \
                                   READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_CSRMV_T_WARP_PERM_HETERO(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float,    \
                                 READ_BF16, WRITE_BF16, 0.0f)
DEFINE_CSRMV_T_WARP_PERM_HETERO(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float,    \
                                 READ_BF16, WRITE_BF16, 0.0f)

// =========================================================================
// FFI Entry Points — Indexed (perm) Heterogeneous Weights
// =========================================================================

#define FFI_CSRMV_NT_AUTO_PERM_HETERO(SUFFIX, WEIGHT_C_T, SPIKE_C_T, SHM_SIZE)       \
void binary_csrmv_nt_auto_perm_hetero##SUFFIX(                                       \
    const BE::Tensor weights, const BE::Tensor indices,                             \
    const BE::Tensor indptr,  const BE::Tensor perm,                                \
    const BE::Tensor vector,  BE::Tensor output,  int64_t stream                    \
) {                                                                                  \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                        \
    int m          = static_cast<int>(indptr.size(0)) - 1;                          \
    int nse        = static_cast<int>(indices.size(0));                             \
    int avg_nnz    = (m > 0) ? (nse / m) : 0;                                       \
    const WEIGHT_C_T* d_w    = static_cast<const WEIGHT_C_T*>(weights.data_ptr());  \
    const int32_t*    d_i    = static_cast<const int32_t*>(indices.data_ptr());     \
    const int32_t*    d_p    = static_cast<const int32_t*>(indptr.data_ptr());      \
    const int32_t*    d_perm = static_cast<const int32_t*>(perm.data_ptr());        \
    const SPIKE_C_T*  d_v    = static_cast<const SPIKE_C_T*>(vector.data_ptr());    \
    WEIGHT_C_T*       d_o    = static_cast<WEIGHT_C_T*>(output.data_ptr());         \
    if (avg_nnz <= 512) {                                                           \
        int blocks = (m + 255) / 256;                                              \
        _csrmv_nt_thread_perm_hetero_kern##SUFFIX<<<blocks, 256, 0, s>>>(           \
            d_w, d_i, d_p, d_perm, d_v, d_o, m);                                    \
    } else {                                                                        \
        _csrmv_nt_block_perm_hetero_kern##SUFFIX<<<m, 256, SHM_SIZE, s>>>(          \
            d_w, d_i, d_p, d_perm, d_v, d_o, m);                                    \
    }                                                                               \
}

#define FFI_CSRMV_T_WARP_PERM_HETERO(SUFFIX, WEIGHT_C_T, SPIKE_C_T)  \
void binary_csrmv_t_warp_perm_hetero##SUFFIX(                        \
    const BE::Tensor weights, const BE::Tensor indices,             \
    const BE::Tensor indptr,  const BE::Tensor perm,                \
    const BE::Tensor vector,  BE::Tensor output,  int64_t stream    \
) {                                                                 \
    cudaStream_t s    = reinterpret_cast<cudaStream_t>(stream);     \
    int m             = static_cast<int>(indptr.size(0)) - 1;       \
    int k             = static_cast<int>(output.size(0));           \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());\
    cudaMemsetAsync(d_out, 0, (size_t)k * sizeof(WEIGHT_C_T), s);   \
    int blocks = (m + 255) / 256;                                   \
    _csrmv_t_warp_perm_hetero_kern##SUFFIX<<<blocks, 256, 0, s>>>(  \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),         \
        static_cast<const int32_t*>(indices.data_ptr()),            \
        static_cast<const int32_t*>(indptr.data_ptr()),             \
        static_cast<const int32_t*>(perm.data_ptr()),               \
        static_cast<const SPIKE_C_T*>(vector.data_ptr()),           \
        d_out, m);                                                  \
}

// float32 perm heterogeneous
// @BE binary_csrmv_nt_auto_perm_hetero_f32_bool
FFI_CSRMV_NT_AUTO_PERM_HETERO(_f32_bool,  float,  int8_t, 8 * sizeof(float))
// @BE binary_csrmv_nt_auto_perm_hetero_f32_float
FFI_CSRMV_NT_AUTO_PERM_HETERO(_f32_float, float,  float,  8 * sizeof(float))
// @BE binary_csrmv_t_warp_perm_hetero_f32_bool
FFI_CSRMV_T_WARP_PERM_HETERO(_f32_bool,  float,  int8_t)
// @BE binary_csrmv_t_warp_perm_hetero_f32_float
FFI_CSRMV_T_WARP_PERM_HETERO(_f32_float, float,  float)

// float64 perm heterogeneous
// @BE binary_csrmv_nt_auto_perm_hetero_f64_bool
FFI_CSRMV_NT_AUTO_PERM_HETERO(_f64_bool,  double, int8_t, 8 * sizeof(double))
// @BE binary_csrmv_nt_auto_perm_hetero_f64_float
FFI_CSRMV_NT_AUTO_PERM_HETERO(_f64_float, double, float,  8 * sizeof(double))
// @BE binary_csrmv_t_warp_perm_hetero_f64_bool
FFI_CSRMV_T_WARP_PERM_HETERO(_f64_bool,  double, int8_t)
// @BE binary_csrmv_t_warp_perm_hetero_f64_float
FFI_CSRMV_T_WARP_PERM_HETERO(_f64_float, double, float)

// float16 perm heterogeneous
// @BE binary_csrmv_nt_auto_perm_hetero_f16_bool
FFI_CSRMV_NT_AUTO_PERM_HETERO(_f16_bool,  __half, int8_t, 8 * sizeof(float))
// @BE binary_csrmv_nt_auto_perm_hetero_f16_float
FFI_CSRMV_NT_AUTO_PERM_HETERO(_f16_float, __half, float,  8 * sizeof(float))
// @BE binary_csrmv_t_warp_perm_hetero_f16_bool
FFI_CSRMV_T_WARP_PERM_HETERO(_f16_bool,  __half, int8_t)
// @BE binary_csrmv_t_warp_perm_hetero_f16_float
FFI_CSRMV_T_WARP_PERM_HETERO(_f16_float, __half, float)

// bfloat16 perm heterogeneous
// @BE binary_csrmv_nt_auto_perm_hetero_bf16_bool
FFI_CSRMV_NT_AUTO_PERM_HETERO(_bf16_bool,  __nv_bfloat16, int8_t, 8 * sizeof(float))
// @BE binary_csrmv_nt_auto_perm_hetero_bf16_float
FFI_CSRMV_NT_AUTO_PERM_HETERO(_bf16_float, __nv_bfloat16, float,  8 * sizeof(float))
// @BE binary_csrmv_t_warp_perm_hetero_bf16_bool
FFI_CSRMV_T_WARP_PERM_HETERO(_bf16_bool,  __nv_bfloat16, int8_t)
// @BE binary_csrmv_t_warp_perm_hetero_bf16_float
FFI_CSRMV_T_WARP_PERM_HETERO(_bf16_float, __nv_bfloat16, float)
