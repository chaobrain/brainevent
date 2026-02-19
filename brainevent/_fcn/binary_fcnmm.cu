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
 * binary_fcnmm.cu — Event-Driven FCN Sparse Matrix-Matrix CUDA Kernels
 * =====================================================================
 *
 * Python API: brainevent.binary_fcnmm(weights, indices, matrix, *, shape, transpose, backend)
 *
 * Event-driven sparse matrix--matrix product with fixed connection number.
 *
 * Computes  Y = W @ M  (or  Y = W^T @ M  when transpose=True)
 * where W is a sparse weight matrix stored in fixed-connection-number
 * format and M is a dense binary event matrix.  Only the connections
 * to active (spiking) entries contribute to the result.
 *
 * Parameters
 * ----------
 * weights : shape (1,) for homogeneous or (num_pre, num_conn) for heterogeneous,
 *           floating-point dtype (float16 / bfloat16 / float32 / float64).
 * indices : shape (num_pre, num_conn), int32 — post-synaptic column indices.
 * matrix  : shape (num_post, n_batch) for gather or (num_pre, n_batch) for scatter.
 *           bool dtype: active when != 0 (stored as uint8).
 *           float dtype: active when > 0.
 * shape   : logical (num_pre, num_post) shape of the dense weight matrix.
 * transpose=False (gather mode):
 *   Y[i,j] = sum_{k} weights[i,k] * 1_{M[indices[i,k],j] active}
 * transpose=True  (scatter mode):
 *   Y[indices[i,k],j] += weights[i,k] * 1_{M[i,j] active}   for all i,k,j
 *
 * Supported weight dtypes: float32 (_f32), float64 (_f64), float16 (_f16), bfloat16 (_bf16).
 * Supported matrix dtypes: bool (_bool_) and float (_float_) matching weight dtype.
 * Float16 and bfloat16 weight accumulation uses float32 for stability.
 * Bfloat16 requires CUDA 11.0+; bfloat16 atomicAdd requires CC 8.0+.
 *
 * Optimization:
 *   Gather warp  (n_conn ≤ 32): branchless, no __ballot_sync.
 *   Gather basic (n_conn > 32): __ballot_sync per k; skips all-zero warp chunks.
 *   Scatter warp (n_conn ≤ 32): tile-level __ballot_sync early exit.
 *   Scatter basic (n_conn > 32): shared-flag row-level early exit.
 *
 * IMPORTANT: weights.data_ptr() is a GPU device pointer — NEVER dereference on host.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// =========================================================================
// Spike active predicates (matrix elements)
// =========================================================================

#define IS_ACTIVE_BOOL(s)       ((s) != 0)
#define IS_ACTIVE_FLOAT_F32(s)  ((s) > 0.0f)
#define IS_ACTIVE_FLOAT_F64(s)  ((s) > 0.0)
#define IS_ACTIVE_FLOAT_F16(s)  (__half2float(s) > 0.0f)
#define IS_ACTIVE_FLOAT_BF16(s) (__bfloat162float(s) > 0.0f)

// =========================================================================
// Per-dtype weight conversion macros: READ_W converts WEIGHT_T -> ACC_T
// =========================================================================

#define READ_F32(x)   (x)
#define WRITE_F32(x)  (x)

#define READ_F64(x)   (x)
#define WRITE_F64(x)  (x)

#define READ_F16(x)   __half2float(x)
#define WRITE_F16(x)  __float2half(x)

#define READ_BF16(x)  __bfloat162float(x)
#define WRITE_BF16(x) __float2bfloat16(x)

// =========================================================================
// GATHER warp kernel macro (n_conn <= 32, branchless)
//
// Grid: (n_pre, ceil(n_batch/32))  Block: 32
// Thread t handles output Y[row, j], j = blockIdx.y*32 + t.
// Branchless loop over k; inactive lanes contribute 0. No __ballot_sync.
// =========================================================================

#define DEFINE_BGM_WARP(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W) \
__global__ void _bgm_warp_kern##SUFFIX(                                                \
    const int32_t* __restrict__ indices,                                               \
    const SPIKE_T* __restrict__ matrix,                                                \
    WEIGHT_T*      __restrict__ output,                                                \
    const WEIGHT_T* __restrict__ weights,                                              \
    int n_pre, int n_conn, int n_batch, int is_homo                                    \
) {                                                                                    \
    int row = blockIdx.x;                                                              \
    int t   = threadIdx.x;                                                             \
    int j   = (int)blockIdx.y * 32 + t;                                               \
    if (row >= n_pre) return;                                                          \
    bool col_valid = (j < n_batch);                                                    \
    int  safe_j    = col_valid ? j : 0;                                               \
    const int32_t*  i_row = indices + (size_t)row * n_conn;                           \
    const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;       \
    ACC_T accum = (ACC_T)0;                                                            \
    for (int k = 0; k < n_conn; k++) {                                                \
        int  src    = i_row[k];                                                        \
        bool active = col_valid && IS_ACTIVE(matrix[(size_t)src * n_batch + safe_j]); \
        accum += active ? (is_homo ? (ACC_T)1 : READ_W(w_row[k])) : (ACC_T)0;        \
    }                                                                                  \
    if (col_valid)                                                                     \
        output[(size_t)row * n_batch + j] =                                           \
            WRITE_W(is_homo ? (READ_W(weights[0]) * accum) : accum);                  \
}

// =========================================================================
// GATHER basic kernel macro (n_conn > 32, with __ballot_sync per k)
//
// Grid: (n_pre, ceil(n_batch/32))  Block: 32
// __ballot_sync per k across 32 j-threads: if no column active for source
// row indices[i,k], skip weight load (event-driven inner loop).
// =========================================================================

#define DEFINE_BGM_BASIC(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W) \
__global__ void _bgm_basic_kern##SUFFIX(                                                \
    const int32_t* __restrict__ indices,                                                \
    const SPIKE_T* __restrict__ matrix,                                                 \
    WEIGHT_T*      __restrict__ output,                                                 \
    const WEIGHT_T* __restrict__ weights,                                               \
    int n_pre, int n_conn, int n_batch, int is_homo                                     \
) {                                                                                     \
    int row = blockIdx.x;                                                               \
    int t   = threadIdx.x;                                                              \
    int j   = (int)blockIdx.y * 32 + t;                                                \
    if (row >= n_pre) return;                                                           \
    bool col_valid = (j < n_batch);                                                     \
    int  safe_j    = col_valid ? j : 0;                                                \
    const int32_t*  i_row = indices + (size_t)row * n_conn;                            \
    const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;        \
    ACC_T accum = (ACC_T)0;                                                             \
    for (int k = 0; k < n_conn; k++) {                                                 \
        int  src    = i_row[k];                                                         \
        bool active = col_valid && IS_ACTIVE(matrix[(size_t)src * n_batch + safe_j]);  \
        unsigned ballot = __ballot_sync(0xffffffff, active);                            \
        if (ballot == 0) continue;                                                      \
        if (active)                                                                     \
            accum += is_homo ? (ACC_T)1 : READ_W(w_row[k]);                            \
    }                                                                                   \
    if (col_valid)                                                                      \
        output[(size_t)row * n_batch + j] =                                            \
            WRITE_W(is_homo ? (READ_W(weights[0]) * accum) : accum);                   \
}

// =========================================================================
// SCATTER warp kernel macro (n_conn <= 32)
//
// Grid: (n_pre, ceil(n_batch/32))  Block: 32
// Tile-level ballot: skip if no active column in tile.
// Active threads loop over k <= 32 connections -> atomicAdd.
// =========================================================================

#define DEFINE_BSM_WARP(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T)                         \
__global__ void _bsm_warp_kern##SUFFIX(                                                \
    const int32_t* __restrict__ indices,                                               \
    const SPIKE_T* __restrict__ matrix,                                                \
    WEIGHT_T*      __restrict__ output,                                                \
    const WEIGHT_T* __restrict__ weights,                                              \
    int n_pre, int n_conn, int n_batch, int is_homo                                    \
) {                                                                                    \
    int row = blockIdx.x;                                                              \
    int t   = threadIdx.x;                                                             \
    int j   = (int)blockIdx.y * 32 + t;                                               \
    if (row >= n_pre) return;                                                          \
    bool col_valid = (j < n_batch);                                                    \
    int  safe_j    = col_valid ? j : 0;                                               \
    bool active    = col_valid && IS_ACTIVE(matrix[(size_t)row * n_batch + safe_j]);  \
    if (__ballot_sync(0xffffffff, active) == 0) return;                               \
    if (!active) return;                                                               \
    const int32_t*  i_row = indices + (size_t)row * n_conn;                           \
    const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;       \
    WEIGHT_T w0 = is_homo ? weights[0] : (WEIGHT_T)0;                                 \
    for (int k = 0; k < n_conn; k++)                                                  \
        atomicAdd(&output[(size_t)i_row[k] * n_batch + j], is_homo ? w0 : w_row[k]); \
}

// =========================================================================
// SCATTER basic kernel macro (n_conn > 32)
//
// Grid: (n_pre,)  Block: 256  Shared: sizeof(int) for row-active flag.
// Row-level early exit: entire block returns if M[i,:] is all-zero.
// For active rows: sequential j-loop picks active columns; 256 threads
// parallelize inner k-loop with atomicAdd.
// =========================================================================

#define DEFINE_BSM_BASIC(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T)                        \
__global__ void _bsm_basic_kern##SUFFIX(                                               \
    const int32_t* __restrict__ indices,                                               \
    const SPIKE_T* __restrict__ matrix,                                                \
    WEIGHT_T*      __restrict__ output,                                                \
    const WEIGHT_T* __restrict__ weights,                                              \
    int n_pre, int n_conn, int n_batch, int is_homo                                    \
) {                                                                                    \
    extern __shared__ int _smem_flag[];                                                \
    int row = blockIdx.x;                                                              \
    if (row >= n_pre) return;                                                          \
    if (threadIdx.x == 0) _smem_flag[0] = 0;                                          \
    __syncthreads();                                                                   \
    for (int j = threadIdx.x; j < n_batch; j += blockDim.x)                          \
        if (IS_ACTIVE(matrix[(size_t)row * n_batch + j])) {                           \
            atomicOr(_smem_flag, 1); break;                                            \
        }                                                                              \
    __syncthreads();                                                                   \
    if (_smem_flag[0] == 0) return;                                                   \
    const int32_t*  i_row = indices + (size_t)row * n_conn;                           \
    const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;       \
    WEIGHT_T w0 = is_homo ? weights[0] : (WEIGHT_T)0;                                 \
    for (int j = 0; j < n_batch; j++) {                                               \
        if (!IS_ACTIVE(matrix[(size_t)row * n_batch + j])) continue;                  \
        for (int k = threadIdx.x; k < n_conn; k += blockDim.x)                       \
            atomicAdd(&output[(size_t)i_row[k] * n_batch + j], is_homo ? w0 : w_row[k]); \
    }                                                                                  \
}

// =========================================================================
// Instantiate device kernels: 4 weight dtypes x 2 spike types = 8 combos,
//                             2 gather + 2 scatter = 4 kernel shapes each
// =========================================================================

// ---- Float32 ----
DEFINE_BGM_WARP(_bool_warp_f32,   uint8_t, IS_ACTIVE_BOOL,      float,          float,  READ_F32,  WRITE_F32)
DEFINE_BGM_WARP(_float_warp_f32,  float,   IS_ACTIVE_FLOAT_F32, float,          float,  READ_F32,  WRITE_F32)
DEFINE_BGM_BASIC(_bool_basic_f32,  uint8_t, IS_ACTIVE_BOOL,      float,          float,  READ_F32,  WRITE_F32)
DEFINE_BGM_BASIC(_float_basic_f32, float,   IS_ACTIVE_FLOAT_F32, float,          float,  READ_F32,  WRITE_F32)
DEFINE_BSM_WARP(_bool_warp_f32,   uint8_t, IS_ACTIVE_BOOL,      float)
DEFINE_BSM_WARP(_float_warp_f32,  float,   IS_ACTIVE_FLOAT_F32, float)
DEFINE_BSM_BASIC(_bool_basic_f32,  uint8_t, IS_ACTIVE_BOOL,      float)
DEFINE_BSM_BASIC(_float_basic_f32, float,   IS_ACTIVE_FLOAT_F32, float)

// ---- Float64 ----
DEFINE_BGM_WARP(_bool_warp_f64,   uint8_t, IS_ACTIVE_BOOL,      double,         double, READ_F64,  WRITE_F64)
DEFINE_BGM_WARP(_float_warp_f64,  double,  IS_ACTIVE_FLOAT_F64, double,         double, READ_F64,  WRITE_F64)
DEFINE_BGM_BASIC(_bool_basic_f64,  uint8_t, IS_ACTIVE_BOOL,      double,         double, READ_F64,  WRITE_F64)
DEFINE_BGM_BASIC(_float_basic_f64, double,  IS_ACTIVE_FLOAT_F64, double,         double, READ_F64,  WRITE_F64)
DEFINE_BSM_WARP(_bool_warp_f64,   uint8_t, IS_ACTIVE_BOOL,      double)
DEFINE_BSM_WARP(_float_warp_f64,  double,  IS_ACTIVE_FLOAT_F64, double)
DEFINE_BSM_BASIC(_bool_basic_f64,  uint8_t, IS_ACTIVE_BOOL,      double)
DEFINE_BSM_BASIC(_float_basic_f64, double,  IS_ACTIVE_FLOAT_F64, double)

// ---- Float16 (accumulate in float32 for stability) ----
DEFINE_BGM_WARP(_bool_warp_f16,   uint8_t, IS_ACTIVE_BOOL,      __half,         float,  READ_F16,  WRITE_F16)
DEFINE_BGM_WARP(_float_warp_f16,  __half,  IS_ACTIVE_FLOAT_F16, __half,         float,  READ_F16,  WRITE_F16)
DEFINE_BGM_BASIC(_bool_basic_f16,  uint8_t, IS_ACTIVE_BOOL,      __half,         float,  READ_F16,  WRITE_F16)
DEFINE_BGM_BASIC(_float_basic_f16, __half,  IS_ACTIVE_FLOAT_F16, __half,         float,  READ_F16,  WRITE_F16)
DEFINE_BSM_WARP(_bool_warp_f16,   uint8_t, IS_ACTIVE_BOOL,      __half)
DEFINE_BSM_WARP(_float_warp_f16,  __half,  IS_ACTIVE_FLOAT_F16, __half)
DEFINE_BSM_BASIC(_bool_basic_f16,  uint8_t, IS_ACTIVE_BOOL,      __half)
DEFINE_BSM_BASIC(_float_basic_f16, __half,  IS_ACTIVE_FLOAT_F16, __half)

// ---- BFloat16 (accumulate in float32 for stability; requires CUDA 11.0+) ----
// Note: bfloat16 atomicAdd requires CC 8.0+ (Ampere or newer).
DEFINE_BGM_WARP(_bool_warp_bf16,   uint8_t,        IS_ACTIVE_BOOL,       __nv_bfloat16, float, READ_BF16, WRITE_BF16)
DEFINE_BGM_WARP(_float_warp_bf16,  __nv_bfloat16,  IS_ACTIVE_FLOAT_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16)
DEFINE_BGM_BASIC(_bool_basic_bf16,  uint8_t,        IS_ACTIVE_BOOL,       __nv_bfloat16, float, READ_BF16, WRITE_BF16)
DEFINE_BGM_BASIC(_float_basic_bf16, __nv_bfloat16,  IS_ACTIVE_FLOAT_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16)
DEFINE_BSM_WARP(_bool_warp_bf16,   uint8_t,        IS_ACTIVE_BOOL,       __nv_bfloat16)
DEFINE_BSM_WARP(_float_warp_bf16,  __nv_bfloat16,  IS_ACTIVE_FLOAT_BF16, __nv_bfloat16)
DEFINE_BSM_BASIC(_bool_basic_bf16,  uint8_t,        IS_ACTIVE_BOOL,       __nv_bfloat16)
DEFINE_BSM_BASIC(_float_basic_bf16, __nv_bfloat16,  IS_ACTIVE_FLOAT_BF16, __nv_bfloat16)

// =========================================================================
// TVM FFI Entry Point Macros
// =========================================================================
// Convention: args = (weights, indices, matrix, output, stream)
//   weights : WEIGHT_C_T, shape (1,) homo or (n_pre, n_conn) hetero
//   indices : int32,   shape (n_pre, n_conn)
//   matrix  : gather → (n_post, n_batch);  scatter → (n_pre, n_batch)
//   output  : gather → (n_pre, n_batch) WEIGHT_C_T, written directly
//             scatter → (n_post, n_batch) WEIGHT_C_T, zeroed via cudaMemsetAsync
//
// IMPORTANT: data_ptr() returns GPU device memory pointers.
// NEVER dereference on host. Pass unchanged to device kernels.

// Gather warp FFI: void binary_fcnmm_gather##SUFFIX(...)
#define FFI_BGM_WARP(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                  \
void binary_fcnmm_gather##SUFFIX(                                                     \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                       \
    tvm::ffi::TensorView matrix,  tvm::ffi::TensorView output, int64_t stream         \
) {                                                                                   \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                         \
    int n_pre   = static_cast<int>(indices.size(0));                                 \
    int n_conn  = static_cast<int>(indices.size(1));                                 \
    int n_batch = static_cast<int>(matrix.size(1));                                  \
    int is_homo = (weights.ndim() == 1) ? 1 : 0;                                    \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());   \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());       \
    const SPIKE_C_T*  d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr());     \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());           \
    int batch_tiles = (n_batch + 31) / 32;                                           \
    dim3 grid(n_pre, batch_tiles);                                                   \
    _bgm_warp_kern##SUFFIX<<<grid, 32, 0, s>>>(                                       \
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo);                  \
}

// Gather basic FFI: void binary_fcnmm_gather##SUFFIX(...)
#define FFI_BGM_BASIC(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                 \
void binary_fcnmm_gather##SUFFIX(                                                     \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                       \
    tvm::ffi::TensorView matrix,  tvm::ffi::TensorView output, int64_t stream         \
) {                                                                                   \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                         \
    int n_pre   = static_cast<int>(indices.size(0));                                 \
    int n_conn  = static_cast<int>(indices.size(1));                                 \
    int n_batch = static_cast<int>(matrix.size(1));                                  \
    int is_homo = (weights.ndim() == 1) ? 1 : 0;                                    \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());   \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());       \
    const SPIKE_C_T*  d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr());     \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());           \
    int batch_tiles = (n_batch + 31) / 32;                                           \
    dim3 grid(n_pre, batch_tiles);                                                   \
    _bgm_basic_kern##SUFFIX<<<grid, 32, 0, s>>>(                                      \
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo);                  \
}

// Scatter warp FFI: void binary_fcnmm_scatter##SUFFIX(...)
#define FFI_BSM_WARP(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                  \
void binary_fcnmm_scatter##SUFFIX(                                                    \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                       \
    tvm::ffi::TensorView matrix,  tvm::ffi::TensorView output, int64_t stream         \
) {                                                                                   \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                         \
    int n_pre   = static_cast<int>(indices.size(0));                                 \
    int n_conn  = static_cast<int>(indices.size(1));                                 \
    int n_post  = static_cast<int>(output.size(0));                                  \
    int n_batch = static_cast<int>(matrix.size(1));                                  \
    int is_homo = (weights.ndim() == 1) ? 1 : 0;                                    \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());   \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());       \
    const SPIKE_C_T*  d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr());     \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());           \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_batch * sizeof(WEIGHT_C_T), s);    \
    int batch_tiles = (n_batch + 31) / 32;                                           \
    dim3 grid(n_pre, batch_tiles);                                                   \
    _bsm_warp_kern##SUFFIX<<<grid, 32, 0, s>>>(                                       \
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo);                  \
}

// Scatter basic FFI: void binary_fcnmm_scatter##SUFFIX(...)
#define FFI_BSM_BASIC(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                 \
void binary_fcnmm_scatter##SUFFIX(                                                    \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                       \
    tvm::ffi::TensorView matrix,  tvm::ffi::TensorView output, int64_t stream         \
) {                                                                                   \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                         \
    int n_pre   = static_cast<int>(indices.size(0));                                 \
    int n_conn  = static_cast<int>(indices.size(1));                                 \
    int n_post  = static_cast<int>(output.size(0));                                  \
    int n_batch = static_cast<int>(matrix.size(1));                                  \
    int is_homo = (weights.ndim() == 1) ? 1 : 0;                                    \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());   \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());       \
    const SPIKE_C_T*  d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr());     \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());           \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_batch * sizeof(WEIGHT_C_T), s);    \
    size_t shm = sizeof(int);                                                        \
    _bsm_basic_kern##SUFFIX<<<n_pre, 256, shm, s>>>(                                  \
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo);                  \
}

// =========================================================================
// Instantiate TVM FFI entry points via macros + @tvm_ffi annotations
// =========================================================================

// ---- Float32 ----
// @tvm_ffi binary_fcnmm_gather_bool_warp_f32
FFI_BGM_WARP(_bool_warp_f32,    float,  uint8_t)
// @tvm_ffi binary_fcnmm_gather_bool_basic_f32
FFI_BGM_BASIC(_bool_basic_f32,  float,  uint8_t)
// @tvm_ffi binary_fcnmm_gather_float_warp_f32
FFI_BGM_WARP(_float_warp_f32,   float,  float)
// @tvm_ffi binary_fcnmm_gather_float_basic_f32
FFI_BGM_BASIC(_float_basic_f32, float,  float)
// @tvm_ffi binary_fcnmm_scatter_bool_warp_f32
FFI_BSM_WARP(_bool_warp_f32,    float,  uint8_t)
// @tvm_ffi binary_fcnmm_scatter_bool_basic_f32
FFI_BSM_BASIC(_bool_basic_f32,  float,  uint8_t)
// @tvm_ffi binary_fcnmm_scatter_float_warp_f32
FFI_BSM_WARP(_float_warp_f32,   float,  float)
// @tvm_ffi binary_fcnmm_scatter_float_basic_f32
FFI_BSM_BASIC(_float_basic_f32, float,  float)

// ---- Float64 ----
// @tvm_ffi binary_fcnmm_gather_bool_warp_f64
FFI_BGM_WARP(_bool_warp_f64,    double, uint8_t)
// @tvm_ffi binary_fcnmm_gather_bool_basic_f64
FFI_BGM_BASIC(_bool_basic_f64,  double, uint8_t)
// @tvm_ffi binary_fcnmm_gather_float_warp_f64
FFI_BGM_WARP(_float_warp_f64,   double, double)
// @tvm_ffi binary_fcnmm_gather_float_basic_f64
FFI_BGM_BASIC(_float_basic_f64, double, double)
// @tvm_ffi binary_fcnmm_scatter_bool_warp_f64
FFI_BSM_WARP(_bool_warp_f64,    double, uint8_t)
// @tvm_ffi binary_fcnmm_scatter_bool_basic_f64
FFI_BSM_BASIC(_bool_basic_f64,  double, uint8_t)
// @tvm_ffi binary_fcnmm_scatter_float_warp_f64
FFI_BSM_WARP(_float_warp_f64,   double, double)
// @tvm_ffi binary_fcnmm_scatter_float_basic_f64
FFI_BSM_BASIC(_float_basic_f64, double, double)

// ---- Float16 (accumulate in float32 for stability) ----
// @tvm_ffi binary_fcnmm_gather_bool_warp_f16
FFI_BGM_WARP(_bool_warp_f16,    __half, uint8_t)
// @tvm_ffi binary_fcnmm_gather_bool_basic_f16
FFI_BGM_BASIC(_bool_basic_f16,  __half, uint8_t)
// @tvm_ffi binary_fcnmm_gather_float_warp_f16
FFI_BGM_WARP(_float_warp_f16,   __half, __half)
// @tvm_ffi binary_fcnmm_gather_float_basic_f16
FFI_BGM_BASIC(_float_basic_f16, __half, __half)
// @tvm_ffi binary_fcnmm_scatter_bool_warp_f16
FFI_BSM_WARP(_bool_warp_f16,    __half, uint8_t)
// @tvm_ffi binary_fcnmm_scatter_bool_basic_f16
FFI_BSM_BASIC(_bool_basic_f16,  __half, uint8_t)
// @tvm_ffi binary_fcnmm_scatter_float_warp_f16
FFI_BSM_WARP(_float_warp_f16,   __half, __half)
// @tvm_ffi binary_fcnmm_scatter_float_basic_f16
FFI_BSM_BASIC(_float_basic_f16, __half, __half)

// ---- BFloat16 (requires CUDA 11.0+; scatter atomicAdd requires CC 8.0+) ----
// @tvm_ffi binary_fcnmm_gather_bool_warp_bf16
FFI_BGM_WARP(_bool_warp_bf16,    __nv_bfloat16, uint8_t)
// @tvm_ffi binary_fcnmm_gather_bool_basic_bf16
FFI_BGM_BASIC(_bool_basic_bf16,  __nv_bfloat16, uint8_t)
// @tvm_ffi binary_fcnmm_gather_float_warp_bf16
FFI_BGM_WARP(_float_warp_bf16,   __nv_bfloat16, __nv_bfloat16)
// @tvm_ffi binary_fcnmm_gather_float_basic_bf16
FFI_BGM_BASIC(_float_basic_bf16, __nv_bfloat16, __nv_bfloat16)
// @tvm_ffi binary_fcnmm_scatter_bool_warp_bf16
FFI_BSM_WARP(_bool_warp_bf16,    __nv_bfloat16, uint8_t)
// @tvm_ffi binary_fcnmm_scatter_bool_basic_bf16
FFI_BSM_BASIC(_bool_basic_bf16,  __nv_bfloat16, uint8_t)
// @tvm_ffi binary_fcnmm_scatter_float_warp_bf16
FFI_BSM_WARP(_float_warp_bf16,   __nv_bfloat16, __nv_bfloat16)
// @tvm_ffi binary_fcnmm_scatter_float_basic_bf16
FFI_BSM_BASIC(_float_basic_bf16, __nv_bfloat16, __nv_bfloat16)
