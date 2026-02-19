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
 * fcnmv.cu — FCN Sparse Matrix-Vector (float) CUDA Kernels
 * =========================================================
 *
 * Python API: brainevent.fcnmv(weights, indices, vector, *, shape, transpose, backend)
 *
 * Sparse matrix--vector product with fixed connection number.
 *
 * Computes  y = W @ v  (or  y = W^T @ v  when transpose=True)
 * where W is a sparse weight matrix stored in fixed-connection-number
 * format and v is a dense floating-point vector.
 *
 * Parameters
 * ----------
 * weights : shape (1,) for homogeneous or (num_pre, num_conn) for heterogeneous,
 *           floating-point dtype (float16 / bfloat16 / float32 / float64).
 * indices : shape (num_pre, num_conn), int32 — post-synaptic column indices.
 * vector  : shape (num_post,) for gather, (num_pre,) for scatter.
 * shape   : logical (num_pre, num_post) shape of the dense weight matrix.
 * transpose=False (gather mode):
 *   y[i] = sum_{k} weights[i,k] * v[indices[i,k]]
 * transpose=True  (scatter mode):
 *   y[indices[i,k]] += weights[i,k] * v[i]   for all i,k
 *
 * Supported dtypes: float32 (default), float64 (_f64), float16 (_f16), bfloat16 (_bf16).
 * vec4 and shared-memory tile paths are float32-only; other dtypes use warp/basic kernels.
 * Float16 and bfloat16 accumulate in float32 for numerical stability.
 * Bfloat16 requires CUDA 11.0+.
 *
 * IMPORTANT: weights.data_ptr() is a GPU device pointer — NEVER dereference on host.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// =========================================================================
// Warp-level reduction helpers
// =========================================================================

__device__ __inline__ float warp_reduce_sum_f32(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __inline__ double warp_reduce_sum_f64(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// =========================================================================
// Per-dtype conversion macros: READ converts WEIGHT_T -> ACC_T,
//                              WRITE converts ACC_T -> WEIGHT_T.
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
// Gather warp kernel macro (one warp per output row)
//
// y[i] = sum_k w[i,k] * v[idx[i,k]]
// is_homo=1: w[i,k] = weights[0] (broadcast), is_homo=0: w[i,k] = weights[i,k]
// =========================================================================

#define DEFINE_GATHER_WARP(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _gather_warp_kern##SUFFIX(                                                \
    const int32_t* __restrict__ indices,                                                  \
    const WEIGHT_T* __restrict__ vector,                                                  \
    WEIGHT_T* __restrict__ output,                                                        \
    const WEIGHT_T* __restrict__ weights,                                                 \
    int n_pre, int n_conn, int is_homo                                                    \
) {                                                                                       \
    int row = blockIdx.x;                                                                 \
    if (row >= n_pre) return;                                                             \
    const int32_t* i_row = indices + (size_t)row * n_conn;                               \
    const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;          \
    ACC_T val = ACC_ZERO;                                                                 \
    for (int k = threadIdx.x; k < n_conn; k += 32)                                       \
        val += is_homo ? READ_W(vector[i_row[k]])                                         \
                       : (READ_W(w_row[k]) * READ_W(vector[i_row[k]]));                  \
    val = WARP_RED(val);                                                                  \
    if (threadIdx.x == 0)                                                                 \
        output[row] = WRITE_W(is_homo ? (READ_W(weights[0]) * val) : val);               \
}

// =========================================================================
// Gather basic kernel macro (one block per output row with block reduction)
// Uses 32*sizeof(ACC_T) bytes of dynamic shared memory.
// Best for 32 < n_conn <= 512.
// =========================================================================

#define DEFINE_GATHER_BASIC(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _gather_basic_kern##SUFFIX(                                                \
    const int32_t* __restrict__ indices,                                                   \
    const WEIGHT_T* __restrict__ vector,                                                   \
    WEIGHT_T* __restrict__ output,                                                         \
    const WEIGHT_T* __restrict__ weights,                                                  \
    int n_pre, int n_conn, int is_homo                                                     \
) {                                                                                        \
    extern __shared__ char _smem_bytes[];                                                  \
    ACC_T* smem_red = reinterpret_cast<ACC_T*>(_smem_bytes);                              \
    int row = blockIdx.x;                                                                  \
    if (row >= n_pre) return;                                                              \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                \
    const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;           \
    ACC_T val = ACC_ZERO;                                                                  \
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x)                                \
        val += is_homo ? READ_W(vector[i_row[k]])                                          \
                       : (READ_W(w_row[k]) * READ_W(vector[i_row[k]]));                   \
    int lane   = threadIdx.x & 31;                                                         \
    int warpid = threadIdx.x >> 5;                                                         \
    val = WARP_RED(val);                                                                   \
    if (lane == 0) smem_red[warpid] = val;                                                 \
    __syncthreads();                                                                        \
    int n_warps = (blockDim.x + 31) >> 5;                                                  \
    val = (threadIdx.x < n_warps) ? smem_red[lane] : ACC_ZERO;                            \
    if (warpid == 0) val = WARP_RED(val);                                                  \
    if (threadIdx.x == 0)                                                                  \
        output[row] = WRITE_W(is_homo ? (READ_W(weights[0]) * val) : val);                \
}

// =========================================================================
// Scatter basic kernel macro (one block per pre-neuron)
// y[idx[i,k]] += w[i,k] * v[i]   (output must be pre-zeroed)
// =========================================================================

#define DEFINE_SCATTER_BASIC(SUFFIX, WEIGHT_T)                                     \
__global__ void _scatter_basic_kern##SUFFIX(                                        \
    const int32_t* __restrict__ indices,                                            \
    const WEIGHT_T* __restrict__ vector,                                            \
    WEIGHT_T* __restrict__ output,                                                  \
    const WEIGHT_T* __restrict__ weights,                                           \
    int n_pre, int n_conn, int is_homo                                              \
) {                                                                                 \
    int row = blockIdx.x;                                                           \
    if (row >= n_pre) return;                                                       \
    WEIGHT_T v = vector[row];                                                       \
    const int32_t* i_row = indices + (size_t)row * n_conn;                         \
    const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;    \
    WEIGHT_T w0 = is_homo ? weights[0] : (WEIGHT_T)0;                              \
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x)                         \
        atomicAdd(&output[i_row[k]], (is_homo ? w0 : w_row[k]) * v);               \
}

// =========================================================================
// Scatter warp kernel macro (8 warps per block, one warp per pre-neuron)
// Grid = ceil(n_pre / 8) blocks.  Best when n_conn <= 32.
// =========================================================================

#define DEFINE_SCATTER_WARP(SUFFIX, WEIGHT_T)                                          \
__global__ void _scatter_warp_kern##SUFFIX(                                             \
    const int32_t* __restrict__ indices,                                                \
    const WEIGHT_T* __restrict__ vector,                                                \
    WEIGHT_T* __restrict__ output,                                                      \
    const WEIGHT_T* __restrict__ weights,                                               \
    int n_pre, int n_conn, int is_homo                                                  \
) {                                                                                     \
    int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;                      \
    int lane_id   = threadIdx.x & 31;                                                   \
    int num_warps = (gridDim.x * blockDim.x) >> 5;                                      \
    for (int row = warp_id; row < n_pre; row += num_warps) {                            \
        WEIGHT_T v = vector[row];                                                        \
        const int32_t* i_row = indices + (size_t)row * n_conn;                          \
        const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;     \
        WEIGHT_T w0 = is_homo ? weights[0] : (WEIGHT_T)0;                               \
        for (int k = lane_id; k < n_conn; k += 32)                                      \
            atomicAdd(&output[i_row[k]], (is_homo ? w0 : w_row[k]) * v);                \
    }                                                                                    \
}

// =========================================================================
// Scatter grid-stride kernel macro (flat (pre, conn) iteration)
// Maximises SM occupancy for large problems.
// =========================================================================

#define DEFINE_SCATTER_GS(SUFFIX, WEIGHT_T)                                            \
__global__ void _scatter_gs_kern##SUFFIX(                                               \
    const int32_t* __restrict__ indices,                                                \
    const WEIGHT_T* __restrict__ vector,                                                \
    WEIGHT_T* __restrict__ output,                                                      \
    const WEIGHT_T* __restrict__ weights,                                               \
    int n_pre, int n_conn, int is_homo                                                  \
) {                                                                                     \
    int total  = n_pre * n_conn;                                                        \
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;                                 \
    int stride = blockDim.x * gridDim.x;                                                \
    for (int idx = tid; idx < total; idx += stride) {                                   \
        int row = idx / n_conn;                                                         \
        WEIGHT_T w = is_homo ? weights[0] : weights[idx];                               \
        atomicAdd(&output[indices[idx]], w * vector[row]);                              \
    }                                                                                   \
}

// =========================================================================
// Instantiate device kernels: 4 weight dtypes
// =========================================================================

// ---- Float32 ----
DEFINE_GATHER_WARP(_f32,  float,          float,  READ_F32,  WRITE_F32,  warp_reduce_sum_f32, 0.0f)
DEFINE_GATHER_BASIC(_f32, float,          float,  READ_F32,  WRITE_F32,  warp_reduce_sum_f32, 0.0f)
DEFINE_SCATTER_BASIC(_f32, float)
DEFINE_SCATTER_WARP(_f32, float)
DEFINE_SCATTER_GS(_f32, float)

// ---- Float64 ----
DEFINE_GATHER_WARP(_f64,  double,         double, READ_F64,  WRITE_F64,  warp_reduce_sum_f64, 0.0)
DEFINE_GATHER_BASIC(_f64, double,         double, READ_F64,  WRITE_F64,  warp_reduce_sum_f64, 0.0)
DEFINE_SCATTER_BASIC(_f64, double)
DEFINE_SCATTER_WARP(_f64, double)
DEFINE_SCATTER_GS(_f64, double)

// ---- Float16 (accumulate in float32 for stability) ----
DEFINE_GATHER_WARP(_f16,  __half,         float,  READ_F16,  WRITE_F16,  warp_reduce_sum_f32, 0.0f)
DEFINE_GATHER_BASIC(_f16, __half,         float,  READ_F16,  WRITE_F16,  warp_reduce_sum_f32, 0.0f)
DEFINE_SCATTER_BASIC(_f16, __half)
DEFINE_SCATTER_WARP(_f16, __half)

// ---- BFloat16 (accumulate in float32 for stability; requires CUDA 11.0+) ----
DEFINE_GATHER_WARP(_bf16,  __nv_bfloat16, float,  READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_GATHER_BASIC(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_SCATTER_BASIC(_bf16, __nv_bfloat16)
DEFINE_SCATTER_WARP(_bf16, __nv_bfloat16)

// =========================================================================
// Float32-only specialised kernels
//
// _gather_shared_kern: tile-loading into shared memory. Best for n_conn > 512.
// _gather_vec4_kern:   float4/int4 vectorised loads.  Best when n_conn % 4 == 0.
// =========================================================================

// One block per row, cooperative tile-load of indices+weights into shared memory.
// Shared memory layout (dynamic):
//   [0 .. blockDim.x*4)              : int32_t s_idx[blockDim.x]
//   [blockDim.x*4 .. 2*blockDim.x*4) : float   s_wt[blockDim.x]
//   [2*blockDim.x*4 .. +32*4)        : float   s_red[32]  (block reduction)
__global__ void _gather_shared_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ vector,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    extern __shared__ char smem_raw[];
    int32_t* s_idx = reinterpret_cast<int32_t*>(smem_raw);
    float*   s_wt  = reinterpret_cast<float*>(smem_raw + blockDim.x * sizeof(int32_t));
    float*   s_red = reinterpret_cast<float*>(smem_raw + blockDim.x * (sizeof(int32_t) + sizeof(float)));

    int row = blockIdx.x;
    if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;

    float val = 0.0f;
    for (int base = 0; base < n_conn; base += blockDim.x) {
        int k = base + threadIdx.x;
        if (k < n_conn) {
            s_idx[threadIdx.x] = i_row[k];
            s_wt[threadIdx.x]  = is_homo ? 1.0f : w_row[k];
        }
        __syncthreads();
        int tile = min((int)blockDim.x, n_conn - base);
        if (threadIdx.x < tile)
            val += s_wt[threadIdx.x] * vector[s_idx[threadIdx.x]];
        __syncthreads();
    }

    int lane   = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;
    val = warp_reduce_sum_f32(val);
    if (lane == 0) s_red[warpid] = val;
    __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < n_warps) ? s_red[lane] : 0.0f;
    if (warpid == 0) val = warp_reduce_sum_f32(val);

    if (threadIdx.x == 0)
        output[row] = is_homo ? (weights[0] * val) : val;
}

// One block per row, float4/int4 vectorised loads.
// Uses 32*sizeof(float) dynamic shared memory for block reduction.
__global__ void _gather_vec4_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ vector,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    extern __shared__ float smem_red[];
    int row = blockIdx.x;
    if (row >= n_pre) return;
    size_t base = (size_t)row * n_conn;
    const int4*   i4 = reinterpret_cast<const int4*>(indices + base);
    const float4* w4 = is_homo ? nullptr : reinterpret_cast<const float4*>(weights + base);
    int n4 = n_conn >> 2;
    float val = 0.0f;
    for (int k = threadIdx.x; k < n4; k += blockDim.x) {
        int4 idx = i4[k];
        if (!is_homo) {
            float4 ww = w4[k];
            val += ww.x * vector[idx.x] + ww.y * vector[idx.y]
                 + ww.z * vector[idx.z] + ww.w * vector[idx.w];
        } else {
            val += vector[idx.x] + vector[idx.y] + vector[idx.z] + vector[idx.w];
        }
    }
    for (int k = (n4 << 2) + threadIdx.x; k < n_conn; k += blockDim.x) {
        float v = vector[indices[base + k]];
        val += is_homo ? v : (weights[base + k] * v);
    }
    int lane   = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;
    val = warp_reduce_sum_f32(val);
    if (lane == 0) smem_red[warpid] = val;
    __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < n_warps) ? smem_red[lane] : 0.0f;
    if (warpid == 0) val = warp_reduce_sum_f32(val);
    if (threadIdx.x == 0)
        output[row] = is_homo ? (weights[0] * val) : val;
}

// =========================================================================
// TVM FFI Entry Point Macros
// =========================================================================
//
// Convention: args = (weights, indices, vector, output, stream)
// weights: shape (1,) for homo or (n_pre, n_conn) for hetero
// indices: shape (n_pre, n_conn), int32
// vector/output: WEIGHT_T arrays
//
// IMPORTANT: data_ptr() returns GPU device pointers — NEVER dereference on host.

// ---- Gather warp FFI macro ----
#define FFI_GATHER_WARP(SUFFIX, WEIGHT_C_T)                                      \
void fcnmv_gather_warp##SUFFIX(                                                   \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                  \
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream    \
) {                                                                               \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                     \
    int n_pre  = static_cast<int>(indices.size(0));                              \
    int n_conn = static_cast<int>(indices.size(1));                              \
    int is_homo = (weights.ndim() == 1) ? 1 : 0;                                \
    const WEIGHT_C_T*  d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*     d_idx = static_cast<const int32_t*>(indices.data_ptr()); \
    const WEIGHT_C_T*  d_vec = static_cast<const WEIGHT_C_T*>(vector.data_ptr()); \
    WEIGHT_C_T*        d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());     \
    _gather_warp_kern##SUFFIX<<<n_pre, 32, 0, s>>>(                              \
        d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);                      \
}

// ---- Gather basic FFI macro ----
#define FFI_GATHER_BASIC(SUFFIX, WEIGHT_C_T, SHM_SIZE)                           \
void fcnmv_gather_basic##SUFFIX(                                                  \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                  \
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream    \
) {                                                                               \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                     \
    int n_pre  = static_cast<int>(indices.size(0));                              \
    int n_conn = static_cast<int>(indices.size(1));                              \
    int is_homo = (weights.ndim() == 1) ? 1 : 0;                                \
    const WEIGHT_C_T*  d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*     d_idx = static_cast<const int32_t*>(indices.data_ptr()); \
    const WEIGHT_C_T*  d_vec = static_cast<const WEIGHT_C_T*>(vector.data_ptr()); \
    WEIGHT_C_T*        d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());     \
    _gather_basic_kern##SUFFIX<<<n_pre, 256, SHM_SIZE, s>>>(                     \
        d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);                      \
}

// ---- Gather auto FFI macro (warp for n_conn<=32, basic otherwise) ----
#define FFI_GATHER_AUTO(SUFFIX, WEIGHT_C_T, SHM_SIZE)                            \
void fcnmv_gather_auto##SUFFIX(                                                   \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                  \
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream    \
) {                                                                               \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                     \
    int n_pre  = static_cast<int>(indices.size(0));                              \
    int n_conn = static_cast<int>(indices.size(1));                              \
    int is_homo = (weights.ndim() == 1) ? 1 : 0;                                \
    const WEIGHT_C_T*  d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*     d_idx = static_cast<const int32_t*>(indices.data_ptr()); \
    const WEIGHT_C_T*  d_vec = static_cast<const WEIGHT_C_T*>(vector.data_ptr()); \
    WEIGHT_C_T*        d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());     \
    if (n_conn <= 32) {                                                           \
        _gather_warp_kern##SUFFIX<<<n_pre, 32, 0, s>>>(                          \
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);                  \
    } else {                                                                      \
        _gather_basic_kern##SUFFIX<<<n_pre, 256, SHM_SIZE, s>>>(                 \
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);                  \
    }                                                                             \
}

// ---- Scatter warp FFI macro ----
#define FFI_SCATTER_WARP(SUFFIX, WEIGHT_C_T)                                     \
void fcnmv_scatter_warp##SUFFIX(                                                  \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                  \
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream    \
) {                                                                               \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                     \
    int n_pre  = static_cast<int>(indices.size(0));                              \
    int n_conn = static_cast<int>(indices.size(1));                              \
    int n_post = static_cast<int>(output.size(0));                               \
    int is_homo = (weights.ndim() == 1) ? 1 : 0;                                \
    const WEIGHT_C_T*  d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*     d_idx = static_cast<const int32_t*>(indices.data_ptr()); \
    const WEIGHT_C_T*  d_vec = static_cast<const WEIGHT_C_T*>(vector.data_ptr()); \
    WEIGHT_C_T*        d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());     \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(WEIGHT_C_T), s);          \
    int blocks = (n_pre + 7) / 8;                                                \
    _scatter_warp_kern##SUFFIX<<<blocks, 256, 0, s>>>(                           \
        d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);                      \
}

// ---- Scatter auto FFI macro ----
#define FFI_SCATTER_AUTO(SUFFIX, WEIGHT_C_T)                                     \
void fcnmv_scatter_auto##SUFFIX(                                                  \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                  \
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream    \
) {                                                                               \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                     \
    int n_pre  = static_cast<int>(indices.size(0));                              \
    int n_conn = static_cast<int>(indices.size(1));                              \
    int n_post = static_cast<int>(output.size(0));                               \
    int is_homo = (weights.ndim() == 1) ? 1 : 0;                                \
    const WEIGHT_C_T*  d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*     d_idx = static_cast<const int32_t*>(indices.data_ptr()); \
    const WEIGHT_C_T*  d_vec = static_cast<const WEIGHT_C_T*>(vector.data_ptr()); \
    WEIGHT_C_T*        d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());     \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(WEIGHT_C_T), s);          \
    if (n_conn <= 32) {                                                           \
        int blocks = (n_pre + 7) / 8;                                            \
        _scatter_warp_kern##SUFFIX<<<blocks, 256, 0, s>>>(                       \
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);                  \
    } else if ((long long)n_pre * n_conn > 262144LL) {                           \
        int blocks = min(1024, (n_pre * n_conn + 255) / 256);                    \
        _scatter_gs_kern##SUFFIX<<<blocks, 256, 0, s>>>(                         \
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);                  \
    } else {                                                                      \
        _scatter_basic_kern##SUFFIX<<<n_pre, 256, 0, s>>>(                       \
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);                  \
    }                                                                             \
}

// =========================================================================
// Instantiate TVM FFI entry points via macros + @tvm_ffi annotations
// =========================================================================

// ---- Float32 (shm: 32 * sizeof(float) = 128 bytes) ----
// @tvm_ffi fcnmv_gather_warp_f32
FFI_GATHER_WARP(_f32, float)
// @tvm_ffi fcnmv_gather_basic_f32
FFI_GATHER_BASIC(_f32, float, 32 * sizeof(float))
// @tvm_ffi fcnmv_scatter_warp_f32
FFI_SCATTER_WARP(_f32, float)
// @tvm_ffi fcnmv_scatter_auto_f32
FFI_SCATTER_AUTO(_f32, float)

// Float32 auto-gather has extra vec4/shared paths — written explicitly below
void fcnmv_gather_vec4_f32(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    _gather_vec4_kern<<<n_pre, 256, 32 * sizeof(float), s>>>(
        d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
}

void fcnmv_gather_auto_f32(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());

    size_t shm_red = 32 * sizeof(float);
    if (n_conn <= 32) {
        _gather_warp_kern_f32<<<n_pre, 32, 0, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    } else if (n_conn % 4 == 0 && n_conn >= 128) {
        _gather_vec4_kern<<<n_pre, 256, shm_red, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    } else if (n_conn > 512) {
        int threads = 256;
        size_t shm = (size_t)threads * (sizeof(int32_t) + sizeof(float)) + 32 * sizeof(float);
        _gather_shared_kern<<<n_pre, threads, shm, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    } else {
        _gather_basic_kern_f32<<<n_pre, 256, shm_red, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    }
}

// ---- Float64 (shm: 32 * sizeof(double) = 256 bytes) ----
// @tvm_ffi fcnmv_gather_warp_f64
FFI_GATHER_WARP(_f64, double)
// @tvm_ffi fcnmv_gather_auto_f64
FFI_GATHER_AUTO(_f64, double, 32 * sizeof(double))
// @tvm_ffi fcnmv_scatter_warp_f64
FFI_SCATTER_WARP(_f64, double)
// @tvm_ffi fcnmv_scatter_auto_f64
FFI_SCATTER_AUTO(_f64, double)

// ---- Float16 (shm: 32 * sizeof(float) = 128 bytes; accumulates in f32) ----
// @tvm_ffi fcnmv_gather_warp_f16
FFI_GATHER_WARP(_f16, __half)
// @tvm_ffi fcnmv_gather_auto_f16
FFI_GATHER_AUTO(_f16, __half, 32 * sizeof(float))
// @tvm_ffi fcnmv_scatter_warp_f16
FFI_SCATTER_WARP(_f16, __half)
// @tvm_ffi fcnmv_scatter_auto_f16
FFI_SCATTER_AUTO(_f16, __half)

// ---- BFloat16 (shm: 32 * sizeof(float) = 128 bytes; accumulates in f32; requires CUDA 11.0+) ----
// @tvm_ffi fcnmv_gather_warp_bf16
FFI_GATHER_WARP(_bf16, __nv_bfloat16)
// @tvm_ffi fcnmv_gather_auto_bf16
FFI_GATHER_AUTO(_bf16, __nv_bfloat16, 32 * sizeof(float))
// @tvm_ffi fcnmv_scatter_warp_bf16
FFI_SCATTER_WARP(_bf16, __nv_bfloat16)
// @tvm_ffi fcnmv_scatter_auto_bf16
FFI_SCATTER_AUTO(_bf16, __nv_bfloat16)
