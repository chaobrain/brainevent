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
 * fcnmm.cu — FCN Sparse Matrix-Matrix (float) CUDA Kernels
 * =========================================================
 *
 * Python API: brainevent.fcnmm(weights, indices, matrix, *, shape, transpose, backend)
 *
 * Sparse matrix--matrix product with fixed connection number.
 *
 * Computes  Y = W @ M  (or  Y = W^T @ M  when transpose=True)
 * where W is a sparse weight matrix stored in fixed-connection-number
 * format and M is a dense floating-point matrix.
 *
 * Parameters
 * ----------
 * weights : shape (1,) for homogeneous or (num_pre, num_conn) for heterogeneous,
 *           floating-point dtype (float16 / bfloat16 / float32 / float64).
 * indices : shape (num_pre, num_conn), int32 — post-synaptic column indices.
 * matrix  : shape (num_post, n_col) for gather, (num_pre, n_col) for scatter.
 * shape   : logical (num_pre, num_post) shape of the dense weight matrix.
 * transpose=False (gather mode):
 *   Y[i,j] = sum_{k} weights[i,k] * M[indices[i,k], j]
 * transpose=True  (scatter mode):
 *   Y[indices[i,k], j] += weights[i,k] * M[i,j]   for all i,k,j
 *
 * Supported dtypes: float32 (_f32), float64 (_f64), float16 (_f16), bfloat16 (_bf16).
 * Float16 and bfloat16 accumulate in float32 for numerical stability.
 * vec4 and shared-memory tile paths are float32-only; other dtypes use basic/block kernels.
 *
 * IMPORTANT: weights.data_ptr() is a GPU device pointer — NEVER dereference on host.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// =========================================================================
// Per-dtype weight conversion macros
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
// FCN Matrix-Matrix GATHER kernels (transpose=False)
//
// Y[i, j] = sum_{k=0}^{n_conn-1} w[i,k] * M[indices[i,k], j]
// indices: [n_pre, n_conn], weights: [1] or [n_pre, n_conn]
// M: [n_post, n_col],  Y: [n_pre, n_col]
// =========================================================================

// -------------------------------------------------------------------------
// GATHER basic kernel macro
// Grid: (n_pre, ceil(n_col / BJ)), Block: (BJ,)
// One thread per output element Y[i,j].
// -------------------------------------------------------------------------

#define DEFINE_MM_GATHER_BASIC(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W)       \
__global__ void _mm_gather_basic_kern##SUFFIX(                                  \
    const int32_t* __restrict__ indices,                                        \
    const WEIGHT_T* __restrict__ matrix,                                        \
    WEIGHT_T*       __restrict__ output,                                        \
    const WEIGHT_T* __restrict__ weights,                                       \
    int n_pre, int n_conn, int n_col, int is_homo                               \
) {                                                                             \
    int i = blockIdx.x;                                                         \
    int j = blockIdx.y * blockDim.x + threadIdx.x;                             \
    if (i >= n_pre || j >= n_col) return;                                       \
    const int32_t*  idx_row = indices + (size_t)i * n_conn;                    \
    const WEIGHT_T* w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn; \
    ACC_T acc = (ACC_T)0;                                                       \
    for (int k = 0; k < n_conn; k++) {                                         \
        ACC_T w = is_homo ? (ACC_T)1 : READ_W(w_row[k]);                       \
        acc += w * READ_W(matrix[(size_t)idx_row[k] * n_col + j]);             \
    }                                                                           \
    output[(size_t)i * n_col + j] = WRITE_W(is_homo ? (READ_W(weights[0]) * acc) : acc); \
}

// -------------------------------------------------------------------------
// SCATTER block kernel macro
// Grid: (n_pre,), Block: (256,)
// One block per pre-neuron; threads stride over j columns.
// -------------------------------------------------------------------------

#define DEFINE_MM_SCATTER_BLOCK(SUFFIX, WEIGHT_T, READ_W)                      \
__global__ void _mm_scatter_block_kern##SUFFIX(                                 \
    const int32_t* __restrict__ indices,                                        \
    const WEIGHT_T* __restrict__ matrix,                                        \
    WEIGHT_T*       __restrict__ output,                                        \
    const WEIGHT_T* __restrict__ weights,                                       \
    int n_pre, int n_conn, int n_col, int is_homo                               \
) {                                                                             \
    int i = blockIdx.x;                                                         \
    if (i >= n_pre) return;                                                     \
    const int32_t*  idx_row = indices + (size_t)i * n_conn;                    \
    const WEIGHT_T* w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn; \
    const WEIGHT_T* m_row   = matrix + (size_t)i * n_col;                      \
    for (int k = 0; k < n_conn; k++) {                                         \
        int tgt = idx_row[k];                                                   \
        WEIGHT_T w = is_homo ? weights[0] : w_row[k];                          \
        WEIGHT_T* out_row = output + (size_t)tgt * n_col;                      \
        for (int j = threadIdx.x; j < n_col; j += blockDim.x)                  \
            atomicAdd(&out_row[j], WRITE_W(READ_W(w) * READ_W(m_row[j])));     \
    }                                                                           \
}

// -------------------------------------------------------------------------
// SCATTER warp kernel macro
// Grid: (min(4096, ceil(n_pre*n_conn/8)),), Block: (256,)
// Each warp handles one (i, k) pair; lanes stride over j columns.
// -------------------------------------------------------------------------

#define DEFINE_MM_SCATTER_WARP(SUFFIX, WEIGHT_T, READ_W)                       \
__global__ void _mm_scatter_warp_kern##SUFFIX(                                  \
    const int32_t* __restrict__ indices,                                        \
    const WEIGHT_T* __restrict__ matrix,                                        \
    WEIGHT_T*       __restrict__ output,                                        \
    const WEIGHT_T* __restrict__ weights,                                       \
    int n_pre, int n_conn, int n_col, int is_homo                               \
) {                                                                             \
    int wid     = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;                \
    int lane    = threadIdx.x & 31;                                             \
    int n_warps = (gridDim.x * blockDim.x) >> 5;                               \
    int n_pairs = n_pre * n_conn;                                               \
    for (int pair = wid; pair < n_pairs; pair += n_warps) {                    \
        int i = pair / n_conn;                                                  \
        int k = pair % n_conn;                                                  \
        int   tgt = indices[(size_t)i * n_conn + k];                           \
        WEIGHT_T w = is_homo ? weights[0] : weights[(size_t)i * n_conn + k];   \
        const WEIGHT_T* m_row   = matrix + (size_t)i * n_col;                  \
        WEIGHT_T*       out_row = output + (size_t)tgt * n_col;                \
        for (int j = lane; j < n_col; j += 32)                                 \
            atomicAdd(&out_row[j], WRITE_W(READ_W(w) * READ_W(m_row[j])));     \
    }                                                                           \
}

// =========================================================================
// Float32-only optimized kernels (vec4 and shared-memory)
// These exploit float4 vectorization and are only practical for float32.
// =========================================================================

#define MMTK 128

// Shared-memory gather: tiles index/weight arrays into shared memory.
// Grid: (n_pre, ceil(n_col / 64)), Block: (64,)
// Shared mem: MMTK * 8 bytes (idx tile + weight tile).
__global__ void _mm_gather_shared_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ matrix,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    extern __shared__ char smem_mm[];
    int32_t* s_idx = reinterpret_cast<int32_t*>(smem_mm);
    float*   s_w   = reinterpret_cast<float*>(smem_mm + MMTK * sizeof(int32_t));

    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= n_pre) return;

    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const float*   w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;

    float acc = 0.0f;
    for (int k0 = 0; k0 < n_conn; k0 += MMTK) {
        int tile = (k0 + MMTK < n_conn) ? MMTK : (n_conn - k0);
        for (int t = threadIdx.x; t < tile; t += blockDim.x) {
            s_idx[t] = idx_row[k0 + t];
            s_w[t]   = is_homo ? 1.0f : w_row[k0 + t];
        }
        __syncthreads();
        if (j < n_col) {
            for (int t = 0; t < tile; t++)
                acc += s_w[t] * matrix[(size_t)s_idx[t] * n_col + j];
        }
        __syncthreads();
    }
    if (j < n_col)
        output[(size_t)i * n_col + j] = is_homo ? (weights[0] * acc) : acc;
}

// Vectorised gather: float4 loads when n_col % 4 == 0.
// Grid: (n_pre, ceil(n_col/4 / 64)), Block: (64,)
__global__ void _mm_gather_vec4_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ matrix,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    int i   = blockIdx.x;
    int j4  = blockIdx.y * blockDim.x + threadIdx.x;
    int nc4 = n_col >> 2;
    if (i >= n_pre || j4 >= nc4) return;

    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const float*   w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;
    const float4*  mat4    = reinterpret_cast<const float4*>(matrix);

    float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = 0; k < n_conn; k++) {
        float  w = is_homo ? weights[0] : w_row[k];
        float4 m = mat4[(size_t)idx_row[k] * nc4 + j4];
        acc.x += w * m.x;
        acc.y += w * m.y;
        acc.z += w * m.z;
        acc.w += w * m.w;
    }
    reinterpret_cast<float4*>(output)[(size_t)i * nc4 + j4] = acc;
}

// Cached scatter for float32 with shared memory.
// Grid: (n_pre, ceil(n_col / MM_SCATTER_BJ)), Block: (MM_SCATTER_BJ,)
#define MM_SCATTER_BJ 128
__global__ void _mm_scatter_cached_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ matrix,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    extern __shared__ float s_m[];
    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= n_pre) return;

    s_m[threadIdx.x] = (j < n_col) ? matrix[(size_t)i * n_col + j] : 0.0f;
    __syncthreads();

    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const float*   w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;

    if (j < n_col) {
        float m_val = s_m[threadIdx.x];
        for (int k = 0; k < n_conn; k++) {
            int   tgt = idx_row[k];
            float w   = is_homo ? weights[0] : w_row[k];
            atomicAdd(&output[(size_t)tgt * n_col + j], w * m_val);
        }
    }
}

// =========================================================================
// Instantiate device kernels for f32, f64, f16, bf16
// (f32 uses explicit optimized kernels above for special paths)
// =========================================================================

DEFINE_MM_GATHER_BASIC(_f32,  float,          float,  READ_F32,  WRITE_F32)
DEFINE_MM_GATHER_BASIC(_f64,  double,         double, READ_F64,  WRITE_F64)
DEFINE_MM_GATHER_BASIC(_f16,  __half,         float,  READ_F16,  WRITE_F16)
DEFINE_MM_GATHER_BASIC(_bf16, __nv_bfloat16,  float,  READ_BF16, WRITE_BF16)

DEFINE_MM_SCATTER_BLOCK(_f32,  float,          READ_F32)
DEFINE_MM_SCATTER_BLOCK(_f64,  double,         READ_F64)
DEFINE_MM_SCATTER_BLOCK(_f16,  __half,         READ_F16)
DEFINE_MM_SCATTER_BLOCK(_bf16, __nv_bfloat16,  READ_BF16)

DEFINE_MM_SCATTER_WARP(_f32,  float,          READ_F32)
DEFINE_MM_SCATTER_WARP(_f64,  double,         READ_F64)
DEFINE_MM_SCATTER_WARP(_f16,  __half,         READ_F16)
DEFINE_MM_SCATTER_WARP(_bf16, __nv_bfloat16,  READ_BF16)

// =========================================================================
// TVM FFI Entry Point Macros
// =========================================================================
// Convention: args = (weights, indices, matrix, output, stream)
// weights: [1] (homo) or [n_pre, n_conn] (hetero)
// indices: [n_pre, n_conn], int32
// Gather:  matrix [n_post, n_col], output [n_pre, n_col]
// Scatter: matrix [n_pre,  n_col], output [n_post, n_col]  (pre-zeroed)
//
// weights.data_ptr() is a GPU device pointer — NEVER dereference on host.
// GPU threads read weights[0] (homo) or weights[i*n_conn+k] (hetero).

// Gather auto FFI macro: selects vec4/shared/basic based on n_conn, n_col.
#define FFI_MM_GATHER_AUTO(SUFFIX, WEIGHT_C_T)                                 \
void fcnmm_gather_auto##SUFFIX(                                                \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                \
    tvm::ffi::TensorView matrix,  tvm::ffi::TensorView output, int64_t stream  \
) {                                                                            \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                  \
    int n_pre  = static_cast<int>(indices.size(0));                           \
    int n_conn = static_cast<int>(indices.size(1));                           \
    int n_col  = static_cast<int>(matrix.size(1));                            \
    int is_homo = (weights.ndim() == 1) ? 1 : 0;                             \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr()); \
    const WEIGHT_C_T* d_mat = static_cast<const WEIGHT_C_T*>(matrix.data_ptr()); \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());    \
    int BJ = 64;                                                               \
    dim3 grid(n_pre, (n_col + BJ - 1) / BJ);                                  \
    _mm_gather_basic_kern##SUFFIX<<<grid, BJ, 0, s>>>(                         \
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);             \
}

// Scatter auto FFI macro: selects warp/block based on n_col.
#define FFI_MM_SCATTER_AUTO(SUFFIX, WEIGHT_C_T)                                \
void fcnmm_scatter_auto##SUFFIX(                                               \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                \
    tvm::ffi::TensorView matrix,  tvm::ffi::TensorView output, int64_t stream  \
) {                                                                            \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                  \
    int n_pre  = static_cast<int>(indices.size(0));                           \
    int n_conn = static_cast<int>(indices.size(1));                           \
    int n_post = static_cast<int>(output.size(0));                            \
    int n_col  = static_cast<int>(matrix.size(1));                            \
    int is_homo = (weights.ndim() == 1) ? 1 : 0;                             \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr()); \
    const WEIGHT_C_T* d_mat = static_cast<const WEIGHT_C_T*>(matrix.data_ptr()); \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());    \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_col * sizeof(WEIGHT_C_T), s); \
    if (n_col <= 64) {                                                         \
        int n_pairs = n_pre * n_conn;                                          \
        int blocks  = min(4096, (n_pairs + 7) / 8);                           \
        _mm_scatter_warp_kern##SUFFIX<<<blocks, 256, 0, s>>>(                  \
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);         \
    } else {                                                                   \
        _mm_scatter_block_kern##SUFFIX<<<n_pre, 256, 0, s>>>(                  \
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);         \
    }                                                                          \
}

// =========================================================================
// TVM FFI Entry Points
// =========================================================================

// --- Float32 gather: uses vec4/shared/basic auto-dispatch ---

void fcnmm_gather_shared_f32(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_mat = static_cast<const float*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    int BJ = 64;
    dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
    size_t shm = MMTK * (sizeof(int32_t) + sizeof(float));
    _mm_gather_shared_kern<<<grid, BJ, shm, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

void fcnmm_gather_vec4_f32(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_mat = static_cast<const float*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    int BJ4 = 64;
    dim3 grid(n_pre, (n_col / 4 + BJ4 - 1) / BJ4);
    _mm_gather_vec4_kern<<<grid, BJ4, 0, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

// Auto-selects vec4/shared/basic gather kernel for float32.
void fcnmm_gather_auto_f32(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_mat = static_cast<const float*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());

    int BJ = 64;
    if (n_col % 4 == 0 && n_col >= 64) {
        dim3 grid(n_pre, (n_col / 4 + BJ - 1) / BJ);
        _mm_gather_vec4_kern<<<grid, BJ, 0, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    } else if (n_conn > 128) {
        dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
        size_t shm = MMTK * (sizeof(int32_t) + sizeof(float));
        _mm_gather_shared_kern<<<grid, BJ, shm, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    } else {
        dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
        _mm_gather_basic_kern_f32<<<grid, BJ, 0, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    }
}

// Auto-selects scatter kernel for float32 with cached shmem path.
void fcnmm_scatter_auto_f32(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_mat = static_cast<const float*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_col * sizeof(float), s);

    if (n_col <= 64) {
        int n_pairs = n_pre * n_conn;
        int blocks  = min(4096, (n_pairs + 7) / 8);
        _mm_scatter_warp_kern_f32<<<blocks, 256, 0, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    } else if (n_conn > 32) {
        int BJ = MM_SCATTER_BJ;
        dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
        size_t shm = BJ * sizeof(float);
        _mm_scatter_cached_kern<<<grid, BJ, shm, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    } else {
        _mm_scatter_block_kern_f32<<<n_pre, 256, 0, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    }
}

// --- Float64 gather/scatter: basic/warp kernels only ---
// @tvm_ffi fcnmm_gather_auto_f64
FFI_MM_GATHER_AUTO(_f64, double)
// @tvm_ffi fcnmm_scatter_auto_f64
FFI_MM_SCATTER_AUTO(_f64, double)

// --- Float16 gather/scatter: basic/warp kernels only ---
// @tvm_ffi fcnmm_gather_auto_f16
FFI_MM_GATHER_AUTO(_f16, __half)
// @tvm_ffi fcnmm_scatter_auto_f16
FFI_MM_SCATTER_AUTO(_f16, __half)

// --- BFloat16 gather/scatter (requires CUDA 11.0+) ---
// @tvm_ffi fcnmm_gather_auto_bf16
FFI_MM_GATHER_AUTO(_bf16, __nv_bfloat16)
// @tvm_ffi fcnmm_scatter_auto_bf16
FFI_MM_SCATTER_AUTO(_bf16, __nv_bfloat16)
