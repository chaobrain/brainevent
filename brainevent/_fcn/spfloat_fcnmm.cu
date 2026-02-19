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
 * spfloat_fcnmm.cu — Sparse-Float Event-Driven FCN Matrix-Matrix CUDA Kernels
 * =============================================================================
 *
 * Python API: brainevent.spfloat_fcnmm(weights, indices, matrix, *, shape, transpose, backend)
 *
 * Sparse-float event-driven matrix--matrix product with fixed connection number.
 *
 * Computes  Y = W @ M  (or  Y = W^T @ M  when transpose=True)
 * where W is a sparse weight matrix stored in fixed-connection-number
 * format and M is a dense matrix whose entries may be sparse-float values.
 * Unlike binary_fcnmm which treats non-zero entries as 1, this variant
 * preserves their actual floating-point values.
 *
 * Parameters
 * ----------
 * weights : shape (1,) for homogeneous or (num_pre, num_conn) for heterogeneous,
 *           floating-point dtype (float16 / float32 / float64).
 * indices : shape (num_pre, num_conn), int32 — post-synaptic column indices.
 * matrix  : shape (num_post, n_col) for gather or (num_pre, n_col) for scatter, float dtype.
 *           Zero entries are skipped; non-zero entries use their actual value.
 * shape   : logical (num_pre, num_post) shape of the dense weight matrix.
 * transpose=False (gather mode):
 *   Y[i,j] = sum_{k: M[indices[i,k],j] != 0} weights[i,k] * M[indices[i,k],j]
 * transpose=True  (scatter mode):
 *   Y[indices[i,k],j] += weights[i,k] * M[i,j]  for all i,k,j where M[i,j] != 0
 *
 * Supported dtypes: float32 (default), float64 (_f64 suffix), float16 (_f16 suffix).
 * vec4 and shared-memory tile paths are float32-only; f16/f64 use basic/warp kernels.
 * Scatter early-exit: tile-level __ballot_sync checks M[i,:] for zeros before atomics.
 *
 * IMPORTANT: weights.data_ptr() is a GPU device pointer — NEVER dereference on host.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp16.h>

// ===========================================================================
// Sparse-Float FCN Matrix-Matrix (spfloat_fcnmm) CUDA Kernels
//
// KEY DIFFERENCE FROM fcnmm: skip zero entries in the input matrix M.
//
// Gather mode (transpose=False):
//   Y[i,j] = sum_k w[i,k] * M[idx[i,k],j]   (skip when M[...] == 0)
//   - Per-element zero check avoids FMA for zero M entries
//
// Scatter mode (transpose=True):
//   Y[idx[i,k],j] += w[i,k] * M[i,j]         (skip when M[i,j] == 0)
//   - TILE-LEVEL EARLY EXIT: if entire M[i, j_tile] is zero, skip all n_conn
//     atomic scatter operations for that tile using warp ballot
//   - At 5% SNN firing rate: ~95% of scatter tiles skipped entirely
//
// IMPORTANT: weights.data_ptr() returns a GPU device pointer.
// NEVER dereference on host.
// ===========================================================================

// ===========================================================================
// GATHER kernels: Y[i,j] = sum_k w[i,k] * M[idx[i,k],j]  (skip M == 0)
// ===========================================================================

// Basic gather: one thread per output element Y[i,j].
// Iterates over n_conn connections; skips FMA when M[idx,j] == 0.
// Grid: (n_pre, ceil(n_col/64)), Block: (64,)
__global__ void _spfloat_mm_gather_basic_kern(
    const int32_t* __restrict__ indices,   // [n_pre, n_conn]
    const float*   __restrict__ matrix,    // [k_dim, n_col]
    float*         __restrict__ output,    // [n_pre, n_col]
    const float*   __restrict__ weights,   // [1] or [n_pre, n_conn]
    int n_pre, int n_conn, int n_col, int is_homo
) {
    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= n_pre || j >= n_col) return;
    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const float*   w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;
    float acc = 0.0f;
    for (int k = 0; k < n_conn; k++) {
        float m_val = matrix[(size_t)idx_row[k] * n_col + j];
        if (m_val != 0.0f) {   // Skip zero M entries (event-driven optimization)
            float w = is_homo ? weights[0] : w_row[k];
            acc += w * m_val;
        }
    }
    output[(size_t)i * n_col + j] = acc;
}

// Shared-memory gather: tiles connection list into shmem to reduce bandwidth.
// Per-connection zero check for event-driven sparsity.
// Grid: (n_pre, ceil(n_col/64)), Block: (64,)
// Shared mem: SPFLOAT_MM_TK * (sizeof(int32_t) + sizeof(float)) bytes
#define SPFLOAT_MM_TK 128
__global__ void _spfloat_mm_gather_shared_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ matrix,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    extern __shared__ char smem_mm[];
    int32_t* s_idx = reinterpret_cast<int32_t*>(smem_mm);
    float*   s_w   = reinterpret_cast<float*>(smem_mm + SPFLOAT_MM_TK * sizeof(int32_t));

    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= n_pre) return;

    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const float*   w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;

    float acc = 0.0f;
    for (int k0 = 0; k0 < n_conn; k0 += SPFLOAT_MM_TK) {
        int tile = (k0 + SPFLOAT_MM_TK < n_conn) ? SPFLOAT_MM_TK : (n_conn - k0);
        // Cooperatively load connection tile into shmem
        for (int t = threadIdx.x; t < tile; t += blockDim.x) {
            s_idx[t] = idx_row[k0 + t];
            s_w[t]   = is_homo ? 1.0f : w_row[k0 + t];
        }
        __syncthreads();
        if (j < n_col) {
            for (int t = 0; t < tile; t++) {
                float m_val = matrix[(size_t)s_idx[t] * n_col + j];
                if (m_val != 0.0f)   // Skip zero M entries
                    acc += s_w[t] * m_val;
            }
        }
        __syncthreads();
    }
    if (j < n_col)
        output[(size_t)i * n_col + j] = is_homo ? (weights[0] * acc) : acc;
}

// Vectorised gather: float4 loads for M and output when n_col % 4 == 0.
// Any non-zero in float4 group → process all 4 elements (coarse zero check).
// Grid: (n_pre, ceil(n_col/4 / 64)), Block: (64,)
__global__ void _spfloat_mm_gather_vec4_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ matrix,   // [k_dim, n_col], n_col%4==0
    float*         __restrict__ output,   // [n_pre, n_col]
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    int i   = blockIdx.x;
    int j4  = blockIdx.y * blockDim.x + threadIdx.x;  // float4 group index
    int nc4 = n_col >> 2;
    if (i >= n_pre || j4 >= nc4) return;

    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const float*   w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;
    const float4*  mat4    = reinterpret_cast<const float4*>(matrix);

    float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = 0; k < n_conn; k++) {
        float  w = is_homo ? weights[0] : w_row[k];
        float4 m = mat4[(size_t)idx_row[k] * nc4 + j4];
        // Coarse zero check: if any component non-zero, accumulate all 4
        if (m.x != 0.0f || m.y != 0.0f || m.z != 0.0f || m.w != 0.0f) {
            acc.x += w * m.x;
            acc.y += w * m.y;
            acc.z += w * m.z;
            acc.w += w * m.w;
        }
    }
    reinterpret_cast<float4*>(output)[(size_t)i * nc4 + j4] = acc;
}

// Auto-selects the best gather kernel based on n_conn and n_col.
void spfloat_fcnmm_gather_auto_device(
    const float* d_w, const int32_t* d_idx, const float* d_mat, float* d_out,
    int n_pre, int n_conn, int n_col, int is_homo, cudaStream_t s
) {
    int BJ = 64;
    if (n_col % 4 == 0 && n_col >= 64) {
        dim3 grid(n_pre, (n_col / 4 + BJ - 1) / BJ);
        _spfloat_mm_gather_vec4_kern<<<grid, BJ, 0, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    } else if (n_conn > 128) {
        dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
        size_t shm = SPFLOAT_MM_TK * (sizeof(int32_t) + sizeof(float));
        _spfloat_mm_gather_shared_kern<<<grid, BJ, shm, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    } else {
        dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
        _spfloat_mm_gather_basic_kern<<<grid, BJ, 0, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    }
}

// ===========================================================================
// SCATTER kernels (transpose=True)
// Y[idx[i,k],j] += w[i,k] * M[i,j]   (Y pre-zeroed, skip when M[i,j]==0)
// KEY OPTIMIZATION: tile-level early exit using warp ballot when M[i,:] == 0
// ===========================================================================

// Block scatter: one block per pre-neuron, per-element zero check.
// Skips atomicAdd when m_val == 0 (avoids false writes for sparse M).
// Grid: (n_pre,), Block: (256,)
__global__ void _spfloat_mm_scatter_block_kern(
    const int32_t* __restrict__ indices,   // [n_pre, n_conn]
    const float*   __restrict__ matrix,    // [n_pre, n_col]
    float*         __restrict__ output,    // [n_post, n_col] (pre-zeroed)
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    int i = blockIdx.x;
    if (i >= n_pre) return;
    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const float*   w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;
    const float*   m_row   = matrix + (size_t)i * n_col;
    for (int k = 0; k < n_conn; k++) {
        int   tgt = idx_row[k];
        float w   = is_homo ? weights[0] : w_row[k];
        float* out_row = output + (size_t)tgt * n_col;
        for (int j = threadIdx.x; j < n_col; j += blockDim.x) {
            float m_val = m_row[j];
            if (m_val != 0.0f)   // Skip zero M entries (no-op atomicAdd)
                atomicAdd(&out_row[j], w * m_val);
        }
    }
}

// Cached scatter: 2D grid — one block per (pre-neuron, n_col tile).
// TILE-LEVEL EARLY EXIT: loads M[i, j_tile] into shmem, then uses warp ballot
// to detect all-zero tiles. If the tile is all-zero, skip ALL n_conn scatter
// operations for this (i, j_tile) combination.
// At 5% SNN firing rate: ~95% of blocks return after the shmem load + ballot.
// Grid: (n_pre, ceil(n_col/BJ)), Block: (BJ,)
// Shared mem: BJ * sizeof(float) + sizeof(unsigned)
#define SPFLOAT_MM_SCATTER_BJ 128
__global__ void _spfloat_mm_scatter_cached_kern(
    const int32_t* __restrict__ indices,   // [n_pre, n_conn]
    const float*   __restrict__ matrix,    // [n_pre, n_col]
    float*         __restrict__ output,    // [n_post, n_col] (pre-zeroed)
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    extern __shared__ float s_m[];   // M[i, j_tile] tile cache
    __shared__ unsigned s_any_nz;    // Tile sparsity flag (static in __global__ is OK)

    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= n_pre) return;

    // Load M[i, j] tile into shmem (single global read per element)
    float m_val = (j < n_col) ? matrix[(size_t)i * n_col + j] : 0.0f;
    s_m[threadIdx.x] = m_val;

    // Initialize sparsity flag
    if (threadIdx.x == 0) s_any_nz = 0u;
    __syncthreads();

    // Warp-ballot based tile sparsity check:
    // Each warp votes on its 32 elements; if any warp detects a non-zero value,
    // it atomically sets the shared flag.
    unsigned warp_nz = __ballot_sync(0xffffffff, m_val != 0.0f);
    if (warp_nz && ((threadIdx.x & 31) == 0))  // warp leader writes
        atomicOr(&s_any_nz, 1u);
    __syncthreads();

    // TILE-LEVEL EARLY EXIT: skip all n_conn scatter ops if tile is all-zero
    // This is the KEY optimization for sparse input matrices (SNNs at 5% rate)
    if (!s_any_nz) return;

    // Scatter for non-zero elements: each thread handles one j column
    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const float*   w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;
    if (j < n_col && m_val != 0.0f) {
        for (int k = 0; k < n_conn; k++) {
            int   tgt = idx_row[k];
            float w   = is_homo ? weights[0] : w_row[k];
            atomicAdd(&output[(size_t)tgt * n_col + j], w * m_val);
        }
    }
}

// Warp scatter: grid-stride over (pre-neuron, connection) pairs.
// Per-element zero check for M[i,j]; avoids atomicAdd when m_val == 0.
// Grid: (min(4096, ceil(n_pre*n_conn/8)),), Block: (256,)
__global__ void _spfloat_mm_scatter_warp_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ matrix,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    int wid     = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane    = threadIdx.x & 31;
    int n_warps = (gridDim.x * blockDim.x) >> 5;
    int n_pairs = n_pre * n_conn;

    for (int pair = wid; pair < n_pairs; pair += n_warps) {
        int i = pair / n_conn;
        int k = pair % n_conn;
        int   tgt = indices[(size_t)i * n_conn + k];
        float w   = is_homo ? weights[0] : weights[(size_t)i * n_conn + k];
        const float* m_row   = matrix + (size_t)i * n_col;
        float*       out_row = output + (size_t)tgt * n_col;
        for (int j = lane; j < n_col; j += 32) {
            float m_val = m_row[j];
            if (m_val != 0.0f)   // Skip zero M entries
                atomicAdd(&out_row[j], w * m_val);
        }
    }
}

// ===========================================================================
// TVM FFI Entry Points
// Convention: args = (weights, indices, matrix, output, stream)
// weights.data_ptr() is a GPU device pointer — NEVER dereference on host.
// Gather:  matrix [n_post, n_col], output [n_pre, n_col]
// Scatter: matrix [n_pre,  n_col], output [n_post, n_col]
// ===========================================================================

void spfloat_fcnmm_gather_shared_f32(
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
    size_t shm = SPFLOAT_MM_TK * (sizeof(int32_t) + sizeof(float));
    _spfloat_mm_gather_shared_kern<<<grid, BJ, shm, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

void spfloat_fcnmm_gather_vec4_f32(
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
    _spfloat_mm_gather_vec4_kern<<<grid, BJ4, 0, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

void spfloat_fcnmm_gather_auto_f32(
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
    spfloat_fcnmm_gather_auto_device(d_w, d_idx, d_mat, d_out, n_pre, n_conn, n_col, is_homo, s);
}

// --- Scatter entry points (output zeroed before kernel launch) ---

// Auto-selects the best scatter kernel based on n_conn and n_col.
void spfloat_fcnmm_scatter_auto_f32(
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
        // Small n_col: warp-per-(i,k) maximises parallelism over connections
        int n_pairs = n_pre * n_conn;
        int blocks  = min(4096, (n_pairs + 7) / 8);
        _spfloat_mm_scatter_warp_kern<<<blocks, 256, 0, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    } else {
        // Larger n_col: cached M[i] in shmem with tile-level sparsity check.
        // At 5% firing rate: 95% of blocks exit before any scatter op.
        int BJ = SPFLOAT_MM_SCATTER_BJ;
        dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
        size_t shm = BJ * sizeof(float) + sizeof(unsigned);
        _spfloat_mm_scatter_cached_kern<<<grid, BJ, shm, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    }
}

// ===========================================================================
// float64 (double) variants for spfloat_fcnmm — no vec4, no shared tiling
// ===========================================================================

__global__ void _spfloat_mm_gather_basic_kern_f64(
    const int32_t* __restrict__ indices,
    const double*  __restrict__ matrix,
    double*        __restrict__ output,
    const double*  __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= n_pre || j >= n_col) return;
    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const double*  w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;
    double acc = 0.0;
    for (int k = 0; k < n_conn; k++) {
        double m_val = matrix[(size_t)idx_row[k] * n_col + j];
        if (m_val != 0.0) {
            double w = is_homo ? weights[0] : w_row[k];
            acc += w * m_val;
        }
    }
    output[(size_t)i * n_col + j] = acc;
}

__global__ void _spfloat_mm_scatter_block_kern_f64(
    const int32_t* __restrict__ indices,
    const double*  __restrict__ matrix,
    double*        __restrict__ output,
    const double*  __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    int i = blockIdx.x;
    if (i >= n_pre) return;
    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const double*  w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;
    const double*  m_row   = matrix + (size_t)i * n_col;
    for (int k = 0; k < n_conn; k++) {
        int    tgt = idx_row[k];
        double w   = is_homo ? weights[0] : w_row[k];
        double* out_row = output + (size_t)tgt * n_col;
        for (int j = threadIdx.x; j < n_col; j += blockDim.x) {
            double m_val = m_row[j];
            if (m_val != 0.0)
                atomicAdd(&out_row[j], w * m_val);
        }
    }
}

__global__ void _spfloat_mm_scatter_warp_kern_f64(
    const int32_t* __restrict__ indices,
    const double*  __restrict__ matrix,
    double*        __restrict__ output,
    const double*  __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    int wid     = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane    = threadIdx.x & 31;
    int n_warps = (gridDim.x * blockDim.x) >> 5;
    int n_pairs = n_pre * n_conn;
    for (int pair = wid; pair < n_pairs; pair += n_warps) {
        int i = pair / n_conn;
        int k = pair % n_conn;
        int    tgt = indices[(size_t)i * n_conn + k];
        double w   = is_homo ? weights[0] : weights[(size_t)i * n_conn + k];
        const double* m_row   = matrix + (size_t)i * n_col;
        double*       out_row = output + (size_t)tgt * n_col;
        for (int j = lane; j < n_col; j += 32) {
            double m_val = m_row[j];
            if (m_val != 0.0)
                atomicAdd(&out_row[j], w * m_val);
        }
    }
}

void spfloat_fcnmm_gather_auto_f64(
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
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const double*  d_mat = static_cast<const double*>(matrix.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    int BJ = 64;
    dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
    _spfloat_mm_gather_basic_kern_f64<<<grid, BJ, 0, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

void spfloat_fcnmm_scatter_auto_f64(
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
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const double*  d_mat = static_cast<const double*>(matrix.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_col * sizeof(double), s);
    if (n_col <= 64) {
        int n_pairs = n_pre * n_conn;
        int blocks  = min(4096, (n_pairs + 7) / 8);
        _spfloat_mm_scatter_warp_kern_f64<<<blocks, 256, 0, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    } else {
        _spfloat_mm_scatter_block_kern_f64<<<n_pre, 256, 0, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    }
}

// ===========================================================================
// float16 (__half) variants for spfloat_fcnmm — accumulate in float32
// ===========================================================================

__global__ void _spfloat_mm_gather_basic_kern_f16(
    const int32_t* __restrict__ indices,
    const __half*  __restrict__ matrix,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= n_pre || j >= n_col) return;
    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const __half*  w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;
    float acc = 0.0f;
    float w0 = is_homo ? __half2float(weights[0]) : 0.0f;
    for (int k = 0; k < n_conn; k++) {
        float m_val = __half2float(matrix[(size_t)idx_row[k] * n_col + j]);
        if (m_val != 0.0f) {
            float w = is_homo ? w0 : __half2float(w_row[k]);
            acc += w * m_val;
        }
    }
    output[(size_t)i * n_col + j] = __float2half(acc);
}

__global__ void _spfloat_mm_scatter_block_kern_f16(
    const int32_t* __restrict__ indices,
    const __half*  __restrict__ matrix,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    int i = blockIdx.x;
    if (i >= n_pre) return;
    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const __half*  w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;
    const __half*  m_row   = matrix + (size_t)i * n_col;
    float w0 = is_homo ? __half2float(weights[0]) : 0.0f;
    for (int k = 0; k < n_conn; k++) {
        int   tgt = idx_row[k];
        float w   = is_homo ? w0 : __half2float(w_row[k]);
        __half* out_row = output + (size_t)tgt * n_col;
        for (int j = threadIdx.x; j < n_col; j += blockDim.x) {
            float m_val = __half2float(m_row[j]);
            if (m_val != 0.0f)
                atomicAdd(&out_row[j], __float2half(w * m_val));
        }
    }
}

__global__ void _spfloat_mm_scatter_warp_kern_f16(
    const int32_t* __restrict__ indices,
    const __half*  __restrict__ matrix,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    int wid     = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane    = threadIdx.x & 31;
    int n_warps = (gridDim.x * blockDim.x) >> 5;
    int n_pairs = n_pre * n_conn;
    float w0 = is_homo ? __half2float(weights[0]) : 0.0f;
    for (int pair = wid; pair < n_pairs; pair += n_warps) {
        int i = pair / n_conn;
        int k = pair % n_conn;
        int   tgt = indices[(size_t)i * n_conn + k];
        float w   = is_homo ? w0 : __half2float(weights[(size_t)i * n_conn + k]);
        const __half* m_row   = matrix + (size_t)i * n_col;
        __half*       out_row = output + (size_t)tgt * n_col;
        for (int j = lane; j < n_col; j += 32) {
            float m_val = __half2float(m_row[j]);
            if (m_val != 0.0f)
                atomicAdd(&out_row[j], __float2half(w * m_val));
        }
    }
}

void spfloat_fcnmm_gather_auto_f16(
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
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const __half*  d_mat = static_cast<const __half*>(matrix.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    int BJ = 64;
    dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
    _spfloat_mm_gather_basic_kern_f16<<<grid, BJ, 0, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

void spfloat_fcnmm_scatter_auto_f16(
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
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const __half*  d_mat = static_cast<const __half*>(matrix.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_col * sizeof(__half), s);
    if (n_col <= 64) {
        int n_pairs = n_pre * n_conn;
        int blocks  = min(4096, (n_pairs + 7) / 8);
        _spfloat_mm_scatter_warp_kern_f16<<<blocks, 256, 0, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    } else {
        _spfloat_mm_scatter_block_kern_f16<<<n_pre, 256, 0, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    }
}
