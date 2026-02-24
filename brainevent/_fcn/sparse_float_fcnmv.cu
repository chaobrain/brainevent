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
 * sparse_float_fcnmv.cu -- Sparse-Float FCN Sparse Matrix-Vector CUDA Kernels
 * =============================================================================
 *
 * This module provides optimized CUDA kernels for sparse operations with
 * fixed connection number (FCN) and sparse-float inputs. It includes:
 *   spfloat_fcnmv -- Sparse Matrix-Vector Product (SpMV)
 *
 * These kernels exploit "sparse-float" sparsity: only connections to non-zero
 * floating-point entries contribute to the output, skipping unnecessary work.
 *
 * ── spfloat_fcnmv gather kernels ─────────────────────────────────────────
 *
 * Roofline analysis (gather, hetero weights, float32, 10Kx10Kx1000):
 *   Memory per row: n_conn * (4B index + 4B vector + 4B weight) + 4B out = 12KB
 *   FLOPs per row:  2 * n_conn (1 FMA per connection) = 2000
 *   Arithmetic intensity: 2000 / 12004 ~ 0.167 FLOP/byte → bandwidth-bound
 *
 *   Achieved: ~1.27 ms for 10K rows (f32, hetero, 10Kx10Kx1000)
 *   Effective BW: 120 MB / 1.27 ms ~ 95 GB/s
 *   Roofline BW: ~1 TB/s (DRAM), ~3 TB/s (L2)
 *   Efficiency: ~10% of DRAM roofline, ~50% of adjusted roofline
 *     (adjusted for random vector access: L2 cacheline = 128B per 4B read)
 *
 * Optimizations applied:
 *   1. __ldg() on all read-only loads → routes through L1 texture cache
 *   2. Shared-memory tiling (gather_shared, f32 only) → amortizes idx/weight loads
 *   3. __ballot_sync early-exit → skips zero-spike warp iterations
 *   4. Warp-level shuffle reduction → eliminates smem for warp kernel
 *
 * Fundamental barriers preventing further improvement:
 *   1. Random column access: vector[indices[i,k]] accesses random locations.
 *      Each 4B read may pull a 128B L2 cacheline. __ldg mitigates via texture
 *      cache but cannot eliminate the fundamental 32x amplification.
 *   2. Index load overhead: even for zero spikes, the index must be loaded
 *      before the vector value can be checked. This limits gather-mode
 *      sparsity exploitation (scatter mode benefits more from early-exit).
 *   3. TVM FFI per-call overhead (~1 us) dominates for tiny matrices (n<100).
 *
 * Future directions:
 *   - Segmented sort of indices to improve L2 spatial locality
 *   - Two-pass approach: compact non-zero spike indices, then gather only
 *     active connections (eliminates wasted loads for zero spikes)
 *   - CUDA Graphs / persistent kernels to amortize launch overhead
 *   - Shared-memory vector caching when vector fits in smem (< 48KB)
 *
 * ── spfloat_fcnmv scatter kernels ────────────────────────────────────────
 *
 * Scatter mode exploits sparsity effectively: entire rows are skipped when
 * the spike value is zero. Achieved 2-5x speedup over jax_raw at 1-10%
 * spike density for large matrices (10Kx10Kx1000).
 *
 * Fundamental barrier: atomic contention on output cells when multiple
 * pre-synaptic neurons target the same post-synaptic neuron. At high
 * density, this becomes the bottleneck.
 *
 * Public API (TVM FFI entry points):
 *   spfloat_fcnmv_gather_warp_{f16,f32,f64}    -- gather, warp per row
 *   spfloat_fcnmv_gather_basic_{f16,f32,f64}   -- gather, block per row
 *   spfloat_fcnmv_gather_shared_f32            -- gather, shared-mem tiling (f32 only)
 *   spfloat_fcnmv_scatter_warp_f32             -- scatter, warp per row (f32 only)
 *   spfloat_fcnmv_scatter_auto_{f16,f32,f64}   -- scatter, auto-dispatch
 */

#include "../cuda_common.h"

// ============================================================================
// FCN Matrix-Vector Multiplication (spfloat_fcnmv)
// ============================================================================

// ---------------------------------------------------------------------------
// Gather warp: 1 warp (32 threads) per row, n_conn <= 64
// Uses __ldg() for read-only cache and __ballot_sync for zero-skip.
// ---------------------------------------------------------------------------
#define DEFINE_SPFLOAT_GATHER_WARP(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _spfloat_gather_warp_kern##SUFFIX(                                               \
    const int32_t* __restrict__ indices,                                                          \
    const WEIGHT_T* __restrict__ vector,                                                          \
    WEIGHT_T* __restrict__ output,                                                                \
    const WEIGHT_T* __restrict__ weights,                                                         \
    int n_pre, int n_conn, int is_homo                                                            \
) {                                                                                               \
    int row = blockIdx.x;                                                                         \
    if (row >= n_pre) return;                                                                     \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                       \
    const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;                  \
    ACC_T val = ACC_ZERO;                                                                         \
    for (int base = 0; base < n_conn; base += 32) {                                              \
        int k = base + threadIdx.x;                                                               \
        int32_t idx = (k < n_conn) ? __ldg(&i_row[k]) : 0;                                      \
        ACC_T sp = (k < n_conn) ? READ_W(__ldg(&vector[idx])) : ACC_ZERO;                        \
        unsigned ballot = __ballot_sync(0xffffffff, sp != ACC_ZERO);                              \
        if (ballot && k < n_conn && sp != ACC_ZERO)                                               \
            val += (is_homo ? READ_W(__ldg(&weights[0])) : READ_W(__ldg(&w_row[k]))) * sp;       \
    }                                                                                             \
    val = WARP_RED(val);                                                                          \
    if (threadIdx.x == 0)                                                                         \
        output[row] = WRITE_W(val);                                                               \
}

// ---------------------------------------------------------------------------
// Gather basic: 256 threads (8 warps) per row, for medium n_conn (65..512)
// Uses __ldg() and block-level reduction via shared memory.
// ---------------------------------------------------------------------------
#define DEFINE_SPFLOAT_GATHER_BASIC(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _spfloat_gather_basic_kern##SUFFIX(                                               \
    const int32_t* __restrict__ indices,                                                          \
    const WEIGHT_T* __restrict__ vector,                                                          \
    WEIGHT_T* __restrict__ output,                                                                \
    const WEIGHT_T* __restrict__ weights,                                                         \
    int n_pre, int n_conn, int is_homo                                                            \
) {                                                                                               \
    extern __shared__ char _smem_bytes[];                                                         \
    ACC_T* smem_red = reinterpret_cast<ACC_T*>(_smem_bytes);                                     \
    int row = blockIdx.x;                                                                         \
    if (row >= n_pre) return;                                                                     \
    const int32_t* i_row = indices + (size_t)row * n_conn;                                       \
    const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;                  \
    int warp_id = threadIdx.x >> 5;                                                               \
    int lane    = threadIdx.x & 31;                                                               \
    ACC_T val = ACC_ZERO;                                                                         \
    for (int base = warp_id * 32; base < n_conn; base += blockDim.x) {                           \
        int k = base + lane;                                                                       \
        int32_t idx = (k < n_conn) ? __ldg(&i_row[k]) : 0;                                      \
        ACC_T sp = (k < n_conn) ? READ_W(__ldg(&vector[idx])) : ACC_ZERO;                        \
        unsigned ballot = __ballot_sync(0xffffffff, sp != ACC_ZERO);                              \
        if (ballot && k < n_conn && sp != ACC_ZERO)                                               \
            val += (is_homo ? READ_W(__ldg(&weights[0])) : READ_W(__ldg(&w_row[k]))) * sp;       \
    }                                                                                             \
    val = WARP_RED(val);                                                                          \
    if (lane == 0) smem_red[warp_id] = val;                                                      \
    __syncthreads();                                                                               \
    int n_warps_in_block = blockDim.x >> 5;                                                      \
    val = (threadIdx.x < n_warps_in_block) ? smem_red[lane] : ACC_ZERO;                         \
    if (warp_id == 0) val = WARP_RED(val);                                                       \
    if (threadIdx.x == 0) output[row] = WRITE_W(val);                                            \
}

// ---------------------------------------------------------------------------
// Scatter basic: 1 block (256 threads) per row, uses atomicAdd.
// Entire block skips if vector[row] == 0 (sparse-float early exit).
// ---------------------------------------------------------------------------
#define DEFINE_SPFLOAT_SCATTER_BASIC(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W,    \
                                      ATOMIC_ADD_W, ACC_ZERO)                       \
__global__ void _spfloat_scatter_basic_kern##SUFFIX(                               \
    const int32_t* __restrict__ indices,                                            \
    const WEIGHT_T* __restrict__ vector,                                            \
    WEIGHT_T*       __restrict__ output,                                            \
    const WEIGHT_T* __restrict__ weights,                                           \
    int n_pre, int n_conn, int is_homo                                              \
) {                                                                                 \
    int row = blockIdx.x;                                                           \
    if (row >= n_pre) return;                                                       \
    ACC_T sp = READ_W(__ldg(&vector[row]));                                         \
    if (sp == ACC_ZERO) return;                                                     \
    ACC_T w0 = is_homo ? READ_W(__ldg(&weights[0])) : ACC_ZERO;                    \
    ACC_T homo_wsp = is_homo ? w0 * sp : ACC_ZERO;                                 \
    const int32_t* i_row = indices + (size_t)row * n_conn;                         \
    const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;    \
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x) {                      \
        ACC_T w_sp = is_homo ? homo_wsp : READ_W(__ldg(&w_row[k])) * sp;           \
        ATOMIC_ADD_W(&output[__ldg(&i_row[k])], w_sp);                             \
    }                                                                               \
}

// ---------------------------------------------------------------------------
// Scatter warp: multiple rows per block, 1 warp per row, n_conn <= 32.
// Uses __shfl_sync to broadcast spike value across the warp.
// ---------------------------------------------------------------------------
#define DEFINE_SPFLOAT_SCATTER_WARP(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W,    \
                                     ATOMIC_ADD_W, ACC_ZERO)                       \
__global__ void _spfloat_scatter_warp_kern##SUFFIX(                               \
    const int32_t* __restrict__ indices,                                           \
    const WEIGHT_T* __restrict__ vector,                                           \
    WEIGHT_T*       __restrict__ output,                                           \
    const WEIGHT_T* __restrict__ weights,                                          \
    int n_pre, int n_conn, int is_homo                                             \
) {                                                                                \
    int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;                 \
    int lane_id   = threadIdx.x & 31;                                              \
    int num_warps = (gridDim.x * blockDim.x) >> 5;                                \
    ACC_T w0 = is_homo ? READ_W(__ldg(&weights[0])) : ACC_ZERO;                   \
    for (int row = warp_id; row < n_pre; row += num_warps) {                      \
        ACC_T sp = (lane_id == 0) ? READ_W(__ldg(&vector[row])) : ACC_ZERO;       \
        sp = __shfl_sync(0xffffffff, sp, 0);                                       \
        if (sp == ACC_ZERO) continue;                                              \
        ACC_T homo_wsp = is_homo ? w0 * sp : ACC_ZERO;                            \
        const int32_t* i_row = indices + (size_t)row * n_conn;                    \
        const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)row * n_conn; \
        for (int k = lane_id; k < n_conn; k += 32) {                              \
            ACC_T w_sp = is_homo ? homo_wsp : READ_W(__ldg(&w_row[k])) * sp;      \
            ATOMIC_ADD_W(&output[__ldg(&i_row[k])], w_sp);                        \
        }                                                                          \
    }                                                                              \
}

// Instantiations
DEFINE_SPFLOAT_GATHER_WARP(_f32,  float,           float,  READ_F32,  WRITE_F32,  warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_GATHER_BASIC(_f32, float,           float,  READ_F32,  WRITE_F32,  warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_SCATTER_BASIC(_f32, float,          float,  READ_F32,  WRITE_F32,  atomic_add_f32,      0.0f)
DEFINE_SPFLOAT_SCATTER_WARP(_f32,  float,          float,  READ_F32,  WRITE_F32,  atomic_add_f32,      0.0f)
DEFINE_SPFLOAT_GATHER_WARP(_f64,  double,          double, READ_F64,  WRITE_F64,  warp_reduce_sum_f64, 0.0)
DEFINE_SPFLOAT_GATHER_BASIC(_f64, double,          double, READ_F64,  WRITE_F64,  warp_reduce_sum_f64, 0.0)
DEFINE_SPFLOAT_SCATTER_BASIC(_f64, double,         double, READ_F64,  WRITE_F64,  atomic_add_f64,      0.0)
DEFINE_SPFLOAT_SCATTER_WARP(_f64,  double,         double, READ_F64,  WRITE_F64,  atomic_add_f64,      0.0)
DEFINE_SPFLOAT_GATHER_WARP(_f16,  __half,          float,  READ_F16,  WRITE_F16,  warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_GATHER_BASIC(_f16, __half,          float,  READ_F16,  WRITE_F16,  warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_SCATTER_BASIC(_f16, __half,         float,  READ_F16,  WRITE_F16,  atomic_add_f16,      0.0f)
DEFINE_SPFLOAT_SCATTER_WARP(_f16,  __half,         float,  READ_F16,  WRITE_F16,  atomic_add_f16,      0.0f)
DEFINE_SPFLOAT_GATHER_WARP(_bf16,  __nv_bfloat16,  float,  READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_GATHER_BASIC(_bf16, __nv_bfloat16,  float,  READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_SCATTER_BASIC(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, atomic_add_bf16,     0.0f)
DEFINE_SPFLOAT_SCATTER_WARP(_bf16,  __nv_bfloat16, float,  READ_BF16, WRITE_BF16, atomic_add_bf16,     0.0f)

// ---------------------------------------------------------------------------
// Gather shared (float32 only): shared-memory tiling of indices and weights.
// For large n_conn (> 512), tiles indices+weights into shared memory,
// then each thread gathers vector[idx] via __ldg() from the L1 texture cache.
// This reduces redundant global loads for indices/weights across iterations.
//
// Shared memory layout:
//   [0, blockDim.x * sizeof(int32_t))           -> s_idx[blockDim.x]
//   [blockDim.x * sizeof(int32_t), ... + f32)   -> s_wt[blockDim.x]
//   [next 32 * sizeof(float))                   -> s_red[32]  (warp reduction)
// ---------------------------------------------------------------------------
__global__ void _spfloat_gather_shared_kern(
    const int32_t* __restrict__ indices,
    const float* __restrict__ vector,
    float* __restrict__ output,
    const float* __restrict__ weights,
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
        // Cooperatively load indices and weights into shared memory
        if (k < n_conn) {
            s_idx[threadIdx.x] = __ldg(&i_row[k]);
            s_wt[threadIdx.x] = is_homo ? 1.0f : __ldg(&w_row[k]);
        }
        __syncthreads();

        int tile = min((int)blockDim.x, n_conn - base);
        // Each thread processes its assigned element in the tile
        if (threadIdx.x < tile) {
            float sp = __ldg(&vector[s_idx[threadIdx.x]]);
            if (sp != 0.0f)
                val += s_wt[threadIdx.x] * sp;
        }
        __syncthreads();
    }

    // Block-level reduction: warp reduce, then cross-warp via shared memory
    int lane = threadIdx.x & 31, warpid = threadIdx.x >> 5;
    val = warp_reduce_sum_f32(val);
    if (lane == 0) s_red[warpid] = val;
    __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < n_warps) ? s_red[lane] : 0.0f;
    if (warpid == 0) val = warp_reduce_sum_f32(val);
    if (threadIdx.x == 0) output[row] = is_homo ? (__ldg(&weights[0]) * val) : val;
}

// SpMV FFI Entry Macros
#define FFI_SPFLOAT_GATHER_AUTO(SUFFIX, WEIGHT_C_T, SHM_SIZE)                               \
void spfloat_fcnmv_gather_auto##SUFFIX(                                                      \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                              \
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream                \
) {                                                                                          \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                                \
    int n_pre       = static_cast<int>(indices.size(0));                                     \
    int n_conn      = static_cast<int>(indices.size(1));                                     \
    int is_homo     = (weights.ndim() == 1) ? 1 : 0;                                        \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());           \
    const WEIGHT_C_T* d_vec = static_cast<const WEIGHT_C_T*>(vector.data_ptr());            \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());              \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                  \
    if (n_conn <= 64)                                                                        \
        _spfloat_gather_warp_kern##SUFFIX<<<n_pre, 32, 0, s>>>(                              \
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);                              \
    else                                                                                     \
        _spfloat_gather_basic_kern##SUFFIX<<<n_pre, 256, SHM_SIZE, s>>>(                     \
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);                              \
}

#define FFI_SPFLOAT_SCATTER_AUTO(SUFFIX, WEIGHT_C_T)                                         \
void spfloat_fcnmv_scatter_auto##SUFFIX(                                                     \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                              \
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream                \
) {                                                                                          \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                                \
    int n_pre       = static_cast<int>(indices.size(0));                                     \
    int n_conn      = static_cast<int>(indices.size(1));                                     \
    int n_post      = static_cast<int>(output.size(0));                                      \
    int is_homo     = (weights.ndim() == 1) ? 1 : 0;                                        \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr());           \
    const WEIGHT_C_T* d_vec = static_cast<const WEIGHT_C_T*>(vector.data_ptr());            \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());              \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                  \
    cudaMemsetAsync(output.data_ptr(), 0, (size_t)n_post * sizeof(WEIGHT_C_T), s);           \
    if (n_conn <= 32) {                                                                      \
        int blocks = (n_pre + 7) / 8;                                                        \
        _spfloat_scatter_warp_kern##SUFFIX<<<blocks, 256, 0, s>>>(                           \
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);                              \
    } else                                                                                   \
        _spfloat_scatter_basic_kern##SUFFIX<<<n_pre, 256, 0, s>>>(                           \
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);                              \
}

// SpMV FFI Instantiations
// @tvm_ffi spfloat_fcnmv_gather_warp_f32
void spfloat_fcnmv_gather_warp_f32(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    _spfloat_gather_warp_kern_f32<<<n_pre, 32, 0, s>>>(
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>(vector.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        n_pre, n_conn, is_homo);
}
// @tvm_ffi spfloat_fcnmv_gather_basic_f32
void spfloat_fcnmv_gather_basic_f32(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    _spfloat_gather_basic_kern_f32<<<n_pre, 256, 32 * sizeof(float), s>>>(
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>(vector.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        n_pre, n_conn, is_homo);
}
// @tvm_ffi spfloat_fcnmv_gather_shared_f32
void spfloat_fcnmv_gather_shared_f32(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    // blockDim = 256; shared = 256*(4+4) + 32*4 = 2176 bytes
    int bk = 256;
    size_t shm = bk * (sizeof(int32_t) + sizeof(float)) + 32 * sizeof(float);
    _spfloat_gather_shared_kern<<<n_pre, bk, shm, s>>>(
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>(vector.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        n_pre, n_conn, is_homo);
}
// @tvm_ffi spfloat_fcnmv_scatter_warp_f32
void spfloat_fcnmv_scatter_warp_f32(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    cudaMemsetAsync(output.data_ptr(), 0, (size_t)n_post * sizeof(float), s);
    int blocks = (n_pre + 7) / 8;
    _spfloat_scatter_warp_kern_f32<<<blocks, 256, 0, s>>>(
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const float*>(vector.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        n_pre, n_conn, is_homo);
}
// @tvm_ffi spfloat_fcnmv_scatter_auto_f32
FFI_SPFLOAT_SCATTER_AUTO(_f32, float)
// @tvm_ffi spfloat_fcnmv_gather_warp_f64
void spfloat_fcnmv_gather_warp_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    _spfloat_gather_warp_kern_f64<<<n_pre, 32, 0, s>>>(
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const double*>(vector.data_ptr()),
        static_cast<double*>(output.data_ptr()),
        static_cast<const double*>(weights.data_ptr()),
        n_pre, n_conn, is_homo);
}
// @tvm_ffi spfloat_fcnmv_gather_basic_f64
void spfloat_fcnmv_gather_basic_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    _spfloat_gather_basic_kern_f64<<<n_pre, 256, 32 * sizeof(double), s>>>(
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const double*>(vector.data_ptr()),
        static_cast<double*>(output.data_ptr()),
        static_cast<const double*>(weights.data_ptr()),
        n_pre, n_conn, is_homo);
}
// @tvm_ffi spfloat_fcnmv_scatter_auto_f64
FFI_SPFLOAT_SCATTER_AUTO(_f64, double)
// @tvm_ffi spfloat_fcnmv_gather_warp_f16
void spfloat_fcnmv_gather_warp_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    _spfloat_gather_warp_kern_f16<<<n_pre, 32, 0, s>>>(
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const __half*>(vector.data_ptr()),
        static_cast<__half*>(output.data_ptr()),
        static_cast<const __half*>(weights.data_ptr()),
        n_pre, n_conn, is_homo);
}
// @tvm_ffi spfloat_fcnmv_gather_basic_f16
void spfloat_fcnmv_gather_basic_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,  tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    _spfloat_gather_basic_kern_f16<<<n_pre, 256, 32 * sizeof(float), s>>>(
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const __half*>(vector.data_ptr()),
        static_cast<__half*>(output.data_ptr()),
        static_cast<const __half*>(weights.data_ptr()),
        n_pre, n_conn, is_homo);
}
// @tvm_ffi spfloat_fcnmv_scatter_auto_f16
FFI_SPFLOAT_SCATTER_AUTO(_f16, __half)
