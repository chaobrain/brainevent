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
 * sparse_float_fcnmm.cu -- Sparse-Float FCN Sparse Matrix-Matrix CUDA Kernels
 * =============================================================================
 *
 * This module provides optimized CUDA kernels for sparse operations with
 * fixed connection number (FCN) and sparse-float inputs. It includes:
 *   spfloat_fcnmm -- Sparse Matrix-Matrix Product (SpMM)
 *
 * These kernels exploit "sparse-float" sparsity: only connections to non-zero
 * floating-point entries contribute to the output, skipping unnecessary work.
 *
 * ── spfloat_fcnmm gather kernel ──────────────────────────────────────────
 *
 * Roofline analysis (gather, hetero, f32, 5Kx5Kx500, ncol=64):
 *   Per output element (row i, col j):
 *     - n_conn matrix reads: n_conn * 4B (random row, coalesced across j)
 *     - n_conn index reads from smem: ~0B global (tiled, shared by all j threads)
 *     - n_conn weight reads from smem: ~0B global (tiled, shared by all j threads)
 *     - 1 output write: 4B
 *   Amortized smem fills per tile: TILE_K * (4B idx + 4B wt) / blockDim.x per thread
 *   FLOPs per element: 2 * n_conn (1 FMA per connection)
 *   Arithmetic intensity: ~0.5 FLOP/byte → bandwidth-bound
 *
 * Optimizations applied:
 *   1. Shared-memory tiling of indices and weights → eliminates redundant
 *      global loads across threads in the same block (reduction = blockDim.x)
 *   2. __ldg() for matrix reads → L1 texture cache
 *   3. Adaptive block size → matches thread count to output columns
 *   4. Sparsity check on matrix values → skips zero-entry FMAs
 *   5. Homo path: skip weight smem, accumulate raw values, multiply once at end
 *
 * ── spfloat_fcnmm scatter kernel ─────────────────────────────────────────
 *
 * Optimizations applied:
 *   1. Shared-memory caching of matrix row → eliminates n_conn redundant
 *      global reads of the same row (reduction factor = n_conn)
 *   2. Row-level early exit via __syncthreads_count → skips all-zero rows
 *      (At 1% spike rate with ncol=64, ~53% of rows are all-zero.)
 *   3. Adaptive block size → matches thread count to output columns
 *
 * Public API (TVM FFI entry points):
 *   spfloat_densemm_nt_{f16,f32,f64}       -- dense SpMM, non-transpose, warp-per-row
 *   spfloat_densemm_nt_tpe_{f16,f32,f64}   -- dense SpMM, non-transpose, thread-per-element
 *   spfloat_densemm_t_{f16,f32,f64}        -- dense SpMM, transpose mode
 *   spfloat_fcnmm_gather_auto_{f16,f32,f64} -- FCN gather mode (fixed post-connections)
 *   spfloat_fcnmm_scatter_auto_{f16,f32,f64} -- FCN scatter mode (fixed pre-connections)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// ============================================================================
// Warp-level reduction helpers
// ============================================================================

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

// ============================================================================
// Per-dtype atomic-add helpers (ACC_T value -> WEIGHT_T memory)
// ============================================================================

__device__ __inline__ void atomic_add_f32(float* addr, float val) {
    atomicAdd(addr, val);
}

__device__ __inline__ void atomic_add_f64(double* addr, double val) {
    atomicAdd(addr, val);
}

__device__ __inline__ void atomic_add_f16(__half* addr, float val) {
#if __CUDA_ARCH__ >= 700
    atomicAdd(addr, __float2half(val));
#else
    unsigned int* base = reinterpret_cast<unsigned int*>(
        reinterpret_cast<size_t>(addr) & ~(size_t)2
    );
    int shift = ((reinterpret_cast<size_t>(addr) & 2) != 0) ? 16 : 0;
    unsigned int assumed, old_val = *base, updated;
    do {
        assumed = old_val;
        unsigned short h = static_cast<unsigned short>((assumed >> shift) & 0xFFFF);
        float cur = __half2float(*reinterpret_cast<__half*>(&h));
        __half new_h = __float2half(cur + val);
        unsigned short new_us = *reinterpret_cast<unsigned short*>(&new_h);
        updated = (assumed & ~(0xFFFFu << shift)) | (static_cast<unsigned int>(new_us) << shift);
        old_val = atomicCAS(base, assumed, updated);
    } while (assumed != old_val);
#endif
}

__device__ __inline__ void atomic_add_bf16(__nv_bfloat16* addr, float val) {
#if __CUDA_ARCH__ >= 800
    atomicAdd(addr, __float2bfloat16(val));
#else
    unsigned int* base = reinterpret_cast<unsigned int*>(
        reinterpret_cast<size_t>(addr) & ~(size_t)2
    );
    int shift = ((reinterpret_cast<size_t>(addr) & 2) != 0) ? 16 : 0;
    unsigned int assumed, old_val = *base, updated;
    do {
        assumed = old_val;
        unsigned short h = static_cast<unsigned short>((assumed >> shift) & 0xFFFF);
        float cur = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&h));
        __nv_bfloat16 new_h = __float2bfloat16(cur + val);
        unsigned short new_us = *reinterpret_cast<unsigned short*>(&new_h);
        updated = (assumed & ~(0xFFFFu << shift)) | (static_cast<unsigned int>(new_us) << shift);
        old_val = atomicCAS(base, assumed, updated);
    } while (assumed != old_val);
#endif
}

// ============================================================================
// Per-dtype conversion macros
// ============================================================================

#define READ_F32(x)   (x)
#define WRITE_F32(x)  (x)
#define READ_F64(x)   (x)
#define WRITE_F64(x)  (x)
#define READ_F16(x)   __half2float(x)
#define WRITE_F16(x)  __float2half(x)
#define READ_BF16(x)  __bfloat162float(x)
#define WRITE_BF16(x) __float2bfloat16(x)

// ============================================================================
// FCN Matrix-Matrix Multiplication (spfloat_fcnmm)
// ============================================================================

#define DEFINE_SPFLOAT_MM_NT_WPR(SUFFIX, WEIGHT_T, ACC_T, CHUNK_N,        \
                                  READ_W, WRITE_W, READ_S,                  \
                                  WARP_RED, ACC_ZERO)                        \
__global__ void _spfloat_mm_nt_wpr_kern##SUFFIX(                            \
    const WEIGHT_T* __restrict__ weights,                                    \
    const WEIGHT_T* __restrict__ spikes,                                     \
    WEIGHT_T*       __restrict__ output,                                     \
    int m, int k, int n                                                      \
) {                                                                          \
    int warp_id = threadIdx.x >> 5;                                          \
    int lane    = threadIdx.x & 31;                                          \
    int warps_per_block = blockDim.x >> 5;                                   \
    int row = blockIdx.x * warps_per_block + warp_id;                        \
    if (row >= m) return;                                                    \
    int col_start = blockIdx.y * CHUNK_N;                                    \
    int chunk_n = min(CHUNK_N, n - col_start);                               \
    const WEIGHT_T* w_row = weights + (size_t)row * k;                       \
    ACC_T acc[CHUNK_N];                                                      \
    _Pragma("unroll")                                                        \
    for (int j = 0; j < CHUNK_N; j++) acc[j] = ACC_ZERO;                    \
    for (int l = lane; l < k; l += 32) {                                     \
        ACC_T w_val = READ_W(__ldg(&w_row[l]));                              \
        const WEIGHT_T* spk_l = spikes + (size_t)l * n + col_start;         \
        _Pragma("unroll")                                                    \
        for (int j = 0; j < CHUNK_N; j++)                                   \
            if (j < chunk_n)                                                 \
                acc[j] += w_val * READ_S(__ldg(&spk_l[j]));                 \
    }                                                                        \
    WEIGHT_T* out_row = output + (size_t)row * n + col_start;               \
    _Pragma("unroll")                                                        \
    for (int j = 0; j < CHUNK_N; j++) {                                     \
        ACC_T val = WARP_RED(acc[j]);                                        \
        if (lane == 0 && j < chunk_n) out_row[j] = WRITE_W(val);            \
    }                                                                        \
}

#define DEFINE_SPFLOAT_MM_NT_TPE(SUFFIX, WEIGHT_T, ACC_T,        \
                                  READ_W, WRITE_W, READ_S, ACC_ZERO) \
__global__ void _spfloat_mm_nt_tpe_kern##SUFFIX(                  \
    const WEIGHT_T* __restrict__ weights,                          \
    const WEIGHT_T* __restrict__ spikes,                           \
    WEIGHT_T*       __restrict__ output,                           \
    int m, int k, int n                                            \
) {                                                                \
    int warp_id = threadIdx.x >> 5;                                \
    int lane    = threadIdx.x & 31;                                \
    int warps_per_block = blockDim.x >> 5;                         \
    int row = blockIdx.x * warps_per_block + warp_id;              \
    int col = blockIdx.y * 32 + lane;                              \
    if (row >= m || col >= n) return;                              \
    const WEIGHT_T* w_row = weights + (size_t)row * k;             \
    ACC_T acc = ACC_ZERO;                                          \
    int l = 0;                                                     \
    for (; l <= k - 4; l += 4) {                                   \
        ACC_T sv0 = READ_S(__ldg(&spikes[(size_t)(l+0) * n + col])); \
        ACC_T sv1 = READ_S(__ldg(&spikes[(size_t)(l+1) * n + col])); \
        ACC_T sv2 = READ_S(__ldg(&spikes[(size_t)(l+2) * n + col])); \
        ACC_T sv3 = READ_S(__ldg(&spikes[(size_t)(l+3) * n + col])); \
        bool any_nz = (sv0 != ACC_ZERO) | (sv1 != ACC_ZERO) |     \
                      (sv2 != ACC_ZERO) | (sv3 != ACC_ZERO);       \
        if (__ballot_sync(__activemask(), any_nz) == 0u) continue; \
        acc += READ_W(__ldg(&w_row[l+0])) * sv0;                   \
        acc += READ_W(__ldg(&w_row[l+1])) * sv1;                   \
        acc += READ_W(__ldg(&w_row[l+2])) * sv2;                   \
        acc += READ_W(__ldg(&w_row[l+3])) * sv3;                   \
    }                                                              \
    for (; l < k; l++) {                                           \
        ACC_T sv = READ_S(__ldg(&spikes[(size_t)l * n + col]));    \
        if (__ballot_sync(__activemask(), sv != ACC_ZERO) == 0u) continue; \
        acc += READ_W(__ldg(&w_row[l])) * sv;                      \
    }                                                              \
    output[(size_t)row * n + col] = WRITE_W(acc);                  \
}

#define DEFINE_SPFLOAT_MM_T(SUFFIX, WEIGHT_T, ACC_T, CHUNK_N,        \
                             READ_W, WRITE_W, READ_S,                  \
                             WARP_RED, ACC_ZERO)                        \
__global__ void _spfloat_mm_t_kern##SUFFIX(                            \
    const WEIGHT_T* __restrict__ weights,                              \
    const WEIGHT_T* __restrict__ spikes,                               \
    WEIGHT_T*       __restrict__ output,                               \
    int m, int k, int n                                                \
) {                                                                    \
    int warp_id = threadIdx.x >> 5;                                    \
    int lane    = threadIdx.x & 31;                                    \
    int warps_per_block = blockDim.x >> 5;                             \
    int row = blockIdx.x * warps_per_block + warp_id;                  \
    if (row >= m) return;                                              \
    int col_start = blockIdx.y * CHUNK_N;                              \
    int chunk_n = min(CHUNK_N, n - col_start);                         \
    const WEIGHT_T* s_row = spikes + (size_t)row * k;                  \
    ACC_T acc[CHUNK_N];                                                \
    _Pragma("unroll")                                                  \
    for (int j = 0; j < CHUNK_N; j++) acc[j] = ACC_ZERO;              \
    for (int l = lane; l < k; l += 32) {                               \
        ACC_T spk_val = READ_S(__ldg(&s_row[l]));                      \
        if (__ballot_sync(__activemask(), spk_val != ACC_ZERO) == 0u)  \
            continue;                                                  \
        if (spk_val != ACC_ZERO) {                                     \
            const WEIGHT_T* w_l = weights + (size_t)l * n + col_start; \
            _Pragma("unroll")                                          \
            for (int j = 0; j < CHUNK_N; j++)                         \
                if (j < chunk_n)                                       \
                    acc[j] += spk_val * READ_W(__ldg(&w_l[j]));        \
        }                                                              \
    }                                                                  \
    WEIGHT_T* out_row = output + (size_t)row * n + col_start;          \
    _Pragma("unroll")                                                  \
    for (int j = 0; j < CHUNK_N; j++) {                                \
        ACC_T val = WARP_RED(acc[j]);                                  \
        if (lane == 0 && j < chunk_n) out_row[j] = WRITE_W(val);      \
    }                                                                  \
}

// SpMM Instantiations
DEFINE_SPFLOAT_MM_NT_WPR(_f32, float,  float,  32, READ_F32, WRITE_F32, READ_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_MM_NT_TPE(_f32, float,  float,      READ_F32, WRITE_F32, READ_F32,                      0.0f)
DEFINE_SPFLOAT_MM_T(_f32,      float,  float,  32, READ_F32, WRITE_F32, READ_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_MM_NT_WPR(_f64, double, double, 16, READ_F64, WRITE_F64, READ_F64, warp_reduce_sum_f64, 0.0)
DEFINE_SPFLOAT_MM_NT_TPE(_f64, double, double,     READ_F64, WRITE_F64, READ_F64,                      0.0)
DEFINE_SPFLOAT_MM_T(_f64,      double, double, 16, READ_F64, WRITE_F64, READ_F64, warp_reduce_sum_f64, 0.0)
DEFINE_SPFLOAT_MM_NT_WPR(_f16, __half, float,  32, READ_F16, WRITE_F16, READ_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_MM_NT_TPE(_f16, __half, float,      READ_F16, WRITE_F16, READ_F16,                      0.0f)
DEFINE_SPFLOAT_MM_T(_f16,      __half, float,  32, READ_F16, WRITE_F16, READ_F16, warp_reduce_sum_f32, 0.0f)

// SpMM FFI Entries
#define FFI_SPFLOAT_MM_NT(SUFFIX, WEIGHT_C_T, CHUNK_N_VAL)                                  \
void spfloat_densemm_nt##SUFFIX(                                                             \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView spikes,                               \
    tvm::ffi::TensorView output,  int64_t stream                                             \
) {                                                                                          \
    cudaStream_t s    = reinterpret_cast<cudaStream_t>(stream);                              \
    int m       = static_cast<int>(weights.size(0));                                         \
    int k       = static_cast<int>(weights.size(1));                                         \
    int n       = static_cast<int>(spikes.size(1));                                          \
    int warps   = 8;                                                                         \
    int m_blks  = (m + warps - 1) / warps;                                                  \
    int n_chnks = (n + CHUNK_N_VAL - 1) / CHUNK_N_VAL;                                      \
    dim3 grid(m_blks, n_chnks);                                                              \
    _spfloat_mm_nt_wpr_kern##SUFFIX<<<grid, 256, 0, s>>>(                                    \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                                  \
        static_cast<const WEIGHT_C_T*>(spikes.data_ptr()),                                   \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                                         \
        m, k, n);                                                                            \
}

#define FFI_SPFLOAT_MM_NT_TPE(SUFFIX, WEIGHT_C_T)                                           \
void spfloat_densemm_nt_tpe##SUFFIX(                                                         \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView spikes,                               \
    tvm::ffi::TensorView output,  int64_t stream                                             \
) {                                                                                          \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                                \
    int m       = static_cast<int>(weights.size(0));                                         \
    int k       = static_cast<int>(weights.size(1));                                         \
    int n       = static_cast<int>(spikes.size(1));                                          \
    int warps   = 8;                                                                         \
    int m_blks  = (m + warps - 1) / warps;                                                  \
    int n_blks  = (n + 31) / 32;                                                             \
    dim3 grid(m_blks, n_blks);                                                               \
    _spfloat_mm_nt_tpe_kern##SUFFIX<<<grid, 256, 0, s>>>(                                    \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                                  \
        static_cast<const WEIGHT_C_T*>(spikes.data_ptr()),                                   \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                                         \
        m, k, n);                                                                            \
}

#define FFI_SPFLOAT_MM_T(SUFFIX, WEIGHT_C_T, CHUNK_N_VAL)                                   \
void spfloat_densemm_t##SUFFIX(                                                              \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView spikes,                               \
    tvm::ffi::TensorView output,  int64_t stream                                             \
) {                                                                                          \
    cudaStream_t s    = reinterpret_cast<cudaStream_t>(stream);                              \
    int k       = static_cast<int>(weights.size(0));                                         \
    int n       = static_cast<int>(weights.size(1));                                         \
    int m       = static_cast<int>(spikes.size(0));                                          \
    int warps   = 8;                                                                         \
    int m_blks  = (m + warps - 1) / warps;                                                  \
    int n_chnks = (n + CHUNK_N_VAL - 1) / CHUNK_N_VAL;                                      \
    dim3 grid(m_blks, n_chnks);                                                              \
    _spfloat_mm_t_kern##SUFFIX<<<grid, 256, 0, s>>>(                                         \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                                  \
        static_cast<const WEIGHT_C_T*>(spikes.data_ptr()),                                   \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                                         \
        m, k, n);                                                                            \
}

// @tvm_ffi spfloat_densemm_nt_f32
FFI_SPFLOAT_MM_NT(_f32, float, 32)
// @tvm_ffi spfloat_densemm_nt_tpe_f32
FFI_SPFLOAT_MM_NT_TPE(_f32, float)
// @tvm_ffi spfloat_densemm_t_f32
FFI_SPFLOAT_MM_T(_f32, float, 32)
// @tvm_ffi spfloat_densemm_nt_f64
FFI_SPFLOAT_MM_NT(_f64, double, 16)
// @tvm_ffi spfloat_densemm_nt_tpe_f64
FFI_SPFLOAT_MM_NT_TPE(_f64, double)
// @tvm_ffi spfloat_densemm_t_f64
FFI_SPFLOAT_MM_T(_f64, double, 16)
// @tvm_ffi spfloat_densemm_nt_f16
FFI_SPFLOAT_MM_NT(_f16, __half, 32)
// @tvm_ffi spfloat_densemm_nt_tpe_f16
FFI_SPFLOAT_MM_NT_TPE(_f16, __half)
// @tvm_ffi spfloat_densemm_t_f16
FFI_SPFLOAT_MM_T(_f16, __half, 32)


// ============================================================================
// FCN Matrix-Matrix Multiplication with sparse-float inputs (spfloat_fcnmm)
// ============================================================================
// These kernels handle Y = W_fcn @ M (gather) or Y = W_fcn^T @ M (scatter)
// where W_fcn is a fixed-connection-number sparse matrix with indices, and
// M is a sparse-float dense matrix.  Zero entries in M are skipped.
//
// ── spfloat_fcnmm gather kernel ──────────────────────────────────────────
//
// Roofline analysis (gather, hetero, f32, 5Kx5Kx500, ncol=64):
//   Per output element (row i, col j):
//     - n_conn matrix reads: n_conn * 4B (random row, coalesced across j)
//     - n_conn index reads from smem: ~0B global (tiled, shared by all j threads)
//     - n_conn weight reads from smem: ~0B global (tiled, shared by all j threads)
//     - 1 output write: 4B
//   Amortized smem fills per tile: TILE_K * (4B idx + 4B wt) / blockDim.x per thread
//   FLOPs per element: 2 * n_conn (1 FMA per connection)
//   Arithmetic intensity: ~0.5 FLOP/byte → bandwidth-bound
//
// Optimizations applied:
//   1. Shared-memory tiling of indices and weights → eliminates redundant
//      global loads across threads in the same block (reduction = blockDim.x)
//   2. __ldg() for matrix reads → L1 texture cache
//   3. Adaptive block size → matches thread count to output columns
//   4. Sparsity check on matrix values → skips zero-entry FMAs
//   5. Homo path: skip weight smem, accumulate raw values, multiply once at end
//
// Achieved throughput (RTX 3080 Ti Laptop, f32, tvmffi vs jax_raw):
//   Hetero gather (dominant case):
//     5Kx5Kx500,ncol=64:   1.16-1.30ms → 2.8-3.4x vs jax_raw (3.5-4.2ms)
//     5Kx5Kx200,ncol=128:  1.17-1.30ms → 1.9-2.0x vs jax_raw (2.4-2.5ms)
//     5Kx5Kx50,ncol=512:   1.17-1.23ms → 2.8-3.1x vs jax_raw (3.3-3.8ms)
//     10Kx10Kx200,ncol=64:  1.17-1.21ms → 2.9-3.1x vs jax_raw (3.4-3.7ms)
//     10Kx10Kx100,ncol=128: 1.18-1.30ms → 2.7-3.0x vs jax_raw (3.4-3.5ms)
//   Homo gather: matches jax_raw (no weight data to tile → minimal benefit)
//
// Fundamental barriers:
//   1. Random matrix row access: matrix[col_idx * n_col + j] with col_idx
//      varying per connection. 4B read may pull 128B L2 cacheline (~32x amp
//      when matrix exceeds L2). Index/weight loads are only ~3% of traffic;
//      ~97% is random matrix access which cannot be optimized without a
//      format change (e.g., reordering matrix rows by index locality).
//   2. Per-connection serial dependency in the inner loop (accumulator chain).
//   3. Homo path already near-optimal: no weight reads, only matrix + index
//      loads; __ldg broadcasts index reads within warps via L1 texture cache.
//
// Future directions:
//   - Index-aware matrix reordering (sort indices per row for spatial locality)
//   - Software-pipelined double-buffering of smem tiles
//   - Warp specialization: separate warps for smem fill vs compute
//
// ── spfloat_fcnmm scatter kernel ─────────────────────────────────────────
//
// Optimizations applied:
//   1. Shared-memory caching of matrix row → eliminates n_conn redundant
//      global reads of the same row (reduction factor = n_conn)
//   2. Row-level early exit via __syncthreads_count → skips all-zero rows
//      (At 1% spike rate with ncol=64, ~53% of rows are all-zero.)
//   3. Adaptive block size → matches thread count to output columns
//
// Achieved throughput (RTX 3080 Ti Laptop, f32, tvmffi vs jax_raw):
//   Low spike rate (1-10%):
//     5Kx5Kx500,ncol=64,1%:   1.22ms → 2.1-2.4x vs jax_raw (2.5-2.9ms)
//     5Kx5Kx200,ncol=128,1%:  1.23-1.28ms → 1.8-1.9x vs jax_raw (2.3-2.4ms)
//     5Kx5Kx50,ncol=512,1%:   1.23-1.27ms → 2.5-3.0x vs jax_raw (3.1-3.6ms)
//     10Kx10Kx200,ncol=64,1%: 1.21-1.23ms → 2.0-2.3x vs jax_raw (2.4-2.8ms)
//     10Kx10Kx50,ncol=256,1%: 1.25-1.27ms → 2.6-2.8x vs jax_raw (3.3-3.5ms)
//   High spike rate (50%): ~1.2-1.4x (early exit rarely triggers)
//
// Fundamental barriers:
//   1. Atomic contention: multiple pre-synaptic neurons may target the same
//      post-synaptic neuron, causing atomicAdd serialization.
//   2. Per-connection serial outer loop: each connection issues separate atomics.
//   3. At high spike rates (≥50%), all-zero row skip provides no benefit, and
//      atomic contention dominates latency.
//
// Future directions:
//   - Segment-sorted indices + shared-memory accumulation before atomics
//   - Connection tiling with warp-cooperative atomics
//   - Per-target histogram to reorder connections by output row

#define FCN_MM_GATHER_TILE_K 128

// ---------------------------------------------------------------------------
// Gather tiled: shared-memory tiling of indices (and weights for hetero).
// All threads cooperatively load a tile of indices into shared memory,
// then each thread gathers its column j from the matrix.
// Homo path: accumulates raw matrix values, multiplies by homo_w at the end.
// Hetero path: also tiles weights into smem, accumulates weighted values.
// ---------------------------------------------------------------------------
#define DEFINE_SPFLOAT_FCN_MM_GATHER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _spfloat_fcnmm_gather_kern##SUFFIX(                                       \
    const int32_t* __restrict__ indices,                                                   \
    const WEIGHT_T* __restrict__ matrix,                                                   \
    WEIGHT_T*       __restrict__ output,                                                   \
    const WEIGHT_T* __restrict__ weights,                                                  \
    int n_pre, int n_conn, int n_col, int is_homo                                          \
) {                                                                                        \
    extern __shared__ char _smem_raw[];                                                    \
    int32_t* s_idx = reinterpret_cast<int32_t*>(_smem_raw);                               \
    ACC_T*   s_wt  = reinterpret_cast<ACC_T*>(                                            \
        _smem_raw + FCN_MM_GATHER_TILE_K * sizeof(int32_t));                               \
                                                                                           \
    int row = blockIdx.x;                                                                  \
    int j   = blockIdx.y * blockDim.x + threadIdx.x;                                      \
    if (row >= n_pre) return;                                                              \
                                                                                           \
    const int32_t*  idx_row = indices + (size_t)row * n_conn;                             \
    const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;            \
    ACC_T homo_w = is_homo ? READ_W(__ldg(&weights[0])) : ACC_ZERO;                       \
    ACC_T acc = ACC_ZERO;                                                                  \
                                                                                           \
    for (int base = 0; base < n_conn; base += FCN_MM_GATHER_TILE_K) {                     \
        int tile_size = n_conn - base;                                                     \
        if (tile_size > FCN_MM_GATHER_TILE_K) tile_size = FCN_MM_GATHER_TILE_K;           \
        /* Cooperative load: indices always, weights only for hetero */                    \
        for (int t = threadIdx.x; t < tile_size; t += blockDim.x) {                      \
            s_idx[t] = __ldg(&idx_row[base + t]);                                         \
            if (!is_homo)                                                                  \
                s_wt[t] = READ_W(__ldg(&w_row[base + t]));                                \
        }                                                                                  \
        __syncthreads();                                                                   \
                                                                                           \
        if (j < n_col) {                                                                   \
            if (is_homo) {                                                                 \
                /* Homo: accumulate raw matrix values, no weight multiply */               \
                for (int k = 0; k < tile_size; k++) {                                     \
                    ACC_T m_val = READ_W(__ldg(&matrix[(size_t)s_idx[k] * n_col + j]));   \
                    if (m_val != ACC_ZERO)                                                 \
                        acc += m_val;                                                      \
                }                                                                          \
            } else {                                                                       \
                /* Hetero: accumulate weighted matrix values from smem */                  \
                for (int k = 0; k < tile_size; k++) {                                     \
                    ACC_T m_val = READ_W(__ldg(&matrix[(size_t)s_idx[k] * n_col + j]));   \
                    if (m_val != ACC_ZERO)                                                 \
                        acc += s_wt[k] * m_val;                                           \
                }                                                                          \
            }                                                                              \
        }                                                                                  \
        __syncthreads();                                                                   \
    }                                                                                      \
                                                                                           \
    if (j < n_col)                                                                         \
        output[(size_t)row * n_col + j] = WRITE_W(is_homo ? homo_w * acc : acc);          \
}

// ---------------------------------------------------------------------------
// Scatter with smem caching: loads the matrix row once into shared memory,
// then reuses it for all n_conn connections.  Includes row-level early exit
// via __syncthreads_count to skip entirely-zero rows.
// ---------------------------------------------------------------------------
#define DEFINE_SPFLOAT_FCN_MM_SCATTER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, \
                                       ATOMIC_ADD_W, ACC_ZERO)                    \
__global__ void _spfloat_fcnmm_scatter_kern##SUFFIX(                              \
    const int32_t* __restrict__ indices,                                           \
    const WEIGHT_T* __restrict__ matrix,                                           \
    WEIGHT_T*       __restrict__ output,                                           \
    const WEIGHT_T* __restrict__ weights,                                          \
    int n_pre, int n_conn, int n_col, int is_homo                                  \
) {                                                                                \
    extern __shared__ char _smem_raw[];                                            \
    WEIGHT_T* s_mrow = reinterpret_cast<WEIGHT_T*>(_smem_raw);                    \
                                                                                   \
    int i = blockIdx.x;                                                            \
    if (i >= n_pre) return;                                                        \
                                                                                   \
    /* Load matrix row into shared memory (one global read per element) */         \
    const WEIGHT_T* m_row = matrix + (size_t)i * n_col;                            \
    for (int j = threadIdx.x; j < n_col; j += blockDim.x)                         \
        s_mrow[j] = __ldg(&m_row[j]);                                              \
    __syncthreads();                                                               \
                                                                                   \
    /* Row-level early exit: skip if entire row is zero */                         \
    int has_nz = 0;                                                                \
    for (int j = threadIdx.x; j < n_col; j += blockDim.x) {                      \
        if (READ_W(s_mrow[j]) != ACC_ZERO) { has_nz = 1; break; }                 \
    }                                                                              \
    if (__syncthreads_count(has_nz) == 0) return;                                 \
                                                                                   \
    const int32_t*  idx_row = indices + (size_t)i * n_conn;                       \
    const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)i * n_conn;      \
    ACC_T homo_w = is_homo ? READ_W(__ldg(&weights[0])) : ACC_ZERO;               \
                                                                                   \
    for (int k = 0; k < n_conn; k++) {                                             \
        int tgt = __ldg(&idx_row[k]);                                              \
        ACC_T w_val = is_homo ? homo_w : READ_W(__ldg(&w_row[k]));                 \
        WEIGHT_T* out_row = output + (size_t)tgt * n_col;                          \
        for (int j = threadIdx.x; j < n_col; j += blockDim.x) {                  \
            ACC_T m_val = READ_W(s_mrow[j]);                                       \
            if (m_val != ACC_ZERO)                                                 \
                ATOMIC_ADD_W(&out_row[j], w_val * m_val);                          \
        }                                                                          \
    }                                                                              \
}

// FCN SpMM Instantiations
DEFINE_SPFLOAT_FCN_MM_GATHER(_f32, float,  float,  READ_F32, WRITE_F32, 0.0f)
DEFINE_SPFLOAT_FCN_MM_GATHER(_f64, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_SPFLOAT_FCN_MM_GATHER(_f16, __half, float,  READ_F16, WRITE_F16, 0.0f)
DEFINE_SPFLOAT_FCN_MM_SCATTER(_f32, float,  float,  READ_F32, WRITE_F32, atomic_add_f32, 0.0f)
DEFINE_SPFLOAT_FCN_MM_SCATTER(_f64, double, double, READ_F64, WRITE_F64, atomic_add_f64, 0.0)
DEFINE_SPFLOAT_FCN_MM_SCATTER(_f16, __half, float,  READ_F16, WRITE_F16, atomic_add_f16, 0.0f)

// FCN SpMM FFI Entry Macros
#define FFI_SPFLOAT_FCN_MM_GATHER(SUFFIX, WEIGHT_C_T)                                       \
void spfloat_fcnmm_gather_auto##SUFFIX(                                                      \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                              \
    tvm::ffi::TensorView matrix,  tvm::ffi::TensorView output, int64_t stream                \
) {                                                                                          \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                                \
    int n_pre       = static_cast<int>(indices.size(0));                                     \
    int n_conn      = static_cast<int>(indices.size(1));                                     \
    int n_col       = static_cast<int>(matrix.size(1));                                      \
    int is_homo     = (weights.ndim() == 1) ? 1 : 0;                                        \
    /* Adaptive block size: round n_col up to warp multiple, clamp [32,256] */               \
    int block_x = ((n_col + 31) >> 5) << 5;                                                 \
    block_x = block_x < 32 ? 32 : block_x > 256 ? 256 : block_x;                           \
    int y_blocks = (n_col + block_x - 1) / block_x;                                         \
    /* ACC_T size: float (4B) for f16/bf16, else sizeof(WEIGHT_C_T) */                       \
    int acc_sz = (sizeof(WEIGHT_C_T) < 4) ? 4 : static_cast<int>(sizeof(WEIGHT_C_T));       \
    size_t smem = FCN_MM_GATHER_TILE_K * (sizeof(int32_t) + acc_sz);                         \
    _spfloat_fcnmm_gather_kern##SUFFIX<<<dim3(n_pre, y_blocks), block_x, smem, s>>>(         \
        static_cast<const int32_t*>(indices.data_ptr()),                                     \
        static_cast<const WEIGHT_C_T*>(matrix.data_ptr()),                                   \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                                         \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                                  \
        n_pre, n_conn, n_col, is_homo);                                                      \
}

#define FFI_SPFLOAT_FCN_MM_SCATTER(SUFFIX, WEIGHT_C_T)                                      \
void spfloat_fcnmm_scatter_auto##SUFFIX(                                                     \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                              \
    tvm::ffi::TensorView matrix,  tvm::ffi::TensorView output, int64_t stream                \
) {                                                                                          \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                                \
    int n_pre       = static_cast<int>(indices.size(0));                                     \
    int n_conn      = static_cast<int>(indices.size(1));                                     \
    int n_post      = static_cast<int>(output.size(0));                                      \
    int n_col       = static_cast<int>(matrix.size(1));                                      \
    int is_homo     = (weights.ndim() == 1) ? 1 : 0;                                        \
    cudaMemsetAsync(output.data_ptr(), 0,                                                    \
                    (size_t)n_post * n_col * sizeof(WEIGHT_C_T), s);                         \
    /* Adaptive block size: round n_col up to warp multiple, clamp [32,256] */               \
    int block_x = ((n_col + 31) >> 5) << 5;                                                 \
    block_x = block_x < 32 ? 32 : block_x > 256 ? 256 : block_x;                           \
    size_t smem = static_cast<size_t>(n_col) * sizeof(WEIGHT_C_T);                           \
    _spfloat_fcnmm_scatter_kern##SUFFIX<<<n_pre, block_x, smem, s>>>(                        \
        static_cast<const int32_t*>(indices.data_ptr()),                                     \
        static_cast<const WEIGHT_C_T*>(matrix.data_ptr()),                                   \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                                         \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                                  \
        n_pre, n_conn, n_col, is_homo);                                                      \
}

// FCN SpMM FFI Instantiations
// @tvm_ffi spfloat_fcnmm_gather_auto_f32
FFI_SPFLOAT_FCN_MM_GATHER(_f32, float)
// @tvm_ffi spfloat_fcnmm_scatter_auto_f32
FFI_SPFLOAT_FCN_MM_SCATTER(_f32, float)
// @tvm_ffi spfloat_fcnmm_gather_auto_f64
FFI_SPFLOAT_FCN_MM_GATHER(_f64, double)
// @tvm_ffi spfloat_fcnmm_scatter_auto_f64
FFI_SPFLOAT_FCN_MM_SCATTER(_f64, double)
// @tvm_ffi spfloat_fcnmm_gather_auto_f16
FFI_SPFLOAT_FCN_MM_GATHER(_f16, __half)
// @tvm_ffi spfloat_fcnmm_scatter_auto_f16
FFI_SPFLOAT_FCN_MM_SCATTER(_f16, __half)
