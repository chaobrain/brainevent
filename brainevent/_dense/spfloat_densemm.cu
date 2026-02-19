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
 * spfloat_densemm.cu -- Sparse-Float Dense Matrix-Matrix CUDA Kernels (v7)
 * =========================================================================
 *
 * Python API: brainevent.spfloat_densemm(weights, spikes, *, transpose, backend)
 *
 * transpose=False (NT): weights[m,k] @ spikes[k,n] -> out[m,n]
 * transpose=True  (T):  spikes[m,k] @ weights[k,n] -> out[m,n]
 *
 * NT Kernel: Multi-Row Warp-Per-Output-Row (branchless)
 * ------------------------------------------------------
 * Grid:  (ceil(m/8), ceil(n/CHUNK_N))   Block: 256 = 8 warps
 *
 * Each warp of 32 threads independently computes one output row and up to
 * CHUNK_N columns.  All 8 warps in a block share the same spike data,
 * giving ~8x L2 traffic reduction via L1 cache / MSHR merging.
 *
 * No shared memory.  No __syncthreads().  Warp-shuffle reduction only.
 * Branchless (no per-element spike check): removes warp-divergence overhead.
 *
 * T Kernel: Multi-Row Warp-Per-Spike-Row (event-driven, no syncthreads)
 * -----------------------------------------------------------------------
 * Grid:  (ceil(m/8), ceil(n/CHUNK_N))   Block: 256 = 8 warps
 *
 * Each warp of 32 threads independently scans one row of the spike matrix.
 * Event-driven: spike check per k-step skips weight reads for zero spikes.
 * No shared memory.  No __syncthreads().  Warp-shuffle reduction only.
 *
 * Advantage over single-row block-per-row (v6 T):
 *   - ~8x fewer blocks  (e.g. m=50,n=10K: 2191 vs 15650 blocks)
 *   - Eliminates __syncthreads() barrier and shared-memory round-trip
 *   - Effective at density <= 1% where event-driven skip saves weight reads
 *
 * CHUNK_N = 32 for f32/f16/bf16,  16 for f64.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

#define MM_BLOCK_SIZE 256

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
// Per-dtype conversion macros
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
// NT Multi-Row Kernel (branchless): weights[m,k] @ spikes[k,n] -> out[m,n]
//
// Grid: (ceil(m/warps_per_block), ceil(n/CHUNK_N))   Block: (256,)
//
// Each warp (32 threads) handles one output row.  8 warps per block share
// the same spike data through L1 cache, reducing L2 traffic ~8x compared
// to one-block-per-row.  No shared memory.  Warp reduction via shuffles.
// =========================================================================

#define DEFINE_SPFLOAT_MM_NT(SUFFIX, WEIGHT_T, ACC_T, CHUNK_N,             \
                              READ_W, WRITE_W, READ_S,                      \
                              WARP_RED, ACC_ZERO)                           \
__global__ void _spfloat_mm_nt_kern##SUFFIX(                                \
    const WEIGHT_T* __restrict__ weights,                                   \
    const WEIGHT_T* __restrict__ spikes,                                    \
    WEIGHT_T*       __restrict__ output,                                    \
    int m, int k, int n                                                     \
) {                                                                         \
    int warp_id = threadIdx.x >> 5;                                        \
    int lane    = threadIdx.x & 31;                                        \
    int warps_per_block = blockDim.x >> 5;                                 \
    int row = blockIdx.x * warps_per_block + warp_id;                      \
    if (row >= m) return;                                                   \
    int col_start = blockIdx.y * CHUNK_N;                                   \
    int chunk_n = min(CHUNK_N, n - col_start);                              \
    const WEIGHT_T* w_row = weights + (size_t)row * k;                     \
    /* Per-thread accumulators */                                           \
    ACC_T acc[CHUNK_N];                                                     \
    for (int j = 0; j < chunk_n; j++) acc[j] = ACC_ZERO;                  \
    /* Main k-loop: branchless, warp-stride */                             \
    for (int l = lane; l < k; l += 32) {                                   \
        ACC_T w_val = READ_W(w_row[l]);                                    \
        const WEIGHT_T* spk_l = spikes + (size_t)l * n + col_start;       \
        for (int j = 0; j < chunk_n; j++)                                  \
            acc[j] += w_val * READ_S(spk_l[j]);                           \
    }                                                                       \
    /* Warp reduction — no shared memory needed */                         \
    WEIGHT_T* out_row = output + (size_t)row * n + col_start;             \
    for (int j = 0; j < chunk_n; j++) {                                    \
        ACC_T val = WARP_RED(acc[j]);                                      \
        if (lane == 0) out_row[j] = WRITE_W(val);                         \
    }                                                                       \
}


// =========================================================================
// T Multi-Row Kernel (event-driven, no syncthreads):
//   spikes[m,k] @ weights[k,n] -> out[m,n]
//
// Grid: (ceil(m/warps_per_block), ceil(n/CHUNK_N))   Block: (256,)
//
// Each warp (32 threads) independently scans one row of spikes.
// Event-driven: skip weight reads for zero spikes.
// No shared memory.  No __syncthreads().  Warp reduction via shuffles.
//
// ~8x fewer blocks than the v6 single-row design, eliminating __syncthreads
// and shared-memory round-trips.  Effective at density <= 1%.
// =========================================================================

#define DEFINE_SPFLOAT_MM_T(SUFFIX, WEIGHT_T, ACC_T, CHUNK_N,              \
                             READ_W, WRITE_W, READ_S,                       \
                             WARP_RED, ACC_ZERO)                            \
__global__ void _spfloat_mm_t_kern##SUFFIX(                                 \
    const WEIGHT_T* __restrict__ weights,                                   \
    const WEIGHT_T* __restrict__ spikes,                                    \
    WEIGHT_T*       __restrict__ output,                                    \
    int m, int k, int n                                                     \
) {                                                                         \
    int warp_id = threadIdx.x >> 5;                                        \
    int lane    = threadIdx.x & 31;                                        \
    int warps_per_block = blockDim.x >> 5;                                 \
    int row = blockIdx.x * warps_per_block + warp_id;                      \
    if (row >= m) return;                                                   \
    int col_start = blockIdx.y * CHUNK_N;                                   \
    int chunk_n = min(CHUNK_N, n - col_start);                              \
    const WEIGHT_T* s_row = spikes + (size_t)row * k;                      \
    /* Per-thread accumulators */                                           \
    ACC_T acc[CHUNK_N];                                                     \
    for (int j = 0; j < chunk_n; j++) acc[j] = ACC_ZERO;                  \
    /* Main k-loop: event-driven (one spike check per k-step) */           \
    for (int l = lane; l < k; l += 32) {                                   \
        ACC_T spk_val = READ_S(s_row[l]);                                  \
        if (spk_val != ACC_ZERO) {                                         \
            const WEIGHT_T* w_l = weights + (size_t)l * n + col_start;    \
            for (int j = 0; j < chunk_n; j++)                              \
                acc[j] += spk_val * READ_W(w_l[j]);                       \
        }                                                                   \
    }                                                                       \
    /* Warp reduction — no shared memory needed */                         \
    WEIGHT_T* out_row = output + (size_t)row * n + col_start;             \
    for (int j = 0; j < chunk_n; j++) {                                    \
        ACC_T val = WARP_RED(acc[j]);                                      \
        if (lane == 0) out_row[j] = WRITE_W(val);                         \
    }                                                                       \
}


// =========================================================================
// Instantiate all kernel variants
// =========================================================================

// ---- Float32: CHUNK_N = 32 ----
DEFINE_SPFLOAT_MM_NT(_f32, float, float, 32,
    READ_F32, WRITE_F32, READ_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_MM_T(_f32, float, float, 32,
    READ_F32, WRITE_F32, READ_F32, warp_reduce_sum_f32, 0.0f)

// ---- Float64: CHUNK_N = 16 ----
DEFINE_SPFLOAT_MM_NT(_f64, double, double, 16,
    READ_F64, WRITE_F64, READ_F64, warp_reduce_sum_f64, 0.0)
DEFINE_SPFLOAT_MM_T(_f64, double, double, 16,
    READ_F64, WRITE_F64, READ_F64, warp_reduce_sum_f64, 0.0)

// ---- Float16: CHUNK_N = 32, accumulates in f32 ----
DEFINE_SPFLOAT_MM_NT(_f16, __half, float, 32,
    READ_F16, WRITE_F16, READ_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_MM_T(_f16, __half, float, 32,
    READ_F16, WRITE_F16, READ_F16, warp_reduce_sum_f32, 0.0f)

// ---- BFloat16: CHUNK_N = 32, accumulates in f32 ----
DEFINE_SPFLOAT_MM_NT(_bf16, __nv_bfloat16, float, 32,
    READ_BF16, WRITE_BF16, READ_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_MM_T(_bf16, __nv_bfloat16, float, 32,
    READ_BF16, WRITE_BF16, READ_BF16, warp_reduce_sum_f32, 0.0f)


// =========================================================================
// TVM FFI Entry Points
// =========================================================================

// NT: multi-row (one warp per row, 8 rows per block)
#define FFI_SPFLOAT_MM_NT(SUFFIX, WEIGHT_C_T, ACC_C_T, CHUNK_N_VAL)        \
void spfloat_densemm_nt##SUFFIX(                                            \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView spikes,             \
    tvm::ffi::TensorView output, int64_t stream                             \
) {                                                                         \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                \
    int m = static_cast<int>(weights.size(0));                              \
    int k = static_cast<int>(weights.size(1));                              \
    int n = static_cast<int>(spikes.size(1));                               \
    int warps_per_block = MM_BLOCK_SIZE / 32;                               \
    int m_blocks  = (m + warps_per_block - 1) / warps_per_block;            \
    int n_chunks  = (n + CHUNK_N_VAL - 1) / CHUNK_N_VAL;                   \
    dim3 grid(m_blocks, n_chunks);                                          \
    const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const WEIGHT_C_T* d_s = static_cast<const WEIGHT_C_T*>(spikes.data_ptr()); \
    WEIGHT_C_T*       d_o = static_cast<WEIGHT_C_T*>(output.data_ptr());    \
    _spfloat_mm_nt_kern##SUFFIX<<<grid, MM_BLOCK_SIZE, 0, s>>>(            \
        d_w, d_s, d_o, m, k, n);                                           \
}

// T: multi-row (one warp per spike row, 8 rows per block, no syncthreads)
#define FFI_SPFLOAT_MM_T(SUFFIX, WEIGHT_C_T, ACC_C_T, CHUNK_N_VAL)         \
void spfloat_densemm_t##SUFFIX(                                             \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView spikes,             \
    tvm::ffi::TensorView output, int64_t stream                             \
) {                                                                         \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                \
    int k = static_cast<int>(weights.size(0));                              \
    int n = static_cast<int>(weights.size(1));                              \
    int m = static_cast<int>(spikes.size(0));                               \
    int warps_per_block = MM_BLOCK_SIZE / 32;                               \
    int m_blocks  = (m + warps_per_block - 1) / warps_per_block;            \
    int n_chunks  = (n + CHUNK_N_VAL - 1) / CHUNK_N_VAL;                   \
    dim3 grid(m_blocks, n_chunks);                                          \
    const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const WEIGHT_C_T* d_s = static_cast<const WEIGHT_C_T*>(spikes.data_ptr()); \
    WEIGHT_C_T*       d_o = static_cast<WEIGHT_C_T*>(output.data_ptr());    \
    _spfloat_mm_t_kern##SUFFIX<<<grid, MM_BLOCK_SIZE, 0, s>>>(             \
        d_w, d_s, d_o, m, k, n);                                           \
}

// =========================================================================
// Instantiate FFI entry points
// =========================================================================

// @tvm_ffi spfloat_densemm_nt_f32
FFI_SPFLOAT_MM_NT(_f32, float, float, 32)
// @tvm_ffi spfloat_densemm_t_f32
FFI_SPFLOAT_MM_T(_f32, float, float, 32)

// @tvm_ffi spfloat_densemm_nt_f64
FFI_SPFLOAT_MM_NT(_f64, double, double, 16)
// @tvm_ffi spfloat_densemm_t_f64
FFI_SPFLOAT_MM_T(_f64, double, double, 16)

// @tvm_ffi spfloat_densemm_nt_f16
FFI_SPFLOAT_MM_NT(_f16, __half, float, 32)
// @tvm_ffi spfloat_densemm_t_f16
FFI_SPFLOAT_MM_T(_f16, __half, float, 32)

// @tvm_ffi spfloat_densemm_nt_bf16
FFI_SPFLOAT_MM_NT(_bf16, __nv_bfloat16, float, 32)
// @tvm_ffi spfloat_densemm_t_bf16
FFI_SPFLOAT_MM_T(_bf16, __nv_bfloat16, float, 32)
