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
 * spfloat_densemm.cu -- Sparse-Float Dense Matrix-Matrix CUDA Kernels (v10)
 * =========================================================================
 *
 * Python API: brainevent.spfloat_densemm(weights, spikes, *, transpose, backend)
 *
 * transpose=False (NT): weights[m,k] @ spikes[k,n] -> out[m,n]
 * transpose=True  (T):  spikes[m,k] @ weights[k,n] -> out[m,n]
 *
 * -------------------------------------------------------------------------
 * NT Kernel — two variants, selected in Python based on n:
 * -------------------------------------------------------------------------
 *
 * NT_WPR (Warp-Per-Row) — for small n (n <= CHUNK_N = 32):
 *   Grid:  (ceil(m/8), ceil(n/CHUNK_N))   Block: 256 = 8 warps
 *   Each warp handles one output row.  For each k-position l (warp-strided,
 *   different lanes handle different l values), reads w[row,l] (coalesced)
 *   and s[l, col_start..end] (non-coalesced but stride = n ≤ 32 × 4 bytes
 *   ≤ 1 cache-line, so penalty is bounded).
 *   Warp-reduce at end; no shared memory needed.
 *   Weight matrix read exactly once per block.y → optimal for n ≤ CHUNK_N.
 *   _Pragma("unroll") forces acc[CHUNK_N] into per-thread registers.
 *
 * NT_TPE (Thread-Per-Element) — for large n (n > CHUNK_N = 32):
 *   Grid:  (ceil(m/8), ceil(n/32))        Block: 256 = 8 warps
 *   Thread (warp_id, lane) → row = bx*8+warp_id, col = by*32+lane.
 *   Each thread independently accumulates ONE output element.
 *
 *   Spike reads: spikes[l*n + col] — fully coalesced (consecutive lanes
 *     read consecutive columns → 1 cache-line per warp instruction for
 *     any n, unlike WPR which has stride-n non-coalescing).
 *   Weight reads: w_row[l] — uniform broadcast (all 32 threads in warp
 *     have the same row and same l → 1 hardware transaction).
 *   No accumulator array, no warp reduction; direct store.
 *
 *   4-unrolled k-loop: issues 4 spike loads before the ballot, giving the
 *   GPU 4 outstanding L2 requests (MLP pipelining) and reducing warp-sync
 *   barriers 4×.  One combined ballot covers all 4 positions.
 *
 *   At d=1%, combined skip rate (all 4×32 entries zero) = (0.99)^128 ≈ 27.7%.
 *   Non-skip rate 72.3% → 4 weight broadcasts + 4 FMAs per non-skipped
 *   4-tuple.  Weight reads are cheap (broadcast scalar from L1/L2),
 *   FP32 FMAs are ~0.25 cycles per warp on A100.
 *
 * -------------------------------------------------------------------------
 * T Kernel v8 — Warp-Ballot + Register Accumulators:
 *   spikes[m,k] @ weights[k,n] -> out[m,n]
 *   Grid:  (ceil(m/8), ceil(n/CHUNK_N))   Block: 256 = 8 warps
 *   Each warp independently scans one spike row (warp-strided k-loop).
 *   __ballot_sync(__activemask()) early exit: if no thread in warp has
 *   nonzero spike at position l, the entire warp skips the weight read.
 *   At density 1%, ~73% of warp steps are all-zero → 73% weight savings.
 *   _Pragma("unroll") keeps acc[CHUNK_N] in per-thread registers.
 *   CHUNK_N=32 for f32/f16/bf16, 16 for f64.
 *
 * -------------------------------------------------------------------------
 * __activemask() used in all ballot calls: safe when k%32≠0 causes
 * partial-warp divergence at the last loop iteration.
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
// NT_WPR Kernel (Warp-Per-Row) — for small n (n <= CHUNK_N = 32):
//   weights[m,k] @ spikes[k,n] -> out[m,n]
//
// Grid: (ceil(m/warps_per_block), ceil(n/CHUNK_N))   Block: (256,)
//
// Each warp handles one output row. Warp-stride over k (each lane handles
// a different k-position). Weight reads are coalesced across warp lanes.
// Spike reads stride by n between lanes: for n ≤ 32, stride ≤ 128 bytes,
// limiting the CL penalty to ≤1 CL per lane.
//
// Weight matrix read exactly once per block.y → optimal when n ≤ CHUNK_N.
// _Pragma("unroll") forces acc[CHUNK_N] into per-thread registers.
// =========================================================================

#define DEFINE_SPFLOAT_MM_NT_WPR(SUFFIX, WEIGHT_T, ACC_T, CHUNK_N,          \
                                  READ_W, WRITE_W, READ_S,                   \
                                  WARP_RED, ACC_ZERO)                        \
__global__ void _spfloat_mm_nt_wpr_kern##SUFFIX(                             \
    const WEIGHT_T* __restrict__ weights,                                    \
    const WEIGHT_T* __restrict__ spikes,                                     \
    WEIGHT_T*       __restrict__ output,                                     \
    int m, int k, int n                                                      \
) {                                                                          \
    int warp_id = threadIdx.x >> 5;                                         \
    int lane    = threadIdx.x & 31;                                         \
    int warps_per_block = blockDim.x >> 5;                                  \
    int row = blockIdx.x * warps_per_block + warp_id;                       \
    if (row >= m) return;                                                    \
    int col_start = blockIdx.y * CHUNK_N;                                   \
    int chunk_n = min(CHUNK_N, n - col_start);                              \
    const WEIGHT_T* w_row = weights + (size_t)row * k;                      \
    /* Register accumulators — _Pragma("unroll") forces acc[] into regs */   \
    ACC_T acc[CHUNK_N];                                                      \
    _Pragma("unroll")                                                        \
    for (int j = 0; j < CHUNK_N; j++) acc[j] = ACC_ZERO;                   \
    /* Main k-loop: branchless, warp-stride */                              \
    for (int l = lane; l < k; l += 32) {                                    \
        ACC_T w_val = READ_W(w_row[l]);                                     \
        const WEIGHT_T* spk_l = spikes + (size_t)l * n + col_start;        \
        _Pragma("unroll")                                                    \
        for (int j = 0; j < CHUNK_N; j++)                                  \
            if (j < chunk_n)                                                \
                acc[j] += w_val * READ_S(spk_l[j]);                        \
    }                                                                        \
    /* Warp reduction — no shared memory needed */                          \
    WEIGHT_T* out_row = output + (size_t)row * n + col_start;              \
    _Pragma("unroll")                                                        \
    for (int j = 0; j < CHUNK_N; j++) {                                    \
        ACC_T val = WARP_RED(acc[j]);                                       \
        if (lane == 0 && j < chunk_n) out_row[j] = WRITE_W(val);           \
    }                                                                        \
}


// =========================================================================
// NT_TPE Kernel (Thread-Per-Element, 4-unrolled) — for large n (n > 32):
//   weights[m,k] @ spikes[k,n] -> out[m,n]
//
// Grid: (ceil(m/warps_per_block), ceil(n/32))   Block: (256,)
//
// Thread (warp_id, lane) -> row = bx*8+warp_id, col = by*32+lane.
// Each thread computes ONE output element independently.
//
// Spike reads: spikes[l*n + col] — always coalesced (consecutive lanes →
//   consecutive columns → 1 cache-line per warp instruction regardless of n).
// Weight reads: w_row[l] — uniform broadcast (all 32 threads same row+l).
//
// 4-unrolled k-loop: issues 4 coalesced spike loads before the ballot,
//   enabling the GPU to pipeline 4 concurrent L2 requests (MLP).  One
//   combined ballot covers all 4 positions → 4× fewer warp-sync barriers.
//
//   Skip rate at d=1%: (0.99)^(4×32) = 27.7% per 4-tuple skipped.
//   Weight reads are cheap broadcast scalars; FP32 FMA throughput is high
//   (~0.25 clocks/warp on A100), so the 2.63× higher FMA count vs per-
//   position ballot is well worth the 4× ballot reduction.
//
// __activemask() is safe for any k (partial warp at last iteration).
// =========================================================================

#define DEFINE_SPFLOAT_MM_NT_TPE(SUFFIX, WEIGHT_T, ACC_T,                   \
                                  READ_W, WRITE_W, READ_S, ACC_ZERO)         \
__global__ void _spfloat_mm_nt_tpe_kern##SUFFIX(                             \
    const WEIGHT_T* __restrict__ weights,                                    \
    const WEIGHT_T* __restrict__ spikes,                                     \
    WEIGHT_T*       __restrict__ output,                                     \
    int m, int k, int n                                                      \
) {                                                                          \
    int warp_id = threadIdx.x >> 5;                                         \
    int lane    = threadIdx.x & 31;                                         \
    int warps_per_block = blockDim.x >> 5;                                  \
    int row = blockIdx.x * warps_per_block + warp_id;                       \
    int col = blockIdx.y * 32 + lane; /* one thread = one output element */ \
    if (row >= m || col >= n) return;                                        \
    const WEIGHT_T* w_row = weights + (size_t)row * k;                      \
    ACC_T acc = ACC_ZERO;                                                    \
    /* 4-unrolled k-loop: issue 4 spike loads before ballot (MLP pipelining)\
     * → 4 outstanding L2 requests, 4× fewer warp-sync barriers.          */\
    int l = 0;                                                               \
    for (; l <= k - 4; l += 4) {                                            \
        /* Issue 4 coalesced spike loads back-to-back before any ballot */  \
        ACC_T sv0 = READ_S(spikes[(size_t)(l+0) * n + col]);               \
        ACC_T sv1 = READ_S(spikes[(size_t)(l+1) * n + col]);               \
        ACC_T sv2 = READ_S(spikes[(size_t)(l+2) * n + col]);               \
        ACC_T sv3 = READ_S(spikes[(size_t)(l+3) * n + col]);               \
        /* Single ballot for all 4: skip if every active lane is zero */    \
        bool any_nz = (sv0 != ACC_ZERO) | (sv1 != ACC_ZERO) |             \
                      (sv2 != ACC_ZERO) | (sv3 != ACC_ZERO);              \
        if (__ballot_sync(__activemask(), any_nz) == 0u) continue;          \
        /* Weight reads are broadcast scalars (all lanes same row+l) */     \
        acc += READ_W(w_row[l+0]) * sv0;                                    \
        acc += READ_W(w_row[l+1]) * sv1;                                    \
        acc += READ_W(w_row[l+2]) * sv2;                                    \
        acc += READ_W(w_row[l+3]) * sv3;                                    \
    }                                                                        \
    /* Remainder: scalar loop for k % 4 tail positions */                   \
    for (; l < k; l++) {                                                    \
        ACC_T sv = READ_S(spikes[(size_t)l * n + col]);                     \
        if (__ballot_sync(__activemask(), sv != ACC_ZERO) == 0u) continue;  \
        acc += READ_W(w_row[l]) * sv;                                       \
    }                                                                        \
    output[(size_t)row * n + col] = WRITE_W(acc);                           \
}


// =========================================================================
// T Kernel v8 (warp-ballot + unrolled registers):
//   spikes[m,k] @ weights[k,n] -> out[m,n]
//
// Grid: (ceil(m/warps_per_block), ceil(n/CHUNK_N))   Block: (256,)
//
// Each warp independently scans one spike row.  For each k-position l:
//   __ballot_sync: if ALL active threads have zero spike, skip weight read.
//   Otherwise: active threads read W[l, col_start..end] and accumulate.
//
// _Pragma("unroll") forces acc[CHUNK_N] into per-thread registers.
// __activemask() is safe when k%32 != 0 (partial-warp at last iteration).
// =========================================================================

#define DEFINE_SPFLOAT_MM_T(SUFFIX, WEIGHT_T, ACC_T, CHUNK_N,               \
                             READ_W, WRITE_W, READ_S,                        \
                             WARP_RED, ACC_ZERO)                             \
__global__ void _spfloat_mm_t_kern##SUFFIX(                                  \
    const WEIGHT_T* __restrict__ weights,                                    \
    const WEIGHT_T* __restrict__ spikes,                                     \
    WEIGHT_T*       __restrict__ output,                                     \
    int m, int k, int n                                                      \
) {                                                                          \
    int warp_id = threadIdx.x >> 5;                                         \
    int lane    = threadIdx.x & 31;                                         \
    int warps_per_block = blockDim.x >> 5;                                  \
    int row = blockIdx.x * warps_per_block + warp_id;                       \
    if (row >= m) return;                                                    \
    int col_start = blockIdx.y * CHUNK_N;                                   \
    int chunk_n = min(CHUNK_N, n - col_start);                              \
    const WEIGHT_T* s_row = spikes + (size_t)row * k;                       \
    /* Register accumulators — _Pragma("unroll") forces acc[] into regs */   \
    ACC_T acc[CHUNK_N];                                                      \
    _Pragma("unroll")                                                        \
    for (int j = 0; j < CHUNK_N; j++) acc[j] = ACC_ZERO;                   \
    /* Main k-loop: warp-ballot early exit + thread-level event-driven */   \
    for (int l = lane; l < k; l += 32) {                                    \
        ACC_T spk_val = READ_S(s_row[l]);                                   \
        /* Warp ballot: skip entire warp step if all active spikes are zero */\
        if (__ballot_sync(__activemask(), spk_val != ACC_ZERO) == 0u)       \
            continue;                                                        \
        /* Thread-level event-driven skip for the remaining active threads */\
        if (spk_val != ACC_ZERO) {                                          \
            const WEIGHT_T* w_l = weights + (size_t)l * n + col_start;     \
            _Pragma("unroll")                                                \
            for (int j = 0; j < CHUNK_N; j++)                              \
                if (j < chunk_n)                                            \
                    acc[j] += spk_val * READ_W(w_l[j]);                    \
        }                                                                    \
    }                                                                        \
    /* Warp reduction — no shared memory needed */                          \
    WEIGHT_T* out_row = output + (size_t)row * n + col_start;              \
    _Pragma("unroll")                                                        \
    for (int j = 0; j < CHUNK_N; j++) {                                    \
        ACC_T val = WARP_RED(acc[j]);                                       \
        if (lane == 0 && j < chunk_n) out_row[j] = WRITE_W(val);           \
    }                                                                        \
}


// =========================================================================
// Instantiate all kernel variants
// =========================================================================

// ---- Float32: NT_WPR CHUNK_N=32, NT_TPE, T CHUNK_N=32 ----
DEFINE_SPFLOAT_MM_NT_WPR(_f32, float, float, 32,
    READ_F32, WRITE_F32, READ_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_MM_NT_TPE(_f32, float, float,
    READ_F32, WRITE_F32, READ_F32, 0.0f)
DEFINE_SPFLOAT_MM_T(_f32, float, float, 32,
    READ_F32, WRITE_F32, READ_F32, warp_reduce_sum_f32, 0.0f)

// ---- Float64: NT_WPR CHUNK_N=16, NT_TPE, T CHUNK_N=16 ----
DEFINE_SPFLOAT_MM_NT_WPR(_f64, double, double, 16,
    READ_F64, WRITE_F64, READ_F64, warp_reduce_sum_f64, 0.0)
DEFINE_SPFLOAT_MM_NT_TPE(_f64, double, double,
    READ_F64, WRITE_F64, READ_F64, 0.0)
DEFINE_SPFLOAT_MM_T(_f64, double, double, 16,
    READ_F64, WRITE_F64, READ_F64, warp_reduce_sum_f64, 0.0)

// ---- Float16: NT_WPR CHUNK_N=32, NT_TPE, T CHUNK_N=32 ----
DEFINE_SPFLOAT_MM_NT_WPR(_f16, __half, float, 32,
    READ_F16, WRITE_F16, READ_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_MM_NT_TPE(_f16, __half, float,
    READ_F16, WRITE_F16, READ_F16, 0.0f)
DEFINE_SPFLOAT_MM_T(_f16, __half, float, 32,
    READ_F16, WRITE_F16, READ_F16, warp_reduce_sum_f32, 0.0f)

// ---- BFloat16: NT_WPR CHUNK_N=32, NT_TPE, T CHUNK_N=32 ----
DEFINE_SPFLOAT_MM_NT_WPR(_bf16, __nv_bfloat16, float, 32,
    READ_BF16, WRITE_BF16, READ_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_SPFLOAT_MM_NT_TPE(_bf16, __nv_bfloat16, float,
    READ_BF16, WRITE_BF16, READ_BF16, 0.0f)
DEFINE_SPFLOAT_MM_T(_bf16, __nv_bfloat16, float, 32,
    READ_BF16, WRITE_BF16, READ_BF16, warp_reduce_sum_f32, 0.0f)


// =========================================================================
// TVM FFI Entry Points
// =========================================================================

// NT_WPR: warp-per-row; grid.y = ceil(n/CHUNK_N_VAL); read weight once
#define FFI_SPFLOAT_MM_NT_WPR(SUFFIX, WEIGHT_C_T, CHUNK_N_VAL)              \
void spfloat_densemm_nt##SUFFIX(                                             \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView spikes,              \
    tvm::ffi::TensorView output, int64_t stream                              \
) {                                                                          \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                 \
    int m = static_cast<int>(weights.size(0));                               \
    int k = static_cast<int>(weights.size(1));                               \
    int n = static_cast<int>(spikes.size(1));                                \
    int warps_per_block = MM_BLOCK_SIZE / 32;                                \
    int m_blocks  = (m + warps_per_block - 1) / warps_per_block;             \
    int n_chunks  = (n + CHUNK_N_VAL - 1) / CHUNK_N_VAL;                    \
    dim3 grid(m_blocks, n_chunks);                                           \
    const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const WEIGHT_C_T* d_s = static_cast<const WEIGHT_C_T*>(spikes.data_ptr()); \
    WEIGHT_C_T*       d_o = static_cast<WEIGHT_C_T*>(output.data_ptr());     \
    _spfloat_mm_nt_wpr_kern##SUFFIX<<<grid, MM_BLOCK_SIZE, 0, s>>>(         \
        d_w, d_s, d_o, m, k, n);                                            \
}

// NT_TPE: thread-per-element; grid.y = ceil(n/32); coalesced spike reads
#define FFI_SPFLOAT_MM_NT_TPE(SUFFIX, WEIGHT_C_T)                           \
void spfloat_densemm_nt_tpe##SUFFIX(                                         \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView spikes,              \
    tvm::ffi::TensorView output, int64_t stream                              \
) {                                                                          \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                 \
    int m = static_cast<int>(weights.size(0));                               \
    int k = static_cast<int>(weights.size(1));                               \
    int n = static_cast<int>(spikes.size(1));                                \
    int warps_per_block = MM_BLOCK_SIZE / 32;                                \
    int m_blocks = (m + warps_per_block - 1) / warps_per_block;             \
    int n_blocks = (n + 31) / 32;   /* 32 = warp size = cols per warp */    \
    dim3 grid(m_blocks, n_blocks);                                           \
    const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const WEIGHT_C_T* d_s = static_cast<const WEIGHT_C_T*>(spikes.data_ptr()); \
    WEIGHT_C_T*       d_o = static_cast<WEIGHT_C_T*>(output.data_ptr());     \
    _spfloat_mm_nt_tpe_kern##SUFFIX<<<grid, MM_BLOCK_SIZE, 0, s>>>(         \
        d_w, d_s, d_o, m, k, n);                                            \
}

// T: warp-per-spike-row
#define FFI_SPFLOAT_MM_T(SUFFIX, WEIGHT_C_T, CHUNK_N_VAL)                   \
void spfloat_densemm_t##SUFFIX(                                              \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView spikes,              \
    tvm::ffi::TensorView output, int64_t stream                              \
) {                                                                          \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                 \
    int k = static_cast<int>(weights.size(0));                               \
    int n = static_cast<int>(weights.size(1));                               \
    int m = static_cast<int>(spikes.size(0));                                \
    int warps_per_block = MM_BLOCK_SIZE / 32;                                \
    int m_blocks  = (m + warps_per_block - 1) / warps_per_block;             \
    int n_chunks  = (n + CHUNK_N_VAL - 1) / CHUNK_N_VAL;                    \
    dim3 grid(m_blocks, n_chunks);                                           \
    const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const WEIGHT_C_T* d_s = static_cast<const WEIGHT_C_T*>(spikes.data_ptr()); \
    WEIGHT_C_T*       d_o = static_cast<WEIGHT_C_T*>(output.data_ptr());     \
    _spfloat_mm_t_kern##SUFFIX<<<grid, MM_BLOCK_SIZE, 0, s>>>(              \
        d_w, d_s, d_o, m, k, n);                                            \
}

// =========================================================================
// Instantiate FFI entry points
// =========================================================================

// @tvm_ffi spfloat_densemm_nt_f32
FFI_SPFLOAT_MM_NT_WPR(_f32, float, 32)
// @tvm_ffi spfloat_densemm_nt_tpe_f32
FFI_SPFLOAT_MM_NT_TPE(_f32, float)
// @tvm_ffi spfloat_densemm_t_f32
FFI_SPFLOAT_MM_T(_f32, float, 32)

// @tvm_ffi spfloat_densemm_nt_f64
FFI_SPFLOAT_MM_NT_WPR(_f64, double, 16)
// @tvm_ffi spfloat_densemm_nt_tpe_f64
FFI_SPFLOAT_MM_NT_TPE(_f64, double)
// @tvm_ffi spfloat_densemm_t_f64
FFI_SPFLOAT_MM_T(_f64, double, 16)

// @tvm_ffi spfloat_densemm_nt_f16
FFI_SPFLOAT_MM_NT_WPR(_f16, __half, 32)
// @tvm_ffi spfloat_densemm_nt_tpe_f16
FFI_SPFLOAT_MM_NT_TPE(_f16, __half)
// @tvm_ffi spfloat_densemm_t_f16
FFI_SPFLOAT_MM_T(_f16, __half, 32)

// @tvm_ffi spfloat_densemm_nt_bf16
FFI_SPFLOAT_MM_NT_WPR(_bf16, __nv_bfloat16, 32)
// @tvm_ffi spfloat_densemm_nt_tpe_bf16
FFI_SPFLOAT_MM_NT_TPE(_bf16, __nv_bfloat16)
// @tvm_ffi spfloat_densemm_t_bf16
FFI_SPFLOAT_MM_T(_bf16, __nv_bfloat16, 32)
