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
 * binary_densemm.cu -- Event-Driven Binary Dense Matrix-Matrix CUDA Kernels
 * ==========================================================================
 *
 * Python API: brainevent.binary_densemm(weights, spikes, *, transpose, backend)
 *
 * Computes event-driven dense matrix-matrix product where the spike matrix
 * is binary (0/1) or float (>0 active). Skips FMA for inactive spikes.
 *
 * Gather mode (transpose=False):
 *   out[i, j] = sum_{l where spikes[l, j] active} weights[i, l]
 *   weights: [m, k] row-major, spikes: [k, n] row-major, output: [m, n]
 *
 * Scatter mode (transpose=True):
 *   out[i, j] = sum_{l where spikes[l, j] active} weights[l, i]
 *   weights: [k, m] row-major, spikes: [k, n] row-major, output: [m, n]
 *
 * Kernel design
 * -------------
 * Both gather and scatter use a 2D-tiled approach:
 *   Grid:  (ceil(n/BN), ceil(m/BM))
 *   Block: (BN, BM/RPT) = (128, 2) = 256 threads
 *   Each thread computes RPT=16 output rows for one output column.
 *
 * BN=128 maximizes weight reuse across columns within a block. The weight
 * tile [BM x BK] is loaded once into shared memory and reused by all 128
 * column-threads. Reducing BN causes column-tiles to redundantly load
 * the weight matrix from DRAM (L2 can't cache the full matrix).
 *
 * Both gather and scatter tile weights into shared memory:
 * - Gather: weights[m,k] row-major. Load weights[i0+bm, k0+bk] with
 *   bk as fast-varying index → stride-1 → coalesced.
 * - Scatter: weights[k,m] row-major. Load weights[k0+bk, i0+bm] with
 *   bm as fast-varying index → stride-1 → coalesced.
 * Both store into transposed shared layout s_W[bk*(BM+1)+bm] with
 * +1 padding to avoid bank conflicts.
 *
 * Spike values are read from global memory (L2-cached across m-tiles).
 * Consecutive threads read consecutive j-values for the same k → coalesced.
 *
 * IMPORTANT: data_ptr() returns GPU device pointers. NEVER dereference
 * on host. Pass to kernels unchanged.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// =========================================================================
// Active-check predicates
// =========================================================================

#define IS_ACTIVE_BOOL(s)  ((s) != 0)
#define IS_ACTIVE_FLOAT(s) ((s) > 0.0f)

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
// Tile parameters
// =========================================================================
//
// BN = output columns per block (= blockDim.x). BN=128 maximizes weight
//      reuse: each column-thread reads from the same weight tile in smem.
// BM = output rows per block = 32.
// RPT = rows per thread = BM / blockDim.y = 32 / 2 = 16.
// BK = k-chunk size for weight tiling = 64.
//
// Block: (128, 2) = 256 threads.
// Shared memory: BK * (BM+1) * sizeof(ACC_T).
//   f32: 64 * 33 * 4 = 8448 bytes per block.
//
// BK=64 halves the number of k-loop iterations (and __syncthreads
// barriers) compared to BK=32, while keeping BM=32 for optimal tile
// size and occupancy (6 blocks/SM = 1536 threads = 100% on GA102).
// Shared memory per block: 8448B * 6 blocks = 50.7KB (fits in 48KB
// default; request 64KB opt-in for 6 blocks).

#define BN  128
#define BM  32
#define RPT 16
#define BK  64
// blockDim.y = BM / RPT = 2

// =========================================================================
// Gather tiled kernel (transpose=False)
//
// Grid:  (ceil(n/BN), ceil(m/BM))
// Block: (BN, BM/RPT) = (128, 2) = 256 threads
//
// Weight tile load: weights[m,k] row-major.
//   bm = idx/BK (slow), bk = idx%BK (fast).
//   A warp of 32 threads reads bk=0..31 for the same bm → 32 consecutive
//   k-values → 128-byte cache line → coalesced.
//
// Shared layout: s_W[bk * (BM+1) + bm].
//   Compute reads s_W[bk * stride + (ty*RPT + ri)] → stride-1 in bm
//   → no bank conflicts.
// =========================================================================

#define DEFINE_GATHER_TILED(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T,   \
                            READ_W, WRITE_W, ACC_ZERO, ACC_SIZE)             \
__global__ void _gather_tiled_kern##SUFFIX(                                 \
    const WEIGHT_T* __restrict__ weights,                                   \
    const SPIKE_T*  __restrict__ spikes,                                    \
    WEIGHT_T*       __restrict__ output,                                    \
    int m, int k, int n                                                     \
) {                                                                         \
    int j0 = blockIdx.x * BN;                                               \
    int i0 = blockIdx.y * BM;                                               \
    int tx = threadIdx.x;                                                    \
    int ty = threadIdx.y;                                                    \
    int j = j0 + tx;                                                        \
    int i_base = i0 + ty * RPT;                                              \
    int tid = ty * BN + tx;                                                  \
    int nthreads = BN * (BM / RPT);  /* = 256 */                           \
                                                                             \
    ACC_T acc[RPT];                                                          \
    for (int ri = 0; ri < RPT; ri++) acc[ri] = ACC_ZERO;                    \
                                                                             \
    extern __shared__ char _smem_bytes[];                                    \
    ACC_T* s_W = reinterpret_cast<ACC_T*>(_smem_bytes);                     \
    const int SW_STRIDE = BM + 1;                                            \
                                                                             \
    for (int k0 = 0; k0 < k; k0 += BK) {                                   \
        int krem = k - k0;                                                   \
        int bk_end = (krem < BK) ? krem : BK;                               \
                                                                             \
        /* Cooperative coalesced weight tile load */                         \
        /* bk = idx % BK (fast) → consecutive k-values → coalesced */       \
        for (int idx = tid; idx < BM * BK; idx += nthreads) {               \
            int bm = idx / BK;                                               \
            int bk = idx % BK;                                               \
            int gi = i0 + bm;                                               \
            int gk = k0 + bk;                                               \
            ACC_T val = ACC_ZERO;                                            \
            if (gi < m && gk < k) {                                          \
                val = READ_W(weights[(size_t)gi * k + gk]);                  \
            }                                                                \
            s_W[bk * SW_STRIDE + bm] = val;                                 \
        }                                                                    \
        __syncthreads();                                                     \
                                                                             \
        /* Event-driven accumulation from shared memory */                  \
        if (j < n) {                                                         \
            for (int bk = 0; bk < bk_end; bk++) {                           \
                SPIKE_T spk = spikes[(size_t)(k0 + bk) * n + j];            \
                if (IS_ACTIVE(spk)) {                                        \
                    for (int ri = 0; ri < RPT; ri++) {                       \
                        acc[ri] += s_W[bk * SW_STRIDE + (ty * RPT + ri)];   \
                    }                                                        \
                }                                                            \
            }                                                                \
        }                                                                    \
        __syncthreads();                                                     \
    }                                                                        \
                                                                             \
    /* Write output */                                                      \
    if (j < n) {                                                             \
        for (int ri = 0; ri < RPT; ri++) {                                   \
            int gi = i_base + ri;                                            \
            if (gi < m) {                                                    \
                output[(size_t)gi * n + j] = WRITE_W(acc[ri]);               \
            }                                                                \
        }                                                                    \
    }                                                                        \
}

// =========================================================================
// Scatter tiled kernel (transpose=True)
//
// Grid:  (ceil(n/BN), ceil(m/BM))
// Block: (BN, BM/RPT) = (128, 2) = 256 threads
//
// Weight tile load: weights[k,m] row-major.
//   bm = idx%BM (fast), bk = idx/BM (slow).
//   A warp of 32 threads reads bm=0..31 for the same bk → 32 consecutive
//   m-values from one k-row → stride-1 → coalesced.
//
// Same transposed shared layout s_W[bk*(BM+1)+bm] as gather.
// Compute phase is identical to gather.
// =========================================================================

#define DEFINE_SCATTER_TILED(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T,  \
                             READ_W, WRITE_W, ACC_ZERO, ACC_SIZE)            \
__global__ void _scatter_tiled_kern##SUFFIX(                                \
    const WEIGHT_T* __restrict__ weights,                                   \
    const SPIKE_T*  __restrict__ spikes,                                    \
    WEIGHT_T*       __restrict__ output,                                    \
    int k, int m, int n                                                     \
) {                                                                         \
    int j0 = blockIdx.x * BN;                                               \
    int i0 = blockIdx.y * BM;                                               \
    int tx = threadIdx.x;                                                    \
    int ty = threadIdx.y;                                                    \
    int j = j0 + tx;                                                        \
    int i_base = i0 + ty * RPT;                                              \
    int tid = ty * BN + tx;                                                  \
    int nthreads = BN * (BM / RPT);                                         \
                                                                             \
    ACC_T acc[RPT];                                                          \
    for (int ri = 0; ri < RPT; ri++) acc[ri] = ACC_ZERO;                    \
                                                                             \
    extern __shared__ char _smem_bytes[];                                    \
    ACC_T* s_W = reinterpret_cast<ACC_T*>(_smem_bytes);                     \
    const int SW_STRIDE = BM + 1;                                            \
                                                                             \
    for (int k0 = 0; k0 < k; k0 += BK) {                                   \
        int krem = k - k0;                                                   \
        int bk_end = (krem < BK) ? krem : BK;                               \
                                                                             \
        /* Cooperative coalesced weight tile load */                         \
        /* bm = idx % BM (fast) → consecutive m-values → coalesced */       \
        for (int idx = tid; idx < BM * BK; idx += nthreads) {               \
            int bm = idx % BM;                                               \
            int bk = idx / BM;                                               \
            int gi = i0 + bm;                                               \
            int gk = k0 + bk;                                               \
            ACC_T val = ACC_ZERO;                                            \
            if (gi < m && gk < k) {                                          \
                val = READ_W(weights[(size_t)gk * m + gi]);                  \
            }                                                                \
            s_W[bk * SW_STRIDE + bm] = val;                                 \
        }                                                                    \
        __syncthreads();                                                     \
                                                                             \
        /* Event-driven accumulation (identical to gather) */               \
        if (j < n) {                                                         \
            for (int bk = 0; bk < bk_end; bk++) {                           \
                SPIKE_T spk = spikes[(size_t)(k0 + bk) * n + j];            \
                if (IS_ACTIVE(spk)) {                                        \
                    for (int ri = 0; ri < RPT; ri++) {                       \
                        acc[ri] += s_W[bk * SW_STRIDE + (ty * RPT + ri)];   \
                    }                                                        \
                }                                                            \
            }                                                                \
        }                                                                    \
        __syncthreads();                                                     \
    }                                                                        \
                                                                             \
    /* Write output */                                                      \
    if (j < n) {                                                             \
        for (int ri = 0; ri < RPT; ri++) {                                   \
            int gi = i_base + ri;                                            \
            if (gi < m) {                                                    \
                output[(size_t)gi * n + j] = WRITE_W(acc[ri]);               \
            }                                                                \
        }                                                                    \
    }                                                                        \
}


// =========================================================================
// Instantiate device kernels: 4 dtypes x 2 spike types x 2 modes
// =========================================================================

// ---- Float32 ----
DEFINE_GATHER_TILED(_f32_bool,   int8_t, IS_ACTIVE_BOOL,  float, float, READ_F32, WRITE_F32, 0.0f, 4)
DEFINE_GATHER_TILED(_f32_float,  float,  IS_ACTIVE_FLOAT, float, float, READ_F32, WRITE_F32, 0.0f, 4)
DEFINE_SCATTER_TILED(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float, float, READ_F32, WRITE_F32, 0.0f, 4)
DEFINE_SCATTER_TILED(_f32_float, float,  IS_ACTIVE_FLOAT, float, float, READ_F32, WRITE_F32, 0.0f, 4)

// ---- Float64 ----
DEFINE_GATHER_TILED(_f64_bool,   int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64, WRITE_F64, 0.0, 8)
DEFINE_GATHER_TILED(_f64_float,  float,  IS_ACTIVE_FLOAT, double, double, READ_F64, WRITE_F64, 0.0, 8)
DEFINE_SCATTER_TILED(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64, WRITE_F64, 0.0, 8)
DEFINE_SCATTER_TILED(_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64, WRITE_F64, 0.0, 8)

// ---- Float16 (accumulate in float32) ----
DEFINE_GATHER_TILED(_f16_bool,   int8_t, IS_ACTIVE_BOOL,  __half, float, READ_F16, WRITE_F16, 0.0f, 4)
DEFINE_GATHER_TILED(_f16_float,  float,  IS_ACTIVE_FLOAT, __half, float, READ_F16, WRITE_F16, 0.0f, 4)
DEFINE_SCATTER_TILED(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float, READ_F16, WRITE_F16, 0.0f, 4)
DEFINE_SCATTER_TILED(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float, READ_F16, WRITE_F16, 0.0f, 4)

// ---- BFloat16 (accumulate in float32; requires CUDA 11.0+) ----
DEFINE_GATHER_TILED(_bf16_bool,   int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f, 4)
DEFINE_GATHER_TILED(_bf16_float,  float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f, 4)
DEFINE_SCATTER_TILED(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f, 4)
DEFINE_SCATTER_TILED(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f, 4)


// =========================================================================
// TVM FFI Entry Point Macros
// =========================================================================
// Both gather and scatter use shared memory for weight tiling.
// Shared memory: BK * (BM + 1) * sizeof(ACC_T)
//   f32/f16/bf16 (ACC_T=float): 64 * 33 * 4 = 8448 bytes
//   f64 (ACC_T=double):         64 * 33 * 8 = 16896 bytes

#define FFI_GATHER(SUFFIX, WEIGHT_C_T, SPIKE_C_T, SHM_SIZE)                \
void binary_densemm_gather_auto##SUFFIX(                                    \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView spikes,             \
    tvm::ffi::TensorView output, int64_t stream                             \
) {                                                                         \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                \
    int m = static_cast<int>(weights.size(0));                              \
    int k = static_cast<int>(weights.size(1));                              \
    int n = static_cast<int>(spikes.size(1));                               \
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);                      \
    dim3 block(BN, BM / RPT);                                               \
    _gather_tiled_kern##SUFFIX<<<grid, block, SHM_SIZE, s>>>(               \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                 \
        static_cast<const SPIKE_C_T*>(spikes.data_ptr()),                   \
        static_cast<WEIGHT_C_T*>(output.data_ptr()), m, k, n);             \
}

#define FFI_SCATTER(SUFFIX, WEIGHT_C_T, SPIKE_C_T, SHM_SIZE)               \
void binary_densemm_scatter_auto##SUFFIX(                                   \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView spikes,             \
    tvm::ffi::TensorView output, int64_t stream                             \
) {                                                                         \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                \
    int k = static_cast<int>(weights.size(0));                              \
    int m = static_cast<int>(weights.size(1));                              \
    int n = static_cast<int>(spikes.size(1));                               \
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);                      \
    dim3 block(BN, BM / RPT);                                               \
    _scatter_tiled_kern##SUFFIX<<<grid, block, SHM_SIZE, s>>>(              \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                 \
        static_cast<const SPIKE_C_T*>(spikes.data_ptr()),                   \
        static_cast<WEIGHT_C_T*>(output.data_ptr()), k, m, n);             \
}


// =========================================================================
// Instantiate TVM FFI entry points
// =========================================================================

// ---- Float32 ----
// @tvm_ffi binary_densemm_gather_auto_f32_bool
FFI_GATHER(_f32_bool,     float,   int8_t, BK * (BM + 1) * sizeof(float))
// @tvm_ffi binary_densemm_gather_auto_f32_float
FFI_GATHER(_f32_float,    float,   float,  BK * (BM + 1) * sizeof(float))
// @tvm_ffi binary_densemm_scatter_auto_f32_bool
FFI_SCATTER(_f32_bool,    float,   int8_t, BK * (BM + 1) * sizeof(float))
// @tvm_ffi binary_densemm_scatter_auto_f32_float
FFI_SCATTER(_f32_float,   float,   float,  BK * (BM + 1) * sizeof(float))

// ---- Float64 ----
// @tvm_ffi binary_densemm_gather_auto_f64_bool
FFI_GATHER(_f64_bool,     double,  int8_t, BK * (BM + 1) * sizeof(double))
// @tvm_ffi binary_densemm_gather_auto_f64_float
FFI_GATHER(_f64_float,    double,  float,  BK * (BM + 1) * sizeof(double))
// @tvm_ffi binary_densemm_scatter_auto_f64_bool
FFI_SCATTER(_f64_bool,    double,  int8_t, BK * (BM + 1) * sizeof(double))
// @tvm_ffi binary_densemm_scatter_auto_f64_float
FFI_SCATTER(_f64_float,   double,  float,  BK * (BM + 1) * sizeof(double))

// ---- Float16 (ACC_T = float) ----
// @tvm_ffi binary_densemm_gather_auto_f16_bool
FFI_GATHER(_f16_bool,     __half,  int8_t, BK * (BM + 1) * sizeof(float))
// @tvm_ffi binary_densemm_gather_auto_f16_float
FFI_GATHER(_f16_float,    __half,  float,  BK * (BM + 1) * sizeof(float))
// @tvm_ffi binary_densemm_scatter_auto_f16_bool
FFI_SCATTER(_f16_bool,    __half,  int8_t, BK * (BM + 1) * sizeof(float))
// @tvm_ffi binary_densemm_scatter_auto_f16_float
FFI_SCATTER(_f16_float,   __half,  float,  BK * (BM + 1) * sizeof(float))

// ---- BFloat16 (ACC_T = float) ----
// @tvm_ffi binary_densemm_gather_auto_bf16_bool
FFI_GATHER(_bf16_bool,    __nv_bfloat16, int8_t, BK * (BM + 1) * sizeof(float))
// @tvm_ffi binary_densemm_gather_auto_bf16_float
FFI_GATHER(_bf16_float,   __nv_bfloat16, float,  BK * (BM + 1) * sizeof(float))
// @tvm_ffi binary_densemm_scatter_auto_bf16_bool
FFI_SCATTER(_bf16_bool,   __nv_bfloat16, int8_t, BK * (BM + 1) * sizeof(float))
// @tvm_ffi binary_densemm_scatter_auto_bf16_float
FFI_SCATTER(_bf16_float,  __nv_bfloat16, float,  BK * (BM + 1) * sizeof(float))
