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
 * This module provides optimized CUDA kernels for event-driven dense
 * matrix-matrix operations (SpMM):
 *
 * 1. binary_densemm_gather_auto  -- weights[m,k] @ spikes[k,n] -> out[m,n]
 *    (transpose=False): tiled shared-memory gather kernel.
 *
 * 2. binary_densemm_scatter_auto  -- spikes[k,n] @ weights[k,m] -> out[m,n]
 *    (transpose=True): tiled shared-memory scatter kernel.
 *
 * Python API (brainevent._dense.binary):
 *   binary_densemm(weights, spikes, transpose=False)
 *     weights : float16/float32/float64/bfloat16 matrix or scalar
 *     spikes  : bool (int8) or float32 spike matrix
 *     returns : output matrix
 *
 * CUDA entry points:
 *   binary_densemm_gather_auto_homo_{dtype}_{spike_dtype}
 *   binary_densemm_gather_auto_hetero_{dtype}_{spike_dtype}
 *   binary_densemm_scatter_auto_homo_{dtype}_{spike_dtype}
 *   binary_densemm_scatter_auto_hetero_{dtype}_{spike_dtype}
 *
 * PERFORMANCE NOTES (RTX 3080 Ti, f32, bool spikes):
 * ===================================================
 *  Config (m x k x n)         Density  cuda    cuBLAS   Speedup
 *  5K x 5K x 100          1%       1.1ms     1.17ms   1.06x  ok
 *  10K x 10K x 100        1%       3.3ms     1.33ms   0.40x  slow
 *  20K x 20K x 100        1%       11ms      5ms      0.44x  slow
 *
 * The event-driven tiled kernel matches cuBLAS at small sizes (<=5K) but
 * falls behind at large sizes (>=10K) where weight reads dominate bandwidth
 * and cannot be skipped.  Use brainevent._csr (CSR format) for 3-5x speedup
 * at high sparsity, or jax_raw backend (cuBLAS) for large dense matrices.
 */

#include "cuda_common.h"
#include "brainevent/common.h"

// =========================================================================
// Dense Matrix-Matrix Multiplication (densemm) -- tiling constants
// =========================================================================

#define BN  128
#define BM  32
#define RPT 16
#define BK  64

// =========================================================================
// Homo gather kernel (scalar weight broadcast to all connections)
// =========================================================================

#define DEFINE_GATHER_TILED_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T,  \
                                 READ_W, WRITE_W, ACC_ZERO, ACC_SIZE)          \
__global__ void _gather_tiled_homo_kern##SUFFIX(                               \
    const WEIGHT_T* __restrict__ weights,                                      \
    const SPIKE_T*  __restrict__ spikes,                                       \
    WEIGHT_T*       __restrict__ output,                                       \
    int m, int k, int n                                                        \
) {                                                                            \
    int j0 = blockIdx.x * BN;                                                  \
    int i0 = blockIdx.y * BM;                                                  \
    int tx = threadIdx.x;                                                      \
    int ty = threadIdx.y;                                                      \
    int j = j0 + tx;                                                           \
    int i_base = i0 + ty * RPT;                                                \
    int tid = ty * BN + tx;                                                    \
    int nthreads = BN * (BM / RPT);                                            \
    ACC_T acc[RPT];                                                            \
    for (int ri = 0; ri < RPT; ri++) acc[ri] = ACC_ZERO;                       \
    extern __shared__ char _smem_bytes[];                                      \
    ACC_T* s_W = reinterpret_cast<ACC_T*>(_smem_bytes);                        \
    const int SW_STRIDE = BM + 1;                                              \
    ACC_T homo_w = READ_W(weights[0]);                                         \
    for (int k0 = 0; k0 < k; k0 += BK) {                                       \
        int krem = k - k0;                                                     \
        int bk_end = (krem < BK) ? krem : BK;                                  \
        for (int idx = tid; idx < BM * BK; idx += nthreads) {                  \
            int bm = idx / BK;                                                 \
            int bk = idx % BK;                                                 \
            int gi = i0 + bm;                                                  \
            int gk = k0 + bk;                                                  \
            s_W[bk * SW_STRIDE + bm] = (gi < m && gk < k) ? homo_w : ACC_ZERO; \
        }                                                                      \
        __syncthreads();                                                       \
        if (j < n) {                                                           \
            for (int bk = 0; bk < bk_end; bk++) {                              \
                SPIKE_T spk = spikes[(size_t)(k0 + bk) * n + j];               \
                if (IS_ACTIVE(spk)) {                                          \
                    for (int ri = 0; ri < RPT; ri++) {                         \
                        acc[ri] += s_W[bk * SW_STRIDE + (ty * RPT + ri)];      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    if (j < n) {                                                               \
        for (int ri = 0; ri < RPT; ri++) {                                     \
            int gi = i_base + ri;                                              \
            if (gi < m) {                                                      \
                output[(size_t)gi * n + j] = WRITE_W(acc[ri]);                 \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

// =========================================================================
// Hetero gather kernel (per-connection weight matrix)
// =========================================================================

#define DEFINE_GATHER_TILED_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, \
                                   READ_W, WRITE_W, ACC_ZERO, ACC_SIZE)         \
__global__ void _gather_tiled_hetero_kern##SUFFIX(                              \
    const WEIGHT_T* __restrict__ weights,                                       \
    const SPIKE_T*  __restrict__ spikes,                                        \
    WEIGHT_T*       __restrict__ output,                                        \
    int m, int k, int n                                                         \
) {                                                                             \
    int j0 = blockIdx.x * BN;                                                   \
    int i0 = blockIdx.y * BM;                                                   \
    int tx = threadIdx.x;                                                       \
    int ty = threadIdx.y;                                                       \
    int j = j0 + tx;                                                            \
    int i_base = i0 + ty * RPT;                                                 \
    int tid = ty * BN + tx;                                                     \
    int nthreads = BN * (BM / RPT);                                             \
    ACC_T acc[RPT];                                                             \
    for (int ri = 0; ri < RPT; ri++) acc[ri] = ACC_ZERO;                        \
    extern __shared__ char _smem_bytes[];                                       \
    ACC_T* s_W = reinterpret_cast<ACC_T*>(_smem_bytes);                         \
    const int SW_STRIDE = BM + 1;                                               \
    for (int k0 = 0; k0 < k; k0 += BK) {                                        \
        int krem = k - k0;                                                      \
        int bk_end = (krem < BK) ? krem : BK;                                   \
        for (int idx = tid; idx < BM * BK; idx += nthreads) {                   \
            int bm = idx / BK;                                                  \
            int bk = idx % BK;                                                  \
            int gi = i0 + bm;                                                   \
            int gk = k0 + bk;                                                   \
            ACC_T val = ACC_ZERO;                                               \
            if (gi < m && gk < k) {                                             \
                val = READ_W(weights[(size_t)gi * k + gk]);                     \
            }                                                                   \
            s_W[bk * SW_STRIDE + bm] = val;                                     \
        }                                                                       \
        __syncthreads();                                                        \
        if (j < n) {                                                            \
            for (int bk = 0; bk < bk_end; bk++) {                               \
                SPIKE_T spk = spikes[(size_t)(k0 + bk) * n + j];                \
                if (IS_ACTIVE(spk)) {                                           \
                    for (int ri = 0; ri < RPT; ri++) {                          \
                        acc[ri] += s_W[bk * SW_STRIDE + (ty * RPT + ri)];       \
                    }                                                           \
                }                                                               \
            }                                                                   \
        }                                                                       \
        __syncthreads();                                                        \
    }                                                                           \
    if (j < n) {                                                                \
        for (int ri = 0; ri < RPT; ri++) {                                      \
            int gi = i_base + ri;                                               \
            if (gi < m) {                                                       \
                output[(size_t)gi * n + j] = WRITE_W(acc[ri]);                  \
            }                                                                   \
        }                                                                       \
    }                                                                           \
}

// =========================================================================
// Homo scatter kernel (scalar weight broadcast to all connections)
// =========================================================================

#define DEFINE_SCATTER_TILED_HOMO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, \
                                  READ_W, WRITE_W, ACC_ZERO, ACC_SIZE)         \
__global__ void _scatter_tiled_homo_kern##SUFFIX(                              \
    const WEIGHT_T* __restrict__ weights,                                      \
    const SPIKE_T*  __restrict__ spikes,                                       \
    WEIGHT_T*       __restrict__ output,                                       \
    int k, int m, int n                                                        \
) {                                                                            \
    int j0 = blockIdx.x * BN;                                                  \
    int i0 = blockIdx.y * BM;                                                  \
    int tx = threadIdx.x;                                                      \
    int ty = threadIdx.y;                                                      \
    int j = j0 + tx;                                                           \
    int i_base = i0 + ty * RPT;                                                \
    int tid = ty * BN + tx;                                                    \
    int nthreads = BN * (BM / RPT);                                            \
    ACC_T acc[RPT];                                                            \
    for (int ri = 0; ri < RPT; ri++) acc[ri] = ACC_ZERO;                       \
    extern __shared__ char _smem_bytes[];                                      \
    ACC_T* s_W = reinterpret_cast<ACC_T*>(_smem_bytes);                        \
    const int SW_STRIDE = BM + 1;                                              \
    ACC_T homo_w = READ_W(weights[0]);                                         \
    for (int k0 = 0; k0 < k; k0 += BK) {                                       \
        int krem = k - k0;                                                     \
        int bk_end = (krem < BK) ? krem : BK;                                  \
        for (int idx = tid; idx < BM * BK; idx += nthreads) {                  \
            int bm = idx % BM;                                                 \
            int bk = idx / BM;                                                 \
            int gi = i0 + bm;                                                  \
            int gk = k0 + bk;                                                  \
            s_W[bk * SW_STRIDE + bm] = (gi < m && gk < k) ? homo_w : ACC_ZERO; \
        }                                                                      \
        __syncthreads();                                                       \
        if (j < n) {                                                           \
            for (int bk = 0; bk < bk_end; bk++) {                              \
                SPIKE_T spk = spikes[(size_t)(k0 + bk) * n + j];               \
                if (IS_ACTIVE(spk)) {                                          \
                    for (int ri = 0; ri < RPT; ri++) {                         \
                        acc[ri] += s_W[bk * SW_STRIDE + (ty * RPT + ri)];      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    if (j < n) {                                                               \
        for (int ri = 0; ri < RPT; ri++) {                                     \
            int gi = i_base + ri;                                              \
            if (gi < m) {                                                      \
                output[(size_t)gi * n + j] = WRITE_W(acc[ri]);                 \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

// =========================================================================
// Hetero scatter kernel (per-connection weight matrix)
// =========================================================================

#define DEFINE_SCATTER_TILED_HETERO(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, \
                                    READ_W, WRITE_W, ACC_ZERO, ACC_SIZE)         \
__global__ void _scatter_tiled_hetero_kern##SUFFIX(                              \
    const WEIGHT_T* __restrict__ weights,                                        \
    const SPIKE_T*  __restrict__ spikes,                                         \
    WEIGHT_T*       __restrict__ output,                                         \
    int k, int m, int n                                                          \
) {                                                                              \
    int j0 = blockIdx.x * BN;                                                    \
    int i0 = blockIdx.y * BM;                                                    \
    int tx = threadIdx.x;                                                        \
    int ty = threadIdx.y;                                                        \
    int j = j0 + tx;                                                             \
    int i_base = i0 + ty * RPT;                                                  \
    int tid = ty * BN + tx;                                                      \
    int nthreads = BN * (BM / RPT);                                              \
    ACC_T acc[RPT];                                                              \
    for (int ri = 0; ri < RPT; ri++) acc[ri] = ACC_ZERO;                         \
    extern __shared__ char _smem_bytes[];                                        \
    ACC_T* s_W = reinterpret_cast<ACC_T*>(_smem_bytes);                          \
    const int SW_STRIDE = BM + 1;                                                \
    for (int k0 = 0; k0 < k; k0 += BK) {                                         \
        int krem = k - k0;                                                       \
        int bk_end = (krem < BK) ? krem : BK;                                    \
        for (int idx = tid; idx < BM * BK; idx += nthreads) {                    \
            int bm = idx % BM;                                                   \
            int bk = idx / BM;                                                   \
            int gi = i0 + bm;                                                    \
            int gk = k0 + bk;                                                    \
            ACC_T val = ACC_ZERO;                                                \
            if (gi < m && gk < k) {                                              \
                val = READ_W(weights[(size_t)gk * m + gi]);                      \
            }                                                                    \
            s_W[bk * SW_STRIDE + bm] = val;                                      \
        }                                                                        \
        __syncthreads();                                                         \
        if (j < n) {                                                             \
            for (int bk = 0; bk < bk_end; bk++) {                                \
                SPIKE_T spk = spikes[(size_t)(k0 + bk) * n + j];                 \
                if (IS_ACTIVE(spk)) {                                            \
                    for (int ri = 0; ri < RPT; ri++) {                           \
                        acc[ri] += s_W[bk * SW_STRIDE + (ty * RPT + ri)];        \
                    }                                                            \
                }                                                                \
            }                                                                    \
        }                                                                        \
        __syncthreads();                                                         \
    }                                                                            \
    if (j < n) {                                                                 \
        for (int ri = 0; ri < RPT; ri++) {                                       \
            int gi = i_base + ri;                                                \
            if (gi < m) {                                                        \
                output[(size_t)gi * n + j] = WRITE_W(acc[ri]);                   \
            }                                                                    \
        }                                                                        \
    }                                                                            \
}

// Homo gather instantiations
DEFINE_GATHER_TILED_HOMO(_f32_bool,   int8_t, IS_ACTIVE_BOOL,  float,          float,  READ_F32,  WRITE_F32,  0.0f, 4)
DEFINE_GATHER_TILED_HOMO(_f32_float,  float,  IS_ACTIVE_FLOAT, float,          float,  READ_F32,  WRITE_F32,  0.0f, 4)
DEFINE_GATHER_TILED_HOMO(_f64_bool,   int8_t, IS_ACTIVE_BOOL,  double,         double, READ_F64,  WRITE_F64,  0.0,  8)
DEFINE_GATHER_TILED_HOMO(_f64_float,  float,  IS_ACTIVE_FLOAT, double,         double, READ_F64,  WRITE_F64,  0.0,  8)
DEFINE_GATHER_TILED_HOMO(_f16_bool,   int8_t, IS_ACTIVE_BOOL,  __half,         float,  READ_F16,  WRITE_F16,  0.0f, 4)
DEFINE_GATHER_TILED_HOMO(_f16_float,  float,  IS_ACTIVE_FLOAT, __half,         float,  READ_F16,  WRITE_F16,  0.0f, 4)
DEFINE_GATHER_TILED_HOMO(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16,  float,  READ_BF16, WRITE_BF16, 0.0f, 4)
DEFINE_GATHER_TILED_HOMO(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16,  float,  READ_BF16, WRITE_BF16, 0.0f, 4)

// Hetero gather instantiations
DEFINE_GATHER_TILED_HETERO(_f32_bool,   int8_t, IS_ACTIVE_BOOL,  float,          float,  READ_F32,  WRITE_F32,  0.0f, 4)
DEFINE_GATHER_TILED_HETERO(_f32_float,  float,  IS_ACTIVE_FLOAT, float,          float,  READ_F32,  WRITE_F32,  0.0f, 4)
DEFINE_GATHER_TILED_HETERO(_f64_bool,   int8_t, IS_ACTIVE_BOOL,  double,         double, READ_F64,  WRITE_F64,  0.0,  8)
DEFINE_GATHER_TILED_HETERO(_f64_float,  float,  IS_ACTIVE_FLOAT, double,         double, READ_F64,  WRITE_F64,  0.0,  8)
DEFINE_GATHER_TILED_HETERO(_f16_bool,   int8_t, IS_ACTIVE_BOOL,  __half,         float,  READ_F16,  WRITE_F16,  0.0f, 4)
DEFINE_GATHER_TILED_HETERO(_f16_float,  float,  IS_ACTIVE_FLOAT, __half,         float,  READ_F16,  WRITE_F16,  0.0f, 4)
DEFINE_GATHER_TILED_HETERO(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16,  float,  READ_BF16, WRITE_BF16, 0.0f, 4)
DEFINE_GATHER_TILED_HETERO(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16,  float,  READ_BF16, WRITE_BF16, 0.0f, 4)

// Homo scatter instantiations
DEFINE_SCATTER_TILED_HOMO(_f32_bool,   int8_t, IS_ACTIVE_BOOL,  float,          float,  READ_F32,  WRITE_F32,  0.0f, 4)
DEFINE_SCATTER_TILED_HOMO(_f32_float,  float,  IS_ACTIVE_FLOAT, float,          float,  READ_F32,  WRITE_F32,  0.0f, 4)
DEFINE_SCATTER_TILED_HOMO(_f64_bool,   int8_t, IS_ACTIVE_BOOL,  double,         double, READ_F64,  WRITE_F64,  0.0,  8)
DEFINE_SCATTER_TILED_HOMO(_f64_float,  float,  IS_ACTIVE_FLOAT, double,         double, READ_F64,  WRITE_F64,  0.0,  8)
DEFINE_SCATTER_TILED_HOMO(_f16_bool,   int8_t, IS_ACTIVE_BOOL,  __half,         float,  READ_F16,  WRITE_F16,  0.0f, 4)
DEFINE_SCATTER_TILED_HOMO(_f16_float,  float,  IS_ACTIVE_FLOAT, __half,         float,  READ_F16,  WRITE_F16,  0.0f, 4)
DEFINE_SCATTER_TILED_HOMO(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16,  float,  READ_BF16, WRITE_BF16, 0.0f, 4)
DEFINE_SCATTER_TILED_HOMO(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16,  float,  READ_BF16, WRITE_BF16, 0.0f, 4)

// Hetero scatter instantiations
DEFINE_SCATTER_TILED_HETERO(_f32_bool,   int8_t, IS_ACTIVE_BOOL,  float,          float,  READ_F32,  WRITE_F32,  0.0f, 4)
DEFINE_SCATTER_TILED_HETERO(_f32_float,  float,  IS_ACTIVE_FLOAT, float,          float,  READ_F32,  WRITE_F32,  0.0f, 4)
DEFINE_SCATTER_TILED_HETERO(_f64_bool,   int8_t, IS_ACTIVE_BOOL,  double,         double, READ_F64,  WRITE_F64,  0.0,  8)
DEFINE_SCATTER_TILED_HETERO(_f64_float,  float,  IS_ACTIVE_FLOAT, double,         double, READ_F64,  WRITE_F64,  0.0,  8)
DEFINE_SCATTER_TILED_HETERO(_f16_bool,   int8_t, IS_ACTIVE_BOOL,  __half,         float,  READ_F16,  WRITE_F16,  0.0f, 4)
DEFINE_SCATTER_TILED_HETERO(_f16_float,  float,  IS_ACTIVE_FLOAT, __half,         float,  READ_F16,  WRITE_F16,  0.0f, 4)
DEFINE_SCATTER_TILED_HETERO(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16,  float,  READ_BF16, WRITE_BF16, 0.0f, 4)
DEFINE_SCATTER_TILED_HETERO(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16,  float,  READ_BF16, WRITE_BF16, 0.0f, 4)

// =========================================================================
// FFI entry points -- Homo gather
// =========================================================================

#define FFI_GATHER_MM_HOMO(SUFFIX, WEIGHT_C_T, SPIKE_C_T, SHM_SIZE) \
void binary_densemm_gather_auto_homo##SUFFIX(                       \
    const BE::Tensor weights, const BE::Tensor spikes,              \
    BE::Tensor output, int64_t stream                               \
) {                                                                 \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);        \
    int m = static_cast<int>(output.size(0));                       \
    int k = static_cast<int>(spikes.size(0));                       \
    int n = static_cast<int>(spikes.size(1));                       \
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);                \
    dim3 block(BN, BM / RPT);                                       \
    _gather_tiled_homo_kern##SUFFIX<<<grid, block, SHM_SIZE, s>>>(  \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),         \
        static_cast<const SPIKE_C_T*>(spikes.data_ptr()),           \
        static_cast<WEIGHT_C_T*>(output.data_ptr()), m, k, n);      \
}

// =========================================================================
// FFI entry points -- Hetero gather
// =========================================================================

#define FFI_GATHER_MM_HETERO(SUFFIX, WEIGHT_C_T, SPIKE_C_T, SHM_SIZE) \
void binary_densemm_gather_auto_hetero##SUFFIX(                       \
    const BE::Tensor weights, const BE::Tensor spikes,                \
    BE::Tensor output, int64_t stream                                 \
) {                                                                   \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);          \
    int m = static_cast<int>(weights.size(0));                        \
    int k = static_cast<int>(weights.size(1));                        \
    int n = static_cast<int>(spikes.size(1));                         \
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);                  \
    dim3 block(BN, BM / RPT);                                         \
    _gather_tiled_hetero_kern##SUFFIX<<<grid, block, SHM_SIZE, s>>>(  \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),           \
        static_cast<const SPIKE_C_T*>(spikes.data_ptr()),             \
        static_cast<WEIGHT_C_T*>(output.data_ptr()), m, k, n);        \
}

// =========================================================================
// FFI entry points -- Homo scatter
// =========================================================================

#define FFI_SCATTER_MM_HOMO(SUFFIX, WEIGHT_C_T, SPIKE_C_T, SHM_SIZE) \
void binary_densemm_scatter_auto_homo##SUFFIX(                       \
    const BE::Tensor weights, const BE::Tensor spikes,               \
    BE::Tensor output, int64_t stream                                \
) {                                                                  \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);         \
    int k = static_cast<int>(spikes.size(0));  /* spikes[k,n] */     \
    int m = static_cast<int>(output.size(0));  /* output[m,n] */     \
    int n = static_cast<int>(spikes.size(1));                        \
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);                 \
    dim3 block(BN, BM / RPT);                                        \
    _scatter_tiled_homo_kern##SUFFIX<<<grid, block, SHM_SIZE, s>>>(  \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),          \
        static_cast<const SPIKE_C_T*>(spikes.data_ptr()),            \
        static_cast<WEIGHT_C_T*>(output.data_ptr()), k, m, n);       \
}

// =========================================================================
// FFI entry points -- Hetero scatter
// =========================================================================

#define FFI_SCATTER_MM_HETERO(SUFFIX, WEIGHT_C_T, SPIKE_C_T, SHM_SIZE) \
void binary_densemm_scatter_auto_hetero##SUFFIX(                       \
    const BE::Tensor weights, const BE::Tensor spikes,                 \
    BE::Tensor output, int64_t stream                                  \
) {                                                                    \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);           \
    int k = static_cast<int>(weights.size(0));                         \
    int m = static_cast<int>(weights.size(1));                         \
    int n = static_cast<int>(spikes.size(1));                          \
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);                   \
    dim3 block(BN, BM / RPT);                                          \
    _scatter_tiled_hetero_kern##SUFFIX<<<grid, block, SHM_SIZE, s>>>(  \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),            \
        static_cast<const SPIKE_C_T*>(spikes.data_ptr()),              \
        static_cast<WEIGHT_C_T*>(output.data_ptr()), k, m, n);         \
}

// Homo gather FFI instantiations
// @BE binary_densemm_gather_auto_homo_f32_bool
FFI_GATHER_MM_HOMO(_f32_bool,    float,          int8_t, BK * (BM + 1) * sizeof(float))
// @BE binary_densemm_gather_auto_homo_f32_float
FFI_GATHER_MM_HOMO(_f32_float,   float,          float,  BK * (BM + 1) * sizeof(float))
// @BE binary_densemm_gather_auto_homo_f64_bool
FFI_GATHER_MM_HOMO(_f64_bool,    double,         int8_t, BK * (BM + 1) * sizeof(double))
// @BE binary_densemm_gather_auto_homo_f64_float
FFI_GATHER_MM_HOMO(_f64_float,   double,         float,  BK * (BM + 1) * sizeof(double))
// @BE binary_densemm_gather_auto_homo_f16_bool
FFI_GATHER_MM_HOMO(_f16_bool,    __half,         int8_t, BK * (BM + 1) * sizeof(float))
// @BE binary_densemm_gather_auto_homo_f16_float
FFI_GATHER_MM_HOMO(_f16_float,   __half,         float,  BK * (BM + 1) * sizeof(float))
// @BE binary_densemm_gather_auto_homo_bf16_bool
FFI_GATHER_MM_HOMO(_bf16_bool,   __nv_bfloat16,  int8_t, BK * (BM + 1) * sizeof(float))
// @BE binary_densemm_gather_auto_homo_bf16_float
FFI_GATHER_MM_HOMO(_bf16_float,  __nv_bfloat16,  float,  BK * (BM + 1) * sizeof(float))

// Hetero gather FFI instantiations
// @BE binary_densemm_gather_auto_hetero_f32_bool
FFI_GATHER_MM_HETERO(_f32_bool,    float,          int8_t, BK * (BM + 1) * sizeof(float))
// @BE binary_densemm_gather_auto_hetero_f32_float
FFI_GATHER_MM_HETERO(_f32_float,   float,          float,  BK * (BM + 1) * sizeof(float))
// @BE binary_densemm_gather_auto_hetero_f64_bool
FFI_GATHER_MM_HETERO(_f64_bool,    double,         int8_t, BK * (BM + 1) * sizeof(double))
// @BE binary_densemm_gather_auto_hetero_f64_float
FFI_GATHER_MM_HETERO(_f64_float,   double,         float,  BK * (BM + 1) * sizeof(double))
// @BE binary_densemm_gather_auto_hetero_f16_bool
FFI_GATHER_MM_HETERO(_f16_bool,    __half,         int8_t, BK * (BM + 1) * sizeof(float))
// @BE binary_densemm_gather_auto_hetero_f16_float
FFI_GATHER_MM_HETERO(_f16_float,   __half,         float,  BK * (BM + 1) * sizeof(float))
// @BE binary_densemm_gather_auto_hetero_bf16_bool
FFI_GATHER_MM_HETERO(_bf16_bool,   __nv_bfloat16,  int8_t, BK * (BM + 1) * sizeof(float))
// @BE binary_densemm_gather_auto_hetero_bf16_float
FFI_GATHER_MM_HETERO(_bf16_float,  __nv_bfloat16,  float,  BK * (BM + 1) * sizeof(float))

// Homo scatter FFI instantiations
// @BE binary_densemm_scatter_auto_homo_f32_bool
FFI_SCATTER_MM_HOMO(_f32_bool,    float,          int8_t, BK * (BM + 1) * sizeof(float))
// @BE binary_densemm_scatter_auto_homo_f32_float
FFI_SCATTER_MM_HOMO(_f32_float,   float,          float,  BK * (BM + 1) * sizeof(float))
// @BE binary_densemm_scatter_auto_homo_f64_bool
FFI_SCATTER_MM_HOMO(_f64_bool,    double,         int8_t, BK * (BM + 1) * sizeof(double))
// @BE binary_densemm_scatter_auto_homo_f64_float
FFI_SCATTER_MM_HOMO(_f64_float,   double,         float,  BK * (BM + 1) * sizeof(double))
// @BE binary_densemm_scatter_auto_homo_f16_bool
FFI_SCATTER_MM_HOMO(_f16_bool,    __half,         int8_t, BK * (BM + 1) * sizeof(float))
// @BE binary_densemm_scatter_auto_homo_f16_float
FFI_SCATTER_MM_HOMO(_f16_float,   __half,         float,  BK * (BM + 1) * sizeof(float))
// @BE binary_densemm_scatter_auto_homo_bf16_bool
FFI_SCATTER_MM_HOMO(_bf16_bool,   __nv_bfloat16,  int8_t, BK * (BM + 1) * sizeof(float))
// @BE binary_densemm_scatter_auto_homo_bf16_float
FFI_SCATTER_MM_HOMO(_bf16_float,  __nv_bfloat16,  float,  BK * (BM + 1) * sizeof(float))

// Hetero scatter FFI instantiations
// @BE binary_densemm_scatter_auto_hetero_f32_bool
FFI_SCATTER_MM_HETERO(_f32_bool,    float,          int8_t, BK * (BM + 1) * sizeof(float))
// @BE binary_densemm_scatter_auto_hetero_f32_float
FFI_SCATTER_MM_HETERO(_f32_float,   float,          float,  BK * (BM + 1) * sizeof(float))
// @BE binary_densemm_scatter_auto_hetero_f64_bool
FFI_SCATTER_MM_HETERO(_f64_bool,    double,         int8_t, BK * (BM + 1) * sizeof(double))
// @BE binary_densemm_scatter_auto_hetero_f64_float
FFI_SCATTER_MM_HETERO(_f64_float,   double,         float,  BK * (BM + 1) * sizeof(double))
// @BE binary_densemm_scatter_auto_hetero_f16_bool
FFI_SCATTER_MM_HETERO(_f16_bool,    __half,         int8_t, BK * (BM + 1) * sizeof(float))
// @BE binary_densemm_scatter_auto_hetero_f16_float
FFI_SCATTER_MM_HETERO(_f16_float,   __half,         float,  BK * (BM + 1) * sizeof(float))
// @BE binary_densemm_scatter_auto_hetero_bf16_bool
FFI_SCATTER_MM_HETERO(_bf16_bool,   __nv_bfloat16,  int8_t, BK * (BM + 1) * sizeof(float))
// @BE binary_densemm_scatter_auto_hetero_bf16_float
FFI_SCATTER_MM_HETERO(_bf16_float,  __nv_bfloat16,  float,  BK * (BM + 1) * sizeof(float))
