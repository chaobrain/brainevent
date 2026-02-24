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
 * float_jitsmm.cu — JIT Scalar Float Matrix-Matrix Product CUDA Kernels
 * ======================================================================
 *
 * Float matrix-matrix product for JIT scalar connectivity.
 *
 * Operation
 * ---------
 * jitsmm — Float matrix-matrix: Y = M @ B
 *   where M[i,j] = w * Bernoulli(prob) is generated on-the-fly.
 *
 * Parameters
 * ----------
 * weight : shape (1,), scalar weight for all connections
 * clen   : shape (1,), connection length = 2/prob (float32)
 * seed   : shape (1,), int32 random seed
 * B      : shape (k, n), input matrix
 * output : shape (m, n), output matrix
 *
 * corder=True  (gather): one thread per output row, no atomics
 *   Dispatches to register-accumulator kernel for n<=16 (2.7x faster),
 *   falls back to thread-per-row kernel for n>16.
 * corder=False (scatter): one thread per input row, uses atomicAdd
 *
 * Supported weight dtypes: float32, float64, float16, bfloat16.
 *
 * Performance notes (RTX 3080 Ti Laptop, SM 8.6):
 *   jitsmm gather n=10: ~1.2ms (regacc, at parity with mv)
 *   jitsmm scatter n=10: ~5.2ms (atomicAdd x 10 columns)
 *
 * IMPORTANT: All data_ptr() returns are GPU device pointers — NEVER dereference on host.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include <cstdint>
#include "cuda_common.h"

// #########################################################################
// ##  jitsmm — Float Matrix-Matrix Product                               ##
// #########################################################################
//
// Performance notes (RTX 3080 Ti Laptop, SM 8.6):
//   Two kernel strategies based on output column count n:
//
//   (A) Register-accumulator gather (n <= 16):
//     Thread-per-row with 16 ACC_T register accumulators. Eliminates
//     inner-loop global R/W (3 ops/col/conn → 1 read/col/conn).
//     256 rows/block for maximum SM occupancy. No curand redundancy.
//     Saves ~2n bytes/conn/row of memory traffic vs global R/W approach.
//     Note: warp-cooperative (32 threads/row) was tested but regressed
//     for small n because redundant curand (32× per row) and poor SM
//     utilization (8 rows/block) dominated over coalesced B-read benefits.
//
//   (B) Thread-per-row gather (n > 16):
//     Original approach with __ldg on B reads. One thread per row,
//     serial column loop with global R/W per connection.
//
//   Fundamental barriers:
//   - curand Philox sequential dependency limits per-row throughput
//   - Random B-row access pattern (determined by curand) prevents
//     prefetch optimization; relies on L2 cache for B matrix

// =========================================================================
// Register-accumulator gather kernel (n <= 16): one thread per row
// Uses 16 register accumulators to eliminate inner-loop global R/W.
// 256 rows/block for maximum SM occupancy; no curand redundancy.
// =========================================================================

#define DEFINE_JITSMM_GATHER_REGACC(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO)    \
__global__ void _jitsmm_gather_regacc_kern##SUFFIX(                                        \
    const WEIGHT_T* __restrict__ weight,                                                   \
    const float*    __restrict__ clen,                                                     \
    const int*      __restrict__ seed,                                                     \
    const WEIGHT_T* __restrict__ B,                                                        \
    WEIGHT_T*       __restrict__ output,                                                   \
    int m, int k, int n                                                                    \
) {                                                                                        \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                         \
    if (i >= m) return;                                                                    \
    ACC_T w0 = READ_W(__ldg(&weight[0]));                                                  \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                       \
    if (cl < 2) cl = 2;                                                                    \
    curandStatePhilox4_32_10_t state;                                                      \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)i, 0ULL, &state); \
    unsigned int j = curand(&state) % cl;                                                  \
    ACC_T acc[16];                                                                         \
    for (int c = 0; c < 16; c++) acc[c] = ACC_ZERO;                                        \
    while (j < (unsigned int)k) {                                                          \
        const WEIGHT_T* br = B + (size_t)j * n;                                            \
        for (int c = 0; c < 16; c++) {                                                     \
            if (c < n) acc[c] += READ_W(__ldg(&br[c]));                                    \
        }                                                                                  \
        j += 1 + (curand(&state) % (cl - 1));                                              \
    }                                                                                      \
    WEIGHT_T* out_row = output + (size_t)i * n;                                            \
    for (int c = 0; c < 16; c++) {                                                         \
        if (c < n) out_row[c] = WRITE_W(w0 * acc[c]);                                      \
    }                                                                                      \
}

DEFINE_JITSMM_GATHER_REGACC(_f32,  float,         float,  READ_F32,  WRITE_F32,  0.0f)
DEFINE_JITSMM_GATHER_REGACC(_f64,  double,        double, READ_F64,  WRITE_F64,  0.0)
DEFINE_JITSMM_GATHER_REGACC(_f16,  __half,        float,  READ_F16,  WRITE_F16,  0.0f)
DEFINE_JITSMM_GATHER_REGACC(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, 0.0f)

// =========================================================================
// Thread-per-row gather kernel (fallback for n > 128)
// Y[i, :] = w * sum_{j in C(i)} B[j, :]
// =========================================================================

#define DEFINE_JITSMM_GATHER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO)           \
__global__ void _jitsmm_gather_kern##SUFFIX(                                               \
    const WEIGHT_T* __restrict__ weight,                                                   \
    const float*    __restrict__ clen,                                                     \
    const int*      __restrict__ seed,                                                     \
    const WEIGHT_T* __restrict__ B,                                                        \
    WEIGHT_T*       __restrict__ output,                                                   \
    int m, int k, int n                                                                    \
) {                                                                                        \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                         \
    if (i >= m) return;                                                                    \
    ACC_T w0 = READ_W(__ldg(&weight[0]));                                                  \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                       \
    if (cl < 2) cl = 2;                                                                    \
    curandStatePhilox4_32_10_t state;                                                      \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)i, 0ULL, &state); \
    unsigned int j = curand(&state) % cl;                                                  \
    /* Loop over connected rows, accumulate all columns */                                 \
    while (j < (unsigned int)k) {                                                          \
        const WEIGHT_T* b_row = B + (size_t)j * n;                                         \
        WEIGHT_T* out_row = output + (size_t)i * n;                                        \
        for (int col = 0; col < n; col++) {                                                \
            ACC_T cur = READ_W(out_row[col]);                                              \
            cur += READ_W(__ldg(&b_row[col]));                                             \
            out_row[col] = WRITE_W(cur);                                                   \
        }                                                                                  \
        j += 1 + (curand(&state) % (cl - 1));                                              \
    }                                                                                      \
    /* Scale by weight */                                                                  \
    WEIGHT_T* out_row = output + (size_t)i * n;                                            \
    for (int col = 0; col < n; col++) {                                                    \
        ACC_T cur = READ_W(out_row[col]);                                                  \
        out_row[col] = WRITE_W(w0 * cur);                                                  \
    }                                                                                      \
}

DEFINE_JITSMM_GATHER(_f32,  float,         float,  READ_F32,  WRITE_F32,  0.0f)
DEFINE_JITSMM_GATHER(_f64,  double,        double, READ_F64,  WRITE_F64,  0.0)
DEFINE_JITSMM_GATHER(_f16,  __half,        float,  READ_F16,  WRITE_F16,  0.0f)
DEFINE_JITSMM_GATHER(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, 0.0f)

// =========================================================================
// Scatter kernel (corder=false): one thread per input row
// For each input row j, atomicAdd w * B[j, col] to Y[connected_rows, col].
// Preloads w*B[j,:] before connectivity loop to avoid re-reading B.
// =========================================================================

#define DEFINE_JITSMM_SCATTER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ATOMIC_ADD)        \
__global__ void _jitsmm_scatter_kern##SUFFIX(                                              \
    const WEIGHT_T* __restrict__ weight,                                                   \
    const float*    __restrict__ clen,                                                     \
    const int*      __restrict__ seed,                                                     \
    const WEIGHT_T* __restrict__ B,                                                        \
    WEIGHT_T*       __restrict__ output,                                                   \
    int m, int k, int n                                                                    \
) {                                                                                        \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                                         \
    if (j >= k) return;                                                                    \
    ACC_T w0 = READ_W(__ldg(&weight[0]));                                                  \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                       \
    if (cl < 2) cl = 2;                                                                    \
    curandStatePhilox4_32_10_t state;                                                      \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)j, 0ULL, &state); \
    unsigned int i = curand(&state) % cl;                                                  \
    const WEIGHT_T* b_row = B + (size_t)j * n;                                             \
    while (i < (unsigned int)m) {                                                          \
        WEIGHT_T* out_row = output + (size_t)i * n;                                        \
        for (int col = 0; col < n; col++) {                                                \
            ACC_T val = w0 * READ_W(__ldg(&b_row[col]));                                   \
            ATOMIC_ADD(&out_row[col], val);                                                \
        }                                                                                  \
        i += 1 + (curand(&state) % (cl - 1));                                              \
    }                                                                                      \
}

DEFINE_JITSMM_SCATTER(_f32,  float,         float,  READ_F32,  WRITE_F32,  atomicAdd_f32)
DEFINE_JITSMM_SCATTER(_f64,  double,        double, READ_F64,  WRITE_F64,  atomicAdd_f64)
DEFINE_JITSMM_SCATTER(_f16,  __half,        float,  READ_F16,  WRITE_F16,  atomicAdd_f16)
DEFINE_JITSMM_SCATTER(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, atomicAdd_bf16)

// ---- TVM FFI: jitsmm gather ----
// Dispatches to register-accumulator kernel for n <= 16, fallback for n > 16.

#define FFI_JITSMM_GATHER(SUFFIX, WEIGHT_C_T)                          \
void jitsmm_gather##SUFFIX(                                            \
    tvm::ffi::TensorView weight,                                       \
    tvm::ffi::TensorView clen,                                         \
    tvm::ffi::TensorView seed,                                         \
    tvm::ffi::TensorView B,                                            \
    tvm::ffi::TensorView output,                                       \
    int64_t stream                                                     \
) {                                                                    \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);           \
    int m = static_cast<int>(output.size(0));                          \
    int n = static_cast<int>(output.size(1));                          \
    int k = static_cast<int>(B.size(0));                               \
    cudaMemsetAsync(output.data_ptr(), 0,                              \
        (size_t)m * n * sizeof(WEIGHT_C_T), s);                        \
    int threads = 256;                                                 \
    int blocks = (m + threads - 1) / threads;                          \
    if (n <= 16) {                                                     \
        /* Register accumulators: 1 thread/row, 256 rows/block */      \
        _jitsmm_gather_regacc_kern##SUFFIX<<<blocks, threads, 0, s>>>( \
            static_cast<const WEIGHT_C_T*>(weight.data_ptr()),         \
            static_cast<const float*>(clen.data_ptr()),                \
            static_cast<const int*>(seed.data_ptr()),                  \
            static_cast<const WEIGHT_C_T*>(B.data_ptr()),              \
            static_cast<WEIGHT_C_T*>(output.data_ptr()),               \
            m, k, n                                                    \
        );                                                             \
    } else {                                                           \
        /* Fallback: thread-per-row with global R/W */                 \
        _jitsmm_gather_kern##SUFFIX<<<blocks, threads, 0, s>>>(        \
            static_cast<const WEIGHT_C_T*>(weight.data_ptr()),         \
            static_cast<const float*>(clen.data_ptr()),                \
            static_cast<const int*>(seed.data_ptr()),                  \
            static_cast<const WEIGHT_C_T*>(B.data_ptr()),              \
            static_cast<WEIGHT_C_T*>(output.data_ptr()),               \
            m, k, n                                                    \
        );                                                             \
    }                                                                  \
}

// @tvm_ffi jitsmm_gather_f32
FFI_JITSMM_GATHER(_f32, float)
// @tvm_ffi jitsmm_gather_f64
FFI_JITSMM_GATHER(_f64, double)
// @tvm_ffi jitsmm_gather_f16
FFI_JITSMM_GATHER(_f16, __half)
// @tvm_ffi jitsmm_gather_bf16
FFI_JITSMM_GATHER(_bf16, __nv_bfloat16)

// ---- TVM FFI: jitsmm scatter ----

#define FFI_JITSMM_SCATTER(SUFFIX, WEIGHT_C_T)               \
void jitsmm_scatter##SUFFIX(                                 \
    tvm::ffi::TensorView weight,                             \
    tvm::ffi::TensorView clen,                               \
    tvm::ffi::TensorView seed,                               \
    tvm::ffi::TensorView B,                                  \
    tvm::ffi::TensorView output,                             \
    int64_t stream                                           \
) {                                                          \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream); \
    int m = static_cast<int>(output.size(0));                \
    int n = static_cast<int>(output.size(1));                \
    int k = static_cast<int>(B.size(0));                     \
    cudaMemsetAsync(output.data_ptr(), 0,                    \
        (size_t)m * n * sizeof(WEIGHT_C_T), s);              \
    int threads = 256;                                       \
    int blocks = (k + threads - 1) / threads;                \
    _jitsmm_scatter_kern##SUFFIX<<<blocks, threads, 0, s>>>( \
        static_cast<const WEIGHT_C_T*>(weight.data_ptr()),   \
        static_cast<const float*>(clen.data_ptr()),          \
        static_cast<const int*>(seed.data_ptr()),            \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),        \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),         \
        m, k, n                                              \
    );                                                       \
}

// @tvm_ffi jitsmm_scatter_f32
FFI_JITSMM_SCATTER(_f32, float)
// @tvm_ffi jitsmm_scatter_f64
FFI_JITSMM_SCATTER(_f64, double)
// @tvm_ffi jitsmm_scatter_f16
FFI_JITSMM_SCATTER(_f16, __half)
// @tvm_ffi jitsmm_scatter_bf16
FFI_JITSMM_SCATTER(_bf16, __nv_bfloat16)
