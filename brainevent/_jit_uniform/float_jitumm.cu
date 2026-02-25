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
 * float_jitumm.cu — JIT Uniform Float Matrix-Matrix Product (jitumm operator)
 * =============================================================================
 *
 * Computes Y = M @ B where M is a JIT-generated uniform random connectivity
 * matrix. Each entry M[i,j] is independently drawn from Uniform(w_low, w_high)
 * with probability prob. Connectivity pattern uses a geometric skip seeded by `seed`.
 *
 * Operations
 * ----------
 * jitumm_gather_{f32,f64,f16,bf16}  — corder=True  (gather, one thread per output row)
 * jitumm_scatter_{f32,f64,f16,bf16} — corder=False (scatter, one thread per input row)
 *
 * Parameters
 * ----------
 * w_low   : shape (1,), lower bound of uniform weight distribution
 * w_high  : shape (1,), upper bound of uniform weight distribution
 * clen    : shape (1,), connection length = ceil(2/prob) (float32)
 * seed    : shape (1,), int32 random seed
 * B       : shape (k, n), input matrix
 * output  : shape (m, n), output matrix
 *
 * Gather kernels zero-initialize output rows in-kernel — no memset needed.
 * Scatter kernels use atomicAdd — output is zeroed via cudaMemsetAsync first.
 *
 * IMPORTANT: All data_ptr() returns are GPU device pointers — NEVER dereference on host.
 */

#include "cuda_common.h"
#include "brainevent/common.h"
#include "curand_common.h"

// #########################################################################
// ##  jitumm — Float Matrix-Matrix Product                               ##
// #########################################################################

// =========================================================================
// Gather kernel (corder=true): one thread per output row i
// Y[i, col] = sum_{j in C(i)} Uniform(w_low, w_high) * B[j, col]
// Each connection j gets a fresh weight sample (same w for all cols of B).
// For n <= 32: uses register accumulators (ACC_T acc[32]) to avoid
// read-modify-write to global memory on every connection. Writes output
// once at the end. For n > 32: falls back to in-kernel zero-init +
// global memory accumulation (avoids register spill pressure).
// =========================================================================

#define DEFINE_JITUMM_GATHER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO)           \
__global__ void _jitumm_gather_kern##SUFFIX(                                               \
    const WEIGHT_T* __restrict__ w_low,                                                    \
    const WEIGHT_T* __restrict__ w_high,                                                   \
    const float*    __restrict__ clen,                                                     \
    const int*      __restrict__ seed,                                                     \
    const WEIGHT_T* __restrict__ B,                                                        \
    WEIGHT_T*       __restrict__ output,                                                   \
    int m, int k, int n                                                                    \
) {                                                                                        \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                         \
    if (i >= m) return;                                                                    \
    ACC_T wlo = READ_W(__ldg(&w_low[0]));                                                  \
    ACC_T range = READ_W(__ldg(&w_high[0])) - wlo;                                         \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                       \
    if (cl < 2) cl = 2;                                                                    \
    curandStatePhilox4_32_10_t state;                                                      \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)i, 0ULL, &state); \
    unsigned int j = curand(&state) % cl;                                                  \
    WEIGHT_T* out_row = output + (size_t)i * n;                                            \
    if (n <= 32) {                                                                         \
        ACC_T acc[32];                                                                     \
        for (int col = 0; col < n; col++) acc[col] = ACC_ZERO;                             \
        while (j < (unsigned int)k) {                                                      \
            float u = curand_uniform(&state);                                              \
            ACC_T w = wlo + (ACC_T)u * range;                                              \
            const WEIGHT_T* b_row = B + (size_t)j * n;                                     \
            for (int col = 0; col < n; col++) {                                            \
                acc[col] += READ_W(__ldg(&b_row[col])) * w;                                \
            }                                                                              \
            j += 1 + (curand(&state) % (cl - 1));                                          \
        }                                                                                  \
        for (int col = 0; col < n; col++) {                                                \
            out_row[col] = WRITE_W(acc[col]);                                              \
        }                                                                                  \
    } else {                                                                               \
        for (int col = 0; col < n; col++) {                                                \
            out_row[col] = WRITE_W(ACC_ZERO);                                              \
        }                                                                                  \
        while (j < (unsigned int)k) {                                                      \
            float u = curand_uniform(&state);                                              \
            ACC_T w = wlo + (ACC_T)u * range;                                              \
            const WEIGHT_T* b_row = B + (size_t)j * n;                                     \
            for (int col = 0; col < n; col++) {                                            \
                ACC_T cur = READ_W(out_row[col]);                                          \
                out_row[col] = WRITE_W(cur + w * READ_W(__ldg(&b_row[col])));              \
            }                                                                              \
            j += 1 + (curand(&state) % (cl - 1));                                          \
        }                                                                                  \
    }                                                                                      \
}

DEFINE_JITUMM_GATHER(_f32,  float,         float,  READ_F32,  WRITE_F32,  0.0f)
DEFINE_JITUMM_GATHER(_f64,  double,        double, READ_F64,  WRITE_F64,  0.0)
DEFINE_JITUMM_GATHER(_f16,  __half,        float,  READ_F16,  WRITE_F16,  0.0f)
DEFINE_JITUMM_GATHER(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, 0.0f)

// =========================================================================
// Scatter kernel (corder=false): one thread per input row j
// For each row j of B, scatter Uniform(w)*B[j,col] to output[connected_i, col].
// =========================================================================

#define DEFINE_JITUMM_SCATTER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ATOMIC_ADD)        \
__global__ void _jitumm_scatter_kern##SUFFIX(                                              \
    const WEIGHT_T* __restrict__ w_low,                                                    \
    const WEIGHT_T* __restrict__ w_high,                                                   \
    const float*    __restrict__ clen,                                                     \
    const int*      __restrict__ seed,                                                     \
    const WEIGHT_T* __restrict__ B,                                                        \
    WEIGHT_T*       __restrict__ output,                                                   \
    int m, int k, int n                                                                    \
) {                                                                                        \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                                         \
    if (j >= k) return;                                                                    \
    ACC_T wlo = READ_W(__ldg(&w_low[0]));                                                  \
    ACC_T range = READ_W(__ldg(&w_high[0])) - wlo;                                         \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                       \
    if (cl < 2) cl = 2;                                                                    \
    curandStatePhilox4_32_10_t state;                                                      \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)j, 0ULL, &state); \
    unsigned int i = curand(&state) % cl;                                                  \
    const WEIGHT_T* b_row = B + (size_t)j * n;                                             \
    while (i < (unsigned int)m) {                                                          \
        float u = curand_uniform(&state);                                                  \
        ACC_T w = wlo + (ACC_T)u * range;                                                  \
        WEIGHT_T* out_row = output + (size_t)i * n;                                        \
        for (int col = 0; col < n; col++) {                                                \
            ACC_T val = w * READ_W(__ldg(&b_row[col]));                                    \
            ATOMIC_ADD(&out_row[col], val);                                                \
        }                                                                                  \
        i += 1 + (curand(&state) % (cl - 1));                                              \
    }                                                                                      \
}

DEFINE_JITUMM_SCATTER(_f32,  float,         float,  READ_F32,  WRITE_F32,  atomic_add_f32)
DEFINE_JITUMM_SCATTER(_f64,  double,        double, READ_F64,  WRITE_F64,  atomic_add_f64)
DEFINE_JITUMM_SCATTER(_f16,  __half,        float,  READ_F16,  WRITE_F16,  atomic_add_f16)
DEFINE_JITUMM_SCATTER(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, atomic_add_bf16)

// ---- CUDA: jitumm gather ----
// No memset needed: gather kernel zero-initializes output rows in-kernel.

#define FFI_JITUMM_GATHER(SUFFIX, WEIGHT_C_T)                \
void jitumm_gather##SUFFIX(                                  \
    const BE::Tensor w_low,                              \
    const BE::Tensor w_high,                             \
    const BE::Tensor clen,                               \
    const BE::Tensor seed,                               \
    const BE::Tensor B,                                  \
    BE::Tensor output,                             \
    int64_t stream                                           \
) {                                                          \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream); \
    int m = static_cast<int>(output.size(0));                \
    int n = static_cast<int>(output.size(1));                \
    int k = static_cast<int>(B.size(0));                     \
    int threads = 256;                                       \
    int blocks = (m + threads - 1) / threads;                \
    _jitumm_gather_kern##SUFFIX<<<blocks, threads, 0, s>>>(  \
        static_cast<const WEIGHT_C_T*>(w_low.data_ptr()),    \
        static_cast<const WEIGHT_C_T*>(w_high.data_ptr()),   \
        static_cast<const float*>(clen.data_ptr()),          \
        static_cast<const int*>(seed.data_ptr()),            \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),        \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),         \
        m, k, n                                              \
    );                                                       \
}

// @BE jitumm_gather_f32
FFI_JITUMM_GATHER(_f32, float)
// @BE jitumm_gather_f64
FFI_JITUMM_GATHER(_f64, double)
// @BE jitumm_gather_f16
FFI_JITUMM_GATHER(_f16, __half)
// @BE jitumm_gather_bf16
FFI_JITUMM_GATHER(_bf16, __nv_bfloat16)

// ---- CUDA: jitumm scatter ----

#define FFI_JITUMM_SCATTER(SUFFIX, WEIGHT_C_T)               \
void jitumm_scatter##SUFFIX(                                 \
    const BE::Tensor w_low,                              \
    const BE::Tensor w_high,                             \
    const BE::Tensor clen,                               \
    const BE::Tensor seed,                               \
    const BE::Tensor B,                                  \
    BE::Tensor output,                             \
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
    _jitumm_scatter_kern##SUFFIX<<<blocks, threads, 0, s>>>( \
        static_cast<const WEIGHT_C_T*>(w_low.data_ptr()),    \
        static_cast<const WEIGHT_C_T*>(w_high.data_ptr()),   \
        static_cast<const float*>(clen.data_ptr()),          \
        static_cast<const int*>(seed.data_ptr()),            \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),        \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),         \
        m, k, n                                              \
    );                                                       \
}

// @BE jitumm_scatter_f32
FFI_JITUMM_SCATTER(_f32, float)
// @BE jitumm_scatter_f64
FFI_JITUMM_SCATTER(_f64, double)
// @BE jitumm_scatter_f16
FFI_JITUMM_SCATTER(_f16, __half)
// @BE jitumm_scatter_bf16
FFI_JITUMM_SCATTER(_bf16, __nv_bfloat16)
