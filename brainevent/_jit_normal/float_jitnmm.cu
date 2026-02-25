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
 * float_jitnmm.cu — JIT Normal Float Matrix-Matrix Product Kernels
 * =================================================================
 *
 * Computes Y = M @ B where M[i,j] = Normal(w_loc, w_scale) * Bernoulli(prob).
 * The matrix M is never materialised; weights are generated on the fly.
 *
 * TVM FFI entry points
 * --------------------
 * jitnmm_gather_{f32,f64,f16,bf16}  — gather (corder=True):  one thread per output row i
 * jitnmm_scatter_{f32,f64,f16,bf16} — scatter (corder=False): one thread per input row j
 *
 * Parameters (common)
 * -------------------
 * w_loc  : shape (1,), mean of normal weight distribution
 * w_scale: shape (1,), std dev of normal weight distribution
 * clen   : shape (1,), connection length = ceil(2/prob) (float32)
 * seed   : shape (1,), int32 random seed
 * B      : shape (k, n), right-hand matrix
 * output : shape (m, n), output matrix (zeroed before writing)
 *
 * Optimizations
 * -------------
 * - __ldg() on all read-only global memory (routes through L1 texture cache)
 * - Register accumulators for n <= 32 in gather kernels (eliminates global R/W)
 *
 * IMPORTANT: All data_ptr() returns are GPU device pointers — NEVER dereference on host.
 */

#include "cuda_common.h"
#include "brainevent/common.h"
#include "curand_common.h"

// #########################################################################
// ##  jitnmm — Float Matrix-Matrix Product                               ##
// #########################################################################

// --- Gather kernel with register accumulators (n<=32) and __ldg ---
#define DEFINE_JITNMM_GATHER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, RNG_FUNC, ACC_ZERO) \
__global__ void _jitnmm_gather_kern##SUFFIX(                                               \
    const WEIGHT_T* __restrict__ w_loc,                                                    \
    const WEIGHT_T* __restrict__ w_scale,                                                  \
    const float*    __restrict__ clen,                                                     \
    const int*      __restrict__ seed,                                                     \
    const WEIGHT_T* __restrict__ B,                                                        \
    WEIGHT_T*       __restrict__ output,                                                   \
    int m, int k, int n                                                                    \
) {                                                                                        \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                         \
    if (i >= m) return;                                                                    \
    ACC_T loc = READ_W(__ldg(&w_loc[0]));                                                  \
    ACC_T scale = READ_W(__ldg(&w_scale[0]));                                              \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                       \
    if (cl < 2) cl = 2;                                                                    \
    curandStatePhilox4_32_10_t state;                                                      \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)i, 0ULL, &state); \
    unsigned int j = curand(&state) % cl;                                                  \
    WEIGHT_T* out_row = output + (size_t)i * n;                                            \
    if (n <= 32) {                                                                         \
        ACC_T acc[32];                                                                     \
        for (int c = 0; c < n; c++) acc[c] = ACC_ZERO;                                     \
        while (j < (unsigned int)k) {                                                      \
            ACC_T num = (ACC_T)RNG_FUNC(&state);                                           \
            ACC_T w = loc + num * scale;                                                   \
            const WEIGHT_T* b_row = B + (size_t)j * n;                                     \
            for (int col = 0; col < n; col++) {                                            \
                acc[col] += w * READ_W(__ldg(&b_row[col]));                                \
            }                                                                              \
            j += 1 + (curand(&state) % (cl - 1));                                          \
        }                                                                                  \
        for (int c = 0; c < n; c++) out_row[c] = WRITE_W(acc[c]);                          \
    } else {                                                                               \
        while (j < (unsigned int)k) {                                                      \
            ACC_T num = (ACC_T)RNG_FUNC(&state);                                           \
            ACC_T w = loc + num * scale;                                                   \
            const WEIGHT_T* b_row = B + (size_t)j * n;                                     \
            for (int col = 0; col < n; col++) {                                            \
                ACC_T cur = READ_W(out_row[col]);                                          \
                out_row[col] = WRITE_W(cur + w * READ_W(__ldg(&b_row[col])));              \
            }                                                                              \
            j += 1 + (curand(&state) % (cl - 1));                                          \
        }                                                                                  \
    }                                                                                      \
}

DEFINE_JITNMM_GATHER(_f32,  float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, 0.0f)
DEFINE_JITNMM_GATHER(_f64,  double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, 0.0)
DEFINE_JITNMM_GATHER(_f16,  __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, 0.0f)
DEFINE_JITNMM_GATHER(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, 0.0f)

// --- Scatter kernel: __ldg on B reads ---
#define DEFINE_JITNMM_SCATTER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, RNG_FUNC, ATOMIC_ADD) \
__global__ void _jitnmm_scatter_kern##SUFFIX(                                                 \
    const WEIGHT_T* __restrict__ w_loc,                                                       \
    const WEIGHT_T* __restrict__ w_scale,                                                     \
    const float*    __restrict__ clen,                                                        \
    const int*      __restrict__ seed,                                                        \
    const WEIGHT_T* __restrict__ B,                                                           \
    WEIGHT_T*       __restrict__ output,                                                      \
    int m, int k, int n                                                                       \
) {                                                                                           \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                                            \
    if (j >= k) return;                                                                       \
    ACC_T loc = READ_W(__ldg(&w_loc[0]));                                                     \
    ACC_T scale = READ_W(__ldg(&w_scale[0]));                                                 \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                          \
    if (cl < 2) cl = 2;                                                                       \
    curandStatePhilox4_32_10_t state;                                                         \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)j, 0ULL, &state);    \
    unsigned int i = curand(&state) % cl;                                                     \
    const WEIGHT_T* b_row = B + (size_t)j * n;                                                \
    while (i < (unsigned int)m) {                                                             \
        ACC_T num = (ACC_T)RNG_FUNC(&state);                                                  \
        ACC_T w = loc + num * scale;                                                          \
        WEIGHT_T* out_row = output + (size_t)i * n;                                           \
        for (int col = 0; col < n; col++) {                                                   \
            ACC_T val = w * READ_W(__ldg(&b_row[col]));                                       \
            ATOMIC_ADD(&out_row[col], val);                                                   \
        }                                                                                     \
        i += 1 + (curand(&state) % (cl - 1));                                                 \
    }                                                                                         \
}

DEFINE_JITNMM_SCATTER(_f32,  float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, atomic_add_f32)
DEFINE_JITNMM_SCATTER(_f64,  double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, atomic_add_f64)
DEFINE_JITNMM_SCATTER(_f16,  __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, atomic_add_f16)
DEFINE_JITNMM_SCATTER(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, atomic_add_bf16)

#define FFI_JITNMM_GATHER(SUFFIX, WEIGHT_C_T)                \
void jitnmm_gather##SUFFIX(                                  \
    const BE::Tensor w_loc,                              \
    const BE::Tensor w_scale,                            \
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
    int blocks = (m + threads - 1) / threads;                \
    _jitnmm_gather_kern##SUFFIX<<<blocks, threads, 0, s>>>(  \
        static_cast<const WEIGHT_C_T*>(w_loc.data_ptr()),    \
        static_cast<const WEIGHT_C_T*>(w_scale.data_ptr()),  \
        static_cast<const float*>(clen.data_ptr()),          \
        static_cast<const int*>(seed.data_ptr()),            \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),        \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),         \
        m, k, n                                              \
    );                                                       \
}

// @BE jitnmm_gather_f32
FFI_JITNMM_GATHER(_f32, float)
// @BE jitnmm_gather_f64
FFI_JITNMM_GATHER(_f64, double)
// @BE jitnmm_gather_f16
FFI_JITNMM_GATHER(_f16, __half)
// @BE jitnmm_gather_bf16
FFI_JITNMM_GATHER(_bf16, __nv_bfloat16)

#define FFI_JITNMM_SCATTER(SUFFIX, WEIGHT_C_T)               \
void jitnmm_scatter##SUFFIX(                                 \
    const BE::Tensor w_loc,                              \
    const BE::Tensor w_scale,                            \
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
    _jitnmm_scatter_kern##SUFFIX<<<blocks, threads, 0, s>>>( \
        static_cast<const WEIGHT_C_T*>(w_loc.data_ptr()),    \
        static_cast<const WEIGHT_C_T*>(w_scale.data_ptr()),  \
        static_cast<const float*>(clen.data_ptr()),          \
        static_cast<const int*>(seed.data_ptr()),            \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),        \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),         \
        m, k, n                                              \
    );                                                       \
}

// @BE jitnmm_scatter_f32
FFI_JITNMM_SCATTER(_f32, float)
// @BE jitnmm_scatter_f64
FFI_JITNMM_SCATTER(_f64, double)
// @BE jitnmm_scatter_f16
FFI_JITNMM_SCATTER(_f16, __half)
// @BE jitnmm_scatter_bf16
FFI_JITNMM_SCATTER(_bf16, __nv_bfloat16)
