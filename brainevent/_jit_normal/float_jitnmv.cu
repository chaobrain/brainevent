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
 * float_jitnmv.cu — JIT Normal Float Matrix-Vector Product Kernels
 * =================================================================
 *
 * Computes y = M @ v where M[i,j] = Normal(w_loc, w_scale) * Bernoulli(prob).
 * The matrix M is never materialised; weights are generated on the fly.
 *
 * TVM FFI entry points
 * --------------------
 * jitnmv_gather_{f32,f64,f16,bf16}  — gather (corder=True):  one thread per output row
 * jitnmv_scatter_{f32,f64,f16,bf16} — scatter (corder=False): one thread per input col
 *
 * Parameters (common)
 * -------------------
 * w_loc  : shape (1,), mean of normal weight distribution
 * w_scale: shape (1,), std dev of normal weight distribution
 * clen   : shape (1,), connection length = ceil(2/prob) (float32)
 * seed   : shape (1,), int32 random seed
 * vector : shape (k,), input vector
 * output : shape (m,), output vector (zeroed before writing)
 *
 * Optimizations
 * -------------
 * - __ldg() on all read-only global memory (routes through L1 texture cache)
 * - Shared memory vector caching for small vectors that fit in device shared memory
 *
 * IMPORTANT: All data_ptr() returns are GPU device pointers — NEVER dereference on host.
 */

#include "cuda_common.h"
#include "brainevent/common.h"
#include "curand_common.h"

// #########################################################################
// ##  jitnmv — Float Matrix-Vector Product                               ##
// #########################################################################

// --- Gather kernel: __ldg on vector reads ---
#define DEFINE_JITNMV_GATHER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, RNG_FUNC, ACC_ZERO) \
__global__ void _jitnmv_gather_kern##SUFFIX(                                               \
    const WEIGHT_T* __restrict__ w_loc,                                                    \
    const WEIGHT_T* __restrict__ w_scale,                                                  \
    const float*    __restrict__ clen,                                                     \
    const int*      __restrict__ seed,                                                     \
    const WEIGHT_T* __restrict__ vector,                                                   \
    WEIGHT_T*       __restrict__ output,                                                   \
    int m, int k                                                                           \
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
    ACC_T acc = ACC_ZERO;                                                                  \
    while (j < (unsigned int)k) {                                                          \
        ACC_T n = (ACC_T)RNG_FUNC(&state);                                                 \
        ACC_T w = loc + n * scale;                                                         \
        acc += READ_W(__ldg(&vector[j])) * w;                                              \
        j += 1 + (curand(&state) % (cl - 1));                                              \
    }                                                                                      \
    output[i] = WRITE_W(acc);                                                              \
}

DEFINE_JITNMV_GATHER(_f32,  float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, 0.0f)
DEFINE_JITNMV_GATHER(_f64,  double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, 0.0)
DEFINE_JITNMV_GATHER(_f16,  __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, 0.0f)
DEFINE_JITNMV_GATHER(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, 0.0f)

// --- Gather kernel with shared memory vector caching ---
#define DEFINE_JITNMV_GATHER_SMEM(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, RNG_FUNC, ACC_ZERO) \
__global__ void _jitnmv_gather_smem_kern##SUFFIX(                                               \
    const WEIGHT_T* __restrict__ w_loc,                                                         \
    const WEIGHT_T* __restrict__ w_scale,                                                       \
    const float*    __restrict__ clen,                                                          \
    const int*      __restrict__ seed,                                                          \
    const WEIGHT_T* __restrict__ vector,                                                        \
    WEIGHT_T*       __restrict__ output,                                                        \
    int m, int k                                                                                \
) {                                                                                             \
    extern __shared__ char _smem_bytes[];                                                       \
    ACC_T* sv = reinterpret_cast<ACC_T*>(_smem_bytes);                                          \
    for (int idx = threadIdx.x; idx < k; idx += blockDim.x) {                                   \
        sv[idx] = READ_W(__ldg(&vector[idx]));                                                  \
    }                                                                                           \
    __syncthreads();                                                                            \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                              \
    if (i >= m) return;                                                                         \
    ACC_T loc = READ_W(__ldg(&w_loc[0]));                                                       \
    ACC_T scale = READ_W(__ldg(&w_scale[0]));                                                   \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                            \
    if (cl < 2) cl = 2;                                                                         \
    curandStatePhilox4_32_10_t state;                                                           \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)i, 0ULL, &state);      \
    unsigned int j = curand(&state) % cl;                                                       \
    ACC_T acc = ACC_ZERO;                                                                       \
    while (j < (unsigned int)k) {                                                               \
        ACC_T n = (ACC_T)RNG_FUNC(&state);                                                      \
        ACC_T w = loc + n * scale;                                                              \
        acc += sv[j] * w;                                                                       \
        j += 1 + (curand(&state) % (cl - 1));                                                   \
    }                                                                                           \
    output[i] = WRITE_W(acc);                                                                   \
}

DEFINE_JITNMV_GATHER_SMEM(_f32,  float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, 0.0f)
DEFINE_JITNMV_GATHER_SMEM(_f64,  double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, 0.0)
DEFINE_JITNMV_GATHER_SMEM(_f16,  __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, 0.0f)
DEFINE_JITNMV_GATHER_SMEM(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, 0.0f)

// --- Scatter kernel: __ldg on vector read ---
#define DEFINE_JITNMV_SCATTER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, RNG_FUNC, ATOMIC_ADD) \
__global__ void _jitnmv_scatter_kern##SUFFIX(                                                 \
    const WEIGHT_T* __restrict__ w_loc,                                                       \
    const WEIGHT_T* __restrict__ w_scale,                                                     \
    const float*    __restrict__ clen,                                                        \
    const int*      __restrict__ seed,                                                        \
    const WEIGHT_T* __restrict__ vector,                                                      \
    WEIGHT_T*       __restrict__ output,                                                      \
    int m, int k                                                                              \
) {                                                                                           \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                                            \
    if (j >= k) return;                                                                       \
    ACC_T loc = READ_W(__ldg(&w_loc[0]));                                                     \
    ACC_T scale = READ_W(__ldg(&w_scale[0]));                                                 \
    ACC_T vj = READ_W(__ldg(&vector[j]));                                                     \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                          \
    if (cl < 2) cl = 2;                                                                       \
    curandStatePhilox4_32_10_t state;                                                         \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)j, 0ULL, &state);    \
    unsigned int i = curand(&state) % cl;                                                     \
    while (i < (unsigned int)m) {                                                             \
        ACC_T n = (ACC_T)RNG_FUNC(&state);                                                    \
        ACC_T w = loc + n * scale;                                                            \
        ATOMIC_ADD(&output[i], w * vj);                                                       \
        i += 1 + (curand(&state) % (cl - 1));                                                 \
    }                                                                                         \
}

DEFINE_JITNMV_SCATTER(_f32,  float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, atomic_add_f32)
DEFINE_JITNMV_SCATTER(_f64,  double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, atomic_add_f64)
DEFINE_JITNMV_SCATTER(_f16,  __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, atomic_add_f16)
DEFINE_JITNMV_SCATTER(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, atomic_add_bf16)

// --- FFI gather: dispatch to smem or global kernel ---
#define FFI_JITNMV_GATHER(SUFFIX, WEIGHT_C_T, ACC_SIZEOF)                     \
void jitnmv_gather##SUFFIX(                                                   \
    const BE::Tensor w_loc,                                               \
    const BE::Tensor w_scale,                                             \
    const BE::Tensor clen,                                                \
    const BE::Tensor seed,                                                \
    const BE::Tensor vector,                                              \
    BE::Tensor output,                                              \
    int64_t stream                                                            \
) {                                                                           \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                  \
    int m = static_cast<int>(output.size(0));                                 \
    int k = static_cast<int>(vector.size(0));                                 \
    cudaMemsetAsync(output.data_ptr(), 0,                                     \
        (size_t)m * sizeof(WEIGHT_C_T), s);                                   \
    int threads = 256;                                                        \
    int blocks = (m + threads - 1) / threads;                                 \
    size_t smem_bytes = (size_t)k * ACC_SIZEOF;                               \
    int _dev = 0; cudaGetDevice(&_dev);                                       \
    int _max_smem = 0;                                                        \
    cudaDeviceGetAttribute(&_max_smem,                                        \
        cudaDevAttrMaxSharedMemoryPerBlock, _dev);                             \
    if (smem_bytes <= (size_t)_max_smem) {                                    \
        _jitnmv_gather_smem_kern##SUFFIX<<<blocks, threads, smem_bytes, s>>>( \
            static_cast<const WEIGHT_C_T*>(w_loc.data_ptr()),                 \
            static_cast<const WEIGHT_C_T*>(w_scale.data_ptr()),               \
            static_cast<const float*>(clen.data_ptr()),                       \
            static_cast<const int*>(seed.data_ptr()),                         \
            static_cast<const WEIGHT_C_T*>(vector.data_ptr()),                \
            static_cast<WEIGHT_C_T*>(output.data_ptr()),                      \
            m, k                                                              \
        );                                                                    \
    } else {                                                                  \
        _jitnmv_gather_kern##SUFFIX<<<blocks, threads, 0, s>>>(               \
            static_cast<const WEIGHT_C_T*>(w_loc.data_ptr()),                 \
            static_cast<const WEIGHT_C_T*>(w_scale.data_ptr()),               \
            static_cast<const float*>(clen.data_ptr()),                       \
            static_cast<const int*>(seed.data_ptr()),                         \
            static_cast<const WEIGHT_C_T*>(vector.data_ptr()),                \
            static_cast<WEIGHT_C_T*>(output.data_ptr()),                      \
            m, k                                                              \
        );                                                                    \
    }                                                                         \
}

// @BE jitnmv_gather_f32
FFI_JITNMV_GATHER(_f32, float, sizeof(float))
// @BE jitnmv_gather_f64
FFI_JITNMV_GATHER(_f64, double, sizeof(double))
// @BE jitnmv_gather_f16
FFI_JITNMV_GATHER(_f16, __half, sizeof(float))
// @BE jitnmv_gather_bf16
FFI_JITNMV_GATHER(_bf16, __nv_bfloat16, sizeof(float))

#define FFI_JITNMV_SCATTER(SUFFIX, WEIGHT_C_T)               \
void jitnmv_scatter##SUFFIX(                                 \
    const BE::Tensor w_loc,                              \
    const BE::Tensor w_scale,                            \
    const BE::Tensor clen,                               \
    const BE::Tensor seed,                               \
    const BE::Tensor vector,                             \
    BE::Tensor output,                             \
    int64_t stream                                           \
) {                                                          \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream); \
    int m = static_cast<int>(output.size(0));                \
    int k = static_cast<int>(vector.size(0));                \
    cudaMemsetAsync(output.data_ptr(), 0,                    \
        (size_t)m * sizeof(WEIGHT_C_T), s);                  \
    int threads = 256;                                       \
    int blocks = (k + threads - 1) / threads;                \
    _jitnmv_scatter_kern##SUFFIX<<<blocks, threads, 0, s>>>( \
        static_cast<const WEIGHT_C_T*>(w_loc.data_ptr()),    \
        static_cast<const WEIGHT_C_T*>(w_scale.data_ptr()),  \
        static_cast<const float*>(clen.data_ptr()),          \
        static_cast<const int*>(seed.data_ptr()),            \
        static_cast<const WEIGHT_C_T*>(vector.data_ptr()),   \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),         \
        m, k                                                 \
    );                                                       \
}

// @BE jitnmv_scatter_f32
FFI_JITNMV_SCATTER(_f32, float)
// @BE jitnmv_scatter_f64
FFI_JITNMV_SCATTER(_f64, double)
// @BE jitnmv_scatter_f16
FFI_JITNMV_SCATTER(_f16, __half)
// @BE jitnmv_scatter_bf16
FFI_JITNMV_SCATTER(_bf16, __nv_bfloat16)
