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
 * float_jitumv.cu — JIT Uniform Float Matrix-Vector Product (jitumv operator)
 * =============================================================================
 *
 * Computes y = M @ v where M is a JIT-generated uniform random connectivity
 * matrix. Each entry M[i,j] is independently drawn from Uniform(w_low, w_high)
 * with probability prob. Connectivity pattern uses a geometric skip seeded by `seed`.
 *
 * Operations
 * ----------
 * jitumv_gather_{f32,f64,f16,bf16}  — corder=True  (gather, one thread per output)
 * jitumv_scatter_{f32,f64,f16,bf16} — corder=False (scatter, one thread per input)
 *
 * Parameters
 * ----------
 * w_low   : shape (1,), lower bound of uniform weight distribution
 * w_high  : shape (1,), upper bound of uniform weight distribution
 * clen    : shape (1,), connection length = ceil(2/prob) (float32)
 * seed    : shape (1,), int32 random seed
 * vector  : shape (k,), input vector
 * output  : shape (m,), output vector
 *
 * Gather kernels write each output element exactly once — no memset needed.
 * Scatter kernels use atomicAdd — output is zeroed via cudaMemsetAsync first.
 *
 * IMPORTANT: All data_ptr() returns are GPU device pointers — NEVER dereference on host.
 */

#include "cuda_common.h"
#include "brainevent/common.h"
#include "curand_common.h"

// #########################################################################
// ##  jitumv — Float Matrix-Vector Product                               ##
// #########################################################################

// =========================================================================
// Gather kernel (corder=true): one thread per output element
// y[i] = sum_{j in C(i)} Uniform(w_low, w_high) * v[j]
// Each connection gets its own weight sample from curand_uniform.
// No memset needed: every output[i] is written exactly once (direct store).
// =========================================================================

#define DEFINE_JITUMV_GATHER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO)           \
__global__ void _jitumv_gather_kern##SUFFIX(                                               \
    const WEIGHT_T* __restrict__ w_low,                                                    \
    const WEIGHT_T* __restrict__ w_high,                                                   \
    const float*    __restrict__ clen,                                                     \
    const int*      __restrict__ seed,                                                     \
    const WEIGHT_T* __restrict__ vector,                                                   \
    WEIGHT_T*       __restrict__ output,                                                   \
    int m, int k                                                                           \
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
    ACC_T acc = ACC_ZERO;                                                                  \
    while (j < (unsigned int)k) {                                                          \
        float u = curand_uniform(&state);                                                  \
        ACC_T w = wlo + (ACC_T)u * range;                                                  \
        acc += READ_W(__ldg(&vector[j])) * w;                                              \
        j += 1 + (curand(&state) % (cl - 1));                                              \
    }                                                                                      \
    output[i] = WRITE_W(acc);                                                              \
}

DEFINE_JITUMV_GATHER(_f32,  float,         float,  READ_F32,  WRITE_F32,  0.0f)
DEFINE_JITUMV_GATHER(_f64,  double,        double, READ_F64,  WRITE_F64,  0.0)
DEFINE_JITUMV_GATHER(_f16,  __half,        float,  READ_F16,  WRITE_F16,  0.0f)
DEFINE_JITUMV_GATHER(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, 0.0f)

// =========================================================================
// Gather kernel with shared memory vector caching (corder=true)
// Cooperatively loads the input vector into shared memory, then all
// subsequent reads use smem (~20 cycle latency vs ~200 from L2).
// Used when k * sizeof(ACC_T) fits in device shared memory.
// =========================================================================

#define DEFINE_JITUMV_GATHER_SMEM(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO)      \
__global__ void _jitumv_gather_smem_kern##SUFFIX(                                          \
    const WEIGHT_T* __restrict__ w_low,                                                    \
    const WEIGHT_T* __restrict__ w_high,                                                   \
    const float*    __restrict__ clen,                                                     \
    const int*      __restrict__ seed,                                                     \
    const WEIGHT_T* __restrict__ vector,                                                   \
    WEIGHT_T*       __restrict__ output,                                                   \
    int m, int k                                                                           \
) {                                                                                        \
    extern __shared__ char _smem_bytes[];                                                  \
    ACC_T* sv = reinterpret_cast<ACC_T*>(_smem_bytes);                                     \
    for (int idx = threadIdx.x; idx < k; idx += blockDim.x) {                              \
        sv[idx] = READ_W(__ldg(&vector[idx]));                                             \
    }                                                                                      \
    __syncthreads();                                                                       \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                         \
    if (i >= m) return;                                                                    \
    ACC_T wlo = READ_W(__ldg(&w_low[0]));                                                  \
    ACC_T range = READ_W(__ldg(&w_high[0])) - wlo;                                         \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                       \
    if (cl < 2) cl = 2;                                                                    \
    curandStatePhilox4_32_10_t state;                                                      \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)i, 0ULL, &state); \
    unsigned int j = curand(&state) % cl;                                                  \
    ACC_T acc = ACC_ZERO;                                                                  \
    while (j < (unsigned int)k) {                                                          \
        float u = curand_uniform(&state);                                                  \
        ACC_T w = wlo + (ACC_T)u * range;                                                  \
        acc += sv[j] * w;                                                                  \
        j += 1 + (curand(&state) % (cl - 1));                                              \
    }                                                                                      \
    output[i] = WRITE_W(acc);                                                              \
}

DEFINE_JITUMV_GATHER_SMEM(_f32,  float,         float,  READ_F32,  WRITE_F32,  0.0f)
DEFINE_JITUMV_GATHER_SMEM(_f64,  double,        double, READ_F64,  WRITE_F64,  0.0)
DEFINE_JITUMV_GATHER_SMEM(_f16,  __half,        float,  READ_F16,  WRITE_F16,  0.0f)
DEFINE_JITUMV_GATHER_SMEM(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, 0.0f)

// =========================================================================
// Scatter kernel (corder=false): one thread per input element
// For each input j, scatter Uniform(w_low, w_high)*v[j] to output[connected_i].
// Uses atomicAdd to handle concurrent writes to the same output element.
// =========================================================================

#define DEFINE_JITUMV_SCATTER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ATOMIC_ADD)        \
__global__ void _jitumv_scatter_kern##SUFFIX(                                              \
    const WEIGHT_T* __restrict__ w_low,                                                    \
    const WEIGHT_T* __restrict__ w_high,                                                   \
    const float*    __restrict__ clen,                                                     \
    const int*      __restrict__ seed,                                                     \
    const WEIGHT_T* __restrict__ vector,                                                   \
    WEIGHT_T*       __restrict__ output,                                                   \
    int m, int k                                                                           \
) {                                                                                        \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                                         \
    if (j >= k) return;                                                                    \
    ACC_T wlo = READ_W(__ldg(&w_low[0]));                                                  \
    ACC_T range = READ_W(__ldg(&w_high[0])) - wlo;                                         \
    ACC_T vj = READ_W(__ldg(&vector[j]));                                                  \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                       \
    if (cl < 2) cl = 2;                                                                    \
    curandStatePhilox4_32_10_t state;                                                      \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)j, 0ULL, &state); \
    unsigned int i = curand(&state) % cl;                                                  \
    while (i < (unsigned int)m) {                                                          \
        float u = curand_uniform(&state);                                                  \
        ACC_T w = wlo + (ACC_T)u * range;                                                  \
        ATOMIC_ADD(&output[i], w * vj);                                                    \
        i += 1 + (curand(&state) % (cl - 1));                                              \
    }                                                                                      \
}

DEFINE_JITUMV_SCATTER(_f32,  float,         float,  READ_F32,  WRITE_F32,  atomic_add_f32)
DEFINE_JITUMV_SCATTER(_f64,  double,        double, READ_F64,  WRITE_F64,  atomic_add_f64)
DEFINE_JITUMV_SCATTER(_f16,  __half,        float,  READ_F16,  WRITE_F16,  atomic_add_f16)
DEFINE_JITUMV_SCATTER(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, atomic_add_bf16)

// ---- CUDA: jitumv gather ----
// Dispatches to shared-memory kernel when vector fits in device smem,
// falls back to global-memory kernel for larger vectors.
// No memset needed: gather kernels write every output element exactly once.

#define FFI_JITUMV_GATHER(SUFFIX, WEIGHT_C_T, ACC_C_T)                        \
void jitumv_gather##SUFFIX(                                                   \
    const BE::Tensor w_low,                                               \
    const BE::Tensor w_high,                                              \
    const BE::Tensor clen,                                                \
    const BE::Tensor seed,                                                \
    const BE::Tensor vector,                                              \
    BE::Tensor output,                                              \
    int64_t stream                                                            \
) {                                                                           \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                  \
    int m = static_cast<int>(output.size(0));                                 \
    int k = static_cast<int>(vector.size(0));                                 \
    int threads = 256;                                                        \
    int blocks = (m + threads - 1) / threads;                                 \
    size_t smem_bytes = (size_t)k * sizeof(ACC_C_T);                          \
    int _dev = 0; cudaGetDevice(&_dev);                                       \
    int _max_smem = 0;                                                        \
    cudaDeviceGetAttribute(&_max_smem,                                        \
        cudaDevAttrMaxSharedMemoryPerBlock, _dev);                             \
    if (smem_bytes <= (size_t)_max_smem) {                                    \
        _jitumv_gather_smem_kern##SUFFIX<<<blocks, threads, smem_bytes, s>>>( \
            static_cast<const WEIGHT_C_T*>(w_low.data_ptr()),                 \
            static_cast<const WEIGHT_C_T*>(w_high.data_ptr()),                \
            static_cast<const float*>(clen.data_ptr()),                       \
            static_cast<const int*>(seed.data_ptr()),                         \
            static_cast<const WEIGHT_C_T*>(vector.data_ptr()),                \
            static_cast<WEIGHT_C_T*>(output.data_ptr()),                      \
            m, k                                                              \
        );                                                                    \
    } else {                                                                  \
        _jitumv_gather_kern##SUFFIX<<<blocks, threads, 0, s>>>(               \
            static_cast<const WEIGHT_C_T*>(w_low.data_ptr()),                 \
            static_cast<const WEIGHT_C_T*>(w_high.data_ptr()),                \
            static_cast<const float*>(clen.data_ptr()),                       \
            static_cast<const int*>(seed.data_ptr()),                         \
            static_cast<const WEIGHT_C_T*>(vector.data_ptr()),                \
            static_cast<WEIGHT_C_T*>(output.data_ptr()),                      \
            m, k                                                              \
        );                                                                    \
    }                                                                         \
}

// @BE jitumv_gather_f32
FFI_JITUMV_GATHER(_f32, float, float)
// @BE jitumv_gather_f64
FFI_JITUMV_GATHER(_f64, double, double)
// @BE jitumv_gather_f16
FFI_JITUMV_GATHER(_f16, __half, float)
// @BE jitumv_gather_bf16
FFI_JITUMV_GATHER(_bf16, __nv_bfloat16, float)

// ---- CUDA: jitumv scatter ----

#define FFI_JITUMV_SCATTER(SUFFIX, WEIGHT_C_T)               \
void jitumv_scatter##SUFFIX(                                 \
    const BE::Tensor w_low,                              \
    const BE::Tensor w_high,                             \
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
    _jitumv_scatter_kern##SUFFIX<<<blocks, threads, 0, s>>>( \
        static_cast<const WEIGHT_C_T*>(w_low.data_ptr()),    \
        static_cast<const WEIGHT_C_T*>(w_high.data_ptr()),   \
        static_cast<const float*>(clen.data_ptr()),          \
        static_cast<const int*>(seed.data_ptr()),            \
        static_cast<const WEIGHT_C_T*>(vector.data_ptr()),   \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),         \
        m, k                                                 \
    );                                                       \
}

// @BE jitumv_scatter_f32
FFI_JITUMV_SCATTER(_f32, float)
// @BE jitumv_scatter_f64
FFI_JITUMV_SCATTER(_f64, double)
// @BE jitumv_scatter_f16
FFI_JITUMV_SCATTER(_f16, __half)
// @BE jitumv_scatter_bf16
FFI_JITUMV_SCATTER(_bf16, __nv_bfloat16)
