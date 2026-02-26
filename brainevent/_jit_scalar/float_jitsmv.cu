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
 * float_jitsmv.cu — JIT Scalar Float Matrix-Vector Product CUDA Kernels
 * ======================================================================
 *
 * Float matrix-vector product for JIT scalar connectivity.
 *
 * Operation
 * ---------
 * jitsmv — Float matrix-vector: y = M @ v
 *   where M[i,j] = w * Bernoulli(prob) is generated on-the-fly.
 *
 * Parameters
 * ----------
 * weight : shape (1,), scalar weight for all connections
 * clen   : shape (1,), connection length = 2/prob (float32)
 * seed   : shape (1,), int32 random seed
 * vector : shape (k,), input vector
 * output : shape (m,), output vector
 *
 * corder=True  (gather): one thread per output element, no atomics
 *   y[i] = w * sum_{j in C(i)} v[j]
 * corder=False (scatter): one thread per input element, uses atomicAdd
 *   For each j, atomicAdd w * v[j] to output[connected_indices]
 *
 * Supported weight dtypes: float32, float64, float16, bfloat16.
 *
 * Performance notes (RTX 3080 Ti Laptop, SM 8.6):
 *   Gather 10Kx10K p=0.1: ~1.3ms min. Curand Philox compute-bound.
 *   Scatter 10Kx10K p=0.1: ~1.3ms min. Curand + atomicAdd.
 *
 * IMPORTANT: All data_ptr() returns are GPU device pointers — NEVER dereference on host.
 */

#include "cuda_common.h"
#include "brainevent/common.h"
#include "curand_common.h"

// #########################################################################
// ##  jitsmv — Float Matrix-Vector Product                               ##
// #########################################################################

// =========================================================================
// Gather kernel (corder=true): one thread per output element
// y[i] = w * sum_{j in C(i)} v[j]
// =========================================================================

#define DEFINE_JITSMV_GATHER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO)           \
__global__ void _jitsmv_gather_kern##SUFFIX(                                               \
    const WEIGHT_T* __restrict__ weight,                                                   \
    const float*    __restrict__ clen,                                                     \
    const int*      __restrict__ seed,                                                     \
    const WEIGHT_T* __restrict__ vector,                                                   \
    WEIGHT_T*       __restrict__ output,                                                   \
    int m, int k                                                                           \
) {                                                                                        \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                         \
    if (i >= m) return;                                                                    \
    ACC_T w0 = READ_W(__ldg(&weight[0]));                                                  \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                       \
    if (cl < 2) cl = 2;                                                                    \
    curandStatePhilox4_32_10_t state;                                                      \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)i, 0ULL, &state); \
    unsigned int j = curand(&state) % cl;                                                  \
    ACC_T acc = ACC_ZERO;                                                                  \
    while (j < (unsigned int)k) {                                                          \
        acc += READ_W(__ldg(&vector[j]));                                                  \
        j += 1 + (curand(&state) % (cl - 1));                                              \
    }                                                                                      \
    output[i] = WRITE_W(w0 * acc);                                                         \
}

DEFINE_JITSMV_GATHER(_f32,  float,         float,  READ_F32,  WRITE_F32,  0.0f)
DEFINE_JITSMV_GATHER(_f64,  double,        double, READ_F64,  WRITE_F64,  0.0)
DEFINE_JITSMV_GATHER(_f16,  __half,        float,  READ_F16,  WRITE_F16,  0.0f)
DEFINE_JITSMV_GATHER(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, 0.0f)

// =========================================================================
// Scatter kernel (corder=false): one thread per input element
// For each input j, atomicAdd w * v[j] to output[connected_indices].
// =========================================================================

#define DEFINE_JITSMV_SCATTER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ATOMIC_ADD)        \
__global__ void _jitsmv_scatter_kern##SUFFIX(                                              \
    const WEIGHT_T* __restrict__ weight,                                                   \
    const float*    __restrict__ clen,                                                     \
    const int*      __restrict__ seed,                                                     \
    const WEIGHT_T* __restrict__ vector,                                                   \
    WEIGHT_T*       __restrict__ output,                                                   \
    int m, int k                                                                           \
) {                                                                                        \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                                         \
    if (j >= k) return;                                                                    \
    ACC_T w0 = READ_W(__ldg(&weight[0]));                                                  \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                       \
    if (cl < 2) cl = 2;                                                                    \
    ACC_T val = w0 * READ_W(__ldg(&vector[j]));                                            \
    curandStatePhilox4_32_10_t state;                                                      \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)j, 0ULL, &state); \
    unsigned int i = curand(&state) % cl;                                                  \
    while (i < (unsigned int)m) {                                                          \
        ATOMIC_ADD(&output[i], val);                                                       \
        i += 1 + (curand(&state) % (cl - 1));                                              \
    }                                                                                      \
}

DEFINE_JITSMV_SCATTER(_f32,  float,         float,  READ_F32,  WRITE_F32,  atomic_add_f32)
DEFINE_JITSMV_SCATTER(_f64,  double,        double, READ_F64,  WRITE_F64,  atomic_add_f64)
DEFINE_JITSMV_SCATTER(_f16,  __half,        float,  READ_F16,  WRITE_F16,  atomic_add_f16)
DEFINE_JITSMV_SCATTER(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, atomic_add_bf16)

// ---- CUDA: jitsmv gather ----

#define FFI_JITSMV_GATHER(SUFFIX, WEIGHT_C_T)                \
void jitsmv_gather##SUFFIX(                                  \
    const BE::Tensor weight,                                 \
    const BE::Tensor clen,                                   \
    const BE::Tensor seed,                                   \
    const BE::Tensor vector,                                 \
    BE::Tensor output,                                       \
    int64_t stream                                           \
) {                                                          \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream); \
    int m = static_cast<int>(output.size(0));                \
    int k = static_cast<int>(vector.size(0));                \
    cudaMemsetAsync(output.data_ptr(), 0,                    \
        (size_t)m * sizeof(WEIGHT_C_T), s);                  \
    int threads = 256;                                       \
    int blocks = (m + threads - 1) / threads;                \
    _jitsmv_gather_kern##SUFFIX<<<blocks, threads, 0, s>>>(  \
        static_cast<const WEIGHT_C_T*>(weight.data_ptr()),   \
        static_cast<const float*>(clen.data_ptr()),          \
        static_cast<const int*>(seed.data_ptr()),            \
        static_cast<const WEIGHT_C_T*>(vector.data_ptr()),   \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),         \
        m, k                                                 \
    );                                                       \
}

// @BE jitsmv_gather_f32
FFI_JITSMV_GATHER(_f32, float)
// @BE jitsmv_gather_f64
FFI_JITSMV_GATHER(_f64, double)
// @BE jitsmv_gather_f16
FFI_JITSMV_GATHER(_f16, __half)
// @BE jitsmv_gather_bf16
FFI_JITSMV_GATHER(_bf16, __nv_bfloat16)

// ---- CUDA: jitsmv scatter ----

#define FFI_JITSMV_SCATTER(SUFFIX, WEIGHT_C_T)               \
void jitsmv_scatter##SUFFIX(                                 \
    const BE::Tensor weight,                                 \
    const BE::Tensor clen,                                   \
    const BE::Tensor seed,                                   \
    const BE::Tensor vector,                                 \
    BE::Tensor output,                                       \
    int64_t stream                                           \
) {                                                          \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream); \
    int m = static_cast<int>(output.size(0));                \
    int k = static_cast<int>(vector.size(0));                \
    cudaMemsetAsync(output.data_ptr(), 0,                    \
        (size_t)m * sizeof(WEIGHT_C_T), s);                  \
    int threads = 256;                                       \
    int blocks = (k + threads - 1) / threads;                \
    _jitsmv_scatter_kern##SUFFIX<<<blocks, threads, 0, s>>>( \
        static_cast<const WEIGHT_C_T*>(weight.data_ptr()),   \
        static_cast<const float*>(clen.data_ptr()),          \
        static_cast<const int*>(seed.data_ptr()),            \
        static_cast<const WEIGHT_C_T*>(vector.data_ptr()),   \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),         \
        m, k                                                 \
    );                                                       \
}

// @BE jitsmv_scatter_f32
FFI_JITSMV_SCATTER(_f32, float)
// @BE jitsmv_scatter_f64
FFI_JITSMV_SCATTER(_f64, double)
// @BE jitsmv_scatter_f16
FFI_JITSMV_SCATTER(_f16, __half)
// @BE jitsmv_scatter_bf16
FFI_JITSMV_SCATTER(_bf16, __nv_bfloat16)
