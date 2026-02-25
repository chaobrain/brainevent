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
 * binary_jitsmv.cu — JIT Scalar Event-Driven Matrix-Vector Product CUDA Kernels
 * ==============================================================================
 *
 * Event-driven (binary spike) matrix-vector product for JIT scalar connectivity.
 *
 * Operation
 * ---------
 * binary_jitsmv — Event-driven mat-vec: y[i] = w * count{j in C(i) : spike[j]}
 *   where M[i,j] = Bernoulli(prob) is generated on-the-fly,
 *   and only active (spiking) inputs are accumulated.
 *
 * Parameters
 * ----------
 * weight : shape (1,), scalar weight for all connections
 * clen   : shape (1,), connection length = 2/prob (float32)
 * seed   : shape (1,), int32 random seed
 * vector : shape (k,), spike input (bool as int8_t, or float)
 * output : shape (m,), output vector
 *
 * corder=True  (gather): one thread per output element, no atomics
 *   y[i] = w * count{j in C(i) : spike[j]}
 * corder=False (scatter): one thread per input element, skip inactive spikes
 *   For each active spike j, atomicAdd w to output[connected_indices]
 *
 * Supported weight dtypes: float32, float64, float16, bfloat16.
 * Supported spike types: bool (int8_t), float.
 *
 * IMPORTANT: All data_ptr() returns are GPU device pointers — NEVER dereference on host.
 */

#include "cuda_common.h"
#include "brainevent/common.h"
#include "curand_common.h"

// #########################################################################
// ##  binary_jitsmv — Event-Driven Matrix-Vector Product                 ##
// #########################################################################

// =========================================================================
// Gather kernel (corder=true): one thread per output element
// y[i] = w * count{j in C(i) : spike[j]}
// =========================================================================

#define DEFINE_BINARY_JITSMV_GATHER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, SPIKE_T, IS_ACTIVE, ACC_ZERO) \
__global__ void _binary_jitsmv_gather_kern##SUFFIX(                                                         \
    const WEIGHT_T* __restrict__ weight,                                                                    \
    const float*    __restrict__ clen,                                                                      \
    const int*      __restrict__ seed,                                                                      \
    const SPIKE_T*  __restrict__ vector,                                                                    \
    WEIGHT_T*       __restrict__ output,                                                                    \
    int m, int k                                                                                            \
) {                                                                                                         \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                                          \
    if (i >= m) return;                                                                                     \
    ACC_T w0 = READ_W(__ldg(&weight[0]));                                                                   \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                                        \
    if (cl < 2) cl = 2;                                                                                     \
    curandStatePhilox4_32_10_t state;                                                                       \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)i, 0ULL, &state);                  \
    unsigned int j = curand(&state) % cl;                                                                   \
    ACC_T acc = ACC_ZERO;                                                                                   \
    while (j < (unsigned int)k) {                                                                           \
        if (IS_ACTIVE(vector[j])) {                                                                         \
            acc += (ACC_T)1.0;                                                                              \
        }                                                                                                   \
        j += 1 + (curand(&state) % (cl - 1));                                                               \
    }                                                                                                       \
    output[i] = WRITE_W(w0 * acc);                                                                          \
}

// f32 weight + bool/float spikes
DEFINE_BINARY_JITSMV_GATHER(_f32_bool,  float,         float,  READ_F32,  WRITE_F32,  int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITSMV_GATHER(_f32_float, float,         float,  READ_F32,  WRITE_F32,  float,  IS_ACTIVE_FLOAT, 0.0f)
// f64 weight + bool/float spikes
DEFINE_BINARY_JITSMV_GATHER(_f64_bool,  double,        double, READ_F64,  WRITE_F64,  int8_t, IS_ACTIVE_BOOL,  0.0)
DEFINE_BINARY_JITSMV_GATHER(_f64_float, double,        double, READ_F64,  WRITE_F64,  float,  IS_ACTIVE_FLOAT, 0.0)
// f16 weight + bool/float spikes
DEFINE_BINARY_JITSMV_GATHER(_f16_bool,  __half,        float,  READ_F16,  WRITE_F16,  int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITSMV_GATHER(_f16_float, __half,        float,  READ_F16,  WRITE_F16,  float,  IS_ACTIVE_FLOAT, 0.0f)
// bf16 weight + bool/float spikes
DEFINE_BINARY_JITSMV_GATHER(_bf16_bool, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITSMV_GATHER(_bf16_float,__nv_bfloat16, float,  READ_BF16, WRITE_BF16, float,  IS_ACTIVE_FLOAT, 0.0f)

// =========================================================================
// Scatter kernel (corder=false): one thread per input element
// Skip inactive spikes entirely (zero-work optimization).
// =========================================================================

#define DEFINE_BINARY_JITSMV_SCATTER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, SPIKE_T, IS_ACTIVE, ATOMIC_ADD) \
__global__ void _binary_jitsmv_scatter_kern##SUFFIX(                                                           \
    const WEIGHT_T* __restrict__ weight,                                                                       \
    const float*    __restrict__ clen,                                                                         \
    const int*      __restrict__ seed,                                                                         \
    const SPIKE_T*  __restrict__ vector,                                                                       \
    WEIGHT_T*       __restrict__ output,                                                                       \
    int m, int k                                                                                               \
) {                                                                                                            \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                                                             \
    if (j >= k) return;                                                                                        \
    if (!IS_ACTIVE(vector[j])) return;                                                                         \
    ACC_T w0 = READ_W(__ldg(&weight[0]));                                                                      \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                                           \
    if (cl < 2) cl = 2;                                                                                        \
    curandStatePhilox4_32_10_t state;                                                                          \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)j, 0ULL, &state);                     \
    unsigned int i = curand(&state) % cl;                                                                      \
    while (i < (unsigned int)m) {                                                                              \
        ATOMIC_ADD(&output[i], w0);                                                                            \
        i += 1 + (curand(&state) % (cl - 1));                                                                  \
    }                                                                                                          \
}

// f32 weight + bool/float spikes
DEFINE_BINARY_JITSMV_SCATTER(_f32_bool,  float,         float,  READ_F32,  WRITE_F32,  int8_t, IS_ACTIVE_BOOL,  atomic_add_f32)
DEFINE_BINARY_JITSMV_SCATTER(_f32_float, float,         float,  READ_F32,  WRITE_F32,  float,  IS_ACTIVE_FLOAT, atomic_add_f32)
// f64 weight + bool/float spikes
DEFINE_BINARY_JITSMV_SCATTER(_f64_bool,  double,        double, READ_F64,  WRITE_F64,  int8_t, IS_ACTIVE_BOOL,  atomic_add_f64)
DEFINE_BINARY_JITSMV_SCATTER(_f64_float, double,        double, READ_F64,  WRITE_F64,  float,  IS_ACTIVE_FLOAT, atomic_add_f64)
// f16 weight + bool/float spikes
DEFINE_BINARY_JITSMV_SCATTER(_f16_bool,  __half,        float,  READ_F16,  WRITE_F16,  int8_t, IS_ACTIVE_BOOL,  atomic_add_f16)
DEFINE_BINARY_JITSMV_SCATTER(_f16_float, __half,        float,  READ_F16,  WRITE_F16,  float,  IS_ACTIVE_FLOAT, atomic_add_f16)
// bf16 weight + bool/float spikes
DEFINE_BINARY_JITSMV_SCATTER(_bf16_bool, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, int8_t, IS_ACTIVE_BOOL,  atomic_add_bf16)
DEFINE_BINARY_JITSMV_SCATTER(_bf16_float,__nv_bfloat16, float,  READ_BF16, WRITE_BF16, float,  IS_ACTIVE_FLOAT, atomic_add_bf16)

// ---- CUDA: binary_jitsmv gather ----

#define FFI_BINARY_JITSMV_GATHER(SUFFIX, WEIGHT_C_T, SPIKE_C_T)    \
void binary_jitsmv_gather##SUFFIX(                                 \
    const BE::Tensor weight,                                   \
    const BE::Tensor clen,                                     \
    const BE::Tensor seed,                                     \
    const BE::Tensor vector,                                   \
    BE::Tensor output,                                   \
    int64_t stream                                                 \
) {                                                                \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);       \
    int m = static_cast<int>(output.size(0));                      \
    int k = static_cast<int>(vector.size(0));                      \
    cudaMemsetAsync(output.data_ptr(), 0,                          \
        (size_t)m * sizeof(WEIGHT_C_T), s);                        \
    int threads = 256;                                             \
    int blocks = (m + threads - 1) / threads;                      \
    _binary_jitsmv_gather_kern##SUFFIX<<<blocks, threads, 0, s>>>( \
        static_cast<const WEIGHT_C_T*>(weight.data_ptr()),         \
        static_cast<const float*>(clen.data_ptr()),                \
        static_cast<const int*>(seed.data_ptr()),                  \
        static_cast<const SPIKE_C_T*>(vector.data_ptr()),          \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),               \
        m, k                                                       \
    );                                                             \
}

// @BE binary_jitsmv_gather_f32_bool
FFI_BINARY_JITSMV_GATHER(_f32_bool,  float,         int8_t)
// @BE binary_jitsmv_gather_f32_float
FFI_BINARY_JITSMV_GATHER(_f32_float, float,         float)
// @BE binary_jitsmv_gather_f64_bool
FFI_BINARY_JITSMV_GATHER(_f64_bool,  double,        int8_t)
// @BE binary_jitsmv_gather_f64_float
FFI_BINARY_JITSMV_GATHER(_f64_float, double,        float)
// @BE binary_jitsmv_gather_f16_bool
FFI_BINARY_JITSMV_GATHER(_f16_bool,  __half,        int8_t)
// @BE binary_jitsmv_gather_f16_float
FFI_BINARY_JITSMV_GATHER(_f16_float, __half,        float)
// @BE binary_jitsmv_gather_bf16_bool
FFI_BINARY_JITSMV_GATHER(_bf16_bool, __nv_bfloat16, int8_t)
// @BE binary_jitsmv_gather_bf16_float
FFI_BINARY_JITSMV_GATHER(_bf16_float,__nv_bfloat16, float)

// ---- CUDA: binary_jitsmv scatter ----

#define FFI_BINARY_JITSMV_SCATTER(SUFFIX, WEIGHT_C_T, SPIKE_C_T)    \
void binary_jitsmv_scatter##SUFFIX(                                 \
    const BE::Tensor weight,                                    \
    const BE::Tensor clen,                                      \
    const BE::Tensor seed,                                      \
    const BE::Tensor vector,                                    \
    BE::Tensor output,                                    \
    int64_t stream                                                  \
) {                                                                 \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);        \
    int m = static_cast<int>(output.size(0));                       \
    int k = static_cast<int>(vector.size(0));                       \
    cudaMemsetAsync(output.data_ptr(), 0,                           \
        (size_t)m * sizeof(WEIGHT_C_T), s);                         \
    int threads = 256;                                              \
    int blocks = (k + threads - 1) / threads;                       \
    _binary_jitsmv_scatter_kern##SUFFIX<<<blocks, threads, 0, s>>>( \
        static_cast<const WEIGHT_C_T*>(weight.data_ptr()),          \
        static_cast<const float*>(clen.data_ptr()),                 \
        static_cast<const int*>(seed.data_ptr()),                   \
        static_cast<const SPIKE_C_T*>(vector.data_ptr()),           \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                \
        m, k                                                        \
    );                                                              \
}

// @BE binary_jitsmv_scatter_f32_bool
FFI_BINARY_JITSMV_SCATTER(_f32_bool,  float,         int8_t)
// @BE binary_jitsmv_scatter_f32_float
FFI_BINARY_JITSMV_SCATTER(_f32_float, float,         float)
// @BE binary_jitsmv_scatter_f64_bool
FFI_BINARY_JITSMV_SCATTER(_f64_bool,  double,        int8_t)
// @BE binary_jitsmv_scatter_f64_float
FFI_BINARY_JITSMV_SCATTER(_f64_float, double,        float)
// @BE binary_jitsmv_scatter_f16_bool
FFI_BINARY_JITSMV_SCATTER(_f16_bool,  __half,        int8_t)
// @BE binary_jitsmv_scatter_f16_float
FFI_BINARY_JITSMV_SCATTER(_f16_float, __half,        float)
// @BE binary_jitsmv_scatter_bf16_bool
FFI_BINARY_JITSMV_SCATTER(_bf16_bool, __nv_bfloat16, int8_t)
// @BE binary_jitsmv_scatter_bf16_float
FFI_BINARY_JITSMV_SCATTER(_bf16_float,__nv_bfloat16, float)
