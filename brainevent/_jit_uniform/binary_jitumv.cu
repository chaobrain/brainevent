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
 * binary_jitumv.cu — JIT Uniform Event-Driven Matrix-Vector Product (binary_jitumv operator)
 * ============================================================================================
 *
 * Computes y[i] = sum_{j in C(i) : spike[j] active} Uniform(w_low, w_high)
 * where C(i) is the JIT-generated uniform random connectivity set for row i.
 * Each active spike contributes a fresh weight sample drawn from Uniform(w_low, w_high).
 *
 * Operations
 * ----------
 * binary_jitumv_gather_{wt}_{sp}  — corder=True  (gather, one thread per output)
 * binary_jitumv_scatter_{wt}_{sp} — corder=False (scatter, one thread per input)
 *
 * Weight dtypes (wt): f32, f64, f16, bf16
 * Spike dtypes  (sp): bool (int8_t), float
 *
 * Parameters
 * ----------
 * w_low   : shape (1,), lower bound of uniform weight distribution
 * w_high  : shape (1,), upper bound of uniform weight distribution
 * clen    : shape (1,), connection length = ceil(2/prob) (float32)
 * seed    : shape (1,), int32 random seed
 * vector  : shape (k,), spike vector (bool or float)
 * output  : shape (m,), output accumulator
 *
 * Gather kernels write each output element exactly once — no memset needed.
 * Scatter kernels use atomicAdd — output is zeroed via cudaMemsetAsync first.
 * Scatter kernels skip inactive spikes entirely (early return optimization).
 *
 * IMPORTANT: All data_ptr() returns are GPU device pointers — NEVER dereference on host.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include <cstdint>
#include "../cuda_common.h"

// =========================================================================
// Spike activity checks (for binary kernels)
// =========================================================================

#undef IS_ACTIVE_BOOL
#undef IS_ACTIVE_FLOAT
#define IS_ACTIVE_BOOL(v, j)  ((v)[j] != 0)
#define IS_ACTIVE_FLOAT(v, j) ((v)[j] > 0.0f)

// =========================================================================
// Shared memory threshold: 48KB default max dynamic shared memory per block
// =========================================================================

#define SMEM_THRESHOLD 49152

// #########################################################################
// ##  binary_jitumv — Event-Driven Matrix-Vector Product                ##
// #########################################################################

// =========================================================================
// Gather kernel (corder=true): one thread per output element
// y[i] = sum_{j in C(i) : spike[j] active} Uniform(w_low, w_high)
// The weight is still sampled from the RNG even if the spike is inactive
// (to preserve the correct RNG stream for subsequent connections).
// No memset needed: every output[i] is written exactly once (direct store).
// =========================================================================

#define DEFINE_BINARY_JITUMV_GATHER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, SPIKE_T, IS_ACTIVE, ACC_ZERO) \
__global__ void _binary_jitumv_gather_kern##SUFFIX(                                                         \
    const WEIGHT_T* __restrict__ w_low,                                                                     \
    const WEIGHT_T* __restrict__ w_high,                                                                    \
    const float*    __restrict__ clen,                                                                      \
    const int*      __restrict__ seed,                                                                      \
    const SPIKE_T*  __restrict__ vector,                                                                    \
    WEIGHT_T*       __restrict__ output,                                                                    \
    int m, int k                                                                                            \
) {                                                                                                         \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                                          \
    if (i >= m) return;                                                                                     \
    ACC_T wlo = READ_W(__ldg(&w_low[0]));                                                                   \
    ACC_T range = READ_W(__ldg(&w_high[0])) - wlo;                                                          \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                                        \
    if (cl < 2) cl = 2;                                                                                     \
    curandStatePhilox4_32_10_t state;                                                                       \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)i, 0ULL, &state);                  \
    unsigned int j = curand(&state) % cl;                                                                   \
    ACC_T acc = ACC_ZERO;                                                                                   \
    while (j < (unsigned int)k) {                                                                           \
        float u = curand_uniform(&state);                                                                   \
        if (IS_ACTIVE(vector, j)) {                                                                         \
            ACC_T w = wlo + (ACC_T)u * range;                                                               \
            acc += w;                                                                                       \
        }                                                                                                   \
        j += 1 + (curand(&state) % (cl - 1));                                                               \
    }                                                                                                       \
    output[i] = WRITE_W(acc);                                                                               \
}

// f32 weight + bool/float spikes
DEFINE_BINARY_JITUMV_GATHER(_f32_bool,  float,         float,  READ_F32,  WRITE_F32,  int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITUMV_GATHER(_f32_float, float,         float,  READ_F32,  WRITE_F32,  float,  IS_ACTIVE_FLOAT, 0.0f)
// f64 weight + bool/float spikes
DEFINE_BINARY_JITUMV_GATHER(_f64_bool,  double,        double, READ_F64,  WRITE_F64,  int8_t, IS_ACTIVE_BOOL,  0.0)
DEFINE_BINARY_JITUMV_GATHER(_f64_float, double,        double, READ_F64,  WRITE_F64,  float,  IS_ACTIVE_FLOAT, 0.0)
// f16 weight + bool/float spikes
DEFINE_BINARY_JITUMV_GATHER(_f16_bool,  __half,        float,  READ_F16,  WRITE_F16,  int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITUMV_GATHER(_f16_float, __half,        float,  READ_F16,  WRITE_F16,  float,  IS_ACTIVE_FLOAT, 0.0f)
// bf16 weight + bool/float spikes
DEFINE_BINARY_JITUMV_GATHER(_bf16_bool, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITUMV_GATHER(_bf16_float,__nv_bfloat16, float,  READ_BF16, WRITE_BF16, float,  IS_ACTIVE_FLOAT, 0.0f)

// =========================================================================
// Gather kernel with shared memory spike caching (corder=true)
// Cooperatively loads the spike vector into shared memory.
// Used when k * sizeof(SPIKE_T) <= 48KB.
// =========================================================================

#define DEFINE_BINARY_JITUMV_GATHER_SMEM(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, SPIKE_T, IS_ACTIVE, ACC_ZERO) \
__global__ void _binary_jitumv_gather_smem_kern##SUFFIX(                                                         \
    const WEIGHT_T* __restrict__ w_low,                                                                          \
    const WEIGHT_T* __restrict__ w_high,                                                                         \
    const float*    __restrict__ clen,                                                                           \
    const int*      __restrict__ seed,                                                                           \
    const SPIKE_T*  __restrict__ vector,                                                                         \
    WEIGHT_T*       __restrict__ output,                                                                         \
    int m, int k                                                                                                 \
) {                                                                                                              \
    extern __shared__ char _smem_bytes[];                                                                        \
    SPIKE_T* sv = reinterpret_cast<SPIKE_T*>(_smem_bytes);                                                       \
    for (int idx = threadIdx.x; idx < k; idx += blockDim.x) {                                                    \
        sv[idx] = __ldg(&vector[idx]);                                                                           \
    }                                                                                                            \
    __syncthreads();                                                                                             \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                                               \
    if (i >= m) return;                                                                                          \
    ACC_T wlo = READ_W(__ldg(&w_low[0]));                                                                        \
    ACC_T range = READ_W(__ldg(&w_high[0])) - wlo;                                                               \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                                             \
    if (cl < 2) cl = 2;                                                                                          \
    curandStatePhilox4_32_10_t state;                                                                            \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)i, 0ULL, &state);                       \
    unsigned int j = curand(&state) % cl;                                                                        \
    ACC_T acc = ACC_ZERO;                                                                                        \
    while (j < (unsigned int)k) {                                                                                \
        float u = curand_uniform(&state);                                                                        \
        if (IS_ACTIVE(sv, j)) {                                                                                  \
            ACC_T w = wlo + (ACC_T)u * range;                                                                    \
            acc += w;                                                                                            \
        }                                                                                                        \
        j += 1 + (curand(&state) % (cl - 1));                                                                    \
    }                                                                                                            \
    output[i] = WRITE_W(acc);                                                                                    \
}

// f32 weight + bool/float spikes
DEFINE_BINARY_JITUMV_GATHER_SMEM(_f32_bool,  float,         float,  READ_F32,  WRITE_F32,  int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITUMV_GATHER_SMEM(_f32_float, float,         float,  READ_F32,  WRITE_F32,  float,  IS_ACTIVE_FLOAT, 0.0f)
// f64 weight + bool/float spikes
DEFINE_BINARY_JITUMV_GATHER_SMEM(_f64_bool,  double,        double, READ_F64,  WRITE_F64,  int8_t, IS_ACTIVE_BOOL,  0.0)
DEFINE_BINARY_JITUMV_GATHER_SMEM(_f64_float, double,        double, READ_F64,  WRITE_F64,  float,  IS_ACTIVE_FLOAT, 0.0)
// f16 weight + bool/float spikes
DEFINE_BINARY_JITUMV_GATHER_SMEM(_f16_bool,  __half,        float,  READ_F16,  WRITE_F16,  int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITUMV_GATHER_SMEM(_f16_float, __half,        float,  READ_F16,  WRITE_F16,  float,  IS_ACTIVE_FLOAT, 0.0f)
// bf16 weight + bool/float spikes
DEFINE_BINARY_JITUMV_GATHER_SMEM(_bf16_bool, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITUMV_GATHER_SMEM(_bf16_float,__nv_bfloat16, float,  READ_BF16, WRITE_BF16, float,  IS_ACTIVE_FLOAT, 0.0f)

// =========================================================================
// Scatter kernel (corder=false): one thread per input element
// Skip inactive spikes entirely (zero-work optimization).
// For active spikes: scatter Uniform(w_low, w_high) to each connected output.
// =========================================================================

#define DEFINE_BINARY_JITUMV_SCATTER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, SPIKE_T, IS_ACTIVE, ATOMIC_ADD) \
__global__ void _binary_jitumv_scatter_kern##SUFFIX(                                                           \
    const WEIGHT_T* __restrict__ w_low,                                                                        \
    const WEIGHT_T* __restrict__ w_high,                                                                       \
    const float*    __restrict__ clen,                                                                         \
    const int*      __restrict__ seed,                                                                         \
    const SPIKE_T*  __restrict__ vector,                                                                       \
    WEIGHT_T*       __restrict__ output,                                                                       \
    int m, int k                                                                                               \
) {                                                                                                            \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                                                             \
    if (j >= k) return;                                                                                        \
    if (!IS_ACTIVE(vector, j)) return;                                                                         \
    ACC_T wlo = READ_W(__ldg(&w_low[0]));                                                                      \
    ACC_T range = READ_W(__ldg(&w_high[0])) - wlo;                                                             \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                                           \
    if (cl < 2) cl = 2;                                                                                        \
    curandStatePhilox4_32_10_t state;                                                                          \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)j, 0ULL, &state);                     \
    unsigned int i = curand(&state) % cl;                                                                      \
    while (i < (unsigned int)m) {                                                                              \
        float u = curand_uniform(&state);                                                                      \
        ACC_T w = wlo + (ACC_T)u * range;                                                                      \
        ATOMIC_ADD(&output[i], w);                                                                             \
        i += 1 + (curand(&state) % (cl - 1));                                                                  \
    }                                                                                                          \
}

// f32 weight + bool/float spikes
DEFINE_BINARY_JITUMV_SCATTER(_f32_bool,  float,         float,  READ_F32,  WRITE_F32,  int8_t, IS_ACTIVE_BOOL,  atomicAdd_f32)
DEFINE_BINARY_JITUMV_SCATTER(_f32_float, float,         float,  READ_F32,  WRITE_F32,  float,  IS_ACTIVE_FLOAT, atomicAdd_f32)
// f64 weight + bool/float spikes
DEFINE_BINARY_JITUMV_SCATTER(_f64_bool,  double,        double, READ_F64,  WRITE_F64,  int8_t, IS_ACTIVE_BOOL,  atomicAdd_f64)
DEFINE_BINARY_JITUMV_SCATTER(_f64_float, double,        double, READ_F64,  WRITE_F64,  float,  IS_ACTIVE_FLOAT, atomicAdd_f64)
// f16 weight + bool/float spikes
DEFINE_BINARY_JITUMV_SCATTER(_f16_bool,  __half,        float,  READ_F16,  WRITE_F16,  int8_t, IS_ACTIVE_BOOL,  atomicAdd_f16)
DEFINE_BINARY_JITUMV_SCATTER(_f16_float, __half,        float,  READ_F16,  WRITE_F16,  float,  IS_ACTIVE_FLOAT, atomicAdd_f16)
// bf16 weight + bool/float spikes
DEFINE_BINARY_JITUMV_SCATTER(_bf16_bool, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, int8_t, IS_ACTIVE_BOOL,  atomicAdd_bf16)
DEFINE_BINARY_JITUMV_SCATTER(_bf16_float,__nv_bfloat16, float,  READ_BF16, WRITE_BF16, float,  IS_ACTIVE_FLOAT, atomicAdd_bf16)

// ---- TVM FFI: binary_jitumv gather ----
// Dispatches to shared-memory kernel when spike vector fits in 48KB smem.
// No memset needed: gather kernels write every output element exactly once.

#define FFI_BINARY_JITUMV_GATHER(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                      \
void binary_jitumv_gather##SUFFIX(                                                   \
    tvm::ffi::TensorView w_low,                                                      \
    tvm::ffi::TensorView w_high,                                                     \
    tvm::ffi::TensorView clen,                                                       \
    tvm::ffi::TensorView seed,                                                       \
    tvm::ffi::TensorView vector,                                                     \
    tvm::ffi::TensorView output,                                                     \
    int64_t stream                                                                   \
) {                                                                                  \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                         \
    int m = static_cast<int>(output.size(0));                                        \
    int k = static_cast<int>(vector.size(0));                                        \
    int threads = 256;                                                               \
    int blocks = (m + threads - 1) / threads;                                        \
    size_t smem_bytes = (size_t)k * sizeof(SPIKE_C_T);                               \
    if (smem_bytes <= SMEM_THRESHOLD) {                                              \
        _binary_jitumv_gather_smem_kern##SUFFIX<<<blocks, threads, smem_bytes, s>>>( \
            static_cast<const WEIGHT_C_T*>(w_low.data_ptr()),                        \
            static_cast<const WEIGHT_C_T*>(w_high.data_ptr()),                       \
            static_cast<const float*>(clen.data_ptr()),                              \
            static_cast<const int*>(seed.data_ptr()),                                \
            static_cast<const SPIKE_C_T*>(vector.data_ptr()),                        \
            static_cast<WEIGHT_C_T*>(output.data_ptr()),                             \
            m, k                                                                     \
        );                                                                           \
    } else {                                                                         \
        _binary_jitumv_gather_kern##SUFFIX<<<blocks, threads, 0, s>>>(               \
            static_cast<const WEIGHT_C_T*>(w_low.data_ptr()),                        \
            static_cast<const WEIGHT_C_T*>(w_high.data_ptr()),                       \
            static_cast<const float*>(clen.data_ptr()),                              \
            static_cast<const int*>(seed.data_ptr()),                                \
            static_cast<const SPIKE_C_T*>(vector.data_ptr()),                        \
            static_cast<WEIGHT_C_T*>(output.data_ptr()),                             \
            m, k                                                                     \
        );                                                                           \
    }                                                                                \
}

// @tvm_ffi binary_jitumv_gather_f32_bool
FFI_BINARY_JITUMV_GATHER(_f32_bool,  float,         int8_t)
// @tvm_ffi binary_jitumv_gather_f32_float
FFI_BINARY_JITUMV_GATHER(_f32_float, float,         float)
// @tvm_ffi binary_jitumv_gather_f64_bool
FFI_BINARY_JITUMV_GATHER(_f64_bool,  double,        int8_t)
// @tvm_ffi binary_jitumv_gather_f64_float
FFI_BINARY_JITUMV_GATHER(_f64_float, double,        float)
// @tvm_ffi binary_jitumv_gather_f16_bool
FFI_BINARY_JITUMV_GATHER(_f16_bool,  __half,        int8_t)
// @tvm_ffi binary_jitumv_gather_f16_float
FFI_BINARY_JITUMV_GATHER(_f16_float, __half,        float)
// @tvm_ffi binary_jitumv_gather_bf16_bool
FFI_BINARY_JITUMV_GATHER(_bf16_bool, __nv_bfloat16, int8_t)
// @tvm_ffi binary_jitumv_gather_bf16_float
FFI_BINARY_JITUMV_GATHER(_bf16_float,__nv_bfloat16, float)

// ---- TVM FFI: binary_jitumv scatter ----

#define FFI_BINARY_JITUMV_SCATTER(SUFFIX, WEIGHT_C_T, SPIKE_C_T)    \
void binary_jitumv_scatter##SUFFIX(                                 \
    tvm::ffi::TensorView w_low,                                     \
    tvm::ffi::TensorView w_high,                                    \
    tvm::ffi::TensorView clen,                                      \
    tvm::ffi::TensorView seed,                                      \
    tvm::ffi::TensorView vector,                                    \
    tvm::ffi::TensorView output,                                    \
    int64_t stream                                                  \
) {                                                                 \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);        \
    int m = static_cast<int>(output.size(0));                       \
    int k = static_cast<int>(vector.size(0));                       \
    cudaMemsetAsync(output.data_ptr(), 0,                           \
        (size_t)m * sizeof(WEIGHT_C_T), s);                         \
    int threads = 256;                                              \
    int blocks = (k + threads - 1) / threads;                       \
    _binary_jitumv_scatter_kern##SUFFIX<<<blocks, threads, 0, s>>>( \
        static_cast<const WEIGHT_C_T*>(w_low.data_ptr()),           \
        static_cast<const WEIGHT_C_T*>(w_high.data_ptr()),          \
        static_cast<const float*>(clen.data_ptr()),                 \
        static_cast<const int*>(seed.data_ptr()),                   \
        static_cast<const SPIKE_C_T*>(vector.data_ptr()),           \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                \
        m, k                                                        \
    );                                                              \
}

// @tvm_ffi binary_jitumv_scatter_f32_bool
FFI_BINARY_JITUMV_SCATTER(_f32_bool,  float,         int8_t)
// @tvm_ffi binary_jitumv_scatter_f32_float
FFI_BINARY_JITUMV_SCATTER(_f32_float, float,         float)
// @tvm_ffi binary_jitumv_scatter_f64_bool
FFI_BINARY_JITUMV_SCATTER(_f64_bool,  double,        int8_t)
// @tvm_ffi binary_jitumv_scatter_f64_float
FFI_BINARY_JITUMV_SCATTER(_f64_float, double,        float)
// @tvm_ffi binary_jitumv_scatter_f16_bool
FFI_BINARY_JITUMV_SCATTER(_f16_bool,  __half,        int8_t)
// @tvm_ffi binary_jitumv_scatter_f16_float
FFI_BINARY_JITUMV_SCATTER(_f16_float, __half,        float)
// @tvm_ffi binary_jitumv_scatter_bf16_bool
FFI_BINARY_JITUMV_SCATTER(_bf16_bool, __nv_bfloat16, int8_t)
// @tvm_ffi binary_jitumv_scatter_bf16_float
FFI_BINARY_JITUMV_SCATTER(_bf16_float,__nv_bfloat16, float)
