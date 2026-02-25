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
 * binary_jitnmv.cu — Event-Driven JIT Normal Matrix-Vector Product Kernels
 * =========================================================================
 *
 * Computes y[i] = sum_{j active} Normal(w_loc, w_scale) * spike[j].
 * The connectivity matrix M[i,j] is never materialised; weights are generated
 * on the fly only for connected (j, i) pairs.
 *
 * TVM FFI entry points
 * --------------------
 * binary_jitnmv_gather_{f32,f64,f16,bf16}_{bool,float}
 *     — gather (corder=True):  one thread per output row i
 * binary_jitnmv_scatter_{f32,f64,f16,bf16}_{bool,float}
 *     — scatter (corder=False): one thread per input col j (skips inactive)
 *
 * Parameters (common)
 * -------------------
 * w_loc  : shape (1,), mean of normal weight distribution
 * w_scale: shape (1,), std dev of normal weight distribution
 * clen   : shape (1,), connection length = ceil(2/prob) (float32)
 * seed   : shape (1,), int32 random seed
 * vector : shape (k,), input spike vector (bool/int8_t or float)
 * output : shape (m,), output vector (zeroed before writing)
 *
 * Optimizations
 * -------------
 * - __ldg() on all read-only global memory (routes through L1 texture cache)
 * - Shared memory spike vector caching for small vectors (k*sizeof(SPIKE_T) <= 4096 bytes)
 * - Scatter kernel early-exits on inactive (zero) spike lanes
 *
 * IMPORTANT: All data_ptr() returns are GPU device pointers — NEVER dereference on host.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include <cstdint>
#include "cuda_common.h"

// =========================================================================
// Spike activity checks (for binary kernels)
// =========================================================================

#undef IS_ACTIVE_BOOL
#undef IS_ACTIVE_FLOAT
#define IS_ACTIVE_BOOL(v, j)  ((v)[j] != 0)
#define IS_ACTIVE_FLOAT(v, j) ((v)[j] > 0.0f)

// =========================================================================
// Shared memory threshold: 4KB (1024 float32 elements).
// =========================================================================

#define SMEM_THRESHOLD 4096

// #########################################################################
// ##  binary_jitnmv — Event-Driven Matrix-Vector Product                 ##
// #########################################################################

// --- Gather kernel: __ldg on parameters ---
#define DEFINE_BINARY_JITNMV_GATHER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, RNG_FUNC, SPIKE_T, IS_ACTIVE, ACC_ZERO) \
__global__ void _binary_jitnmv_gather_kern##SUFFIX(                                                                   \
    const WEIGHT_T* __restrict__ w_loc,                                                                               \
    const WEIGHT_T* __restrict__ w_scale,                                                                             \
    const float*    __restrict__ clen,                                                                                \
    const int*      __restrict__ seed,                                                                                \
    const SPIKE_T*  __restrict__ vector,                                                                              \
    WEIGHT_T*       __restrict__ output,                                                                              \
    int m, int k                                                                                                      \
) {                                                                                                                   \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                                                    \
    if (i >= m) return;                                                                                               \
    ACC_T loc = READ_W(__ldg(&w_loc[0]));                                                                             \
    ACC_T scale = READ_W(__ldg(&w_scale[0]));                                                                         \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                                                  \
    if (cl < 2) cl = 2;                                                                                               \
    curandStatePhilox4_32_10_t state;                                                                                 \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)i, 0ULL, &state);                            \
    unsigned int j = curand(&state) % cl;                                                                             \
    ACC_T acc = ACC_ZERO;                                                                                             \
    while (j < (unsigned int)k) {                                                                                     \
        ACC_T n = (ACC_T)RNG_FUNC(&state);                                                                            \
        if (IS_ACTIVE(vector, j)) {                                                                                   \
            ACC_T w = loc + n * scale;                                                                                \
            acc += w;                                                                                                 \
        }                                                                                                             \
        j += 1 + (curand(&state) % (cl - 1));                                                                         \
    }                                                                                                                 \
    output[i] = WRITE_W(acc);                                                                                         \
}

DEFINE_BINARY_JITNMV_GATHER(_f32_bool,   float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITNMV_GATHER(_f32_float,  float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, float,  IS_ACTIVE_FLOAT, 0.0f)
DEFINE_BINARY_JITNMV_GATHER(_f64_bool,   double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, int8_t, IS_ACTIVE_BOOL,  0.0)
DEFINE_BINARY_JITNMV_GATHER(_f64_float,  double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, float,  IS_ACTIVE_FLOAT, 0.0)
DEFINE_BINARY_JITNMV_GATHER(_f16_bool,   __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITNMV_GATHER(_f16_float,  __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, float,  IS_ACTIVE_FLOAT, 0.0f)
DEFINE_BINARY_JITNMV_GATHER(_bf16_bool,  __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITNMV_GATHER(_bf16_float, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, float,  IS_ACTIVE_FLOAT, 0.0f)

// --- Gather kernel with shared memory spike vector caching ---
#define DEFINE_BINARY_JITNMV_GATHER_SMEM(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, RNG_FUNC, SPIKE_T, IS_ACTIVE, ACC_ZERO) \
__global__ void _binary_jitnmv_gather_smem_kern##SUFFIX(                                                                   \
    const WEIGHT_T* __restrict__ w_loc,                                                                                    \
    const WEIGHT_T* __restrict__ w_scale,                                                                                  \
    const float*    __restrict__ clen,                                                                                     \
    const int*      __restrict__ seed,                                                                                     \
    const SPIKE_T*  __restrict__ vector,                                                                                   \
    WEIGHT_T*       __restrict__ output,                                                                                   \
    int m, int k                                                                                                           \
) {                                                                                                                        \
    extern __shared__ char _smem_bytes[];                                                                                  \
    SPIKE_T* sv = reinterpret_cast<SPIKE_T*>(_smem_bytes);                                                                 \
    for (int idx = threadIdx.x; idx < k; idx += blockDim.x) {                                                              \
        sv[idx] = vector[idx];                                                                                             \
    }                                                                                                                      \
    __syncthreads();                                                                                                       \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                                                         \
    if (i >= m) return;                                                                                                    \
    ACC_T loc = READ_W(__ldg(&w_loc[0]));                                                                                  \
    ACC_T scale = READ_W(__ldg(&w_scale[0]));                                                                              \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                                                       \
    if (cl < 2) cl = 2;                                                                                                    \
    curandStatePhilox4_32_10_t state;                                                                                      \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)i, 0ULL, &state);                                 \
    unsigned int j = curand(&state) % cl;                                                                                  \
    ACC_T acc = ACC_ZERO;                                                                                                  \
    while (j < (unsigned int)k) {                                                                                          \
        ACC_T n = (ACC_T)RNG_FUNC(&state);                                                                                 \
        if (IS_ACTIVE(sv, j)) {                                                                                            \
            ACC_T w = loc + n * scale;                                                                                     \
            acc += w;                                                                                                      \
        }                                                                                                                  \
        j += 1 + (curand(&state) % (cl - 1));                                                                              \
    }                                                                                                                      \
    output[i] = WRITE_W(acc);                                                                                              \
}

DEFINE_BINARY_JITNMV_GATHER_SMEM(_f32_bool,   float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITNMV_GATHER_SMEM(_f32_float,  float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, float,  IS_ACTIVE_FLOAT, 0.0f)
DEFINE_BINARY_JITNMV_GATHER_SMEM(_f64_bool,   double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, int8_t, IS_ACTIVE_BOOL,  0.0)
DEFINE_BINARY_JITNMV_GATHER_SMEM(_f64_float,  double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, float,  IS_ACTIVE_FLOAT, 0.0)
DEFINE_BINARY_JITNMV_GATHER_SMEM(_f16_bool,   __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITNMV_GATHER_SMEM(_f16_float,  __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, float,  IS_ACTIVE_FLOAT, 0.0f)
DEFINE_BINARY_JITNMV_GATHER_SMEM(_bf16_bool,  __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITNMV_GATHER_SMEM(_bf16_float, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, float,  IS_ACTIVE_FLOAT, 0.0f)

// --- Scatter kernel: __ldg on parameters, early-exit on inactive spikes ---
#define DEFINE_BINARY_JITNMV_SCATTER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, RNG_FUNC, SPIKE_T, IS_ACTIVE, ATOMIC_ADD) \
__global__ void _binary_jitnmv_scatter_kern##SUFFIX(                                                                     \
    const WEIGHT_T* __restrict__ w_loc,                                                                                  \
    const WEIGHT_T* __restrict__ w_scale,                                                                                \
    const float*    __restrict__ clen,                                                                                   \
    const int*      __restrict__ seed,                                                                                   \
    const SPIKE_T*  __restrict__ vector,                                                                                 \
    WEIGHT_T*       __restrict__ output,                                                                                 \
    int m, int k                                                                                                         \
) {                                                                                                                      \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                                                                       \
    if (j >= k) return;                                                                                                  \
    if (!IS_ACTIVE(vector, j)) return;                                                                                   \
    ACC_T loc = READ_W(__ldg(&w_loc[0]));                                                                                \
    ACC_T scale = READ_W(__ldg(&w_scale[0]));                                                                            \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                                                     \
    if (cl < 2) cl = 2;                                                                                                  \
    curandStatePhilox4_32_10_t state;                                                                                    \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)j, 0ULL, &state);                               \
    unsigned int i = curand(&state) % cl;                                                                                \
    while (i < (unsigned int)m) {                                                                                        \
        ACC_T n = (ACC_T)RNG_FUNC(&state);                                                                               \
        ACC_T w = loc + n * scale;                                                                                       \
        ATOMIC_ADD(&output[i], w);                                                                                       \
        i += 1 + (curand(&state) % (cl - 1));                                                                            \
    }                                                                                                                    \
}

DEFINE_BINARY_JITNMV_SCATTER(_f32_bool,   float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, int8_t, IS_ACTIVE_BOOL,  atomic_add_f32)
DEFINE_BINARY_JITNMV_SCATTER(_f32_float,  float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, float,  IS_ACTIVE_FLOAT, atomic_add_f32)
DEFINE_BINARY_JITNMV_SCATTER(_f64_bool,   double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, int8_t, IS_ACTIVE_BOOL,  atomicAdd_f64)
DEFINE_BINARY_JITNMV_SCATTER(_f64_float,  double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, float,  IS_ACTIVE_FLOAT, atomicAdd_f64)
DEFINE_BINARY_JITNMV_SCATTER(_f16_bool,   __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, int8_t, IS_ACTIVE_BOOL,  atomicAdd_f16)
DEFINE_BINARY_JITNMV_SCATTER(_f16_float,  __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, float,  IS_ACTIVE_FLOAT, atomicAdd_f16)
DEFINE_BINARY_JITNMV_SCATTER(_bf16_bool,  __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, int8_t, IS_ACTIVE_BOOL,  atomicAdd_bf16)
DEFINE_BINARY_JITNMV_SCATTER(_bf16_float, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, float,  IS_ACTIVE_FLOAT, atomicAdd_bf16)

// --- FFI gather: dispatch to smem or global kernel ---
#define FFI_BINARY_JITNMV_GATHER(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                      \
void binary_jitnmv_gather##SUFFIX(                                                   \
    tvm::ffi::TensorView w_loc,                                                      \
    tvm::ffi::TensorView w_scale,                                                    \
    tvm::ffi::TensorView clen,                                                       \
    tvm::ffi::TensorView seed,                                                       \
    tvm::ffi::TensorView vector,                                                     \
    tvm::ffi::TensorView output,                                                     \
    int64_t stream                                                                   \
) {                                                                                  \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                         \
    int m = static_cast<int>(output.size(0));                                        \
    int k = static_cast<int>(vector.size(0));                                        \
    cudaMemsetAsync(output.data_ptr(), 0,                                            \
        (size_t)m * sizeof(WEIGHT_C_T), s);                                          \
    int threads = 256;                                                               \
    int blocks = (m + threads - 1) / threads;                                        \
    size_t smem_bytes = (size_t)k * sizeof(SPIKE_C_T);                               \
    if (smem_bytes <= SMEM_THRESHOLD) {                                              \
        _binary_jitnmv_gather_smem_kern##SUFFIX<<<blocks, threads, smem_bytes, s>>>( \
            static_cast<const WEIGHT_C_T*>(w_loc.data_ptr()),                        \
            static_cast<const WEIGHT_C_T*>(w_scale.data_ptr()),                      \
            static_cast<const float*>(clen.data_ptr()),                              \
            static_cast<const int*>(seed.data_ptr()),                                \
            static_cast<const SPIKE_C_T*>(vector.data_ptr()),                        \
            static_cast<WEIGHT_C_T*>(output.data_ptr()),                             \
            m, k                                                                     \
        );                                                                           \
    } else {                                                                         \
        _binary_jitnmv_gather_kern##SUFFIX<<<blocks, threads, 0, s>>>(               \
            static_cast<const WEIGHT_C_T*>(w_loc.data_ptr()),                        \
            static_cast<const WEIGHT_C_T*>(w_scale.data_ptr()),                      \
            static_cast<const float*>(clen.data_ptr()),                              \
            static_cast<const int*>(seed.data_ptr()),                                \
            static_cast<const SPIKE_C_T*>(vector.data_ptr()),                        \
            static_cast<WEIGHT_C_T*>(output.data_ptr()),                             \
            m, k                                                                     \
        );                                                                           \
    }                                                                                \
}

// @tvm_ffi binary_jitnmv_gather_f32_bool
FFI_BINARY_JITNMV_GATHER(_f32_bool,   float,         int8_t)
// @tvm_ffi binary_jitnmv_gather_f32_float
FFI_BINARY_JITNMV_GATHER(_f32_float,  float,         float)
// @tvm_ffi binary_jitnmv_gather_f64_bool
FFI_BINARY_JITNMV_GATHER(_f64_bool,   double,        int8_t)
// @tvm_ffi binary_jitnmv_gather_f64_float
FFI_BINARY_JITNMV_GATHER(_f64_float,  double,        float)
// @tvm_ffi binary_jitnmv_gather_f16_bool
FFI_BINARY_JITNMV_GATHER(_f16_bool,   __half,        int8_t)
// @tvm_ffi binary_jitnmv_gather_f16_float
FFI_BINARY_JITNMV_GATHER(_f16_float,  __half,        float)
// @tvm_ffi binary_jitnmv_gather_bf16_bool
FFI_BINARY_JITNMV_GATHER(_bf16_bool,  __nv_bfloat16, int8_t)
// @tvm_ffi binary_jitnmv_gather_bf16_float
FFI_BINARY_JITNMV_GATHER(_bf16_float, __nv_bfloat16, float)

#define FFI_BINARY_JITNMV_SCATTER(SUFFIX, WEIGHT_C_T, SPIKE_C_T)    \
void binary_jitnmv_scatter##SUFFIX(                                 \
    tvm::ffi::TensorView w_loc,                                     \
    tvm::ffi::TensorView w_scale,                                   \
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
    _binary_jitnmv_scatter_kern##SUFFIX<<<blocks, threads, 0, s>>>( \
        static_cast<const WEIGHT_C_T*>(w_loc.data_ptr()),           \
        static_cast<const WEIGHT_C_T*>(w_scale.data_ptr()),         \
        static_cast<const float*>(clen.data_ptr()),                 \
        static_cast<const int*>(seed.data_ptr()),                   \
        static_cast<const SPIKE_C_T*>(vector.data_ptr()),           \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                \
        m, k                                                        \
    );                                                              \
}

// @tvm_ffi binary_jitnmv_scatter_f32_bool
FFI_BINARY_JITNMV_SCATTER(_f32_bool,   float,         int8_t)
// @tvm_ffi binary_jitnmv_scatter_f32_float
FFI_BINARY_JITNMV_SCATTER(_f32_float,  float,         float)
// @tvm_ffi binary_jitnmv_scatter_f64_bool
FFI_BINARY_JITNMV_SCATTER(_f64_bool,   double,        int8_t)
// @tvm_ffi binary_jitnmv_scatter_f64_float
FFI_BINARY_JITNMV_SCATTER(_f64_float,  double,        float)
// @tvm_ffi binary_jitnmv_scatter_f16_bool
FFI_BINARY_JITNMV_SCATTER(_f16_bool,   __half,        int8_t)
// @tvm_ffi binary_jitnmv_scatter_f16_float
FFI_BINARY_JITNMV_SCATTER(_f16_float,  __half,        float)
// @tvm_ffi binary_jitnmv_scatter_bf16_bool
FFI_BINARY_JITNMV_SCATTER(_bf16_bool,  __nv_bfloat16, int8_t)
// @tvm_ffi binary_jitnmv_scatter_bf16_float
FFI_BINARY_JITNMV_SCATTER(_bf16_float, __nv_bfloat16, float)
