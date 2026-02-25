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
 * binary_jitsmm.cu — JIT Scalar Event-Driven Matrix-Matrix Product CUDA Kernels
 * ==============================================================================
 *
 * Event-driven (binary spike) matrix-matrix product for JIT scalar connectivity.
 *
 * Operation
 * ---------
 * binary_jitsmm — Event-driven mat-mat:
 *   Y[i,:] = w * count{j in C(i) : B[j,:] active}
 *   where M[i,j] = Bernoulli(prob) is generated on-the-fly.
 *
 * Parameters
 * ----------
 * weight : shape (1,), scalar weight for all connections
 * clen   : shape (1,), connection length = 2/prob (float32)
 * seed   : shape (1,), int32 random seed
 * B      : shape (k, n), spike input matrix (bool as int8_t, or float)
 * output : shape (m, n), output matrix
 *
 * corder=True  (gather): one thread per output row, no atomics
 *   Dispatches to register-accumulator kernel for n<=16 (2.7x faster),
 *   falls back to thread-per-row kernel for n>16.
 * corder=False (scatter): one thread per input row, atomicAdd for active columns
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
// ##  binary_jitsmm — Event-Driven Matrix-Matrix Product                 ##
// #########################################################################

// =========================================================================
// Register-accumulator gather kernel (n <= 16): one thread per row
// Uses 16 register accumulators for spike counts per column.
// =========================================================================

#define DEFINE_BINARY_JITSMM_GATHER_REGACC(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, SPIKE_T, IS_ACTIVE, ACC_ZERO) \
__global__ void _binary_jitsmm_gather_regacc_kern##SUFFIX(                                                         \
    const WEIGHT_T* __restrict__ weight,                                                                           \
    const float*    __restrict__ clen,                                                                             \
    const int*      __restrict__ seed,                                                                             \
    const SPIKE_T*  __restrict__ B,                                                                                \
    WEIGHT_T*       __restrict__ output,                                                                           \
    int m, int k, int n                                                                                            \
) {                                                                                                                \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                                                 \
    if (i >= m) return;                                                                                            \
    ACC_T w0 = READ_W(__ldg(&weight[0]));                                                                          \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                                               \
    if (cl < 2) cl = 2;                                                                                            \
    curandStatePhilox4_32_10_t state;                                                                              \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)i, 0ULL, &state);                         \
    unsigned int j = curand(&state) % cl;                                                                          \
    ACC_T acc[16];                                                                                                 \
    for (int c = 0; c < 16; c++) acc[c] = ACC_ZERO;                                                                \
    while (j < (unsigned int)k) {                                                                                  \
        const SPIKE_T* br = B + (size_t)j * n;                                                                     \
        for (int c = 0; c < 16; c++) {                                                                             \
            if (c < n && IS_ACTIVE(br[c])) acc[c] += (ACC_T)1.0;                                                   \
        }                                                                                                          \
        j += 1 + (curand(&state) % (cl - 1));                                                                      \
    }                                                                                                              \
    WEIGHT_T* out_row = output + (size_t)i * n;                                                                    \
    for (int c = 0; c < 16; c++) {                                                                                 \
        if (c < n) out_row[c] = WRITE_W(w0 * acc[c]);                                                              \
    }                                                                                                              \
}

// f32 + bool/float
DEFINE_BINARY_JITSMM_GATHER_REGACC(_f32_bool,  float,         float,  READ_F32,  WRITE_F32,  int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITSMM_GATHER_REGACC(_f32_float, float,         float,  READ_F32,  WRITE_F32,  float,  IS_ACTIVE_FLOAT, 0.0f)
// f64 + bool/float
DEFINE_BINARY_JITSMM_GATHER_REGACC(_f64_bool,  double,        double, READ_F64,  WRITE_F64,  int8_t, IS_ACTIVE_BOOL,  0.0)
DEFINE_BINARY_JITSMM_GATHER_REGACC(_f64_float, double,        double, READ_F64,  WRITE_F64,  float,  IS_ACTIVE_FLOAT, 0.0)
// f16 + bool/float
DEFINE_BINARY_JITSMM_GATHER_REGACC(_f16_bool,  __half,        float,  READ_F16,  WRITE_F16,  int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITSMM_GATHER_REGACC(_f16_float, __half,        float,  READ_F16,  WRITE_F16,  float,  IS_ACTIVE_FLOAT, 0.0f)
// bf16 + bool/float
DEFINE_BINARY_JITSMM_GATHER_REGACC(_bf16_bool, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITSMM_GATHER_REGACC(_bf16_float,__nv_bfloat16, float,  READ_BF16, WRITE_BF16, float,  IS_ACTIVE_FLOAT, 0.0f)

// =========================================================================
// Thread-per-row gather kernel (fallback for n > 128)
// Y[i, col] = w * sum_{j in C(i)} active(B[j, col])
// =========================================================================

#define DEFINE_BINARY_JITSMM_GATHER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, SPIKE_T, IS_ACTIVE, ACC_ZERO) \
__global__ void _binary_jitsmm_gather_kern##SUFFIX(                                                         \
    const WEIGHT_T* __restrict__ weight,                                                                    \
    const float*    __restrict__ clen,                                                                      \
    const int*      __restrict__ seed,                                                                      \
    const SPIKE_T*  __restrict__ B,                                                                         \
    WEIGHT_T*       __restrict__ output,                                                                    \
    int m, int k, int n                                                                                     \
) {                                                                                                         \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                                          \
    if (i >= m) return;                                                                                     \
    ACC_T w0 = READ_W(__ldg(&weight[0]));                                                                   \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                                        \
    if (cl < 2) cl = 2;                                                                                     \
    curandStatePhilox4_32_10_t state;                                                                       \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)i, 0ULL, &state);                  \
    unsigned int j = curand(&state) % cl;                                                                   \
    WEIGHT_T* out_row = output + (size_t)i * n;                                                             \
    while (j < (unsigned int)k) {                                                                           \
        const SPIKE_T* b_row = B + (size_t)j * n;                                                           \
        for (int col = 0; col < n; col++) {                                                                 \
            if (IS_ACTIVE(b_row[col])) {                                                                    \
                ACC_T cur = READ_W(out_row[col]);                                                           \
                out_row[col] = WRITE_W(cur + (ACC_T)1.0);                                                   \
            }                                                                                               \
        }                                                                                                   \
        j += 1 + (curand(&state) % (cl - 1));                                                               \
    }                                                                                                       \
    /* Scale by weight */                                                                                   \
    for (int col = 0; col < n; col++) {                                                                     \
        ACC_T cur = READ_W(out_row[col]);                                                                   \
        out_row[col] = WRITE_W(w0 * cur);                                                                   \
    }                                                                                                       \
}

// f32 + bool/float
DEFINE_BINARY_JITSMM_GATHER(_f32_bool,  float,         float,  READ_F32,  WRITE_F32,  int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITSMM_GATHER(_f32_float, float,         float,  READ_F32,  WRITE_F32,  float,  IS_ACTIVE_FLOAT, 0.0f)
// f64 + bool/float
DEFINE_BINARY_JITSMM_GATHER(_f64_bool,  double,        double, READ_F64,  WRITE_F64,  int8_t, IS_ACTIVE_BOOL,  0.0)
DEFINE_BINARY_JITSMM_GATHER(_f64_float, double,        double, READ_F64,  WRITE_F64,  float,  IS_ACTIVE_FLOAT, 0.0)
// f16 + bool/float
DEFINE_BINARY_JITSMM_GATHER(_f16_bool,  __half,        float,  READ_F16,  WRITE_F16,  int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITSMM_GATHER(_f16_float, __half,        float,  READ_F16,  WRITE_F16,  float,  IS_ACTIVE_FLOAT, 0.0f)
// bf16 + bool/float
DEFINE_BINARY_JITSMM_GATHER(_bf16_bool, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITSMM_GATHER(_bf16_float,__nv_bfloat16, float,  READ_BF16, WRITE_BF16, float,  IS_ACTIVE_FLOAT, 0.0f)

// =========================================================================
// Scatter kernel (corder=false): one thread per input row
// For each input row j, generate connections and scatter weight for active columns.
// =========================================================================

#define DEFINE_BINARY_JITSMM_SCATTER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, SPIKE_T, IS_ACTIVE, ATOMIC_ADD) \
__global__ void _binary_jitsmm_scatter_kern##SUFFIX(                                                           \
    const WEIGHT_T* __restrict__ weight,                                                                       \
    const float*    __restrict__ clen,                                                                         \
    const int*      __restrict__ seed,                                                                         \
    const SPIKE_T*  __restrict__ B,                                                                            \
    WEIGHT_T*       __restrict__ output,                                                                       \
    int m, int k, int n                                                                                        \
) {                                                                                                            \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                                                             \
    if (j >= k) return;                                                                                        \
    ACC_T w0 = READ_W(__ldg(&weight[0]));                                                                      \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                                           \
    if (cl < 2) cl = 2;                                                                                        \
    const SPIKE_T* b_row = B + (size_t)j * n;                                                                  \
    curandStatePhilox4_32_10_t state;                                                                          \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)j, 0ULL, &state);                     \
    unsigned int i = curand(&state) % cl;                                                                      \
    while (i < (unsigned int)m) {                                                                              \
        WEIGHT_T* out_row = output + (size_t)i * n;                                                            \
        for (int col = 0; col < n; col++) {                                                                    \
            if (IS_ACTIVE(b_row[col])) {                                                                       \
                ATOMIC_ADD(&out_row[col], w0);                                                                 \
            }                                                                                                  \
        }                                                                                                      \
        i += 1 + (curand(&state) % (cl - 1));                                                                  \
    }                                                                                                          \
}

// f32 + bool/float
DEFINE_BINARY_JITSMM_SCATTER(_f32_bool,  float,         float,  READ_F32,  WRITE_F32,  int8_t, IS_ACTIVE_BOOL,  atomic_add_f32)
DEFINE_BINARY_JITSMM_SCATTER(_f32_float, float,         float,  READ_F32,  WRITE_F32,  float,  IS_ACTIVE_FLOAT, atomic_add_f32)
// f64 + bool/float
DEFINE_BINARY_JITSMM_SCATTER(_f64_bool,  double,        double, READ_F64,  WRITE_F64,  int8_t, IS_ACTIVE_BOOL,  atomic_add_f64)
DEFINE_BINARY_JITSMM_SCATTER(_f64_float, double,        double, READ_F64,  WRITE_F64,  float,  IS_ACTIVE_FLOAT, atomic_add_f64)
// f16 + bool/float
DEFINE_BINARY_JITSMM_SCATTER(_f16_bool,  __half,        float,  READ_F16,  WRITE_F16,  int8_t, IS_ACTIVE_BOOL,  atomic_add_f16)
DEFINE_BINARY_JITSMM_SCATTER(_f16_float, __half,        float,  READ_F16,  WRITE_F16,  float,  IS_ACTIVE_FLOAT, atomic_add_f16)
// bf16 + bool/float
DEFINE_BINARY_JITSMM_SCATTER(_bf16_bool, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, int8_t, IS_ACTIVE_BOOL,  atomic_add_bf16)
DEFINE_BINARY_JITSMM_SCATTER(_bf16_float,__nv_bfloat16, float,  READ_BF16, WRITE_BF16, float,  IS_ACTIVE_FLOAT, atomic_add_bf16)

// ---- CUDA: binary_jitsmm gather ----
// Dispatches to register-accumulator kernel for n <= 16, fallback for n > 16.

#define FFI_BINARY_JITSMM_GATHER(SUFFIX, WEIGHT_C_T, SPIKE_C_T)               \
void binary_jitsmm_gather##SUFFIX(                                            \
    const BE::Tensor weight,                                              \
    const BE::Tensor clen,                                                \
    const BE::Tensor seed,                                                \
    const BE::Tensor B,                                                   \
    BE::Tensor output,                                              \
    int64_t stream                                                            \
) {                                                                           \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                  \
    int m = static_cast<int>(output.size(0));                                 \
    int n = static_cast<int>(output.size(1));                                 \
    int k = static_cast<int>(B.size(0));                                      \
    cudaMemsetAsync(output.data_ptr(), 0,                                     \
        (size_t)m * n * sizeof(WEIGHT_C_T), s);                               \
    int threads = 256;                                                        \
    int blocks = (m + threads - 1) / threads;                                 \
    if (n <= 16) {                                                            \
        _binary_jitsmm_gather_regacc_kern##SUFFIX<<<blocks, threads, 0, s>>>( \
            static_cast<const WEIGHT_C_T*>(weight.data_ptr()),                \
            static_cast<const float*>(clen.data_ptr()),                       \
            static_cast<const int*>(seed.data_ptr()),                         \
            static_cast<const SPIKE_C_T*>(B.data_ptr()),                      \
            static_cast<WEIGHT_C_T*>(output.data_ptr()),                      \
            m, k, n                                                           \
        );                                                                    \
    } else {                                                                  \
        _binary_jitsmm_gather_kern##SUFFIX<<<blocks, threads, 0, s>>>(        \
            static_cast<const WEIGHT_C_T*>(weight.data_ptr()),                \
            static_cast<const float*>(clen.data_ptr()),                       \
            static_cast<const int*>(seed.data_ptr()),                         \
            static_cast<const SPIKE_C_T*>(B.data_ptr()),                      \
            static_cast<WEIGHT_C_T*>(output.data_ptr()),                      \
            m, k, n                                                           \
        );                                                                    \
    }                                                                         \
}

// @BE binary_jitsmm_gather_f32_bool
FFI_BINARY_JITSMM_GATHER(_f32_bool,  float,         int8_t)
// @BE binary_jitsmm_gather_f32_float
FFI_BINARY_JITSMM_GATHER(_f32_float, float,         float)
// @BE binary_jitsmm_gather_f64_bool
FFI_BINARY_JITSMM_GATHER(_f64_bool,  double,        int8_t)
// @BE binary_jitsmm_gather_f64_float
FFI_BINARY_JITSMM_GATHER(_f64_float, double,        float)
// @BE binary_jitsmm_gather_f16_bool
FFI_BINARY_JITSMM_GATHER(_f16_bool,  __half,        int8_t)
// @BE binary_jitsmm_gather_f16_float
FFI_BINARY_JITSMM_GATHER(_f16_float, __half,        float)
// @BE binary_jitsmm_gather_bf16_bool
FFI_BINARY_JITSMM_GATHER(_bf16_bool, __nv_bfloat16, int8_t)
// @BE binary_jitsmm_gather_bf16_float
FFI_BINARY_JITSMM_GATHER(_bf16_float,__nv_bfloat16, float)

// ---- CUDA: binary_jitsmm scatter ----

#define FFI_BINARY_JITSMM_SCATTER(SUFFIX, WEIGHT_C_T, SPIKE_C_T)    \
void binary_jitsmm_scatter##SUFFIX(                                 \
    const BE::Tensor weight,                                    \
    const BE::Tensor clen,                                      \
    const BE::Tensor seed,                                      \
    const BE::Tensor B,                                         \
    BE::Tensor output,                                    \
    int64_t stream                                                  \
) {                                                                 \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);        \
    int m = static_cast<int>(output.size(0));                       \
    int n = static_cast<int>(output.size(1));                       \
    int k = static_cast<int>(B.size(0));                            \
    cudaMemsetAsync(output.data_ptr(), 0,                           \
        (size_t)m * n * sizeof(WEIGHT_C_T), s);                     \
    int threads = 256;                                              \
    int blocks = (k + threads - 1) / threads;                       \
    _binary_jitsmm_scatter_kern##SUFFIX<<<blocks, threads, 0, s>>>( \
        static_cast<const WEIGHT_C_T*>(weight.data_ptr()),          \
        static_cast<const float*>(clen.data_ptr()),                 \
        static_cast<const int*>(seed.data_ptr()),                   \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                \
        m, k, n                                                     \
    );                                                              \
}

// @BE binary_jitsmm_scatter_f32_bool
FFI_BINARY_JITSMM_SCATTER(_f32_bool,  float,         int8_t)
// @BE binary_jitsmm_scatter_f32_float
FFI_BINARY_JITSMM_SCATTER(_f32_float, float,         float)
// @BE binary_jitsmm_scatter_f64_bool
FFI_BINARY_JITSMM_SCATTER(_f64_bool,  double,        int8_t)
// @BE binary_jitsmm_scatter_f64_float
FFI_BINARY_JITSMM_SCATTER(_f64_float, double,        float)
// @BE binary_jitsmm_scatter_f16_bool
FFI_BINARY_JITSMM_SCATTER(_f16_bool,  __half,        int8_t)
// @BE binary_jitsmm_scatter_f16_float
FFI_BINARY_JITSMM_SCATTER(_f16_float, __half,        float)
// @BE binary_jitsmm_scatter_bf16_bool
FFI_BINARY_JITSMM_SCATTER(_bf16_bool, __nv_bfloat16, int8_t)
// @BE binary_jitsmm_scatter_bf16_float
FFI_BINARY_JITSMM_SCATTER(_bf16_float,__nv_bfloat16, float)
