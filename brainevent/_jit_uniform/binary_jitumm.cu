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
 * binary_jitumm.cu — JIT Uniform Event-Driven Matrix-Matrix Product (binary_jitumm operator)
 * ==============================================================================================
 *
 * Computes Y[i, col] = sum_{j in C(i)} Uniform(w_low, w_high) * active(B[j, col])
 * where C(i) is the JIT-generated uniform random connectivity set for row i.
 * For each connected j, one weight is sampled and applied to all active B columns.
 *
 * Operations
 * ----------
 * binary_jitumm_gather_{wt}_{sp}  — corder=True  (gather, one thread per output row)
 * binary_jitumm_scatter_{wt}_{sp} — corder=False (scatter, one thread per input row)
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
 * B       : shape (k, n), input spike matrix (bool or float)
 * output  : shape (m, n), output accumulator
 *
 * Gather kernels zero-initialize output rows in-kernel — no memset needed.
 * Scatter kernels use atomicAdd — output is zeroed via cudaMemsetAsync first.
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

// #########################################################################
// ##  binary_jitumm — Event-Driven Matrix-Matrix Product                ##
// #########################################################################

// =========================================================================
// Gather kernel (corder=true): one thread per output row i
// Y[i, col] = sum_{j in C(i)} Uniform(w_low, w_high) * active(B[j, col])
// For each connected j, one weight is sampled and applied to all active B columns.
// For n <= 32: uses register accumulators to avoid read-modify-write to
// global memory on every connection. For n > 32: falls back to in-kernel
// zero-init + global memory accumulation.
// =========================================================================

#define DEFINE_BINARY_JITUMM_GATHER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, SPIKE_T, IS_ACTIVE, ACC_ZERO) \
__global__ void _binary_jitumm_gather_kern##SUFFIX(                                                         \
    const WEIGHT_T* __restrict__ w_low,                                                                     \
    const WEIGHT_T* __restrict__ w_high,                                                                    \
    const float*    __restrict__ clen,                                                                      \
    const int*      __restrict__ seed,                                                                      \
    const SPIKE_T*  __restrict__ B,                                                                         \
    WEIGHT_T*       __restrict__ output,                                                                    \
    int m, int k, int n                                                                                     \
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
    WEIGHT_T* out_row = output + (size_t)i * n;                                                             \
    if (n <= 32) {                                                                                          \
        ACC_T acc[32];                                                                                      \
        for (int col = 0; col < n; col++) acc[col] = ACC_ZERO;                                              \
        while (j < (unsigned int)k) {                                                                       \
            float u = curand_uniform(&state);                                                               \
            ACC_T w = wlo + (ACC_T)u * range;                                                               \
            const SPIKE_T* b_row = B + (size_t)j * n;                                                       \
            for (int col = 0; col < n; col++) {                                                             \
                if (IS_ACTIVE(b_row, col)) {                                                                \
                    acc[col] += w;                                                                          \
                }                                                                                           \
            }                                                                                               \
            j += 1 + (curand(&state) % (cl - 1));                                                           \
        }                                                                                                   \
        for (int col = 0; col < n; col++) {                                                                 \
            out_row[col] = WRITE_W(acc[col]);                                                               \
        }                                                                                                   \
    } else {                                                                                                \
        for (int col = 0; col < n; col++) {                                                                 \
            out_row[col] = WRITE_W(ACC_ZERO);                                                               \
        }                                                                                                   \
        while (j < (unsigned int)k) {                                                                       \
            float u = curand_uniform(&state);                                                               \
            ACC_T w = wlo + (ACC_T)u * range;                                                               \
            const SPIKE_T* b_row = B + (size_t)j * n;                                                       \
            for (int col = 0; col < n; col++) {                                                             \
                if (IS_ACTIVE(b_row, col)) {                                                                \
                    ACC_T cur = READ_W(out_row[col]);                                                       \
                    out_row[col] = WRITE_W(cur + w);                                                        \
                }                                                                                           \
            }                                                                                               \
            j += 1 + (curand(&state) % (cl - 1));                                                           \
        }                                                                                                   \
    }                                                                                                       \
}

// f32 + bool/float
DEFINE_BINARY_JITUMM_GATHER(_f32_bool,  float,         float,  READ_F32,  WRITE_F32,  int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITUMM_GATHER(_f32_float, float,         float,  READ_F32,  WRITE_F32,  float,  IS_ACTIVE_FLOAT, 0.0f)
// f64 + bool/float
DEFINE_BINARY_JITUMM_GATHER(_f64_bool,  double,        double, READ_F64,  WRITE_F64,  int8_t, IS_ACTIVE_BOOL,  0.0)
DEFINE_BINARY_JITUMM_GATHER(_f64_float, double,        double, READ_F64,  WRITE_F64,  float,  IS_ACTIVE_FLOAT, 0.0)
// f16 + bool/float
DEFINE_BINARY_JITUMM_GATHER(_f16_bool,  __half,        float,  READ_F16,  WRITE_F16,  int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITUMM_GATHER(_f16_float, __half,        float,  READ_F16,  WRITE_F16,  float,  IS_ACTIVE_FLOAT, 0.0f)
// bf16 + bool/float
DEFINE_BINARY_JITUMM_GATHER(_bf16_bool, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITUMM_GATHER(_bf16_float,__nv_bfloat16, float,  READ_BF16, WRITE_BF16, float,  IS_ACTIVE_FLOAT, 0.0f)

// =========================================================================
// Scatter kernel (corder=false): one thread per input row j
// For each active B[j, col], scatter Uniform(w) to connected output rows.
// =========================================================================

#define DEFINE_BINARY_JITUMM_SCATTER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, SPIKE_T, IS_ACTIVE, ATOMIC_ADD) \
__global__ void _binary_jitumm_scatter_kern##SUFFIX(                                                           \
    const WEIGHT_T* __restrict__ w_low,                                                                        \
    const WEIGHT_T* __restrict__ w_high,                                                                       \
    const float*    __restrict__ clen,                                                                         \
    const int*      __restrict__ seed,                                                                         \
    const SPIKE_T*  __restrict__ B,                                                                            \
    WEIGHT_T*       __restrict__ output,                                                                       \
    int m, int k, int n                                                                                        \
) {                                                                                                            \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                                                             \
    if (j >= k) return;                                                                                        \
    ACC_T wlo = READ_W(__ldg(&w_low[0]));                                                                      \
    ACC_T range = READ_W(__ldg(&w_high[0])) - wlo;                                                             \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                                           \
    if (cl < 2) cl = 2;                                                                                        \
    const SPIKE_T* b_row = B + (size_t)j * n;                                                                  \
    curandStatePhilox4_32_10_t state;                                                                          \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)j, 0ULL, &state);                     \
    unsigned int i = curand(&state) % cl;                                                                      \
    while (i < (unsigned int)m) {                                                                              \
        float u = curand_uniform(&state);                                                                      \
        ACC_T w = wlo + (ACC_T)u * range;                                                                      \
        WEIGHT_T* out_row = output + (size_t)i * n;                                                            \
        for (int col = 0; col < n; col++) {                                                                    \
            if (IS_ACTIVE(b_row, col)) {                                                                       \
                ATOMIC_ADD(&out_row[col], w);                                                                  \
            }                                                                                                  \
        }                                                                                                      \
        i += 1 + (curand(&state) % (cl - 1));                                                                  \
    }                                                                                                          \
}

// f32 + bool/float
DEFINE_BINARY_JITUMM_SCATTER(_f32_bool,  float,         float,  READ_F32,  WRITE_F32,  int8_t, IS_ACTIVE_BOOL,  atomicAdd_f32)
DEFINE_BINARY_JITUMM_SCATTER(_f32_float, float,         float,  READ_F32,  WRITE_F32,  float,  IS_ACTIVE_FLOAT, atomicAdd_f32)
// f64 + bool/float
DEFINE_BINARY_JITUMM_SCATTER(_f64_bool,  double,        double, READ_F64,  WRITE_F64,  int8_t, IS_ACTIVE_BOOL,  atomicAdd_f64)
DEFINE_BINARY_JITUMM_SCATTER(_f64_float, double,        double, READ_F64,  WRITE_F64,  float,  IS_ACTIVE_FLOAT, atomicAdd_f64)
// f16 + bool/float
DEFINE_BINARY_JITUMM_SCATTER(_f16_bool,  __half,        float,  READ_F16,  WRITE_F16,  int8_t, IS_ACTIVE_BOOL,  atomicAdd_f16)
DEFINE_BINARY_JITUMM_SCATTER(_f16_float, __half,        float,  READ_F16,  WRITE_F16,  float,  IS_ACTIVE_FLOAT, atomicAdd_f16)
// bf16 + bool/float
DEFINE_BINARY_JITUMM_SCATTER(_bf16_bool, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, int8_t, IS_ACTIVE_BOOL,  atomicAdd_bf16)
DEFINE_BINARY_JITUMM_SCATTER(_bf16_float,__nv_bfloat16, float,  READ_BF16, WRITE_BF16, float,  IS_ACTIVE_FLOAT, atomicAdd_bf16)

// ---- TVM FFI: binary_jitumm gather ----
// No memset needed: gather kernel zero-initializes output rows in-kernel.

#define FFI_BINARY_JITUMM_GATHER(SUFFIX, WEIGHT_C_T, SPIKE_C_T)    \
void binary_jitumm_gather##SUFFIX(                                 \
    tvm::ffi::TensorView w_low,                                    \
    tvm::ffi::TensorView w_high,                                   \
    tvm::ffi::TensorView clen,                                     \
    tvm::ffi::TensorView seed,                                     \
    tvm::ffi::TensorView B,                                        \
    tvm::ffi::TensorView output,                                   \
    int64_t stream                                                 \
) {                                                                \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);       \
    int m = static_cast<int>(output.size(0));                      \
    int n = static_cast<int>(output.size(1));                      \
    int k = static_cast<int>(B.size(0));                           \
    int threads = 256;                                             \
    int blocks = (m + threads - 1) / threads;                      \
    _binary_jitumm_gather_kern##SUFFIX<<<blocks, threads, 0, s>>>( \
        static_cast<const WEIGHT_C_T*>(w_low.data_ptr()),          \
        static_cast<const WEIGHT_C_T*>(w_high.data_ptr()),         \
        static_cast<const float*>(clen.data_ptr()),                \
        static_cast<const int*>(seed.data_ptr()),                  \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),               \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),               \
        m, k, n                                                    \
    );                                                             \
}

// @tvm_ffi binary_jitumm_gather_f32_bool
FFI_BINARY_JITUMM_GATHER(_f32_bool,  float,         int8_t)
// @tvm_ffi binary_jitumm_gather_f32_float
FFI_BINARY_JITUMM_GATHER(_f32_float, float,         float)
// @tvm_ffi binary_jitumm_gather_f64_bool
FFI_BINARY_JITUMM_GATHER(_f64_bool,  double,        int8_t)
// @tvm_ffi binary_jitumm_gather_f64_float
FFI_BINARY_JITUMM_GATHER(_f64_float, double,        float)
// @tvm_ffi binary_jitumm_gather_f16_bool
FFI_BINARY_JITUMM_GATHER(_f16_bool,  __half,        int8_t)
// @tvm_ffi binary_jitumm_gather_f16_float
FFI_BINARY_JITUMM_GATHER(_f16_float, __half,        float)
// @tvm_ffi binary_jitumm_gather_bf16_bool
FFI_BINARY_JITUMM_GATHER(_bf16_bool, __nv_bfloat16, int8_t)
// @tvm_ffi binary_jitumm_gather_bf16_float
FFI_BINARY_JITUMM_GATHER(_bf16_float,__nv_bfloat16, float)

// ---- TVM FFI: binary_jitumm scatter ----

#define FFI_BINARY_JITUMM_SCATTER(SUFFIX, WEIGHT_C_T, SPIKE_C_T)    \
void binary_jitumm_scatter##SUFFIX(                                 \
    tvm::ffi::TensorView w_low,                                     \
    tvm::ffi::TensorView w_high,                                    \
    tvm::ffi::TensorView clen,                                      \
    tvm::ffi::TensorView seed,                                      \
    tvm::ffi::TensorView B,                                         \
    tvm::ffi::TensorView output,                                    \
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
    _binary_jitumm_scatter_kern##SUFFIX<<<blocks, threads, 0, s>>>( \
        static_cast<const WEIGHT_C_T*>(w_low.data_ptr()),           \
        static_cast<const WEIGHT_C_T*>(w_high.data_ptr()),          \
        static_cast<const float*>(clen.data_ptr()),                 \
        static_cast<const int*>(seed.data_ptr()),                   \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                \
        m, k, n                                                     \
    );                                                              \
}

// @tvm_ffi binary_jitumm_scatter_f32_bool
FFI_BINARY_JITUMM_SCATTER(_f32_bool,  float,         int8_t)
// @tvm_ffi binary_jitumm_scatter_f32_float
FFI_BINARY_JITUMM_SCATTER(_f32_float, float,         float)
// @tvm_ffi binary_jitumm_scatter_f64_bool
FFI_BINARY_JITUMM_SCATTER(_f64_bool,  double,        int8_t)
// @tvm_ffi binary_jitumm_scatter_f64_float
FFI_BINARY_JITUMM_SCATTER(_f64_float, double,        float)
// @tvm_ffi binary_jitumm_scatter_f16_bool
FFI_BINARY_JITUMM_SCATTER(_f16_bool,  __half,        int8_t)
// @tvm_ffi binary_jitumm_scatter_f16_float
FFI_BINARY_JITUMM_SCATTER(_f16_float, __half,        float)
// @tvm_ffi binary_jitumm_scatter_bf16_bool
FFI_BINARY_JITUMM_SCATTER(_bf16_bool, __nv_bfloat16, int8_t)
// @tvm_ffi binary_jitumm_scatter_bf16_float
FFI_BINARY_JITUMM_SCATTER(_bf16_float,__nv_bfloat16, float)
