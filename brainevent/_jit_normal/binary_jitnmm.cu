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
 * binary_jitnmm.cu — Event-Driven JIT Normal Matrix-Matrix Product Kernels
 * =========================================================================
 *
 * Computes Y[i,:] = sum_{j active} Normal(w_loc, w_scale) * B[j,:].
 * The connectivity matrix M[i,j] is never materialised; weights are generated
 * on the fly only for connected (i, j) pairs.
 *
 * TVM FFI entry points
 * --------------------
 * binary_jitnmm_gather_{f32,f64,f16,bf16}_{bool,float}
 *     — gather (corder=True):  one thread per output row i
 * binary_jitnmm_scatter_{f32,f64,f16,bf16}_{bool,float}
 *     — scatter (corder=False): one thread per input row j
 *
 * Parameters (common)
 * -------------------
 * w_loc  : shape (1,), mean of normal weight distribution
 * w_scale: shape (1,), std dev of normal weight distribution
 * clen   : shape (1,), connection length = ceil(2/prob) (float32)
 * seed   : shape (1,), int32 random seed
 * B      : shape (k, n), right-hand spike/event matrix (bool/int8_t or float)
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
#include "curand_common.h"

// #########################################################################
// ##  binary_jitnmm — Event-Driven Matrix-Matrix Product                 ##
// #########################################################################

// --- Gather kernel with register accumulators (n<=32) and __ldg ---
#define DEFINE_BINARY_JITNMM_GATHER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, RNG_FUNC, SPIKE_T, IS_ACTIVE, ACC_ZERO) \
__global__ void _binary_jitnmm_gather_kern##SUFFIX(                                                                   \
    const WEIGHT_T* __restrict__ w_loc,                                                                               \
    const WEIGHT_T* __restrict__ w_scale,                                                                             \
    const float*    __restrict__ clen,                                                                                \
    const int*      __restrict__ seed,                                                                                \
    const SPIKE_T*  __restrict__ B,                                                                                   \
    WEIGHT_T*       __restrict__ output,                                                                              \
    int m, int k, int n                                                                                               \
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
    WEIGHT_T* out_row = output + (size_t)i * n;                                                                       \
    if (n <= 32) {                                                                                                    \
        ACC_T acc[32];                                                                                                \
        for (int c = 0; c < n; c++) acc[c] = ACC_ZERO;                                                                \
        while (j < (unsigned int)k) {                                                                                 \
            ACC_T num = (ACC_T)RNG_FUNC(&state);                                                                      \
            ACC_T w = loc + num * scale;                                                                              \
            const SPIKE_T* b_row = B + (size_t)j * n;                                                                 \
            for (int col = 0; col < n; col++) {                                                                       \
                if (IS_ACTIVE(b_row[col])) {                                                                          \
                    acc[col] += w;                                                                                    \
                }                                                                                                     \
            }                                                                                                         \
            j += 1 + (curand(&state) % (cl - 1));                                                                     \
        }                                                                                                             \
        for (int c = 0; c < n; c++) out_row[c] = WRITE_W(acc[c]);                                                     \
    } else {                                                                                                          \
        while (j < (unsigned int)k) {                                                                                 \
            ACC_T num = (ACC_T)RNG_FUNC(&state);                                                                      \
            ACC_T w = loc + num * scale;                                                                              \
            const SPIKE_T* b_row = B + (size_t)j * n;                                                                 \
            for (int col = 0; col < n; col++) {                                                                       \
                if (IS_ACTIVE(b_row[col])) {                                                                          \
                    ACC_T cur = READ_W(out_row[col]);                                                                 \
                    out_row[col] = WRITE_W(cur + w);                                                                  \
                }                                                                                                     \
            }                                                                                                         \
            j += 1 + (curand(&state) % (cl - 1));                                                                     \
        }                                                                                                             \
    }                                                                                                                 \
}

DEFINE_BINARY_JITNMM_GATHER(_f32_bool,   float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITNMM_GATHER(_f32_float,  float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, float,  IS_ACTIVE_FLOAT, 0.0f)
DEFINE_BINARY_JITNMM_GATHER(_f64_bool,   double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, int8_t, IS_ACTIVE_BOOL,  0.0)
DEFINE_BINARY_JITNMM_GATHER(_f64_float,  double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, float,  IS_ACTIVE_FLOAT, 0.0)
DEFINE_BINARY_JITNMM_GATHER(_f16_bool,   __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITNMM_GATHER(_f16_float,  __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, float,  IS_ACTIVE_FLOAT, 0.0f)
DEFINE_BINARY_JITNMM_GATHER(_bf16_bool,  __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITNMM_GATHER(_bf16_float, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, float,  IS_ACTIVE_FLOAT, 0.0f)

// --- Scatter kernel: __ldg on parameters ---
#define DEFINE_BINARY_JITNMM_SCATTER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, RNG_FUNC, SPIKE_T, IS_ACTIVE, ATOMIC_ADD) \
__global__ void _binary_jitnmm_scatter_kern##SUFFIX(                                                                     \
    const WEIGHT_T* __restrict__ w_loc,                                                                                  \
    const WEIGHT_T* __restrict__ w_scale,                                                                                \
    const float*    __restrict__ clen,                                                                                   \
    const int*      __restrict__ seed,                                                                                   \
    const SPIKE_T*  __restrict__ B,                                                                                      \
    WEIGHT_T*       __restrict__ output,                                                                                 \
    int m, int k, int n                                                                                                  \
) {                                                                                                                      \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                                                                       \
    if (j >= k) return;                                                                                                  \
    ACC_T loc = READ_W(__ldg(&w_loc[0]));                                                                                \
    ACC_T scale = READ_W(__ldg(&w_scale[0]));                                                                            \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                                                     \
    if (cl < 2) cl = 2;                                                                                                  \
    const SPIKE_T* b_row = B + (size_t)j * n;                                                                            \
    curandStatePhilox4_32_10_t state;                                                                                    \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)j, 0ULL, &state);                               \
    unsigned int i = curand(&state) % cl;                                                                                \
    while (i < (unsigned int)m) {                                                                                        \
        ACC_T num = (ACC_T)RNG_FUNC(&state);                                                                             \
        ACC_T w = loc + num * scale;                                                                                     \
        WEIGHT_T* out_row = output + (size_t)i * n;                                                                      \
        for (int col = 0; col < n; col++) {                                                                              \
            if (IS_ACTIVE(b_row[col])) {                                                                                 \
                ATOMIC_ADD(&out_row[col], w);                                                                            \
            }                                                                                                            \
        }                                                                                                                \
        i += 1 + (curand(&state) % (cl - 1));                                                                            \
    }                                                                                                                    \
}

DEFINE_BINARY_JITNMM_SCATTER(_f32_bool,   float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, int8_t, IS_ACTIVE_BOOL,  atomic_add_f32)
DEFINE_BINARY_JITNMM_SCATTER(_f32_float,  float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, float,  IS_ACTIVE_FLOAT, atomic_add_f32)
DEFINE_BINARY_JITNMM_SCATTER(_f64_bool,   double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, int8_t, IS_ACTIVE_BOOL,  atomic_add_f64)
DEFINE_BINARY_JITNMM_SCATTER(_f64_float,  double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, float,  IS_ACTIVE_FLOAT, atomic_add_f64)
DEFINE_BINARY_JITNMM_SCATTER(_f16_bool,   __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, int8_t, IS_ACTIVE_BOOL,  atomic_add_f16)
DEFINE_BINARY_JITNMM_SCATTER(_f16_float,  __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, float,  IS_ACTIVE_FLOAT, atomic_add_f16)
DEFINE_BINARY_JITNMM_SCATTER(_bf16_bool,  __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, int8_t, IS_ACTIVE_BOOL,  atomic_add_bf16)
DEFINE_BINARY_JITNMM_SCATTER(_bf16_float, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, float,  IS_ACTIVE_FLOAT, atomic_add_bf16)

#define FFI_BINARY_JITNMM_GATHER(SUFFIX, WEIGHT_C_T, SPIKE_C_T)    \
void binary_jitnmm_gather##SUFFIX(                                 \
    tvm::ffi::TensorView w_loc,                                    \
    tvm::ffi::TensorView w_scale,                                  \
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
    cudaMemsetAsync(output.data_ptr(), 0,                          \
        (size_t)m * n * sizeof(WEIGHT_C_T), s);                    \
    int threads = 256;                                             \
    int blocks = (m + threads - 1) / threads;                      \
    _binary_jitnmm_gather_kern##SUFFIX<<<blocks, threads, 0, s>>>( \
        static_cast<const WEIGHT_C_T*>(w_loc.data_ptr()),          \
        static_cast<const WEIGHT_C_T*>(w_scale.data_ptr()),        \
        static_cast<const float*>(clen.data_ptr()),                \
        static_cast<const int*>(seed.data_ptr()),                  \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),               \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),               \
        m, k, n                                                    \
    );                                                             \
}

// @tvm_ffi binary_jitnmm_gather_f32_bool
FFI_BINARY_JITNMM_GATHER(_f32_bool,   float,         int8_t)
// @tvm_ffi binary_jitnmm_gather_f32_float
FFI_BINARY_JITNMM_GATHER(_f32_float,  float,         float)
// @tvm_ffi binary_jitnmm_gather_f64_bool
FFI_BINARY_JITNMM_GATHER(_f64_bool,   double,        int8_t)
// @tvm_ffi binary_jitnmm_gather_f64_float
FFI_BINARY_JITNMM_GATHER(_f64_float,  double,        float)
// @tvm_ffi binary_jitnmm_gather_f16_bool
FFI_BINARY_JITNMM_GATHER(_f16_bool,   __half,        int8_t)
// @tvm_ffi binary_jitnmm_gather_f16_float
FFI_BINARY_JITNMM_GATHER(_f16_float,  __half,        float)
// @tvm_ffi binary_jitnmm_gather_bf16_bool
FFI_BINARY_JITNMM_GATHER(_bf16_bool,  __nv_bfloat16, int8_t)
// @tvm_ffi binary_jitnmm_gather_bf16_float
FFI_BINARY_JITNMM_GATHER(_bf16_float, __nv_bfloat16, float)

#define FFI_BINARY_JITNMM_SCATTER(SUFFIX, WEIGHT_C_T, SPIKE_C_T)    \
void binary_jitnmm_scatter##SUFFIX(                                 \
    tvm::ffi::TensorView w_loc,                                     \
    tvm::ffi::TensorView w_scale,                                   \
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
    _binary_jitnmm_scatter_kern##SUFFIX<<<blocks, threads, 0, s>>>( \
        static_cast<const WEIGHT_C_T*>(w_loc.data_ptr()),           \
        static_cast<const WEIGHT_C_T*>(w_scale.data_ptr()),         \
        static_cast<const float*>(clen.data_ptr()),                 \
        static_cast<const int*>(seed.data_ptr()),                   \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                \
        m, k, n                                                     \
    );                                                              \
}

// @tvm_ffi binary_jitnmm_scatter_f32_bool
FFI_BINARY_JITNMM_SCATTER(_f32_bool,   float,         int8_t)
// @tvm_ffi binary_jitnmm_scatter_f32_float
FFI_BINARY_JITNMM_SCATTER(_f32_float,  float,         float)
// @tvm_ffi binary_jitnmm_scatter_f64_bool
FFI_BINARY_JITNMM_SCATTER(_f64_bool,   double,        int8_t)
// @tvm_ffi binary_jitnmm_scatter_f64_float
FFI_BINARY_JITNMM_SCATTER(_f64_float,  double,        float)
// @tvm_ffi binary_jitnmm_scatter_f16_bool
FFI_BINARY_JITNMM_SCATTER(_f16_bool,   __half,        int8_t)
// @tvm_ffi binary_jitnmm_scatter_f16_float
FFI_BINARY_JITNMM_SCATTER(_f16_float,  __half,        float)
// @tvm_ffi binary_jitnmm_scatter_bf16_bool
FFI_BINARY_JITNMM_SCATTER(_bf16_bool,  __nv_bfloat16, int8_t)
// @tvm_ffi binary_jitnmm_scatter_bf16_float
FFI_BINARY_JITNMM_SCATTER(_bf16_float, __nv_bfloat16, float)
