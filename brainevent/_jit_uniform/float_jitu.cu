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
 * float_jitu.cu — JIT Uniform Dense Matrix Generation (jitu operator)
 * ====================================================================
 *
 * Generates a dense random connectivity matrix where each entry W[i,j] is
 * independently drawn from Uniform(w_low, w_high) with probability prob,
 * and zero otherwise. Connectivity pattern is determined by a geometric
 * skip seeded by `seed`.
 *
 * Operations
 * ----------
 * jitu_corder_true_{f32,f64,f16,bf16}  — corder=True  (row-major, one thread per row)
 * jitu_corder_false_{f32,f64,f16,bf16} — corder=False (col-major, one thread per col)
 *
 * Parameters
 * ----------
 * w_low   : shape (1,), lower bound of uniform weight distribution
 * w_high  : shape (1,), upper bound of uniform weight distribution
 * clen    : shape (1,), connection length = ceil(2/prob) (float32)
 * seed    : shape (1,), int32 random seed
 * output  : shape (n_rows, n_cols), output matrix
 *
 * IMPORTANT: All data_ptr() returns are GPU device pointers — NEVER dereference on host.
 */

#include "cuda_common.h"
#include "brainevent/common.h"
#include "curand_common.h"


// #########################################################################
// ##  jitu — Dense Matrix Generation                                     ##
// #########################################################################

// =========================================================================
// corder=true kernel: one thread per row
// output[row, col] = Uniform(w_low, w_high) for connected (row, col) pairs.
// =========================================================================

#define DEFINE_JITU_CORDER_TRUE(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W)                    \
__global__ void _jitu_corder_true_kern##SUFFIX(                                              \
    const WEIGHT_T* __restrict__ w_low,                                                      \
    const WEIGHT_T* __restrict__ w_high,                                                     \
    const float*    __restrict__ clen,                                                       \
    const int*      __restrict__ seed,                                                       \
    WEIGHT_T*       __restrict__ output,                                                     \
    int n_rows, int n_cols                                                                   \
) {                                                                                          \
    int row = blockIdx.x * blockDim.x + threadIdx.x;                                         \
    if (row >= n_rows) return;                                                               \
    ACC_T wlo = READ_W(__ldg(&w_low[0]));                                                    \
    ACC_T range = READ_W(__ldg(&w_high[0])) - wlo;                                           \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                         \
    if (cl < 2) cl = 2;                                                                      \
    curandStatePhilox4_32_10_t state;                                                        \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)row, 0ULL, &state); \
    unsigned int col = curand(&state) % cl;                                                  \
    while (col < (unsigned int)n_cols) {                                                     \
        float u = curand_uniform(&state);                                                    \
        output[(size_t)row * n_cols + col] = WRITE_W(wlo + (ACC_T)u * range);                \
        col += 1 + (curand(&state) % (cl - 1));                                              \
    }                                                                                        \
}

DEFINE_JITU_CORDER_TRUE(_f32,  float,         float,  READ_F32,  WRITE_F32)
DEFINE_JITU_CORDER_TRUE(_f64,  double,        double, READ_F64,  WRITE_F64)
DEFINE_JITU_CORDER_TRUE(_f16,  __half,        float,  READ_F16,  WRITE_F16)
DEFINE_JITU_CORDER_TRUE(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16)

// =========================================================================
// corder=false kernel: one thread per column
// output[row, col] = Uniform(w_low, w_high) for connected (row, col) pairs.
// =========================================================================

#define DEFINE_JITU_CORDER_FALSE(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W)                   \
__global__ void _jitu_corder_false_kern##SUFFIX(                                             \
    const WEIGHT_T* __restrict__ w_low,                                                      \
    const WEIGHT_T* __restrict__ w_high,                                                     \
    const float*    __restrict__ clen,                                                       \
    const int*      __restrict__ seed,                                                       \
    WEIGHT_T*       __restrict__ output,                                                     \
    int n_rows, int n_cols                                                                   \
) {                                                                                          \
    int col = blockIdx.x * blockDim.x + threadIdx.x;                                         \
    if (col >= n_cols) return;                                                               \
    ACC_T wlo = READ_W(__ldg(&w_low[0]));                                                    \
    ACC_T range = READ_W(__ldg(&w_high[0])) - wlo;                                           \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                         \
    if (cl < 2) cl = 2;                                                                      \
    curandStatePhilox4_32_10_t state;                                                        \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)col, 0ULL, &state); \
    unsigned int row = curand(&state) % cl;                                                  \
    while (row < (unsigned int)n_rows) {                                                     \
        float u = curand_uniform(&state);                                                    \
        output[(size_t)row * n_cols + col] = WRITE_W(wlo + (ACC_T)u * range);                \
        row += 1 + (curand(&state) % (cl - 1));                                              \
    }                                                                                        \
}

DEFINE_JITU_CORDER_FALSE(_f32,  float,         float,  READ_F32,  WRITE_F32)
DEFINE_JITU_CORDER_FALSE(_f64,  double,        double, READ_F64,  WRITE_F64)
DEFINE_JITU_CORDER_FALSE(_f16,  __half,        float,  READ_F16,  WRITE_F16)
DEFINE_JITU_CORDER_FALSE(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16)

// ---- CUDA: jitu corder=true ----

#define FFI_JITU_CORDER_TRUE(SUFFIX, WEIGHT_C_T)               \
void jitu_corder_true##SUFFIX(                                 \
    const BE::Tensor w_low,                                \
    const BE::Tensor w_high,                               \
    const BE::Tensor clen,                                 \
    const BE::Tensor seed,                                 \
    BE::Tensor output,                               \
    int64_t stream                                             \
) {                                                            \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);   \
    int n_rows = static_cast<int>(output.size(0));             \
    int n_cols = static_cast<int>(output.size(1));             \
    cudaMemsetAsync(output.data_ptr(), 0,                      \
        (size_t)n_rows * n_cols * sizeof(WEIGHT_C_T), s);      \
    int threads = 256;                                         \
    int blocks = (n_rows + threads - 1) / threads;             \
    _jitu_corder_true_kern##SUFFIX<<<blocks, threads, 0, s>>>( \
        static_cast<const WEIGHT_C_T*>(w_low.data_ptr()),      \
        static_cast<const WEIGHT_C_T*>(w_high.data_ptr()),     \
        static_cast<const float*>(clen.data_ptr()),            \
        static_cast<const int*>(seed.data_ptr()),              \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),           \
        n_rows, n_cols                                         \
    );                                                         \
}

// @BE jitu_corder_true_f32
FFI_JITU_CORDER_TRUE(_f32, float)
// @BE jitu_corder_true_f64
FFI_JITU_CORDER_TRUE(_f64, double)
// @BE jitu_corder_true_f16
FFI_JITU_CORDER_TRUE(_f16, __half)
// @BE jitu_corder_true_bf16
FFI_JITU_CORDER_TRUE(_bf16, __nv_bfloat16)

// ---- CUDA: jitu corder=false ----

#define FFI_JITU_CORDER_FALSE(SUFFIX, WEIGHT_C_T)               \
void jitu_corder_false##SUFFIX(                                 \
    const BE::Tensor w_low,                                 \
    const BE::Tensor w_high,                                \
    const BE::Tensor clen,                                  \
    const BE::Tensor seed,                                  \
    BE::Tensor output,                                \
    int64_t stream                                              \
) {                                                             \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);    \
    int n_rows = static_cast<int>(output.size(0));              \
    int n_cols = static_cast<int>(output.size(1));              \
    cudaMemsetAsync(output.data_ptr(), 0,                       \
        (size_t)n_rows * n_cols * sizeof(WEIGHT_C_T), s);       \
    int threads = 256;                                          \
    int blocks = (n_cols + threads - 1) / threads;              \
    _jitu_corder_false_kern##SUFFIX<<<blocks, threads, 0, s>>>( \
        static_cast<const WEIGHT_C_T*>(w_low.data_ptr()),       \
        static_cast<const WEIGHT_C_T*>(w_high.data_ptr()),      \
        static_cast<const float*>(clen.data_ptr()),             \
        static_cast<const int*>(seed.data_ptr()),               \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),            \
        n_rows, n_cols                                          \
    );                                                          \
}

// @BE jitu_corder_false_f32
FFI_JITU_CORDER_FALSE(_f32, float)
// @BE jitu_corder_false_f64
FFI_JITU_CORDER_FALSE(_f64, double)
// @BE jitu_corder_false_f16
FFI_JITU_CORDER_FALSE(_f16, __half)
// @BE jitu_corder_false_bf16
FFI_JITU_CORDER_FALSE(_bf16, __nv_bfloat16)
