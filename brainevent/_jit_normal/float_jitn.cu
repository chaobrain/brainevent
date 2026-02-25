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
 * float_jitn.cu — JIT Normal Dense Matrix Generation Kernels
 * ===========================================================
 *
 * Generates a dense matrix M[i,j] = Normal(w_loc, w_scale) * Bernoulli(prob).
 * Connectivity pattern is determined by a geometric skip seeded by `seed`.
 *
 * TVM FFI entry points
 * --------------------
 * jitn_corder_true_{f32,f64,f16,bf16}   — row-parallel gather (corder=True)
 * jitn_corder_false_{f32,f64,f16,bf16}  — col-parallel scatter (corder=False)
 *
 * Parameters (common)
 * -------------------
 * w_loc  : shape (1,), mean of normal weight distribution
 * w_scale: shape (1,), std dev of normal weight distribution
 * clen   : shape (1,), connection length = ceil(2/prob) (float32)
 * seed   : shape (1,), int32 random seed
 * output : shape (n_rows, n_cols), output matrix (zeroed before writing)
 *
 * IMPORTANT: All data_ptr() returns are GPU device pointers — NEVER dereference on host.
 */

#include "cuda_common.h"
#include "curand_common.h"

// #########################################################################
// ##  jitn — Dense Matrix Generation                                      ##
// #########################################################################

#define DEFINE_JITN_CORDER_TRUE(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, RNG_FUNC)          \
__global__ void _jitn_corder_true_kern##SUFFIX(                                              \
    const WEIGHT_T* __restrict__ w_loc,                                                      \
    const WEIGHT_T* __restrict__ w_scale,                                                    \
    const float*    __restrict__ clen,                                                       \
    const int*      __restrict__ seed,                                                       \
    WEIGHT_T*       __restrict__ output,                                                     \
    int n_rows, int n_cols                                                                   \
) {                                                                                          \
    int row = blockIdx.x * blockDim.x + threadIdx.x;                                         \
    if (row >= n_rows) return;                                                               \
    ACC_T loc = READ_W(__ldg(&w_loc[0]));                                                    \
    ACC_T scale = READ_W(__ldg(&w_scale[0]));                                                \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                         \
    if (cl < 2) cl = 2;                                                                      \
    curandStatePhilox4_32_10_t state;                                                        \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)row, 0ULL, &state); \
    unsigned int col = curand(&state) % cl;                                                  \
    while (col < (unsigned int)n_cols) {                                                     \
        ACC_T n = (ACC_T)RNG_FUNC(&state);                                                   \
        output[(size_t)row * n_cols + col] = WRITE_W(loc + n * scale);                       \
        col += 1 + (curand(&state) % (cl - 1));                                              \
    }                                                                                        \
}

DEFINE_JITN_CORDER_TRUE(_f32,  float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32)
DEFINE_JITN_CORDER_TRUE(_f64,  double,        double, READ_F64,  WRITE_F64,  curand_normal_f64)
DEFINE_JITN_CORDER_TRUE(_f16,  __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32)
DEFINE_JITN_CORDER_TRUE(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32)

#define DEFINE_JITN_CORDER_FALSE(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, RNG_FUNC)         \
__global__ void _jitn_corder_false_kern##SUFFIX(                                             \
    const WEIGHT_T* __restrict__ w_loc,                                                      \
    const WEIGHT_T* __restrict__ w_scale,                                                    \
    const float*    __restrict__ clen,                                                       \
    const int*      __restrict__ seed,                                                       \
    WEIGHT_T*       __restrict__ output,                                                     \
    int n_rows, int n_cols                                                                   \
) {                                                                                          \
    int col = blockIdx.x * blockDim.x + threadIdx.x;                                         \
    if (col >= n_cols) return;                                                               \
    ACC_T loc = READ_W(__ldg(&w_loc[0]));                                                    \
    ACC_T scale = READ_W(__ldg(&w_scale[0]));                                                \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                         \
    if (cl < 2) cl = 2;                                                                      \
    curandStatePhilox4_32_10_t state;                                                        \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)col, 0ULL, &state); \
    unsigned int row = curand(&state) % cl;                                                  \
    while (row < (unsigned int)n_rows) {                                                     \
        ACC_T n = (ACC_T)RNG_FUNC(&state);                                                   \
        output[(size_t)row * n_cols + col] = WRITE_W(loc + n * scale);                       \
        row += 1 + (curand(&state) % (cl - 1));                                              \
    }                                                                                        \
}

DEFINE_JITN_CORDER_FALSE(_f32,  float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32)
DEFINE_JITN_CORDER_FALSE(_f64,  double,        double, READ_F64,  WRITE_F64,  curand_normal_f64)
DEFINE_JITN_CORDER_FALSE(_f16,  __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32)
DEFINE_JITN_CORDER_FALSE(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32)

#define FFI_JITN_CORDER_TRUE(SUFFIX, WEIGHT_C_T)               \
void jitn_corder_true##SUFFIX(                                 \
    tvm::ffi::TensorView w_loc,                                \
    tvm::ffi::TensorView w_scale,                              \
    tvm::ffi::TensorView clen,                                 \
    tvm::ffi::TensorView seed,                                 \
    tvm::ffi::TensorView output,                               \
    int64_t stream                                             \
) {                                                            \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);   \
    int n_rows = static_cast<int>(output.size(0));             \
    int n_cols = static_cast<int>(output.size(1));             \
    cudaMemsetAsync(output.data_ptr(), 0,                      \
        (size_t)n_rows * n_cols * sizeof(WEIGHT_C_T), s);      \
    int threads = 256;                                         \
    int blocks = (n_rows + threads - 1) / threads;             \
    _jitn_corder_true_kern##SUFFIX<<<blocks, threads, 0, s>>>( \
        static_cast<const WEIGHT_C_T*>(w_loc.data_ptr()),      \
        static_cast<const WEIGHT_C_T*>(w_scale.data_ptr()),    \
        static_cast<const float*>(clen.data_ptr()),            \
        static_cast<const int*>(seed.data_ptr()),              \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),           \
        n_rows, n_cols                                         \
    );                                                         \
}

// @tvm_ffi jitn_corder_true_f32
FFI_JITN_CORDER_TRUE(_f32, float)
// @tvm_ffi jitn_corder_true_f64
FFI_JITN_CORDER_TRUE(_f64, double)
// @tvm_ffi jitn_corder_true_f16
FFI_JITN_CORDER_TRUE(_f16, __half)
// @tvm_ffi jitn_corder_true_bf16
FFI_JITN_CORDER_TRUE(_bf16, __nv_bfloat16)

#define FFI_JITN_CORDER_FALSE(SUFFIX, WEIGHT_C_T)               \
void jitn_corder_false##SUFFIX(                                 \
    tvm::ffi::TensorView w_loc,                                 \
    tvm::ffi::TensorView w_scale,                               \
    tvm::ffi::TensorView clen,                                  \
    tvm::ffi::TensorView seed,                                  \
    tvm::ffi::TensorView output,                                \
    int64_t stream                                              \
) {                                                             \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);    \
    int n_rows = static_cast<int>(output.size(0));              \
    int n_cols = static_cast<int>(output.size(1));              \
    cudaMemsetAsync(output.data_ptr(), 0,                       \
        (size_t)n_rows * n_cols * sizeof(WEIGHT_C_T), s);       \
    int threads = 256;                                          \
    int blocks = (n_cols + threads - 1) / threads;              \
    _jitn_corder_false_kern##SUFFIX<<<blocks, threads, 0, s>>>( \
        static_cast<const WEIGHT_C_T*>(w_loc.data_ptr()),       \
        static_cast<const WEIGHT_C_T*>(w_scale.data_ptr()),     \
        static_cast<const float*>(clen.data_ptr()),             \
        static_cast<const int*>(seed.data_ptr()),               \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),            \
        n_rows, n_cols                                          \
    );                                                          \
}

// @tvm_ffi jitn_corder_false_f32
FFI_JITN_CORDER_FALSE(_f32, float)
// @tvm_ffi jitn_corder_false_f64
FFI_JITN_CORDER_FALSE(_f64, double)
// @tvm_ffi jitn_corder_false_f16
FFI_JITN_CORDER_FALSE(_f16, __half)
// @tvm_ffi jitn_corder_false_bf16
FFI_JITN_CORDER_FALSE(_bf16, __nv_bfloat16)
