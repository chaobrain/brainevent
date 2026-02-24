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
 * float_jits.cu — JIT Scalar Dense Matrix Generation CUDA Kernels
 * ===============================================================
 *
 * Dense matrix generation for JIT scalar connectivity.
 *
 * Operation
 * ---------
 * jits — Dense matrix generation: M[i,j] = w * Bernoulli(prob)
 *
 * Parameters
 * ----------
 * weight : shape (1,), scalar weight for all connections
 * clen   : shape (1,), connection length = 2/prob (float32)
 * seed   : shape (1,), int32 random seed
 *
 * Geometric sampling pattern:
 *   curand_init(seed, thread_id, 0, &state);
 *   unsigned int i = curand(&state) % clen;
 *   while (i < dim) { process(i); i += 1 + (curand(&state) % (clen - 1)); }
 *
 * corder=True  (gather): one thread per output row, writes columns
 * corder=False (scatter): one thread per output column, writes rows
 *
 * Supported weight dtypes: float32, float64, float16, bfloat16.
 *
 * IMPORTANT: All data_ptr() returns are GPU device pointers — NEVER dereference on host.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include <cstdint>

// =========================================================================
// Per-dtype conversion macros
// =========================================================================

#define READ_F32(x)   (x)
#define WRITE_F32(x)  (x)

#define READ_F64(x)   (x)
#define WRITE_F64(x)  (x)

#define READ_F16(x)   __half2float(x)
#define WRITE_F16(x)  __float2half(x)

#define READ_BF16(x)  __bfloat162float(x)
#define WRITE_BF16(x) __float2bfloat16(x)


// #########################################################################
// ##  jits — Dense Matrix Generation                                     ##
// #########################################################################

// =========================================================================
// corder=true kernel: one thread per row
// output[i, col] = weight for connected columns, 0 otherwise.
// =========================================================================

#define DEFINE_JITS_CORDER_TRUE(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W)     \
__global__ void _jits_corder_true_kern##SUFFIX(                                \
    const WEIGHT_T* __restrict__ weight,                                       \
    const float*    __restrict__ clen,                                         \
    const int*      __restrict__ seed,                                         \
    WEIGHT_T*       __restrict__ output,                                       \
    int n_rows, int n_cols                                                     \
) {                                                                            \
    int row = blockIdx.x * blockDim.x + threadIdx.x;                           \
    if (row >= n_rows) return;                                                 \
    WEIGHT_T w = __ldg(&weight[0]);                                            \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                           \
    if (cl < 2) cl = 2;                                                        \
    curandStatePhilox4_32_10_t state;                                           \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)row, 0ULL, &state); \
    unsigned int col = curand(&state) % cl;                                     \
    while (col < (unsigned int)n_cols) {                                        \
        output[(size_t)row * n_cols + col] = w;                                 \
        col += 1 + (curand(&state) % (cl - 1));                                \
    }                                                                           \
}

DEFINE_JITS_CORDER_TRUE(_f32,  float,         float,  READ_F32,  WRITE_F32)
DEFINE_JITS_CORDER_TRUE(_f64,  double,        double, READ_F64,  WRITE_F64)
DEFINE_JITS_CORDER_TRUE(_f16,  __half,        float,  READ_F16,  WRITE_F16)
DEFINE_JITS_CORDER_TRUE(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16)

// =========================================================================
// corder=false kernel: one thread per column
// output[row, col] = weight for connected rows.
// =========================================================================

#define DEFINE_JITS_CORDER_FALSE(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W)    \
__global__ void _jits_corder_false_kern##SUFFIX(                               \
    const WEIGHT_T* __restrict__ weight,                                       \
    const float*    __restrict__ clen,                                         \
    const int*      __restrict__ seed,                                         \
    WEIGHT_T*       __restrict__ output,                                       \
    int n_rows, int n_cols                                                     \
) {                                                                            \
    int col = blockIdx.x * blockDim.x + threadIdx.x;                           \
    if (col >= n_cols) return;                                                 \
    WEIGHT_T w = __ldg(&weight[0]);                                            \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                           \
    if (cl < 2) cl = 2;                                                        \
    curandStatePhilox4_32_10_t state;                                           \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)col, 0ULL, &state); \
    unsigned int row = curand(&state) % cl;                                     \
    while (row < (unsigned int)n_rows) {                                        \
        output[(size_t)row * n_cols + col] = w;                                 \
        row += 1 + (curand(&state) % (cl - 1));                                \
    }                                                                           \
}

DEFINE_JITS_CORDER_FALSE(_f32,  float,         float,  READ_F32,  WRITE_F32)
DEFINE_JITS_CORDER_FALSE(_f64,  double,        double, READ_F64,  WRITE_F64)
DEFINE_JITS_CORDER_FALSE(_f16,  __half,        float,  READ_F16,  WRITE_F16)
DEFINE_JITS_CORDER_FALSE(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16)

// ---- TVM FFI: jits corder=true ----

#define FFI_JITS_CORDER_TRUE(SUFFIX, WEIGHT_C_T)                              \
void jits_corder_true##SUFFIX(                                                 \
    tvm::ffi::TensorView weight,                                               \
    tvm::ffi::TensorView clen,                                                 \
    tvm::ffi::TensorView seed,                                                 \
    tvm::ffi::TensorView output,                                               \
    int64_t stream                                                             \
) {                                                                            \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                   \
    int n_rows = static_cast<int>(output.size(0));                             \
    int n_cols = static_cast<int>(output.size(1));                             \
    cudaMemsetAsync(output.data_ptr(), 0,                                      \
        (size_t)n_rows * n_cols * sizeof(WEIGHT_C_T), s);                      \
    int threads = 256;                                                         \
    int blocks = (n_rows + threads - 1) / threads;                             \
    _jits_corder_true_kern##SUFFIX<<<blocks, threads, 0, s>>>(                 \
        static_cast<const WEIGHT_C_T*>(weight.data_ptr()),                     \
        static_cast<const float*>(clen.data_ptr()),                            \
        static_cast<const int*>(seed.data_ptr()),                              \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                           \
        n_rows, n_cols                                                         \
    );                                                                         \
}

// @tvm_ffi jits_corder_true_f32
FFI_JITS_CORDER_TRUE(_f32, float)
// @tvm_ffi jits_corder_true_f64
FFI_JITS_CORDER_TRUE(_f64, double)
// @tvm_ffi jits_corder_true_f16
FFI_JITS_CORDER_TRUE(_f16, __half)
// @tvm_ffi jits_corder_true_bf16
FFI_JITS_CORDER_TRUE(_bf16, __nv_bfloat16)

// ---- TVM FFI: jits corder=false ----

#define FFI_JITS_CORDER_FALSE(SUFFIX, WEIGHT_C_T)                             \
void jits_corder_false##SUFFIX(                                                \
    tvm::ffi::TensorView weight,                                               \
    tvm::ffi::TensorView clen,                                                 \
    tvm::ffi::TensorView seed,                                                 \
    tvm::ffi::TensorView output,                                               \
    int64_t stream                                                             \
) {                                                                            \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                   \
    int n_rows = static_cast<int>(output.size(0));                             \
    int n_cols = static_cast<int>(output.size(1));                             \
    cudaMemsetAsync(output.data_ptr(), 0,                                      \
        (size_t)n_rows * n_cols * sizeof(WEIGHT_C_T), s);                      \
    int threads = 256;                                                         \
    int blocks = (n_cols + threads - 1) / threads;                             \
    _jits_corder_false_kern##SUFFIX<<<blocks, threads, 0, s>>>(                \
        static_cast<const WEIGHT_C_T*>(weight.data_ptr()),                     \
        static_cast<const float*>(clen.data_ptr()),                            \
        static_cast<const int*>(seed.data_ptr()),                              \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                           \
        n_rows, n_cols                                                         \
    );                                                                         \
}

// @tvm_ffi jits_corder_false_f32
FFI_JITS_CORDER_FALSE(_f32, float)
// @tvm_ffi jits_corder_false_f64
FFI_JITS_CORDER_FALSE(_f64, double)
// @tvm_ffi jits_corder_false_f16
FFI_JITS_CORDER_FALSE(_f16, __half)
// @tvm_ffi jits_corder_false_bf16
FFI_JITS_CORDER_FALSE(_bf16, __nv_bfloat16)
