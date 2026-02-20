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
 * jit_normal.cu — JIT Normal Connectivity CUDA Kernels (all operations)
 * =========================================================================
 *
 * Unified CUDA kernel file for all JIT normal connectivity operations.
 * Each entry W[i,j] is independently set to a value drawn from
 * Normal(w_loc, w_scale) with probability `prob`, and zero otherwise.
 * Connectivity is determined by a geometric skip pattern seeded by `seed`.
 *
 * Operations
 * ----------
 * 1. jitn          — Dense matrix generation: M[i,j] = Normal(w_loc,w_scale)*Bernoulli(prob)
 * 2. jitnmv        — Float matrix-vector:     y = M @ v
 * 3. jitnmm        — Float matrix-matrix:     Y = M @ B
 * 4. binary_jitnmv — Event-driven mat-vec:    y[i] = sum_{j active} Normal()*B[i,j]
 * 5. binary_jitnmm — Event-driven mat-mat:    Y[i,:] = sum_{j active} Normal()*B[i,k]
 *
 * Parameters (common to all)
 * --------------------------
 * w_loc  : shape (1,), mean of normal weight distribution
 * w_scale: shape (1,), std dev of normal weight distribution
 * clen   : shape (1,), connection length = ceil(2/prob) (float32)
 * seed   : shape (1,), int32 random seed
 *
 * Random sampling pattern (shared by all kernels):
 *   curand_init(seed, thread_id, 0, &state);
 *   unsigned int i = curand(&state) % clen;        // first connected index
 *   while (i < dim) {
 *       float n = curand_normal(&state);            // weight sample N(0,1)
 *       float w = w_loc + n * w_scale;             // scale to N(loc, scale)
 *       process(i, w);
 *       i += 1 + (curand(&state) % (clen - 1));    // geometric skip to next
 *   }
 *
 * corder=True  (gather): one thread per output element, no atomics
 * corder=False (scatter): one thread per input element, uses atomicAdd
 *
 * Supported weight dtypes: float32, float64, float16, bfloat16.
 * Half-precision accumulates in float32 for numerical stability.
 * Binary kernels support bool (int8_t) and float spike types.
 *
 * IMPORTANT: All data_ptr() returns are GPU device pointers — NEVER dereference on host.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include <cstdint>

// =========================================================================
// Per-dtype read/write conversion macros
// =========================================================================

#define READ_F32(x)   (x)
#define WRITE_F32(x)  (x)

#define READ_F64(x)   (x)
#define WRITE_F64(x)  (x)

#define READ_F16(x)   __half2float(x)
#define WRITE_F16(x)  __float2half(x)

#define READ_BF16(x)  __bfloat162float(x)
#define WRITE_BF16(x) __float2bfloat16(x)

// =========================================================================
// Spike activity checks (for binary kernels)
// =========================================================================

#define IS_ACTIVE_BOOL(v, j)  ((v)[j] != 0)
#define IS_ACTIVE_FLOAT(v, j) ((v)[j] > 0.0f)

// =========================================================================
// atomicAdd helpers for f16/bf16 (CAS-based for pre-Volta GPUs)
// =========================================================================

__device__ __inline__ void atomicAdd_f32(float* addr, float val) {
    atomicAdd(addr, val);
}

__device__ __inline__ void atomicAdd_f64(double* addr, double val) {
    atomicAdd(addr, val);
}

__device__ __inline__ void atomicAdd_f16(__half* addr, float val) {
    unsigned short int* addr_as_usi = (unsigned short int*)addr;
    unsigned short int old = *addr_as_usi;
    unsigned short int assumed;
    do {
        assumed = old;
        float old_f = __half2float(*reinterpret_cast<const __half*>(&assumed));
        __half new_h = __float2half(old_f + val);
        unsigned short int new_val = *reinterpret_cast<unsigned short int*>(&new_h);
        old = atomicCAS(addr_as_usi, assumed, new_val);
    } while (assumed != old);
}

__device__ __inline__ void atomicAdd_bf16(__nv_bfloat16* addr, float val) {
    unsigned short int* addr_as_usi = (unsigned short int*)addr;
    unsigned short int old = *addr_as_usi;
    unsigned short int assumed;
    do {
        assumed = old;
        float old_f = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&assumed));
        __nv_bfloat16 new_h = __float2bfloat16(old_f + val);
        unsigned short int new_val = *reinterpret_cast<unsigned short int*>(&new_h);
        old = atomicCAS(addr_as_usi, assumed, new_val);
    } while (assumed != old);
}

// =========================================================================
// RNG Helpers for Normal Distribution
// =========================================================================

__device__ __inline__ float curand_normal_f32(curandStatePhilox4_32_10_t* state) {
    return curand_normal(state);
}

__device__ __inline__ double curand_normal_f64(curandStatePhilox4_32_10_t* state) {
    return curand_normal_double(state);
}

// #########################################################################
// ##  1. jitn — Dense Matrix Generation                                  ##
// #########################################################################

#define DEFINE_JITN_CORDER_TRUE(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, RNG_FUNC) \
__global__ void _jitn_corder_true_kern##SUFFIX(                                \
    const WEIGHT_T* __restrict__ w_loc,                                        \
    const WEIGHT_T* __restrict__ w_scale,                                      \
    const float*    __restrict__ clen,                                         \
    const int*      __restrict__ seed,                                         \
    WEIGHT_T*       __restrict__ output,                                       \
    int n_rows, int n_cols                                                     \
) {                                                                            \
    int row = blockIdx.x * blockDim.x + threadIdx.x;                           \
    if (row >= n_rows) return;                                                 \
    ACC_T loc = READ_W(w_loc[0]);                                              \
    ACC_T scale = READ_W(w_scale[0]);                                          \
    unsigned int cl = (unsigned int)clen[0];                                    \
    if (cl < 2) cl = 2;                                                        \
    curandStatePhilox4_32_10_t state;                                           \
    curand_init((unsigned long long)seed[0], (unsigned long long)row, 0ULL, &state); \
    unsigned int col = curand(&state) % cl;                                     \
    while (col < (unsigned int)n_cols) {                                        \
        ACC_T n = (ACC_T)RNG_FUNC(&state);                                     \
        output[(size_t)row * n_cols + col] = WRITE_W(loc + n * scale);         \
        col += 1 + (curand(&state) % (cl - 1));                                \
    }                                                                           \
}

DEFINE_JITN_CORDER_TRUE(_f32,  float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32)
DEFINE_JITN_CORDER_TRUE(_f64,  double,        double, READ_F64,  WRITE_F64,  curand_normal_f64)
DEFINE_JITN_CORDER_TRUE(_f16,  __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32)
DEFINE_JITN_CORDER_TRUE(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32)

#define DEFINE_JITN_CORDER_FALSE(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, RNG_FUNC) \
__global__ void _jitn_corder_false_kern##SUFFIX(                               \
    const WEIGHT_T* __restrict__ w_loc,                                        \
    const WEIGHT_T* __restrict__ w_scale,                                      \
    const float*    __restrict__ clen,                                         \
    const int*      __restrict__ seed,                                         \
    WEIGHT_T*       __restrict__ output,                                       \
    int n_rows, int n_cols                                                     \
) {                                                                            \
    int col = blockIdx.x * blockDim.x + threadIdx.x;                           \
    if (col >= n_cols) return;                                                 \
    ACC_T loc = READ_W(w_loc[0]);                                              \
    ACC_T scale = READ_W(w_scale[0]);                                          \
    unsigned int cl = (unsigned int)clen[0];                                    \
    if (cl < 2) cl = 2;                                                        \
    curandStatePhilox4_32_10_t state;                                           \
    curand_init((unsigned long long)seed[0], (unsigned long long)col, 0ULL, &state); \
    unsigned int row = curand(&state) % cl;                                     \
    while (row < (unsigned int)n_rows) {                                        \
        ACC_T n = (ACC_T)RNG_FUNC(&state);                                     \
        output[(size_t)row * n_cols + col] = WRITE_W(loc + n * scale);         \
        row += 1 + (curand(&state) % (cl - 1));                                \
    }                                                                           \
}

DEFINE_JITN_CORDER_FALSE(_f32,  float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32)
DEFINE_JITN_CORDER_FALSE(_f64,  double,        double, READ_F64,  WRITE_F64,  curand_normal_f64)
DEFINE_JITN_CORDER_FALSE(_f16,  __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32)
DEFINE_JITN_CORDER_FALSE(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32)

#define FFI_JITN_CORDER_TRUE(SUFFIX, WEIGHT_C_T)                              \
void jitn_corder_true##SUFFIX(                                                \
    tvm::ffi::TensorView w_loc,                                                \
    tvm::ffi::TensorView w_scale,                                              \
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
    _jitn_corder_true_kern##SUFFIX<<<blocks, threads, 0, s>>>(                 \
        static_cast<const WEIGHT_C_T*>(w_loc.data_ptr()),                      \
        static_cast<const WEIGHT_C_T*>(w_scale.data_ptr()),                    \
        static_cast<const float*>(clen.data_ptr()),                            \
        static_cast<const int*>(seed.data_ptr()),                              \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                           \
        n_rows, n_cols                                                         \
    );                                                                         \
}

// @tvm_ffi jitn_corder_true_f32
FFI_JITN_CORDER_TRUE(_f32, float)
// @tvm_ffi jitn_corder_true_f64
FFI_JITN_CORDER_TRUE(_f64, double)
// @tvm_ffi jitn_corder_true_f16
FFI_JITN_CORDER_TRUE(_f16, __half)
// @tvm_ffi jitn_corder_true_bf16
FFI_JITN_CORDER_TRUE(_bf16, __nv_bfloat16)

#define FFI_JITN_CORDER_FALSE(SUFFIX, WEIGHT_C_T)                             \
void jitn_corder_false##SUFFIX(                                               \
    tvm::ffi::TensorView w_loc,                                                \
    tvm::ffi::TensorView w_scale,                                              \
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
    _jitn_corder_false_kern##SUFFIX<<<blocks, threads, 0, s>>>(                \
        static_cast<const WEIGHT_C_T*>(w_loc.data_ptr()),                      \
        static_cast<const WEIGHT_C_T*>(w_scale.data_ptr()),                    \
        static_cast<const float*>(clen.data_ptr()),                            \
        static_cast<const int*>(seed.data_ptr()),                              \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                           \
        n_rows, n_cols                                                         \
    );                                                                         \
}

// @tvm_ffi jitn_corder_false_f32
FFI_JITN_CORDER_FALSE(_f32, float)
// @tvm_ffi jitn_corder_false_f64
FFI_JITN_CORDER_FALSE(_f64, double)
// @tvm_ffi jitn_corder_false_f16
FFI_JITN_CORDER_FALSE(_f16, __half)
// @tvm_ffi jitn_corder_false_bf16
FFI_JITN_CORDER_FALSE(_bf16, __nv_bfloat16)


// #########################################################################
// ##  2. jitnmv — Float Matrix-Vector Product                            ##
// #########################################################################

#define DEFINE_JITNMV_GATHER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, RNG_FUNC, ACC_ZERO) \
__global__ void _jitnmv_gather_kern##SUFFIX(                                      \
    const WEIGHT_T* __restrict__ w_loc,                                           \
    const WEIGHT_T* __restrict__ w_scale,                                         \
    const float*    __restrict__ clen,                                            \
    const int*      __restrict__ seed,                                            \
    const WEIGHT_T* __restrict__ vector,                                          \
    WEIGHT_T*       __restrict__ output,                                          \
    int m, int k                                                                  \
) {                                                                               \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                \
    if (i >= m) return;                                                           \
    ACC_T loc = READ_W(w_loc[0]);                                                 \
    ACC_T scale = READ_W(w_scale[0]);                                             \
    unsigned int cl = (unsigned int)clen[0];                                       \
    if (cl < 2) cl = 2;                                                           \
    curandStatePhilox4_32_10_t state;                                              \
    curand_init((unsigned long long)seed[0], (unsigned long long)i, 0ULL, &state); \
    unsigned int j = curand(&state) % cl;                                          \
    ACC_T acc = ACC_ZERO;                                                          \
    while (j < (unsigned int)k) {                                                  \
        ACC_T n = (ACC_T)RNG_FUNC(&state);                                        \
        ACC_T w = loc + n * scale;                                                \
        acc += READ_W(vector[j]) * w;                                             \
        j += 1 + (curand(&state) % (cl - 1));                                     \
    }                                                                              \
    output[i] = WRITE_W(acc);                                                      \
}

DEFINE_JITNMV_GATHER(_f32,  float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, 0.0f)
DEFINE_JITNMV_GATHER(_f64,  double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, 0.0)
DEFINE_JITNMV_GATHER(_f16,  __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, 0.0f)
DEFINE_JITNMV_GATHER(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, 0.0f)

#define DEFINE_JITNMV_SCATTER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, RNG_FUNC, ATOMIC_ADD) \
__global__ void _jitnmv_scatter_kern##SUFFIX(                                         \
    const WEIGHT_T* __restrict__ w_loc,                                               \
    const WEIGHT_T* __restrict__ w_scale,                                             \
    const float*    __restrict__ clen,                                                \
    const int*      __restrict__ seed,                                                \
    const WEIGHT_T* __restrict__ vector,                                              \
    WEIGHT_T*       __restrict__ output,                                              \
    int m, int k                                                                      \
) {                                                                                   \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                                    \
    if (j >= k) return;                                                               \
    ACC_T loc = READ_W(w_loc[0]);                                                     \
    ACC_T scale = READ_W(w_scale[0]);                                                 \
    ACC_T vj = READ_W(vector[j]);                                                     \
    unsigned int cl = (unsigned int)clen[0];                                           \
    if (cl < 2) cl = 2;                                                               \
    curandStatePhilox4_32_10_t state;                                                  \
    curand_init((unsigned long long)seed[0], (unsigned long long)j, 0ULL, &state);     \
    unsigned int i = curand(&state) % cl;                                              \
    while (i < (unsigned int)m) {                                                      \
        ACC_T n = (ACC_T)RNG_FUNC(&state);                                            \
        ACC_T w = loc + n * scale;                                                    \
        ATOMIC_ADD(&output[i], w * vj);                                                \
        i += 1 + (curand(&state) % (cl - 1));                                         \
    }                                                                                  \
}

DEFINE_JITNMV_SCATTER(_f32,  float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, atomicAdd_f32)
DEFINE_JITNMV_SCATTER(_f64,  double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, atomicAdd_f64)
DEFINE_JITNMV_SCATTER(_f16,  __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, atomicAdd_f16)
DEFINE_JITNMV_SCATTER(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, atomicAdd_bf16)

#define FFI_JITNMV_GATHER(SUFFIX, WEIGHT_C_T)                               \
void jitnmv_gather##SUFFIX(                                                  \
    tvm::ffi::TensorView w_loc,                                              \
    tvm::ffi::TensorView w_scale,                                            \
    tvm::ffi::TensorView clen,                                               \
    tvm::ffi::TensorView seed,                                               \
    tvm::ffi::TensorView vector,                                             \
    tvm::ffi::TensorView output,                                             \
    int64_t stream                                                           \
) {                                                                          \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                 \
    int m = static_cast<int>(output.size(0));                                \
    int k = static_cast<int>(vector.size(0));                                \
    cudaMemsetAsync(output.data_ptr(), 0,                                    \
        (size_t)m * sizeof(WEIGHT_C_T), s);                                  \
    int threads = 256;                                                       \
    int blocks = (m + threads - 1) / threads;                                \
    _jitnmv_gather_kern##SUFFIX<<<blocks, threads, 0, s>>>(                  \
        static_cast<const WEIGHT_C_T*>(w_loc.data_ptr()),                    \
        static_cast<const WEIGHT_C_T*>(w_scale.data_ptr()),                  \
        static_cast<const float*>(clen.data_ptr()),                          \
        static_cast<const int*>(seed.data_ptr()),                            \
        static_cast<const WEIGHT_C_T*>(vector.data_ptr()),                   \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                         \
        m, k                                                                 \
    );                                                                       \
}

// @tvm_ffi jitnmv_gather_f32
FFI_JITNMV_GATHER(_f32, float)
// @tvm_ffi jitnmv_gather_f64
FFI_JITNMV_GATHER(_f64, double)
// @tvm_ffi jitnmv_gather_f16
FFI_JITNMV_GATHER(_f16, __half)
// @tvm_ffi jitnmv_gather_bf16
FFI_JITNMV_GATHER(_bf16, __nv_bfloat16)

#define FFI_JITNMV_SCATTER(SUFFIX, WEIGHT_C_T)                              \
void jitnmv_scatter##SUFFIX(                                                 \
    tvm::ffi::TensorView w_loc,                                               \
    tvm::ffi::TensorView w_scale,                                             \
    tvm::ffi::TensorView clen,                                                \
    tvm::ffi::TensorView seed,                                                \
    tvm::ffi::TensorView vector,                                              \
    tvm::ffi::TensorView output,                                              \
    int64_t stream                                                            \
) {                                                                           \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                  \
    int m = static_cast<int>(output.size(0));                                 \
    int k = static_cast<int>(vector.size(0));                                 \
    cudaMemsetAsync(output.data_ptr(), 0,                                     \
        (size_t)m * sizeof(WEIGHT_C_T), s);                                   \
    int threads = 256;                                                        \
    int blocks = (k + threads - 1) / threads;                                 \
    _jitnmv_scatter_kern##SUFFIX<<<blocks, threads, 0, s>>>(                  \
        static_cast<const WEIGHT_C_T*>(w_loc.data_ptr()),                     \
        static_cast<const WEIGHT_C_T*>(w_scale.data_ptr()),                   \
        static_cast<const float*>(clen.data_ptr()),                           \
        static_cast<const int*>(seed.data_ptr()),                             \
        static_cast<const WEIGHT_C_T*>(vector.data_ptr()),                    \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                          \
        m, k                                                                  \
    );                                                                        \
}

// @tvm_ffi jitnmv_scatter_f32
FFI_JITNMV_SCATTER(_f32, float)
// @tvm_ffi jitnmv_scatter_f64
FFI_JITNMV_SCATTER(_f64, double)
// @tvm_ffi jitnmv_scatter_f16
FFI_JITNMV_SCATTER(_f16, __half)
// @tvm_ffi jitnmv_scatter_bf16
FFI_JITNMV_SCATTER(_bf16, __nv_bfloat16)


// #########################################################################
// ##  3. jitnmm — Float Matrix-Matrix Product                            ##
// #########################################################################

#define DEFINE_JITNMM_GATHER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, RNG_FUNC, ACC_ZERO) \
__global__ void _jitnmm_gather_kern##SUFFIX(                                      \
    const WEIGHT_T* __restrict__ w_loc,                                           \
    const WEIGHT_T* __restrict__ w_scale,                                         \
    const float*    __restrict__ clen,                                            \
    const int*      __restrict__ seed,                                            \
    const WEIGHT_T* __restrict__ B,                                               \
    WEIGHT_T*       __restrict__ output,                                          \
    int m, int k, int n                                                           \
) {                                                                               \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                \
    if (i >= m) return;                                                           \
    ACC_T loc = READ_W(w_loc[0]);                                                 \
    ACC_T scale = READ_W(w_scale[0]);                                             \
    unsigned int cl = (unsigned int)clen[0];                                       \
    if (cl < 2) cl = 2;                                                           \
    curandStatePhilox4_32_10_t state;                                              \
    curand_init((unsigned long long)seed[0], (unsigned long long)i, 0ULL, &state); \
    unsigned int j = curand(&state) % cl;                                          \
    WEIGHT_T* out_row = output + (size_t)i * n;                                    \
    while (j < (unsigned int)k) {                                                  \
        ACC_T num = (ACC_T)RNG_FUNC(&state);                                      \
        ACC_T w = loc + num * scale;                                              \
        const WEIGHT_T* b_row = B + (size_t)j * n;                                \
        for (int col = 0; col < n; col++) {                                        \
            ACC_T cur = READ_W(out_row[col]);                                      \
            out_row[col] = WRITE_W(cur + w * READ_W(b_row[col]));                  \
        }                                                                          \
        j += 1 + (curand(&state) % (cl - 1));                                     \
    }                                                                              \
}

DEFINE_JITNMM_GATHER(_f32,  float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, 0.0f)
DEFINE_JITNMM_GATHER(_f64,  double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, 0.0)
DEFINE_JITNMM_GATHER(_f16,  __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, 0.0f)
DEFINE_JITNMM_GATHER(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, 0.0f)

#define DEFINE_JITNMM_SCATTER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, RNG_FUNC, ATOMIC_ADD) \
__global__ void _jitnmm_scatter_kern##SUFFIX(                                         \
    const WEIGHT_T* __restrict__ w_loc,                                               \
    const WEIGHT_T* __restrict__ w_scale,                                             \
    const float*    __restrict__ clen,                                                \
    const int*      __restrict__ seed,                                                \
    const WEIGHT_T* __restrict__ B,                                                   \
    WEIGHT_T*       __restrict__ output,                                              \
    int m, int k, int n                                                               \
) {                                                                                   \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                                    \
    if (j >= k) return;                                                               \
    ACC_T loc = READ_W(w_loc[0]);                                                     \
    ACC_T scale = READ_W(w_scale[0]);                                                 \
    unsigned int cl = (unsigned int)clen[0];                                           \
    if (cl < 2) cl = 2;                                                               \
    curandStatePhilox4_32_10_t state;                                                  \
    curand_init((unsigned long long)seed[0], (unsigned long long)j, 0ULL, &state);     \
    unsigned int i = curand(&state) % cl;                                              \
    const WEIGHT_T* b_row = B + (size_t)j * n;                                        \
    while (i < (unsigned int)m) {                                                      \
        ACC_T num = (ACC_T)RNG_FUNC(&state);                                          \
        ACC_T w = loc + num * scale;                                                  \
        WEIGHT_T* out_row = output + (size_t)i * n;                                    \
        for (int col = 0; col < n; col++) {                                            \
            ACC_T val = w * READ_W(b_row[col]);                                        \
            ATOMIC_ADD(&out_row[col], val);                                            \
        }                                                                              \
        i += 1 + (curand(&state) % (cl - 1));                                         \
    }                                                                                  \
}

DEFINE_JITNMM_SCATTER(_f32,  float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, atomicAdd_f32)
DEFINE_JITNMM_SCATTER(_f64,  double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, atomicAdd_f64)
DEFINE_JITNMM_SCATTER(_f16,  __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, atomicAdd_f16)
DEFINE_JITNMM_SCATTER(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, atomicAdd_bf16)

#define FFI_JITNMM_GATHER(SUFFIX, WEIGHT_C_T)                                \
void jitnmm_gather##SUFFIX(                                                  \
    tvm::ffi::TensorView w_loc,                                               \
    tvm::ffi::TensorView w_scale,                                             \
    tvm::ffi::TensorView clen,                                                \
    tvm::ffi::TensorView seed,                                                \
    tvm::ffi::TensorView B,                                                   \
    tvm::ffi::TensorView output,                                              \
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
    _jitnmm_gather_kern##SUFFIX<<<blocks, threads, 0, s>>>(                   \
        static_cast<const WEIGHT_C_T*>(w_loc.data_ptr()),                     \
        static_cast<const WEIGHT_C_T*>(w_scale.data_ptr()),                   \
        static_cast<const float*>(clen.data_ptr()),                           \
        static_cast<const int*>(seed.data_ptr()),                             \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                         \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                          \
        m, k, n                                                               \
    );                                                                        \
}

// @tvm_ffi jitnmm_gather_f32
FFI_JITNMM_GATHER(_f32, float)
// @tvm_ffi jitnmm_gather_f64
FFI_JITNMM_GATHER(_f64, double)
// @tvm_ffi jitnmm_gather_f16
FFI_JITNMM_GATHER(_f16, __half)
// @tvm_ffi jitnmm_gather_bf16
FFI_JITNMM_GATHER(_bf16, __nv_bfloat16)

#define FFI_JITNMM_SCATTER(SUFFIX, WEIGHT_C_T)                               \
void jitnmm_scatter##SUFFIX(                                                 \
    tvm::ffi::TensorView w_loc,                                               \
    tvm::ffi::TensorView w_scale,                                             \
    tvm::ffi::TensorView clen,                                                \
    tvm::ffi::TensorView seed,                                                \
    tvm::ffi::TensorView B,                                                   \
    tvm::ffi::TensorView output,                                              \
    int64_t stream                                                            \
) {                                                                            \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                   \
    int m = static_cast<int>(output.size(0));                                  \
    int n = static_cast<int>(output.size(1));                                  \
    int k = static_cast<int>(B.size(0));                                       \
    cudaMemsetAsync(output.data_ptr(), 0,                                      \
        (size_t)m * n * sizeof(WEIGHT_C_T), s);                                \
    int threads = 256;                                                         \
    int blocks = (k + threads - 1) / threads;                                  \
    _jitnmm_scatter_kern##SUFFIX<<<blocks, threads, 0, s>>>(                   \
        static_cast<const WEIGHT_C_T*>(w_loc.data_ptr()),                      \
        static_cast<const WEIGHT_C_T*>(w_scale.data_ptr()),                    \
        static_cast<const float*>(clen.data_ptr()),                            \
        static_cast<const int*>(seed.data_ptr()),                              \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                          \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                           \
        m, k, n                                                                \
    );                                                                         \
}

// @tvm_ffi jitnmm_scatter_f32
FFI_JITNMM_SCATTER(_f32, float)
// @tvm_ffi jitnmm_scatter_f64
FFI_JITNMM_SCATTER(_f64, double)
// @tvm_ffi jitnmm_scatter_f16
FFI_JITNMM_SCATTER(_f16, __half)
// @tvm_ffi jitnmm_scatter_bf16
FFI_JITNMM_SCATTER(_bf16, __nv_bfloat16)


// #########################################################################
// ##  4. binary_jitnmv — Event-Driven Matrix-Vector Product              ##
// #########################################################################

#define DEFINE_BINARY_JITNMV_GATHER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, RNG_FUNC, SPIKE_T, IS_ACTIVE, ACC_ZERO) \
__global__ void _binary_jitnmv_gather_kern##SUFFIX(                             \
    const WEIGHT_T* __restrict__ w_loc,                                         \
    const WEIGHT_T* __restrict__ w_scale,                                       \
    const float*    __restrict__ clen,                                          \
    const int*      __restrict__ seed,                                          \
    const SPIKE_T*  __restrict__ vector,                                        \
    WEIGHT_T*       __restrict__ output,                                        \
    int m, int k                                                                \
) {                                                                             \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                              \
    if (i >= m) return;                                                         \
    ACC_T loc = READ_W(w_loc[0]);                                               \
    ACC_T scale = READ_W(w_scale[0]);                                           \
    unsigned int cl = (unsigned int)clen[0];                                     \
    if (cl < 2) cl = 2;                                                         \
    curandStatePhilox4_32_10_t state;                                            \
    curand_init((unsigned long long)seed[0], (unsigned long long)i, 0ULL, &state);\
    unsigned int j = curand(&state) % cl;                                        \
    ACC_T acc = ACC_ZERO;                                                        \
    while (j < (unsigned int)k) {                                                \
        ACC_T n = (ACC_T)RNG_FUNC(&state);                                      \
        if (IS_ACTIVE(vector, j)) {                                              \
            ACC_T w = loc + n * scale;                                          \
            acc += w;                                                            \
        }                                                                        \
        j += 1 + (curand(&state) % (cl - 1));                                   \
    }                                                                            \
    output[i] = WRITE_W(acc);                                                    \
}

DEFINE_BINARY_JITNMV_GATHER(_f32_bool,  float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITNMV_GATHER(_f32_float, float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, float,  IS_ACTIVE_FLOAT, 0.0f)
DEFINE_BINARY_JITNMV_GATHER(_f64_bool,  double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, int8_t, IS_ACTIVE_BOOL,  0.0)
DEFINE_BINARY_JITNMV_GATHER(_f64_float, double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, float,  IS_ACTIVE_FLOAT, 0.0)
DEFINE_BINARY_JITNMV_GATHER(_f16_bool,  __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITNMV_GATHER(_f16_float, __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, float,  IS_ACTIVE_FLOAT, 0.0f)
DEFINE_BINARY_JITNMV_GATHER(_bf16_bool, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITNMV_GATHER(_bf16_float,__nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, float,  IS_ACTIVE_FLOAT, 0.0f)

#define DEFINE_BINARY_JITNMV_SCATTER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, RNG_FUNC, SPIKE_T, IS_ACTIVE, ATOMIC_ADD) \
__global__ void _binary_jitnmv_scatter_kern##SUFFIX(                            \
    const WEIGHT_T* __restrict__ w_loc,                                         \
    const WEIGHT_T* __restrict__ w_scale,                                       \
    const float*    __restrict__ clen,                                          \
    const int*      __restrict__ seed,                                          \
    const SPIKE_T*  __restrict__ vector,                                        \
    WEIGHT_T*       __restrict__ output,                                        \
    int m, int k                                                                \
) {                                                                             \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                              \
    if (j >= k) return;                                                         \
    if (!IS_ACTIVE(vector, j)) return;                                          \
    ACC_T loc = READ_W(w_loc[0]);                                               \
    ACC_T scale = READ_W(w_scale[0]);                                           \
    unsigned int cl = (unsigned int)clen[0];                                     \
    if (cl < 2) cl = 2;                                                         \
    curandStatePhilox4_32_10_t state;                                            \
    curand_init((unsigned long long)seed[0], (unsigned long long)j, 0ULL, &state);\
    unsigned int i = curand(&state) % cl;                                        \
    while (i < (unsigned int)m) {                                                \
        ACC_T n = (ACC_T)RNG_FUNC(&state);                                      \
        ACC_T w = loc + n * scale;                                              \
        ATOMIC_ADD(&output[i], w);                                               \
        i += 1 + (curand(&state) % (cl - 1));                                   \
    }                                                                            \
}

DEFINE_BINARY_JITNMV_SCATTER(_f32_bool,  float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, int8_t, IS_ACTIVE_BOOL,  atomicAdd_f32)
DEFINE_BINARY_JITNMV_SCATTER(_f32_float, float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, float,  IS_ACTIVE_FLOAT, atomicAdd_f32)
DEFINE_BINARY_JITNMV_SCATTER(_f64_bool,  double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, int8_t, IS_ACTIVE_BOOL,  atomicAdd_f64)
DEFINE_BINARY_JITNMV_SCATTER(_f64_float, double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, float,  IS_ACTIVE_FLOAT, atomicAdd_f64)
DEFINE_BINARY_JITNMV_SCATTER(_f16_bool,  __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, int8_t, IS_ACTIVE_BOOL,  atomicAdd_f16)
DEFINE_BINARY_JITNMV_SCATTER(_f16_float, __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, float,  IS_ACTIVE_FLOAT, atomicAdd_f16)
DEFINE_BINARY_JITNMV_SCATTER(_bf16_bool, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, int8_t, IS_ACTIVE_BOOL,  atomicAdd_bf16)
DEFINE_BINARY_JITNMV_SCATTER(_bf16_float,__nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, float,  IS_ACTIVE_FLOAT, atomicAdd_bf16)

#define FFI_BINARY_JITNMV_GATHER(SUFFIX, WEIGHT_C_T, SPIKE_C_T)              \
void binary_jitnmv_gather##SUFFIX(                                            \
    tvm::ffi::TensorView w_loc,                                               \
    tvm::ffi::TensorView w_scale,                                             \
    tvm::ffi::TensorView clen,                                                \
    tvm::ffi::TensorView seed,                                                \
    tvm::ffi::TensorView vector,                                              \
    tvm::ffi::TensorView output,                                              \
    int64_t stream                                                            \
) {                                                                           \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                  \
    int m = static_cast<int>(output.size(0));                                 \
    int k = static_cast<int>(vector.size(0));                                 \
    cudaMemsetAsync(output.data_ptr(), 0,                                     \
        (size_t)m * sizeof(WEIGHT_C_T), s);                                   \
    int threads = 256;                                                        \
    int blocks = (m + threads - 1) / threads;                                 \
    _binary_jitnmv_gather_kern##SUFFIX<<<blocks, threads, 0, s>>>(            \
        static_cast<const WEIGHT_C_T*>(w_loc.data_ptr()),                     \
        static_cast<const WEIGHT_C_T*>(w_scale.data_ptr()),                   \
        static_cast<const float*>(clen.data_ptr()),                           \
        static_cast<const int*>(seed.data_ptr()),                             \
        static_cast<const SPIKE_C_T*>(vector.data_ptr()),                     \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                          \
        m, k                                                                  \
    );                                                                        \
}

// @tvm_ffi binary_jitnmv_gather_f32_bool
FFI_BINARY_JITNMV_GATHER(_f32_bool,  float,         int8_t)
// @tvm_ffi binary_jitnmv_gather_f32_float
FFI_BINARY_JITNMV_GATHER(_f32_float, float,         float)
// @tvm_ffi binary_jitnmv_gather_f64_bool
FFI_BINARY_JITNMV_GATHER(_f64_bool,  double,        int8_t)
// @tvm_ffi binary_jitnmv_gather_f64_float
FFI_BINARY_JITNMV_GATHER(_f64_float, double,        float)
// @tvm_ffi binary_jitnmv_gather_f16_bool
FFI_BINARY_JITNMV_GATHER(_f16_bool,  __half,        int8_t)
// @tvm_ffi binary_jitnmv_gather_f16_float
FFI_BINARY_JITNMV_GATHER(_f16_float, __half,        float)
// @tvm_ffi binary_jitnmv_gather_bf16_bool
FFI_BINARY_JITNMV_GATHER(_bf16_bool, __nv_bfloat16, int8_t)
// @tvm_ffi binary_jitnmv_gather_bf16_float
FFI_BINARY_JITNMV_GATHER(_bf16_float,__nv_bfloat16, float)

#define FFI_BINARY_JITNMV_SCATTER(SUFFIX, WEIGHT_C_T, SPIKE_C_T)             \
void binary_jitnmv_scatter##SUFFIX(                                           \
    tvm::ffi::TensorView w_loc,                                               \
    tvm::ffi::TensorView w_scale,                                             \
    tvm::ffi::TensorView clen,                                                \
    tvm::ffi::TensorView seed,                                                \
    tvm::ffi::TensorView vector,                                              \
    tvm::ffi::TensorView output,                                              \
    int64_t stream                                                            \
) {                                                                           \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                  \
    int m = static_cast<int>(output.size(0));                                 \
    int k = static_cast<int>(vector.size(0));                                 \
    cudaMemsetAsync(output.data_ptr(), 0,                                     \
        (size_t)m * sizeof(WEIGHT_C_T), s);                                   \
    int threads = 256;                                                        \
    int blocks = (k + threads - 1) / threads;                                 \
    _binary_jitnmv_scatter_kern##SUFFIX<<<blocks, threads, 0, s>>>(           \
        static_cast<const WEIGHT_C_T*>(w_loc.data_ptr()),                     \
        static_cast<const WEIGHT_C_T*>(w_scale.data_ptr()),                   \
        static_cast<const float*>(clen.data_ptr()),                           \
        static_cast<const int*>(seed.data_ptr()),                             \
        static_cast<const SPIKE_C_T*>(vector.data_ptr()),                     \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                          \
        m, k                                                                  \
    );                                                                        \
}

// @tvm_ffi binary_jitnmv_scatter_f32_bool
FFI_BINARY_JITNMV_SCATTER(_f32_bool,  float,         int8_t)
// @tvm_ffi binary_jitnmv_scatter_f32_float
FFI_BINARY_JITNMV_SCATTER(_f32_float, float,         float)
// @tvm_ffi binary_jitnmv_scatter_f64_bool
FFI_BINARY_JITNMV_SCATTER(_f64_bool,  double,        int8_t)
// @tvm_ffi binary_jitnmv_scatter_f64_float
FFI_BINARY_JITNMV_SCATTER(_f64_float, double,        float)
// @tvm_ffi binary_jitnmv_scatter_f16_bool
FFI_BINARY_JITNMV_SCATTER(_f16_bool,  __half,        int8_t)
// @tvm_ffi binary_jitnmv_scatter_f16_float
FFI_BINARY_JITNMV_SCATTER(_f16_float, __half,        float)
// @tvm_ffi binary_jitnmv_scatter_bf16_bool
FFI_BINARY_JITNMV_SCATTER(_bf16_bool, __nv_bfloat16, int8_t)
// @tvm_ffi binary_jitnmv_scatter_bf16_float
FFI_BINARY_JITNMV_SCATTER(_bf16_float,__nv_bfloat16, float)


// #########################################################################
// ##  5. binary_jitnmm — Event-Driven Matrix-Matrix Product              ##
// #########################################################################

#define DEFINE_BINARY_JITNMM_GATHER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, RNG_FUNC, SPIKE_T, IS_ACTIVE, ACC_ZERO) \
__global__ void _binary_jitnmm_gather_kern##SUFFIX(                             \
    const WEIGHT_T* __restrict__ w_loc,                                         \
    const WEIGHT_T* __restrict__ w_scale,                                       \
    const float*    __restrict__ clen,                                          \
    const int*      __restrict__ seed,                                          \
    const SPIKE_T*  __restrict__ B,                                             \
    WEIGHT_T*       __restrict__ output,                                        \
    int m, int k, int n                                                         \
) {                                                                             \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                              \
    if (i >= m) return;                                                         \
    ACC_T loc = READ_W(w_loc[0]);                                               \
    ACC_T scale = READ_W(w_scale[0]);                                           \
    unsigned int cl = (unsigned int)clen[0];                                     \
    if (cl < 2) cl = 2;                                                         \
    curandStatePhilox4_32_10_t state;                                            \
    curand_init((unsigned long long)seed[0], (unsigned long long)i, 0ULL, &state);\
    unsigned int j = curand(&state) % cl;                                        \
    WEIGHT_T* out_row = output + (size_t)i * n;                                  \
    while (j < (unsigned int)k) {                                                \
        ACC_T num = (ACC_T)RNG_FUNC(&state);                                    \
        ACC_T w = loc + num * scale;                                            \
        const SPIKE_T* b_row = B + (size_t)j * n;                               \
        for (int col = 0; col < n; col++) {                                      \
            if (IS_ACTIVE(b_row, col)) {                                         \
                ACC_T cur = READ_W(out_row[col]);                                \
                out_row[col] = WRITE_W(cur + w);                                 \
            }                                                                    \
        }                                                                        \
        j += 1 + (curand(&state) % (cl - 1));                                   \
    }                                                                            \
}

DEFINE_BINARY_JITNMM_GATHER(_f32_bool,  float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITNMM_GATHER(_f32_float, float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, float,  IS_ACTIVE_FLOAT, 0.0f)
DEFINE_BINARY_JITNMM_GATHER(_f64_bool,  double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, int8_t, IS_ACTIVE_BOOL,  0.0)
DEFINE_BINARY_JITNMM_GATHER(_f64_float, double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, float,  IS_ACTIVE_FLOAT, 0.0)
DEFINE_BINARY_JITNMM_GATHER(_f16_bool,  __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITNMM_GATHER(_f16_float, __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, float,  IS_ACTIVE_FLOAT, 0.0f)
DEFINE_BINARY_JITNMM_GATHER(_bf16_bool, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, int8_t, IS_ACTIVE_BOOL,  0.0f)
DEFINE_BINARY_JITNMM_GATHER(_bf16_float,__nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, float,  IS_ACTIVE_FLOAT, 0.0f)

#define DEFINE_BINARY_JITNMM_SCATTER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, RNG_FUNC, SPIKE_T, IS_ACTIVE, ATOMIC_ADD) \
__global__ void _binary_jitnmm_scatter_kern##SUFFIX(                            \
    const WEIGHT_T* __restrict__ w_loc,                                         \
    const WEIGHT_T* __restrict__ w_scale,                                       \
    const float*    __restrict__ clen,                                          \
    const int*      __restrict__ seed,                                          \
    const SPIKE_T*  __restrict__ B,                                             \
    WEIGHT_T*       __restrict__ output,                                        \
    int m, int k, int n                                                         \
) {                                                                             \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                              \
    if (j >= k) return;                                                         \
    ACC_T loc = READ_W(w_loc[0]);                                               \
    ACC_T scale = READ_W(w_scale[0]);                                           \
    unsigned int cl = (unsigned int)clen[0];                                     \
    if (cl < 2) cl = 2;                                                         \
    const SPIKE_T* b_row = B + (size_t)j * n;                                   \
    curandStatePhilox4_32_10_t state;                                            \
    curand_init((unsigned long long)seed[0], (unsigned long long)j, 0ULL, &state);\
    unsigned int i = curand(&state) % cl;                                        \
    while (i < (unsigned int)m) {                                                \
        ACC_T num = (ACC_T)RNG_FUNC(&state);                                    \
        ACC_T w = loc + num * scale;                                            \
        WEIGHT_T* out_row = output + (size_t)i * n;                              \
        for (int col = 0; col < n; col++) {                                      \
            if (IS_ACTIVE(b_row, col)) {                                         \
                ATOMIC_ADD(&out_row[col], w);                                    \
            }                                                                    \
        }                                                                        \
        i += 1 + (curand(&state) % (cl - 1));                                   \
    }                                                                            \
}

DEFINE_BINARY_JITNMM_SCATTER(_f32_bool,  float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, int8_t, IS_ACTIVE_BOOL,  atomicAdd_f32)
DEFINE_BINARY_JITNMM_SCATTER(_f32_float, float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, float,  IS_ACTIVE_FLOAT, atomicAdd_f32)
DEFINE_BINARY_JITNMM_SCATTER(_f64_bool,  double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, int8_t, IS_ACTIVE_BOOL,  atomicAdd_f64)
DEFINE_BINARY_JITNMM_SCATTER(_f64_float, double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, float,  IS_ACTIVE_FLOAT, atomicAdd_f64)
DEFINE_BINARY_JITNMM_SCATTER(_f16_bool,  __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, int8_t, IS_ACTIVE_BOOL,  atomicAdd_f16)
DEFINE_BINARY_JITNMM_SCATTER(_f16_float, __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, float,  IS_ACTIVE_FLOAT, atomicAdd_f16)
DEFINE_BINARY_JITNMM_SCATTER(_bf16_bool, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, int8_t, IS_ACTIVE_BOOL,  atomicAdd_bf16)
DEFINE_BINARY_JITNMM_SCATTER(_bf16_float,__nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, float,  IS_ACTIVE_FLOAT, atomicAdd_bf16)

#define FFI_BINARY_JITNMM_GATHER(SUFFIX, WEIGHT_C_T, SPIKE_C_T)              \
void binary_jitnmm_gather##SUFFIX(                                            \
    tvm::ffi::TensorView w_loc,                                               \
    tvm::ffi::TensorView w_scale,                                             \
    tvm::ffi::TensorView clen,                                                \
    tvm::ffi::TensorView seed,                                                \
    tvm::ffi::TensorView B,                                                   \
    tvm::ffi::TensorView output,                                              \
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
    _binary_jitnmm_gather_kern##SUFFIX<<<blocks, threads, 0, s>>>(            \
        static_cast<const WEIGHT_C_T*>(w_loc.data_ptr()),                     \
        static_cast<const WEIGHT_C_T*>(w_scale.data_ptr()),                   \
        static_cast<const float*>(clen.data_ptr()),                           \
        static_cast<const int*>(seed.data_ptr()),                             \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                          \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                          \
        m, k, n                                                               \
    );                                                                        \
}

// @tvm_ffi binary_jitnmm_gather_f32_bool
FFI_BINARY_JITNMM_GATHER(_f32_bool,  float,         int8_t)
// @tvm_ffi binary_jitnmm_gather_f32_float
FFI_BINARY_JITNMM_GATHER(_f32_float, float,         float)
// @tvm_ffi binary_jitnmm_gather_f64_bool
FFI_BINARY_JITNMM_GATHER(_f64_bool,  double,        int8_t)
// @tvm_ffi binary_jitnmm_gather_f64_float
FFI_BINARY_JITNMM_GATHER(_f64_float, double,        float)
// @tvm_ffi binary_jitnmm_gather_f16_bool
FFI_BINARY_JITNMM_GATHER(_f16_bool,  __half,        int8_t)
// @tvm_ffi binary_jitnmm_gather_f16_float
FFI_BINARY_JITNMM_GATHER(_f16_float, __half,        float)
// @tvm_ffi binary_jitnmm_gather_bf16_bool
FFI_BINARY_JITNMM_GATHER(_bf16_bool, __nv_bfloat16, int8_t)
// @tvm_ffi binary_jitnmm_gather_bf16_float
FFI_BINARY_JITNMM_GATHER(_bf16_float,__nv_bfloat16, float)

#define FFI_BINARY_JITNMM_SCATTER(SUFFIX, WEIGHT_C_T, SPIKE_C_T)             \
void binary_jitnmm_scatter##SUFFIX(                                           \
    tvm::ffi::TensorView w_loc,                                               \
    tvm::ffi::TensorView w_scale,                                             \
    tvm::ffi::TensorView clen,                                                \
    tvm::ffi::TensorView seed,                                                \
    tvm::ffi::TensorView B,                                                   \
    tvm::ffi::TensorView output,                                              \
    int64_t stream                                                            \
) {                                                                           \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                  \
    int m = static_cast<int>(output.size(0));                                 \
    int n = static_cast<int>(output.size(1));                                 \
    int k = static_cast<int>(B.size(0));                                      \
    cudaMemsetAsync(output.data_ptr(), 0,                                     \
        (size_t)m * n * sizeof(WEIGHT_C_T), s);                               \
    int threads = 256;                                                        \
    int blocks = (k + threads - 1) / threads;                                 \
    _binary_jitnmm_scatter_kern##SUFFIX<<<blocks, threads, 0, s>>>(           \
        static_cast<const WEIGHT_C_T*>(w_loc.data_ptr()),                     \
        static_cast<const WEIGHT_C_T*>(w_scale.data_ptr()),                   \
        static_cast<const float*>(clen.data_ptr()),                           \
        static_cast<const int*>(seed.data_ptr()),                             \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                          \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                          \
        m, k, n                                                               \
    );                                                                        \
}

// @tvm_ffi binary_jitnmm_scatter_f32_bool
FFI_BINARY_JITNMM_SCATTER(_f32_bool,  float,         int8_t)
// @tvm_ffi binary_jitnmm_scatter_f32_float
FFI_BINARY_JITNMM_SCATTER(_f32_float, float,         float)
// @tvm_ffi binary_jitnmm_scatter_f64_bool
FFI_BINARY_JITNMM_SCATTER(_f64_bool,  double,        int8_t)
// @tvm_ffi binary_jitnmm_scatter_f64_float
FFI_BINARY_JITNMM_SCATTER(_f64_float, double,        float)
// @tvm_ffi binary_jitnmm_scatter_f16_bool
FFI_BINARY_JITNMM_SCATTER(_f16_bool,  __half,        int8_t)
// @tvm_ffi binary_jitnmm_scatter_f16_float
FFI_BINARY_JITNMM_SCATTER(_f16_float, __half,        float)
// @tvm_ffi binary_jitnmm_scatter_bf16_bool
FFI_BINARY_JITNMM_SCATTER(_bf16_bool, __nv_bfloat16, int8_t)
// @tvm_ffi binary_jitnmm_scatter_bf16_float
FFI_BINARY_JITNMM_SCATTER(_bf16_float,__nv_bfloat16, float)
