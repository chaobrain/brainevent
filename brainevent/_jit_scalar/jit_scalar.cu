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
 * jit_scalar.cu — JIT Scalar Connectivity CUDA Kernels (all operations)
 * ======================================================================
 *
 * Unified CUDA kernel file for all JIT scalar connectivity operations.
 * All kernels share the same cuRAND Philox geometric sampling pattern
 * for consistent random connectivity generation.
 *
 * Operations
 * ----------
 * 1. jits          — Dense matrix generation: M[i,j] = w * Bernoulli(prob)
 * 2. jitsmv        — Float matrix-vector:     y = M @ v
 * 3. jitsmm        — Float matrix-matrix:     Y = M @ B
 * 4. binary_jitsmv — Event-driven mat-vec:    y[i] = w * count{j in C(i) : spike[j]}
 * 5. binary_jitsmm — Event-driven mat-mat:    Y[i,:] = w * count{j in C(i) : B[j,:] active}
 *
 * Parameters (common to all)
 * --------------------------
 * weight : shape (1,), scalar weight for all connections
 * clen   : shape (1,), connection length = 2/prob (float32)
 * seed   : shape (1,), int32 random seed
 *
 * Geometric sampling pattern (shared by all kernels):
 *   curand_init(seed, thread_id, 0, &state);
 *   unsigned int i = curand(&state) % clen;
 *   while (i < dim) { process(i); i += 1 + (curand(&state) % (clen - 1)); }
 *
 * corder=True  (gather): one thread per output element, no atomics
 * corder=False (scatter): one thread per input element, uses atomicAdd
 *
 * Supported weight dtypes: float32, float64, float16, bfloat16.
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

// =========================================================================
// Spike activity checks (for binary kernels)
// =========================================================================

#define IS_ACTIVE_BOOL(v, j)  ((v)[j] != 0)
#define IS_ACTIVE_FLOAT(v, j) ((v)[j] > 0.0f)

// =========================================================================
// atomicAdd helpers for f16/bf16 (CAS-based)
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
        float old_f = __half2float(*reinterpret_cast<__half*>(&assumed));
        unsigned short int new_val = *reinterpret_cast<unsigned short int*>(
            &(*reinterpret_cast<__half*>(&assumed) = __float2half(old_f + val))
        );
        old = atomicCAS(addr_as_usi, assumed, new_val);
    } while (assumed != old);
}

__device__ __inline__ void atomicAdd_bf16(__nv_bfloat16* addr, float val) {
    unsigned short int* addr_as_usi = (unsigned short int*)addr;
    unsigned short int old = *addr_as_usi;
    unsigned short int assumed;
    do {
        assumed = old;
        float old_f = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&assumed));
        unsigned short int new_val = *reinterpret_cast<unsigned short int*>(
            &(*reinterpret_cast<__nv_bfloat16*>(&assumed) = __float2bfloat16(old_f + val))
        );
        old = atomicCAS(addr_as_usi, assumed, new_val);
    } while (assumed != old);
}


// #########################################################################
// ##  1. jits — Dense Matrix Generation                                  ##
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
    WEIGHT_T w = weight[0];                                                    \
    unsigned int cl = (unsigned int)clen[0];                                    \
    if (cl < 2) cl = 2;                                                        \
    curandStatePhilox4_32_10_t state;                                           \
    curand_init((unsigned long long)seed[0], (unsigned long long)row, 0ULL, &state); \
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
    WEIGHT_T w = weight[0];                                                    \
    unsigned int cl = (unsigned int)clen[0];                                    \
    if (cl < 2) cl = 2;                                                        \
    curandStatePhilox4_32_10_t state;                                           \
    curand_init((unsigned long long)seed[0], (unsigned long long)col, 0ULL, &state); \
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


// #########################################################################
// ##  2. jitsmv — Float Matrix-Vector Product                            ##
// #########################################################################

// =========================================================================
// Gather kernel (corder=true): one thread per output element
// y[i] = w * sum_{j in C(i)} v[j]
// =========================================================================

#define DEFINE_JITSMV_GATHER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _jitsmv_gather_kern##SUFFIX(                                      \
    const WEIGHT_T* __restrict__ weight,                                          \
    const float*    __restrict__ clen,                                            \
    const int*      __restrict__ seed,                                            \
    const WEIGHT_T* __restrict__ vector,                                          \
    WEIGHT_T*       __restrict__ output,                                          \
    int m, int k                                                                  \
) {                                                                               \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                \
    if (i >= m) return;                                                           \
    ACC_T w0 = READ_W(weight[0]);                                                 \
    unsigned int cl = (unsigned int)clen[0];                                       \
    if (cl < 2) cl = 2;                                                           \
    curandStatePhilox4_32_10_t state;                                              \
    curand_init((unsigned long long)seed[0], (unsigned long long)i, 0ULL, &state); \
    unsigned int j = curand(&state) % cl;                                          \
    ACC_T acc = ACC_ZERO;                                                          \
    while (j < (unsigned int)k) {                                                  \
        acc += READ_W(vector[j]);                                                  \
        j += 1 + (curand(&state) % (cl - 1));                                     \
    }                                                                              \
    output[i] = WRITE_W(w0 * acc);                                                 \
}

DEFINE_JITSMV_GATHER(_f32,  float,         float,  READ_F32,  WRITE_F32,  0.0f)
DEFINE_JITSMV_GATHER(_f64,  double,        double, READ_F64,  WRITE_F64,  0.0)
DEFINE_JITSMV_GATHER(_f16,  __half,        float,  READ_F16,  WRITE_F16,  0.0f)
DEFINE_JITSMV_GATHER(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, 0.0f)

// =========================================================================
// Scatter kernel (corder=false): one thread per input element
// For each input j, atomicAdd w * v[j] to output[connected_indices].
// =========================================================================

#define DEFINE_JITSMV_SCATTER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ATOMIC_ADD) \
__global__ void _jitsmv_scatter_kern##SUFFIX(                                         \
    const WEIGHT_T* __restrict__ weight,                                              \
    const float*    __restrict__ clen,                                                \
    const int*      __restrict__ seed,                                                \
    const WEIGHT_T* __restrict__ vector,                                              \
    WEIGHT_T*       __restrict__ output,                                              \
    int m, int k                                                                      \
) {                                                                                   \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                                    \
    if (j >= k) return;                                                               \
    ACC_T w0 = READ_W(weight[0]);                                                     \
    unsigned int cl = (unsigned int)clen[0];                                           \
    if (cl < 2) cl = 2;                                                               \
    ACC_T val = w0 * READ_W(vector[j]);                                                \
    curandStatePhilox4_32_10_t state;                                                  \
    curand_init((unsigned long long)seed[0], (unsigned long long)j, 0ULL, &state);     \
    unsigned int i = curand(&state) % cl;                                              \
    while (i < (unsigned int)m) {                                                      \
        ATOMIC_ADD(&output[i], val);                                                   \
        i += 1 + (curand(&state) % (cl - 1));                                         \
    }                                                                                  \
}

DEFINE_JITSMV_SCATTER(_f32,  float,         float,  READ_F32,  WRITE_F32,  atomicAdd_f32)
DEFINE_JITSMV_SCATTER(_f64,  double,        double, READ_F64,  WRITE_F64,  atomicAdd_f64)
DEFINE_JITSMV_SCATTER(_f16,  __half,        float,  READ_F16,  WRITE_F16,  atomicAdd_f16)
DEFINE_JITSMV_SCATTER(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, atomicAdd_bf16)

// ---- TVM FFI: jitsmv gather ----

#define FFI_JITSMV_GATHER(SUFFIX, WEIGHT_C_T)                               \
void jitsmv_gather##SUFFIX(                                                  \
    tvm::ffi::TensorView weight,                                             \
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
    _jitsmv_gather_kern##SUFFIX<<<blocks, threads, 0, s>>>(                  \
        static_cast<const WEIGHT_C_T*>(weight.data_ptr()),                   \
        static_cast<const float*>(clen.data_ptr()),                          \
        static_cast<const int*>(seed.data_ptr()),                            \
        static_cast<const WEIGHT_C_T*>(vector.data_ptr()),                   \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                         \
        m, k                                                                 \
    );                                                                       \
}

// @tvm_ffi jitsmv_gather_f32
FFI_JITSMV_GATHER(_f32, float)
// @tvm_ffi jitsmv_gather_f64
FFI_JITSMV_GATHER(_f64, double)
// @tvm_ffi jitsmv_gather_f16
FFI_JITSMV_GATHER(_f16, __half)
// @tvm_ffi jitsmv_gather_bf16
FFI_JITSMV_GATHER(_bf16, __nv_bfloat16)

// ---- TVM FFI: jitsmv scatter ----

#define FFI_JITSMV_SCATTER(SUFFIX, WEIGHT_C_T)                               \
void jitsmv_scatter##SUFFIX(                                                  \
    tvm::ffi::TensorView weight,                                              \
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
    _jitsmv_scatter_kern##SUFFIX<<<blocks, threads, 0, s>>>(                  \
        static_cast<const WEIGHT_C_T*>(weight.data_ptr()),                    \
        static_cast<const float*>(clen.data_ptr()),                           \
        static_cast<const int*>(seed.data_ptr()),                             \
        static_cast<const WEIGHT_C_T*>(vector.data_ptr()),                    \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                          \
        m, k                                                                  \
    );                                                                        \
}

// @tvm_ffi jitsmv_scatter_f32
FFI_JITSMV_SCATTER(_f32, float)
// @tvm_ffi jitsmv_scatter_f64
FFI_JITSMV_SCATTER(_f64, double)
// @tvm_ffi jitsmv_scatter_f16
FFI_JITSMV_SCATTER(_f16, __half)
// @tvm_ffi jitsmv_scatter_bf16
FFI_JITSMV_SCATTER(_bf16, __nv_bfloat16)


// #########################################################################
// ##  3. jitsmm — Float Matrix-Matrix Product                            ##
// #########################################################################

// =========================================================================
// Gather kernel (corder=true): one thread per output row
// Y[i, :] = w * sum_{j in C(i)} B[j, :]
// =========================================================================

#define DEFINE_JITSMM_GATHER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _jitsmm_gather_kern##SUFFIX(                                      \
    const WEIGHT_T* __restrict__ weight,                                          \
    const float*    __restrict__ clen,                                            \
    const int*      __restrict__ seed,                                            \
    const WEIGHT_T* __restrict__ B,                                               \
    WEIGHT_T*       __restrict__ output,                                          \
    int m, int k, int n                                                           \
) {                                                                               \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                \
    if (i >= m) return;                                                           \
    ACC_T w0 = READ_W(weight[0]);                                                 \
    unsigned int cl = (unsigned int)clen[0];                                       \
    if (cl < 2) cl = 2;                                                           \
    curandStatePhilox4_32_10_t state;                                              \
    curand_init((unsigned long long)seed[0], (unsigned long long)i, 0ULL, &state); \
    unsigned int j = curand(&state) % cl;                                          \
    /* Loop over connected rows, accumulate all columns */                         \
    while (j < (unsigned int)k) {                                                  \
        const WEIGHT_T* b_row = B + (size_t)j * n;                                \
        WEIGHT_T* out_row = output + (size_t)i * n;                                \
        for (int col = 0; col < n; col++) {                                        \
            ACC_T cur = READ_W(out_row[col]);                                      \
            cur += READ_W(b_row[col]);                                             \
            out_row[col] = WRITE_W(cur);                                           \
        }                                                                          \
        j += 1 + (curand(&state) % (cl - 1));                                     \
    }                                                                              \
    /* Scale by weight */                                                          \
    WEIGHT_T* out_row = output + (size_t)i * n;                                    \
    for (int col = 0; col < n; col++) {                                            \
        ACC_T cur = READ_W(out_row[col]);                                          \
        out_row[col] = WRITE_W(w0 * cur);                                          \
    }                                                                              \
}

DEFINE_JITSMM_GATHER(_f32,  float,         float,  READ_F32,  WRITE_F32,  0.0f)
DEFINE_JITSMM_GATHER(_f64,  double,        double, READ_F64,  WRITE_F64,  0.0)
DEFINE_JITSMM_GATHER(_f16,  __half,        float,  READ_F16,  WRITE_F16,  0.0f)
DEFINE_JITSMM_GATHER(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, 0.0f)

// =========================================================================
// Scatter kernel (corder=false): one thread per input row
// For each input row j, atomicAdd w * B[j, col] to Y[connected_rows, col].
// =========================================================================

#define DEFINE_JITSMM_SCATTER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ATOMIC_ADD) \
__global__ void _jitsmm_scatter_kern##SUFFIX(                                         \
    const WEIGHT_T* __restrict__ weight,                                              \
    const float*    __restrict__ clen,                                                \
    const int*      __restrict__ seed,                                                \
    const WEIGHT_T* __restrict__ B,                                                   \
    WEIGHT_T*       __restrict__ output,                                              \
    int m, int k, int n                                                               \
) {                                                                                   \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                                    \
    if (j >= k) return;                                                               \
    ACC_T w0 = READ_W(weight[0]);                                                     \
    unsigned int cl = (unsigned int)clen[0];                                           \
    if (cl < 2) cl = 2;                                                               \
    curandStatePhilox4_32_10_t state;                                                  \
    curand_init((unsigned long long)seed[0], (unsigned long long)j, 0ULL, &state);     \
    unsigned int i = curand(&state) % cl;                                              \
    const WEIGHT_T* b_row = B + (size_t)j * n;                                        \
    while (i < (unsigned int)m) {                                                      \
        WEIGHT_T* out_row = output + (size_t)i * n;                                    \
        for (int col = 0; col < n; col++) {                                            \
            ACC_T val = w0 * READ_W(b_row[col]);                                       \
            ATOMIC_ADD(&out_row[col], val);                                            \
        }                                                                              \
        i += 1 + (curand(&state) % (cl - 1));                                         \
    }                                                                                  \
}

DEFINE_JITSMM_SCATTER(_f32,  float,         float,  READ_F32,  WRITE_F32,  atomicAdd_f32)
DEFINE_JITSMM_SCATTER(_f64,  double,        double, READ_F64,  WRITE_F64,  atomicAdd_f64)
DEFINE_JITSMM_SCATTER(_f16,  __half,        float,  READ_F16,  WRITE_F16,  atomicAdd_f16)
DEFINE_JITSMM_SCATTER(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, atomicAdd_bf16)

// ---- TVM FFI: jitsmm gather ----

#define FFI_JITSMM_GATHER(SUFFIX, WEIGHT_C_T)                                \
void jitsmm_gather##SUFFIX(                                                   \
    tvm::ffi::TensorView weight,                                              \
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
    _jitsmm_gather_kern##SUFFIX<<<blocks, threads, 0, s>>>(                   \
        static_cast<const WEIGHT_C_T*>(weight.data_ptr()),                    \
        static_cast<const float*>(clen.data_ptr()),                           \
        static_cast<const int*>(seed.data_ptr()),                             \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                         \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                          \
        m, k, n                                                               \
    );                                                                        \
}

// @tvm_ffi jitsmm_gather_f32
FFI_JITSMM_GATHER(_f32, float)
// @tvm_ffi jitsmm_gather_f64
FFI_JITSMM_GATHER(_f64, double)
// @tvm_ffi jitsmm_gather_f16
FFI_JITSMM_GATHER(_f16, __half)
// @tvm_ffi jitsmm_gather_bf16
FFI_JITSMM_GATHER(_bf16, __nv_bfloat16)

// ---- TVM FFI: jitsmm scatter ----

#define FFI_JITSMM_SCATTER(SUFFIX, WEIGHT_C_T)                                \
void jitsmm_scatter##SUFFIX(                                                   \
    tvm::ffi::TensorView weight,                                               \
    tvm::ffi::TensorView clen,                                                 \
    tvm::ffi::TensorView seed,                                                 \
    tvm::ffi::TensorView B,                                                    \
    tvm::ffi::TensorView output,                                               \
    int64_t stream                                                             \
) {                                                                            \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                   \
    int m = static_cast<int>(output.size(0));                                  \
    int n = static_cast<int>(output.size(1));                                  \
    int k = static_cast<int>(B.size(0));                                       \
    cudaMemsetAsync(output.data_ptr(), 0,                                      \
        (size_t)m * n * sizeof(WEIGHT_C_T), s);                                \
    int threads = 256;                                                         \
    int blocks = (k + threads - 1) / threads;                                  \
    _jitsmm_scatter_kern##SUFFIX<<<blocks, threads, 0, s>>>(                   \
        static_cast<const WEIGHT_C_T*>(weight.data_ptr()),                     \
        static_cast<const float*>(clen.data_ptr()),                            \
        static_cast<const int*>(seed.data_ptr()),                              \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                          \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                           \
        m, k, n                                                                \
    );                                                                         \
}

// @tvm_ffi jitsmm_scatter_f32
FFI_JITSMM_SCATTER(_f32, float)
// @tvm_ffi jitsmm_scatter_f64
FFI_JITSMM_SCATTER(_f64, double)
// @tvm_ffi jitsmm_scatter_f16
FFI_JITSMM_SCATTER(_f16, __half)
// @tvm_ffi jitsmm_scatter_bf16
FFI_JITSMM_SCATTER(_bf16, __nv_bfloat16)


// #########################################################################
// ##  4. binary_jitsmv — Event-Driven Matrix-Vector Product              ##
// #########################################################################

// =========================================================================
// Gather kernel (corder=true): one thread per output element
// y[i] = w * count{j in C(i) : spike[j]}
// =========================================================================

#define DEFINE_BINARY_JITSMV_GATHER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, SPIKE_T, IS_ACTIVE, ACC_ZERO) \
__global__ void _binary_jitsmv_gather_kern##SUFFIX(                             \
    const WEIGHT_T* __restrict__ weight,                                        \
    const float*    __restrict__ clen,                                          \
    const int*      __restrict__ seed,                                          \
    const SPIKE_T*  __restrict__ vector,                                        \
    WEIGHT_T*       __restrict__ output,                                        \
    int m, int k                                                                \
) {                                                                             \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                              \
    if (i >= m) return;                                                         \
    ACC_T w0 = READ_W(weight[0]);                                               \
    unsigned int cl = (unsigned int)clen[0];                                     \
    if (cl < 2) cl = 2;                                                         \
    curandStatePhilox4_32_10_t state;                                            \
    curand_init((unsigned long long)seed[0], (unsigned long long)i, 0ULL, &state);\
    unsigned int j = curand(&state) % cl;                                        \
    ACC_T acc = ACC_ZERO;                                                        \
    while (j < (unsigned int)k) {                                                \
        if (IS_ACTIVE(vector, j)) {                                              \
            acc += (ACC_T)1.0;                                                   \
        }                                                                        \
        j += 1 + (curand(&state) % (cl - 1));                                   \
    }                                                                            \
    output[i] = WRITE_W(w0 * acc);                                               \
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
__global__ void _binary_jitsmv_scatter_kern##SUFFIX(                            \
    const WEIGHT_T* __restrict__ weight,                                        \
    const float*    __restrict__ clen,                                          \
    const int*      __restrict__ seed,                                          \
    const SPIKE_T*  __restrict__ vector,                                        \
    WEIGHT_T*       __restrict__ output,                                        \
    int m, int k                                                                \
) {                                                                             \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                              \
    if (j >= k) return;                                                         \
    if (!IS_ACTIVE(vector, j)) return;                                          \
    ACC_T w0 = READ_W(weight[0]);                                               \
    unsigned int cl = (unsigned int)clen[0];                                     \
    if (cl < 2) cl = 2;                                                         \
    curandStatePhilox4_32_10_t state;                                            \
    curand_init((unsigned long long)seed[0], (unsigned long long)j, 0ULL, &state);\
    unsigned int i = curand(&state) % cl;                                        \
    while (i < (unsigned int)m) {                                                \
        ATOMIC_ADD(&output[i], w0);                                              \
        i += 1 + (curand(&state) % (cl - 1));                                   \
    }                                                                            \
}

// f32 weight + bool/float spikes
DEFINE_BINARY_JITSMV_SCATTER(_f32_bool,  float,         float,  READ_F32,  WRITE_F32,  int8_t, IS_ACTIVE_BOOL,  atomicAdd_f32)
DEFINE_BINARY_JITSMV_SCATTER(_f32_float, float,         float,  READ_F32,  WRITE_F32,  float,  IS_ACTIVE_FLOAT, atomicAdd_f32)
// f64 weight + bool/float spikes
DEFINE_BINARY_JITSMV_SCATTER(_f64_bool,  double,        double, READ_F64,  WRITE_F64,  int8_t, IS_ACTIVE_BOOL,  atomicAdd_f64)
DEFINE_BINARY_JITSMV_SCATTER(_f64_float, double,        double, READ_F64,  WRITE_F64,  float,  IS_ACTIVE_FLOAT, atomicAdd_f64)
// f16 weight + bool/float spikes
DEFINE_BINARY_JITSMV_SCATTER(_f16_bool,  __half,        float,  READ_F16,  WRITE_F16,  int8_t, IS_ACTIVE_BOOL,  atomicAdd_f16)
DEFINE_BINARY_JITSMV_SCATTER(_f16_float, __half,        float,  READ_F16,  WRITE_F16,  float,  IS_ACTIVE_FLOAT, atomicAdd_f16)
// bf16 weight + bool/float spikes
DEFINE_BINARY_JITSMV_SCATTER(_bf16_bool, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, int8_t, IS_ACTIVE_BOOL,  atomicAdd_bf16)
DEFINE_BINARY_JITSMV_SCATTER(_bf16_float,__nv_bfloat16, float,  READ_BF16, WRITE_BF16, float,  IS_ACTIVE_FLOAT, atomicAdd_bf16)

// ---- TVM FFI: binary_jitsmv gather ----

#define FFI_BINARY_JITSMV_GATHER(SUFFIX, WEIGHT_C_T, SPIKE_C_T)              \
void binary_jitsmv_gather##SUFFIX(                                            \
    tvm::ffi::TensorView weight,                                              \
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
    _binary_jitsmv_gather_kern##SUFFIX<<<blocks, threads, 0, s>>>(            \
        static_cast<const WEIGHT_C_T*>(weight.data_ptr()),                    \
        static_cast<const float*>(clen.data_ptr()),                           \
        static_cast<const int*>(seed.data_ptr()),                             \
        static_cast<const SPIKE_C_T*>(vector.data_ptr()),                     \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                          \
        m, k                                                                  \
    );                                                                        \
}

// @tvm_ffi binary_jitsmv_gather_f32_bool
FFI_BINARY_JITSMV_GATHER(_f32_bool,  float,         int8_t)
// @tvm_ffi binary_jitsmv_gather_f32_float
FFI_BINARY_JITSMV_GATHER(_f32_float, float,         float)
// @tvm_ffi binary_jitsmv_gather_f64_bool
FFI_BINARY_JITSMV_GATHER(_f64_bool,  double,        int8_t)
// @tvm_ffi binary_jitsmv_gather_f64_float
FFI_BINARY_JITSMV_GATHER(_f64_float, double,        float)
// @tvm_ffi binary_jitsmv_gather_f16_bool
FFI_BINARY_JITSMV_GATHER(_f16_bool,  __half,        int8_t)
// @tvm_ffi binary_jitsmv_gather_f16_float
FFI_BINARY_JITSMV_GATHER(_f16_float, __half,        float)
// @tvm_ffi binary_jitsmv_gather_bf16_bool
FFI_BINARY_JITSMV_GATHER(_bf16_bool, __nv_bfloat16, int8_t)
// @tvm_ffi binary_jitsmv_gather_bf16_float
FFI_BINARY_JITSMV_GATHER(_bf16_float,__nv_bfloat16, float)

// ---- TVM FFI: binary_jitsmv scatter ----

#define FFI_BINARY_JITSMV_SCATTER(SUFFIX, WEIGHT_C_T, SPIKE_C_T)             \
void binary_jitsmv_scatter##SUFFIX(                                           \
    tvm::ffi::TensorView weight,                                              \
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
    _binary_jitsmv_scatter_kern##SUFFIX<<<blocks, threads, 0, s>>>(           \
        static_cast<const WEIGHT_C_T*>(weight.data_ptr()),                    \
        static_cast<const float*>(clen.data_ptr()),                           \
        static_cast<const int*>(seed.data_ptr()),                             \
        static_cast<const SPIKE_C_T*>(vector.data_ptr()),                     \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                          \
        m, k                                                                  \
    );                                                                        \
}

// @tvm_ffi binary_jitsmv_scatter_f32_bool
FFI_BINARY_JITSMV_SCATTER(_f32_bool,  float,         int8_t)
// @tvm_ffi binary_jitsmv_scatter_f32_float
FFI_BINARY_JITSMV_SCATTER(_f32_float, float,         float)
// @tvm_ffi binary_jitsmv_scatter_f64_bool
FFI_BINARY_JITSMV_SCATTER(_f64_bool,  double,        int8_t)
// @tvm_ffi binary_jitsmv_scatter_f64_float
FFI_BINARY_JITSMV_SCATTER(_f64_float, double,        float)
// @tvm_ffi binary_jitsmv_scatter_f16_bool
FFI_BINARY_JITSMV_SCATTER(_f16_bool,  __half,        int8_t)
// @tvm_ffi binary_jitsmv_scatter_f16_float
FFI_BINARY_JITSMV_SCATTER(_f16_float, __half,        float)
// @tvm_ffi binary_jitsmv_scatter_bf16_bool
FFI_BINARY_JITSMV_SCATTER(_bf16_bool, __nv_bfloat16, int8_t)
// @tvm_ffi binary_jitsmv_scatter_bf16_float
FFI_BINARY_JITSMV_SCATTER(_bf16_float,__nv_bfloat16, float)


// #########################################################################
// ##  5. binary_jitsmm — Event-Driven Matrix-Matrix Product              ##
// #########################################################################

// =========================================================================
// Gather kernel (corder=true): one thread per output row
// Y[i, col] = w * sum_{j in C(i)} active(B[j, col])
// =========================================================================

#define DEFINE_BINARY_JITSMM_GATHER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, SPIKE_T, IS_ACTIVE, ACC_ZERO) \
__global__ void _binary_jitsmm_gather_kern##SUFFIX(                             \
    const WEIGHT_T* __restrict__ weight,                                        \
    const float*    __restrict__ clen,                                          \
    const int*      __restrict__ seed,                                          \
    const SPIKE_T*  __restrict__ B,                                             \
    WEIGHT_T*       __restrict__ output,                                        \
    int m, int k, int n                                                         \
) {                                                                             \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                              \
    if (i >= m) return;                                                         \
    ACC_T w0 = READ_W(weight[0]);                                               \
    unsigned int cl = (unsigned int)clen[0];                                     \
    if (cl < 2) cl = 2;                                                         \
    curandStatePhilox4_32_10_t state;                                            \
    curand_init((unsigned long long)seed[0], (unsigned long long)i, 0ULL, &state);\
    unsigned int j = curand(&state) % cl;                                        \
    WEIGHT_T* out_row = output + (size_t)i * n;                                  \
    while (j < (unsigned int)k) {                                                \
        const SPIKE_T* b_row = B + (size_t)j * n;                               \
        for (int col = 0; col < n; col++) {                                      \
            if (IS_ACTIVE(b_row, col)) {                                         \
                ACC_T cur = READ_W(out_row[col]);                                \
                out_row[col] = WRITE_W(cur + (ACC_T)1.0);                        \
            }                                                                    \
        }                                                                        \
        j += 1 + (curand(&state) % (cl - 1));                                   \
    }                                                                            \
    /* Scale by weight */                                                        \
    for (int col = 0; col < n; col++) {                                          \
        ACC_T cur = READ_W(out_row[col]);                                        \
        out_row[col] = WRITE_W(w0 * cur);                                        \
    }                                                                            \
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
__global__ void _binary_jitsmm_scatter_kern##SUFFIX(                            \
    const WEIGHT_T* __restrict__ weight,                                        \
    const float*    __restrict__ clen,                                          \
    const int*      __restrict__ seed,                                          \
    const SPIKE_T*  __restrict__ B,                                             \
    WEIGHT_T*       __restrict__ output,                                        \
    int m, int k, int n                                                         \
) {                                                                             \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                              \
    if (j >= k) return;                                                         \
    ACC_T w0 = READ_W(weight[0]);                                               \
    unsigned int cl = (unsigned int)clen[0];                                     \
    if (cl < 2) cl = 2;                                                         \
    const SPIKE_T* b_row = B + (size_t)j * n;                                   \
    curandStatePhilox4_32_10_t state;                                            \
    curand_init((unsigned long long)seed[0], (unsigned long long)j, 0ULL, &state);\
    unsigned int i = curand(&state) % cl;                                        \
    while (i < (unsigned int)m) {                                                \
        WEIGHT_T* out_row = output + (size_t)i * n;                              \
        for (int col = 0; col < n; col++) {                                      \
            if (IS_ACTIVE(b_row, col)) {                                         \
                ATOMIC_ADD(&out_row[col], w0);                                   \
            }                                                                    \
        }                                                                        \
        i += 1 + (curand(&state) % (cl - 1));                                   \
    }                                                                            \
}

// f32 + bool/float
DEFINE_BINARY_JITSMM_SCATTER(_f32_bool,  float,         float,  READ_F32,  WRITE_F32,  int8_t, IS_ACTIVE_BOOL,  atomicAdd_f32)
DEFINE_BINARY_JITSMM_SCATTER(_f32_float, float,         float,  READ_F32,  WRITE_F32,  float,  IS_ACTIVE_FLOAT, atomicAdd_f32)
// f64 + bool/float
DEFINE_BINARY_JITSMM_SCATTER(_f64_bool,  double,        double, READ_F64,  WRITE_F64,  int8_t, IS_ACTIVE_BOOL,  atomicAdd_f64)
DEFINE_BINARY_JITSMM_SCATTER(_f64_float, double,        double, READ_F64,  WRITE_F64,  float,  IS_ACTIVE_FLOAT, atomicAdd_f64)
// f16 + bool/float
DEFINE_BINARY_JITSMM_SCATTER(_f16_bool,  __half,        float,  READ_F16,  WRITE_F16,  int8_t, IS_ACTIVE_BOOL,  atomicAdd_f16)
DEFINE_BINARY_JITSMM_SCATTER(_f16_float, __half,        float,  READ_F16,  WRITE_F16,  float,  IS_ACTIVE_FLOAT, atomicAdd_f16)
// bf16 + bool/float
DEFINE_BINARY_JITSMM_SCATTER(_bf16_bool, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, int8_t, IS_ACTIVE_BOOL,  atomicAdd_bf16)
DEFINE_BINARY_JITSMM_SCATTER(_bf16_float,__nv_bfloat16, float,  READ_BF16, WRITE_BF16, float,  IS_ACTIVE_FLOAT, atomicAdd_bf16)

// ---- TVM FFI: binary_jitsmm gather ----

#define FFI_BINARY_JITSMM_GATHER(SUFFIX, WEIGHT_C_T, SPIKE_C_T)              \
void binary_jitsmm_gather##SUFFIX(                                            \
    tvm::ffi::TensorView weight,                                              \
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
    _binary_jitsmm_gather_kern##SUFFIX<<<blocks, threads, 0, s>>>(            \
        static_cast<const WEIGHT_C_T*>(weight.data_ptr()),                    \
        static_cast<const float*>(clen.data_ptr()),                           \
        static_cast<const int*>(seed.data_ptr()),                             \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                          \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                          \
        m, k, n                                                               \
    );                                                                        \
}

// @tvm_ffi binary_jitsmm_gather_f32_bool
FFI_BINARY_JITSMM_GATHER(_f32_bool,  float,         int8_t)
// @tvm_ffi binary_jitsmm_gather_f32_float
FFI_BINARY_JITSMM_GATHER(_f32_float, float,         float)
// @tvm_ffi binary_jitsmm_gather_f64_bool
FFI_BINARY_JITSMM_GATHER(_f64_bool,  double,        int8_t)
// @tvm_ffi binary_jitsmm_gather_f64_float
FFI_BINARY_JITSMM_GATHER(_f64_float, double,        float)
// @tvm_ffi binary_jitsmm_gather_f16_bool
FFI_BINARY_JITSMM_GATHER(_f16_bool,  __half,        int8_t)
// @tvm_ffi binary_jitsmm_gather_f16_float
FFI_BINARY_JITSMM_GATHER(_f16_float, __half,        float)
// @tvm_ffi binary_jitsmm_gather_bf16_bool
FFI_BINARY_JITSMM_GATHER(_bf16_bool, __nv_bfloat16, int8_t)
// @tvm_ffi binary_jitsmm_gather_bf16_float
FFI_BINARY_JITSMM_GATHER(_bf16_float,__nv_bfloat16, float)

// ---- TVM FFI: binary_jitsmm scatter ----

#define FFI_BINARY_JITSMM_SCATTER(SUFFIX, WEIGHT_C_T, SPIKE_C_T)             \
void binary_jitsmm_scatter##SUFFIX(                                           \
    tvm::ffi::TensorView weight,                                              \
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
    _binary_jitsmm_scatter_kern##SUFFIX<<<blocks, threads, 0, s>>>(           \
        static_cast<const WEIGHT_C_T*>(weight.data_ptr()),                    \
        static_cast<const float*>(clen.data_ptr()),                           \
        static_cast<const int*>(seed.data_ptr()),                             \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                          \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                          \
        m, k, n                                                               \
    );                                                                        \
}

// @tvm_ffi binary_jitsmm_scatter_f32_bool
FFI_BINARY_JITSMM_SCATTER(_f32_bool,  float,         int8_t)
// @tvm_ffi binary_jitsmm_scatter_f32_float
FFI_BINARY_JITSMM_SCATTER(_f32_float, float,         float)
// @tvm_ffi binary_jitsmm_scatter_f64_bool
FFI_BINARY_JITSMM_SCATTER(_f64_bool,  double,        int8_t)
// @tvm_ffi binary_jitsmm_scatter_f64_float
FFI_BINARY_JITSMM_SCATTER(_f64_float, double,        float)
// @tvm_ffi binary_jitsmm_scatter_f16_bool
FFI_BINARY_JITSMM_SCATTER(_f16_bool,  __half,        int8_t)
// @tvm_ffi binary_jitsmm_scatter_f16_float
FFI_BINARY_JITSMM_SCATTER(_f16_float, __half,        float)
// @tvm_ffi binary_jitsmm_scatter_bf16_bool
FFI_BINARY_JITSMM_SCATTER(_bf16_bool, __nv_bfloat16, int8_t)
// @tvm_ffi binary_jitsmm_scatter_bf16_float
FFI_BINARY_JITSMM_SCATTER(_bf16_float,__nv_bfloat16, float)
