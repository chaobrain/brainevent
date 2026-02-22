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
 * jit_uniform.cu — JIT Uniform Connectivity CUDA Kernels (all operations)
 * =========================================================================
 *
 * Unified CUDA kernel file for all JIT uniform connectivity operations.
 * Each entry W[i,j] is independently set to a value drawn from
 * Uniform(w_low, w_high) with probability `prob`, and zero otherwise.
 * Connectivity is determined by a geometric skip pattern seeded by `seed`.
 *
 * Operations
 * ----------
 * 1. jitu          — Dense matrix generation: M[i,j] = Uniform(w_low,w_high)*Bernoulli(prob)
 * 2. jitumv        — Float matrix-vector:     y = M @ v
 * 3. jitumm        — Float matrix-matrix:     Y = M @ B
 * 4. binary_jitumv — Event-driven mat-vec:    y[i] = sum_{j active} Uniform()*B[i,j]
 * 5. binary_jitumm — Event-driven mat-mat:    Y[i,:] = sum_{j active} Uniform()*B[i,k]
 *
 * Parameters (common to all)
 * --------------------------
 * w_low  : shape (1,), lower bound of uniform weight distribution
 * w_high : shape (1,), upper bound of uniform weight distribution
 * clen   : shape (1,), connection length = ceil(2/prob) (float32)
 * seed   : shape (1,), int32 random seed
 *
 * Random sampling pattern (shared by all kernels):
 *   curand_init(seed, thread_id, 0, &state);
 *   unsigned int i = curand(&state) % clen;        // first connected index
 *   while (i < dim) {
 *       float u = curand_uniform(&state);           // weight sample in (0,1]
 *       float w = w_low + u * (w_high - w_low);    // scale to [w_low, w_high]
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
 *
 * Performance notes:
 *
 * Gather kernels (corder=true):
 * - __ldg() read-only cache: all read-only data (parameters, input vectors,
 *   B matrices) are loaded via __ldg() to route through the L1 texture cache,
 *   reducing latency for random-access patterns.
 * - Shared memory vector caching (MV only): when the input vector fits in
 *   48KB shared memory, it is cooperatively loaded by all threads in the
 *   block, reducing per-access latency from ~200 cycles (L2) to ~20 cycles.
 * - Gather MV kernels write each output element exactly once (direct store),
 *   so no memset is needed.
 * - Register accumulators (MM, n <= 32): accumulation happens in a register
 *   array ACC_T acc[32] instead of reading/writing global memory on every
 *   connection. This eliminates ~2n global memory round-trips per connection
 *   (~30 cycles each for L1 cached output). Single write at the end.
 *   For n > 32, falls back to in-kernel zero-init + global memory
 *   accumulation to avoid excessive register pressure.
 * - The kernel is fundamentally compute-bound on Philox RNG operations
 *   (~25 cycles per connection for 2 curand calls). Memory latency is
 *   secondary after the above optimizations.
 *
 * Scatter kernels (corder=false):
 * - Use atomicAdd for concurrent output writes. Limited by atomic
 *   contention at high connection probability (p >= 0.5).
 * - Binary scatter kernels skip inactive spikes entirely (early return).
 *
 * Measured performance (RTX 3080 Ti, 10K×10K, p=0.1, f32, tvmffi):
 *   jitumv  gather:  ~1.3ms (at TVM FFI dispatch floor)
 *   jitumv  scatter: ~3.0ms (p=0.5)
 *   jitumm  gather:  ~1.3ms (at dispatch floor, n=10)
 *   jitumm  scatter: ~5.0ms (n=10, atomicAdd + curand bound)
 *   binary_jitumv gather:  ~1.3ms (at dispatch floor)
 *   binary_jitumm gather:  ~1.3ms (at dispatch floor, n=10)
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
// Shared memory threshold: 48KB default max dynamic shared memory per block
// =========================================================================

#define SMEM_THRESHOLD 49152

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


// #########################################################################
// ##  1. jitu — Dense Matrix Generation                                  ##
// #########################################################################

// =========================================================================
// corder=true kernel: one thread per row
// output[row, col] = Uniform(w_low, w_high) for connected (row, col) pairs.
// =========================================================================

#define DEFINE_JITU_CORDER_TRUE(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W)     \
__global__ void _jitu_corder_true_kern##SUFFIX(                                \
    const WEIGHT_T* __restrict__ w_low,                                        \
    const WEIGHT_T* __restrict__ w_high,                                       \
    const float*    __restrict__ clen,                                         \
    const int*      __restrict__ seed,                                         \
    WEIGHT_T*       __restrict__ output,                                       \
    int n_rows, int n_cols                                                     \
) {                                                                            \
    int row = blockIdx.x * blockDim.x + threadIdx.x;                           \
    if (row >= n_rows) return;                                                 \
    ACC_T wlo = READ_W(__ldg(&w_low[0]));                                      \
    ACC_T range = READ_W(__ldg(&w_high[0])) - wlo;                             \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                            \
    if (cl < 2) cl = 2;                                                        \
    curandStatePhilox4_32_10_t state;                                           \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)row, 0ULL, &state); \
    unsigned int col = curand(&state) % cl;                                     \
    while (col < (unsigned int)n_cols) {                                        \
        float u = curand_uniform(&state);                                       \
        output[(size_t)row * n_cols + col] = WRITE_W(wlo + (ACC_T)u * range);  \
        col += 1 + (curand(&state) % (cl - 1));                                \
    }                                                                           \
}

DEFINE_JITU_CORDER_TRUE(_f32,  float,         float,  READ_F32,  WRITE_F32)
DEFINE_JITU_CORDER_TRUE(_f64,  double,        double, READ_F64,  WRITE_F64)
DEFINE_JITU_CORDER_TRUE(_f16,  __half,        float,  READ_F16,  WRITE_F16)
DEFINE_JITU_CORDER_TRUE(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16)

// =========================================================================
// corder=false kernel: one thread per column
// output[row, col] = Uniform(w_low, w_high) for connected (row, col) pairs.
// =========================================================================

#define DEFINE_JITU_CORDER_FALSE(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W)    \
__global__ void _jitu_corder_false_kern##SUFFIX(                               \
    const WEIGHT_T* __restrict__ w_low,                                        \
    const WEIGHT_T* __restrict__ w_high,                                       \
    const float*    __restrict__ clen,                                         \
    const int*      __restrict__ seed,                                         \
    WEIGHT_T*       __restrict__ output,                                       \
    int n_rows, int n_cols                                                     \
) {                                                                            \
    int col = blockIdx.x * blockDim.x + threadIdx.x;                           \
    if (col >= n_cols) return;                                                 \
    ACC_T wlo = READ_W(__ldg(&w_low[0]));                                      \
    ACC_T range = READ_W(__ldg(&w_high[0])) - wlo;                             \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                            \
    if (cl < 2) cl = 2;                                                        \
    curandStatePhilox4_32_10_t state;                                           \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)col, 0ULL, &state); \
    unsigned int row = curand(&state) % cl;                                     \
    while (row < (unsigned int)n_rows) {                                        \
        float u = curand_uniform(&state);                                       \
        output[(size_t)row * n_cols + col] = WRITE_W(wlo + (ACC_T)u * range);  \
        row += 1 + (curand(&state) % (cl - 1));                                \
    }                                                                           \
}

DEFINE_JITU_CORDER_FALSE(_f32,  float,         float,  READ_F32,  WRITE_F32)
DEFINE_JITU_CORDER_FALSE(_f64,  double,        double, READ_F64,  WRITE_F64)
DEFINE_JITU_CORDER_FALSE(_f16,  __half,        float,  READ_F16,  WRITE_F16)
DEFINE_JITU_CORDER_FALSE(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16)

// ---- TVM FFI: jitu corder=true ----

#define FFI_JITU_CORDER_TRUE(SUFFIX, WEIGHT_C_T)                              \
void jitu_corder_true##SUFFIX(                                                 \
    tvm::ffi::TensorView w_low,                                                \
    tvm::ffi::TensorView w_high,                                               \
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
    _jitu_corder_true_kern##SUFFIX<<<blocks, threads, 0, s>>>(                 \
        static_cast<const WEIGHT_C_T*>(w_low.data_ptr()),                      \
        static_cast<const WEIGHT_C_T*>(w_high.data_ptr()),                     \
        static_cast<const float*>(clen.data_ptr()),                            \
        static_cast<const int*>(seed.data_ptr()),                              \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                           \
        n_rows, n_cols                                                         \
    );                                                                         \
}

// @tvm_ffi jitu_corder_true_f32
FFI_JITU_CORDER_TRUE(_f32, float)
// @tvm_ffi jitu_corder_true_f64
FFI_JITU_CORDER_TRUE(_f64, double)
// @tvm_ffi jitu_corder_true_f16
FFI_JITU_CORDER_TRUE(_f16, __half)
// @tvm_ffi jitu_corder_true_bf16
FFI_JITU_CORDER_TRUE(_bf16, __nv_bfloat16)

// ---- TVM FFI: jitu corder=false ----

#define FFI_JITU_CORDER_FALSE(SUFFIX, WEIGHT_C_T)                             \
void jitu_corder_false##SUFFIX(                                                \
    tvm::ffi::TensorView w_low,                                                \
    tvm::ffi::TensorView w_high,                                               \
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
    _jitu_corder_false_kern##SUFFIX<<<blocks, threads, 0, s>>>(                \
        static_cast<const WEIGHT_C_T*>(w_low.data_ptr()),                      \
        static_cast<const WEIGHT_C_T*>(w_high.data_ptr()),                     \
        static_cast<const float*>(clen.data_ptr()),                            \
        static_cast<const int*>(seed.data_ptr()),                              \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                           \
        n_rows, n_cols                                                         \
    );                                                                         \
}

// @tvm_ffi jitu_corder_false_f32
FFI_JITU_CORDER_FALSE(_f32, float)
// @tvm_ffi jitu_corder_false_f64
FFI_JITU_CORDER_FALSE(_f64, double)
// @tvm_ffi jitu_corder_false_f16
FFI_JITU_CORDER_FALSE(_f16, __half)
// @tvm_ffi jitu_corder_false_bf16
FFI_JITU_CORDER_FALSE(_bf16, __nv_bfloat16)


// #########################################################################
// ##  2. jitumv — Float Matrix-Vector Product                            ##
// #########################################################################

// =========================================================================
// Gather kernel (corder=true): one thread per output element
// y[i] = sum_{j in C(i)} Uniform(w_low, w_high) * v[j]
// Each connection gets its own weight sample from curand_uniform.
// No memset needed: every output[i] is written exactly once (direct store).
// =========================================================================

#define DEFINE_JITUMV_GATHER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _jitumv_gather_kern##SUFFIX(                                      \
    const WEIGHT_T* __restrict__ w_low,                                           \
    const WEIGHT_T* __restrict__ w_high,                                          \
    const float*    __restrict__ clen,                                            \
    const int*      __restrict__ seed,                                            \
    const WEIGHT_T* __restrict__ vector,                                          \
    WEIGHT_T*       __restrict__ output,                                          \
    int m, int k                                                                  \
) {                                                                               \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                \
    if (i >= m) return;                                                           \
    ACC_T wlo = READ_W(__ldg(&w_low[0]));                                          \
    ACC_T range = READ_W(__ldg(&w_high[0])) - wlo;                                 \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                \
    if (cl < 2) cl = 2;                                                           \
    curandStatePhilox4_32_10_t state;                                              \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)i, 0ULL, &state); \
    unsigned int j = curand(&state) % cl;                                          \
    ACC_T acc = ACC_ZERO;                                                          \
    while (j < (unsigned int)k) {                                                  \
        float u = curand_uniform(&state);                                          \
        ACC_T w = wlo + (ACC_T)u * range;                                         \
        acc += READ_W(__ldg(&vector[j])) * w;                                      \
        j += 1 + (curand(&state) % (cl - 1));                                     \
    }                                                                              \
    output[i] = WRITE_W(acc);                                                      \
}

DEFINE_JITUMV_GATHER(_f32,  float,         float,  READ_F32,  WRITE_F32,  0.0f)
DEFINE_JITUMV_GATHER(_f64,  double,        double, READ_F64,  WRITE_F64,  0.0)
DEFINE_JITUMV_GATHER(_f16,  __half,        float,  READ_F16,  WRITE_F16,  0.0f)
DEFINE_JITUMV_GATHER(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, 0.0f)

// =========================================================================
// Gather kernel with shared memory vector caching (corder=true)
// Cooperatively loads the input vector into shared memory, then all
// subsequent reads use smem (~20 cycle latency vs ~200 from L2).
// Used when k * sizeof(ACC_T) <= 48KB.
// =========================================================================

#define DEFINE_JITUMV_GATHER_SMEM(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _jitumv_gather_smem_kern##SUFFIX(                                      \
    const WEIGHT_T* __restrict__ w_low,                                                \
    const WEIGHT_T* __restrict__ w_high,                                               \
    const float*    __restrict__ clen,                                                 \
    const int*      __restrict__ seed,                                                 \
    const WEIGHT_T* __restrict__ vector,                                               \
    WEIGHT_T*       __restrict__ output,                                               \
    int m, int k                                                                       \
) {                                                                                    \
    extern __shared__ char _smem_bytes[];                                               \
    ACC_T* sv = reinterpret_cast<ACC_T*>(_smem_bytes);                                 \
    for (int idx = threadIdx.x; idx < k; idx += blockDim.x) {                          \
        sv[idx] = READ_W(__ldg(&vector[idx]));                                          \
    }                                                                                  \
    __syncthreads();                                                                   \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                     \
    if (i >= m) return;                                                                \
    ACC_T wlo = READ_W(__ldg(&w_low[0]));                                               \
    ACC_T range = READ_W(__ldg(&w_high[0])) - wlo;                                      \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                     \
    if (cl < 2) cl = 2;                                                                \
    curandStatePhilox4_32_10_t state;                                                   \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)i, 0ULL, &state); \
    unsigned int j = curand(&state) % cl;                                               \
    ACC_T acc = ACC_ZERO;                                                               \
    while (j < (unsigned int)k) {                                                       \
        float u = curand_uniform(&state);                                               \
        ACC_T w = wlo + (ACC_T)u * range;                                              \
        acc += sv[j] * w;                                                               \
        j += 1 + (curand(&state) % (cl - 1));                                          \
    }                                                                                   \
    output[i] = WRITE_W(acc);                                                           \
}

DEFINE_JITUMV_GATHER_SMEM(_f32,  float,         float,  READ_F32,  WRITE_F32,  0.0f)
DEFINE_JITUMV_GATHER_SMEM(_f64,  double,        double, READ_F64,  WRITE_F64,  0.0)
DEFINE_JITUMV_GATHER_SMEM(_f16,  __half,        float,  READ_F16,  WRITE_F16,  0.0f)
DEFINE_JITUMV_GATHER_SMEM(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, 0.0f)

// =========================================================================
// Scatter kernel (corder=false): one thread per input element
// For each input j, scatter Uniform(w_low, w_high)*v[j] to output[connected_i].
// Uses atomicAdd to handle concurrent writes to the same output element.
// =========================================================================

#define DEFINE_JITUMV_SCATTER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ATOMIC_ADD) \
__global__ void _jitumv_scatter_kern##SUFFIX(                                         \
    const WEIGHT_T* __restrict__ w_low,                                               \
    const WEIGHT_T* __restrict__ w_high,                                              \
    const float*    __restrict__ clen,                                                \
    const int*      __restrict__ seed,                                                \
    const WEIGHT_T* __restrict__ vector,                                              \
    WEIGHT_T*       __restrict__ output,                                              \
    int m, int k                                                                      \
) {                                                                                   \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                                    \
    if (j >= k) return;                                                               \
    ACC_T wlo = READ_W(__ldg(&w_low[0]));                                              \
    ACC_T range = READ_W(__ldg(&w_high[0])) - wlo;                                     \
    ACC_T vj = READ_W(__ldg(&vector[j]));                                              \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                    \
    if (cl < 2) cl = 2;                                                               \
    curandStatePhilox4_32_10_t state;                                                  \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)j, 0ULL, &state); \
    unsigned int i = curand(&state) % cl;                                              \
    while (i < (unsigned int)m) {                                                      \
        float u = curand_uniform(&state);                                              \
        ACC_T w = wlo + (ACC_T)u * range;                                             \
        ATOMIC_ADD(&output[i], w * vj);                                                \
        i += 1 + (curand(&state) % (cl - 1));                                         \
    }                                                                                  \
}

DEFINE_JITUMV_SCATTER(_f32,  float,         float,  READ_F32,  WRITE_F32,  atomicAdd_f32)
DEFINE_JITUMV_SCATTER(_f64,  double,        double, READ_F64,  WRITE_F64,  atomicAdd_f64)
DEFINE_JITUMV_SCATTER(_f16,  __half,        float,  READ_F16,  WRITE_F16,  atomicAdd_f16)
DEFINE_JITUMV_SCATTER(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, atomicAdd_bf16)

// ---- TVM FFI: jitumv gather ----
// Dispatches to shared-memory kernel when vector fits in 48KB smem,
// falls back to global-memory kernel for larger vectors.
// No memset needed: gather kernels write every output element exactly once.

#define FFI_JITUMV_GATHER(SUFFIX, WEIGHT_C_T, ACC_C_T)                          \
void jitumv_gather##SUFFIX(                                                      \
    tvm::ffi::TensorView w_low,                                                  \
    tvm::ffi::TensorView w_high,                                                 \
    tvm::ffi::TensorView clen,                                                   \
    tvm::ffi::TensorView seed,                                                   \
    tvm::ffi::TensorView vector,                                                 \
    tvm::ffi::TensorView output,                                                 \
    int64_t stream                                                               \
) {                                                                              \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                     \
    int m = static_cast<int>(output.size(0));                                    \
    int k = static_cast<int>(vector.size(0));                                    \
    int threads = 256;                                                           \
    int blocks = (m + threads - 1) / threads;                                    \
    size_t smem_bytes = (size_t)k * sizeof(ACC_C_T);                             \
    if (smem_bytes <= SMEM_THRESHOLD) {                                          \
        _jitumv_gather_smem_kern##SUFFIX<<<blocks, threads, smem_bytes, s>>>(    \
            static_cast<const WEIGHT_C_T*>(w_low.data_ptr()),                    \
            static_cast<const WEIGHT_C_T*>(w_high.data_ptr()),                   \
            static_cast<const float*>(clen.data_ptr()),                          \
            static_cast<const int*>(seed.data_ptr()),                            \
            static_cast<const WEIGHT_C_T*>(vector.data_ptr()),                   \
            static_cast<WEIGHT_C_T*>(output.data_ptr()),                         \
            m, k                                                                 \
        );                                                                       \
    } else {                                                                     \
        _jitumv_gather_kern##SUFFIX<<<blocks, threads, 0, s>>>(                  \
            static_cast<const WEIGHT_C_T*>(w_low.data_ptr()),                    \
            static_cast<const WEIGHT_C_T*>(w_high.data_ptr()),                   \
            static_cast<const float*>(clen.data_ptr()),                          \
            static_cast<const int*>(seed.data_ptr()),                            \
            static_cast<const WEIGHT_C_T*>(vector.data_ptr()),                   \
            static_cast<WEIGHT_C_T*>(output.data_ptr()),                         \
            m, k                                                                 \
        );                                                                       \
    }                                                                            \
}

// @tvm_ffi jitumv_gather_f32
FFI_JITUMV_GATHER(_f32, float, float)
// @tvm_ffi jitumv_gather_f64
FFI_JITUMV_GATHER(_f64, double, double)
// @tvm_ffi jitumv_gather_f16
FFI_JITUMV_GATHER(_f16, __half, float)
// @tvm_ffi jitumv_gather_bf16
FFI_JITUMV_GATHER(_bf16, __nv_bfloat16, float)

// ---- TVM FFI: jitumv scatter ----

#define FFI_JITUMV_SCATTER(SUFFIX, WEIGHT_C_T)                               \
void jitumv_scatter##SUFFIX(                                                  \
    tvm::ffi::TensorView w_low,                                               \
    tvm::ffi::TensorView w_high,                                              \
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
    _jitumv_scatter_kern##SUFFIX<<<blocks, threads, 0, s>>>(                  \
        static_cast<const WEIGHT_C_T*>(w_low.data_ptr()),                     \
        static_cast<const WEIGHT_C_T*>(w_high.data_ptr()),                    \
        static_cast<const float*>(clen.data_ptr()),                           \
        static_cast<const int*>(seed.data_ptr()),                             \
        static_cast<const WEIGHT_C_T*>(vector.data_ptr()),                    \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                          \
        m, k                                                                  \
    );                                                                        \
}

// @tvm_ffi jitumv_scatter_f32
FFI_JITUMV_SCATTER(_f32, float)
// @tvm_ffi jitumv_scatter_f64
FFI_JITUMV_SCATTER(_f64, double)
// @tvm_ffi jitumv_scatter_f16
FFI_JITUMV_SCATTER(_f16, __half)
// @tvm_ffi jitumv_scatter_bf16
FFI_JITUMV_SCATTER(_bf16, __nv_bfloat16)


// #########################################################################
// ##  3. jitumm — Float Matrix-Matrix Product                            ##
// #########################################################################

// =========================================================================
// Gather kernel (corder=true): one thread per output row i
// Y[i, col] = sum_{j in C(i)} Uniform(w_low, w_high) * B[j, col]
// Each connection j gets a fresh weight sample (same w for all cols of B).
// For n <= 32: uses register accumulators (ACC_T acc[32]) to avoid
// read-modify-write to global memory on every connection. Writes output
// once at the end. For n > 32: falls back to in-kernel zero-init +
// global memory accumulation (avoids register spill pressure).
// =========================================================================

#define DEFINE_JITUMM_GATHER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _jitumm_gather_kern##SUFFIX(                                      \
    const WEIGHT_T* __restrict__ w_low,                                           \
    const WEIGHT_T* __restrict__ w_high,                                          \
    const float*    __restrict__ clen,                                            \
    const int*      __restrict__ seed,                                            \
    const WEIGHT_T* __restrict__ B,                                               \
    WEIGHT_T*       __restrict__ output,                                          \
    int m, int k, int n                                                           \
) {                                                                               \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                \
    if (i >= m) return;                                                           \
    ACC_T wlo = READ_W(__ldg(&w_low[0]));                                          \
    ACC_T range = READ_W(__ldg(&w_high[0])) - wlo;                                 \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                \
    if (cl < 2) cl = 2;                                                           \
    curandStatePhilox4_32_10_t state;                                              \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)i, 0ULL, &state); \
    unsigned int j = curand(&state) % cl;                                          \
    WEIGHT_T* out_row = output + (size_t)i * n;                                    \
    if (n <= 32) {                                                                 \
        ACC_T acc[32];                                                             \
        for (int col = 0; col < n; col++) acc[col] = ACC_ZERO;                    \
        while (j < (unsigned int)k) {                                              \
            float u = curand_uniform(&state);                                      \
            ACC_T w = wlo + (ACC_T)u * range;                                     \
            const WEIGHT_T* b_row = B + (size_t)j * n;                            \
            for (int col = 0; col < n; col++) {                                    \
                acc[col] += READ_W(__ldg(&b_row[col])) * w;                       \
            }                                                                      \
            j += 1 + (curand(&state) % (cl - 1));                                 \
        }                                                                          \
        for (int col = 0; col < n; col++) {                                        \
            out_row[col] = WRITE_W(acc[col]);                                      \
        }                                                                          \
    } else {                                                                       \
        for (int col = 0; col < n; col++) {                                        \
            out_row[col] = WRITE_W(ACC_ZERO);                                      \
        }                                                                          \
        while (j < (unsigned int)k) {                                              \
            float u = curand_uniform(&state);                                      \
            ACC_T w = wlo + (ACC_T)u * range;                                     \
            const WEIGHT_T* b_row = B + (size_t)j * n;                            \
            for (int col = 0; col < n; col++) {                                    \
                ACC_T cur = READ_W(out_row[col]);                                  \
                out_row[col] = WRITE_W(cur + w * READ_W(__ldg(&b_row[col])));     \
            }                                                                      \
            j += 1 + (curand(&state) % (cl - 1));                                 \
        }                                                                          \
    }                                                                              \
}

DEFINE_JITUMM_GATHER(_f32,  float,         float,  READ_F32,  WRITE_F32,  0.0f)
DEFINE_JITUMM_GATHER(_f64,  double,        double, READ_F64,  WRITE_F64,  0.0)
DEFINE_JITUMM_GATHER(_f16,  __half,        float,  READ_F16,  WRITE_F16,  0.0f)
DEFINE_JITUMM_GATHER(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, 0.0f)

// =========================================================================
// Scatter kernel (corder=false): one thread per input row j
// For each row j of B, scatter Uniform(w)*B[j,col] to output[connected_i, col].
// =========================================================================

#define DEFINE_JITUMM_SCATTER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ATOMIC_ADD) \
__global__ void _jitumm_scatter_kern##SUFFIX(                                         \
    const WEIGHT_T* __restrict__ w_low,                                               \
    const WEIGHT_T* __restrict__ w_high,                                              \
    const float*    __restrict__ clen,                                                \
    const int*      __restrict__ seed,                                                \
    const WEIGHT_T* __restrict__ B,                                                   \
    WEIGHT_T*       __restrict__ output,                                              \
    int m, int k, int n                                                               \
) {                                                                                   \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                                    \
    if (j >= k) return;                                                               \
    ACC_T wlo = READ_W(__ldg(&w_low[0]));                                              \
    ACC_T range = READ_W(__ldg(&w_high[0])) - wlo;                                     \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                    \
    if (cl < 2) cl = 2;                                                               \
    curandStatePhilox4_32_10_t state;                                                  \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)j, 0ULL, &state); \
    unsigned int i = curand(&state) % cl;                                              \
    const WEIGHT_T* b_row = B + (size_t)j * n;                                        \
    while (i < (unsigned int)m) {                                                      \
        float u = curand_uniform(&state);                                              \
        ACC_T w = wlo + (ACC_T)u * range;                                             \
        WEIGHT_T* out_row = output + (size_t)i * n;                                    \
        for (int col = 0; col < n; col++) {                                            \
            ACC_T val = w * READ_W(__ldg(&b_row[col]));                                \
            ATOMIC_ADD(&out_row[col], val);                                            \
        }                                                                              \
        i += 1 + (curand(&state) % (cl - 1));                                         \
    }                                                                                  \
}

DEFINE_JITUMM_SCATTER(_f32,  float,         float,  READ_F32,  WRITE_F32,  atomicAdd_f32)
DEFINE_JITUMM_SCATTER(_f64,  double,        double, READ_F64,  WRITE_F64,  atomicAdd_f64)
DEFINE_JITUMM_SCATTER(_f16,  __half,        float,  READ_F16,  WRITE_F16,  atomicAdd_f16)
DEFINE_JITUMM_SCATTER(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, atomicAdd_bf16)

// ---- TVM FFI: jitumm gather ----
// No memset needed: gather kernel zero-initializes output rows in-kernel.

#define FFI_JITUMM_GATHER(SUFFIX, WEIGHT_C_T)                                \
void jitumm_gather##SUFFIX(                                                   \
    tvm::ffi::TensorView w_low,                                               \
    tvm::ffi::TensorView w_high,                                              \
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
    int threads = 256;                                                        \
    int blocks = (m + threads - 1) / threads;                                 \
    _jitumm_gather_kern##SUFFIX<<<blocks, threads, 0, s>>>(                   \
        static_cast<const WEIGHT_C_T*>(w_low.data_ptr()),                     \
        static_cast<const WEIGHT_C_T*>(w_high.data_ptr()),                    \
        static_cast<const float*>(clen.data_ptr()),                           \
        static_cast<const int*>(seed.data_ptr()),                             \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                         \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                          \
        m, k, n                                                               \
    );                                                                        \
}

// @tvm_ffi jitumm_gather_f32
FFI_JITUMM_GATHER(_f32, float)
// @tvm_ffi jitumm_gather_f64
FFI_JITUMM_GATHER(_f64, double)
// @tvm_ffi jitumm_gather_f16
FFI_JITUMM_GATHER(_f16, __half)
// @tvm_ffi jitumm_gather_bf16
FFI_JITUMM_GATHER(_bf16, __nv_bfloat16)

// ---- TVM FFI: jitumm scatter ----

#define FFI_JITUMM_SCATTER(SUFFIX, WEIGHT_C_T)                                \
void jitumm_scatter##SUFFIX(                                                   \
    tvm::ffi::TensorView w_low,                                                \
    tvm::ffi::TensorView w_high,                                               \
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
    _jitumm_scatter_kern##SUFFIX<<<blocks, threads, 0, s>>>(                   \
        static_cast<const WEIGHT_C_T*>(w_low.data_ptr()),                      \
        static_cast<const WEIGHT_C_T*>(w_high.data_ptr()),                     \
        static_cast<const float*>(clen.data_ptr()),                            \
        static_cast<const int*>(seed.data_ptr()),                              \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                          \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                           \
        m, k, n                                                                \
    );                                                                         \
}

// @tvm_ffi jitumm_scatter_f32
FFI_JITUMM_SCATTER(_f32, float)
// @tvm_ffi jitumm_scatter_f64
FFI_JITUMM_SCATTER(_f64, double)
// @tvm_ffi jitumm_scatter_f16
FFI_JITUMM_SCATTER(_f16, __half)
// @tvm_ffi jitumm_scatter_bf16
FFI_JITUMM_SCATTER(_bf16, __nv_bfloat16)


// #########################################################################
// ##  4. binary_jitumv — Event-Driven Matrix-Vector Product              ##
// #########################################################################

// =========================================================================
// Gather kernel (corder=true): one thread per output element
// y[i] = sum_{j in C(i) : spike[j] active} Uniform(w_low, w_high)
// The weight is still sampled from the RNG even if the spike is inactive
// (to preserve the correct RNG stream for subsequent connections).
// No memset needed: every output[i] is written exactly once (direct store).
// =========================================================================

#define DEFINE_BINARY_JITUMV_GATHER(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, SPIKE_T, IS_ACTIVE, ACC_ZERO) \
__global__ void _binary_jitumv_gather_kern##SUFFIX(                             \
    const WEIGHT_T* __restrict__ w_low,                                         \
    const WEIGHT_T* __restrict__ w_high,                                        \
    const float*    __restrict__ clen,                                          \
    const int*      __restrict__ seed,                                          \
    const SPIKE_T*  __restrict__ vector,                                        \
    WEIGHT_T*       __restrict__ output,                                        \
    int m, int k                                                                \
) {                                                                             \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                              \
    if (i >= m) return;                                                         \
    ACC_T wlo = READ_W(__ldg(&w_low[0]));                                        \
    ACC_T range = READ_W(__ldg(&w_high[0])) - wlo;                               \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                              \
    if (cl < 2) cl = 2;                                                         \
    curandStatePhilox4_32_10_t state;                                            \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)i, 0ULL, &state); \
    unsigned int j = curand(&state) % cl;                                        \
    ACC_T acc = ACC_ZERO;                                                        \
    while (j < (unsigned int)k) {                                                \
        float u = curand_uniform(&state);                                        \
        if (IS_ACTIVE(vector, j)) {                                              \
            ACC_T w = wlo + (ACC_T)u * range;                                   \
            acc += w;                                                            \
        }                                                                        \
        j += 1 + (curand(&state) % (cl - 1));                                   \
    }                                                                            \
    output[i] = WRITE_W(acc);                                                    \
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
__global__ void _binary_jitumv_gather_smem_kern##SUFFIX(                        \
    const WEIGHT_T* __restrict__ w_low,                                         \
    const WEIGHT_T* __restrict__ w_high,                                        \
    const float*    __restrict__ clen,                                          \
    const int*      __restrict__ seed,                                          \
    const SPIKE_T*  __restrict__ vector,                                        \
    WEIGHT_T*       __restrict__ output,                                        \
    int m, int k                                                                \
) {                                                                             \
    extern __shared__ char _smem_bytes[];                                        \
    SPIKE_T* sv = reinterpret_cast<SPIKE_T*>(_smem_bytes);                      \
    for (int idx = threadIdx.x; idx < k; idx += blockDim.x) {                   \
        sv[idx] = __ldg(&vector[idx]);                                           \
    }                                                                           \
    __syncthreads();                                                            \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                              \
    if (i >= m) return;                                                         \
    ACC_T wlo = READ_W(__ldg(&w_low[0]));                                        \
    ACC_T range = READ_W(__ldg(&w_high[0])) - wlo;                               \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                              \
    if (cl < 2) cl = 2;                                                         \
    curandStatePhilox4_32_10_t state;                                            \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)i, 0ULL, &state); \
    unsigned int j = curand(&state) % cl;                                        \
    ACC_T acc = ACC_ZERO;                                                        \
    while (j < (unsigned int)k) {                                                \
        float u = curand_uniform(&state);                                        \
        if (IS_ACTIVE(sv, j)) {                                                 \
            ACC_T w = wlo + (ACC_T)u * range;                                   \
            acc += w;                                                            \
        }                                                                        \
        j += 1 + (curand(&state) % (cl - 1));                                   \
    }                                                                            \
    output[i] = WRITE_W(acc);                                                    \
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
__global__ void _binary_jitumv_scatter_kern##SUFFIX(                            \
    const WEIGHT_T* __restrict__ w_low,                                         \
    const WEIGHT_T* __restrict__ w_high,                                        \
    const float*    __restrict__ clen,                                          \
    const int*      __restrict__ seed,                                          \
    const SPIKE_T*  __restrict__ vector,                                        \
    WEIGHT_T*       __restrict__ output,                                        \
    int m, int k                                                                \
) {                                                                             \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                              \
    if (j >= k) return;                                                         \
    if (!IS_ACTIVE(vector, j)) return;                                          \
    ACC_T wlo = READ_W(__ldg(&w_low[0]));                                        \
    ACC_T range = READ_W(__ldg(&w_high[0])) - wlo;                               \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                              \
    if (cl < 2) cl = 2;                                                         \
    curandStatePhilox4_32_10_t state;                                            \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)j, 0ULL, &state); \
    unsigned int i = curand(&state) % cl;                                        \
    while (i < (unsigned int)m) {                                                \
        float u = curand_uniform(&state);                                        \
        ACC_T w = wlo + (ACC_T)u * range;                                       \
        ATOMIC_ADD(&output[i], w);                                               \
        i += 1 + (curand(&state) % (cl - 1));                                   \
    }                                                                            \
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

#define FFI_BINARY_JITUMV_GATHER(SUFFIX, WEIGHT_C_T, SPIKE_C_T)               \
void binary_jitumv_gather##SUFFIX(                                              \
    tvm::ffi::TensorView w_low,                                                 \
    tvm::ffi::TensorView w_high,                                                \
    tvm::ffi::TensorView clen,                                                  \
    tvm::ffi::TensorView seed,                                                  \
    tvm::ffi::TensorView vector,                                                \
    tvm::ffi::TensorView output,                                                \
    int64_t stream                                                              \
) {                                                                             \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                    \
    int m = static_cast<int>(output.size(0));                                   \
    int k = static_cast<int>(vector.size(0));                                   \
    int threads = 256;                                                          \
    int blocks = (m + threads - 1) / threads;                                   \
    size_t smem_bytes = (size_t)k * sizeof(SPIKE_C_T);                          \
    if (smem_bytes <= SMEM_THRESHOLD) {                                         \
        _binary_jitumv_gather_smem_kern##SUFFIX<<<blocks, threads, smem_bytes, s>>>( \
            static_cast<const WEIGHT_C_T*>(w_low.data_ptr()),                   \
            static_cast<const WEIGHT_C_T*>(w_high.data_ptr()),                  \
            static_cast<const float*>(clen.data_ptr()),                         \
            static_cast<const int*>(seed.data_ptr()),                           \
            static_cast<const SPIKE_C_T*>(vector.data_ptr()),                   \
            static_cast<WEIGHT_C_T*>(output.data_ptr()),                        \
            m, k                                                                \
        );                                                                      \
    } else {                                                                    \
        _binary_jitumv_gather_kern##SUFFIX<<<blocks, threads, 0, s>>>(          \
            static_cast<const WEIGHT_C_T*>(w_low.data_ptr()),                   \
            static_cast<const WEIGHT_C_T*>(w_high.data_ptr()),                  \
            static_cast<const float*>(clen.data_ptr()),                         \
            static_cast<const int*>(seed.data_ptr()),                           \
            static_cast<const SPIKE_C_T*>(vector.data_ptr()),                   \
            static_cast<WEIGHT_C_T*>(output.data_ptr()),                        \
            m, k                                                                \
        );                                                                      \
    }                                                                           \
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

#define FFI_BINARY_JITUMV_SCATTER(SUFFIX, WEIGHT_C_T, SPIKE_C_T)             \
void binary_jitumv_scatter##SUFFIX(                                           \
    tvm::ffi::TensorView w_low,                                               \
    tvm::ffi::TensorView w_high,                                              \
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
    _binary_jitumv_scatter_kern##SUFFIX<<<blocks, threads, 0, s>>>(           \
        static_cast<const WEIGHT_C_T*>(w_low.data_ptr()),                     \
        static_cast<const WEIGHT_C_T*>(w_high.data_ptr()),                    \
        static_cast<const float*>(clen.data_ptr()),                           \
        static_cast<const int*>(seed.data_ptr()),                             \
        static_cast<const SPIKE_C_T*>(vector.data_ptr()),                     \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                          \
        m, k                                                                  \
    );                                                                        \
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


// #########################################################################
// ##  5. binary_jitumm — Event-Driven Matrix-Matrix Product              ##
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
__global__ void _binary_jitumm_gather_kern##SUFFIX(                             \
    const WEIGHT_T* __restrict__ w_low,                                         \
    const WEIGHT_T* __restrict__ w_high,                                        \
    const float*    __restrict__ clen,                                          \
    const int*      __restrict__ seed,                                          \
    const SPIKE_T*  __restrict__ B,                                             \
    WEIGHT_T*       __restrict__ output,                                        \
    int m, int k, int n                                                         \
) {                                                                             \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                              \
    if (i >= m) return;                                                         \
    ACC_T wlo = READ_W(__ldg(&w_low[0]));                                        \
    ACC_T range = READ_W(__ldg(&w_high[0])) - wlo;                               \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                              \
    if (cl < 2) cl = 2;                                                         \
    curandStatePhilox4_32_10_t state;                                            \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)i, 0ULL, &state); \
    unsigned int j = curand(&state) % cl;                                        \
    WEIGHT_T* out_row = output + (size_t)i * n;                                  \
    if (n <= 32) {                                                               \
        ACC_T acc[32];                                                           \
        for (int col = 0; col < n; col++) acc[col] = ACC_ZERO;                  \
        while (j < (unsigned int)k) {                                            \
            float u = curand_uniform(&state);                                    \
            ACC_T w = wlo + (ACC_T)u * range;                                   \
            const SPIKE_T* b_row = B + (size_t)j * n;                           \
            for (int col = 0; col < n; col++) {                                  \
                if (IS_ACTIVE(b_row, col)) {                                     \
                    acc[col] += w;                                               \
                }                                                                \
            }                                                                    \
            j += 1 + (curand(&state) % (cl - 1));                               \
        }                                                                        \
        for (int col = 0; col < n; col++) {                                      \
            out_row[col] = WRITE_W(acc[col]);                                    \
        }                                                                        \
    } else {                                                                     \
        for (int col = 0; col < n; col++) {                                      \
            out_row[col] = WRITE_W(ACC_ZERO);                                    \
        }                                                                        \
        while (j < (unsigned int)k) {                                            \
            float u = curand_uniform(&state);                                    \
            ACC_T w = wlo + (ACC_T)u * range;                                   \
            const SPIKE_T* b_row = B + (size_t)j * n;                           \
            for (int col = 0; col < n; col++) {                                  \
                if (IS_ACTIVE(b_row, col)) {                                     \
                    ACC_T cur = READ_W(out_row[col]);                            \
                    out_row[col] = WRITE_W(cur + w);                             \
                }                                                                \
            }                                                                    \
            j += 1 + (curand(&state) % (cl - 1));                               \
        }                                                                        \
    }                                                                            \
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
__global__ void _binary_jitumm_scatter_kern##SUFFIX(                            \
    const WEIGHT_T* __restrict__ w_low,                                         \
    const WEIGHT_T* __restrict__ w_high,                                        \
    const float*    __restrict__ clen,                                          \
    const int*      __restrict__ seed,                                          \
    const SPIKE_T*  __restrict__ B,                                             \
    WEIGHT_T*       __restrict__ output,                                        \
    int m, int k, int n                                                         \
) {                                                                             \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                              \
    if (j >= k) return;                                                         \
    ACC_T wlo = READ_W(__ldg(&w_low[0]));                                        \
    ACC_T range = READ_W(__ldg(&w_high[0])) - wlo;                               \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                              \
    if (cl < 2) cl = 2;                                                         \
    const SPIKE_T* b_row = B + (size_t)j * n;                                   \
    curandStatePhilox4_32_10_t state;                                            \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)j, 0ULL, &state); \
    unsigned int i = curand(&state) % cl;                                        \
    while (i < (unsigned int)m) {                                                \
        float u = curand_uniform(&state);                                        \
        ACC_T w = wlo + (ACC_T)u * range;                                       \
        WEIGHT_T* out_row = output + (size_t)i * n;                              \
        for (int col = 0; col < n; col++) {                                      \
            if (IS_ACTIVE(b_row, col)) {                                         \
                ATOMIC_ADD(&out_row[col], w);                                    \
            }                                                                    \
        }                                                                        \
        i += 1 + (curand(&state) % (cl - 1));                                   \
    }                                                                            \
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

#define FFI_BINARY_JITUMM_GATHER(SUFFIX, WEIGHT_C_T, SPIKE_C_T)              \
void binary_jitumm_gather##SUFFIX(                                            \
    tvm::ffi::TensorView w_low,                                               \
    tvm::ffi::TensorView w_high,                                              \
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
    int threads = 256;                                                        \
    int blocks = (m + threads - 1) / threads;                                 \
    _binary_jitumm_gather_kern##SUFFIX<<<blocks, threads, 0, s>>>(            \
        static_cast<const WEIGHT_C_T*>(w_low.data_ptr()),                     \
        static_cast<const WEIGHT_C_T*>(w_high.data_ptr()),                    \
        static_cast<const float*>(clen.data_ptr()),                           \
        static_cast<const int*>(seed.data_ptr()),                             \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                          \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                          \
        m, k, n                                                               \
    );                                                                        \
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

#define FFI_BINARY_JITUMM_SCATTER(SUFFIX, WEIGHT_C_T, SPIKE_C_T)             \
void binary_jitumm_scatter##SUFFIX(                                           \
    tvm::ffi::TensorView w_low,                                               \
    tvm::ffi::TensorView w_high,                                              \
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
    _binary_jitumm_scatter_kern##SUFFIX<<<blocks, threads, 0, s>>>(           \
        static_cast<const WEIGHT_C_T*>(w_low.data_ptr()),                     \
        static_cast<const WEIGHT_C_T*>(w_high.data_ptr()),                    \
        static_cast<const float*>(clen.data_ptr()),                           \
        static_cast<const int*>(seed.data_ptr()),                             \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                          \
        static_cast<WEIGHT_C_T*>(output.data_ptr()),                          \
        m, k, n                                                               \
    );                                                                        \
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
