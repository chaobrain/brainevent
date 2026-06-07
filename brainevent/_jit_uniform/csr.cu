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
 * csr.cu — Direct JIT-connectivity → CSR materialization, uniform weights (CUDA)
 * =============================================================================
 *
 * Generates a Compressed Sparse Row (CSR) representation of a just-in-time
 * connectivity (JITC) matrix whose non-zero weights are drawn from a uniform
 * distribution ``U(low, high)``, *without* ever allocating the dense matrix.
 *
 * The connectivity walk reproduces the dense ``jitu`` kernels (see
 * ``float_jitu.cu``) bit-for-bit: same ``curand_init`` seeding, same geometric
 * skip ``col += 1 + curand % (clen - 1)``, and the same per-connection
 * ``curand_uniform`` weight draw.  Because the two CSR passes share that walk,
 * the CSR they emit reproduces exactly the matrix returned by ``.todense()`` on
 * the CUDA backend.
 *
 * Two passes (``nnz`` is data dependent, XLA needs static shapes):
 *
 *   1. count : per-row non-zero counts  -> row_counts  (int32, shape (n_rows,))
 *   2. fill  : column indices + values  -> indices (int32), data (WEIGHT_T)
 *
 * The count pass *replicates and discards* the per-connection ``curand_uniform``
 * draw so its pseudo-random stream stays aligned with the fill pass; otherwise
 * the two passes would disagree on connection positions.
 *
 * Layout
 * ------
 * corder=true  (gather):  one thread per row, walks columns.  Each row writes
 *              its CSR slice ``[indptr[r], indptr[r+1])`` sequentially in
 *              increasing column order -> rows are column-sorted.
 * corder=false (scatter): one thread per column, walks rows.  A per-row write
 *              cursor (``wptr``, a device copy of ``indptr[:n_rows]``) is
 *              advanced with ``atomicAdd``; column order within a row is not
 *              guaranteed (each (row, col) is still written exactly once).
 *
 * Parameters
 * ----------
 * w0, w1 : shape (1,), weight parameters (low, high).  Ignored by the count kernels.
 * clen   : shape (1,), connection length = ceil(2/prob) (read as float, matching
 *          the dense kernels).
 * seed   : shape (1,), int32 random seed.
 * indptr : shape (n_rows + 1,), int32 CSR row pointers (fill pass input).
 * n_cols : scalar attribute, number of matrix columns.
 *
 * Supported weight dtypes: float32, float64, float16, bfloat16.
 *
 * IMPORTANT: All data_ptr() returns are GPU device pointers — NEVER dereference on host.
 */

#include "cuda_common.h"
#include "brainevent/common.h"
#include "curand_common.h"


// #########################################################################
// ##  Count pass — per-row non-zero counts                               ##
// #########################################################################

// ---- count, corder=true: one thread per row (with discarded weight draw) ----
#define DEFINE_COUNT_CT_DRAW(NAME, DRAW_FN)                                                  \
__global__ void NAME(                                                                        \
    const float* __restrict__ clen,                                                          \
    const int*   __restrict__ seed,                                                          \
    int*         __restrict__ row_counts,                                                    \
    int n_rows, int n_cols                                                                   \
) {                                                                                          \
    int row = blockIdx.x * blockDim.x + threadIdx.x;                                         \
    if (row >= n_rows) return;                                                               \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                         \
    if (cl < 2) cl = 2;                                                                      \
    curandStatePhilox4_32_10_t state;                                                        \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)row, 0ULL, &state); \
    unsigned int col = curand(&state) % cl;                                                  \
    int cnt = 0;                                                                             \
    while (col < (unsigned int)n_cols) {                                                     \
        (void)DRAW_FN(&state);                                                               \
        cnt += 1;                                                                            \
        col += 1 + (curand(&state) % (cl - 1));                                              \
    }                                                                                        \
    row_counts[row] = cnt;                                                                   \
}

// ---- count, corder=false: one thread per column (with discarded weight draw) ----
#define DEFINE_COUNT_CF_DRAW(NAME, DRAW_FN)                                                  \
__global__ void NAME(                                                                        \
    const float* __restrict__ clen,                                                          \
    const int*   __restrict__ seed,                                                          \
    int*         __restrict__ row_counts,                                                    \
    int n_rows, int n_cols                                                                   \
) {                                                                                          \
    int col = blockIdx.x * blockDim.x + threadIdx.x;                                         \
    if (col >= n_cols) return;                                                               \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                         \
    if (cl < 2) cl = 2;                                                                      \
    curandStatePhilox4_32_10_t state;                                                        \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)col, 0ULL, &state); \
    unsigned int row = curand(&state) % cl;                                                  \
    while (row < (unsigned int)n_rows) {                                                     \
        (void)DRAW_FN(&state);                                                               \
        atomicAdd(&row_counts[row], 1);                                                      \
        row += 1 + (curand(&state) % (cl - 1));                                              \
    }                                                                                        \
}

// uniform: dense draws a single float32 ``curand_uniform`` per connection.
DEFINE_COUNT_CT_DRAW(_count_ct_uniform, curand_uniform)
DEFINE_COUNT_CF_DRAW(_count_cf_uniform, curand_uniform)


// #########################################################################
// ##  Fill pass — column indices and values                              ##
// #########################################################################

// ---- fill, uniform, corder=true ----
#define DEFINE_FILL_U_CT(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W)                           \
__global__ void _fill_u_ct##SUFFIX(                                                          \
    const WEIGHT_T* __restrict__ w0,                                                         \
    const WEIGHT_T* __restrict__ w1,                                                         \
    const float*    __restrict__ clen,                                                       \
    const int*      __restrict__ seed,                                                       \
    const int*      __restrict__ indptr,                                                     \
    int*            __restrict__ indices,                                                    \
    WEIGHT_T*       __restrict__ data,                                                       \
    int n_rows, int n_cols                                                                   \
) {                                                                                          \
    int row = blockIdx.x * blockDim.x + threadIdx.x;                                         \
    if (row >= n_rows) return;                                                               \
    ACC_T wlo = READ_W(__ldg(&w0[0]));                                                       \
    ACC_T range = READ_W(__ldg(&w1[0])) - wlo;                                               \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                         \
    if (cl < 2) cl = 2;                                                                      \
    curandStatePhilox4_32_10_t state;                                                        \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)row, 0ULL, &state); \
    unsigned int col = curand(&state) % cl;                                                  \
    int pos = indptr[row];                                                                   \
    while (col < (unsigned int)n_cols) {                                                     \
        float uu = curand_uniform(&state);                                                   \
        indices[pos] = (int)col;                                                             \
        data[pos] = WRITE_W(wlo + (ACC_T)uu * range);                                        \
        pos += 1;                                                                            \
        col += 1 + (curand(&state) % (cl - 1));                                              \
    }                                                                                        \
}

// ---- fill, uniform, corder=false ----
#define DEFINE_FILL_U_CF(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W)                           \
__global__ void _fill_u_cf##SUFFIX(                                                          \
    const WEIGHT_T* __restrict__ w0,                                                         \
    const WEIGHT_T* __restrict__ w1,                                                         \
    const float*    __restrict__ clen,                                                       \
    const int*      __restrict__ seed,                                                       \
    int*            __restrict__ wptr,                                                       \
    int*            __restrict__ indices,                                                    \
    WEIGHT_T*       __restrict__ data,                                                       \
    int n_rows, int n_cols                                                                   \
) {                                                                                          \
    int col = blockIdx.x * blockDim.x + threadIdx.x;                                         \
    if (col >= n_cols) return;                                                               \
    ACC_T wlo = READ_W(__ldg(&w0[0]));                                                       \
    ACC_T range = READ_W(__ldg(&w1[0])) - wlo;                                               \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                         \
    if (cl < 2) cl = 2;                                                                      \
    curandStatePhilox4_32_10_t state;                                                        \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)col, 0ULL, &state); \
    unsigned int row = curand(&state) % cl;                                                  \
    while (row < (unsigned int)n_rows) {                                                     \
        float uu = curand_uniform(&state);                                                   \
        int pos = atomicAdd(&wptr[row], 1);                                                  \
        indices[pos] = (int)col;                                                             \
        data[pos] = WRITE_W(wlo + (ACC_T)uu * range);                                        \
        row += 1 + (curand(&state) % (cl - 1));                                              \
    }                                                                                        \
}

// uniform fill globals
DEFINE_FILL_U_CT(_f32,  float,         float,  READ_F32,  WRITE_F32)
DEFINE_FILL_U_CT(_f64,  double,        double, READ_F64,  WRITE_F64)
DEFINE_FILL_U_CT(_f16,  __half,        float,  READ_F16,  WRITE_F16)
DEFINE_FILL_U_CT(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16)
DEFINE_FILL_U_CF(_f32,  float,         float,  READ_F32,  WRITE_F32)
DEFINE_FILL_U_CF(_f64,  double,        double, READ_F64,  WRITE_F64)
DEFINE_FILL_U_CF(_f16,  __half,        float,  READ_F16,  WRITE_F16)
DEFINE_FILL_U_CF(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16)


// #########################################################################
// ##  FFI entry points                                                   ##
// #########################################################################

// ---- FFI: count, corder=true (launch over rows; each row writes its count) ----
#define FFI_COUNT_CT(FNAME, GLOBAL)                                                          \
void FNAME(                                                                                  \
    const BE::Tensor w0, const BE::Tensor w1,                                                \
    const BE::Tensor clen, const BE::Tensor seed,                                            \
    BE::Tensor row_counts, int n_cols, int64_t stream                                        \
) {                                                                                          \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                 \
    (void)w0; (void)w1;                                                                      \
    int n_rows = static_cast<int>(row_counts.size(0));                                       \
    int threads = 256;                                                                       \
    int blocks = (n_rows + threads - 1) / threads;                                           \
    GLOBAL<<<blocks, threads, 0, s>>>(                                                       \
        static_cast<const float*>(clen.data_ptr()),                                          \
        static_cast<const int*>(seed.data_ptr()),                                            \
        static_cast<int*>(row_counts.data_ptr()),                                            \
        n_rows, n_cols                                                                       \
    );                                                                                       \
}

// ---- FFI: count, corder=false (launch over cols; atomic scatter into counts) ----
#define FFI_COUNT_CF(FNAME, GLOBAL)                                                          \
void FNAME(                                                                                  \
    const BE::Tensor w0, const BE::Tensor w1,                                                \
    const BE::Tensor clen, const BE::Tensor seed,                                            \
    BE::Tensor row_counts, int n_cols, int64_t stream                                        \
) {                                                                                          \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                 \
    (void)w0; (void)w1;                                                                      \
    int n_rows = static_cast<int>(row_counts.size(0));                                       \
    cudaMemsetAsync(row_counts.data_ptr(), 0, (size_t)n_rows * sizeof(int), s);              \
    int threads = 256;                                                                       \
    int blocks = (n_cols + threads - 1) / threads;                                           \
    GLOBAL<<<blocks, threads, 0, s>>>(                                                       \
        static_cast<const float*>(clen.data_ptr()),                                          \
        static_cast<const int*>(seed.data_ptr()),                                            \
        static_cast<int*>(row_counts.data_ptr()),                                            \
        n_rows, n_cols                                                                       \
    );                                                                                       \
}

// ---- FFI: fill, corder=true (launch over rows; sequential, column-sorted) ----
#define FFI_FILL_CT(FNAME, GLOBAL, WEIGHT_C_T)                                               \
void FNAME(                                                                                  \
    const BE::Tensor w0, const BE::Tensor w1,                                                \
    const BE::Tensor clen, const BE::Tensor seed, const BE::Tensor indptr,                   \
    BE::Tensor indices, BE::Tensor data, int n_cols, int64_t stream                          \
) {                                                                                          \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                 \
    int n_rows = static_cast<int>(indptr.size(0)) - 1;                                       \
    int threads = 256;                                                                       \
    int blocks = (n_rows + threads - 1) / threads;                                           \
    GLOBAL<<<blocks, threads, 0, s>>>(                                                       \
        static_cast<const WEIGHT_C_T*>(w0.data_ptr()),                                       \
        static_cast<const WEIGHT_C_T*>(w1.data_ptr()),                                       \
        static_cast<const float*>(clen.data_ptr()),                                          \
        static_cast<const int*>(seed.data_ptr()),                                            \
        static_cast<const int*>(indptr.data_ptr()),                                          \
        static_cast<int*>(indices.data_ptr()),                                               \
        static_cast<WEIGHT_C_T*>(data.data_ptr()),                                           \
        n_rows, n_cols                                                                       \
    );                                                                                       \
}

// ---- FFI: fill, corder=false (launch over cols; atomic write-cursor per row) ----
#define FFI_FILL_CF(FNAME, GLOBAL, WEIGHT_C_T)                                               \
void FNAME(                                                                                  \
    const BE::Tensor w0, const BE::Tensor w1,                                                \
    const BE::Tensor clen, const BE::Tensor seed, const BE::Tensor indptr,                   \
    BE::Tensor indices, BE::Tensor data, int n_cols, int64_t stream                          \
) {                                                                                          \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                 \
    int n_rows = static_cast<int>(indptr.size(0)) - 1;                                       \
    int* wptr = nullptr;                                                                     \
    cudaMalloc((void**)&wptr, (size_t)n_rows * sizeof(int));                                 \
    cudaMemcpyAsync(wptr, indptr.data_ptr(), (size_t)n_rows * sizeof(int),                   \
                    cudaMemcpyDeviceToDevice, s);                                            \
    int threads = 256;                                                                       \
    int blocks = (n_cols + threads - 1) / threads;                                           \
    GLOBAL<<<blocks, threads, 0, s>>>(                                                       \
        static_cast<const WEIGHT_C_T*>(w0.data_ptr()),                                       \
        static_cast<const WEIGHT_C_T*>(w1.data_ptr()),                                       \
        static_cast<const float*>(clen.data_ptr()),                                          \
        static_cast<const int*>(seed.data_ptr()),                                            \
        wptr,                                                                                \
        static_cast<int*>(indices.data_ptr()),                                               \
        static_cast<WEIGHT_C_T*>(data.data_ptr()),                                           \
        n_rows, n_cols                                                                       \
    );                                                                                       \
    cudaFreeAsync(wptr, s);                                                                  \
}

// ====================== uniform — count ======================
// @BE count_corder_true_f32
FFI_COUNT_CT(count_corder_true_f32,  _count_ct_uniform)
// @BE count_corder_true_f64
FFI_COUNT_CT(count_corder_true_f64,  _count_ct_uniform)
// @BE count_corder_true_f16
FFI_COUNT_CT(count_corder_true_f16,  _count_ct_uniform)
// @BE count_corder_true_bf16
FFI_COUNT_CT(count_corder_true_bf16, _count_ct_uniform)
// @BE count_corder_false_f32
FFI_COUNT_CF(count_corder_false_f32,  _count_cf_uniform)
// @BE count_corder_false_f64
FFI_COUNT_CF(count_corder_false_f64,  _count_cf_uniform)
// @BE count_corder_false_f16
FFI_COUNT_CF(count_corder_false_f16,  _count_cf_uniform)
// @BE count_corder_false_bf16
FFI_COUNT_CF(count_corder_false_bf16, _count_cf_uniform)

// ====================== uniform — fill ======================
// @BE fill_corder_true_f32
FFI_FILL_CT(fill_corder_true_f32,  _fill_u_ct_f32,  float)
// @BE fill_corder_true_f64
FFI_FILL_CT(fill_corder_true_f64,  _fill_u_ct_f64,  double)
// @BE fill_corder_true_f16
FFI_FILL_CT(fill_corder_true_f16,  _fill_u_ct_f16,  __half)
// @BE fill_corder_true_bf16
FFI_FILL_CT(fill_corder_true_bf16, _fill_u_ct_bf16, __nv_bfloat16)
// @BE fill_corder_false_f32
FFI_FILL_CF(fill_corder_false_f32,  _fill_u_cf_f32,  float)
// @BE fill_corder_false_f64
FFI_FILL_CF(fill_corder_false_f64,  _fill_u_cf_f64,  double)
// @BE fill_corder_false_f16
FFI_FILL_CF(fill_corder_false_f16,  _fill_u_cf_f16,  __half)
// @BE fill_corder_false_bf16
FFI_FILL_CF(fill_corder_false_bf16, _fill_u_cf_bf16, __nv_bfloat16)
