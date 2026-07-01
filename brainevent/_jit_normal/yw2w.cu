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
 * yw2w.cu — Direct JIT-normal y*w materialization (CUDA)
 * ======================================================
 *
 * Fills one flat value per structural non-zero of a normal just-in-time
 * connectivity matrix. The connectivity and normal-weight streams match
 * csr.cu; this kernel writes sampled_weight * y[row] for non-transpose mode
 * and sampled_weight * y[col] for transpose mode directly into CSR flat data
 * order.
 */

#include "cuda_common.h"
#include "brainevent/common.h"
#include "curand_common.h"


// #########################################################################
// ##  Fill pass — per-synapse y * w values                               ##
// #########################################################################

// ---- fill, normal, corder=true ----
#define DEFINE_YW2W_N_CT(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, RNG_FUNC, TRANSPOSE)      \
__global__ void _yw2w_n_ct##SUFFIX(                                                          \
    const WEIGHT_T* __restrict__ w0,                                                         \
    const WEIGHT_T* __restrict__ w1,                                                         \
    const float*    __restrict__ clen,                                                       \
    const WEIGHT_T* __restrict__ y,                                                          \
    const int*      __restrict__ seed,                                                       \
    const int*      __restrict__ indptr,                                                     \
    WEIGHT_T*       __restrict__ data,                                                       \
    int n_rows, int n_cols                                                                   \
) {                                                                                          \
    int row = blockIdx.x * blockDim.x + threadIdx.x;                                         \
    if (row >= n_rows) return;                                                               \
    ACC_T loc = READ_W(__ldg(&w0[0]));                                                       \
    ACC_T scale = READ_W(__ldg(&w1[0]));                                                     \
    ACC_T y_row = TRANSPOSE ? (ACC_T)0 : READ_W(__ldg(&y[row]));                             \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                         \
    if (cl < 2) cl = 2;                                                                      \
    curandStatePhilox4_32_10_t state;                                                        \
    curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)row, 0ULL, &state); \
    unsigned int col = curand(&state) % cl;                                                  \
    int pos = indptr[row];                                                                   \
    while (col < (unsigned int)n_cols) {                                                     \
        ACC_T y_value = TRANSPOSE ? READ_W(__ldg(&y[col])) : y_row;                          \
        ACC_T nn = (ACC_T)RNG_FUNC(&state);                                                  \
        data[pos] = WRITE_W((loc + nn * scale) * y_value);                                   \
        pos += 1;                                                                            \
        col += 1 + (curand(&state) % (cl - 1));                                              \
    }                                                                                        \
}

// ---- fill, normal, corder=false ----
#define DEFINE_YW2W_N_CF(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, RNG_FUNC, TRANSPOSE)      \
__global__ void _yw2w_n_cf##SUFFIX(                                                          \
    const WEIGHT_T* __restrict__ w0,                                                         \
    const WEIGHT_T* __restrict__ w1,                                                         \
    const float*    __restrict__ clen,                                                       \
    const WEIGHT_T* __restrict__ y,                                                          \
    const int*      __restrict__ seed,                                                       \
    const int*      __restrict__ indptr,                                                     \
    WEIGHT_T*       __restrict__ data,                                                       \
    int n_rows, int n_cols                                                                   \
) {                                                                                          \
    int row = blockIdx.x * blockDim.x + threadIdx.x;                                         \
    if (row >= n_rows) return;                                                               \
    ACC_T loc = READ_W(__ldg(&w0[0]));                                                       \
    ACC_T scale = READ_W(__ldg(&w1[0]));                                                     \
    unsigned int cl = (unsigned int)__ldg(&clen[0]);                                         \
    if (cl < 2) cl = 2;                                                                      \
    int pos = indptr[row];                                                                   \
    ACC_T y_row = TRANSPOSE ? (ACC_T)0 : READ_W(__ldg(&y[row]));                             \
    for (int col = 0; col < n_cols; ++col) {                                                 \
        curandStatePhilox4_32_10_t state;                                                    \
        curand_init((unsigned long long)__ldg(&seed[0]), (unsigned long long)col, 0ULL, &state); \
        unsigned int rr = curand(&state) % cl;                                               \
        while (rr < (unsigned int)row) {                                                     \
            (void)RNG_FUNC(&state);                                                          \
            rr += 1 + (curand(&state) % (cl - 1));                                           \
        }                                                                                    \
        if (rr == (unsigned int)row) {                                                       \
            ACC_T y_value = TRANSPOSE ? READ_W(__ldg(&y[col])) : y_row;                      \
            ACC_T nn = (ACC_T)RNG_FUNC(&state);                                              \
            data[pos] = WRITE_W((loc + nn * scale) * y_value);                               \
            pos += 1;                                                                        \
        }                                                                                    \
    }                                                                                        \
}

DEFINE_YW2W_N_CT(_nt_f32,  float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, false)
DEFINE_YW2W_N_CT(_t_f32,   float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, true)
DEFINE_YW2W_N_CT(_nt_f64,  double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, false)
DEFINE_YW2W_N_CT(_t_f64,   double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, true)
DEFINE_YW2W_N_CT(_nt_f16,  __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, false)
DEFINE_YW2W_N_CT(_t_f16,   __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, true)
DEFINE_YW2W_N_CT(_nt_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, false)
DEFINE_YW2W_N_CT(_t_bf16,  __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, true)

DEFINE_YW2W_N_CF(_nt_f32,  float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, false)
DEFINE_YW2W_N_CF(_t_f32,   float,         float,  READ_F32,  WRITE_F32,  curand_normal_f32, true)
DEFINE_YW2W_N_CF(_nt_f64,  double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, false)
DEFINE_YW2W_N_CF(_t_f64,   double,        double, READ_F64,  WRITE_F64,  curand_normal_f64, true)
DEFINE_YW2W_N_CF(_nt_f16,  __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, false)
DEFINE_YW2W_N_CF(_t_f16,   __half,        float,  READ_F16,  WRITE_F16,  curand_normal_f32, true)
DEFINE_YW2W_N_CF(_nt_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, false)
DEFINE_YW2W_N_CF(_t_bf16,  __nv_bfloat16, float,  READ_BF16, WRITE_BF16, curand_normal_f32, true)


// #########################################################################
// ##  FFI entry points                                                    ##
// #########################################################################

#define FFI_YW2W_FILL_CT(FNAME, GLOBAL, WEIGHT_C_T)                                         \
void FNAME(                                                                                 \
    const BE::Tensor w0, const BE::Tensor w1,                                                \
    const BE::Tensor clen, const BE::Tensor y, const BE::Tensor seed,                        \
    const BE::Tensor indptr, BE::Tensor data, int n_cols, int64_t stream                     \
) {                                                                                         \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                \
    int n_rows = static_cast<int>(indptr.size(0)) - 1;                                      \
    int threads = 256;                                                                      \
    int blocks = (n_rows + threads - 1) / threads;                                          \
    GLOBAL<<<blocks, threads, 0, s>>>(                                                      \
        static_cast<const WEIGHT_C_T*>(w0.data_ptr()),                                      \
        static_cast<const WEIGHT_C_T*>(w1.data_ptr()),                                      \
        static_cast<const float*>(clen.data_ptr()),                                         \
        static_cast<const WEIGHT_C_T*>(y.data_ptr()),                                       \
        static_cast<const int*>(seed.data_ptr()),                                           \
        static_cast<const int*>(indptr.data_ptr()),                                         \
        static_cast<WEIGHT_C_T*>(data.data_ptr()),                                          \
        n_rows, n_cols                                                                      \
    );                                                                                      \
}

#define FFI_YW2W_FILL_CF(FNAME, GLOBAL, WEIGHT_C_T)                                         \
void FNAME(                                                                                 \
    const BE::Tensor w0, const BE::Tensor w1,                                                \
    const BE::Tensor clen, const BE::Tensor y, const BE::Tensor seed,                        \
    const BE::Tensor indptr, BE::Tensor data, int n_cols, int64_t stream                     \
) {                                                                                         \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                \
    int n_rows = static_cast<int>(indptr.size(0)) - 1;                                      \
    int threads = 256;                                                                      \
    int blocks = (n_rows + threads - 1) / threads;                                          \
    GLOBAL<<<blocks, threads, 0, s>>>(                                                      \
        static_cast<const WEIGHT_C_T*>(w0.data_ptr()),                                      \
        static_cast<const WEIGHT_C_T*>(w1.data_ptr()),                                      \
        static_cast<const float*>(clen.data_ptr()),                                         \
        static_cast<const WEIGHT_C_T*>(y.data_ptr()),                                       \
        static_cast<const int*>(seed.data_ptr()),                                           \
        static_cast<const int*>(indptr.data_ptr()),                                         \
        static_cast<WEIGHT_C_T*>(data.data_ptr()),                                          \
        n_rows, n_cols                                                                      \
    );                                                                                      \
}

// ====================== normal — fill, corder=true ======================
// @BE fill_corder_true_nt_f32
FFI_YW2W_FILL_CT(fill_corder_true_nt_f32,  _yw2w_n_ct_nt_f32,  float)
// @BE fill_corder_true_t_f32
FFI_YW2W_FILL_CT(fill_corder_true_t_f32,   _yw2w_n_ct_t_f32,   float)
// @BE fill_corder_true_nt_f64
FFI_YW2W_FILL_CT(fill_corder_true_nt_f64,  _yw2w_n_ct_nt_f64,  double)
// @BE fill_corder_true_t_f64
FFI_YW2W_FILL_CT(fill_corder_true_t_f64,   _yw2w_n_ct_t_f64,   double)
// @BE fill_corder_true_nt_f16
FFI_YW2W_FILL_CT(fill_corder_true_nt_f16,  _yw2w_n_ct_nt_f16,  __half)
// @BE fill_corder_true_t_f16
FFI_YW2W_FILL_CT(fill_corder_true_t_f16,   _yw2w_n_ct_t_f16,   __half)
// @BE fill_corder_true_nt_bf16
FFI_YW2W_FILL_CT(fill_corder_true_nt_bf16, _yw2w_n_ct_nt_bf16, __nv_bfloat16)
// @BE fill_corder_true_t_bf16
FFI_YW2W_FILL_CT(fill_corder_true_t_bf16,  _yw2w_n_ct_t_bf16,  __nv_bfloat16)

// ====================== normal — fill, corder=false ======================
// @BE fill_corder_false_nt_f32
FFI_YW2W_FILL_CF(fill_corder_false_nt_f32,  _yw2w_n_cf_nt_f32,  float)
// @BE fill_corder_false_t_f32
FFI_YW2W_FILL_CF(fill_corder_false_t_f32,   _yw2w_n_cf_t_f32,   float)
// @BE fill_corder_false_nt_f64
FFI_YW2W_FILL_CF(fill_corder_false_nt_f64,  _yw2w_n_cf_nt_f64,  double)
// @BE fill_corder_false_t_f64
FFI_YW2W_FILL_CF(fill_corder_false_t_f64,   _yw2w_n_cf_t_f64,   double)
// @BE fill_corder_false_nt_f16
FFI_YW2W_FILL_CF(fill_corder_false_nt_f16,  _yw2w_n_cf_nt_f16,  __half)
// @BE fill_corder_false_t_f16
FFI_YW2W_FILL_CF(fill_corder_false_t_f16,   _yw2w_n_cf_t_f16,   __half)
// @BE fill_corder_false_nt_bf16
FFI_YW2W_FILL_CF(fill_corder_false_nt_bf16, _yw2w_n_cf_nt_bf16, __nv_bfloat16)
// @BE fill_corder_false_t_bf16
FFI_YW2W_FILL_CF(fill_corder_false_t_bf16,  _yw2w_n_cf_t_bf16,  __nv_bfloat16)
