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
 * plasticity_row_driven.cu
 * ========================
 * Favorable (row-driven) ELL plasticity weight update on binary spikes.
 *
 * For each row r with active spike[r]:
 *     out_w[r*n_conn + k] += trace[indices[r*n_conn + k]]   for k in [0, n_conn)
 *
 * ELL stores n_conn entries per row contiguously, so this is the CSR pre kernel
 * with indptr[r] == r*n_conn. Each (r,k) is written once -> no atomics.
 *
 * Entry points: update_fcn_row_{wt}_{spk}, wt in {f16,bf16,f32,f64}, spk in {bool,float}.
 */

#include "cuda_common.h"
#include "brainevent/common.h"

#define DEFINE_FCN_ROW_THREAD(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W) \
__global__ void __launch_bounds__(256)                                                       \
_fcn_row_thread_kern##SUFFIX(                                                                 \
    WEIGHT_T*       __restrict__ out_w,                                                       \
    const SPIKE_T*  __restrict__ spike,                                                       \
    const WEIGHT_T* __restrict__ trace,                                                       \
    const int32_t*  __restrict__ indices,                                                     \
    int n_row, int n_conn                                                                     \
) {                                                                                           \
    int row = (int)(blockIdx.x * (uint32_t)blockDim.x) + threadIdx.x;                         \
    int safe_row = (row < n_row) ? row : (n_row - 1);                                         \
    bool active = (row < n_row) && IS_ACTIVE(__ldg(&spike[safe_row]));                        \
    if (__ballot_sync(0xFFFFFFFF, active) == 0) return;                                       \
    if (!active) return;                                                                      \
    int start = row * n_conn;                                                                 \
    int end   = start + n_conn;                                                               \
    for (int pos = start; pos < end; ++pos) {                                                 \
        int col = __ldg(&indices[pos]);                                                       \
        ACC_T val = READ_W(out_w[pos]) + READ_W(__ldg(&trace[col]));                          \
        out_w[pos] = WRITE_W(val);                                                            \
    }                                                                                         \
}

#define DEFINE_FCN_ROW_WARP(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W) \
__global__ void __launch_bounds__(256)                                                     \
_fcn_row_warp_kern##SUFFIX(                                                                 \
    WEIGHT_T*       __restrict__ out_w,                                                     \
    const SPIKE_T*  __restrict__ spike,                                                     \
    const WEIGHT_T* __restrict__ trace,                                                     \
    const int32_t*  __restrict__ indices,                                                   \
    int n_row, int n_conn                                                                   \
) {                                                                                         \
    int warp_id = (int)(blockIdx.x * (blockDim.x / 32u)) + (int)(threadIdx.x / 32u);        \
    int lane    = (int)(threadIdx.x & 31u);                                                 \
    if (warp_id >= n_row) return;                                                           \
    bool active = IS_ACTIVE(__ldg(&spike[warp_id]));                                        \
    if (__ballot_sync(0xFFFFFFFF, active) == 0) return;                                     \
    if (!active) return;                                                                    \
    int start = warp_id * n_conn;                                                           \
    int end   = start + n_conn;                                                             \
    for (int pos = start + lane; pos < end; pos += 32) {                                    \
        int col = __ldg(&indices[pos]);                                                     \
        ACC_T val = READ_W(out_w[pos]) + READ_W(__ldg(&trace[col]));                        \
        out_w[pos] = WRITE_W(val);                                                          \
    }                                                                                       \
}

#define DEFINE_FCN_ROW_BLOCK(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W) \
__global__ void __launch_bounds__(256)                                                      \
_fcn_row_block_kern##SUFFIX(                                                                 \
    WEIGHT_T*       __restrict__ out_w,                                                      \
    const SPIKE_T*  __restrict__ spike,                                                      \
    const WEIGHT_T* __restrict__ trace,                                                      \
    const int32_t*  __restrict__ indices,                                                    \
    int n_row, int n_conn                                                                    \
) {                                                                                          \
    int row = (int)blockIdx.x;                                                               \
    if (row >= n_row) return;                                                                \
    if (!IS_ACTIVE(__ldg(&spike[row]))) return;                                              \
    int start = row * n_conn;                                                                \
    int end   = start + n_conn;                                                              \
    for (int pos = start + (int)threadIdx.x; pos < end; pos += 256) {                        \
        int col = __ldg(&indices[pos]);                                                      \
        ACC_T val = READ_W(out_w[pos]) + READ_W(__ldg(&trace[col]));                         \
        out_w[pos] = WRITE_W(val);                                                           \
    }                                                                                        \
}

#define DEFINE_FCN_ROW_ALL(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W) \
    DEFINE_FCN_ROW_THREAD(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W)   \
    DEFINE_FCN_ROW_WARP(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W)     \
    DEFINE_FCN_ROW_BLOCK(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W)

DEFINE_FCN_ROW_ALL(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,         float,  READ_F32,  WRITE_F32)
DEFINE_FCN_ROW_ALL(_f32_float, float,  IS_ACTIVE_FLOAT, float,         float,  READ_F32,  WRITE_F32)
DEFINE_FCN_ROW_ALL(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double,        double, READ_F64,  WRITE_F64)
DEFINE_FCN_ROW_ALL(_f64_float, float,  IS_ACTIVE_FLOAT, double,        double, READ_F64,  WRITE_F64)
DEFINE_FCN_ROW_ALL(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half,        float,  READ_F16,  WRITE_F16)
DEFINE_FCN_ROW_ALL(_f16_float, float,  IS_ACTIVE_FLOAT, __half,        float,  READ_F16,  WRITE_F16)
DEFINE_FCN_ROW_ALL(_bf16_bool, int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float,  READ_BF16, WRITE_BF16)
DEFINE_FCN_ROW_ALL(_bf16_float,float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float,  READ_BF16, WRITE_BF16)

#define FFI_FCN_ROW(SUFFIX, WEIGHT_C_T, SPIKE_C_T)            \
void update_fcn_row##SUFFIX(                                  \
    const BE::Tensor data,                                    \
    const BE::Tensor indices,                                 \
    const BE::Tensor spike,                                   \
    const BE::Tensor trace,                                   \
    BE::Tensor out_weight,                                    \
    int64_t stream                                            \
) {                                                           \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);  \
    int n_row  = static_cast<int>(out_weight.size(0));        \
    int n_conn = static_cast<int>(out_weight.size(1));        \
    if (n_row <= 0 || n_conn <= 0) return;                    \
    WEIGHT_C_T*       d_w   = static_cast<WEIGHT_C_T*>(out_weight.data_ptr());      \
    const SPIKE_C_T*  d_spk = static_cast<const SPIKE_C_T*>(spike.data_ptr());      \
    const WEIGHT_C_T* d_tr  = static_cast<const WEIGHT_C_T*>(trace.data_ptr());     \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());      \
    if (n_conn < 32) {                                        \
        int grid = (n_row + 255) / 256;                       \
        _fcn_row_thread_kern##SUFFIX<<<grid, 256, 0, s>>>(d_w, d_spk, d_tr, d_idx, n_row, n_conn); \
    } else if (n_conn < 256) {                                \
        int grid = (n_row + 7) / 8;                           \
        _fcn_row_warp_kern##SUFFIX<<<grid, 256, 0, s>>>(d_w, d_spk, d_tr, d_idx, n_row, n_conn);   \
    } else {                                                  \
        _fcn_row_block_kern##SUFFIX<<<n_row, 256, 0, s>>>(d_w, d_spk, d_tr, d_idx, n_row, n_conn); \
    }                                                         \
}

// @BE update_fcn_row_f32_bool
FFI_FCN_ROW(_f32_bool,  float,         int8_t)
// @BE update_fcn_row_f32_float
FFI_FCN_ROW(_f32_float, float,         float)
// @BE update_fcn_row_f64_bool
FFI_FCN_ROW(_f64_bool,  double,        int8_t)
// @BE update_fcn_row_f64_float
FFI_FCN_ROW(_f64_float, double,        float)
// @BE update_fcn_row_f16_bool
FFI_FCN_ROW(_f16_bool,  __half,        int8_t)
// @BE update_fcn_row_f16_float
FFI_FCN_ROW(_f16_float, __half,        float)
// @BE update_fcn_row_bf16_bool
FFI_FCN_ROW(_bf16_bool, __nv_bfloat16, int8_t)
// @BE update_fcn_row_bf16_float
FFI_FCN_ROW(_bf16_float,__nv_bfloat16, float)
