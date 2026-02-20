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
 * plasticity_binary.cu -- COO Plasticity Update CUDA Kernels
 * ==========================================================
 *
 * This module provides optimized CUDA kernels for synaptic weight updates
 * driven by binary spike events in Coordinate (COO) sparse format.
 * It includes both pre-synaptic and post-synaptic update rules, typically
 * used in spike-timing-dependent plasticity (STDP) models.
 *
 * Supported Operations:
 * --------------------
 * 1. update_coo_on_pre: weight[i] += post_trace[post_ids[i]] if pre_spike[pre_ids[i]]
 * 2. update_coo_on_post: weight[i] += pre_trace[pre_ids[i]] if post_spike[post_ids[i]]
 *
 * Optimization Features:
 * ----------------------
 * - Warp-Ballot Early Exit: Entire warps (32 threads) skip processing if all
 *   corresponding neurons are inactive, drastically reducing overhead in
 *   sparse spike regimes.
 * - Coalesced Memory Access: Synapse-level data (weights, indices) are read
 *   sequentially to maximize L1/L2 cache efficiency.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// =========================================================================
// Active-check predicates
// =========================================================================

#define IS_ACTIVE_BOOL(s)   ((s) != 0)
#define IS_ACTIVE_FLOAT(s)  ((s) != 0.0f)

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
// COO Pre-Synaptic Plasticity Kernel
// =========================================================================

#define DEFINE_COO_ON_PRE(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T,     \
                          READ_W, WRITE_W)                                  \
__global__ void __launch_bounds__(256) _coo_on_pre_kern##SUFFIX(            \
    WEIGHT_T*         __restrict__ out_w,                                   \
    const SPIKE_T*    __restrict__ spike,                                   \
    const WEIGHT_T*   __restrict__ trace,                                   \
    const int32_t*    __restrict__ pre_ids,                                 \
    const int32_t*    __restrict__ post_ids,                                \
    int n_syn                                                               \
) {                                                                         \
    int i = (int)(blockIdx.x * (uint32_t)blockDim.x) + threadIdx.x;        \
    int safe_i = (i < n_syn) ? i : (n_syn - 1);                            \
    bool my_active = (i < n_syn) && IS_ACTIVE(spike[pre_ids[safe_i]]);      \
    unsigned int ballot = __ballot_sync(0xFFFFFFFF, my_active);             \
    if (ballot == 0) return;                                                \
    if (my_active) {                                                        \
        ACC_T val = READ_W(out_w[i]) + READ_W(trace[post_ids[i]]);         \
        out_w[i] = WRITE_W(val);                                            \
    }                                                                       \
}

// Instantiations
DEFINE_COO_ON_PRE(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,          float,  READ_F32,  WRITE_F32)
DEFINE_COO_ON_PRE(_f32_float, float,  IS_ACTIVE_FLOAT, float,          float,  READ_F32,  WRITE_F32)
DEFINE_COO_ON_PRE(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double,         double, READ_F64,  WRITE_F64)
DEFINE_COO_ON_PRE(_f64_float, float,  IS_ACTIVE_FLOAT, double,         double, READ_F64,  WRITE_F64)
DEFINE_COO_ON_PRE(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half,         float,  READ_F16,  WRITE_F16)
DEFINE_COO_ON_PRE(_f16_float, float,  IS_ACTIVE_FLOAT, __half,         float,  READ_F16,  WRITE_F16)
DEFINE_COO_ON_PRE(_bf16_bool, int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16,  float,  READ_BF16, WRITE_BF16)
DEFINE_COO_ON_PRE(_bf16_float,float,  IS_ACTIVE_FLOAT, __nv_bfloat16,  float,  READ_BF16, WRITE_BF16)

// =========================================================================
// COO Post-Synaptic Plasticity Kernel
// =========================================================================

#define DEFINE_COO_ON_POST(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T,    \
                           READ_W, WRITE_W)                                 \
__global__ void __launch_bounds__(256) _coo_on_post_kern##SUFFIX(           \
    WEIGHT_T*         __restrict__ out_w,                                   \
    const SPIKE_T*    __restrict__ spike,                                   \
    const WEIGHT_T*   __restrict__ trace,                                   \
    const int32_t*    __restrict__ pre_ids,                                 \
    const int32_t*    __restrict__ post_ids,                                \
    int n_syn                                                               \
) {                                                                         \
    int i = (int)(blockIdx.x * (uint32_t)blockDim.x) + threadIdx.x;        \
    int safe_i = (i < n_syn) ? i : (n_syn - 1);                            \
    bool my_active = (i < n_syn) && IS_ACTIVE(spike[post_ids[safe_i]]);     \
    unsigned int ballot = __ballot_sync(0xFFFFFFFF, my_active);             \
    if (ballot == 0) return;                                                \
    if (my_active) {                                                        \
        ACC_T val = READ_W(out_w[i]) + READ_W(trace[pre_ids[i]]);          \
        out_w[i] = WRITE_W(val);                                            \
    }                                                                       \
}

// Instantiations
DEFINE_COO_ON_POST(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,          float,  READ_F32,  WRITE_F32)
DEFINE_COO_ON_POST(_f32_float, float,  IS_ACTIVE_FLOAT, float,          float,  READ_F32,  WRITE_F32)
DEFINE_COO_ON_POST(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double,         double, READ_F64,  WRITE_F64)
DEFINE_COO_ON_POST(_f64_float, float,  IS_ACTIVE_FLOAT, double,         double, READ_F64,  WRITE_F64)
DEFINE_COO_ON_POST(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half,         float,  READ_F16,  WRITE_F16)
DEFINE_COO_ON_POST(_f16_float, float,  IS_ACTIVE_FLOAT, __half,         float,  READ_F16,  WRITE_F16)
DEFINE_COO_ON_POST(_bf16_bool, int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16,  float,  READ_BF16, WRITE_BF16)
DEFINE_COO_ON_POST(_bf16_float,float,  IS_ACTIVE_FLOAT, __nv_bfloat16,  float,  READ_BF16, WRITE_BF16)

// =========================================================================
// TVM FFI Entry Points
// =========================================================================

#define FFI_COO_ON_PRE(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                      \
void update_coo_on_pre##SUFFIX(                                             \
    tvm::ffi::TensorView weight,                                            \
    tvm::ffi::TensorView pre_ids,                                           \
    tvm::ffi::TensorView post_ids,                                          \
    tvm::ffi::TensorView spike,                                             \
    tvm::ffi::TensorView trace,                                             \
    tvm::ffi::TensorView out_weight,                                        \
    int64_t stream                                                          \
) {                                                                         \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                \
    int n_syn = static_cast<int>(out_weight.size(0));                       \
    if (n_syn == 0) return;                                                 \
    WEIGHT_C_T*        d_w    = static_cast<WEIGHT_C_T*>(                   \
                                    out_weight.data_ptr());                 \
    const SPIKE_C_T*   d_spk  = static_cast<const SPIKE_C_T*>(             \
                                    spike.data_ptr());                      \
    const WEIGHT_C_T*  d_tr   = static_cast<const WEIGHT_C_T*>(            \
                                    trace.data_ptr());                      \
    const int32_t*     d_pre  = static_cast<const int32_t*>(               \
                                    pre_ids.data_ptr());                    \
    const int32_t*     d_post = static_cast<const int32_t*>(               \
                                    post_ids.data_ptr());                   \
    int grid_size = (n_syn + 255) / 256;                                    \
    _coo_on_pre_kern##SUFFIX<<<grid_size, 256, 0, s>>>(                     \
        d_w, d_spk, d_tr, d_pre, d_post, n_syn);                           \
}

#define FFI_COO_ON_POST(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                     \
void update_coo_on_post##SUFFIX(                                            \
    tvm::ffi::TensorView weight,                                            \
    tvm::ffi::TensorView pre_ids,                                           \
    tvm::ffi::TensorView post_ids,                                          \
    tvm::ffi::TensorView trace,                                             \
    tvm::ffi::TensorView spike,                                             \
    tvm::ffi::TensorView out_weight,                                        \
    int64_t stream                                                          \
) {                                                                         \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                \
    int n_syn = static_cast<int>(out_weight.size(0));                       \
    if (n_syn == 0) return;                                                 \
    WEIGHT_C_T*        d_w    = static_cast<WEIGHT_C_T*>(                   \
                                    out_weight.data_ptr());                 \
    const SPIKE_C_T*   d_spk  = static_cast<const SPIKE_C_T*>(             \
                                    spike.data_ptr());                      \
    const WEIGHT_C_T*  d_tr   = static_cast<const WEIGHT_C_T*>(            \
                                    trace.data_ptr());                      \
    const int32_t*     d_pre  = static_cast<const int32_t*>(               \
                                    pre_ids.data_ptr());                    \
    const int32_t*     d_post = static_cast<const int32_t*>(               \
                                    post_ids.data_ptr());                   \
    int grid_size = (n_syn + 255) / 256;                                    \
    _coo_on_post_kern##SUFFIX<<<grid_size, 256, 0, s>>>(                    \
        d_w, d_spk, d_tr, d_pre, d_post, n_syn);                           \
}

// @tvm_ffi update_coo_on_pre_f32_bool
FFI_COO_ON_PRE(_f32_bool,  float,         int8_t)
// @tvm_ffi update_coo_on_pre_f32_float
FFI_COO_ON_PRE(_f32_float, float,         float)
// @tvm_ffi update_coo_on_pre_f64_bool
FFI_COO_ON_PRE(_f64_bool,  double,        int8_t)
// @tvm_ffi update_coo_on_pre_f64_float
FFI_COO_ON_PRE(_f64_float, double,        float)
// @tvm_ffi update_coo_on_pre_f16_bool
FFI_COO_ON_PRE(_f16_bool,  __half,        int8_t)
// @tvm_ffi update_coo_on_pre_f16_float
FFI_COO_ON_PRE(_f16_float, __half,        float)
// @tvm_ffi update_coo_on_pre_bf16_bool
FFI_COO_ON_PRE(_bf16_bool, __nv_bfloat16, int8_t)
// @tvm_ffi update_coo_on_pre_bf16_float
FFI_COO_ON_PRE(_bf16_float,__nv_bfloat16, float)

// @tvm_ffi update_coo_on_post_f32_bool
FFI_COO_ON_POST(_f32_bool,  float,         int8_t)
// @tvm_ffi update_coo_on_post_f32_float
FFI_COO_ON_POST(_f32_float, float,         float)
// @tvm_ffi update_coo_on_post_f64_bool
FFI_COO_ON_POST(_f64_bool,  double,        int8_t)
// @tvm_ffi update_coo_on_post_f64_float
FFI_COO_ON_POST(_f64_float, double,        float)
// @tvm_ffi update_coo_on_post_f16_bool
FFI_COO_ON_POST(_f16_bool,  __half,        int8_t)
// @tvm_ffi update_coo_on_post_f16_float
FFI_COO_ON_POST(_f16_float, __half,        float)
// @tvm_ffi update_coo_on_post_bf16_bool
FFI_COO_ON_POST(_bf16_bool, __nv_bfloat16, int8_t)
// @tvm_ffi update_coo_on_post_bf16_float
FFI_COO_ON_POST(_bf16_float,__nv_bfloat16, float)
