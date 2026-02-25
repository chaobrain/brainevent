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
 * plasticity_binary_update_coo_on_binary_pre.cu
 * =============================================
 *
 * CUDA kernels for COO pre-synaptic plasticity weight updates driven by
 * binary (spike) events.
 *
 * Operation:
 *   update_coo_on_pre:
 *     weight[i] += post_trace[post_ids[i]]  if pre_spike[pre_ids[i]] != 0
 *
 * Optimization Features:
 * - Two-Level Early Exit: out-of-bounds threads skip ALL memory reads,
 *   saving bandwidth; warps with zero in-bounds threads return immediately.
 * - Warp-Ballot Early Exit: entire warps skip processing when all
 *   corresponding neurons are inactive (sparse spike regime).
 * - Read-Only Cache (__ldg): spike and trace reads routed through the
 *   texture/read-only cache for improved hit rates on random access.
 * - Coalesced Memory Access: synapse-level data (weights, indices) are
 *   read sequentially to maximise L1/L2 cache efficiency.
 * - Block Size 512: improves occupancy and amortises warp scheduling.
 *
 * Python API:
 *   update_coo_on_binary_pre(weight, pre_ids, post_ids, pre_spike,
 *                            post_trace, w_min=None, w_max=None,
 *                            backend=None)
 *
 * TVM FFI Entry Points (one per dtype combination):
 *   update_coo_on_pre_{wt}_{spk}  where
 *     wt  in {f16, bf16, f32, f64}
 *     spk in {bool, float}
 */

#include "cuda_common.h"
#include "brainevent/common.h"

// =========================================================================
// COO Pre-Synaptic Plasticity Kernel
// =========================================================================

#define DEFINE_COO_ON_PRE(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, \
                          READ_W, WRITE_W)                             \
__global__ void __launch_bounds__(512) _coo_on_pre_kern##SUFFIX(       \
    WEIGHT_T*         __restrict__ out_w,                              \
    const SPIKE_T*    __restrict__ spike,                              \
    const WEIGHT_T*   __restrict__ trace,                              \
    const int32_t*    __restrict__ pre_ids,                            \
    const int32_t*    __restrict__ post_ids,                           \
    int n_syn                                                          \
) {                                                                    \
    int i = (int)(blockIdx.x * (uint32_t)blockDim.x) + threadIdx.x;    \
    bool in_bounds = (i < n_syn);                                      \
                                                                       \
    /* Early exit for out-of-bounds warps */                           \
    unsigned int bounds_ballot = __ballot_sync(0xFFFFFFFF, in_bounds); \
    if (bounds_ballot == 0) return;                                    \
                                                                       \
    bool my_active = false;                                            \
    int32_t pre_id, post_id;                                           \
    WEIGHT_T trace_val;                                                \
                                                                       \
    if (in_bounds) {                                                   \
        pre_id = __ldg(&pre_ids[i]);                                   \
        SPIKE_T spike_val = __ldg(&spike[pre_id]);                     \
        my_active = IS_ACTIVE(spike_val);                              \
                                                                       \
        if (my_active) {                                               \
            post_id = __ldg(&post_ids[i]);                             \
            trace_val = __ldg(&trace[post_id]);                        \
        }                                                              \
    }                                                                  \
                                                                       \
    /* Early exit for inactive warps */                                \
    unsigned int active_ballot = __ballot_sync(0xFFFFFFFF, my_active); \
    if (active_ballot == 0) return;                                    \
                                                                       \
    if (my_active) {                                                   \
        ACC_T val = READ_W(out_w[i]) + READ_W(trace_val);              \
        out_w[i] = WRITE_W(val);                                       \
    }                                                                  \
}

// Instantiations
DEFINE_COO_ON_PRE(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,          float,  READ_F32,  WRITE_F32)
DEFINE_COO_ON_PRE(_f32_float, float,  IS_ACTIVE_FLOAT, float,          float,  READ_F32,  WRITE_F32)
DEFINE_COO_ON_PRE(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double,         double, READ_F64,  WRITE_F64)
DEFINE_COO_ON_PRE(_f64_float, float,  IS_ACTIVE_FLOAT, double,         double, READ_F64,  WRITE_F64)
DEFINE_COO_ON_PRE(_f16_bool,  int8_t,         IS_ACTIVE_BOOL,  __half,         float,  READ_F16,  WRITE_F16)
DEFINE_COO_ON_PRE(_f16_float, __half,         IS_ACTIVE_F16,   __half,         float,  READ_F16,  WRITE_F16)
DEFINE_COO_ON_PRE(_bf16_bool, int8_t,         IS_ACTIVE_BOOL,  __nv_bfloat16,  float,  READ_BF16, WRITE_BF16)
DEFINE_COO_ON_PRE(_bf16_float,__nv_bfloat16,  IS_ACTIVE_BF16,  __nv_bfloat16,  float,  READ_BF16, WRITE_BF16)

// =========================================================================
// TVM FFI Entry Points
// =========================================================================

#define FFI_COO_ON_PRE(SUFFIX, WEIGHT_C_T, SPIKE_C_T)           \
void update_coo_on_pre##SUFFIX(                                 \
    const BE::Tensor weight,                                \
    const BE::Tensor pre_ids,                               \
    const BE::Tensor post_ids,                              \
    const BE::Tensor spike,                                 \
    const BE::Tensor trace,                                 \
    const BE::Tensor out_weight,                            \
    int64_t stream                                              \
) {                                                             \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);    \
    int n_syn = static_cast<int>(out_weight.size(0));           \
    if (n_syn == 0) return;                                     \
    const WEIGHT_C_T*  d_w_in = static_cast<const WEIGHT_C_T*>( \
                                    weight.data_ptr());         \
    WEIGHT_C_T*        d_w    = static_cast<WEIGHT_C_T*>(       \
                                    out_weight.data_ptr());     \
    const SPIKE_C_T*   d_spk  = static_cast<const SPIKE_C_T*>(  \
                                    spike.data_ptr());          \
    const WEIGHT_C_T*  d_tr   = static_cast<const WEIGHT_C_T*>( \
                                    trace.data_ptr());          \
    const int32_t*     d_pre  = static_cast<const int32_t*>(    \
                                    pre_ids.data_ptr());        \
    const int32_t*     d_post = static_cast<const int32_t*>(    \
                                    post_ids.data_ptr());       \
    cudaMemcpyAsync(d_w, d_w_in, (size_t)n_syn * sizeof(WEIGHT_C_T), \
                    cudaMemcpyDeviceToDevice, s);               \
    int grid_size = (n_syn + 511) / 512;                        \
    _coo_on_pre_kern##SUFFIX<<<grid_size, 512, 0, s>>>(         \
        d_w, d_spk, d_tr, d_pre, d_post, n_syn);                \
}

// @BE update_coo_on_pre_f32_bool
FFI_COO_ON_PRE(_f32_bool,  float,         int8_t)
// @BE update_coo_on_pre_f32_float
FFI_COO_ON_PRE(_f32_float, float,         float)
// @BE update_coo_on_pre_f64_bool
FFI_COO_ON_PRE(_f64_bool,  double,        int8_t)
// @BE update_coo_on_pre_f64_float
FFI_COO_ON_PRE(_f64_float, double,        float)
// @BE update_coo_on_pre_f16_bool
FFI_COO_ON_PRE(_f16_bool,  __half,        int8_t)
// @BE update_coo_on_pre_f16_float
FFI_COO_ON_PRE(_f16_float, __half,        __half)
// @BE update_coo_on_pre_bf16_bool
FFI_COO_ON_PRE(_bf16_bool, __nv_bfloat16, int8_t)
// @BE update_coo_on_pre_bf16_float
FFI_COO_ON_PRE(_bf16_float,__nv_bfloat16, __nv_bfloat16)
