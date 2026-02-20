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
 * plasticity_binary_on_post.cu -- COO Post-Synaptic Plasticity Update CUDA Kernels
 * ===========================================================================
 *
 * Python API:
 *   brainevent.update_coo_on_binary_post(
 *       weight, pre_ids, post_ids, pre_trace, post_spike, *, backend='tvmffi')
 *
 * Operation (in-place, COO sparse format):
 *   For each synapse i where post_spike[post_ids[i]] is active:
 *       weight[i] += pre_trace[pre_ids[i]]
 *
 * Kernel Design:
 *   Symmetric to the pre-synaptic kernel: spike lookup uses post_ids[i],
 *   trace lookup uses pre_ids[i]. All optimizations (warp-ballot early exit,
 *   coalesced sequential access, L2 cache reuse) apply identically.
 *
 * Parameters (TVM FFI tensors):
 *   weight    [n_syn]   float* (aliased as input+output)
 *   pre_ids   [n_syn]   int32_t* (presynaptic index per synapse)
 *   post_ids  [n_syn]   int32_t* (postsynaptic index per synapse)
 *   trace     [n_pre]   float*   (presynaptic eligibility traces)
 *   spike     [n_post]  spike_t* (bool as int8, or float)
 *   out_weight[n_syn]   float*   (output buffer; aliased to weight)
 *
 * Dtype variants: f32, f64, f16, bf16  x  bool-spike, float-spike
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// =========================================================================
// Active-check predicates  (spike is stored as int8 for bool dtype)
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
// COO Post-Synaptic Plasticity Kernel
//
// Symmetric to the pre-synaptic kernel.  The only difference is:
//   - spike is indexed by post_ids[i]   (postsynaptic spike indicator)
//   - trace is indexed by pre_ids[i]    (presynaptic eligibility trace)
//
// All optimizations apply identically:
//   - Warp-ballot early exit for sparse post-synaptic spike trains.
//   - Coalesced sequential access for weight[], pre_ids[], post_ids[].
//   - Small spike[] and trace[] arrays stay in L2 cache.
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
                                                                            \
    /* Guard out-of-range threads while keeping all lanes in the vote */    \
    int safe_i = (i < n_syn) ? i : (n_syn - 1);                            \
    bool my_active = (i < n_syn) && IS_ACTIVE(spike[post_ids[safe_i]]);     \
                                                                            \
    /* Warp-ballot: skip the entire warp if no active synapse */            \
    unsigned int ballot = __ballot_sync(0xFFFFFFFF, my_active);             \
    if (ballot == 0) return;                                                \
                                                                            \
    if (my_active) {                                                        \
        ACC_T val = READ_W(out_w[i]) + READ_W(trace[pre_ids[i]]);          \
        out_w[i] = WRITE_W(val);                                            \
    }                                                                       \
}

// ---- Instantiate all dtype combinations ----
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
//
// Argument order matches jax.ffi.ffi_call invocation:
//   (weight, pre_ids, post_ids, trace, spike, out_weight, stream)
// NEVER dereference data_ptr() on the host — pass to kernel unchanged.
// =========================================================================

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
