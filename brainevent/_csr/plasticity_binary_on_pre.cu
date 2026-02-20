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
 * plasticity_binary_on_pre.cu -- CSR Pre-Synaptic Plasticity Update CUDA Kernels
 * =================================================================================
 *
 * Python API:
 *   brainevent.update_csr_on_binary_pre(
 *       weight, indices, indptr, pre_spike, post_trace,
 *       shape=(n_pre, n_post), backend='tvmffi')
 *
 * Operation (in-place, CSR sparse format):
 *   For each pre-synaptic neuron i where pre_spike[i] is active:
 *       for pos in range(indptr[i], indptr[i+1]):
 *           weight[pos] += post_trace[indices[pos]]
 *
 * This implements the presynaptic STDP component: when a pre-neuron fires,
 * all its outgoing synaptic weights are updated by the postsynaptic traces.
 *
 * Kernel Variants (auto-dispatched by avg_nnz_per_row = nse / n_pre):
 * -------------------------------------------------------------------
 *   _thread_kern:  1 thread per row (256 threads/block).
 *                  Warp-ballot early exit: skip warp if all 32 rows inactive.
 *                  Best when avg_nnz < 32 (very sparse connectivity).
 *
 *   _warp_kern:    1 warp (32 threads) per row (256 threads/block = 8 warps).
 *                  Threads in a warp cooperate over the row's nonzeros with
 *                  stride-32 access, yielding coalesced writes.
 *                  Best when 32 <= avg_nnz < 256 (medium connectivity).
 *
 *   _block_kern:   1 block (256 threads) per row.
 *                  All 256 threads cooperate over one row with stride-256.
 *                  Best when avg_nnz >= 256 (dense connectivity).
 *
 *   _auto:         Host-side dispatch to thread/warp/block based on avg_nnz.
 *                  This is the recommended entry point.
 *
 * Parameters (TVM FFI tensors):
 *   weight    [nse]     float* (input, aliased as output out_weight)
 *   indices   [nse]     int32_t* (column indices, CSR format)
 *   indptr    [n_pre+1] int32_t* (row pointers, CSR format)
 *   spike     [n_pre]   spike_t* (bool as int8, or float)
 *   trace     [n_post]  float*   (postsynaptic eligibility traces)
 *   out_weight[nse]     float*   (output buffer, aliased to weight)
 *
 * Memory access pattern:
 *   - indptr[row], indptr[row+1]: two sequential reads per row (cached)
 *   - spike[row]: broadcast read per row (small array, stays in L2)
 *   - out_w[indptr[row]:indptr[row+1]]: coalesced sequential writes
 *   - indices[indptr[row]:indptr[row+1]]: coalesced sequential reads
 *   - trace[indices[...]]: scattered reads (trace is small, L2 cached)
 *
 * Dtype variants: f32, f64, f16, bf16  x  bool-spike, float-spike
 *
 * IMPORTANT: All data_ptr() values are GPU device pointers.
 *            NEVER dereference on the host. Extract only metadata (size, ndim).
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// =========================================================================
// Active-check predicates  (bool spikes stored as int8)
// =========================================================================

#define IS_ACTIVE_BOOL(s)   ((s) != 0)
#define IS_ACTIVE_FLOAT(s)  ((s) != 0.0f)

// =========================================================================
// Per-dtype read/write conversion macros
// (f16/bf16 accumulate in float32 for numerical stability)
// =========================================================================

#define READ_F32(x)    (x)
#define WRITE_F32(x)   (x)

#define READ_F64(x)    (x)
#define WRITE_F64(x)   (x)

#define READ_F16(x)    __half2float(x)
#define WRITE_F16(x)   __float2half(x)

#define READ_BF16(x)   __bfloat162float(x)
#define WRITE_BF16(x)  __float2bfloat16(x)

// =========================================================================
// Variant 1: Thread-per-row kernel
//
// One thread handles one row of the CSR matrix.
//
// Warp-ballot early exit:
//   Before entering the per-row loop, all 32 threads in a warp vote on
//   whether their assigned row has an active spike via __ballot_sync.
//   If none of the 32 rows are active (ballot == 0), the entire warp
//   exits immediately. This is highly effective for sparse spike trains
//   (density < 3%) where the vast majority of warps are fully inactive.
//
// Memory access:
//   - out_w[indptr[row]..indptr[row+1]-1]: sequential per thread, but
//     non-coalesced across threads (different rows at different offsets).
//   - trace[indices[pos]]: scattered, but trace[] is small -> L2 hit.
//   Preferred regime: avg_nnz < 32 where each thread's loop is short.
// =========================================================================

#define DEFINE_CSR_ON_PRE_THREAD(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, \
                                  READ_W, WRITE_W)                              \
__global__ void __launch_bounds__(256)                                          \
_csr_on_pre_thread_kern##SUFFIX(                                                \
    WEIGHT_T*        __restrict__ out_w,                                        \
    const SPIKE_T*   __restrict__ spike,                                        \
    const WEIGHT_T*  __restrict__ trace,                                        \
    const int32_t*   __restrict__ indices,                                      \
    const int32_t*   __restrict__ indptr,                                       \
    int n_pre                                                                   \
) {                                                                             \
    int row = (int)(blockIdx.x * (uint32_t)blockDim.x) + threadIdx.x;          \
    /* Clamp for warp-ballot: out-of-range threads check an in-range spike. */ \
    int safe_row = (row < n_pre) ? row : (n_pre - 1);                          \
    bool my_active = (row < n_pre) && IS_ACTIVE(spike[safe_row]);              \
    /* Warp-ballot: skip entire warp if no row is active */                     \
    unsigned int ballot = __ballot_sync(0xFFFFFFFF, my_active);                 \
    if (ballot == 0) return;                                                    \
    if (!my_active) return;                                                     \
    int start = indptr[row];                                                    \
    int end   = indptr[row + 1];                                                \
    for (int pos = start; pos < end; ++pos) {                                   \
        ACC_T val = READ_W(out_w[pos]) + READ_W(trace[indices[pos]]);           \
        out_w[pos] = WRITE_W(val);                                              \
    }                                                                           \
}

// =========================================================================
// Variant 2: Warp-per-row kernel
//
// One warp (32 threads) cooperates over a single row.
// Block size: 256 threads = 8 warps per block.
// Grid size:  ceil(n_pre / 8).
//
// All 32 threads in the warp check the same spike[row], so the ballot
// degenerates to a simple branch.  The coalescing benefit is the key:
// lane 0..31 accesses out_w[start+0], out_w[start+1], ..., out_w[start+31]
// in one coalesced transaction, then out_w[start+32..63], etc.
//
// For rows longer than 32 elements this gives full memory bandwidth
// utilization because each warp-step accesses 32 consecutive floats.
//
// Memory access:
//   - out_w[start + lane], [start + lane + 32], ...: fully coalesced.
//   - trace[indices[start + lane]]: scattered, but small -> L2 hit.
//   Preferred regime: 32 <= avg_nnz < 256.
// =========================================================================

#define DEFINE_CSR_ON_PRE_WARP(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, \
                                READ_W, WRITE_W)                              \
__global__ void __launch_bounds__(256)                                        \
_csr_on_pre_warp_kern##SUFFIX(                                                \
    WEIGHT_T*        __restrict__ out_w,                                      \
    const SPIKE_T*   __restrict__ spike,                                      \
    const WEIGHT_T*  __restrict__ trace,                                      \
    const int32_t*   __restrict__ indices,                                    \
    const int32_t*   __restrict__ indptr,                                     \
    int n_pre                                                                 \
) {                                                                           \
    /* 8 warps per block; each warp owns one row */                           \
    int warp_id = (int)(blockIdx.x * (blockDim.x / 32u))                     \
                  + (int)(threadIdx.x / 32u);                                 \
    int lane    = (int)(threadIdx.x & 31u);                                   \
    if (warp_id >= n_pre) return;                                             \
    if (!IS_ACTIVE(spike[warp_id])) return;                                   \
    int start = indptr[warp_id];                                              \
    int end   = indptr[warp_id + 1];                                          \
    /* Stride-32 loop: all 32 lanes access consecutive weight/index slots */ \
    for (int pos = start + lane; pos < end; pos += 32) {                      \
        ACC_T val = READ_W(out_w[pos]) + READ_W(trace[indices[pos]]);         \
        out_w[pos] = WRITE_W(val);                                            \
    }                                                                         \
}

// =========================================================================
// Variant 3: Block-per-row kernel
//
// One block (256 threads) cooperates over a single row.
// Grid size: n_pre blocks.
//
// Thread tid accesses out_w[start+tid], [start+tid+256], ...
// Provides maximum parallelism for very dense rows.
// If the row has 0 active pre-spike, the entire block returns immediately.
//
// Memory access:
//   - out_w[start + tid], [start + tid + 256], ...: fully coalesced per step.
//   - trace[indices[...]]: scattered, L2 cached for small trace[].
//   Preferred regime: avg_nnz >= 256 (high-connectivity layers).
// =========================================================================

#define DEFINE_CSR_ON_PRE_BLOCK(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, \
                                 READ_W, WRITE_W)                              \
__global__ void __launch_bounds__(256)                                         \
_csr_on_pre_block_kern##SUFFIX(                                                \
    WEIGHT_T*        __restrict__ out_w,                                       \
    const SPIKE_T*   __restrict__ spike,                                       \
    const WEIGHT_T*  __restrict__ trace,                                       \
    const int32_t*   __restrict__ indices,                                     \
    const int32_t*   __restrict__ indptr,                                      \
    int n_pre                                                                  \
) {                                                                            \
    int row = (int)blockIdx.x;                                                 \
    if (row >= n_pre) return;                                                  \
    if (!IS_ACTIVE(spike[row])) return;                                        \
    int start = indptr[row];                                                   \
    int end   = indptr[row + 1];                                               \
    int tid   = (int)threadIdx.x;                                              \
    for (int pos = start + tid; pos < end; pos += 256) {                       \
        ACC_T val = READ_W(out_w[pos]) + READ_W(trace[indices[pos]]);          \
        out_w[pos] = WRITE_W(val);                                             \
    }                                                                          \
}

// ---- Instantiate all three variants for all dtype combinations ----

DEFINE_CSR_ON_PRE_THREAD(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,          float,  READ_F32,  WRITE_F32)
DEFINE_CSR_ON_PRE_THREAD(_f32_float, float,  IS_ACTIVE_FLOAT, float,          float,  READ_F32,  WRITE_F32)
DEFINE_CSR_ON_PRE_THREAD(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double,         double, READ_F64,  WRITE_F64)
DEFINE_CSR_ON_PRE_THREAD(_f64_float, float,  IS_ACTIVE_FLOAT, double,         double, READ_F64,  WRITE_F64)
DEFINE_CSR_ON_PRE_THREAD(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half,         float,  READ_F16,  WRITE_F16)
DEFINE_CSR_ON_PRE_THREAD(_f16_float, float,  IS_ACTIVE_FLOAT, __half,         float,  READ_F16,  WRITE_F16)
DEFINE_CSR_ON_PRE_THREAD(_bf16_bool, int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16,  float,  READ_BF16, WRITE_BF16)
DEFINE_CSR_ON_PRE_THREAD(_bf16_float,float,  IS_ACTIVE_FLOAT, __nv_bfloat16,  float,  READ_BF16, WRITE_BF16)

DEFINE_CSR_ON_PRE_WARP(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,          float,  READ_F32,  WRITE_F32)
DEFINE_CSR_ON_PRE_WARP(_f32_float, float,  IS_ACTIVE_FLOAT, float,          float,  READ_F32,  WRITE_F32)
DEFINE_CSR_ON_PRE_WARP(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double,         double, READ_F64,  WRITE_F64)
DEFINE_CSR_ON_PRE_WARP(_f64_float, float,  IS_ACTIVE_FLOAT, double,         double, READ_F64,  WRITE_F64)
DEFINE_CSR_ON_PRE_WARP(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half,         float,  READ_F16,  WRITE_F16)
DEFINE_CSR_ON_PRE_WARP(_f16_float, float,  IS_ACTIVE_FLOAT, __half,         float,  READ_F16,  WRITE_F16)
DEFINE_CSR_ON_PRE_WARP(_bf16_bool, int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16,  float,  READ_BF16, WRITE_BF16)
DEFINE_CSR_ON_PRE_WARP(_bf16_float,float,  IS_ACTIVE_FLOAT, __nv_bfloat16,  float,  READ_BF16, WRITE_BF16)

DEFINE_CSR_ON_PRE_BLOCK(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,          float,  READ_F32,  WRITE_F32)
DEFINE_CSR_ON_PRE_BLOCK(_f32_float, float,  IS_ACTIVE_FLOAT, float,          float,  READ_F32,  WRITE_F32)
DEFINE_CSR_ON_PRE_BLOCK(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double,         double, READ_F64,  WRITE_F64)
DEFINE_CSR_ON_PRE_BLOCK(_f64_float, float,  IS_ACTIVE_FLOAT, double,         double, READ_F64,  WRITE_F64)
DEFINE_CSR_ON_PRE_BLOCK(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half,         float,  READ_F16,  WRITE_F16)
DEFINE_CSR_ON_PRE_BLOCK(_f16_float, float,  IS_ACTIVE_FLOAT, __half,         float,  READ_F16,  WRITE_F16)
DEFINE_CSR_ON_PRE_BLOCK(_bf16_bool, int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16,  float,  READ_BF16, WRITE_BF16)
DEFINE_CSR_ON_PRE_BLOCK(_bf16_float,float,  IS_ACTIVE_FLOAT, __nv_bfloat16,  float,  READ_BF16, WRITE_BF16)


// =========================================================================
// TVM FFI Entry Points (_auto: host-side dispatch to thread/warp/block)
//
// Auto-dispatch thresholds (avg_nnz = nse / n_pre):
//   avg_nnz <  32   -> thread_kern (256 rows per block, warp-ballot exit)
//   avg_nnz <  256  -> warp_kern   (8 rows per block, stride-32 coalesced)
//   avg_nnz >= 256  -> block_kern  (1 row per block, stride-256 coalesced)
//
// Host-only safe operations: size(0), size(1), ndim() on TensorView.
// NEVER dereference data_ptr() on the host.
// =========================================================================

#define FFI_CSR_ON_PRE(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                        \
void update_csr_on_pre##SUFFIX(                                               \
    tvm::ffi::TensorView weight,                                              \
    tvm::ffi::TensorView indices,                                             \
    tvm::ffi::TensorView indptr,                                              \
    tvm::ffi::TensorView spike,                                               \
    tvm::ffi::TensorView trace,                                               \
    tvm::ffi::TensorView out_weight,                                          \
    int64_t stream                                                            \
) {                                                                           \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                  \
    int nse   = static_cast<int>(out_weight.size(0));                         \
    int n_pre = static_cast<int>(indptr.size(0)) - 1;                         \
    if (n_pre <= 0 || nse == 0) return;                                       \
    WEIGHT_C_T*       d_w   = static_cast<WEIGHT_C_T*>(                       \
                                  out_weight.data_ptr());                     \
    const SPIKE_C_T*  d_spk = static_cast<const SPIKE_C_T*>(                  \
                                  spike.data_ptr());                          \
    const WEIGHT_C_T* d_tr  = static_cast<const WEIGHT_C_T*>(                 \
                                  trace.data_ptr());                          \
    const int32_t*    d_idx = static_cast<const int32_t*>(                    \
                                  indices.data_ptr());                        \
    const int32_t*    d_ipt = static_cast<const int32_t*>(                    \
                                  indptr.data_ptr());                         \
    /* Auto-dispatch based on average nonzeros per row */                     \
    int avg_nnz = nse / n_pre;                                                \
    if (avg_nnz < 32) {                                                       \
        /* Thread-per-row: 256 threads/block */                               \
        int grid = (n_pre + 255) / 256;                                       \
        _csr_on_pre_thread_kern##SUFFIX<<<grid, 256, 0, s>>>(                 \
            d_w, d_spk, d_tr, d_idx, d_ipt, n_pre);                          \
    } else if (avg_nnz < 256) {                                               \
        /* Warp-per-row: 256 threads/block = 8 warps -> 8 rows per block */  \
        int grid = (n_pre + 7) / 8;                                           \
        _csr_on_pre_warp_kern##SUFFIX<<<grid, 256, 0, s>>>(                   \
            d_w, d_spk, d_tr, d_idx, d_ipt, n_pre);                          \
    } else {                                                                  \
        /* Block-per-row: n_pre blocks, 256 threads each */                  \
        _csr_on_pre_block_kern##SUFFIX<<<n_pre, 256, 0, s>>>(                 \
            d_w, d_spk, d_tr, d_idx, d_ipt, n_pre);                          \
    }                                                                         \
}

// @tvm_ffi update_csr_on_pre_f32_bool
FFI_CSR_ON_PRE(_f32_bool,  float,          int8_t)
// @tvm_ffi update_csr_on_pre_f32_float
FFI_CSR_ON_PRE(_f32_float, float,          float)
// @tvm_ffi update_csr_on_pre_f64_bool
FFI_CSR_ON_PRE(_f64_bool,  double,         int8_t)
// @tvm_ffi update_csr_on_pre_f64_float
FFI_CSR_ON_PRE(_f64_float, double,         float)
// @tvm_ffi update_csr_on_pre_f16_bool
FFI_CSR_ON_PRE(_f16_bool,  __half,         int8_t)
// @tvm_ffi update_csr_on_pre_f16_float
FFI_CSR_ON_PRE(_f16_float, __half,         float)
// @tvm_ffi update_csr_on_pre_bf16_bool
FFI_CSR_ON_PRE(_bf16_bool, __nv_bfloat16,  int8_t)
// @tvm_ffi update_csr_on_pre_bf16_float
FFI_CSR_ON_PRE(_bf16_float,__nv_bfloat16,  float)
