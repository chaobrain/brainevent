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
 * plasticity_binary.cu -- CSR Plasticity Update CUDA Kernels
 * ==========================================================
 *
 * This module provides optimized CUDA kernels for synaptic weight updates
 * in Compressed Sparse Row (CSR) format triggered by binary spike events.
 * It includes both pre-synaptic and post-synaptic update rules.
 *
 * Supported Operations:
 * --------------------
 * 1. update_csr_on_pre: Triggered by presynaptic spikes. 
 *    Updates outgoing synaptic weights using postsynaptic traces.
 *    Optimized via thread, warp, and block variants based on row density.
 *
 * 2. update_csr_on_post: Triggered by postsynaptic spikes.
 *    Updates incoming synaptic weights using presynaptic traces.
 *    Uses a CSC-like indexing structure into the CSR weight array.
 *
 * Optimization Features:
 * ----------------------
 * - Warp-Ballot Early Exit: Skips processing for inactive neurons to reduce overhead.
 * - Multi-level Parallelism: Auto-dispatch to thread, warp, or block variants
 *   based on sparsity to maximize throughput.
 * - Coalesced Memory Access: Leverages CSR/CSC layouts for sequential indexing.
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
// Per-dtype conversion macros
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
// Atomic-add helpers
// =========================================================================

__device__ __forceinline__ void atomic_add_f32(float* addr, float val) {
    atomicAdd(addr, val);
}
__device__ __forceinline__ void atomic_add_f64(double* addr, double val) {
    atomicAdd(addr, val);
}
__device__ __forceinline__ void atomic_add_f16(__half* addr, float delta) {
#if __CUDA_ARCH__ >= 700
    atomicAdd(addr, __float2half(delta));
#else
    unsigned int* addr32 = reinterpret_cast<unsigned int*>(reinterpret_cast<uintptr_t>(addr) & ~3u);
    unsigned int old32 = *addr32;
    unsigned int assumed;
    do {
        assumed = old32;
        unsigned int shift = (reinterpret_cast<uintptr_t>(addr) & 2u) ? 16u : 0u;
        __half old_half = __ushort_as_half(static_cast<unsigned short>((assumed >> shift) & 0xFFFFu));
        float new_f = __half2float(old_half) + delta;
        unsigned short new_ush = __half_as_ushort(__float2half(new_f));
        unsigned int new32 = (assumed & ~(0xFFFFu << shift)) | (static_cast<unsigned int>(new_ush) << shift);
        old32 = atomicCAS(addr32, assumed, new32);
    } while (old32 != assumed);
#endif
}
__device__ __forceinline__ void atomic_add_bf16(__nv_bfloat16* addr, float delta) {
#if __CUDA_ARCH__ >= 800
    atomicAdd(addr, __float2bfloat16(delta));
#else
    unsigned int* addr32 = reinterpret_cast<unsigned int*>(reinterpret_cast<uintptr_t>(addr) & ~3u);
    unsigned int old32 = *addr32;
    unsigned int assumed;
    do {
        assumed = old32;
        unsigned int shift = (reinterpret_cast<uintptr_t>(addr) & 2u) ? 16u : 0u;
        __nv_bfloat16 old_bf = __ushort_as_bfloat16(static_cast<unsigned short>((assumed >> shift) & 0xFFFFu));
        float new_f = __bfloat162float(old_bf) + delta;
        unsigned short new_ush = __bfloat16_as_ushort(__float2bfloat16(new_f));
        unsigned int new32 = (assumed & ~(0xFFFFu << shift)) | (static_cast<unsigned int>(new_ush) << shift);
        old32 = atomicCAS(addr32, assumed, new32);
    } while (old32 != assumed);
#endif
}

// =========================================================================
// CSR Pre-Synaptic Plasticity Kernels
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
    int safe_row = (row < n_pre) ? row : (n_pre - 1);                          \
    bool my_active = (row < n_pre) && IS_ACTIVE(spike[safe_row]);              \
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
    int warp_id = (int)(blockIdx.x * (blockDim.x / 32u))                     \
                  + (int)(threadIdx.x / 32u);                                 \
    int lane    = (int)(threadIdx.x & 31u);                                   \
    if (warp_id >= n_pre) return;                                             \
    if (!IS_ACTIVE(spike[warp_id])) return;                                   \
    int start = indptr[warp_id];                                              \
    int end   = indptr[warp_id + 1];                                          \
    for (int pos = start + lane; pos < end; pos += 32) {                      \
        ACC_T val = READ_W(out_w[pos]) + READ_W(trace[indices[pos]]);         \
        out_w[pos] = WRITE_W(val);                                            \
    }                                                                         \
}

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

// Sp-Pre Instantiations
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
// CSR Post-Synaptic Plasticity Kernels
// =========================================================================

#define DEFINE_CSR_ON_POST_THREAD(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, \
                                   READ_W, WRITE_W, ATOMIC_ADD)                 \
__global__ void __launch_bounds__(256)                                           \
_csr_on_post_thread_kern##SUFFIX(                                                \
    WEIGHT_T*        __restrict__ out_w,                                         \
    const SPIKE_T*   __restrict__ spike,                                         \
    const WEIGHT_T*  __restrict__ trace,                                         \
    const int32_t*   __restrict__ indices,                                       \
    const int32_t*   __restrict__ indptr,                                        \
    const int32_t*   __restrict__ weight_indices,                                \
    int n_post                                                                   \
) {                                                                              \
    int col = (int)(blockIdx.x * (uint32_t)blockDim.x) + threadIdx.x;           \
    int safe_col = (col < n_post) ? col : (n_post - 1);                         \
    bool my_active = (col < n_post) && IS_ACTIVE(spike[safe_col]);              \
    unsigned int ballot = __ballot_sync(0xFFFFFFFF, my_active);                  \
    if (ballot == 0) return;                                                     \
    if (!my_active) return;                                                      \
    int start = indptr[col];                                                     \
    int end   = indptr[col + 1];                                                 \
    for (int pos = start; pos < end; ++pos) {                                    \
        ACC_T delta = READ_W(trace[indices[pos]]);                               \
        ATOMIC_ADD(&out_w[weight_indices[pos]], delta);                          \
    }                                                                            \
}

#define DEFINE_CSR_ON_POST_WARP(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, \
                                 READ_W, WRITE_W, ATOMIC_ADD)                 \
__global__ void __launch_bounds__(256)                                        \
_csr_on_post_warp_kern##SUFFIX(                                               \
    WEIGHT_T*        __restrict__ out_w,                                      \
    const SPIKE_T*   __restrict__ spike,                                      \
    const WEIGHT_T*  __restrict__ trace,                                      \
    const int32_t*   __restrict__ indices,                                    \
    const int32_t*   __restrict__ indptr,                                     \
    const int32_t*   __restrict__ weight_indices,                             \
    int n_post                                                                \
) {                                                                           \
    int warp_id = (int)(blockIdx.x * (blockDim.x / 32u))                     \
                  + (int)(threadIdx.x / 32u);                                 \
    int lane    = (int)(threadIdx.x & 31u);                                   \
    if (warp_id >= n_post) return;                                            \
    if (!IS_ACTIVE(spike[warp_id])) return;                                   \
    int start = indptr[warp_id];                                              \
    int end   = indptr[warp_id + 1];                                          \
    for (int pos = start + lane; pos < end; pos += 32) {                      \
        ACC_T delta = READ_W(trace[indices[pos]]);                            \
        ATOMIC_ADD(&out_w[weight_indices[pos]], delta);                       \
    }                                                                         \
}

#define DEFINE_CSR_ON_POST_BLOCK(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, \
                                  READ_W, WRITE_W, ATOMIC_ADD)                 \
__global__ void __launch_bounds__(256)                                          \
_csr_on_post_block_kern##SUFFIX(                                                \
    WEIGHT_T*        __restrict__ out_w,                                        \
    const SPIKE_T*   __restrict__ spike,                                        \
    const WEIGHT_T*  __restrict__ trace,                                        \
    const int32_t*   __restrict__ indices,                                      \
    const int32_t*   __restrict__ indptr,                                       \
    const int32_t*   __restrict__ weight_indices,                               \
    int n_post                                                                  \
) {                                                                             \
    int col = (int)blockIdx.x;                                                  \
    if (col >= n_post) return;                                                  \
    if (!IS_ACTIVE(spike[col])) return;                                         \
    int start = indptr[col];                                                    \
    int end   = indptr[col + 1];                                                \
    int tid   = (int)threadIdx.x;                                               \
    for (int pos = start + tid; pos < end; pos += 256) {                        \
        ACC_T delta = READ_W(trace[indices[pos]]);                              \
        ATOMIC_ADD(&out_w[weight_indices[pos]], delta);                         \
    }                                                                           \
}

// Sp-Post Instantiations
DEFINE_CSR_ON_POST_THREAD(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,         float,  READ_F32,  WRITE_F32,  atomic_add_f32)
DEFINE_CSR_ON_POST_THREAD(_f32_float, float,  IS_ACTIVE_FLOAT, float,         float,  READ_F32,  WRITE_F32,  atomic_add_f32)
DEFINE_CSR_ON_POST_THREAD(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double,        double, READ_F64,  WRITE_F64,  atomic_add_f64)
DEFINE_CSR_ON_POST_THREAD(_f64_float, float,  IS_ACTIVE_FLOAT, double,        double, READ_F64,  WRITE_F64,  atomic_add_f64)
DEFINE_CSR_ON_POST_THREAD(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half,        float,  READ_F16,  WRITE_F16,  atomic_add_f16)
DEFINE_CSR_ON_POST_THREAD(_f16_float, float,  IS_ACTIVE_FLOAT, __half,        float,  READ_F16,  WRITE_F16,  atomic_add_f16)
DEFINE_CSR_ON_POST_THREAD(_bf16_bool, int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float,  READ_BF16, WRITE_BF16, atomic_add_bf16)
DEFINE_CSR_ON_POST_THREAD(_bf16_float,float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, atomic_add_bf16)
DEFINE_CSR_ON_POST_WARP(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,         float,  READ_F32,  WRITE_F32,  atomic_add_f32)
DEFINE_CSR_ON_POST_WARP(_f32_float, float,  IS_ACTIVE_FLOAT, float,         float,  READ_F32,  WRITE_F32,  atomic_add_f32)
DEFINE_CSR_ON_POST_WARP(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double,        double, READ_F64,  WRITE_F64,  atomic_add_f64)
DEFINE_CSR_ON_POST_WARP(_f64_float, float,  IS_ACTIVE_FLOAT, double,        double, READ_F64,  WRITE_F64,  atomic_add_f64)
DEFINE_CSR_ON_POST_WARP(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half,        float,  READ_F16,  WRITE_F16,  atomic_add_f16)
DEFINE_CSR_ON_POST_WARP(_f16_float, float,  IS_ACTIVE_FLOAT, __half,        float,  READ_F16,  WRITE_F16,  atomic_add_f16)
DEFINE_CSR_ON_POST_WARP(_bf16_bool, int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float,  READ_BF16, WRITE_BF16, atomic_add_bf16)
DEFINE_CSR_ON_POST_WARP(_bf16_float,float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, atomic_add_bf16)
DEFINE_CSR_ON_POST_BLOCK(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,         float,  READ_F32,  WRITE_F32,  atomic_add_f32)
DEFINE_CSR_ON_POST_BLOCK(_f32_float, float,  IS_ACTIVE_FLOAT, float,         float,  READ_F32,  WRITE_F32,  atomic_add_f32)
DEFINE_CSR_ON_POST_BLOCK(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double,        double, READ_F64,  WRITE_F64,  atomic_add_f64)
DEFINE_CSR_ON_POST_BLOCK(_f64_float, float,  IS_ACTIVE_FLOAT, double,        double, READ_F64,  WRITE_F64,  atomic_add_f64)
DEFINE_CSR_ON_POST_BLOCK(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half,        float,  READ_F16,  WRITE_F16,  atomic_add_f16)
DEFINE_CSR_ON_POST_BLOCK(_f16_float, float,  IS_ACTIVE_FLOAT, __half,        float,  READ_F16,  WRITE_F16,  atomic_add_f16)
DEFINE_CSR_ON_POST_BLOCK(_bf16_bool, int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float,  READ_BF16, WRITE_BF16, atomic_add_bf16)
DEFINE_CSR_ON_POST_BLOCK(_bf16_float,float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float,  READ_BF16, WRITE_BF16, atomic_add_bf16)

// =========================================================================
// TVM FFI Entry Points
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
    int avg_nnz = nse / n_pre;                                                \
    if (avg_nnz < 32) {                                                       \
        int grid = (n_pre + 255) / 256;                                       \
        _csr_on_pre_thread_kern##SUFFIX<<<grid, 256, 0, s>>>(                 \
            d_w, d_spk, d_tr, d_idx, d_ipt, n_pre);                          \
    } else if (avg_nnz < 256) {                                               \
        int grid = (n_pre + 7) / 8;                                           \
        _csr_on_pre_warp_kern##SUFFIX<<<grid, 256, 0, s>>>(                   \
            d_w, d_spk, d_tr, d_idx, d_ipt, n_pre);                          \
    } else {                                                                  \
        _csr_on_pre_block_kern##SUFFIX<<<n_pre, 256, 0, s>>>(                 \
            d_w, d_spk, d_tr, d_idx, d_ipt, n_pre);                          \
    }                                                                         \
}

#define FFI_CSR_ON_POST(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                       \
void update_csr_on_post##SUFFIX(                                              \
    tvm::ffi::TensorView weight,                                              \
    tvm::ffi::TensorView indices,                                             \
    tvm::ffi::TensorView indptr,                                              \
    tvm::ffi::TensorView weight_indices,                                      \
    tvm::ffi::TensorView trace,                                               \
    tvm::ffi::TensorView spike,                                               \
    tvm::ffi::TensorView out_weight,                                          \
    int64_t stream                                                            \
) {                                                                           \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                  \
    int nse    = static_cast<int>(out_weight.size(0));                        \
    int n_post = static_cast<int>(indptr.size(0)) - 1;                        \
    if (n_post <= 0 || nse == 0) return;                                      \
    WEIGHT_C_T*       d_w    = static_cast<WEIGHT_C_T*>(                      \
                                   out_weight.data_ptr());                    \
    const SPIKE_C_T*  d_spk  = static_cast<const SPIKE_C_T*>(                 \
                                   spike.data_ptr());                         \
    const WEIGHT_C_T* d_tr   = static_cast<const WEIGHT_C_T*>(                \
                                   trace.data_ptr());                         \
    const int32_t*    d_idx  = static_cast<const int32_t*>(                   \
                                   indices.data_ptr());                       \
    const int32_t*    d_ipt  = static_cast<const int32_t*>(                   \
                                   indptr.data_ptr());                        \
    const int32_t*    d_widx = static_cast<const int32_t*>(                   \
                                   weight_indices.data_ptr());                \
    int avg_nnz = nse / n_post;                                               \
    if (avg_nnz < 32) {                                                       \
        int grid = (n_post + 255) / 256;                                      \
        _csr_on_post_thread_kern##SUFFIX<<<grid, 256, 0, s>>>(                \
            d_w, d_spk, d_tr, d_idx, d_ipt, d_widx, n_post);                 \
    } else if (avg_nnz < 256) {                                               \
        int grid = (n_post + 7) / 8;                                          \
        _csr_on_post_warp_kern##SUFFIX<<<grid, 256, 0, s>>>(                  \
            d_w, d_spk, d_tr, d_idx, d_ipt, d_widx, n_post);                 \
    } else {                                                                  \
        _csr_on_post_block_kern##SUFFIX<<<n_post, 256, 0, s>>>(               \
            d_w, d_spk, d_tr, d_idx, d_ipt, d_widx, n_post);                 \
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

// @tvm_ffi update_csr_on_post_f32_bool
FFI_CSR_ON_POST(_f32_bool,  float,          int8_t)
// @tvm_ffi update_csr_on_post_f32_float
FFI_CSR_ON_POST(_f32_float, float,          float)
// @tvm_ffi update_csr_on_post_f64_bool
FFI_CSR_ON_POST(_f64_bool,  double,         int8_t)
// @tvm_ffi update_csr_on_post_f64_float
FFI_CSR_ON_POST(_f64_float, double,         float)
// @tvm_ffi update_csr_on_post_f16_bool
FFI_CSR_ON_POST(_f16_bool,  __half,         int8_t)
// @tvm_ffi update_csr_on_post_f16_float
FFI_CSR_ON_POST(_f16_float, __half,         float)
// @tvm_ffi update_csr_on_post_bf16_bool
FFI_CSR_ON_POST(_bf16_bool, __nv_bfloat16,  int8_t)
// @tvm_ffi update_csr_on_post_bf16_float
FFI_CSR_ON_POST(_bf16_float,__nv_bfloat16,  float)
