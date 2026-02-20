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
 * plasticity_binary.cu -- Dense Plasticity Update CUDA Kernels
 * ============================================================
 *
 * This module provides optimized CUDA kernels for synaptic weight updates
 * in dense format triggered by binary spike events. It includes both
 * pre-synaptic and post-synaptic update rules.
 *
 * Supported Operations:
 * --------------------
 * 1. update_dense_on_pre: weight[i, :] += post_trace if pre_spike[i] is active
 * 2. update_dense_on_post: weight[:, j] += pre_trace if post_spike[j] is active
 *
 * Optimization Features:
 * ----------------------
 * - Warp-Cooperative Execution: Threads in a warp cooperate to update rows/columns
 *   efficiently, maximizing bandwidth and minimizing divergence.
 * - Shared Memory Tiling: Active indices are gathered into shared memory to
 *   distribute work across threads in a block.
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

#define READ_F32(x)   (x)
#define WRITE_F32(x)  (x)
#define READ_F64(x)   (x)
#define WRITE_F64(x)  (x)
#define READ_F16(x)   __half2float(x)
#define WRITE_F16(x)  __float2half(x)
#define READ_BF16(x)  __bfloat162float(x)
#define WRITE_BF16(x) __float2bfloat16(x)

// =========================================================================
// Dense Pre-Synaptic Plasticity Kernels
// =========================================================================

#define DEFINE_ON_PRE_FINAL(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T,  \
                             READ_W, WRITE_W)                               \
__global__ void __launch_bounds__(256) _on_pre_final_kern##SUFFIX(         \
    WEIGHT_T*       __restrict__ out_w,                                     \
    const SPIKE_T*  __restrict__ spike,                                     \
    const WEIGHT_T* __restrict__ trace,                                     \
    int n_pre, int n_post                                                   \
) {                                                                         \
    __shared__ int active_rows[32];                                         \
    __shared__ int n_act;                                                   \
    if (threadIdx.x == 0) n_act = 0;                                        \
    __syncthreads();                                                        \
    int row_base = blockIdx.y * 32;                                         \
    if (threadIdx.x < 32) {                                                 \
        int r = row_base + threadIdx.x;                                    \
        if (r < n_pre && IS_ACTIVE(spike[r])) {                             \
            int pos = atomicAdd(&n_act, 1);                                 \
            active_rows[pos] = r;                                           \
        }                                                                   \
    }                                                                       \
    __syncthreads();                                                        \
    int count = n_act;                                                      \
    if (count == 0) return;                                                 \
    int col_tile_base = blockIdx.x * 1024;                                  \
    size_t stride = (size_t)n_post;                                         \
    for (int i = 0; i < count; ++i) {                                       \
        int row = active_rows[i];                                           \
        WEIGHT_T* w_row = out_w + (size_t)row * stride;                     \
        for (int j = threadIdx.x; j < 1024; j += 256) {                     \
            int col = col_tile_base + j;                                    \
            if (col < n_post) {                                             \
                ACC_T val = READ_W(w_row[col]) + READ_W(trace[col]);        \
                w_row[col] = WRITE_W(val);                                  \
            }                                                               \
        }                                                                   \
    }                                                                       \
}

// Instantiations
DEFINE_ON_PRE_FINAL(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float, float, READ_F32, WRITE_F32)
DEFINE_ON_PRE_FINAL(_f32_float, float,  IS_ACTIVE_FLOAT, float, float, READ_F32, WRITE_F32)
DEFINE_ON_PRE_FINAL(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64, WRITE_F64)
DEFINE_ON_PRE_FINAL(_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64, WRITE_F64)
DEFINE_ON_PRE_FINAL(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float, READ_F16, WRITE_F16)
DEFINE_ON_PRE_FINAL(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float, READ_F16, WRITE_F16)
DEFINE_ON_PRE_FINAL(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, WRITE_BF16)
DEFINE_ON_PRE_FINAL(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, WRITE_BF16)

// =========================================================================
// Dense Post-Synaptic Plasticity Kernels
// =========================================================================

#define ON_POST_ROW_TILE 512

#define DEFINE_ON_POST_WARP(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T,  \
                             READ_W, WRITE_W)                               \
__global__ void __launch_bounds__(256) _on_post_warp_kern##SUFFIX(         \
    WEIGHT_T*       __restrict__ out_w,                                     \
    const WEIGHT_T* __restrict__ trace,                                     \
    const SPIKE_T*  __restrict__ spike,                                     \
    int n_pre, int n_post                                                   \
) {                                                                         \
    int tx = threadIdx.x & 31;                                              \
    int warp_in_block = threadIdx.x >> 5;                                   \
    int col_tile_base = blockIdx.x * 32;                                    \
    __shared__ int active_cols[8][32];                                      \
    int c = col_tile_base + tx;                                             \
    bool active = (c < n_post && IS_ACTIVE(spike[c]));                      \
    unsigned int mask = __ballot_sync(0xFFFFFFFF, active);                  \
    if (mask == 0) return;                                                  \
    int num_active = __popc(mask);                                          \
    if (active) {                                                           \
        int pos = __popc(mask & ((1u << tx) - 1));                         \
        active_cols[warp_in_block][pos] = c;                                \
    }                                                                       \
    int row_tile_start = blockIdx.y * ON_POST_ROW_TILE;                     \
    int row_tile_end   = min(row_tile_start + ON_POST_ROW_TILE, n_pre);      \
    int rows_in_tile   = row_tile_end - row_tile_start;                     \
    int rows_per_warp = (rows_in_tile + 7) / 8;                             \
    int my_row_start = row_tile_start + warp_in_block * rows_per_warp;      \
    int my_row_end   = min(my_row_start + rows_per_warp, row_tile_end);      \
    if (my_row_start >= my_row_end) return;                                 \
    int my_rows = my_row_end - my_row_start;                                \
    int total_work = num_active * my_rows;                                  \
    size_t stride = (size_t)n_post;                                         \
    for (int i = tx; i < total_work; i += 32) {                             \
        int col_idx = i % num_active;                                       \
        int row_off = i / num_active;                                       \
        int row = my_row_start + row_off;                                   \
        int global_col = active_cols[warp_in_block][col_idx];               \
        size_t offset = (size_t)row * stride + global_col;                  \
        out_w[offset] = WRITE_W(READ_W(out_w[offset]) + READ_W(trace[row]));\
    }                                                                       \
}

// Instantiations
DEFINE_ON_POST_WARP(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float, float, READ_F32, WRITE_F32)
DEFINE_ON_POST_WARP(_f32_float, float,  IS_ACTIVE_FLOAT, float, float, READ_F32, WRITE_F32)
DEFINE_ON_POST_WARP(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64, WRITE_F64)
DEFINE_ON_POST_WARP(_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64, WRITE_F64)
DEFINE_ON_POST_WARP(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float, READ_F16, WRITE_F16)
DEFINE_ON_POST_WARP(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float, READ_F16, WRITE_F16)
DEFINE_ON_POST_WARP(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, WRITE_BF16)
DEFINE_ON_POST_WARP(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, WRITE_BF16)

// =========================================================================
// TVM FFI Entry Points
// =========================================================================

#define FFI_ON_PRE(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                          \
void update_dense_on_pre##SUFFIX(                                           \
    tvm::ffi::TensorView weight,                                            \
    tvm::ffi::TensorView spike,                                             \
    tvm::ffi::TensorView trace,                                             \
    tvm::ffi::TensorView out_weight,                                        \
    int64_t stream                                                          \
) {                                                                         \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                \
    int n_pre  = static_cast<int>(out_weight.size(0));                     \
    int n_post = static_cast<int>(out_weight.size(1));                     \
    WEIGHT_C_T*       d_w     = static_cast<WEIGHT_C_T*>(out_weight.data_ptr()); \
    const SPIKE_C_T*  d_spk   = static_cast<const SPIKE_C_T*>(spike.data_ptr()); \
    const WEIGHT_C_T* d_trace = static_cast<const WEIGHT_C_T*>(trace.data_ptr()); \
    int n_col_blocks = (n_post + 1023) / 1024;                              \
    int n_row_blocks = (n_pre + 31) / 32;                                   \
    dim3 grid(n_col_blocks, n_row_blocks);                                  \
    _on_pre_final_kern##SUFFIX<<<grid, 256, 0, s>>>(                        \
        d_w, d_spk, d_trace, n_pre, n_post);                                \
}

#define FFI_ON_POST(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                         \
void update_dense_on_post##SUFFIX(                                          \
    tvm::ffi::TensorView weight,                                            \
    tvm::ffi::TensorView trace,                                             \
    tvm::ffi::TensorView spike,                                             \
    tvm::ffi::TensorView out_weight,                                        \
    int64_t stream                                                          \
) {                                                                         \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                \
    int n_pre  = static_cast<int>(out_weight.size(0));                     \
    int n_post = static_cast<int>(out_weight.size(1));                     \
    WEIGHT_C_T*       d_w     = static_cast<WEIGHT_C_T*>(out_weight.data_ptr()); \
    const WEIGHT_C_T* d_trace = static_cast<const WEIGHT_C_T*>(trace.data_ptr()); \
    const SPIKE_C_T*  d_spk   = static_cast<const SPIKE_C_T*>(spike.data_ptr()); \
    int n_col_blocks = (n_post + 31) / 32;                                 \
    int n_row_blocks = (n_pre + ON_POST_ROW_TILE - 1) / ON_POST_ROW_TILE;  \
    dim3 grid(n_col_blocks, n_row_blocks);                                  \
    _on_post_warp_kern##SUFFIX<<<grid, 256, 0, s>>>(                        \
        d_w, d_trace, d_spk, n_pre, n_post);                               \
}

// @tvm_ffi update_dense_on_pre_f32_bool
FFI_ON_PRE(_f32_bool,   float,          int8_t)
// @tvm_ffi update_dense_on_pre_f32_float
FFI_ON_PRE(_f32_float,  float,          float)
// @tvm_ffi update_dense_on_pre_f64_bool
FFI_ON_PRE(_f64_bool,   double,         int8_t)
// @tvm_ffi update_dense_on_pre_f64_float
FFI_ON_PRE(_f64_float,  double,         float)
// @tvm_ffi update_dense_on_pre_f16_bool
FFI_ON_PRE(_f16_bool,   __half,         int8_t)
// @tvm_ffi update_dense_on_pre_f16_float
FFI_ON_PRE(_f16_float,  __half,         float)
// @tvm_ffi update_dense_on_pre_bf16_bool
FFI_ON_PRE(_bf16_bool,  __nv_bfloat16,  int8_t)
// @tvm_ffi update_dense_on_pre_bf16_float
FFI_ON_PRE(_bf16_float, __nv_bfloat16,  float)

// @tvm_ffi update_dense_on_post_f32_bool
FFI_ON_POST(_f32_bool,   float,          int8_t)
// @tvm_ffi update_dense_on_post_f32_float
FFI_ON_POST(_f32_float,  float,          float)
// @tvm_ffi update_dense_on_post_f64_bool
FFI_ON_POST(_f64_bool,   double,         int8_t)
// @tvm_ffi update_dense_on_post_f64_float
FFI_ON_POST(_f64_float,  double,         float)
// @tvm_ffi update_dense_on_post_f16_bool
FFI_ON_POST(_f16_bool,   __half,         int8_t)
// @tvm_ffi update_dense_on_post_f16_float
FFI_ON_POST(_f16_float,  __half,         float)
// @tvm_ffi update_dense_on_post_bf16_bool
FFI_ON_POST(_bf16_bool,  __nv_bfloat16,  int8_t)
// @tvm_ffi update_dense_on_post_bf16_float
FFI_ON_POST(_bf16_float, __nv_bfloat16,  float)
