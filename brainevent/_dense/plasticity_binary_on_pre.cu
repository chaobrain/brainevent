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
 * plasticity_binary_on_pre.cu -- Dense Pre-Synaptic Plasticity Update CUDA Kernels
 * ==================================================================================
 *
 * Operation: weight[i, :] += post_trace for each active pre_spike[i]
 *
 * Public Python API
 * -----------------
 * update_dense_on_pre_<wt_sfx>_<spk_sfx>(weight, spike, trace, out_weight, stream)
 *   weight     : (n_pre, n_post) weight matrix (read-only, aliased to out_weight)
 *   spike      : (n_pre,) binary spike array (int8_t or float)
 *   trace      : (n_post,) postsynaptic trace (same dtype as weight)
 *   out_weight : (n_pre, n_post) updated weight matrix (output)
 *   stream     : int64_t CUDA stream handle
 *
 * Weight dtype suffixes (_wt_sfx): _f16, _bf16, _f32, _f64
 * Spike dtype suffixes (_spk_sfx): _bool (int8_t), _float (float)
 *
 * Optimization Features
 * ---------------------
 * - Shared memory caching of post-synaptic trace vector (COL_TILE_SIZE elements)
 * - Active row gathering into shared memory for warp-cooperative processing
 * - 4-way manual loop unrolling for instruction-level parallelism
 */

#include "cuda_common.h"
#include "brainevent/common.h"

// =========================================================================
// Dense Pre-Synaptic Plasticity Kernels
// =========================================================================

#define COL_TILE_SIZE 1024

#define DEFINE_ON_PRE_FINAL(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T,              \
                             READ_W, WRITE_W)                                         \
__global__ void __launch_bounds__(256) _on_pre_final_kern##SUFFIX(                    \
    WEIGHT_T*       __restrict__ out_w,                                               \
    const SPIKE_T*  __restrict__ spike,                                               \
    const WEIGHT_T* __restrict__ trace,                                               \
    int n_pre, int n_post                                                             \
) {                                                                                   \
    __shared__ int active_rows[32];                                                   \
    __shared__ int n_act;                                                             \
    __shared__ ACC_T trace_cache[COL_TILE_SIZE];                                      \
    if (threadIdx.x == 0) n_act = 0;                                                  \
    __syncthreads();                                                                  \
    int row_base = blockIdx.y * 32;                                                   \
    if (threadIdx.x < 32) {                                                           \
        int r = row_base + threadIdx.x;                                               \
        if (r < n_pre && IS_ACTIVE(spike[r])) {                                       \
            int pos = atomicAdd(&n_act, 1);                                           \
            active_rows[pos] = r;                                                     \
        }                                                                             \
    }                                                                                 \
    int col_tile_base = blockIdx.x * COL_TILE_SIZE;                                   \
    int tile_cols = min(COL_TILE_SIZE, n_post - col_tile_base);                       \
    for (int j = threadIdx.x; j < tile_cols; j += 256) {                              \
        int col = col_tile_base + j;                                                  \
        trace_cache[j] = READ_W(trace[col]);                                          \
    }                                                                                 \
    __syncthreads();                                                                  \
    int count = n_act;                                                                \
    if (count == 0) return;                                                           \
    size_t stride = (size_t)n_post;                                                   \
    for (int i = 0; i < count; ++i) {                                                 \
        int row = active_rows[i];                                                     \
        WEIGHT_T* w_row = out_w + (size_t)row * stride;                               \
        int j = threadIdx.x;                                                          \
        for (; j + 512 <= tile_cols; j += 1024) {                                     \
            ACC_T v0 = READ_W(w_row[col_tile_base + j])       + trace_cache[j];       \
            ACC_T v1 = READ_W(w_row[col_tile_base + j + 256]) + trace_cache[j + 256]; \
            ACC_T v2 = READ_W(w_row[col_tile_base + j + 512]) + trace_cache[j + 512]; \
            ACC_T v3 = READ_W(w_row[col_tile_base + j + 768]) + trace_cache[j + 768]; \
            w_row[col_tile_base + j]       = WRITE_W(v0);                             \
            w_row[col_tile_base + j + 256] = WRITE_W(v1);                             \
            w_row[col_tile_base + j + 512] = WRITE_W(v2);                             \
            w_row[col_tile_base + j + 768] = WRITE_W(v3);                             \
        }                                                                             \
        for (; j < tile_cols; j += 256) {                                             \
            int col = col_tile_base + j;                                              \
            ACC_T val = READ_W(w_row[col]) + trace_cache[j];                          \
            w_row[col] = WRITE_W(val);                                                \
        }                                                                             \
    }                                                                                 \
}

// Instantiations
DEFINE_ON_PRE_FINAL(_f32_bool,   int8_t, IS_ACTIVE_BOOL,  float,          float,  READ_F32,  WRITE_F32)
DEFINE_ON_PRE_FINAL(_f32_float,  float,  IS_ACTIVE_FLOAT, float,          float,  READ_F32,  WRITE_F32)
DEFINE_ON_PRE_FINAL(_f64_bool,   int8_t, IS_ACTIVE_BOOL,  double,         double, READ_F64,  WRITE_F64)
DEFINE_ON_PRE_FINAL(_f64_float,  float,  IS_ACTIVE_FLOAT, double,         double, READ_F64,  WRITE_F64)
DEFINE_ON_PRE_FINAL(_f16_bool,   int8_t, IS_ACTIVE_BOOL,  __half,         float,  READ_F16,  WRITE_F16)
DEFINE_ON_PRE_FINAL(_f16_float,  float,  IS_ACTIVE_FLOAT, __half,         float,  READ_F16,  WRITE_F16)
DEFINE_ON_PRE_FINAL(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16,  float,  READ_BF16, WRITE_BF16)
DEFINE_ON_PRE_FINAL(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16,  float,  READ_BF16, WRITE_BF16)

// =========================================================================
// TVM FFI Entry Points
// =========================================================================

#define FFI_ON_PRE(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                                 \
void update_dense_on_pre##SUFFIX(                                                 \
    const BE::Tensor weight,                                                  \
    const BE::Tensor spike,                                                   \
    const BE::Tensor trace,                                                   \
    const BE::Tensor out_weight,                                              \
    int64_t stream                                                                \
) {                                                                               \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                      \
    int n_pre  = static_cast<int>(out_weight.size(0));                            \
    int n_post = static_cast<int>(out_weight.size(1));                            \
    WEIGHT_C_T*       d_w     = static_cast<WEIGHT_C_T*>(out_weight.data_ptr());  \
    const SPIKE_C_T*  d_spk   = static_cast<const SPIKE_C_T*>(spike.data_ptr());  \
    const WEIGHT_C_T* d_trace = static_cast<const WEIGHT_C_T*>(trace.data_ptr()); \
    int n_col_blocks = (n_post + 1023) / 1024;                                    \
    int n_row_blocks = (n_pre + 31) / 32;                                         \
    dim3 grid(n_col_blocks, n_row_blocks);                                        \
    _on_pre_final_kern##SUFFIX<<<grid, 256, 0, s>>>(                              \
        d_w, d_spk, d_trace, n_pre, n_post);                                      \
}

// @BE update_dense_on_pre_f32_bool
FFI_ON_PRE(_f32_bool,   float,          int8_t)
// @BE update_dense_on_pre_f32_float
FFI_ON_PRE(_f32_float,  float,          float)
// @BE update_dense_on_pre_f64_bool
FFI_ON_PRE(_f64_bool,   double,         int8_t)
// @BE update_dense_on_pre_f64_float
FFI_ON_PRE(_f64_float,  double,         float)
// @BE update_dense_on_pre_f16_bool
FFI_ON_PRE(_f16_bool,   __half,         int8_t)
// @BE update_dense_on_pre_f16_float
FFI_ON_PRE(_f16_float,  __half,         float)
// @BE update_dense_on_pre_bf16_bool
FFI_ON_PRE(_bf16_bool,  __nv_bfloat16,  int8_t)
// @BE update_dense_on_pre_bf16_float
FFI_ON_PRE(_bf16_float, __nv_bfloat16,  float)
