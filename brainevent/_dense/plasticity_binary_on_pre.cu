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
// Tiling configuration
// =========================================================================

#define ON_PRE_ROW_TILE 32

// =========================================================================
// Warp-Cooperative Vectorized Kernel (Redux)
//
// 1. Each block handles ON_PRE_ROW_TILE (32) rows.
// 2. We use __ballot_sync to identify active rows.
// 3. We gather active row indices into shared memory.
// 4. We distribute the column updates across all threads.
// 5. This ensures that if even only one row is active, all 256 threads
//    in the block are working on it together.
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
                                                                            \
    if (threadIdx.x == 0) n_act = 0;                                        \
    __syncthreads();                                                        \
                                                                            \
    int row_base = blockIdx.y * 32;                                         \
    if (threadIdx.x < 32) {                                                 \
        int r = row_base + threadIdx.x;                                    \
        if (r < n_pre && IS_ACTIVE(spike[r])) {                             \
            int pos = atomicAdd(&n_act, 1);                                 \
            active_rows[pos] = r;                                           \
        }                                                                   \
    }                                                                       \
    __syncthreads();                                                        \
                                                                            \
    int count = n_act;                                                      \
    if (count == 0) return;                                                 \
                                                                            \
    int col_tile_base = blockIdx.x * 1024;                                  \
    size_t stride = (size_t)n_post;                                         \
                                                                            \
    /* All threads cooperate to update the gathered active rows */          \
    for (int i = 0; i < count; ++i) {                                       \
        int row = active_rows[i];                                           \
        WEIGHT_T* w_row = out_w + (size_t)row * stride;                     \
                                                                            \
        /* Each block processes a 1024-column chunk of the row */           \
        for (int j = threadIdx.x; j < 1024; j += 256) {                     \
            int col = col_tile_base + j;                                    \
            if (col < n_post) {                                             \
                ACC_T val = READ_W(w_row[col]) + READ_W(trace[col]);        \
                w_row[col] = WRITE_W(val);                                  \
            }                                                               \
        }                                                                   \
    }                                                                       \
}

// ---- Instantiate kernels ----
DEFINE_ON_PRE_FINAL(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float, float, READ_F32, WRITE_F32)
DEFINE_ON_PRE_FINAL(_f32_float, float,  IS_ACTIVE_FLOAT, float, float, READ_F32, WRITE_F32)
DEFINE_ON_PRE_FINAL(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64, WRITE_F64)
DEFINE_ON_PRE_FINAL(_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64, WRITE_F64)
DEFINE_ON_PRE_FINAL(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float, READ_F16, WRITE_F16)
DEFINE_ON_PRE_FINAL(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float, READ_F16, WRITE_F16)
DEFINE_ON_PRE_FINAL(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, WRITE_BF16)
DEFINE_ON_PRE_FINAL(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, WRITE_BF16)

// ---- FFI Entry Points ----
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
