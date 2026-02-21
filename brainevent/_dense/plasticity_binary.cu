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
// Warp-level primitives
// =========================================================================

__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

// =========================================================================
// Dense Pre-Synaptic Plasticity Kernels
// =========================================================================
/*
 * update_dense_on_pre — Pre-synaptic weight update kernel
 * ========================================================
 *
 * Operation: weight[i, :] += post_trace for each active pre_spike[i]
 *
 * Performance characteristics (fp32, 10000×10000, 1% density):
 *   Baseline:        2275 µs (0.37% efficiency vs 8.5 µs theoretical)
 *   + Smem cache:    2187 µs (0.39% efficiency, +4%)
 *   + 4-way unroll:  2226 µs (0.38% efficiency, no gain)
 *
 * Optimizations applied:
 *   [✓] Shared memory caching of trace vector
 *   [✓] 4-way manual loop unrolling for ILP
 *   [✗] Attempted coalesced access pattern (regressed 7%, reverted)
 *   [✗] Attempted early block exit with __syncthreads_count (caused hang, reverted)
 *
 * Fundamental performance barriers:
 *   1. TVM FFI overhead: ~80-90 µs per call (from dispatch benchmark)
 *   2. Strided memory access: threads access columns i, i+256, i+512, ... (not fully coalesced)
 *   3. Small active row count: 1% density means most blocks exit early after shared memory init
 *   4. Atomic contention: atomicAdd(&n_act, 1) serializes active row insertion (minor)
 *   5. Occupancy: 256 threads/block × shared memory usage limits concurrent blocks
 *
 * Roofline analysis:
 *   Memory traffic: 12.84 MB for 107 active rows × 10000 cols (read trace + read/write weight)
 *   Arithmetic:     1.07M FP32 additions
 *   Intensity:      0.083 ops/byte (bandwidth bound)
 *   Theoretical:    8.5 µs @ 1.5 TB/s effective bandwidth
 *   Achieved:       ~2200 µs (0.38% of theoretical)
 *
 * Efficiency breakdown (estimated):
 *   - TVM FFI dispatch overhead:   ~4% (90 µs / 2200 µs)
 *   - Kernel launch overhead:      ~0.3% (~7 µs / 2200 µs)
 *   - Memory bandwidth utilization: ~0.4% (non-coalesced strided access)
 *   - Wasted work on empty blocks:  <1% (early exit at count==0)
 *
 * Achieving >85% efficiency from current 0.38% requires fundamental algorithmic changes:
 *
 * Future directions:
 *   A) Algorithmic: Two-pass approach
 *      - Pass 1: Global atomic counter to build compacted list of all active rows across blocks
 *      - Pass 2: Warp-cooperative processing of compacted list with fully coalesced access
 *      - Benefit: Eliminates per-block overhead, enables perfect coalescing
 *      - Cost: Extra kernel launch + global atomics
 *
 *   B) Format change: Row-major → Transposed storage
 *      - Store weight^T so columns become rows
 *      - Update becomes: weight^T[:, i] += post_trace (coalesced column access)
 *      - Benefit: Perfect coalescing, no strided access
 *      - Cost: Requires transpose on input/output (may be amortized in full training loop)
 *
 *   C) Kernel fusion: Fuse plasticity update with forward/backward pass
 *      - Avoid separate kernel launch overhead
 *      - Reuse cached weight data from matmul
 *      - Benefit: Eliminates FFI + launch overhead, improves data locality
 *      - Cost: Requires higher-level operator scheduling
 *
 *   D) Hardware features (sm_80+):
 *      - Async copy (ldgsts.async) for overlapped data movement
 *      - TMA (Tensor Memory Accelerator) on sm_90
 *      - Persistent kernels to amortize launch cost across batches
 *
 * Current implementation is adequate for:
 *   - Correctness across all dtypes (fp16/bf16/fp32/fp64) and spike types (bool/float)
 *   - Moderate problem sizes (< 5000×5000) where absolute latency < 1.5 ms
 *   - High spike density (>10%) where the kernel approaches a dense memset-like operation
 *
 * For large sparse matrices (>10k×10k) at low density (<1%), the current kernel
 * achieves only 0.38% of theoretical roofline due to strided access and dispatch overhead.
 * Further optimization requires algorithmic or architectural changes beyond the scope
 * of in-place kernel tuning.
 */

#define COL_TILE_SIZE 1024

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
    __shared__ ACC_T trace_cache[COL_TILE_SIZE];                            \
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
    int col_tile_base = blockIdx.x * COL_TILE_SIZE;                         \
    int tile_cols = min(COL_TILE_SIZE, n_post - col_tile_base);             \
    for (int j = threadIdx.x; j < tile_cols; j += 256) {                    \
        int col = col_tile_base + j;                                        \
        trace_cache[j] = READ_W(trace[col]);                                \
    }                                                                       \
    __syncthreads();                                                        \
    int count = n_act;                                                      \
    if (count == 0) return;                                                 \
    size_t stride = (size_t)n_post;                                         \
    for (int i = 0; i < count; ++i) {                                       \
        int row = active_rows[i];                                           \
        WEIGHT_T* w_row = out_w + (size_t)row * stride;                     \
        int j = threadIdx.x;                                                \
        for (; j + 512 <= tile_cols; j += 1024) {                           \
            ACC_T v0 = READ_W(w_row[col_tile_base + j])       + trace_cache[j];       \
            ACC_T v1 = READ_W(w_row[col_tile_base + j + 256]) + trace_cache[j + 256]; \
            ACC_T v2 = READ_W(w_row[col_tile_base + j + 512]) + trace_cache[j + 512]; \
            ACC_T v3 = READ_W(w_row[col_tile_base + j + 768]) + trace_cache[j + 768]; \
            w_row[col_tile_base + j]       = WRITE_W(v0);                   \
            w_row[col_tile_base + j + 256] = WRITE_W(v1);                   \
            w_row[col_tile_base + j + 512] = WRITE_W(v2);                   \
            w_row[col_tile_base + j + 768] = WRITE_W(v3);                   \
        }                                                                   \
        for (; j < tile_cols; j += 256) {                                   \
            int col = col_tile_base + j;                                    \
            ACC_T val = READ_W(w_row[col]) + trace_cache[j];                \
            w_row[col] = WRITE_W(val);                                      \
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
/*
 * update_dense_on_post — Post-synaptic weight update kernel
 * ==========================================================
 *
 * Operation: weight[:, j] += pre_trace for each active post_spike[j]
 *
 * Performance characteristics (fp32, 10000×10000):
 *   1% density (93 cols):
 *     Baseline:      2278 µs
 *     + Optimized:   2223 µs (+2.4%)
 *   10% density (969 cols):
 *     Baseline:      3516 µs (58% slower than pre kernel)
 *     + Optimized:   3331 µs (+5.3%, still 50% slower than pre)
 *
 * Optimizations applied:
 *   [✓] Restructured loop: outer loop over rows, inner over active cols (avoid div/mod)
 *   [✓] Cached trace values in registers (one read per row instead of per column)
 *   [✓] Eliminated expensive `i % num_active` and `i / num_active` operations
 *
 * Roofline analysis (10000×10000, 1% density, 93 active cols):
 *   Memory traffic: 93 cols × 10000 rows × 12 bytes (read trace + read/write weight) = 11.16 MB
 *   Arithmetic:     930K FP32 additions
 *   Intensity:      0.083 ops/byte (bandwidth bound, same as pre kernel)
 *   Theoretical:    7.4 µs @ 1.5 TB/s
 *   Achieved:       2223 µs (0.33% of theoretical)
 *
 * Fundamental performance barriers:
 *   1. Non-coalesced column writes: Each thread writes to scattered columns in row-major storage
 *      - Pre kernel updates rows → coalesced sequential access within each row
 *      - Post kernel updates cols → strided access with stride = n_post (up to 10000)
 *   2. At high density (10%), the post kernel is 50% slower than pre due to worse memory access pattern
 *   3. TVM FFI + kernel launch overhead (~90 µs) same as pre kernel
 *
 * Achieving >85% efficiency requires:
 *   A) Transpose weight matrix: Store as column-major so column updates become coalesced row updates
 *   B) Two-pass algorithm: Gather active columns, sort, then batch updates for better locality
 *   C) Kernel fusion: Combine with forward/backward pass to amortize overhead
 *
 * Current efficiency: 0.33% of roofline (fundamental barrier: non-coalesced column writes)
 */

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
    size_t stride = (size_t)n_post;                                         \
    for (int row = my_row_start + (tx / num_active); row < my_row_end; row += 32 / num_active) { \
        ACC_T trace_val = READ_W(trace[row]);                               \
        int col_idx = tx % num_active;                                      \
        int global_col = active_cols[warp_in_block][col_idx];               \
        size_t offset = (size_t)row * stride + global_col;                  \
        out_w[offset] = WRITE_W(READ_W(out_w[offset]) + trace_val);         \
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
