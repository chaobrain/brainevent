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
 * plasticity_binary_on_post.cu -- Dense Post-Synaptic Plasticity Update CUDA Kernels
 * ===========================================================================
 *
 * Python API:
 *   brainevent.update_dense_on_binary_post(weight, pre_trace, post_spike,
 *                                          *, backend='tvmffi')
 *
 * Operation (in-place):
 *   For each postsynaptic neuron j where post_spike[j] is active:
 *       weight[:, j] += pre_trace
 *
 *   Equivalently:
 *       weight[i, j] += pre_trace[i]   for all i, if post_spike[j] active
 *
 * Parameters
 * ----------
 * weight    : dense float matrix [n_pre, n_post].  Updated in-place.
 * pre_trace : 1-D float vector [n_pre], same dtype as weight.
 * post_spike: 1-D vector [n_post].
 *             bool  (int8): active when != 0.
 *             float (f32):  active when != 0.0f.
 *
 * Kernel variant: 2D-Tiled Column-Parallel (_on_post_tiled_kern)
 * ---------------------------------------------------------------
 * Tile size: TILE_COLS = 32 columns (one warp-width per row-group).
 * Row tile:  ON_POST_ROW_TILE rows per block in the Y dimension.
 *
 *   grid  = (ceil(n_post / 32), ceil(n_pre / ON_POST_ROW_TILE))
 *   block = (256,) = 8 row-groups × 32 column-lanes
 *
 *   col_in_tile = threadIdx.x & 31   (0..31, column offset within tile)
 *   row_group   = threadIdx.x >> 5   (0..7,  row striding group)
 *   col         = blockIdx.x * 32 + col_in_tile
 *   row_start   = blockIdx.y * ON_POST_ROW_TILE
 *   row_end     = min(row_start + ON_POST_ROW_TILE, n_pre)
 *
 * Spike activity pre-fetch (shared memory):
 *   Threads 0..31 each load one spike value into spk_active[32].
 *   After __syncthreads(), every thread checks spk_active[col_in_tile].
 *   Inactive-column threads return immediately — entire columns skipped
 *   with a single shared-memory load rather than per-row global reads.
 *
 * Row iteration (coalesced writes):
 *   for row = row_start + row_group; row < row_end; row += 8:
 *     out_w[row * n_post + col] += trace[row]
 *
 *   Memory access pattern:
 *     CUDA warp k (tx = 32k .. 32k+31): row_group = k, col_in_tile = 0..31
 *     → Accesses out_w[k * n_post + col_start .. col_start+31]
 *     → 32 consecutive addresses → FULLY COALESCED.
 *   All 8 warps simultaneously update 8 different rows of the same 32-col
 *   tile → 8 × 32 = 256 elements updated per kernel iteration with
 *   maximum memory-level parallelism.
 *
 * 2D grid advantage over 1D:
 *   The 1D grid (ceil(n_post/32),) gives very few blocks for large n_pre.
 *   Example: 10000×10000 → 313 blocks, each looping over all 10000 rows.
 *   The 2D grid splits the row dimension into tiles of ON_POST_ROW_TILE
 *   rows, giving 313×79 = ~25K blocks → much better GPU occupancy.
 *
 * Float16 and bfloat16 kernels accumulate in float32 for numerical stability.
 * Bfloat16 requires CUDA 11.0+.
 *
 * IMPORTANT: weight.data_ptr() / out_weight.data_ptr() are GPU device
 * pointers -- NEVER dereference on the host.  The host-side FFI entry
 * reads only metadata (size(0), size(1)); the device pointer is passed
 * unchanged to the CUDA kernel.
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
// READ converts WEIGHT_T -> ACC_T for computation.
// WRITE converts ACC_T -> WEIGHT_T for storage.
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
// Row-tile size for 2D grid Y-dimension.
// Each block handles ON_POST_ROW_TILE rows × 32 columns.
// 128 rows per tile: balances GPU occupancy (many blocks) against
// per-block amortization (16 row iterations per row_group).
// =========================================================================

#define ON_POST_ROW_TILE 128

// =========================================================================
// 2D-tiled column-parallel on_post kernel macro
//
// TILE_COLS = 32  (one warp width)
// BLOCK_SIZE = 256 = 8 row_groups × 32 col_lanes
//
// grid  = (ceil(n_post / 32), ceil(n_pre / ON_POST_ROW_TILE))
// block = (256,)
//
// Layout: col_in_tile = tx & 31,  row_group = tx >> 5
//   col       = blockIdx.x * 32 + col_in_tile
//   row_start = blockIdx.y * ON_POST_ROW_TILE
//   row_end   = min(row_start + ON_POST_ROW_TILE, n_pre)
//
// Shared memory: spk_active[32]
//   Loaded in parallel by the first warp (tx < 32).
//   Represents the spike activity for the 32 columns handled by this block.
//   Enables O(1) block-level early-exit when all 32 columns are inactive.
//
// Coalesced write guarantee:
//   Within each row iteration, warp w (tx=32w..32w+31) accesses:
//     out_w[w * n_post + col_start .. col_start+31]
//   = 32 consecutive float/half/double elements → fully coalesced.
// =========================================================================

#define DEFINE_ON_POST_TILED(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T,  \
                             READ_W, WRITE_W)                               \
__global__ void _on_post_tiled_kern##SUFFIX(                                \
    WEIGHT_T*       __restrict__ out_w,                                     \
    const WEIGHT_T* __restrict__ trace,                                     \
    const SPIKE_T*  __restrict__ spike,                                     \
    int n_pre, int n_post                                                   \
) {                                                                         \
    /* Shared memory: spike activity for the 32-column tile */              \
    __shared__ int spk_active[32];                                          \
    int col_in_tile = threadIdx.x & 31;   /* 0..31 */                      \
    int row_group   = threadIdx.x >> 5;   /* 0..7  */                      \
    int col = blockIdx.x * 32 + col_in_tile;                               \
    /* 2D grid: blockIdx.y selects the row tile */                          \
    int row_start = blockIdx.y * ON_POST_ROW_TILE;                          \
    int row_end   = row_start + ON_POST_ROW_TILE;                           \
    if (row_end > n_pre) row_end = n_pre;                                   \
    /* Load spike flags for this tile's 32 columns into shared memory */   \
    if (threadIdx.x < 32) {                                                \
        int c = blockIdx.x * 32 + threadIdx.x;                            \
        spk_active[threadIdx.x] = (c < n_post && IS_ACTIVE(spike[c])) ? 1 : 0; \
    }                                                                       \
    __syncthreads();                                                        \
    /* Early exit: this column is out-of-bounds or its spike is inactive */ \
    if (col >= n_post || !spk_active[col_in_tile]) return;                 \
    /* Stride over rows in [row_start, row_end): each row_group handles */ \
    /* rows row_start+row_group, +8, +16, ...  Coalesced within warp. */   \
    for (int row = row_start + row_group; row < row_end; row += 8) {       \
        ACC_T updated = READ_W(out_w[(size_t)row * n_post + col])          \
                      + READ_W(trace[row]);                                 \
        out_w[(size_t)row * n_post + col] = WRITE_W(updated);             \
    }                                                                       \
}

// =========================================================================
// Instantiate kernels: 4 weight dtypes x 2 spike types = 8 kernels
// =========================================================================

// ---- Float32 ----
DEFINE_ON_POST_TILED(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float, float, READ_F32, WRITE_F32)
DEFINE_ON_POST_TILED(_f32_float, float,  IS_ACTIVE_FLOAT, float, float, READ_F32, WRITE_F32)

// ---- Float64 ----
DEFINE_ON_POST_TILED(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64, WRITE_F64)
DEFINE_ON_POST_TILED(_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64, WRITE_F64)

// ---- Float16 (accumulate in float32 for numerical stability) ----
DEFINE_ON_POST_TILED(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float, READ_F16, WRITE_F16)
DEFINE_ON_POST_TILED(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float, READ_F16, WRITE_F16)

// ---- BFloat16 (accumulate in float32; requires CUDA 11.0+) ----
DEFINE_ON_POST_TILED(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, WRITE_BF16)
DEFINE_ON_POST_TILED(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, WRITE_BF16)


// =========================================================================
// TVM FFI Entry Point Macro
// =========================================================================
//
// Convention: args = (weight, trace, spike, out_weight, stream)
//   weight    [n_pre, n_post]  — GPU device ptr (same buffer as out_weight)
//   trace     [n_pre]          — GPU device ptr
//   spike     [n_post]         — GPU device ptr
//   out_weight[n_pre, n_post]  — GPU device ptr (aliased to weight input)
//   stream    int64_t          — cudaStream_t cast to int64_t
//
// The 'weight' input and 'out_weight' output share the same GPU memory
// buffer via JAX input_output_aliases={0: 0}.  The kernel reads and writes
// through out_weight's data_ptr exclusively.
//
// IMPORTANT: data_ptr() returns GPU device pointers.
// NEVER dereference on the host.  Pass to kernels unchanged.
//
// Grid = (ceil(n_post / 32), ceil(n_pre / ON_POST_ROW_TILE)), block = 256.
// =========================================================================

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
    /* out_weight.data_ptr() is aliased to weight.data_ptr() */            \
    WEIGHT_C_T*       d_w     = static_cast<WEIGHT_C_T*>(out_weight.data_ptr()); \
    const WEIGHT_C_T* d_trace = static_cast<const WEIGHT_C_T*>(trace.data_ptr()); \
    const SPIKE_C_T*  d_spk   = static_cast<const SPIKE_C_T*>(spike.data_ptr()); \
    int n_col_blocks = (n_post + 31) / 32;                                 \
    int n_row_blocks = (n_pre + ON_POST_ROW_TILE - 1) / ON_POST_ROW_TILE;  \
    dim3 grid(n_col_blocks, n_row_blocks);                                  \
    _on_post_tiled_kern##SUFFIX<<<grid, 256, 0, s>>>(                       \
        d_w, d_trace, d_spk, n_pre, n_post);                               \
}

// =========================================================================
// Instantiate TVM FFI entry points via macros + @tvm_ffi annotations
// =========================================================================

// ---- Float32 ----
// @tvm_ffi update_dense_on_post_f32_bool
FFI_ON_POST(_f32_bool,   float,          int8_t)
// @tvm_ffi update_dense_on_post_f32_float
FFI_ON_POST(_f32_float,  float,          float)

// ---- Float64 ----
// @tvm_ffi update_dense_on_post_f64_bool
FFI_ON_POST(_f64_bool,   double,         int8_t)
// @tvm_ffi update_dense_on_post_f64_float
FFI_ON_POST(_f64_float,  double,         float)

// ---- Float16 ----
// @tvm_ffi update_dense_on_post_f16_bool
FFI_ON_POST(_f16_bool,   __half,         int8_t)
// @tvm_ffi update_dense_on_post_f16_float
FFI_ON_POST(_f16_float,  __half,         float)

// ---- BFloat16 ----
// @tvm_ffi update_dense_on_post_bf16_bool
FFI_ON_POST(_bf16_bool,  __nv_bfloat16,  int8_t)
// @tvm_ffi update_dense_on_post_bf16_float
FFI_ON_POST(_bf16_float, __nv_bfloat16,  float)
