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
 * ==========================================================================
 *
 * Python API:
 *   brainevent.update_dense_on_binary_pre(weight, pre_spike, post_trace,
 *                                         *, backend='tvmffi')
 *
 * Operation (in-place):
 *   For each presynaptic neuron i where pre_spike[i] is active:
 *       weight[i, :] += post_trace
 *
 *   Equivalently:
 *       weight[i, j] += post_trace[j]   for all j, if pre_spike[i] active
 *
 * Parameters
 * ----------
 * weight    : dense float matrix [n_pre, n_post].  Updated in-place.
 * pre_spike : 1-D vector [n_pre].
 *             bool  (int8): active when != 0.
 *             float (f32):  active when != 0.0f.
 * post_trace: 1-D float vector [n_post], same dtype as weight.
 *
 * Kernel variants
 * ---------------
 * _on_pre_kern:
 *   grid  = (n_pre,)          — one block per presynaptic neuron
 *   block = (BLOCK_COL,)      — auto-selected: 32 if n_post <= 64, else 256
 *
 *   Each block first checks its spike; if inactive the entire block returns
 *   immediately with zero work.  If active, all threads in the block stride
 *   over the n_post weight columns with coalesced writes:
 *       thread tx writes columns  tx,  tx+BLOCK_COL,  tx+2*BLOCK_COL, ...
 *
 *   Memory access pattern (row-major weight):
 *     weight[i, j] = weight_ptr + i * n_post + j
 *     → consecutive threads j=0,1,...,BLOCK_COL-1 access consecutive memory
 *       addresses within row i → fully coalesced global reads and writes.
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
// Row-parallel on_pre kernel macro
//
// grid  = (n_pre,)  — one block per presynaptic neuron
// block = (BLOCK_COL,)   — threads tile over n_post columns
//
// Each block:
//   1. Reads pre_spike[row].  If inactive → immediate warp-coherent exit.
//   2. Strides over n_post columns: out_w[row, j] += trace[j]
//
// Memory layout (row-major):  out_w[row, j] = out_w_ptr + row * n_post + j
//   → threads tx=0..BLOCK_COL-1 access addresses  (row*n_post + 0), ...,
//     (row*n_post + BLOCK_COL-1): BLOCK_COL-wide contiguous → coalesced.
// =========================================================================

#define DEFINE_ON_PRE(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T,        \
                      READ_W, WRITE_W)                                      \
__global__ void _on_pre_kern##SUFFIX(                                       \
    WEIGHT_T*       __restrict__ out_w,                                     \
    const SPIKE_T*  __restrict__ spike,                                     \
    const WEIGHT_T* __restrict__ trace,                                     \
    int n_pre, int n_post                                                   \
) {                                                                         \
    int row = blockIdx.x;                                                   \
    if (row >= n_pre || !IS_ACTIVE(spike[row])) return;                    \
    WEIGHT_T* w_row = out_w + (size_t)row * n_post;                        \
    for (int j = threadIdx.x; j < n_post; j += blockDim.x) {              \
        ACC_T updated = READ_W(w_row[j]) + READ_W(trace[j]);               \
        w_row[j] = WRITE_W(updated);                                       \
    }                                                                       \
}

// =========================================================================
// Instantiate kernels: 4 weight dtypes x 2 spike types = 8 kernels
// =========================================================================

// ---- Float32 ----
DEFINE_ON_PRE(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float, float, READ_F32, WRITE_F32)
DEFINE_ON_PRE(_f32_float, float,  IS_ACTIVE_FLOAT, float, float, READ_F32, WRITE_F32)

// ---- Float64 ----
DEFINE_ON_PRE(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64, WRITE_F64)
DEFINE_ON_PRE(_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64, WRITE_F64)

// ---- Float16 (accumulate in float32 for numerical stability) ----
DEFINE_ON_PRE(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float, READ_F16, WRITE_F16)
DEFINE_ON_PRE(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float, READ_F16, WRITE_F16)

// ---- BFloat16 (accumulate in float32; requires CUDA 11.0+) ----
DEFINE_ON_PRE(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, WRITE_BF16)
DEFINE_ON_PRE(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, WRITE_BF16)


// =========================================================================
// TVM FFI Entry Point Macro
// =========================================================================
//
// Convention: args = (weight, spike, trace, out_weight, stream)
//   weight    [n_pre, n_post]  — GPU device ptr (same buffer as out_weight)
//   spike     [n_pre]          — GPU device ptr
//   trace     [n_post]         — GPU device ptr
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
// Block size selection:
//   n_post <= 64  → 32 threads  (warp-level, no loop overhead)
//   n_post > 64   → 256 threads (block-level, best for large n_post)
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
    /* out_weight.data_ptr() is aliased to weight.data_ptr() */            \
    WEIGHT_C_T*       d_w     = static_cast<WEIGHT_C_T*>(out_weight.data_ptr()); \
    const SPIKE_C_T*  d_spk   = static_cast<const SPIKE_C_T*>(spike.data_ptr()); \
    const WEIGHT_C_T* d_trace = static_cast<const WEIGHT_C_T*>(trace.data_ptr()); \
    int block_col = (n_post <= 64) ? 32 : 256;                             \
    _on_pre_kern##SUFFIX<<<n_pre, block_col, 0, s>>>(                       \
        d_w, d_spk, d_trace, n_pre, n_post);                               \
}

// =========================================================================
// Instantiate TVM FFI entry points via macros + @tvm_ffi annotations
// =========================================================================

// ---- Float32 ----
// @tvm_ffi update_dense_on_pre_f32_bool
FFI_ON_PRE(_f32_bool,   float,          int8_t)
// @tvm_ffi update_dense_on_pre_f32_float
FFI_ON_PRE(_f32_float,  float,          float)

// ---- Float64 ----
// @tvm_ffi update_dense_on_pre_f64_bool
FFI_ON_PRE(_f64_bool,   double,         int8_t)
// @tvm_ffi update_dense_on_pre_f64_float
FFI_ON_PRE(_f64_float,  double,         float)

// ---- Float16 ----
// @tvm_ffi update_dense_on_pre_f16_bool
FFI_ON_PRE(_f16_bool,   __half,         int8_t)
// @tvm_ffi update_dense_on_pre_f16_float
FFI_ON_PRE(_f16_float,  __half,         float)

// ---- BFloat16 ----
// @tvm_ffi update_dense_on_pre_bf16_bool
FFI_ON_PRE(_bf16_bool,  __nv_bfloat16,  int8_t)
// @tvm_ffi update_dense_on_pre_bf16_float
FFI_ON_PRE(_bf16_float, __nv_bfloat16,  float)
