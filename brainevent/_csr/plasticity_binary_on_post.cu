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
 * plasticity_binary_on_post.cu -- CSR Post-Synaptic Plasticity Update CUDA Kernels
 * ===================================================================================
 *
 * Python API:
 *   brainevent.update_csr_on_binary_post(
 *       weight, indices, indptr, weight_indices, pre_trace, post_spike,
 *       shape=(n_pre, n_post), backend='tvmffi')
 *
 * Operation (in-place, CSC-indexed into CSR weight array):
 *   For each post-synaptic neuron j where post_spike[j] is active:
 *       for pos in range(indptr[j], indptr[j+1]):
 *           weight[weight_indices[pos]] += pre_trace[indices[pos]]
 *
 * This implements the postsynaptic STDP component: when a post-neuron fires,
 * all its incoming synaptic weights (located via weight_indices in the CSR
 * weight array) are updated by the presynaptic eligibility traces.
 *
 * Data layout note:
 *   - indices[pos]        : presynaptic neuron id for CSC entry pos
 *   - indptr[j..j+1]      : range of CSC entries for postsynaptic neuron j
 *   - weight_indices[pos] : corresponding position in the CSR weight array
 *
 * Correctness note on concurrent writes:
 *   In a valid CSC decomposition of a CSR matrix, weight_indices is an
 *   injective mapping (each CSR weight position appears at most once).
 *   Therefore no two active postsynaptic neurons write to the same weight
 *   position, and concurrent parallel writes are race-free.
 *   atomicAdd is used as a safety guard for malformed inputs.
 *
 * Kernel Variants (auto-dispatched by avg_nnz_per_col = nse / n_post):
 * --------------------------------------------------------------------
 *   _thread_kern:  1 thread per column (post-neuron), 256 threads/block.
 *                  Warp-ballot early exit: skip warp if all 32 cols inactive.
 *                  Best when avg_nnz < 32 (very sparse connectivity).
 *
 *   _warp_kern:    1 warp (32 threads) per column, 256 threads/block = 8 warps.
 *                  Lane-stride-32 iteration with atomicAdd scatter writes.
 *                  Best when 32 <= avg_nnz < 256 (medium connectivity).
 *
 *   _block_kern:   1 block (256 threads) per column.
 *                  All 256 threads cooperate, stride-256 iteration.
 *                  Best when avg_nnz >= 256 (dense connectivity).
 *
 *   _auto:         Host-side dispatch based on avg_nnz. Recommended entry.
 *
 * Parameters (TVM FFI tensors):
 *   weight        [nse]      float* (input, aliased as output)
 *   indices       [nse]      int32_t* (pre-neuron ids in CSC format)
 *   indptr        [n_post+1] int32_t* (column pointers, CSC format)
 *   weight_indices[nse]      int32_t* (CSR weight positions for each CSC entry)
 *   trace         [n_pre]    float*   (presynaptic eligibility traces)
 *   spike         [n_post]   spike_t* (bool as int8, or float)
 *   out_weight    [nse]      float*   (output buffer, aliased to weight)
 *
 * Memory access pattern:
 *   - spike[col]: broadcast read per column (small, L2 cached)
 *   - indptr[col], indptr[col+1]: two sequential reads (cached)
 *   - indices[pos]: sequential reads within a column (coalesced for warp/block)
 *   - weight_indices[pos]: sequential reads, scattered write targets
 *   - trace[indices[pos]]: scattered reads (trace[] is small, L2 cached)
 *   - out_w[weight_indices[pos]]: scattered writes (injective by construction)
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
// Atomic-add helpers for mixed-precision dtypes
//
// CUDA's atomicAdd natively supports float, double, unsigned int,
// unsigned long long. For __half (fp16, sm_70+) and __nv_bfloat16
// (bf16, sm_80+), we use the native CUDA atomicAdd overloads when
// available, or a CAS loop fallback for older compute capabilities.
// For fp16/bf16 the accumulator type is float32, so we store back
// via WRITE_F16/WRITE_BF16.
// =========================================================================

// float and double: direct atomicAdd
__device__ __forceinline__ void
atomic_add_f32(float* addr, float val) {
    atomicAdd(addr, val);
}
__device__ __forceinline__ void
atomic_add_f64(double* addr, double val) {
    atomicAdd(addr, val);
}

// __half: read-modify via float32 accumulator, atomicAdd on the float32
// result, then write back as __half using CAS loop for atomicity.
__device__ __forceinline__ void
atomic_add_f16(__half* addr, float delta) {
    // CAS-based atomic update for fp16
    unsigned int* addr32 = reinterpret_cast<unsigned int*>(
        reinterpret_cast<uintptr_t>(addr) & ~3u);
    unsigned int old32 = *addr32;
    unsigned int assumed;
    do {
        assumed = old32;
        unsigned int shift = (reinterpret_cast<uintptr_t>(addr) & 2u) ? 16u : 0u;
        __half old_half = __ushort_as_half(static_cast<unsigned short>(
            (assumed >> shift) & 0xFFFFu));
        float new_f = __half2float(old_half) + delta;
        unsigned short new_ush = __half_as_ushort(__float2half(new_f));
        unsigned int new32 = (assumed & ~(0xFFFFu << shift))
                             | (static_cast<unsigned int>(new_ush) << shift);
        old32 = atomicCAS(addr32, assumed, new32);
    } while (old32 != assumed);
}

// __nv_bfloat16: same CAS approach
__device__ __forceinline__ void
atomic_add_bf16(__nv_bfloat16* addr, float delta) {
    unsigned int* addr32 = reinterpret_cast<unsigned int*>(
        reinterpret_cast<uintptr_t>(addr) & ~3u);
    unsigned int old32 = *addr32;
    unsigned int assumed;
    do {
        assumed = old32;
        unsigned int shift = (reinterpret_cast<uintptr_t>(addr) & 2u) ? 16u : 0u;
        __nv_bfloat16 old_bf = __ushort_as_bfloat16(static_cast<unsigned short>(
            (assumed >> shift) & 0xFFFFu));
        float new_f = __bfloat162float(old_bf) + delta;
        unsigned short new_ush = __bfloat16_as_ushort(__float2bfloat16(new_f));
        unsigned int new32 = (assumed & ~(0xFFFFu << shift))
                             | (static_cast<unsigned int>(new_ush) << shift);
        old32 = atomicCAS(addr32, assumed, new32);
    } while (old32 != assumed);
}

// =========================================================================
// Variant 1: Thread-per-column kernel
//
// One thread handles one column (postsynaptic neuron).
//
// Warp-ballot early exit:
//   32 threads each check a different post-neuron's spike. If the warp-wide
//   ballot is zero (all inactive), the entire warp exits early, saving the
//   CSC range load and the scatter-write loop.
//
// Scatter writes:
//   out_w[weight_indices[pos]] += trace[indices[pos]]
//   Since weight_indices is injective, no race conditions occur.
//   atomicAdd used for safety.
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

// =========================================================================
// Variant 2: Warp-per-column kernel
//
// One warp (32 threads) cooperates on one postsynaptic neuron.
// All 32 threads in the warp check the same spike[col].
// Lane `lane` accesses pos = start + lane, start + lane + 32, ...
//
// The reads of indices[] and weight_indices[] are coalesced (consecutive
// positions within one column accessed by lane 0..31 simultaneously).
// The scatter writes to out_w[weight_indices[...]] are NOT coalesced
// (scattered by construction) but are L2-friendly for small weight arrays.
// =========================================================================

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

// =========================================================================
// Variant 3: Block-per-column kernel
//
// One block (256 threads) cooperates on one postsynaptic neuron.
// Thread tid accesses pos = start + tid, start + tid + 256, ...
// =========================================================================

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

// ---- Instantiate all three variants for all dtype combinations ----

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
// TVM FFI Entry Points (_auto: host-side dispatch to thread/warp/block)
//
// Signature: (weight, indices, indptr, weight_indices, trace, spike, out_weight, stream)
//
// Auto-dispatch thresholds (avg_nnz = nse / n_post):
//   avg_nnz <  32   -> thread_kern (256 cols per block, warp-ballot exit)
//   avg_nnz <  256  -> warp_kern   (8 cols per block, stride-32)
//   avg_nnz >= 256  -> block_kern  (1 col per block, stride-256)
// =========================================================================

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
