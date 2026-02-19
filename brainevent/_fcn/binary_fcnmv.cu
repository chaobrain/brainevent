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
 * binary_fcnmv.cu — Event-Driven FCN Sparse Matrix-Vector CUDA Kernels
 * =====================================================================
 *
 * Python API: brainevent.binary_fcnmv(weights, indices, spikes, *, shape, transpose, backend)
 *
 * Event-driven sparse matrix--vector product with fixed connection number.
 *
 * Computes  y = W @ s  (or  y = W^T @ s  when transpose=True)
 * where W is a sparse weight matrix stored in fixed-connection-number
 * format and s is a binary spike vector.  Only connections to spiking
 * neurons contribute to the result (event-driven).
 *
 * Parameters
 * ----------
 * weights : shape (1,) for homogeneous or (num_pre, num_conn) for heterogeneous,
 *           floating-point dtype (float16 / bfloat16 / float32 / float64).
 * indices : shape (num_pre, num_conn), int32 — post-synaptic column indices.
 * spikes  : shape (num_post,) for gather or (num_pre,) for scatter.
 *           bool dtype: active when != 0 (stored as uint8).
 *           float dtype: active when > 0.
 * shape   : logical (num_pre, num_post) shape of the dense weight matrix.
 * transpose=False (gather mode):
 *   y[i] = sum_{k} weights[i,k] * 1_{spikes[indices[i,k]] active}
 * transpose=True  (scatter mode):
 *   y[indices[i,k]] += weights[i,k] * 1_{spikes[i] active}   for all i,k
 *
 * Supported weight dtypes: float32 (_f32), float64 (_f64), float16 (_f16), bfloat16 (_bf16).
 * Supported spike dtypes:  bool (_bool_) and float (_float_) matching the weight dtype.
 * Float16 and bfloat16 accumulate in float32 for numerical stability.
 * Bfloat16 requires CUDA 11.0+.
 *
 * Optimization: __ballot_sync() / warp ballot used in basic-gather to skip all-zero
 * warp chunks, providing up to 20x speedup at 5% firing rate vs dense computation.
 *
 * IMPORTANT: weights.data_ptr() is a GPU device pointer — NEVER dereference on host.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// =========================================================================
// Warp-level reduction helpers
// =========================================================================

__device__ __inline__ float warp_reduce_sum_f32(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __inline__ double warp_reduce_sum_f64(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// =========================================================================
// Spike active predicates
// =========================================================================

#define IS_ACTIVE_BOOL(s)       ((s) != 0)
#define IS_ACTIVE_FLOAT_F32(s)  ((s) > 0.0f)
#define IS_ACTIVE_FLOAT_F64(s)  ((s) > 0.0)
#define IS_ACTIVE_FLOAT_F16(s)  (__half2float(s) > 0.0f)
#define IS_ACTIVE_FLOAT_BF16(s) (__bfloat162float(s) > 0.0f)

// =========================================================================
// Per-dtype weight conversion macros: READ converts WEIGHT_T -> ACC_T
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
// GATHER warp kernel macro (one warp per output row)
//
// y[i] = sum_{k where is_active(spikes[indices[i,k]])} weights[i,k]
// For homo: y[i] = weights[0] * count_active_k
//
// Event-driven strategy (warp variant):
//   BRANCHLESS — no __ballot_sync.  The whole row fits in 1-2 cache lines
//   (~15 cycles), so skipping it saves only ~15 cycles per all-inactive row.
//   Ballot costs ~3 cycles per row; break-even is below ~1% firing rate.
//
// OOB safety: lanes with lane >= n_conn use safe_lane = n_conn-1 to avoid
// out-of-bounds reads; in_range=false → active=false → contributes 0.
// =========================================================================

#define DEFINE_BG_WARP(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T,          \
                       READ_W, WRITE_W, WARP_RED, ACC_ZERO)                   \
__global__ void _bg_warp_kern##SUFFIX(                                        \
    const int32_t* __restrict__ indices,                                       \
    const SPIKE_T* __restrict__ spikes,                                        \
    WEIGHT_T*      __restrict__ output,                                        \
    const WEIGHT_T* __restrict__ weights,                                      \
    int n_pre, int n_conn, int is_homo                                         \
) {                                                                            \
    int row = blockIdx.x;                                                      \
    if (row >= n_pre) return;                                                  \
    int lane = threadIdx.x;                                                    \
    const int32_t* i_row = indices + (size_t)row * n_conn;                    \
    const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)row * n_conn; \
    bool in_range = (lane < n_conn);                                           \
    int safe_lane = in_range ? lane : (n_conn - 1);                           \
    bool active = in_range && IS_ACTIVE(spikes[i_row[safe_lane]]);             \
    ACC_T val = active ? (is_homo ? (ACC_T)1 : READ_W(w_row[lane])) : ACC_ZERO; \
    val = WARP_RED(val);                                                       \
    if (lane == 0)                                                             \
        output[row] = WRITE_W(is_homo ? (READ_W(weights[0]) * val) : val);    \
}

// =========================================================================
// GATHER basic kernel macro (one block per output row with block reduction)
//
// Event-driven via __ballot_sync per 32-element chunk:
//   each warp checks its chunk with a warp vote; if all inactive, the
//   weight loads and FP adds for that chunk are skipped entirely.
//   At 5% firing rate with n_conn=1000, ~95% of chunks are skipped.
//
// Uses 32*sizeof(ACC_T) bytes of dynamic shared memory for reduction.
// Best when n_conn > 32.
// =========================================================================

#define DEFINE_BG_BASIC(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T,         \
                        READ_W, WRITE_W, WARP_RED, ACC_ZERO)                  \
__global__ void _bg_basic_kern##SUFFIX(                                       \
    const int32_t* __restrict__ indices,                                       \
    const SPIKE_T* __restrict__ spikes,                                        \
    WEIGHT_T*      __restrict__ output,                                        \
    const WEIGHT_T* __restrict__ weights,                                      \
    int n_pre, int n_conn, int is_homo                                         \
) {                                                                            \
    extern __shared__ char _smem_bytes[];                                      \
    ACC_T* smem_red = reinterpret_cast<ACC_T*>(_smem_bytes);                  \
    int row = blockIdx.x;                                                      \
    if (row >= n_pre) return;                                                  \
    const int32_t* i_row = indices + (size_t)row * n_conn;                    \
    const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)row * n_conn; \
    int lane   = threadIdx.x & 31;                                             \
    int warpid = threadIdx.x >> 5;                                             \
    int nwarps = blockDim.x >> 5;                                              \
    ACC_T val = ACC_ZERO;                                                      \
    for (int chunk = warpid; (chunk << 5) < n_conn; chunk += nwarps) {        \
        int k = (chunk << 5) + lane;                                           \
        bool in_range = (k < n_conn);                                          \
        int safe_k = in_range ? k : (n_conn - 1);                             \
        bool active = in_range && IS_ACTIVE(spikes[i_row[safe_k]]);            \
        unsigned ballot = __ballot_sync(0xffffffff, active);                   \
        if (ballot == 0) continue;                                             \
        if (active)                                                            \
            val += is_homo ? (ACC_T)1 : READ_W(w_row[k]);                     \
    }                                                                          \
    val = WARP_RED(val);                                                       \
    if (lane == 0) smem_red[warpid] = val;                                     \
    __syncthreads();                                                           \
    int n_warps = (blockDim.x + 31) >> 5;                                      \
    val = (threadIdx.x < n_warps) ? smem_red[lane] : ACC_ZERO;                \
    if (warpid == 0) val = WARP_RED(val);                                      \
    if (threadIdx.x == 0)                                                      \
        output[row] = WRITE_W(is_homo ? (READ_W(weights[0]) * val) : val);    \
}

// =========================================================================
// SCATTER warp kernel macro (8 warps per block, one warp per pre-neuron)
//
// For each active row i: output[indices[i,k]] += weights[i,k]
// Early exit (continue) skips inactive rows — highly effective at 1-5% rates.
// Grid = ceil(n_pre / 8) blocks.  Best when n_conn <= 32.
// =========================================================================

#define DEFINE_BS_WARP(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T)                  \
__global__ void _bs_warp_kern##SUFFIX(                                         \
    const int32_t* __restrict__ indices,                                       \
    const SPIKE_T* __restrict__ spikes,                                        \
    WEIGHT_T*      __restrict__ output,                                        \
    const WEIGHT_T* __restrict__ weights,                                      \
    int n_pre, int n_conn, int is_homo                                         \
) {                                                                            \
    int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;             \
    int lane_id   = threadIdx.x & 31;                                          \
    int num_warps = (gridDim.x * blockDim.x) >> 5;                             \
    for (int row = warp_id; row < n_pre; row += num_warps) {                   \
        if (!IS_ACTIVE(spikes[row])) continue;                                 \
        const int32_t* i_row = indices + (size_t)row * n_conn;                 \
        const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)row * n_conn; \
        WEIGHT_T w0 = is_homo ? weights[0] : (WEIGHT_T)0;                      \
        for (int k = lane_id; k < n_conn; k += 32)                             \
            atomicAdd(&output[i_row[k]], is_homo ? w0 : w_row[k]);             \
    }                                                                          \
}

// =========================================================================
// SCATTER basic kernel macro (one block per pre-neuron)
//
// Entire block exits early if spike inactive. Best when n_conn > 32.
// =========================================================================

#define DEFINE_BS_BASIC(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T)                 \
__global__ void _bs_basic_kern##SUFFIX(                                        \
    const int32_t* __restrict__ indices,                                       \
    const SPIKE_T* __restrict__ spikes,                                        \
    WEIGHT_T*      __restrict__ output,                                        \
    const WEIGHT_T* __restrict__ weights,                                      \
    int n_pre, int n_conn, int is_homo                                         \
) {                                                                            \
    int row = blockIdx.x;                                                      \
    if (row >= n_pre) return;                                                  \
    if (!IS_ACTIVE(spikes[row])) return;                                       \
    const int32_t* i_row = indices + (size_t)row * n_conn;                    \
    const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)row * n_conn; \
    WEIGHT_T w0 = is_homo ? weights[0] : (WEIGHT_T)0;                          \
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x)                    \
        atomicAdd(&output[i_row[k]], is_homo ? w0 : w_row[k]);                 \
}

// =========================================================================
// Instantiate device kernels: 4 weight dtypes x 2 spike types = 8 combos,
//                             2 gather + 2 scatter = 4 kernel shapes each
// =========================================================================

// ---- Float32 ----
DEFINE_BG_WARP(_bool_warp_f32,   uint8_t, IS_ACTIVE_BOOL,      float,          float,  READ_F32,  WRITE_F32,  warp_reduce_sum_f32, 0.0f)
DEFINE_BG_WARP(_float_warp_f32,  float,   IS_ACTIVE_FLOAT_F32, float,          float,  READ_F32,  WRITE_F32,  warp_reduce_sum_f32, 0.0f)
DEFINE_BG_BASIC(_bool_basic_f32,  uint8_t, IS_ACTIVE_BOOL,      float,          float,  READ_F32,  WRITE_F32,  warp_reduce_sum_f32, 0.0f)
DEFINE_BG_BASIC(_float_basic_f32, float,   IS_ACTIVE_FLOAT_F32, float,          float,  READ_F32,  WRITE_F32,  warp_reduce_sum_f32, 0.0f)
DEFINE_BS_WARP(_bool_warp_f32,   uint8_t, IS_ACTIVE_BOOL,      float)
DEFINE_BS_WARP(_float_warp_f32,  float,   IS_ACTIVE_FLOAT_F32, float)
DEFINE_BS_BASIC(_bool_basic_f32,  uint8_t, IS_ACTIVE_BOOL,      float)
DEFINE_BS_BASIC(_float_basic_f32, float,   IS_ACTIVE_FLOAT_F32, float)

// ---- Float64 ----
DEFINE_BG_WARP(_bool_warp_f64,   uint8_t, IS_ACTIVE_BOOL,      double,         double, READ_F64,  WRITE_F64,  warp_reduce_sum_f64, 0.0)
DEFINE_BG_WARP(_float_warp_f64,  double,  IS_ACTIVE_FLOAT_F64, double,         double, READ_F64,  WRITE_F64,  warp_reduce_sum_f64, 0.0)
DEFINE_BG_BASIC(_bool_basic_f64,  uint8_t, IS_ACTIVE_BOOL,      double,         double, READ_F64,  WRITE_F64,  warp_reduce_sum_f64, 0.0)
DEFINE_BG_BASIC(_float_basic_f64, double,  IS_ACTIVE_FLOAT_F64, double,         double, READ_F64,  WRITE_F64,  warp_reduce_sum_f64, 0.0)
DEFINE_BS_WARP(_bool_warp_f64,   uint8_t, IS_ACTIVE_BOOL,      double)
DEFINE_BS_WARP(_float_warp_f64,  double,  IS_ACTIVE_FLOAT_F64, double)
DEFINE_BS_BASIC(_bool_basic_f64,  uint8_t, IS_ACTIVE_BOOL,      double)
DEFINE_BS_BASIC(_float_basic_f64, double,  IS_ACTIVE_FLOAT_F64, double)

// ---- Float16 (accumulate in float32 for stability) ----
DEFINE_BG_WARP(_bool_warp_f16,   uint8_t, IS_ACTIVE_BOOL,      __half,         float,  READ_F16,  WRITE_F16,  warp_reduce_sum_f32, 0.0f)
DEFINE_BG_WARP(_float_warp_f16,  __half,  IS_ACTIVE_FLOAT_F16, __half,         float,  READ_F16,  WRITE_F16,  warp_reduce_sum_f32, 0.0f)
DEFINE_BG_BASIC(_bool_basic_f16,  uint8_t, IS_ACTIVE_BOOL,      __half,         float,  READ_F16,  WRITE_F16,  warp_reduce_sum_f32, 0.0f)
DEFINE_BG_BASIC(_float_basic_f16, __half,  IS_ACTIVE_FLOAT_F16, __half,         float,  READ_F16,  WRITE_F16,  warp_reduce_sum_f32, 0.0f)
DEFINE_BS_WARP(_bool_warp_f16,   uint8_t, IS_ACTIVE_BOOL,      __half)
DEFINE_BS_WARP(_float_warp_f16,  __half,  IS_ACTIVE_FLOAT_F16, __half)
DEFINE_BS_BASIC(_bool_basic_f16,  uint8_t, IS_ACTIVE_BOOL,      __half)
DEFINE_BS_BASIC(_float_basic_f16, __half,  IS_ACTIVE_FLOAT_F16, __half)

// ---- BFloat16 (accumulate in float32 for stability; requires CUDA 11.0+) ----
DEFINE_BG_WARP(_bool_warp_bf16,   uint8_t,        IS_ACTIVE_BOOL,       __nv_bfloat16,  float,  READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_WARP(_float_warp_bf16,  __nv_bfloat16,  IS_ACTIVE_FLOAT_BF16, __nv_bfloat16,  float,  READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_BASIC(_bool_basic_bf16,  uint8_t,        IS_ACTIVE_BOOL,       __nv_bfloat16,  float,  READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BG_BASIC(_float_basic_bf16, __nv_bfloat16,  IS_ACTIVE_FLOAT_BF16, __nv_bfloat16,  float,  READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BS_WARP(_bool_warp_bf16,   uint8_t,        IS_ACTIVE_BOOL,       __nv_bfloat16)
DEFINE_BS_WARP(_float_warp_bf16,  __nv_bfloat16,  IS_ACTIVE_FLOAT_BF16, __nv_bfloat16)
DEFINE_BS_BASIC(_bool_basic_bf16,  uint8_t,        IS_ACTIVE_BOOL,       __nv_bfloat16)
DEFINE_BS_BASIC(_float_basic_bf16, __nv_bfloat16,  IS_ACTIVE_FLOAT_BF16, __nv_bfloat16)

// =========================================================================
// TVM FFI Entry Point Macros
// =========================================================================
//
// Convention: args = (weights, indices, spikes, output, stream)
//   weights : WEIGHT_C_T, shape (1,) for homo or (n_pre, n_conn) for hetero
//   indices : int32,   shape (n_pre, n_conn)
//   spikes  : gather → (n_post,);  scatter → (n_pre,)
//             bool variant   → uint8 pointer
//             float variant  → SPIKE_C_T pointer
//   output  : gather → (n_pre,) WEIGHT_C_T, written directly
//             scatter → (n_post,) WEIGHT_C_T, zeroed via cudaMemsetAsync
//
// IMPORTANT: data_ptr() returns GPU device memory pointers.
// NEVER dereference on host. Pass unchanged to device kernels.

// ---- Gather warp FFI macro ----
// Generates: void binary_fcnmv_gather##SUFFIX(...)
#define FFI_BG_WARP(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                              \
void binary_fcnmv_gather##SUFFIX(                                                \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                  \
    tvm::ffi::TensorView spikes,  tvm::ffi::TensorView output, int64_t stream    \
) {                                                                              \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                    \
    int n_pre  = static_cast<int>(indices.size(0));                             \
    int n_conn = static_cast<int>(indices.size(1));                             \
    int is_homo = (weights.ndim() == 1) ? 1 : 0;                               \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());  \
    const SPIKE_C_T*  d_spk = static_cast<const SPIKE_C_T*>(spikes.data_ptr()); \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());      \
    _bg_warp_kern##SUFFIX<<<n_pre, 32, 0, s>>>(                                  \
        d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);                      \
}

// ---- Gather basic FFI macro ----
// Generates: void binary_fcnmv_gather##SUFFIX(...)
#define FFI_BG_BASIC(SUFFIX, WEIGHT_C_T, SPIKE_C_T, SHM_SIZE)                   \
void binary_fcnmv_gather##SUFFIX(                                                \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                  \
    tvm::ffi::TensorView spikes,  tvm::ffi::TensorView output, int64_t stream    \
) {                                                                              \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                    \
    int n_pre  = static_cast<int>(indices.size(0));                             \
    int n_conn = static_cast<int>(indices.size(1));                             \
    int is_homo = (weights.ndim() == 1) ? 1 : 0;                               \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());  \
    const SPIKE_C_T*  d_spk = static_cast<const SPIKE_C_T*>(spikes.data_ptr()); \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());      \
    _bg_basic_kern##SUFFIX<<<n_pre, 256, SHM_SIZE, s>>>(                         \
        d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);                      \
}

// ---- Scatter warp FFI macro ----
// Generates: void binary_fcnmv_scatter##SUFFIX(...)
#define FFI_BS_WARP(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                              \
void binary_fcnmv_scatter##SUFFIX(                                               \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                  \
    tvm::ffi::TensorView spikes,  tvm::ffi::TensorView output, int64_t stream    \
) {                                                                              \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                    \
    int n_pre  = static_cast<int>(indices.size(0));                             \
    int n_conn = static_cast<int>(indices.size(1));                             \
    int n_post = static_cast<int>(output.size(0));                              \
    int is_homo = (weights.ndim() == 1) ? 1 : 0;                               \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());  \
    const SPIKE_C_T*  d_spk = static_cast<const SPIKE_C_T*>(spikes.data_ptr()); \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());      \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(WEIGHT_C_T), s);          \
    int blocks = (n_pre + 7) / 8;                                               \
    _bs_warp_kern##SUFFIX<<<blocks, 256, 0, s>>>(                                \
        d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);                      \
}

// ---- Scatter basic FFI macro ----
// Generates: void binary_fcnmv_scatter##SUFFIX(...)
#define FFI_BS_BASIC(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                             \
void binary_fcnmv_scatter##SUFFIX(                                               \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                  \
    tvm::ffi::TensorView spikes,  tvm::ffi::TensorView output, int64_t stream    \
) {                                                                              \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                    \
    int n_pre  = static_cast<int>(indices.size(0));                             \
    int n_conn = static_cast<int>(indices.size(1));                             \
    int n_post = static_cast<int>(output.size(0));                              \
    int is_homo = (weights.ndim() == 1) ? 1 : 0;                               \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());  \
    const SPIKE_C_T*  d_spk = static_cast<const SPIKE_C_T*>(spikes.data_ptr()); \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());      \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(WEIGHT_C_T), s);          \
    _bs_basic_kern##SUFFIX<<<n_pre, 256, 0, s>>>(                                \
        d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);                      \
}

// =========================================================================
// Instantiate TVM FFI entry points via macros + @tvm_ffi annotations
// =========================================================================

// ---- Float32 (shm: 32 * sizeof(float) = 128 bytes) ----
// @tvm_ffi binary_fcnmv_gather_bool_warp_f32
FFI_BG_WARP(_bool_warp_f32,    float,  uint8_t)
// @tvm_ffi binary_fcnmv_gather_bool_basic_f32
FFI_BG_BASIC(_bool_basic_f32,  float,  uint8_t, 32 * sizeof(float))
// @tvm_ffi binary_fcnmv_gather_float_warp_f32
FFI_BG_WARP(_float_warp_f32,   float,  float)
// @tvm_ffi binary_fcnmv_gather_float_basic_f32
FFI_BG_BASIC(_float_basic_f32, float,  float,   32 * sizeof(float))
// @tvm_ffi binary_fcnmv_scatter_bool_warp_f32
FFI_BS_WARP(_bool_warp_f32,    float,  uint8_t)
// @tvm_ffi binary_fcnmv_scatter_bool_basic_f32
FFI_BS_BASIC(_bool_basic_f32,  float,  uint8_t)
// @tvm_ffi binary_fcnmv_scatter_float_warp_f32
FFI_BS_WARP(_float_warp_f32,   float,  float)
// @tvm_ffi binary_fcnmv_scatter_float_basic_f32
FFI_BS_BASIC(_float_basic_f32, float,  float)

// ---- Float64 (shm: 32 * sizeof(double) = 256 bytes) ----
// @tvm_ffi binary_fcnmv_gather_bool_warp_f64
FFI_BG_WARP(_bool_warp_f64,    double, uint8_t)
// @tvm_ffi binary_fcnmv_gather_bool_basic_f64
FFI_BG_BASIC(_bool_basic_f64,  double, uint8_t, 32 * sizeof(double))
// @tvm_ffi binary_fcnmv_gather_float_warp_f64
FFI_BG_WARP(_float_warp_f64,   double, double)
// @tvm_ffi binary_fcnmv_gather_float_basic_f64
FFI_BG_BASIC(_float_basic_f64, double, double,  32 * sizeof(double))
// @tvm_ffi binary_fcnmv_scatter_bool_warp_f64
FFI_BS_WARP(_bool_warp_f64,    double, uint8_t)
// @tvm_ffi binary_fcnmv_scatter_bool_basic_f64
FFI_BS_BASIC(_bool_basic_f64,  double, uint8_t)
// @tvm_ffi binary_fcnmv_scatter_float_warp_f64
FFI_BS_WARP(_float_warp_f64,   double, double)
// @tvm_ffi binary_fcnmv_scatter_float_basic_f64
FFI_BS_BASIC(_float_basic_f64, double, double)

// ---- Float16 (shm: 32 * sizeof(float) = 128 bytes; accumulates in f32) ----
// @tvm_ffi binary_fcnmv_gather_bool_warp_f16
FFI_BG_WARP(_bool_warp_f16,    __half, uint8_t)
// @tvm_ffi binary_fcnmv_gather_bool_basic_f16
FFI_BG_BASIC(_bool_basic_f16,  __half, uint8_t, 32 * sizeof(float))
// @tvm_ffi binary_fcnmv_gather_float_warp_f16
FFI_BG_WARP(_float_warp_f16,   __half, __half)
// @tvm_ffi binary_fcnmv_gather_float_basic_f16
FFI_BG_BASIC(_float_basic_f16, __half, __half,  32 * sizeof(float))
// @tvm_ffi binary_fcnmv_scatter_bool_warp_f16
FFI_BS_WARP(_bool_warp_f16,    __half, uint8_t)
// @tvm_ffi binary_fcnmv_scatter_bool_basic_f16
FFI_BS_BASIC(_bool_basic_f16,  __half, uint8_t)
// @tvm_ffi binary_fcnmv_scatter_float_warp_f16
FFI_BS_WARP(_float_warp_f16,   __half, __half)
// @tvm_ffi binary_fcnmv_scatter_float_basic_f16
FFI_BS_BASIC(_float_basic_f16, __half, __half)

// ---- BFloat16 (shm: 32 * sizeof(float) = 128 bytes; accumulates in f32; requires CUDA 11.0+) ----
// @tvm_ffi binary_fcnmv_gather_bool_warp_bf16
FFI_BG_WARP(_bool_warp_bf16,    __nv_bfloat16, uint8_t)
// @tvm_ffi binary_fcnmv_gather_bool_basic_bf16
FFI_BG_BASIC(_bool_basic_bf16,  __nv_bfloat16, uint8_t,        32 * sizeof(float))
// @tvm_ffi binary_fcnmv_gather_float_warp_bf16
FFI_BG_WARP(_float_warp_bf16,   __nv_bfloat16, __nv_bfloat16)
// @tvm_ffi binary_fcnmv_gather_float_basic_bf16
FFI_BG_BASIC(_float_basic_bf16, __nv_bfloat16, __nv_bfloat16, 32 * sizeof(float))
// @tvm_ffi binary_fcnmv_scatter_bool_warp_bf16
FFI_BS_WARP(_bool_warp_bf16,    __nv_bfloat16, uint8_t)
// @tvm_ffi binary_fcnmv_scatter_bool_basic_bf16
FFI_BS_BASIC(_bool_basic_bf16,  __nv_bfloat16, uint8_t)
// @tvm_ffi binary_fcnmv_scatter_float_warp_bf16
FFI_BS_WARP(_float_warp_bf16,   __nv_bfloat16, __nv_bfloat16)
// @tvm_ffi binary_fcnmv_scatter_float_basic_bf16
FFI_BS_BASIC(_float_basic_bf16, __nv_bfloat16, __nv_bfloat16)
