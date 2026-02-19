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
 *           floating-point dtype (float16 / float32 / float64).
 * indices : shape (num_pre, num_conn), int32 — post-synaptic column indices.
 * spikes  : shape (num_post,) for gather or (num_pre,) for scatter.
 *           bool dtype: active when True.
 *           float dtype: active when > 0.
 * shape   : logical (num_pre, num_post) shape of the dense weight matrix.
 * transpose=False (gather mode):
 *   y[i] = sum_{k} weights[i,k] * 1_{spikes[indices[i,k]] active}
 * transpose=True  (scatter mode):
 *   y[indices[i,k]] += weights[i,k] * 1_{spikes[i] active}   for all i,k
 *
 * Supported weight dtypes: float32 (default), float64 (_f64 suffix), float16 (_f16 suffix).
 * Supported spike dtypes: bool (_bool_ kernels) and float (_float_ kernels).
 * Optimization: __ballot_sync() / warp ballot used to skip all-zero warp chunks,
 * providing up to 20x speedup at 5% firing rate vs dense computation.
 *
 * IMPORTANT: weights.data_ptr() is a GPU device pointer — NEVER dereference on host.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// =========================================================================
// Warp reduction helper
// =========================================================================

__device__ __inline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// =========================================================================
// GATHER kernels  (transpose=False)
//
// y[i] = sum_{k=0}^{n_conn-1} weights[i,k] * is_active(spikes[indices[i,k]])
//
//   is_active(s) for bool spikes (uint8):  s != 0
//   is_active(s) for float spikes:         s > 0.0f
//
// For homogeneous weights (is_homo=1):
//   y[i] = weights[0] * count_k( is_active(spikes[indices[i,k]]) )
//
// Event-driven strategy:
//
//   Warp variant  (n_conn <= 32): BRANCHLESS — no __ballot_sync.
//     The whole row fits in 1-2 cache lines (~15 cycles of work).  A ballot
//     costs ~3 cycles on every row to save ~15 cycles on all-inactive rows;
//     at typical SNN rates (5-10 %) only ~3-19 % of rows are all-inactive,
//     so the expected gain is ≈ 0 cycles.  Ballot only wins below ~1 % firing
//     rate, which is outside the normal operating range.
//
//   Basic variant (n_conn >  32): __ballot_sync PER 32-ELEMENT CHUNK.
//     Each skipped chunk avoids loading a 128-byte weight tile.  At 5 %
//     firing rate with n_conn=1000 (~31 chunks/row), ~95 % of chunks are
//     all-inactive → substantial memory traffic reduction.
//
// OOB safety: lanes past n_conn use safe_lane/safe_k = (n_conn-1) to avoid
// out-of-bounds reads; those lanes have in_range=false so they contribute 0.
// =========================================================================

// ---- Gather / Bool spikes ----

// One warp (32 threads) per output row.  Branchless formulation: inactive
// lanes contribute 0 rather than using __ballot_sync for an early exit.
//
// Why no ballot here?
//   The entire row (n_conn ≤ 32) fits in one or two cache lines, so skipping
//   it entirely saves only ~15 cycles per all-inactive row.  The ballot
//   instruction itself costs ~3 cycles on every row, making the net gain
//   negligible at typical SNN firing rates (5–10 %) and only meaningful
//   below ~1 %.  The simpler branchless path is faster in practice.
//
// OOB safety: lanes with lane >= n_conn use safe_lane = n_conn-1 to avoid
// out-of-bounds reads; those lanes have in_range=false so active=false and
// they contribute 0 to the reduction.
__global__ void _bg_bool_warp_kern(
    const int32_t* __restrict__ indices,   // [n_pre, n_conn]
    const uint8_t* __restrict__ spikes,    // [n_post], bool stored as uint8
    float*         __restrict__ output,    // [n_pre]
    const float*   __restrict__ weights,   // [1] or [n_pre, n_conn]
    int n_pre, int n_conn, int is_homo
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;
    int lane = threadIdx.x;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    bool in_range = (lane < n_conn);
    int safe_lane = in_range ? lane : (n_conn - 1);
    bool active = in_range && (spikes[i_row[safe_lane]] != 0);
    float val = active ? (is_homo ? 1.0f : w_row[lane]) : 0.0f;
    val = warp_reduce_sum(val);
    if (lane == 0)
        output[row] = is_homo ? (weights[0] * val) : val;
}

// One block (256 threads = 8 warps) per output row with inline block reduction.
// Event-driven via __ballot_sync per 32-element chunk of connections:
//   each warp checks its chunk with a warp vote; if the ballot is zero the
//   weight loads and FP adds for that chunk are skipped entirely.
// Uses 32*sizeof(float) of dynamic shared memory for the reduction scratchpad.
// Best when n_conn > 32.
__global__ void _bg_bool_basic_kern(
    const int32_t* __restrict__ indices,   // [n_pre, n_conn]
    const uint8_t* __restrict__ spikes,    // [n_post]
    float*         __restrict__ output,    // [n_pre]
    const float*   __restrict__ weights,   // [1] or [n_pre, n_conn]
    int n_pre, int n_conn, int is_homo
) {
    extern __shared__ float smem_red[];   // 32 floats for block reduction
    int row = blockIdx.x;
    if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;

    int lane   = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;   // 8 for blockDim.x=256

    float val = 0.0f;
    // Each warp processes its assigned 32-element chunk(s) of connections
    for (int chunk = warpid; (chunk << 5) < n_conn; chunk += nwarps) {
        int k = (chunk << 5) + lane;
        bool in_range = (k < n_conn);
        int safe_k = in_range ? k : (n_conn - 1);   // OOB-safe index
        bool active = in_range && (spikes[i_row[safe_k]] != 0);

        // Chunk-level vote: skip entirely if no spike is active
        unsigned ballot = __ballot_sync(0xffffffff, active);
        if (ballot == 0) continue;   // no weight loads, no FP adds

        if (active)
            val += is_homo ? 1.0f : w_row[k];
    }

    // Inline block reduction via dynamic shared memory
    val = warp_reduce_sum(val);
    if (lane == 0) smem_red[warpid] = val;
    __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < n_warps) ? smem_red[lane] : 0.0f;
    if (warpid == 0) val = warp_reduce_sum(val);
    if (threadIdx.x == 0)
        output[row] = is_homo ? (weights[0] * val) : val;
}

// ---- Gather / Float spikes ----

// One warp (32 threads) per output row.  Branchless — no __ballot_sync.
// Same rationale as the bool variant above: the ballot overhead per row
// exceeds the savings from skipping all-inactive rows at typical firing rates.
__global__ void _bg_float_warp_kern(
    const int32_t* __restrict__ indices,   // [n_pre, n_conn]
    const float*   __restrict__ spikes,    // [n_post], float spikes
    float*         __restrict__ output,    // [n_pre]
    const float*   __restrict__ weights,   // [1] or [n_pre, n_conn]
    int n_pre, int n_conn, int is_homo
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;
    int lane = threadIdx.x;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    bool in_range = (lane < n_conn);
    int safe_lane = in_range ? lane : (n_conn - 1);
    bool active = in_range && (spikes[i_row[safe_lane]] > 0.0f);
    float val = active ? (is_homo ? 1.0f : w_row[lane]) : 0.0f;
    val = warp_reduce_sum(val);
    if (lane == 0)
        output[row] = is_homo ? (weights[0] * val) : val;
}

// One block (256 threads) per output row with inline block reduction.
// Event-driven via __ballot_sync per 32-element chunk.
// Best when n_conn > 32.
__global__ void _bg_float_basic_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ spikes,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    extern __shared__ float smem_red[];
    int row = blockIdx.x;
    if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;

    int lane   = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;

    float val = 0.0f;
    for (int chunk = warpid; (chunk << 5) < n_conn; chunk += nwarps) {
        int k = (chunk << 5) + lane;
        bool in_range = (k < n_conn);
        int safe_k = in_range ? k : (n_conn - 1);
        bool active = in_range && (spikes[i_row[safe_k]] > 0.0f);

        unsigned ballot = __ballot_sync(0xffffffff, active);
        if (ballot == 0) continue;

        if (active)
            val += is_homo ? 1.0f : w_row[k];
    }

    val = warp_reduce_sum(val);
    if (lane == 0) smem_red[warpid] = val;
    __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < n_warps) ? smem_red[lane] : 0.0f;
    if (warpid == 0) val = warp_reduce_sum(val);
    if (threadIdx.x == 0)
        output[row] = is_homo ? (weights[0] * val) : val;
}

// =========================================================================
// SCATTER kernels  (transpose=True)
//
// For each active row i (is_active(spikes[i]) == true):
//   output[indices[i, k]] += weights[i, k]   for k = 0 .. n_conn-1
//
// Output buffer must be pre-zeroed; this is done via cudaMemsetAsync
// in the TVM FFI entry functions below.
//
// Key optimisation: early exit (return / continue) when a pre-neuron is
// inactive.  With typical SNN firing rates of 1-5 %, this skips 95-99 %
// of all blocks/warps entirely.
// =========================================================================

// ---- Scatter / Bool spikes ----

// 8 warps per block (256 threads), one warp per pre-neuron.
// Grid = ceil(n_pre / 8) blocks.
// All 32 threads in a warp handle the same row — no intra-warp divergence
// on the "skip if inactive" check.
// Best when n_conn <= 32.
__global__ void _bs_bool_warp_kern(
    const int32_t* __restrict__ indices,   // [n_pre, n_conn]
    const uint8_t* __restrict__ spikes,    // [n_pre]
    float*         __restrict__ output,    // [n_post]
    const float*   __restrict__ weights,   // [1] or [n_pre, n_conn]
    int n_pre, int n_conn, int is_homo
) {
    int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane_id   = threadIdx.x & 31;
    int num_warps = (gridDim.x * blockDim.x) >> 5;
    for (int row = warp_id; row < n_pre; row += num_warps) {
        if (!spikes[row]) continue;   // all 32 threads in warp take same branch
        const int32_t* i_row = indices + (size_t)row * n_conn;
        const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
        float w0 = is_homo ? weights[0] : 0.0f;
        for (int k = lane_id; k < n_conn; k += 32)
            atomicAdd(&output[i_row[k]], is_homo ? w0 : w_row[k]);
    }
}

// One block (256 threads) per pre-neuron.  Entire block exits early
// if the neuron is inactive.
// Best when n_conn > 32.
__global__ void _bs_bool_basic_kern(
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ spikes,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;
    if (!spikes[row]) return;   // skip entire block if neuron is inactive
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float w0 = is_homo ? weights[0] : 0.0f;
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x)
        atomicAdd(&output[i_row[k]], is_homo ? w0 : w_row[k]);
}

// ---- Scatter / Float spikes ----

// 8 warps per block, one warp per pre-neuron.  Best when n_conn <= 32.
__global__ void _bs_float_warp_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ spikes,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane_id   = threadIdx.x & 31;
    int num_warps = (gridDim.x * blockDim.x) >> 5;
    for (int row = warp_id; row < n_pre; row += num_warps) {
        if (!(spikes[row] > 0.0f)) continue;
        const int32_t* i_row = indices + (size_t)row * n_conn;
        const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
        float w0 = is_homo ? weights[0] : 0.0f;
        for (int k = lane_id; k < n_conn; k += 32)
            atomicAdd(&output[i_row[k]], is_homo ? w0 : w_row[k]);
    }
}

// One block per pre-neuron.  Best when n_conn > 32.
__global__ void _bs_float_basic_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ spikes,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;
    if (!(spikes[row] > 0.0f)) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float w0 = is_homo ? weights[0] : 0.0f;
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x)
        atomicAdd(&output[i_row[k]], is_homo ? w0 : w_row[k]);
}

// =========================================================================
// TVM FFI Entry Points
//
// Convention: args = (weights, indices, spikes, output, stream)
//   weights : float32, shape (1,) for homo or (n_pre, n_conn) for hetero
//   indices : int32,   shape (n_pre, n_conn)
//   spikes  : gather → (n_post,);  scatter → (n_pre,)
//             bool variant   → uint8 pointer
//             float variant  → float32 pointer
//   output  : gather → (n_pre,) float32, written directly (no pre-zero)
//             scatter → (n_post,) float32, zeroed here via cudaMemsetAsync
//
// IMPORTANT: data_ptr() returns a GPU device memory pointer.
// NEVER dereference it in host C++ code (causes SIGSEGV).
// Pass it unchanged to device kernels; GPU threads read from it.
// Only ndim() and size() are host-safe metadata reads.
// =========================================================================

// ---- Gather / Bool ----

void binary_fcnmv_gather_bool_warp(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;   // host-safe metadata
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());   // device ptr
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const uint8_t* d_spk = static_cast<const uint8_t*>(spikes.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    _bg_bool_warp_kern<<<n_pre, 32, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

void binary_fcnmv_gather_bool_basic(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const uint8_t* d_spk = static_cast<const uint8_t*>(spikes.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    size_t shm = 32 * sizeof(float);
    _bg_bool_basic_kern<<<n_pre, 256, shm, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

// ---- Gather / Float ----

void binary_fcnmv_gather_float_warp(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_spk = static_cast<const float*>(spikes.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    _bg_float_warp_kern<<<n_pre, 32, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

void binary_fcnmv_gather_float_basic(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_spk = static_cast<const float*>(spikes.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    size_t shm = 32 * sizeof(float);
    _bg_float_basic_kern<<<n_pre, 256, shm, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

// ---- Scatter / Bool (output pre-zeroed via cudaMemsetAsync) ----

void binary_fcnmv_scatter_bool_warp(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const uint8_t* d_spk = static_cast<const uint8_t*>(spikes.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(float), s);
    int blocks = (n_pre + 7) / 8;
    _bs_bool_warp_kern<<<blocks, 256, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

void binary_fcnmv_scatter_bool_basic(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const uint8_t* d_spk = static_cast<const uint8_t*>(spikes.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(float), s);
    _bs_bool_basic_kern<<<n_pre, 256, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

// ---- Scatter / Float ----

void binary_fcnmv_scatter_float_warp(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_spk = static_cast<const float*>(spikes.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(float), s);
    int blocks = (n_pre + 7) / 8;
    _bs_float_warp_kern<<<blocks, 256, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

void binary_fcnmv_scatter_float_basic(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_spk = static_cast<const float*>(spikes.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(float), s);
    _bs_float_basic_kern<<<n_pre, 256, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

// =========================================================================
// float64 (double) variants for binary_fcnmv
// =========================================================================

__device__ __inline__ double bfcnmv_warp_reduce_sum_f64(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void _bg_bool_warp_kern_f64(
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ spikes,
    double*        __restrict__ output,
    const double*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;
    int lane = threadIdx.x;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const double*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    bool in_range = (lane < n_conn);
    int safe_lane = in_range ? lane : (n_conn - 1);
    bool active = in_range && (spikes[i_row[safe_lane]] != 0);
    double val = active ? (is_homo ? 1.0 : w_row[lane]) : 0.0;
    val = bfcnmv_warp_reduce_sum_f64(val);
    if (lane == 0)
        output[row] = is_homo ? (weights[0] * val) : val;
}

__global__ void _bg_bool_basic_kern_f64(
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ spikes,
    double*        __restrict__ output,
    const double*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    extern __shared__ double smem_red_bfmv_f64[];
    int row = blockIdx.x;
    if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const double*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    int lane   = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;
    double val = 0.0;
    for (int chunk = warpid; (chunk << 5) < n_conn; chunk += nwarps) {
        int k = (chunk << 5) + lane;
        bool in_range = (k < n_conn);
        int safe_k = in_range ? k : (n_conn - 1);
        bool active = in_range && (spikes[i_row[safe_k]] != 0);
        unsigned ballot = __ballot_sync(0xffffffff, active);
        if (ballot == 0) continue;
        if (active)
            val += is_homo ? 1.0 : w_row[k];
    }
    val = bfcnmv_warp_reduce_sum_f64(val);
    if (lane == 0) smem_red_bfmv_f64[warpid] = val;
    __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < n_warps) ? smem_red_bfmv_f64[lane] : 0.0;
    if (warpid == 0) val = bfcnmv_warp_reduce_sum_f64(val);
    if (threadIdx.x == 0)
        output[row] = is_homo ? (weights[0] * val) : val;
}

__global__ void _bg_float_warp_kern_f64(
    const int32_t* __restrict__ indices,
    const double*  __restrict__ spikes,
    double*        __restrict__ output,
    const double*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;
    int lane = threadIdx.x;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const double*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    bool in_range = (lane < n_conn);
    int safe_lane = in_range ? lane : (n_conn - 1);
    bool active = in_range && (spikes[i_row[safe_lane]] > 0.0);
    double val = active ? (is_homo ? 1.0 : w_row[lane]) : 0.0;
    val = bfcnmv_warp_reduce_sum_f64(val);
    if (lane == 0)
        output[row] = is_homo ? (weights[0] * val) : val;
}

__global__ void _bg_float_basic_kern_f64(
    const int32_t* __restrict__ indices,
    const double*  __restrict__ spikes,
    double*        __restrict__ output,
    const double*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    extern __shared__ double smem_red_bfmv_f64[];
    int row = blockIdx.x;
    if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const double*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    int lane   = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;
    double val = 0.0;
    for (int chunk = warpid; (chunk << 5) < n_conn; chunk += nwarps) {
        int k = (chunk << 5) + lane;
        bool in_range = (k < n_conn);
        int safe_k = in_range ? k : (n_conn - 1);
        bool active = in_range && (spikes[i_row[safe_k]] > 0.0);
        unsigned ballot = __ballot_sync(0xffffffff, active);
        if (ballot == 0) continue;
        if (active)
            val += is_homo ? 1.0 : w_row[k];
    }
    val = bfcnmv_warp_reduce_sum_f64(val);
    if (lane == 0) smem_red_bfmv_f64[warpid] = val;
    __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < n_warps) ? smem_red_bfmv_f64[lane] : 0.0;
    if (warpid == 0) val = bfcnmv_warp_reduce_sum_f64(val);
    if (threadIdx.x == 0)
        output[row] = is_homo ? (weights[0] * val) : val;
}

__global__ void _bs_bool_warp_kern_f64(
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ spikes,
    double*        __restrict__ output,
    const double*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane_id   = threadIdx.x & 31;
    int num_warps = (gridDim.x * blockDim.x) >> 5;
    for (int row = warp_id; row < n_pre; row += num_warps) {
        if (!spikes[row]) continue;
        const int32_t* i_row = indices + (size_t)row * n_conn;
        const double*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
        double w0 = is_homo ? weights[0] : 0.0;
        for (int k = lane_id; k < n_conn; k += 32)
            atomicAdd(&output[i_row[k]], is_homo ? w0 : w_row[k]);
    }
}

__global__ void _bs_bool_basic_kern_f64(
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ spikes,
    double*        __restrict__ output,
    const double*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;
    if (!spikes[row]) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const double*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    double w0 = is_homo ? weights[0] : 0.0;
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x)
        atomicAdd(&output[i_row[k]], is_homo ? w0 : w_row[k]);
}

__global__ void _bs_float_warp_kern_f64(
    const int32_t* __restrict__ indices,
    const double*  __restrict__ spikes,
    double*        __restrict__ output,
    const double*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane_id   = threadIdx.x & 31;
    int num_warps = (gridDim.x * blockDim.x) >> 5;
    for (int row = warp_id; row < n_pre; row += num_warps) {
        if (!(spikes[row] > 0.0)) continue;
        const int32_t* i_row = indices + (size_t)row * n_conn;
        const double*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
        double w0 = is_homo ? weights[0] : 0.0;
        for (int k = lane_id; k < n_conn; k += 32)
            atomicAdd(&output[i_row[k]], is_homo ? w0 : w_row[k]);
    }
}

__global__ void _bs_float_basic_kern_f64(
    const int32_t* __restrict__ indices,
    const double*  __restrict__ spikes,
    double*        __restrict__ output,
    const double*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;
    if (!(spikes[row] > 0.0)) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const double*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    double w0 = is_homo ? weights[0] : 0.0;
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x)
        atomicAdd(&output[i_row[k]], is_homo ? w0 : w_row[k]);
}

void binary_fcnmv_gather_bool_warp_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const uint8_t* d_spk = static_cast<const uint8_t*>(spikes.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    _bg_bool_warp_kern_f64<<<n_pre, 32, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

void binary_fcnmv_gather_bool_basic_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const uint8_t* d_spk = static_cast<const uint8_t*>(spikes.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    size_t shm = 32 * sizeof(double);
    _bg_bool_basic_kern_f64<<<n_pre, 256, shm, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

void binary_fcnmv_gather_float_warp_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const double*  d_spk = static_cast<const double*>(spikes.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    _bg_float_warp_kern_f64<<<n_pre, 32, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

void binary_fcnmv_gather_float_basic_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const double*  d_spk = static_cast<const double*>(spikes.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    size_t shm = 32 * sizeof(double);
    _bg_float_basic_kern_f64<<<n_pre, 256, shm, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

void binary_fcnmv_scatter_bool_warp_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const uint8_t* d_spk = static_cast<const uint8_t*>(spikes.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(double), s);
    int blocks = (n_pre + 7) / 8;
    _bs_bool_warp_kern_f64<<<blocks, 256, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

void binary_fcnmv_scatter_bool_basic_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const uint8_t* d_spk = static_cast<const uint8_t*>(spikes.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(double), s);
    _bs_bool_basic_kern_f64<<<n_pre, 256, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

void binary_fcnmv_scatter_float_warp_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const double*  d_spk = static_cast<const double*>(spikes.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(double), s);
    int blocks = (n_pre + 7) / 8;
    _bs_float_warp_kern_f64<<<blocks, 256, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

void binary_fcnmv_scatter_float_basic_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const double*  d_spk = static_cast<const double*>(spikes.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(double), s);
    _bs_float_basic_kern_f64<<<n_pre, 256, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

// =========================================================================
// float16 (__half) variants for binary_fcnmv — accumulate in float32
// =========================================================================

__global__ void _bg_bool_warp_kern_f16(
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ spikes,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;
    int lane = threadIdx.x;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const __half*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    bool in_range = (lane < n_conn);
    int safe_lane = in_range ? lane : (n_conn - 1);
    bool active = in_range && (spikes[i_row[safe_lane]] != 0);
    float val = active ? (is_homo ? 1.0f : __half2float(w_row[lane])) : 0.0f;
    val = warp_reduce_sum(val);
    if (lane == 0)
        output[row] = __float2half(is_homo ? (__half2float(weights[0]) * val) : val);
}

__global__ void _bg_bool_basic_kern_f16(
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ spikes,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    extern __shared__ float smem_red_bfmv_f16h[];
    int row = blockIdx.x;
    if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const __half*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    int lane   = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;
    float val = 0.0f;
    for (int chunk = warpid; (chunk << 5) < n_conn; chunk += nwarps) {
        int k = (chunk << 5) + lane;
        bool in_range = (k < n_conn);
        int safe_k = in_range ? k : (n_conn - 1);
        bool active = in_range && (spikes[i_row[safe_k]] != 0);
        unsigned ballot = __ballot_sync(0xffffffff, active);
        if (ballot == 0) continue;
        if (active)
            val += is_homo ? 1.0f : __half2float(w_row[k]);
    }
    val = warp_reduce_sum(val);
    if (lane == 0) smem_red_bfmv_f16h[warpid] = val;
    __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < n_warps) ? smem_red_bfmv_f16h[lane] : 0.0f;
    if (warpid == 0) val = warp_reduce_sum(val);
    if (threadIdx.x == 0)
        output[row] = __float2half(is_homo ? (__half2float(weights[0]) * val) : val);
}

__global__ void _bg_float_warp_kern_f16(
    const int32_t* __restrict__ indices,
    const __half*  __restrict__ spikes,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;
    int lane = threadIdx.x;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const __half*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    bool in_range = (lane < n_conn);
    int safe_lane = in_range ? lane : (n_conn - 1);
    bool active = in_range && (__half2float(spikes[i_row[safe_lane]]) > 0.0f);
    float val = active ? (is_homo ? 1.0f : __half2float(w_row[lane])) : 0.0f;
    val = warp_reduce_sum(val);
    if (lane == 0)
        output[row] = __float2half(is_homo ? (__half2float(weights[0]) * val) : val);
}

__global__ void _bg_float_basic_kern_f16(
    const int32_t* __restrict__ indices,
    const __half*  __restrict__ spikes,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    extern __shared__ float smem_red_bfmv_f16h[];
    int row = blockIdx.x;
    if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const __half*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    int lane   = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;
    float val = 0.0f;
    for (int chunk = warpid; (chunk << 5) < n_conn; chunk += nwarps) {
        int k = (chunk << 5) + lane;
        bool in_range = (k < n_conn);
        int safe_k = in_range ? k : (n_conn - 1);
        bool active = in_range && (__half2float(spikes[i_row[safe_k]]) > 0.0f);
        unsigned ballot = __ballot_sync(0xffffffff, active);
        if (ballot == 0) continue;
        if (active)
            val += is_homo ? 1.0f : __half2float(w_row[k]);
    }
    val = warp_reduce_sum(val);
    if (lane == 0) smem_red_bfmv_f16h[warpid] = val;
    __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < n_warps) ? smem_red_bfmv_f16h[lane] : 0.0f;
    if (warpid == 0) val = warp_reduce_sum(val);
    if (threadIdx.x == 0)
        output[row] = __float2half(is_homo ? (__half2float(weights[0]) * val) : val);
}

__global__ void _bs_bool_warp_kern_f16(
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ spikes,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane_id   = threadIdx.x & 31;
    int num_warps = (gridDim.x * blockDim.x) >> 5;
    for (int row = warp_id; row < n_pre; row += num_warps) {
        if (!spikes[row]) continue;
        const int32_t* i_row = indices + (size_t)row * n_conn;
        const __half*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
        float w0 = is_homo ? __half2float(weights[0]) : 0.0f;
        for (int k = lane_id; k < n_conn; k += 32)
            atomicAdd(&output[i_row[k]], __float2half(is_homo ? w0 : __half2float(w_row[k])));
    }
}

__global__ void _bs_bool_basic_kern_f16(
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ spikes,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;
    if (!spikes[row]) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const __half*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float w0 = is_homo ? __half2float(weights[0]) : 0.0f;
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x)
        atomicAdd(&output[i_row[k]], __float2half(is_homo ? w0 : __half2float(w_row[k])));
}

__global__ void _bs_float_warp_kern_f16(
    const int32_t* __restrict__ indices,
    const __half*  __restrict__ spikes,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane_id   = threadIdx.x & 31;
    int num_warps = (gridDim.x * blockDim.x) >> 5;
    for (int row = warp_id; row < n_pre; row += num_warps) {
        if (!(__half2float(spikes[row]) > 0.0f)) continue;
        const int32_t* i_row = indices + (size_t)row * n_conn;
        const __half*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
        float w0 = is_homo ? __half2float(weights[0]) : 0.0f;
        for (int k = lane_id; k < n_conn; k += 32)
            atomicAdd(&output[i_row[k]], __float2half(is_homo ? w0 : __half2float(w_row[k])));
    }
}

__global__ void _bs_float_basic_kern_f16(
    const int32_t* __restrict__ indices,
    const __half*  __restrict__ spikes,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;
    if (!(__half2float(spikes[row]) > 0.0f)) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const __half*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float w0 = is_homo ? __half2float(weights[0]) : 0.0f;
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x)
        atomicAdd(&output[i_row[k]], __float2half(is_homo ? w0 : __half2float(w_row[k])));
}

void binary_fcnmv_gather_bool_warp_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const uint8_t* d_spk = static_cast<const uint8_t*>(spikes.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    _bg_bool_warp_kern_f16<<<n_pre, 32, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

void binary_fcnmv_gather_bool_basic_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const uint8_t* d_spk = static_cast<const uint8_t*>(spikes.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    size_t shm = 32 * sizeof(float);
    _bg_bool_basic_kern_f16<<<n_pre, 256, shm, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

void binary_fcnmv_gather_float_warp_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const __half*  d_spk = static_cast<const __half*>(spikes.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    _bg_float_warp_kern_f16<<<n_pre, 32, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

void binary_fcnmv_gather_float_basic_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const __half*  d_spk = static_cast<const __half*>(spikes.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    size_t shm = 32 * sizeof(float);
    _bg_float_basic_kern_f16<<<n_pre, 256, shm, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

void binary_fcnmv_scatter_bool_warp_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const uint8_t* d_spk = static_cast<const uint8_t*>(spikes.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(__half), s);
    int blocks = (n_pre + 7) / 8;
    _bs_bool_warp_kern_f16<<<blocks, 256, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

void binary_fcnmv_scatter_bool_basic_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const uint8_t* d_spk = static_cast<const uint8_t*>(spikes.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(__half), s);
    _bs_bool_basic_kern_f16<<<n_pre, 256, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

void binary_fcnmv_scatter_float_warp_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const __half*  d_spk = static_cast<const __half*>(spikes.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(__half), s);
    int blocks = (n_pre + 7) / 8;
    _bs_float_warp_kern_f16<<<blocks, 256, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}

void binary_fcnmv_scatter_float_basic_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView spikes, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const __half*  d_spk = static_cast<const __half*>(spikes.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(__half), s);
    _bs_float_basic_kern_f16<<<n_pre, 256, 0, s>>>(d_idx, d_spk, d_out, d_w, n_pre, n_conn, is_homo);
}
