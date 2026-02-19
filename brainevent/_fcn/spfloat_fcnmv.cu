/*
 * spfloat_fcnmv.cu — Sparse-Float Event-Driven FCN Matrix-Vector CUDA Kernels
 * =============================================================================
 *
 * Python API: brainevent.spfloat_fcnmv(weights, indices, spikes, *, shape, transpose, backend)
 *
 * Sparse-float event-driven matrix--vector product with fixed connection number.
 *
 * Computes  y = W @ s  (or  y = W^T @ s  when transpose=True)
 * where W is a sparse weight matrix stored in fixed-connection-number
 * format and s is a sparse-float vector.  Unlike binary_fcnmv which treats
 * non-zero entries as 1, this variant preserves their actual floating-point
 * value, combining sparsity skipping with float-valued spike amplitudes.
 *
 * Parameters
 * ----------
 * weights : shape (1,) for homogeneous or (num_pre, num_conn) for heterogeneous,
 *           floating-point dtype (float16 / float32 / float64).
 * indices : shape (num_pre, num_conn), int32 — post-synaptic column indices.
 * spikes  : shape (num_post,) for gather or (num_pre,) for scatter, float dtype.
 *           Zero entries are skipped; non-zero entries use their actual value.
 * shape   : logical (num_pre, num_post) shape of the dense weight matrix.
 * transpose=False (gather mode):
 *   y[i] = sum_{k: spikes[indices[i,k]] != 0} weights[i,k] * spikes[indices[i,k]]
 * transpose=True  (scatter mode):
 *   y[indices[i,k]] += weights[i,k] * spikes[i]  for all i,k where spikes[i] != 0
 *
 * Supported dtypes: float32 (default), float64 (_f64 suffix), float16 (_f16 suffix).
 * shared-memory gather path is float32-only; f16/f64 use basic/warp kernels.
 *
 * IMPORTANT: weights.data_ptr() is a GPU device pointer — NEVER dereference on host.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp16.h>

// ===========================================================================
// Sparse-Float FCN Matrix-Vector (spfloat_fcnmv) CUDA Kernels
//
// KEY DIFFERENCE FROM fcnmv: The input vector (spikes) is SPARSE.
// Optimization strategies:
//   Gather (transpose=False): y[i] = sum_k w[i,k]*s[idx[i,k]]
//     - __ballot_sync detects all-zero 32-connection chunks, skips weight load
//     - At 5% SNN firing rate: ~95% fewer FMA operations
//   Scatter (transpose=True): y[idx[i,k]] += w[i,k]*s[i]
//     - Per-pre-neuron early exit when s[i] == 0
//     - At 5% SNN firing rate: ~95% of pre-neurons skipped entirely
//
// IMPORTANT: weights.data_ptr() returns a GPU device pointer.
// NEVER dereference on host. GPU threads read weights[0] (homo)
// or weights[row*n_conn+k] (hetero).
// ===========================================================================

__device__ __inline__ float spfloat_mv_warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ===========================================================================
// GATHER kernels: y[i] = sum_k w[i,k] * s[idx[i,k]]  (skip s == 0)
// ===========================================================================

// Warp gather: one warp (32 threads) per output row.
// Processes 32 consecutive connections per ballot cycle.
// __ballot_sync detects all-zero chunks → skip weight load/multiply entirely.
// Dynamic shared mem: 0 bytes.  Best for n_conn <= 64.
__global__ void _spfloat_gather_warp_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ vector,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float val = 0.0f;
    // All 32 warp threads process 32 consecutive connections per ballot cycle.
    // Out-of-bounds k gives sp = 0 (contributes false to ballot, no divergence).
    for (int base = 0; base < n_conn; base += 32) {
        int k = base + threadIdx.x;
        float sp = (k < n_conn) ? vector[i_row[k]] : 0.0f;
        // __ballot_sync: if all 32 threads have zero spike, skip weight load
        unsigned ballot = __ballot_sync(0xffffffff, sp != 0.0f);
        if (ballot && k < n_conn && sp != 0.0f)
            val += (is_homo ? weights[0] : w_row[k]) * sp;
    }
    val = spfloat_mv_warp_reduce_sum(val);
    if (threadIdx.x == 0)
        output[row] = val;
}

// Basic gather: one block (256 threads) per output row.
// Each of 8 warps processes 32 consecutive connections with ballot-skip.
// Warp-coalesced access: consecutive lanes access consecutive connections.
// Dynamic shared mem: 32 * sizeof(float) for block reduction.
// Best for 64 < n_conn <= 512.
__global__ void _spfloat_gather_basic_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ vector,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    extern __shared__ float smem_red[];   // 32 floats for block reduction
    int row = blockIdx.x;
    if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;

    // Warp-level loop: each warp processes 32 consecutive connections per iter.
    // All threads in a warp share the same warp_id → same base → enter/exit together.
    int warp_id = threadIdx.x >> 5;   // 0..7
    int lane    = threadIdx.x & 31;
    float val = 0.0f;
    for (int base = warp_id * 32; base < n_conn; base += blockDim.x) {
        int k = base + lane;
        float sp = (k < n_conn) ? vector[i_row[k]] : 0.0f;
        // Per-warp ballot: skip weight multiply when all 32 spikes are zero
        unsigned ballot = __ballot_sync(0xffffffff, sp != 0.0f);
        if (ballot && k < n_conn && sp != 0.0f)
            val += (is_homo ? weights[0] : w_row[k]) * sp;
    }

    // Inline block reduction via dynamic shared memory
    val = spfloat_mv_warp_reduce_sum(val);
    if (lane == 0) smem_red[warp_id] = val;
    __syncthreads();
    int n_warps = blockDim.x >> 5;   // 8
    val = (threadIdx.x < n_warps) ? smem_red[lane] : 0.0f;
    if (warp_id == 0) val = spfloat_mv_warp_reduce_sum(val);
    if (threadIdx.x == 0)
        output[row] = val;
}

// Shared-memory gather: tiles idx+weights into shmem to reduce bandwidth.
// Per-warp ballot-skip for zero-spike tiles.
// Dynamic shared mem: blockDim.x*(sizeof(int32_t)+sizeof(float)) + 32*4 bytes.
// Best for n_conn > 512.
__global__ void _spfloat_gather_shared_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ vector,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    extern __shared__ char smem_raw[];
    int32_t* s_idx = reinterpret_cast<int32_t*>(smem_raw);
    float*   s_wt  = reinterpret_cast<float*>(smem_raw + blockDim.x * sizeof(int32_t));
    float*   s_red = reinterpret_cast<float*>(
        smem_raw + blockDim.x * (sizeof(int32_t) + sizeof(float)));

    int row = blockIdx.x;
    if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;

    float val = 0.0f;
    for (int base = 0; base < n_conn; base += blockDim.x) {
        int k = base + threadIdx.x;
        // Cooperatively load connection tile into shmem (coalesced access)
        if (k < n_conn) {
            s_idx[threadIdx.x] = i_row[k];
            s_wt[threadIdx.x]  = is_homo ? 1.0f : w_row[k];
        }
        __syncthreads();
        int tile = min((int)blockDim.x, n_conn - base);
        // Per-warp ballot: skip if all spikes in this 32-element chunk are zero
        float sp = (threadIdx.x < tile) ? vector[s_idx[threadIdx.x]] : 0.0f;
        unsigned ballot = __ballot_sync(0xffffffff, sp != 0.0f);
        if (ballot && threadIdx.x < tile && sp != 0.0f)
            val += s_wt[threadIdx.x] * sp;
        __syncthreads();
    }

    // Inline block reduction
    int lane   = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;
    val = spfloat_mv_warp_reduce_sum(val);
    if (lane == 0) s_red[warpid] = val;
    __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < n_warps) ? s_red[lane] : 0.0f;
    if (warpid == 0) val = spfloat_mv_warp_reduce_sum(val);
    if (threadIdx.x == 0)
        output[row] = is_homo ? (weights[0] * val) : val;
}

// ===========================================================================
// SCATTER kernels: y[idx[i,k]] += w[i,k]*s[i]
// KEY OPTIMIZATION: entire pre-neuron skipped when s[i] == 0
// At 5% SNN firing rate: 95% of pre-neurons retired in < 10 instructions
// ===========================================================================

// Basic scatter: one block per pre-neuron.
// Early exit when s[row] == 0 → 95% of blocks retire immediately at 5% rate.
// Pre-computes homo_wsp = weights[0] * sp once per active neuron.
__global__ void _spfloat_scatter_basic_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ vector,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;
    float sp = vector[row];
    if (sp == 0.0f) return;   // EARLY EXIT: neuron inactive
    float homo_wsp = is_homo ? weights[0] * sp : 0.0f;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x) {
        float w_sp = is_homo ? homo_wsp : w_row[k] * sp;
        atomicAdd(&output[i_row[k]], w_sp);
    }
}

// Warp scatter: 8 warps per block (256 threads), one warp per pre-neuron.
// __shfl_sync broadcasts s[row] from lane 0 to all 32 lanes → entire warp
// skips if s[row] == 0, without any shared memory or serialised load.
// Best for small-to-medium n_conn (avoids atomicAdd launch overhead).
__global__ void _spfloat_scatter_warp_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ vector,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane_id   = threadIdx.x & 31;
    int num_warps = (gridDim.x * blockDim.x) >> 5;
    for (int row = warp_id; row < n_pre; row += num_warps) {
        // Lane 0 loads spike value; broadcast to all 32 lanes via __shfl_sync
        float sp = (lane_id == 0) ? vector[row] : 0.0f;
        sp = __shfl_sync(0xffffffff, sp, 0);   // broadcast from lane 0
        if (sp == 0.0f) continue;              // ENTIRE WARP SKIPS
        float homo_wsp = is_homo ? weights[0] * sp : 0.0f;
        const int32_t* i_row = indices + (size_t)row * n_conn;
        const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
        for (int k = lane_id; k < n_conn; k += 32) {
            float w_sp = is_homo ? homo_wsp : w_row[k] * sp;
            atomicAdd(&output[i_row[k]], w_sp);
        }
    }
}

// ===========================================================================
// TVM FFI Entry Points
// Convention: args = (weights, indices, vector, output, stream)
// weights.data_ptr() is a GPU device pointer — NEVER dereference on host.
// GPU threads read weights[0] (homo) or weights[row*n_conn+k] (hetero).
// ===========================================================================

void spfloat_fcnmv_gather_warp(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    _spfloat_gather_warp_kern<<<n_pre, 32, 0, s>>>(
        d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
}

void spfloat_fcnmv_gather_basic(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    size_t shm = 32 * sizeof(float);
    _spfloat_gather_basic_kern<<<n_pre, 256, shm, s>>>(
        d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
}

void spfloat_fcnmv_gather_shared(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    int threads = 256;
    size_t shm = (size_t)threads * (sizeof(int32_t) + sizeof(float)) + 32 * sizeof(float);
    _spfloat_gather_shared_kern<<<n_pre, threads, shm, s>>>(
        d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
}

// Auto-selects the best gather kernel based on n_conn.
void spfloat_fcnmv_gather_auto(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    size_t shm_red = 32 * sizeof(float);
    if (n_conn <= 64) {
        // Small n_conn: one warp per row, ballot skip zero chunks
        _spfloat_gather_warp_kern<<<n_pre, 32, 0, s>>>(
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    } else if (n_conn > 512) {
        // Large n_conn: shared-memory tiling amortises index/weight bandwidth
        int threads = 256;
        size_t shm = (size_t)threads * (sizeof(int32_t) + sizeof(float)) + 32 * sizeof(float);
        _spfloat_gather_shared_kern<<<n_pre, threads, shm, s>>>(
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    } else {
        // Medium n_conn: one block per row, 8-warp ballot skip
        _spfloat_gather_basic_kern<<<n_pre, 256, shm_red, s>>>(
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    }
}

// Scatter entry points (output zeroed before kernel launch)

void spfloat_fcnmv_scatter_basic(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,
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
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(float), s);
    // One block per pre-neuron: 95% exit immediately at 5% firing rate
    _spfloat_scatter_basic_kern<<<n_pre, 256, 0, s>>>(
        d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
}

void spfloat_fcnmv_scatter_warp(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,
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
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(float), s);
    // 256 threads = 8 warps; ceil(n_pre / 8) blocks
    int blocks = (n_pre + 7) / 8;
    _spfloat_scatter_warp_kern<<<blocks, 256, 0, s>>>(
        d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
}

// Auto-selects the best scatter kernel based on n_conn.
void spfloat_fcnmv_scatter_auto(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,
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
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(float), s);
    if (n_conn <= 32) {
        // Small n_conn: one warp per neuron, warp-level early exit
        int blocks = (n_pre + 7) / 8;
        _spfloat_scatter_warp_kern<<<blocks, 256, 0, s>>>(
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    } else {
        // Larger n_conn: one block per neuron, block-level early exit
        _spfloat_scatter_basic_kern<<<n_pre, 256, 0, s>>>(
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    }
}

// ===========================================================================
// float64 (double) variants — no vec4, no shared-mem tiling
// ===========================================================================

__device__ __inline__ double spfloat_mv_warp_reduce_sum_f64(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void _spfloat_gather_warp_kern_f64(
    const int32_t* __restrict__ indices,
    const double*  __restrict__ vector,
    double*        __restrict__ output,
    const double*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const double*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    double val = 0.0;
    for (int base = 0; base < n_conn; base += 32) {
        int k = base + threadIdx.x;
        double sp = (k < n_conn) ? vector[i_row[k]] : 0.0;
        unsigned ballot = __ballot_sync(0xffffffff, sp != 0.0);
        if (ballot && k < n_conn && sp != 0.0)
            val += (is_homo ? weights[0] : w_row[k]) * sp;
    }
    val = spfloat_mv_warp_reduce_sum_f64(val);
    if (threadIdx.x == 0)
        output[row] = val;
}

__global__ void _spfloat_gather_basic_kern_f64(
    const int32_t* __restrict__ indices,
    const double*  __restrict__ vector,
    double*        __restrict__ output,
    const double*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    extern __shared__ double smem_red_sfmv_f64[];
    int row = blockIdx.x;
    if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const double*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    double val = 0.0;
    for (int base = warp_id * 32; base < n_conn; base += blockDim.x) {
        int k = base + lane;
        double sp = (k < n_conn) ? vector[i_row[k]] : 0.0;
        unsigned ballot = __ballot_sync(0xffffffff, sp != 0.0);
        if (ballot && k < n_conn && sp != 0.0)
            val += (is_homo ? weights[0] : w_row[k]) * sp;
    }
    val = spfloat_mv_warp_reduce_sum_f64(val);
    if (lane == 0) smem_red_sfmv_f64[warp_id] = val;
    __syncthreads();
    int n_warps = blockDim.x >> 5;
    val = (threadIdx.x < n_warps) ? smem_red_sfmv_f64[lane] : 0.0;
    if (warp_id == 0) val = spfloat_mv_warp_reduce_sum_f64(val);
    if (threadIdx.x == 0)
        output[row] = val;
}

__global__ void _spfloat_scatter_basic_kern_f64(
    const int32_t* __restrict__ indices,
    const double*  __restrict__ vector,
    double*        __restrict__ output,
    const double*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;
    double sp = vector[row];
    if (sp == 0.0) return;
    double homo_wsp = is_homo ? weights[0] * sp : 0.0;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const double*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x) {
        double w_sp = is_homo ? homo_wsp : w_row[k] * sp;
        atomicAdd(&output[i_row[k]], w_sp);
    }
}

__global__ void _spfloat_scatter_warp_kern_f64(
    const int32_t* __restrict__ indices,
    const double*  __restrict__ vector,
    double*        __restrict__ output,
    const double*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane_id   = threadIdx.x & 31;
    int num_warps = (gridDim.x * blockDim.x) >> 5;
    for (int row = warp_id; row < n_pre; row += num_warps) {
        double sp = (lane_id == 0) ? vector[row] : 0.0;
        // __shfl_sync on double: broadcast via two 32-bit shuffles
        unsigned lo = __shfl_sync(0xffffffff, __double2loint(sp), 0);
        unsigned hi = __shfl_sync(0xffffffff, __double2hiint(sp), 0);
        sp = __hiloint2double(hi, lo);
        if (sp == 0.0) continue;
        double homo_wsp = is_homo ? weights[0] * sp : 0.0;
        const int32_t* i_row = indices + (size_t)row * n_conn;
        const double*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
        for (int k = lane_id; k < n_conn; k += 32) {
            double w_sp = is_homo ? homo_wsp : w_row[k] * sp;
            atomicAdd(&output[i_row[k]], w_sp);
        }
    }
}

void spfloat_fcnmv_gather_warp_f64(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const double*  d_vec = static_cast<const double*>(vector.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    _spfloat_gather_warp_kern_f64<<<n_pre, 32, 0, s>>>(
        d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
}

void spfloat_fcnmv_gather_basic_f64(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const double*  d_vec = static_cast<const double*>(vector.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    size_t shm = 32 * sizeof(double);
    _spfloat_gather_basic_kern_f64<<<n_pre, 256, shm, s>>>(
        d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
}

void spfloat_fcnmv_gather_auto_f64(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const double*  d_vec = static_cast<const double*>(vector.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    if (n_conn <= 64) {
        _spfloat_gather_warp_kern_f64<<<n_pre, 32, 0, s>>>(
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    } else {
        size_t shm = 32 * sizeof(double);
        _spfloat_gather_basic_kern_f64<<<n_pre, 256, shm, s>>>(
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    }
}

void spfloat_fcnmv_scatter_basic_f64(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const double*  d_vec = static_cast<const double*>(vector.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(double), s);
    _spfloat_scatter_basic_kern_f64<<<n_pre, 256, 0, s>>>(
        d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
}

void spfloat_fcnmv_scatter_warp_f64(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const double*  d_vec = static_cast<const double*>(vector.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(double), s);
    int blocks = (n_pre + 7) / 8;
    _spfloat_scatter_warp_kern_f64<<<blocks, 256, 0, s>>>(
        d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
}

void spfloat_fcnmv_scatter_auto_f64(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const double*  d_vec = static_cast<const double*>(vector.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(double), s);
    if (n_conn <= 32) {
        int blocks = (n_pre + 7) / 8;
        _spfloat_scatter_warp_kern_f64<<<blocks, 256, 0, s>>>(
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    } else {
        _spfloat_scatter_basic_kern_f64<<<n_pre, 256, 0, s>>>(
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    }
}

// ===========================================================================
// float16 (__half) variants — accumulate in float32, no vec4, no shared tiling
// ===========================================================================

__global__ void _spfloat_gather_warp_kern_f16(
    const int32_t* __restrict__ indices,
    const __half*  __restrict__ vector,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const __half*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float val = 0.0f;
    float w0 = is_homo ? __half2float(weights[0]) : 0.0f;
    for (int base = 0; base < n_conn; base += 32) {
        int k = base + threadIdx.x;
        float sp = (k < n_conn) ? __half2float(vector[i_row[k]]) : 0.0f;
        unsigned ballot = __ballot_sync(0xffffffff, sp != 0.0f);
        if (ballot && k < n_conn && sp != 0.0f) {
            float w = is_homo ? w0 : __half2float(w_row[k]);
            val += w * sp;
        }
    }
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    if (threadIdx.x == 0)
        output[row] = __float2half(val);
}

__global__ void _spfloat_gather_basic_kern_f16(
    const int32_t* __restrict__ indices,
    const __half*  __restrict__ vector,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    extern __shared__ float smem_red_sfmv_f16h[];
    int row = blockIdx.x;
    if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const __half*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    float val = 0.0f;
    float w0 = is_homo ? __half2float(weights[0]) : 0.0f;
    for (int base = warp_id * 32; base < n_conn; base += blockDim.x) {
        int k = base + lane;
        float sp = (k < n_conn) ? __half2float(vector[i_row[k]]) : 0.0f;
        unsigned ballot = __ballot_sync(0xffffffff, sp != 0.0f);
        if (ballot && k < n_conn && sp != 0.0f) {
            float w = is_homo ? w0 : __half2float(w_row[k]);
            val += w * sp;
        }
    }
    // warp reduce
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    if (lane == 0) smem_red_sfmv_f16h[warp_id] = val;
    __syncthreads();
    int n_warps = blockDim.x >> 5;
    val = (threadIdx.x < n_warps) ? smem_red_sfmv_f16h[lane] : 0.0f;
    if (warp_id == 0) {
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
    }
    if (threadIdx.x == 0)
        output[row] = __float2half(val);
}

__global__ void _spfloat_scatter_basic_kern_f16(
    const int32_t* __restrict__ indices,
    const __half*  __restrict__ vector,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;
    float sp = __half2float(vector[row]);
    if (sp == 0.0f) return;
    float w0 = is_homo ? __half2float(weights[0]) : 0.0f;
    float homo_wsp = is_homo ? w0 * sp : 0.0f;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const __half*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x) {
        float w_sp = is_homo ? homo_wsp : __half2float(w_row[k]) * sp;
        atomicAdd(&output[i_row[k]], __float2half(w_sp));
    }
}

__global__ void _spfloat_scatter_warp_kern_f16(
    const int32_t* __restrict__ indices,
    const __half*  __restrict__ vector,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane_id   = threadIdx.x & 31;
    int num_warps = (gridDim.x * blockDim.x) >> 5;
    float w0 = is_homo ? __half2float(weights[0]) : 0.0f;
    for (int row = warp_id; row < n_pre; row += num_warps) {
        float sp = (lane_id == 0) ? __half2float(vector[row]) : 0.0f;
        sp = __shfl_sync(0xffffffff, sp, 0);
        if (sp == 0.0f) continue;
        float homo_wsp = is_homo ? w0 * sp : 0.0f;
        const int32_t* i_row = indices + (size_t)row * n_conn;
        const __half*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
        for (int k = lane_id; k < n_conn; k += 32) {
            float w_sp = is_homo ? homo_wsp : __half2float(w_row[k]) * sp;
            atomicAdd(&output[i_row[k]], __float2half(w_sp));
        }
    }
}

void spfloat_fcnmv_gather_warp_f16(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const __half*  d_vec = static_cast<const __half*>(vector.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    _spfloat_gather_warp_kern_f16<<<n_pre, 32, 0, s>>>(
        d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
}

void spfloat_fcnmv_gather_basic_f16(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const __half*  d_vec = static_cast<const __half*>(vector.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    size_t shm = 32 * sizeof(float);
    _spfloat_gather_basic_kern_f16<<<n_pre, 256, shm, s>>>(
        d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
}

void spfloat_fcnmv_gather_auto_f16(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const __half*  d_vec = static_cast<const __half*>(vector.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    if (n_conn <= 64) {
        _spfloat_gather_warp_kern_f16<<<n_pre, 32, 0, s>>>(
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    } else {
        size_t shm = 32 * sizeof(float);
        _spfloat_gather_basic_kern_f16<<<n_pre, 256, shm, s>>>(
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    }
}

void spfloat_fcnmv_scatter_basic_f16(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const __half*  d_vec = static_cast<const __half*>(vector.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(__half), s);
    _spfloat_scatter_basic_kern_f16<<<n_pre, 256, 0, s>>>(
        d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
}

void spfloat_fcnmv_scatter_warp_f16(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const __half*  d_vec = static_cast<const __half*>(vector.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(__half), s);
    int blocks = (n_pre + 7) / 8;
    _spfloat_scatter_warp_kern_f16<<<blocks, 256, 0, s>>>(
        d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
}

void spfloat_fcnmv_scatter_auto_f16(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const __half*  d_vec = static_cast<const __half*>(vector.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(__half), s);
    if (n_conn <= 32) {
        int blocks = (n_pre + 7) / 8;
        _spfloat_scatter_warp_kern_f16<<<blocks, 256, 0, s>>>(
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    } else {
        _spfloat_scatter_basic_kern_f16<<<n_pre, 256, 0, s>>>(
            d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    }
}
