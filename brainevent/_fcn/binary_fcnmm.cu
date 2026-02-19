/*
 * binary_fcnmm.cu — Event-Driven FCN Sparse Matrix-Matrix CUDA Kernels
 * =====================================================================
 *
 * Python API: brainevent.binary_fcnmm(weights, indices, matrix, *, shape, transpose, backend)
 *
 * Event-driven sparse matrix--matrix product with fixed connection number.
 *
 * Computes  Y = W @ M  (or  Y = W^T @ M  when transpose=True)
 * where W is a sparse weight matrix stored in fixed-connection-number
 * format and M is a dense binary event matrix.  Only the connections
 * to active (spiking) entries contribute to the result.
 *
 * Parameters
 * ----------
 * weights : shape (1,) for homogeneous or (num_pre, num_conn) for heterogeneous,
 *           floating-point dtype (float16 / float32 / float64).
 * indices : shape (num_pre, num_conn), int32 — post-synaptic column indices.
 * matrix  : shape (num_post, n_col) for gather or (num_pre, n_col) for scatter.
 *           bool dtype: active when True.
 *           float dtype: active when > 0.
 * shape   : logical (num_pre, num_post) shape of the dense weight matrix.
 * transpose=False (gather mode):
 *   Y[i,j] = sum_{k} weights[i,k] * 1_{M[indices[i,k],j] active}
 * transpose=True  (scatter mode):
 *   Y[indices[i,k],j] += weights[i,k] * 1_{M[i,j] active}   for all i,k,j
 *
 * Supported weight dtypes: float32 (default), float64 (_f64 suffix), float16 (_f16 suffix).
 * Supported matrix dtypes: bool (_bool_ kernels) and float (_float_ kernels).
 * Optimization: warp ballot + tile-level early exit to skip all-zero warp columns.
 *
 * IMPORTANT: weights.data_ptr() is a GPU device pointer — NEVER dereference on host.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// =========================================================================
// BINARY FCNMM — Fixed Connection Number Matrix-Matrix CUDA Kernels
//
// FCN format:
//   indices : int32  [n_pre, n_conn]  — post-synaptic (column) indices
//   weights : float32 [1] (homo) or [n_pre, n_conn] (hetero)
//
// GATHER mode (transpose=False):
//   matrix M : [n_post, n_batch]
//   output Y : [n_pre,  n_batch]
//   Y[i, j] = sum_{k=0}^{n_conn-1} weights[i,k] * is_active(M[indices[i,k], j])
//
// SCATTER mode (transpose=True):
//   matrix M : [n_pre,  n_batch]
//   output Y : [n_post, n_batch]  — pre-zeroed via cudaMemsetAsync
//   if is_active(M[i,j]):  Y[indices[i,k], j] += weights[i,k]  for all k
//
// Activity:  bool matrix (uint8): s != 0    float matrix: s > 0.0f
//
// Event-driven strategy:
//
//   GATHER, warp variant (n_conn <= 32):
//     Grid (n_pre, ceil(n_batch/32)), Block 32.
//     Thread t handles output column j = blockIdx.y*32 + t.
//     Branchless loop over k (no __ballot_sync): entire row fits in 1-2
//     cache lines; ballot overhead exceeds savings at 5-10% firing rates.
//
//   GATHER, basic variant (n_conn > 32):
//     Same grid/block.  __ballot_sync per k across 32 j-threads:
//     if no batch column is active for source row indices[i,k], skip weight load.
//
//   SCATTER, warp variant (n_conn <= 32):
//     Grid (n_pre, ceil(n_batch/32)), Block 32.
//     Tile-level ballot: skip entire block if no j-column active in tile.
//     Active threads loop over k <= 32 connections -> atomicAdd.
//
//   SCATTER, basic variant (n_conn > 32):
//     Grid (n_pre,), Block 256.  Shared int flag for row-level early exit.
//     For active rows: sequential j-loop picks active columns; 256 threads
//     parallelize inner k-loop with atomicAdd.
//
// Memory safety:
//   bool matrix stored as uint8.
//   Scatter output pre-zeroed via cudaMemsetAsync in TVM FFI entry functions.
//   Host entry functions only read ndim()/size() metadata.
//   data_ptr() passed to kernels, never dereferenced on host (SIGSEGV risk).
// =========================================================================

// =========================================================================
// GATHER kernels (transpose=False)
// =========================================================================

// ---- Gather / Bool matrix, warp variant (n_conn <= 32) ----
// Grid: (n_pre, ceil(n_batch/32))  Block: 32
// Thread t handles output Y[row, j], j = blockIdx.y*32 + t.
// Branchless loop over k; inactive lanes contribute 0. No __ballot_sync.
__global__ void _bgm_bool_gather_warp_kern(
    const int32_t* __restrict__ indices,   // [n_pre, n_conn]
    const uint8_t* __restrict__ matrix,    // [n_post, n_batch] bool->uint8
    float*         __restrict__ output,    // [n_pre, n_batch]
    const float*   __restrict__ weights,   // [1] or [n_pre, n_conn]
    int n_pre, int n_conn, int n_batch, int is_homo
) {
    int row = blockIdx.x;
    int t   = threadIdx.x;
    int j   = (int)blockIdx.y * 32 + t;
    if (row >= n_pre) return;
    bool col_valid = (j < n_batch);
    int  safe_j    = col_valid ? j : 0;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float accum = 0.0f;
    for (int k = 0; k < n_conn; k++) {
        int src    = i_row[k];
        bool active = col_valid && (matrix[(size_t)src * n_batch + safe_j] != 0);
        accum += active ? (is_homo ? 1.0f : w_row[k]) : 0.0f;
    }
    if (col_valid)
        output[(size_t)row * n_batch + j] = is_homo ? (weights[0] * accum) : accum;
}

// ---- Gather / Bool matrix, basic variant (n_conn > 32) ----
// Grid: (n_pre, ceil(n_batch/32))  Block: 32
// __ballot_sync per k across 32 j-threads: if no column is active for source
// row indices[i,k], skip the weight load (event-driven inner loop).
__global__ void _bgm_bool_gather_basic_kern(
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ matrix,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int n_batch, int is_homo
) {
    int row = blockIdx.x;
    int t   = threadIdx.x;
    int j   = (int)blockIdx.y * 32 + t;
    if (row >= n_pre) return;
    bool col_valid = (j < n_batch);
    int  safe_j    = col_valid ? j : 0;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float accum = 0.0f;
    for (int k = 0; k < n_conn; k++) {
        int  src    = i_row[k];
        bool active = col_valid && (matrix[(size_t)src * n_batch + safe_j] != 0);
        // Skip weight load if no batch column is active for this source row
        unsigned ballot = __ballot_sync(0xffffffff, active);
        if (ballot == 0) continue;
        if (active)
            accum += is_homo ? 1.0f : w_row[k];
    }
    if (col_valid)
        output[(size_t)row * n_batch + j] = is_homo ? (weights[0] * accum) : accum;
}

// ---- Gather / Float matrix, warp variant (n_conn <= 32) ----
// Grid: (n_pre, ceil(n_batch/32))  Block: 32  Branchless.
__global__ void _bgm_float_gather_warp_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ matrix,    // [n_post, n_batch]
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int n_batch, int is_homo
) {
    int row = blockIdx.x;
    int t   = threadIdx.x;
    int j   = (int)blockIdx.y * 32 + t;
    if (row >= n_pre) return;
    bool col_valid = (j < n_batch);
    int  safe_j    = col_valid ? j : 0;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float accum = 0.0f;
    for (int k = 0; k < n_conn; k++) {
        int  src    = i_row[k];
        bool active = col_valid && (matrix[(size_t)src * n_batch + safe_j] > 0.0f);
        accum += active ? (is_homo ? 1.0f : w_row[k]) : 0.0f;
    }
    if (col_valid)
        output[(size_t)row * n_batch + j] = is_homo ? (weights[0] * accum) : accum;
}

// ---- Gather / Float matrix, basic variant (n_conn > 32) ----
// Grid: (n_pre, ceil(n_batch/32))  Block: 32
// __ballot_sync per k (same event-driven strategy as bool variant).
__global__ void _bgm_float_gather_basic_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ matrix,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int n_batch, int is_homo
) {
    int row = blockIdx.x;
    int t   = threadIdx.x;
    int j   = (int)blockIdx.y * 32 + t;
    if (row >= n_pre) return;
    bool col_valid = (j < n_batch);
    int  safe_j    = col_valid ? j : 0;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float accum = 0.0f;
    for (int k = 0; k < n_conn; k++) {
        int  src    = i_row[k];
        bool active = col_valid && (matrix[(size_t)src * n_batch + safe_j] > 0.0f);
        unsigned ballot = __ballot_sync(0xffffffff, active);
        if (ballot == 0) continue;
        if (active)
            accum += is_homo ? 1.0f : w_row[k];
    }
    if (col_valid)
        output[(size_t)row * n_batch + j] = is_homo ? (weights[0] * accum) : accum;
}

// =========================================================================
// SCATTER kernels (transpose=True)
//
// if is_active(M[i, j]):  Y[indices[i,k], j] += weights[i,k]  for all k
//
// Output buffer must be pre-zeroed; done via cudaMemsetAsync below.
//
// Key optimisation: event-driven early exit avoids all atomicAdds for
// inactive pre-neurons / batch tiles.
// =========================================================================

// ---- Scatter / Bool matrix, warp variant (n_conn <= 32) ----
// Grid: (n_pre, ceil(n_batch/32))  Block: 32
// Each block: one pre-neuron row + one 32-column batch tile.
// Tile-level ballot: skip if no active column in tile.
// Active threads loop over k <= 32 connections -> atomicAdd to output.
__global__ void _bgm_bool_scatter_warp_kern(
    const int32_t* __restrict__ indices,   // [n_pre, n_conn]
    const uint8_t* __restrict__ matrix,    // [n_pre, n_batch]
    float*         __restrict__ output,    // [n_post, n_batch]
    const float*   __restrict__ weights,   // [1] or [n_pre, n_conn]
    int n_pre, int n_conn, int n_batch, int is_homo
) {
    int row = blockIdx.x;
    int t   = threadIdx.x;
    int j   = (int)blockIdx.y * 32 + t;
    if (row >= n_pre) return;
    bool col_valid = (j < n_batch);
    int  safe_j    = col_valid ? j : 0;
    bool active    = col_valid && (matrix[(size_t)row * n_batch + safe_j] != 0);
    // Tile-level event-driven exit: skip if no active column in this tile
    if (__ballot_sync(0xffffffff, active) == 0) return;
    if (!active) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float w0 = is_homo ? weights[0] : 0.0f;
    for (int k = 0; k < n_conn; k++)
        atomicAdd(&output[(size_t)i_row[k] * n_batch + j], is_homo ? w0 : w_row[k]);
}

// ---- Scatter / Bool matrix, basic variant (n_conn > 32) ----
// Grid: (n_pre,)  Block: 256  Shared memory: sizeof(int) for row-active flag.
// Row-level early exit: entire block returns if M[i,:] is all-zero.
// For active rows: sequential j-loop picks active columns; 256 threads
// parallelise the inner k-loop (stride=blockDim.x) with atomicAdd.
__global__ void _bgm_bool_scatter_basic_kern(
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ matrix,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int n_batch, int is_homo
) {
    extern __shared__ int smem_flag[];
    int row = blockIdx.x;
    if (row >= n_pre) return;
    if (threadIdx.x == 0) smem_flag[0] = 0;
    __syncthreads();
    // Check if any M[row, j] is active (all threads scan their portion of j)
    for (int j = threadIdx.x; j < n_batch; j += blockDim.x)
        if (matrix[(size_t)row * n_batch + j] != 0) { atomicOr(smem_flag, 1); break; }
    __syncthreads();
    if (smem_flag[0] == 0) return;   // row entirely inactive -> skip all atomicAdds
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float w0 = is_homo ? weights[0] : 0.0f;
    for (int j = 0; j < n_batch; j++) {
        if (!matrix[(size_t)row * n_batch + j]) continue;
        for (int k = threadIdx.x; k < n_conn; k += blockDim.x)
            atomicAdd(&output[(size_t)i_row[k] * n_batch + j], is_homo ? w0 : w_row[k]);
    }
}

// ---- Scatter / Float matrix, warp variant (n_conn <= 32) ----
// Grid: (n_pre, ceil(n_batch/32))  Block: 32  Same as bool, check > 0.0f.
__global__ void _bgm_float_scatter_warp_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ matrix,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int n_batch, int is_homo
) {
    int row = blockIdx.x;
    int t   = threadIdx.x;
    int j   = (int)blockIdx.y * 32 + t;
    if (row >= n_pre) return;
    bool col_valid = (j < n_batch);
    int  safe_j    = col_valid ? j : 0;
    bool active    = col_valid && (matrix[(size_t)row * n_batch + safe_j] > 0.0f);
    if (__ballot_sync(0xffffffff, active) == 0) return;
    if (!active) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float w0 = is_homo ? weights[0] : 0.0f;
    for (int k = 0; k < n_conn; k++)
        atomicAdd(&output[(size_t)i_row[k] * n_batch + j], is_homo ? w0 : w_row[k]);
}

// ---- Scatter / Float matrix, basic variant (n_conn > 32) ----
// Grid: (n_pre,)  Block: 256  Shared: sizeof(int).
__global__ void _bgm_float_scatter_basic_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ matrix,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int n_batch, int is_homo
) {
    extern __shared__ int smem_flag[];
    int row = blockIdx.x;
    if (row >= n_pre) return;
    if (threadIdx.x == 0) smem_flag[0] = 0;
    __syncthreads();
    for (int j = threadIdx.x; j < n_batch; j += blockDim.x)
        if (matrix[(size_t)row * n_batch + j] > 0.0f) { atomicOr(smem_flag, 1); break; }
    __syncthreads();
    if (smem_flag[0] == 0) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float w0 = is_homo ? weights[0] : 0.0f;
    for (int j = 0; j < n_batch; j++) {
        if (!(matrix[(size_t)row * n_batch + j] > 0.0f)) continue;
        for (int k = threadIdx.x; k < n_conn; k += blockDim.x)
            atomicAdd(&output[(size_t)i_row[k] * n_batch + j], is_homo ? w0 : w_row[k]);
    }
}

// =========================================================================
// TVM FFI Entry Points
//
// Convention: args = (weights, indices, matrix, output, stream)
//   weights : float32, shape (1,) homo or (n_pre, n_conn) hetero
//   indices : int32,   shape (n_pre, n_conn)
//   matrix  : gather -> (n_post, n_batch);  scatter -> (n_pre, n_batch)
//             bool variant -> uint8;  float variant -> float32
//   output  : gather -> (n_pre, n_batch) float32, written directly
//             scatter -> (n_post, n_batch) float32, zeroed via cudaMemsetAsync
//
// IMPORTANT: data_ptr() returns a GPU device memory pointer.
// NEVER dereference it in host C++ code (causes SIGSEGV).
// Pass it to device kernels; GPU threads read from it.
// Only ndim() and size() are host-safe metadata reads.
// =========================================================================

// ---- Gather / Bool ----

void binary_fcnmm_gather_bool_warp(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t   s       = reinterpret_cast<cudaStream_t>(stream);
    int n_pre   = static_cast<int>(indices.size(0));
    int n_conn  = static_cast<int>(indices.size(1));
    int n_batch = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const uint8_t* d_mat = static_cast<const uint8_t*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    int batch_tiles = (n_batch + 31) / 32;
    dim3 grid(n_pre, batch_tiles);
    _bgm_bool_gather_warp_kern<<<grid, 32, 0, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo);
}

void binary_fcnmm_gather_bool_basic(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t   s       = reinterpret_cast<cudaStream_t>(stream);
    int n_pre   = static_cast<int>(indices.size(0));
    int n_conn  = static_cast<int>(indices.size(1));
    int n_batch = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const uint8_t* d_mat = static_cast<const uint8_t*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    int batch_tiles = (n_batch + 31) / 32;
    dim3 grid(n_pre, batch_tiles);
    _bgm_bool_gather_basic_kern<<<grid, 32, 0, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo);
}

// ---- Gather / Float ----

void binary_fcnmm_gather_float_warp(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t   s       = reinterpret_cast<cudaStream_t>(stream);
    int n_pre   = static_cast<int>(indices.size(0));
    int n_conn  = static_cast<int>(indices.size(1));
    int n_batch = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_mat = static_cast<const float*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    int batch_tiles = (n_batch + 31) / 32;
    dim3 grid(n_pre, batch_tiles);
    _bgm_float_gather_warp_kern<<<grid, 32, 0, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo);
}

void binary_fcnmm_gather_float_basic(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t   s       = reinterpret_cast<cudaStream_t>(stream);
    int n_pre   = static_cast<int>(indices.size(0));
    int n_conn  = static_cast<int>(indices.size(1));
    int n_batch = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_mat = static_cast<const float*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    int batch_tiles = (n_batch + 31) / 32;
    dim3 grid(n_pre, batch_tiles);
    _bgm_float_gather_basic_kern<<<grid, 32, 0, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo);
}

// ---- Scatter / Bool (output pre-zeroed via cudaMemsetAsync) ----

void binary_fcnmm_scatter_bool_warp(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t   s       = reinterpret_cast<cudaStream_t>(stream);
    int n_pre   = static_cast<int>(indices.size(0));
    int n_conn  = static_cast<int>(indices.size(1));
    int n_post  = static_cast<int>(output.size(0));
    int n_batch = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const uint8_t* d_mat = static_cast<const uint8_t*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_batch * sizeof(float), s);
    int batch_tiles = (n_batch + 31) / 32;
    dim3 grid(n_pre, batch_tiles);
    _bgm_bool_scatter_warp_kern<<<grid, 32, 0, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo);
}

void binary_fcnmm_scatter_bool_basic(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t   s       = reinterpret_cast<cudaStream_t>(stream);
    int n_pre   = static_cast<int>(indices.size(0));
    int n_conn  = static_cast<int>(indices.size(1));
    int n_post  = static_cast<int>(output.size(0));
    int n_batch = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const uint8_t* d_mat = static_cast<const uint8_t*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_batch * sizeof(float), s);
    size_t shm = sizeof(int);
    _bgm_bool_scatter_basic_kern<<<n_pre, 256, shm, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo);
}

// ---- Scatter / Float ----

void binary_fcnmm_scatter_float_warp(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t   s       = reinterpret_cast<cudaStream_t>(stream);
    int n_pre   = static_cast<int>(indices.size(0));
    int n_conn  = static_cast<int>(indices.size(1));
    int n_post  = static_cast<int>(output.size(0));
    int n_batch = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_mat = static_cast<const float*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_batch * sizeof(float), s);
    int batch_tiles = (n_batch + 31) / 32;
    dim3 grid(n_pre, batch_tiles);
    _bgm_float_scatter_warp_kern<<<grid, 32, 0, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo);
}

void binary_fcnmm_scatter_float_basic(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t   s       = reinterpret_cast<cudaStream_t>(stream);
    int n_pre   = static_cast<int>(indices.size(0));
    int n_conn  = static_cast<int>(indices.size(1));
    int n_post  = static_cast<int>(output.size(0));
    int n_batch = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_mat = static_cast<const float*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_batch * sizeof(float), s);
    size_t shm = sizeof(int);
    _bgm_float_scatter_basic_kern<<<n_pre, 256, shm, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo);
}

// =========================================================================
// float64 (double) variants for binary_fcnmm
// =========================================================================

__global__ void _bgm_bool_gather_warp_kern_f64(
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ matrix,
    double*        __restrict__ output,
    const double*  __restrict__ weights,
    int n_pre, int n_conn, int n_batch, int is_homo
) {
    int row = blockIdx.x;
    int t   = threadIdx.x;
    int j   = (int)blockIdx.y * 32 + t;
    if (row >= n_pre) return;
    bool col_valid = (j < n_batch);
    int  safe_j    = col_valid ? j : 0;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const double*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    double accum = 0.0;
    for (int k = 0; k < n_conn; k++) {
        int src    = i_row[k];
        bool active = col_valid && (matrix[(size_t)src * n_batch + safe_j] != 0);
        accum += active ? (is_homo ? 1.0 : w_row[k]) : 0.0;
    }
    if (col_valid)
        output[(size_t)row * n_batch + j] = is_homo ? (weights[0] * accum) : accum;
}

__global__ void _bgm_bool_gather_basic_kern_f64(
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ matrix,
    double*        __restrict__ output,
    const double*  __restrict__ weights,
    int n_pre, int n_conn, int n_batch, int is_homo
) {
    int row = blockIdx.x;
    int t   = threadIdx.x;
    int j   = (int)blockIdx.y * 32 + t;
    if (row >= n_pre) return;
    bool col_valid = (j < n_batch);
    int  safe_j    = col_valid ? j : 0;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const double*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    double accum = 0.0;
    for (int k = 0; k < n_conn; k++) {
        int  src    = i_row[k];
        bool active = col_valid && (matrix[(size_t)src * n_batch + safe_j] != 0);
        unsigned ballot = __ballot_sync(0xffffffff, active);
        if (ballot == 0) continue;
        if (active)
            accum += is_homo ? 1.0 : w_row[k];
    }
    if (col_valid)
        output[(size_t)row * n_batch + j] = is_homo ? (weights[0] * accum) : accum;
}

__global__ void _bgm_float_gather_warp_kern_f64(
    const int32_t* __restrict__ indices,
    const double*  __restrict__ matrix,
    double*        __restrict__ output,
    const double*  __restrict__ weights,
    int n_pre, int n_conn, int n_batch, int is_homo
) {
    int row = blockIdx.x;
    int t   = threadIdx.x;
    int j   = (int)blockIdx.y * 32 + t;
    if (row >= n_pre) return;
    bool col_valid = (j < n_batch);
    int  safe_j    = col_valid ? j : 0;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const double*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    double accum = 0.0;
    for (int k = 0; k < n_conn; k++) {
        int  src    = i_row[k];
        bool active = col_valid && (matrix[(size_t)src * n_batch + safe_j] > 0.0);
        accum += active ? (is_homo ? 1.0 : w_row[k]) : 0.0;
    }
    if (col_valid)
        output[(size_t)row * n_batch + j] = is_homo ? (weights[0] * accum) : accum;
}

__global__ void _bgm_float_gather_basic_kern_f64(
    const int32_t* __restrict__ indices,
    const double*  __restrict__ matrix,
    double*        __restrict__ output,
    const double*  __restrict__ weights,
    int n_pre, int n_conn, int n_batch, int is_homo
) {
    int row = blockIdx.x;
    int t   = threadIdx.x;
    int j   = (int)blockIdx.y * 32 + t;
    if (row >= n_pre) return;
    bool col_valid = (j < n_batch);
    int  safe_j    = col_valid ? j : 0;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const double*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    double accum = 0.0;
    for (int k = 0; k < n_conn; k++) {
        int  src    = i_row[k];
        bool active = col_valid && (matrix[(size_t)src * n_batch + safe_j] > 0.0);
        unsigned ballot = __ballot_sync(0xffffffff, active);
        if (ballot == 0) continue;
        if (active)
            accum += is_homo ? 1.0 : w_row[k];
    }
    if (col_valid)
        output[(size_t)row * n_batch + j] = is_homo ? (weights[0] * accum) : accum;
}

__global__ void _bgm_bool_scatter_warp_kern_f64(
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ matrix,
    double*        __restrict__ output,
    const double*  __restrict__ weights,
    int n_pre, int n_conn, int n_batch, int is_homo
) {
    int row = blockIdx.x;
    int t   = threadIdx.x;
    int j   = (int)blockIdx.y * 32 + t;
    if (row >= n_pre) return;
    bool col_valid = (j < n_batch);
    int  safe_j    = col_valid ? j : 0;
    bool active    = col_valid && (matrix[(size_t)row * n_batch + safe_j] != 0);
    if (__ballot_sync(0xffffffff, active) == 0) return;
    if (!active) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const double*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    double w0 = is_homo ? weights[0] : 0.0;
    for (int k = 0; k < n_conn; k++)
        atomicAdd(&output[(size_t)i_row[k] * n_batch + j], is_homo ? w0 : w_row[k]);
}

__global__ void _bgm_bool_scatter_basic_kern_f64(
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ matrix,
    double*        __restrict__ output,
    const double*  __restrict__ weights,
    int n_pre, int n_conn, int n_batch, int is_homo
) {
    extern __shared__ int smem_flag_f64[];
    int row = blockIdx.x;
    if (row >= n_pre) return;
    if (threadIdx.x == 0) smem_flag_f64[0] = 0;
    __syncthreads();
    for (int j = threadIdx.x; j < n_batch; j += blockDim.x)
        if (matrix[(size_t)row * n_batch + j] != 0) { atomicOr(smem_flag_f64, 1); break; }
    __syncthreads();
    if (smem_flag_f64[0] == 0) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const double*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    double w0 = is_homo ? weights[0] : 0.0;
    for (int j = 0; j < n_batch; j++) {
        if (!matrix[(size_t)row * n_batch + j]) continue;
        for (int k = threadIdx.x; k < n_conn; k += blockDim.x)
            atomicAdd(&output[(size_t)i_row[k] * n_batch + j], is_homo ? w0 : w_row[k]);
    }
}

__global__ void _bgm_float_scatter_warp_kern_f64(
    const int32_t* __restrict__ indices,
    const double*  __restrict__ matrix,
    double*        __restrict__ output,
    const double*  __restrict__ weights,
    int n_pre, int n_conn, int n_batch, int is_homo
) {
    int row = blockIdx.x;
    int t   = threadIdx.x;
    int j   = (int)blockIdx.y * 32 + t;
    if (row >= n_pre) return;
    bool col_valid = (j < n_batch);
    int  safe_j    = col_valid ? j : 0;
    bool active    = col_valid && (matrix[(size_t)row * n_batch + safe_j] > 0.0);
    if (__ballot_sync(0xffffffff, active) == 0) return;
    if (!active) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const double*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    double w0 = is_homo ? weights[0] : 0.0;
    for (int k = 0; k < n_conn; k++)
        atomicAdd(&output[(size_t)i_row[k] * n_batch + j], is_homo ? w0 : w_row[k]);
}

__global__ void _bgm_float_scatter_basic_kern_f64(
    const int32_t* __restrict__ indices,
    const double*  __restrict__ matrix,
    double*        __restrict__ output,
    const double*  __restrict__ weights,
    int n_pre, int n_conn, int n_batch, int is_homo
) {
    extern __shared__ int smem_flag_f64[];
    int row = blockIdx.x;
    if (row >= n_pre) return;
    if (threadIdx.x == 0) smem_flag_f64[0] = 0;
    __syncthreads();
    for (int j = threadIdx.x; j < n_batch; j += blockDim.x)
        if (matrix[(size_t)row * n_batch + j] > 0.0) { atomicOr(smem_flag_f64, 1); break; }
    __syncthreads();
    if (smem_flag_f64[0] == 0) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const double*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    double w0 = is_homo ? weights[0] : 0.0;
    for (int j = 0; j < n_batch; j++) {
        if (!(matrix[(size_t)row * n_batch + j] > 0.0)) continue;
        for (int k = threadIdx.x; k < n_conn; k += blockDim.x)
            atomicAdd(&output[(size_t)i_row[k] * n_batch + j], is_homo ? w0 : w_row[k]);
    }
}

void binary_fcnmm_gather_bool_warp_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre   = static_cast<int>(indices.size(0));
    int n_conn  = static_cast<int>(indices.size(1));
    int n_batch = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const uint8_t* d_mat = static_cast<const uint8_t*>(matrix.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    dim3 grid(n_pre, (n_batch + 31) / 32);
    _bgm_bool_gather_warp_kern_f64<<<grid, 32, 0, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo);
}

void binary_fcnmm_gather_bool_basic_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre   = static_cast<int>(indices.size(0));
    int n_conn  = static_cast<int>(indices.size(1));
    int n_batch = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const uint8_t* d_mat = static_cast<const uint8_t*>(matrix.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    dim3 grid(n_pre, (n_batch + 31) / 32);
    _bgm_bool_gather_basic_kern_f64<<<grid, 32, 0, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo);
}

void binary_fcnmm_gather_float_warp_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre   = static_cast<int>(indices.size(0));
    int n_conn  = static_cast<int>(indices.size(1));
    int n_batch = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const double*  d_mat = static_cast<const double*>(matrix.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    dim3 grid(n_pre, (n_batch + 31) / 32);
    _bgm_float_gather_warp_kern_f64<<<grid, 32, 0, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo);
}

void binary_fcnmm_gather_float_basic_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre   = static_cast<int>(indices.size(0));
    int n_conn  = static_cast<int>(indices.size(1));
    int n_batch = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const double*  d_mat = static_cast<const double*>(matrix.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    dim3 grid(n_pre, (n_batch + 31) / 32);
    _bgm_float_gather_basic_kern_f64<<<grid, 32, 0, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo);
}

void binary_fcnmm_scatter_bool_warp_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre   = static_cast<int>(indices.size(0));
    int n_conn  = static_cast<int>(indices.size(1));
    int n_post  = static_cast<int>(output.size(0));
    int n_batch = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const uint8_t* d_mat = static_cast<const uint8_t*>(matrix.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_batch * sizeof(double), s);
    dim3 grid(n_pre, (n_batch + 31) / 32);
    _bgm_bool_scatter_warp_kern_f64<<<grid, 32, 0, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo);
}

void binary_fcnmm_scatter_bool_basic_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre   = static_cast<int>(indices.size(0));
    int n_conn  = static_cast<int>(indices.size(1));
    int n_post  = static_cast<int>(output.size(0));
    int n_batch = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const uint8_t* d_mat = static_cast<const uint8_t*>(matrix.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_batch * sizeof(double), s);
    size_t shm = sizeof(int);
    _bgm_bool_scatter_basic_kern_f64<<<n_pre, 256, shm, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo);
}

void binary_fcnmm_scatter_float_warp_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre   = static_cast<int>(indices.size(0));
    int n_conn  = static_cast<int>(indices.size(1));
    int n_post  = static_cast<int>(output.size(0));
    int n_batch = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const double*  d_mat = static_cast<const double*>(matrix.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_batch * sizeof(double), s);
    dim3 grid(n_pre, (n_batch + 31) / 32);
    _bgm_float_scatter_warp_kern_f64<<<grid, 32, 0, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo);
}

void binary_fcnmm_scatter_float_basic_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre   = static_cast<int>(indices.size(0));
    int n_conn  = static_cast<int>(indices.size(1));
    int n_post  = static_cast<int>(output.size(0));
    int n_batch = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const double*  d_mat = static_cast<const double*>(matrix.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_batch * sizeof(double), s);
    size_t shm = sizeof(int);
    _bgm_float_scatter_basic_kern_f64<<<n_pre, 256, shm, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo);
}

// =========================================================================
// float16 (__half) variants for binary_fcnmm — accumulate in float32
// =========================================================================

__global__ void _bgm_bool_gather_warp_kern_f16(
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ matrix,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int n_batch, int is_homo
) {
    int row = blockIdx.x;
    int t   = threadIdx.x;
    int j   = (int)blockIdx.y * 32 + t;
    if (row >= n_pre) return;
    bool col_valid = (j < n_batch);
    int  safe_j    = col_valid ? j : 0;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const __half*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float accum = 0.0f;
    for (int k = 0; k < n_conn; k++) {
        int  src    = i_row[k];
        bool active = col_valid && (matrix[(size_t)src * n_batch + safe_j] != 0);
        accum += active ? (is_homo ? 1.0f : __half2float(w_row[k])) : 0.0f;
    }
    if (col_valid)
        output[(size_t)row * n_batch + j] = __float2half(is_homo ? (__half2float(weights[0]) * accum) : accum);
}

__global__ void _bgm_bool_gather_basic_kern_f16(
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ matrix,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int n_batch, int is_homo
) {
    int row = blockIdx.x;
    int t   = threadIdx.x;
    int j   = (int)blockIdx.y * 32 + t;
    if (row >= n_pre) return;
    bool col_valid = (j < n_batch);
    int  safe_j    = col_valid ? j : 0;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const __half*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float accum = 0.0f;
    for (int k = 0; k < n_conn; k++) {
        int  src    = i_row[k];
        bool active = col_valid && (matrix[(size_t)src * n_batch + safe_j] != 0);
        unsigned ballot = __ballot_sync(0xffffffff, active);
        if (ballot == 0) continue;
        if (active)
            accum += is_homo ? 1.0f : __half2float(w_row[k]);
    }
    if (col_valid)
        output[(size_t)row * n_batch + j] = __float2half(is_homo ? (__half2float(weights[0]) * accum) : accum);
}

__global__ void _bgm_float_gather_warp_kern_f16(
    const int32_t* __restrict__ indices,
    const __half*  __restrict__ matrix,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int n_batch, int is_homo
) {
    int row = blockIdx.x;
    int t   = threadIdx.x;
    int j   = (int)blockIdx.y * 32 + t;
    if (row >= n_pre) return;
    bool col_valid = (j < n_batch);
    int  safe_j    = col_valid ? j : 0;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const __half*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float accum = 0.0f;
    for (int k = 0; k < n_conn; k++) {
        int  src    = i_row[k];
        bool active = col_valid && (__half2float(matrix[(size_t)src * n_batch + safe_j]) > 0.0f);
        accum += active ? (is_homo ? 1.0f : __half2float(w_row[k])) : 0.0f;
    }
    if (col_valid)
        output[(size_t)row * n_batch + j] = __float2half(is_homo ? (__half2float(weights[0]) * accum) : accum);
}

__global__ void _bgm_float_gather_basic_kern_f16(
    const int32_t* __restrict__ indices,
    const __half*  __restrict__ matrix,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int n_batch, int is_homo
) {
    int row = blockIdx.x;
    int t   = threadIdx.x;
    int j   = (int)blockIdx.y * 32 + t;
    if (row >= n_pre) return;
    bool col_valid = (j < n_batch);
    int  safe_j    = col_valid ? j : 0;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const __half*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float accum = 0.0f;
    for (int k = 0; k < n_conn; k++) {
        int  src    = i_row[k];
        bool active = col_valid && (__half2float(matrix[(size_t)src * n_batch + safe_j]) > 0.0f);
        unsigned ballot = __ballot_sync(0xffffffff, active);
        if (ballot == 0) continue;
        if (active)
            accum += is_homo ? 1.0f : __half2float(w_row[k]);
    }
    if (col_valid)
        output[(size_t)row * n_batch + j] = __float2half(is_homo ? (__half2float(weights[0]) * accum) : accum);
}

__global__ void _bgm_bool_scatter_warp_kern_f16(
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ matrix,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int n_batch, int is_homo
) {
    int row = blockIdx.x;
    int t   = threadIdx.x;
    int j   = (int)blockIdx.y * 32 + t;
    if (row >= n_pre) return;
    bool col_valid = (j < n_batch);
    int  safe_j    = col_valid ? j : 0;
    bool active    = col_valid && (matrix[(size_t)row * n_batch + safe_j] != 0);
    if (__ballot_sync(0xffffffff, active) == 0) return;
    if (!active) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const __half*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float w0 = is_homo ? __half2float(weights[0]) : 0.0f;
    for (int k = 0; k < n_conn; k++)
        atomicAdd(&output[(size_t)i_row[k] * n_batch + j],
                  __float2half(is_homo ? w0 : __half2float(w_row[k])));
}

__global__ void _bgm_bool_scatter_basic_kern_f16(
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ matrix,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int n_batch, int is_homo
) {
    extern __shared__ int smem_flag_f16i[];
    int row = blockIdx.x;
    if (row >= n_pre) return;
    if (threadIdx.x == 0) smem_flag_f16i[0] = 0;
    __syncthreads();
    for (int j = threadIdx.x; j < n_batch; j += blockDim.x)
        if (matrix[(size_t)row * n_batch + j] != 0) { atomicOr(smem_flag_f16i, 1); break; }
    __syncthreads();
    if (smem_flag_f16i[0] == 0) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const __half*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float w0 = is_homo ? __half2float(weights[0]) : 0.0f;
    for (int j = 0; j < n_batch; j++) {
        if (!matrix[(size_t)row * n_batch + j]) continue;
        for (int k = threadIdx.x; k < n_conn; k += blockDim.x)
            atomicAdd(&output[(size_t)i_row[k] * n_batch + j],
                      __float2half(is_homo ? w0 : __half2float(w_row[k])));
    }
}

__global__ void _bgm_float_scatter_warp_kern_f16(
    const int32_t* __restrict__ indices,
    const __half*  __restrict__ matrix,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int n_batch, int is_homo
) {
    int row = blockIdx.x;
    int t   = threadIdx.x;
    int j   = (int)blockIdx.y * 32 + t;
    if (row >= n_pre) return;
    bool col_valid = (j < n_batch);
    int  safe_j    = col_valid ? j : 0;
    bool active    = col_valid && (__half2float(matrix[(size_t)row * n_batch + safe_j]) > 0.0f);
    if (__ballot_sync(0xffffffff, active) == 0) return;
    if (!active) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const __half*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float w0 = is_homo ? __half2float(weights[0]) : 0.0f;
    for (int k = 0; k < n_conn; k++)
        atomicAdd(&output[(size_t)i_row[k] * n_batch + j],
                  __float2half(is_homo ? w0 : __half2float(w_row[k])));
}

__global__ void _bgm_float_scatter_basic_kern_f16(
    const int32_t* __restrict__ indices,
    const __half*  __restrict__ matrix,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int n_batch, int is_homo
) {
    extern __shared__ int smem_flag_f16i[];
    int row = blockIdx.x;
    if (row >= n_pre) return;
    if (threadIdx.x == 0) smem_flag_f16i[0] = 0;
    __syncthreads();
    for (int j = threadIdx.x; j < n_batch; j += blockDim.x)
        if (__half2float(matrix[(size_t)row * n_batch + j]) > 0.0f) { atomicOr(smem_flag_f16i, 1); break; }
    __syncthreads();
    if (smem_flag_f16i[0] == 0) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const __half*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float w0 = is_homo ? __half2float(weights[0]) : 0.0f;
    for (int j = 0; j < n_batch; j++) {
        if (!(__half2float(matrix[(size_t)row * n_batch + j]) > 0.0f)) continue;
        for (int k = threadIdx.x; k < n_conn; k += blockDim.x)
            atomicAdd(&output[(size_t)i_row[k] * n_batch + j],
                      __float2half(is_homo ? w0 : __half2float(w_row[k])));
    }
}

void binary_fcnmm_gather_bool_warp_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre   = static_cast<int>(indices.size(0));
    int n_conn  = static_cast<int>(indices.size(1));
    int n_batch = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const uint8_t* d_mat = static_cast<const uint8_t*>(matrix.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    dim3 grid(n_pre, (n_batch + 31) / 32);
    _bgm_bool_gather_warp_kern_f16<<<grid, 32, 0, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo);
}

void binary_fcnmm_gather_bool_basic_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre   = static_cast<int>(indices.size(0));
    int n_conn  = static_cast<int>(indices.size(1));
    int n_batch = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const uint8_t* d_mat = static_cast<const uint8_t*>(matrix.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    dim3 grid(n_pre, (n_batch + 31) / 32);
    _bgm_bool_gather_basic_kern_f16<<<grid, 32, 0, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo);
}

void binary_fcnmm_gather_float_warp_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre   = static_cast<int>(indices.size(0));
    int n_conn  = static_cast<int>(indices.size(1));
    int n_batch = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const __half*  d_mat = static_cast<const __half*>(matrix.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    dim3 grid(n_pre, (n_batch + 31) / 32);
    _bgm_float_gather_warp_kern_f16<<<grid, 32, 0, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo);
}

void binary_fcnmm_gather_float_basic_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre   = static_cast<int>(indices.size(0));
    int n_conn  = static_cast<int>(indices.size(1));
    int n_batch = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const __half*  d_mat = static_cast<const __half*>(matrix.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    dim3 grid(n_pre, (n_batch + 31) / 32);
    _bgm_float_gather_basic_kern_f16<<<grid, 32, 0, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo);
}

void binary_fcnmm_scatter_bool_warp_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre   = static_cast<int>(indices.size(0));
    int n_conn  = static_cast<int>(indices.size(1));
    int n_post  = static_cast<int>(output.size(0));
    int n_batch = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const uint8_t* d_mat = static_cast<const uint8_t*>(matrix.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_batch * sizeof(__half), s);
    dim3 grid(n_pre, (n_batch + 31) / 32);
    _bgm_bool_scatter_warp_kern_f16<<<grid, 32, 0, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo);
}

void binary_fcnmm_scatter_bool_basic_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre   = static_cast<int>(indices.size(0));
    int n_conn  = static_cast<int>(indices.size(1));
    int n_post  = static_cast<int>(output.size(0));
    int n_batch = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const uint8_t* d_mat = static_cast<const uint8_t*>(matrix.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_batch * sizeof(__half), s);
    size_t shm = sizeof(int);
    _bgm_bool_scatter_basic_kern_f16<<<n_pre, 256, shm, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo);
}

void binary_fcnmm_scatter_float_warp_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre   = static_cast<int>(indices.size(0));
    int n_conn  = static_cast<int>(indices.size(1));
    int n_post  = static_cast<int>(output.size(0));
    int n_batch = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const __half*  d_mat = static_cast<const __half*>(matrix.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_batch * sizeof(__half), s);
    dim3 grid(n_pre, (n_batch + 31) / 32);
    _bgm_float_scatter_warp_kern_f16<<<grid, 32, 0, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo);
}

void binary_fcnmm_scatter_float_basic_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre   = static_cast<int>(indices.size(0));
    int n_conn  = static_cast<int>(indices.size(1));
    int n_post  = static_cast<int>(output.size(0));
    int n_batch = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const __half*  d_mat = static_cast<const __half*>(matrix.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_batch * sizeof(__half), s);
    size_t shm = sizeof(int);
    _bgm_float_scatter_basic_kern_f16<<<n_pre, 256, shm, s>>>(d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo);
}
