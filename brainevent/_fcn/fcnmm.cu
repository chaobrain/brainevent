/*
 * fcnmm.cu — FCN Sparse Matrix-Matrix (float) CUDA Kernels
 * =========================================================
 *
 * Python API: brainevent.fcnmm(weights, indices, matrix, *, shape, transpose, backend)
 *
 * Sparse matrix--matrix product with fixed connection number.
 *
 * Computes  Y = W @ M  (or  Y = W^T @ M  when transpose=True)
 * where W is a sparse weight matrix stored in fixed-connection-number
 * format and M is a dense floating-point matrix.
 *
 * Parameters
 * ----------
 * weights : shape (1,) for homogeneous or (num_pre, num_conn) for heterogeneous,
 *           floating-point dtype (float16 / float32 / float64).
 * indices : shape (num_pre, num_conn), int32 — post-synaptic column indices.
 * matrix  : shape (num_post, n_col) for gather, (num_pre, n_col) for scatter.
 * shape   : logical (num_pre, num_post) shape of the dense weight matrix.
 * transpose=False (gather mode):
 *   Y[i,j] = sum_{k} weights[i,k] * M[indices[i,k], j]
 * transpose=True  (scatter mode):
 *   Y[indices[i,k], j] += weights[i,k] * M[i,j]   for all i,k,j
 *
 * Supported dtypes: float32 (default), float64 (_f64 suffix), float16 (_f16 suffix).
 * vec4 and shared-memory tile paths are float32-only; f16/f64 use basic/warp kernels.
 *
 * IMPORTANT: weights.data_ptr() is a GPU device pointer — NEVER dereference on host.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// =========================================================================
// FCN Matrix-Matrix product CUDA kernels
//
// Gather mode (transpose=False):
//   Y[i, j] = sum_{k=0}^{n_conn-1} w[i,k] * M[indices[i,k], j]
//   indices: [n_pre, n_conn], weights: [1] or [n_pre, n_conn]
//   M: [n_post, n_col],  Y: [n_pre, n_col]
//
// Scatter mode (transpose=True):
//   Y[indices[i,k], j] += w[i,k] * M[i, j]
//   M: [n_pre, n_col],  Y: [n_post, n_col]  (Y pre-zeroed at launch)
//
// IMPORTANT: weights.data_ptr() returns a GPU device pointer.
// NEVER dereference on host. GPU threads read weights[0] (homo)
// or weights[i*n_conn+k] (hetero).
// =========================================================================

// =========================================================================
// Gather kernels (transpose=False)
// =========================================================================

// Basic gather: one thread per output element Y[i,j].
// Grid: (n_pre, ceil(n_col / 64)), Block: (64,)
// Threads in a warp read consecutive j positions of M[same_row, j..j+31]
// giving coalesced loads within each k iteration.
__global__ void _mm_gather_basic_kern(
    const int32_t* __restrict__ indices,   // [n_pre, n_conn]
    const float*   __restrict__ matrix,    // [k_dim, n_col]
    float*         __restrict__ output,    // [n_pre, n_col]
    const float*   __restrict__ weights,   // [1] or [n_pre, n_conn]
    int n_pre, int n_conn, int n_col, int is_homo
) {
    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= n_pre || j >= n_col) return;
    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const float*   w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;
    float acc = 0.0f;
    for (int k = 0; k < n_conn; k++) {
        float w = is_homo ? weights[0] : w_row[k];
        acc += w * matrix[(size_t)idx_row[k] * n_col + j];
    }
    output[(size_t)i * n_col + j] = acc;
}

// Shared-memory gather: tiles the index and weight arrays into shared memory
// to reduce global memory traffic for the connection lists.
// Grid: (n_pre, ceil(n_col / 64)), Block: (64,)
// Shared mem: MMTK * 8 bytes (idx tile + weight tile).
// Best when n_conn is large (> 128) and index/weight bandwidth dominates.
#define MMTK 128
__global__ void _mm_gather_shared_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ matrix,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    extern __shared__ char smem_mm[];
    int32_t* s_idx = reinterpret_cast<int32_t*>(smem_mm);
    float*   s_w   = reinterpret_cast<float*>(smem_mm + MMTK * sizeof(int32_t));

    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= n_pre) return;

    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const float*   w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;

    float acc = 0.0f;
    for (int k0 = 0; k0 < n_conn; k0 += MMTK) {
        int tile = (k0 + MMTK < n_conn) ? MMTK : (n_conn - k0);
        // Cooperatively load the connection tile into shared memory.
        for (int t = threadIdx.x; t < tile; t += blockDim.x) {
            s_idx[t] = idx_row[k0 + t];
            s_w[t]   = is_homo ? 1.0f : w_row[k0 + t];
        }
        __syncthreads();
        // Accumulate contributions from this tile.
        if (j < n_col) {
            for (int t = 0; t < tile; t++)
                acc += s_w[t] * matrix[(size_t)s_idx[t] * n_col + j];
        }
        __syncthreads();
    }
    if (j < n_col)
        output[(size_t)i * n_col + j] = is_homo ? (weights[0] * acc) : acc;
}

// Vectorised gather: float4 loads for M and output when n_col % 4 == 0.
// Each thread processes 4 consecutive j columns simultaneously => 4x throughput.
// Grid: (n_pre, ceil(n_col/4 / 64)), Block: (64,)
// Best when n_col is divisible by 4 and >= 64 (memory-bandwidth bound).
__global__ void _mm_gather_vec4_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ matrix,   // [k_dim, n_col], n_col%4==0
    float*         __restrict__ output,   // [n_pre, n_col]
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    int i   = blockIdx.x;
    int j4  = blockIdx.y * blockDim.x + threadIdx.x;  // float4 group index
    int nc4 = n_col >> 2;
    if (i >= n_pre || j4 >= nc4) return;

    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const float*   w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;
    const float4*  mat4    = reinterpret_cast<const float4*>(matrix);

    float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = 0; k < n_conn; k++) {
        float  w = is_homo ? weights[0] : w_row[k];
        float4 m = mat4[(size_t)idx_row[k] * nc4 + j4];
        acc.x += w * m.x;
        acc.y += w * m.y;
        acc.z += w * m.z;
        acc.w += w * m.w;
    }
    reinterpret_cast<float4*>(output)[(size_t)i * nc4 + j4] = acc;
}

// =========================================================================
// Scatter kernels (transpose=True)
// Y[indices[i,k], j] += w[i,k] * M[i, j]   (Y pre-zeroed before launch)
// =========================================================================

// Block scatter: one block per pre-neuron, threads stride over j columns.
// For each i, sequentially iterates over n_conn connections and atomically
// accumulates to the target output row. Good for large n_col.
// Grid: (n_pre,), Block: (256,)
__global__ void _mm_scatter_block_kern(
    const int32_t* __restrict__ indices,   // [n_pre, n_conn]
    const float*   __restrict__ matrix,    // [n_pre, n_col]
    float*         __restrict__ output,    // [n_post, n_col] (pre-zeroed)
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    int i = blockIdx.x;
    if (i >= n_pre) return;
    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const float*   w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;
    const float*   m_row   = matrix + (size_t)i * n_col;
    for (int k = 0; k < n_conn; k++) {
        int   tgt = idx_row[k];
        float w   = is_homo ? weights[0] : w_row[k];
        float* out_row = output + (size_t)tgt * n_col;
        for (int j = threadIdx.x; j < n_col; j += blockDim.x)
            atomicAdd(&out_row[j], w * m_row[j]);
    }
}

// Cached scatter: 2D grid — one block per (pre-neuron, n_col tile).
// Tiles M[i] into shared memory once, then performs all n_conn atomic
// scatter operations from shmem. Eliminates repeated DRAM reads of M[i]
// and is especially efficient for large n_conn with moderate n_col.
// Grid: (n_pre, ceil(n_col / BLOCK_J)), Block: (BLOCK_J,)
// Shared mem: BLOCK_J * sizeof(float) bytes
#define MM_SCATTER_BJ 128
__global__ void _mm_scatter_cached_kern(
    const int32_t* __restrict__ indices,   // [n_pre, n_conn]
    const float*   __restrict__ matrix,    // [n_pre, n_col]
    float*         __restrict__ output,    // [n_post, n_col] (pre-zeroed)
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    extern __shared__ float s_m[];  // cache one M[i] column tile
    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= n_pre) return;

    // Load M[i, j] tile into shared memory once.
    s_m[threadIdx.x] = (j < n_col) ? matrix[(size_t)i * n_col + j] : 0.0f;
    __syncthreads();

    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const float*   w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;

    if (j < n_col) {
        // All connections from row i scatter to their targets using shmem value.
        float m_val = s_m[threadIdx.x];
        for (int k = 0; k < n_conn; k++) {
            int   tgt = idx_row[k];
            float w   = is_homo ? weights[0] : w_row[k];
            atomicAdd(&output[(size_t)tgt * n_col + j], w * m_val);
        }
    }
}

// Warp scatter: grid-stride over (pre-neuron, connection) pairs.
// Each warp handles one (i, k) pair; lanes stride over j columns.
// Maximises parallelism over both i and k dimensions. Good for small n_col.
// Grid: (min(4096, ceil(n_pre*n_conn/8)),), Block: (256,)  [8 warps per block]
__global__ void _mm_scatter_warp_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ matrix,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    int wid     = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;  // global warp id
    int lane    = threadIdx.x & 31;
    int n_warps = (gridDim.x * blockDim.x) >> 5;
    int n_pairs = n_pre * n_conn;

    for (int pair = wid; pair < n_pairs; pair += n_warps) {
        int i = pair / n_conn;
        int k = pair % n_conn;
        int   tgt = indices[(size_t)i * n_conn + k];
        float w   = is_homo ? weights[0] : weights[(size_t)i * n_conn + k];
        const float* m_row   = matrix + (size_t)i * n_col;
        float*       out_row = output + (size_t)tgt * n_col;
        for (int j = lane; j < n_col; j += 32)
            atomicAdd(&out_row[j], w * m_row[j]);
    }
}

// =========================================================================
// TVM FFI Entry Points
// =========================================================================
// Convention: args = (weights, indices, matrix, output, stream)
// weights: [1] (homo) or [n_pre, n_conn] (hetero), float32
// indices: [n_pre, n_conn], int32
// Gather:  matrix [n_post, n_col], output [n_pre, n_col]
// Scatter: matrix [n_pre,  n_col], output [n_post, n_col]
//
// weights.data_ptr() is a GPU device pointer — NEVER dereference on host.
// GPU threads read weights[0] (homo) or weights[i*n_conn+k] (hetero).

// --- Gather entry points ---

void fcnmm_gather_basic(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_mat = static_cast<const float*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    int BJ = 64;
    dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
    _mm_gather_basic_kern<<<grid, BJ, 0, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

void fcnmm_gather_shared(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_mat = static_cast<const float*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    int BJ = 64;
    dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
    size_t shm = MMTK * (sizeof(int32_t) + sizeof(float));
    _mm_gather_shared_kern<<<grid, BJ, shm, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

void fcnmm_gather_vec4(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_mat = static_cast<const float*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    int BJ4 = 64;
    dim3 grid(n_pre, (n_col / 4 + BJ4 - 1) / BJ4);
    _mm_gather_vec4_kern<<<grid, BJ4, 0, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

// Auto-selects the best gather kernel based on n_conn and n_col.
void fcnmm_gather_auto(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_mat = static_cast<const float*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());

    int BJ = 64;
    if (n_col % 4 == 0 && n_col >= 64) {
        // Vectorised float4 path: 4x throughput for aligned n_col.
        dim3 grid(n_pre, (n_col / 4 + BJ - 1) / BJ);
        _mm_gather_vec4_kern<<<grid, BJ, 0, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    } else if (n_conn > 128) {
        // Shared-memory path: amortises index/weight bandwidth for large n_conn.
        dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
        size_t shm = MMTK * (sizeof(int32_t) + sizeof(float));
        _mm_gather_shared_kern<<<grid, BJ, shm, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    } else {
        // Basic path: good general-purpose option.
        dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
        _mm_gather_basic_kern<<<grid, BJ, 0, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    }
}

// --- Scatter entry points (output zeroed before kernel launch) ---

void fcnmm_scatter_block(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_mat = static_cast<const float*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_col * sizeof(float), s);
    _mm_scatter_block_kern<<<n_pre, 256, 0, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

void fcnmm_scatter_warp(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_mat = static_cast<const float*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_col * sizeof(float), s);
    int n_pairs = n_pre * n_conn;
    int blocks  = min(4096, (n_pairs + 7) / 8);
    _mm_scatter_warp_kern<<<blocks, 256, 0, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

void fcnmm_scatter_cached(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_mat = static_cast<const float*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_col * sizeof(float), s);
    int BJ = MM_SCATTER_BJ;
    dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
    size_t shm = BJ * sizeof(float);
    _mm_scatter_cached_kern<<<grid, BJ, shm, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

// Auto-selects the best scatter kernel based on problem size.
void fcnmm_scatter_auto(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const float*   d_w   = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_mat = static_cast<const float*>(matrix.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_col * sizeof(float), s);

    if (n_col <= 64) {
        // Small n_col: one warp per (i,k) pair maximises SM utilisation.
        int n_pairs = n_pre * n_conn;
        int blocks  = min(4096, (n_pairs + 7) / 8);
        _mm_scatter_warp_kern<<<blocks, 256, 0, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    } else if (n_conn > 32) {
        // Moderate-to-large n_conn: cached M[i] in shmem cuts DRAM reads.
        int BJ = MM_SCATTER_BJ;
        dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
        size_t shm = BJ * sizeof(float);
        _mm_scatter_cached_kern<<<grid, BJ, shm, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    } else {
        // Small n_conn: simple block-per-pre-neuron with thread stride.
        _mm_scatter_block_kern<<<n_pre, 256, 0, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    }
}

// =========================================================================
// float64 (double) matrix-matrix variants
// =========================================================================

__global__ void _mm_gather_basic_kern_f64(
    const int32_t* __restrict__ indices,
    const double*  __restrict__ matrix,
    double*        __restrict__ output,
    const double*  __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= n_pre || j >= n_col) return;
    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const double*  w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;
    double acc = 0.0;
    for (int k = 0; k < n_conn; k++) {
        double w = is_homo ? weights[0] : w_row[k];
        acc += w * matrix[(size_t)idx_row[k] * n_col + j];
    }
    output[(size_t)i * n_col + j] = acc;
}

__global__ void _mm_scatter_block_kern_f64(
    const int32_t* __restrict__ indices,
    const double*  __restrict__ matrix,
    double*        __restrict__ output,
    const double*  __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    int i = blockIdx.x;
    if (i >= n_pre) return;
    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const double*  w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;
    const double*  m_row   = matrix + (size_t)i * n_col;
    for (int k = 0; k < n_conn; k++) {
        int   tgt = idx_row[k];
        double w   = is_homo ? weights[0] : w_row[k];
        double* out_row = output + (size_t)tgt * n_col;
        for (int j = threadIdx.x; j < n_col; j += blockDim.x)
            atomicAdd(&out_row[j], w * m_row[j]);
    }
}

__global__ void _mm_scatter_warp_kern_f64(
    const int32_t* __restrict__ indices,
    const double*  __restrict__ matrix,
    double*        __restrict__ output,
    const double*  __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    int wid     = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane    = threadIdx.x & 31;
    int n_warps = (gridDim.x * blockDim.x) >> 5;
    int n_pairs = n_pre * n_conn;
    for (int pair = wid; pair < n_pairs; pair += n_warps) {
        int i = pair / n_conn;
        int k = pair % n_conn;
        int    tgt = indices[(size_t)i * n_conn + k];
        double w   = is_homo ? weights[0] : weights[(size_t)i * n_conn + k];
        const double* m_row   = matrix + (size_t)i * n_col;
        double*       out_row = output + (size_t)tgt * n_col;
        for (int j = lane; j < n_col; j += 32)
            atomicAdd(&out_row[j], w * m_row[j]);
    }
}

void fcnmm_gather_basic_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const double*  d_mat = static_cast<const double*>(matrix.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    int BJ = 64;
    dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
    _mm_gather_basic_kern_f64<<<grid, BJ, 0, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

void fcnmm_gather_auto_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const double*  d_mat = static_cast<const double*>(matrix.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    int BJ = 64;
    dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
    _mm_gather_basic_kern_f64<<<grid, BJ, 0, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

void fcnmm_scatter_block_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const double*  d_mat = static_cast<const double*>(matrix.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_col * sizeof(double), s);
    _mm_scatter_block_kern_f64<<<n_pre, 256, 0, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

void fcnmm_scatter_warp_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const double*  d_mat = static_cast<const double*>(matrix.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_col * sizeof(double), s);
    int n_pairs = n_pre * n_conn;
    int blocks  = min(4096, (n_pairs + 7) / 8);
    _mm_scatter_warp_kern_f64<<<blocks, 256, 0, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

void fcnmm_scatter_auto_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const double*  d_mat = static_cast<const double*>(matrix.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_col * sizeof(double), s);
    if (n_col <= 64) {
        int n_pairs = n_pre * n_conn;
        int blocks  = min(4096, (n_pairs + 7) / 8);
        _mm_scatter_warp_kern_f64<<<blocks, 256, 0, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    } else {
        _mm_scatter_block_kern_f64<<<n_pre, 256, 0, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    }
}

// =========================================================================
// float16 (__half) matrix-matrix variants — accumulate in float32
// =========================================================================

__global__ void _mm_gather_basic_kern_f16(
    const int32_t* __restrict__ indices,
    const __half*  __restrict__ matrix,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= n_pre || j >= n_col) return;
    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const __half*  w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;
    float acc = 0.0f;
    for (int k = 0; k < n_conn; k++) {
        float w = is_homo ? __half2float(weights[0]) : __half2float(w_row[k]);
        acc += w * __half2float(matrix[(size_t)idx_row[k] * n_col + j]);
    }
    output[(size_t)i * n_col + j] = __float2half(acc);
}

__global__ void _mm_scatter_block_kern_f16(
    const int32_t* __restrict__ indices,
    const __half*  __restrict__ matrix,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    int i = blockIdx.x;
    if (i >= n_pre) return;
    const int32_t* idx_row = indices + (size_t)i * n_conn;
    const __half*  w_row   = is_homo ? nullptr : weights + (size_t)i * n_conn;
    const __half*  m_row   = matrix + (size_t)i * n_col;
    for (int k = 0; k < n_conn; k++) {
        int   tgt = idx_row[k];
        float w   = is_homo ? __half2float(weights[0]) : __half2float(w_row[k]);
        __half* out_row = output + (size_t)tgt * n_col;
        for (int j = threadIdx.x; j < n_col; j += blockDim.x)
            atomicAdd(&out_row[j], __float2half(w * __half2float(m_row[j])));
    }
}

__global__ void _mm_scatter_warp_kern_f16(
    const int32_t* __restrict__ indices,
    const __half*  __restrict__ matrix,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int n_col, int is_homo
) {
    int wid     = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane    = threadIdx.x & 31;
    int n_warps = (gridDim.x * blockDim.x) >> 5;
    int n_pairs = n_pre * n_conn;
    for (int pair = wid; pair < n_pairs; pair += n_warps) {
        int i = pair / n_conn;
        int k = pair % n_conn;
        int   tgt = indices[(size_t)i * n_conn + k];
        float w   = is_homo ? __half2float(weights[0]) : __half2float(weights[(size_t)i * n_conn + k]);
        const __half* m_row   = matrix + (size_t)i * n_col;
        __half*       out_row = output + (size_t)tgt * n_col;
        for (int j = lane; j < n_col; j += 32)
            atomicAdd(&out_row[j], __float2half(w * __half2float(m_row[j])));
    }
}

void fcnmm_gather_basic_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const __half*  d_mat = static_cast<const __half*>(matrix.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    int BJ = 64;
    dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
    _mm_gather_basic_kern_f16<<<grid, BJ, 0, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

void fcnmm_gather_auto_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const __half*  d_mat = static_cast<const __half*>(matrix.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    int BJ = 64;
    dim3 grid(n_pre, (n_col + BJ - 1) / BJ);
    _mm_gather_basic_kern_f16<<<grid, BJ, 0, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

void fcnmm_scatter_block_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const __half*  d_mat = static_cast<const __half*>(matrix.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_col * sizeof(__half), s);
    _mm_scatter_block_kern_f16<<<n_pre, 256, 0, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

void fcnmm_scatter_warp_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const __half*  d_mat = static_cast<const __half*>(matrix.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_col * sizeof(__half), s);
    int n_pairs = n_pre * n_conn;
    int blocks  = min(4096, (n_pairs + 7) / 8);
    _mm_scatter_warp_kern_f16<<<blocks, 256, 0, s>>>(
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
}

void fcnmm_scatter_auto_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView matrix, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int n_post = static_cast<int>(output.size(0));
    int n_col  = static_cast<int>(matrix.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const __half*  d_mat = static_cast<const __half*>(matrix.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_col * sizeof(__half), s);
    if (n_col <= 64) {
        int n_pairs = n_pre * n_conn;
        int blocks  = min(4096, (n_pairs + 7) / 8);
        _mm_scatter_warp_kern_f16<<<blocks, 256, 0, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    } else {
        _mm_scatter_block_kern_f16<<<n_pre, 256, 0, s>>>(
            d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_col, is_homo);
    }
}
