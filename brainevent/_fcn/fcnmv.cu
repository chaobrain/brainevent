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
 * fcnmv.cu — FCN Sparse Matrix-Vector (float) CUDA Kernels
 * =========================================================
 *
 * Python API: brainevent.fcnmv(weights, indices, vector, *, shape, transpose, backend)
 *
 * Sparse matrix--vector product with fixed connection number.
 *
 * Computes  y = W @ v  (or  y = W^T @ v  when transpose=True)
 * where W is a sparse weight matrix stored in fixed-connection-number
 * format and v is a dense floating-point vector.
 *
 * Parameters
 * ----------
 * weights : shape (1,) for homogeneous or (num_pre, num_conn) for heterogeneous,
 *           floating-point dtype (float16 / float32 / float64).
 * indices : shape (num_pre, num_conn), int32 — post-synaptic column indices.
 * vector  : shape (num_post,) for gather, (num_pre,) for scatter.
 * shape   : logical (num_pre, num_post) shape of the dense weight matrix.
 * transpose=False (gather mode):
 *   y[i] = sum_{k} weights[i,k] * v[indices[i,k]]
 * transpose=True  (scatter mode):
 *   y[indices[i,k]] += weights[i,k] * v[i]   for all i,k
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
// Warp reduction helper
// =========================================================================

__device__ __inline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// =========================================================================
// Gather device kernels (transpose=False)
// y[i] = sum_k w[i,k] * v[idx[i,k]]
//
// weights: device pointer. is_homo=1 => shape (1,); is_homo=0 => shape (n_pre,n_conn)
// GPU threads read weights[0] for homo, weights[row*n_conn+k] for hetero.
// =========================================================================

// One warp (32 threads) per output row. Handles any n_conn via loop.
__global__ void _gather_warp_kern(
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
    for (int k = threadIdx.x; k < n_conn; k += 32)
        val += is_homo ? vector[i_row[k]] : (w_row[k] * vector[i_row[k]]);
    val = warp_reduce_sum(val);
    if (threadIdx.x == 0)
        output[row] = is_homo ? (weights[0] * val) : val;
}

// One block (256 threads) per output row with inline block reduction.
// Uses 32*sizeof(float) of dynamic shared memory for the reduction scratchpad.
// Best for 32 < n_conn <= 512.
__global__ void _gather_basic_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ vector,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    extern __shared__ float smem_red[];  // 32 floats for block reduction
    int row = blockIdx.x;
    if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float val = 0.0f;
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x)
        val += is_homo ? vector[i_row[k]] : (w_row[k] * vector[i_row[k]]);
    // Inline block reduction via dynamic shared memory
    int lane   = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) smem_red[warpid] = val;
    __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < n_warps) ? smem_red[lane] : 0.0f;
    if (warpid == 0) val = warp_reduce_sum(val);
    if (threadIdx.x == 0)
        output[row] = is_homo ? (weights[0] * val) : val;
}

// One block per row, cooperative tile-load of indices+weights into shared memory.
// Shared memory layout (dynamic, size = blockDim.x*(4+4) + 32*4 bytes):
//   [0 .. blockDim.x*4)              : int32_t s_idx[blockDim.x]
//   [blockDim.x*4 .. 2*blockDim.x*4) : float   s_wt[blockDim.x]
//   [2*blockDim.x*4 .. +32*4)        : float   s_red[32]  (block reduction)
// Best for n_conn > 512.
__global__ void _gather_shared_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ vector,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    extern __shared__ char smem_raw[];
    int32_t* s_idx = reinterpret_cast<int32_t*>(smem_raw);
    float*   s_wt  = reinterpret_cast<float*>(smem_raw + blockDim.x * sizeof(int32_t));
    float*   s_red = reinterpret_cast<float*>(smem_raw + blockDim.x * (sizeof(int32_t) + sizeof(float)));

    int row = blockIdx.x;
    if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;

    float val = 0.0f;
    for (int base = 0; base < n_conn; base += blockDim.x) {
        int k = base + threadIdx.x;
        if (k < n_conn) {
            s_idx[threadIdx.x] = i_row[k];
            s_wt[threadIdx.x]  = is_homo ? 1.0f : w_row[k];
        }
        __syncthreads();
        int tile = min((int)blockDim.x, n_conn - base);
        if (threadIdx.x < tile)
            val += s_wt[threadIdx.x] * vector[s_idx[threadIdx.x]];
        __syncthreads();
    }

    // Inline block reduction using s_red from dynamic shared memory
    int lane   = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) s_red[warpid] = val;
    __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < n_warps) ? s_red[lane] : 0.0f;
    if (warpid == 0) val = warp_reduce_sum(val);

    if (threadIdx.x == 0)
        output[row] = is_homo ? (weights[0] * val) : val;
}

// One block per row, float4/int4 vectorised loads with inline block reduction.
// Uses 32*sizeof(float) of dynamic shared memory for the reduction scratchpad.
// Best when n_conn % 4 == 0.
__global__ void _gather_vec4_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ vector,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    extern __shared__ float smem_red[];  // 32 floats for block reduction
    int row = blockIdx.x;
    if (row >= n_pre) return;
    size_t base = (size_t)row * n_conn;
    const int4*   i4 = reinterpret_cast<const int4*>(indices + base);
    const float4* w4 = is_homo ? nullptr : reinterpret_cast<const float4*>(weights + base);
    int n4 = n_conn >> 2;
    float val = 0.0f;
    for (int k = threadIdx.x; k < n4; k += blockDim.x) {
        int4 idx = i4[k];
        if (!is_homo) {
            float4 ww = w4[k];
            val += ww.x * vector[idx.x] + ww.y * vector[idx.y]
                 + ww.z * vector[idx.z] + ww.w * vector[idx.w];
        } else {
            val += vector[idx.x] + vector[idx.y] + vector[idx.z] + vector[idx.w];
        }
    }
    // Scalar remainder (when n_conn % 4 != 0)
    for (int k = (n4 << 2) + threadIdx.x; k < n_conn; k += blockDim.x) {
        float v = vector[indices[base + k]];
        val += is_homo ? v : (weights[base + k] * v);
    }
    // Inline block reduction via dynamic shared memory
    int lane   = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;
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
// Scatter device kernels (transpose=True)
// y[idx[i,k]] += w[i,k] * v[i]   (output must be pre-zeroed)
// =========================================================================

// One block per pre-neuron, threads stride over n_conn.
__global__ void _scatter_basic_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ vector,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;
    float v = vector[row];
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x)
        atomicAdd(&output[i_row[k]], (is_homo ? weights[0] : w_row[k]) * v);
}

// 8 warps per block (256 threads), one warp per pre-neuron.
// Grid = ceil(n_pre / 8) blocks.
__global__ void _scatter_warp_kern(
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
        float v = vector[row];
        const int32_t* i_row = indices + (size_t)row * n_conn;
        const float*   w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
        for (int k = lane_id; k < n_conn; k += 32)
            atomicAdd(&output[i_row[k]], (is_homo ? weights[0] : w_row[k]) * v);
    }
}

// Flat grid-stride over all (pre, conn) pairs. Maximises SM occupancy.
__global__ void _scatter_gs_kern(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ vector,
    float*         __restrict__ output,
    const float*   __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int total  = n_pre * n_conn;
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int idx = tid; idx < total; idx += stride) {
        int row = idx / n_conn;
        float w = is_homo ? weights[0] : weights[idx];
        atomicAdd(&output[indices[idx]], w * vector[row]);
    }
}

// =========================================================================
// TVM FFI Entry Points
// =========================================================================
// Convention: args = (weights, indices, vector, output, stream)
// weights: shape (1,) for homo or (n_pre, n_conn) for hetero, float32
// indices: shape (n_pre, n_conn), int32
// vector:  shape (n_post,) for gather, (n_pre,) for scatter, float32
// output:  shape (n_pre,) for gather, (n_post,) for scatter, float32
//
// IMPORTANT: weights.data_ptr() is a GPU device pointer.
// NEVER dereference it on the host. Pass it to kernels unchanged.
// GPU threads read weights[0] (homo) or weights[row*n_conn+k] (hetero).

// --- Gather entry points ---

void fcnmv_gather_warp(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector,
    tvm::ffi::TensorView output,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;   // metadata: host-safe
    const float*   d_weights = static_cast<const float*>(weights.data_ptr());  // device ptr, not dereferenced
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    _gather_warp_kern<<<n_pre, 32, 0, s>>>(d_idx, d_vec, d_out, d_weights, n_pre, n_conn, is_homo);
}

void fcnmv_gather_basic(
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
    const float*   d_weights = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    size_t shm = 32 * sizeof(float);
    _gather_basic_kern<<<n_pre, 256, shm, s>>>(d_idx, d_vec, d_out, d_weights, n_pre, n_conn, is_homo);
}

void fcnmv_gather_shared(
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
    const float*   d_weights = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    int threads = 256;
    // Dynamic shared mem: s_idx[threads] + s_wt[threads] + s_red[32]
    size_t shm = (size_t)threads * (sizeof(int32_t) + sizeof(float)) + 32 * sizeof(float);
    _gather_shared_kern<<<n_pre, threads, shm, s>>>(d_idx, d_vec, d_out, d_weights, n_pre, n_conn, is_homo);
}

void fcnmv_gather_vec4(
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
    const float*   d_weights = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    size_t shm = 32 * sizeof(float);
    _gather_vec4_kern<<<n_pre, 256, shm, s>>>(d_idx, d_vec, d_out, d_weights, n_pre, n_conn, is_homo);
}

// Auto-selects the best gather kernel based on n_conn.
void fcnmv_gather_auto(
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
    const float*   d_weights = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());

    size_t shm_red = 32 * sizeof(float);
    if (n_conn <= 32) {
        _gather_warp_kern<<<n_pre, 32, 0, s>>>(d_idx, d_vec, d_out, d_weights, n_pre, n_conn, is_homo);
    } else if (n_conn % 4 == 0 && n_conn >= 128) {
        _gather_vec4_kern<<<n_pre, 256, shm_red, s>>>(d_idx, d_vec, d_out, d_weights, n_pre, n_conn, is_homo);
    } else if (n_conn > 512) {
        int threads = 256;
        size_t shm = (size_t)threads * (sizeof(int32_t) + sizeof(float)) + 32 * sizeof(float);
        _gather_shared_kern<<<n_pre, threads, shm, s>>>(d_idx, d_vec, d_out, d_weights, n_pre, n_conn, is_homo);
    } else {
        _gather_basic_kern<<<n_pre, 256, shm_red, s>>>(d_idx, d_vec, d_out, d_weights, n_pre, n_conn, is_homo);
    }
}

// --- Scatter entry points (output zeroed before kernel launch) ---

void fcnmv_scatter_basic(
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
    const float*   d_weights = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(float), s);
    _scatter_basic_kern<<<n_pre, 256, 0, s>>>(d_idx, d_vec, d_out, d_weights, n_pre, n_conn, is_homo);
}

void fcnmv_scatter_warp(
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
    const float*   d_weights = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(float), s);
    // 256 threads per block = 8 warps; grid = ceil(n_pre / 8)
    int blocks = (n_pre + 7) / 8;
    _scatter_warp_kern<<<blocks, 256, 0, s>>>(d_idx, d_vec, d_out, d_weights, n_pre, n_conn, is_homo);
}

void fcnmv_scatter_gridstride(
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
    const float*   d_weights = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(float), s);
    int blocks = min(1024, (n_pre * n_conn + 255) / 256);
    _scatter_gs_kern<<<blocks, 256, 0, s>>>(d_idx, d_vec, d_out, d_weights, n_pre, n_conn, is_homo);
}

// Auto-selects the best scatter kernel based on problem size.
void fcnmv_scatter_auto(
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
    const float*   d_weights = static_cast<const float*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const float*   d_vec = static_cast<const float*>(vector.data_ptr());
    float*         d_out = static_cast<float*>(output.data_ptr());
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(float), s);

    if (n_conn <= 32) {
        // One warp per pre-neuron
        int blocks = (n_pre + 7) / 8;
        _scatter_warp_kern<<<blocks, 256, 0, s>>>(d_idx, d_vec, d_out, d_weights, n_pre, n_conn, is_homo);
    } else if ((long long)n_pre * n_conn > 262144LL) {
        // Large problem: grid-stride maximises occupancy
        int blocks = min(1024, (n_pre * n_conn + 255) / 256);
        _scatter_gs_kern<<<blocks, 256, 0, s>>>(d_idx, d_vec, d_out, d_weights, n_pre, n_conn, is_homo);
    } else {
        // Medium problem: one block per pre-neuron
        _scatter_basic_kern<<<n_pre, 256, 0, s>>>(d_idx, d_vec, d_out, d_weights, n_pre, n_conn, is_homo);
    }
}

// =========================================================================
// float64 (double) variants
// =========================================================================

__device__ __inline__ double warp_reduce_sum_f64(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void _gather_warp_kern_f64(
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
    for (int k = threadIdx.x; k < n_conn; k += 32)
        val += is_homo ? vector[i_row[k]] : (w_row[k] * vector[i_row[k]]);
    val = warp_reduce_sum_f64(val);
    if (threadIdx.x == 0)
        output[row] = is_homo ? (weights[0] * val) : val;
}

__global__ void _gather_basic_kern_f64(
    const int32_t* __restrict__ indices,
    const double*  __restrict__ vector,
    double*        __restrict__ output,
    const double*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    extern __shared__ double smem_red_f64[];
    int row = blockIdx.x;
    if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const double*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    double val = 0.0;
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x)
        val += is_homo ? vector[i_row[k]] : (w_row[k] * vector[i_row[k]]);
    int lane   = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;
    val = warp_reduce_sum_f64(val);
    if (lane == 0) smem_red_f64[warpid] = val;
    __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < n_warps) ? smem_red_f64[lane] : 0.0;
    if (warpid == 0) val = warp_reduce_sum_f64(val);
    if (threadIdx.x == 0)
        output[row] = is_homo ? (weights[0] * val) : val;
}

__global__ void _scatter_basic_kern_f64(
    const int32_t* __restrict__ indices,
    const double*  __restrict__ vector,
    double*        __restrict__ output,
    const double*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;
    double v = vector[row];
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const double*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x)
        atomicAdd(&output[i_row[k]], (is_homo ? weights[0] : w_row[k]) * v);
}

__global__ void _scatter_warp_kern_f64(
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
        double v = vector[row];
        const int32_t* i_row = indices + (size_t)row * n_conn;
        const double*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
        for (int k = lane_id; k < n_conn; k += 32)
            atomicAdd(&output[i_row[k]], (is_homo ? weights[0] : w_row[k]) * v);
    }
}

void fcnmv_gather_warp_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const double*  d_vec = static_cast<const double*>(vector.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    _gather_warp_kern_f64<<<n_pre, 32, 0, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
}

void fcnmv_gather_basic_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector, tvm::ffi::TensorView output, int64_t stream
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
    _gather_basic_kern_f64<<<n_pre, 256, shm, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
}

void fcnmv_gather_auto_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const double*  d_w   = static_cast<const double*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const double*  d_vec = static_cast<const double*>(vector.data_ptr());
    double*        d_out = static_cast<double*>(output.data_ptr());
    if (n_conn <= 32) {
        _gather_warp_kern_f64<<<n_pre, 32, 0, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    } else {
        size_t shm = 32 * sizeof(double);
        _gather_basic_kern_f64<<<n_pre, 256, shm, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    }
}

void fcnmv_scatter_basic_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector, tvm::ffi::TensorView output, int64_t stream
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
    _scatter_basic_kern_f64<<<n_pre, 256, 0, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
}

void fcnmv_scatter_warp_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector, tvm::ffi::TensorView output, int64_t stream
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
    _scatter_warp_kern_f64<<<blocks, 256, 0, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
}

void fcnmv_scatter_auto_f64(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector, tvm::ffi::TensorView output, int64_t stream
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
        _scatter_warp_kern_f64<<<blocks, 256, 0, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    } else {
        _scatter_basic_kern_f64<<<n_pre, 256, 0, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    }
}

// =========================================================================
// float16 (__half) variants — accumulate in float32 for numerical stability
// Requires compute capability >= 7.0 for atomicAdd(__half*)
// =========================================================================

__global__ void _gather_warp_kern_f16(
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
    for (int k = threadIdx.x; k < n_conn; k += 32) {
        float v = __half2float(vector[i_row[k]]);
        val += is_homo ? v : (__half2float(w_row[k]) * v);
    }
    val = warp_reduce_sum(val);
    if (threadIdx.x == 0)
        output[row] = __float2half(is_homo ? (__half2float(weights[0]) * val) : val);
}

__global__ void _gather_basic_kern_f16(
    const int32_t* __restrict__ indices,
    const __half*  __restrict__ vector,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    extern __shared__ float smem_red_f16h[];
    int row = blockIdx.x;
    if (row >= n_pre) return;
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const __half*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float val = 0.0f;
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x) {
        float v = __half2float(vector[i_row[k]]);
        val += is_homo ? v : (__half2float(w_row[k]) * v);
    }
    int lane   = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) smem_red_f16h[warpid] = val;
    __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < n_warps) ? smem_red_f16h[lane] : 0.0f;
    if (warpid == 0) val = warp_reduce_sum(val);
    if (threadIdx.x == 0)
        output[row] = __float2half(is_homo ? (__half2float(weights[0]) * val) : val);
}

__global__ void _scatter_basic_kern_f16(
    const int32_t* __restrict__ indices,
    const __half*  __restrict__ vector,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int row = blockIdx.x;
    if (row >= n_pre) return;
    float v = __half2float(vector[row]);
    const int32_t* i_row = indices + (size_t)row * n_conn;
    const __half*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
    float w0 = is_homo ? __half2float(weights[0]) : 0.0f;
    for (int k = threadIdx.x; k < n_conn; k += blockDim.x) {
        float w = is_homo ? w0 : __half2float(w_row[k]);
        atomicAdd(&output[i_row[k]], __float2half(w * v));
    }
}

__global__ void _scatter_warp_kern_f16(
    const int32_t* __restrict__ indices,
    const __half*  __restrict__ vector,
    __half*        __restrict__ output,
    const __half*  __restrict__ weights,
    int n_pre, int n_conn, int is_homo
) {
    int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane_id   = threadIdx.x & 31;
    int num_warps = (gridDim.x * blockDim.x) >> 5;
    for (int row = warp_id; row < n_pre; row += num_warps) {
        float v = __half2float(vector[row]);
        const int32_t* i_row = indices + (size_t)row * n_conn;
        const __half*  w_row = is_homo ? nullptr : weights + (size_t)row * n_conn;
        float w0 = is_homo ? __half2float(weights[0]) : 0.0f;
        for (int k = lane_id; k < n_conn; k += 32) {
            float w = is_homo ? w0 : __half2float(w_row[k]);
            atomicAdd(&output[i_row[k]], __float2half(w * v));
        }
    }
}

void fcnmv_gather_warp_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const __half*  d_vec = static_cast<const __half*>(vector.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    _gather_warp_kern_f16<<<n_pre, 32, 0, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
}

void fcnmv_gather_basic_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector, tvm::ffi::TensorView output, int64_t stream
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
    _gather_basic_kern_f16<<<n_pre, 256, shm, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
}

void fcnmv_gather_auto_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector, tvm::ffi::TensorView output, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int n_pre  = static_cast<int>(indices.size(0));
    int n_conn = static_cast<int>(indices.size(1));
    int is_homo = (weights.ndim() == 1) ? 1 : 0;
    const __half*  d_w   = static_cast<const __half*>(weights.data_ptr());
    const int32_t* d_idx = static_cast<const int32_t*>(indices.data_ptr());
    const __half*  d_vec = static_cast<const __half*>(vector.data_ptr());
    __half*        d_out = static_cast<__half*>(output.data_ptr());
    if (n_conn <= 32) {
        _gather_warp_kern_f16<<<n_pre, 32, 0, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    } else {
        size_t shm = 32 * sizeof(float);
        _gather_basic_kern_f16<<<n_pre, 256, shm, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    }
}

void fcnmv_scatter_basic_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector, tvm::ffi::TensorView output, int64_t stream
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
    _scatter_basic_kern_f16<<<n_pre, 256, 0, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
}

void fcnmv_scatter_warp_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector, tvm::ffi::TensorView output, int64_t stream
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
    _scatter_warp_kern_f16<<<blocks, 256, 0, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
}

void fcnmv_scatter_auto_f16(
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,
    tvm::ffi::TensorView vector, tvm::ffi::TensorView output, int64_t stream
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
        _scatter_warp_kern_f16<<<blocks, 256, 0, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    } else {
        _scatter_basic_kern_f16<<<n_pre, 256, 0, s>>>(d_idx, d_vec, d_out, d_w, n_pre, n_conn, is_homo);
    }
}
