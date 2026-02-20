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
 * csrmv.cu -- Float-Weighted CSR Sparse Matrix-Vector CUDA Kernels
 * ================================================================
 *
 * Python API:
 *   brainevent.csrmv(data, indices, indptr, v, shape=(m,k), transpose=False)
 *
 * Computes y = A @ v  (transpose=False) or  y = A.T @ v  (transpose=True)
 * where A is stored in CSR format and v is a dense float vector.
 *
 * Unlike binary_csrmv.cu, every element of v contributes regardless of
 * sign or magnitude -- no active-event predicate is applied.
 *
 * Non-transpose (NT, gather mode):
 *   A[m,k] @ v[k]  ->  out[m]
 *   out[i] = sum_{j in nz(i)} A[i,j] * v[indices[j]]
 *
 * Transpose (T, scatter mode):
 *   A.T[k,m] @ v[m]  ->  out[k]
 *   out[indices[j]] += A[i,j] * v[i]   for all j in row i, all i
 *
 * Kernel variants
 * ---------------
 * Non-transpose:
 *   NT_thread:  1 thread per row (256 threads/block). Best for avg_nnz < 8.
 *   NT_warp:    1 warp  per row (32 threads/block).  Best for avg_nnz 8-512.
 *   NT_block:   1 block per row (256 threads/block). Best for avg_nnz > 512.
 *   NT_auto:    Host-side auto-dispatch to thread/warp/block.
 *
 * Transpose:
 *   T_warp:     1 warp per row; warp-stride atomicAdd scatter to output.
 *
 * Weight modes (runtime, not compile-time):
 *   is_homo=1:  homogeneous — all nonzeros share data[0].
 *   is_homo=0:  heterogeneous — nonzero j uses data[j].
 *
 * Dtype support
 * -------------
 *   Weight and vector dtypes: float32, float64, float16, bfloat16.
 *   Float16 and bfloat16 accumulate in float32 for numerical stability.
 *   Bfloat16 transpose atomicAdd requires sm_80+ (Ampere or newer).
 *   Float16  transpose atomicAdd requires sm_70+ (Volta or newer).
 *
 * IMPORTANT: All data_ptr() values are GPU device pointers.
 *            NEVER dereference on the host. Extract only metadata (size, ndim).
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
// Per-dtype conversion macros (READ: WEIGHT_T -> ACC_T, WRITE: ACC_T -> WEIGHT_T)
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
// Non-transpose Thread kernel
//
// One thread per output row.  Each thread serially accumulates over all
// nonzeros in its row.  Best for very sparse rows (avg_nnz < 8).
//
// Grid: (ceil(m / 256), 1, 1)   Block: (256, 1, 1)
// =========================================================================

#define DEFINE_CSRMV_NT_THREAD(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO)  \
__global__ void _csrmv_nt_thread_kern##SUFFIX(                                        \
    const WEIGHT_T* __restrict__ weights,                                             \
    const int32_t*  __restrict__ indices,                                             \
    const int32_t*  __restrict__ indptr,                                              \
    const WEIGHT_T* __restrict__ vector,                                              \
    WEIGHT_T*       __restrict__ output,                                              \
    int m, int is_homo                                                                \
) {                                                                                    \
    int row = blockIdx.x * blockDim.x + threadIdx.x;                                 \
    if (row >= m) return;                                                             \
    int start = indptr[row], end = indptr[row + 1];                                   \
    ACC_T acc = ACC_ZERO;                                                             \
    if (is_homo) {                                                                    \
        ACC_T w = READ_W(weights[0]);                                                 \
        for (int j = start; j < end; j++) {                                           \
            acc += w * READ_W(vector[indices[j]]);                                    \
        }                                                                              \
    } else {                                                                          \
        for (int j = start; j < end; j++) {                                           \
            acc += READ_W(weights[j]) * READ_W(vector[indices[j]]);                   \
        }                                                                              \
    }                                                                                  \
    output[row] = WRITE_W(acc);                                                       \
}

// =========================================================================
// Non-transpose Warp kernel
//
// One warp (32 threads) per output row.  Threads stride by 32 over the
// row's nonzeros and accumulate into a warp-level register, then reduce
// with __shfl_down_sync.  Best for moderate-density rows (avg_nnz 8-512).
//
// Grid: (m, 1, 1)   Block: (32, 1, 1)
// =========================================================================

#define DEFINE_CSRMV_NT_WARP(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO)  \
__global__ void _csrmv_nt_warp_kern##SUFFIX(                                                  \
    const WEIGHT_T* __restrict__ weights,                                                     \
    const int32_t*  __restrict__ indices,                                                     \
    const int32_t*  __restrict__ indptr,                                                      \
    const WEIGHT_T* __restrict__ vector,                                                      \
    WEIGHT_T*       __restrict__ output,                                                      \
    int m, int is_homo                                                                        \
) {                                                                                            \
    int row = blockIdx.x;                                                                     \
    if (row >= m) return;                                                                     \
    int start = indptr[row], end = indptr[row + 1];                                           \
    ACC_T acc = ACC_ZERO;                                                                     \
    if (is_homo) {                                                                            \
        ACC_T w = READ_W(weights[0]);                                                         \
        for (int j = start + (int)threadIdx.x; j < end; j += 32) {                           \
            acc += w * READ_W(vector[indices[j]]);                                            \
        }                                                                                      \
    } else {                                                                                  \
        for (int j = start + (int)threadIdx.x; j < end; j += 32) {                           \
            acc += READ_W(weights[j]) * READ_W(vector[indices[j]]);                           \
        }                                                                                      \
    }                                                                                          \
    acc = WARP_RED(acc);                                                                      \
    if (threadIdx.x == 0) output[row] = WRITE_W(acc);                                        \
}

// =========================================================================
// Non-transpose Block kernel
//
// One block (256 threads) per output row.  Threads stride by 256 over the
// row's nonzeros.  A two-level reduction (warp shuffle + shared memory)
// produces the final row sum.  Best for dense rows (avg_nnz > 512).
//
// Grid: (m, 1, 1)   Block: (256, 1, 1)
// Dynamic shared memory: 8 * sizeof(ACC_T)  (one slot per warp)
// =========================================================================

#define DEFINE_CSRMV_NT_BLOCK(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO)  \
__global__ void _csrmv_nt_block_kern##SUFFIX(                                                  \
    const WEIGHT_T* __restrict__ weights,                                                      \
    const int32_t*  __restrict__ indices,                                                      \
    const int32_t*  __restrict__ indptr,                                                       \
    const WEIGHT_T* __restrict__ vector,                                                       \
    WEIGHT_T*       __restrict__ output,                                                       \
    int m, int is_homo                                                                         \
) {                                                                                             \
    extern __shared__ char _smem_bytes[];                                                      \
    ACC_T* smem_red = reinterpret_cast<ACC_T*>(_smem_bytes);                                   \
    int row = blockIdx.x;                                                                      \
    if (row >= m) return;                                                                      \
    int start = indptr[row], end = indptr[row + 1];                                            \
    ACC_T acc = ACC_ZERO;                                                                      \
    if (is_homo) {                                                                             \
        ACC_T w = READ_W(weights[0]);                                                          \
        for (int j = start + (int)threadIdx.x; j < end; j += blockDim.x) {                    \
            acc += w * READ_W(vector[indices[j]]);                                             \
        }                                                                                       \
    } else {                                                                                   \
        for (int j = start + (int)threadIdx.x; j < end; j += blockDim.x) {                    \
            acc += READ_W(weights[j]) * READ_W(vector[indices[j]]);                            \
        }                                                                                       \
    }                                                                                           \
    /* Two-level block reduction: warp shuffle + shared memory */                              \
    int lane   = threadIdx.x & 31;                                                             \
    int warpid = threadIdx.x >> 5;                                                             \
    acc = WARP_RED(acc);                                                                       \
    if (lane == 0) smem_red[warpid] = acc;                                                     \
    __syncthreads();                                                                            \
    int n_warps = (blockDim.x + 31) >> 5;  /* = 8 for blockDim.x=256 */                       \
    acc = (threadIdx.x < n_warps) ? smem_red[lane] : ACC_ZERO;                                \
    if (warpid == 0) acc = WARP_RED(acc);                                                      \
    if (threadIdx.x == 0) output[row] = WRITE_W(acc);                                         \
}

// =========================================================================
// Transpose Warp kernel (float-valued scatter)
//
// One warp (32 threads) per source row i.  All threads load v[row] and
// scatter the weighted contribution via atomicAdd to the output vector.
// No early-exit since every row contributes (unlike binary event kernels).
//
// Half-precision notes:
//   float16  atomicAdd requires sm_70+ (Volta).
//   bfloat16 atomicAdd requires sm_80+ (Ampere).
//
// Grid: (m, 1, 1)   Block: (32, 1, 1)
// =========================================================================

#define DEFINE_CSRMV_T_WARP(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO)  \
__global__ void _csrmv_t_warp_kern##SUFFIX(                                        \
    const WEIGHT_T* __restrict__ weights,                                          \
    const int32_t*  __restrict__ indices,                                          \
    const int32_t*  __restrict__ indptr,                                           \
    const WEIGHT_T* __restrict__ vector,                                           \
    WEIGHT_T*       __restrict__ output,                                           \
    int m, int is_homo                                                             \
) {                                                                                 \
    int row = blockIdx.x;                                                          \
    if (row >= m) return;                                                          \
    ACC_T v_val = READ_W(vector[row]);                                             \
    int start = indptr[row], end = indptr[row + 1];                                \
    if (is_homo) {                                                                 \
        WEIGHT_T contrib = WRITE_W(READ_W(weights[0]) * v_val);                   \
        for (int j = start + (int)threadIdx.x; j < end; j += 32) {                \
            atomicAdd(&output[indices[j]], contrib);                               \
        }                                                                           \
    } else {                                                                       \
        for (int j = start + (int)threadIdx.x; j < end; j += 32) {                \
            atomicAdd(&output[indices[j]], WRITE_W(READ_W(weights[j]) * v_val));   \
        }                                                                           \
    }                                                                               \
}

// =========================================================================
// Kernel instantiations: 4 weight dtypes
// =========================================================================

// ---- Float32 ----
DEFINE_CSRMV_NT_THREAD(_f32, float,  float,  READ_F32, WRITE_F32, 0.0f)
DEFINE_CSRMV_NT_WARP(_f32,  float,  float,  READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_CSRMV_NT_BLOCK(_f32, float,  float,  READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)
DEFINE_CSRMV_T_WARP(_f32,   float,  float,  READ_F32, WRITE_F32, 0.0f)

// ---- Float64 ----
DEFINE_CSRMV_NT_THREAD(_f64, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_CSRMV_NT_WARP(_f64,   double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_CSRMV_NT_BLOCK(_f64,  double, double, READ_F64, WRITE_F64, warp_reduce_sum_f64, 0.0)
DEFINE_CSRMV_T_WARP(_f64,    double, double, READ_F64, WRITE_F64, 0.0)

// ---- Float16 (accumulate in float32; transpose atomicAdd requires sm_70+) ----
DEFINE_CSRMV_NT_THREAD(_f16, __half, float,  READ_F16, WRITE_F16, 0.0f)
DEFINE_CSRMV_NT_WARP(_f16,   __half, float,  READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_CSRMV_NT_BLOCK(_f16,  __half, float,  READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
DEFINE_CSRMV_T_WARP(_f16,    __half, float,  READ_F16, WRITE_F16, 0.0f)

// ---- BFloat16 (accumulate in float32; transpose atomicAdd requires sm_80+) ----
DEFINE_CSRMV_NT_THREAD(_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_CSRMV_NT_WARP(_bf16,   __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_CSRMV_NT_BLOCK(_bf16,  __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_CSRMV_T_WARP(_bf16,    __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)


// =========================================================================
// TVM FFI Entry Point Macros
// =========================================================================
//
// Convention: args = (weights, indices, indptr, vector, output, stream)
//   NT: weights [1 or nse], indices [nse], indptr [m+1], vector [k], output [m]
//   T:  weights [1 or nse], indices [nse], indptr [m+1], vector [m], output [k]
//
// Host-safe metadata extracted from TensorViews:
//   m        = indptr.size(0) - 1  (number of CSR rows)
//   nse      = indices.size(0)     (number of stored elements)
//   is_homo  = (weights.size(0) == 1) ? 1 : 0
//   avg_nnz  = nse / max(m, 1)     (for NT_auto dispatch)
//
// IMPORTANT: data_ptr() is a GPU pointer. Never dereference on the host.
// =========================================================================

// ---- FFI macro: non-transpose thread kernel ----
#define FFI_CSRMV_NT_THREAD(SUFFIX, WEIGHT_C_T)                                \
void csrmv_nt_thread##SUFFIX(                                                   \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                 \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView vector,                  \
    tvm::ffi::TensorView output,  int64_t stream                                \
) {                                                                              \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                    \
    int m       = static_cast<int>(indptr.size(0)) - 1;                         \
    int is_homo = (weights.size(0) == 1) ? 1 : 0;                              \
    int blocks  = (m + 255) / 256;                                              \
    _csrmv_nt_thread_kern##SUFFIX<<<blocks, 256, 0, s>>>(                       \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                     \
        static_cast<const int32_t*>(indices.data_ptr()),                        \
        static_cast<const int32_t*>(indptr.data_ptr()),                         \
        static_cast<const WEIGHT_C_T*>(vector.data_ptr()),                      \
        static_cast<WEIGHT_C_T*>(output.data_ptr()), m, is_homo);               \
}

// ---- FFI macro: non-transpose warp kernel ----
#define FFI_CSRMV_NT_WARP(SUFFIX, WEIGHT_C_T)                                  \
void csrmv_nt_warp##SUFFIX(                                                     \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                 \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView vector,                  \
    tvm::ffi::TensorView output,  int64_t stream                                \
) {                                                                              \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                    \
    int m       = static_cast<int>(indptr.size(0)) - 1;                         \
    int is_homo = (weights.size(0) == 1) ? 1 : 0;                              \
    _csrmv_nt_warp_kern##SUFFIX<<<m, 32, 0, s>>>(                               \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                     \
        static_cast<const int32_t*>(indices.data_ptr()),                        \
        static_cast<const int32_t*>(indptr.data_ptr()),                         \
        static_cast<const WEIGHT_C_T*>(vector.data_ptr()),                      \
        static_cast<WEIGHT_C_T*>(output.data_ptr()), m, is_homo);               \
}

// ---- FFI macro: non-transpose block kernel ----
#define FFI_CSRMV_NT_BLOCK(SUFFIX, WEIGHT_C_T, SHM_SIZE)                       \
void csrmv_nt_block##SUFFIX(                                                    \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                 \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView vector,                  \
    tvm::ffi::TensorView output,  int64_t stream                                \
) {                                                                              \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                    \
    int m       = static_cast<int>(indptr.size(0)) - 1;                         \
    int is_homo = (weights.size(0) == 1) ? 1 : 0;                              \
    _csrmv_nt_block_kern##SUFFIX<<<m, 256, SHM_SIZE, s>>>(                      \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                     \
        static_cast<const int32_t*>(indices.data_ptr()),                        \
        static_cast<const int32_t*>(indptr.data_ptr()),                         \
        static_cast<const WEIGHT_C_T*>(vector.data_ptr()),                      \
        static_cast<WEIGHT_C_T*>(output.data_ptr()), m, is_homo);               \
}

// ---- FFI macro: non-transpose auto (selects thread/warp/block) ----
//
// Dispatch thresholds (tuned empirically):
//   avg_nnz < 8   -> NT_thread (256 threads/block, 1 thread/row)
//   avg_nnz < 512 -> NT_warp   (32 threads/block,  1 warp/row)
//   else          -> NT_block  (256 threads/block,  1 block/row)
//
#define FFI_CSRMV_NT_AUTO(SUFFIX, WEIGHT_C_T, SHM_SIZE)                        \
void csrmv_nt_auto##SUFFIX(                                                     \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                 \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView vector,                  \
    tvm::ffi::TensorView output,  int64_t stream                                \
) {                                                                              \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                   \
    int m        = static_cast<int>(indptr.size(0)) - 1;                        \
    int nse      = static_cast<int>(indices.size(0));                           \
    int is_homo  = (weights.size(0) == 1) ? 1 : 0;                             \
    int avg_nnz  = (m > 0) ? (nse / m) : 0;                                    \
    const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*    d_i = static_cast<const int32_t*>(indices.data_ptr());    \
    const int32_t*    d_p = static_cast<const int32_t*>(indptr.data_ptr());     \
    const WEIGHT_C_T* d_v = static_cast<const WEIGHT_C_T*>(vector.data_ptr()); \
    WEIGHT_C_T*       d_o = static_cast<WEIGHT_C_T*>(output.data_ptr());        \
    if (avg_nnz < 8) {                                                          \
        int blocks = (m + 255) / 256;                                           \
        _csrmv_nt_thread_kern##SUFFIX<<<blocks, 256, 0, s>>>(                   \
            d_w, d_i, d_p, d_v, d_o, m, is_homo);                              \
    } else if (avg_nnz < 512) {                                                 \
        _csrmv_nt_warp_kern##SUFFIX<<<m, 32, 0, s>>>(                           \
            d_w, d_i, d_p, d_v, d_o, m, is_homo);                              \
    } else {                                                                    \
        _csrmv_nt_block_kern##SUFFIX<<<m, 256, SHM_SIZE, s>>>(                  \
            d_w, d_i, d_p, d_v, d_o, m, is_homo);                              \
    }                                                                            \
}

// ---- FFI macro: transpose warp kernel (float-valued scatter) ----
//
// NOTE: The transpose kernel uses atomicAdd to scatter weights into the output
// vector.  JAX's ffi_call does NOT guarantee zero-initialised output buffers,
// so we must explicitly zero the output before launching the scatter kernel.
// We use cudaMemsetAsync to zero on the same stream as the kernel launch.
//
#define FFI_CSRMV_T_WARP(SUFFIX, WEIGHT_C_T)                                   \
void csrmv_t_warp##SUFFIX(                                                      \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                 \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView vector,                  \
    tvm::ffi::TensorView output,  int64_t stream                                \
) {                                                                              \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                   \
    int m        = static_cast<int>(indptr.size(0)) - 1;                        \
    int k        = static_cast<int>(output.size(0));                            \
    int is_homo  = (weights.size(0) == 1) ? 1 : 0;                             \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());            \
    /* Zero output before scatter atomicAdds */                                 \
    cudaMemsetAsync(d_out, 0, (size_t)k * sizeof(WEIGHT_C_T), s);              \
    _csrmv_t_warp_kern##SUFFIX<<<m, 32, 0, s>>>(                                \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                     \
        static_cast<const int32_t*>(indices.data_ptr()),                        \
        static_cast<const int32_t*>(indptr.data_ptr()),                         \
        static_cast<const WEIGHT_C_T*>(vector.data_ptr()),                      \
        d_out, m, is_homo);                                                     \
}


// =========================================================================
// Instantiate TVM FFI entry points via macros + @tvm_ffi annotations
// =========================================================================

// ---- Float32 (shm: 8 * sizeof(float) = 32 bytes) ----
// @tvm_ffi csrmv_nt_thread_f32
FFI_CSRMV_NT_THREAD(_f32, float)
// @tvm_ffi csrmv_nt_warp_f32
FFI_CSRMV_NT_WARP(_f32, float)
// @tvm_ffi csrmv_nt_block_f32
FFI_CSRMV_NT_BLOCK(_f32, float, 8 * sizeof(float))
// @tvm_ffi csrmv_nt_auto_f32
FFI_CSRMV_NT_AUTO(_f32, float, 8 * sizeof(float))
// @tvm_ffi csrmv_t_warp_f32
FFI_CSRMV_T_WARP(_f32, float)

// ---- Float64 (shm: 8 * sizeof(double) = 64 bytes) ----
// @tvm_ffi csrmv_nt_thread_f64
FFI_CSRMV_NT_THREAD(_f64, double)
// @tvm_ffi csrmv_nt_warp_f64
FFI_CSRMV_NT_WARP(_f64, double)
// @tvm_ffi csrmv_nt_block_f64
FFI_CSRMV_NT_BLOCK(_f64, double, 8 * sizeof(double))
// @tvm_ffi csrmv_nt_auto_f64
FFI_CSRMV_NT_AUTO(_f64, double, 8 * sizeof(double))
// @tvm_ffi csrmv_t_warp_f64
FFI_CSRMV_T_WARP(_f64, double)

// ---- Float16 (accumulates in f32; shm: 8 * sizeof(float) = 32 bytes) ----
// @tvm_ffi csrmv_nt_thread_f16
FFI_CSRMV_NT_THREAD(_f16, __half)
// @tvm_ffi csrmv_nt_warp_f16
FFI_CSRMV_NT_WARP(_f16, __half)
// @tvm_ffi csrmv_nt_block_f16
FFI_CSRMV_NT_BLOCK(_f16, __half, 8 * sizeof(float))
// @tvm_ffi csrmv_nt_auto_f16
FFI_CSRMV_NT_AUTO(_f16, __half, 8 * sizeof(float))
// @tvm_ffi csrmv_t_warp_f16
FFI_CSRMV_T_WARP(_f16, __half)

// ---- BFloat16 (accumulates in f32; shm: 8 * sizeof(float) = 32 bytes) ----
// @tvm_ffi csrmv_nt_thread_bf16
FFI_CSRMV_NT_THREAD(_bf16, __nv_bfloat16)
// @tvm_ffi csrmv_nt_warp_bf16
FFI_CSRMV_NT_WARP(_bf16, __nv_bfloat16)
// @tvm_ffi csrmv_nt_block_bf16
FFI_CSRMV_NT_BLOCK(_bf16, __nv_bfloat16, 8 * sizeof(float))
// @tvm_ffi csrmv_nt_auto_bf16
FFI_CSRMV_NT_AUTO(_bf16, __nv_bfloat16, 8 * sizeof(float))
// @tvm_ffi csrmv_t_warp_bf16
FFI_CSRMV_T_WARP(_bf16, __nv_bfloat16)
