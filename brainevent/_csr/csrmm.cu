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
 * csrmm.cu -- Float-Weighted CSR Sparse Matrix-Matrix CUDA Kernels
 * ================================================================
 *
 * Python API:
 *   brainevent.csrmm(data, indices, indptr, B, shape=(m,k), transpose=False)
 *
 * Computes C = A @ B  (transpose=False) or  C = A.T @ B  (transpose=True)
 * where A is stored in CSR format and B is a dense float matrix.
 *
 * Non-transpose (NT, gather mode):
 *   A[m,k] @ B[k,n]  ->  C[m,n]
 *   C[i,l] = sum_{j in nz(i)} A[i,j] * B[indices[j], l]
 *   where nz(i) = {j : indptr[i] <= j < indptr[i+1]}.
 *
 * Transpose (T, scatter mode):
 *   A.T[k,m] @ B[m,n]  ->  C[k,n]
 *   C[indices[j], l] += A[i,j] * B[i, l]  for all i in 0..m, j in row i
 *
 * Parallelisation strategy
 * ------------------------
 * Both modes decompose along two dimensions:
 *   1. Rows (m for NT, m source rows for T): one warp or block per row.
 *   2. Columns of B (n): 32-wide blocks aligned to warp width.
 *      Grid dim-1 = ceil(n / 32).  Thread t in the warp is exclusively
 *      responsible for output column blockIdx.y*32+t.  This gives fully
 *      coalesced reads of B and writes of C within each warp.
 *
 * Kernel variants
 * ---------------
 * Non-transpose:
 *   NT_warp:   1 warp (32 threads) per (row, col-block).  Each thread
 *              serially scans the row's nonzeros, reads B[col_j, c] and
 *              multiplies by weight.  Best for avg_nnz <= 256.
 *              Grid: (m, ceil(n/32))  Block: (32,)
 *
 *   NT_block:  1 block (256 threads) per (row, col-block).  The 256
 *              threads are logically arranged as 8 nnz-strips × 32 cols;
 *              strip s handles nonzeros s, s+8, s+16, ...  Partial sums
 *              are accumulated in shared memory and reduced by strip-0.
 *              Best for avg_nnz > 256.
 *              Grid: (m, ceil(n/32))  Block: (256,)  Smem: 8*32*sizeof(ACC_T)
 *
 *   NT_auto:   Host-side dispatch:
 *              avg_nnz <= 256  ->  NT_warp
 *              avg_nnz >  256  ->  NT_block
 *
 * Transpose:
 *   T_warp:    1 warp (32 threads) per (source row, col-block).  Thread t
 *              reads B[row, col_start+t] as a float value.  For every
 *              nonzero j in source row `row`, the thread atomicAdds
 *              w[j] * B_val into C[indices[j], col_start+t].
 *              The output buffer is zeroed with cudaMemsetAsync before launch.
 *              Grid: (m, ceil(n/32))  Block: (32,)
 *
 * Weight modes (runtime):
 *   is_homo=1:  all nonzeros share data[0].
 *   is_homo=0:  nonzero j uses data[j].
 *   Detected as: is_homo = (weights.size(0) == 1) ? 1 : 0.
 *
 * Dtype support
 * -------------
 *   Weight dtypes: float32, float64, float16, bfloat16.
 *   B dtype always matches weight dtype.
 *   Float16/bfloat16 accumulate in float32 (NT variants).
 *   T_warp atomicAdd into __half  requires sm_70+ (Volta or newer).
 *   T_warp atomicAdd into __nv_bfloat16 requires sm_80+ (Ampere or newer).
 *
 * Memory access patterns
 * ----------------------
 *   NT read of B:  B[indices[j] * n + col_start + lane]
 *     -> 32 threads access 32 consecutive elements of a B row (coalesced).
 *   NT write of C: C[row * n + col_start + lane]
 *     -> 32 threads write 32 consecutive elements of a C row (coalesced).
 *   T read of B:   B[row * n + col_start + lane]
 *     -> 32 consecutive B elements, coalesced read.
 *   T atomicAdd C: C[indices[j] * n + col_start + lane]
 *     -> 32 consecutive C elements per atomicAdd, coalesced within warp.
 *
 * Index dtype
 * -----------
 *   This kernel requires int32 column indices.  The Python wrapper ensures
 *   indices/indptr are int32 before dispatching to this backend.
 *
 * IMPORTANT: All data_ptr() values are GPU device pointers.
 *            NEVER dereference on the host. Extract only metadata (size, ndim).
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

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
// atomicAdd wrappers for each output type
// (T_warp writes accumulator (float) back to WEIGHT_T via atomicAdd)
// =========================================================================

// f32 and f64: atomicAdd accepts the native type directly
#define ATOMIC_ADD_F32(ptr, v)   atomicAdd(ptr, v)
#define ATOMIC_ADD_F64(ptr, v)   atomicAdd(ptr, v)

// f16: accumulate in float, convert back to __half for atomicAdd (sm_70+)
#define ATOMIC_ADD_F16(ptr, v)   atomicAdd(ptr, __float2half(v))

// bf16: accumulate in float, convert back to __nv_bfloat16 (sm_80+)
#define ATOMIC_ADD_BF16(ptr, v)  atomicAdd(ptr, __float2bfloat16(v))


// =========================================================================
// Non-transpose Warp kernel
//
// One warp (32 threads) per (output row, 32-wide column block).
// Thread t exclusively handles output column (col_block * 32 + t).
// The warp serially scans all nonzeros in the row, reading B[col_j, c]
// and multiplying by the weight, accumulating in a per-thread register.
//
// Memory access:
//   Read  B: B[indices[j] * n + col_start + lane]  -- 32 coalesced reads
//             (one per nonzero; warp reads 32 consecutive B-row elements)
//   Write C: C[row * n + col_start + lane]          -- 32 coalesced writes
//
// Grid: (m, ceil(n/32), 1)  Block: (32, 1, 1)
// Shared memory: 0
// =========================================================================

#define DEFINE_CSRMM_NT_WARP(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _csrmm_nt_warp_kern##SUFFIX(                                      \
    const WEIGHT_T* __restrict__ weights,                                         \
    const int32_t*  __restrict__ indices,                                         \
    const int32_t*  __restrict__ indptr,                                          \
    const WEIGHT_T* __restrict__ B,                                               \
    WEIGHT_T*       __restrict__ C,                                               \
    int m, int n, int is_homo                                                     \
) {                                                                                \
    int row       = blockIdx.x;                                                   \
    int col_start = blockIdx.y * 32;                                              \
    int c         = col_start + (int)threadIdx.x;                                 \
    if (row >= m || c >= n) return;                                               \
    int start = indptr[row], end = indptr[row + 1];                               \
    ACC_T acc = ACC_ZERO;                                                         \
    if (is_homo) {                                                                \
        ACC_T w = READ_W(weights[0]);                                             \
        for (int j = start; j < end; j++) {                                       \
            acc += w * READ_W(B[indices[j] * n + c]);                            \
        }                                                                          \
    } else {                                                                      \
        for (int j = start; j < end; j++) {                                       \
            acc += READ_W(weights[j]) * READ_W(B[indices[j] * n + c]);          \
        }                                                                          \
    }                                                                              \
    C[row * n + c] = WRITE_W(acc);                                               \
}

// =========================================================================
// Non-transpose Block kernel
//
// One block (256 threads) per (output row, 32-wide column block).
// The 256 threads are logically partitioned into 8 strips × 32 columns:
//   lane  = threadIdx.x & 31   -- output column offset within the block
//   strip = threadIdx.x >> 5   -- nonzero strip (0..7)
// Strip s handles nonzeros s, s+8, s+16, ...  Partial sums are stored
// in shared memory smem[strip * 32 + lane], then reduced by the strip-0
// threads and written coalesced to C[row, col_start..col_start+31].
//
// The 8x parallelism over nonzeros reduces per-thread work for rows with
// many connections (avg_nnz > 256), at the cost of a shared-memory
// reduction (8 * 32 * sizeof(ACC_T) bytes, one barrier, 8 additions).
//
// Memory access:
//   Read  B: B[indices[j] * n + col_start + lane]  -- coalesced (stride-1)
//   Write C: C[row * n + col_start + lane]          -- coalesced
//
// Grid: (m, ceil(n/32), 1)  Block: (256, 1, 1)
// Shared memory: 8 * 32 * sizeof(ACC_T)
// =========================================================================

#define DEFINE_CSRMM_NT_BLOCK(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _csrmm_nt_block_kern##SUFFIX(                                      \
    const WEIGHT_T* __restrict__ weights,                                          \
    const int32_t*  __restrict__ indices,                                          \
    const int32_t*  __restrict__ indptr,                                           \
    const WEIGHT_T* __restrict__ B,                                                \
    WEIGHT_T*       __restrict__ C,                                                \
    int m, int n, int is_homo                                                      \
) {                                                                                 \
    extern __shared__ char _smem_bytes[];                                          \
    ACC_T* smem = reinterpret_cast<ACC_T*>(_smem_bytes);                          \
    /* smem layout: smem[strip * 32 + lane] for 8 strips, 32 cols */             \
    int row       = blockIdx.x;                                                    \
    int col_start = blockIdx.y * 32;                                               \
    int lane      = threadIdx.x & 31;   /* output column offset: 0..31 */        \
    int strip     = threadIdx.x >> 5;   /* nonzero strip: 0..7  */               \
    int c         = col_start + lane;                                              \
    if (row >= m) return;                                                          \
    int start = indptr[row], end = indptr[row + 1];                                \
    ACC_T acc = ACC_ZERO;                                                          \
    /* Accumulate (threads with c>=n contribute ACC_ZERO) */                     \
    if (c < n) {                                                                   \
        if (is_homo) {                                                             \
            ACC_T w = READ_W(weights[0]);                                          \
            for (int j = start + strip; j < end; j += 8) {                        \
                acc += w * READ_W(B[indices[j] * n + c]);                         \
            }                                                                       \
        } else {                                                                   \
            for (int j = start + strip; j < end; j += 8) {                        \
                acc += READ_W(weights[j]) * READ_W(B[indices[j] * n + c]);       \
            }                                                                       \
        }                                                                           \
    }                                                                               \
    /* All 256 threads write to shared memory, then barrier */                   \
    smem[strip * 32 + lane] = acc;                                                 \
    __syncthreads();                                                                \
    /* Strip-0 threads reduce 8 partial sums and write output */                 \
    if (strip == 0 && c < n) {                                                     \
        acc = ACC_ZERO;                                                            \
        for (int s = 0; s < 8; s++) acc += smem[s * 32 + lane];                  \
        C[row * n + c] = WRITE_W(acc);                                             \
    }                                                                               \
}

// =========================================================================
// Transpose Warp kernel (dense scatter)
//
// One warp (32 threads) per (source row, 32-wide column block).
// Thread t handles output column (col_block * 32 + t) independently.
// Each thread reads B[row, c] once (coalesced across the warp), then
// for every nonzero j in source row `row`, atomicAdds
//   w[j] * B[row, c]  into  C[indices[j], c].
//
// Unlike the binary transpose variant, there is no event-driven skip.
// Every source row has at least 1 nonzero, so all threads always scatter.
//
// The scatter atomicAdds for different j in a row target potentially
// different C rows C[indices[j], c], causing write conflicts between
// warps that share column indices.  atomicAdd handles these correctly.
// Within a single warp, the 32 threads write to 32 consecutive addresses
// C[indices[j]*n + col_start + 0..31] (same destination row), which is
// coalesced.
//
// NOTE: The output buffer C must be zeroed before the scatter kernel runs.
//       The FFI entry point performs cudaMemsetAsync(C, 0, k*n*sizeof, s).
//
// Half-precision notes:
//   float16  atomicAdd requires sm_70+ (Volta).
//   bfloat16 atomicAdd requires sm_80+ (Ampere).
//
// Grid: (m, ceil(n/32), 1)  Block: (32, 1, 1)
// Shared memory: 0
// =========================================================================

#define DEFINE_CSRMM_T_WARP(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W,          \
                             ATOMIC_ADD_W, ACC_ZERO)                              \
__global__ void _csrmm_t_warp_kern##SUFFIX(                                       \
    const WEIGHT_T* __restrict__ weights,                                         \
    const int32_t*  __restrict__ indices,                                         \
    const int32_t*  __restrict__ indptr,                                          \
    const WEIGHT_T* __restrict__ B,                                               \
    WEIGHT_T*       __restrict__ C,                                               \
    int m, int n, int is_homo                                                     \
) {                                                                                \
    int row       = blockIdx.x;                                                   \
    int col_start = blockIdx.y * 32;                                              \
    int c         = col_start + (int)threadIdx.x;                                 \
    if (row >= m || c >= n) return;                                               \
    /* Read the B value once; all nonzeros in this row scatter it */            \
    ACC_T b_val = READ_W(B[row * n + c]);                                         \
    int start = indptr[row], end = indptr[row + 1];                               \
    if (is_homo) {                                                                \
        ACC_T w = READ_W(weights[0]);                                             \
        ACC_T contrib = w * b_val;                                                \
        for (int j = start; j < end; j++) {                                       \
            ATOMIC_ADD_W(&C[indices[j] * n + c], contrib);                        \
        }                                                                          \
    } else {                                                                      \
        for (int j = start; j < end; j++) {                                       \
            ATOMIC_ADD_W(&C[indices[j] * n + c], READ_W(weights[j]) * b_val);   \
        }                                                                          \
    }                                                                              \
}

// =========================================================================
// Kernel instantiations: 4 weight dtypes = 4 groups, 3+1 variants each
//                        = 16 device kernels total
// =========================================================================

// ---- Float32 ----
DEFINE_CSRMM_NT_WARP(_f32,  float,  float,  READ_F32, WRITE_F32, 0.0f)
DEFINE_CSRMM_NT_BLOCK(_f32, float,  float,  READ_F32, WRITE_F32, 0.0f)
DEFINE_CSRMM_T_WARP(_f32,   float,  float,  READ_F32, WRITE_F32, ATOMIC_ADD_F32, 0.0f)

// ---- Float64 ----
DEFINE_CSRMM_NT_WARP(_f64,  double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_CSRMM_NT_BLOCK(_f64, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_CSRMM_T_WARP(_f64,   double, double, READ_F64, WRITE_F64, ATOMIC_ADD_F64, 0.0)

// ---- Float16 (NT accumulates in float32; T atomicAdd requires sm_70+) ----
DEFINE_CSRMM_NT_WARP(_f16,  __half, float,  READ_F16, WRITE_F16, 0.0f)
DEFINE_CSRMM_NT_BLOCK(_f16, __half, float,  READ_F16, WRITE_F16, 0.0f)
DEFINE_CSRMM_T_WARP(_f16,   __half, float,  READ_F16, WRITE_F16, ATOMIC_ADD_F16, 0.0f)

// ---- BFloat16 (NT accumulates in float32; T atomicAdd requires sm_80+) ----
DEFINE_CSRMM_NT_WARP(_bf16,  __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_CSRMM_NT_BLOCK(_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_CSRMM_T_WARP(_bf16,   __nv_bfloat16, float, READ_BF16, WRITE_BF16, ATOMIC_ADD_BF16, 0.0f)


// =========================================================================
// TVM FFI Entry Point Macros
// =========================================================================
//
// Convention for NT kernels:
//   args = (weights, indices, indptr, B, C, stream)
//   weights: [1] (homo) or [nse] (hetero)  WEIGHT_T
//   indices: [nse]                          int32
//   indptr:  [m+1]                          int32
//   B:       [k, n]                         WEIGHT_T  (row-major)
//   C:       [m, n]                         WEIGHT_T  (row-major, output)
//
// Convention for T kernels:
//   args = (weights, indices, indptr, B, C, stream)
//   weights: [1] (homo) or [nse] (hetero)  WEIGHT_T
//   indices: [nse]                          int32
//   indptr:  [m+1]                          int32
//   B:       [m, n]                         WEIGHT_T  (row-major, source)
//   C:       [k, n]                         WEIGHT_T  (row-major, output; zeroed before kernel)
//
// Host-safe metadata:
//   m        = indptr.size(0) - 1
//   n        = B.size(1)
//   nse      = indices.size(0)
//   is_homo  = (weights.size(0) == 1) ? 1 : 0
//   avg_nnz  = nse / max(m, 1)    (for NT_auto dispatch)
//   k (T)    = C.size(0)
//
// IMPORTANT: data_ptr() is a GPU pointer. Never dereference on the host.
// =========================================================================

// ---- FFI macro: non-transpose warp kernel ----
#define FFI_CSRMM_NT_WARP(SUFFIX, WEIGHT_C_T)                                   \
void csrmm_nt_warp##SUFFIX(                                                      \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                  \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView B,                        \
    tvm::ffi::TensorView C,       int64_t stream                                 \
) {                                                                               \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                    \
    int m        = static_cast<int>(indptr.size(0)) - 1;                         \
    int n        = static_cast<int>(B.size(1));                                  \
    int is_homo  = (weights.size(0) == 1) ? 1 : 0;                              \
    int c_blocks = (n + 31) / 32;                                                \
    dim3 grid(m, c_blocks);                                                      \
    _csrmm_nt_warp_kern##SUFFIX<<<grid, 32, 0, s>>>(                             \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                      \
        static_cast<const int32_t*>(indices.data_ptr()),                         \
        static_cast<const int32_t*>(indptr.data_ptr()),                          \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                            \
        static_cast<WEIGHT_C_T*>(C.data_ptr()),                                  \
        m, n, is_homo);                                                           \
}

// ---- FFI macro: non-transpose block kernel ----
#define FFI_CSRMM_NT_BLOCK(SUFFIX, WEIGHT_C_T, SHM_SIZE)                        \
void csrmm_nt_block##SUFFIX(                                                     \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                  \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView B,                        \
    tvm::ffi::TensorView C,       int64_t stream                                 \
) {                                                                               \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                    \
    int m        = static_cast<int>(indptr.size(0)) - 1;                         \
    int n        = static_cast<int>(B.size(1));                                  \
    int is_homo  = (weights.size(0) == 1) ? 1 : 0;                              \
    int c_blocks = (n + 31) / 32;                                                \
    dim3 grid(m, c_blocks);                                                      \
    _csrmm_nt_block_kern##SUFFIX<<<grid, 256, SHM_SIZE, s>>>(                    \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                      \
        static_cast<const int32_t*>(indices.data_ptr()),                         \
        static_cast<const int32_t*>(indptr.data_ptr()),                          \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                            \
        static_cast<WEIGHT_C_T*>(C.data_ptr()),                                  \
        m, n, is_homo);                                                           \
}

// ---- FFI macro: non-transpose auto (selects warp/block based on avg_nnz) ----
//
// Dispatch thresholds (tuned for typical SNN workloads):
//   avg_nnz <= 256  ->  NT_warp   (1 warp/col-block; low overhead)
//   avg_nnz >  256  ->  NT_block  (8-strip reduction; 8x nnz parallelism)
//
#define FFI_CSRMM_NT_AUTO(SUFFIX, WEIGHT_C_T, SHM_SIZE)                         \
void csrmm_nt_auto##SUFFIX(                                                      \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                  \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView B,                        \
    tvm::ffi::TensorView C,       int64_t stream                                 \
) {                                                                               \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                    \
    int m        = static_cast<int>(indptr.size(0)) - 1;                         \
    int nse      = static_cast<int>(indices.size(0));                            \
    int n        = static_cast<int>(B.size(1));                                  \
    int is_homo  = (weights.size(0) == 1) ? 1 : 0;                              \
    int avg_nnz  = (m > 0) ? (nse / m) : 0;                                     \
    int c_blocks = (n + 31) / 32;                                                \
    dim3 grid(m, c_blocks);                                                      \
    const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*    d_i = static_cast<const int32_t*>(indices.data_ptr());    \
    const int32_t*    d_p = static_cast<const int32_t*>(indptr.data_ptr());     \
    const WEIGHT_C_T* d_b = static_cast<const WEIGHT_C_T*>(B.data_ptr());      \
    WEIGHT_C_T*       d_c = static_cast<WEIGHT_C_T*>(C.data_ptr());            \
    if (avg_nnz <= 256) {                                                        \
        _csrmm_nt_warp_kern##SUFFIX<<<grid, 32, 0, s>>>(                         \
            d_w, d_i, d_p, d_b, d_c, m, n, is_homo);                            \
    } else {                                                                     \
        _csrmm_nt_block_kern##SUFFIX<<<grid, 256, SHM_SIZE, s>>>(                \
            d_w, d_i, d_p, d_b, d_c, m, n, is_homo);                            \
    }                                                                             \
}

// ---- FFI macro: transpose warp kernel (dense scatter) ----
//
// NOTE: C must be zeroed before this kernel is launched.
//       cudaMemsetAsync(C, 0, k*n*sizeof(WEIGHT_C_T), stream) ensures correct
//       sequencing on the same stream.
//
#define FFI_CSRMM_T_WARP(SUFFIX, WEIGHT_C_T)                                    \
void csrmm_t_warp##SUFFIX(                                                       \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,                  \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView B,                        \
    tvm::ffi::TensorView C,       int64_t stream                                 \
) {                                                                               \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                    \
    int m        = static_cast<int>(indptr.size(0)) - 1;                         \
    int n        = static_cast<int>(B.size(1));                                  \
    int k        = static_cast<int>(C.size(0));                                  \
    int is_homo  = (weights.size(0) == 1) ? 1 : 0;                              \
    WEIGHT_C_T* d_c = static_cast<WEIGHT_C_T*>(C.data_ptr());                  \
    /* Zero output buffer before scatter (atomicAdds assume zero base) */      \
    cudaMemsetAsync(d_c, 0, (size_t)k * (size_t)n * sizeof(WEIGHT_C_T), s);   \
    int c_blocks = (n + 31) / 32;                                                \
    dim3 grid(m, c_blocks);                                                      \
    _csrmm_t_warp_kern##SUFFIX<<<grid, 32, 0, s>>>(                              \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                      \
        static_cast<const int32_t*>(indices.data_ptr()),                         \
        static_cast<const int32_t*>(indptr.data_ptr()),                          \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                            \
        d_c, m, n, is_homo);                                                     \
}


// =========================================================================
// Instantiate TVM FFI entry points via macros + @tvm_ffi annotations
// =========================================================================

// ---- Float32 (shm for NT_block: 8*32*sizeof(float) = 1024 bytes) ----
// @tvm_ffi csrmm_nt_warp_f32
FFI_CSRMM_NT_WARP(_f32, float)
// @tvm_ffi csrmm_nt_block_f32
FFI_CSRMM_NT_BLOCK(_f32, float, 8 * 32 * sizeof(float))
// @tvm_ffi csrmm_nt_auto_f32
FFI_CSRMM_NT_AUTO(_f32, float, 8 * 32 * sizeof(float))
// @tvm_ffi csrmm_t_warp_f32
FFI_CSRMM_T_WARP(_f32, float)

// ---- Float64 (shm for NT_block: 8*32*sizeof(double) = 2048 bytes) ----
// @tvm_ffi csrmm_nt_warp_f64
FFI_CSRMM_NT_WARP(_f64, double)
// @tvm_ffi csrmm_nt_block_f64
FFI_CSRMM_NT_BLOCK(_f64, double, 8 * 32 * sizeof(double))
// @tvm_ffi csrmm_nt_auto_f64
FFI_CSRMM_NT_AUTO(_f64, double, 8 * 32 * sizeof(double))
// @tvm_ffi csrmm_t_warp_f64
FFI_CSRMM_T_WARP(_f64, double)

// ---- Float16 (shm: 8*32*sizeof(float); T atomicAdd sm_70+) ----
// @tvm_ffi csrmm_nt_warp_f16
FFI_CSRMM_NT_WARP(_f16, __half)
// @tvm_ffi csrmm_nt_block_f16
FFI_CSRMM_NT_BLOCK(_f16, __half, 8 * 32 * sizeof(float))
// @tvm_ffi csrmm_nt_auto_f16
FFI_CSRMM_NT_AUTO(_f16, __half, 8 * 32 * sizeof(float))
// @tvm_ffi csrmm_t_warp_f16
FFI_CSRMM_T_WARP(_f16, __half)

// ---- BFloat16 (shm: 8*32*sizeof(float); T atomicAdd sm_80+) ----
// @tvm_ffi csrmm_nt_warp_bf16
FFI_CSRMM_NT_WARP(_bf16, __nv_bfloat16)
// @tvm_ffi csrmm_nt_block_bf16
FFI_CSRMM_NT_BLOCK(_bf16, __nv_bfloat16, 8 * 32 * sizeof(float))
// @tvm_ffi csrmm_nt_auto_bf16
FFI_CSRMM_NT_AUTO(_bf16, __nv_bfloat16, 8 * 32 * sizeof(float))
// @tvm_ffi csrmm_t_warp_bf16
FFI_CSRMM_T_WARP(_bf16, __nv_bfloat16)
