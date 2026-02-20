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
 * spfloat_csrmm.cu -- Sparse-Float CSR Sparse Matrix-Matrix CUDA Kernels
 * =======================================================================
 *
 * Python API:
 *   brainevent.spfloat_csrmm(data, indices, indptr, B, shape=(m,k), transpose=False)
 *
 * Computes C = A @ B  (transpose=False) or  C = A.T @ B  (transpose=True)
 * where A is stored in CSR format with float-valued weights and B is a
 * sparse float matrix (most entries exactly zero).
 *
 * Computation
 * -----------
 * Non-transpose (NT, gather mode):
 *   A[m,k] @ B[k,n]  ->  C[m,n]
 *   C[i,l] = sum_{j in nz(i)} data[j] * B[indices[j], l],
 *            where B[indices[j], l] != 0   (zero entries skipped)
 *
 * Transpose (T, scatter mode, event-driven):
 *   A.T[k,m] @ B[m,n]  ->  C[k,n]
 *   if B[i,l] != 0:  C[indices[j], l] += data[j] * B[i, l]
 *                    for all j in row i
 *   Threads where B[i,l] == 0 return immediately (event-driven execution).
 *
 * Parallelisation strategy
 * ------------------------
 * Both modes decompose the output matrix along two dimensions:
 *   1. Source rows (m): one warp or block per row.
 *   2. Columns of B (n): 32-wide column blocks aligned to warp width.
 *      Grid dim-1 = ceil(n / 32).  Thread t in the warp handles exclusively
 *      output column blockIdx.y*32+t.  This gives fully coalesced reads of B
 *      and writes of C within each warp.
 *
 * Kernel variants
 * ---------------
 * Non-transpose:
 *   NT_warp:   1 warp (32 threads) per (output row, col-block).  Each thread
 *              serially scans the row's nonzeros; for each nonzero j it reads
 *              B[indices[j], c] and skips if zero, otherwise accumulates
 *              weight[j] * B_val.  Best for avg_nnz <= 256.
 *              Grid: (m, ceil(n/32))  Block: (32,)
 *
 *   NT_block:  1 block (256 threads) per (output row, col-block).  The 256
 *              threads are logically arranged as 8 nnz-strips × 32 cols;
 *              strip s handles nonzeros s, s+8, s+16, ...  Each thread checks
 *              whether B[indices[j], c] is zero before accumulating.  Partial
 *              sums are stored in shared memory (8×32 × sizeof(ACC_T)) and
 *              reduced by strip-0 threads.  Best for avg_nnz > 256.
 *              Grid: (m, ceil(n/32))  Block: (256,)  Smem: 8*32*sizeof(ACC_T)
 *
 *   NT_auto:   Host-side auto-dispatch:
 *              avg_nnz <= 256  ->  NT_warp
 *              avg_nnz >  256  ->  NT_block
 *
 * Transpose:
 *   T_warp:    1 warp (32 threads) per (source row, col-block).  Thread t
 *              reads B[row, col_start+t].  If B_val == 0, thread returns
 *              immediately (event-driven skip, proportional to B sparsity).
 *              Active threads scatter weight contribution for every nonzero in
 *              source row via atomicAdd into C.
 *              The output buffer is zeroed with cudaMemsetAsync before launch.
 *              Grid: (m, ceil(n/32))  Block: (32,)
 *
 *   T_auto:    Alias for T_warp (consistent naming with NT_auto).
 *
 * Weight modes (runtime, not compile-time):
 *   is_homo=1: homogeneous — all nonzeros share data[0].
 *   is_homo=0: heterogeneous — nonzero j uses data[j].
 *   Determined from TensorView metadata on the host (weights.size(0)==1).
 *
 * Dtype support
 * -------------
 *   Weight/B matrix dtypes: float32, float64, float16, bfloat16.
 *   Float16 and bfloat16 accumulate in float32 for numerical stability.
 *   Bfloat16 T_warp atomicAdd requires sm_80+ (Ampere or newer).
 *   Float16  T_warp atomicAdd requires sm_70+ (Volta or newer).
 *   Both weight and B use the same dtype.
 *
 * Memory access patterns
 * ----------------------
 *   NT read of B:  B[indices[j] * n + col_start + lane]
 *     -> 32 threads access 32 consecutive elements of a B row (coalesced).
 *   NT write of C: C[row * n + col_start + lane]
 *     -> 32 threads write 32 consecutive C elements (coalesced).
 *   T read of B:   B[row * n + col_start + lane]
 *     -> 32 consecutive B elements, coalesced read.
 *   T atomicAdd C: C[indices[j] * n + col_start + lane]
 *     -> 32 consecutive C elements per atomicAdd, coalesced within warp.
 *
 * Index dtype
 * -----------
 *   Requires int32 column indices and row pointers.  The Python wrapper
 *   asserts int32 before dispatching to this backend.
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
// atomicAdd wrappers (T_warp writes float accumulator back to WEIGHT_T)
// =========================================================================

// f32/f64: accept native type directly
#define ATOMIC_ADD_F32(ptr, v)   atomicAdd(ptr, v)
#define ATOMIC_ADD_F64(ptr, v)   atomicAdd(ptr, v)

// f16: accumulate in float, convert back to __half (sm_70+)
#define ATOMIC_ADD_F16(ptr, v)   atomicAdd(ptr, __float2half(v))

// bf16: accumulate in float, convert back to __nv_bfloat16 (sm_80+)
#define ATOMIC_ADD_BF16(ptr, v)  atomicAdd(ptr, __float2bfloat16(v))


// =========================================================================
// Non-transpose Warp kernel
//
// One warp (32 threads) per (output row, 32-wide column block).
// Thread t exclusively handles output column (col_block * 32 + t).
//
// For each nonzero j in the CSR row, the thread reads B[indices[j], c]
// and checks whether it is zero (sparse-float event-driven optimization).
// Only non-zero B entries contribute to the accumulator.
//
// Memory access:
//   Read  B: B[indices[j] * n + col_start + lane]  -- 32 coalesced reads
//             (one per nonzero; warp reads 32 consecutive B-row elements)
//   Write C: C[row * n + col_start + lane]          -- 32 coalesced writes
//
// Grid: (m, ceil(n/32), 1)  Block: (32, 1, 1)
// Shared memory: 0
// =========================================================================

#define DEFINE_SPFLOAT_CSRMM_NT_WARP(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _spfloat_csrmm_nt_warp_kern##SUFFIX(                                      \
    const WEIGHT_T* __restrict__ weights,                                                  \
    const int32_t*  __restrict__ indices,                                                  \
    const int32_t*  __restrict__ indptr,                                                   \
    const WEIGHT_T* __restrict__ B,                                                        \
    WEIGHT_T*       __restrict__ C,                                                        \
    int m, int n, int is_homo                                                              \
) {                                                                                         \
    int row       = blockIdx.x;                                                            \
    int col_start = blockIdx.y * 32;                                                       \
    int c         = col_start + (int)threadIdx.x;                                          \
    if (row >= m || c >= n) return;                                                        \
    int start = indptr[row], end = indptr[row + 1];                                        \
    ACC_T acc = ACC_ZERO;                                                                  \
    if (is_homo) {                                                                         \
        ACC_T w = READ_W(weights[0]);                                                      \
        for (int j = start; j < end; j++) {                                                \
            ACC_T b_val = READ_W(B[indices[j] * n + c]);                                  \
            /* Event-driven: skip zero B entries */                                        \
            if (b_val != ACC_ZERO) acc += w * b_val;                                      \
        }                                                                                   \
    } else {                                                                               \
        for (int j = start; j < end; j++) {                                                \
            ACC_T b_val = READ_W(B[indices[j] * n + c]);                                  \
            /* Event-driven: skip zero B entries */                                        \
            if (b_val != ACC_ZERO) acc += READ_W(weights[j]) * b_val;                    \
        }                                                                                   \
    }                                                                                       \
    C[row * n + c] = WRITE_W(acc);                                                        \
}

// =========================================================================
// Non-transpose Block kernel
//
// One block (256 threads) per (output row, 32-wide column block).
// The 256 threads are logically partitioned into 8 strips × 32 columns:
//   lane  = threadIdx.x & 31   -- output column offset within the block
//   strip = threadIdx.x >> 5   -- nonzero strip (0..7)
// Strip s handles nonzeros s, s+8, s+16, ...  For each nonzero, the thread
// checks B[indices[j], c] against zero before accumulating (sparse-float
// optimization).  Partial sums stored in shared memory, reduced by strip-0.
//
// The 8x parallelism over nonzeros benefits dense CSR rows (avg_nnz > 256).
//
// Memory access:
//   Read  B: B[indices[j] * n + col_start + lane]  -- coalesced (stride-1)
//   Write C: C[row * n + col_start + lane]          -- coalesced
//
// Grid: (m, ceil(n/32), 1)  Block: (256, 1, 1)
// Shared memory: 8 * 32 * sizeof(ACC_T)
// =========================================================================

#define DEFINE_SPFLOAT_CSRMM_NT_BLOCK(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W, ACC_ZERO) \
__global__ void _spfloat_csrmm_nt_block_kern##SUFFIX(                                      \
    const WEIGHT_T* __restrict__ weights,                                                   \
    const int32_t*  __restrict__ indices,                                                   \
    const int32_t*  __restrict__ indptr,                                                    \
    const WEIGHT_T* __restrict__ B,                                                         \
    WEIGHT_T*       __restrict__ C,                                                         \
    int m, int n, int is_homo                                                               \
) {                                                                                          \
    extern __shared__ char _smem_bytes[];                                                   \
    ACC_T* smem = reinterpret_cast<ACC_T*>(_smem_bytes);                                   \
    /* smem layout: smem[strip * 32 + lane] for 8 strips, 32 cols */                      \
    int row       = blockIdx.x;                                                             \
    int col_start = blockIdx.y * 32;                                                        \
    int lane      = threadIdx.x & 31;   /* output column offset: 0..31 */                 \
    int strip     = threadIdx.x >> 5;   /* nonzero strip: 0..7  */                        \
    int c         = col_start + lane;                                                       \
    if (row >= m) return;                                                                   \
    int start = indptr[row], end = indptr[row + 1];                                         \
    ACC_T acc = ACC_ZERO;                                                                   \
    /* Accumulate (threads with c>=n contribute ACC_ZERO) */                              \
    if (c < n) {                                                                            \
        if (is_homo) {                                                                      \
            ACC_T w = READ_W(weights[0]);                                                   \
            for (int j = start + strip; j < end; j += 8) {                                 \
                ACC_T b_val = READ_W(B[indices[j] * n + c]);                               \
                /* Event-driven: skip zero B entries */                                     \
                if (b_val != ACC_ZERO) acc += w * b_val;                                   \
            }                                                                                \
        } else {                                                                            \
            for (int j = start + strip; j < end; j += 8) {                                 \
                ACC_T b_val = READ_W(B[indices[j] * n + c]);                               \
                /* Event-driven: skip zero B entries */                                     \
                if (b_val != ACC_ZERO) acc += READ_W(weights[j]) * b_val;                 \
            }                                                                                \
        }                                                                                    \
    }                                                                                        \
    /* All 256 threads write to shared memory, then barrier */                            \
    smem[strip * 32 + lane] = acc;                                                          \
    __syncthreads();                                                                         \
    /* Strip-0 threads reduce 8 partial sums and write output */                          \
    if (strip == 0 && c < n) {                                                              \
        acc = ACC_ZERO;                                                                     \
        for (int s = 0; s < 8; s++) acc += smem[s * 32 + lane];                            \
        C[row * n + c] = WRITE_W(acc);                                                      \
    }                                                                                        \
}

// =========================================================================
// Transpose Warp kernel (event-driven scatter)
//
// One warp (32 threads) per (source row, 32-wide column block).
// Thread t handles output column (col_block * 32 + t) independently.
//
// Event-driven property:
//   Each thread reads B[row, c] once (coalesced across warp).
//   If B[row, c] == 0, the thread returns immediately without scanning any
//   nonzeros or performing any atomicAdds.  For sparse B matrices, most
//   threads across the grid return early, eliminating work proportional to
//   the B zero fraction × total CSR nonzeros.
//
// Active threads scatter weight contribution for every nonzero j in source
// row `row` to C[indices[j], c] via atomicAdd.
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

#define DEFINE_SPFLOAT_CSRMM_T_WARP(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W,      \
                                     ATOMIC_ADD_W, ACC_ZERO)                          \
__global__ void _spfloat_csrmm_t_warp_kern##SUFFIX(                                   \
    const WEIGHT_T* __restrict__ weights,                                              \
    const int32_t*  __restrict__ indices,                                              \
    const int32_t*  __restrict__ indptr,                                               \
    const WEIGHT_T* __restrict__ B,                                                    \
    WEIGHT_T*       __restrict__ C,                                                    \
    int m, int n, int is_homo                                                          \
) {                                                                                     \
    int row       = blockIdx.x;                                                        \
    int col_start = blockIdx.y * 32;                                                   \
    int c         = col_start + (int)threadIdx.x;                                      \
    if (row >= m || c >= n) return;                                                    \
    /* Read B once per thread; skip zero (event-driven optimization) */              \
    ACC_T b_val = READ_W(B[row * n + c]);                                              \
    if (b_val == ACC_ZERO) return;                                                     \
    int start = indptr[row], end = indptr[row + 1];                                    \
    /* Active: scatter weight contribution to all connected output rows */           \
    if (is_homo) {                                                                     \
        ACC_T contrib = READ_W(weights[0]) * b_val;                                    \
        for (int j = start; j < end; j++) {                                            \
            ATOMIC_ADD_W(&C[indices[j] * n + c], contrib);                             \
        }                                                                               \
    } else {                                                                           \
        for (int j = start; j < end; j++) {                                            \
            ATOMIC_ADD_W(&C[indices[j] * n + c], READ_W(weights[j]) * b_val);        \
        }                                                                               \
    }                                                                                   \
}

// =========================================================================
// Kernel instantiations: 4 weight/B dtypes, 3 variants each = 12 kernels
// =========================================================================

// ---- Float32 ----
DEFINE_SPFLOAT_CSRMM_NT_WARP(_f32,  float,  float,  READ_F32, WRITE_F32, 0.0f)
DEFINE_SPFLOAT_CSRMM_NT_BLOCK(_f32, float,  float,  READ_F32, WRITE_F32, 0.0f)
DEFINE_SPFLOAT_CSRMM_T_WARP(_f32,   float,  float,  READ_F32, WRITE_F32, ATOMIC_ADD_F32, 0.0f)

// ---- Float64 ----
DEFINE_SPFLOAT_CSRMM_NT_WARP(_f64,  double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_SPFLOAT_CSRMM_NT_BLOCK(_f64, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_SPFLOAT_CSRMM_T_WARP(_f64,   double, double, READ_F64, WRITE_F64, ATOMIC_ADD_F64, 0.0)

// ---- Float16 (NT accumulates in float32; T atomicAdd requires sm_70+) ----
DEFINE_SPFLOAT_CSRMM_NT_WARP(_f16,  __half, float,  READ_F16, WRITE_F16, 0.0f)
DEFINE_SPFLOAT_CSRMM_NT_BLOCK(_f16, __half, float,  READ_F16, WRITE_F16, 0.0f)
DEFINE_SPFLOAT_CSRMM_T_WARP(_f16,   __half, float,  READ_F16, WRITE_F16, ATOMIC_ADD_F16, 0.0f)

// ---- BFloat16 (NT accumulates in float32; T atomicAdd requires sm_80+) ----
DEFINE_SPFLOAT_CSRMM_NT_WARP(_bf16,  __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_SPFLOAT_CSRMM_NT_BLOCK(_bf16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_SPFLOAT_CSRMM_T_WARP(_bf16,   __nv_bfloat16, float, READ_BF16, WRITE_BF16, ATOMIC_ADD_BF16, 0.0f)


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
// Host-safe metadata extracted from TensorViews:
//   m       = indptr.size(0) - 1  (number of CSR rows)
//   n       = B.size(1)           (number of columns)
//   nse     = indices.size(0)     (number of stored elements)
//   is_homo = (weights.size(0) == 1) ? 1 : 0
//   avg_nnz = nse / max(m, 1)     (for NT_auto dispatch)
//   k (T)   = C.size(0)           (CSR cols / T output rows)
//
// IMPORTANT: data_ptr() is a GPU pointer. Never dereference on the host.
// =========================================================================

// ---- FFI macro: non-transpose warp kernel ----
#define FFI_SPFLOAT_CSRMM_NT_WARP(SUFFIX, WEIGHT_C_T)                           \
void spfloat_csrmm_nt_warp##SUFFIX(                                              \
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
    _spfloat_csrmm_nt_warp_kern##SUFFIX<<<grid, 32, 0, s>>>(                     \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                      \
        static_cast<const int32_t*>(indices.data_ptr()),                         \
        static_cast<const int32_t*>(indptr.data_ptr()),                          \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                            \
        static_cast<WEIGHT_C_T*>(C.data_ptr()),                                  \
        m, n, is_homo);                                                           \
}

// ---- FFI macro: non-transpose block kernel ----
#define FFI_SPFLOAT_CSRMM_NT_BLOCK(SUFFIX, WEIGHT_C_T, SHM_SIZE)                \
void spfloat_csrmm_nt_block##SUFFIX(                                             \
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
    _spfloat_csrmm_nt_block_kern##SUFFIX<<<grid, 256, SHM_SIZE, s>>>(            \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                      \
        static_cast<const int32_t*>(indices.data_ptr()),                         \
        static_cast<const int32_t*>(indptr.data_ptr()),                          \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                            \
        static_cast<WEIGHT_C_T*>(C.data_ptr()),                                  \
        m, n, is_homo);                                                           \
}

// ---- FFI macro: non-transpose auto-dispatch ----
//
// Dispatch thresholds (tuned for typical SNN workloads; consistent with csrmm.cu):
//   avg_nnz <= 256  ->  NT_warp   (1 warp/col-block; low reduction overhead)
//   avg_nnz >  256  ->  NT_block  (8-strip reduction; 8x nnz parallelism)
//
#define FFI_SPFLOAT_CSRMM_NT_AUTO(SUFFIX, WEIGHT_C_T, SHM_SIZE)                 \
void spfloat_csrmm_nt_auto##SUFFIX(                                              \
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
        _spfloat_csrmm_nt_warp_kern##SUFFIX<<<grid, 32, 0, s>>>(                 \
            d_w, d_i, d_p, d_b, d_c, m, n, is_homo);                            \
    } else {                                                                     \
        _spfloat_csrmm_nt_block_kern##SUFFIX<<<grid, 256, SHM_SIZE, s>>>(        \
            d_w, d_i, d_p, d_b, d_c, m, n, is_homo);                            \
    }                                                                             \
}

// ---- FFI macro: transpose warp kernel (event-driven scatter) ----
//
// NOTE: C must be zeroed before this kernel is launched.
//       cudaMemsetAsync(C, 0, k*n*sizeof(WEIGHT_C_T), stream) ensures correct
//       sequencing on the same stream.
//
#define FFI_SPFLOAT_CSRMM_T_WARP(SUFFIX, WEIGHT_C_T)                            \
void spfloat_csrmm_t_warp##SUFFIX(                                               \
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
    _spfloat_csrmm_t_warp_kern##SUFFIX<<<grid, 32, 0, s>>>(                      \
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
// @tvm_ffi spfloat_csrmm_nt_warp_f32
FFI_SPFLOAT_CSRMM_NT_WARP(_f32, float)
// @tvm_ffi spfloat_csrmm_nt_block_f32
FFI_SPFLOAT_CSRMM_NT_BLOCK(_f32, float, 8 * 32 * sizeof(float))
// @tvm_ffi spfloat_csrmm_nt_auto_f32
FFI_SPFLOAT_CSRMM_NT_AUTO(_f32, float, 8 * 32 * sizeof(float))
// @tvm_ffi spfloat_csrmm_t_warp_f32
FFI_SPFLOAT_CSRMM_T_WARP(_f32, float)

// ---- Float64 (shm for NT_block: 8*32*sizeof(double) = 2048 bytes) ----
// @tvm_ffi spfloat_csrmm_nt_warp_f64
FFI_SPFLOAT_CSRMM_NT_WARP(_f64, double)
// @tvm_ffi spfloat_csrmm_nt_block_f64
FFI_SPFLOAT_CSRMM_NT_BLOCK(_f64, double, 8 * 32 * sizeof(double))
// @tvm_ffi spfloat_csrmm_nt_auto_f64
FFI_SPFLOAT_CSRMM_NT_AUTO(_f64, double, 8 * 32 * sizeof(double))
// @tvm_ffi spfloat_csrmm_t_warp_f64
FFI_SPFLOAT_CSRMM_T_WARP(_f64, double)

// ---- Float16 (NT accumulates in f32; shm: 8*32*sizeof(float); T atomicAdd sm_70+) ----
// @tvm_ffi spfloat_csrmm_nt_warp_f16
FFI_SPFLOAT_CSRMM_NT_WARP(_f16, __half)
// @tvm_ffi spfloat_csrmm_nt_block_f16
FFI_SPFLOAT_CSRMM_NT_BLOCK(_f16, __half, 8 * 32 * sizeof(float))
// @tvm_ffi spfloat_csrmm_nt_auto_f16
FFI_SPFLOAT_CSRMM_NT_AUTO(_f16, __half, 8 * 32 * sizeof(float))
// @tvm_ffi spfloat_csrmm_t_warp_f16
FFI_SPFLOAT_CSRMM_T_WARP(_f16, __half)

// ---- BFloat16 (NT accumulates in f32; shm: 8*32*sizeof(float); T atomicAdd sm_80+) ----
// @tvm_ffi spfloat_csrmm_nt_warp_bf16
FFI_SPFLOAT_CSRMM_NT_WARP(_bf16, __nv_bfloat16)
// @tvm_ffi spfloat_csrmm_nt_block_bf16
FFI_SPFLOAT_CSRMM_NT_BLOCK(_bf16, __nv_bfloat16, 8 * 32 * sizeof(float))
// @tvm_ffi spfloat_csrmm_nt_auto_bf16
FFI_SPFLOAT_CSRMM_NT_AUTO(_bf16, __nv_bfloat16, 8 * 32 * sizeof(float))
// @tvm_ffi spfloat_csrmm_t_warp_bf16
FFI_SPFLOAT_CSRMM_T_WARP(_bf16, __nv_bfloat16)
