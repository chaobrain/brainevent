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
 * binary_csrmm.cu -- Event-Driven Binary CSR Sparse Matrix-Matrix CUDA Kernels
 * ==============================================================================
 *
 * Python API:
 *   brainevent.binary_csrmm(data, indices, indptr, B, shape=(m,k), transpose=False)
 *
 * Computes C = A @ B  (transpose=False) or  C = A.T @ B  (transpose=True)
 * where A is stored in CSR format and B is a binary event matrix.
 *
 * Active-event semantics
 * ----------------------
 * For bool spikes (int8):  active when != 0.
 * For float spikes (f32):  active when  > 0.
 *
 * Non-transpose (NT, gather mode):
 *   A[m,k] @ B[k,n]  ->  C[m,n]
 *   C[i,l] = sum_{j in nz(i)} A[i,j] * e(B[indices[j], l])
 *   where nz(i) = {j : indptr[i] <= j < indptr[i+1]}, e() is the event indicator.
 *
 * Transpose (T, scatter mode):
 *   A.T[k,m] @ B[m,n]  ->  C[k,n]
 *   if B[i,l] is active:  C[indices[j], l] += A[i,j]  for all j in row i
 *
 * Parallelisation strategy
 * ------------------------
 * Both modes decompose the output matrix along two dimensions:
 *   1. Rows (m for NT, m source rows for T): one warp or block per row.
 *   2. Columns of B (n): 32-wide column blocks aligned to warp width.
 *      Grid dim-1 = ceil(n / 32).  Each block/warp is responsible for
 *      output columns [blockIdx.y*32, blockIdx.y*32+32).  Thread t in the
 *      warp is exclusively responsible for column blockIdx.y*32+t.  This
 *      gives fully coalesced reads of B and writes of C within each warp.
 *
 * Kernel variants
 * ---------------
 * Non-transpose:
 *   NT_warp:   1 warp (32 threads) per (row, col-block).  Each thread
 *              serially scans the row's nonzeros and accumulates for its
 *              one column.  Best for avg_nnz <= 256.
 *              Grid: (m, ceil(n/32))  Block: (32,)
 *
 *   NT_block:  1 block (256 threads) per (row, col-block).  The 256
 *              threads are logically arranged as 8 nnz-strips × 32 cols;
 *              strip s handles nonzeros s, s+8, s+16, ...  Partial sums
 *              are accumulated in shared memory (8×32 × sizeof(ACC_T))
 *              then reduced by the strip-0 threads.  Effective parallelism
 *              over both nonzeros and output columns.
 *              Grid: (m, ceil(n/32))  Block: (256,)  Smem: 8*32*sizeof(ACC_T)
 *
 *   NT_auto:   Host-side auto-dispatch:
 *              avg_nnz <= 256  ->  NT_warp
 *              avg_nnz >  256  ->  NT_block
 *
 * Transpose:
 *   T_warp:    1 warp (32 threads) per (row, col-block).  Thread t checks
 *              whether B[row, col_start+t] is active.  If active, it
 *              serially scatters A[row, j] (for all j in row) to C via
 *              atomicAdd, avoiding inter-thread contention within the warp.
 *              The output buffer is zeroed with cudaMemsetAsync before launch.
 *              Grid: (m, ceil(n/32))  Block: (32,)
 *
 *   T_auto:    Alias for T_warp (consistent naming with NT_auto).
 *
 * Weight modes (runtime, not compile-time):
 *   is_homo=1:  homogeneous — all nonzeros share data[0].
 *   is_homo=0:  heterogeneous — nonzero j uses data[j].
 *   Detected as: is_homo = (weights.size(0) == 1) ? 1 : 0.
 *
 * Dtype support
 * -------------
 *   Weight dtypes: float32, float64, float16, bfloat16.
 *   Float16 and bfloat16 accumulate in float32 (NT variants).
 *   Bfloat16 T_warp atomicAdd requires sm_80+ (Ampere or newer).
 *   Float16  T_warp atomicAdd requires sm_70+ (Volta or newer).
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
 *   This kernel requires int32 column indices.  The Python wrapper should
 *   ensure indices/indptr are int32 before dispatching to this backend.
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
// Active-check predicates
// =========================================================================

#define IS_ACTIVE_BOOL(s)  ((s) != 0)
#define IS_ACTIVE_FLOAT(s) ((s) > 0.0f)

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
// Non-transpose Warp kernel
//
// One warp (32 threads) per (output row, 32-wide column block).
// Thread t exclusively handles output column (col_block * 32 + t).
// The warp serially scans all nonzeros in the row, each time checking
// the event at B[indices[j], col] and conditionally accumulating the
// weight into an independent per-thread register.
//
// Memory access:
//   Read  B: B[indices[j] * n + col_start + lane]  -- 32 coalesced reads
//   Write C: C[row * n + col_start + lane]          -- 32 coalesced writes
//
// Grid: (m, ceil(n/32), 1)  Block: (32, 1, 1)
// Shared memory: 0
// =========================================================================

#define DEFINE_CSRMM_NT_WARP(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T,  \
                              READ_W, WRITE_W, ACC_ZERO)                     \
__global__ void _csrmm_nt_warp_kern##SUFFIX(                                 \
    const WEIGHT_T* __restrict__ weights,                                    \
    const int32_t*  __restrict__ indices,                                    \
    const int32_t*  __restrict__ indptr,                                     \
    const SPIKE_T*  __restrict__ B,                                          \
    WEIGHT_T*       __restrict__ C,                                          \
    int m, int n, int is_homo                                                \
) {                                                                           \
    int row       = blockIdx.x;                                              \
    int col_start = blockIdx.y * 32;                                         \
    int c         = col_start + (int)threadIdx.x;                            \
    if (row >= m || c >= n) return;                                          \
    int start = indptr[row], end = indptr[row + 1];                          \
    ACC_T acc = ACC_ZERO;                                                    \
    if (is_homo) {                                                           \
        ACC_T w = READ_W(weights[0]);                                        \
        for (int j = start; j < end; j++) {                                  \
            if (IS_ACTIVE(B[indices[j] * n + c])) acc += w;                 \
        }                                                                     \
    } else {                                                                 \
        for (int j = start; j < end; j++) {                                  \
            if (IS_ACTIVE(B[indices[j] * n + c])) acc += READ_W(weights[j]);\
        }                                                                     \
    }                                                                         \
    C[row * n + c] = WRITE_W(acc);                                           \
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
// threads.  The final result is written coalesced to C[row, col_start..].
//
// The 8x parallelism over nonzeros reduces per-thread work for dense rows,
// at the cost of a shared-memory reduction (32 bytes of smem writes per
// warp, barrier, 8 additions per output element).
//
// Memory access:
//   Read  B: B[indices[j] * n + col_start + lane]  -- coalesced (stride-1)
//   Write C: C[row * n + col_start + lane]          -- coalesced
//
// Grid: (m, ceil(n/32), 1)  Block: (256, 1, 1)
// Shared memory: 8 * 32 * sizeof(ACC_T)
// =========================================================================

#define DEFINE_CSRMM_NT_BLOCK(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T,  \
                               READ_W, WRITE_W, ACC_ZERO)                    \
__global__ void _csrmm_nt_block_kern##SUFFIX(                                \
    const WEIGHT_T* __restrict__ weights,                                    \
    const int32_t*  __restrict__ indices,                                    \
    const int32_t*  __restrict__ indptr,                                     \
    const SPIKE_T*  __restrict__ B,                                          \
    WEIGHT_T*       __restrict__ C,                                          \
    int m, int n, int is_homo                                                \
) {                                                                           \
    extern __shared__ char _smem_bytes[];                                    \
    ACC_T* smem = reinterpret_cast<ACC_T*>(_smem_bytes);                     \
    /* smem layout: smem[strip * 32 + lane] for 8 strips, 32 cols */        \
    int row       = blockIdx.x;                                              \
    int col_start = blockIdx.y * 32;                                         \
    int lane      = threadIdx.x & 31;   /* output column offset: 0..31 */   \
    int strip     = threadIdx.x >> 5;   /* nonzero strip:         0..7  */  \
    int c         = col_start + lane;                                        \
    if (row >= m) return;                                                    \
    int start = indptr[row], end = indptr[row + 1];                          \
    ACC_T acc = ACC_ZERO;                                                    \
    /* Accumulate (threads with c>=n contribute ACC_ZERO) */                \
    if (c < n) {                                                             \
        if (is_homo) {                                                       \
            ACC_T w = READ_W(weights[0]);                                    \
            for (int j = start + strip; j < end; j += 8) {                  \
                if (IS_ACTIVE(B[indices[j] * n + c])) acc += w;             \
            }                                                                 \
        } else {                                                             \
            for (int j = start + strip; j < end; j += 8) {                  \
                if (IS_ACTIVE(B[indices[j] * n + c])) acc += READ_W(weights[j]);\
            }                                                                 \
        }                                                                     \
    }                                                                         \
    /* All 256 threads write to shared memory, then barrier */              \
    smem[strip * 32 + lane] = acc;                                           \
    __syncthreads();                                                          \
    /* Strip-0 threads reduce 8 partial sums and write output */            \
    if (strip == 0 && c < n) {                                              \
        acc = ACC_ZERO;                                                      \
        for (int s = 0; s < 8; s++) acc += smem[s * 32 + lane];             \
        C[row * n + c] = WRITE_W(acc);                                       \
    }                                                                         \
}

// =========================================================================
// Transpose Warp kernel (event-driven scatter)
//
// One warp (32 threads) per (source row, 32-wide column block).
// Thread t handles output column (col_block * 32 + t) independently.
// If B[row, c] is active, the thread scatters the weight contribution
// for every nonzero in source row `row` to C[indices[j], c] via atomicAdd.
//
// Event-driven property: threads where B[row, c] is inactive return
// immediately without scanning any nonzeros or performing any atomicAdds.
// For low spike densities, most threads in the grid return early.
//
// The scatter atomicAdds for different nonzeros in a row write to
// C[indices[j], c], where j varies.  Warps processing different rows
// may produce conflicting writes only if their rows share column indices;
// atomicAdd handles this correctly.  Within a single warp, the 32 threads
// write to 32 consecutive addresses C[indices[j]*n + col_start + 0..31]
// (same destination row, consecutive columns), which is coalesced.
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

#define DEFINE_CSRMM_T_WARP(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T,   \
                             READ_W, WRITE_W, ACC_ZERO)                      \
__global__ void _csrmm_t_warp_kern##SUFFIX(                                  \
    const WEIGHT_T* __restrict__ weights,                                    \
    const int32_t*  __restrict__ indices,                                    \
    const int32_t*  __restrict__ indptr,                                     \
    const SPIKE_T*  __restrict__ B,                                          \
    WEIGHT_T*       __restrict__ C,                                          \
    int m, int n, int is_homo                                                \
) {                                                                           \
    int row       = blockIdx.x;                                              \
    int col_start = blockIdx.y * 32;                                         \
    int c         = col_start + (int)threadIdx.x;                            \
    if (row >= m || c >= n) return;                                          \
    /* Each thread independently checks its column event */                 \
    if (!IS_ACTIVE(B[row * n + c])) return;                                  \
    int start = indptr[row], end = indptr[row + 1];                          \
    /* Active: scatter weight to all connected output rows */               \
    if (is_homo) {                                                           \
        WEIGHT_T w_out = weights[0];                                         \
        for (int j = start; j < end; j++) {                                  \
            atomicAdd(&C[indices[j] * n + c], w_out);                        \
        }                                                                     \
    } else {                                                                 \
        for (int j = start; j < end; j++) {                                  \
            atomicAdd(&C[indices[j] * n + c], weights[j]);                   \
        }                                                                     \
    }                                                                         \
}

// =========================================================================
// Kernel instantiations: 4 weight dtypes x 2 spike types = 8 groups,
//                        3 variants each = 24 device kernels
// =========================================================================

// ---- Float32 ----
DEFINE_CSRMM_NT_WARP(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_CSRMM_NT_WARP(_f32_float, float,  IS_ACTIVE_FLOAT, float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_CSRMM_NT_BLOCK(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_CSRMM_NT_BLOCK(_f32_float, float,  IS_ACTIVE_FLOAT, float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_CSRMM_T_WARP(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float, float, READ_F32, WRITE_F32, 0.0f)
DEFINE_CSRMM_T_WARP(_f32_float, float,  IS_ACTIVE_FLOAT, float, float, READ_F32, WRITE_F32, 0.0f)

// ---- Float64 ----
DEFINE_CSRMM_NT_WARP(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_CSRMM_NT_WARP(_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_CSRMM_NT_BLOCK(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_CSRMM_NT_BLOCK(_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_CSRMM_T_WARP(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64, WRITE_F64, 0.0)
DEFINE_CSRMM_T_WARP(_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64, WRITE_F64, 0.0)

// ---- Float16 (accumulate in float32; transpose atomicAdd requires sm_70+) ----
DEFINE_CSRMM_NT_WARP(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_CSRMM_NT_WARP(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_CSRMM_NT_BLOCK(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_CSRMM_NT_BLOCK(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_CSRMM_T_WARP(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float, READ_F16, WRITE_F16, 0.0f)
DEFINE_CSRMM_T_WARP(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float, READ_F16, WRITE_F16, 0.0f)

// ---- BFloat16 (accumulate in float32; transpose atomicAdd requires sm_80+) ----
DEFINE_CSRMM_NT_WARP(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_CSRMM_NT_WARP(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_CSRMM_NT_BLOCK(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_CSRMM_NT_BLOCK(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_CSRMM_T_WARP(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)
DEFINE_CSRMM_T_WARP(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, WRITE_BF16, 0.0f)


// =========================================================================
// TVM FFI Entry Point Macros
// =========================================================================
//
// Convention for NT kernels:
//   args = (weights, indices, indptr, B, C, stream)
//   weights: [1] (homo) or [nse] (hetero)  WEIGHT_T
//   indices: [nse]                          int32
//   indptr:  [m+1]                          int32
//   B:       [k, n]                         SPIKE_T  (row-major)
//   C:       [m, n]                         WEIGHT_T (row-major, output)
//
// Convention for T kernels:
//   args = (weights, indices, indptr, B, C, stream)
//   weights: [1] (homo) or [nse] (hetero)  WEIGHT_T
//   indices: [nse]                          int32
//   indptr:  [m+1]                          int32
//   B:       [m, n]                         SPIKE_T  (row-major, source)
//   C:       [k, n]                         WEIGHT_T (row-major, output; zeroed before kernel)
//
// Host-safe metadata (all from TensorView):
//   m        = indptr.size(0) - 1      (CSR rows / NT output rows)
//   n        = B.size(1)               (output columns, same as B columns)
//   nse      = indices.size(0)         (number of stored elements)
//   is_homo  = (weights.size(0) == 1) ? 1 : 0
//   avg_nnz  = nse / max(m, 1)         (for NT_auto dispatch)
//   k (T)    = C.size(0)               (CSR cols / T output rows)
//
// IMPORTANT: data_ptr() is a GPU pointer. Never dereference on the host.
// =========================================================================

// ---- FFI macro: non-transpose warp kernel ----
#define FFI_CSRMM_NT_WARP(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                    \
void binary_csrmm_nt_warp##SUFFIX(                                           \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,              \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView B,                   \
    tvm::ffi::TensorView C,       int64_t stream                             \
) {                                                                           \
    cudaStream_t s   = reinterpret_cast<cudaStream_t>(stream);               \
    int m        = static_cast<int>(indptr.size(0)) - 1;                     \
    int n        = static_cast<int>(B.size(1));                              \
    int is_homo  = (weights.size(0) == 1) ? 1 : 0;                          \
    int c_blocks = (n + 31) / 32;                                            \
    dim3 grid(m, c_blocks);                                                  \
    _csrmm_nt_warp_kern##SUFFIX<<<grid, 32, 0, s>>>(                         \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                  \
        static_cast<const int32_t*>(indices.data_ptr()),                     \
        static_cast<const int32_t*>(indptr.data_ptr()),                      \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                         \
        static_cast<WEIGHT_C_T*>(C.data_ptr()),                              \
        m, n, is_homo);                                                       \
}

// ---- FFI macro: non-transpose block kernel ----
#define FFI_CSRMM_NT_BLOCK(SUFFIX, WEIGHT_C_T, SPIKE_C_T, SHM_SIZE)         \
void binary_csrmm_nt_block##SUFFIX(                                          \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,              \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView B,                   \
    tvm::ffi::TensorView C,       int64_t stream                             \
) {                                                                           \
    cudaStream_t s   = reinterpret_cast<cudaStream_t>(stream);               \
    int m        = static_cast<int>(indptr.size(0)) - 1;                     \
    int n        = static_cast<int>(B.size(1));                              \
    int is_homo  = (weights.size(0) == 1) ? 1 : 0;                          \
    int c_blocks = (n + 31) / 32;                                            \
    dim3 grid(m, c_blocks);                                                  \
    _csrmm_nt_block_kern##SUFFIX<<<grid, 256, SHM_SIZE, s>>>(                \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                  \
        static_cast<const int32_t*>(indices.data_ptr()),                     \
        static_cast<const int32_t*>(indptr.data_ptr()),                      \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                         \
        static_cast<WEIGHT_C_T*>(C.data_ptr()),                              \
        m, n, is_homo);                                                       \
}

// ---- FFI macro: non-transpose auto (selects warp/block based on avg_nnz) ----
//
// Dispatch thresholds (tuned for typical SNN workloads):
//   avg_nnz <= 256  ->  NT_warp   (1 warp/col-block; minimal reduction overhead)
//   avg_nnz >  256  ->  NT_block  (8-strip reduction; 8x nnz parallelism)
//
#define FFI_CSRMM_NT_AUTO(SUFFIX, WEIGHT_C_T, SPIKE_C_T, SHM_SIZE)          \
void binary_csrmm_nt_auto##SUFFIX(                                           \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,              \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView B,                   \
    tvm::ffi::TensorView C,       int64_t stream                             \
) {                                                                           \
    cudaStream_t s   = reinterpret_cast<cudaStream_t>(stream);               \
    int m        = static_cast<int>(indptr.size(0)) - 1;                     \
    int nse      = static_cast<int>(indices.size(0));                        \
    int n        = static_cast<int>(B.size(1));                              \
    int is_homo  = (weights.size(0) == 1) ? 1 : 0;                          \
    int avg_nnz  = (m > 0) ? (nse / m) : 0;                                 \
    int c_blocks = (n + 31) / 32;                                            \
    dim3 grid(m, c_blocks);                                                  \
    const WEIGHT_C_T* d_w = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*    d_i = static_cast<const int32_t*>(indices.data_ptr()); \
    const int32_t*    d_p = static_cast<const int32_t*>(indptr.data_ptr());  \
    const SPIKE_C_T*  d_b = static_cast<const SPIKE_C_T*>(B.data_ptr());    \
    WEIGHT_C_T*       d_c = static_cast<WEIGHT_C_T*>(C.data_ptr());         \
    if (avg_nnz <= 256) {                                                    \
        _csrmm_nt_warp_kern##SUFFIX<<<grid, 32, 0, s>>>(                     \
            d_w, d_i, d_p, d_b, d_c, m, n, is_homo);                        \
    } else {                                                                 \
        _csrmm_nt_block_kern##SUFFIX<<<grid, 256, SHM_SIZE, s>>>(            \
            d_w, d_i, d_p, d_b, d_c, m, n, is_homo);                        \
    }                                                                         \
}

// ---- FFI macro: transpose warp kernel (event-driven scatter) ----
//
// NOTE: The transpose kernel uses atomicAdd to scatter weights into C.
//       JAX's ffi_call does NOT guarantee zero-initialised output buffers,
//       so we must explicitly zero C before launching the scatter kernel.
//       cudaMemsetAsync(C, 0, k*n*sizeof, stream) is called on the same
//       stream to ensure correct sequencing.
//
#define FFI_CSRMM_T_WARP(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                     \
void binary_csrmm_t_warp##SUFFIX(                                            \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices,              \
    tvm::ffi::TensorView indptr,  tvm::ffi::TensorView B,                   \
    tvm::ffi::TensorView C,       int64_t stream                             \
) {                                                                           \
    cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);                \
    int m        = static_cast<int>(indptr.size(0)) - 1;                     \
    int n        = static_cast<int>(B.size(1));                              \
    int k        = static_cast<int>(C.size(0));                              \
    int is_homo  = (weights.size(0) == 1) ? 1 : 0;                          \
    WEIGHT_C_T* d_c = static_cast<WEIGHT_C_T*>(C.data_ptr());               \
    /* Zero output buffer before scatter (atomicAdds assume zero base) */   \
    cudaMemsetAsync(d_c, 0, (size_t)k * (size_t)n * sizeof(WEIGHT_C_T), s); \
    int c_blocks = (n + 31) / 32;                                            \
    dim3 grid(m, c_blocks);                                                  \
    _csrmm_t_warp_kern##SUFFIX<<<grid, 32, 0, s>>>(                          \
        static_cast<const WEIGHT_C_T*>(weights.data_ptr()),                  \
        static_cast<const int32_t*>(indices.data_ptr()),                     \
        static_cast<const int32_t*>(indptr.data_ptr()),                      \
        static_cast<const SPIKE_C_T*>(B.data_ptr()),                         \
        d_c, m, n, is_homo);                                                 \
}


// =========================================================================
// Instantiate TVM FFI entry points via macros + @tvm_ffi annotations
// =========================================================================

// ---- Float32 (shm for NT_block: 8*32*sizeof(float) = 1024 bytes) ----
// @tvm_ffi binary_csrmm_nt_warp_f32_bool
FFI_CSRMM_NT_WARP(_f32_bool,  float,  int8_t)
// @tvm_ffi binary_csrmm_nt_warp_f32_float
FFI_CSRMM_NT_WARP(_f32_float, float,  float)
// @tvm_ffi binary_csrmm_nt_block_f32_bool
FFI_CSRMM_NT_BLOCK(_f32_bool,  float,  int8_t, 8 * 32 * sizeof(float))
// @tvm_ffi binary_csrmm_nt_block_f32_float
FFI_CSRMM_NT_BLOCK(_f32_float, float,  float,  8 * 32 * sizeof(float))
// @tvm_ffi binary_csrmm_nt_auto_f32_bool
FFI_CSRMM_NT_AUTO(_f32_bool,  float,  int8_t, 8 * 32 * sizeof(float))
// @tvm_ffi binary_csrmm_nt_auto_f32_float
FFI_CSRMM_NT_AUTO(_f32_float, float,  float,  8 * 32 * sizeof(float))
// @tvm_ffi binary_csrmm_t_warp_f32_bool
FFI_CSRMM_T_WARP(_f32_bool,  float,  int8_t)
// @tvm_ffi binary_csrmm_t_warp_f32_float
FFI_CSRMM_T_WARP(_f32_float, float,  float)

// ---- Float64 (shm for NT_block: 8*32*sizeof(double) = 2048 bytes) ----
// @tvm_ffi binary_csrmm_nt_auto_f64_bool
FFI_CSRMM_NT_AUTO(_f64_bool,  double, int8_t, 8 * 32 * sizeof(double))
// @tvm_ffi binary_csrmm_nt_auto_f64_float
FFI_CSRMM_NT_AUTO(_f64_float, double, float,  8 * 32 * sizeof(double))
// @tvm_ffi binary_csrmm_t_warp_f64_bool
FFI_CSRMM_T_WARP(_f64_bool,  double, int8_t)
// @tvm_ffi binary_csrmm_t_warp_f64_float
FFI_CSRMM_T_WARP(_f64_float, double, float)

// ---- Float16 (accumulates in f32; shm: 8*32*sizeof(float); atomicAdd sm_70+) ----
// @tvm_ffi binary_csrmm_nt_auto_f16_bool
FFI_CSRMM_NT_AUTO(_f16_bool,  __half, int8_t, 8 * 32 * sizeof(float))
// @tvm_ffi binary_csrmm_nt_auto_f16_float
FFI_CSRMM_NT_AUTO(_f16_float, __half, float,  8 * 32 * sizeof(float))
// @tvm_ffi binary_csrmm_t_warp_f16_bool
FFI_CSRMM_T_WARP(_f16_bool,  __half, int8_t)
// @tvm_ffi binary_csrmm_t_warp_f16_float
FFI_CSRMM_T_WARP(_f16_float, __half, float)

// ---- BFloat16 (accumulates in f32; shm: 8*32*sizeof(float); atomicAdd sm_80+) ----
// @tvm_ffi binary_csrmm_nt_auto_bf16_bool
FFI_CSRMM_NT_AUTO(_bf16_bool,  __nv_bfloat16, int8_t, 8 * 32 * sizeof(float))
// @tvm_ffi binary_csrmm_nt_auto_bf16_float
FFI_CSRMM_NT_AUTO(_bf16_float, __nv_bfloat16, float,  8 * 32 * sizeof(float))
// @tvm_ffi binary_csrmm_t_warp_bf16_bool
FFI_CSRMM_T_WARP(_bf16_bool,  __nv_bfloat16, int8_t)
// @tvm_ffi binary_csrmm_t_warp_bf16_float
FFI_CSRMM_T_WARP(_bf16_float, __nv_bfloat16, float)
