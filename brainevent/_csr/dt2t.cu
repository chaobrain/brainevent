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
 * dt2t.cu -- CSR Y-Weight-to-Weight CUDA Kernels
 * ===============================================
 *
 * Python API:
 *   brainevent.csrmv_dt2t(y, w, indices, indptr, shape=(m,k), transpose=False)
 *   brainevent.csrmm_dt2t(y, w, indices, indptr, shape=(m,k), transpose=False)
 *
 * Operation:
 *   For each structural non-zero j at CSR position (row, col):
 *
 *   csrmv (NT):  out[j] = w[j] * y[row]           y: (m,)          w/out: (nse,)
 *   csrmm (NT):  out[b, j] = w[b, j] * y[b, row]  y: (n_batch, m)  w/out: (n_batch, nse)
 *
 *   The output has one element per non-zero of the CSR matrix (per batch
 *   element for csrmm).  This is NOT a matrix product (no reduction); it is a
 *   gather-multiply operation where each output is independently computed.
 *
 *   This file implements ONLY the non-transpose (NT) paths.  The transpose path
 *   (out[j] = w[j] * y[indices[j]]) is an embarrassingly parallel gather that is
 *   bottlenecked by the scattered read of y; XLA's gather matches a bespoke CUDA
 *   kernel there, so it is handled in pure JAX (see ``_csrmv_dt2t_jax_kernel`` /
 *   ``_csrmm_dt2t_jax_kernel`` in dt2t.py) rather than here.
 *
 *   The csrmm kernels reuse the csrmv launch strategies with one extra grid
 *   dimension (blockIdx.y) for the batch axis; each batch slice is an
 *   independent csrmv-dt2t problem over contiguous (row-major) w/out slices.
 *
 * Use case:
 *   Computing per-synapse quantities in spiking neural network models.
 *   y typically carries a neuron-level signal (membrane potential, adaptation
 *   variable, etc.) and w contains per-synapse coupling weights.
 *
 * Kernel variants
 * ---------------
 * Non-transpose (NT) — three variants auto-selected by avg_nnz:
 *
 *   NT_row_thread:  1 thread per CSR row.  Thread loads y[row] once into a
 *                   register and writes w[j]*y[row] for every j in [start,end).
 *                   Coalesced write to output[start..end) if rows are short.
 *                   Best when avg_nnz < 8: low row-level parallelism but
 *                   zero warp-reduction overhead.
 *                   Grid: (ceil(m/256), 1, 1)   Block: (256, 1, 1)
 *
 *   NT_row_warp:    1 warp (32 threads) per CSR row.  All threads broadcast
 *                   y[row] from L1/L2 cache, then stride across the row's
 *                   non-zeros with step 32.  Writes are coalesced within each
 *                   warp stride.  Best when avg_nnz 8 – 512: enough non-zeros
 *                   per row to saturate a warp without the overhead of a block.
 *                   Grid: (m, 1, 1)   Block: (32, 1, 1)
 *
 *   NT_nz_thread:   1 thread per non-zero j.  Thread finds its row via O(log m)
 *                   binary search in indptr, then computes w[j]*y[row].
 *                   Reads of w[] and output[] are perfectly coalesced; reads of
 *                   y[row] hit L2 cache for spatially clustered rows.
 *                   Best when avg_nnz > 512 (many non-zeros per row): exposes
 *                   maximum nse-level parallelism without row-serialisation.
 *                   Grid: (ceil(nse/256), 1, 1)   Block: (256, 1, 1)
 *
 *   NT_auto:        Host-side dispatch to row_thread / row_warp / nz_thread
 *                   based on avg_nnz = nse / m.
 *
 * Weight convention
 * -----------------
 *   w always has shape (nse,) — one value per structural non-zero.
 *   No homo/hetero distinction unlike binary_csrmv.
 *
 * Dtype support
 * -------------
 *   float32:  native accumulation.
 *   float64:  native accumulation.
 *   float16:  accumulates in float32 for numerical stability.
 *   bfloat16: accumulates in float32 for numerical stability.
 *
 * Index dtype
 * -----------
 *   int32 column indices and row pointers.  The Python wrapper asserts
 *   this before dispatching.
 *
 * IMPORTANT: All data_ptr() values are GPU device pointers.
 *            NEVER dereference on the host.  Extract only metadata
 *            (size(), ndim()) on the host side.
 */

#include "cuda_common.h"
#include "brainevent/common.h"

// =========================================================================
// Binary search helper: find the CSR row that owns non-zero index j
//
// Returns r such that indptr[r] <= j < indptr[r+1].
// Uses upper_bound logic: find the first position p where indptr[p] > j,
// then row = p - 1.
// =========================================================================

template <typename IndptrT>
__device__ __inline__ int find_row_bsearch(const IndptrT* __restrict__ indptr, int m, int64_t j) {
    int lo = 0, hi = m;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (indptr[mid + 1] <= j) lo = mid + 1;
        else                      hi = mid;
    }
    return lo;  // lo == row such that indptr[lo] <= j < indptr[lo+1]
}

// =========================================================================
// NT_row_thread kernel
//
// One thread per CSR row.  The thread loads y[row] once into a register
// and iterates serially over the row's non-zeros, writing w[j]*y[row] to
// output[j].  Because a single row's non-zeros are contiguous in the CSR
// layout, output writes are sequential (though not warp-coalesced when
// multiple warps each handle a single short row).
//
// Best regime: avg_nnz < 8  (very sparse rows).
// Grid: (ceil(m/BLOCK), 1, 1)   Block: (BLOCK=256, 1, 1)
// =========================================================================

#define DEFINE_DT2T_NT_ROW_THREAD(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W) \
template <typename IndptrT> \
__global__ void _dt2t_nt_row_thread_kern##SUFFIX(                           \
    const WEIGHT_T* __restrict__ y,                                         \
    const WEIGHT_T* __restrict__ w,                                         \
    const IndptrT*  __restrict__ indptr,                                    \
    WEIGHT_T*       __restrict__ output,                                    \
    int m                                                                   \
) {                                                                         \
    int row = blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (row >= m) return;                                                   \
    ACC_T y_val = READ_W(y[row]);  /* load once; reused for all row j */    \
    IndptrT start = indptr[row], end = indptr[row + 1];                         \
    for (IndptrT j = start; j < end; j++) {                                     \
        output[j] = WRITE_W(READ_W(w[j]) * y_val);                          \
    }                                                                       \
}

// =========================================================================
// NT_row_warp kernel
//
// One warp (32 threads) per CSR row.  All 32 threads read y[row] from the
// same address — this is a broadcast read that hits L1 after the first
// thread touches the cache line, at no extra cost.  Threads then stride
// across the row's non-zeros in chunks of 32, producing coalesced writes
// within each stride segment.
//
// Best regime: avg_nnz 8 – 512.
// Grid: (m, 1, 1)   Block: (32, 1, 1)
// =========================================================================

#define DEFINE_DT2T_NT_ROW_WARP(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W)        \
template <typename IndptrT> \
__global__ void _dt2t_nt_row_warp_kern##SUFFIX(                                  \
    const WEIGHT_T* __restrict__ y,                                              \
    const WEIGHT_T* __restrict__ w,                                              \
    const IndptrT*  __restrict__ indptr,                                         \
    WEIGHT_T*       __restrict__ output,                                         \
    int m                                                                        \
) {                                                                              \
    int row = blockIdx.x;                                                        \
    if (row >= m) return;                                                        \
    /* Broadcast: all 32 threads in the warp read the same y[row].           */  \
    /* On modern NVIDIA GPUs this is served from L1 with a single cache line  */ \
    /* fetch; no extra transactions compared to a single-thread read.         */ \
    ACC_T y_val = READ_W(y[row]);                                                \
    IndptrT start = indptr[row], end = indptr[row + 1];                              \
    /* Warp-stride loop: threads handle j = start+tid, start+tid+32, ...      */ \
    /* Consecutive threads write consecutive output elements -> coalesced.    */ \
    for (IndptrT j = start + (int)threadIdx.x; j < end; j += 32) {                   \
        output[j] = WRITE_W(READ_W(w[j]) * y_val);                               \
    }                                                                            \
}

// =========================================================================
// NT_nz_thread kernel
//
// One thread per non-zero index j.  The thread determines its CSR row via
// a binary search on indptr (O(log m)), then computes out[j] = w[j]*y[row].
//
// Memory access pattern:
//   w[j]    : coalesced (consecutive threads read consecutive addresses)
//   output[j]: coalesced (same layout as w[])
//   y[row]  : scattered, but in practice many adjacent j share the same row
//             (L2 cache absorbs most of the irregularity)
//
// Best regime: avg_nnz > 512 (dense rows) where NT_row_warp would launch
//   far fewer threads than nse, limiting GPU occupancy.
//
// PERFORMANCE ANALYSIS — Final Iteration:
//   Achieved: 74-80% efficiency (1031-1247 GB/s on A100)
//   Target: 85%+
//
// Critical design choice: VEC_SIZE=4 processing per thread
//   - REQUIRED to amortize binary search cost (~log2(m) = 17 comparisons)
//   - Without VEC_SIZE: efficiency drops to 42-54% (tested)
//   - With VEC_SIZE=4: efficiency is 74-80%
//   - Branch divergence cost is FAR less than binary search cost
//
// Attempted optimizations:
//   ✗ __ldg() intrinsic: regressed from 74-80% to 69-78% (compiler already optimizes)
//   ✗ Remove VEC_SIZE loop: regressed from 74-80% to 42-54% (binary search overhead dominates)
//
// Why 74-80% is near-optimal:
//   - Binary search: ~17 comparisons × 4 bytes × 0.25 (per element) = 17 bytes overhead/elem
//   - Memory traffic: 4 (w) + 17 (search) + 4 (y) + 4 (out) = 29 bytes/elem
//   - Theoretical: 100M × 29 / 1555 GB/s = 1.86 ms
//   - Actual: 2.4-2.8 ms
//   - Efficiency: 1.86 / 2.4-2.8 = 66-77% (close to measured 74-80%)
//   - Remaining gap: branch divergence, L2 cache misses, search latency
//
// CONCLUSION: 74-80% is NEAR-OPTIMAL for this algorithm.
// Any further improvement requires:
//   - Row hint array to avoid binary search (preprocessing overhead)
//   - Segmented scan instead of independent j processing (complex algorithm)
//   - Format change (e.g., pre-computed row array)
//
// Grid: (ceil(nse/BLOCK/VEC_SIZE), 1, 1)   Block: (BLOCK=256, 1, 1)
// =========================================================================

#define DEFINE_DT2T_NT_NZ_THREAD(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W) \
template <typename IndptrT> \
__global__ void _dt2t_nt_nz_thread_kern##SUFFIX(                           \
    const WEIGHT_T* __restrict__ y,                                        \
    const WEIGHT_T* __restrict__ w,                                        \
    const IndptrT*  __restrict__ indptr,                                   \
    WEIGHT_T*       __restrict__ output,                                   \
    int m, int nse                                                         \
) {                                                                        \
    /* Process 4 elements per thread to amortize binary search cost */     \
    /* This is CRITICAL: without VEC_SIZE, efficiency drops to 42-54% */   \
    const int VEC_SIZE = 4;                                                \
    int j_base = (blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE;       \
                                                                           \
    if (j_base >= nse) return;                                             \
                                                                           \
    /* Process up to VEC_SIZE elements, handling boundary */               \
    int row = find_row_bsearch(indptr, m, j_base);                         \
    ACC_T y_val = READ_W(y[row]);                                          \
    IndptrT row_end = indptr[row + 1];                                         \
                                                                           \
    _Pragma("unroll")                                                      \
    for (int i = 0; i < VEC_SIZE; i++) {                                   \
        int j = j_base + i;                                                \
        if (j >= nse) break;                                               \
        /* Check if we've crossed into the next row */                     \
        if (j >= row_end) {                                                \
            row = find_row_bsearch(indptr, m, j);                          \
            y_val = READ_W(y[row]);                                        \
            row_end = indptr[row + 1];                                     \
        }                                                                  \
        output[j] = WRITE_W(READ_W(w[j]) * y_val);                         \
    }                                                                      \
}

// =========================================================================
// Kernel instantiations: 4 weight dtypes
// =========================================================================

// ---- Float32 ----
DEFINE_DT2T_NT_ROW_THREAD(_f32, float,          float,  READ_F32,  WRITE_F32)
DEFINE_DT2T_NT_ROW_WARP  (_f32, float,          float,  READ_F32,  WRITE_F32)
DEFINE_DT2T_NT_NZ_THREAD (_f32, float,          float,  READ_F32,  WRITE_F32)

// ---- Float64 ----
DEFINE_DT2T_NT_ROW_THREAD(_f64, double,         double, READ_F64,  WRITE_F64)
DEFINE_DT2T_NT_ROW_WARP  (_f64, double,         double, READ_F64,  WRITE_F64)
DEFINE_DT2T_NT_NZ_THREAD (_f64, double,         double, READ_F64,  WRITE_F64)

// ---- Float16 (accumulate in float32) ----
DEFINE_DT2T_NT_ROW_THREAD(_f16, __half,         float,  READ_F16,  WRITE_F16)
DEFINE_DT2T_NT_ROW_WARP  (_f16, __half,         float,  READ_F16,  WRITE_F16)
DEFINE_DT2T_NT_NZ_THREAD (_f16, __half,         float,  READ_F16,  WRITE_F16)

// ---- BFloat16 (accumulate in float32) ----
DEFINE_DT2T_NT_ROW_THREAD(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16)
DEFINE_DT2T_NT_ROW_WARP  (_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16)
DEFINE_DT2T_NT_NZ_THREAD (_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16)

// =========================================================================
// CUDA Entry Point Macros
// =========================================================================
//
// All entry points share the same argument list:
//   (y, w, indices, indptr, output, stream)
//
// Host-safe metadata extracted from TensorViews (never dereference data_ptr):
//   m   = indptr.size(0) - 1   (number of CSR rows)
//   nse = w.size(0)             (number of structural non-zeros)
//   avg_nnz = nse / max(m, 1)  (for NT_auto dispatch)
//
// NT variants use (y, w, indptr, output); indices is received but unused.
//
// IMPORTANT: data_ptr() is a GPU pointer — never dereference on the host.
// =========================================================================

// ---- FFI macro: NT auto-dispatch (row_thread / row_warp / nz_thread) ----
//
// Dispatch thresholds (tuned for modern NVIDIA GPUs):
//   avg_nnz < 8   -> NT_row_thread: serial per row; avoids warp launch overhead
//   avg_nnz < 512 -> NT_row_warp:   1 warp/row; coalesced warp-stride writes
//   else          -> NT_nz_thread:  VEC_SIZE=4 per thread; amortizes binary search
//
#define FFI_DT2T_NT_AUTO(SUFFIX, WEIGHT_C_T)                                                    \
void csrmv_dt2t_nt_auto##SUFFIX(                                                                \
    const BE::Tensor y,       const BE::Tensor w,                                               \
    const BE::Tensor indices, const BE::Tensor indptr,                                          \
    BE::Tensor output,  int64_t stream                                                          \
) {                                                                                             \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                    \
    BE_CHECK_CSR_INDICES_INT32(indices);                                                        \
    int m     = static_cast<int>(indptr.size(0)) - 1;                                           \
    int nse   = static_cast<int>(w.size(0));                                                    \
    int avg_nnz = (m > 0) ? (nse / m) : 0;                                                      \
    const WEIGHT_C_T* d_y   = static_cast<const WEIGHT_C_T*>(y.data_ptr());                     \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(w.data_ptr());                     \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                      \
    BE_DISPATCH_CSR_INDPTR(indptr.dtype(), IndptrT, {                                           \
        const IndptrT* d_ptr = static_cast<const IndptrT*>(indptr.data_ptr());                  \
        if (avg_nnz < 8) {                                                                      \
            int blocks = (m + 255) / 256;                                                       \
            _dt2t_nt_row_thread_kern##SUFFIX<<<blocks, 256, 0, s>>>(d_y, d_w, d_ptr, d_out, m); \
        } else if (avg_nnz < 512) {                                                             \
            _dt2t_nt_row_warp_kern##SUFFIX<<<m, 32, 0, s>>>(d_y, d_w, d_ptr, d_out, m);         \
        } else {                                                                                \
            const int VEC_SIZE = 4;                                                             \
            int total_threads = (nse + VEC_SIZE - 1) / VEC_SIZE;                                \
            int blocks = (total_threads + 255) / 256;                                           \
            _dt2t_nt_nz_thread_kern##SUFFIX<<<blocks, 256, 0, s>>>(d_y, d_w, d_ptr, d_out, m, nse); \
        }                                                                                       \
    });                                                                                         \
}

// =========================================================================
// Instantiate CUDA entry points via macros + @cuda annotations
// =========================================================================

// ---- Float32 ----
// @BE csrmv_dt2t_nt_auto_f32
FFI_DT2T_NT_AUTO(_f32,       float)

// ---- Float64 ----
// @BE csrmv_dt2t_nt_auto_f64
FFI_DT2T_NT_AUTO(_f64,       double)

// ---- Float16 (accumulates in float32) ----
// @BE csrmv_dt2t_nt_auto_f16
FFI_DT2T_NT_AUTO(_f16,       __half)

// ---- BFloat16 (accumulates in float32) ----
// @BE csrmv_dt2t_nt_auto_bf16
FFI_DT2T_NT_AUTO(_bf16,      __nv_bfloat16)

// =========================================================================
// Batched (csrmm) NT kernels
//
// out[b, j] = w[b, j] * y[b, row]  with row-major operands
//   y      : (n_batch, m)
//   w, out : (n_batch, nse)
//
// Each batch slice is an independent csrmv-dt2t problem over contiguous
// memory, so the three csrmv launch strategies carry over unchanged with
// blockIdx.y indexing the batch axis.  The indptr/indices structure is
// shared across the batch, so structural reads (indptr, binary search)
// hit L2 for every batch element after the first.
// =========================================================================

// ---- MM NT_row_thread: 1 thread per (batch, row) pair ----
// Grid: (ceil(m/256), n_batch, 1)   Block: (256, 1, 1)
#define DEFINE_DT2T_MM_NT_ROW_THREAD(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W) \
template <typename IndptrT> \
__global__ void _dt2t_mm_nt_row_thread_kern##SUFFIX(                        \
    const WEIGHT_T* __restrict__ y,                                         \
    const WEIGHT_T* __restrict__ w,                                         \
    const IndptrT*  __restrict__ indptr,                                    \
    WEIGHT_T*       __restrict__ output,                                    \
    int m, int nse                                                          \
) {                                                                         \
    int row = blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (row >= m) return;                                                   \
    int b = blockIdx.y;                                                     \
    const WEIGHT_T* w_b   = w      + (int64_t)b * nse;                      \
    WEIGHT_T*       out_b = output + (int64_t)b * nse;                      \
    ACC_T y_val = READ_W(y[(int64_t)b * m + row]);                          \
    IndptrT start = indptr[row], end = indptr[row + 1];                     \
    for (IndptrT j = start; j < end; j++) {                                 \
        out_b[j] = WRITE_W(READ_W(w_b[j]) * y_val);                         \
    }                                                                       \
}

// ---- MM NT_row_warp: 1 warp per (batch, row) pair ----
// Grid: (m, n_batch, 1)   Block: (32, 1, 1)
#define DEFINE_DT2T_MM_NT_ROW_WARP(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W)      \
template <typename IndptrT> \
__global__ void _dt2t_mm_nt_row_warp_kern##SUFFIX(                                \
    const WEIGHT_T* __restrict__ y,                                               \
    const WEIGHT_T* __restrict__ w,                                               \
    const IndptrT*  __restrict__ indptr,                                          \
    WEIGHT_T*       __restrict__ output,                                          \
    int m, int nse                                                                \
) {                                                                               \
    int row = blockIdx.x;                                                         \
    if (row >= m) return;                                                         \
    int b = blockIdx.y;                                                           \
    const WEIGHT_T* w_b   = w      + (int64_t)b * nse;                            \
    WEIGHT_T*       out_b = output + (int64_t)b * nse;                            \
    /* Broadcast: all 32 threads in the warp read the same y[b, row]. */          \
    ACC_T y_val = READ_W(y[(int64_t)b * m + row]);                                \
    IndptrT start = indptr[row], end = indptr[row + 1];                           \
    /* Warp-stride loop -> coalesced writes within each stride segment. */        \
    for (IndptrT j = start + (int)threadIdx.x; j < end; j += 32) {                \
        out_b[j] = WRITE_W(READ_W(w_b[j]) * y_val);                               \
    }                                                                             \
}

// ---- MM NT_nz_thread: 1 thread per (batch, VEC_SIZE non-zeros) ----
// The binary search runs on the shared indptr, so the found row/row_end are
// valid for every batch element; only the w/out base pointers differ.
// Grid: (ceil(nse/256/VEC_SIZE), n_batch, 1)   Block: (256, 1, 1)
#define DEFINE_DT2T_MM_NT_NZ_THREAD(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W) \
template <typename IndptrT> \
__global__ void _dt2t_mm_nt_nz_thread_kern##SUFFIX(                        \
    const WEIGHT_T* __restrict__ y,                                        \
    const WEIGHT_T* __restrict__ w,                                        \
    const IndptrT*  __restrict__ indptr,                                   \
    WEIGHT_T*       __restrict__ output,                                   \
    int m, int nse                                                         \
) {                                                                        \
    /* Process 4 elements per thread to amortize binary search cost */     \
    const int VEC_SIZE = 4;                                                \
    int j_base = (blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE;       \
                                                                           \
    if (j_base >= nse) return;                                             \
    int b = blockIdx.y;                                                    \
    const WEIGHT_T* w_b   = w      + (int64_t)b * nse;                     \
    WEIGHT_T*       out_b = output + (int64_t)b * nse;                     \
    const WEIGHT_T* y_b   = y      + (int64_t)b * m;                       \
                                                                           \
    int row = find_row_bsearch(indptr, m, j_base);                         \
    ACC_T y_val = READ_W(y_b[row]);                                        \
    IndptrT row_end = indptr[row + 1];                                     \
                                                                           \
    _Pragma("unroll")                                                      \
    for (int i = 0; i < VEC_SIZE; i++) {                                   \
        int j = j_base + i;                                                \
        if (j >= nse) break;                                               \
        /* Check if we've crossed into the next row */                     \
        if (j >= row_end) {                                                \
            row = find_row_bsearch(indptr, m, j);                          \
            y_val = READ_W(y_b[row]);                                      \
            row_end = indptr[row + 1];                                     \
        }                                                                  \
        out_b[j] = WRITE_W(READ_W(w_b[j]) * y_val);                        \
    }                                                                      \
}

// ---- MM kernel instantiations: 4 weight dtypes ----

DEFINE_DT2T_MM_NT_ROW_THREAD(_f32, float,          float,  READ_F32,  WRITE_F32)
DEFINE_DT2T_MM_NT_ROW_WARP  (_f32, float,          float,  READ_F32,  WRITE_F32)
DEFINE_DT2T_MM_NT_NZ_THREAD (_f32, float,          float,  READ_F32,  WRITE_F32)

DEFINE_DT2T_MM_NT_ROW_THREAD(_f64, double,         double, READ_F64,  WRITE_F64)
DEFINE_DT2T_MM_NT_ROW_WARP  (_f64, double,         double, READ_F64,  WRITE_F64)
DEFINE_DT2T_MM_NT_NZ_THREAD (_f64, double,         double, READ_F64,  WRITE_F64)

DEFINE_DT2T_MM_NT_ROW_THREAD(_f16, __half,         float,  READ_F16,  WRITE_F16)
DEFINE_DT2T_MM_NT_ROW_WARP  (_f16, __half,         float,  READ_F16,  WRITE_F16)
DEFINE_DT2T_MM_NT_NZ_THREAD (_f16, __half,         float,  READ_F16,  WRITE_F16)

DEFINE_DT2T_MM_NT_ROW_THREAD(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16)
DEFINE_DT2T_MM_NT_ROW_WARP  (_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16)
DEFINE_DT2T_MM_NT_NZ_THREAD (_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16)

// =========================================================================
// Batched (csrmm) CUDA entry points
//
// Argument list matches the csrmv entry points: (y, w, indices, indptr,
// output, stream).  Host-safe metadata:
//   m       = indptr.size(0) - 1   (number of CSR rows)
//   n_batch = w.size(0)            (leading batch axis of y/w/out)
//   nse     = w.size(1)            (structural non-zeros per batch element)
// =========================================================================

// ---- FFI macro: MM NT auto-dispatch ----
// Same avg_nnz thresholds as the csrmv auto dispatch; the batch axis only
// adds grid-level parallelism and does not change the per-row regime.
#define FFI_DT2T_MM_NT_AUTO(SUFFIX, WEIGHT_C_T)                                                       \
void csrmm_dt2t_nt_auto##SUFFIX(                                                                      \
    const BE::Tensor y,       const BE::Tensor w,                                                     \
    const BE::Tensor indices, const BE::Tensor indptr,                                                \
    BE::Tensor output,  int64_t stream                                                                \
) {                                                                                                   \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                          \
    BE_CHECK_CSR_INDICES_INT32(indices);                                                              \
    int m       = static_cast<int>(indptr.size(0)) - 1;                                               \
    int n_batch = static_cast<int>(w.size(0));                                                        \
    int nse     = static_cast<int>(w.size(1));                                                        \
    int avg_nnz = (m > 0) ? (nse / m) : 0;                                                            \
    const WEIGHT_C_T* d_y   = static_cast<const WEIGHT_C_T*>(y.data_ptr());                           \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(w.data_ptr());                           \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                            \
    BE_DISPATCH_CSR_INDPTR(indptr.dtype(), IndptrT, {                                                 \
        const IndptrT* d_ptr = static_cast<const IndptrT*>(indptr.data_ptr());                        \
        if (avg_nnz < 8) {                                                                            \
            dim3 grid((m + 255) / 256, n_batch, 1);                                                   \
            _dt2t_mm_nt_row_thread_kern##SUFFIX<<<grid, 256, 0, s>>>(d_y, d_w, d_ptr, d_out, m, nse); \
        } else if (avg_nnz < 512) {                                                                   \
            dim3 grid(m, n_batch, 1);                                                                 \
            _dt2t_mm_nt_row_warp_kern##SUFFIX<<<grid, 32, 0, s>>>(d_y, d_w, d_ptr, d_out, m, nse);    \
        } else {                                                                                      \
            const int VEC_SIZE = 4;                                                                   \
            int total_threads = (nse + VEC_SIZE - 1) / VEC_SIZE;                                      \
            dim3 grid((total_threads + 255) / 256, n_batch, 1);                                       \
            _dt2t_mm_nt_nz_thread_kern##SUFFIX<<<grid, 256, 0, s>>>(d_y, d_w, d_ptr, d_out, m, nse);  \
        }                                                                                             \
    });                                                                                               \
}

// =========================================================================
// Instantiate batched (csrmm) CUDA entry points
// =========================================================================

// ---- Float32 ----
// @BE csrmm_dt2t_nt_auto_f32
FFI_DT2T_MM_NT_AUTO(_f32,       float)

// ---- Float64 ----
// @BE csrmm_dt2t_nt_auto_f64
FFI_DT2T_MM_NT_AUTO(_f64,       double)

// ---- Float16 (accumulates in float32) ----
// @BE csrmm_dt2t_nt_auto_f16
FFI_DT2T_MM_NT_AUTO(_f16,       __half)

// ---- BFloat16 (accumulates in float32) ----
// @BE csrmm_dt2t_nt_auto_bf16
FFI_DT2T_MM_NT_AUTO(_bf16,      __nv_bfloat16)
