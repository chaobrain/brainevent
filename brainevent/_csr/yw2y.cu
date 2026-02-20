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
 * csrmv_yw2y.cu -- CSR Weighted-to-Nonzero CUDA Kernels
 * ======================================================
 *
 * Python API:
 *   brainevent.csrmv_yw2y(y, w, indices, indptr, shape=(m,k), transpose=False)
 *
 * Operation:
 *   For each structural non-zero j at CSR position (row, col):
 *
 *   Non-transpose (NT):  out[j] = w[j] * y[row]
 *   Transpose    (T):    out[j] = w[j] * y[col]   (col = indices[j])
 *
 *   The output has shape (nse,), one element per non-zero of the CSR matrix.
 *   This is NOT a matrix-vector product (no reduction); it is a gather-multiply
 *   operation where each output is independently computed.
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
 * Transpose (T):
 *   T_nz_thread:    1 thread per non-zero j.  Computes out[j] = w[j]*y[indices[j]]
 *                   directly — no scatter, no atomics.  Reads of w[], indices[],
 *                   and output[] are coalesced; reads of y[] are scattered
 *                   (indirect gather).  This is the only T variant because the
 *                   operation is embarrassingly parallel.
 *                   Grid: (ceil(nse/256), 1, 1)   Block: (256, 1, 1)
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
// Binary search helper: find the CSR row that owns non-zero index j
//
// Returns r such that indptr[r] <= j < indptr[r+1].
// Uses upper_bound logic: find the first position p where indptr[p] > j,
// then row = p - 1.
// =========================================================================

__device__ __inline__ int find_row_bsearch(const int32_t* __restrict__ indptr, int m, int j) {
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

#define DEFINE_YW2Y_NT_ROW_THREAD(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W)   \
__global__ void _yw2y_nt_row_thread_kern##SUFFIX(                              \
    const WEIGHT_T* __restrict__ y,                                            \
    const WEIGHT_T* __restrict__ w,                                            \
    const int32_t*  __restrict__ indptr,                                       \
    WEIGHT_T*       __restrict__ output,                                       \
    int m                                                                      \
) {                                                                             \
    int row = blockIdx.x * blockDim.x + threadIdx.x;                          \
    if (row >= m) return;                                                      \
    ACC_T y_val = READ_W(y[row]);  /* load once; reused for all row j */      \
    int start = indptr[row], end = indptr[row + 1];                            \
    for (int j = start; j < end; j++) {                                        \
        output[j] = WRITE_W(READ_W(w[j]) * y_val);                            \
    }                                                                           \
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

#define DEFINE_YW2Y_NT_ROW_WARP(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W)    \
__global__ void _yw2y_nt_row_warp_kern##SUFFIX(                               \
    const WEIGHT_T* __restrict__ y,                                            \
    const WEIGHT_T* __restrict__ w,                                            \
    const int32_t*  __restrict__ indptr,                                       \
    WEIGHT_T*       __restrict__ output,                                       \
    int m                                                                      \
) {                                                                             \
    int row = blockIdx.x;                                                      \
    if (row >= m) return;                                                      \
    /* Broadcast: all 32 threads in the warp read the same y[row].           */ \
    /* On modern NVIDIA GPUs this is served from L1 with a single cache line  */ \
    /* fetch; no extra transactions compared to a single-thread read.         */ \
    ACC_T y_val = READ_W(y[row]);                                              \
    int start = indptr[row], end = indptr[row + 1];                            \
    /* Warp-stride loop: threads handle j = start+tid, start+tid+32, ...      */ \
    /* Consecutive threads write consecutive output elements -> coalesced.    */ \
    for (int j = start + (int)threadIdx.x; j < end; j += 32) {                \
        output[j] = WRITE_W(READ_W(w[j]) * y_val);                            \
    }                                                                           \
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
// Grid: (ceil(nse/BLOCK), 1, 1)   Block: (BLOCK=256, 1, 1)
// =========================================================================

#define DEFINE_YW2Y_NT_NZ_THREAD(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W)   \
__global__ void _yw2y_nt_nz_thread_kern##SUFFIX(                              \
    const WEIGHT_T* __restrict__ y,                                            \
    const WEIGHT_T* __restrict__ w,                                            \
    const int32_t*  __restrict__ indptr,                                       \
    WEIGHT_T*       __restrict__ output,                                       \
    int m, int nse                                                             \
) {                                                                             \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                            \
    if (j >= nse) return;                                                      \
    int row = find_row_bsearch(indptr, m, j);                                  \
    output[j] = WRITE_W(READ_W(w[j]) * READ_W(y[row]));                       \
}

// =========================================================================
// T_nz_thread kernel (transpose)
//
// One thread per non-zero j.  Computes out[j] = w[j] * y[indices[j]].
// This is the transpose variant: the column index (indices[j]) addresses y
// rather than the row index.
//
// Memory access pattern:
//   w[j]       : coalesced
//   indices[j] : coalesced
//   y[indices[j]]: scattered (indirect gather); quality depends on sparsity
//   output[j]  : coalesced
//
// No atomics needed (unlike binary_csrmv transpose): each output element
// out[j] is owned exclusively by thread j.
//
// This is the only T variant; being embarrassingly parallel, no further
// specialisation is needed.
// Grid: (ceil(nse/BLOCK), 1, 1)   Block: (BLOCK=256, 1, 1)
// =========================================================================

#define DEFINE_YW2Y_T_NZ_THREAD(SUFFIX, WEIGHT_T, ACC_T, READ_W, WRITE_W)    \
__global__ void _yw2y_t_nz_thread_kern##SUFFIX(                               \
    const WEIGHT_T* __restrict__ y,                                            \
    const WEIGHT_T* __restrict__ w,                                            \
    const int32_t*  __restrict__ indices,                                      \
    WEIGHT_T*       __restrict__ output,                                       \
    int nse                                                                    \
) {                                                                             \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                            \
    if (j >= nse) return;                                                      \
    /* Gather y at the column index of this non-zero */                        \
    output[j] = WRITE_W(READ_W(w[j]) * READ_W(y[indices[j]]));                \
}

// =========================================================================
// Kernel instantiations: 4 weight dtypes
// =========================================================================

// ---- Float32 ----
DEFINE_YW2Y_NT_ROW_THREAD(_f32, float,          float,  READ_F32,  WRITE_F32)
DEFINE_YW2Y_NT_ROW_WARP  (_f32, float,          float,  READ_F32,  WRITE_F32)
DEFINE_YW2Y_NT_NZ_THREAD (_f32, float,          float,  READ_F32,  WRITE_F32)
DEFINE_YW2Y_T_NZ_THREAD  (_f32, float,          float,  READ_F32,  WRITE_F32)

// ---- Float64 ----
DEFINE_YW2Y_NT_ROW_THREAD(_f64, double,         double, READ_F64,  WRITE_F64)
DEFINE_YW2Y_NT_ROW_WARP  (_f64, double,         double, READ_F64,  WRITE_F64)
DEFINE_YW2Y_NT_NZ_THREAD (_f64, double,         double, READ_F64,  WRITE_F64)
DEFINE_YW2Y_T_NZ_THREAD  (_f64, double,         double, READ_F64,  WRITE_F64)

// ---- Float16 (accumulate in float32) ----
DEFINE_YW2Y_NT_ROW_THREAD(_f16, __half,         float,  READ_F16,  WRITE_F16)
DEFINE_YW2Y_NT_ROW_WARP  (_f16, __half,         float,  READ_F16,  WRITE_F16)
DEFINE_YW2Y_NT_NZ_THREAD (_f16, __half,         float,  READ_F16,  WRITE_F16)
DEFINE_YW2Y_T_NZ_THREAD  (_f16, __half,         float,  READ_F16,  WRITE_F16)

// ---- BFloat16 (accumulate in float32) ----
DEFINE_YW2Y_NT_ROW_THREAD(_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16)
DEFINE_YW2Y_NT_ROW_WARP  (_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16)
DEFINE_YW2Y_NT_NZ_THREAD (_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16)
DEFINE_YW2Y_T_NZ_THREAD  (_bf16, __nv_bfloat16, float,  READ_BF16, WRITE_BF16)

// =========================================================================
// TVM FFI Entry Point Macros
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
// T  variant uses (y, w, indices, output); indptr is received but unused.
//
// IMPORTANT: data_ptr() is a GPU pointer — never dereference on the host.
// =========================================================================

// ---- FFI macro: NT row-thread kernel ----
#define FFI_YW2Y_NT_ROW_THREAD(SUFFIX, WEIGHT_C_T)                            \
void csrmv_yw2y_nt_row_thread##SUFFIX(                                         \
    tvm::ffi::TensorView y,       tvm::ffi::TensorView w,                     \
    tvm::ffi::TensorView indices, tvm::ffi::TensorView indptr,                 \
    tvm::ffi::TensorView output,  int64_t stream                               \
) {                                                                             \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                   \
    int m     = static_cast<int>(indptr.size(0)) - 1;                         \
    int blocks = (m + 255) / 256;                                              \
    _yw2y_nt_row_thread_kern##SUFFIX<<<blocks, 256, 0, s>>>(                   \
        static_cast<const WEIGHT_C_T*>(y.data_ptr()),                          \
        static_cast<const WEIGHT_C_T*>(w.data_ptr()),                          \
        static_cast<const int32_t*>(indptr.data_ptr()),                        \
        static_cast<WEIGHT_C_T*>(output.data_ptr()), m);                       \
}

// ---- FFI macro: NT row-warp kernel ----
#define FFI_YW2Y_NT_ROW_WARP(SUFFIX, WEIGHT_C_T)                              \
void csrmv_yw2y_nt_row_warp##SUFFIX(                                           \
    tvm::ffi::TensorView y,       tvm::ffi::TensorView w,                     \
    tvm::ffi::TensorView indices, tvm::ffi::TensorView indptr,                 \
    tvm::ffi::TensorView output,  int64_t stream                               \
) {                                                                             \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                   \
    int m = static_cast<int>(indptr.size(0)) - 1;                             \
    _yw2y_nt_row_warp_kern##SUFFIX<<<m, 32, 0, s>>>(                           \
        static_cast<const WEIGHT_C_T*>(y.data_ptr()),                          \
        static_cast<const WEIGHT_C_T*>(w.data_ptr()),                          \
        static_cast<const int32_t*>(indptr.data_ptr()),                        \
        static_cast<WEIGHT_C_T*>(output.data_ptr()), m);                       \
}

// ---- FFI macro: NT nz-thread kernel ----
#define FFI_YW2Y_NT_NZ_THREAD(SUFFIX, WEIGHT_C_T)                             \
void csrmv_yw2y_nt_nz_thread##SUFFIX(                                          \
    tvm::ffi::TensorView y,       tvm::ffi::TensorView w,                     \
    tvm::ffi::TensorView indices, tvm::ffi::TensorView indptr,                 \
    tvm::ffi::TensorView output,  int64_t stream                               \
) {                                                                             \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                   \
    int m   = static_cast<int>(indptr.size(0)) - 1;                           \
    int nse = static_cast<int>(w.size(0));                                     \
    int blocks = (nse + 255) / 256;                                            \
    _yw2y_nt_nz_thread_kern##SUFFIX<<<blocks, 256, 0, s>>>(                    \
        static_cast<const WEIGHT_C_T*>(y.data_ptr()),                          \
        static_cast<const WEIGHT_C_T*>(w.data_ptr()),                          \
        static_cast<const int32_t*>(indptr.data_ptr()),                        \
        static_cast<WEIGHT_C_T*>(output.data_ptr()), m, nse);                  \
}

// ---- FFI macro: NT auto-dispatch (row_thread / row_warp / nz_thread) ----
//
// Dispatch thresholds (tuned for modern NVIDIA GPUs):
//   avg_nnz < 8   -> NT_row_thread: serial per row; avoids warp launch overhead
//   avg_nnz < 512 -> NT_row_warp:   1 warp/row; coalesced warp-stride writes
//   else          -> NT_nz_thread:  1 thread/nz; max occupancy for dense rows
//
#define FFI_YW2Y_NT_AUTO(SUFFIX, WEIGHT_C_T)                                   \
void csrmv_yw2y_nt_auto##SUFFIX(                                               \
    tvm::ffi::TensorView y,       tvm::ffi::TensorView w,                     \
    tvm::ffi::TensorView indices, tvm::ffi::TensorView indptr,                 \
    tvm::ffi::TensorView output,  int64_t stream                               \
) {                                                                             \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                   \
    int m     = static_cast<int>(indptr.size(0)) - 1;                         \
    int nse   = static_cast<int>(w.size(0));                                   \
    int avg_nnz = (m > 0) ? (nse / m) : 0;                                    \
    const WEIGHT_C_T* d_y   = static_cast<const WEIGHT_C_T*>(y.data_ptr());   \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(w.data_ptr());   \
    const int32_t*    d_ptr = static_cast<const int32_t*>(indptr.data_ptr()); \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());    \
    if (avg_nnz < 8) {                                                         \
        int blocks = (m + 255) / 256;                                          \
        _yw2y_nt_row_thread_kern##SUFFIX<<<blocks, 256, 0, s>>>(               \
            d_y, d_w, d_ptr, d_out, m);                                        \
    } else if (avg_nnz < 512) {                                                \
        _yw2y_nt_row_warp_kern##SUFFIX<<<m, 32, 0, s>>>(                       \
            d_y, d_w, d_ptr, d_out, m);                                        \
    } else {                                                                   \
        int blocks = (nse + 255) / 256;                                        \
        _yw2y_nt_nz_thread_kern##SUFFIX<<<blocks, 256, 0, s>>>(                \
            d_y, d_w, d_ptr, d_out, m, nse);                                   \
    }                                                                           \
}

// ---- FFI macro: T nz-thread kernel (transpose) ----
#define FFI_YW2Y_T_NZ_THREAD(SUFFIX, WEIGHT_C_T)                              \
void csrmv_yw2y_t_nz_thread##SUFFIX(                                           \
    tvm::ffi::TensorView y,       tvm::ffi::TensorView w,                     \
    tvm::ffi::TensorView indices, tvm::ffi::TensorView indptr,                 \
    tvm::ffi::TensorView output,  int64_t stream                               \
) {                                                                             \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                   \
    int nse = static_cast<int>(w.size(0));                                     \
    int blocks = (nse + 255) / 256;                                            \
    _yw2y_t_nz_thread_kern##SUFFIX<<<blocks, 256, 0, s>>>(                     \
        static_cast<const WEIGHT_C_T*>(y.data_ptr()),                          \
        static_cast<const WEIGHT_C_T*>(w.data_ptr()),                          \
        static_cast<const int32_t*>(indices.data_ptr()),                       \
        static_cast<WEIGHT_C_T*>(output.data_ptr()), nse);                     \
}


// =========================================================================
// Instantiate TVM FFI entry points via macros + @tvm_ffi annotations
// =========================================================================

// ---- Float32 ----
// @tvm_ffi csrmv_yw2y_nt_row_thread_f32
FFI_YW2Y_NT_ROW_THREAD(_f32,  float)
// @tvm_ffi csrmv_yw2y_nt_row_warp_f32
FFI_YW2Y_NT_ROW_WARP(_f32,   float)
// @tvm_ffi csrmv_yw2y_nt_nz_thread_f32
FFI_YW2Y_NT_NZ_THREAD(_f32,  float)
// @tvm_ffi csrmv_yw2y_nt_auto_f32
FFI_YW2Y_NT_AUTO(_f32,       float)
// @tvm_ffi csrmv_yw2y_t_nz_thread_f32
FFI_YW2Y_T_NZ_THREAD(_f32,   float)

// ---- Float64 ----
// @tvm_ffi csrmv_yw2y_nt_auto_f64
FFI_YW2Y_NT_AUTO(_f64,       double)
// @tvm_ffi csrmv_yw2y_t_nz_thread_f64
FFI_YW2Y_T_NZ_THREAD(_f64,   double)

// ---- Float16 (accumulates in float32) ----
// @tvm_ffi csrmv_yw2y_nt_auto_f16
FFI_YW2Y_NT_AUTO(_f16,       __half)
// @tvm_ffi csrmv_yw2y_t_nz_thread_f16
FFI_YW2Y_T_NZ_THREAD(_f16,   __half)

// ---- BFloat16 (accumulates in float32) ----
// @tvm_ffi csrmv_yw2y_nt_auto_bf16
FFI_YW2Y_NT_AUTO(_bf16,      __nv_bfloat16)
// @tvm_ffi csrmv_yw2y_t_nz_thread_bf16
FFI_YW2Y_T_NZ_THREAD(_bf16,  __nv_bfloat16)
