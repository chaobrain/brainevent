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
 * coomm.cu -- Float-Weighted COO Sparse Matrix-Matrix CUDA Kernels
 * ================================================================
 *
 * Python API: brainevent.coomm(data, row, col, B, *, shape, transpose, backend)
 *
 * Standard (non-event-driven) COO sparse matrix-matrix product where the
 * dense matrix ``B`` contains continuous floating-point values.  Every NNZ
 * entry contributes to the output; there is no activity predicate.
 *
 * Parameters
 * ----------
 * data   : 1-D float tensor of shape [1] (homogeneous) or [nnz] (heterogeneous).
 * row    : 1-D int32 tensor of shape [nnz] -- row indices of the sparse matrix.
 * col    : 1-D int32 tensor of shape [nnz] -- column indices of the sparse matrix.
 * B      : 2-D float tensor of shape [k, n] (NT) or [m, n] (T).
 * output : 2-D float tensor of shape [m, n] (NT) or [k, n] (T).
 *
 * Semantics
 * ---------
 * NT mode (transpose=False):
 *   out[row[s], j] += data[s] * B[col[s], j]
 *   Sparse A: [m, k].  B: [k, n].  Output: [m, n].
 *
 * T mode (transpose=True):
 *   out[col[s], j] += data[s] * B[row[s], j]
 *   Sparse A: [m, k].  B: [m, n].  Output: [k, n].
 *
 * Kernel Variants
 * ---------------
 * Two complementary kernel families cover the parameter space:
 *
 * Variant 1 — Column-Tiled (CT):
 *   Grid: (ceil(nnz/BLOCK_K), ceil(n/BLOCK_N))    Block: (BLOCK_N=32)
 *   One warp per thread block.  Serializes over BLOCK_K=32 NNZ entries;
 *   all 32 threads handle BLOCK_N=32 output columns simultaneously.
 *   Coalesced reads of B[src, n_tile].
 *   Best for: large nnz, small/moderate n (n ≤ 64).
 *
 * Variant 2 — Warp-Per-Entry (WPE):
 *   Grid: (ceil(nnz/WARPS_PER_BLOCK), ceil(n/32))   Block: (256=8 warps)
 *   Each warp handles a single NNZ entry and 32 consecutive output columns.
 *   No serial loop over NNZ: maximum parallelism.
 *   Best for: large n (n > 64), moderate or large nnz.
 *
 * Homogeneous vs. Heterogeneous weights:
 *   Detected at runtime via data.size(0)==1.  Warp-uniform branch: no divergence.
 *
 * Dtype support:
 *   B and weights: float32, float64, float16 (sm_70+), bfloat16 (sm_80+).
 *   f16 and bf16 accumulate in float32; final atomic write back to f16/bf16.
 *
 * Output initialization:
 *   The output buffer is zeroed via cudaMemsetAsync before the kernel launches.
 *
 * Index safety:
 *   All strides involving large dimensions use int64_t to avoid int32 overflow.
 *
 * IMPORTANT: data_ptr() returns GPU device pointers.  NEVER dereference on the host.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// ============================================================================
// Per-dtype conversion macros: READ converts WEIGHT_T -> ACC_T
// ============================================================================

#define READ_F32(x)   (x)
#define READ_F64(x)   (x)
#define READ_F16(x)   __half2float(x)
#define READ_BF16(x)  __bfloat162float(x)

// ============================================================================
// Per-dtype atomic-add helpers (ACC_T value -> WEIGHT_T memory)
//
// f32 / f64 : native atomicAdd, universally available.
// f16       : native on sm_70+; CAS-based fallback for older arches.
// bf16      : native on sm_80+; CAS-based fallback.
// ============================================================================

__device__ __inline__ void atomic_add_f32(float* addr, float val) {
    atomicAdd(addr, val);
}

__device__ __inline__ void atomic_add_f64(double* addr, double val) {
    atomicAdd(addr, val);
}

__device__ __inline__ void atomic_add_f16(__half* addr, float val) {
#if __CUDA_ARCH__ >= 700
    atomicAdd(addr, __float2half(val));
#else
    unsigned int* base = reinterpret_cast<unsigned int*>(
        reinterpret_cast<size_t>(addr) & ~(size_t)2
    );
    int shift = ((reinterpret_cast<size_t>(addr) & 2) != 0) ? 16 : 0;
    unsigned int assumed, old_val = *base, updated;
    do {
        assumed = old_val;
        unsigned short h = static_cast<unsigned short>((assumed >> shift) & 0xFFFF);
        float cur = __half2float(*reinterpret_cast<__half*>(&h));
        __half new_h = __float2half(cur + val);
        unsigned short new_us = *reinterpret_cast<unsigned short*>(&new_h);
        updated = (assumed & ~(0xFFFFu << shift)) | (static_cast<unsigned int>(new_us) << shift);
        old_val = atomicCAS(base, assumed, updated);
    } while (assumed != old_val);
#endif
}

__device__ __inline__ void atomic_add_bf16(__nv_bfloat16* addr, float val) {
#if __CUDA_ARCH__ >= 800
    atomicAdd(addr, __float2bfloat16(val));
#else
    unsigned int* base = reinterpret_cast<unsigned int*>(
        reinterpret_cast<size_t>(addr) & ~(size_t)2
    );
    int shift = ((reinterpret_cast<size_t>(addr) & 2) != 0) ? 16 : 0;
    unsigned int assumed, old_val = *base, updated;
    do {
        assumed = old_val;
        unsigned short h = static_cast<unsigned short>((assumed >> shift) & 0xFFFF);
        float cur = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&h));
        __nv_bfloat16 new_h = __float2bfloat16(cur + val);
        unsigned short new_us = *reinterpret_cast<unsigned short*>(&new_h);
        updated = (assumed & ~(0xFFFFu << shift)) | (static_cast<unsigned int>(new_us) << shift);
        old_val = atomicCAS(base, assumed, updated);
    } while (assumed != old_val);
#endif
}

// ============================================================================
// Tiling constants (match binary_coomm.cu for consistency)
// ============================================================================

#define COOMM_CT_BLOCK_K   32   // NNZ entries serialized per CT block
#define COOMM_CT_BLOCK_N   32   // Output columns per CT block (= warp width)
#define COOMM_WPE_WARPS    8    // Warps per block in WPE variant (8×32 = 256 threads)
#define COOMM_WPE_COLS     32   // Output columns per warp in WPE variant

// ============================================================================
// Variant 1: Column-Tiled (CT) — NT direction
//
// out[row[s], j] += data[s] * B[col[s], j]
//
// Grid:  (ceil(nnz / BLOCK_K), ceil(n / BLOCK_N))
// Block: (BLOCK_N=32,)  — one warp
//
// Thread t covers output column (blockIdx.y * BLOCK_N + t).
// Each block serializes over up to BLOCK_K NNZ entries.
// For each NNZ entry:
//   - All 32 threads coalesce-read 32 consecutive B[src, *] values.
//   - Each active thread atomicAdds weight * B[src, my_col] into out[dst, my_col].
//
// The weight read is warp-uniform (same value for all 32 threads, cached in L1).
// ============================================================================

#define DEFINE_COOMM_CT_NT(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)             \
__global__ void _coomm_ct_nt_kern##SUFFIX(                                             \
    const WEIGHT_T* __restrict__ data,                                                 \
    const int32_t*  __restrict__ row,                                                  \
    const int32_t*  __restrict__ col,                                                  \
    const WEIGHT_T* __restrict__ B,    /* [k, n] row-major */                         \
    WEIGHT_T*                    out,  /* [m, n] row-major */                         \
    int nnz, int n, int is_homo                                                        \
) {                                                                                    \
    int nnz_start = blockIdx.x * COOMM_CT_BLOCK_K;                                    \
    int col_start = blockIdx.y * COOMM_CT_BLOCK_N;                                    \
    int t         = threadIdx.x;                                                       \
    int my_col    = col_start + t;                                                     \
    bool col_valid = (my_col < n);                                                     \
    /* Pre-load homo weight (warp-uniform, hits L1) */                                 \
    ACC_T homo_w = READ_W(data[0]);                                                    \
    int nnz_end = nnz_start + COOMM_CT_BLOCK_K;                                       \
    if (nnz_end > nnz) nnz_end = nnz;                                                 \
    for (int s = nnz_start; s < nnz_end; s++) {                                       \
        int src = col[s];   /* source row in B (NT mode) */                            \
        int dst = row[s];   /* dest   row in out */                                    \
        if (!col_valid) continue;                                                      \
        /* Coalesced read: 32 threads load 32 consecutive B columns */                 \
        ACC_T b_val = READ_W(B[(int64_t)src * n + my_col]);                           \
        ACC_T w = is_homo ? homo_w : READ_W(data[s]);                                  \
        ATOMIC_ADD_W(out + (int64_t)dst * n + my_col, w * b_val);                     \
    }                                                                                  \
}

// ============================================================================
// Variant 1: Column-Tiled (CT) — T direction
//
// out[col[s], j] += data[s] * B[row[s], j]
// ============================================================================

#define DEFINE_COOMM_CT_T(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)              \
__global__ void _coomm_ct_t_kern##SUFFIX(                                              \
    const WEIGHT_T* __restrict__ data,                                                 \
    const int32_t*  __restrict__ row,                                                  \
    const int32_t*  __restrict__ col,                                                  \
    const WEIGHT_T* __restrict__ B,    /* [m, n] row-major */                         \
    WEIGHT_T*                    out,  /* [k, n] row-major */                         \
    int nnz, int n, int is_homo                                                        \
) {                                                                                    \
    int nnz_start = blockIdx.x * COOMM_CT_BLOCK_K;                                    \
    int col_start = blockIdx.y * COOMM_CT_BLOCK_N;                                    \
    int t         = threadIdx.x;                                                       \
    int my_col    = col_start + t;                                                     \
    bool col_valid = (my_col < n);                                                     \
    ACC_T homo_w = READ_W(data[0]);                                                    \
    int nnz_end = nnz_start + COOMM_CT_BLOCK_K;                                       \
    if (nnz_end > nnz) nnz_end = nnz;                                                 \
    for (int s = nnz_start; s < nnz_end; s++) {                                       \
        int src = row[s];   /* source row in B (T mode) */                             \
        int dst = col[s];   /* dest   row in out (T mode) */                           \
        if (!col_valid) continue;                                                      \
        ACC_T b_val = READ_W(B[(int64_t)src * n + my_col]);                           \
        ACC_T w = is_homo ? homo_w : READ_W(data[s]);                                  \
        ATOMIC_ADD_W(out + (int64_t)dst * n + my_col, w * b_val);                     \
    }                                                                                  \
}

// ============================================================================
// Variant 2: Warp-Per-Entry (WPE) — NT direction
//
// out[row[s], j] += data[s] * B[col[s], j]
//
// Grid:  (ceil(nnz / WARPS_PER_BLOCK), ceil(n / COLS_PER_WARP))
// Block: (WARPS_PER_BLOCK * 32 = 256,)  — 8 warps
//
// Each warp handles exactly one NNZ entry and 32 consecutive output columns.
// All 32 lanes coalesce-read B[src, n_tile] and write out[dst, n_tile] atomically.
// No serial loop: maximum GPU utilisation for large nnz and large n.
// ============================================================================

#define DEFINE_COOMM_WPE_NT(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)            \
__global__ void _coomm_wpe_nt_kern##SUFFIX(                                            \
    const WEIGHT_T* __restrict__ data,                                                 \
    const int32_t*  __restrict__ row,                                                  \
    const int32_t*  __restrict__ col,                                                  \
    const WEIGHT_T* __restrict__ B,    /* [k, n] row-major */                         \
    WEIGHT_T*                    out,  /* [m, n] row-major */                         \
    int nnz, int n, int is_homo                                                        \
) {                                                                                    \
    int warp_id   = (int)(blockIdx.x * COOMM_WPE_WARPS) + (int)(threadIdx.x >> 5);   \
    int lane      = threadIdx.x & 31;                                                  \
    int col_start = blockIdx.y * COOMM_WPE_COLS;                                      \
    int my_col    = col_start + lane;                                                  \
    if (warp_id >= nnz) return;                                                        \
    bool col_valid = (my_col < n);                                                     \
    int s   = warp_id;                                                                 \
    int src = col[s];   /* source row in B */                                          \
    int dst = row[s];   /* dest   row in out */                                        \
    if (!col_valid) return;                                                            \
    /* Coalesced read: 32 consecutive B columns per warp */                            \
    ACC_T b_val = READ_W(B[(int64_t)src * n + my_col]);                               \
    ACC_T w = is_homo ? READ_W(data[0]) : READ_W(data[s]);                             \
    /* Coalesced write: 32 consecutive out columns per warp */                         \
    ATOMIC_ADD_W(out + (int64_t)dst * n + my_col, w * b_val);                         \
}

// ============================================================================
// Variant 2: Warp-Per-Entry (WPE) — T direction
//
// out[col[s], j] += data[s] * B[row[s], j]
// ============================================================================

#define DEFINE_COOMM_WPE_T(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)             \
__global__ void _coomm_wpe_t_kern##SUFFIX(                                             \
    const WEIGHT_T* __restrict__ data,                                                 \
    const int32_t*  __restrict__ row,                                                  \
    const int32_t*  __restrict__ col,                                                  \
    const WEIGHT_T* __restrict__ B,    /* [m, n] row-major */                         \
    WEIGHT_T*                    out,  /* [k, n] row-major */                         \
    int nnz, int n, int is_homo                                                        \
) {                                                                                    \
    int warp_id   = (int)(blockIdx.x * COOMM_WPE_WARPS) + (int)(threadIdx.x >> 5);   \
    int lane      = threadIdx.x & 31;                                                  \
    int col_start = blockIdx.y * COOMM_WPE_COLS;                                      \
    int my_col    = col_start + lane;                                                  \
    if (warp_id >= nnz) return;                                                        \
    bool col_valid = (my_col < n);                                                     \
    int s   = warp_id;                                                                 \
    int src = row[s];   /* source row in B (T mode) */                                 \
    int dst = col[s];   /* dest   row in out (T mode) */                               \
    if (!col_valid) return;                                                            \
    ACC_T b_val = READ_W(B[(int64_t)src * n + my_col]);                               \
    ACC_T w = is_homo ? READ_W(data[0]) : READ_W(data[s]);                             \
    ATOMIC_ADD_W(out + (int64_t)dst * n + my_col, w * b_val);                         \
}

// ============================================================================
// Kernel instantiations: 4 dtypes × 4 variants = 16 kernels
// ============================================================================

// ---- Float32 ----
DEFINE_COOMM_CT_NT (_f32, float,          float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_CT_T  (_f32, float,          float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_WPE_NT(_f32, float,          float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_WPE_T (_f32, float,          float,  READ_F32,  atomic_add_f32)

// ---- Float64 ----
DEFINE_COOMM_CT_NT (_f64, double,         double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_CT_T  (_f64, double,         double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_WPE_NT(_f64, double,         double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_WPE_T (_f64, double,         double, READ_F64,  atomic_add_f64)

// ---- Float16 (accumulate in f32 for numerical stability) ----
DEFINE_COOMM_CT_NT (_f16, __half,          float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_CT_T  (_f16, __half,          float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_WPE_NT(_f16, __half,          float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_WPE_T (_f16, __half,          float,  READ_F16,  atomic_add_f16)

// ---- BFloat16 (accumulate in f32 for numerical stability) ----
DEFINE_COOMM_CT_NT (_bf16, __nv_bfloat16,  float,  READ_BF16, atomic_add_bf16)
DEFINE_COOMM_CT_T  (_bf16, __nv_bfloat16,  float,  READ_BF16, atomic_add_bf16)
DEFINE_COOMM_WPE_NT(_bf16, __nv_bfloat16,  float,  READ_BF16, atomic_add_bf16)
DEFINE_COOMM_WPE_T (_bf16, __nv_bfloat16,  float,  READ_BF16, atomic_add_bf16)


// ============================================================================
// TVM FFI Entry Point Macros
// ============================================================================
//
// Convention: args = (data, row_idx, col_idx, B, output, stream)
//   data    : [1] (homo) or [nnz] (hetero) -- weight values
//   row_idx : [nnz], int32 -- row indices
//   col_idx : [nnz], int32 -- column indices
//   B       : [k, n] (NT) or [m, n] (T)
//   output  : [m, n] (NT) or [k, n] (T) -- zero-initialized here
//   stream  : CUDA stream handle (int64)
//
// IMPORTANT: data_ptr() returns GPU device pointers. Never dereference on host.
//
// CT dispatch:  block=(32,), grid=(ceil(nnz/32), ceil(n/32)).
// WPE dispatch: block=(256,), grid=(ceil(nnz/8), ceil(n/32)).
// ============================================================================

// ---- FFI macro: CT-NT ----
#define FFI_COOMM_CT_NT(SUFFIX, WEIGHT_C_T, OUT_BYTES_PER_ELEM)                       \
void coomm_ct_nt##SUFFIX(                                                              \
    tvm::ffi::TensorView data,                                                         \
    tvm::ffi::TensorView row_idx,                                                      \
    tvm::ffi::TensorView col_idx,                                                      \
    tvm::ffi::TensorView B,                                                            \
    tvm::ffi::TensorView output,                                                       \
    int64_t stream                                                                     \
) {                                                                                    \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                          \
    int nnz  = static_cast<int>(row_idx.size(0));                                     \
    int n    = static_cast<int>(B.size(1));                                           \
    int m    = static_cast<int>(output.size(0));                                      \
    int is_homo = (data.size(0) == 1) ? 1 : 0;                                       \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                 \
    cudaMemsetAsync(d_out, 0, (size_t)m * n * OUT_BYTES_PER_ELEM, s);                \
    if (nnz == 0) return;                                                              \
    dim3 block(COOMM_CT_BLOCK_N, 1, 1);                                               \
    dim3 grid(                                                                         \
        (nnz + COOMM_CT_BLOCK_K - 1) / COOMM_CT_BLOCK_K,                             \
        (n   + COOMM_CT_BLOCK_N - 1) / COOMM_CT_BLOCK_N,                             \
        1                                                                              \
    );                                                                                 \
    _coomm_ct_nt_kern##SUFFIX<<<grid, block, 0, s>>>(                                 \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                              \
        static_cast<const int32_t*>(row_idx.data_ptr()),                              \
        static_cast<const int32_t*>(col_idx.data_ptr()),                              \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                                 \
        d_out, nnz, n, is_homo                                                        \
    );                                                                                 \
}

// ---- FFI macro: CT-T ----
#define FFI_COOMM_CT_T(SUFFIX, WEIGHT_C_T, OUT_BYTES_PER_ELEM)                        \
void coomm_ct_t##SUFFIX(                                                               \
    tvm::ffi::TensorView data,                                                         \
    tvm::ffi::TensorView row_idx,                                                      \
    tvm::ffi::TensorView col_idx,                                                      \
    tvm::ffi::TensorView B,                                                            \
    tvm::ffi::TensorView output,                                                       \
    int64_t stream                                                                     \
) {                                                                                    \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                          \
    int nnz   = static_cast<int>(row_idx.size(0));                                    \
    int n     = static_cast<int>(B.size(1));                                          \
    int k_out = static_cast<int>(output.size(0));                                     \
    int is_homo = (data.size(0) == 1) ? 1 : 0;                                       \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                 \
    cudaMemsetAsync(d_out, 0, (size_t)k_out * n * OUT_BYTES_PER_ELEM, s);            \
    if (nnz == 0) return;                                                              \
    dim3 block(COOMM_CT_BLOCK_N, 1, 1);                                               \
    dim3 grid(                                                                         \
        (nnz + COOMM_CT_BLOCK_K - 1) / COOMM_CT_BLOCK_K,                             \
        (n   + COOMM_CT_BLOCK_N - 1) / COOMM_CT_BLOCK_N,                             \
        1                                                                              \
    );                                                                                 \
    _coomm_ct_t_kern##SUFFIX<<<grid, block, 0, s>>>(                                  \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                              \
        static_cast<const int32_t*>(row_idx.data_ptr()),                              \
        static_cast<const int32_t*>(col_idx.data_ptr()),                              \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                                 \
        d_out, nnz, n, is_homo                                                        \
    );                                                                                 \
}

// ---- FFI macro: WPE-NT ----
#define FFI_COOMM_WPE_NT(SUFFIX, WEIGHT_C_T, OUT_BYTES_PER_ELEM)                      \
void coomm_wpe_nt##SUFFIX(                                                             \
    tvm::ffi::TensorView data,                                                         \
    tvm::ffi::TensorView row_idx,                                                      \
    tvm::ffi::TensorView col_idx,                                                      \
    tvm::ffi::TensorView B,                                                            \
    tvm::ffi::TensorView output,                                                       \
    int64_t stream                                                                     \
) {                                                                                    \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                          \
    int nnz   = static_cast<int>(row_idx.size(0));                                    \
    int n     = static_cast<int>(B.size(1));                                          \
    int m     = static_cast<int>(output.size(0));                                     \
    int is_homo = (data.size(0) == 1) ? 1 : 0;                                       \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                 \
    cudaMemsetAsync(d_out, 0, (size_t)m * n * OUT_BYTES_PER_ELEM, s);                \
    if (nnz == 0) return;                                                              \
    dim3 block(COOMM_WPE_WARPS * 32, 1, 1);    /* 256 threads = 8 warps */           \
    dim3 grid(                                                                         \
        (nnz + COOMM_WPE_WARPS - 1) / COOMM_WPE_WARPS,                               \
        (n   + COOMM_WPE_COLS  - 1) / COOMM_WPE_COLS,                                \
        1                                                                              \
    );                                                                                 \
    _coomm_wpe_nt_kern##SUFFIX<<<grid, block, 0, s>>>(                                \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                              \
        static_cast<const int32_t*>(row_idx.data_ptr()),                              \
        static_cast<const int32_t*>(col_idx.data_ptr()),                              \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                                 \
        d_out, nnz, n, is_homo                                                        \
    );                                                                                 \
}

// ---- FFI macro: WPE-T ----
#define FFI_COOMM_WPE_T(SUFFIX, WEIGHT_C_T, OUT_BYTES_PER_ELEM)                       \
void coomm_wpe_t##SUFFIX(                                                              \
    tvm::ffi::TensorView data,                                                         \
    tvm::ffi::TensorView row_idx,                                                      \
    tvm::ffi::TensorView col_idx,                                                      \
    tvm::ffi::TensorView B,                                                            \
    tvm::ffi::TensorView output,                                                       \
    int64_t stream                                                                     \
) {                                                                                    \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                          \
    int nnz   = static_cast<int>(row_idx.size(0));                                    \
    int n     = static_cast<int>(B.size(1));                                          \
    int k_out = static_cast<int>(output.size(0));                                     \
    int is_homo = (data.size(0) == 1) ? 1 : 0;                                       \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                 \
    cudaMemsetAsync(d_out, 0, (size_t)k_out * n * OUT_BYTES_PER_ELEM, s);            \
    if (nnz == 0) return;                                                              \
    dim3 block(COOMM_WPE_WARPS * 32, 1, 1);                                           \
    dim3 grid(                                                                         \
        (nnz + COOMM_WPE_WARPS - 1) / COOMM_WPE_WARPS,                               \
        (n   + COOMM_WPE_COLS  - 1) / COOMM_WPE_COLS,                                \
        1                                                                              \
    );                                                                                 \
    _coomm_wpe_t_kern##SUFFIX<<<grid, block, 0, s>>>(                                 \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                              \
        static_cast<const int32_t*>(row_idx.data_ptr()),                              \
        static_cast<const int32_t*>(col_idx.data_ptr()),                              \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                                 \
        d_out, nnz, n, is_homo                                                        \
    );                                                                                 \
}

// ============================================================================
// Instantiate TVM FFI entry points
// ============================================================================
// Naming convention:
//   coomm_{variant}_{direction}_{weight_dtype}
//   variant:   ct (column-tiled) | wpe (warp-per-entry)
//   direction: nt (transpose=False) | t (transpose=True)
//   dtype:     f32 | f64 | f16 | bf16
// ============================================================================

// ============================================================================
// CT-NT: Column-Tiled, Non-Transpose
// ============================================================================

// @tvm_ffi coomm_ct_nt_f32
FFI_COOMM_CT_NT(_f32, float,          sizeof(float))
// @tvm_ffi coomm_ct_nt_f64
FFI_COOMM_CT_NT(_f64, double,         sizeof(double))
// @tvm_ffi coomm_ct_nt_f16
FFI_COOMM_CT_NT(_f16, __half,         sizeof(__half))
// @tvm_ffi coomm_ct_nt_bf16
FFI_COOMM_CT_NT(_bf16, __nv_bfloat16, sizeof(__nv_bfloat16))

// ============================================================================
// CT-T: Column-Tiled, Transpose
// ============================================================================

// @tvm_ffi coomm_ct_t_f32
FFI_COOMM_CT_T(_f32, float,          sizeof(float))
// @tvm_ffi coomm_ct_t_f64
FFI_COOMM_CT_T(_f64, double,         sizeof(double))
// @tvm_ffi coomm_ct_t_f16
FFI_COOMM_CT_T(_f16, __half,         sizeof(__half))
// @tvm_ffi coomm_ct_t_bf16
FFI_COOMM_CT_T(_bf16, __nv_bfloat16, sizeof(__nv_bfloat16))

// ============================================================================
// WPE-NT: Warp-Per-Entry, Non-Transpose
// ============================================================================

// @tvm_ffi coomm_wpe_nt_f32
FFI_COOMM_WPE_NT(_f32, float,          sizeof(float))
// @tvm_ffi coomm_wpe_nt_f64
FFI_COOMM_WPE_NT(_f64, double,         sizeof(double))
// @tvm_ffi coomm_wpe_nt_f16
FFI_COOMM_WPE_NT(_f16, __half,         sizeof(__half))
// @tvm_ffi coomm_wpe_nt_bf16
FFI_COOMM_WPE_NT(_bf16, __nv_bfloat16, sizeof(__nv_bfloat16))

// ============================================================================
// WPE-T: Warp-Per-Entry, Transpose
// ============================================================================

// @tvm_ffi coomm_wpe_t_f32
FFI_COOMM_WPE_T(_f32, float,          sizeof(float))
// @tvm_ffi coomm_wpe_t_f64
FFI_COOMM_WPE_T(_f64, double,         sizeof(double))
// @tvm_ffi coomm_wpe_t_f16
FFI_COOMM_WPE_T(_f16, __half,         sizeof(__half))
// @tvm_ffi coomm_wpe_t_bf16
FFI_COOMM_WPE_T(_bf16, __nv_bfloat16, sizeof(__nv_bfloat16))
