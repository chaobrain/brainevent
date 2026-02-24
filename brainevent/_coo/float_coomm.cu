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
 * float_coomm.cu -- Float-Weighted COO Sparse Matrix-Matrix CUDA Kernels
 * =======================================================================
 *
 * This module provides high-performance CUDA kernels for standard (non-event-driven)
 * sparse matrix-matrix (SpMM) multiplications where the sparse matrix is in
 * Coordinate (COO) format.
 *
 * Supported Operations:
 * --------------------
 * coomm (SpMM): out = A @ B  or  out = A.T @ B
 *   - Column-Tiled (CT) Variant: Optimized for small number of columns (n <= 64).
 *     Block=(32 threads), Grid=(ceil(nnz/32), ceil(n/32)).
 *   - Warp-Per-Entry (WPE) Variant: Optimized for large number of columns (n > 64).
 *     Block=(256 threads, 8 warps), Grid=(ceil(nnz/8), ceil(n/32)).
 *
 * Data Types:
 * ----------
 * - float32, float64, float16 (sm_70+), bfloat16 (sm_80+)
 * - For f16/bf16, accumulation is performed in float32 for numerical stability.
 *
 * TVM FFI Entry Points:
 * --------------------
 * coomm_homo_ct_nt_{f32,f64,f16,bf16}   -- CT kernel, non-transposed SpMM (homo weights)
 * coomm_homo_ct_t_{f32,f64,f16,bf16}    -- CT kernel, transposed SpMM (homo weights)
 * coomm_homo_wpe_nt_{f32,f64,f16,bf16}  -- WPE kernel, non-transposed SpMM (homo weights)
 * coomm_homo_wpe_t_{f32,f64,f16,bf16}   -- WPE kernel, transposed SpMM (homo weights)
 * coomm_hetero_ct_nt_{f32,f64,f16,bf16} -- CT kernel, non-transposed SpMM (hetero weights)
 * coomm_hetero_ct_t_{f32,f64,f16,bf16}  -- CT kernel, transposed SpMM (hetero weights)
 * coomm_hetero_wpe_nt_{f32,f64,f16,bf16}-- WPE kernel, non-transposed SpMM (hetero weights)
 * coomm_hetero_wpe_t_{f32,f64,f16,bf16} -- WPE kernel, transposed SpMM (hetero weights)
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
// COO Matrix-Matrix Multiplication (coomm) -- block/warp constants
// ============================================================================

#define COOMM_CT_BLOCK_K   32
#define COOMM_CT_BLOCK_N   32
#define COOMM_WPE_WARPS    8
#define COOMM_WPE_COLS     32

// ============================================================================
// Homo CT kernels (scalar weight broadcast to all connections)
// ============================================================================

#define DEFINE_COOMM_HOMO_CT_NT(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)       \
__global__ void _coomm_homo_ct_nt_kern##SUFFIX(                                       \
    const WEIGHT_T* __restrict__ data,                                                \
    const int32_t*  __restrict__ row,                                                 \
    const int32_t*  __restrict__ col,                                                 \
    const WEIGHT_T* __restrict__ B,                                                   \
    WEIGHT_T*                    out,                                                 \
    int nnz, int n                                                                    \
) {                                                                                   \
    int nnz_start  = blockIdx.x * COOMM_CT_BLOCK_K;                                  \
    int col_start  = blockIdx.y * COOMM_CT_BLOCK_N;                                  \
    int t          = threadIdx.x;                                                     \
    int my_col     = col_start + t;                                                   \
    bool col_valid = (my_col < n);                                                    \
    ACC_T homo_w   = READ_W(data[0]);                                                 \
    int nnz_end = nnz_start + COOMM_CT_BLOCK_K;                                      \
    if (nnz_end > nnz) nnz_end = nnz;                                                \
    for (int s = nnz_start; s < nnz_end; s++) {                                      \
        int src = col[s];                                                             \
        int dst = row[s];                                                             \
        if (!col_valid) continue;                                                     \
        ACC_T b_val = READ_W(B[(int64_t)src * n + my_col]);                          \
        ATOMIC_ADD_W(out + (int64_t)dst * n + my_col, homo_w * b_val);              \
    }                                                                                 \
}

#define DEFINE_COOMM_HOMO_CT_T(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)        \
__global__ void _coomm_homo_ct_t_kern##SUFFIX(                                        \
    const WEIGHT_T* __restrict__ data,                                                \
    const int32_t*  __restrict__ row,                                                 \
    const int32_t*  __restrict__ col,                                                 \
    const WEIGHT_T* __restrict__ B,                                                   \
    WEIGHT_T*                    out,                                                 \
    int nnz, int n                                                                    \
) {                                                                                   \
    int nnz_start  = blockIdx.x * COOMM_CT_BLOCK_K;                                  \
    int col_start  = blockIdx.y * COOMM_CT_BLOCK_N;                                  \
    int t          = threadIdx.x;                                                     \
    int my_col     = col_start + t;                                                   \
    bool col_valid = (my_col < n);                                                    \
    ACC_T homo_w   = READ_W(data[0]);                                                 \
    int nnz_end = nnz_start + COOMM_CT_BLOCK_K;                                      \
    if (nnz_end > nnz) nnz_end = nnz;                                                \
    for (int s = nnz_start; s < nnz_end; s++) {                                      \
        int src = row[s];                                                             \
        int dst = col[s];                                                             \
        if (!col_valid) continue;                                                     \
        ACC_T b_val = READ_W(B[(int64_t)src * n + my_col]);                          \
        ATOMIC_ADD_W(out + (int64_t)dst * n + my_col, homo_w * b_val);              \
    }                                                                                 \
}

// ============================================================================
// Hetero CT kernels (per-connection weight array)
// ============================================================================

#define DEFINE_COOMM_HETERO_CT_NT(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)     \
__global__ void _coomm_hetero_ct_nt_kern##SUFFIX(                                     \
    const WEIGHT_T* __restrict__ data,                                                \
    const int32_t*  __restrict__ row,                                                 \
    const int32_t*  __restrict__ col,                                                 \
    const WEIGHT_T* __restrict__ B,                                                   \
    WEIGHT_T*                    out,                                                 \
    int nnz, int n                                                                    \
) {                                                                                   \
    int nnz_start  = blockIdx.x * COOMM_CT_BLOCK_K;                                  \
    int col_start  = blockIdx.y * COOMM_CT_BLOCK_N;                                  \
    int t          = threadIdx.x;                                                     \
    int my_col     = col_start + t;                                                   \
    bool col_valid = (my_col < n);                                                    \
    int nnz_end = nnz_start + COOMM_CT_BLOCK_K;                                      \
    if (nnz_end > nnz) nnz_end = nnz;                                                \
    for (int s = nnz_start; s < nnz_end; s++) {                                      \
        int src = col[s];                                                             \
        int dst = row[s];                                                             \
        if (!col_valid) continue;                                                     \
        ACC_T b_val = READ_W(B[(int64_t)src * n + my_col]);                          \
        ACC_T w     = READ_W(data[s]);                                                \
        ATOMIC_ADD_W(out + (int64_t)dst * n + my_col, w * b_val);                   \
    }                                                                                 \
}

#define DEFINE_COOMM_HETERO_CT_T(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)      \
__global__ void _coomm_hetero_ct_t_kern##SUFFIX(                                      \
    const WEIGHT_T* __restrict__ data,                                                \
    const int32_t*  __restrict__ row,                                                 \
    const int32_t*  __restrict__ col,                                                 \
    const WEIGHT_T* __restrict__ B,                                                   \
    WEIGHT_T*                    out,                                                 \
    int nnz, int n                                                                    \
) {                                                                                   \
    int nnz_start  = blockIdx.x * COOMM_CT_BLOCK_K;                                  \
    int col_start  = blockIdx.y * COOMM_CT_BLOCK_N;                                  \
    int t          = threadIdx.x;                                                     \
    int my_col     = col_start + t;                                                   \
    bool col_valid = (my_col < n);                                                    \
    int nnz_end = nnz_start + COOMM_CT_BLOCK_K;                                      \
    if (nnz_end > nnz) nnz_end = nnz;                                                \
    for (int s = nnz_start; s < nnz_end; s++) {                                      \
        int src = row[s];                                                             \
        int dst = col[s];                                                             \
        if (!col_valid) continue;                                                     \
        ACC_T b_val = READ_W(B[(int64_t)src * n + my_col]);                          \
        ACC_T w     = READ_W(data[s]);                                                \
        ATOMIC_ADD_W(out + (int64_t)dst * n + my_col, w * b_val);                   \
    }                                                                                 \
}

// ============================================================================
// Homo WPE kernels (scalar weight broadcast to all connections)
// ============================================================================

#define DEFINE_COOMM_HOMO_WPE_NT(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)      \
__global__ void _coomm_homo_wpe_nt_kern##SUFFIX(                                      \
    const WEIGHT_T* __restrict__ data,                                                \
    const int32_t*  __restrict__ row,                                                 \
    const int32_t*  __restrict__ col,                                                 \
    const WEIGHT_T* __restrict__ B,                                                   \
    WEIGHT_T*                    out,                                                 \
    int nnz, int n                                                                    \
) {                                                                                   \
    int warp_id   = (int)(blockIdx.x * COOMM_WPE_WARPS) + (int)(threadIdx.x >> 5);  \
    int lane      = threadIdx.x & 31;                                                 \
    int col_start = blockIdx.y * COOMM_WPE_COLS;                                     \
    int my_col    = col_start + lane;                                                 \
    if (warp_id >= nnz) return;                                                       \
    bool col_valid = (my_col < n);                                                    \
    int s   = warp_id;                                                                \
    int src = col[s];                                                                 \
    int dst = row[s];                                                                 \
    if (!col_valid) return;                                                           \
    ACC_T b_val  = READ_W(B[(int64_t)src * n + my_col]);                             \
    ACC_T homo_w = READ_W(data[0]);                                                   \
    ATOMIC_ADD_W(out + (int64_t)dst * n + my_col, homo_w * b_val);                  \
}

#define DEFINE_COOMM_HOMO_WPE_T(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)       \
__global__ void _coomm_homo_wpe_t_kern##SUFFIX(                                       \
    const WEIGHT_T* __restrict__ data,                                                \
    const int32_t*  __restrict__ row,                                                 \
    const int32_t*  __restrict__ col,                                                 \
    const WEIGHT_T* __restrict__ B,                                                   \
    WEIGHT_T*                    out,                                                 \
    int nnz, int n                                                                    \
) {                                                                                   \
    int warp_id   = (int)(blockIdx.x * COOMM_WPE_WARPS) + (int)(threadIdx.x >> 5);  \
    int lane      = threadIdx.x & 31;                                                 \
    int col_start = blockIdx.y * COOMM_WPE_COLS;                                     \
    int my_col    = col_start + lane;                                                 \
    if (warp_id >= nnz) return;                                                       \
    bool col_valid = (my_col < n);                                                    \
    int s   = warp_id;                                                                \
    int src = row[s];                                                                 \
    int dst = col[s];                                                                 \
    if (!col_valid) return;                                                           \
    ACC_T b_val  = READ_W(B[(int64_t)src * n + my_col]);                             \
    ACC_T homo_w = READ_W(data[0]);                                                   \
    ATOMIC_ADD_W(out + (int64_t)dst * n + my_col, homo_w * b_val);                  \
}

// ============================================================================
// Hetero WPE kernels (per-connection weight array)
// ============================================================================

#define DEFINE_COOMM_HETERO_WPE_NT(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)    \
__global__ void _coomm_hetero_wpe_nt_kern##SUFFIX(                                    \
    const WEIGHT_T* __restrict__ data,                                                \
    const int32_t*  __restrict__ row,                                                 \
    const int32_t*  __restrict__ col,                                                 \
    const WEIGHT_T* __restrict__ B,                                                   \
    WEIGHT_T*                    out,                                                 \
    int nnz, int n                                                                    \
) {                                                                                   \
    int warp_id   = (int)(blockIdx.x * COOMM_WPE_WARPS) + (int)(threadIdx.x >> 5);  \
    int lane      = threadIdx.x & 31;                                                 \
    int col_start = blockIdx.y * COOMM_WPE_COLS;                                     \
    int my_col    = col_start + lane;                                                 \
    if (warp_id >= nnz) return;                                                       \
    bool col_valid = (my_col < n);                                                    \
    int s   = warp_id;                                                                \
    int src = col[s];                                                                 \
    int dst = row[s];                                                                 \
    if (!col_valid) return;                                                           \
    ACC_T b_val = READ_W(B[(int64_t)src * n + my_col]);                              \
    ACC_T w     = READ_W(data[s]);                                                    \
    ATOMIC_ADD_W(out + (int64_t)dst * n + my_col, w * b_val);                       \
}

#define DEFINE_COOMM_HETERO_WPE_T(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)     \
__global__ void _coomm_hetero_wpe_t_kern##SUFFIX(                                     \
    const WEIGHT_T* __restrict__ data,                                                \
    const int32_t*  __restrict__ row,                                                 \
    const int32_t*  __restrict__ col,                                                 \
    const WEIGHT_T* __restrict__ B,                                                   \
    WEIGHT_T*                    out,                                                 \
    int nnz, int n                                                                    \
) {                                                                                   \
    int warp_id   = (int)(blockIdx.x * COOMM_WPE_WARPS) + (int)(threadIdx.x >> 5);  \
    int lane      = threadIdx.x & 31;                                                 \
    int col_start = blockIdx.y * COOMM_WPE_COLS;                                     \
    int my_col    = col_start + lane;                                                 \
    if (warp_id >= nnz) return;                                                       \
    bool col_valid = (my_col < n);                                                    \
    int s   = warp_id;                                                                \
    int src = row[s];                                                                 \
    int dst = col[s];                                                                 \
    if (!col_valid) return;                                                           \
    ACC_T b_val = READ_W(B[(int64_t)src * n + my_col]);                              \
    ACC_T w     = READ_W(data[s]);                                                    \
    ATOMIC_ADD_W(out + (int64_t)dst * n + my_col, w * b_val);                       \
}

// Homo instantiations
DEFINE_COOMM_HOMO_CT_NT (_f32,  float,          float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_HOMO_CT_T  (_f32,  float,          float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_HOMO_WPE_NT(_f32,  float,          float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_HOMO_WPE_T (_f32,  float,          float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_HOMO_CT_NT (_f64,  double,         double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_HOMO_CT_T  (_f64,  double,         double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_HOMO_WPE_NT(_f64,  double,         double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_HOMO_WPE_T (_f64,  double,         double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_HOMO_CT_NT (_f16,  __half,         float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_HOMO_CT_T  (_f16,  __half,         float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_HOMO_WPE_NT(_f16,  __half,         float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_HOMO_WPE_T (_f16,  __half,         float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_HOMO_CT_NT (_bf16, __nv_bfloat16,  float,  READ_BF16, atomic_add_bf16)
DEFINE_COOMM_HOMO_CT_T  (_bf16, __nv_bfloat16,  float,  READ_BF16, atomic_add_bf16)
DEFINE_COOMM_HOMO_WPE_NT(_bf16, __nv_bfloat16,  float,  READ_BF16, atomic_add_bf16)
DEFINE_COOMM_HOMO_WPE_T (_bf16, __nv_bfloat16,  float,  READ_BF16, atomic_add_bf16)

// Hetero instantiations
DEFINE_COOMM_HETERO_CT_NT (_f32,  float,          float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_HETERO_CT_T  (_f32,  float,          float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_HETERO_WPE_NT(_f32,  float,          float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_HETERO_WPE_T (_f32,  float,          float,  READ_F32,  atomic_add_f32)
DEFINE_COOMM_HETERO_CT_NT (_f64,  double,         double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_HETERO_CT_T  (_f64,  double,         double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_HETERO_WPE_NT(_f64,  double,         double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_HETERO_WPE_T (_f64,  double,         double, READ_F64,  atomic_add_f64)
DEFINE_COOMM_HETERO_CT_NT (_f16,  __half,         float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_HETERO_CT_T  (_f16,  __half,         float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_HETERO_WPE_NT(_f16,  __half,         float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_HETERO_WPE_T (_f16,  __half,         float,  READ_F16,  atomic_add_f16)
DEFINE_COOMM_HETERO_CT_NT (_bf16, __nv_bfloat16,  float,  READ_BF16, atomic_add_bf16)
DEFINE_COOMM_HETERO_CT_T  (_bf16, __nv_bfloat16,  float,  READ_BF16, atomic_add_bf16)
DEFINE_COOMM_HETERO_WPE_NT(_bf16, __nv_bfloat16,  float,  READ_BF16, atomic_add_bf16)
DEFINE_COOMM_HETERO_WPE_T (_bf16, __nv_bfloat16,  float,  READ_BF16, atomic_add_bf16)

// ============================================================================
// FFI entry points -- Homo CT
// ============================================================================

#define FFI_COOMM_HOMO_CT_NT(SUFFIX, WEIGHT_C_T, OUT_BYTES_PER_ELEM)          \
void coomm_homo_ct_nt##SUFFIX(                                                 \
    tvm::ffi::TensorView data,                                                 \
    tvm::ffi::TensorView row_idx,                                              \
    tvm::ffi::TensorView col_idx,                                              \
    tvm::ffi::TensorView B,                                                    \
    tvm::ffi::TensorView output,                                               \
    int64_t stream                                                             \
) {                                                                            \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                  \
    int nnz = static_cast<int>(row_idx.size(0));                              \
    int n   = static_cast<int>(B.size(1));                                    \
    int m   = static_cast<int>(output.size(0));                               \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());         \
    cudaMemsetAsync(d_out, 0, (size_t)m * n * OUT_BYTES_PER_ELEM, s);        \
    if (nnz == 0) return;                                                      \
    dim3 block(COOMM_CT_BLOCK_N, 1, 1);                                       \
    dim3 grid(                                                                 \
        (nnz + COOMM_CT_BLOCK_K - 1) / COOMM_CT_BLOCK_K,                     \
        (n   + COOMM_CT_BLOCK_N - 1) / COOMM_CT_BLOCK_N,                     \
        1                                                                      \
    );                                                                         \
    _coomm_homo_ct_nt_kern##SUFFIX<<<grid, block, 0, s>>>(                    \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                      \
        static_cast<const int32_t*>(row_idx.data_ptr()),                      \
        static_cast<const int32_t*>(col_idx.data_ptr()),                      \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                         \
        d_out, nnz, n                                                          \
    );                                                                         \
}

#define FFI_COOMM_HOMO_CT_T(SUFFIX, WEIGHT_C_T, OUT_BYTES_PER_ELEM)           \
void coomm_homo_ct_t##SUFFIX(                                                  \
    tvm::ffi::TensorView data,                                                 \
    tvm::ffi::TensorView row_idx,                                              \
    tvm::ffi::TensorView col_idx,                                              \
    tvm::ffi::TensorView B,                                                    \
    tvm::ffi::TensorView output,                                               \
    int64_t stream                                                             \
) {                                                                            \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                  \
    int nnz   = static_cast<int>(row_idx.size(0));                            \
    int n     = static_cast<int>(B.size(1));                                  \
    int k_out = static_cast<int>(output.size(0));                             \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());         \
    cudaMemsetAsync(d_out, 0, (size_t)k_out * n * OUT_BYTES_PER_ELEM, s);    \
    if (nnz == 0) return;                                                      \
    dim3 block(COOMM_CT_BLOCK_N, 1, 1);                                       \
    dim3 grid(                                                                 \
        (nnz + COOMM_CT_BLOCK_K - 1) / COOMM_CT_BLOCK_K,                     \
        (n   + COOMM_CT_BLOCK_N - 1) / COOMM_CT_BLOCK_N,                     \
        1                                                                      \
    );                                                                         \
    _coomm_homo_ct_t_kern##SUFFIX<<<grid, block, 0, s>>>(                     \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                      \
        static_cast<const int32_t*>(row_idx.data_ptr()),                      \
        static_cast<const int32_t*>(col_idx.data_ptr()),                      \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                         \
        d_out, nnz, n                                                          \
    );                                                                         \
}

// ============================================================================
// FFI entry points -- Homo WPE
// ============================================================================

#define FFI_COOMM_HOMO_WPE_NT(SUFFIX, WEIGHT_C_T, OUT_BYTES_PER_ELEM)         \
void coomm_homo_wpe_nt##SUFFIX(                                                \
    tvm::ffi::TensorView data,                                                 \
    tvm::ffi::TensorView row_idx,                                              \
    tvm::ffi::TensorView col_idx,                                              \
    tvm::ffi::TensorView B,                                                    \
    tvm::ffi::TensorView output,                                               \
    int64_t stream                                                             \
) {                                                                            \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                  \
    int nnz = static_cast<int>(row_idx.size(0));                              \
    int n   = static_cast<int>(B.size(1));                                    \
    int m   = static_cast<int>(output.size(0));                               \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());         \
    cudaMemsetAsync(d_out, 0, (size_t)m * n * OUT_BYTES_PER_ELEM, s);        \
    if (nnz == 0) return;                                                      \
    dim3 block(COOMM_WPE_WARPS * 32, 1, 1);                                   \
    dim3 grid(                                                                 \
        (nnz + COOMM_WPE_WARPS - 1) / COOMM_WPE_WARPS,                       \
        (n   + COOMM_WPE_COLS  - 1) / COOMM_WPE_COLS,                        \
        1                                                                      \
    );                                                                         \
    _coomm_homo_wpe_nt_kern##SUFFIX<<<grid, block, 0, s>>>(                   \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                      \
        static_cast<const int32_t*>(row_idx.data_ptr()),                      \
        static_cast<const int32_t*>(col_idx.data_ptr()),                      \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                         \
        d_out, nnz, n                                                          \
    );                                                                         \
}

#define FFI_COOMM_HOMO_WPE_T(SUFFIX, WEIGHT_C_T, OUT_BYTES_PER_ELEM)          \
void coomm_homo_wpe_t##SUFFIX(                                                 \
    tvm::ffi::TensorView data,                                                 \
    tvm::ffi::TensorView row_idx,                                              \
    tvm::ffi::TensorView col_idx,                                              \
    tvm::ffi::TensorView B,                                                    \
    tvm::ffi::TensorView output,                                               \
    int64_t stream                                                             \
) {                                                                            \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                  \
    int nnz   = static_cast<int>(row_idx.size(0));                            \
    int n     = static_cast<int>(B.size(1));                                  \
    int k_out = static_cast<int>(output.size(0));                             \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());         \
    cudaMemsetAsync(d_out, 0, (size_t)k_out * n * OUT_BYTES_PER_ELEM, s);    \
    if (nnz == 0) return;                                                      \
    dim3 block(COOMM_WPE_WARPS * 32, 1, 1);                                   \
    dim3 grid(                                                                 \
        (nnz + COOMM_WPE_WARPS - 1) / COOMM_WPE_WARPS,                       \
        (n   + COOMM_WPE_COLS  - 1) / COOMM_WPE_COLS,                        \
        1                                                                      \
    );                                                                         \
    _coomm_homo_wpe_t_kern##SUFFIX<<<grid, block, 0, s>>>(                    \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                      \
        static_cast<const int32_t*>(row_idx.data_ptr()),                      \
        static_cast<const int32_t*>(col_idx.data_ptr()),                      \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                         \
        d_out, nnz, n                                                          \
    );                                                                         \
}

// ============================================================================
// FFI entry points -- Hetero CT
// ============================================================================

#define FFI_COOMM_HETERO_CT_NT(SUFFIX, WEIGHT_C_T, OUT_BYTES_PER_ELEM)        \
void coomm_hetero_ct_nt##SUFFIX(                                               \
    tvm::ffi::TensorView data,                                                 \
    tvm::ffi::TensorView row_idx,                                              \
    tvm::ffi::TensorView col_idx,                                              \
    tvm::ffi::TensorView B,                                                    \
    tvm::ffi::TensorView output,                                               \
    int64_t stream                                                             \
) {                                                                            \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                  \
    int nnz = static_cast<int>(row_idx.size(0));                              \
    int n   = static_cast<int>(B.size(1));                                    \
    int m   = static_cast<int>(output.size(0));                               \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());         \
    cudaMemsetAsync(d_out, 0, (size_t)m * n * OUT_BYTES_PER_ELEM, s);        \
    if (nnz == 0) return;                                                      \
    dim3 block(COOMM_CT_BLOCK_N, 1, 1);                                       \
    dim3 grid(                                                                 \
        (nnz + COOMM_CT_BLOCK_K - 1) / COOMM_CT_BLOCK_K,                     \
        (n   + COOMM_CT_BLOCK_N - 1) / COOMM_CT_BLOCK_N,                     \
        1                                                                      \
    );                                                                         \
    _coomm_hetero_ct_nt_kern##SUFFIX<<<grid, block, 0, s>>>(                  \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                      \
        static_cast<const int32_t*>(row_idx.data_ptr()),                      \
        static_cast<const int32_t*>(col_idx.data_ptr()),                      \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                         \
        d_out, nnz, n                                                          \
    );                                                                         \
}

#define FFI_COOMM_HETERO_CT_T(SUFFIX, WEIGHT_C_T, OUT_BYTES_PER_ELEM)         \
void coomm_hetero_ct_t##SUFFIX(                                                \
    tvm::ffi::TensorView data,                                                 \
    tvm::ffi::TensorView row_idx,                                              \
    tvm::ffi::TensorView col_idx,                                              \
    tvm::ffi::TensorView B,                                                    \
    tvm::ffi::TensorView output,                                               \
    int64_t stream                                                             \
) {                                                                            \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                  \
    int nnz   = static_cast<int>(row_idx.size(0));                            \
    int n     = static_cast<int>(B.size(1));                                  \
    int k_out = static_cast<int>(output.size(0));                             \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());         \
    cudaMemsetAsync(d_out, 0, (size_t)k_out * n * OUT_BYTES_PER_ELEM, s);    \
    if (nnz == 0) return;                                                      \
    dim3 block(COOMM_CT_BLOCK_N, 1, 1);                                       \
    dim3 grid(                                                                 \
        (nnz + COOMM_CT_BLOCK_K - 1) / COOMM_CT_BLOCK_K,                     \
        (n   + COOMM_CT_BLOCK_N - 1) / COOMM_CT_BLOCK_N,                     \
        1                                                                      \
    );                                                                         \
    _coomm_hetero_ct_t_kern##SUFFIX<<<grid, block, 0, s>>>(                   \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                      \
        static_cast<const int32_t*>(row_idx.data_ptr()),                      \
        static_cast<const int32_t*>(col_idx.data_ptr()),                      \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                         \
        d_out, nnz, n                                                          \
    );                                                                         \
}

// ============================================================================
// FFI entry points -- Hetero WPE
// ============================================================================

#define FFI_COOMM_HETERO_WPE_NT(SUFFIX, WEIGHT_C_T, OUT_BYTES_PER_ELEM)       \
void coomm_hetero_wpe_nt##SUFFIX(                                              \
    tvm::ffi::TensorView data,                                                 \
    tvm::ffi::TensorView row_idx,                                              \
    tvm::ffi::TensorView col_idx,                                              \
    tvm::ffi::TensorView B,                                                    \
    tvm::ffi::TensorView output,                                               \
    int64_t stream                                                             \
) {                                                                            \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                  \
    int nnz = static_cast<int>(row_idx.size(0));                              \
    int n   = static_cast<int>(B.size(1));                                    \
    int m   = static_cast<int>(output.size(0));                               \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());         \
    cudaMemsetAsync(d_out, 0, (size_t)m * n * OUT_BYTES_PER_ELEM, s);        \
    if (nnz == 0) return;                                                      \
    dim3 block(COOMM_WPE_WARPS * 32, 1, 1);                                   \
    dim3 grid(                                                                 \
        (nnz + COOMM_WPE_WARPS - 1) / COOMM_WPE_WARPS,                       \
        (n   + COOMM_WPE_COLS  - 1) / COOMM_WPE_COLS,                        \
        1                                                                      \
    );                                                                         \
    _coomm_hetero_wpe_nt_kern##SUFFIX<<<grid, block, 0, s>>>(                 \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                      \
        static_cast<const int32_t*>(row_idx.data_ptr()),                      \
        static_cast<const int32_t*>(col_idx.data_ptr()),                      \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                         \
        d_out, nnz, n                                                          \
    );                                                                         \
}

#define FFI_COOMM_HETERO_WPE_T(SUFFIX, WEIGHT_C_T, OUT_BYTES_PER_ELEM)        \
void coomm_hetero_wpe_t##SUFFIX(                                               \
    tvm::ffi::TensorView data,                                                 \
    tvm::ffi::TensorView row_idx,                                              \
    tvm::ffi::TensorView col_idx,                                              \
    tvm::ffi::TensorView B,                                                    \
    tvm::ffi::TensorView output,                                               \
    int64_t stream                                                             \
) {                                                                            \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                  \
    int nnz   = static_cast<int>(row_idx.size(0));                            \
    int n     = static_cast<int>(B.size(1));                                  \
    int k_out = static_cast<int>(output.size(0));                             \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());         \
    cudaMemsetAsync(d_out, 0, (size_t)k_out * n * OUT_BYTES_PER_ELEM, s);    \
    if (nnz == 0) return;                                                      \
    dim3 block(COOMM_WPE_WARPS * 32, 1, 1);                                   \
    dim3 grid(                                                                 \
        (nnz + COOMM_WPE_WARPS - 1) / COOMM_WPE_WARPS,                       \
        (n   + COOMM_WPE_COLS  - 1) / COOMM_WPE_COLS,                        \
        1                                                                      \
    );                                                                         \
    _coomm_hetero_wpe_t_kern##SUFFIX<<<grid, block, 0, s>>>(                  \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                      \
        static_cast<const int32_t*>(row_idx.data_ptr()),                      \
        static_cast<const int32_t*>(col_idx.data_ptr()),                      \
        static_cast<const WEIGHT_C_T*>(B.data_ptr()),                         \
        d_out, nnz, n                                                          \
    );                                                                         \
}

// Homo CT instantiations
// @tvm_ffi coomm_homo_ct_nt_f32
FFI_COOMM_HOMO_CT_NT(_f32,  float,          sizeof(float))
// @tvm_ffi coomm_homo_ct_t_f32
FFI_COOMM_HOMO_CT_T (_f32,  float,          sizeof(float))
// @tvm_ffi coomm_homo_ct_nt_f64
FFI_COOMM_HOMO_CT_NT(_f64,  double,         sizeof(double))
// @tvm_ffi coomm_homo_ct_t_f64
FFI_COOMM_HOMO_CT_T (_f64,  double,         sizeof(double))
// @tvm_ffi coomm_homo_ct_nt_f16
FFI_COOMM_HOMO_CT_NT(_f16,  __half,         sizeof(__half))
// @tvm_ffi coomm_homo_ct_t_f16
FFI_COOMM_HOMO_CT_T (_f16,  __half,         sizeof(__half))
// @tvm_ffi coomm_homo_ct_nt_bf16
FFI_COOMM_HOMO_CT_NT(_bf16, __nv_bfloat16,  sizeof(__nv_bfloat16))
// @tvm_ffi coomm_homo_ct_t_bf16
FFI_COOMM_HOMO_CT_T (_bf16, __nv_bfloat16,  sizeof(__nv_bfloat16))

// Homo WPE instantiations
// @tvm_ffi coomm_homo_wpe_nt_f32
FFI_COOMM_HOMO_WPE_NT(_f32,  float,          sizeof(float))
// @tvm_ffi coomm_homo_wpe_t_f32
FFI_COOMM_HOMO_WPE_T (_f32,  float,          sizeof(float))
// @tvm_ffi coomm_homo_wpe_nt_f64
FFI_COOMM_HOMO_WPE_NT(_f64,  double,         sizeof(double))
// @tvm_ffi coomm_homo_wpe_t_f64
FFI_COOMM_HOMO_WPE_T (_f64,  double,         sizeof(double))
// @tvm_ffi coomm_homo_wpe_nt_f16
FFI_COOMM_HOMO_WPE_NT(_f16,  __half,         sizeof(__half))
// @tvm_ffi coomm_homo_wpe_t_f16
FFI_COOMM_HOMO_WPE_T (_f16,  __half,         sizeof(__half))
// @tvm_ffi coomm_homo_wpe_nt_bf16
FFI_COOMM_HOMO_WPE_NT(_bf16, __nv_bfloat16,  sizeof(__nv_bfloat16))
// @tvm_ffi coomm_homo_wpe_t_bf16
FFI_COOMM_HOMO_WPE_T (_bf16, __nv_bfloat16,  sizeof(__nv_bfloat16))

// Hetero CT instantiations
// @tvm_ffi coomm_hetero_ct_nt_f32
FFI_COOMM_HETERO_CT_NT(_f32,  float,          sizeof(float))
// @tvm_ffi coomm_hetero_ct_t_f32
FFI_COOMM_HETERO_CT_T (_f32,  float,          sizeof(float))
// @tvm_ffi coomm_hetero_ct_nt_f64
FFI_COOMM_HETERO_CT_NT(_f64,  double,         sizeof(double))
// @tvm_ffi coomm_hetero_ct_t_f64
FFI_COOMM_HETERO_CT_T (_f64,  double,         sizeof(double))
// @tvm_ffi coomm_hetero_ct_nt_f16
FFI_COOMM_HETERO_CT_NT(_f16,  __half,         sizeof(__half))
// @tvm_ffi coomm_hetero_ct_t_f16
FFI_COOMM_HETERO_CT_T (_f16,  __half,         sizeof(__half))
// @tvm_ffi coomm_hetero_ct_nt_bf16
FFI_COOMM_HETERO_CT_NT(_bf16, __nv_bfloat16,  sizeof(__nv_bfloat16))
// @tvm_ffi coomm_hetero_ct_t_bf16
FFI_COOMM_HETERO_CT_T (_bf16, __nv_bfloat16,  sizeof(__nv_bfloat16))

// Hetero WPE instantiations
// @tvm_ffi coomm_hetero_wpe_nt_f32
FFI_COOMM_HETERO_WPE_NT(_f32,  float,          sizeof(float))
// @tvm_ffi coomm_hetero_wpe_t_f32
FFI_COOMM_HETERO_WPE_T (_f32,  float,          sizeof(float))
// @tvm_ffi coomm_hetero_wpe_nt_f64
FFI_COOMM_HETERO_WPE_NT(_f64,  double,         sizeof(double))
// @tvm_ffi coomm_hetero_wpe_t_f64
FFI_COOMM_HETERO_WPE_T (_f64,  double,         sizeof(double))
// @tvm_ffi coomm_hetero_wpe_nt_f16
FFI_COOMM_HETERO_WPE_NT(_f16,  __half,         sizeof(__half))
// @tvm_ffi coomm_hetero_wpe_t_f16
FFI_COOMM_HETERO_WPE_T (_f16,  __half,         sizeof(__half))
// @tvm_ffi coomm_hetero_wpe_nt_bf16
FFI_COOMM_HETERO_WPE_NT(_bf16, __nv_bfloat16,  sizeof(__nv_bfloat16))
// @tvm_ffi coomm_hetero_wpe_t_bf16
FFI_COOMM_HETERO_WPE_T (_bf16, __nv_bfloat16,  sizeof(__nv_bfloat16))
