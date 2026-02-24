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
 * float_coomv.cu -- Float-Weighted COO Sparse Matrix-Vector CUDA Kernels
 * =======================================================================
 *
 * This module provides high-performance CUDA kernels for standard (non-event-driven)
 * sparse matrix-vector (SpMV) multiplications where the sparse matrix is in
 * Coordinate (COO) format.
 *
 * Supported Operations:
 * --------------------
 * coomv (SpMV): out = A @ v  or  out = A.T @ v
 *   - Vectorized loads (float4/int4) for 4-way ILP to improve memory-level
 *     parallelism and hide latency. Falls back to scalar path for remainder.
 *   - Block size of 1024 threads (max) for better SM occupancy.
 *
 * Data Types:
 * ----------
 * - float32, float64, float16 (sm_70+), bfloat16 (sm_80+)
 * - For f16/bf16, accumulation is performed in float32 for numerical stability.
 *
 * TVM FFI Entry Points:
 * --------------------
 * coomv_homo_atomic_nt_{f32,f64,f16,bf16}   -- non-transposed SpMV (homo weights)
 * coomv_homo_atomic_t_{f32,f64,f16,bf16}    -- transposed SpMV (homo weights)
 * coomv_hetero_atomic_nt_{f32,f64,f16,bf16} -- non-transposed SpMV (hetero weights)
 * coomv_hetero_atomic_t_{f32,f64,f16,bf16}  -- transposed SpMV (hetero weights)
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
// Homo COO SpMV kernels (scalar weight broadcast to all connections)
// ============================================================================

#define DEFINE_COOMV_HOMO_ATOMIC_NT(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)   \
__global__ void _coomv_homo_atomic_nt_kern##SUFFIX(                                   \
    const WEIGHT_T* __restrict__ data,                                                \
    const int32_t*  __restrict__ row,                                                 \
    const int32_t*  __restrict__ col,                                                 \
    const WEIGHT_T* __restrict__ v,                                                   \
    WEIGHT_T*                    out,                                                 \
    int nnz                                                                           \
) {                                                                                   \
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;                        \
    const int stride = gridDim.x  * blockDim.x;                                      \
    ACC_T homo_w = READ_W(__ldg(data));                                               \
                                                                                      \
    /* Vectorized path: process 4 elements per iteration */                          \
    const int nnz_vec = (nnz >> 2) << 2;  /* round down to multiple of 4 */         \
    for (int k_base = tid * 4; k_base < nnz_vec; k_base += stride * 4) {            \
        /* Load 4 column and row indices as int4 */                                  \
        int4 c4 = __ldg(reinterpret_cast<const int4*>(col + k_base));               \
        int4 r4 = __ldg(reinterpret_cast<const int4*>(row + k_base));               \
                                                                                      \
        /* Gather vector values */                                                   \
        ACC_T v0 = READ_W(__ldg(v + c4.x));                                          \
        ACC_T v1 = READ_W(__ldg(v + c4.y));                                          \
        ACC_T v2 = READ_W(__ldg(v + c4.z));                                          \
        ACC_T v3 = READ_W(__ldg(v + c4.w));                                          \
                                                                                      \
        /* Compute and scatter */                                                    \
        ATOMIC_ADD_W(out + r4.x, homo_w * v0);                                       \
        ATOMIC_ADD_W(out + r4.y, homo_w * v1);                                       \
        ATOMIC_ADD_W(out + r4.z, homo_w * v2);                                       \
        ATOMIC_ADD_W(out + r4.w, homo_w * v3);                                       \
    }                                                                                 \
                                                                                      \
    /* Scalar path: handle remaining elements (0-3) */                               \
    for (int k = nnz_vec + tid; k < nnz; k += stride) {                              \
        int c = __ldg(col + k);                                                       \
        ACC_T v_val = READ_W(__ldg(v + c));                                           \
        ATOMIC_ADD_W(out + __ldg(row + k), homo_w * v_val);                          \
    }                                                                                 \
}

#define DEFINE_COOMV_HOMO_ATOMIC_T(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)    \
__global__ void _coomv_homo_atomic_t_kern##SUFFIX(                                    \
    const WEIGHT_T* __restrict__ data,                                                \
    const int32_t*  __restrict__ row,                                                 \
    const int32_t*  __restrict__ col,                                                 \
    const WEIGHT_T* __restrict__ v,                                                   \
    WEIGHT_T*                    out,                                                 \
    int nnz                                                                           \
) {                                                                                   \
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;                        \
    const int stride = gridDim.x  * blockDim.x;                                      \
    ACC_T homo_w = READ_W(__ldg(data));                                               \
                                                                                      \
    /* Vectorized path: process 4 elements per iteration */                          \
    const int nnz_vec = (nnz >> 2) << 2;  /* round down to multiple of 4 */         \
    for (int k_base = tid * 4; k_base < nnz_vec; k_base += stride * 4) {            \
        /* Load 4 row and column indices */                                          \
        int4 r4 = __ldg(reinterpret_cast<const int4*>(row + k_base));               \
        int4 c4 = __ldg(reinterpret_cast<const int4*>(col + k_base));               \
                                                                                      \
        /* Gather vector values (transpose: read from row indices) */                \
        ACC_T v0 = READ_W(__ldg(v + r4.x));                                          \
        ACC_T v1 = READ_W(__ldg(v + r4.y));                                          \
        ACC_T v2 = READ_W(__ldg(v + r4.z));                                          \
        ACC_T v3 = READ_W(__ldg(v + r4.w));                                          \
                                                                                      \
        /* Compute and scatter (transpose: write to col indices) */                  \
        ATOMIC_ADD_W(out + c4.x, homo_w * v0);                                       \
        ATOMIC_ADD_W(out + c4.y, homo_w * v1);                                       \
        ATOMIC_ADD_W(out + c4.z, homo_w * v2);                                       \
        ATOMIC_ADD_W(out + c4.w, homo_w * v3);                                       \
    }                                                                                 \
                                                                                      \
    /* Scalar path: handle remainder */                                              \
    for (int k = nnz_vec + tid; k < nnz; k += stride) {                              \
        int r = __ldg(row + k);                                                       \
        ACC_T v_val = READ_W(__ldg(v + r));                                           \
        ATOMIC_ADD_W(out + __ldg(col + k), homo_w * v_val);                          \
    }                                                                                 \
}

// ============================================================================
// Hetero COO SpMV kernels (per-connection weight array)
// ============================================================================

#define DEFINE_COOMV_HETERO_ATOMIC_NT(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _coomv_hetero_atomic_nt_kern##SUFFIX(                                 \
    const WEIGHT_T* __restrict__ data,                                                \
    const int32_t*  __restrict__ row,                                                 \
    const int32_t*  __restrict__ col,                                                 \
    const WEIGHT_T* __restrict__ v,                                                   \
    WEIGHT_T*                    out,                                                 \
    int nnz                                                                           \
) {                                                                                   \
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;                        \
    const int stride = gridDim.x  * blockDim.x;                                      \
                                                                                      \
    /* Vectorized path: process 4 elements per iteration */                          \
    const int nnz_vec = (nnz >> 2) << 2;  /* round down to multiple of 4 */         \
    for (int k_base = tid * 4; k_base < nnz_vec; k_base += stride * 4) {            \
        /* Load 4 column and row indices as int4 */                                  \
        int4 c4 = __ldg(reinterpret_cast<const int4*>(col + k_base));               \
        int4 r4 = __ldg(reinterpret_cast<const int4*>(row + k_base));               \
                                                                                      \
        /* Gather vector values */                                                   \
        ACC_T v0 = READ_W(__ldg(v + c4.x));                                          \
        ACC_T v1 = READ_W(__ldg(v + c4.y));                                          \
        ACC_T v2 = READ_W(__ldg(v + c4.z));                                          \
        ACC_T v3 = READ_W(__ldg(v + c4.w));                                          \
                                                                                      \
        /* Load 4 weights */                                                         \
        ACC_T w0 = READ_W(__ldg(data + k_base + 0));                                 \
        ACC_T w1 = READ_W(__ldg(data + k_base + 1));                                 \
        ACC_T w2 = READ_W(__ldg(data + k_base + 2));                                 \
        ACC_T w3 = READ_W(__ldg(data + k_base + 3));                                 \
                                                                                      \
        /* Compute and scatter */                                                    \
        ATOMIC_ADD_W(out + r4.x, w0 * v0);                                           \
        ATOMIC_ADD_W(out + r4.y, w1 * v1);                                           \
        ATOMIC_ADD_W(out + r4.z, w2 * v2);                                           \
        ATOMIC_ADD_W(out + r4.w, w3 * v3);                                           \
    }                                                                                 \
                                                                                      \
    /* Scalar path: handle remaining elements (0-3) */                               \
    for (int k = nnz_vec + tid; k < nnz; k += stride) {                              \
        int c = __ldg(col + k);                                                       \
        ACC_T v_val = READ_W(__ldg(v + c));                                           \
        ACC_T w     = READ_W(__ldg(data + k));                                        \
        ATOMIC_ADD_W(out + __ldg(row + k), w * v_val);                               \
    }                                                                                 \
}

#define DEFINE_COOMV_HETERO_ATOMIC_T(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)  \
__global__ void _coomv_hetero_atomic_t_kern##SUFFIX(                                  \
    const WEIGHT_T* __restrict__ data,                                                \
    const int32_t*  __restrict__ row,                                                 \
    const int32_t*  __restrict__ col,                                                 \
    const WEIGHT_T* __restrict__ v,                                                   \
    WEIGHT_T*                    out,                                                 \
    int nnz                                                                           \
) {                                                                                   \
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;                        \
    const int stride = gridDim.x  * blockDim.x;                                      \
                                                                                      \
    /* Vectorized path: process 4 elements per iteration */                          \
    const int nnz_vec = (nnz >> 2) << 2;  /* round down to multiple of 4 */         \
    for (int k_base = tid * 4; k_base < nnz_vec; k_base += stride * 4) {            \
        /* Load 4 row and column indices */                                          \
        int4 r4 = __ldg(reinterpret_cast<const int4*>(row + k_base));               \
        int4 c4 = __ldg(reinterpret_cast<const int4*>(col + k_base));               \
                                                                                      \
        /* Gather vector values (transpose: read from row indices) */                \
        ACC_T v0 = READ_W(__ldg(v + r4.x));                                          \
        ACC_T v1 = READ_W(__ldg(v + r4.y));                                          \
        ACC_T v2 = READ_W(__ldg(v + r4.z));                                          \
        ACC_T v3 = READ_W(__ldg(v + r4.w));                                          \
                                                                                      \
        /* Load 4 weights */                                                         \
        ACC_T w0 = READ_W(__ldg(data + k_base + 0));                                 \
        ACC_T w1 = READ_W(__ldg(data + k_base + 1));                                 \
        ACC_T w2 = READ_W(__ldg(data + k_base + 2));                                 \
        ACC_T w3 = READ_W(__ldg(data + k_base + 3));                                 \
                                                                                      \
        /* Compute and scatter (transpose: write to col indices) */                  \
        ATOMIC_ADD_W(out + c4.x, w0 * v0);                                           \
        ATOMIC_ADD_W(out + c4.y, w1 * v1);                                           \
        ATOMIC_ADD_W(out + c4.z, w2 * v2);                                           \
        ATOMIC_ADD_W(out + c4.w, w3 * v3);                                           \
    }                                                                                 \
                                                                                      \
    /* Scalar path: handle remainder */                                              \
    for (int k = nnz_vec + tid; k < nnz; k += stride) {                              \
        int r = __ldg(row + k);                                                       \
        ACC_T v_val = READ_W(__ldg(v + r));                                           \
        ACC_T w     = READ_W(__ldg(data + k));                                        \
        ATOMIC_ADD_W(out + __ldg(col + k), w * v_val);                               \
    }                                                                                 \
}

// Homo instantiations
DEFINE_COOMV_HOMO_ATOMIC_NT(_f32,  float,          float,  READ_F32,  atomic_add_f32)
DEFINE_COOMV_HOMO_ATOMIC_T (_f32,  float,          float,  READ_F32,  atomic_add_f32)
DEFINE_COOMV_HOMO_ATOMIC_NT(_f64,  double,         double, READ_F64,  atomic_add_f64)
DEFINE_COOMV_HOMO_ATOMIC_T (_f64,  double,         double, READ_F64,  atomic_add_f64)
DEFINE_COOMV_HOMO_ATOMIC_NT(_f16,  __half,         float,  READ_F16,  atomic_add_f16)
DEFINE_COOMV_HOMO_ATOMIC_T (_f16,  __half,         float,  READ_F16,  atomic_add_f16)
DEFINE_COOMV_HOMO_ATOMIC_NT(_bf16, __nv_bfloat16,  float,  READ_BF16, atomic_add_bf16)
DEFINE_COOMV_HOMO_ATOMIC_T (_bf16, __nv_bfloat16,  float,  READ_BF16, atomic_add_bf16)

// Hetero instantiations
DEFINE_COOMV_HETERO_ATOMIC_NT(_f32,  float,          float,  READ_F32,  atomic_add_f32)
DEFINE_COOMV_HETERO_ATOMIC_T (_f32,  float,          float,  READ_F32,  atomic_add_f32)
DEFINE_COOMV_HETERO_ATOMIC_NT(_f64,  double,         double, READ_F64,  atomic_add_f64)
DEFINE_COOMV_HETERO_ATOMIC_T (_f64,  double,         double, READ_F64,  atomic_add_f64)
DEFINE_COOMV_HETERO_ATOMIC_NT(_f16,  __half,         float,  READ_F16,  atomic_add_f16)
DEFINE_COOMV_HETERO_ATOMIC_T (_f16,  __half,         float,  READ_F16,  atomic_add_f16)
DEFINE_COOMV_HETERO_ATOMIC_NT(_bf16, __nv_bfloat16,  float,  READ_BF16, atomic_add_bf16)
DEFINE_COOMV_HETERO_ATOMIC_T (_bf16, __nv_bfloat16,  float,  READ_BF16, atomic_add_bf16)

// ============================================================================
// FFI entry points -- Homo
// ============================================================================

#define FFI_COOMV_HOMO_ATOMIC_NT(SUFFIX, WEIGHT_C_T, OUT_BYTES_PER_ELEM)      \
void coomv_homo_atomic_nt##SUFFIX(                                             \
    tvm::ffi::TensorView data,                                                 \
    tvm::ffi::TensorView row_idx,                                              \
    tvm::ffi::TensorView col_idx,                                              \
    tvm::ffi::TensorView v,                                                    \
    tvm::ffi::TensorView output,                                               \
    int64_t stream                                                             \
) {                                                                            \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                  \
    int nnz = static_cast<int>(row_idx.size(0));                              \
    int m   = static_cast<int>(output.size(0));                               \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());         \
    cudaMemsetAsync(d_out, 0, (size_t)m * OUT_BYTES_PER_ELEM, s);            \
    if (nnz == 0) return;                                                      \
    int block = 1024;                                                          \
    int grid  = (nnz + block * 4 - 1) / (block * 4);                         \
    _coomv_homo_atomic_nt_kern##SUFFIX<<<grid, block, 0, s>>>(                \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                      \
        static_cast<const int32_t*>(row_idx.data_ptr()),                      \
        static_cast<const int32_t*>(col_idx.data_ptr()),                      \
        static_cast<const WEIGHT_C_T*>(v.data_ptr()),                         \
        d_out, nnz                                                             \
    );                                                                         \
}

#define FFI_COOMV_HOMO_ATOMIC_T(SUFFIX, WEIGHT_C_T, OUT_BYTES_PER_ELEM)       \
void coomv_homo_atomic_t##SUFFIX(                                              \
    tvm::ffi::TensorView data,                                                 \
    tvm::ffi::TensorView row_idx,                                              \
    tvm::ffi::TensorView col_idx,                                              \
    tvm::ffi::TensorView v,                                                    \
    tvm::ffi::TensorView output,                                               \
    int64_t stream                                                             \
) {                                                                            \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                  \
    int nnz = static_cast<int>(row_idx.size(0));                              \
    int k   = static_cast<int>(output.size(0));                               \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());         \
    cudaMemsetAsync(d_out, 0, (size_t)k * OUT_BYTES_PER_ELEM, s);            \
    if (nnz == 0) return;                                                      \
    int block = 1024;                                                          \
    int grid  = (nnz + block * 4 - 1) / (block * 4);                         \
    _coomv_homo_atomic_t_kern##SUFFIX<<<grid, block, 0, s>>>(                 \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                      \
        static_cast<const int32_t*>(row_idx.data_ptr()),                      \
        static_cast<const int32_t*>(col_idx.data_ptr()),                      \
        static_cast<const WEIGHT_C_T*>(v.data_ptr()),                         \
        d_out, nnz                                                             \
    );                                                                         \
}

// ============================================================================
// FFI entry points -- Hetero
// ============================================================================

#define FFI_COOMV_HETERO_ATOMIC_NT(SUFFIX, WEIGHT_C_T, OUT_BYTES_PER_ELEM)    \
void coomv_hetero_atomic_nt##SUFFIX(                                           \
    tvm::ffi::TensorView data,                                                 \
    tvm::ffi::TensorView row_idx,                                              \
    tvm::ffi::TensorView col_idx,                                              \
    tvm::ffi::TensorView v,                                                    \
    tvm::ffi::TensorView output,                                               \
    int64_t stream                                                             \
) {                                                                            \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                  \
    int nnz = static_cast<int>(row_idx.size(0));                              \
    int m   = static_cast<int>(output.size(0));                               \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());         \
    cudaMemsetAsync(d_out, 0, (size_t)m * OUT_BYTES_PER_ELEM, s);            \
    if (nnz == 0) return;                                                      \
    int block = 1024;                                                          \
    int grid  = (nnz + block * 4 - 1) / (block * 4);                         \
    _coomv_hetero_atomic_nt_kern##SUFFIX<<<grid, block, 0, s>>>(              \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                      \
        static_cast<const int32_t*>(row_idx.data_ptr()),                      \
        static_cast<const int32_t*>(col_idx.data_ptr()),                      \
        static_cast<const WEIGHT_C_T*>(v.data_ptr()),                         \
        d_out, nnz                                                             \
    );                                                                         \
}

#define FFI_COOMV_HETERO_ATOMIC_T(SUFFIX, WEIGHT_C_T, OUT_BYTES_PER_ELEM)     \
void coomv_hetero_atomic_t##SUFFIX(                                            \
    tvm::ffi::TensorView data,                                                 \
    tvm::ffi::TensorView row_idx,                                              \
    tvm::ffi::TensorView col_idx,                                              \
    tvm::ffi::TensorView v,                                                    \
    tvm::ffi::TensorView output,                                               \
    int64_t stream                                                             \
) {                                                                            \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                  \
    int nnz = static_cast<int>(row_idx.size(0));                              \
    int k   = static_cast<int>(output.size(0));                               \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());         \
    cudaMemsetAsync(d_out, 0, (size_t)k * OUT_BYTES_PER_ELEM, s);            \
    if (nnz == 0) return;                                                      \
    int block = 1024;                                                          \
    int grid  = (nnz + block * 4 - 1) / (block * 4);                         \
    _coomv_hetero_atomic_t_kern##SUFFIX<<<grid, block, 0, s>>>(               \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                      \
        static_cast<const int32_t*>(row_idx.data_ptr()),                      \
        static_cast<const int32_t*>(col_idx.data_ptr()),                      \
        static_cast<const WEIGHT_C_T*>(v.data_ptr()),                         \
        d_out, nnz                                                             \
    );                                                                         \
}

// @tvm_ffi coomv_homo_atomic_nt_f32
FFI_COOMV_HOMO_ATOMIC_NT(_f32,  float,          sizeof(float))
// @tvm_ffi coomv_homo_atomic_t_f32
FFI_COOMV_HOMO_ATOMIC_T (_f32,  float,          sizeof(float))
// @tvm_ffi coomv_homo_atomic_nt_f64
FFI_COOMV_HOMO_ATOMIC_NT(_f64,  double,         sizeof(double))
// @tvm_ffi coomv_homo_atomic_t_f64
FFI_COOMV_HOMO_ATOMIC_T (_f64,  double,         sizeof(double))
// @tvm_ffi coomv_homo_atomic_nt_f16
FFI_COOMV_HOMO_ATOMIC_NT(_f16,  __half,         sizeof(__half))
// @tvm_ffi coomv_homo_atomic_t_f16
FFI_COOMV_HOMO_ATOMIC_T (_f16,  __half,         sizeof(__half))
// @tvm_ffi coomv_homo_atomic_nt_bf16
FFI_COOMV_HOMO_ATOMIC_NT(_bf16, __nv_bfloat16,  sizeof(__nv_bfloat16))
// @tvm_ffi coomv_homo_atomic_t_bf16
FFI_COOMV_HOMO_ATOMIC_T (_bf16, __nv_bfloat16,  sizeof(__nv_bfloat16))

// @tvm_ffi coomv_hetero_atomic_nt_f32
FFI_COOMV_HETERO_ATOMIC_NT(_f32,  float,          sizeof(float))
// @tvm_ffi coomv_hetero_atomic_t_f32
FFI_COOMV_HETERO_ATOMIC_T (_f32,  float,          sizeof(float))
// @tvm_ffi coomv_hetero_atomic_nt_f64
FFI_COOMV_HETERO_ATOMIC_NT(_f64,  double,         sizeof(double))
// @tvm_ffi coomv_hetero_atomic_t_f64
FFI_COOMV_HETERO_ATOMIC_T (_f64,  double,         sizeof(double))
// @tvm_ffi coomv_hetero_atomic_nt_f16
FFI_COOMV_HETERO_ATOMIC_NT(_f16,  __half,         sizeof(__half))
// @tvm_ffi coomv_hetero_atomic_t_f16
FFI_COOMV_HETERO_ATOMIC_T (_f16,  __half,         sizeof(__half))
// @tvm_ffi coomv_hetero_atomic_nt_bf16
FFI_COOMV_HETERO_ATOMIC_NT(_bf16, __nv_bfloat16,  sizeof(__nv_bfloat16))
// @tvm_ffi coomv_hetero_atomic_t_bf16
FFI_COOMV_HETERO_ATOMIC_T (_bf16, __nv_bfloat16,  sizeof(__nv_bfloat16))
