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
 * binary_coomv.cu -- Event-Driven Binary COO Sparse Matrix-Vector CUDA Kernels
 * =============================================================================
 *
 * This module provides high-performance, event-driven CUDA kernels for sparse
 * matrix-vector (SpMV) multiplication where the sparse matrix is in Coordinate
 * (COO) format and the dense vector contains binary events (spikes).
 *
 * Event-Driven Optimization:
 * -------------------------
 * In SNN simulations, dense vectors are often very sparse in time (most
 * elements are zero/inactive). These kernels exploit this by checking the
 * activity of the dense vector before performing expensive atomic accumulations.
 * This "event-driven" approach significantly reduces memory traffic and
 * contention on the output buffer.
 *
 * Kernel Variants:
 * ---------------
 * Each variant is provided in two weight modes:
 *   - Homogeneous (homo): a single scalar weight data[0] is broadcast to all
 *     connections, eliminating per-NNZ weight loads from the inner loop.
 *   - Heterogeneous (hetero): per-connection weights data[k] are loaded for
 *     each NNZ entry.
 *
 * Supported Operations:
 * --------------------
 * binary_coomv (SpMV): out = A @ v  or  out = A.T @ v
 *   - Uses a grid-stride loop with atomic additions.
 *   - Optimized for various data types (f32, f64, f16, bf16).
 *
 * Data Types and Numerical Stability:
 * ----------------------------------
 * - Supports float32, float64, float16 (sm_70+), and bfloat16 (sm_80+).
 * - For reduced-precision types (f16, bf16), accumulation is performed in
 *   float32 to maintain numerical precision, with results written back
 *   atomically.
 *
 * TVM FFI Integration:
 * -------------------
 * All kernels are exposed via TVM FFI with @tvm_ffi annotations for seamless
 * integration with JAX.  Homo vs. hetero dispatch is resolved at compile time
 * on the Python side (based on weight_info.size), so there is no runtime
 * is_homo branch in the kernels.
 */

#include "../cuda_common.h"

// ============================================================================
// Per-dtype atomic-add helpers (accumulator value -> weight memory)
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
// Homogeneous kernels — scalar weight data[0] broadcast to all connections
// ============================================================================

#define DEFINE_COOMV_HOMO_ATOMIC_NT(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _coomv_homo_atomic_nt_kern##SUFFIX(                                          \
    const WEIGHT_T* __restrict__ data,                                                        \
    const int32_t*  __restrict__ row,                                                         \
    const int32_t*  __restrict__ col,                                                         \
    const SPIKE_T*  __restrict__ v,                                                           \
    WEIGHT_T*                    out,                                                         \
    int nnz                                                                                   \
) {                                                                                           \
    ACC_T homo_w = READ_W(data[0]);                                                           \
    int k = blockIdx.x * blockDim.x + threadIdx.x;                                           \
    const int stride = gridDim.x * blockDim.x;                                               \
    while (k < nnz) {                                                                         \
        bool active = IS_ACTIVE(v[col[k]]);                                                   \
        uint32_t ballot = __ballot_sync(0xffffffff, active);                                  \
        if (ballot == 0u) { k += stride; continue; }                                          \
        if (active) { ATOMIC_ADD_W(out + row[k], homo_w); }                                  \
        k += stride;                                                                          \
    }                                                                                         \
}

#define DEFINE_COOMV_HOMO_ATOMIC_T(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _coomv_homo_atomic_t_kern##SUFFIX(                                           \
    const WEIGHT_T* __restrict__ data,                                                        \
    const int32_t*  __restrict__ row,                                                         \
    const int32_t*  __restrict__ col,                                                         \
    const SPIKE_T*  __restrict__ v,                                                           \
    WEIGHT_T*                    out,                                                         \
    int nnz                                                                                   \
) {                                                                                           \
    ACC_T homo_w = READ_W(data[0]);                                                           \
    int k = blockIdx.x * blockDim.x + threadIdx.x;                                           \
    const int stride = gridDim.x * blockDim.x;                                               \
    while (k < nnz) {                                                                         \
        bool active = IS_ACTIVE(v[row[k]]);                                                   \
        uint32_t ballot = __ballot_sync(0xffffffff, active);                                  \
        if (ballot == 0u) { k += stride; continue; }                                          \
        if (active) { ATOMIC_ADD_W(out + col[k], homo_w); }                                  \
        k += stride;                                                                          \
    }                                                                                         \
}

// ============================================================================
// Heterogeneous kernels — per-connection weight data[k]
// ============================================================================

#define DEFINE_COOMV_HETERO_ATOMIC_NT(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _coomv_hetero_atomic_nt_kern##SUFFIX(                                        \
    const WEIGHT_T* __restrict__ data,                                                        \
    const int32_t*  __restrict__ row,                                                         \
    const int32_t*  __restrict__ col,                                                         \
    const SPIKE_T*  __restrict__ v,                                                           \
    WEIGHT_T*                    out,                                                         \
    int nnz                                                                                   \
) {                                                                                           \
    int k = blockIdx.x * blockDim.x + threadIdx.x;                                           \
    const int stride = gridDim.x * blockDim.x;                                               \
    while (k < nnz) {                                                                         \
        bool active = IS_ACTIVE(v[col[k]]);                                                   \
        uint32_t ballot = __ballot_sync(0xffffffff, active);                                  \
        if (ballot == 0u) { k += stride; continue; }                                          \
        if (active) { ATOMIC_ADD_W(out + row[k], READ_W(data[k])); }                         \
        k += stride;                                                                          \
    }                                                                                         \
}

#define DEFINE_COOMV_HETERO_ATOMIC_T(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _coomv_hetero_atomic_t_kern##SUFFIX(                                         \
    const WEIGHT_T* __restrict__ data,                                                        \
    const int32_t*  __restrict__ row,                                                         \
    const int32_t*  __restrict__ col,                                                         \
    const SPIKE_T*  __restrict__ v,                                                           \
    WEIGHT_T*                    out,                                                         \
    int nnz                                                                                   \
) {                                                                                           \
    int k = blockIdx.x * blockDim.x + threadIdx.x;                                           \
    const int stride = gridDim.x * blockDim.x;                                               \
    while (k < nnz) {                                                                         \
        bool active = IS_ACTIVE(v[row[k]]);                                                   \
        uint32_t ballot = __ballot_sync(0xffffffff, active);                                  \
        if (ballot == 0u) { k += stride; continue; }                                          \
        if (active) { ATOMIC_ADD_W(out + col[k], READ_W(data[k])); }                         \
        k += stride;                                                                          \
    }                                                                                         \
}

// ============================================================================
// Kernel instantiations — homogeneous
// ============================================================================

DEFINE_COOMV_HOMO_ATOMIC_NT(_f32_bool,   int8_t,        IS_ACTIVE_BOOL,  float,         float,  READ_F32,  atomic_add_f32)
DEFINE_COOMV_HOMO_ATOMIC_NT(_f32_float,  float,         IS_ACTIVE_FLOAT, float,         float,  READ_F32,  atomic_add_f32)
DEFINE_COOMV_HOMO_ATOMIC_T (_f32_bool,   int8_t,        IS_ACTIVE_BOOL,  float,         float,  READ_F32,  atomic_add_f32)
DEFINE_COOMV_HOMO_ATOMIC_T (_f32_float,  float,         IS_ACTIVE_FLOAT, float,         float,  READ_F32,  atomic_add_f32)
DEFINE_COOMV_HOMO_ATOMIC_NT(_f64_bool,   int8_t,        IS_ACTIVE_BOOL,  double,        double, READ_F64,  atomic_add_f64)
DEFINE_COOMV_HOMO_ATOMIC_NT(_f64_float,  float,         IS_ACTIVE_FLOAT, double,        double, READ_F64,  atomic_add_f64)
DEFINE_COOMV_HOMO_ATOMIC_T (_f64_bool,   int8_t,        IS_ACTIVE_BOOL,  double,        double, READ_F64,  atomic_add_f64)
DEFINE_COOMV_HOMO_ATOMIC_T (_f64_float,  float,         IS_ACTIVE_FLOAT, double,        double, READ_F64,  atomic_add_f64)
DEFINE_COOMV_HOMO_ATOMIC_NT(_f16_bool,   int8_t,        IS_ACTIVE_BOOL,  __half,        float,  READ_F16,  atomic_add_f16)
DEFINE_COOMV_HOMO_ATOMIC_NT(_f16_float,  float,         IS_ACTIVE_FLOAT, __half,        float,  READ_F16,  atomic_add_f16)
DEFINE_COOMV_HOMO_ATOMIC_T (_f16_bool,   int8_t,        IS_ACTIVE_BOOL,  __half,        float,  READ_F16,  atomic_add_f16)
DEFINE_COOMV_HOMO_ATOMIC_T (_f16_float,  float,         IS_ACTIVE_FLOAT, __half,        float,  READ_F16,  atomic_add_f16)
DEFINE_COOMV_HOMO_ATOMIC_NT(_bf16_bool,  int8_t,        IS_ACTIVE_BOOL,  __nv_bfloat16, float,  READ_BF16, atomic_add_bf16)
DEFINE_COOMV_HOMO_ATOMIC_NT(_bf16_float, float,         IS_ACTIVE_FLOAT, __nv_bfloat16, float,  READ_BF16, atomic_add_bf16)
DEFINE_COOMV_HOMO_ATOMIC_T (_bf16_bool,  int8_t,        IS_ACTIVE_BOOL,  __nv_bfloat16, float,  READ_BF16, atomic_add_bf16)
DEFINE_COOMV_HOMO_ATOMIC_T (_bf16_float, float,         IS_ACTIVE_FLOAT, __nv_bfloat16, float,  READ_BF16, atomic_add_bf16)

// ============================================================================
// Kernel instantiations — heterogeneous
// ============================================================================

DEFINE_COOMV_HETERO_ATOMIC_NT(_f32_bool,   int8_t,        IS_ACTIVE_BOOL,  float,         float,  READ_F32,  atomic_add_f32)
DEFINE_COOMV_HETERO_ATOMIC_NT(_f32_float,  float,         IS_ACTIVE_FLOAT, float,         float,  READ_F32,  atomic_add_f32)
DEFINE_COOMV_HETERO_ATOMIC_T (_f32_bool,   int8_t,        IS_ACTIVE_BOOL,  float,         float,  READ_F32,  atomic_add_f32)
DEFINE_COOMV_HETERO_ATOMIC_T (_f32_float,  float,         IS_ACTIVE_FLOAT, float,         float,  READ_F32,  atomic_add_f32)
DEFINE_COOMV_HETERO_ATOMIC_NT(_f64_bool,   int8_t,        IS_ACTIVE_BOOL,  double,        double, READ_F64,  atomic_add_f64)
DEFINE_COOMV_HETERO_ATOMIC_NT(_f64_float,  float,         IS_ACTIVE_FLOAT, double,        double, READ_F64,  atomic_add_f64)
DEFINE_COOMV_HETERO_ATOMIC_T (_f64_bool,   int8_t,        IS_ACTIVE_BOOL,  double,        double, READ_F64,  atomic_add_f64)
DEFINE_COOMV_HETERO_ATOMIC_T (_f64_float,  float,         IS_ACTIVE_FLOAT, double,        double, READ_F64,  atomic_add_f64)
DEFINE_COOMV_HETERO_ATOMIC_NT(_f16_bool,   int8_t,        IS_ACTIVE_BOOL,  __half,        float,  READ_F16,  atomic_add_f16)
DEFINE_COOMV_HETERO_ATOMIC_NT(_f16_float,  float,         IS_ACTIVE_FLOAT, __half,        float,  READ_F16,  atomic_add_f16)
DEFINE_COOMV_HETERO_ATOMIC_T (_f16_bool,   int8_t,        IS_ACTIVE_BOOL,  __half,        float,  READ_F16,  atomic_add_f16)
DEFINE_COOMV_HETERO_ATOMIC_T (_f16_float,  float,         IS_ACTIVE_FLOAT, __half,        float,  READ_F16,  atomic_add_f16)
DEFINE_COOMV_HETERO_ATOMIC_NT(_bf16_bool,  int8_t,        IS_ACTIVE_BOOL,  __nv_bfloat16, float,  READ_BF16, atomic_add_bf16)
DEFINE_COOMV_HETERO_ATOMIC_NT(_bf16_float, float,         IS_ACTIVE_FLOAT, __nv_bfloat16, float,  READ_BF16, atomic_add_bf16)
DEFINE_COOMV_HETERO_ATOMIC_T (_bf16_bool,  int8_t,        IS_ACTIVE_BOOL,  __nv_bfloat16, float,  READ_BF16, atomic_add_bf16)
DEFINE_COOMV_HETERO_ATOMIC_T (_bf16_float, float,         IS_ACTIVE_FLOAT, __nv_bfloat16, float,  READ_BF16, atomic_add_bf16)

// ============================================================================
// FFI entry point macros — homogeneous
// ============================================================================

#define FFI_COOMV_HOMO_ATOMIC_NT(SUFFIX, WEIGHT_C_T, SPIKE_C_T, OUT_BYTES_PER_ELEM)         \
void binary_coomv_homo_atomic_nt##SUFFIX(                                                     \
    tvm::ffi::TensorView data,                                                                \
    tvm::ffi::TensorView row_idx,                                                             \
    tvm::ffi::TensorView col_idx,                                                             \
    tvm::ffi::TensorView v,                                                                   \
    tvm::ffi::TensorView output,                                                              \
    int64_t stream                                                                            \
) {                                                                                           \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                  \
    int nnz = static_cast<int>(row_idx.size(0));                                              \
    int m   = static_cast<int>(output.size(0));                                               \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                         \
    cudaMemsetAsync(d_out, 0, (size_t)m * OUT_BYTES_PER_ELEM, s);                            \
    if (nnz == 0) return;                                                                     \
    int block = 256;                                                                          \
    int grid  = (nnz + block - 1) / block;                                                   \
    _coomv_homo_atomic_nt_kern##SUFFIX<<<grid, block, 0, s>>>(                                \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                                      \
        static_cast<const int32_t*>(row_idx.data_ptr()),                                      \
        static_cast<const int32_t*>(col_idx.data_ptr()),                                      \
        static_cast<const SPIKE_C_T*>(v.data_ptr()),                                          \
        d_out, nnz                                                                            \
    );                                                                                        \
}

#define FFI_COOMV_HOMO_ATOMIC_T(SUFFIX, WEIGHT_C_T, SPIKE_C_T, OUT_BYTES_PER_ELEM)          \
void binary_coomv_homo_atomic_t##SUFFIX(                                                      \
    tvm::ffi::TensorView data,                                                                \
    tvm::ffi::TensorView row_idx,                                                             \
    tvm::ffi::TensorView col_idx,                                                             \
    tvm::ffi::TensorView v,                                                                   \
    tvm::ffi::TensorView output,                                                              \
    int64_t stream                                                                            \
) {                                                                                           \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                  \
    int nnz = static_cast<int>(row_idx.size(0));                                              \
    int k   = static_cast<int>(output.size(0));                                               \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                         \
    cudaMemsetAsync(d_out, 0, (size_t)k * OUT_BYTES_PER_ELEM, s);                            \
    if (nnz == 0) return;                                                                     \
    int block = 256;                                                                          \
    int grid  = (nnz + block - 1) / block;                                                   \
    _coomv_homo_atomic_t_kern##SUFFIX<<<grid, block, 0, s>>>(                                 \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                                      \
        static_cast<const int32_t*>(row_idx.data_ptr()),                                      \
        static_cast<const int32_t*>(col_idx.data_ptr()),                                      \
        static_cast<const SPIKE_C_T*>(v.data_ptr()),                                          \
        d_out, nnz                                                                            \
    );                                                                                        \
}

// ============================================================================
// FFI entry point macros — heterogeneous
// ============================================================================

#define FFI_COOMV_HETERO_ATOMIC_NT(SUFFIX, WEIGHT_C_T, SPIKE_C_T, OUT_BYTES_PER_ELEM)       \
void binary_coomv_hetero_atomic_nt##SUFFIX(                                                   \
    tvm::ffi::TensorView data,                                                                \
    tvm::ffi::TensorView row_idx,                                                             \
    tvm::ffi::TensorView col_idx,                                                             \
    tvm::ffi::TensorView v,                                                                   \
    tvm::ffi::TensorView output,                                                              \
    int64_t stream                                                                            \
) {                                                                                           \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                  \
    int nnz = static_cast<int>(row_idx.size(0));                                              \
    int m   = static_cast<int>(output.size(0));                                               \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                         \
    cudaMemsetAsync(d_out, 0, (size_t)m * OUT_BYTES_PER_ELEM, s);                            \
    if (nnz == 0) return;                                                                     \
    int block = 256;                                                                          \
    int grid  = (nnz + block - 1) / block;                                                   \
    _coomv_hetero_atomic_nt_kern##SUFFIX<<<grid, block, 0, s>>>(                              \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                                      \
        static_cast<const int32_t*>(row_idx.data_ptr()),                                      \
        static_cast<const int32_t*>(col_idx.data_ptr()),                                      \
        static_cast<const SPIKE_C_T*>(v.data_ptr()),                                          \
        d_out, nnz                                                                            \
    );                                                                                        \
}

#define FFI_COOMV_HETERO_ATOMIC_T(SUFFIX, WEIGHT_C_T, SPIKE_C_T, OUT_BYTES_PER_ELEM)        \
void binary_coomv_hetero_atomic_t##SUFFIX(                                                    \
    tvm::ffi::TensorView data,                                                                \
    tvm::ffi::TensorView row_idx,                                                             \
    tvm::ffi::TensorView col_idx,                                                             \
    tvm::ffi::TensorView v,                                                                   \
    tvm::ffi::TensorView output,                                                              \
    int64_t stream                                                                            \
) {                                                                                           \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                  \
    int nnz = static_cast<int>(row_idx.size(0));                                              \
    int k   = static_cast<int>(output.size(0));                                               \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                         \
    cudaMemsetAsync(d_out, 0, (size_t)k * OUT_BYTES_PER_ELEM, s);                            \
    if (nnz == 0) return;                                                                     \
    int block = 256;                                                                          \
    int grid  = (nnz + block - 1) / block;                                                   \
    _coomv_hetero_atomic_t_kern##SUFFIX<<<grid, block, 0, s>>>(                               \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                                      \
        static_cast<const int32_t*>(row_idx.data_ptr()),                                      \
        static_cast<const int32_t*>(col_idx.data_ptr()),                                      \
        static_cast<const SPIKE_C_T*>(v.data_ptr()),                                          \
        d_out, nnz                                                                            \
    );                                                                                        \
}

// ============================================================================
// FFI instantiations — homogeneous NT
// ============================================================================

// @tvm_ffi binary_coomv_homo_atomic_nt_f32_bool
FFI_COOMV_HOMO_ATOMIC_NT(_f32_bool,   float,         int8_t, sizeof(float))
// @tvm_ffi binary_coomv_homo_atomic_nt_f32_float
FFI_COOMV_HOMO_ATOMIC_NT(_f32_float,  float,         float,  sizeof(float))
// @tvm_ffi binary_coomv_homo_atomic_nt_f64_bool
FFI_COOMV_HOMO_ATOMIC_NT(_f64_bool,   double,        int8_t, sizeof(double))
// @tvm_ffi binary_coomv_homo_atomic_nt_f64_float
FFI_COOMV_HOMO_ATOMIC_NT(_f64_float,  double,        float,  sizeof(double))
// @tvm_ffi binary_coomv_homo_atomic_nt_f16_bool
FFI_COOMV_HOMO_ATOMIC_NT(_f16_bool,   __half,        int8_t, sizeof(__half))
// @tvm_ffi binary_coomv_homo_atomic_nt_f16_float
FFI_COOMV_HOMO_ATOMIC_NT(_f16_float,  __half,        float,  sizeof(__half))
// @tvm_ffi binary_coomv_homo_atomic_nt_bf16_bool
FFI_COOMV_HOMO_ATOMIC_NT(_bf16_bool,  __nv_bfloat16, int8_t, sizeof(__nv_bfloat16))
// @tvm_ffi binary_coomv_homo_atomic_nt_bf16_float
FFI_COOMV_HOMO_ATOMIC_NT(_bf16_float, __nv_bfloat16, float,  sizeof(__nv_bfloat16))

// ============================================================================
// FFI instantiations — homogeneous T
// ============================================================================

// @tvm_ffi binary_coomv_homo_atomic_t_f32_bool
FFI_COOMV_HOMO_ATOMIC_T(_f32_bool,   float,         int8_t, sizeof(float))
// @tvm_ffi binary_coomv_homo_atomic_t_f32_float
FFI_COOMV_HOMO_ATOMIC_T(_f32_float,  float,         float,  sizeof(float))
// @tvm_ffi binary_coomv_homo_atomic_t_f64_bool
FFI_COOMV_HOMO_ATOMIC_T(_f64_bool,   double,        int8_t, sizeof(double))
// @tvm_ffi binary_coomv_homo_atomic_t_f64_float
FFI_COOMV_HOMO_ATOMIC_T(_f64_float,  double,        float,  sizeof(double))
// @tvm_ffi binary_coomv_homo_atomic_t_f16_bool
FFI_COOMV_HOMO_ATOMIC_T(_f16_bool,   __half,        int8_t, sizeof(__half))
// @tvm_ffi binary_coomv_homo_atomic_t_f16_float
FFI_COOMV_HOMO_ATOMIC_T(_f16_float,  __half,        float,  sizeof(__half))
// @tvm_ffi binary_coomv_homo_atomic_t_bf16_bool
FFI_COOMV_HOMO_ATOMIC_T(_bf16_bool,  __nv_bfloat16, int8_t, sizeof(__nv_bfloat16))
// @tvm_ffi binary_coomv_homo_atomic_t_bf16_float
FFI_COOMV_HOMO_ATOMIC_T(_bf16_float, __nv_bfloat16, float,  sizeof(__nv_bfloat16))

// ============================================================================
// FFI instantiations — heterogeneous NT
// ============================================================================

// @tvm_ffi binary_coomv_hetero_atomic_nt_f32_bool
FFI_COOMV_HETERO_ATOMIC_NT(_f32_bool,   float,         int8_t, sizeof(float))
// @tvm_ffi binary_coomv_hetero_atomic_nt_f32_float
FFI_COOMV_HETERO_ATOMIC_NT(_f32_float,  float,         float,  sizeof(float))
// @tvm_ffi binary_coomv_hetero_atomic_nt_f64_bool
FFI_COOMV_HETERO_ATOMIC_NT(_f64_bool,   double,        int8_t, sizeof(double))
// @tvm_ffi binary_coomv_hetero_atomic_nt_f64_float
FFI_COOMV_HETERO_ATOMIC_NT(_f64_float,  double,        float,  sizeof(double))
// @tvm_ffi binary_coomv_hetero_atomic_nt_f16_bool
FFI_COOMV_HETERO_ATOMIC_NT(_f16_bool,   __half,        int8_t, sizeof(__half))
// @tvm_ffi binary_coomv_hetero_atomic_nt_f16_float
FFI_COOMV_HETERO_ATOMIC_NT(_f16_float,  __half,        float,  sizeof(__half))
// @tvm_ffi binary_coomv_hetero_atomic_nt_bf16_bool
FFI_COOMV_HETERO_ATOMIC_NT(_bf16_bool,  __nv_bfloat16, int8_t, sizeof(__nv_bfloat16))
// @tvm_ffi binary_coomv_hetero_atomic_nt_bf16_float
FFI_COOMV_HETERO_ATOMIC_NT(_bf16_float, __nv_bfloat16, float,  sizeof(__nv_bfloat16))

// ============================================================================
// FFI instantiations — heterogeneous T
// ============================================================================

// @tvm_ffi binary_coomv_hetero_atomic_t_f32_bool
FFI_COOMV_HETERO_ATOMIC_T(_f32_bool,   float,         int8_t, sizeof(float))
// @tvm_ffi binary_coomv_hetero_atomic_t_f32_float
FFI_COOMV_HETERO_ATOMIC_T(_f32_float,  float,         float,  sizeof(float))
// @tvm_ffi binary_coomv_hetero_atomic_t_f64_bool
FFI_COOMV_HETERO_ATOMIC_T(_f64_bool,   double,        int8_t, sizeof(double))
// @tvm_ffi binary_coomv_hetero_atomic_t_f64_float
FFI_COOMV_HETERO_ATOMIC_T(_f64_float,  double,        float,  sizeof(double))
// @tvm_ffi binary_coomv_hetero_atomic_t_f16_bool
FFI_COOMV_HETERO_ATOMIC_T(_f16_bool,   __half,        int8_t, sizeof(__half))
// @tvm_ffi binary_coomv_hetero_atomic_t_f16_float
FFI_COOMV_HETERO_ATOMIC_T(_f16_float,  __half,        float,  sizeof(__half))
// @tvm_ffi binary_coomv_hetero_atomic_t_bf16_bool
FFI_COOMV_HETERO_ATOMIC_T(_bf16_bool,  __nv_bfloat16, int8_t, sizeof(__nv_bfloat16))
// @tvm_ffi binary_coomv_hetero_atomic_t_bf16_float
FFI_COOMV_HETERO_ATOMIC_T(_bf16_float, __nv_bfloat16, float,  sizeof(__nv_bfloat16))
