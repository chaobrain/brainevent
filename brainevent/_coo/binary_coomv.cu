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
 * ============================================================================
 *
 * Python API: brainevent.binary_coomv(data, row, col, v, *, shape, transpose, backend)
 *
 * Event-driven COO (Coordinate) sparse matrix-vector product where the
 * dense vector ``v`` contains binary events (spikes). Only entries whose
 * corresponding ``v`` element is active contribute to the output, enabling
 * event-driven sparsity exploitation beyond the structural sparsity of the
 * COO matrix.
 *
 * Parameters
 * ----------
 * data : 1-D float tensor of shape [1] (homogeneous) or [nnz] (heterogeneous).
 * row  : 1-D int32 tensor of shape [nnz] -- row indices.
 * col  : 1-D int32 tensor of shape [nnz] -- column indices.
 * v    : 1-D tensor of shape [k] -- spike (event) vector.
 *        bool  (int8): active when != 0.
 *        float (f32):  active when  > 0.
 *
 * Semantics
 * ---------
 * NT mode (transpose=False):
 *   out[row[k]] += data[k]  for each k where v[col[k]] is active
 *   Output shape: [m].  v shape: [k_cols].
 *
 * T mode (transpose=True):
 *   out[col[k]] += data[k]  for each k where v[row[k]] is active
 *   Output shape: [k_cols].  v shape: [m].
 *
 * Kernel variants
 * ---------------
 * Atomic (universal, transpose=False and True):
 *   _coomv_atomic_nt: Grid-stride loop over NNZ, one thread per entry.
 *                     Checks v[col[k]] (NT) or v[row[k]] (T), performs
 *                     atomicAdd to the output. Coalesced reads of row[],
 *                     col[], data[] arrays; random cached reads of v[];
 *                     random atomic writes to out[].
 *                     Works for any input (sorted or unsorted).
 *
 *   _coomv_atomic_t: Same but for transpose=True.
 *
 * Homogeneous vs Heterogeneous weights:
 *   Detected at runtime from data tensor: data.size(0)==1 => homogeneous.
 *   Homogeneous: single scalar weight broadcast over all NNZ.
 *   Heterogeneous: per-NNZ weight array data[k].
 *
 * Dtype support:
 *   Weights and output: float32, float64, float16 (sm_70+), bfloat16 (sm_80+).
 *   f16 and bf16 accumulate in float32 for numerical stability; atomic
 *   writes back as f16/bf16 using native atomicAdd where available
 *   (sm_70+/sm_80+) or CAS-loop fallback for older architectures.
 *
 * Output initialization:
 *   The output buffer must be zeroed before the atomic scatter.
 *   The TVM FFI entry function calls cudaMemsetAsync prior to kernel launch.
 *
 * IMPORTANT: All data_ptr() calls return GPU device pointers.
 * NEVER dereference on the host. Pass to kernels unchanged.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// ============================================================================
// Warp-level reduction helpers (used by future segmented-reduction kernels)
// ============================================================================

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

// ============================================================================
// Active-check predicates
// ============================================================================

#define IS_ACTIVE_BOOL(s)  ((s) != 0)
#define IS_ACTIVE_FLOAT(s) ((s) > 0.0f)

// ============================================================================
// Per-dtype conversion macros: READ converts WEIGHT_T -> ACC_T
// ============================================================================

#define READ_F32(x)   (x)
#define READ_F64(x)   (x)
#define READ_F16(x)   __half2float(x)
#define READ_BF16(x)  __bfloat162float(x)

// ============================================================================
// Per-dtype atomic-add helpers (accumulator value -> weight memory)
//
// For f32/f64: native atomicAdd, universally supported (sm_60+ for f64).
// For f16: atomicAdd(__half*, __half) on sm_70+; CAS-based fallback below.
// For bf16: atomicAdd(__nv_bfloat16*, ...) on sm_80+; CAS-based fallback.
//
// The CAS fallbacks handle two packed halves stored in a 32-bit word,
// using atomicCAS on the aligned 4-byte word.
// ============================================================================

__device__ __inline__ void atomic_add_f32(float* addr, float val) {
    atomicAdd(addr, val);
}

__device__ __inline__ void atomic_add_f64(double* addr, double val) {
    atomicAdd(addr, val);
}

// Float16 atomic add: native on sm_70+, CAS fallback otherwise.
__device__ __inline__ void atomic_add_f16(__half* addr, float val) {
#if __CUDA_ARCH__ >= 700
    atomicAdd(addr, __float2half(val));
#else
    // CAS-based fallback: pack two f16 values in a 32-bit word.
    // The address must be 2-byte aligned (guaranteed for __half arrays).
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

// BFloat16 atomic add: native on sm_80+, CAS fallback otherwise.
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
// Atomic NT kernel macro
//
// Semantics: out[row[k]] += data[k] * (v[col[k]] active)  for k in [0, nnz)
//
// Grid-stride loop: each thread processes one or more NNZ entries.
// Coalesced reads: row[], col[], data[] are read sequentially.
// Event-driven: the IS_ACTIVE check skips the atomicAdd for inactive spikes,
//               saving one round-trip to the (potentially contended) output.
// Homo/hetero: is_homo=1 broadcasts data[0]; is_homo=0 reads data[k] per entry.
//              The condition is warp-uniform (same for all threads in any kernel
//              launch), so there is no warp divergence.
// ============================================================================

#define DEFINE_COOMV_ATOMIC_NT(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _coomv_atomic_nt_kern##SUFFIX(                                                      \
    const WEIGHT_T* __restrict__ data,                                                              \
    const int32_t*  __restrict__ row,                                                               \
    const int32_t*  __restrict__ col,                                                               \
    const SPIKE_T*  __restrict__ v,                                                                 \
    WEIGHT_T*                    out,                                                               \
    int nnz, int is_homo                                                                            \
) {                                                                                                 \
    /* Pre-load homo weight: valid read since data[0] always exists (nnz > 0). */                  \
    ACC_T homo_w = READ_W(data[0]);                                                                 \
    int k = blockIdx.x * blockDim.x + threadIdx.x;                                                 \
    const int stride = gridDim.x * blockDim.x;                                                     \
    while (k < nnz) {                                                                               \
        if (IS_ACTIVE(v[col[k]])) {                                                                 \
            ACC_T w = is_homo ? homo_w : READ_W(data[k]);                                           \
            ATOMIC_ADD_W(out + row[k], w);                                                          \
        }                                                                                           \
        k += stride;                                                                                \
    }                                                                                               \
}

// ============================================================================
// Atomic T kernel macro
//
// Semantics: out[col[k]] += data[k] * (v[row[k]] active)  for k in [0, nnz)
//
// Identical structure to NT, but reading v[row[k]] and scatter-adding to out[col[k]].
// ============================================================================

#define DEFINE_COOMV_ATOMIC_T(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W) \
__global__ void _coomv_atomic_t_kern##SUFFIX(                                                      \
    const WEIGHT_T* __restrict__ data,                                                             \
    const int32_t*  __restrict__ row,                                                              \
    const int32_t*  __restrict__ col,                                                              \
    const SPIKE_T*  __restrict__ v,                                                                \
    WEIGHT_T*                    out,                                                              \
    int nnz, int is_homo                                                                           \
) {                                                                                                \
    ACC_T homo_w = READ_W(data[0]);                                                                \
    int k = blockIdx.x * blockDim.x + threadIdx.x;                                                \
    const int stride = gridDim.x * blockDim.x;                                                    \
    while (k < nnz) {                                                                              \
        if (IS_ACTIVE(v[row[k]])) {                                                                \
            ACC_T w = is_homo ? homo_w : READ_W(data[k]);                                          \
            ATOMIC_ADD_W(out + col[k], w);                                                         \
        }                                                                                          \
        k += stride;                                                                               \
    }                                                                                              \
}

// ============================================================================
// Kernel instantiations: 4 dtypes x 2 spike types x 2 directions = 16 kernels
// ============================================================================

// ---- Float32 ----
DEFINE_COOMV_ATOMIC_NT(_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMV_ATOMIC_NT(_f32_float, float,  IS_ACTIVE_FLOAT, float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMV_ATOMIC_T (_f32_bool,  int8_t, IS_ACTIVE_BOOL,  float,  float,  READ_F32,  atomic_add_f32)
DEFINE_COOMV_ATOMIC_T (_f32_float, float,  IS_ACTIVE_FLOAT, float,  float,  READ_F32,  atomic_add_f32)

// ---- Float64 ----
DEFINE_COOMV_ATOMIC_NT(_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMV_ATOMIC_NT(_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMV_ATOMIC_T (_f64_bool,  int8_t, IS_ACTIVE_BOOL,  double, double, READ_F64,  atomic_add_f64)
DEFINE_COOMV_ATOMIC_T (_f64_float, float,  IS_ACTIVE_FLOAT, double, double, READ_F64,  atomic_add_f64)

// ---- Float16 (accumulate in f32, write back as f16) ----
DEFINE_COOMV_ATOMIC_NT(_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMV_ATOMIC_NT(_f16_float, float,  IS_ACTIVE_FLOAT, __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMV_ATOMIC_T (_f16_bool,  int8_t, IS_ACTIVE_BOOL,  __half, float,  READ_F16,  atomic_add_f16)
DEFINE_COOMV_ATOMIC_T (_f16_float, float,  IS_ACTIVE_FLOAT, __half, float,  READ_F16,  atomic_add_f16)

// ---- BFloat16 (accumulate in f32, write back as bf16) ----
DEFINE_COOMV_ATOMIC_NT(_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMV_ATOMIC_NT(_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMV_ATOMIC_T (_bf16_bool,  int8_t, IS_ACTIVE_BOOL,  __nv_bfloat16, float, READ_BF16, atomic_add_bf16)
DEFINE_COOMV_ATOMIC_T (_bf16_float, float,  IS_ACTIVE_FLOAT, __nv_bfloat16, float, READ_BF16, atomic_add_bf16)


// ============================================================================
// TVM FFI Entry Point Macros
// ============================================================================
//
// Convention: args = (data, row_idx, col_idx, v, output, stream)
//   data    : [1] (homo) or [nnz] (hetero) -- weight values
//   row_idx : [nnz], int32                 -- row indices
//   col_idx : [nnz], int32                 -- column indices
//   v       : [k] (NT) or [m] (T)          -- spike vector
//   output  : [m] (NT) or [k_cols] (T)     -- result (zero-initialized here)
//   stream  : CUDA stream handle (int64)
//
// IMPORTANT: data_ptr() returns GPU device pointers.
// NEVER dereference on the host. Only extract metadata (ndim, size).
//
// Output zeroing: cudaMemsetAsync is called before the kernel to ensure
// the atomicAdd scatter begins from a clean state.
//
// Block size: 256 threads per block provides good occupancy across Ampere/
// Turing/Volta architectures with typical register pressure (~16 regs/thread).
// Grid size: ceil(nnz / 256). For nnz < 256 the grid has one block with
// many idle threads, which is fine since the guard `k < nnz` handles it.
// ============================================================================

// ---- FFI macro: atomic NT ----
#define FFI_COOMV_ATOMIC_NT(SUFFIX, WEIGHT_C_T, SPIKE_C_T, OUT_BYTES_PER_ELEM)              \
void binary_coomv_atomic_nt##SUFFIX(                                                          \
    tvm::ffi::TensorView data,                                                                \
    tvm::ffi::TensorView row_idx,                                                             \
    tvm::ffi::TensorView col_idx,                                                             \
    tvm::ffi::TensorView v,                                                                   \
    tvm::ffi::TensorView output,                                                              \
    int64_t stream                                                                            \
) {                                                                                           \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                 \
    int nnz = static_cast<int>(row_idx.size(0));                                             \
    int m   = static_cast<int>(output.size(0));                                              \
    int is_homo = (data.size(0) == 1) ? 1 : 0;                                              \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                        \
    /* Zero output before atomic scatter */                                                   \
    cudaMemsetAsync(d_out, 0, (size_t)m * OUT_BYTES_PER_ELEM, s);                           \
    if (nnz == 0) return;                                                                     \
    int block = 256;                                                                          \
    int grid  = (nnz + block - 1) / block;                                                  \
    _coomv_atomic_nt_kern##SUFFIX<<<grid, block, 0, s>>>(                                    \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                                     \
        static_cast<const int32_t*>(row_idx.data_ptr()),                                     \
        static_cast<const int32_t*>(col_idx.data_ptr()),                                     \
        static_cast<const SPIKE_C_T*>(v.data_ptr()),                                         \
        d_out, nnz, is_homo                                                                  \
    );                                                                                        \
}

// ---- FFI macro: atomic T ----
#define FFI_COOMV_ATOMIC_T(SUFFIX, WEIGHT_C_T, SPIKE_C_T, OUT_BYTES_PER_ELEM)               \
void binary_coomv_atomic_t##SUFFIX(                                                           \
    tvm::ffi::TensorView data,                                                                \
    tvm::ffi::TensorView row_idx,                                                             \
    tvm::ffi::TensorView col_idx,                                                             \
    tvm::ffi::TensorView v,                                                                   \
    tvm::ffi::TensorView output,                                                              \
    int64_t stream                                                                            \
) {                                                                                           \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                 \
    int nnz = static_cast<int>(row_idx.size(0));                                             \
    int k   = static_cast<int>(output.size(0));                                              \
    int is_homo = (data.size(0) == 1) ? 1 : 0;                                              \
    WEIGHT_C_T* d_out = static_cast<WEIGHT_C_T*>(output.data_ptr());                        \
    /* Zero output before atomic scatter */                                                   \
    cudaMemsetAsync(d_out, 0, (size_t)k * OUT_BYTES_PER_ELEM, s);                           \
    if (nnz == 0) return;                                                                     \
    int block = 256;                                                                          \
    int grid  = (nnz + block - 1) / block;                                                  \
    _coomv_atomic_t_kern##SUFFIX<<<grid, block, 0, s>>>(                                     \
        static_cast<const WEIGHT_C_T*>(data.data_ptr()),                                     \
        static_cast<const int32_t*>(row_idx.data_ptr()),                                     \
        static_cast<const int32_t*>(col_idx.data_ptr()),                                     \
        static_cast<const SPIKE_C_T*>(v.data_ptr()),                                         \
        d_out, nnz, is_homo                                                                  \
    );                                                                                        \
}

// ============================================================================
// Instantiate TVM FFI entry points via macros + @tvm_ffi annotations
// ============================================================================

// ---- Float32 (4 bytes/elem) ----
// @tvm_ffi binary_coomv_atomic_nt_f32_bool
FFI_COOMV_ATOMIC_NT(_f32_bool,  float,  int8_t, sizeof(float))
// @tvm_ffi binary_coomv_atomic_nt_f32_float
FFI_COOMV_ATOMIC_NT(_f32_float, float,  float,  sizeof(float))
// @tvm_ffi binary_coomv_atomic_t_f32_bool
FFI_COOMV_ATOMIC_T (_f32_bool,  float,  int8_t, sizeof(float))
// @tvm_ffi binary_coomv_atomic_t_f32_float
FFI_COOMV_ATOMIC_T (_f32_float, float,  float,  sizeof(float))

// ---- Float64 (8 bytes/elem) ----
// @tvm_ffi binary_coomv_atomic_nt_f64_bool
FFI_COOMV_ATOMIC_NT(_f64_bool,  double, int8_t, sizeof(double))
// @tvm_ffi binary_coomv_atomic_nt_f64_float
FFI_COOMV_ATOMIC_NT(_f64_float, double, float,  sizeof(double))
// @tvm_ffi binary_coomv_atomic_t_f64_bool
FFI_COOMV_ATOMIC_T (_f64_bool,  double, int8_t, sizeof(double))
// @tvm_ffi binary_coomv_atomic_t_f64_float
FFI_COOMV_ATOMIC_T (_f64_float, double, float,  sizeof(double))

// ---- Float16 (2 bytes/elem; accumulates in f32) ----
// @tvm_ffi binary_coomv_atomic_nt_f16_bool
FFI_COOMV_ATOMIC_NT(_f16_bool,  __half, int8_t, sizeof(__half))
// @tvm_ffi binary_coomv_atomic_nt_f16_float
FFI_COOMV_ATOMIC_NT(_f16_float, __half, float,  sizeof(__half))
// @tvm_ffi binary_coomv_atomic_t_f16_bool
FFI_COOMV_ATOMIC_T (_f16_bool,  __half, int8_t, sizeof(__half))
// @tvm_ffi binary_coomv_atomic_t_f16_float
FFI_COOMV_ATOMIC_T (_f16_float, __half, float,  sizeof(__half))

// ---- BFloat16 (2 bytes/elem; requires CUDA 11.0+; accumulates in f32) ----
// @tvm_ffi binary_coomv_atomic_nt_bf16_bool
FFI_COOMV_ATOMIC_NT(_bf16_bool,  __nv_bfloat16, int8_t, sizeof(__nv_bfloat16))
// @tvm_ffi binary_coomv_atomic_nt_bf16_float
FFI_COOMV_ATOMIC_NT(_bf16_float, __nv_bfloat16, float,  sizeof(__nv_bfloat16))
// @tvm_ffi binary_coomv_atomic_t_bf16_bool
FFI_COOMV_ATOMIC_T (_bf16_bool,  __nv_bfloat16, int8_t, sizeof(__nv_bfloat16))
// @tvm_ffi binary_coomv_atomic_t_bf16_float
FFI_COOMV_ATOMIC_T (_bf16_float, __nv_bfloat16, float,  sizeof(__nv_bfloat16))
