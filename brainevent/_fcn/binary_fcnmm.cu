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
 * binary_fcnmm.cu -- Event-Driven Binary FCN Sparse Matrix-Matrix CUDA Kernels
 * ==============================================================================
 *
 * This module provides optimized CUDA kernels for event-driven sparse
 * matrix-matrix multiplication with fixed connection number (FCN).
 *
 * Operator: binary_fcnmm
 *   - Gather mode (transpose=False): output[i,j] = sum_k weights[i,k] * is_active(matrix[indices[i,k], j])
 *   - Scatter mode (transpose=True): output[indices[i,k], j] += weights[i,k] * is_active(matrix[i,j])
 *
 * Supports weight dtypes: float32, float64, float16, bfloat16
 * Supports spike dtypes:  bool (uint8), float32, float64, float16, bfloat16
 * Supports homo (scalar weight) and hetero (per-connection weight array) modes.
 *
 * TVM FFI entry points (all named binary_fcnmm_<mode>_<spike>_<kernel>_<dtype>):
 *   binary_fcnmm_gather_bool_warp_{f32,f64,f16,bf16}
 *   binary_fcnmm_gather_bool_basic_{f32,f64,f16,bf16}
 *   binary_fcnmm_gather_float_warp_{f32,f64,f16,bf16}
 *   binary_fcnmm_gather_float_basic_{f32,f64,f16,bf16}
 *   binary_fcnmm_scatter_bool_warp_{f32,f64,f16,bf16}
 *   binary_fcnmm_scatter_bool_basic_{f32,f64,f16,bf16}
 *   binary_fcnmm_scatter_float_warp_{f32,f64,f16,bf16}
 *   binary_fcnmm_scatter_float_basic_{f32,f64,f16,bf16}
 *
 * Parameters (all FFI entry points share this signature):
 *   weights  -- [n_pre, n_conn] (hetero) or [1] (homo) weight tensor
 *   indices  -- [n_pre, n_conn] int32 connectivity indices
 *   matrix   -- [n_post, n_batch] (gather) or [n_pre, n_batch] (scatter) spike/activity matrix
 *   output   -- [n_pre, n_batch] (gather) or [n_post, n_batch] (scatter) result matrix
 *   stream   -- CUDA stream handle (int64_t)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// ============================================================================
// Warp-level reduction helpers
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
// Spike active predicates
// ============================================================================

#define IS_ACTIVE_BOOL(s)       ((s) != 0)
#define IS_ACTIVE_FLOAT_F32(s)  ((s) > 0.0f)
#define IS_ACTIVE_FLOAT_F64(s)  ((s) > 0.0)
#define IS_ACTIVE_FLOAT_F16(s)  (__half2float(s) > 0.0f)
#define IS_ACTIVE_FLOAT_BF16(s) (__bfloat162float(s) > 0.0f)

// ============================================================================
// Per-dtype conversion macros
// ============================================================================

#define READ_F32(x)   (x)
#define WRITE_F32(x)  (x)
#define READ_F64(x)   (x)
#define WRITE_F64(x)  (x)
#define READ_F16(x)   __half2float(x)
#define WRITE_F16(x)  __float2half(x)
#define READ_BF16(x)  __bfloat162float(x)
#define WRITE_BF16(x) __float2bfloat16(x)

// ============================================================================
// FCN Matrix-Matrix Multiplication (fcnmm) — Optimized CUDA Kernels
// ============================================================================
//
// Performance Status (10000x10000, 10% spike rate, n_batch=32, RTX 3080 Ti):
//   Theoretical (BW-bound): 0.089 ms (81.5 MB / 912 GB/s)
//
//   Gather SpMM (transpose=False):
//   ┌────────────────────────────────────┬──────────┬─────────┬──────────┐
//   │ Config                             │ tvmffi   │ pallas  │ vs pallas│
//   ├────────────────────────────────────┼──────────┼─────────┼──────────┤
//   │ homo, float, n_conn=1000           │ 1.57 ms  │ 1.57 ms │ 1.00x    │
//   │ homo, bool,  n_conn=1000           │ 1.69 ms  │ 1.59 ms │ 0.94x    │
//   │ homo, float, n_conn=500            │ 2.88 ms  │ 2.91 ms │ 1.01x  * │
//   │ homo, bool,  n_conn=500            │ 2.57 ms  │ 2.65 ms │ 1.03x  * │
//   │ hetero, float, n_conn=500          │ 1.57 ms  │ 1.51 ms │ 0.96x    │
//   │ hetero, bool,  n_conn=500          │ 4.46 ms  │ 4.01 ms │ 0.90x    │
//   │ hetero, float, n_conn=1000 [gap]   │ 2.94 ms  │ 1.57 ms │ 0.53x    │
//   │ hetero, bool,  n_conn=1000 [gap]   │ 2.92 ms  │ 1.91 ms │ 0.65x    │
//   └────────────────────────────────────┴──────────┴─────────┴──────────┘
//   (* = tvmffi beats pallas)
//
//   Scatter SpMM (transpose=True):
//   ┌────────────────────────────────────┬──────────┬─────────┬──────────┐
//   │ Config                             │ tvmffi   │ jax_raw │ vs raw   │
//   ├────────────────────────────────────┼──────────┼─────────┼──────────┤
//   │ hetero, float, n_conn=1000         │ 1.57 ms  │ 5.28 ms │ 3.37x  * │
//   │ hetero, bool,  n_conn=1000         │ 1.54 ms  │ 5.37 ms │ 3.48x  * │
//   │ homo, float, n_conn=1000           │ 1.58 ms  │ 5.44 ms │ 3.44x  * │
//   │ homo, bool,  n_conn=1000           │ 1.71 ms  │ 5.44 ms │ 3.17x  * │
//   └────────────────────────────────────┴──────────┴─────────┴──────────┘
//
// Gather SpMM Optimizations Applied:
//   [x] __ldg() for all read-only data (indices, matrix, weights)
//   [x] Multi-warp connection parallelism: 256 threads/block (8 warps),
//       warps divide n_conn dimension for 8x memory-level parallelism
//   [x] Shared memory index caching: cooperative bulk DRAM load of indices,
//       breaking the dependent load chain (DRAM 400cy → smem 30cy for indices)
//   [x] Manual 4x loop unrolling: 4 concurrent L2 matrix reads per warp
//       per iteration, maximizing outstanding memory requests
//   [x] Conditional weight read: skip weight load for inactive spikes
//
// Fundamental Barriers (gather mode, hetero weights, large n_conn):
//   1. **Triton vectorized gather advantage**: Pallas/Triton compiles
//      matrix_ref[ind, :] to block_k=128 concurrent gather loads per block,
//      vs our 8 warps × 4 unroll = 32 concurrent loads. This 4x difference
//      in memory-level parallelism explains the hetero n_conn=1000 gap.
//      NVRTC scalar __ldg() cannot match Triton's vectorized gather ISA.
//   2. **Weight array DRAM streaming**: At n_conn=1000 with 10K rows, the
//      weight array is 40 MB — far exceeding L2 (6 MB). Each active
//      connection requires a 400-cycle DRAM weight read. Index caching via
//      shared memory eliminates 40 MB of index DRAM traffic, but hetero
//      weights still stream from DRAM.
//   3. **Low arithmetic intensity**: ~0.025 FLOP/byte. Fundamentally
//      memory-latency-bound, not compute or bandwidth-bound.
//
// TODO: Future optimization ideas for closing the hetero n_conn=1000 gap:
//   - Kernel fusion: fuse weight read with matrix read in a single DRAM pass
//   - Cooperative groups: use cooperative launch for cross-block index sharing
//   - Vectorized loads: use float4/int4 loads for coalesced weight streaming
//   - Triton backend: port gather kernel to Pallas/Triton for vectorized gather

// Gather warp kernel: 1 block per row, 32 threads per block (for n_conn <= 32)
#define DEFINE_BGM_WARP(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W) \
__global__ void _bgm_warp_kern##SUFFIX( \
    const int32_t* __restrict__ indices, \
    const SPIKE_T* __restrict__ matrix, \
    WEIGHT_T*      __restrict__ output, \
    const WEIGHT_T* __restrict__ weights, \
    int n_pre, int n_conn, int n_batch, int is_homo \
) { \
    int row = blockIdx.x; \
    int t   = threadIdx.x; \
    int j   = (int)blockIdx.y * 32 + t; \
    if (row >= n_pre) return; \
    bool col_valid = (j < n_batch); \
    int  safe_j    = col_valid ? j : 0; \
    const int32_t*  i_row = indices + (size_t)row * n_conn; \
    const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)row * n_conn; \
    ACC_T accum = (ACC_T)0; \
    for (int k = 0; k < n_conn; k++) { \
        int  src    = __ldg(&i_row[k]); \
        bool active = col_valid && IS_ACTIVE(__ldg(&matrix[(size_t)src * n_batch + safe_j])); \
        if (active) \
            accum += is_homo ? (ACC_T)1 : READ_W(__ldg(&w_row[k])); \
    } \
    if (col_valid) \
        output[(size_t)row * n_batch + j] = \
            WRITE_W(is_homo ? (READ_W(__ldg(&weights[0])) * accum) : accum); \
}

// Gather multi-warp kernel with shared memory index caching.
// 1 block per row, 256 threads (8 warps).
//
// Phase 1: All 256 threads cooperatively load indices[row, 0..n_conn-1] into
//   shared memory using coalesced DRAM reads. This converts scattered serial
//   index loads into a bulk streaming transfer that the memory controller can
//   pipeline efficiently.
//
// Phase 2: Each warp processes n_conn/8 connections. Index reads come from
//   shared memory (~30 cycle latency) instead of DRAM (~400 cycles), breaking
//   the dependent load chain: index_load(smem,30) -> matrix_load(L2,200)
//   instead of index_load(DRAM,400) -> matrix_load(L2,200).
//
// Phase 3: Partial sums reduced across warps via shared memory (reused).
//
// Shared memory: max(n_conn * 4B, nwarps * 32 * sizeof(ACC_T))
#define DEFINE_BGM_BASIC(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, READ_W, WRITE_W, WARP_RED, ACC_ZERO) \
__global__ void _bgm_basic_kern##SUFFIX( \
    const int32_t* __restrict__ indices, \
    const SPIKE_T* __restrict__ matrix, \
    WEIGHT_T*      __restrict__ output, \
    const WEIGHT_T* __restrict__ weights, \
    int n_pre, int n_conn, int n_batch, int is_homo \
) { \
    extern __shared__ char _smem_bytes[]; \
    int32_t* s_idx = reinterpret_cast<int32_t*>(_smem_bytes); \
    int row = blockIdx.x; \
    if (row >= n_pre) return; \
    int lane   = threadIdx.x & 31; \
    int warpid = threadIdx.x >> 5; \
    int nwarps = blockDim.x >> 5; \
    int j = (int)blockIdx.y * 32 + lane; \
    bool col_valid = (j < n_batch); \
    int  safe_j    = col_valid ? j : 0; \
    const int32_t*  i_row = indices + (size_t)row * n_conn; \
    const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)row * n_conn; \
    /* Phase 1: Cooperatively load ALL indices into shared memory */ \
    for (int i = threadIdx.x; i < n_conn; i += blockDim.x) \
        s_idx[i] = __ldg(&i_row[i]); \
    __syncthreads(); \
    /* Phase 2: Each warp processes its share from shared memory.             */ \
    /* Manual 4x unroll: issue 4 matrix loads before processing any,        */ \
    /* quadrupling outstanding L2 reads per warp for better latency hiding. */ \
    ACC_T accum = ACC_ZERO; \
    int step4 = nwarps << 2; \
    int k = warpid; \
    for (; k + 3 * nwarps < n_conn; k += step4) { \
        int  src0   = s_idx[k]; \
        int  src1   = s_idx[k + nwarps]; \
        int  src2   = s_idx[k + 2 * nwarps]; \
        int  src3   = s_idx[k + 3 * nwarps]; \
        SPIKE_T m0  = __ldg(&matrix[(size_t)src0 * n_batch + safe_j]); \
        SPIKE_T m1  = __ldg(&matrix[(size_t)src1 * n_batch + safe_j]); \
        SPIKE_T m2  = __ldg(&matrix[(size_t)src2 * n_batch + safe_j]); \
        SPIKE_T m3  = __ldg(&matrix[(size_t)src3 * n_batch + safe_j]); \
        if (col_valid && IS_ACTIVE(m0)) \
            accum += is_homo ? (ACC_T)1 : READ_W(__ldg(&w_row[k])); \
        if (col_valid && IS_ACTIVE(m1)) \
            accum += is_homo ? (ACC_T)1 : READ_W(__ldg(&w_row[k + nwarps])); \
        if (col_valid && IS_ACTIVE(m2)) \
            accum += is_homo ? (ACC_T)1 : READ_W(__ldg(&w_row[k + 2 * nwarps])); \
        if (col_valid && IS_ACTIVE(m3)) \
            accum += is_homo ? (ACC_T)1 : READ_W(__ldg(&w_row[k + 3 * nwarps])); \
    } \
    /* Handle remaining 1-3 connections with simple loop */ \
    for (; k < n_conn; k += nwarps) { \
        int  src    = s_idx[k]; \
        bool active = col_valid && IS_ACTIVE(__ldg(&matrix[(size_t)src * n_batch + safe_j])); \
        if (active) \
            accum += is_homo ? (ACC_T)1 : READ_W(__ldg(&w_row[k])); \
    } \
    /* Phase 3: Warp reduction (reuse shared memory) */ \
    __syncthreads(); \
    ACC_T* smem_red = reinterpret_cast<ACC_T*>(_smem_bytes); \
    smem_red[warpid * 32 + lane] = accum; \
    __syncthreads(); \
    if (warpid == 0) { \
        ACC_T sum = ACC_ZERO; \
        for (int w = 0; w < nwarps; w++) \
            sum += smem_red[w * 32 + lane]; \
        if (col_valid) \
            output[(size_t)row * n_batch + j] = \
                WRITE_W(is_homo ? (READ_W(__ldg(&weights[0])) * sum) : sum); \
    } \
}

#define DEFINE_BSM_WARP(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, READ_W, ATOMIC_ADD_W) \
__global__ void _bsm_warp_kern##SUFFIX( \
    const int32_t* __restrict__ indices, \
    const SPIKE_T* __restrict__ matrix, \
    WEIGHT_T*      __restrict__ output, \
    const WEIGHT_T* __restrict__ weights, \
    int n_pre, int n_conn, int n_batch, int is_homo \
) { \
    int row = blockIdx.x; \
    int t   = threadIdx.x; \
    int j   = (int)blockIdx.y * 32 + t; \
    if (row >= n_pre) return; \
    bool col_valid = (j < n_batch); \
    int  safe_j    = col_valid ? j : 0; \
    bool active    = col_valid && IS_ACTIVE(__ldg(&matrix[(size_t)row * n_batch + safe_j])); \
    if (__ballot_sync(0xffffffff, active) == 0) return; \
    if (!active) return; \
    const int32_t*  i_row = indices + (size_t)row * n_conn; \
    const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)row * n_conn; \
    float w0 = is_homo ? READ_W(__ldg(&weights[0])) : 0.0f; \
    for (int k = 0; k < n_conn; k++) \
        ATOMIC_ADD_W(&output[(size_t)__ldg(&i_row[k]) * n_batch + j], \
                     is_homo ? w0 : READ_W(__ldg(&w_row[k]))); \
}

#define DEFINE_BSM_BASIC(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, READ_W, ATOMIC_ADD_W) \
__global__ void _bsm_basic_kern##SUFFIX( \
    const int32_t* __restrict__ indices, \
    const SPIKE_T* __restrict__ matrix, \
    WEIGHT_T*      __restrict__ output, \
    const WEIGHT_T* __restrict__ weights, \
    int n_pre, int n_conn, int n_batch, int is_homo \
) { \
    extern __shared__ int _smem_flag[]; \
    int row = blockIdx.x; \
    if (row >= n_pre) return; \
    if (threadIdx.x == 0) _smem_flag[0] = 0; \
    __syncthreads(); \
    for (int j = threadIdx.x; j < n_batch; j += blockDim.x) \
        if (IS_ACTIVE(__ldg(&matrix[(size_t)row * n_batch + j]))) { \
            atomicOr(_smem_flag, 1); break; \
        } \
    __syncthreads(); \
    if (_smem_flag[0] == 0) return; \
    const int32_t*  i_row = indices + (size_t)row * n_conn; \
    const WEIGHT_T* w_row = is_homo ? nullptr : weights + (size_t)row * n_conn; \
    float w0 = is_homo ? READ_W(__ldg(&weights[0])) : 0.0f; \
    for (int j = 0; j < n_batch; j++) { \
        if (!IS_ACTIVE(__ldg(&matrix[(size_t)row * n_batch + j]))) continue; \
        for (int k = threadIdx.x; k < n_conn; k += blockDim.x) \
            ATOMIC_ADD_W(&output[(size_t)__ldg(&i_row[k]) * n_batch + j], \
                         is_homo ? w0 : READ_W(__ldg(&w_row[k]))); \
    } \
}

// SpMM Instantiations
DEFINE_BGM_WARP(_bool_warp_f32,   uint8_t, IS_ACTIVE_BOOL,      float,          float,  READ_F32,  WRITE_F32)
DEFINE_BGM_WARP(_float_warp_f32,  float,   IS_ACTIVE_FLOAT_F32, float,          float,  READ_F32,  WRITE_F32)
DEFINE_BGM_BASIC(_bool_basic_f32,  uint8_t, IS_ACTIVE_BOOL,      float,          float,  READ_F32,  WRITE_F32,  warp_reduce_sum_f32, 0.0f)
DEFINE_BGM_BASIC(_float_basic_f32, float,   IS_ACTIVE_FLOAT_F32, float,          float,  READ_F32,  WRITE_F32,  warp_reduce_sum_f32, 0.0f)
DEFINE_BSM_WARP(_bool_warp_f32,   uint8_t, IS_ACTIVE_BOOL,      float,  READ_F32,  atomic_add_f32)
DEFINE_BSM_WARP(_float_warp_f32,  float,   IS_ACTIVE_FLOAT_F32, float,  READ_F32,  atomic_add_f32)
DEFINE_BSM_BASIC(_bool_basic_f32,  uint8_t, IS_ACTIVE_BOOL,      float,  READ_F32,  atomic_add_f32)
DEFINE_BSM_BASIC(_float_basic_f32, float,   IS_ACTIVE_FLOAT_F32, float,  READ_F32,  atomic_add_f32)
DEFINE_BGM_WARP(_bool_warp_f64,   uint8_t, IS_ACTIVE_BOOL,      double,         double, READ_F64,  WRITE_F64)
DEFINE_BGM_WARP(_float_warp_f64,  double,  IS_ACTIVE_FLOAT_F64, double,         double, READ_F64,  WRITE_F64)
DEFINE_BGM_BASIC(_bool_basic_f64,  uint8_t, IS_ACTIVE_BOOL,      double,         double, READ_F64,  WRITE_F64,  warp_reduce_sum_f64, 0.0)
DEFINE_BGM_BASIC(_float_basic_f64, double,  IS_ACTIVE_FLOAT_F64, double,         double, READ_F64,  WRITE_F64,  warp_reduce_sum_f64, 0.0)
DEFINE_BSM_WARP(_bool_warp_f64,   uint8_t, IS_ACTIVE_BOOL,      double, READ_F64,  atomic_add_f64)
DEFINE_BSM_WARP(_float_warp_f64,  double,  IS_ACTIVE_FLOAT_F64, double, READ_F64,  atomic_add_f64)
DEFINE_BSM_BASIC(_bool_basic_f64,  uint8_t, IS_ACTIVE_BOOL,      double, READ_F64,  atomic_add_f64)
DEFINE_BSM_BASIC(_float_basic_f64, double,  IS_ACTIVE_FLOAT_F64, double, READ_F64,  atomic_add_f64)
DEFINE_BGM_WARP(_bool_warp_f16,   uint8_t, IS_ACTIVE_BOOL,      __half,         float,  READ_F16,  WRITE_F16)
DEFINE_BGM_WARP(_float_warp_f16,  __half,  IS_ACTIVE_FLOAT_F16, __half,         float,  READ_F16,  WRITE_F16)
DEFINE_BGM_BASIC(_bool_basic_f16,  uint8_t, IS_ACTIVE_BOOL,      __half,         float,  READ_F16,  WRITE_F16,  warp_reduce_sum_f32, 0.0f)
DEFINE_BGM_BASIC(_float_basic_f16, __half,  IS_ACTIVE_FLOAT_F16, __half,         float,  READ_F16,  WRITE_F16,  warp_reduce_sum_f32, 0.0f)
DEFINE_BSM_WARP(_bool_warp_f16,   uint8_t, IS_ACTIVE_BOOL,      __half,  READ_F16,  atomic_add_f16)
DEFINE_BSM_WARP(_float_warp_f16,  __half,  IS_ACTIVE_FLOAT_F16, __half,  READ_F16,  atomic_add_f16)
DEFINE_BSM_BASIC(_bool_basic_f16,  uint8_t, IS_ACTIVE_BOOL,      __half,  READ_F16,  atomic_add_f16)
DEFINE_BSM_BASIC(_float_basic_f16, __half,  IS_ACTIVE_FLOAT_F16, __half,  READ_F16,  atomic_add_f16)
DEFINE_BGM_WARP(_bool_warp_bf16,   uint8_t,        IS_ACTIVE_BOOL,       __nv_bfloat16, float, READ_BF16, WRITE_BF16)
DEFINE_BGM_WARP(_float_warp_bf16,  __nv_bfloat16,  IS_ACTIVE_FLOAT_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16)
DEFINE_BGM_BASIC(_bool_basic_bf16,  uint8_t,        IS_ACTIVE_BOOL,       __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BGM_BASIC(_float_basic_bf16, __nv_bfloat16,  IS_ACTIVE_FLOAT_BF16, __nv_bfloat16, float, READ_BF16, WRITE_BF16, warp_reduce_sum_f32, 0.0f)
DEFINE_BSM_WARP(_bool_warp_bf16,   uint8_t,        IS_ACTIVE_BOOL,       __nv_bfloat16,  READ_BF16, atomic_add_bf16)
DEFINE_BSM_WARP(_float_warp_bf16,  __nv_bfloat16,  IS_ACTIVE_FLOAT_BF16, __nv_bfloat16,  READ_BF16, atomic_add_bf16)
DEFINE_BSM_BASIC(_bool_basic_bf16,  uint8_t,        IS_ACTIVE_BOOL,       __nv_bfloat16,  READ_BF16, atomic_add_bf16)
DEFINE_BSM_BASIC(_float_basic_bf16, __nv_bfloat16,  IS_ACTIVE_FLOAT_BF16, __nv_bfloat16,  READ_BF16, atomic_add_bf16)

// FFI Macros for SpMM
#define FFI_BGM_WARP(SUFFIX, WEIGHT_C_T, SPIKE_C_T) \
void binary_fcnmm_gather##SUFFIX( \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices, \
    tvm::ffi::TensorView matrix,  tvm::ffi::TensorView output, int64_t stream \
) { \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream); \
    int n_pre   = static_cast<int>(indices.size(0)); \
    int n_conn  = static_cast<int>(indices.size(1)); \
    int n_batch = static_cast<int>(matrix.size(1)); \
    int is_homo = (weights.ndim() == 1) ? 1 : 0; \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr()); \
    const SPIKE_C_T*  d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr()); \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr()); \
    int batch_tiles = (n_batch + 31) / 32; \
    dim3 grid(n_pre, batch_tiles); \
    _bgm_warp_kern##SUFFIX<<<grid, 32, 0, s>>>( \
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo); \
}

#define FFI_BGM_BASIC(SUFFIX, WEIGHT_C_T, SPIKE_C_T, ACC_SIZE) \
void binary_fcnmm_gather##SUFFIX( \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices, \
    tvm::ffi::TensorView matrix,  tvm::ffi::TensorView output, int64_t stream \
) { \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream); \
    int n_pre   = static_cast<int>(indices.size(0)); \
    int n_conn  = static_cast<int>(indices.size(1)); \
    int n_batch = static_cast<int>(matrix.size(1)); \
    int is_homo = (weights.ndim() == 1) ? 1 : 0; \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr()); \
    const SPIKE_C_T*  d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr()); \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr()); \
    int bsz = 256; \
    int nwarps = bsz >> 5; \
    size_t idx_bytes = (size_t)n_conn * sizeof(int32_t); \
    size_t red_bytes = (size_t)nwarps * 32 * ACC_SIZE; \
    size_t shm = (idx_bytes > red_bytes) ? idx_bytes : red_bytes; \
    int batch_tiles = (n_batch + 31) / 32; \
    dim3 grid(n_pre, batch_tiles); \
    _bgm_basic_kern##SUFFIX<<<grid, bsz, shm, s>>>( \
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo); \
}

#define FFI_BSM_WARP(SUFFIX, WEIGHT_C_T, SPIKE_C_T) \
void binary_fcnmm_scatter##SUFFIX( \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices, \
    tvm::ffi::TensorView matrix,  tvm::ffi::TensorView output, int64_t stream \
) { \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream); \
    int n_pre   = static_cast<int>(indices.size(0)); \
    int n_conn  = static_cast<int>(indices.size(1)); \
    int n_post  = static_cast<int>(output.size(0)); \
    int n_batch = static_cast<int>(matrix.size(1)); \
    int is_homo = (weights.ndim() == 1) ? 1 : 0; \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr()); \
    const SPIKE_C_T*  d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr()); \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr()); \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_batch * sizeof(WEIGHT_C_T), s); \
    int batch_tiles = (n_batch + 31) / 32; \
    dim3 grid(n_pre, batch_tiles); \
    _bsm_warp_kern##SUFFIX<<<grid, 32, 0, s>>>( \
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo); \
}

#define FFI_BSM_BASIC(SUFFIX, WEIGHT_C_T, SPIKE_C_T) \
void binary_fcnmm_scatter##SUFFIX( \
    tvm::ffi::TensorView weights, tvm::ffi::TensorView indices, \
    tvm::ffi::TensorView matrix,  tvm::ffi::TensorView output, int64_t stream \
) { \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream); \
    int n_pre   = static_cast<int>(indices.size(0)); \
    int n_conn  = static_cast<int>(indices.size(1)); \
    int n_post  = static_cast<int>(output.size(0)); \
    int n_batch = static_cast<int>(matrix.size(1)); \
    int is_homo = (weights.ndim() == 1) ? 1 : 0; \
    const WEIGHT_C_T* d_w   = static_cast<const WEIGHT_C_T*>(weights.data_ptr()); \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr()); \
    const SPIKE_C_T*  d_mat = static_cast<const SPIKE_C_T*>(matrix.data_ptr()); \
    WEIGHT_C_T*       d_out = static_cast<WEIGHT_C_T*>(output.data_ptr()); \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * n_batch * sizeof(WEIGHT_C_T), s); \
    size_t shm = sizeof(int); \
    _bsm_basic_kern##SUFFIX<<<n_pre, 256, shm, s>>>( \
        d_idx, d_mat, d_out, d_w, n_pre, n_conn, n_batch, is_homo); \
}

// SpMM FFI Instantiations
// @tvm_ffi binary_fcnmm_gather_bool_warp_f32
FFI_BGM_WARP(_bool_warp_f32,    float,  uint8_t)
// @tvm_ffi binary_fcnmm_gather_bool_basic_f32
FFI_BGM_BASIC(_bool_basic_f32,  float,  uint8_t, sizeof(float))
// @tvm_ffi binary_fcnmm_gather_float_warp_f32
FFI_BGM_WARP(_float_warp_f32,   float,  float)
// @tvm_ffi binary_fcnmm_gather_float_basic_f32
FFI_BGM_BASIC(_float_basic_f32, float,  float, sizeof(float))
// @tvm_ffi binary_fcnmm_scatter_bool_warp_f32
FFI_BSM_WARP(_bool_warp_f32,    float,  uint8_t)
// @tvm_ffi binary_fcnmm_scatter_bool_basic_f32
FFI_BSM_BASIC(_bool_basic_f32,  float,  uint8_t)
// @tvm_ffi binary_fcnmm_scatter_float_warp_f32
FFI_BSM_WARP(_float_warp_f32,   float,  float)
// @tvm_ffi binary_fcnmm_scatter_float_basic_f32
FFI_BSM_BASIC(_float_basic_f32, float,  float)
// @tvm_ffi binary_fcnmm_gather_bool_warp_f64
FFI_BGM_WARP(_bool_warp_f64,    double, uint8_t)
// @tvm_ffi binary_fcnmm_gather_bool_basic_f64
FFI_BGM_BASIC(_bool_basic_f64,  double, uint8_t, sizeof(double))
// @tvm_ffi binary_fcnmm_gather_float_warp_f64
FFI_BGM_WARP(_float_warp_f64,   double, double)
// @tvm_ffi binary_fcnmm_gather_float_basic_f64
FFI_BGM_BASIC(_float_basic_f64, double, double, sizeof(double))
// @tvm_ffi binary_fcnmm_scatter_bool_warp_f64
FFI_BSM_WARP(_bool_warp_f64,    double, uint8_t)
// @tvm_ffi binary_fcnmm_scatter_bool_basic_f64
FFI_BSM_BASIC(_bool_basic_f64,  double, uint8_t)
// @tvm_ffi binary_fcnmm_scatter_float_warp_f64
FFI_BSM_WARP(_float_warp_f64,   double, double)
// @tvm_ffi binary_fcnmm_scatter_float_basic_f64
FFI_BSM_BASIC(_float_basic_f64, double, double)
// @tvm_ffi binary_fcnmm_gather_bool_warp_f16
FFI_BGM_WARP(_bool_warp_f16,    __half, uint8_t)
// @tvm_ffi binary_fcnmm_gather_bool_basic_f16
FFI_BGM_BASIC(_bool_basic_f16,  __half, uint8_t, sizeof(float))
// @tvm_ffi binary_fcnmm_gather_float_warp_f16
FFI_BGM_WARP(_float_warp_f16,   __half, __half)
// @tvm_ffi binary_fcnmm_gather_float_basic_f16
FFI_BGM_BASIC(_float_basic_f16, __half, __half, sizeof(float))
// @tvm_ffi binary_fcnmm_scatter_bool_warp_f16
FFI_BSM_WARP(_bool_warp_f16,    __half, uint8_t)
// @tvm_ffi binary_fcnmm_scatter_bool_basic_f16
FFI_BSM_BASIC(_bool_basic_f16,  __half, uint8_t)
// @tvm_ffi binary_fcnmm_scatter_float_warp_f16
FFI_BSM_WARP(_float_warp_f16,   __half, __half)
// @tvm_ffi binary_fcnmm_scatter_float_basic_f16
FFI_BSM_BASIC(_float_basic_f16, __half, __half)
// @tvm_ffi binary_fcnmm_gather_bool_warp_bf16
FFI_BGM_WARP(_bool_warp_bf16,    __nv_bfloat16, uint8_t)
// @tvm_ffi binary_fcnmm_gather_bool_basic_bf16
FFI_BGM_BASIC(_bool_basic_bf16,  __nv_bfloat16, uint8_t, sizeof(float))
// @tvm_ffi binary_fcnmm_gather_float_warp_bf16
FFI_BGM_WARP(_float_warp_bf16,   __nv_bfloat16, __nv_bfloat16)
// @tvm_ffi binary_fcnmm_gather_float_basic_bf16
FFI_BGM_BASIC(_float_basic_bf16, __nv_bfloat16, __nv_bfloat16, sizeof(float))
// @tvm_ffi binary_fcnmm_scatter_bool_warp_bf16
FFI_BSM_WARP(_bool_warp_bf16,    __nv_bfloat16, uint8_t)
// @tvm_ffi binary_fcnmm_scatter_bool_basic_bf16
FFI_BSM_BASIC(_bool_basic_bf16,  __nv_bfloat16, uint8_t)
// @tvm_ffi binary_fcnmm_scatter_float_warp_bf16
FFI_BSM_WARP(_float_warp_bf16,   __nv_bfloat16, __nv_bfloat16)
// @tvm_ffi binary_fcnmm_scatter_float_basic_bf16
FFI_BSM_BASIC(_float_basic_bf16, __nv_bfloat16, __nv_bfloat16)
