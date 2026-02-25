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
 * cuda_common.h -- Common CUDA Helpers for BrainEvent Sparse Operations
 * ======================================================================
 *
 * This header provides shared utilities for all sparse matrix operations
 * across the BrainEvent library:
 *
 * - Warp-level reduction primitives (sum, max, min)
 * - Active-check predicates for event-driven computation
 * - Per-dtype conversion macros for multi-precision support (fp16, bf16, fp32, fp64)
 *
 * Usage:
 *   #include "../cuda_common.h"  // from submodule (e.g., _csr/, _coo/)
 *   #include "cuda_common.h"     // from brainevent/ root
 *
 * All functions and macros are designed for use in CUDA device code.
 */

#ifndef BRAINEVENT_CUDA_COMMON_H_
#define BRAINEVENT_CUDA_COMMON_H_

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// =========================================================================
// Warp-level Reduction Primitives
// =========================================================================

/*
 * NOTE ON FP16/BF16 REDUCTIONS:
 * Reduction helpers are defined for accumulator types (f32/f64), not storage
 * types. fp16/bf16 kernels upcast with READ_F16/READ_BF16 and use float
 * accumulators (ACC_T_F16/ACC_T_BF16), so they intentionally call f32
 * reductions (warp_reduce_*_f32).
 */

/**
 * Warp-level sum reduction for float32.
 *
 * Reduces a value across all 32 threads in a warp using shuffle-down
 * instructions. The result is valid only in lane 0.
 *
 * Algorithm: Tree reduction with log2(32) = 5 steps
 *   Step 1: lanes [0..15] += lanes [16..31]
 *   Step 2: lanes [0..7]  += lanes [8..15]
 *   Step 3: lanes [0..3]  += lanes [4..7]
 *   Step 4: lanes [0..1]  += lanes [2..3]
 *   Step 5: lane  0       += lane  1
 *
 * @param val  Input value from this thread
 * @return     Sum of all values across the warp (valid in lane 0 only)
 */
__device__ __inline__ float warp_reduce_sum_f32(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

/**
 * Warp-level sum reduction for float64.
 *
 * Reduces a value across all 32 threads in a warp using shuffle-down
 * instructions. The result is valid only in lane 0.
 *
 * @param val  Input value from this thread
 * @return     Sum of all values across the warp (valid in lane 0 only)
 */
__device__ __inline__ double warp_reduce_sum_f64(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

/**
 * Warp-level max reduction for float32.
 *
 * Reduces a value across all 32 threads in a warp using shuffle-down
 * instructions. The result is valid only in lane 0.
 *
 * @param val  Input value from this thread
 * @return     Maximum of all values across the warp (valid in lane 0 only)
 */
__device__ __inline__ float warp_reduce_max_f32(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

/**
 * Warp-level max reduction for float64.
 *
 * Reduces a value across all 32 threads in a warp using shuffle-down
 * instructions. The result is valid only in lane 0.
 *
 * @param val  Input value from this thread
 * @return     Maximum of all values across the warp (valid in lane 0 only)
 */
__device__ __inline__ double warp_reduce_max_f64(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

/**
 * Warp-level min reduction for float32.
 *
 * Reduces a value across all 32 threads in a warp using shuffle-down
 * instructions. The result is valid only in lane 0.
 *
 * @param val  Input value from this thread
 * @return     Minimum of all values across the warp (valid in lane 0 only)
 */
__device__ __inline__ float warp_reduce_min_f32(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

/**
 * Warp-level min reduction for float64.
 *
 * Reduces a value across all 32 threads in a warp using shuffle-down
 * instructions. The result is valid only in lane 0.
 *
 * @param val  Input value from this thread
 * @return     Minimum of all values across the warp (valid in lane 0 only)
 */
__device__ __inline__ double warp_reduce_min_f64(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmin(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// =========================================================================
// Active-Check Predicates
// =========================================================================

/**
 * Check if a boolean spike is active.
 *
 * For event-driven computation, a boolean spike is considered active
 * if it is non-zero (true).
 *
 * @param s  Spike value (int8_t representing bool)
 * @return   1 if active, 0 otherwise
 */
#define IS_ACTIVE_BOOL(s)  ((s) != 0)

/**
 * Check if a float32 spike is active.
 *
 * For event-driven computation, a float spike is considered active
 * if it is strictly positive.
 *
 * @param s  Spike value (float)
 * @return   1 if active, 0 otherwise
 */
#define IS_ACTIVE_FLOAT(s) ((s) > 0.0f)

/**
 * Check if a float32 spike is active (explicit).
 *
 * Alias for IS_ACTIVE_FLOAT for consistency with dtype naming.
 *
 * @param s  Spike value (float)
 * @return   1 if active, 0 otherwise
 */
#define IS_ACTIVE_F32(s) ((s) > 0.0f)

/**
 * Check if a float64 spike is active.
 *
 * For event-driven computation, a double spike is considered active
 * if it is strictly positive.
 *
 * @param s  Spike value (double)
 * @return   1 if active, 0 otherwise
 */
#define IS_ACTIVE_F64(s) ((s) > 0.0)

/**
 * Check if a float16 spike is active.
 *
 * For event-driven computation, a half spike is considered active
 * if it is strictly positive. Converts to float for comparison.
 *
 * @param s  Spike value (__half)
 * @return   1 if active, 0 otherwise
 */
#define IS_ACTIVE_F16(s) (__half2float(s) > 0.0f)

/**
 * Check if a bfloat16 spike is active.
 *
 * For event-driven computation, a bfloat16 spike is considered active
 * if it is strictly positive. Converts to float for comparison.
 *
 * @param s  Spike value (__nv_bfloat16)
 * @return   1 if active, 0 otherwise
 */
#define IS_ACTIVE_BF16(s) (__bfloat162float(s) > 0.0f)

// Aliases with explicit FLOAT_ prefix for dtype-parameterized kernels
#define IS_ACTIVE_FLOAT_F32(s)  IS_ACTIVE_F32(s)
#define IS_ACTIVE_FLOAT_F64(s)  IS_ACTIVE_F64(s)
#define IS_ACTIVE_FLOAT_F16(s)  IS_ACTIVE_F16(s)
#define IS_ACTIVE_FLOAT_BF16(s) IS_ACTIVE_BF16(s)

// =========================================================================
// Per-Dtype Conversion Macros
// =========================================================================

/**
 * float32: identity conversions
 *
 * No conversion needed - float32 is the native accumulator type.
 */
#define READ_F32(x)   (x)
#define WRITE_F32(x)  (x)

/**
 * float64: identity conversions
 *
 * No conversion needed - float64 accumulates natively.
 */
#define READ_F64(x)   (x)
#define WRITE_F64(x)  (x)

/**
 * float16: convert to/from float32 for computation
 *
 * float16 (__half) is converted to float32 for accumulation to maintain
 * numerical stability. Results are converted back to float16 for storage.
 * Consequently, fp16 paths use float warp reductions instead of a dedicated
 * warp_reduce_*_f16 implementation.
 */
#define READ_F16(x)   __half2float(x)
#define WRITE_F16(x)  __float2half(x)

/**
 * bfloat16: convert to/from float32 for computation
 *
 * bfloat16 (__nv_bfloat16) is converted to float32 for accumulation to
 * maintain numerical stability. Results are converted back to bfloat16
 * for storage. Consequently, bf16 paths use float warp reductions instead of
 * a dedicated warp_reduce_*_bf16 implementation.
 */
#define READ_BF16(x)  __bfloat162float(x)
#define WRITE_BF16(x) __float2bfloat16(x)

// =========================================================================
// Accumulator Type Selection
// =========================================================================

/**
 * Accumulator type for float16 weights.
 *
 * float16 accumulates in float32 for numerical stability.
 */
#define ACC_T_F16  float

/**
 * Accumulator type for bfloat16 weights.
 *
 * bfloat16 accumulates in float32 for numerical stability.
 */
#define ACC_T_BF16 float

/**
 * Accumulator type for float32 weights.
 *
 * float32 accumulates natively.
 */
#define ACC_T_F32  float

/**
 * Accumulator type for float64 weights.
 *
 * float64 accumulates natively.
 */
#define ACC_T_F64  double

// =========================================================================
// Zero Constants
// =========================================================================

#define ZERO_F32  0.0f
#define ZERO_F64  0.0
#define ZERO_F16  0.0f  // accumulator is float32
#define ZERO_BF16 0.0f  // accumulator is float32

// =========================================================================
// Atomic Add Helpers (with CUDA arch guards)
// =========================================================================

/**
 * Atomic add for float32.
 *
 * Native atomic operation supported on all CUDA architectures.
 *
 * @param addr  Pointer to memory location
 * @param val   Value to add
 */
__device__ __inline__ void atomic_add_f32(float* addr, float val) {
    atomicAdd(addr, val);
}

/**
 * Atomic add for float64.
 *
 * Native atomic operation supported on all CUDA architectures.
 *
 * @param addr  Pointer to memory location
 * @param val   Value to add
 */
__device__ __inline__ void atomic_add_f64(double* addr, double val) {
    atomicAdd(addr, val);
}

/**
 * Atomic add for float16.
 *
 * Uses native atomicAdd on sm_70+ (Volta and newer).
 * Falls back to CAS-based emulation on older architectures.
 *
 * @param addr  Pointer to memory location
 * @param val   Value to add (float32, will be converted to float16)
 */
__device__ __inline__ void atomic_add_f16(__half* addr, float val) {
#if __CUDA_ARCH__ >= 700
    atomicAdd(addr, __float2half(val));
#else
    // Emulate with CAS on older architectures
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

/**
 * Atomic add for bfloat16.
 *
 * Uses native atomicAdd on sm_80+ (Ampere and newer).
 * Falls back to CAS-based emulation on older architectures.
 *
 * @param addr  Pointer to memory location
 * @param val   Value to add (float32, will be converted to bfloat16)
 */
__device__ __inline__ void atomic_add_bf16(__nv_bfloat16* addr, float val) {
#if __CUDA_ARCH__ >= 800
    atomicAdd(addr, __float2bfloat16(val));
#else
    // Emulate with CAS on older architectures
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

#endif  // BRAINEVENT_CUDA_COMMON_H_
