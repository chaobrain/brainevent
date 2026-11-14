// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

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

#include "brainevent/cuda_traits.cuh"

// =========================================================================
// Warp-level Reduction Primitives
// =========================================================================

/*
 * PRECONDITION (all warp_reduce_* helpers below): FULLY CONVERGED WARP.
 * Every helper uses __shfl_down_sync(0xffffffff, ...), i.e. it asserts that
 * ALL 32 lanes of the warp participate.  Calling any of these from a path that
 * a subset of lanes reached -- e.g. after `if (tid >= n) return;` or inside a
 * data-dependent branch -- is undefined behaviour: it may hang or fold in
 * garbage from inactive/retired lanes (M17).
 *
 * Callers with partial warps MUST mask off out-of-range lanes BEFORE the
 * reduction (pad the inactive lanes with the reduction identity -- 0 for sum
 * -- and let all 32 lanes call the helper), rather than
 * early-returning. Computing __activemask() INSIDE the helper is intentionally
 * NOT done: it would silently fold over whatever lanes happen to be active and
 * mask the real bug instead of surfacing it.
 */

/*
 * NOTE ON FP16/BF16 REDUCTIONS:
 * Reduction helpers are defined for accumulator types (f32/f64), not storage
 * types. fp16/bf16 kernels upcast with READ_F16/READ_BF16 and accumulate in
 * float32, so they intentionally call the f32 reductions
 * (warp_reduce_sum_f32).
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
 * if it is non-zero (negative values and NaN are active).
 *
 * @param s  Spike value (float)
 * @return   1 if active, 0 otherwise
 */
#define IS_ACTIVE_FLOAT(s) ((s) != 0.0f)

/**
 * Check if a float32 spike is active (explicit).
 *
 * Alias for IS_ACTIVE_FLOAT for consistency with dtype naming.
 *
 * @param s  Spike value (float)
 * @return   1 if active, 0 otherwise
 */
#define IS_ACTIVE_F32(s) ((s) != 0.0f)

/**
 * Check if a float64 spike is active.
 *
 * For event-driven computation, a double spike is considered active
 * if it is non-zero (negative values and NaN are active).
 *
 * @param s  Spike value (double)
 * @return   1 if active, 0 otherwise
 */
#define IS_ACTIVE_F64(s) ((s) != 0.0)

/**
 * Check if a float16 spike is active.
 *
 * For event-driven computation, a half spike is considered active
 * if it is non-zero (negative values and NaN are active). Converts to float for comparison.
 *
 * @param s  Spike value (__half)
 * @return   1 if active, 0 otherwise
 */
#define IS_ACTIVE_F16(s) (__half2float(s) != 0.0f)

/**
 * Check if a bfloat16 spike is active.
 *
 * For event-driven computation, a bfloat16 spike is considered active
 * if it is non-zero (negative values and NaN are active). Converts to float for comparison.
 *
 * @param s  Spike value (__nv_bfloat16)
 * @return   1 if active, 0 otherwise
 */
#define IS_ACTIVE_BF16(s) (__bfloat162float(s) != 0.0f)

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
// Zero Constants
// =========================================================================

#define ZERO_F32  0.0f
#define ZERO_F64  0.0
#define ZERO_F16  0.0f  // accumulator is float32
#define ZERO_BF16 0.0f  // accumulator is float32

// =========================================================================
// Launch Geometry — one warp per row (SpMV, row slicing, dt2t)
// =========================================================================

/**
 * Warps per block for kernels that map one row to one warp.
 *
 * 8 warps == 256 threads, so one block retires 8 rows per sweep. Measured
 * 2-3x faster than the one-warp-per-block (`<<<m, 32>>>`) shape at low average
 * row length, and never slower.
 */
#define BE_WARP_PER_ROW_WPB 8

/** Upper bound on the warp-tier grid; the kernels are grid-strided. */
#define BE_WARP_PER_ROW_MAX_GRID 4096

/**
 * Grid size covering @p m rows at ::BE_WARP_PER_ROW_WPB rows per
 * block, capped at ::BE_WARP_PER_ROW_MAX_GRID so the grid tracks the resident
 * block count instead of scaling with the row count.
 *
 * Kernels launched with this must be grid-strided, since the cap can make the
 * grid smaller than the row count. Never returns 0: an empty matrix still gets
 * one block (which exits immediately), because a zero-sized grid is an invalid
 * launch configuration.
 *
 * @param m  Number of rows.
 * @return   Number of blocks to launch, in [1, ::BE_WARP_PER_ROW_MAX_GRID].
 */
__host__ __inline__ int be_warp_per_row_grid(int m) {
    int blocks = (m + BE_WARP_PER_ROW_WPB - 1) / BE_WARP_PER_ROW_WPB;
    if (blocks > BE_WARP_PER_ROW_MAX_GRID) blocks = BE_WARP_PER_ROW_MAX_GRID;
    return blocks > 0 ? blocks : 1;
}

/** Convenience spelling of ::be_warp_per_row_grid for use inside launch macros. */
#define BE_WARP_PER_ROW_GRID(m) be_warp_per_row_grid(m)

// =========================================================================
// Launch Geometry — warp-per-row SpMM
// =========================================================================
//
// SpMM warp kernels map lanes to *columns* (32 columns per warp) and warps to
// rows, so the row tiling is a separate knob from the CSRMV one above.

/** Maximum grid.x for SpMM warp kernels; 4096 x 128 threads saturates all SMs. */
#define CSRMM_MAX_GRID_X 4096

/** Rows per block for SpMM warp kernels (4 warps x 32 lanes = 128 threads). */
#define CSRMM_WARP_RPB 4

// =========================================================================
// Atomic Add Helpers (with CUDA arch guards)
// =========================================================================

// atomic_add_{f32,f64,f16,bf16} (including the pre-sm_70/sm_80 CAS emulation
// and its M16 alignment precondition) are defined once in cuda_traits.cuh and
// re-exported here for kernels that call them unqualified.
#ifdef __CUDACC__
using cuda_traits::atomic_add_f32;
using cuda_traits::atomic_add_f64;
using cuda_traits::atomic_add_f16;
using cuda_traits::atomic_add_bf16;
#endif

#endif  // BRAINEVENT_CUDA_COMMON_H_
