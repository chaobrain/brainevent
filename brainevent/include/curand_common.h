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
 * curand_common.h -- Common CURAND Helpers for BrainEvent Random Number Generation
 * =================================================================================
 *
 * This header provides shared utilities for random number generation in CUDA kernels
 * across the BrainEvent library:
 *
 * - Normal distribution RNG helpers (float32, float64)
 * - Uniform distribution RNG helpers (float32, float64)
 *
 * Usage:
 *   #include "../curand_common.h"  // from submodule (e.g., _jit_normal/, _jit_uniform/)
 *   #include "curand_common.h"     // from brainevent/ root
 *
 * All functions are designed for use in CUDA device code with curandStatePhilox4_32_10_t.
 */

#ifndef BRAINEVENT_CURAND_COMMON_H_
#define BRAINEVENT_CURAND_COMMON_H_

#include <curand_kernel.h>

// =========================================================================
// Normal Distribution RNG Helpers
// =========================================================================

/**
 * Generate a normally distributed float32 random number.
 *
 * Wrapper around curand_normal for consistent API across dtypes.
 *
 * @param state  Pointer to curandStatePhilox4_32_10_t state
 * @return       Normally distributed float32 value (mean=0, stddev=1)
 */
__device__ __inline__ float curand_normal_f32(curandStatePhilox4_32_10_t* state) {
    return curand_normal(state);
}

/**
 * Generate a normally distributed float64 random number.
 *
 * Wrapper around curand_normal_double for consistent API across dtypes.
 *
 * @param state  Pointer to curandStatePhilox4_32_10_t state
 * @return       Normally distributed float64 value (mean=0, stddev=1)
 */
__device__ __inline__ double curand_normal_f64(curandStatePhilox4_32_10_t* state) {
    return curand_normal_double(state);
}

// =========================================================================
// Uniform Distribution RNG Helpers
// =========================================================================

/**
 * Generate a uniformly distributed float32 random number.
 *
 * Wrapper around curand_uniform for consistent API across dtypes.
 *
 * @param state  Pointer to curandStatePhilox4_32_10_t state
 * @return       Uniformly distributed float32 value in (0, 1]
 */
__device__ __inline__ float curand_uniform_f32(curandStatePhilox4_32_10_t* state) {
    return curand_uniform(state);
}

/**
 * Generate a uniformly distributed float64 random number.
 *
 * Wrapper around curand_uniform_double for consistent API across dtypes.
 *
 * @param state  Pointer to curandStatePhilox4_32_10_t state
 * @return       Uniformly distributed float64 value in (0, 1]
 */
__device__ __inline__ double curand_uniform_f64(curandStatePhilox4_32_10_t* state) {
    return curand_uniform_double(state);
}

#endif  // BRAINEVENT_CURAND_COMMON_H_
