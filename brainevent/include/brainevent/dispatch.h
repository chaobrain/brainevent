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

#pragma once
/// @file dispatch.h
/// @brief C++ type aliases for all JKB dtypes and runtime dispatch macros.
///
/// Include via ``jkb/common.h`` — this header is user-facing.

#include "brainevent/tensor.h"
#include <cstdio>

// ── Half-precision type aliases (CUDA only) ─────────────────────────────

#if defined(__CUDACC__) || defined(__CUDA__)
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

namespace BE {

// -- Type aliases ---------------------------------------------------------

#if defined(__CUDACC__) || defined(__CUDA__)
/// IEEE 754 half-precision float (CUDA ``__half``).
using float16_t  = __half;
/// Google Brain bfloat16 (CUDA ``__nv_bfloat16``).
using bfloat16_t = __nv_bfloat16;
#else
/// Opaque 16-bit storage for float16 on CPU.
/// Cast to a proper half type before arithmetic.
struct float16_t  { uint16_t bits; };
/// Opaque 16-bit storage for bfloat16 on CPU.
struct bfloat16_t { uint16_t bits; };
#endif

/// Complex number stored as a pair of float32.
struct complex64_t {
    float real;
    float imag;
};

/// Complex number stored as a pair of float64.
struct complex128_t {
    double real;
    double imag;
};

}  // namespace BE

// ── Dispatch macros ─────────────────────────────────────────────────────
//
// Usage:
//   BE_DISPATCH_FLOATING(tensor.dtype(), scalar_t, {
//       my_kernel<scalar_t><<<...>>>(
//           static_cast<const scalar_t*>(tensor.data_ptr()), ...);
//   });
//
// Each macro creates an immediately-invoked lambda with a switch
// statement.  The ``SCALAR_VAR`` name becomes a ``using`` alias
// inside the body block.

/// Dispatch over common floating-point dtypes (float32, float64).
#define BE_DISPATCH_FLOATING(DTYPE, SCALAR_VAR, ...)                    \
    [&] {                                                                \
        switch (DTYPE) {                                                 \
            case BE::DType::Float32: {                                  \
                using SCALAR_VAR = float;                                \
                __VA_ARGS__                                              \
                break;                                                   \
            }                                                            \
            case BE::DType::Float64: {                                  \
                using SCALAR_VAR = double;                               \
                __VA_ARGS__                                              \
                break;                                                   \
            }                                                            \
            default:                                                     \
                fprintf(stderr,                                          \
                    "[be] BE_DISPATCH_FLOATING: unsupported dtype %s\n", \
                    BE::dtype_name(DTYPE));                             \
                abort();                                                 \
        }                                                                \
    }()

/// Dispatch over all floating-point dtypes including half-precision.
/// float16 and bfloat16 branches are only available in CUDA compilation.
#if defined(__CUDACC__) || defined(__CUDA__)
#define BE_DISPATCH_FLOATING_AND_HALF(DTYPE, SCALAR_VAR, ...)           \
    [&] {                                                                \
        switch (DTYPE) {                                                 \
            case BE::DType::Float16: {                                  \
                using SCALAR_VAR = BE::float16_t;                       \
                __VA_ARGS__                                              \
                break;                                                   \
            }                                                            \
            case BE::DType::BFloat16: {                                 \
                using SCALAR_VAR = BE::bfloat16_t;                      \
                __VA_ARGS__                                              \
                break;                                                   \
            }                                                            \
            case BE::DType::Float32: {                                  \
                using SCALAR_VAR = float;                                \
                __VA_ARGS__                                              \
                break;                                                   \
            }                                                            \
            case BE::DType::Float64: {                                  \
                using SCALAR_VAR = double;                               \
                __VA_ARGS__                                              \
                break;                                                   \
            }                                                            \
            default:                                                     \
                fprintf(stderr,                                          \
                    "[be] BE_DISPATCH_FLOATING_AND_HALF: unsupported dtype %s\n", \
                    BE::dtype_name(DTYPE));                             \
                abort();                                                 \
        }                                                                \
    }()
#else
#define BE_DISPATCH_FLOATING_AND_HALF BE_DISPATCH_FLOATING
#endif

/// Dispatch over integer dtypes (signed, unsigned, and bool).
#define BE_DISPATCH_INTEGRAL(DTYPE, SCALAR_VAR, ...)                    \
    [&] {                                                                \
        switch (DTYPE) {                                                 \
            case BE::DType::Bool:   { using SCALAR_VAR = bool;     __VA_ARGS__ break; } \
            case BE::DType::Int8:   { using SCALAR_VAR = int8_t;   __VA_ARGS__ break; } \
            case BE::DType::Int16:  { using SCALAR_VAR = int16_t;  __VA_ARGS__ break; } \
            case BE::DType::Int32:  { using SCALAR_VAR = int32_t;  __VA_ARGS__ break; } \
            case BE::DType::Int64:  { using SCALAR_VAR = int64_t;  __VA_ARGS__ break; } \
            case BE::DType::UInt8:  { using SCALAR_VAR = uint8_t;  __VA_ARGS__ break; } \
            case BE::DType::UInt16: { using SCALAR_VAR = uint16_t; __VA_ARGS__ break; } \
            case BE::DType::UInt32: { using SCALAR_VAR = uint32_t; __VA_ARGS__ break; } \
            case BE::DType::UInt64: { using SCALAR_VAR = uint64_t; __VA_ARGS__ break; } \
            default:                                                     \
                fprintf(stderr,                                          \
                    "[be] BE_DISPATCH_INTEGRAL: unsupported dtype %s\n", \
                    BE::dtype_name(DTYPE));                             \
                abort();                                                 \
        }                                                                \
    }()

/// Dispatch over complex dtypes.
#define BE_DISPATCH_COMPLEX(DTYPE, SCALAR_VAR, ...)                     \
    [&] {                                                                \
        switch (DTYPE) {                                                 \
            case BE::DType::Complex64: {                                \
                using SCALAR_VAR = BE::complex64_t;                     \
                __VA_ARGS__                                              \
                break;                                                   \
            }                                                            \
            case BE::DType::Complex128: {                               \
                using SCALAR_VAR = BE::complex128_t;                    \
                __VA_ARGS__                                              \
                break;                                                   \
            }                                                            \
            default:                                                     \
                fprintf(stderr,                                          \
                    "[be] BE_DISPATCH_COMPLEX: unsupported dtype %s\n", \
                    BE::dtype_name(DTYPE));                             \
                abort();                                                 \
        }                                                                \
    }()

/// Dispatch over floating-point and integer dtypes (no complex).
#define BE_DISPATCH_ALL_TYPES(DTYPE, SCALAR_VAR, ...)                   \
    [&] {                                                                \
        switch (DTYPE) {                                                 \
            case BE::DType::Float32:  { using SCALAR_VAR = float;    __VA_ARGS__ break; } \
            case BE::DType::Float64:  { using SCALAR_VAR = double;   __VA_ARGS__ break; } \
            case BE::DType::Bool:     { using SCALAR_VAR = bool;     __VA_ARGS__ break; } \
            case BE::DType::Int8:     { using SCALAR_VAR = int8_t;   __VA_ARGS__ break; } \
            case BE::DType::Int16:    { using SCALAR_VAR = int16_t;  __VA_ARGS__ break; } \
            case BE::DType::Int32:    { using SCALAR_VAR = int32_t;  __VA_ARGS__ break; } \
            case BE::DType::Int64:    { using SCALAR_VAR = int64_t;  __VA_ARGS__ break; } \
            case BE::DType::UInt8:    { using SCALAR_VAR = uint8_t;  __VA_ARGS__ break; } \
            case BE::DType::UInt16:   { using SCALAR_VAR = uint16_t; __VA_ARGS__ break; } \
            case BE::DType::UInt32:   { using SCALAR_VAR = uint32_t; __VA_ARGS__ break; } \
            case BE::DType::UInt64:   { using SCALAR_VAR = uint64_t; __VA_ARGS__ break; } \
            default:                                                     \
                fprintf(stderr,                                          \
                    "[be] BE_DISPATCH_ALL_TYPES: unsupported dtype %s\n", \
                    BE::dtype_name(DTYPE));                             \
                abort();                                                 \
        }                                                                \
    }()

/// Dispatch over every supported dtype including half-precision and complex.
#if defined(__CUDACC__) || defined(__CUDA__)
#define BE_DISPATCH_ALL(DTYPE, SCALAR_VAR, ...)                         \
    [&] {                                                                \
        switch (DTYPE) {                                                 \
            case BE::DType::Float16:    { using SCALAR_VAR = BE::float16_t;    __VA_ARGS__ break; } \
            case BE::DType::BFloat16:   { using SCALAR_VAR = BE::bfloat16_t;   __VA_ARGS__ break; } \
            case BE::DType::Float32:    { using SCALAR_VAR = float;              __VA_ARGS__ break; } \
            case BE::DType::Float64:    { using SCALAR_VAR = double;             __VA_ARGS__ break; } \
            case BE::DType::Bool:       { using SCALAR_VAR = bool;               __VA_ARGS__ break; } \
            case BE::DType::Int8:       { using SCALAR_VAR = int8_t;             __VA_ARGS__ break; } \
            case BE::DType::Int16:      { using SCALAR_VAR = int16_t;            __VA_ARGS__ break; } \
            case BE::DType::Int32:      { using SCALAR_VAR = int32_t;            __VA_ARGS__ break; } \
            case BE::DType::Int64:      { using SCALAR_VAR = int64_t;            __VA_ARGS__ break; } \
            case BE::DType::UInt8:      { using SCALAR_VAR = uint8_t;            __VA_ARGS__ break; } \
            case BE::DType::UInt16:     { using SCALAR_VAR = uint16_t;           __VA_ARGS__ break; } \
            case BE::DType::UInt32:     { using SCALAR_VAR = uint32_t;           __VA_ARGS__ break; } \
            case BE::DType::UInt64:     { using SCALAR_VAR = uint64_t;           __VA_ARGS__ break; } \
            case BE::DType::Complex64:  { using SCALAR_VAR = BE::complex64_t;  __VA_ARGS__ break; } \
            case BE::DType::Complex128: { using SCALAR_VAR = BE::complex128_t; __VA_ARGS__ break; } \
            default:                                                     \
                fprintf(stderr,                                          \
                    "[be] BE_DISPATCH_ALL: unsupported dtype %s\n",    \
                    BE::dtype_name(DTYPE));                             \
                abort();                                                 \
        }                                                                \
    }()
#else
// CPU: half types use opaque storage; complex types are always available.
#define BE_DISPATCH_ALL(DTYPE, SCALAR_VAR, ...)                         \
    [&] {                                                                \
        switch (DTYPE) {                                                 \
            case BE::DType::Float16:    { using SCALAR_VAR = BE::float16_t;    __VA_ARGS__ break; } \
            case BE::DType::BFloat16:   { using SCALAR_VAR = BE::bfloat16_t;   __VA_ARGS__ break; } \
            case BE::DType::Float32:    { using SCALAR_VAR = float;              __VA_ARGS__ break; } \
            case BE::DType::Float64:    { using SCALAR_VAR = double;             __VA_ARGS__ break; } \
            case BE::DType::Bool:       { using SCALAR_VAR = bool;               __VA_ARGS__ break; } \
            case BE::DType::Int8:       { using SCALAR_VAR = int8_t;             __VA_ARGS__ break; } \
            case BE::DType::Int16:      { using SCALAR_VAR = int16_t;            __VA_ARGS__ break; } \
            case BE::DType::Int32:      { using SCALAR_VAR = int32_t;            __VA_ARGS__ break; } \
            case BE::DType::Int64:      { using SCALAR_VAR = int64_t;            __VA_ARGS__ break; } \
            case BE::DType::UInt8:      { using SCALAR_VAR = uint8_t;            __VA_ARGS__ break; } \
            case BE::DType::UInt16:     { using SCALAR_VAR = uint16_t;           __VA_ARGS__ break; } \
            case BE::DType::UInt32:     { using SCALAR_VAR = uint32_t;           __VA_ARGS__ break; } \
            case BE::DType::UInt64:     { using SCALAR_VAR = uint64_t;           __VA_ARGS__ break; } \
            case BE::DType::Complex64:  { using SCALAR_VAR = BE::complex64_t;  __VA_ARGS__ break; } \
            case BE::DType::Complex128: { using SCALAR_VAR = BE::complex128_t; __VA_ARGS__ break; } \
            default:                                                     \
                fprintf(stderr,                                          \
                    "[be] BE_DISPATCH_ALL: unsupported dtype %s\n",    \
                    BE::dtype_name(DTYPE));                             \
                abort();                                                 \
        }                                                                \
    }()
#endif
