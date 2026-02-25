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
/// @file dtypes.h
/// @brief XLA DataType ↔ JKB::DType conversion and dispatch macros.
///
/// INTERNAL HEADER — included only in auto-generated FFI wrappers,
/// never directly by user CUDA code (it pulls in XLA FFI headers).

#include "jkb/tensor.h"
#include "xla/ffi/api/ffi.h"

namespace JKB {

/// Convert an XLA FFI DataType to JKB::DType.
inline DType xla_to_jkb_dtype(xla::ffi::DataType dt) noexcept {
    using D = xla::ffi::DataType;
    switch (dt) {
        case D::F16:  return DType::Float16;
        case D::F32:  return DType::Float32;
        case D::F64:  return DType::Float64;
        case D::BF16: return DType::BFloat16;
        case D::S8:   return DType::Int8;
        case D::S16:  return DType::Int16;
        case D::S32:  return DType::Int32;
        case D::S64:  return DType::Int64;
        case D::U8:   return DType::UInt8;
        case D::U16:  return DType::UInt16;
        case D::U32:  return DType::UInt32;
        case D::U64:  return DType::UInt64;
        case D::PRED: return DType::Bool;
        case D::C64:  return DType::Complex64;
        case D::C128: return DType::Complex128;
        default:      return DType::Invalid;
    }
}

// Dispatch macros have moved to the user-facing header jkb/dispatch.h
// (included via jkb/common.h).  This internal header only provides
// the XLA DataType ↔ JKB::DType conversion.

}  // namespace JKB
