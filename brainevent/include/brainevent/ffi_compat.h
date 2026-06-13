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
/// @file ffi_compat.h
/// @brief Converts XLA FFI buffer types to BE::Tensor.
///
/// INTERNAL HEADER - included only in auto-generated FFI wrappers,
/// never directly by user CUDA code.

#include "xla/ffi/api/ffi.h"
#include "brainevent/tensor.h"
#include "brainevent/dtypes.h"
// cudaStream_t is provided automatically by nvcc for .cu files.
// Include it explicitly here so the generated FFI wrapper compiles when
// the user's source does not include <cuda_runtime.h>.
#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#endif

namespace BE {
namespace internal {

/// Build a Tensor from an XLA FFI input buffer.
///
/// Throws ``std::invalid_argument`` (host only) for a dtype with no BE::DType
/// mapping (f8/f4/sub-byte/token), instead of building a ``DType::Invalid``
/// tensor whose ``element_size()`` is silently 0 (L11).  The generated FFI
/// wrapper catches this and returns an ``xla::ffi::Error``.
inline Tensor buffer_to_tensor(xla::ffi::AnyBuffer buf) {
    void* data = buf.untyped_data();
    auto dims = buf.dimensions();
    int ndim = static_cast<int>(dims.size());
    DType dtype = xla_to_jkb_dtype(buf.element_type());
#ifndef __CUDA_ARCH__
    if (dtype == DType::Invalid) {
        throw std::invalid_argument(
            "BE: unsupported XLA FFI buffer dtype (XLA DataType enum " +
            std::to_string(static_cast<int>(buf.element_type())) +
            "); no BE::DType mapping exists for this type");
    }
#endif
    return Tensor(data, dims.begin(), ndim, dtype);
}

/// Build a Tensor from an XLA FFI output (result) buffer.
inline Tensor result_buffer_to_tensor(
    xla::ffi::Result<xla::ffi::AnyBuffer> res) {
    xla::ffi::AnyBuffer& buf = *res;
    return buffer_to_tensor(buf);
}

}  // namespace internal
}  // namespace BE
