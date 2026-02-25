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
/// @brief Converts XLA FFI buffer types to JKB::Tensor.
///
/// INTERNAL HEADER — included only in auto-generated FFI wrappers,
/// never directly by user CUDA code.

#include "xla/ffi/api/ffi.h"
#include "jkb/tensor.h"
#include "jkb/dtypes.h"
// cudaStream_t is provided automatically by nvcc for .cu files.
// Include it explicitly here so the generated FFI wrapper compiles when
// the user's source does not include <cuda_runtime.h>.
#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#endif

namespace JKB {
namespace internal {

/// Build a Tensor from an XLA FFI input buffer.
inline Tensor buffer_to_tensor(xla::ffi::AnyBuffer buf) {
    void* data = buf.untyped_data();
    auto dims = buf.dimensions();
    int ndim = static_cast<int>(dims.size());
    DType dtype = xla_to_jkb_dtype(buf.element_type());
    return Tensor(data, dims.begin(), ndim, dtype);
}

/// Build a Tensor from an XLA FFI output (result) buffer.
inline Tensor result_buffer_to_tensor(
    xla::ffi::Result<xla::ffi::AnyBuffer> res) {
    xla::ffi::AnyBuffer& buf = *res;
    return buffer_to_tensor(buf);
}

}  // namespace internal
}  // namespace JKB
