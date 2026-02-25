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
/// @file tensor.h
/// @brief JKB::Tensor — lightweight, self-contained tensor descriptor.
///
/// Tensor does NOT own the data it points to. It stores shape and
/// C-contiguous strides internally (up to kMaxDim dimensions) so that a
/// single value copy carries all the metadata a CUDA kernel needs.

#include <cstdint>
#include <cstddef>

namespace JKB {

/// Maximum number of dimensions supported.
static constexpr int kMaxDim = 8;

/// Data-type tag mirroring JAX / NumPy dtypes.
/// Numeric values are arbitrary; the mapping to XLA DataType lives in
/// dtypes.h (internal header, never included by user code).
enum class DType : uint8_t {
    Float16    = 0,
    Float32    = 1,
    Float64    = 2,
    BFloat16   = 3,
    Int8       = 4,
    Int16      = 5,
    Int32      = 6,
    Int64      = 7,
    UInt8      = 8,
    UInt16     = 9,
    UInt32     = 10,
    UInt64     = 11,
    Bool       = 12,
    Complex64  = 13,
    Complex128 = 14,
    Invalid    = 255,
};

/// Byte width of a single element for each DType.
inline size_t dtype_size(DType dt) noexcept {
    switch (dt) {
        case DType::Bool:       return 1;
        case DType::Int8:       return 1;
        case DType::UInt8:      return 1;
        case DType::Int16:      return 2;
        case DType::UInt16:     return 2;
        case DType::Float16:    return 2;
        case DType::BFloat16:   return 2;
        case DType::Int32:      return 4;
        case DType::UInt32:     return 4;
        case DType::Float32:    return 4;
        case DType::Int64:      return 8;
        case DType::UInt64:     return 8;
        case DType::Float64:    return 8;
        case DType::Complex64:  return 8;
        case DType::Complex128: return 16;
        default:                return 0;
    }
}

/// Human-readable dtype name (for error messages).
inline const char* dtype_name(DType dt) noexcept {
    switch (dt) {
        case DType::Float16:    return "float16";
        case DType::Float32:    return "float32";
        case DType::Float64:    return "float64";
        case DType::BFloat16:   return "bfloat16";
        case DType::Int8:       return "int8";
        case DType::Int16:      return "int16";
        case DType::Int32:      return "int32";
        case DType::Int64:      return "int64";
        case DType::UInt8:      return "uint8";
        case DType::UInt16:     return "uint16";
        case DType::UInt32:     return "uint32";
        case DType::UInt64:     return "uint64";
        case DType::Bool:       return "bool";
        case DType::Complex64:  return "complex64";
        case DType::Complex128: return "complex128";
        default:                return "invalid";
    }
}

/// Lightweight, non-owning view over a contiguous tensor buffer.
///
/// Stores shape and strides internally so the object is trivially copyable
/// and can be passed by value into CUDA kernel argument lists.
class Tensor {
public:
    /// Default-construct an empty (invalid) view.
    Tensor()
        : data_(nullptr), ndim_(0), dtype_(DType::Invalid) {
        for (int i = 0; i < kMaxDim; ++i) {
            shape_[i] = 0;
            strides_[i] = 0;
        }
    }

    /// Construct from raw components.  Strides are computed assuming
    /// C-contiguous (row-major) layout.
    Tensor(void* data, const int64_t* shape, int ndim, DType dtype)
        : data_(data), ndim_(ndim), dtype_(dtype) {
        for (int i = 0; i < kMaxDim; ++i) {
            shape_[i] = (i < ndim) ? shape[i] : 0;
            strides_[i] = 0;
        }
        // C-contiguous strides (element counts, not bytes).
        if (ndim > 0) {
            strides_[ndim - 1] = 1;
            for (int i = ndim - 2; i >= 0; --i) {
                strides_[i] = strides_[i + 1] * shape_[i + 1];
            }
        }
    }

    // -- data access --------------------------------------------------

    /// Raw data pointer (void*).  Compatible with ``tvm::ffi::TensorView::data_ptr()``.
    /// For a typed pointer, use the template overload: ``data_ptr<float>()``.
    void* data_ptr() const noexcept { return data_; }

    /// Typed data pointer (caller must ensure the correct type).
    ///
    /// Example::
    ///
    ///     float* p = tensor.data_ptr<float>();
    ///     const float* cp = tensor.data_ptr<const float>();
    template <typename T>
    T* data_ptr() const noexcept { return static_cast<T*>(data_); }

    /// Alias for ``data_ptr()`` — kept for backward compatibility.
    void* data() const noexcept { return data_; }

    // -- shape / strides ----------------------------------------------

    /// Number of dimensions.
    int ndim() const noexcept { return ndim_; }

    /// Size along dimension @p i.
    int64_t shape(int i) const noexcept { return shape_[i]; }

    /// Stride (in elements) along dimension @p i.
    int64_t stride(int i) const noexcept { return strides_[i]; }

    /// Pointer to the internal shape array.
    const int64_t* shape_ptr() const noexcept { return shape_; }

    /// Pointer to the internal strides array.
    const int64_t* strides_ptr() const noexcept { return strides_; }

    /// Alias: size(i) == shape(i).
    int64_t size(int i) const noexcept { return shape_[i]; }

    // -- dtype --------------------------------------------------------

    /// Element data type.
    DType dtype() const noexcept { return dtype_; }

    /// Bytes per element.
    size_t element_size() const noexcept { return dtype_size(dtype_); }

    // -- aggregate queries --------------------------------------------

    /// Total number of elements.
    int64_t numel() const noexcept {
        if (ndim_ == 0) return 0;
        int64_t n = 1;
        for (int i = 0; i < ndim_; ++i) n *= shape_[i];
        return n;
    }

    /// Total number of bytes.
    size_t nbytes() const noexcept {
        return static_cast<size_t>(numel()) * element_size();
    }

    /// Whether the tensor is C-contiguous.
    bool is_contiguous() const noexcept {
        if (ndim_ <= 1) return true;
        int64_t expected = 1;
        for (int i = ndim_ - 1; i >= 0; --i) {
            if (strides_[i] != expected) return false;
            expected *= shape_[i];
        }
        return true;
    }

private:
    void*    data_;
    int64_t  shape_[kMaxDim];
    int64_t  strides_[kMaxDim];
    int      ndim_;
    DType    dtype_;
};

}  // namespace JKB
