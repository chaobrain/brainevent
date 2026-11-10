// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

#ifndef DEVING_OPERATORS_COMMON_OP_TRAITS_CUH
#define DEVING_OPERATORS_COMMON_OP_TRAITS_CUH
// BrainEvent dtype mapping; scalar CUDA operations live in cuda_traits.cuh.
#include "cuda_traits.cuh"
#include "cuda_common.h"
#include "brainevent/common.h"
namespace op_traits {
template<class T> struct ValueTraits;
template<> struct ValueTraits<float> : cuda_traits::ValueTraits<float> { static constexpr BE::DType dtype = BE::DType::Float32; };
template<> struct ValueTraits<double> : cuda_traits::ValueTraits<double> { static constexpr BE::DType dtype = BE::DType::Float64; };
template<> struct ValueTraits<__half> : cuda_traits::ValueTraits<__half> { static constexpr BE::DType dtype = BE::DType::Float16; };
template<> struct ValueTraits<__nv_bfloat16> : cuda_traits::ValueTraits<__nv_bfloat16> { static constexpr BE::DType dtype = BE::DType::BFloat16; };
template<class T> struct EventTraits;
template<> struct EventTraits<int8_t> : cuda_traits::EventTraits<int8_t> { static constexpr BE::DType dtype = BE::DType::Bool; };
template<> struct EventTraits<float> : cuda_traits::EventTraits<float> { static constexpr BE::DType dtype = BE::DType::Float32; };
inline bool is_signed_index(BE::DType dtype) {
    return dtype == BE::DType::Int32 || dtype == BE::DType::Int64;
}
} // namespace op_traits
#endif
