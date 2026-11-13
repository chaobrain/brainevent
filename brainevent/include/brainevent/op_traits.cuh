// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

#ifndef BRAINEVENT_OP_TRAITS_CUH
#define BRAINEVENT_OP_TRAITS_CUH
// BrainEvent dtype mapping; scalar CUDA operations live in cuda_traits.cuh.
#include "brainevent/cuda_traits.cuh"
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
// Accumulator of ValueT (float for half/bfloat16) and its traits: scatter kernels
// add every edge in AccType and the caller rounds each finished sum once.
template<class ValueT> using AccType = typename ValueTraits<ValueT>::AccT;
template<class ValueT> using AccTraits = ValueTraits<AccType<ValueT>>;
// dtype of an event buffer: the binary event type, or the value dtype for
// weighted kernels, which read the event as a scale.
template<class ValueT, class EventT, bool Weighted>
constexpr BE::DType event_dtype() {
    if constexpr (Weighted) return ValueTraits<ValueT>::dtype;
    else return EventTraits<EventT>::dtype;
}
inline bool is_signed_index(BE::DType dtype) {
    return dtype == BE::DType::Int32 || dtype == BE::DType::Int64;
}
} // namespace op_traits
#endif
