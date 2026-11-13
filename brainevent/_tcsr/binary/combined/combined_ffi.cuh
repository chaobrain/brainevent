// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

#ifndef COMBINED_FFI_CUH
#define COMBINED_FFI_CUH
#include "combined_kernels.cuh"
#include "brainevent/op_traits.cuh"
namespace
{
  template <class V, class E, class O, bool Homo, bool Weighted>
  constexpr auto combined_launcher()
  {
    if constexpr (Weighted)
      return Homo ? combined::launch_weighted_homo<V, O> : combined::launch_weighted<V, O>;
    else
      return Homo ? combined::launch_homo<V, E, O> : combined::launch<V, E, O>;
  }

  template <class V, class E, bool Homo, bool Weighted = false>
  void combined_ffi(const BE::Tensor &values, const BE::Tensor &local, const BE::Tensor &offsets,
                    const BE::Tensor &events, BE::Tensor &output, BE::Tensor &active,
                    BE::Tensor &workspace, int64_t width, int64_t type, int64_t stream)
  {
    constexpr auto event_dtype = [] {
      if constexpr (Weighted) return op_traits::ValueTraits<V>::dtype;
      else return op_traits::EventTraits<E>::dtype;
    }();
    BE_CHECK(values.dtype() == op_traits::ValueTraits<V>::dtype && output.dtype() == values.dtype() &&
             events.dtype() == event_dtype && local.dtype() == BE::DType::UInt16 &&
             op_traits::is_signed_index(offsets.dtype()) && active.dtype() == BE::DType::Int32 &&
             workspace.dtype() == BE::DType::UInt32)
        << "combined: invalid dtype";
    BE_CHECK(values.ndim() == 1 && local.ndim() == 1 && offsets.ndim() == 1 && events.ndim() == 2 &&
             output.ndim() == 2 && active.ndim() == 1 && workspace.ndim() == 1)
        << "combined: invalid rank";
    const int64_t rows = events.size(1), cols = output.size(1), batch = events.size(0);
    BE_CHECK(rows <= INT32_MAX && cols <= INT32_MAX && batch <= 65535 && output.size(0) == batch &&
             type == static_cast<int>(cuda_traits::ValueTraits<V>::type))
        << "combined: invalid plan";
    combined::Plan plan;
    BE_CUDA_CHECK(combined::make_plan(rows, cols, batch, local.numel(), static_cast<combined::ValueType>(type), &plan));
    BE_CHECK(width == plan.tile_width) << "combined: invalid plan tile width";
    const int64_t tiles = (cols + width - 1) / width;
    BE_CHECK(values.numel() == (Homo ? 1 : plan.nnz)) << "combined: invalid weight length";
    BE_CHECK(offsets.numel() == rows * (tiles + 1) &&
             active.numel() >= batch * rows && workspace.numel() * sizeof(uint32_t) >= combined::workspace_bytes(rows, batch))
        << "combined: invalid metadata or scratch size";
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (offsets.dtype() == BE::DType::Int32)
    {
      constexpr auto launch = combined_launcher<V, E, int32_t, Homo, Weighted>();
      BE_CUDA_CHECK((launch(values.data_ptr<const V>(), local.data_ptr<const uint16_t>(),
                            offsets.data_ptr<const int32_t>(), events.data_ptr<const E>(), output.data_ptr<V>(),
                            active.data_ptr<int32_t>(), workspace.data_ptr<uint32_t>(), plan, cuda_stream)));
    }
    else
    {
      constexpr auto launch = combined_launcher<V, E, int64_t, Homo, Weighted>();
      BE_CUDA_CHECK((launch(values.data_ptr<const V>(), local.data_ptr<const uint16_t>(),
                            offsets.data_ptr<const int64_t>(), events.data_ptr<const E>(), output.data_ptr<V>(),
                            active.data_ptr<int32_t>(), workspace.data_ptr<uint32_t>(), plan, cuda_stream)));
    }
  }
}
#endif
