// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

#ifndef COMBINED_COL_MAJOR_FFI_CUH
#define COMBINED_COL_MAJOR_FFI_CUH
#include "combined_col_major.cuh"
#include "brainevent/op_traits.cuh"
namespace
{
template <class V, class E, class O>
constexpr auto combined_col_major_launcher()
{
  return combined_col_major::launch<V, E, O>;
}

template <class V, class E>
void combined_col_major_ffi(const BE::Tensor &values, const BE::Tensor &local,
    const BE::Tensor &offsets, const BE::Tensor &perm, const BE::Tensor &events,
    BE::Tensor &output, BE::Tensor &active, BE::Tensor &workspace,
    int64_t width, int64_t type, int64_t stream)
{
  constexpr auto event_dtype = op_traits::EventTraits<E>::dtype;
  BE_CHECK(values.dtype() == op_traits::ValueTraits<V>::dtype && output.dtype() == values.dtype() &&
           events.dtype() == event_dtype && local.dtype() == BE::DType::UInt16 &&
           op_traits::is_signed_index(offsets.dtype()) && perm.dtype() == offsets.dtype() &&
           active.dtype() == BE::DType::Int32 && workspace.dtype() == BE::DType::UInt32)
      << "combined col major: invalid dtype (perm must match tile_offsets)";
  BE_CHECK(values.ndim() == 1 && local.ndim() == 1 && offsets.ndim() == 1 && perm.ndim() == 1 &&
           events.ndim() == 2 && output.ndim() == 2 && active.ndim() == 1 && workspace.ndim() == 1)
      << "combined col major: invalid rank";
  const int64_t rows = events.size(1), cols = output.size(1), batch = events.size(0);
  BE_CHECK(rows <= INT32_MAX && cols <= INT32_MAX && batch <= 65535 && output.size(0) == batch &&
           type == static_cast<int>(cuda_traits::ValueTraits<V>::type))
      << "combined col major: invalid plan";
  combined::Plan plan;
  BE_CUDA_CHECK(combined::make_plan(rows, cols, batch, local.numel(),
                                   static_cast<combined::ValueType>(type), &plan));
  BE_CHECK(width == plan.tile_width) << "combined col major: invalid plan tile width";
  const int64_t tiles = (cols + width - 1) / width;
  BE_CHECK(values.numel() == plan.nnz && perm.numel() == plan.nnz)
      << "combined col major: hetero weights and perm length must equal nnz";
  BE_CHECK(offsets.numel() == rows * (tiles + 1) && active.numel() >= batch * rows &&
           workspace.numel() * sizeof(uint32_t) >= combined::workspace_bytes(rows, batch))
      << "combined col major: invalid metadata or scratch size";
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  BE_DISPATCH_SIGNED_INDEX(offsets.dtype(), O, {
    constexpr auto launch = combined_col_major_launcher<V, E, O>();
    BE_CUDA_CHECK((launch(values.data_ptr<const V>(),
        local.data_ptr<const uint16_t>(), offsets.data_ptr<const O>(), perm.data_ptr<const O>(),
        events.data_ptr<const E>(), output.data_ptr<V>(), active.data_ptr<int32_t>(),
        workspace.data_ptr<uint32_t>(), plan, cuda_stream)));
  });
}
} // namespace

#endif
