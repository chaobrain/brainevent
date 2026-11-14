// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

#ifndef HYBRID_BATCHEDV3_SPMM_FFI_CUH
#define HYBRID_BATCHEDV3_SPMM_FFI_CUH

// JAX FFI implementation shared by every hybrid v3_gz SpMM target (BN events ->
// BN output, overwritten): row-major binary/weighted (hetero or homo values) and
// column-major mirror binary/weighted (hetero values through perm). Each .cu
// names its targets with a one-line macro around batched_spmm_v3_ffi, so the
// generated bindings see the exact signatures. Index dtypes (int32/int64) are
// dispatched at run time. Edges accumulate in OutT (float for half/bfloat16);
// the caller rounds each finished sum once to ValueT.
#include "hybrid_batchedv3_spmm_col_major.cuh"
#include "brainevent/common.h"
#include "brainevent/op_traits.cuh"

#include <cstdint>
#include <limits>
#include <type_traits>

namespace
{
  // Prepared plans use the device's L2 size. Check the executing device before
  // touching scratch; zero-valued expectations are reserved for raw FFI tests.
  inline void check_prepared_spmm_device(const char *name, int64_t expected_sms,
                                         int64_t expected_l2_bytes)
  {
    if (expected_sms == 0 && expected_l2_bytes == 0)
      return;
    BE_CHECK(expected_sms > 0 && expected_l2_bytes > 0)
        << name << ": invalid prepared device information";
    hybrid::Device actual;
    BE_CUDA_CHECK(hybrid::device_info(&actual));
    BE_CHECK(actual.sms == expected_sms && actual.l2_bytes == expected_l2_bytes)
        << name << ": prepared CSRConfig device does not match the executing GPU; prepare it for this device";
  }

  // Validate the static plan attributes of a nonempty call on the executing
  // device and build its launch plan.
  inline hybrid::Plan plan_from_attrs(const char *name, int64_t batch, int64_t nnz, int64_t group,
                                      int64_t long_threshold, int64_t chunk, int64_t long_tasks,
                                      int64_t task_capacity, int64_t expected_sms, int64_t expected_l2_bytes)
  {
    check_prepared_spmm_device(name, expected_sms, expected_l2_bytes);
    BE_CHECK((group == 4 || group == 8 || group == 16 || group == 32) && long_threshold >= 0 &&
             chunk >= 1 && chunk <= batch && chunk <= hybrid::kMaxGroupBatches && long_tasks >= 0 &&
             long_tasks <= std::numeric_limits<int64_t>::max() / chunk && task_capacity == chunk * long_tasks)
        << name << ": invalid plan";
    hybrid::Plan plan;
    plan.group = static_cast<int>(group);
    plan.long_threshold = long_threshold;
    plan.chunk = static_cast<int>(chunk);
    plan.long_tasks = long_tasks;
    plan.task_capacity = task_capacity;
    plan.nnz = nnz;
    return plan;
  }

  // Weighted events share ValueT; binary events are EventT (int8 bool or float).
  // perm is the mirror's weight permutation for ColMajor (always hetero), else null.
  template <class ValueT, class EventT, bool Weighted, bool Homo, bool ColMajor>
  void batched_spmm_v3_ffi(
      const char *name, const BE::Tensor &values, const BE::Tensor &indices, const BE::Tensor &indptr,
      const BE::Tensor *perm, const BE::Tensor &long_mask, const BE::Tensor &events, BE::Tensor &output,
      BE::Tensor &workspace, BE::Tensor &short_rows, int64_t group, int64_t long_threshold, int64_t chunk,
      int64_t long_tasks, int64_t task_capacity, int64_t expected_sms, int64_t expected_l2_bytes,
      int64_t stream)
  {
    static_assert(!(ColMajor && Homo), "column-major hybrid SpMM is heterogeneous only");
    using OutT = hybrid::OutT<ValueT>;
    constexpr BE::DType value_dtype = op_traits::ValueTraits<ValueT>::dtype;
    constexpr BE::DType out_dtype = op_traits::ValueTraits<OutT>::dtype;
    constexpr BE::DType event_dtype = op_traits::event_dtype<ValueT, EventT, Weighted>();
    BE_CHECK(values.dtype() == value_dtype && output.dtype() == out_dtype &&
             events.dtype() == event_dtype && op_traits::is_signed_index(indices.dtype()) &&
             op_traits::is_signed_index(indptr.dtype()) && long_mask.dtype() == BE::DType::UInt32 &&
             workspace.dtype() == BE::DType::UInt32 && short_rows.dtype() == BE::DType::Int32)
        << name << ": invalid dtype";
    BE_CHECK(values.ndim() == 1 && indices.ndim() == 1 && indptr.ndim() == 1 && long_mask.ndim() == 1 &&
             events.ndim() == 2 && output.ndim() == 2 && workspace.ndim() == 1 && short_rows.ndim() == 1)
        << name << ": invalid rank";
    const int64_t batch = events.size(0), rows = events.size(1), cols = output.size(1), nnz = indices.numel();
    if constexpr (ColMajor)
    {
      BE_CHECK(perm->dtype() == indptr.dtype()) << name << ": perm dtype must match indptr";
      BE_CHECK(perm->ndim() == 1) << name << ": perm must be rank-1";
      BE_CHECK(values.numel() == nnz && perm->numel() == nnz)
          << name << ": weights/perm/indices lengths must match (hetero only)";
    }
    BE_CHECK(output.size(0) == batch && indptr.numel() == rows + 1 && values.numel() == (Homo ? 1 : nnz) &&
             long_mask.numel() == (rows + 31) / 32)
        << name << ": invalid shape";
    BE_CHECK(rows <= std::numeric_limits<int>::max() && cols <= std::numeric_limits<int>::max() &&
             batch <= std::numeric_limits<int>::max())
        << name << ": dimensions exceed int32";
    BE_CHECK(indptr.dtype() != BE::DType::Int32 || nnz <= std::numeric_limits<int32_t>::max())
        << name << ": int32 indptr cannot address nnz";
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (batch == 0 || cols == 0)
      return;
    if (rows == 0 || nnz == 0)
    {
      // Nothing to scatter: the overwrite contract still requires a zero output.
      BE_CUDA_CHECK(cudaMemsetAsync(output.data_ptr<OutT>(), 0,
                                    static_cast<size_t>(batch) * cols * sizeof(OutT), cuda_stream));
      return;
    }
    const hybrid::Plan plan = plan_from_attrs(name, batch, nnz, group, long_threshold, chunk, long_tasks,
                                              task_capacity, expected_sms, expected_l2_bytes);
    BE_CHECK(static_cast<size_t>(workspace.numel()) * sizeof(uint32_t) >=
                 hybrid::workspace_bytes(plan, static_cast<int>(batch), Weighted ? sizeof(ValueT) : 0) &&
             static_cast<int64_t>(short_rows.numel()) >= chunk * rows)
        << name << ": scratch too small";
    using EventBufT = std::conditional_t<Weighted, ValueT, EventT>;
    BE_DISPATCH_SIGNED_INDEX_PAIR(indices.dtype(), indptr.dtype(), IndexT, OffsetT, {
      const auto *v = values.data_ptr<const ValueT>();
      const auto *idx = indices.data_ptr<const IndexT>();
      const auto *ptr = indptr.data_ptr<const OffsetT>();
      const auto *mask = long_mask.data_ptr<const uint32_t>();
      const auto *ev = events.data_ptr<const EventBufT>();
      auto *out = output.data_ptr<OutT>();
      auto *short_list = short_rows.data_ptr<int32_t>();
      auto *scratch = workspace.data_ptr<uint32_t>();
      const int r = static_cast<int>(rows), c = static_cast<int>(cols), b = static_cast<int>(batch);
      if constexpr (ColMajor)
      {
        const auto *p = perm->data_ptr<const OffsetT>();
        if constexpr (Weighted)
        {
          BE_CUDA_CHECK((hybrid_col_major::launch_group_zeroed_weighted<ValueT, IndexT, OffsetT>(
              v, idx, ptr, p, mask, ev, out, short_list, scratch, r, c, b, plan, cuda_stream)));
        }
        else
        {
          BE_CUDA_CHECK((hybrid_col_major::launch_group_zeroed<ValueT, EventT, IndexT, OffsetT>(
              v, idx, ptr, p, mask, ev, out, short_list, scratch, r, c, b, plan, cuda_stream)));
        }
      }
      else if constexpr (Weighted)
      {
        constexpr auto launch = Homo ? hybrid::launch_group_zeroed_weighted_homo<ValueT, IndexT, OffsetT>
                                     : hybrid::launch_group_zeroed_weighted<ValueT, IndexT, OffsetT>;
        BE_CUDA_CHECK((launch(v, idx, ptr, mask, ev, out, short_list, scratch, r, c, b, plan, cuda_stream)));
      }
      else
      {
        constexpr auto launch = Homo ? hybrid::launch_group_zeroed_homo<ValueT, EventT, IndexT, OffsetT>
                                     : hybrid::launch_group_zeroed<ValueT, EventT, IndexT, OffsetT>;
        BE_CUDA_CHECK((launch(v, idx, ptr, mask, ev, out, short_list, scratch, r, c, b, plan, cuda_stream)));
      }
    });
  }
} // namespace

#endif
