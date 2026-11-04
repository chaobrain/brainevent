#ifndef HYBRID_BATCHEDV3_SPMM_COL_MAJOR_FFI_CUH
#define HYBRID_BATCHEDV3_SPMM_COL_MAJOR_FFI_CUH

#include "hybrid_batchedv3_spmm_col_major.cuh"
#include "_common/op_traits.cuh"
#include "hybrid_batchedv3_spmm_ffi_device.cuh"
#include <limits>

namespace
{
  template <class ValueT, class EventT, bool Weighted>
  void batched_spmm_v3_col_major_impl(
      const char *name, const BE::Tensor &values, const BE::Tensor &indices,
      const BE::Tensor &indptr, const BE::Tensor &perm, const BE::Tensor &long_mask,
      const BE::Tensor &events, BE::Tensor &output, BE::Tensor &workspace, BE::Tensor &short_rows,
      int64_t group, int64_t long_threshold, int64_t chunk, int64_t long_tasks,
      int64_t task_capacity, int64_t expected_sms, int64_t expected_l2_bytes, int64_t stream)
  {
    constexpr BE::DType value_dtype = op_traits::ValueTraits<ValueT>::dtype;
    constexpr BE::DType event_dtype = [] {
      if constexpr (Weighted)
        return op_traits::ValueTraits<ValueT>::dtype;
      else
        return op_traits::EventTraits<EventT>::dtype;
    }();
    BE_CHECK(values.dtype() == value_dtype && output.dtype() == value_dtype &&
             events.dtype() == event_dtype && op_traits::is_signed_index(indices.dtype()) &&
             op_traits::is_signed_index(indptr.dtype()) && long_mask.dtype() == BE::DType::UInt32 &&
             workspace.dtype() == BE::DType::UInt32 && short_rows.dtype() == BE::DType::Int32)
        << name << ": invalid dtype";
    BE_CHECK(perm.dtype() == indptr.dtype()) << name << ": perm dtype must match indptr";
    BE_CHECK(values.ndim() == 1 && indices.ndim() == 1 && indptr.ndim() == 1 && long_mask.ndim() == 1 &&
             events.ndim() == 2 && output.ndim() == 2 && workspace.ndim() == 1 && short_rows.ndim() == 1)
        << name << ": invalid rank";
    BE_CHECK(perm.ndim() == 1) << name << ": perm must be rank-1";
    const int64_t batch = events.size(0), rows = events.size(1), cols = output.size(1), nnz = indices.numel();
    BE_CHECK(values.numel() == nnz && perm.numel() == nnz)
        << name << ": weights/perm/indices lengths must match (hetero only)";
    BE_CHECK(output.size(0) == batch && indptr.numel() == rows + 1 && long_mask.numel() == (rows + 31) / 32)
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
      BE_CUDA_CHECK(cudaMemsetAsync(output.data_ptr<ValueT>(), 0,
                                    static_cast<size_t>(batch) * cols * sizeof(ValueT), cuda_stream));
      return;
    }
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
    BE_CHECK(static_cast<size_t>(workspace.numel()) * sizeof(uint32_t) >=
                 hybrid::workspace_bytes(plan, static_cast<int>(batch), Weighted ? sizeof(ValueT) : 0) &&
             static_cast<int64_t>(short_rows.numel()) >= batch * rows)
        << name << ": scratch too small";
    BE_DISPATCH_SIGNED_INDEX_PAIR(indices.dtype(), indptr.dtype(), IndexT, OffsetT, {
      if constexpr (Weighted)
      {
        BE_CUDA_CHECK((hybrid_col_major::launch_group_zeroed_weighted<ValueT, IndexT, OffsetT>(
            values.data_ptr<const ValueT>(), indices.data_ptr<const IndexT>(), indptr.data_ptr<const OffsetT>(),
            perm.data_ptr<const OffsetT>(), long_mask.data_ptr<const uint32_t>(), events.data_ptr<const ValueT>(),
            output.data_ptr<ValueT>(), short_rows.data_ptr<int32_t>(), workspace.data_ptr<uint32_t>(),
            static_cast<int>(rows), static_cast<int>(cols), static_cast<int>(batch), plan, cuda_stream)));
      }
      else
      {
        BE_CUDA_CHECK((hybrid_col_major::launch_group_zeroed<ValueT, EventT, IndexT, OffsetT>(
            values.data_ptr<const ValueT>(), indices.data_ptr<const IndexT>(), indptr.data_ptr<const OffsetT>(),
            perm.data_ptr<const OffsetT>(), long_mask.data_ptr<const uint32_t>(), events.data_ptr<const EventT>(),
            output.data_ptr<ValueT>(), short_rows.data_ptr<int32_t>(), workspace.data_ptr<uint32_t>(),
            static_cast<int>(rows), static_cast<int>(cols), static_cast<int>(batch), plan, cuda_stream)));
      }
    });
  }
} // namespace

#endif
