// hybrid (v3) binary SpMM: BN spikes, BN output. Kernels and the schedule live in hybrid_batchedv3_spmm_impl.cuh
// (shared with hybrid_batchedv3_spmm_weighted.cu); geometry, planning and sizing in hybrid_batchedv3_spmm_plan.cuh.
// launch_accumulate(): hybrid_v3, the caller zeroes the whole output beforehand.
// launch_group_zeroed(): hybrid_v3_gz, K1 also zeroes its group's output slice right before K2/K3
//   accumulate into it, so the zeros are still in L2 (no separate memset; no precondition on the output).
// Homo variants read one device value shared by all edges.
// Types: values/output ValueT in {half, bfloat16, float, double}; events int8 (bool) or float,
// active iff != 0; indices IndexT and row_ptr OffsetT in {int32, int64}.
#include "hybrid_batchedv3_spmm_impl.cuh"

namespace hybrid
{
  // hybrid_v3: output_bn must be all zero on entry (the caller zeroes it); the operator only accumulates.
  // long_mask: long_row_mask() of the same matrix and plan, on the device.
  template <typename ValueT, typename EventT, typename IndexT, typename OffsetT>
  cudaError_t launch_accumulate(const ValueT *values, const IndexT *indices, const OffsetT *row_ptr,
                                const uint32_t *long_mask, const EventT *spike_bn, ValueT *output_bn,
                                int32_t *short_rows, void *workspace, int rows, int cols, int batch, const Plan &plan,
                                cudaStream_t stream)
  {
    return launch_impl<false, false, ValueT, EventT, void, IndexT, OffsetT>(values, indices, row_ptr, long_mask, spike_bn, output_bn, short_rows, workspace,
                                     rows, cols, batch, plan, stream);
  }

  // hybrid_v3_gz: output_bn may hold anything; each group's K1 zeroes that group's output rows first.
  template <typename ValueT, typename EventT, typename IndexT, typename OffsetT>
  cudaError_t launch_group_zeroed(const ValueT *values, const IndexT *indices, const OffsetT *row_ptr,
                                  const uint32_t *long_mask, const EventT *spike_bn, ValueT *output_bn,
                                  int32_t *short_rows, void *workspace, int rows, int cols, int batch,
                                  const Plan &plan, cudaStream_t stream)
  {
    return launch_impl<true, false, ValueT, EventT, void, IndexT, OffsetT>(values, indices, row_ptr, long_mask, spike_bn, output_bn, short_rows, workspace,
                                    rows, cols, batch, plan, stream);
  }

  // hybrid_v3: output_bn must be all zero on entry (the caller zeroes it); the operator only accumulates.
  // long_mask: long_row_mask() of the same matrix and plan, on the device.
  template <typename ValueT, typename EventT, typename IndexT, typename OffsetT>
  cudaError_t launch_accumulate_homo(const ValueT *values, const IndexT *indices, const OffsetT *row_ptr,
                                     const uint32_t *long_mask, const EventT *spike_bn, ValueT *output_bn,
                                     int32_t *short_rows, void *workspace, int rows, int cols, int batch, const Plan &plan,
                                     cudaStream_t stream)
  {
    return launch_impl<false, true, ValueT, EventT, void, IndexT, OffsetT>(values, indices, row_ptr, long_mask, spike_bn, output_bn, short_rows, workspace,
                                    rows, cols, batch, plan, stream);
  }

  // hybrid_v3_gz: output_bn may hold anything; each group's K1 zeroes that group's output rows first.
  template <typename ValueT, typename EventT, typename IndexT, typename OffsetT>
  cudaError_t launch_group_zeroed_homo(const ValueT *values, const IndexT *indices, const OffsetT *row_ptr,
                                       const uint32_t *long_mask, const EventT *spike_bn, ValueT *output_bn,
                                       int32_t *short_rows, void *workspace, int rows, int cols, int batch,
                                       const Plan &plan, cudaStream_t stream)
  {
    return launch_impl<true, true, ValueT, EventT, void, IndexT, OffsetT>(values, indices, row_ptr, long_mask, spike_bn, output_bn, short_rows, workspace,
                                   rows, cols, batch, plan, stream);
  }
} // namespace hybrid

namespace hybrid
{
  // Definitions live in this translation unit; ordinary C++ callers link these instances.
#define INSTANTIATE_V3(V, E, I, O)                                                                                                                                                          \
  template cudaError_t launch_accumulate<V, E, I, O>(const V *, const I *, const O *, const uint32_t *, const E *, V *, int32_t *, void *, int, int, int, const Plan &, cudaStream_t);      \
  template cudaError_t launch_group_zeroed<V, E, I, O>(const V *, const I *, const O *, const uint32_t *, const E *, V *, int32_t *, void *, int, int, int, const Plan &, cudaStream_t);    \
  template cudaError_t launch_accumulate_homo<V, E, I, O>(const V *, const I *, const O *, const uint32_t *, const E *, V *, int32_t *, void *, int, int, int, const Plan &, cudaStream_t); \
  template cudaError_t launch_group_zeroed_homo<V, E, I, O>(const V *, const I *, const O *, const uint32_t *, const E *, V *, int32_t *, void *, int, int, int, const Plan &, cudaStream_t);
#define INSTANTIATE_V3_INDICES(V, E)     \
  INSTANTIATE_V3(V, E, int32_t, int32_t) \
  INSTANTIATE_V3(V, E, int32_t, int64_t) \
  INSTANTIATE_V3(V, E, int64_t, int32_t) \
  INSTANTIATE_V3(V, E, int64_t, int64_t)
  INSTANTIATE_V3_INDICES(__half, int8_t)
  INSTANTIATE_V3_INDICES(__half, float)
  INSTANTIATE_V3_INDICES(__nv_bfloat16, int8_t)
  INSTANTIATE_V3_INDICES(__nv_bfloat16, float)
  INSTANTIATE_V3_INDICES(float, int8_t)
  INSTANTIATE_V3_INDICES(float, float)
  INSTANTIATE_V3_INDICES(double, int8_t)
  INSTANTIATE_V3_INDICES(double, float)
#undef INSTANTIATE_V3_INDICES
#undef INSTANTIATE_V3
}

#ifndef BE_SPMM_STANDALONE
// JAX FFI entry points for the hybrid v3_gz binary SpMM (BN events -> BN output, overwritten).
// batched_spmm_v3_{f16,bf16,f32,f64}_{bool,float}: the value suffix names the values/output dtype,
// the event suffix the events dtype. Index dtypes (int32/int64) are dispatched at run time.
// batched_spmm_v3_device_info: SM count and L2 bytes of the current device, for host-side planning.
#include <cstdint>
#include <limits>

#include "brainevent/common.h"
#include "cuda_common.h"
#include "_common/op_traits.cuh"
#include "hybrid_batchedv3_spmm_ffi_device.cuh"

namespace
{
  template <class ValueT, class EventT, bool Homo>
  void batched_spmm_v3_impl(const char *name, const BE::Tensor &values, const BE::Tensor &indices,
                            const BE::Tensor &indptr, const BE::Tensor &long_mask, const BE::Tensor &events,
                            BE::Tensor &output, BE::Tensor &workspace, BE::Tensor &short_rows,
                            int64_t group, int64_t long_threshold, int64_t chunk, int64_t long_tasks,
                            int64_t task_capacity, int64_t expected_sms, int64_t expected_l2_bytes, int64_t stream)
  {
    constexpr BE::DType value_dtype = op_traits::ValueTraits<ValueT>::dtype;
    BE_CHECK(values.dtype() == value_dtype && output.dtype() == value_dtype &&
             events.dtype() == op_traits::EventTraits<EventT>::dtype &&
             op_traits::is_signed_index(indices.dtype()) && op_traits::is_signed_index(indptr.dtype()) &&
             long_mask.dtype() == BE::DType::UInt32 && workspace.dtype() == BE::DType::UInt32 &&
             short_rows.dtype() == BE::DType::Int32)
        << name << ": invalid dtype";
    BE_CHECK(values.ndim() == 1 && indices.ndim() == 1 && indptr.ndim() == 1 && long_mask.ndim() == 1 &&
             events.ndim() == 2 && output.ndim() == 2 && workspace.ndim() == 1 && short_rows.ndim() == 1)
        << name << ": invalid rank";
    const int64_t batch = events.size(0), rows = events.size(1), cols = output.size(1), nnz = indices.numel();
    BE_CHECK(output.size(0) == batch && indptr.numel() == rows + 1 && values.numel() == (Homo ? 1 : nnz) &&
             long_mask.numel() == (rows + 31) / 32)
        << name << ": invalid shape";
    BE_CHECK(rows <= std::numeric_limits<int>::max() && cols <= std::numeric_limits<int>::max() &&
             batch <= std::numeric_limits<int>::max())
        << name << ": dimensions exceed int32";
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (batch == 0 || cols == 0)
      return;
    if (rows == 0 || nnz == 0)
    {
      // Nothing to scatter: the overwrite contract still requires a zero output.
      BE_CUDA_CHECK(cudaMemsetAsync(output.data_ptr<ValueT>(), 0,
                                    static_cast<size_t>(batch) * cols * sizeof(ValueT), cuda_stream));
      return;
    }
    check_prepared_spmm_device(name, expected_sms, expected_l2_bytes);
    BE_CHECK((group == 4 || group == 8 || group == 16 || group == 32) && long_threshold >= 0 &&
             chunk >= 1 && chunk <= batch && chunk <= hybrid::kMaxGroupBatches && long_tasks >= 0 &&
             task_capacity == chunk * long_tasks)
        << name << ": invalid plan";
    hybrid::Plan plan;
    plan.group = static_cast<int>(group);
    plan.long_threshold = long_threshold;
    plan.chunk = static_cast<int>(chunk);
    plan.long_tasks = long_tasks;
    plan.task_capacity = task_capacity;
    plan.nnz = nnz;
    BE_CHECK(static_cast<size_t>(workspace.numel()) * sizeof(uint32_t) >=
                 hybrid::workspace_bytes(plan, static_cast<int>(batch)) &&
             static_cast<int64_t>(short_rows.numel()) >= batch * rows)
        << name << ": scratch too small";
    BE_DISPATCH_SIGNED_INDEX_PAIR(indices.dtype(), indptr.dtype(), IndexT, OffsetT, {
      constexpr auto launch = Homo ? hybrid::launch_group_zeroed_homo<ValueT, EventT, IndexT, OffsetT>
                                   : hybrid::launch_group_zeroed<ValueT, EventT, IndexT, OffsetT>;
      BE_CUDA_CHECK((launch(
          values.data_ptr<const ValueT>(), indices.data_ptr<const IndexT>(), indptr.data_ptr<const OffsetT>(),
          long_mask.data_ptr<const uint32_t>(), events.data_ptr<const EventT>(), output.data_ptr<ValueT>(),
          short_rows.data_ptr<int32_t>(), workspace.data_ptr<uint32_t>(), static_cast<int>(rows),
          static_cast<int>(cols), static_cast<int>(batch), plan, cuda_stream)));
    });
  }
} // namespace

#define DEFINE_BATCHED_SPMM_V3(SUFFIX, VALUE_T, EVENT_T, HOMO)                                                  \
  void batched_spmm_v3##SUFFIX(const BE::Tensor values, const BE::Tensor indices, const BE::Tensor indptr,      \
                               const BE::Tensor long_mask, const BE::Tensor events, BE::Tensor output,          \
                               BE::Tensor workspace, BE::Tensor short_rows, int64_t group,                      \
                               int64_t long_threshold, int64_t chunk, int64_t long_tasks,                       \
                               int64_t task_capacity, int64_t expected_sms, int64_t expected_l2_bytes, int64_t stream)                                           \
  {                                                                                                             \
    batched_spmm_v3_impl<VALUE_T, EVENT_T, HOMO>("batched_spmm_v3" #SUFFIX, values, indices, indptr, long_mask, \
                                                 events, output, workspace, short_rows, group, long_threshold,  \
                                                 chunk, long_tasks, task_capacity, expected_sms, expected_l2_bytes, stream);                     \
  }

// @BE batched_spmm_v3_f16_bool
DEFINE_BATCHED_SPMM_V3(_f16_bool, __half, int8_t, false)
// @BE batched_spmm_v3_f16_float
DEFINE_BATCHED_SPMM_V3(_f16_float, __half, float, false)
// @BE batched_spmm_v3_bf16_bool
DEFINE_BATCHED_SPMM_V3(_bf16_bool, __nv_bfloat16, int8_t, false)
// @BE batched_spmm_v3_bf16_float
DEFINE_BATCHED_SPMM_V3(_bf16_float, __nv_bfloat16, float, false)
// @BE batched_spmm_v3_f32_bool
DEFINE_BATCHED_SPMM_V3(_f32_bool, float, int8_t, false)
// @BE batched_spmm_v3_f32_float
DEFINE_BATCHED_SPMM_V3(_f32_float, float, float, false)
// @BE batched_spmm_v3_f64_bool
DEFINE_BATCHED_SPMM_V3(_f64_bool, double, int8_t, false)
// @BE batched_spmm_v3_f64_float
DEFINE_BATCHED_SPMM_V3(_f64_float, double, float, false)
// @BE batched_spmm_v3_homo_f16_bool
DEFINE_BATCHED_SPMM_V3(_homo_f16_bool, __half, int8_t, true)
// @BE batched_spmm_v3_homo_f16_float
DEFINE_BATCHED_SPMM_V3(_homo_f16_float, __half, float, true)
// @BE batched_spmm_v3_homo_bf16_bool
DEFINE_BATCHED_SPMM_V3(_homo_bf16_bool, __nv_bfloat16, int8_t, true)
// @BE batched_spmm_v3_homo_bf16_float
DEFINE_BATCHED_SPMM_V3(_homo_bf16_float, __nv_bfloat16, float, true)
// @BE batched_spmm_v3_homo_f32_bool
DEFINE_BATCHED_SPMM_V3(_homo_f32_bool, float, int8_t, true)
// @BE batched_spmm_v3_homo_f32_float
DEFINE_BATCHED_SPMM_V3(_homo_f32_float, float, float, true)
// @BE batched_spmm_v3_homo_f64_bool
DEFINE_BATCHED_SPMM_V3(_homo_f64_bool, double, int8_t, true)
// @BE batched_spmm_v3_homo_f64_float
DEFINE_BATCHED_SPMM_V3(_homo_f64_float, double, float, true)

#undef DEFINE_BATCHED_SPMM_V3

// @BE batched_spmm_v3_device_info
void batched_spmm_v3_device_info(const BE::Tensor dummy, BE::Tensor info, int64_t stream)
{
  BE_CHECK(info.dtype() == BE::DType::Int32 && info.ndim() == 1 && info.numel() == 2) << "expects int32[2]";
  hybrid::Device device;
  BE_CUDA_CHECK(hybrid::device_info(&device));
  const int32_t host[2] = {device.sms, device.l2_bytes};
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  BE_CUDA_CHECK(cudaMemcpyAsync(info.data_ptr<int32_t>(), host, sizeof(host), cudaMemcpyHostToDevice, cuda_stream));
  // host is a stack array: finish the copy before returning.
  BE_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
}

#endif
