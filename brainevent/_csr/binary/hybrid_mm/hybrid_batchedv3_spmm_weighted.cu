// hybrid (v3) weighted SpMM: the binary operator's schedule (hybrid_batchedv3_spmm_impl.cuh) with QueueT = ValueT, so
// each active event value scales its contributions:
//   output[b, col] += events[b, row] * w   over events[b, row] != 0   (w = values[j] or values[0]).
// Events, values and output share ValueT in {half, bfloat16, float, double}; products and sums use
// AccT. Negative and NaN events are active. Event values follow the vw standard (no row lookup):
// short rows re-read events[b, row] from the short list's row; long-row tasks carry their value in
// the task value queue, so the workspace is workspace_bytes(plan, batch, sizeof(ValueT)).
#include "hybrid_batchedv3_spmm_impl.cuh"

namespace hybrid
{
#define DEFINE_WEIGHTED_LAUNCH(NAME, ZERO, HOMO)                                                                 \
  template <typename ValueT, typename IndexT, typename OffsetT>                                                  \
  cudaError_t NAME(const ValueT *values, const IndexT *indices, const OffsetT *row_ptr, const uint32_t *long_mask, \
                   const ValueT *events, ValueT *output, int32_t *short_rows, void *workspace, int rows, int cols,  \
                   int batch, const Plan &plan, cudaStream_t stream)                                             \
  {                                                                                                              \
    return launch_impl<ZERO, HOMO, ValueT, ValueT, ValueT, IndexT, OffsetT>(                                     \
        values, indices, row_ptr, long_mask, events, output, short_rows, workspace, rows, cols, batch, plan, stream); \
  }
  // accumulate: output must be all zero on entry; group_zeroed: K1 zeroes each group's output rows.
  DEFINE_WEIGHTED_LAUNCH(launch_accumulate_weighted, false, false)
  DEFINE_WEIGHTED_LAUNCH(launch_group_zeroed_weighted, true, false)
  DEFINE_WEIGHTED_LAUNCH(launch_accumulate_weighted_homo, false, true)
  DEFINE_WEIGHTED_LAUNCH(launch_group_zeroed_weighted_homo, true, true)
#undef DEFINE_WEIGHTED_LAUNCH

  // Definitions live in this translation unit; ordinary C++ callers link these instances.
#define INSTANTIATE_WEIGHTED(V, I, O)                                                                                                                                                         \
  template cudaError_t launch_accumulate_weighted<V, I, O>(const V *, const I *, const O *, const uint32_t *, const V *, V *, int32_t *, void *, int, int, int, const Plan &, cudaStream_t);      \
  template cudaError_t launch_group_zeroed_weighted<V, I, O>(const V *, const I *, const O *, const uint32_t *, const V *, V *, int32_t *, void *, int, int, int, const Plan &, cudaStream_t);    \
  template cudaError_t launch_accumulate_weighted_homo<V, I, O>(const V *, const I *, const O *, const uint32_t *, const V *, V *, int32_t *, void *, int, int, int, const Plan &, cudaStream_t); \
  template cudaError_t launch_group_zeroed_weighted_homo<V, I, O>(const V *, const I *, const O *, const uint32_t *, const V *, V *, int32_t *, void *, int, int, int, const Plan &, cudaStream_t);
#define INSTANTIATE_WEIGHTED_INDICES(V)    \
  INSTANTIATE_WEIGHTED(V, int32_t, int32_t) \
  INSTANTIATE_WEIGHTED(V, int32_t, int64_t) \
  INSTANTIATE_WEIGHTED(V, int64_t, int32_t) \
  INSTANTIATE_WEIGHTED(V, int64_t, int64_t)
  INSTANTIATE_WEIGHTED_INDICES(__half)
  INSTANTIATE_WEIGHTED_INDICES(__nv_bfloat16)
  INSTANTIATE_WEIGHTED_INDICES(float)
  INSTANTIATE_WEIGHTED_INDICES(double)
#undef INSTANTIATE_WEIGHTED_INDICES
#undef INSTANTIATE_WEIGHTED
} // namespace hybrid

#ifndef BE_SPMM_STANDALONE
// JAX FFI entry points for the hybrid v3_gz weighted SpMM (BN events -> BN output, overwritten).
// batched_spmm_v3_weighted[_homo]_{f16,bf16,f32,f64}: events, values and output share the suffix
// dtype. Index dtypes (int32/int64) are dispatched at run time.
#include <cstdint>
#include <limits>

#include "brainevent/common.h"
#include "cuda_common.h"
#include "_common/op_traits.cuh"
#include "hybrid_batchedv3_spmm_ffi_device.cuh"

namespace
{
  template <class ValueT, bool Homo>
  void batched_spmm_v3_weighted_impl(const char *name, const BE::Tensor &values, const BE::Tensor &indices,
                            const BE::Tensor &indptr, const BE::Tensor &long_mask, const BE::Tensor &events,
                            BE::Tensor &output, BE::Tensor &workspace, BE::Tensor &short_rows,
                            int64_t group, int64_t long_threshold, int64_t chunk, int64_t long_tasks,
                            int64_t task_capacity, int64_t expected_sms, int64_t expected_l2_bytes, int64_t stream)
  {
    constexpr BE::DType value_dtype = op_traits::ValueTraits<ValueT>::dtype;
    BE_CHECK(values.dtype() == value_dtype && output.dtype() == value_dtype &&
             events.dtype() == value_dtype &&
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
                 hybrid::workspace_bytes(plan, static_cast<int>(batch), sizeof(ValueT)) &&
             static_cast<int64_t>(short_rows.numel()) >= batch * rows)
        << name << ": scratch too small";
    BE_DISPATCH_SIGNED_INDEX_PAIR(indices.dtype(), indptr.dtype(), IndexT, OffsetT, {
      constexpr auto launch = Homo ? hybrid::launch_group_zeroed_weighted_homo<ValueT, IndexT, OffsetT>
                                   : hybrid::launch_group_zeroed_weighted<ValueT, IndexT, OffsetT>;
      BE_CUDA_CHECK((launch(
          values.data_ptr<const ValueT>(), indices.data_ptr<const IndexT>(), indptr.data_ptr<const OffsetT>(),
          long_mask.data_ptr<const uint32_t>(), events.data_ptr<const ValueT>(), output.data_ptr<ValueT>(),
          short_rows.data_ptr<int32_t>(), workspace.data_ptr<uint32_t>(), static_cast<int>(rows),
          static_cast<int>(cols), static_cast<int>(batch), plan, cuda_stream)));
    });
  }
} // namespace

#define DEFINE_BATCHED_SPMM_V3_WEIGHTED(SUFFIX, VALUE_T, HOMO)                                                    \
  void batched_spmm_v3_weighted##SUFFIX(const BE::Tensor values, const BE::Tensor indices,                       \
                                        const BE::Tensor indptr, const BE::Tensor long_mask,                     \
                                        const BE::Tensor events, BE::Tensor output, BE::Tensor workspace,        \
                                        BE::Tensor short_rows, int64_t group, int64_t long_threshold,            \
                                        int64_t chunk, int64_t long_tasks, int64_t task_capacity, int64_t expected_sms, int64_t expected_l2_bytes, int64_t stream) \
  {                                                                                                               \
    batched_spmm_v3_weighted_impl<VALUE_T, HOMO>("batched_spmm_v3_weighted" #SUFFIX, values, indices, indptr,    \
                                                 long_mask, events, output, workspace, short_rows, group,        \
                                                 long_threshold, chunk, long_tasks, task_capacity, expected_sms, expected_l2_bytes, stream);      \
  }

// @BE batched_spmm_v3_weighted_f16
DEFINE_BATCHED_SPMM_V3_WEIGHTED(_f16, __half, false)
// @BE batched_spmm_v3_weighted_bf16
DEFINE_BATCHED_SPMM_V3_WEIGHTED(_bf16, __nv_bfloat16, false)
// @BE batched_spmm_v3_weighted_f32
DEFINE_BATCHED_SPMM_V3_WEIGHTED(_f32, float, false)
// @BE batched_spmm_v3_weighted_f64
DEFINE_BATCHED_SPMM_V3_WEIGHTED(_f64, double, false)
// @BE batched_spmm_v3_weighted_homo_f16
DEFINE_BATCHED_SPMM_V3_WEIGHTED(_homo_f16, __half, true)
// @BE batched_spmm_v3_weighted_homo_bf16
DEFINE_BATCHED_SPMM_V3_WEIGHTED(_homo_bf16, __nv_bfloat16, true)
// @BE batched_spmm_v3_weighted_homo_f32
DEFINE_BATCHED_SPMM_V3_WEIGHTED(_homo_f32, float, true)
// @BE batched_spmm_v3_weighted_homo_f64
DEFINE_BATCHED_SPMM_V3_WEIGHTED(_homo_f64, double, true)

#undef DEFINE_BATCHED_SPMM_V3_WEIGHTED

#endif
