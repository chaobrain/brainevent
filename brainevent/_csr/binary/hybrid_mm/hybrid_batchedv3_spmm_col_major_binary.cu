// Heterogeneous column-major binary SpMM: each active mirror column scatters
// values[perm[j]] to output[b, indices[j]]. Events are bool or float32 (!= 0).
#include "hybrid_batchedv3_spmm_col_major.cuh"
#include "hybrid_batchedv3_spmm_impl.cuh"

namespace hybrid_col_major
{
#define DEFINE_COL_MAJOR_BINARY_LAUNCH(NAME, ZERO)                                                             \
  template <class ValueT, class EventT, class IndexT, class OffsetT>                                            \
  cudaError_t NAME(const ValueT *values, const IndexT *indices, const OffsetT *row_ptr, const OffsetT *perm,     \
                   const uint32_t *long_mask, const EventT *events, ValueT *output, int32_t *short_rows,        \
                   void *workspace, int rows, int cols, int batch, const Plan &plan, cudaStream_t stream)      \
  {                                                                                                          \
    return hybrid::launch_impl<ZERO, false, ValueT, EventT, void, IndexT, OffsetT, true>(                        \
        values, indices, row_ptr, long_mask, events, output, short_rows, workspace,                            \
        rows, cols, batch, plan, stream, perm);                                                               \
  }
  DEFINE_COL_MAJOR_BINARY_LAUNCH(launch_accumulate, false)
  DEFINE_COL_MAJOR_BINARY_LAUNCH(launch_group_zeroed, true)
#undef DEFINE_COL_MAJOR_BINARY_LAUNCH

#define INSTANTIATE_COL_MAJOR_BINARY(V, E, I, O)                                                               \
  template cudaError_t launch_accumulate<V, E, I, O>(const V *, const I *, const O *, const O *,                \
      const uint32_t *, const E *, V *, int32_t *, void *, int, int, int, const Plan &, cudaStream_t);           \
  template cudaError_t launch_group_zeroed<V, E, I, O>(const V *, const I *, const O *, const O *,              \
      const uint32_t *, const E *, V *, int32_t *, void *, int, int, int, const Plan &, cudaStream_t);
#define INSTANTIATE_COL_MAJOR_BINARY_INDICES(V, E)      \
  INSTANTIATE_COL_MAJOR_BINARY(V, E, int32_t, int32_t)   \
  INSTANTIATE_COL_MAJOR_BINARY(V, E, int32_t, int64_t)   \
  INSTANTIATE_COL_MAJOR_BINARY(V, E, int64_t, int32_t)   \
  INSTANTIATE_COL_MAJOR_BINARY(V, E, int64_t, int64_t)
  INSTANTIATE_COL_MAJOR_BINARY_INDICES(__half, int8_t)
  INSTANTIATE_COL_MAJOR_BINARY_INDICES(__half, float)
  INSTANTIATE_COL_MAJOR_BINARY_INDICES(__nv_bfloat16, int8_t)
  INSTANTIATE_COL_MAJOR_BINARY_INDICES(__nv_bfloat16, float)
  INSTANTIATE_COL_MAJOR_BINARY_INDICES(float, int8_t)
  INSTANTIATE_COL_MAJOR_BINARY_INDICES(float, float)
  INSTANTIATE_COL_MAJOR_BINARY_INDICES(double, int8_t)
  INSTANTIATE_COL_MAJOR_BINARY_INDICES(double, float)
#undef INSTANTIATE_COL_MAJOR_BINARY_INDICES
#undef INSTANTIATE_COL_MAJOR_BINARY
} // namespace hybrid_col_major

#ifndef BE_SPMM_STANDALONE
#include "hybrid_batchedv3_spmm_col_major_ffi.cuh"

#define DEFINE_COL_MAJOR_BINARY(SUFFIX, VALUE_T, EVENT_T)                                                        \
  void batched_spmm_v3_col_major_binary_hetero##SUFFIX(                                                         \
      const BE::Tensor values, const BE::Tensor indices, const BE::Tensor indptr, const BE::Tensor perm,       \
      const BE::Tensor long_mask, const BE::Tensor events, BE::Tensor output, BE::Tensor workspace,            \
      BE::Tensor short_rows, int64_t group, int64_t long_threshold, int64_t chunk, int64_t long_tasks,          \
      int64_t task_capacity, int64_t expected_sms, int64_t expected_l2_bytes, int64_t stream)                                                                   \
  {                                                                                                           \
    batched_spmm_v3_col_major_impl<VALUE_T, EVENT_T, false>(                                                    \
        "batched_spmm_v3_col_major_binary_hetero" #SUFFIX, values, indices, indptr, perm, long_mask, events,   \
        output, workspace, short_rows, group, long_threshold, chunk, long_tasks, task_capacity, expected_sms, expected_l2_bytes, stream);       \
  }

// @BE batched_spmm_v3_col_major_binary_hetero_f16_bool
DEFINE_COL_MAJOR_BINARY(_f16_bool, __half, int8_t)
// @BE batched_spmm_v3_col_major_binary_hetero_f16_float
DEFINE_COL_MAJOR_BINARY(_f16_float, __half, float)
// @BE batched_spmm_v3_col_major_binary_hetero_bf16_bool
DEFINE_COL_MAJOR_BINARY(_bf16_bool, __nv_bfloat16, int8_t)
// @BE batched_spmm_v3_col_major_binary_hetero_bf16_float
DEFINE_COL_MAJOR_BINARY(_bf16_float, __nv_bfloat16, float)
// @BE batched_spmm_v3_col_major_binary_hetero_f32_bool
DEFINE_COL_MAJOR_BINARY(_f32_bool, float, int8_t)
// @BE batched_spmm_v3_col_major_binary_hetero_f32_float
DEFINE_COL_MAJOR_BINARY(_f32_float, float, float)
// @BE batched_spmm_v3_col_major_binary_hetero_f64_bool
DEFINE_COL_MAJOR_BINARY(_f64_bool, double, int8_t)
// @BE batched_spmm_v3_col_major_binary_hetero_f64_float
DEFINE_COL_MAJOR_BINARY(_f64_float, double, float)
#undef DEFINE_COL_MAJOR_BINARY

#endif
