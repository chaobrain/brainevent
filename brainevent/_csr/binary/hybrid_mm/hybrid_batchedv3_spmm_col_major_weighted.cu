// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

// Heterogeneous column-major weighted SpMM: each nonzero event multiplies
// values[perm[j]]. Short columns re-read events; long tasks carry their value.
#include "hybrid_batchedv3_spmm_col_major.cuh"
#include "hybrid_batchedv3_spmm_impl.cuh"

namespace hybrid_col_major
{
#define DEFINE_COL_MAJOR_WEIGHTED_LAUNCH(NAME, ZERO)                                                           \
  template <class ValueT, class IndexT, class OffsetT>                                                          \
  cudaError_t NAME(const ValueT *values, const IndexT *indices, const OffsetT *row_ptr, const OffsetT *perm,     \
                   const uint32_t *long_mask, const ValueT *events, ValueT *output, int32_t *short_rows,        \
                   void *workspace, int rows, int cols, int batch, const Plan &plan, cudaStream_t stream)      \
  {                                                                                                          \
    return hybrid::launch_impl<ZERO, false, ValueT, ValueT, ValueT, IndexT, OffsetT, true>(                      \
        values, indices, row_ptr, long_mask, events, output, short_rows, workspace,                            \
        rows, cols, batch, plan, stream, perm);                                                               \
  }
  DEFINE_COL_MAJOR_WEIGHTED_LAUNCH(launch_accumulate_weighted, false)
  DEFINE_COL_MAJOR_WEIGHTED_LAUNCH(launch_group_zeroed_weighted, true)
#undef DEFINE_COL_MAJOR_WEIGHTED_LAUNCH

#define INSTANTIATE_COL_MAJOR_WEIGHTED(V, I, O)                                                                \
  template cudaError_t launch_accumulate_weighted<V, I, O>(const V *, const I *, const O *, const O *,         \
      const uint32_t *, const V *, V *, int32_t *, void *, int, int, int, const Plan &, cudaStream_t);           \
  template cudaError_t launch_group_zeroed_weighted<V, I, O>(const V *, const I *, const O *, const O *,       \
      const uint32_t *, const V *, V *, int32_t *, void *, int, int, int, const Plan &, cudaStream_t);
#define INSTANTIATE_COL_MAJOR_WEIGHTED_INDICES(V)     \
  INSTANTIATE_COL_MAJOR_WEIGHTED(V, int32_t, int32_t)  \
  INSTANTIATE_COL_MAJOR_WEIGHTED(V, int32_t, int64_t)  \
  INSTANTIATE_COL_MAJOR_WEIGHTED(V, int64_t, int32_t)  \
  INSTANTIATE_COL_MAJOR_WEIGHTED(V, int64_t, int64_t)
  INSTANTIATE_COL_MAJOR_WEIGHTED_INDICES(__half)
  INSTANTIATE_COL_MAJOR_WEIGHTED_INDICES(__nv_bfloat16)
  INSTANTIATE_COL_MAJOR_WEIGHTED_INDICES(float)
  INSTANTIATE_COL_MAJOR_WEIGHTED_INDICES(double)
#undef INSTANTIATE_COL_MAJOR_WEIGHTED_INDICES
#undef INSTANTIATE_COL_MAJOR_WEIGHTED
} // namespace hybrid_col_major

#ifndef BE_SPMM_STANDALONE
#include "hybrid_batchedv3_spmm_col_major_ffi.cuh"

#define DEFINE_COL_MAJOR_WEIGHTED(SUFFIX, VALUE_T)                                                             \
  void batched_spmm_v3_col_major_weighted_hetero##SUFFIX(                                                       \
      const BE::Tensor values, const BE::Tensor indices, const BE::Tensor indptr, const BE::Tensor perm,       \
      const BE::Tensor long_mask, const BE::Tensor events, BE::Tensor output, BE::Tensor workspace,            \
      BE::Tensor short_rows, int64_t group, int64_t long_threshold, int64_t chunk, int64_t long_tasks,          \
      int64_t task_capacity, int64_t expected_sms, int64_t expected_l2_bytes, int64_t stream)                                                                   \
  {                                                                                                           \
    batched_spmm_v3_col_major_impl<VALUE_T, VALUE_T, true>(                                                     \
        "batched_spmm_v3_col_major_weighted_hetero" #SUFFIX, values, indices, indptr, perm, long_mask, events, \
        output, workspace, short_rows, group, long_threshold, chunk, long_tasks, task_capacity, expected_sms, expected_l2_bytes, stream);       \
  }

// @BE batched_spmm_v3_col_major_weighted_hetero_f16
DEFINE_COL_MAJOR_WEIGHTED(_f16, __half)
// @BE batched_spmm_v3_col_major_weighted_hetero_bf16
DEFINE_COL_MAJOR_WEIGHTED(_bf16, __nv_bfloat16)
// @BE batched_spmm_v3_col_major_weighted_hetero_f32
DEFINE_COL_MAJOR_WEIGHTED(_f32, float)
// @BE batched_spmm_v3_col_major_weighted_hetero_f64
DEFINE_COL_MAJOR_WEIGHTED(_f64, double)
#undef DEFINE_COL_MAJOR_WEIGHTED
#endif
