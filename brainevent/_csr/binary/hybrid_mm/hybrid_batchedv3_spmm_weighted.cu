// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

// hybrid (v3) weighted SpMM: the binary operator's schedule (hybrid_batchedv3_spmm_impl.cuh) with QueueT = ValueT, so
// each active event value scales its contributions:
//   output[b, col] += events[b, row] * w   over events[b, row] != 0   (w = values[j] or values[0]).
// Events and values share ValueT in {half, bfloat16, float, double}; products, sums and the output
// use AccT (OutT<ValueT>). Negative and NaN events are active. Event values follow the vw standard (no row lookup):
// short rows re-read events[b, row] from the short list's row; long-row tasks carry their value in
// the task value queue, so the workspace is workspace_bytes(plan, batch, sizeof(ValueT)).
#include "hybrid_batchedv3_spmm_impl.cuh"

namespace hybrid
{
#define DEFINE_WEIGHTED_LAUNCH(NAME, ZERO, HOMO)                                                                 \
  template <typename ValueT, typename IndexT, typename OffsetT>                                                  \
  cudaError_t NAME(const ValueT *values, const IndexT *indices, const OffsetT *row_ptr, const uint32_t *long_mask, \
                   const ValueT *events, OutT<ValueT> *output, int32_t *short_rows, void *workspace, int rows, int cols,  \
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
  template cudaError_t launch_accumulate_weighted<V, I, O>(const V *, const I *, const O *, const uint32_t *, const V *, OutT<V> *, int32_t *, void *, int, int, int, const Plan &, cudaStream_t);      \
  template cudaError_t launch_group_zeroed_weighted<V, I, O>(const V *, const I *, const O *, const uint32_t *, const V *, OutT<V> *, int32_t *, void *, int, int, int, const Plan &, cudaStream_t);    \
  template cudaError_t launch_accumulate_weighted_homo<V, I, O>(const V *, const I *, const O *, const uint32_t *, const V *, OutT<V> *, int32_t *, void *, int, int, int, const Plan &, cudaStream_t); \
  template cudaError_t launch_group_zeroed_weighted_homo<V, I, O>(const V *, const I *, const O *, const uint32_t *, const V *, OutT<V> *, int32_t *, void *, int, int, int, const Plan &, cudaStream_t);
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

// JAX FFI entry points for the hybrid v3_gz weighted SpMM (BN events -> BN output, overwritten).
// batched_spmm_v3_weighted[_homo]_{f16,bf16,f32,f64}: events, values and output share the suffix
// dtype. Index dtypes (int32/int64) are dispatched at run time.
#include "hybrid_batchedv3_spmm_ffi.cuh"

#define DEFINE_BATCHED_SPMM_V3_WEIGHTED(SUFFIX, VALUE_T, HOMO)                                                    \
  void batched_spmm_v3_weighted##SUFFIX(const BE::Tensor values, const BE::Tensor indices,                       \
                                        const BE::Tensor indptr, const BE::Tensor long_mask,                     \
                                        const BE::Tensor events, BE::Tensor output, BE::Tensor workspace,        \
                                        BE::Tensor short_rows, int64_t group, int64_t long_threshold,            \
                                        int64_t chunk, int64_t long_tasks, int64_t task_capacity, int64_t expected_sms, int64_t expected_l2_bytes, int64_t stream) \
  {                                                                                                               \
    batched_spmm_v3_ffi<VALUE_T, VALUE_T, true, HOMO, false>(                                                     \
        "batched_spmm_v3_weighted" #SUFFIX, values, indices, indptr, nullptr, long_mask, events, output,          \
        workspace, short_rows, group, long_threshold, chunk, long_tasks, task_capacity, expected_sms,             \
        expected_l2_bytes, stream);                                                                               \
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
