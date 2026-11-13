// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

// hybrid (v3) binary SpMM: BN spikes, BN output. Kernels and the schedule live in hybrid_batchedv3_spmm_impl.cuh
// (shared with hybrid_batchedv3_spmm_weighted.cu); geometry, planning and sizing in hybrid_batchedv3_spmm_plan.cuh.
// launch_accumulate(): hybrid_v3, the caller zeroes the whole output beforehand.
// launch_group_zeroed(): hybrid_v3_gz, K1 also zeroes its group's output slice right before K2/K3
//   accumulate into it, so the zeros are still in L2 (no separate memset; no precondition on the output).
// Homo variants read one device value shared by all edges.
// Types: values ValueT in {half, bfloat16, float, double}, output OutT<ValueT> (float for half/bfloat16);
// events int8 (bool) or float,
// active iff != 0; indices IndexT and row_ptr OffsetT in {int32, int64}.
#include "hybrid_batchedv3_spmm_impl.cuh"

namespace hybrid
{
  // hybrid_v3: output_bn must be all zero on entry (the caller zeroes it); the operator only accumulates.
  // long_mask: long_row_mask() of the same matrix and plan, on the device.
  template <typename ValueT, typename EventT, typename IndexT, typename OffsetT>
  cudaError_t launch_accumulate(const ValueT *values, const IndexT *indices, const OffsetT *row_ptr,
                                const uint32_t *long_mask, const EventT *spike_bn, OutT<ValueT> *output_bn,
                                int32_t *short_rows, void *workspace, int rows, int cols, int batch, const Plan &plan,
                                cudaStream_t stream)
  {
    return launch_impl<false, false, ValueT, EventT, void, IndexT, OffsetT>(values, indices, row_ptr, long_mask, spike_bn, output_bn, short_rows, workspace,
                                     rows, cols, batch, plan, stream);
  }

  // hybrid_v3_gz: output_bn may hold anything; each group's K1 zeroes that group's output rows first.
  template <typename ValueT, typename EventT, typename IndexT, typename OffsetT>
  cudaError_t launch_group_zeroed(const ValueT *values, const IndexT *indices, const OffsetT *row_ptr,
                                  const uint32_t *long_mask, const EventT *spike_bn, OutT<ValueT> *output_bn,
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
                                     const uint32_t *long_mask, const EventT *spike_bn, OutT<ValueT> *output_bn,
                                     int32_t *short_rows, void *workspace, int rows, int cols, int batch, const Plan &plan,
                                     cudaStream_t stream)
  {
    return launch_impl<false, true, ValueT, EventT, void, IndexT, OffsetT>(values, indices, row_ptr, long_mask, spike_bn, output_bn, short_rows, workspace,
                                    rows, cols, batch, plan, stream);
  }

  // hybrid_v3_gz: output_bn may hold anything; each group's K1 zeroes that group's output rows first.
  template <typename ValueT, typename EventT, typename IndexT, typename OffsetT>
  cudaError_t launch_group_zeroed_homo(const ValueT *values, const IndexT *indices, const OffsetT *row_ptr,
                                       const uint32_t *long_mask, const EventT *spike_bn, OutT<ValueT> *output_bn,
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
  template cudaError_t launch_accumulate<V, E, I, O>(const V *, const I *, const O *, const uint32_t *, const E *, OutT<V> *, int32_t *, void *, int, int, int, const Plan &, cudaStream_t);      \
  template cudaError_t launch_group_zeroed<V, E, I, O>(const V *, const I *, const O *, const uint32_t *, const E *, OutT<V> *, int32_t *, void *, int, int, int, const Plan &, cudaStream_t);    \
  template cudaError_t launch_accumulate_homo<V, E, I, O>(const V *, const I *, const O *, const uint32_t *, const E *, OutT<V> *, int32_t *, void *, int, int, int, const Plan &, cudaStream_t); \
  template cudaError_t launch_group_zeroed_homo<V, E, I, O>(const V *, const I *, const O *, const uint32_t *, const E *, OutT<V> *, int32_t *, void *, int, int, int, const Plan &, cudaStream_t);
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

// JAX FFI entry points for the hybrid v3_gz binary SpMM (BN events -> BN output, overwritten).
// batched_spmm_v3_{f16,bf16,f32,f64}_{bool,float}: the value suffix names the values/output dtype,
// the event suffix the events dtype. Index dtypes (int32/int64) are dispatched at run time.
#include "hybrid_batchedv3_spmm_ffi.cuh"

#define DEFINE_BATCHED_SPMM_V3(SUFFIX, VALUE_T, EVENT_T, HOMO)                                                  \
  void batched_spmm_v3##SUFFIX(const BE::Tensor values, const BE::Tensor indices, const BE::Tensor indptr,      \
                               const BE::Tensor long_mask, const BE::Tensor events, BE::Tensor output,          \
                               BE::Tensor workspace, BE::Tensor short_rows, int64_t group,                      \
                               int64_t long_threshold, int64_t chunk, int64_t long_tasks,                       \
                               int64_t task_capacity, int64_t expected_sms, int64_t expected_l2_bytes, int64_t stream)                                           \
  {                                                                                                             \
    batched_spmm_v3_ffi<VALUE_T, EVENT_T, false, HOMO, false>(                                                  \
        "batched_spmm_v3" #SUFFIX, values, indices, indptr, nullptr, long_mask, events, output, workspace,       \
        short_rows, group, long_threshold, chunk, long_tasks, task_capacity, expected_sms, expected_l2_bytes,    \
        stream);                                                                                                \
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
