// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

#ifndef HYBRID_BATCHEDV3_SPMM_COL_MAJOR_CUH
#define HYBRID_BATCHEDV3_SPMM_COL_MAJOR_CUH

// Heterogeneous column-major SpMM over a structural mirror of W(M, N).
// events: [batch, N], output: [batch, M]. The mirror supplies row_ptr[N + 1],
// indices[nnz] and perm[nnz]; structural edge j reads values[perm[j]].
// Values remain in their original CSR order. Plan rows=N, cols=M; build its
// long mask from the mirror pointers. No weight gather buffer is required.
// Link the binary and/or weighted .cu; BE_SPMM_STANDALONE omits FFI bindings.
#include "hybrid_batchedv3_spmm.cuh"

namespace hybrid_col_major
{
  using hybrid::Device;
  using hybrid::Plan;
  using hybrid::device_info;
  using hybrid::long_row_mask;
  using hybrid::make_plan;
  using hybrid::workspace_bytes;

  // Caller-owned scratch: workspace_bytes(plan, batch), short_rows[plan.chunk * rows].
  // output[batch, cols] is hybrid::OutT<ValueT> (float for half/bfloat16 values).
  // accumulate adds to output; group_zeroed overwrites each batch group's output.
  // perm has the same element type as row_ptr, independently of indices.
  template <class ValueT, class EventT, class IndexT, class OffsetT>
  cudaError_t launch_accumulate(const ValueT *values, const IndexT *indices, const OffsetT *row_ptr,
                                const OffsetT *perm, const uint32_t *long_mask, const EventT *events,
                                hybrid::OutT<ValueT> *output, int32_t *short_rows, void *workspace,
                                int rows, int cols, int batch, const Plan &plan, cudaStream_t stream);
  template <class ValueT, class EventT, class IndexT, class OffsetT>
  cudaError_t launch_group_zeroed(const ValueT *values, const IndexT *indices, const OffsetT *row_ptr,
                                  const OffsetT *perm, const uint32_t *long_mask, const EventT *events,
                                  hybrid::OutT<ValueT> *output, int32_t *short_rows, void *workspace,
                                  int rows, int cols, int batch, const Plan &plan, cudaStream_t stream);

  // Weighted events share ValueT; scratch includes the long-task event queue:
  // workspace_bytes(plan, batch, sizeof(ValueT)). Zero events skip all edges.
  template <class ValueT, class IndexT, class OffsetT>
  cudaError_t launch_accumulate_weighted(const ValueT *values, const IndexT *indices, const OffsetT *row_ptr,
                                         const OffsetT *perm, const uint32_t *long_mask, const ValueT *events,
                                         hybrid::OutT<ValueT> *output, int32_t *short_rows, void *workspace,
                                         int rows, int cols, int batch, const Plan &plan, cudaStream_t stream);
  template <class ValueT, class IndexT, class OffsetT>
  cudaError_t launch_group_zeroed_weighted(const ValueT *values, const IndexT *indices, const OffsetT *row_ptr,
                                           const OffsetT *perm, const uint32_t *long_mask, const ValueT *events,
                                           hybrid::OutT<ValueT> *output, int32_t *short_rows, void *workspace,
                                           int rows, int cols, int batch, const Plan &plan, cudaStream_t stream);
} // namespace hybrid_col_major

#endif
