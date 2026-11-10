// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

#ifndef HYBRID_BATCHEDV3_SPMM_CUH
#define HYBRID_BATCHEDV3_SPMM_CUH
// Public API. Link hybrid_batchedv3_spmm_binary.cu (binary launchers) and/or hybrid_batchedv3_spmm_weighted.cu
// (weighted launchers). Define BE_SPMM_STANDALONE to omit BrainEvent bindings. Planning and
// sizing are inline (hybrid_batchedv3_spmm_plan.cuh), so both objects can share one program.
#include "_common/cuda_traits.cuh"
#include <cstddef>
#include <vector>
namespace hybrid
{
    constexpr int kMaxGroupBatches = 65535;
    struct Device
    {
        int sms = 0;
        int l2_bytes = 0;
    };
    struct Plan
    {
        int group = 32;
        int64_t long_threshold = 0;
        int chunk = 1;
        int64_t long_tasks = 0;
        int64_t task_capacity = 0;
        int64_t nnz = -1; // Set by make_plan; -1 preserves manually constructed schedules.
    };
} // namespace hybrid

// device_info, make_plan, long_row_mask, workspace_bytes(plan, batch, value_bytes = 0).
#include "hybrid_batchedv3_spmm_plan.cuh"

namespace hybrid
{
    // Both launchers use caller-owned workspace and short_rows[batch * rows].
    // launch_accumulate requires a prezeroed output for an independent product.
    template <class ValueT, class EventT, class IndexT, class OffsetT>
    cudaError_t launch_accumulate(const ValueT *values, const IndexT *indices, const OffsetT *row_ptr,
                                  const uint32_t *long_mask, const EventT *events, ValueT *output,
                                  int32_t *short_rows, void *workspace, int rows, int cols, int batch,
                                  const Plan &plan, cudaStream_t stream);
    template <class ValueT, class EventT, class IndexT, class OffsetT>
    cudaError_t launch_group_zeroed(const ValueT *values, const IndexT *indices, const OffsetT *row_ptr,
                                    const uint32_t *long_mask, const EventT *events, ValueT *output,
                                    int32_t *short_rows, void *workspace, int rows, int cols, int batch,
                                    const Plan &plan, cudaStream_t stream);
    // Homo variants read one device value shared by all edges. Plan/workspace are unchanged.
    template <class ValueT, class EventT, class IndexT, class OffsetT>
    cudaError_t launch_accumulate_homo(const ValueT *values, const IndexT *indices, const OffsetT *row_ptr,
                                       const uint32_t *long_mask, const EventT *events, ValueT *output,
                                       int32_t *short_rows, void *workspace, int rows, int cols, int batch,
                                       const Plan &plan, cudaStream_t stream);
    template <class ValueT, class EventT, class IndexT, class OffsetT>
    cudaError_t launch_group_zeroed_homo(const ValueT *values, const IndexT *indices, const OffsetT *row_ptr,
                                         const uint32_t *long_mask, const EventT *events, ValueT *output,
                                         int32_t *short_rows, void *workspace, int rows, int cols, int batch,
                                         const Plan &plan, cudaStream_t stream);

    // Weighted variants (hybrid_batchedv3_spmm_weighted.cu): output[b, col] += events[b, row] * w for
    // active events[b, row] != 0, with events, values and output sharing ValueT. Plan, long_mask,
    // short_rows and the output contracts are those of the binary launchers above. The workspace
    // adds a task value queue (the vw standard: each long-row task stores its event value, short
    // rows re-read events[b, row]), so it must hold workspace_bytes(plan, batch, sizeof(ValueT)).
    template <class ValueT, class IndexT, class OffsetT>
    cudaError_t launch_accumulate_weighted(const ValueT *values, const IndexT *indices, const OffsetT *row_ptr,
                                           const uint32_t *long_mask, const ValueT *events, ValueT *output,
                                           int32_t *short_rows, void *workspace, int rows, int cols, int batch,
                                           const Plan &plan, cudaStream_t stream);
    template <class ValueT, class IndexT, class OffsetT>
    cudaError_t launch_group_zeroed_weighted(const ValueT *values, const IndexT *indices, const OffsetT *row_ptr,
                                             const uint32_t *long_mask, const ValueT *events, ValueT *output,
                                             int32_t *short_rows, void *workspace, int rows, int cols, int batch,
                                             const Plan &plan, cudaStream_t stream);
    template <class ValueT, class IndexT, class OffsetT>
    cudaError_t launch_accumulate_weighted_homo(const ValueT *values, const IndexT *indices, const OffsetT *row_ptr,
                                                const uint32_t *long_mask, const ValueT *events, ValueT *output,
                                                int32_t *short_rows, void *workspace, int rows, int cols, int batch,
                                                const Plan &plan, cudaStream_t stream);
    template <class ValueT, class IndexT, class OffsetT>
    cudaError_t launch_group_zeroed_weighted_homo(const ValueT *values, const IndexT *indices, const OffsetT *row_ptr,
                                                  const uint32_t *long_mask, const ValueT *events, ValueT *output,
                                                  int32_t *short_rows, void *workspace, int rows, int cols, int batch,
                                                  const Plan &plan, cudaStream_t stream);
} // namespace hybrid
#endif
