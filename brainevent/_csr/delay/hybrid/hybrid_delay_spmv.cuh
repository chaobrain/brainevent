// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.
#ifndef DEVING_HYBRID_DELAY_SPMV_CUH
#define DEVING_HYBRID_DELAY_SPMV_CUH

// Public CUDA API; usable from ordinary C++ translation units.
// Link hybrid_delay_spmv_binary.cu for launch_binary,
// and hybrid_delay_spmv_weighted.cu for launch_weighted.
// Define BE_REGION_DELAY_STANDALONE when compiling the .cu files to omit FFI.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace region_delay
{
    // All pointers refer to device memory on stream. No allocation or host
    // synchronization occurs. The caller owns buffers and validates CSR values,
    // column bounds, delay offsets and allocation sizes. Rows/cols fit int32;
    // segments/slots are positive int32. Ring byte size must fit int64, and
    // its element count must fit RingT. IndexT/EdgeT/RingT independently support
    // int32_t/int64_t. ValueT is __half, __nv_bfloat16, float or double.
    // Homo=true reads weights[0]; otherwise weights has one value per CSR edge.
    // CSR rows are sorted by delay segment. delay_count is [rows, segments] in
    // the EdgeT dtype; delay_off[k] = delay_steps[k] * cols in RingT. Every
    // delay step is in [0, slots). head points to one int32 slot in [0, slots).
    // buffer is an initialized [slots, cols] ring. Each call adds contributions
    // to its existing contents, including the current head when delay is zero.
    // No spike vector is extracted and head is never advanced. Head is required
    // even for an empty graph; that graph leaves the entire ring unchanged.
    // Consumed slots are cleared by the caller before delivery when needed.
    //
    // Whole-row classification: <=128 edges use one thread; heavier rows split
    // each delay segment into <=4096-edge tasks. With the default build:
    //   row_task_count[r] = row_nnz[r] <= 128 ? 0 :
    //                       sum_k ceil(delay_count[r,k] / 4096)
    //   task_capacity >= sum_r row_task_count[r], regardless of activity.
    // Match these counts to BE_DELAY_TPR_THRESHOLD/BE_DELAY_TASK_NNZ if overridden.
    // begin/end/out_off each hold task_capacity entries. Tasks never cross a
    // segment. status is int32[2] {task_count, error_code}, reset on every call,
    // including rows=0. Error codes: 1=reservation/capacity, 2=head, 3=metadata.
    // Read status only after stream completion; any error invalidates the result
    // (short rows may already have written). Task arrays need no clearing and
    // may be null when capacity=0. Empty rows do not dereference weights/indices.
    // Returns cudaErrorInvalidValue for invalid scalar arguments/missing required
    // buffers, otherwise CUDA memset/launch status. Execution errors are async.

    // Binary: nonzero events gate weight delivery (negative/NaN are active).
    // EventT is int8_t (bool activity) or float, independently of ValueT.
    template <bool Homo, typename ValueT, typename EventT,
              typename IndexT, typename EdgeT, typename RingT>
    cudaError_t launch_binary(
        const ValueT *weights, const IndexT *indices, const EdgeT *indptr,
        const EventT *events, const EdgeT *delay_count, const RingT *delay_off,
        const int32_t *row_task_count, const int32_t *head,
        ValueT *buffer, EdgeT *task_begin, EdgeT *task_end, RingT *task_out_off,
        int32_t *status, int64_t rows, int64_t segments, int64_t cols, int64_t slots,
        int64_t task_capacity, cudaStream_t stream = nullptr);

    // Weighted: deliver weights[e] * events[row] for events[row] != 0.
    // Events, weights, ring and task_value share ValueT. Each heavy task carries
    // the event read at extraction in task_value[task]; BlockScatter never reads
    // events or indptr. task_value has task_capacity entries and may be null for
    // capacity=0. Half products use float32, then existing half ring atomics;
    // float64 stays double throughout. Zero events skip even nonfinite weights.
    template <bool Homo, typename ValueT, typename IndexT, typename EdgeT, typename RingT>
    cudaError_t launch_weighted(
        const ValueT *weights, const IndexT *indices, const EdgeT *indptr,
        const ValueT *events, const EdgeT *delay_count, const RingT *delay_off,
        const int32_t *row_task_count, const int32_t *head,
        ValueT *buffer, EdgeT *task_begin, EdgeT *task_end, RingT *task_out_off,
        ValueT *task_value, int32_t *status,
        int64_t rows, int64_t segments, int64_t cols, int64_t slots,
        int64_t task_capacity, cudaStream_t stream = nullptr);


}

#endif // DEVING_HYBRID_DELAY_SPMV_CUH
