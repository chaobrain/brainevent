// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

#ifndef DEVING_HYBRID_SDDMV_CUH
#define DEVING_HYBRID_SDDMV_CUH

// Link hybrid_sddmv_binary.cu for launch_binary and
// hybrid_sddmv_weighted.cu for launch_weighted. Define
// BE_SDDMV_STANDALONE when compiling the .cu files to omit BrainEvent/FFI bindings.
// This shared header also supports ordinary C++ callers.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace hybrid_sddmv
{
    // Common contract for launch_binary and launch_weighted:
    // All pointers refer to device memory. dweight has nnz entries and must be
    // zeroed by the caller before each independent product: inactive slots are
    // untouched. No allocation or host synchronization is performed.
    // task_begin/task_end have task_capacity entries, with the same type as indptr.
    // status has two int32 entries {active_task_count, overflow_flag}; it is cleared
    // on every call, including empty inputs. Task buffers may be null if capacity=0.
    // Capacity must cover every row (including inactive ones):
    //   sum(row_nnz > BE_HYBRID_TPR_THRESHOLD ?
    //       ceil(row_nnz / BE_HYBRID_TASK_NNZ) : 0).
    // If BE_HYBRID_TPR_THRESHOLD / BE_HYBRID_TASK_NNZ are overridden when compiling
    // the .cu, compute capacity with the same values. Read status after stream
    // completion; an overflow flag means capacity was insufficient, output incomplete.
    // indices and indptr independently support int32_t/int64_t; m is in [0, INT_MAX].
    // CSR structure, column bounds and allocation sizes are the caller's responsibility.
    // Returns cudaErrorInvalidValue for invalid scalar arguments or missing buffers,
    // otherwise the CUDA memset/launch status. Execution errors surface asynchronously.

    // Binary semantics: dweight[j] = ct[indices[j]] for vector[row] != 0.
    // ValueT: __half, __nv_bfloat16, float, double. EventT: int8_t (bool/int8), float.
    // ValueT=double additionally supports EventT=double. All event types only gate
    // activity; negative and NaN values are active. FFI distinguishes bool/int8.
    template <typename ValueT, typename EventT, typename IndexT, typename OffsetT>
    cudaError_t launch_binary(
        const ValueT *ct, const IndexT *indices, const OffsetT *indptr,
        const EventT *vector, ValueT *dweight, OffsetT *task_begin,
        OffsetT *task_end, int32_t *status, int m, int64_t nnz,
        int64_t task_capacity, cudaStream_t stream);

    // Weighted semantics: dweight[j] = vector[row] * ct[indices[j]] when vector[row]
    // is nonzero. ct, vector and dweight share ValueT (__half, __nv_bfloat16,
    // float or double). Half values multiply in float32 and round on storage;
    // double stays float64. Each active slot is overwritten, not accumulated.
    // Negative and NaN events are active.
    // Value queue (the vw standard): task_value has task_capacity ValueT entries;
    // each heavy task stores the event value read at extraction, so tasks never
    // look up their row. It may be null if capacity=0, like task_begin/task_end.
    // ValueT can be deduced from pointers; the default preserves existing f32 calls.
    template <typename IndexT, typename OffsetT, typename ValueT = float>
    cudaError_t launch_weighted(
        const ValueT *ct, const IndexT *indices, const OffsetT *indptr,
        const ValueT *vector, ValueT *dweight, OffsetT *task_begin,
        OffsetT *task_end, ValueT *task_value, int32_t *status, int m, int64_t nnz,
        int64_t task_capacity, cudaStream_t stream);
} // namespace hybrid_sddmv

#endif // DEVING_HYBRID_SDDMV_CUH
