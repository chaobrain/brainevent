// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

/*
 * hybrid_sddmv_weighted.cu -- Weighted hybrid SDDMV relying on a caller-zeroed dweight
 * =============================================================================
 *
 * Weighted counterpart of the binary file: same scheduling, but the event value
 * is carried into the product:
 *
 *   dweight[j] = vector[row] * ct[indices[j]]
 *     for active row where vector[row] != 0
 *     for j in indptr[row]:indptr[row + 1]
 *   dweight[j] = 0 for every slot of an inactive row
 *
 * Strategy (default tuning values; configurable through CSRConfig):
 *   0. Clear status only; dweight arrives zeroed (aliased dweight_in).
 *   1. Extract active rows. Each thread reads its event value once. Rows with
 *      nnz <= 128 are gathered immediately by that thread, scaled by the value.
 *   2. Heavy active rows reserve contiguous block tasks once with atomicAdd.
 *      Each task stores begin/end offsets for up to BE_HYBRID_TASK_NNZ slots and, in the
 *      value queue, the event value read in step 1.
 *   3. Launch a fixed number of 256-thread blocks. Each block reads the
 *      active task count on device and processes tasks with a grid-stride
 *      loop, avoiding D2H status reads and host synchronization.
 *
 * Value queue: heavy tasks carry their event value in
 * task_value[task] instead of recovering the row, so the block kernel never
 * reads indptr or vector and needs no shared memory or block barriers.
 *
 * Integration contract:
 *
 *     ct             : f16/bf16/f32/f64[k] cotangent
 *     indices        : s32/s64[nnz] global columns in slot order
 *     indptr         : s32/s64[m + 1]
 *     vector         : same dtype as ct [m]; active means != 0
 *     task_begin/end : same dtype as indptr, task_capacity entries
 *     task_value     : ct dtype [task_capacity] event value queue
 *     dweight_in     : ct dtype [nnz] caller-zeroed buffer aliased to dweight
 *     status         : s32[2] scratch/output {active_task_count, overflow_flag}
 *     dweight        : ct dtype [nnz] output
 *
 *   task_capacity bounds the tasks required by all heavy rows. The Python
 *   caller computes this bound from shapes and the captured CSRConfig.
 */

#include "hybrid_sddmv_impl.cuh"

namespace hybrid_sddmv
{

    template <typename IndexT, typename OffsetT, typename ValueT>
    cudaError_t launch_weighted(
        const ValueT *ct,
        const IndexT *indices,
        const OffsetT *indptr,
        const ValueT *vector,
        ValueT *dweight,
        OffsetT *task_begin,
        OffsetT *task_end,
        ValueT *task_value,
        int32_t *device_status,
        int m,
        int64_t nnz,
        int64_t task_capacity,
        cudaStream_t stream)
    {
        return launch_sddmv<true, ValueT, ValueT, IndexT, OffsetT>(
            ct, indices, indptr, vector, dweight, task_begin, task_end, task_value,
            device_status, m, nnz, task_capacity, stream);
    }

#define INSTANTIATE_HYBRID_WEIGHTED(VALUE_T, INDEX_T, OFFSET_T)              \
    template cudaError_t launch_weighted<INDEX_T, OFFSET_T, VALUE_T>(        \
        const VALUE_T *, const INDEX_T *, const OFFSET_T *, const VALUE_T *, \
        VALUE_T *, OFFSET_T *, OFFSET_T *, VALUE_T *, int32_t *, int, int64_t, int64_t, cudaStream_t);
#define INSTANTIATE_HYBRID_INDICES(VALUE_T)                \
    INSTANTIATE_HYBRID_WEIGHTED(VALUE_T, int32_t, int32_t) \
    INSTANTIATE_HYBRID_WEIGHTED(VALUE_T, int32_t, int64_t) \
    INSTANTIATE_HYBRID_WEIGHTED(VALUE_T, int64_t, int32_t) \
    INSTANTIATE_HYBRID_WEIGHTED(VALUE_T, int64_t, int64_t)

    INSTANTIATE_HYBRID_INDICES(__half)
    INSTANTIATE_HYBRID_INDICES(__nv_bfloat16)
    INSTANTIATE_HYBRID_INDICES(float)
    INSTANTIATE_HYBRID_INDICES(double)
#undef INSTANTIATE_HYBRID_INDICES
#undef INSTANTIATE_HYBRID_WEIGHTED

} // namespace hybrid_sddmv

#ifndef BE_SDDMV_STANDALONE
#include "cuda_common.h"
#include "brainevent/common.h"
#include "brainevent/dtypes.h"
#include "brainevent/op_traits.cuh"

#ifndef BE_HYBRID_SDDMV_CHECK
#define BE_HYBRID_SDDMV_CHECK(condition, message)  \
    do                                             \
    {                                              \
        BE_CHECK(condition) << (message);          \
    } while (0)
#endif

#define DEFINE_HYBRID_SDDMV_WEIGHTED(SUFFIX, VALUE_T)                                                       \
    void hybrid_sddmv_weighted##SUFFIX(                                                                     \
        const BE::Tensor ct,                                                                                \
        const BE::Tensor indices,                                                                           \
        const BE::Tensor indptr,                                                                            \
        const BE::Tensor vector,                                                                            \
        const BE::Tensor dweight_in,                                                                        \
        const BE::Tensor task_begin_scratch,                                                                \
        const BE::Tensor task_end_scratch,                                                                  \
        const BE::Tensor task_value_scratch,                                                                \
        const BE::Tensor status_scratch,                                                                    \
        BE::Tensor dweight,                                                                                 \
        BE::Tensor task_begin_output,                                                                       \
        BE::Tensor task_end_output,                                                                         \
        BE::Tensor task_value_output,                                                                       \
        BE::Tensor status_output,                                                                           \
        int64_t task_capacity,                                                                              \
        int64_t stream)                                                                                     \
    {                                                                                                       \
        constexpr BE::DType value_dtype = op_traits::ValueTraits<VALUE_T>::dtype;                           \
        BE_HYBRID_SDDMV_CHECK(                                                                              \
            ct.dtype() == value_dtype && vector.dtype() == value_dtype &&                                   \
                dweight_in.dtype() == value_dtype && dweight.dtype() == value_dtype,                        \
            "hybrid_sddmv_weighted" #SUFFIX " expects matching ct, vector and dweight dtype");              \
        BE_HYBRID_SDDMV_CHECK(                                                                              \
            op_traits::is_signed_index(indices.dtype()) && op_traits::is_signed_index(indptr.dtype()),      \
            "hybrid_sddmv_weighted" #SUFFIX " expects indices/indptr=s32/s64");                             \
        BE_HYBRID_SDDMV_CHECK(                                                                              \
            task_begin_scratch.dtype() == indptr.dtype() && task_end_scratch.dtype() == indptr.dtype() &&   \
                task_begin_output.dtype() == indptr.dtype() && task_end_output.dtype() == indptr.dtype() && \
                task_value_scratch.dtype() == value_dtype && task_value_output.dtype() == value_dtype &&    \
                status_scratch.dtype() == BE::DType::Int32 && status_output.dtype() == BE::DType::Int32,    \
            "hybrid_sddmv_weighted" #SUFFIX                                                                 \
            " expects task ranges in the indptr dtype, task values in the ct dtype and status=s32");        \
        BE_HYBRID_SDDMV_CHECK(                                                                              \
            ct.ndim() == 1 && indices.ndim() == 1 && indptr.ndim() == 1 && vector.ndim() == 1 &&            \
                dweight_in.ndim() == 1 && dweight.ndim() == 1 &&                                            \
                task_begin_scratch.ndim() == 1 && task_end_scratch.ndim() == 1 &&                           \
                task_value_scratch.ndim() == 1 && status_scratch.ndim() == 1 &&                             \
                task_begin_output.ndim() == 1 && task_end_output.ndim() == 1 &&                             \
                task_value_output.ndim() == 1 && status_output.ndim() == 1,                                 \
            "hybrid_sddmv_weighted" #SUFFIX " expects rank-1 buffers");                                     \
        BE_HYBRID_SDDMV_CHECK(                                                                              \
            dweight_in.numel() == indices.numel() && dweight.numel() == indices.numel(),                    \
            "hybrid_sddmv_weighted" #SUFFIX " expects dweight_in, dweight and indices lengths to match");   \
        BE_HYBRID_SDDMV_CHECK(                                                                              \
            indptr.numel() == vector.numel() + 1,                                                           \
            "hybrid_sddmv_weighted" #SUFFIX " expects indptr length to be vector length plus one");         \
        const size_t capacity = static_cast<size_t>(task_capacity);                                         \
        BE_HYBRID_SDDMV_CHECK(                                                                              \
            task_capacity >= 0 &&                                                                           \
                task_begin_scratch.numel() == capacity && task_end_scratch.numel() == capacity &&           \
                task_value_scratch.numel() == capacity && task_begin_output.numel() == capacity &&          \
                task_end_output.numel() == capacity && task_value_output.numel() == capacity,               \
            "hybrid_sddmv_weighted" #SUFFIX " expects task scratch lengths to match task_capacity");        \
        BE_HYBRID_SDDMV_CHECK(                                                                              \
            status_scratch.numel() == 2 && status_output.numel() == 2,                                      \
            "hybrid_sddmv_weighted" #SUFFIX " expects status scratch length 2");                            \
        BE_HYBRID_SDDMV_CHECK(                                                                              \
            indptr.numel() - 1 <= static_cast<size_t>(std::numeric_limits<int>::max()),                     \
            "hybrid_sddmv_weighted" #SUFFIX " expects at most INT32_MAX rows");                             \
        cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);                                  \
        cudaError_t launch_status = cudaSuccess;                                                            \
        BE_DISPATCH_SIGNED_INDEX_PAIR(indices.dtype(), indptr.dtype(), IndexT, OffsetT, {                   \
            launch_status = hybrid_sddmv::launch_weighted<IndexT, OffsetT, VALUE_T>(                        \
                ct.data_ptr<const VALUE_T>(),                                                               \
                indices.data_ptr<const IndexT>(),                                                           \
                indptr.data_ptr<const OffsetT>(),                                                           \
                vector.data_ptr<const VALUE_T>(),                                                           \
                dweight.data_ptr<VALUE_T>(),                                                                \
                task_begin_output.data_ptr<OffsetT>(),                                                      \
                task_end_output.data_ptr<OffsetT>(),                                                        \
                task_value_output.data_ptr<VALUE_T>(),                                                      \
                status_output.data_ptr<int32_t>(),                                                          \
                static_cast<int>(indptr.numel()) - 1,                                                       \
                static_cast<int64_t>(indices.numel()),                                                      \
                task_capacity,                                                                              \
                cuda_stream);                                                                               \
        });                                                                                                 \
        BE_HYBRID_SDDMV_CHECK(launch_status == cudaSuccess,                                                 \
                              "hybrid_sddmv_weighted" #SUFFIX " launch failed");                            \
    }

// @BE hybrid_sddmv_weighted_f16
DEFINE_HYBRID_SDDMV_WEIGHTED(_f16, __half)

// @BE hybrid_sddmv_weighted_bf16
DEFINE_HYBRID_SDDMV_WEIGHTED(_bf16, __nv_bfloat16)

// @BE hybrid_sddmv_weighted_f32
DEFINE_HYBRID_SDDMV_WEIGHTED(_f32, float)

// @BE hybrid_sddmv_weighted_f64
DEFINE_HYBRID_SDDMV_WEIGHTED(_f64, double)

#endif // BE_SDDMV_STANDALONE
