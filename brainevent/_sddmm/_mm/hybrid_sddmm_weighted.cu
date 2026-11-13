// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

// Weighted SDDMM: the binary operator's schedule (hybrid_sddmm_impl.cuh) with QueueT = ValueT, so each
// active event value scales its contributions:
//   dweight[j] += sum_b events[b, row] * ct[b, indices[j]]   over events[b, row] != 0.
// Output is shared dweight, never zeroed. Types: events, ct and dweight share ValueT in
// {half, bfloat16, float, double}; indices and indptr int32/int64. Negative and NaN events are active.
// Event values follow the vw standard (no row lookup): short rows re-read events[b, row] from
// the short list's row; long-row tasks carry their value in the task value queue.
// Half types use the shared float32 product/accumulator path and round on final output write.
#include "hybrid_sddmm_impl.cuh"

namespace batched_v3
{
    template <class ValueT, class IndexT, class OffsetT>
    cudaError_t launch_weighted(const ValueT *events, const ValueT *ct, const IndexT *indices, const OffsetT *ptr,
                                ValueT *dweight, const uint32_t *mask, int32_t *selected, void *workspace, int rows,
                                int cols, int64_t nnz, const Plan &p, cudaStream_t stream)
    {
        return detail::launch_impl<ValueT, ValueT, ValueT, IndexT, OffsetT>(
            events, ct, indices, ptr, dweight, mask, selected, workspace, rows, cols, nnz, p, stream);
    }

    // Definitions live in this translation unit; ordinary C++ callers link these instances.
#define INSTANTIATE_WEIGHTED(V, I, O)                                                                   \
    template cudaError_t launch_weighted<V, I, O>(const V *, const V *, const I *, const O *, V *,      \
                                                  const uint32_t *, int32_t *, void *, int, int, int64_t, \
                                                  const Plan &, cudaStream_t);
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
} // namespace batched_v3

#ifndef BE_SDDMM_STANDALONE
// JAX FFI entry points for the batched v3 weighted SDDMM: the caller's stream, no output clearing.
// batched_v3_sddmm_weighted_{f16,bf16,f32,f64}: events, ct and dweight share the suffix dtype.
// Index dtypes (int32/int64) are dispatched at run time.
#include "brainevent/common.h"
#include "cuda_common.h"
#include "brainevent/op_traits.cuh"

namespace
{
    template <class ValueT>
    void batched_v3_sddmm_weighted_impl(const char *name, const BE::Tensor &events, const BE::Tensor &ct,
                                        const BE::Tensor &indices, const BE::Tensor &indptr,
                                        const BE::Tensor &long_mask, const BE::Tensor &dweight_in,
                                        BE::Tensor &dweight, BE::Tensor &workspace, BE::Tensor &selected,
                                        int64_t group, int64_t chunk, int64_t short_count, int64_t long_count,
                                        int64_t capacity, int64_t stream)
    {
        using namespace batched_v3;
        constexpr BE::DType value_dtype = op_traits::ValueTraits<ValueT>::dtype;
        BE_CHECK(events.dtype() == value_dtype && ct.dtype() == value_dtype &&
                 op_traits::is_signed_index(indices.dtype()) && op_traits::is_signed_index(indptr.dtype()) &&
                 long_mask.dtype() == BE::DType::UInt32 && dweight_in.dtype() == value_dtype &&
                 dweight.dtype() == value_dtype && workspace.dtype() == BE::DType::UInt32 &&
                 selected.dtype() == BE::DType::Int32)
            << name << ": invalid dtype";
        BE_CHECK(events.ndim() == 2 && ct.ndim() == 2 && indices.ndim() == 1 && indptr.ndim() == 1 &&
                 long_mask.ndim() == 1 && dweight_in.ndim() == 1 && dweight.ndim() == 1 && workspace.ndim() == 1 &&
                 selected.ndim() == 1)
            << name << ": invalid rank";
        const int64_t batch = events.size(0), rows = events.size(1), cols = ct.size(1), nnz = indices.numel();
        BE_CHECK(batch >= 1 && batch <= 65535 && rows >= 1 && rows <= INT32_MAX && cols >= 1 && cols <= INT32_MAX &&
                 ct.size(0) == batch && indptr.numel() == rows + 1 && long_mask.numel() == (rows + 31) / 32 &&
                 dweight.numel() == nnz && dweight_in.numel() == nnz)
            << name << ": invalid shape";
        BE_CHECK((group == 4 || group == 8 || group == 16 || group == 32) && chunk >= 1 && chunk <= batch &&
                 short_count >= 0 && long_count >= 0 && short_count + long_count <= rows && capacity >= 0)
            << name << ": invalid plan";
        int device = 0, sms = 0;
        BE_CUDA_CHECK(cudaGetDevice(&device));
        BE_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
        const Plan p{static_cast<int>(group), static_cast<int>(chunk), static_cast<int>(batch), static_cast<int>(sms),
                     short_count, long_count, capacity};
        const int64_t scratch_bytes = workspace_bytes(p, sizeof(ValueT)) + accumulator_bytes<ValueT>(nnz);
        BE_CHECK(static_cast<int64_t>(workspace.numel()) * 4 >= scratch_bytes &&
                 static_cast<int64_t>(selected.numel()) >= list_entries(static_cast<int>(rows), p))
            << name << ": scratch too small";
        BE_CHECK(dweight_in.data_ptr<const ValueT>() == dweight.data_ptr<ValueT>()) << name << ": FFI requires output alias";
        const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        BE_DISPATCH_SIGNED_INDEX_PAIR(indices.dtype(), indptr.dtype(), IndexT, OffsetT, {
            BE_CUDA_CHECK((launch_weighted<ValueT, IndexT, OffsetT>(
                events.data_ptr<const ValueT>(), ct.data_ptr<const ValueT>(), indices.data_ptr<const IndexT>(),
                indptr.data_ptr<const OffsetT>(), dweight.data_ptr<ValueT>(), long_mask.data_ptr<const uint32_t>(),
                selected.data_ptr<int32_t>(), workspace.data_ptr<uint32_t>(), static_cast<int>(rows),
                static_cast<int>(cols), nnz, p, cuda_stream)));
        });
    }
} // namespace

#define DEFINE_BATCHED_V3_SDDMM_WEIGHTED(SUFFIX, VALUE_T)                                                     \
    void batched_v3_sddmm_weighted##SUFFIX(const BE::Tensor events, const BE::Tensor ct,                      \
                                           const BE::Tensor indices, const BE::Tensor indptr,                 \
                                           const BE::Tensor long_mask, const BE::Tensor dweight_in,           \
                                           BE::Tensor dweight, BE::Tensor workspace, BE::Tensor selected,     \
                                           int64_t group, int64_t chunk, int64_t short_count,                 \
                                           int64_t long_count, int64_t capacity, int64_t stream)              \
    {                                                                                                         \
        batched_v3_sddmm_weighted_impl<VALUE_T>("batched_v3_sddmm_weighted" #SUFFIX, events, ct, indices,     \
                                                indptr, long_mask, dweight_in, dweight, workspace, selected,  \
                                                group, chunk, short_count, long_count, capacity, stream);     \
    }

// @BE batched_v3_sddmm_weighted_f16
DEFINE_BATCHED_V3_SDDMM_WEIGHTED(_f16, __half)
// @BE batched_v3_sddmm_weighted_bf16
DEFINE_BATCHED_V3_SDDMM_WEIGHTED(_bf16, __nv_bfloat16)
// @BE batched_v3_sddmm_weighted_f32
DEFINE_BATCHED_V3_SDDMM_WEIGHTED(_f32, float)
// @BE batched_v3_sddmm_weighted_f64
DEFINE_BATCHED_V3_SDDMM_WEIGHTED(_f64, double)

#undef DEFINE_BATCHED_V3_SDDMM_WEIGHTED

#endif // BE_SDDMM_STANDALONE
