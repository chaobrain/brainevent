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
#include "hybrid_sddmm_ffi.cuh"

#define DEFINE_BATCHED_V3_SDDMM_WEIGHTED(SUFFIX, VALUE_T)                                                     \
    void batched_v3_sddmm_weighted##SUFFIX(const BE::Tensor events, const BE::Tensor ct,                      \
                                           const BE::Tensor indices, const BE::Tensor indptr,                 \
                                           const BE::Tensor long_mask, const BE::Tensor dweight_in,           \
                                           BE::Tensor dweight, BE::Tensor workspace, BE::Tensor selected,     \
                                           int64_t group, int64_t chunk, int64_t short_count,                 \
                                           int64_t long_count, int64_t capacity, int64_t stream)              \
    {                                                                                                         \
        batched_v3_sddmm_ffi<VALUE_T, VALUE_T, op_traits::ValueTraits<VALUE_T>::dtype, true>(                \
            "batched_v3_sddmm_weighted" #SUFFIX, events, ct, indices, indptr, long_mask, dweight_in, dweight, \
            workspace, selected, group, chunk, short_count, long_count, capacity, stream);                    \
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
