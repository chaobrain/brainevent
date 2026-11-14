// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

// Binary SDDMM: compact independent batches, cooperate within rows, split long rows.
// Adapted from tile hybrid v3 scheduling; output is shared dweight, never zeroed.
// Types: ct/dweight ValueT in {half, bfloat16, float, double}; events int8 (bool) or float,
// with double events additionally instantiated for double ct/dweight. FFI exposes int8
// separately from bool for double ct/dweight; both use the same native int8_t launcher.
// active iff != 0; indices and indptr int32/int64. Kernels and the schedule live in
// hybrid_sddmm_impl.cuh (shared with hybrid_sddmm_weighted.cu); planning and sizing in hybrid_sddmm_plan.cuh.
#include "hybrid_sddmm_impl.cuh"

namespace batched_v3
{
    template <class ValueT, class EventT, class IndexT, class OffsetT>
    cudaError_t launch(const EventT *events, const ValueT *ct, const IndexT *indices, const OffsetT *ptr,
                       ValueT *dweight, const uint32_t *mask, int32_t *selected, void *workspace, int rows, int cols,
                       int64_t nnz, const Plan &p, cudaStream_t stream)
    {
        return detail::launch_impl<ValueT, EventT, void, IndexT, OffsetT>(
            events, ct, indices, ptr, dweight, mask, selected, workspace, rows, cols, nnz, p, stream);
    }

    // Definitions live in this translation unit; ordinary C++ callers link these instances.
#define INSTANTIATE_BINARY(V, E, I, O)                                                                     \
    template cudaError_t launch<V, E, I, O>(const E *, const V *, const I *, const O *, V *, const uint32_t *, \
                                            int32_t *, void *, int, int, int64_t, const Plan &, cudaStream_t);
#define INSTANTIATE_BINARY_INDICES(V, E)     \
    INSTANTIATE_BINARY(V, E, int32_t, int32_t) \
    INSTANTIATE_BINARY(V, E, int32_t, int64_t) \
    INSTANTIATE_BINARY(V, E, int64_t, int32_t) \
    INSTANTIATE_BINARY(V, E, int64_t, int64_t)
    INSTANTIATE_BINARY_INDICES(__half, int8_t)
    INSTANTIATE_BINARY_INDICES(__half, float)
    INSTANTIATE_BINARY_INDICES(__nv_bfloat16, int8_t)
    INSTANTIATE_BINARY_INDICES(__nv_bfloat16, float)
    INSTANTIATE_BINARY_INDICES(float, int8_t)
    INSTANTIATE_BINARY_INDICES(float, float)
    INSTANTIATE_BINARY_INDICES(double, int8_t)
    INSTANTIATE_BINARY_INDICES(double, float)
    INSTANTIATE_BINARY_INDICES(double, double)
#undef INSTANTIATE_BINARY_INDICES
#undef INSTANTIATE_BINARY
} // namespace batched_v3

#ifndef BE_SDDMM_STANDALONE
// JAX FFI entry points for the batched v3 binary SDDMM: the caller's stream, no output clearing.
// batched_v3_sddmm_{f16,bf16,f32,f64}_{bool,float}: the value suffix names the ct/dweight dtype,
// the event suffix the events dtype. f64 additionally exposes {int8,double} events.
// Index dtypes (int32/int64) are dispatched at run time.
#include "hybrid_sddmm_ffi.cuh"

#define DEFINE_BATCHED_V3_SDDMM(SUFFIX, VALUE_T, EVENT_T, EVENT_DTYPE)                                      \
    void batched_v3_sddmm##SUFFIX(const BE::Tensor events, const BE::Tensor ct, const BE::Tensor indices,   \
                                  const BE::Tensor indptr, const BE::Tensor long_mask,                      \
                                  const BE::Tensor dweight_in, BE::Tensor dweight, BE::Tensor workspace,    \
                                  BE::Tensor selected, int64_t group, int64_t chunk, int64_t short_count,   \
                                  int64_t long_count, int64_t capacity, int64_t stream)                     \
    {                                                                                                       \
        batched_v3_sddmm_ffi<VALUE_T, EVENT_T, BE::DType::EVENT_DTYPE, false>(                              \
            "batched_v3_sddmm" #SUFFIX, events, ct, indices, indptr, long_mask, dweight_in, dweight,          \
            workspace, selected, group, chunk, short_count, long_count, capacity, stream);                    \
    }

// @BE batched_v3_sddmm_f16_bool
DEFINE_BATCHED_V3_SDDMM(_f16_bool, __half, int8_t, Bool)
// @BE batched_v3_sddmm_f16_float
DEFINE_BATCHED_V3_SDDMM(_f16_float, __half, float, Float32)
// @BE batched_v3_sddmm_bf16_bool
DEFINE_BATCHED_V3_SDDMM(_bf16_bool, __nv_bfloat16, int8_t, Bool)
// @BE batched_v3_sddmm_bf16_float
DEFINE_BATCHED_V3_SDDMM(_bf16_float, __nv_bfloat16, float, Float32)
// @BE batched_v3_sddmm_f32_bool
DEFINE_BATCHED_V3_SDDMM(_f32_bool, float, int8_t, Bool)
// @BE batched_v3_sddmm_f32_float
DEFINE_BATCHED_V3_SDDMM(_f32_float, float, float, Float32)
// @BE batched_v3_sddmm_f64_bool
DEFINE_BATCHED_V3_SDDMM(_f64_bool, double, int8_t, Bool)
// @BE batched_v3_sddmm_f64_int8
DEFINE_BATCHED_V3_SDDMM(_f64_int8, double, int8_t, Int8)
// @BE batched_v3_sddmm_f64_float
DEFINE_BATCHED_V3_SDDMM(_f64_float, double, float, Float32)
// @BE batched_v3_sddmm_f64_double
DEFINE_BATCHED_V3_SDDMM(_f64_double, double, double, Float64)

#undef DEFINE_BATCHED_V3_SDDMM

#endif // BE_SDDMM_STANDALONE
