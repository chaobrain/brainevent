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
#include "brainevent/common.h"
#include "cuda_common.h"
#include "brainevent/op_traits.cuh"

namespace
{
    // Bool and Int8 share int8_t storage, but are distinct FFI dtypes. Specify
    // the event dtype per entry point without changing shared op_traits mappings.
    template <class ValueT, class EventT, BE::DType EventDType>
    void batched_v3_sddmm_impl(const char *name, const BE::Tensor &events, const BE::Tensor &ct,
                               const BE::Tensor &indices, const BE::Tensor &indptr, const BE::Tensor &long_mask,
                               const BE::Tensor &dweight_in, BE::Tensor &dweight, BE::Tensor &workspace,
                               BE::Tensor &selected, int64_t group, int64_t chunk, int64_t short_count,
                               int64_t long_count, int64_t capacity, int64_t stream)
    {
        using namespace batched_v3;
        constexpr BE::DType value_dtype = op_traits::ValueTraits<ValueT>::dtype;
        BE_CHECK(events.dtype() == EventDType && ct.dtype() == value_dtype &&
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
        const int64_t scratch_bytes = workspace_bytes(p) + accumulator_bytes<ValueT>(nnz);
        BE_CHECK(static_cast<int64_t>(workspace.numel()) * 4 >= scratch_bytes &&
                 static_cast<int64_t>(selected.numel()) >= list_entries(static_cast<int>(rows), p))
            << name << ": scratch too small";
        BE_CHECK(dweight_in.data_ptr<const ValueT>() == dweight.data_ptr<ValueT>()) << name << ": FFI requires output alias";
        const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        BE_DISPATCH_SIGNED_INDEX_PAIR(indices.dtype(), indptr.dtype(), IndexT, OffsetT, {
            BE_CUDA_CHECK((launch<ValueT, EventT, IndexT, OffsetT>(
                events.data_ptr<const EventT>(), ct.data_ptr<const ValueT>(), indices.data_ptr<const IndexT>(),
                indptr.data_ptr<const OffsetT>(), dweight.data_ptr<ValueT>(), long_mask.data_ptr<const uint32_t>(),
                selected.data_ptr<int32_t>(), workspace.data_ptr<uint32_t>(), static_cast<int>(rows),
                static_cast<int>(cols), nnz, p, cuda_stream)));
        });
    }
} // namespace

#define DEFINE_BATCHED_V3_SDDMM(SUFFIX, VALUE_T, EVENT_T, EVENT_DTYPE)                                      \
    void batched_v3_sddmm##SUFFIX(const BE::Tensor events, const BE::Tensor ct, const BE::Tensor indices,   \
                                  const BE::Tensor indptr, const BE::Tensor long_mask,                      \
                                  const BE::Tensor dweight_in, BE::Tensor dweight, BE::Tensor workspace,    \
                                  BE::Tensor selected, int64_t group, int64_t chunk, int64_t short_count,   \
                                  int64_t long_count, int64_t capacity, int64_t stream)                     \
    {                                                                                                       \
        batched_v3_sddmm_impl<VALUE_T, EVENT_T, BE::DType::EVENT_DTYPE>(                                    \
                                                "batched_v3_sddmm" #SUFFIX, events, ct, indices, indptr, \
                                                long_mask, dweight_in, dweight, workspace, selected, group, \
                                                chunk, short_count, long_count, capacity, stream);          \
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
