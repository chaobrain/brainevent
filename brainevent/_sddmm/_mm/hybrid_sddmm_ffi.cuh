// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

#ifndef DEVING_BATCHED_V3_HYBRID_SDDMM_FFI_CUH
#define DEVING_BATCHED_V3_HYBRID_SDDMM_FFI_CUH

// JAX FFI implementation shared by the binary and weighted batched v3 SDDMM
// targets: the caller's stream, no output clearing (dweight aliases dweight_in).
// Each .cu names its targets with a one-line macro around batched_v3_sddmm_ffi,
// so the generated bindings see the exact signatures. Index dtypes
// (int32/int64) are dispatched at run time.
#include "hybrid_sddmm.cuh"
#include "brainevent/common.h"
#include "cuda_common.h"
#include "brainevent/op_traits.cuh"

#include <atomic>

namespace
{
    // SM count of the current device, queried once per device ordinal.
    inline int current_device_sms()
    {
        constexpr int kMaxDevices = 64;
        static std::atomic<int> cached[kMaxDevices];
        int device = 0;
        BE_CUDA_CHECK(cudaGetDevice(&device));
        int sms = device < kMaxDevices ? cached[device].load(std::memory_order_relaxed) : 0;
        if (sms == 0)
        {
            BE_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
            if (device < kMaxDevices)
                cached[device].store(sms, std::memory_order_relaxed);
        }
        return sms;
    }

    // Binary events are EventT with FFI dtype EventDType (bool and int8 share
    // int8_t storage but are distinct FFI dtypes); weighted events share ValueT.
    template <class ValueT, class EventT, BE::DType EventDType, bool Weighted>
    void batched_v3_sddmm_ffi(const char *name, const BE::Tensor &events, const BE::Tensor &ct,
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
        const Plan p{static_cast<int>(group), static_cast<int>(chunk), static_cast<int>(batch),
                     current_device_sms(), short_count, long_count, capacity};
        const int64_t scratch_bytes =
            workspace_bytes(p, Weighted ? sizeof(ValueT) : 0) + accumulator_bytes<ValueT>(nnz);
        BE_CHECK(static_cast<int64_t>(workspace.numel()) * 4 >= scratch_bytes &&
                 static_cast<int64_t>(selected.numel()) >= list_entries(static_cast<int>(rows), p))
            << name << ": scratch too small";
        BE_CHECK(dweight_in.data_ptr<const ValueT>() == dweight.data_ptr<ValueT>()) << name << ": FFI requires output alias";
        const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        BE_DISPATCH_SIGNED_INDEX_PAIR(indices.dtype(), indptr.dtype(), IndexT, OffsetT, {
            const auto *ev = events.data_ptr<const EventT>();
            const auto *values = ct.data_ptr<const ValueT>();
            const auto *idx = indices.data_ptr<const IndexT>();
            const auto *ptr = indptr.data_ptr<const OffsetT>();
            auto *out = dweight.data_ptr<ValueT>();
            const auto *mask = long_mask.data_ptr<const uint32_t>();
            auto *list = selected.data_ptr<int32_t>();
            auto *scratch = workspace.data_ptr<uint32_t>();
            if constexpr (Weighted)
            {
                BE_CUDA_CHECK((launch_weighted<ValueT, IndexT, OffsetT>(
                    ev, values, idx, ptr, out, mask, list, scratch, static_cast<int>(rows), static_cast<int>(cols),
                    nnz, p, cuda_stream)));
            }
            else
            {
                BE_CUDA_CHECK((launch<ValueT, EventT, IndexT, OffsetT>(
                    ev, values, idx, ptr, out, mask, list, scratch, static_cast<int>(rows), static_cast<int>(cols),
                    nnz, p, cuda_stream)));
            }
        });
    }
} // namespace

#endif // DEVING_BATCHED_V3_HYBRID_SDDMM_FFI_CUH
