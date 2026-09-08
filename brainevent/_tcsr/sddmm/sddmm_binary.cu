// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "cuda_common.h"
#include "brainevent/common.h"
#include "brainevent/dispatch.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace
{

    constexpr int kThreadsPerBlock = 256;
    constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;
    constexpr int kGroupSize = 16;
    constexpr int kGroupsPerBlock = kThreadsPerBlock / kGroupSize;
    constexpr int kBaseTileSize = 8192;
    constexpr int kBaseTilesPerMacro = 4;
    constexpr int kBatchPerPhase = 128;
    constexpr int kWordsPerPhase = kBatchPerPhase / 32;
    constexpr int kRowsPerChunk = 128;

    template <typename EventT>
    __device__ __forceinline__ bool is_active(EventT value);

    template <typename T>
    struct TensorDType;

    template <>
    struct TensorDType<bool>
    {
        static constexpr BE::DType value = BE::DType::Bool;
    };

    template <>
    struct TensorDType<int8_t>
    {
        static constexpr BE::DType value = BE::DType::Int8;
    };

    template <>
    struct TensorDType<float>
    {
        static constexpr BE::DType value = BE::DType::Float32;
    };

    template <>
    struct TensorDType<double>
    {
        static constexpr BE::DType value = BE::DType::Float64;
    };

    template <>
    __device__ __forceinline__ bool is_active<bool>(bool value)
    {
        return value;
    }

    template <>
    __device__ __forceinline__ bool is_active<float>(float value)
    {
        return value > 0.0f;
    }

    template <>
    __device__ __forceinline__ bool is_active<int8_t>(int8_t value)
    {
        return value > 0;
    }

    template <>
    __device__ __forceinline__ bool is_active<double>(double value)
    {
        return value > 0.0;
    }

    template <int Batch>
    struct BatchTraits
    {
        static_assert(
            Batch == 4 || Batch == 8 || Batch == 16 || Batch == 32 ||
                Batch == 64 || Batch == 128 || Batch == 256 || Batch == 512,
            "unsupported persistent macro-tile SDDMM batch size");
        static constexpr int kPhases =
            (Batch + kBatchPerPhase - 1) / kBatchPerPhase;
    };

    template <int Batch, typename EventT>
    __global__ void build_phase_masks_kernel(
        const EventT *__restrict__ event,
        uint32_t *__restrict__ masks,
        int64_t rows)
    {
        constexpr int kPhases = BatchTraits<Batch>::kPhases;
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int64_t stride = static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;
        for (int64_t task =
                 static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
             task < rows * kPhases;
             task += stride)
        {
            const int phase = static_cast<int>(task / rows);
            const int64_t row = task - static_cast<int64_t>(phase) * rows;
#pragma unroll
            for (int word = 0; word < kWordsPerPhase; ++word)
            {
                const int batch_index =
                    phase * kBatchPerPhase + word * 32 + lane;
                const bool active = batch_index < Batch && is_active<EventT>(
                                                               event[static_cast<size_t>(row) * Batch + batch_index]);
                const uint32_t word_mask =
                    __ballot_sync(0xffffffffU, active);
                if (lane == 0)
                {
                    masks[(task * kWordsPerPhase) + word] = word_mask;
                }
            }
        }
    }

    template <int Batch, typename ValueT, typename IndptrT>
    __global__ void persistent_macrotile_phase_kernel(
        const ValueT *__restrict__ ct,
        const uint16_t *__restrict__ local_targets,
        const IndptrT *__restrict__ indptr,
        const int32_t *__restrict__ tile_offsets,
        const uint32_t *__restrict__ masks,
        ValueT *__restrict__ dweight,
        int64_t rows,
        int64_t cols,
        int tile_count,
        int macro_tile,
        int phase)
    {
        const int group = threadIdx.x / kGroupSize;
        const int group_lane = threadIdx.x & (kGroupSize - 1);
        const uint32_t group_mask =
            0xffffU << ((threadIdx.x & kGroupSize) == 0 ? 0 : kGroupSize);
        const int64_t chunks = (rows + kRowsPerChunk - 1) / kRowsPerChunk;
        const int boundaries = tile_count + 1;

        for (int64_t chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x)
        {
            const int64_t chunk_row_begin = chunk * kRowsPerChunk;
            const int64_t chunk_row_end =
                min(chunk_row_begin + kRowsPerChunk, rows);
            for (int64_t row = chunk_row_begin + group;
                 row < chunk_row_end;
                 row += kGroupsPerBlock)
            {
                const size_t mask_offset =
                    (static_cast<size_t>(phase) * rows + row) * kWordsPerPhase;
                uint32_t row_masks[kWordsPerPhase];
                uint32_t any_active = 0;
#pragma unroll
                for (int word = 0; word < kWordsPerPhase; ++word)
                {
                    uint32_t value = group_lane == 0 ? masks[mask_offset + word] : 0;
                    value = __shfl_sync(group_mask, value, 0, kGroupSize);
                    row_masks[word] = value;
                    any_active |= value;
                }
                if (any_active == 0)
                    continue;

                const IndptrT row_begin = indptr[row];
#pragma unroll
                for (int subtile = 0; subtile < kBaseTilesPerMacro; ++subtile)
                {
                    const int base_tile = macro_tile * kBaseTilesPerMacro + subtile;
                    if (base_tile >= tile_count)
                        continue;
                    const size_t boundary_offset =
                        static_cast<size_t>(row) * boundaries + base_tile;
                    const int begin = tile_offsets[boundary_offset];
                    const int end = tile_offsets[boundary_offset + 1];
                    const int tile_begin = base_tile * kBaseTileSize;
                    for (int relative = begin + group_lane;
                         relative < end;
                         relative += kGroupSize)
                    {
                        const IndptrT slot = row_begin + relative;
                        const int col = tile_begin + local_targets[slot];
                        ValueT partial = ValueT(0);
#pragma unroll
                        for (int word = 0; word < kWordsPerPhase; ++word)
                        {
                            uint32_t active_bits = row_masks[word];
                            while (active_bits != 0)
                            {
                                const int active_lane =
                                    __ffs(static_cast<int>(active_bits)) - 1;
                                const int batch_index =
                                    phase * kBatchPerPhase + word * 32 + active_lane;
                                partial += ct[static_cast<size_t>(batch_index) * cols + col];
                                active_bits &= active_bits - 1;
                            }
                        }
                        dweight[slot] += partial;
                    }
                }
            }
        }
    }

    template <int Batch, typename EventT, typename ValueT, typename IndptrT>
    void launch_fixed_batch(
        const BE::Tensor event,
        const BE::Tensor ct,
        const BE::Tensor local_targets,
        const BE::Tensor indptr,
        const BE::Tensor tile_offsets,
        BE::Tensor dweight,
        BE::Tensor masks,
        int64_t rows,
        int64_t cols,
        cudaStream_t stream)
    {
        BE_CHECK(masks.size(0) == BatchTraits<Batch>::kPhases &&
                 masks.size(1) == rows &&
                 masks.size(2) == kWordsPerPhase)
            << "persistent macro-tile mask scratch shape mismatch";
        if (dweight.numel() == 0)
            return;
        BE_CUDA_CHECK(cudaMemsetAsync(
            dweight.data_ptr<ValueT>(),
            0,
            static_cast<size_t>(dweight.numel()) * sizeof(ValueT),
            stream));

        int device = 0;
        int sm_count = 0;
        BE_CUDA_CHECK(cudaGetDevice(&device));
        BE_CUDA_CHECK(cudaDeviceGetAttribute(
            &sm_count, cudaDevAttrMultiProcessorCount, device));
        const int64_t row_chunks = (rows + kRowsPerChunk - 1) / kRowsPerChunk;
        BE_CHECK(row_chunks <= std::numeric_limits<int>::max())
            << "persistent macro-tile row-chunk grid exceeds CUDA launch range";
        const int persistent_blocks = static_cast<int>(row_chunks);
        const int64_t mask_tasks = rows * BatchTraits<Batch>::kPhases;
        const int mask_blocks = static_cast<int>(std::min<int64_t>(
            (mask_tasks + kWarpsPerBlock - 1) / kWarpsPerBlock,
            static_cast<int64_t>(sm_count * 2)));

        build_phase_masks_kernel<Batch, EventT>
            <<<mask_blocks, kThreadsPerBlock, 0, stream>>>(
                event.data_ptr<const EventT>(),
                masks.data_ptr<uint32_t>(),
                rows);
        BE_CHECK_KERNEL_LAUNCH();

        const int tile_count = static_cast<int>(tile_offsets.size(1) - 1);
        const int macro_tiles =
            (tile_count + kBaseTilesPerMacro - 1) / kBaseTilesPerMacro;
        for (int macro_tile = 0; macro_tile < macro_tiles; ++macro_tile)
        {
            for (int phase = 0; phase < BatchTraits<Batch>::kPhases; ++phase)
            {
                persistent_macrotile_phase_kernel<Batch, ValueT, IndptrT>
                    <<<persistent_blocks, kThreadsPerBlock, 0, stream>>>(
                        ct.data_ptr<const ValueT>(),
                        local_targets.data_ptr<const uint16_t>(),
                        indptr.data_ptr<const IndptrT>(),
                        tile_offsets.data_ptr<const int32_t>(),
                        masks.data_ptr<const uint32_t>(),
                        dweight.data_ptr<ValueT>(),
                        rows,
                        cols,
                        tile_count,
                        macro_tile,
                        phase);
                BE_CHECK_KERNEL_LAUNCH();
            }
        }
    }

    template <typename EventT, typename ValueT>
    void launch_persistent_macrotile(
        const BE::Tensor event,
        const BE::Tensor ct,
        const BE::Tensor local_targets,
        const BE::Tensor indptr,
        const BE::Tensor tile_offsets,
        BE::Tensor dweight,
        BE::Tensor masks,
        int64_t stream)
    {
        BE_CHECK(event.dtype() == TensorDType<EventT>::value)
            << "persistent macro-tile event dtype mismatch";
        BE_CHECK(ct.dtype() == TensorDType<ValueT>::value &&
                 dweight.dtype() == TensorDType<ValueT>::value)
            << "persistent macro-tile CT and output dtype mismatch";
        BE_CHECK(local_targets.dtype() == BE::DType::UInt16)
            << "persistent macro-tile expects uint16 local targets";
        BE_CHECK(tile_offsets.dtype() == BE::DType::Int32)
            << "persistent macro-tile expects int32 tile offsets";
        BE_CHECK(indptr.dtype() == BE::DType::Int32 ||
                 indptr.dtype() == BE::DType::Int64)
            << "persistent macro-tile expects int32 or int64 indptr";
        BE_CHECK(masks.dtype() == BE::DType::UInt32)
            << "persistent macro-tile expects uint32 mask scratch";
        BE_CHECK(event.ndim() == 2 && ct.ndim() == 2 &&
                 tile_offsets.ndim() == 2 && masks.ndim() == 3)
            << "persistent macro-tile dense and tile ranks are invalid";
        BE_CHECK(local_targets.ndim() == 1 && indptr.ndim() == 1 &&
                 dweight.ndim() == 1)
            << "persistent macro-tile CSR ranks are invalid";

        const int64_t rows = indptr.size(0) - 1;
        const int64_t batch = event.size(1);
        const int64_t cols = ct.size(1);
        BE_CHECK(rows > 0 && cols > 0)
            << "persistent macro-tile expects positive dense dimensions";
        BE_CHECK(event.size(0) == rows && ct.size(0) == batch)
            << "persistent macro-tile dense dimensions are inconsistent";
        BE_CHECK(local_targets.numel() == dweight.numel())
            << "persistent macro-tile slot arrays must match";
        const int64_t tile_count = (cols + kBaseTileSize - 1) / kBaseTileSize;
        BE_CHECK(tile_offsets.size(0) == rows &&
                 tile_offsets.size(1) == tile_count + 1)
            << "persistent macro-tile boundary shape mismatch";

        cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
#define LAUNCH_BATCH(BATCH)                                                 \
    BE_DISPATCH_CSR_INDPTR(indptr.dtype(), IndptrT, {                       \
        launch_fixed_batch<BATCH, EventT, ValueT, IndptrT>(                 \
            event, ct, local_targets, indptr, tile_offsets, dweight, masks, \
            rows, cols, cuda_stream);                                       \
    })
        switch (batch)
        {
        case 4:
            LAUNCH_BATCH(4);
            break;
        case 8:
            LAUNCH_BATCH(8);
            break;
        case 16:
            LAUNCH_BATCH(16);
            break;
        case 32:
            LAUNCH_BATCH(32);
            break;
        case 64:
            LAUNCH_BATCH(64);
            break;
        case 128:
            LAUNCH_BATCH(128);
            break;
        case 256:
            LAUNCH_BATCH(256);
            break;
        case 512:
            LAUNCH_BATCH(512);
            break;
        default:
            BE_CHECK(false) << "unsupported persistent macro-tile batch";
        }
#undef LAUNCH_BATCH
    }

} // namespace

// @BE tcsr_sddmm_dweight_binary_f32_bool_t arg arg arg arg arg ret ret stream
void tcsr_sddmm_dweight_binary_f32_bool_t(
    const BE::Tensor event,
    const BE::Tensor ct,
    const BE::Tensor local_targets,
    const BE::Tensor indptr,
    const BE::Tensor tile_offsets,
    BE::Tensor dweight,
    BE::Tensor masks,
    int64_t stream)
{
    launch_persistent_macrotile<bool, float>(
        event, ct, local_targets, indptr, tile_offsets, dweight, masks, stream);
}

// @BE tcsr_sddmm_dweight_binary_f32_float_t arg arg arg arg arg ret ret stream
void tcsr_sddmm_dweight_binary_f32_float_t(
    const BE::Tensor event,
    const BE::Tensor ct,
    const BE::Tensor local_targets,
    const BE::Tensor indptr,
    const BE::Tensor tile_offsets,
    BE::Tensor dweight,
    BE::Tensor masks,
    int64_t stream)
{
    launch_persistent_macrotile<float, float>(
        event, ct, local_targets, indptr, tile_offsets, dweight, masks, stream);
}

// @BE tcsr_sddmm_dweight_binary_f64_bool_t arg arg arg arg arg ret ret stream
void tcsr_sddmm_dweight_binary_f64_bool_t(
    const BE::Tensor event,
    const BE::Tensor ct,
    const BE::Tensor local_targets,
    const BE::Tensor indptr,
    const BE::Tensor tile_offsets,
    BE::Tensor dweight,
    BE::Tensor masks,
    int64_t stream)
{
    launch_persistent_macrotile<bool, double>(
        event, ct, local_targets, indptr, tile_offsets, dweight, masks, stream);
}

// @BE tcsr_sddmm_dweight_binary_f64_int8_t arg arg arg arg arg ret ret stream
void tcsr_sddmm_dweight_binary_f64_int8_t(
    const BE::Tensor event,
    const BE::Tensor ct,
    const BE::Tensor local_targets,
    const BE::Tensor indptr,
    const BE::Tensor tile_offsets,
    BE::Tensor dweight,
    BE::Tensor masks,
    int64_t stream)
{
    launch_persistent_macrotile<int8_t, double>(
        event, ct, local_targets, indptr, tile_offsets, dweight, masks, stream);
}

// @BE tcsr_sddmm_dweight_binary_f64_float_t arg arg arg arg arg ret ret stream
void tcsr_sddmm_dweight_binary_f64_float_t(
    const BE::Tensor event,
    const BE::Tensor ct,
    const BE::Tensor local_targets,
    const BE::Tensor indptr,
    const BE::Tensor tile_offsets,
    BE::Tensor dweight,
    BE::Tensor masks,
    int64_t stream)
{
    launch_persistent_macrotile<float, double>(
        event, ct, local_targets, indptr, tile_offsets, dweight, masks, stream);
}

// @BE tcsr_sddmm_dweight_binary_f64_double_t arg arg arg arg arg ret ret stream
void tcsr_sddmm_dweight_binary_f64_double_t(
    const BE::Tensor event,
    const BE::Tensor ct,
    const BE::Tensor local_targets,
    const BE::Tensor indptr,
    const BE::Tensor tile_offsets,
    BE::Tensor dweight,
    BE::Tensor masks,
    int64_t stream)
{
    launch_persistent_macrotile<double, double>(
        event, ct, local_targets, indptr, tile_offsets, dweight, masks, stream);
}
