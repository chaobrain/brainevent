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

    template <typename ValueT>
    struct TensorDType;

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

    template <typename ValueT>
    __device__ __forceinline__ bool is_active(ValueT value)
    {
        return value != ValueT(0);
    }

    template <int Batch>
    struct BatchTraits
    {
        static_assert(
            Batch == 4 || Batch == 8 || Batch == 16 || Batch == 32 ||
                Batch == 64 || Batch == 128 || Batch == 256 || Batch == 512,
            "unsupported Batch-N float SDDMM batch size");
        static constexpr int kPhases =
            (Batch + kBatchPerPhase - 1) / kBatchPerPhase;
    };

    template <int Batch, typename ValueT>
    __global__ void build_phase_masks_kernel(
        const ValueT *__restrict__ B,
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
                const bool active = batch_index < Batch && is_active(
                                                               B[static_cast<size_t>(row) * Batch + batch_index]);
                const uint32_t word_mask =
                    __ballot_sync(0xffffffffU, active);
                if (lane == 0)
                {
                    masks[(task * kWordsPerPhase) + word] = word_mask;
                }
            }
        }
    }

    template <int Batch>
    __global__ void find_first_active_phase_kernel(
        const uint32_t *__restrict__ masks,
        uint8_t *__restrict__ first_phase,
        int64_t rows)
    {
        constexpr int kPhases = BatchTraits<Batch>::kPhases;
        const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
        for (int64_t row =
                 static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             row < rows;
             row += stride)
        {
            uint8_t first = 0xffU;
#pragma unroll
            for (int phase = 0; phase < kPhases; ++phase)
            {
                const size_t mask_offset =
                    (static_cast<size_t>(phase) * rows + row) * kWordsPerPhase;
                uint32_t any_active = 0;
#pragma unroll
                for (int word = 0; word < kWordsPerPhase; ++word)
                {
                    any_active |= masks[mask_offset + word];
                }
                if (any_active != 0)
                {
                    first = static_cast<uint8_t>(phase);
                    break;
                }
            }
            first_phase[row] = first;
        }
    }

    template <int Batch>
    __global__ void compact_active_rows_kernel(
        const uint32_t *__restrict__ masks,
        uint16_t *__restrict__ active_counts,
        uint8_t *__restrict__ active_rows,
        int64_t rows)
    {
        constexpr int kPhases = BatchTraits<Batch>::kPhases;
        __shared__ int warp_counts[4];
        __shared__ int warp_offsets[4];

        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int64_t row_chunks = (rows + kRowsPerChunk - 1) / kRowsPerChunk;
        const int64_t task = blockIdx.x;
        if (task >= static_cast<int64_t>(kPhases) * row_chunks)
            return;
        const int phase = static_cast<int>(task / row_chunks);
        const int64_t chunk = task - static_cast<int64_t>(phase) * row_chunks;
        const int64_t row = chunk * kRowsPerChunk + threadIdx.x;

        uint32_t any_active = 0;
        if (row < rows)
        {
            const size_t mask_offset =
                (static_cast<size_t>(phase) * rows + row) * kWordsPerPhase;
#pragma unroll
            for (int word = 0; word < kWordsPerPhase; ++word)
            {
                any_active |= masks[mask_offset + word];
            }
        }
        const uint32_t active_mask =
            __ballot_sync(0xffffffffU, any_active != 0);
        const uint32_t lower_lanes =
            lane == 0 ? 0U : ((1U << lane) - 1U);
        const int rank_in_warp = __popc(active_mask & lower_lanes);
        if (lane == 0)
            warp_counts[warp] = __popc(active_mask);
        __syncthreads();

        if (threadIdx.x < 4)
        {
            int offset = 0;
            for (int previous = 0; previous < threadIdx.x; ++previous)
                offset += warp_counts[previous];
            warp_offsets[threadIdx.x] = offset;
        }
        __syncthreads();

        const int rank = warp_offsets[warp] + rank_in_warp;
        const size_t active_offset = static_cast<size_t>(task) * kRowsPerChunk;
        if (any_active != 0)
            active_rows[active_offset + rank] = static_cast<uint8_t>(threadIdx.x);
        if (threadIdx.x == 0)
        {
            active_counts[task] = static_cast<uint16_t>(
                warp_counts[0] + warp_counts[1] + warp_counts[2] + warp_counts[3]);
        }
    }

    template <int Batch, typename ValueT, typename IndptrT>
    __global__ void b_bit_driven_kernel(
        const ValueT *__restrict__ B,
        const ValueT *__restrict__ ct,
        const uint16_t *__restrict__ local_targets,
        const IndptrT *__restrict__ indptr,
        const int32_t *__restrict__ tile_offsets,
        const uint32_t *__restrict__ masks,
        const uint8_t *__restrict__ first_phase,
        const uint16_t *__restrict__ active_counts,
        const uint8_t *__restrict__ active_rows,
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
            const int row_tasks = static_cast<int>(active_counts[static_cast<size_t>(phase) * chunks + chunk]);
            for (int position = group;
                 position < row_tasks;
                 position += kGroupsPerBlock)
            {
                const size_t list_offset =
                    (static_cast<size_t>(phase) * chunks + chunk) *
                        kRowsPerChunk +
                    position;
                const int lane_row_offset = group_lane == 0
                                                ? static_cast<int>(active_rows[list_offset])
                                                : 0;
                const int row_offset = __shfl_sync(
                    group_mask, lane_row_offset, 0, kGroupSize);
                const int64_t row = chunk_row_begin + row_offset;
                const size_t mask_offset =
                    (static_cast<size_t>(phase) * rows + row) * kWordsPerPhase;
                uint32_t row_masks[kWordsPerPhase];
#pragma unroll
                for (int word = 0; word < kWordsPerPhase; ++word)
                {
                    uint32_t value =
                        group_lane == 0 ? masks[mask_offset + word] : 0;
                    row_masks[word] = __shfl_sync(
                        group_mask, value, 0, kGroupSize);
                }

                const uint8_t lane_first_phase = group_lane == 0
                                                     ? first_phase[row]
                                                     : 0xffU;
                const uint8_t row_first_phase = static_cast<uint8_t>(__shfl_sync(
                    group_mask,
                    static_cast<unsigned int>(lane_first_phase),
                    0,
                    kGroupSize));
                bool first_bit = row_first_phase == static_cast<uint8_t>(phase);
                const IndptrT row_begin = indptr[row];

#pragma unroll
                for (int word = 0; word < kWordsPerPhase; ++word)
                {
                    uint32_t active_bits = row_masks[word];
                    while (active_bits != 0)
                    {
                        const int bit = __ffs(static_cast<int>(active_bits)) - 1;
                        const int batch_index =
                            phase * kBatchPerPhase + word * 32 + bit;
                        const ValueT lane_B_value = group_lane == 0
                                                        ? B[static_cast<size_t>(row) * Batch + batch_index]
                                                        : ValueT(0);
                        const ValueT B_value =
                            __shfl_sync(group_mask, lane_B_value, 0, kGroupSize);

#pragma unroll
                        for (int subtile = 0;
                             subtile < kBaseTilesPerMacro;
                             ++subtile)
                        {
                            const int base_tile =
                                macro_tile * kBaseTilesPerMacro + subtile;
                            if (base_tile >= tile_count)
                                continue;
                            const size_t boundary_offset =
                                static_cast<size_t>(row) * boundaries + base_tile;
                            const int begin = tile_offsets[boundary_offset];
                            const int end = tile_offsets[boundary_offset + 1];
                            const int tile_begin = base_tile * kBaseTileSize;
                            for (int relative0 = begin + group_lane;
                                 relative0 < end;
                                 relative0 += 2 * kGroupSize)
                            {
                                const int relative1 = relative0 + kGroupSize;
                                const bool valid1 = relative1 < end;
                                const IndptrT slot0 = row_begin + relative0;
                                const IndptrT slot1 = row_begin + relative1;
                                const int col0 =
                                    tile_begin + local_targets[slot0];
                                const int col1 = valid1
                                                     ? tile_begin + local_targets[slot1]
                                                     : 0;
                                const size_t ct_offset =
                                    static_cast<size_t>(batch_index) * cols;
                                const ValueT value0 = B_value * ct[ct_offset + col0];
                                const ValueT value1 = valid1
                                                          ? B_value * ct[ct_offset + col1]
                                                          : ValueT(0);
                                if (first_bit)
                                {
                                    dweight[slot0] = value0;
                                    if (valid1)
                                        dweight[slot1] = value1;
                                }
                                else
                                {
                                    dweight[slot0] += value0;
                                    if (valid1)
                                        dweight[slot1] += value1;
                                }
                            }
                        }
                        first_bit = false;
                        active_bits &= active_bits - 1;
                    }
                }
            }
        }
    }

    template <int Batch, typename ValueT, typename IndptrT>
    void launch_fixed_batch(
        const BE::Tensor B,
        const BE::Tensor ct,
        const BE::Tensor local_targets,
        const BE::Tensor indptr,
        const BE::Tensor tile_offsets,
        BE::Tensor dweight,
        BE::Tensor masks,
        BE::Tensor first_phase,
        BE::Tensor active_counts,
        BE::Tensor active_rows,
        int64_t rows,
        int64_t cols,
        cudaStream_t stream)
    {
        constexpr int kPhases = BatchTraits<Batch>::kPhases;
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
            << "Batch-N float row-chunk grid exceeds CUDA launch range";
        const int persistent_blocks = static_cast<int>(row_chunks);
        const int64_t mask_tasks = rows * kPhases;
        const int mask_blocks = static_cast<int>(std::min<int64_t>(
            (mask_tasks + kWarpsPerBlock - 1) / kWarpsPerBlock,
            static_cast<int64_t>(sm_count * 2)));

        build_phase_masks_kernel<Batch, ValueT>
            <<<mask_blocks, kThreadsPerBlock, 0, stream>>>(
                B.data_ptr<const ValueT>(),
                masks.data_ptr<uint32_t>(),
                rows);
        BE_CHECK_KERNEL_LAUNCH();

        const int first_blocks = static_cast<int>(std::min<int64_t>(
            (rows + kThreadsPerBlock - 1) / kThreadsPerBlock,
            static_cast<int64_t>(sm_count * 2)));
        find_first_active_phase_kernel<Batch>
            <<<first_blocks, kThreadsPerBlock, 0, stream>>>(
                masks.data_ptr<const uint32_t>(),
                first_phase.data_ptr<uint8_t>(),
                rows);
        BE_CHECK_KERNEL_LAUNCH();

        const int64_t compact_tasks = row_chunks * kPhases;
        BE_CHECK(compact_tasks <= std::numeric_limits<int>::max())
            << "Batch-N float compaction grid exceeds CUDA launch range";
        compact_active_rows_kernel<Batch>
            <<<static_cast<int>(compact_tasks), kRowsPerChunk, 0, stream>>>(
                masks.data_ptr<const uint32_t>(),
                active_counts.data_ptr<uint16_t>(),
                active_rows.data_ptr<uint8_t>(),
                rows);
        BE_CHECK_KERNEL_LAUNCH();

        const int tile_count = static_cast<int>(tile_offsets.size(1) - 1);
        const int macro_tiles =
            (tile_count + kBaseTilesPerMacro - 1) / kBaseTilesPerMacro;
        for (int macro_tile = 0; macro_tile < macro_tiles; ++macro_tile)
        {
            for (int phase = 0; phase < kPhases; ++phase)
            {
                b_bit_driven_kernel<Batch, ValueT, IndptrT>
                    <<<persistent_blocks, kThreadsPerBlock, 0, stream>>>(
                        B.data_ptr<const ValueT>(),
                        ct.data_ptr<const ValueT>(),
                        local_targets.data_ptr<const uint16_t>(),
                        indptr.data_ptr<const IndptrT>(),
                        tile_offsets.data_ptr<const int32_t>(),
                        masks.data_ptr<const uint32_t>(),
                        first_phase.data_ptr<const uint8_t>(),
                        active_counts.data_ptr<const uint16_t>(),
                        active_rows.data_ptr<const uint8_t>(),
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

    template <typename ValueT>
    void launch_sddmm_float(
        const BE::Tensor B,
        const BE::Tensor ct,
        const BE::Tensor local_targets,
        const BE::Tensor indptr,
        const BE::Tensor tile_offsets,
        BE::Tensor dweight,
        BE::Tensor masks,
        BE::Tensor first_phase,
        BE::Tensor active_counts,
        BE::Tensor active_rows,
        int64_t stream)
    {
        BE_CHECK(B.dtype() == TensorDType<ValueT>::value &&
                 ct.dtype() == TensorDType<ValueT>::value &&
                 dweight.dtype() == TensorDType<ValueT>::value)
            << "Batch-N float SDDMM dense value and output dtype mismatch";
        BE_CHECK(local_targets.dtype() == BE::DType::UInt16)
            << "Batch-N float SDDMM expects uint16 local targets";
        BE_CHECK(tile_offsets.dtype() == BE::DType::Int32)
            << "Batch-N float SDDMM expects int32 tile offsets";
        BE_CHECK(indptr.dtype() == BE::DType::Int32 ||
                 indptr.dtype() == BE::DType::Int64)
            << "Batch-N float SDDMM expects int32 or int64 indptr";
        BE_CHECK(masks.dtype() == BE::DType::UInt32 &&
                 first_phase.dtype() == BE::DType::UInt8 &&
                 active_counts.dtype() == BE::DType::UInt16 &&
                 active_rows.dtype() == BE::DType::UInt8)
            << "Batch-N float SDDMM scratch dtype mismatch";

        BE_CHECK(B.ndim() == 2 && ct.ndim() == 2 &&
                 tile_offsets.ndim() == 2 && masks.ndim() == 3 &&
                 active_counts.ndim() == 2 && active_rows.ndim() == 3)
            << "Batch-N float SDDMM dense or scratch rank mismatch";
        BE_CHECK(local_targets.ndim() == 1 && indptr.ndim() == 1 &&
                 dweight.ndim() == 1 && first_phase.ndim() == 1)
            << "Batch-N float SDDMM sparse rank mismatch";

        const int64_t rows = indptr.size(0) - 1;
        const int64_t batch = B.size(1);
        const int64_t cols = ct.size(1);
        BE_CHECK(rows > 0 && cols > 0)
            << "Batch-N float SDDMM expects positive dense dimensions";
        BE_CHECK(
            batch == 4 || batch == 8 || batch == 16 || batch == 32 ||
            batch == 64 || batch == 128 || batch == 256 || batch == 512)
            << "Batch-N float SDDMM received an unsupported batch size";
        BE_CHECK(B.size(0) == rows && ct.size(0) == batch)
            << "Batch-N float SDDMM dense dimensions are inconsistent";
        BE_CHECK(local_targets.numel() == dweight.numel())
            << "Batch-N float SDDMM slot arrays must match";

        const int64_t tile_count = (cols + kBaseTileSize - 1) / kBaseTileSize;
        const int64_t row_chunks = (rows + kRowsPerChunk - 1) / kRowsPerChunk;
        const int64_t phases = (batch + kBatchPerPhase - 1) / kBatchPerPhase;
        BE_CHECK(tile_offsets.size(0) == rows &&
                 tile_offsets.size(1) == tile_count + 1)
            << "batched float SDDMM tile boundary shape mismatch";
        BE_CHECK(masks.size(0) == phases && masks.size(1) == rows &&
                 masks.size(2) == kWordsPerPhase && first_phase.size(0) == rows)
            << "batched float SDDMM mask scratch shape mismatch";
        BE_CHECK(active_counts.size(0) == phases &&
                 active_counts.size(1) == row_chunks &&
                 active_rows.size(0) == phases &&
                 active_rows.size(1) == row_chunks &&
                 active_rows.size(2) == kRowsPerChunk)
            << "batched float SDDMM compaction scratch shape mismatch";

        cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
#define LAUNCH_BATCH(BATCH)                                                    \
    BE_DISPATCH_CSR_INDPTR(indptr.dtype(), IndptrT, {                          \
        launch_fixed_batch<BATCH, ValueT, IndptrT>(                            \
            B, ct, local_targets, indptr, tile_offsets, dweight, masks,        \
            first_phase, active_counts, active_rows, rows, cols, cuda_stream); \
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
            BE_CHECK(false) << "unsupported batched float SDDMM batch";
        }
#undef LAUNCH_BATCH
    }

} // namespace

// @BE tcsr_sddmm_dweight_float_f32_t arg arg arg arg arg ret ret ret ret ret stream
void tcsr_sddmm_dweight_float_f32_t(
    const BE::Tensor B,
    const BE::Tensor ct,
    const BE::Tensor local_targets,
    const BE::Tensor indptr,
    const BE::Tensor tile_offsets,
    BE::Tensor dweight,
    BE::Tensor masks,
    BE::Tensor first_phase,
    BE::Tensor active_counts,
    BE::Tensor active_rows,
    int64_t stream)
{
    launch_sddmm_float<float>(
        B,
        ct,
        local_targets,
        indptr,
        tile_offsets,
        dweight,
        masks,
        first_phase,
        active_counts,
        active_rows,
        stream);
}

// @BE tcsr_sddmm_dweight_float_f64_t arg arg arg arg arg ret ret ret ret ret stream
void tcsr_sddmm_dweight_float_f64_t(
    const BE::Tensor B,
    const BE::Tensor ct,
    const BE::Tensor local_targets,
    const BE::Tensor indptr,
    const BE::Tensor tile_offsets,
    BE::Tensor dweight,
    BE::Tensor masks,
    BE::Tensor first_phase,
    BE::Tensor active_counts,
    BE::Tensor active_rows,
    int64_t stream)
{
    launch_sddmm_float<double>(
        B,
        ct,
        local_targets,
        indptr,
        tile_offsets,
        dweight,
        masks,
        first_phase,
        active_counts,
        active_rows,
        stream);
}
