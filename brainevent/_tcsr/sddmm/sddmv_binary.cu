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
    constexpr int kRowsPerCompactBlock = kThreadsPerBlock;
    constexpr int kBaseTileSize = 8192;
    constexpr int kBaseTilesPerMacro = 4;

    template <typename EventT>
    __device__ __forceinline__ bool event_active(EventT value);

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
    __device__ __forceinline__ bool event_active<bool>(bool value)
    {
        return value;
    }

    template <>
    __device__ __forceinline__ bool event_active<float>(float value)
    {
        return value > 0.0f;
    }

    template <>
    __device__ __forceinline__ bool event_active<int8_t>(int8_t value)
    {
        return value > 0;
    }

    template <>
    __device__ __forceinline__ bool event_active<double>(double value)
    {
        return value > 0.0;
    }

    template <typename EventT>
    __global__ void compact_active_rows_kernel(
        const EventT *__restrict__ event,
        int32_t *__restrict__ active_rows,
        int32_t *__restrict__ active_count,
        int64_t rows)
    {
        __shared__ int warp_counts[kWarpsPerBlock];
        __shared__ int warp_offsets[kWarpsPerBlock];
        __shared__ int block_output_begin;

        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int64_t row =
            static_cast<int64_t>(blockIdx.x) * kRowsPerCompactBlock + threadIdx.x;
        const bool active = row < rows && event_active<EventT>(event[row]);
        const uint32_t active_bits = __ballot_sync(0xffffffffU, active);
        const uint32_t lower_lanes =
            lane == 0 ? 0U : ((1U << lane) - 1U);
        const int rank_in_warp = __popc(active_bits & lower_lanes);

        if (lane == 0)
            warp_counts[warp] = __popc(active_bits);
        __syncthreads();

        if (threadIdx.x == 0)
        {
            int block_count = 0;
#pragma unroll
            for (int previous_warp = 0;
                 previous_warp < kWarpsPerBlock;
                 ++previous_warp)
            {
                warp_offsets[previous_warp] = block_count;
                block_count += warp_counts[previous_warp];
            }
            block_output_begin = block_count == 0
                                     ? 0
                                     : atomicAdd(active_count, block_count);
        }
        __syncthreads();

        if (active)
        {
            const int output =
                block_output_begin + warp_offsets[warp] + rank_in_warp;
            active_rows[output] = static_cast<int32_t>(row);
        }
    }

    template <typename ValueT, typename IndptrT>
    __global__ void single_sddmm_panel_kernel(
        const ValueT *__restrict__ ct,
        const uint16_t *__restrict__ local_targets,
        const IndptrT *__restrict__ indptr,
        const int32_t *__restrict__ tile_offsets,
        const int32_t *__restrict__ active_rows,
        const int32_t *__restrict__ active_count,
        ValueT *__restrict__ dweight,
        int tile_count,
        int macro_tile)
    {
        const int group = threadIdx.x / kGroupSize;
        const int group_lane = threadIdx.x & (kGroupSize - 1);
        const int count = *active_count;
        const int boundaries = tile_count + 1;
        const int task_stride = gridDim.x * kGroupsPerBlock;

        for (int position = blockIdx.x * kGroupsPerBlock + group;
             position < count;
             position += task_stride)
        {
            const int32_t row = active_rows[position];
            const IndptrT row_begin = indptr[row];
#pragma unroll
            for (int subtile = 0; subtile < kBaseTilesPerMacro; ++subtile)
            {
                const int base_tile = macro_tile * kBaseTilesPerMacro + subtile;
                if (base_tile >= tile_count)
                    continue;
                const size_t boundary_offset =
                    static_cast<size_t>(row) * boundaries + base_tile;
                const int relative_begin = tile_offsets[boundary_offset];
                const int relative_end = tile_offsets[boundary_offset + 1];
                const int tile_begin = base_tile * kBaseTileSize;

                for (int relative0 = relative_begin + group_lane;
                     relative0 < relative_end;
                     relative0 += 2 * kGroupSize)
                {
                    const int relative1 = relative0 + kGroupSize;
                    const bool valid1 = relative1 < relative_end;
                    const IndptrT slot0 = row_begin + relative0;
                    const IndptrT slot1 = row_begin + relative1;
                    const int col0 = tile_begin + local_targets[slot0];
                    const int col1 = valid1
                                         ? tile_begin + local_targets[slot1]
                                         : 0;
                    dweight[slot0] = ct[col0];
                    if (valid1)
                        dweight[slot1] = ct[col1];
                }
            }
        }
    }

    template <typename EventT, typename ValueT>
    void launch_single_sddmm(
        const BE::Tensor event,
        const BE::Tensor ct,
        const BE::Tensor local_targets,
        const BE::Tensor indptr,
        const BE::Tensor tile_offsets,
        BE::Tensor dweight,
        BE::Tensor active_rows,
        BE::Tensor active_count,
        int64_t stream)
    {
        BE_CHECK(event.dtype() == TensorDType<EventT>::value)
            << "single-batch TileCSR SDDMM event dtype mismatch";
        BE_CHECK(ct.dtype() == TensorDType<ValueT>::value &&
                 dweight.dtype() == TensorDType<ValueT>::value)
            << "single-batch TileCSR SDDMM CT and dweight dtype mismatch";
        BE_CHECK(local_targets.dtype() == BE::DType::UInt16)
            << "single-batch TileCSR SDDMM expects uint16 local targets";
        BE_CHECK(tile_offsets.dtype() == BE::DType::Int32)
            << "single-batch TileCSR SDDMM expects int32 tile offsets";
        BE_CHECK(indptr.dtype() == BE::DType::Int32 ||
                 indptr.dtype() == BE::DType::Int64)
            << "single-batch TileCSR SDDMM expects int32 or int64 indptr";
        BE_CHECK(active_rows.dtype() == BE::DType::Int32 &&
                 active_count.dtype() == BE::DType::Int32)
            << "single-batch TileCSR SDDMM active scratch must be int32";

        BE_CHECK(event.ndim() == 1 && ct.ndim() == 1 &&
                 local_targets.ndim() == 1 && indptr.ndim() == 1 &&
                 dweight.ndim() == 1 && active_rows.ndim() == 1 &&
                 active_count.ndim() == 1)
            << "single-batch TileCSR SDDMM expects rank-one vectors";
        BE_CHECK(tile_offsets.ndim() == 2)
            << "single-batch TileCSR SDDMM expects rank-two tile offsets";

        const int64_t rows = indptr.size(0) - 1;
        const int64_t cols = ct.size(0);
        const int64_t tile_count64 =
            (cols + kBaseTileSize - 1) / kBaseTileSize;
        BE_CHECK(rows > 0 && cols > 0)
            << "single-batch TileCSR SDDMM expects positive dimensions";
        BE_CHECK(event.numel() == rows && active_rows.numel() == rows)
            << "single-batch TileCSR SDDMM row lengths must match";
        BE_CHECK(local_targets.numel() == dweight.numel())
            << "single-batch TileCSR SDDMM slot lengths must match";
        BE_CHECK(active_count.numel() == 1)
            << "single-batch TileCSR SDDMM active count shape mismatch";
        BE_CHECK(tile_offsets.size(0) == rows &&
                 tile_offsets.size(1) == tile_count64 + 1)
            << "single-batch TileCSR SDDMM tile boundary shape mismatch";
        BE_CHECK(tile_count64 <= std::numeric_limits<int>::max())
            << "single-batch TileCSR SDDMM tile count exceeds int32";

        cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        BE_CUDA_CHECK(cudaMemsetAsync(
            active_count.data_ptr<int32_t>(),
            0,
            sizeof(int32_t),
            cuda_stream));
        if (dweight.numel() == 0)
            return;
        BE_CUDA_CHECK(cudaMemsetAsync(
            dweight.data_ptr<ValueT>(),
            0,
            static_cast<size_t>(dweight.numel()) * sizeof(ValueT),
            cuda_stream));

        const int64_t compact_blocks64 =
            (rows + kRowsPerCompactBlock - 1) / kRowsPerCompactBlock;
        BE_CHECK(compact_blocks64 <=
                 static_cast<int64_t>(std::numeric_limits<unsigned int>::max()))
            << "single-batch TileCSR SDDMM compact grid is too large";
        compact_active_rows_kernel<EventT>
            <<<static_cast<unsigned int>(compact_blocks64),
               kThreadsPerBlock,
               0,
               cuda_stream>>>(
                event.data_ptr<const EventT>(),
                active_rows.data_ptr<int32_t>(),
                active_count.data_ptr<int32_t>(),
                rows);
        BE_CHECK_KERNEL_LAUNCH();

        int device = 0;
        int sm_count = 0;
        BE_CUDA_CHECK(cudaGetDevice(&device));
        BE_CUDA_CHECK(cudaDeviceGetAttribute(
            &sm_count,
            cudaDevAttrMultiProcessorCount,
            device));
        const int64_t maximum_task_blocks =
            (rows + kGroupsPerBlock - 1) / kGroupsPerBlock;
        const int persistent_blocks = static_cast<int>(std::min<int64_t>(
            maximum_task_blocks,
            static_cast<int64_t>(sm_count) * 2));
        const int tile_count = static_cast<int>(tile_count64);
        const int macro_tiles =
            (tile_count + kBaseTilesPerMacro - 1) / kBaseTilesPerMacro;

        BE_DISPATCH_CSR_INDPTR(indptr.dtype(), IndptrT, {
            for (int macro_tile = 0; macro_tile < macro_tiles; ++macro_tile)
            {
                single_sddmm_panel_kernel<ValueT, IndptrT>
                    <<<persistent_blocks,
                       kThreadsPerBlock,
                       0,
                       cuda_stream>>>(
                        ct.data_ptr<const ValueT>(),
                        local_targets.data_ptr<const uint16_t>(),
                        indptr.data_ptr<const IndptrT>(),
                        tile_offsets.data_ptr<const int32_t>(),
                        active_rows.data_ptr<const int32_t>(),
                        active_count.data_ptr<const int32_t>(),
                        dweight.data_ptr<ValueT>(),
                        tile_count,
                        macro_tile);
                BE_CHECK_KERNEL_LAUNCH();
            }
        });
    }

} // namespace

// @BE tcsr_sddmv_dweight_binary_f32_bool_t arg arg arg arg arg ret ret ret stream
void tcsr_sddmv_dweight_binary_f32_bool_t(
    const BE::Tensor event,
    const BE::Tensor ct,
    const BE::Tensor local_targets,
    const BE::Tensor indptr,
    const BE::Tensor tile_offsets,
    BE::Tensor dweight,
    BE::Tensor active_rows,
    BE::Tensor active_count,
    int64_t stream)
{
    launch_single_sddmm<bool, float>(
        event,
        ct,
        local_targets,
        indptr,
        tile_offsets,
        dweight,
        active_rows,
        active_count,
        stream);
}

// @BE tcsr_sddmv_dweight_binary_f32_float_t arg arg arg arg arg ret ret ret stream
void tcsr_sddmv_dweight_binary_f32_float_t(
    const BE::Tensor event,
    const BE::Tensor ct,
    const BE::Tensor local_targets,
    const BE::Tensor indptr,
    const BE::Tensor tile_offsets,
    BE::Tensor dweight,
    BE::Tensor active_rows,
    BE::Tensor active_count,
    int64_t stream)
{
    launch_single_sddmm<float, float>(
        event,
        ct,
        local_targets,
        indptr,
        tile_offsets,
        dweight,
        active_rows,
        active_count,
        stream);
}

// @BE tcsr_sddmv_dweight_binary_f64_bool_t arg arg arg arg arg ret ret ret stream
void tcsr_sddmv_dweight_binary_f64_bool_t(
    const BE::Tensor event,
    const BE::Tensor ct,
    const BE::Tensor local_targets,
    const BE::Tensor indptr,
    const BE::Tensor tile_offsets,
    BE::Tensor dweight,
    BE::Tensor active_rows,
    BE::Tensor active_count,
    int64_t stream)
{
    launch_single_sddmm<bool, double>(
        event, ct, local_targets, indptr, tile_offsets, dweight, active_rows,
        active_count, stream);
}

// @BE tcsr_sddmv_dweight_binary_f64_int8_t arg arg arg arg arg ret ret ret stream
void tcsr_sddmv_dweight_binary_f64_int8_t(
    const BE::Tensor event,
    const BE::Tensor ct,
    const BE::Tensor local_targets,
    const BE::Tensor indptr,
    const BE::Tensor tile_offsets,
    BE::Tensor dweight,
    BE::Tensor active_rows,
    BE::Tensor active_count,
    int64_t stream)
{
    launch_single_sddmm<int8_t, double>(
        event, ct, local_targets, indptr, tile_offsets, dweight, active_rows,
        active_count, stream);
}

// @BE tcsr_sddmv_dweight_binary_f64_float_t arg arg arg arg arg ret ret ret stream
void tcsr_sddmv_dweight_binary_f64_float_t(
    const BE::Tensor event,
    const BE::Tensor ct,
    const BE::Tensor local_targets,
    const BE::Tensor indptr,
    const BE::Tensor tile_offsets,
    BE::Tensor dweight,
    BE::Tensor active_rows,
    BE::Tensor active_count,
    int64_t stream)
{
    launch_single_sddmm<float, double>(
        event, ct, local_targets, indptr, tile_offsets, dweight, active_rows,
        active_count, stream);
}

// @BE tcsr_sddmv_dweight_binary_f64_double_t arg arg arg arg arg ret ret ret stream
void tcsr_sddmv_dweight_binary_f64_double_t(
    const BE::Tensor event,
    const BE::Tensor ct,
    const BE::Tensor local_targets,
    const BE::Tensor indptr,
    const BE::Tensor tile_offsets,
    BE::Tensor dweight,
    BE::Tensor active_rows,
    BE::Tensor active_count,
    int64_t stream)
{
    launch_single_sddmm<double, double>(
        event, ct, local_targets, indptr, tile_offsets, dweight, active_rows,
        active_count, stream);
}
