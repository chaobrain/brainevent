// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
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

#include <cstdint>
#include <limits>
#include <type_traits>

namespace
{

constexpr int kThreadsPerBlock = 256;
constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;
constexpr int kBaseTileSize = 8192;
constexpr int kRowsPerChunk = 1024;

template <typename EventT>
__device__ __forceinline__ bool event_active(EventT value);

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

template <typename EventT, typename IndptrT>
__global__ void csr_bpsa_dinput_single_kernel(
    const float *__restrict__ weights,
    const uint16_t *__restrict__ local_targets,
    const IndptrT *__restrict__ indptr,
    const int32_t *__restrict__ tile_offsets,
    const EventT *__restrict__ event,
    const float *__restrict__ ct,
    float *__restrict__ db,
    int64_t rows,
    int tile_count)
{
    __shared__ uint16_t active_offsets[kRowsPerChunk];
    __shared__ int active_count;

    if (threadIdx.x == 0)
        active_count = 0;
    __syncthreads();

    const int64_t chunk_begin =
        static_cast<int64_t>(blockIdx.x) * kRowsPerChunk;
    const int64_t remaining = rows - chunk_begin;
    const int valid_rows = static_cast<int>(
        remaining < kRowsPerChunk ? remaining : kRowsPerChunk);

    for (int row_offset = threadIdx.x;
         row_offset < valid_rows;
         row_offset += kThreadsPerBlock)
    {
        const int64_t row = chunk_begin + row_offset;
        db[row] = 0.0f;
        if (event_active<EventT>(event[row]))
        {
            const int slot = atomicAdd(&active_count, 1);
            active_offsets[slot] = static_cast<uint16_t>(row_offset);
        }
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int boundaries = tile_count + 1;
    for (int task = warp; task < active_count; task += kWarpsPerBlock)
    {
        const int64_t row = chunk_begin + active_offsets[task];
        const IndptrT row_begin = indptr[row];
        float partial = 0.0f;
        for (int tile = 0; tile < tile_count; ++tile)
        {
            const size_t boundary_offset =
                static_cast<size_t>(row) * boundaries + tile;
            const int relative_begin = tile_offsets[boundary_offset];
            const int relative_end = tile_offsets[boundary_offset + 1];
            const int tile_begin = tile * kBaseTileSize;
            for (int relative = relative_begin + lane;
                 relative < relative_end;
                 relative += 32)
            {
                const IndptrT slot = row_begin + relative;
                const int column = tile_begin + local_targets[slot];
                partial += weights[slot] * ct[column];
            }
        }
        partial = warp_reduce_sum_f32(partial);
        if (lane == 0)
            db[row] = partial;
    }
}

template <typename EventT>
void launch_csr_bpsa_dinput_single(
    const BE::Tensor weights,
    const BE::Tensor local_targets,
    const BE::Tensor indptr,
    const BE::Tensor tile_offsets,
    const BE::Tensor event,
    const BE::Tensor ct,
    BE::Tensor db,
    int64_t stream)
{
    const BE::DType expected_event_dtype =
        std::is_same<EventT, bool>::value
            ? BE::DType::Bool
            : BE::DType::Float32;
    BE_CHECK(weights.dtype() == BE::DType::Float32)
        << "single-batch TileCSR BPSA expects float32 weights";
    BE_CHECK(local_targets.dtype() == BE::DType::UInt16)
        << "single-batch TileCSR BPSA expects uint16 local targets";
    BE_CHECK(indptr.dtype() == BE::DType::Int32 ||
             indptr.dtype() == BE::DType::Int64)
        << "single-batch TileCSR BPSA expects int32 or int64 indptr";
    BE_CHECK(tile_offsets.dtype() == BE::DType::Int32)
        << "single-batch TileCSR BPSA expects int32 tile offsets";
    BE_CHECK(event.dtype() == expected_event_dtype)
        << "single-batch TileCSR BPSA event dtype does not match its target";
    BE_CHECK(ct.dtype() == BE::DType::Float32 &&
             db.dtype() == BE::DType::Float32)
        << "single-batch TileCSR BPSA expects float32 CT and dB";

    BE_CHECK(weights.ndim() == 1 && local_targets.ndim() == 1 &&
             indptr.ndim() == 1 && event.ndim() == 1 &&
             ct.ndim() == 1 && db.ndim() == 1)
        << "single-batch TileCSR BPSA expects rank-one vector operands";
    BE_CHECK(tile_offsets.ndim() == 2)
        << "single-batch TileCSR BPSA expects rank-two tile offsets";

    const int64_t rows = indptr.size(0) - 1;
    const int64_t cols = ct.size(0);
    const int64_t tile_count64 =
        (cols + kBaseTileSize - 1) / kBaseTileSize;
    BE_CHECK(rows > 0 && cols > 0)
        << "single-batch TileCSR BPSA expects positive dimensions";
    BE_CHECK(weights.numel() == local_targets.numel())
        << "single-batch TileCSR BPSA slot lengths must match";
    BE_CHECK(event.numel() == rows && db.numel() == rows)
        << "single-batch TileCSR BPSA row lengths must match";
    BE_CHECK(tile_offsets.size(0) == rows &&
             tile_offsets.size(1) == tile_count64 + 1)
        << "single-batch TileCSR BPSA tile boundary shape mismatch";
    BE_CHECK(tile_count64 <= std::numeric_limits<int>::max())
        << "single-batch TileCSR BPSA tile count exceeds int32";

    const int64_t row_chunks =
        (rows + kRowsPerChunk - 1) / kRowsPerChunk;
    BE_CHECK(
        row_chunks <= static_cast<int64_t>(
            std::numeric_limits<unsigned int>::max()))
        << "single-batch TileCSR BPSA row grid is too large";

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const int tile_count = static_cast<int>(tile_count64);
    BE_DISPATCH_CSR_INDPTR(indptr.dtype(), IndptrT, {
        csr_bpsa_dinput_single_kernel<EventT, IndptrT>
            <<<static_cast<unsigned int>(row_chunks),
               kThreadsPerBlock,
               0,
               cuda_stream>>>(
                weights.data_ptr<const float>(),
                local_targets.data_ptr<const uint16_t>(),
                indptr.data_ptr<const IndptrT>(),
                tile_offsets.data_ptr<const int32_t>(),
                event.data_ptr<const EventT>(),
                ct.data_ptr<const float>(),
                db.data_ptr<float>(),
                rows,
                tile_count);
    });
    BE_CHECK_KERNEL_LAUNCH();
}

} // namespace

// @BE csr_bpsa_dinput_single_f32_bool arg arg arg arg arg arg ret stream
void csr_bpsa_dinput_single_f32_bool(
    const BE::Tensor weights,
    const BE::Tensor local_targets,
    const BE::Tensor indptr,
    const BE::Tensor tile_offsets,
    const BE::Tensor event,
    const BE::Tensor ct,
    BE::Tensor db,
    int64_t stream)
{
    launch_csr_bpsa_dinput_single<bool>(
        weights,
        local_targets,
        indptr,
        tile_offsets,
        event,
        ct,
        db,
        stream);
}

// @BE csr_bpsa_dinput_single_f32_float arg arg arg arg arg arg ret stream
void csr_bpsa_dinput_single_f32_float(
    const BE::Tensor weights,
    const BE::Tensor local_targets,
    const BE::Tensor indptr,
    const BE::Tensor tile_offsets,
    const BE::Tensor event,
    const BE::Tensor ct,
    BE::Tensor db,
    int64_t stream)
{
    launch_csr_bpsa_dinput_single<float>(
        weights,
        local_targets,
        indptr,
        tile_offsets,
        event,
        ct,
        db,
        stream);
}
