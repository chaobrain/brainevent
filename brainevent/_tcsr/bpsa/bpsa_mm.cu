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

#include <cstdint>
#include <limits>

namespace
{

constexpr int kThreadsPerBlock = 256;
constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;
constexpr int kBaseTileSize = 8192;
constexpr int kRowsPerChunk = 8192;

template <typename IndptrT>
__global__ void csr_bpsa_dinput_kernel(
    const float *__restrict__ weights,
    const IndptrT *__restrict__ indptr,
    const uint16_t *__restrict__ local_targets,
    const int32_t *__restrict__ tile_offsets,
    const uint8_t *__restrict__ mask_bn,
    const float *__restrict__ ct_bn,
    float *__restrict__ db_bn,
    int64_t rows,
    int64_t cols,
    int64_t mask_width,
    int tile_count,
    int tile)
{
    __shared__ uint16_t active_offsets[kRowsPerChunk];
    __shared__ int active_count;

    if (threadIdx.x == 0)
        active_count = 0;
    __syncthreads();

    const int64_t batch = blockIdx.y;
    const int64_t chunk_begin =
        static_cast<int64_t>(blockIdx.x) * kRowsPerChunk;
    const int64_t remaining = rows - chunk_begin;
    const int valid_rows = static_cast<int>(
        remaining < kRowsPerChunk ? remaining : kRowsPerChunk);
    const int valid_bytes = (valid_rows + 7) / 8;

    for (int byte_offset = threadIdx.x;
         byte_offset < valid_bytes;
         byte_offset += kThreadsPerBlock)
    {
        const int64_t mask_byte = chunk_begin / 8 + byte_offset;
        uint8_t active_bits = mask_bn[
            static_cast<size_t>(batch) * mask_width + mask_byte];
        while (active_bits != 0)
        {
            const int bit = __ffs(static_cast<int>(active_bits)) - 1;
            const int row_offset = byte_offset * 8 + bit;
            if (row_offset < valid_rows)
            {
                const int slot = atomicAdd(&active_count, 1);
                active_offsets[slot] = static_cast<uint16_t>(row_offset);
            }
            active_bits = static_cast<uint8_t>(active_bits & (active_bits - 1));
        }
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int boundaries = tile_count + 1;
    const int tile_begin = tile * kBaseTileSize;
    for (int task = warp; task < active_count; task += kWarpsPerBlock)
    {
        const int64_t row = chunk_begin + active_offsets[task];
        const size_t boundary_offset =
            static_cast<size_t>(row) * boundaries + tile;
        const int relative_begin = tile_offsets[boundary_offset];
        const int relative_end = tile_offsets[boundary_offset + 1];
        const IndptrT row_begin = indptr[row];
        float partial = 0.0f;
        for (int relative = relative_begin + lane;
             relative < relative_end;
             relative += 32)
        {
            const IndptrT slot = row_begin + relative;
            const int col = tile_begin + local_targets[slot];
            partial += weights[slot] * ct_bn[
                static_cast<size_t>(batch) * cols + col];
        }
        partial = warp_reduce_sum_f32(partial);
        if (lane == 0)
            db_bn[static_cast<size_t>(batch) * rows + row] += partial;
    }
}

} // namespace

// @BE csr_bpsa_dinput_f32 arg arg arg arg arg arg ret stream
void csr_bpsa_dinput_f32(
    const BE::Tensor weights,
    const BE::Tensor indptr,
    const BE::Tensor local_targets,
    const BE::Tensor tile_offsets,
    const BE::Tensor mask_bn,
    const BE::Tensor ct_bn,
    BE::Tensor db_bn,
    int64_t stream)
{
    BE_CHECK(weights.dtype() == BE::DType::Float32)
        << "BPSA dB expects float32 weights";
    BE_CHECK(indptr.dtype() == BE::DType::Int32 ||
             indptr.dtype() == BE::DType::Int64)
        << "BPSA dB expects int32 or int64 indptr";
    BE_CHECK(local_targets.dtype() == BE::DType::UInt16)
        << "BPSA dB expects uint16 local_targets";
    BE_CHECK(tile_offsets.dtype() == BE::DType::Int32)
        << "BPSA dB expects int32 tile_offsets";
    BE_CHECK(mask_bn.dtype() == BE::DType::UInt8)
        << "BPSA dB expects uint8 mask_bn";
    BE_CHECK(ct_bn.dtype() == BE::DType::Float32 &&
             db_bn.dtype() == BE::DType::Float32)
        << "BPSA dB expects float32 ct_bn and db_bn";

    BE_CHECK(weights.ndim() == 1 && indptr.ndim() == 1 &&
             local_targets.ndim() == 1 && tile_offsets.ndim() == 2)
        << "BPSA dB expects rank-1 slots and rank-2 tile_offsets";
    BE_CHECK(mask_bn.ndim() == 2 && ct_bn.ndim() == 2 && db_bn.ndim() == 2)
        << "BPSA dB expects rank-2 dense operands";

    const int64_t rows = indptr.size(0) - 1;
    const int64_t batch = mask_bn.size(0);
    const int64_t cols = ct_bn.size(1);
    const int64_t mask_width = (rows + 7) / 8;
    const int64_t tile_count64 = (cols + kBaseTileSize - 1) / kBaseTileSize;
    BE_CHECK(rows > 0 && batch > 0 && batch <= 65535 && cols > 0)
        << "BPSA dB received invalid dimensions";
    BE_CHECK(weights.numel() == local_targets.numel())
        << "BPSA dB weights and local_targets sizes must match";
    BE_CHECK(mask_bn.size(1) == mask_width)
        << "BPSA dB mask width mismatch";
    BE_CHECK(ct_bn.size(0) == batch)
        << "BPSA dB cotangent batch mismatch";
    BE_CHECK(tile_offsets.size(0) == rows &&
             tile_offsets.size(1) == tile_count64 + 1)
        << "BPSA dB tile_offsets shape mismatch";
    BE_CHECK(db_bn.size(0) == batch && db_bn.size(1) == rows)
        << "BPSA dB output shape mismatch";
    BE_CHECK(tile_count64 <= std::numeric_limits<int>::max())
        << "BPSA dB tile grid is too large";

    const int64_t row_chunks = (rows + kRowsPerChunk - 1) / kRowsPerChunk;
    BE_CHECK(
        row_chunks <= static_cast<int64_t>(
            std::numeric_limits<unsigned int>::max()))
        << "BPSA dB row grid is too large";

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    BE_CUDA_CHECK(cudaMemsetAsync(
        db_bn.data_ptr<float>(),
        0,
        static_cast<size_t>(db_bn.numel()) * sizeof(float),
        cuda_stream));

    const int tile_count = static_cast<int>(tile_count64);
    BE_DISPATCH_CSR_INDPTR(indptr.dtype(), IndptrT, {
        for (int tile = 0; tile < tile_count; ++tile)
        {
            csr_bpsa_dinput_kernel<IndptrT>
                <<<dim3(static_cast<unsigned int>(row_chunks),
                        static_cast<unsigned int>(batch)),
                   kThreadsPerBlock,
                   0,
                   cuda_stream>>>(
                    weights.data_ptr<const float>(),
                    indptr.data_ptr<const IndptrT>(),
                    local_targets.data_ptr<const uint16_t>(),
                    tile_offsets.data_ptr<const int32_t>(),
                    mask_bn.data_ptr<const uint8_t>(),
                    ct_bn.data_ptr<const float>(),
                    db_bn.data_ptr<float>(),
                    rows,
                    cols,
                    mask_width,
                    tile_count,
                    tile);
            BE_CHECK_KERNEL_LAUNCH();
        }
    });
}
