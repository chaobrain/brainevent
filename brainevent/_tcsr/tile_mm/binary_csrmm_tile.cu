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

#include "brainevent/common.h"
#include "bn_tile.cuh"

template <bool Homogeneous>
void binary_csrmm_tile_impl(
    const BE::Tensor values,
    const BE::Tensor row_ptr,
    const BE::Tensor local_targets,
    const BE::Tensor tile_offsets,
    const BE::Tensor spike_bn,
    BE::Tensor output_bn,
    BE::Tensor active_rows,
    BE::Tensor workspace,
    int64_t stream)
{
    const char *operation = Homogeneous
                                ? "binary_csrmm_tile_homo_f32"
                                : "binary_csrmm_tile_f32";
    BE_CHECK(values.dtype() == BE::DType::Float32)
        << operation << " expects float32 values";
    BE_CHECK(row_ptr.dtype() == BE::DType::Int64)
        << operation << " expects int64 row_ptr";
    BE_CHECK(local_targets.dtype() == BE::DType::UInt16)
        << operation << " expects uint16 local_targets";
    BE_CHECK(tile_offsets.dtype() == BE::DType::Int32)
        << operation << " expects int32 tile_offsets";
    BE_CHECK(spike_bn.dtype() == BE::DType::Int8)
        << operation << " expects int8 spike_bn";
    BE_CHECK(output_bn.dtype() == BE::DType::Float32)
        << operation << " expects float32 output_bn";
    BE_CHECK(active_rows.dtype() == BE::DType::Int32)
        << operation << " expects int32 active_rows";
    BE_CHECK(workspace.dtype() == BE::DType::Int32)
        << operation << " expects int32 workspace";

    BE_CHECK(values.ndim() == 1 && row_ptr.ndim() == 1 &&
             local_targets.ndim() == 1 && tile_offsets.ndim() == 2)
        << operation << " expects rank-1 layout arrays and rank-2 tile_offsets";
    BE_CHECK(spike_bn.ndim() == 2 && output_bn.ndim() == 2 &&
             active_rows.ndim() == 2 && workspace.ndim() == 2)
        << operation << " expects rank-2 runtime tensors";

    const int batch = static_cast<int>(spike_bn.size(0));
    const int rows = static_cast<int>(spike_bn.size(1));
    const int cols = static_cast<int>(output_bn.size(1));
    const int tile_count = (cols + bn_tile::kTileSize - 1) / bn_tile::kTileSize;
    const int chunks = bn_tile::row_chunks(rows);

    BE_CHECK(batch > 0 && batch <= 65535 && rows > 0 && cols > 0)
        << operation << " received invalid dimensions";
    if (Homogeneous)
    {
        BE_CHECK(values.numel() == 1 && local_targets.numel() > 0)
            << operation << " expects one value and nonempty targets";
    }
    else
    {
        BE_CHECK(values.numel() > 0 && values.numel() == local_targets.numel())
            << operation << " expects matching nonempty values and targets";
    }
    BE_CHECK(row_ptr.numel() == static_cast<size_t>(rows + 1))
        << operation << " row_ptr length must be rows + 1";
    BE_CHECK(tile_offsets.size(0) == rows &&
             tile_offsets.size(1) == tile_count + 1)
        << operation << " tile_offsets shape mismatch";
    BE_CHECK(output_bn.size(0) == batch)
        << operation << " output batch mismatch";
    BE_CHECK(active_rows.size(0) == batch && active_rows.size(1) == rows)
        << operation << " active_rows shape mismatch";
    BE_CHECK(workspace.size(0) == batch && workspace.size(1) == 2 + chunks)
        << operation << " workspace shape mismatch";

    const auto launch = Homogeneous ? bn_tile::launch_homo : bn_tile::launch;
    BE_CUDA_CHECK(launch(
        values.data_ptr<const float>(),
        local_targets.data_ptr<const uint16_t>(),
        row_ptr.data_ptr<const int64_t>(),
        tile_offsets.data_ptr<const int32_t>(),
        spike_bn.data_ptr<const int8_t>(),
        output_bn.data_ptr<float>(),
        active_rows.data_ptr<int32_t>(),
        workspace.data_ptr(),
        rows,
        cols,
        batch,
        reinterpret_cast<cudaStream_t>(stream)));
}

// @BE binary_csrmm_tile_f32 arg arg arg arg arg ret ret ret stream
void binary_csrmm_tile_f32(
    const BE::Tensor values,
    const BE::Tensor row_ptr,
    const BE::Tensor local_targets,
    const BE::Tensor tile_offsets,
    const BE::Tensor spike_bn,
    BE::Tensor output_bn,
    BE::Tensor active_rows,
    BE::Tensor workspace,
    int64_t stream)
{
    binary_csrmm_tile_impl<false>(
        values, row_ptr, local_targets, tile_offsets, spike_bn, output_bn,
        active_rows, workspace, stream);
}

// @BE binary_csrmm_tile_homo_f32 arg arg arg arg arg ret ret ret stream
void binary_csrmm_tile_homo_f32(
    const BE::Tensor values,
    const BE::Tensor row_ptr,
    const BE::Tensor local_targets,
    const BE::Tensor tile_offsets,
    const BE::Tensor spike_bn,
    BE::Tensor output_bn,
    BE::Tensor active_rows,
    BE::Tensor workspace,
    int64_t stream)
{
    binary_csrmm_tile_impl<true>(
        values, row_ptr, local_targets, tile_offsets, spike_bn, output_bn,
        active_rows, workspace, stream);
}
