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

/*
 * Canonical TCSR float matrix-matrix kernels with BN dense layout.
 *
 * X @ W owns one shared output tile per (batch, tile). W @ X owns one shared
 * input tile per (batch, 4096-row chunk, tile). Both preserve dense values.
 */

#include <cstddef>
#include <cstdint>

#include "cuda_common.h"
#include "brainevent/common.h"

namespace {

constexpr int kMetadataTileSize = 8192;
constexpr int kSharedBytes = 32 * 1024;
constexpr int kBlockSize = 512;
constexpr int kGroupSize = 16;
constexpr int kGroupsPerBlock = kBlockSize / kGroupSize;
constexpr int kRowsPerChunk = 4096;

template <typename ValueT>
__device__ __forceinline__ ValueT group_reduce_sum(ValueT value) {
#pragma unroll
  for (int offset = kGroupSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(__activemask(), value, offset, kGroupSize);
  }
  return value;
}

template <typename ValueT>
__device__ __forceinline__ void tile_bounds(
    const int64_t* __restrict__ row_ptr,
    const uint16_t* __restrict__ local_targets,
    const int32_t* __restrict__ tile_offsets,
    int row,
    int metadata_tile,
    int metadata_subtile,
    int metadata_tile_count,
    int lane,
    int* begin_out,
    int* end_out) {
  constexpr int kComputeTileSize = kSharedBytes / sizeof(ValueT);
  constexpr int kSubtilesPerMetadata = kMetadataTileSize / kComputeTileSize;
  const size_t offset =
      static_cast<size_t>(row) * (metadata_tile_count + 1) + metadata_tile;
  int begin = tile_offsets[offset];
  int end = tile_offsets[offset + 1];
  if (kSubtilesPerMetadata == 2) {
    int midpoint = begin;
    if (lane == 0) {
      int low = begin;
      int high = end;
      const int64_t sparse_begin = row_ptr[row];
      while (low < high) {
        const int middle = low + (high - low) / 2;
        if (local_targets[sparse_begin + middle] < kComputeTileSize) {
          low = middle + 1;
        } else {
          high = middle;
        }
      }
      midpoint = low;
    }
    midpoint = __shfl_sync(__activemask(), midpoint, 0, kGroupSize);
    if (metadata_subtile == 0) {
      end = midpoint;
    } else {
      begin = midpoint;
    }
  }
  *begin_out = begin;
  *end_out = end;
}

template <typename ValueT, bool Homogeneous>
__global__ void xw_tile_kernel(
    const ValueT* __restrict__ values,
    const int64_t* __restrict__ row_ptr,
    const uint16_t* __restrict__ local_targets,
    const int32_t* __restrict__ tile_offsets,
    const ValueT* __restrict__ x_bn,
    ValueT* __restrict__ y_bn,
    int rows,
    int cols,
    int metadata_tile_count) {
  constexpr int kComputeTileSize = kSharedBytes / sizeof(ValueT);
  constexpr int kSubtilesPerMetadata = kMetadataTileSize / kComputeTileSize;
  __shared__ ValueT output_tile[kComputeTileSize];

  const int compute_tile = blockIdx.x;
  const int batch = blockIdx.y;
  const int metadata_tile = compute_tile / kSubtilesPerMetadata;
  const int metadata_subtile = compute_tile % kSubtilesPerMetadata;
  const int local_tile_begin = metadata_subtile * kComputeTileSize;
  const int output_tile_begin = compute_tile * kComputeTileSize;
  const int tile_elements = min(kComputeTileSize, cols - output_tile_begin);
  for (int local = threadIdx.x; local < kComputeTileSize; local += kBlockSize) {
    output_tile[local] = ValueT(0);
  }
  __syncthreads();

  const int group = threadIdx.x / kGroupSize;
  const int lane = threadIdx.x & (kGroupSize - 1);
  const ValueT homogeneous_value = Homogeneous ? values[0] : ValueT(0);
  const ValueT* x = x_bn + static_cast<size_t>(batch) * rows;
  for (int row = group; row < rows; row += kGroupsPerBlock) {
    int begin;
    int end;
    tile_bounds<ValueT>(
        row_ptr, local_targets, tile_offsets, row, metadata_tile,
        metadata_subtile, metadata_tile_count, lane, &begin, &end);
    const int64_t sparse_begin = row_ptr[row];
    const ValueT dense_value = x[row];
    for (int64_t relative = static_cast<int64_t>(begin) + lane;
         relative < static_cast<int64_t>(end); relative += kGroupSize) {
      const int64_t entry = sparse_begin + relative;
      const int local = static_cast<int>(local_targets[entry]) - local_tile_begin;
      const ValueT weight = Homogeneous ? homogeneous_value : values[entry];
      atomicAdd(&output_tile[local], weight * dense_value);
    }
  }
  __syncthreads();

  ValueT* y = y_bn + static_cast<size_t>(batch) * cols;
  for (int local = threadIdx.x; local < tile_elements; local += kBlockSize) {
    y[output_tile_begin + local] = output_tile[local];
  }
}

template <typename ValueT, bool Homogeneous>
__global__ void wx_tile_kernel(
    const ValueT* __restrict__ values,
    const int64_t* __restrict__ row_ptr,
    const uint16_t* __restrict__ local_targets,
    const int32_t* __restrict__ tile_offsets,
    const ValueT* __restrict__ x_bn,
    ValueT* __restrict__ y_bn,
    int rows,
    int cols,
    int metadata_tile_count) {
  constexpr int kComputeTileSize = kSharedBytes / sizeof(ValueT);
  constexpr int kSubtilesPerMetadata = kMetadataTileSize / kComputeTileSize;
  __shared__ ValueT x_tile[kComputeTileSize];

  const int row_chunk = blockIdx.x;
  const int compute_tile = blockIdx.y;
  const int batch = blockIdx.z;
  const int metadata_tile = compute_tile / kSubtilesPerMetadata;
  const int metadata_subtile = compute_tile % kSubtilesPerMetadata;
  const int local_tile_begin = metadata_subtile * kComputeTileSize;
  const int dense_tile_begin = compute_tile * kComputeTileSize;
  const ValueT* x = x_bn + static_cast<size_t>(batch) * cols;
  for (int local = threadIdx.x; local < kComputeTileSize; local += kBlockSize) {
    const int col = dense_tile_begin + local;
    x_tile[local] = col < cols ? x[col] : ValueT(0);
  }
  __syncthreads();

  const int group = threadIdx.x / kGroupSize;
  const int lane = threadIdx.x & (kGroupSize - 1);
  const int row_begin = row_chunk * kRowsPerChunk;
  const int row_end = min(rows, row_begin + kRowsPerChunk);
  const ValueT homogeneous_value = Homogeneous ? values[0] : ValueT(0);
  ValueT* y = y_bn + static_cast<size_t>(batch) * rows;
  for (int row = row_begin + group; row < row_end; row += kGroupsPerBlock) {
    int begin;
    int end;
    tile_bounds<ValueT>(
        row_ptr, local_targets, tile_offsets, row, metadata_tile,
        metadata_subtile, metadata_tile_count, lane, &begin, &end);
    ValueT sum = ValueT(0);
    const int64_t sparse_begin = row_ptr[row];
    for (int64_t relative = static_cast<int64_t>(begin) + lane;
         relative < static_cast<int64_t>(end); relative += kGroupSize) {
      const int64_t entry = sparse_begin + relative;
      const int local = static_cast<int>(local_targets[entry]) - local_tile_begin;
      const ValueT weight = Homogeneous ? homogeneous_value : values[entry];
      sum += weight * x_tile[local];
    }
    sum = group_reduce_sum(sum);
    if (lane == 0) atomicAdd(&y[row], sum);
  }
}

template <typename ValueT, BE::DType DType, bool Homogeneous, bool XW>
void launch_mm(
    const BE::Tensor values,
    const BE::Tensor row_ptr,
    const BE::Tensor local_targets,
    const BE::Tensor tile_offsets,
    const BE::Tensor x_bn,
    BE::Tensor y_bn,
    int64_t stream) {
  const char* operation = XW ? "csrmm_xw_tile" : "csrmm_wx_tile";
  BE_CHECK(values.dtype() == DType && x_bn.dtype() == DType &&
           y_bn.dtype() == DType)
      << operation << " requires matching value, x, and y dtypes";
  BE_CHECK(row_ptr.dtype() == BE::DType::Int64)
      << operation << " requires int64 row_ptr";
  BE_CHECK(local_targets.dtype() == BE::DType::UInt16)
      << operation << " requires uint16 local_targets";
  BE_CHECK(tile_offsets.dtype() == BE::DType::Int32)
      << operation << " requires int32 tile_offsets";
  BE_CHECK(values.ndim() == 1 && row_ptr.ndim() == 1 &&
           local_targets.ndim() == 1 && tile_offsets.ndim() == 2 &&
           x_bn.ndim() == 2 && y_bn.ndim() == 2)
      << operation << " received an invalid tensor rank";

  const int rows = static_cast<int>(row_ptr.size(0)) - 1;
  const int batch = static_cast<int>(x_bn.size(0));
  const int cols = static_cast<int>(XW ? y_bn.size(1) : x_bn.size(1));
  const int metadata_tiles = (cols + kMetadataTileSize - 1) / kMetadataTileSize;
  const size_t nnz = local_targets.numel();
  BE_CHECK(rows >= 0 && batch >= 0 && batch <= 65535)
      << operation << " received invalid dimensions";
  BE_CHECK(y_bn.size(0) == batch)
      << operation << " output batch mismatch";
  BE_CHECK((XW && x_bn.size(1) == rows) || (!XW && y_bn.size(1) == rows))
      << operation << " neuron dimension mismatch";
  BE_CHECK(tile_offsets.size(0) == rows &&
           tile_offsets.size(1) == metadata_tiles + 1)
      << operation << " tile_offsets shape mismatch";
  BE_CHECK((Homogeneous && values.numel() == 1) ||
           (!Homogeneous && values.numel() == nnz))
      << operation << " values length mismatch";

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  if (y_bn.numel() > 0) {
    BE_CUDA_CHECK(cudaMemsetAsync(
        y_bn.data_ptr<ValueT>(), 0, y_bn.numel() * sizeof(ValueT), cuda_stream));
  }
  if (rows == 0 || cols == 0 || batch == 0 || nnz == 0) return;
  constexpr int compute_tile_size = kSharedBytes / sizeof(ValueT);
  const int compute_tiles = (cols + compute_tile_size - 1) / compute_tile_size;
  if (XW) {
    xw_tile_kernel<ValueT, Homogeneous>
        <<<dim3(compute_tiles, batch), kBlockSize, 0, cuda_stream>>>(
            values.data_ptr<const ValueT>(), row_ptr.data_ptr<const int64_t>(),
            local_targets.data_ptr<const uint16_t>(),
            tile_offsets.data_ptr<const int32_t>(),
            x_bn.data_ptr<const ValueT>(), y_bn.data_ptr<ValueT>(), rows, cols,
            metadata_tiles);
  } else {
    const int row_chunks = (rows + kRowsPerChunk - 1) / kRowsPerChunk;
    wx_tile_kernel<ValueT, Homogeneous>
        <<<dim3(row_chunks, compute_tiles, batch), kBlockSize, 0, cuda_stream>>>(
            values.data_ptr<const ValueT>(), row_ptr.data_ptr<const int64_t>(),
            local_targets.data_ptr<const uint16_t>(),
            tile_offsets.data_ptr<const int32_t>(),
            x_bn.data_ptr<const ValueT>(), y_bn.data_ptr<ValueT>(), rows, cols,
            metadata_tiles);
  }
  BE_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

#define DEFINE_CSRMM_HANDLERS(SUFFIX, VALUE_T, DTYPE)                         \
  void csrmm_xw_tile##SUFFIX(                                                  \
      const BE::Tensor values, const BE::Tensor row_ptr,                      \
      const BE::Tensor local_targets, const BE::Tensor tile_offsets,          \
      const BE::Tensor x, BE::Tensor y, int64_t stream) {                     \
    launch_mm<VALUE_T, DTYPE, false, true>(                                   \
        values, row_ptr, local_targets, tile_offsets, x, y, stream);          \
  }                                                                            \
  void csrmm_xw_tile_homo##SUFFIX(                                             \
      const BE::Tensor values, const BE::Tensor row_ptr,                      \
      const BE::Tensor local_targets, const BE::Tensor tile_offsets,          \
      const BE::Tensor x, BE::Tensor y, int64_t stream) {                     \
    launch_mm<VALUE_T, DTYPE, true, true>(                                    \
        values, row_ptr, local_targets, tile_offsets, x, y, stream);          \
  }                                                                            \
  void csrmm_wx_tile##SUFFIX(                                                  \
      const BE::Tensor values, const BE::Tensor row_ptr,                      \
      const BE::Tensor local_targets, const BE::Tensor tile_offsets,          \
      const BE::Tensor x, BE::Tensor y, int64_t stream) {                     \
    launch_mm<VALUE_T, DTYPE, false, false>(                                  \
        values, row_ptr, local_targets, tile_offsets, x, y, stream);          \
  }                                                                            \
  void csrmm_wx_tile_homo##SUFFIX(                                             \
      const BE::Tensor values, const BE::Tensor row_ptr,                      \
      const BE::Tensor local_targets, const BE::Tensor tile_offsets,          \
      const BE::Tensor x, BE::Tensor y, int64_t stream) {                     \
    launch_mm<VALUE_T, DTYPE, true, false>(                                   \
        values, row_ptr, local_targets, tile_offsets, x, y, stream);          \
  }

// @BE csrmm_xw_tile_f32 arg arg arg arg arg ret stream
// @BE csrmm_xw_tile_homo_f32 arg arg arg arg arg ret stream
// @BE csrmm_wx_tile_f32 arg arg arg arg arg ret stream
// @BE csrmm_wx_tile_homo_f32 arg arg arg arg arg ret stream
DEFINE_CSRMM_HANDLERS(_f32, float, BE::DType::Float32)

// @BE csrmm_xw_tile_f64 arg arg arg arg arg ret stream
// @BE csrmm_xw_tile_homo_f64 arg arg arg arg arg ret stream
// @BE csrmm_wx_tile_f64 arg arg arg arg arg ret stream
// @BE csrmm_wx_tile_homo_f64 arg arg arg arg arg ret stream
DEFINE_CSRMM_HANDLERS(_f64, double, BE::DType::Float64)

#undef DEFINE_CSRMM_HANDLERS
