// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

#ifndef COMBINED_KERNELS_CUH
#define COMBINED_KERNELS_CUH
// CUDA/C++ API. Link combined_spmm_binary.cu; define BE_SPMM_STANDALONE for native-only builds.
#include "brainevent/cuda_traits.cuh"
#include "planner.h"
#include <cstddef>
namespace combined {
using ValueType = cuda_traits::ValueType;
constexpr int kTileWidth = 24576; // Maximum f16/bf16/f32 width; use make_plan for metadata.
template<class V> using TileConfig = cuda_traits::TileConfig<V>;
struct Plan {
    int rows = 0, cols = 0, batch = 0;
    int64_t nnz = 0;
    ValueType value_type = ValueType::f32;
    int tile_width = kTileWidth; // Actual width, bound to cols, batch and value_type.
};
int tile_width(ValueType type); // Dtype maximum, not the prepared schedule's width.
cudaError_t make_plan(int rows, int cols, int batch, int64_t nnz, ValueType type, Plan* plan);
size_t workspace_bytes(int rows, int batch);
// Metadata is sorted within each row. Offsets are absolute edge indices;
// local_targets[e] = column[e] % plan.tile_width. active_rows needs batch * rows entries.
// Output is overwritten; scratch is initialized by launch, on the supplied stream.
template<class ValueT, class EventT, class OffsetT>
cudaError_t launch(const ValueT* values, const uint16_t* local_targets,
                   const OffsetT* tile_offsets, const EventT* events, ValueT* output,
                   int32_t* active_rows, void* workspace, const Plan& plan, cudaStream_t stream);
// One device value shared by all edges; the same sorted metadata and workspace apply.
template<class ValueT, class EventT, class OffsetT>
cudaError_t launch_homo(const ValueT* values, const uint16_t* local_targets,
                   const OffsetT* tile_offsets, const EventT* events, ValueT* output,
                   int32_t* active_rows, void* workspace, const Plan& plan, cudaStream_t stream);
// Mirror/tile edge e reads values[perm[e]]. All metadata uses the same edge order.
// perm and absolute tile offsets share OffsetT; values contains plan.nnz elements.
template<class ValueT, class EventT, class OffsetT>
cudaError_t launch_permuted(const ValueT* values, const uint16_t* local_targets,
                   const OffsetT* tile_offsets, const OffsetT* perm, const EventT* events,
                   ValueT* output, int32_t* active_rows, void* workspace,
                   const Plan& plan, cudaStream_t stream);
} // namespace combined
#endif
