// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

#ifndef COMBINED_COL_MAJOR_CUH
#define COMBINED_COL_MAJOR_CUH
// W(M, N): events[B, N] -> output[B, M], values[perm[e]] in original weight order.
// Link combined_spmm_col_major_binary.cu and ../combined_spmm_binary.cu with BE_SPMM_STANDALONE.
#include "../combined_kernels.cuh"

namespace combined_col_major
{
using combined::Plan;
using combined::ValueType;
using combined::make_plan;
using combined::tile_width;
using combined::workspace_bytes;

// Plan rows=N, cols=M. Sorted mirror edges define local_targets[e] and perm[e].
// Absolute tile_offsets[N * (ceil(M / tile_width) + 1)] and perm share OffsetT.
// Output is overwritten; active_rows needs B*N entries; scratch is caller-owned.
template <class ValueT, class EventT, class OffsetT>
cudaError_t launch(const ValueT *values, const uint16_t *local_targets,
                   const OffsetT *tile_offsets, const OffsetT *perm, const EventT *events,
                   ValueT *output, int32_t *active_rows, void *workspace,
                   const Plan &plan, cudaStream_t stream);
// Weighted counterpart: same metadata/scratch, events share ValueT.
// Link combined_spmm_col_major_weighted.cu, ../combined_spmm_weighted.cu and ../combined_spmm_binary.cu.
template <class ValueT, class OffsetT>
cudaError_t launch_weighted(const ValueT *values, const uint16_t *local_targets,
                   const OffsetT *tile_offsets, const OffsetT *perm, const ValueT *events,
                   ValueT *output, int32_t *active_rows, void *workspace,
                   const Plan &plan, cudaStream_t stream);
} // namespace combined_col_major
#endif
