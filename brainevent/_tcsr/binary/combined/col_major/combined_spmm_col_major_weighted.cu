// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

// Column-major weighted wrappers; canonical weights are read through perm.
#include "combined_col_major.cuh"

namespace combined_col_major
{
template <class V, class O>
cudaError_t launch_weighted(const V *values, const uint16_t *local_targets,
    const O *tile_offsets, const O *perm, const V *events, V *output,
    int32_t *active_rows, void *workspace, const Plan &plan, cudaStream_t stream)
{
  return combined::launch_weighted_permuted(values, local_targets, tile_offsets, perm,
                                           events, output, active_rows, workspace, plan, stream);
}
#define INSTANTIATE_WEIGHTED(V, O) \
  template cudaError_t launch_weighted<V, O>(const V *, const uint16_t *, const O *, const O *, \
      const V *, V *, int32_t *, void *, const Plan &, cudaStream_t);
#define INSTANTIATE_WEIGHTED_OFFSETS(V) \
  INSTANTIATE_WEIGHTED(V, int32_t) \
  INSTANTIATE_WEIGHTED(V, int64_t)
INSTANTIATE_WEIGHTED_OFFSETS(__half)
INSTANTIATE_WEIGHTED_OFFSETS(__nv_bfloat16)
INSTANTIATE_WEIGHTED_OFFSETS(float)
INSTANTIATE_WEIGHTED_OFFSETS(double)
#undef INSTANTIATE_WEIGHTED_OFFSETS
#undef INSTANTIATE_WEIGHTED
} // namespace combined_col_major

#ifndef BE_SPMM_STANDALONE
#include "combined_col_major_ffi.cuh"
#define DEFINE_COL_WEIGHTED(SUFFIX, V) \
  void combined_col_major_weighted_hetero##SUFFIX(const BE::Tensor values, const BE::Tensor local_targets, \
      const BE::Tensor tile_offsets, const BE::Tensor perm, const BE::Tensor events, BE::Tensor output, \
      BE::Tensor active_rows, BE::Tensor workspace, int64_t tile_width, int64_t value_type, int64_t stream) \
  { \
    combined_col_major_ffi<V, V, true>(values, local_targets, tile_offsets, perm, events, output, \
                                      active_rows, workspace, tile_width, value_type, stream); \
  }
// @BE combined_col_major_weighted_hetero_f16
DEFINE_COL_WEIGHTED(_f16, __half)
// @BE combined_col_major_weighted_hetero_bf16
DEFINE_COL_WEIGHTED(_bf16, __nv_bfloat16)
// @BE combined_col_major_weighted_hetero_f32
DEFINE_COL_WEIGHTED(_f32, float)
// @BE combined_col_major_weighted_hetero_f64
DEFINE_COL_WEIGHTED(_f64, double)
#undef DEFINE_COL_WEIGHTED
#endif
