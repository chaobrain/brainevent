// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

// Weighted tile SpMM: nonzero events scale each edge contribution in AccT.
// The ordered active-row list supplies source row IDs; no value queue is needed.
#include "combined_impl.cuh"

namespace combined
{
#define DEFINE_WEIGHTED_LAUNCH(NAME, HOMO) \
  template <class V, class O> \
  cudaError_t NAME(const V *values, const uint16_t *local_targets, const O *tile_offsets, \
                   const V *events, V *output, int32_t *active_rows, void *workspace, \
                   const Plan &plan, cudaStream_t stream) \
  { \
    return launch_impl<HOMO, V, V, O, false, true>(values, local_targets, tile_offsets, events, \
                                                  output, active_rows, workspace, plan, stream); \
  }
DEFINE_WEIGHTED_LAUNCH(launch_weighted, false)
DEFINE_WEIGHTED_LAUNCH(launch_weighted_homo, true)
#undef DEFINE_WEIGHTED_LAUNCH

template <class V, class O>
cudaError_t launch_weighted_permuted(const V *values, const uint16_t *local_targets,
    const O *tile_offsets, const O *perm, const V *events, V *output,
    int32_t *active_rows, void *workspace, const Plan &plan, cudaStream_t stream)
{
  return launch_impl<false, V, V, O, true, true>(values, local_targets, tile_offsets, events,
                                               output, active_rows, workspace, plan, stream, perm);
}
#define INSTANTIATE_WEIGHTED(V, O) \
  template cudaError_t launch_weighted<V, O>(const V *, const uint16_t *, const O *, const V *, \
      V *, int32_t *, void *, const Plan &, cudaStream_t); \
  template cudaError_t launch_weighted_homo<V, O>(const V *, const uint16_t *, const O *, const V *, \
      V *, int32_t *, void *, const Plan &, cudaStream_t); \
  template cudaError_t launch_weighted_permuted<V, O>(const V *, const uint16_t *, const O *, const O *, \
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
} // namespace combined

#ifndef BE_SPMM_STANDALONE
#include "combined_ffi.cuh"
#define DEFINE_WEIGHTED(SUFFIX, V, HOMO) \
  void combined_weighted##SUFFIX(const BE::Tensor values, const BE::Tensor local_targets, \
      const BE::Tensor tile_offsets, const BE::Tensor events, BE::Tensor output, \
      BE::Tensor active_rows, BE::Tensor workspace, int64_t tile_width, int64_t value_type, int64_t stream) \
  { \
    combined_ffi<V, V, HOMO, true>(values, local_targets, tile_offsets, events, output, \
                                  active_rows, workspace, tile_width, value_type, stream); \
  }
// @BE combined_weighted_f16
DEFINE_WEIGHTED(_f16, __half, false)
// @BE combined_weighted_bf16
DEFINE_WEIGHTED(_bf16, __nv_bfloat16, false)
// @BE combined_weighted_f32
DEFINE_WEIGHTED(_f32, float, false)
// @BE combined_weighted_f64
DEFINE_WEIGHTED(_f64, double, false)
// @BE combined_weighted_homo_f16
DEFINE_WEIGHTED(_homo_f16, __half, true)
// @BE combined_weighted_homo_bf16
DEFINE_WEIGHTED(_homo_bf16, __nv_bfloat16, true)
// @BE combined_weighted_homo_f32
DEFINE_WEIGHTED(_homo_f32, float, true)
// @BE combined_weighted_homo_f64
DEFINE_WEIGHTED(_homo_f64, double, true)
#undef DEFINE_WEIGHTED
#endif
