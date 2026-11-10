// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

// Binary tile SpMM launchers; kernels are shared with combined_spmm_weighted.cu.
#include "combined_impl.cuh"

namespace combined
{
  int tile_width(ValueType type)
  {
    switch (type)
    {
    case ValueType::f16:
    case ValueType::bf16:
    case ValueType::f32:
      return 24576;
    case ValueType::f64:
      return 12288;
    default:
      return 0;
    }
  }
  cudaError_t make_plan(int rows, int cols, int batch, int64_t nnz, ValueType type, Plan *plan)
  {
    const int width = tile_width(type);
    if (!plan || rows < 0 || cols < 0 || batch < 0 || batch > 65535 || nnz < 0 || !width ||
        ((rows == 0 || cols == 0) && nnz != 0))
      return cudaErrorInvalidValue;
    const auto schedule = make_tile_schedule(cols, batch, type == ValueType::f64 ? 8 : 4);
    *plan = {rows, cols, batch, nnz, type, schedule.tile_width};
    return cudaSuccess;
  }
  size_t workspace_bytes(int rows, int batch)
  {
    return ordered_workspace_bytes(rows, batch);
  }
  template <class V, class E, class O>
  cudaError_t launch(const V *values, const uint16_t *local_targets,
                     const O *tile_offsets, const E *events, V *output,
                     int32_t *active_rows, void *workspace, const Plan &plan, cudaStream_t stream)
  {
    return launch_impl<false>(values, local_targets, tile_offsets, events, output,
                              active_rows, workspace, plan, stream);
  }
  template <class V, class E, class O>
  cudaError_t launch_homo(const V *values, const uint16_t *local_targets,
                     const O *tile_offsets, const E *events, V *output,
                     int32_t *active_rows, void *workspace, const Plan &plan, cudaStream_t stream)
  {
    return launch_impl<true>(values, local_targets, tile_offsets, events, output,
                              active_rows, workspace, plan, stream);
  }
  template <class V, class E, class O>
  cudaError_t launch_permuted(const V *values, const uint16_t *local_targets,
                     const O *tile_offsets, const O *perm, const E *events, V *output,
                     int32_t *active_rows, void *workspace, const Plan &plan, cudaStream_t stream)
  {
    return launch_impl<false, V, E, O, true>(values, local_targets, tile_offsets, events, output,
                                           active_rows, workspace, plan, stream, perm);
  }
#define INSTANTIATE_COMBINED(V, E, O) \
  template cudaError_t launch<V, E, O>(const V *, const uint16_t *, const O *, const E *, V *, int32_t *, void *, const Plan &, cudaStream_t); \
  template cudaError_t launch_homo<V, E, O>(const V *, const uint16_t *, const O *, const E *, V *, int32_t *, void *, const Plan &, cudaStream_t); \
  template cudaError_t launch_permuted<V, E, O>(const V *, const uint16_t *, const O *, const O *, const E *, V *, int32_t *, void *, const Plan &, cudaStream_t);
#define INSTANTIATE_COMBINED_INDICES(V, E) \
  INSTANTIATE_COMBINED(V, E, int32_t) \
  INSTANTIATE_COMBINED(V, E, int64_t)
INSTANTIATE_COMBINED_INDICES(__half, int8_t)
INSTANTIATE_COMBINED_INDICES(__half, float)
INSTANTIATE_COMBINED_INDICES(__nv_bfloat16, int8_t)
INSTANTIATE_COMBINED_INDICES(__nv_bfloat16, float)
INSTANTIATE_COMBINED_INDICES(float, int8_t)
INSTANTIATE_COMBINED_INDICES(float, float)
INSTANTIATE_COMBINED_INDICES(double, int8_t)
INSTANTIATE_COMBINED_INDICES(double, float)
#undef INSTANTIATE_COMBINED_INDICES
#undef INSTANTIATE_COMBINED

}
#ifndef BE_SPMM_STANDALONE
#include "combined_ffi.cuh"
#define DEFINE_COMBINED(SUFFIX, V, E, HOMO)                                                                                                \
  void combined##SUFFIX(const BE::Tensor values, const BE::Tensor local_targets, const BE::Tensor tile_offsets,                      \
                        const BE::Tensor events, BE::Tensor output, BE::Tensor active_rows, BE::Tensor workspace,                    \
                        int64_t tile_width, int64_t value_type, int64_t stream)                                                      \
  {                                                                                                                                  \
    combined_ffi<V, E, HOMO>(values, local_targets, tile_offsets, events, output, active_rows, workspace, tile_width, value_type, stream); \
  }
// @BE combined_f16_bool
DEFINE_COMBINED(_f16_bool, __half, int8_t, false)
// @BE combined_f16_float
DEFINE_COMBINED(_f16_float, __half, float, false)
// @BE combined_bf16_bool
DEFINE_COMBINED(_bf16_bool, __nv_bfloat16, int8_t, false)
// @BE combined_bf16_float
DEFINE_COMBINED(_bf16_float, __nv_bfloat16, float, false)
// @BE combined_f32_bool
DEFINE_COMBINED(_f32_bool, float, int8_t, false)
// @BE combined_f32_float
DEFINE_COMBINED(_f32_float, float, float, false)
// @BE combined_f64_bool
DEFINE_COMBINED(_f64_bool, double, int8_t, false)
// @BE combined_f64_float
DEFINE_COMBINED(_f64_float, double, float, false)
// @BE combined_homo_f16_bool
DEFINE_COMBINED(_homo_f16_bool, __half, int8_t, true)
// @BE combined_homo_f16_float
DEFINE_COMBINED(_homo_f16_float, __half, float, true)
// @BE combined_homo_bf16_bool
DEFINE_COMBINED(_homo_bf16_bool, __nv_bfloat16, int8_t, true)
// @BE combined_homo_bf16_float
DEFINE_COMBINED(_homo_bf16_float, __nv_bfloat16, float, true)
// @BE combined_homo_f32_bool
DEFINE_COMBINED(_homo_f32_bool, float, int8_t, true)
// @BE combined_homo_f32_float
DEFINE_COMBINED(_homo_f32_float, float, float, true)
// @BE combined_homo_f64_bool
DEFINE_COMBINED(_homo_f64_bool, double, int8_t, true)
// @BE combined_homo_f64_float
DEFINE_COMBINED(_homo_f64_float, double, float, true)

#undef DEFINE_COMBINED

#endif
