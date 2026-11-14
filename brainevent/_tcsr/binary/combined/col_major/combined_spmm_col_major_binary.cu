// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

// Column-major binary SpMM FFI entry points. A self-contained module: the shared
// launchers and kernels come from combined_impl.cuh.
#include "../combined_impl.cuh"
#include "combined_col_major_ffi.cuh"

namespace combined_col_major
{
template <class V, class E, class O>
cudaError_t launch(const V *values, const uint16_t *local_targets,
                   const O *tile_offsets, const O *perm, const E *events,
                   V *output, int32_t *active_rows, void *workspace,
                   const Plan &plan, cudaStream_t stream)
{
  return combined::launch_permuted(values, local_targets, tile_offsets, perm, events,
                                   output, active_rows, workspace, plan, stream);
}

#define INSTANTIATE_COL_MAJOR(V, E, O) \
  template cudaError_t launch<V, E, O>(const V *, const uint16_t *, const O *, const O *, \
      const E *, V *, int32_t *, void *, const Plan &, cudaStream_t);
#define INSTANTIATE_COL_MAJOR_OFFSETS(V, E) \
  INSTANTIATE_COL_MAJOR(V, E, int32_t) \
  INSTANTIATE_COL_MAJOR(V, E, int64_t)
INSTANTIATE_COL_MAJOR_OFFSETS(__half, int8_t)
INSTANTIATE_COL_MAJOR_OFFSETS(__half, float)
INSTANTIATE_COL_MAJOR_OFFSETS(__nv_bfloat16, int8_t)
INSTANTIATE_COL_MAJOR_OFFSETS(__nv_bfloat16, float)
INSTANTIATE_COL_MAJOR_OFFSETS(float, int8_t)
INSTANTIATE_COL_MAJOR_OFFSETS(float, float)
INSTANTIATE_COL_MAJOR_OFFSETS(double, int8_t)
INSTANTIATE_COL_MAJOR_OFFSETS(double, float)
#undef INSTANTIATE_COL_MAJOR_OFFSETS
#undef INSTANTIATE_COL_MAJOR
} // namespace combined_col_major

#define DEFINE_COL_MAJOR(SUFFIX, V, E) \
  void combined_col_major_binary_hetero##SUFFIX(const BE::Tensor values, const BE::Tensor local_targets, \
      const BE::Tensor tile_offsets, const BE::Tensor perm, const BE::Tensor events, BE::Tensor output, \
      BE::Tensor active_rows, BE::Tensor workspace, int64_t tile_width, int64_t value_type, int64_t stream) \
  { \
    combined_col_major_ffi<V, E>(values, local_targets, tile_offsets, perm, events, output, \
                                active_rows, workspace, tile_width, value_type, stream); \
  }
// @BE combined_col_major_binary_hetero_f16_bool
DEFINE_COL_MAJOR(_f16_bool, __half, int8_t)
// @BE combined_col_major_binary_hetero_f16_float
DEFINE_COL_MAJOR(_f16_float, __half, float)
// @BE combined_col_major_binary_hetero_bf16_bool
DEFINE_COL_MAJOR(_bf16_bool, __nv_bfloat16, int8_t)
// @BE combined_col_major_binary_hetero_bf16_float
DEFINE_COL_MAJOR(_bf16_float, __nv_bfloat16, float)
// @BE combined_col_major_binary_hetero_f32_bool
DEFINE_COL_MAJOR(_f32_bool, float, int8_t)
// @BE combined_col_major_binary_hetero_f32_float
DEFINE_COL_MAJOR(_f32_float, float, float)
// @BE combined_col_major_binary_hetero_f64_bool
DEFINE_COL_MAJOR(_f64_bool, double, int8_t)
// @BE combined_col_major_binary_hetero_f64_float
DEFINE_COL_MAJOR(_f64_float, double, float)
#undef DEFINE_COL_MAJOR
