// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

// Binary tile SpMM FFI entry points (direct and homogeneous). The C++ launchers
// and their explicit instantiations come from combined_impl.cuh.
#include "combined_impl.cuh"
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
