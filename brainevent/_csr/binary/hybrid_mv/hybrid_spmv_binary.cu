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
// ==============================================================================

// Binary hybrid SpMV: nonzero bool/float32 events gate weight delivery.
// Shared kernels, launchers and FFI validation live in hybrid_spmv_impl.cuh.
#include "hybrid_spmv_impl.cuh"

#define DEFINE_HYBRID_SPMV_BINARY(SUFFIX, WEIGHT_T, EVENT_T, MODE) \
    void binary_csrmv_wat_hybrid##SUFFIX( \
        const BE::Tensor weights, const BE::Tensor indices, const BE::Tensor indptr, \
        const BE::Tensor vector, const BE::Tensor task_begin_scratch, \
        const BE::Tensor task_end_scratch, const BE::Tensor status_scratch, \
        BE::Tensor output, BE::Tensor task_begin_output, BE::Tensor task_end_output, \
        BE::Tensor status_output, int64_t task_capacity, int64_t stream) \
    { \
        HybridSpmv##MODE<WEIGHT_T, EVENT_T, false>( \
            "binary_csrmv_wat_hybrid" #SUFFIX, weights, indices, indptr, vector, \
            task_begin_scratch, task_end_scratch, status_scratch, \
            output, task_begin_output, task_end_output, status_output, task_capacity, stream); \
    }

// @BE binary_csrmv_wat_hybrid_hetero_f32_bool
DEFINE_HYBRID_SPMV_BINARY(_hetero_f32_bool, float, int8_t, Hetero)
// @BE binary_csrmv_wat_hybrid_hetero_f32_float
DEFINE_HYBRID_SPMV_BINARY(_hetero_f32_float, float, float, Hetero)
// @BE binary_csrmv_wat_hybrid_hetero_f64_bool
DEFINE_HYBRID_SPMV_BINARY(_hetero_f64_bool, double, int8_t, Hetero)
// @BE binary_csrmv_wat_hybrid_hetero_f64_float
DEFINE_HYBRID_SPMV_BINARY(_hetero_f64_float, double, float, Hetero)
// @BE binary_csrmv_wat_hybrid_hetero_f16_bool
DEFINE_HYBRID_SPMV_BINARY(_hetero_f16_bool, __half, int8_t, Hetero)
// @BE binary_csrmv_wat_hybrid_hetero_f16_float
DEFINE_HYBRID_SPMV_BINARY(_hetero_f16_float, __half, float, Hetero)
// @BE binary_csrmv_wat_hybrid_hetero_bf16_bool
DEFINE_HYBRID_SPMV_BINARY(_hetero_bf16_bool, __nv_bfloat16, int8_t, Hetero)
// @BE binary_csrmv_wat_hybrid_hetero_bf16_float
DEFINE_HYBRID_SPMV_BINARY(_hetero_bf16_float, __nv_bfloat16, float, Hetero)
// @BE binary_csrmv_wat_hybrid_homo_f32_bool
DEFINE_HYBRID_SPMV_BINARY(_homo_f32_bool, float, int8_t, Homo)
// @BE binary_csrmv_wat_hybrid_homo_f32_float
DEFINE_HYBRID_SPMV_BINARY(_homo_f32_float, float, float, Homo)
// @BE binary_csrmv_wat_hybrid_homo_f64_bool
DEFINE_HYBRID_SPMV_BINARY(_homo_f64_bool, double, int8_t, Homo)
// @BE binary_csrmv_wat_hybrid_homo_f64_float
DEFINE_HYBRID_SPMV_BINARY(_homo_f64_float, double, float, Homo)
// @BE binary_csrmv_wat_hybrid_homo_f16_bool
DEFINE_HYBRID_SPMV_BINARY(_homo_f16_bool, __half, int8_t, Homo)
// @BE binary_csrmv_wat_hybrid_homo_f16_float
DEFINE_HYBRID_SPMV_BINARY(_homo_f16_float, __half, float, Homo)
// @BE binary_csrmv_wat_hybrid_homo_bf16_bool
DEFINE_HYBRID_SPMV_BINARY(_homo_bf16_bool, __nv_bfloat16, int8_t, Homo)
// @BE binary_csrmv_wat_hybrid_homo_bf16_float
DEFINE_HYBRID_SPMV_BINARY(_homo_bf16_float, __nv_bfloat16, float, Homo)

#undef DEFINE_HYBRID_SPMV_BINARY
