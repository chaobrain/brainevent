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

/*
 * dummy.cu -- Scatter-only dummy FCNMV CUDA kernels
 * =================================================
 *
 * These kernels are benchmark-only backends used to isolate preprocessing and
 * launch overheads from the real sparse computation.
 *
 * Design rules:
 *   - scatter only
 *   - homo weights only
 *   - no-op device kernels (no representation reads, no sparse updates)
 *   - FFI still zeroes the output so the call is well-defined
 *   - full-launch vs active-launch differences are preserved in FFI launch dims
 */

#include "cuda_common.h"
#include "brainevent/common.h"

// ============================================================================
// generic no-op launch kernels
// ============================================================================

__global__ void _dummy_tpr_launch_kern(int n_launch) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_launch) return;
}

__global__ void _dummy_wpr_launch_kern(int n_launch) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = tid / 32;
    if (row >= n_launch) return;
}

// ============================================================================
// FFI entry points
// ============================================================================

#define FFI_DUMMY_BINARY_SCATTER_HOMO(SUFFIX, WEIGHT_C_T, SPIKE_C_T)                 \
void dummy_binary_fcnmv_scatter_homo##SUFFIX(                                         \
    const BE::Tensor weights, const BE::Tensor indices,                               \
    const BE::Tensor spikes, BE::Tensor output, int64_t stream                        \
) {                                                                                    \
    (void)weights;                                                                     \
    (void)spikes;                                                                      \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                           \
    int n_pre  = static_cast<int>(indices.size(0));                                    \
    int n_conn = static_cast<int>(indices.size(1));                                    \
    int n_post = static_cast<int>(output.size(0));                                     \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());         \
    (void)d_idx;                                                                        \
    void* d_out = output.data_ptr();                                                    \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(WEIGHT_C_T), s);                 \
    if (n_pre == 0 || n_conn <= 0) return;                                             \
    int bsz = 256;                                                                      \
    if ((int64_t)n_conn * 2084000 > (int64_t)n_pre * 1539) {                           \
        int warps_per_block = bsz / 32;                                                 \
        int n_blocks = (n_pre + warps_per_block - 1) / warps_per_block;                \
        _dummy_wpr_launch_kern<<<n_blocks, bsz, 0, s>>>(n_pre);                        \
    } else {                                                                            \
        int n_blocks = (n_pre + bsz - 1) / bsz;                                         \
        _dummy_tpr_launch_kern<<<n_blocks, bsz, 0, s>>>(n_pre);                        \
    }                                                                                   \
    BE_CHECK_KERNEL_LAUNCH();                                                           \
}

#define FFI_DUMMY_BITPACK_SCATTER_HOMO(SUFFIX, WEIGHT_C_T)                             \
void dummy_bitpack_binary_fcnmv_scatter_homo##SUFFIX(                                   \
    const BE::Tensor weights, const BE::Tensor indices,                                 \
    const BE::Tensor packed, BE::Tensor output,                                         \
    int64_t pack_axis, int64_t stream                                                   \
) {                                                                                     \
    (void)weights;                                                                       \
    (void)packed;                                                                        \
    (void)pack_axis;                                                                     \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                            \
    int n_pre  = static_cast<int>(indices.size(0));                                     \
    int n_conn = static_cast<int>(indices.size(1));                                     \
    int n_post = static_cast<int>(output.size(0));                                      \
    const int32_t*    d_idx = static_cast<const int32_t*>(indices.data_ptr());          \
    (void)d_idx;                                                                         \
    void* d_out = output.data_ptr();                                                     \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(WEIGHT_C_T), s);                  \
    if (n_pre == 0 || n_conn <= 0) return;                                              \
    int bsz = 256;                                                                       \
    int n_blocks = (n_pre + bsz - 1) / bsz;                                             \
    _dummy_tpr_launch_kern<<<n_blocks, bsz, 0, s>>>(n_pre);                             \
    BE_CHECK_KERNEL_LAUNCH();                                                            \
}

#define FFI_DUMMY_COMPACT_PACKED_SCATTER_HOMO(SUFFIX, WEIGHT_C_T)                       \
void dummy_compact_binary_fcnmv_scatter_homo##SUFFIX(                                    \
    const BE::Tensor weights, const BE::Tensor indices,                                  \
    const BE::Tensor packed, const BE::Tensor active_ids,                                \
    const BE::Tensor n_active, BE::Tensor output,                                        \
    int64_t stream                                                                       \
) {                                                                                      \
    (void)weights;                                                                        \
    (void)packed;                                                                         \
    (void)active_ids;                                                                     \
    (void)n_active;                                                                       \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                             \
    int n_orig = static_cast<int>(active_ids.size(0));                                   \
    int n_conn = static_cast<int>(indices.size(1));                                      \
    int n_post = static_cast<int>(output.size(0));                                       \
    const int32_t*    d_idx  = static_cast<const int32_t*>(indices.data_ptr());          \
    (void)d_idx;                                                                          \
    void* d_out = output.data_ptr();                                                      \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(WEIGHT_C_T), s);                   \
    if (n_orig == 0 || n_conn <= 0) return;                                              \
    int bsz = 256;                                                                        \
    int n_blocks = (n_orig + bsz - 1) / bsz;                                             \
    _dummy_tpr_launch_kern<<<n_blocks, bsz, 0, s>>>(n_orig);                             \
    BE_CHECK_KERNEL_LAUNCH();                                                             \
}

#define FFI_DUMMY_COMPACT_VECTOR_FULL_SCATTER_HOMO(SUFFIX, WEIGHT_C_T)                   \
void dummy_compact_binary_fcnmv_scatter_homo_vector_full##SUFFIX(                         \
    const BE::Tensor weights, const BE::Tensor indices,                                   \
    const BE::Tensor packed, const BE::Tensor active_ids,                                 \
    const BE::Tensor n_active, BE::Tensor output,                                         \
    int64_t stream                                                                        \
) {                                                                                       \
    (void)weights;                                                                         \
    (void)packed;                                                                          \
    (void)n_active;                                                                        \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                              \
    int n_orig = static_cast<int>(active_ids.size(0));                                    \
    int n_conn = static_cast<int>(indices.size(1));                                       \
    int n_post = static_cast<int>(output.size(0));                                        \
    const int32_t*    d_idx  = static_cast<const int32_t*>(indices.data_ptr());           \
    const int32_t*    d_aids = static_cast<const int32_t*>(active_ids.data_ptr());        \
    (void)d_idx;                                                                           \
    (void)d_aids;                                                                          \
    void* d_out = output.data_ptr();                                                       \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(WEIGHT_C_T), s);                    \
    if (n_orig == 0 || n_conn <= 0) return;                                               \
    int bsz = 256;                                                                         \
    int n_blocks = (n_orig + bsz - 1) / bsz;                                              \
    _dummy_tpr_launch_kern<<<n_blocks, bsz, 0, s>>>(n_orig);                              \
    BE_CHECK_KERNEL_LAUNCH();                                                              \
}

#define FFI_DUMMY_COMPACT_VECTOR_ACTIVE_SCATTER_HOMO(SUFFIX, WEIGHT_C_T)                 \
void dummy_compact_binary_fcnmv_scatter_homo_vector_active##SUFFIX(                       \
    const BE::Tensor weights, const BE::Tensor indices,                                   \
    const BE::Tensor packed, const BE::Tensor active_ids,                                 \
    const BE::Tensor n_active, BE::Tensor output,                                         \
    int64_t stream                                                                        \
) {                                                                                       \
    (void)weights;                                                                         \
    (void)packed;                                                                          \
    (void)active_ids;                                                                      \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                              \
    int n_conn = static_cast<int>(indices.size(1));                                       \
    int n_post = static_cast<int>(output.size(0));                                        \
    const int32_t*    d_idx  = static_cast<const int32_t*>(indices.data_ptr());           \
    const int32_t*    d_na   = static_cast<const int32_t*>(n_active.data_ptr());          \
    (void)d_idx;                                                                           \
    void* d_out = output.data_ptr();                                                       \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(WEIGHT_C_T), s);                    \
    if (n_conn <= 0) return;                                                               \
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;                  \
    BE_CUDA_CHECK(cudaStreamIsCapturing(s, &capture_status));                              \
    if (capture_status != cudaStreamCaptureStatusNone) {                                   \
        fprintf(stderr,                                                                    \
                "[be] dummy compact vector_active backend does not support CUDA graph capture.\n"); \
        fflush(stderr);                                                                    \
        abort();                                                                           \
    }                                                                                      \
    int32_t host_n_active = 0;                                                             \
    BE_CUDA_CHECK(cudaMemcpyAsync(                                                         \
        &host_n_active, d_na, sizeof(int32_t), cudaMemcpyDeviceToHost, s));                \
    BE_CUDA_CHECK(cudaStreamSynchronize(s));                                               \
    if (host_n_active <= 0) return;                                                        \
    int bsz = 256;                                                                         \
    int n_blocks = (host_n_active + bsz - 1) / bsz;                                       \
    _dummy_tpr_launch_kern<<<n_blocks, bsz, 0, s>>>(host_n_active);                       \
    BE_CHECK_KERNEL_LAUNCH();                                                              \
}

// ============================================================================
// FFI instantiations
// ============================================================================

// @BE dummy_binary_fcnmv_scatter_homo_bool_f32 arg arg arg ret stream
FFI_DUMMY_BINARY_SCATTER_HOMO(_bool_f32, float, uint8_t)
// @BE dummy_binary_fcnmv_scatter_homo_float_f32 arg arg arg ret stream
FFI_DUMMY_BINARY_SCATTER_HOMO(_float_f32, float, float)
// @BE dummy_binary_fcnmv_scatter_homo_bool_f64 arg arg arg ret stream
FFI_DUMMY_BINARY_SCATTER_HOMO(_bool_f64, double, uint8_t)
// @BE dummy_binary_fcnmv_scatter_homo_float_f64 arg arg arg ret stream
FFI_DUMMY_BINARY_SCATTER_HOMO(_float_f64, double, double)
// @BE dummy_binary_fcnmv_scatter_homo_bool_f16 arg arg arg ret stream
FFI_DUMMY_BINARY_SCATTER_HOMO(_bool_f16, __half, uint8_t)
// @BE dummy_binary_fcnmv_scatter_homo_float_f16 arg arg arg ret stream
FFI_DUMMY_BINARY_SCATTER_HOMO(_float_f16, __half, __half)
// @BE dummy_binary_fcnmv_scatter_homo_bool_bf16 arg arg arg ret stream
FFI_DUMMY_BINARY_SCATTER_HOMO(_bool_bf16, __nv_bfloat16, uint8_t)
// @BE dummy_binary_fcnmv_scatter_homo_float_bf16 arg arg arg ret stream
FFI_DUMMY_BINARY_SCATTER_HOMO(_float_bf16, __nv_bfloat16, __nv_bfloat16)

// @BE dummy_bitpack_binary_fcnmv_scatter_homo_f32 arg arg arg ret attr.pack_axis:int64 stream
FFI_DUMMY_BITPACK_SCATTER_HOMO(_f32, float)
// @BE dummy_bitpack_binary_fcnmv_scatter_homo_f64 arg arg arg ret attr.pack_axis:int64 stream
FFI_DUMMY_BITPACK_SCATTER_HOMO(_f64, double)
// @BE dummy_bitpack_binary_fcnmv_scatter_homo_f16 arg arg arg ret attr.pack_axis:int64 stream
FFI_DUMMY_BITPACK_SCATTER_HOMO(_f16, __half)
// @BE dummy_bitpack_binary_fcnmv_scatter_homo_bf16 arg arg arg ret attr.pack_axis:int64 stream
FFI_DUMMY_BITPACK_SCATTER_HOMO(_bf16, __nv_bfloat16)

// @BE dummy_compact_binary_fcnmv_scatter_homo_f32 arg arg arg arg arg ret stream
FFI_DUMMY_COMPACT_PACKED_SCATTER_HOMO(_f32, float)
// @BE dummy_compact_binary_fcnmv_scatter_homo_vector_full_f32 arg arg arg arg arg ret stream
FFI_DUMMY_COMPACT_VECTOR_FULL_SCATTER_HOMO(_f32, float)
// @BE dummy_compact_binary_fcnmv_scatter_homo_vector_active_f32 arg arg arg arg arg ret stream
FFI_DUMMY_COMPACT_VECTOR_ACTIVE_SCATTER_HOMO(_f32, float)

// @BE dummy_compact_binary_fcnmv_scatter_homo_f64 arg arg arg arg arg ret stream
FFI_DUMMY_COMPACT_PACKED_SCATTER_HOMO(_f64, double)
// @BE dummy_compact_binary_fcnmv_scatter_homo_vector_full_f64 arg arg arg arg arg ret stream
FFI_DUMMY_COMPACT_VECTOR_FULL_SCATTER_HOMO(_f64, double)
// @BE dummy_compact_binary_fcnmv_scatter_homo_vector_active_f64 arg arg arg arg arg ret stream
FFI_DUMMY_COMPACT_VECTOR_ACTIVE_SCATTER_HOMO(_f64, double)

// @BE dummy_compact_binary_fcnmv_scatter_homo_f16 arg arg arg arg arg ret stream
FFI_DUMMY_COMPACT_PACKED_SCATTER_HOMO(_f16, __half)
// @BE dummy_compact_binary_fcnmv_scatter_homo_vector_full_f16 arg arg arg arg arg ret stream
FFI_DUMMY_COMPACT_VECTOR_FULL_SCATTER_HOMO(_f16, __half)
// @BE dummy_compact_binary_fcnmv_scatter_homo_vector_active_f16 arg arg arg arg arg ret stream
FFI_DUMMY_COMPACT_VECTOR_ACTIVE_SCATTER_HOMO(_f16, __half)

// @BE dummy_compact_binary_fcnmv_scatter_homo_bf16 arg arg arg arg arg ret stream
FFI_DUMMY_COMPACT_PACKED_SCATTER_HOMO(_bf16, __nv_bfloat16)
// @BE dummy_compact_binary_fcnmv_scatter_homo_vector_full_bf16 arg arg arg arg arg ret stream
FFI_DUMMY_COMPACT_VECTOR_FULL_SCATTER_HOMO(_bf16, __nv_bfloat16)
// @BE dummy_compact_binary_fcnmv_scatter_homo_vector_active_bf16 arg arg arg arg arg ret stream
FFI_DUMMY_COMPACT_VECTOR_ACTIVE_SCATTER_HOMO(_bf16, __nv_bfloat16)
