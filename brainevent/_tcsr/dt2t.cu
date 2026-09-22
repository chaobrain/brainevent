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
// ==============================================================================

// TCSR diagonal expansion. NT traverses canonical rows directly. T traverses
// mirror rows and uses permutation[mirror_slot] to preserve canonical slots.

#include "cuda_common.h"
#include "brainevent/common.h"

namespace {

template <typename T>
struct ValueTraits;

template <>
struct ValueTraits<float> {
    using AccT = float;
    static constexpr BE::DType dtype = BE::DType::Float32;
    __device__ static AccT read(float value) { return value; }
    __device__ static float write(AccT value) { return value; }
};

template <>
struct ValueTraits<double> {
    using AccT = double;
    static constexpr BE::DType dtype = BE::DType::Float64;
    __device__ static AccT read(double value) { return value; }
    __device__ static double write(AccT value) { return value; }
};

template <>
struct ValueTraits<__half> {
    using AccT = float;
    static constexpr BE::DType dtype = BE::DType::Float16;
    __device__ static AccT read(__half value) { return __half2float(value); }
    __device__ static __half write(AccT value) { return __float2half(value); }
};

template <>
struct ValueTraits<__nv_bfloat16> {
    using AccT = float;
    static constexpr BE::DType dtype = BE::DType::BFloat16;
    __device__ static AccT read(__nv_bfloat16 value) {
        return __bfloat162float(value);
    }
    __device__ static __nv_bfloat16 write(AccT value) {
        return __float2bfloat16(value);
    }
};

template <typename ValueT, bool Indexed>
__global__ void Dt2tMvKernel(
    const ValueT* __restrict__ y,
    const ValueT* __restrict__ weights,
    const int64_t* __restrict__ indptr,
    const int64_t* __restrict__ permutation,
    ValueT* __restrict__ output,
    int64_t rows)
{
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * warps_per_block
                           + warp_in_block;
         row < rows;
         row += static_cast<int64_t>(gridDim.x) * warps_per_block) {
        const typename ValueTraits<ValueT>::AccT y_value =
            ValueTraits<ValueT>::read(y[row]);
        const int64_t begin = indptr[row];
        const int64_t end = indptr[row + 1];
        for (int64_t slot = begin + lane; slot < end; slot += 32) {
            const int64_t value_slot = Indexed ? permutation[slot] : slot;
            output[value_slot] = ValueTraits<ValueT>::write(
                ValueTraits<ValueT>::read(weights[value_slot]) * y_value);
        }
    }
}

template <typename ValueT, bool Indexed>
__global__ void Dt2tMmKernel(
    const ValueT* __restrict__ y,
    const ValueT* __restrict__ weights,
    const int64_t* __restrict__ indptr,
    const int64_t* __restrict__ permutation,
    ValueT* __restrict__ output,
    int64_t rows,
    int64_t nnz,
    int64_t batch)
{
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int64_t task_count = batch * rows;
    for (int64_t task = static_cast<int64_t>(blockIdx.x) * warps_per_block
                            + warp_in_block;
         task < task_count;
         task += static_cast<int64_t>(gridDim.x) * warps_per_block) {
        const int64_t batch_id = task / rows;
        const int64_t row = task - batch_id * rows;
        const typename ValueTraits<ValueT>::AccT y_value =
            ValueTraits<ValueT>::read(y[batch_id * rows + row]);
        const ValueT* batch_weights = weights + batch_id * nnz;
        ValueT* batch_output = output + batch_id * nnz;
        const int64_t begin = indptr[row];
        const int64_t end = indptr[row + 1];
        for (int64_t slot = begin + lane; slot < end; slot += 32) {
            const int64_t value_slot = Indexed ? permutation[slot] : slot;
            batch_output[value_slot] = ValueTraits<ValueT>::write(
                ValueTraits<ValueT>::read(batch_weights[value_slot]) * y_value);
        }
    }
}

template <typename ValueT, bool Indexed>
void LaunchMv(
    const BE::Tensor y,
    const BE::Tensor weights,
    const BE::Tensor indptr,
    const BE::Tensor permutation,
    BE::Tensor output,
    cudaStream_t stream)
{
    const int64_t rows = static_cast<int64_t>(indptr.numel()) - 1;
    if (rows <= 0 || weights.numel() == 0) return;
    const int threads = 256;
    Dt2tMvKernel<ValueT, Indexed><<<BE_WARP_PER_ROW_GRID(rows), threads, 0, stream>>>(
        y.data_ptr<const ValueT>(),
        weights.data_ptr<const ValueT>(),
        indptr.data_ptr<const int64_t>(),
        permutation.data_ptr<const int64_t>(),
        output.data_ptr<ValueT>(),
        rows);
    BE_CHECK_KERNEL_LAUNCH();
}

template <typename ValueT, bool Indexed>
void LaunchMm(
    const BE::Tensor y,
    const BE::Tensor weights,
    const BE::Tensor indptr,
    const BE::Tensor permutation,
    BE::Tensor output,
    cudaStream_t stream)
{
    const int64_t rows = static_cast<int64_t>(indptr.numel()) - 1;
    const int64_t batch = static_cast<int64_t>(y.size(0));
    const int64_t nnz = static_cast<int64_t>(weights.size(1));
    if (rows <= 0 || batch <= 0 || nnz == 0) return;
    const int threads = 256;
    int64_t block_count = (batch * rows + 7) / 8;
    if (block_count > BE_WARP_PER_ROW_MAX_GRID) {
        block_count = BE_WARP_PER_ROW_MAX_GRID;
    }
    Dt2tMmKernel<ValueT, Indexed><<<static_cast<int>(block_count), threads, 0, stream>>>(
        y.data_ptr<const ValueT>(),
        weights.data_ptr<const ValueT>(),
        indptr.data_ptr<const int64_t>(),
        permutation.data_ptr<const int64_t>(),
        output.data_ptr<ValueT>(),
        rows,
        nnz,
        batch);
    BE_CHECK_KERNEL_LAUNCH();
}

#define CHECK_COMMON_DT2T(ValueT, IS_INDEXED, IS_MM, NAME)                 \
    BE_CHECK(y.dtype() == ValueTraits<ValueT>::dtype) << NAME              \
        << " expects matching y dtype";                                    \
    BE_CHECK(weights.dtype() == ValueTraits<ValueT>::dtype) << NAME        \
        << " expects matching weights dtype";                              \
    BE_CHECK(output.dtype() == ValueTraits<ValueT>::dtype) << NAME         \
        << " expects matching output dtype";                               \
    BE_CHECK(indices.dtype() == BE::DType::Int32) << NAME                  \
        << " expects indices=int32";                                       \
    BE_CHECK(indptr.dtype() == BE::DType::Int64) << NAME                   \
        << " expects indptr=int64";                                        \
    BE_CHECK(permutation.dtype() == BE::DType::Int64) << NAME              \
        << " expects permutation=int64";                                   \
    BE_CHECK(indices.ndim() == 1 && indptr.ndim() == 1 &&                  \
             permutation.ndim() == 1) << NAME                              \
        << " expects rank-1 structure";                                    \
    BE_CHECK(y.ndim() == ((IS_MM) ? 2 : 1) &&                              \
             weights.ndim() == ((IS_MM) ? 2 : 1) &&                        \
             output.ndim() == ((IS_MM) ? 2 : 1)) << NAME                   \
        << " received invalid data ranks";                                 \
    BE_CHECK(weights.numel() == output.numel()) << NAME                    \
        << " expects output shape to match weights";                       \
    BE_CHECK(indices.numel() == (IS_MM ? weights.size(1) : weights.size(0))) \
        << NAME << " expects one weight per sparse slot";                   \
    BE_CHECK(permutation.numel() == ((IS_INDEXED) ? indices.numel() : 0))  \
        << NAME << " received invalid permutation length";                 \
    BE_CHECK(indptr.numel() == (IS_MM ? y.size(1) : y.size(0)) + 1)        \
        << NAME << " expects indptr length to match neuron dimension";     \
    BE_CHECK(!(IS_MM) || weights.size(0) == y.size(0)) << NAME             \
        << " expects matching BN batch dimensions"

#define DEFINE_MV_ENTRY(NAME, SUFFIX, ValueT, IS_INDEXED)                  \
void NAME##SUFFIX(                                                         \
    const BE::Tensor y,                                                    \
    const BE::Tensor weights,                                              \
    const BE::Tensor indices,                                              \
    const BE::Tensor indptr,                                               \
    const BE::Tensor permutation,                                          \
    BE::Tensor output,                                                     \
    int64_t stream)                                                        \
{                                                                          \
    CHECK_COMMON_DT2T(ValueT, IS_INDEXED, false, #NAME #SUFFIX);           \
    LaunchMv<ValueT, IS_INDEXED>(                                          \
        y, weights, indptr, permutation, output,                           \
        reinterpret_cast<cudaStream_t>(stream));                           \
}

#define DEFINE_MM_ENTRY(NAME, SUFFIX, ValueT, IS_INDEXED)                  \
void NAME##SUFFIX(                                                         \
    const BE::Tensor y,                                                    \
    const BE::Tensor weights,                                              \
    const BE::Tensor indices,                                              \
    const BE::Tensor indptr,                                               \
    const BE::Tensor permutation,                                          \
    BE::Tensor output,                                                     \
    int64_t stream)                                                        \
{                                                                          \
    CHECK_COMMON_DT2T(ValueT, IS_INDEXED, true, #NAME #SUFFIX);            \
    LaunchMm<ValueT, IS_INDEXED>(                                          \
        y, weights, indptr, permutation, output,                           \
        reinterpret_cast<cudaStream_t>(stream));                           \
}

}  // namespace

// @BE csrmv_dt2t_nt_f16
DEFINE_MV_ENTRY(csrmv_dt2t_nt, _f16, __half, false)
// @BE csrmv_dt2t_nt_bf16
DEFINE_MV_ENTRY(csrmv_dt2t_nt, _bf16, __nv_bfloat16, false)
// @BE csrmv_dt2t_nt_f32
DEFINE_MV_ENTRY(csrmv_dt2t_nt, _f32, float, false)
// @BE csrmv_dt2t_nt_f64
DEFINE_MV_ENTRY(csrmv_dt2t_nt, _f64, double, false)

// @BE csrmv_dt2t_t_indexed_f16
DEFINE_MV_ENTRY(csrmv_dt2t_t_indexed, _f16, __half, true)
// @BE csrmv_dt2t_t_indexed_bf16
DEFINE_MV_ENTRY(csrmv_dt2t_t_indexed, _bf16, __nv_bfloat16, true)
// @BE csrmv_dt2t_t_indexed_f32
DEFINE_MV_ENTRY(csrmv_dt2t_t_indexed, _f32, float, true)
// @BE csrmv_dt2t_t_indexed_f64
DEFINE_MV_ENTRY(csrmv_dt2t_t_indexed, _f64, double, true)

// @BE csrmm_dt2t_nt_f16
DEFINE_MM_ENTRY(csrmm_dt2t_nt, _f16, __half, false)
// @BE csrmm_dt2t_nt_bf16
DEFINE_MM_ENTRY(csrmm_dt2t_nt, _bf16, __nv_bfloat16, false)
// @BE csrmm_dt2t_nt_f32
DEFINE_MM_ENTRY(csrmm_dt2t_nt, _f32, float, false)
// @BE csrmm_dt2t_nt_f64
DEFINE_MM_ENTRY(csrmm_dt2t_nt, _f64, double, false)

// @BE csrmm_dt2t_t_indexed_f16
DEFINE_MM_ENTRY(csrmm_dt2t_t_indexed, _f16, __half, true)
// @BE csrmm_dt2t_t_indexed_bf16
DEFINE_MM_ENTRY(csrmm_dt2t_t_indexed, _bf16, __nv_bfloat16, true)
// @BE csrmm_dt2t_t_indexed_f32
DEFINE_MM_ENTRY(csrmm_dt2t_t_indexed, _f32, float, true)
// @BE csrmm_dt2t_t_indexed_f64
DEFINE_MM_ENTRY(csrmm_dt2t_t_indexed, _f64, double, true)
