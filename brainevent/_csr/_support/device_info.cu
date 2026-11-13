// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

// Query the current CUDA device without compiling the CSR MM kernels.
#include <cstdint>
#include <cuda_runtime.h>

#include "brainevent/common.h"
#include "cuda_common.h"

// @BE csr_device_info
void csr_device_info(const BE::Tensor dummy, BE::Tensor info, int64_t stream)
{
  (void)dummy;
  BE_CHECK(info.dtype() == BE::DType::Int32 && info.ndim() == 1 && info.numel() == 2) << "expects int32[2]";

  int device = 0;
  int sms = 0;
  int l2_bytes = 0;
  BE_CUDA_CHECK(cudaGetDevice(&device));
  BE_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  BE_CUDA_CHECK(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, device));

  const int32_t host[2] = {sms, l2_bytes};
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  BE_CUDA_CHECK(cudaMemcpyAsync(info.data_ptr<int32_t>(), host, sizeof(host), cudaMemcpyHostToDevice, cuda_stream));
  // The source array is on the stack; finish the copy before returning.
  BE_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
}
