// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

#ifndef HYBRID_BATCHEDV3_SPMM_FFI_DEVICE_CUH
#define HYBRID_BATCHEDV3_SPMM_FFI_DEVICE_CUH

// Prepared plans use the device's L2 size. Check the executing device before
// touching scratch; zero-valued expectations are reserved for raw FFI tests.
inline void check_prepared_spmm_device(const char *name, int64_t expected_sms,
                                       int64_t expected_l2_bytes)
{
  if (expected_sms == 0 && expected_l2_bytes == 0)
    return;
  BE_CHECK(expected_sms > 0 && expected_l2_bytes > 0)
      << name << ": invalid prepared device information";
  hybrid::Device actual;
  BE_CUDA_CHECK(hybrid::device_info(&actual));
  BE_CHECK(actual.sms == expected_sms && actual.l2_bytes == expected_l2_bytes)
      << name << ": prepared CSRConfig device does not match the executing GPU; prepare it for this device";
}

#endif
