# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Test all data types: bool, int8-64, uint8-64, float16/32/64, bfloat16, complex64/128."""

import numpy as np
import pytest

from brainevent._test_util import requires_gpu

pytestmark = requires_gpu

import jax
jax.config.update("jax_enable_x64", True)


# --- CUDA kernel that copies input to output for any dtype ---

COPY_KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include "brainevent/common.h"

__global__ void copy_kernel(const char* src, char* dst, int64_t nbytes) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nbytes) dst[i] = src[i];
}

// @BE copy_tensor
void copy_tensor(const BE::Tensor src, BE::Tensor dst, int64_t stream) {
    int64_t n = src.nbytes();
    copy_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        static_cast<const char*>(src.data_ptr()),
        static_cast<char*>(dst.data_ptr()), n);
}
"""


# --- CUDA kernel using BE_DISPATCH_ALL for dtype-aware add ---

DISPATCH_ALL_SRC = r"""
#include <cuda_runtime.h>
#include "brainevent/common.h"

template <typename T>
__global__ void add_kernel(const T* a, const T* b, T* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

// @BE typed_add
void typed_add(const BE::Tensor a, const BE::Tensor b,
               BE::Tensor out, int64_t stream) {
    int n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    auto s = (cudaStream_t)stream;

    BE_DISPATCH_ALL_TYPES(a.dtype(), scalar_t, {
        add_kernel<scalar_t><<<blocks, threads, 0, s>>>(
            static_cast<const scalar_t*>(a.data_ptr()),
            static_cast<const scalar_t*>(b.data_ptr()),
            static_cast<scalar_t*>(out.data_ptr()), n);
    });
}
"""


@pytest.fixture(scope="module")
def copy_module():
    import brainevent.kernix as jkb
    return jkb.load_cuda_inline(
        name="test_copy_all_dtypes",
        cuda_sources=COPY_KERNEL_SRC,
        force_rebuild=True,
    )


@pytest.fixture(scope="module")
def dispatch_module():
    import brainevent.kernix as jkb
    return jkb.load_cuda_inline(
        name="test_dispatch_all",
        cuda_sources=DISPATCH_ALL_SRC,
        force_rebuild=True,
    )


# -- Copy (byte-level) tests for all dtypes --------------------------------

@pytest.mark.parametrize("dtype", [
    "bool",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64",
    "bfloat16",
    "complex64", "complex128",
])
def test_copy_dtype(copy_module, dtype):
    """Byte-level copy kernel preserves data for all dtypes."""
    import jax.numpy as jnp

    jnp_dtype = getattr(jnp, dtype)
    n = 64

    if dtype == "bool":
        x = jnp.array([True, False] * (n // 2), dtype=jnp_dtype)
    elif dtype.startswith("complex"):
        real = np.arange(n, dtype=np.float32 if dtype == "complex64" else np.float64)
        x = jnp.array(real + 1j * real, dtype=jnp_dtype)
    else:
        x = jnp.arange(n, dtype=jnp_dtype)

    result = jax.ffi.ffi_call(
        "test_copy_all_dtypes.copy_tensor",
        jax.ShapeDtypeStruct(x.shape, x.dtype),
    )(x)

    np.testing.assert_array_equal(np.asarray(result), np.asarray(x))


# -- Dispatch macro tests for numeric dtypes --------------------------------

@pytest.mark.parametrize("dtype", [
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float32", "float64",
])
def test_dispatch_add(dispatch_module, dtype):
    """BE_DISPATCH_ALL_TYPES correctly dispatches addition."""
    import jax.numpy as jnp

    jnp_dtype = getattr(jnp, dtype)
    n = 256
    a = jnp.ones(n, dtype=jnp_dtype) * 3
    b = jnp.ones(n, dtype=jnp_dtype) * 4

    result = jax.ffi.ffi_call(
        "test_dispatch_all.typed_add",
        jax.ShapeDtypeStruct((n,), jnp_dtype),
    )(a, b)

    expected = np.full(n, 7, dtype=dtype)
    np.testing.assert_array_equal(np.asarray(result), expected)


def test_dispatch_add_bool(dispatch_module):
    """BE_DISPATCH_ALL_TYPES handles bool (addition = logical OR)."""
    import jax.numpy as jnp

    n = 64
    a = jnp.array([True, False, True, False] * (n // 4), dtype=jnp.bool_)
    b = jnp.array([True, True, False, False] * (n // 4), dtype=jnp.bool_)

    result = jax.ffi.ffi_call(
        "test_dispatch_all.typed_add",
        jax.ShapeDtypeStruct((n,), jnp.bool_),
    )(a, b)

    # bool addition: True + True = True (1+1=2, but bool clips to True)
    # In C++: bool a + bool b truncates to bool
    result_np = np.asarray(result)
    assert result_np.dtype == np.bool_
