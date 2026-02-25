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

"""Test multi-dtype support (float32, float64)."""

import pytest
import numpy as np

import jax as _jax
import pytest as _pytest

requires_gpu = _pytest.mark.skipif(
    not (bool(_jax.devices("gpu")) if True else False),
    reason="No GPU detected via jax.devices('gpu')",
)

pytestmark = requires_gpu

# Enable 64-bit types in JAX
import jax
jax.config.update("jax_enable_x64", True)

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include "brainevent/common.h"

template <typename T>
__global__ void add_kernel(const T* a, const T* b, T* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

void typed_add(BE::Tensor a, BE::Tensor b,
               BE::Tensor out, int64_t stream) {
    int n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    auto s = (cudaStream_t)stream;

    // Dispatch on dtype
    switch (a.dtype()) {
        case BE::DType::Float32:
            add_kernel<float><<<blocks, threads, 0, s>>>(
                static_cast<const float*>(a.data_ptr()),
                static_cast<const float*>(b.data_ptr()),
                static_cast<float*>(out.data_ptr()), n);
            break;
        case BE::DType::Float64:
            add_kernel<double><<<blocks, threads, 0, s>>>(
                static_cast<const double*>(a.data_ptr()),
                static_cast<const double*>(b.data_ptr()),
                static_cast<double*>(out.data_ptr()), n);
            break;
        default:
            break;
    }
}
"""


@pytest.fixture(scope="module")
def typed_add_module():
    import brainevent.kernix as jkb
    return jkb.load_cuda_inline(
        name="test_typed_add",
        cuda_sources=CUDA_SRC,
        functions={"typed_add": ["arg", "arg", "ret", "stream"]},
        force_rebuild=True,
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_typed_add(typed_add_module, dtype):
    """Vector addition works for float32 and float64."""
    import jax
    import jax.numpy as jnp

    jnp_dtype = getattr(jnp, dtype)
    n = 1024
    a = jnp.ones(n, dtype=jnp_dtype) * 3.0
    b = jnp.ones(n, dtype=jnp_dtype) * 4.0

    result = jax.ffi.ffi_call(
        "test_typed_add.typed_add",
        jax.ShapeDtypeStruct((n,), jnp_dtype),
    )(a, b)

    expected = np.full(n, 7.0, dtype=dtype)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-5)


def test_2d_tensor(typed_add_module):
    """Works with multi-dimensional tensors."""
    import jax
    import jax.numpy as jnp

    a = jnp.ones((32, 64), dtype=jnp.float32)
    b = jnp.full((32, 64), 5.0, dtype=jnp.float32)

    result = jax.ffi.ffi_call(
        "test_typed_add.typed_add",
        jax.ShapeDtypeStruct((32, 64), jnp.float32),
    )(a, b)

    expected = np.full((32, 64), 6.0, dtype=np.float32)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-5)
