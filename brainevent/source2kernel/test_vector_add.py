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

"""End-to-end test: compile a vector_add CUDA kernel and call it via JAX FFI."""

import pytest
import numpy as np

import jax as _jax
import pytest as _pytest

requires_gpu = _pytest.mark.skipif(
    not (bool(_jax.devices("gpu")) if True else False),
    reason="No GPU detected via jax.devices('gpu')",
)

pytestmark = requires_gpu

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include "brainevent/common.h"

__global__ void vector_add_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

void vector_add(BE::Tensor a, BE::Tensor b,
                BE::Tensor out, int64_t stream) {
    int n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        static_cast<const float*>(a.data_ptr()),
        static_cast<const float*>(b.data_ptr()),
        static_cast<float*>(out.data_ptr()), n);
    BE_CUDA_CHECK(cudaGetLastError());
}
"""


@pytest.fixture(scope="module")
def vadd_module():
    """Compile the vector_add kernel once for all tests in this module."""
    import brainevent.source2kernel as jkb
    return jkb.load_cuda_inline(
        name="test_vadd",
        cuda_sources=CUDA_SRC,
        functions={"vector_add": ["arg", "arg", "ret", "stream"]},
        force_rebuild=True,
        verbose=True,
    )


def test_basic_vector_add(vadd_module):
    """Basic correctness: a + b == expected."""
    import jax
    import jax.numpy as jnp

    a = jnp.arange(1024, dtype=jnp.float32)
    b = jnp.full(1024, 2.0, dtype=jnp.float32)

    result = jax.ffi.ffi_call(
        "test_vadd.vector_add",
        jax.ShapeDtypeStruct((1024,), jnp.float32),
    )(a, b)

    expected = np.arange(1024, dtype=np.float32) + 2.0
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-5)


def test_jit_vector_add(vadd_module):
    """Works under @jax.jit."""
    import jax
    import jax.numpy as jnp

    @jax.jit
    def add_jit(x, y):
        return jax.ffi.ffi_call(
            "test_vadd.vector_add",
            jax.ShapeDtypeStruct(x.shape, x.dtype),
        )(x, y)

    a = jnp.ones(512, dtype=jnp.float32)
    b = jnp.ones(512, dtype=jnp.float32) * 3.0
    result = add_jit(a, b)
    np.testing.assert_allclose(np.asarray(result), np.full(512, 4.0), rtol=1e-5)


def test_large_array(vadd_module):
    """Works with large arrays (1M elements)."""
    import jax
    import jax.numpy as jnp

    n = 1_000_000
    a = jnp.ones(n, dtype=jnp.float32)
    b = jnp.ones(n, dtype=jnp.float32) * 7.0

    result = jax.ffi.ffi_call(
        "test_vadd.vector_add",
        jax.ShapeDtypeStruct((n,), jnp.float32),
    )(a, b)

    np.testing.assert_allclose(np.asarray(result), np.full(n, 8.0), rtol=1e-5)


def test_module_attributes(vadd_module):
    """CompiledModule exposes expected attributes."""
    import sys
    ext = ".dylib" if sys.platform == "darwin" else ".dll" if sys.platform == "win32" else ".so"
    assert "vector_add" in vadd_module.function_names
    assert vadd_module.path.endswith(ext)


def test_list_registered_targets(vadd_module):
    """Targets appear in the global registry."""
    import brainevent.source2kernel as jkb
    targets = jkb.list_registered_targets()
    assert "test_vadd.vector_add" in targets
