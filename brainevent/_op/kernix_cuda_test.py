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

"""Kernix CUDA end-to-end tests."""

import pytest
import numpy as np

from brainevent._test_util import requires_gpu

VADD_CUDA_SRC = r"""
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
    import brainevent
    return brainevent.load_cuda_inline(
        name="test_vadd",
        cuda_sources=VADD_CUDA_SRC,
        functions={"vector_add": ["arg", "arg", "ret", "stream"]},
        force_rebuild=True,
        verbose=True,
    )


@requires_gpu
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


@requires_gpu
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


@requires_gpu
def test_module_attributes(vadd_module):
    """CompiledModule exposes expected attributes."""
    import sys
    ext = ".dylib" if sys.platform == "darwin" else ".dll" if sys.platform == "win32" else ".so"
    assert "vector_add" in vadd_module.function_names
    assert vadd_module.path.endswith(ext)


@requires_gpu
def test_list_registered_targets(vadd_module):
    """Targets appear in the global registry."""
    import brainevent
    targets = brainevent.list_registered_targets()
    assert "test_vadd.vector_add" in targets


# ---------------------------------------------------------------------------
# Multiple output CUDA FFI tests
# ---------------------------------------------------------------------------

MULTI_OUT_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include "brainevent/common.h"

__global__ void split_kernel(const float* x, float* lo, float* hi,
                             int n, int split) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < split) lo[idx] = x[idx];
    if (idx < n - split) hi[idx] = x[split + idx];
}

void min_max(BE::Tensor x, BE::Tensor out_min,
             BE::Tensor out_max, int64_t stream) {
    // Simple test: copy first half to out_min, second half to out_max
    int n = x.numel();
    int half = n / 2;
    split_kernel<<<(n+255)/256, 256, 0, (cudaStream_t)stream>>>(
        static_cast<const float*>(x.data_ptr()),
        static_cast<float*>(out_min.data_ptr()),
        static_cast<float*>(out_max.data_ptr()),
        n, half);
}
"""


@pytest.fixture(scope="module")
def multi_out_module():
    import brainevent
    return brainevent.load_cuda_inline(
        name="test_multi_out",
        cuda_sources=MULTI_OUT_CUDA_SRC,
        functions={
            "min_max": ["arg", "ret", "ret", "stream"],
        },
        force_rebuild=True,
    )


@requires_gpu
def test_two_outputs(multi_out_module):
    """Function with two output buffers."""
    import jax
    import jax.numpy as jnp

    n = 256
    x = jnp.arange(n, dtype=jnp.float32)

    lo, hi = jax.ffi.ffi_call(
        "test_multi_out.min_max",
        (
            jax.ShapeDtypeStruct((n // 2,), jnp.float32),
            jax.ShapeDtypeStruct((n // 2,), jnp.float32),
        ),
    )(x)

    np.testing.assert_allclose(
        np.asarray(lo), np.arange(n // 2, dtype=np.float32)
    )
    np.testing.assert_allclose(
        np.asarray(hi), np.arange(n // 2, n, dtype=np.float32)
    )
