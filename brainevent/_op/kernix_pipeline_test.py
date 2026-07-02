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

"""Kernix pipeline integration tests."""

import os
import platform
import time
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
from brainevent._error import KernelError
from brainevent._op.kernix_pipeline import _cache_header_paths
from brainevent._test_util import requires_gpu


PIPELINE_CUDA_SRC = r"""
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
    vector_add_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        static_cast<const float*>(a.data_ptr()),
        static_cast<const float*>(b.data_ptr()),
        static_cast<float*>(out.data_ptr()), n);
    BE_CUDA_CHECK(cudaGetLastError());
}
"""


@pytest.fixture(scope="module")
def pipeline_cuda_module():
    return brainevent.load_cuda_inline(
        name="test_pipeline_vadd",
        cuda_sources=PIPELINE_CUDA_SRC,
        functions={"vector_add": ["arg", "arg", "ret", "stream"]},
        force_rebuild=True,
    )


@requires_gpu
def test_load_cuda_inline_registers_and_runs_kernel(pipeline_cuda_module):
    a = jnp.arange(1024, dtype=jnp.float32)
    b = jnp.full(1024, 2.0, dtype=jnp.float32)

    result = jax.ffi.ffi_call(
        "test_pipeline_vadd.vector_add",
        jax.ShapeDtypeStruct((1024,), jnp.float32),
    )(a, b)

    expected = np.arange(1024, dtype=np.float32) + 2.0
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-5)


PIPELINE_CPP_SRC = r"""
#include "brainevent/common.h"

void add_one_cpu(const BE::Tensor x, BE::Tensor y) {
    int n = x.numel();
    const float* in_ptr = static_cast<const float*>(x.data_ptr());
    float* out_ptr = static_cast<float*>(y.data_ptr());
    for (int i = 0; i < n; ++i) {
        out_ptr[i] = in_ptr[i] + 1.0f;
    }
}
"""


@pytest.fixture(scope="module")
def pipeline_cpp_module():
    return brainevent.load_cpp_inline(
        name="test_pipeline_cpu_add_one",
        cpp_sources=PIPELINE_CPP_SRC,
        functions=["add_one_cpu"],
        force_rebuild=True,
    )


@pytest.mark.skipif(
    platform.platform().startswith("Windows"),
    reason="Windows is not supported yet.",
)
def test_load_cpp_inline_registers_and_runs_cpu_kernel(pipeline_cpp_module):
    cpu = jax.devices("cpu")[0]
    x = jax.device_put(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32), cpu)

    result = jax.ffi.ffi_call(
        "test_pipeline_cpu_add_one.add_one_cpu",
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        vmap_method="broadcast_all",
    )(x)

    expected = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(result), expected)


def test_empty_functions_guard():
    with pytest.raises(KernelError):
        brainevent.load_cpp_inline("noop", "int main(){return 0;}", functions={})


PIPELINE_ANNOTATION_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include "brainevent/common.h"

__global__ void add_k(const float* a, const float* b, float* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = a[i] + b[i];
}

// @BE vector_add
void vector_add(const BE::Tensor a, const BE::Tensor b,
                BE::Tensor out, int64_t stream) {
    int n = a.numel();
    add_k<<<(n+255)/256, 256, 0, (cudaStream_t)stream>>>(
        static_cast<const float*>(a.data_ptr()),
        static_cast<const float*>(b.data_ptr()),
        static_cast<float*>(out.data_ptr()), n);
}
"""


@pytest.fixture(scope="module")
def annotation_module():
    return brainevent.load_cuda_inline(
        name="test_pipeline_annotation_vadd",
        cuda_sources=PIPELINE_ANNOTATION_CUDA_SRC,
        force_rebuild=True,
    )


@requires_gpu
def test_annotations_feed_cuda_pipeline(annotation_module):
    a = jnp.ones(512, dtype=jnp.float32)
    b = jnp.full(512, 2.0, dtype=jnp.float32)

    result = jax.ffi.ffi_call(
        "test_pipeline_annotation_vadd.vector_add",
        jax.ShapeDtypeStruct((512,), jnp.float32),
    )(a, b)

    expected = np.full(512, 3.0, dtype=np.float32)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-5)


PIPELINE_CACHE_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include "brainevent/common.h"

__global__ void scale_kernel(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = x[idx] * 2.0f;
}

void scale2x(BE::Tensor x, BE::Tensor out, int64_t stream) {
    int n = x.numel();
    scale_kernel<<<(n+255)/256, 256, 0, (cudaStream_t)stream>>>(
        static_cast<const float*>(x.data_ptr()),
        static_cast<float*>(out.data_ptr()), n);
}
"""


@requires_gpu
def test_cache_hit_is_faster():
    brainevent.clear_cache("test_pipeline_cache_speed")

    t0 = time.monotonic()
    brainevent.load_cuda_inline(
        name="test_pipeline_cache_speed",
        cuda_sources=PIPELINE_CACHE_CUDA_SRC,
        functions={"scale2x": ["arg", "ret", "stream"]},
        target_prefix="pipeline_cache_a",
    )
    t_first = time.monotonic() - t0

    t0 = time.monotonic()
    brainevent.load_cuda_inline(
        name="test_pipeline_cache_speed",
        cuda_sources=PIPELINE_CACHE_CUDA_SRC,
        functions={"scale2x": ["arg", "ret", "stream"]},
        target_prefix="pipeline_cache_b",
    )
    t_second = time.monotonic() - t0

    assert t_second < t_first * 0.5, (
        f"Cache not effective: first={t_first:.2f}s, second={t_second:.2f}s"
    )


@requires_gpu
def test_force_rebuild():
    mod = brainevent.load_cuda_inline(
        name="test_pipeline_force_rb",
        cuda_sources=PIPELINE_CACHE_CUDA_SRC,
        functions={"scale2x": ["arg", "ret", "stream"]},
        force_rebuild=True,
        target_prefix="pipeline_force_rb_a",
    )
    assert "scale2x" in mod.function_names

    mod2 = brainevent.load_cuda_inline(
        name="test_pipeline_force_rb",
        cuda_sources=PIPELINE_CACHE_CUDA_SRC,
        functions={"scale2x": ["arg", "ret", "stream"]},
        force_rebuild=True,
        target_prefix="pipeline_force_rb_b",
    )
    assert "scale2x" in mod2.function_names


@requires_gpu
def test_clear_cache():
    brainevent.load_cuda_inline(
        name="test_pipeline_clear",
        cuda_sources=PIPELINE_CACHE_CUDA_SRC,
        functions={"scale2x": ["arg", "ret", "stream"]},
        target_prefix="pipeline_clear_test",
    )

    removed = brainevent.clear_cache("test_pipeline_clear")
    assert removed >= 1


def test_cache_header_paths_cover_injected_headers():
    be_inc = os.path.join(os.path.dirname(brainevent.__file__), "include")
    toolchain = types.SimpleNamespace(
        brainevent_include_dir=be_inc,
        xla_ffi_include_dir="/nonexistent-xla-include",
    )

    paths = _cache_header_paths(toolchain)
    names = {os.path.basename(path) for path in paths}

    for required in ("ffi_compat.h", "check.h", "tensor.h", "dtypes.h", "cuda_common.h"):
        assert required in names, f"{required} missing from cache header set: {names}"
    for path in paths:
        assert os.path.isfile(path)


@requires_gpu
def test_diagnostics_runs():
    brainevent.print_diagnostics()
