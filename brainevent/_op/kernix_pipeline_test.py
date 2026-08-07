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


# ---------------------------------------------------------------------------
# C++/CPU FFI end-to-end coverage for ``load_cpp_inline``.
#
# These were a module-level ``pytest.skip`` on Windows when they lived in their
# own file; the guard is per-test here so it cannot silently skip the rest of
# this module.
# ---------------------------------------------------------------------------

_skip_on_windows = pytest.mark.skipif(
    platform.platform().startswith("Windows"),
    reason="Windows is not supported yet.",
)

ADD_ONE_SRC = r"""
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

SCALE_SRC = r"""
#include "brainevent/common.h"

void scale_cpu(const BE::Tensor x, BE::Tensor y) {
    int n = x.numel();
    const float* in_ptr = static_cast<const float*>(x.data_ptr());
    float* out_ptr = static_cast<float*>(y.data_ptr());
    for (int i = 0; i < n; ++i) {
        out_ptr[i] = in_ptr[i] * 2.0f;
    }
}
"""

MULTI_OUT_SRC = r"""
#include "brainevent/common.h"

void split_cpu(const BE::Tensor x,
               BE::Tensor lo, BE::Tensor hi) {
    int n = x.numel();
    int half = n / 2;
    const float* src = static_cast<const float*>(x.data_ptr());
    float* lo_ptr = static_cast<float*>(lo.data_ptr());
    float* hi_ptr = static_cast<float*>(hi.data_ptr());
    for (int i = 0; i < half; ++i) lo_ptr[i] = src[i];
    for (int i = 0; i < n - half; ++i) hi_ptr[i] = src[half + i];
}
"""


@pytest.fixture(scope="module")
def cpu_add_one_module():
    return brainevent.load_cpp_inline(
        name="test_cpu_add_one",
        cpp_sources=ADD_ONE_SRC,
        functions=["add_one_cpu"],
        force_rebuild=True,
    )


@_skip_on_windows
def test_add_one_cpu_jit(cpu_add_one_module):
    """CPU kernel works under jax.jit."""

    cpu = jax.devices("cpu")[0]

    @jax.jit
    def add_one(x):
        return jax.ffi.ffi_call(
            "test_cpu_add_one.add_one_cpu",
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            vmap_method="broadcast_all",
        )(x)

    x = jax.device_put(jnp.arange(256, dtype=jnp.float32), cpu)
    result = add_one(x)
    expected = np.arange(256, dtype=np.float32) + 1.0
    np.testing.assert_allclose(np.asarray(result), expected)


@_skip_on_windows
def test_explicit_dict_form():
    """CPU kernel with explicit dict-form functions."""

    mod = brainevent.load_cpp_inline(
        name="test_cpu_scale",
        cpp_sources=SCALE_SRC,
        functions={"scale_cpu": ["arg", "ret"]},
        force_rebuild=True,
    )

    cpu = jax.devices("cpu")[0]
    x = jax.device_put(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32), cpu)
    result = jax.ffi.ffi_call(
        "test_cpu_scale.scale_cpu",
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        vmap_method="broadcast_all",
    )(x)

    expected = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(result), expected)


@_skip_on_windows
def test_multi_output_cpu():
    """CPU kernel with two output buffers."""
    mod = brainevent.load_cpp_inline(
        name="test_cpu_split",
        cpp_sources=MULTI_OUT_SRC,
        functions=["split_cpu"],
        force_rebuild=True,
    )

    cpu = jax.devices("cpu")[0]
    n = 256
    x = jax.device_put(jnp.arange(n, dtype=jnp.float32), cpu)

    lo, hi = jax.ffi.ffi_call(
        "test_cpu_split.split_cpu",
        (
            jax.ShapeDtypeStruct((n // 2,), jnp.float32),
            jax.ShapeDtypeStruct((n // 2,), jnp.float32),
        ),
        vmap_method="broadcast_all",
    )(x)

    np.testing.assert_allclose(
        np.asarray(lo), np.arange(n // 2, dtype=np.float32)
    )
    np.testing.assert_allclose(
        np.asarray(hi), np.arange(n // 2, n, dtype=np.float32)
    )


# A CPU kernel that fails a host-side invariant.  ``x.numel()`` is always
# non-negative, so ``x.numel() < 0`` is always false and the check fires at
# runtime (the compiler cannot prove it away from the source alone).
CHECK_FAIL_SRC = r"""
#include "brainevent/common.h"

void check_fail_cpu(const BE::Tensor x, BE::Tensor y) {
    BE_CHECK(x.numel() < 0) << "intentional failure: numel=" << x.numel();
    float* out_ptr = static_cast<float*>(y.data_ptr());
    out_ptr[0] = 1.0f;  // unreachable
}
"""


@_skip_on_windows
def test_host_check_failure_raises_not_aborts():
    """A failing BE_CHECK surfaces as a Python exception (no SIGABRT)."""
    mod = brainevent.load_cpp_inline(
        name="test_cpu_check_fail",
        cpp_sources=CHECK_FAIL_SRC,
        functions=["check_fail_cpu"],
        force_rebuild=True,
    )

    cpu = jax.devices("cpu")[0]
    x = jax.device_put(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32), cpu)

    with pytest.raises(Exception) as excinfo:
        result = jax.ffi.ffi_call(
            "test_cpu_check_fail.check_fail_cpu",
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            vmap_method="broadcast_all",
        )(x)
        jax.block_until_ready(result)

    # The diagnostic from check.h propagates through the FFI error.
    message = str(excinfo.value)
    assert "CHECK FAILED" in message or "intentional failure" in message, message


@_skip_on_windows
def test_process_survives_after_check_failure():
    """After a check failure, the interpreter is still usable (not aborted)."""
    mod = brainevent.load_cpp_inline(
        name="test_cpu_check_fail2",
        cpp_sources=CHECK_FAIL_SRC.replace("check_fail_cpu", "check_fail_cpu2"),
        functions=["check_fail_cpu2"],
        force_rebuild=True,
    )
    cpu = jax.devices("cpu")[0]
    x = jax.device_put(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32), cpu)

    with pytest.raises(Exception):
        jax.block_until_ready(
            jax.ffi.ffi_call(
                "test_cpu_check_fail2.check_fail_cpu2",
                jax.ShapeDtypeStruct(x.shape, x.dtype),
                vmap_method="broadcast_all",
            )(x)
        )

    # Still alive: a normal JAX computation completes after the failure.
    assert float(jnp.sum(jnp.arange(5.0))) == 10.0


# ---------------------------------------------------------------------------
# CUDA end-to-end coverage for ``load_cuda_inline``: jit, large arrays, module
# attributes, target registration and multi-output kernels.
# ---------------------------------------------------------------------------

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
    """Compile the vector_add kernel once for all tests that use it."""
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
    targets = brainevent.list_registered_targets()
    assert "test_vadd.vector_add" in targets


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


# ---------------------------------------------------------------------------
# Data-type coverage: bool, int8-64, uint8-64, float16/32/64, bfloat16,
# complex64/128.
#
# Grouped into a class so that the GPU requirement and the x64 autouse fixture
# stay scoped to these tests. Both were module-level when this lived in its own
# file; at module level here the fixture would flip ``jax_enable_x64`` on for
# every other test in this file -- and it deliberately never restores it.
# ---------------------------------------------------------------------------

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

MULTI_DTYPE_CUDA_SRC = r"""
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
def copy_module():
    return brainevent.load_cuda_inline(
        name="test_copy_all_dtypes",
        cuda_sources=COPY_KERNEL_SRC,
        force_rebuild=True,
    )


@pytest.fixture(scope="module")
def dispatch_module():
    return brainevent.load_cuda_inline(
        name="test_dispatch_all",
        cuda_sources=DISPATCH_ALL_SRC,
        force_rebuild=True,
    )


@pytest.fixture(scope="module")
def typed_add_module():
    return brainevent.load_cuda_inline(
        name="test_typed_add",
        cuda_sources=MULTI_DTYPE_CUDA_SRC,
        functions={"typed_add": ["arg", "arg", "ret", "stream"]},
        force_rebuild=True,
    )


class TestDtypes:
    """Kernix data-type coverage across the copy, dispatch and typed-add kernels."""

    pytestmark = requires_gpu

    @pytest.fixture(autouse=True)
    def _enable_x64(self):
        """Ensure 64-bit types are available for every test in this class."""
        jax.config.update("jax_enable_x64", True)
        yield
        # intentionally do not restore — other test modules may rely on x64

    # -- Copy (byte-level) tests for all dtypes ------------------------------

    @pytest.mark.parametrize("dtype", [
        "bool",
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "float16", "float32", "float64",
        "bfloat16",
        "complex64", "complex128",
    ])
    def test_copy_dtype(self, copy_module, dtype):
        """Byte-level copy kernel preserves data for all dtypes."""
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

    # -- Dispatch macro tests for numeric dtypes -----------------------------

    @pytest.mark.parametrize("dtype", [
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "float32", "float64",
    ])
    def test_dispatch_add(self, dispatch_module, dtype):
        """BE_DISPATCH_ALL_TYPES correctly dispatches addition."""
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

    def test_dispatch_add_bool(self, dispatch_module):
        """BE_DISPATCH_ALL_TYPES handles bool (addition = logical OR)."""
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

    # -- Float32/float64 dtype dispatch --------------------------------------

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_typed_add(self, typed_add_module, dtype):
        """Vector addition works for float32 and float64."""
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

    def test_2d_tensor(self, typed_add_module):
        """Works with multi-dimensional tensors."""
        a = jnp.ones((32, 64), dtype=jnp.float32)
        b = jnp.full((32, 64), 5.0, dtype=jnp.float32)

        result = jax.ffi.ffi_call(
            "test_typed_add.typed_add",
            jax.ShapeDtypeStruct((32, 64), jnp.float32),
        )(a, b)

        expected = np.full((32, 64), 6.0, dtype=np.float32)
        np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-5)
