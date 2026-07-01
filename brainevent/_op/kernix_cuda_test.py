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
from brainevent._op import kernix_toolchain as kt

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


# ---------------------------------------------------------------------------
# Compute-capability resolution tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("8.6", "sm_86"), ("86", "sm_86"), ("sm_86", "sm_86"),
    ("compute_86", "sm_86"), (" 8.6 ", "sm_86"), ("9.0a", "sm_90a"),
    ("90a", "sm_90a"), ("12.0", "sm_120"), ("120", "sm_120"),
])
def test_normalize_arch_ok(raw, expected):
    assert kt.normalize_arch(raw) == expected


@pytest.mark.parametrize("bad", ["", "   ", "abc", "x", "8", "sm_", ".."])
def test_normalize_arch_bad(bad):
    with pytest.raises(ValueError):
        kt.normalize_arch(bad)


def test_resolve_explicit_wins(monkeypatch):
    monkeypatch.setattr(kt, "_arch_from_jax", lambda: ["sm_99"])
    assert kt.resolve_compute_capabilities("8.6") == ["sm_86"]
    assert kt.resolve_compute_capabilities(["8.6", "9.0"]) == ["sm_86", "sm_90"]


def test_resolve_explicit_comma_string(monkeypatch):
    monkeypatch.setattr(kt, "_arch_from_jax", lambda: ["sm_99"])
    assert kt.resolve_compute_capabilities("8.0,8.6") == ["sm_80", "sm_86"]
    assert kt.resolve_compute_capabilities(" 8.0 , , 8.6 ") == ["sm_80", "sm_86"]
    assert kt.resolve_compute_capabilities(["8.0,8.6", "9.0"]) == [
        "sm_80", "sm_86", "sm_90"]


def test_resolve_precedence_config_over_env(monkeypatch):
    monkeypatch.setenv("BRAINEVENT_COMPUTE_CAPABILITIES", "8.0")
    monkeypatch.setattr(kt, "_arch_from_jax", lambda: ["sm_99"])
    kt.set_compute_capabilities("8.6")
    try:
        assert kt.resolve_compute_capabilities() == ["sm_86"]
    finally:
        kt.set_compute_capabilities(None)


def test_resolve_env_over_jax(monkeypatch):
    monkeypatch.setenv("BRAINEVENT_COMPUTE_CAPABILITIES", " 8.0 , , 8.6 ")
    monkeypatch.setattr(kt, "_arch_from_jax", lambda: ["sm_99"])
    assert kt.resolve_compute_capabilities() == ["sm_80", "sm_86"]


def test_resolve_jax_over_smi(monkeypatch):
    monkeypatch.delenv("BRAINEVENT_COMPUTE_CAPABILITIES", raising=False)
    monkeypatch.setattr(kt, "_arch_from_jax", lambda: ["sm_86"])
    monkeypatch.setattr(kt, "_arch_from_nvidia_smi", lambda: ["sm_70"])
    assert kt.resolve_compute_capabilities() == ["sm_86"]


def test_resolve_raises_when_all_absent(monkeypatch):
    from brainevent._error import GpuArchDetectionError
    monkeypatch.delenv("BRAINEVENT_COMPUTE_CAPABILITIES", raising=False)
    monkeypatch.setattr(kt, "_arch_from_jax", lambda: None)
    monkeypatch.setattr(kt, "_arch_from_nvidia_smi", lambda: None)
    with pytest.raises(GpuArchDetectionError):
        kt.resolve_compute_capabilities()


def test_gencode_single():
    assert kt.gencode_flags(["sm_86"]) == [
        "-gencode", "arch=compute_86,code=sm_86",
        "-gencode", "arch=compute_86,code=compute_86",
    ]


def test_gencode_multi_ptx_for_highest():
    out = kt.gencode_flags(["sm_80", "sm_90", "8.6"])
    assert "arch=compute_80,code=sm_80" in out
    assert "arch=compute_86,code=sm_86" in out
    assert "arch=compute_90,code=sm_90" in out
    assert out[-1] == "arch=compute_90,code=compute_90"


def test_gencode_empty_raises():
    with pytest.raises(ValueError):
        kt.gencode_flags([])


def test_cache_uses_platform_ext(tmp_path, monkeypatch):
    from brainevent._op import kernix_cache as kcache
    from brainevent._op import kernix_toolchain as ktool
    monkeypatch.setattr(ktool.sys, "platform", "win32")
    c = kcache.CompilationCache(base_dir=str(tmp_path))
    assert c.lookup("m", "deadbeef") is None
    src = tmp_path / "src.dll"
    src.write_bytes(b"x")
    dest = c.store("m", "deadbeef", str(src))
    assert dest.name == "m.dll"
    assert c.lookup("m", "deadbeef") == dest


def test_cache_store_atomic_move(tmp_path):
    from brainevent._op import kernix_cache as kcache
    c = kcache.CompilationCache(base_dir=str(tmp_path))
    build = tmp_path / "build"
    build.mkdir()
    so = build / "m.so"
    so.write_bytes(b"hello")
    dest = c.store("m", "k", str(so))
    assert dest.read_bytes() == b"hello"
    assert not so.exists()


def test_config_set_compute_capability():
    import brainevent
    from brainevent._op import kernix_toolchain as ktool
    brainevent.config.set_compute_capability("8.6")
    try:
        assert brainevent.config.get_compute_capability() == ["sm_86"]
        assert ktool.resolve_compute_capabilities() == ["sm_86"]
    finally:
        brainevent.config.set_compute_capability(None)
    assert brainevent.config.get_compute_capability() is None


def test_set_compute_capabilities_comma_string():
    kt.set_compute_capabilities("8.6,8.0")
    try:
        assert kt.get_compute_capabilities() == ["sm_86", "sm_80"]
    finally:
        kt.set_compute_capabilities(None)
    assert kt.get_compute_capabilities() is None


def test_config_set_compute_capability_comma():
    import brainevent
    brainevent.config.set_compute_capability("8.6, 8.0")
    try:
        assert brainevent.config.get_compute_capability() == ["sm_86", "sm_80"]
    finally:
        brainevent.config.set_compute_capability(None)
