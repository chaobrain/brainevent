# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Kernix C++/CPU FFI end-to-end tests."""

import platform

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
from brainevent._error import KernelRegistrationError

if platform.platform().startswith('Windows'):
    pytest.skip(reason="Windows is not supported yet.", allow_module_level=True)


@pytest.fixture
def isolated_cache(tmp_path):
    """Point the shared compilation cache at a scratch dir for one test.

    F3/F10 exercise *cache-hit* behaviour (no ``force_rebuild``), so they need a
    private cache dir that other tests cannot pre-populate.
    """
    old = brainevent.get_cache_dir()
    brainevent.set_cache_dir(str(tmp_path / "becache"))
    try:
        yield
    finally:
        brainevent.set_cache_dir(old)

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


# ---------------------------------------------------------------------------
# Host-side check propagation tests
# ---------------------------------------------------------------------------

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
# Cluster D end-to-end tests (C++/CPU path)
# ---------------------------------------------------------------------------

# Source defining TWO functions; used to exercise the finding-3 scenario where
# the same source is loaded first with one function, then with both.
TWO_FN_SRC = r"""
#include "brainevent/common.h"

void addf(const BE::Tensor x, BE::Tensor y) {
    int n = x.numel();
    const float* i = static_cast<const float*>(x.data_ptr());
    float* o = static_cast<float*>(y.data_ptr());
    for (int k = 0; k < n; ++k) o[k] = i[k] + 1.0f;
}

void addg(const BE::Tensor x, BE::Tensor y) {
    int n = x.numel();
    const float* i = static_cast<const float*>(x.data_ptr());
    float* o = static_cast<float*>(y.data_ptr());
    for (int k = 0; k < n; ++k) o[k] = i[k] + 100.0f;
}
"""


def test_f3_specs_participate_in_cache_key_e2e(isolated_cache):
    """F3: same source + a superset ``functions`` mapping must recompile.

    Before the fix the cache key hashed only the user source, so the second load
    (which needs a ``be_addg`` wrapper) would hit the first load's ``.so`` — which
    lacks that symbol — and fail with the misleading "Did the compilation
    succeed?" error.  With specs in the key the two loads get distinct entries.
    """
    cpu = jax.devices("cpu")[0]
    x = jax.device_put(jnp.arange(8, dtype=jnp.float32), cpu)

    # Load 1: only addf (seeds the cache under this name).
    m1 = brainevent.load_cpp_inline(
        name="bef3_mod", cpp_sources=TWO_FN_SRC,
        functions={"addf": ["arg", "ret"]}, auto_register=False,
    )
    assert m1.function_names == ["addf"]

    # Load 2: SAME source + name, now with BOTH functions.  Must not cache-hit
    # the single-function artefact.
    m2 = brainevent.load_cpp_inline(
        name="bef3_mod", cpp_sources=TWO_FN_SRC,
        functions={"addf": ["arg", "ret"], "addg": ["arg", "ret"]},
        target_prefix="bef3_run",
    )
    assert m1.path != m2.path, "specs must change the cache entry (different .so)"
    assert set(m2.function_names) == {"addf", "addg"}

    # Both wrappers behave correctly.
    rf = jax.ffi.ffi_call("bef3_run.addf", jax.ShapeDtypeStruct(x.shape, x.dtype),
                          vmap_method="broadcast_all")(x)
    rg = jax.ffi.ffi_call("bef3_run.addg", jax.ShapeDtypeStruct(x.shape, x.dtype),
                          vmap_method="broadcast_all")(x)
    np.testing.assert_allclose(np.asarray(rf), np.arange(8) + 1.0)
    np.testing.assert_allclose(np.asarray(rg), np.arange(8) + 100.0)


HDR_SRC = r"""
#include "brainevent/common.h"
#include "bef10_addend.h"

void add_hdr(const BE::Tensor x, BE::Tensor y) {
    int n = x.numel();
    const float* i = static_cast<const float*>(x.data_ptr());
    float* o = static_cast<float*>(y.data_ptr());
    for (int k = 0; k < n; ++k) o[k] = i[k] + (float)BEF10_ADDEND;
}
"""


def test_f10_extra_include_header_edit_rebuilds(isolated_cache, tmp_path):
    """F10: editing a header under ``extra_include_paths`` must invalidate cache."""
    inc = tmp_path / "inc"
    inc.mkdir()
    header = inc / "bef10_addend.h"
    header.write_text("#define BEF10_ADDEND 1\n")

    cpu = jax.devices("cpu")[0]
    x = jax.device_put(jnp.zeros(4, dtype=jnp.float32), cpu)

    m1 = brainevent.load_cpp_inline(
        name="bef10_mod", cpp_sources=HDR_SRC, functions=["add_hdr"],
        extra_include_paths=[str(inc)], target_prefix="bef10_a",
    )
    r1 = jax.ffi.ffi_call("bef10_a.add_hdr", jax.ShapeDtypeStruct(x.shape, x.dtype),
                          vmap_method="broadcast_all")(x)
    np.testing.assert_allclose(np.asarray(r1), np.full(4, 1.0))

    # Edit the header contents (no force_rebuild): cache must miss and recompile.
    header.write_text("#define BEF10_ADDEND 2\n")
    m2 = brainevent.load_cpp_inline(
        name="bef10_mod", cpp_sources=HDR_SRC, functions=["add_hdr"],
        extra_include_paths=[str(inc)], target_prefix="bef10_b",
    )
    assert m1.path != m2.path, "header edit must produce a new cache entry"
    r2 = jax.ffi.ffi_call("bef10_b.add_hdr", jax.ShapeDtypeStruct(x.shape, x.dtype),
                          vmap_method="broadcast_all")(x)
    np.testing.assert_allclose(np.asarray(r2), np.full(4, 2.0))


def _src_plus(delta: float) -> str:
    return (
        '#include "brainevent/common.h"\n'
        "void bump(const BE::Tensor x, BE::Tensor y) {\n"
        "  int n = x.numel();\n"
        "  const float* i = static_cast<const float*>(x.data_ptr());\n"
        "  float* o = static_cast<float*>(y.data_ptr());\n"
        f"  for (int k = 0; k < n; ++k) o[k] = i[k] + {delta}f;\n"
        "}\n"
    )


def test_f12_stem_collision_without_replace_errors(isolated_cache):
    """F12: a second, different artefact under the same target name → clear error.

    Models two sources that share a file stem (both target ``bef12.bump``).  The
    error must name BOTH remedies: replace=True and a distinct name/target_prefix.
    """
    brainevent.load_cpp_inline(
        name="bef12", cpp_sources=_src_plus(1.0), functions=["bump"],
    )
    with pytest.raises(KernelRegistrationError) as ei:
        brainevent.load_cpp_inline(
            name="bef12", cpp_sources=_src_plus(2.0), functions=["bump"],
        )
    msg = str(ei.value)
    assert "replace=True" in msg
    assert "target_prefix" in msg or "name=" in msg


def test_f12_replace_refused_deterministically(isolated_cache):
    """F12/F4: replace=True with changed content raises a clean, actionable error.

    A live re-point cannot be verified on this JAX (probed: the Host registry
    rejects a differing bundle address; the CUDA registry silently keeps the old
    handler), so ``replace=True`` refuses deterministically and directs the user
    to a distinct name — it never silently keeps serving the stale kernel.
    """
    brainevent.load_cpp_inline(
        name="bef12r", cpp_sources=_src_plus(1.0), functions=["bump"],
    )
    with pytest.raises(KernelRegistrationError) as ei:
        brainevent.load_cpp_inline(
            name="bef12r", cpp_sources=_src_plus(2.0), functions=["bump"],
            replace=True,
        )
    assert "distinct name" in str(ei.value) or "target_prefix" in str(ei.value)


def test_f4_force_rebuild_new_code_via_distinct_name(isolated_cache):
    """F4 e2e: force_rebuild refuses to silently serve stale code; a distinct
    target name reliably dispatches the edited kernel.

    On jax 0.10.2 the CPU registry cannot re-point a live target, so the honest
    outcome is: (1) force_rebuild with changed content under the same name raises
    (no silent stale dispatch — finding 4 fixed), and (2) the edited kernel runs
    when registered under a new target name.
    """
    cpu = jax.devices("cpu")[0]
    x = jax.device_put(jnp.zeros(4, dtype=jnp.float32), cpu)

    brainevent.load_cpp_inline(
        name="bef4", cpp_sources=_src_plus(1.0), functions=["bump"],
        target_prefix="bef4_v1",
    )
    r1 = jax.ffi.ffi_call("bef4_v1.bump", jax.ShapeDtypeStruct(x.shape, x.dtype),
                          vmap_method="broadcast_all")(x)
    np.testing.assert_allclose(np.asarray(r1), np.full(4, 1.0))

    # Edit + force_rebuild under the SAME target prefix → refuses (no silent stale).
    with pytest.raises(KernelRegistrationError):
        brainevent.load_cpp_inline(
            name="bef4", cpp_sources=_src_plus(2.0), functions=["bump"],
            force_rebuild=True, target_prefix="bef4_v1",
        )

    # Distinct target prefix → the edited kernel dispatches correctly.
    brainevent.load_cpp_inline(
        name="bef4", cpp_sources=_src_plus(2.0), functions=["bump"],
        target_prefix="bef4_v2",
    )
    r2 = jax.ffi.ffi_call("bef4_v2.bump", jax.ShapeDtypeStruct(x.shape, x.dtype),
                          vmap_method="broadcast_all")(x)
    np.testing.assert_allclose(np.asarray(r2), np.full(4, 2.0))
