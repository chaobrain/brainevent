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

"""Audit reproduction tests for ``kernix_runtime``.

These tests are hermetic: they never load a real native module.  The JAX FFI
surface (``jax.ffi.register_ffi_target`` and ``jax.ffi.pycapsule``) is
monkeypatched so the registration bridge can be exercised without a compiled
``.so``.

Covered findings (see ``dev/2026-06-13-op-issues.md``):

* **M5** -- FFI-target registration race + silent overwrite; no unload.
* **L6** -- ``_format_load_error`` only recognises POSIX dlopen wording.
"""

import threading

import pytest

from brainevent._error import KernelRegistrationError
from brainevent._op import kernix_runtime as kr


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class _FakeModule:
    """Minimal stand-in for :class:`CompiledModule`.

    Only the surface that :func:`register_ffi_target` touches is implemented:
    a :attr:`path` (the ``so_path`` used for equivalence) and
    :meth:`get_handler` (returning an opaque sentinel that the patched
    ``pycapsule`` accepts).
    """

    def __init__(self, so_path: str, functions):
        self._so_path = str(so_path)
        self._functions = {name: object() for name in functions}

    @property
    def path(self) -> str:
        return self._so_path

    @property
    def function_names(self):
        return list(self._functions)

    def get_handler(self, name: str):
        return self._functions[name]


@pytest.fixture
def clean_registry(monkeypatch):
    """Isolate the module-global registries and count real FFI registrations.

    Yields a ``calls`` list; each successful ``jax.ffi.register_ffi_target``
    invocation appends its ``target_name``.  The ``_LIVE_MODULES`` /
    ``_REGISTERED_TARGETS`` containers are swapped for fresh ones so the test
    does not see (or pollute) global state.
    """
    calls = []

    def fake_register(target_name, capsule, platform="CUDA"):
        calls.append(target_name)

    def fake_pycapsule(fn_ptr):
        return ("capsule", fn_ptr)

    monkeypatch.setattr(kr.jax.ffi, "register_ffi_target", fake_register)
    monkeypatch.setattr(kr.jax.ffi, "pycapsule", fake_pycapsule)
    monkeypatch.setattr(kr, "_LIVE_MODULES", {}, raising=False)
    monkeypatch.setattr(kr, "_REGISTERED_TARGETS", set(), raising=False)
    return calls


# ---------------------------------------------------------------------------
# M5 -- registration race, idempotency, and silent-overwrite protection
# ---------------------------------------------------------------------------

def test_m5_registration_lock_exists():
    """A module-level ``threading.Lock`` must guard the registry."""
    assert hasattr(kr, "_REGISTRATION_LOCK"), (
        "kernix_runtime must expose a module-level registration lock"
    )
    assert isinstance(kr._REGISTRATION_LOCK, type(threading.Lock()))


def test_m5_same_module_reregistration_is_idempotent(clean_registry):
    """(a) Re-registering the *same* module under a name is a no-op.

    The second call must neither invoke ``jax.ffi.register_ffi_target`` again
    nor overwrite the live keep-alive entry.
    """
    calls = clean_registry
    mod = _FakeModule("/tmp/libfake.so", ["noop"])

    kr.register_ffi_target("dup.noop", mod, "noop", platform="cpu")
    live_after_first = kr._LIVE_MODULES["dup.noop"]

    # Same module, same name, same platform -> equivalent -> idempotent.
    kr.register_ffi_target("dup.noop", mod, "noop", platform="cpu")

    assert calls == ["dup.noop"], "equivalent re-registration must not re-call FFI"
    assert kr._LIVE_MODULES["dup.noop"] is live_after_first, (
        "_LIVE_MODULES must not be overwritten on idempotent re-registration"
    )


def test_m5_concurrent_registration_single_call(clean_registry):
    """(b) Concurrent registration of one name yields exactly one registration."""
    calls = clean_registry
    mod = _FakeModule("/tmp/libfake.so", ["noop"])

    n_threads = 32
    start = threading.Barrier(n_threads)
    errors = []

    def worker():
        start.wait()
        try:
            kr.register_ffi_target("race.noop", mod, "noop", platform="cpu")
        except Exception as exc:  # pragma: no cover - surfaced via assert below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent equivalent registration raised: {errors}"
    assert calls == ["race.noop"], (
        f"expected exactly one FFI registration, got {len(calls)}: {calls}"
    )
    assert kr._REGISTERED_TARGETS == {"race.noop"}


def test_m5_different_module_same_name_raises(clean_registry):
    """(c) A *different* module under an existing name must not silently clobber."""
    calls = clean_registry
    mod_a = _FakeModule("/tmp/liba.so", ["noop"])
    mod_b = _FakeModule("/tmp/libb.so", ["noop"])  # different so_path

    kr.register_ffi_target("conf.noop", mod_a, "noop", platform="cpu")
    live_after_first = kr._LIVE_MODULES["conf.noop"]

    with pytest.raises(KernelRegistrationError):
        kr.register_ffi_target("conf.noop", mod_b, "noop", platform="cpu")

    # The original keep-alive survives; the conflicting one was rejected.
    assert kr._LIVE_MODULES["conf.noop"] is live_after_first
    assert calls == ["conf.noop"], "conflicting registration must not re-call FFI"


def test_m5_different_platform_same_name_raises(clean_registry):
    """(c) Same module + name but a *different platform* is not equivalent."""
    mod = _FakeModule("/tmp/libfake.so", ["noop"])
    kr.register_ffi_target("plat.noop", mod, "noop", platform="cpu")
    with pytest.raises(KernelRegistrationError):
        kr.register_ffi_target("plat.noop", mod, "noop", platform="CUDA")


def test_m5_different_func_same_name_raises(clean_registry):
    """(c) Same module + name but a *different function* is not equivalent."""
    mod = _FakeModule("/tmp/libfake.so", ["noop", "other"])
    kr.register_ffi_target("fn.noop", mod, "noop", platform="cpu")
    with pytest.raises(KernelRegistrationError):
        kr.register_ffi_target("fn.noop", mod, "other", platform="cpu")


# ---------------------------------------------------------------------------
# L6 -- Windows loader wording must produce the helpful hint
# ---------------------------------------------------------------------------

# (error string, set of Windows-specific tokens at least one of which the hint
#  must contain).  These tokens are *absent* from the POSIX generic fallback
#  line, so the assertions genuinely require new Windows heuristics rather than
#  passing on the catch-all bullet.
WINDOWS_DEP_ERRORS = [
    # FormatMessage text for ERROR_MOD_NOT_FOUND (126).
    "[WinError 126] The specified module could not be found",
    # FormatMessage text for ERROR_PROC_NOT_FOUND (127).
    "[WinError 127] The specified procedure could not be found",
    # Bare numeric codes that ctypes/loaders sometimes surface.
    "error 126",
    "Error 127 while loading dependent DLLs",
]

# Tokens that only the Windows-aware branch emits (none appear in the POSIX
# generic fallback line).
_WIN_DEP_TOKENS = ("dll", "path")
_WIN_ARCH_TOKENS = ("32-bit", "64-bit", "bitness", "win32 application")

# The POSIX generic catch-all bullet; the Windows branch must NOT be this.
_GENERIC = "Verify the build succeeded and dependent libraries are available"


@pytest.mark.parametrize("err_text", WINDOWS_DEP_ERRORS)
def test_l6_windows_missing_dll_gives_hint(err_text):
    """Windows missing-dependency wording yields a DLL/PATH-aware hint."""
    out = kr._format_load_error("C:\\build\\kernel.dll", OSError(err_text))

    assert "E-LOAD" in out
    assert "C:\\build\\kernel.dll" in out
    assert "How to fix:" in out
    low = out.lower()
    assert any(tok in low for tok in _WIN_DEP_TOKENS), (
        f"no Windows DLL/PATH hint produced for {err_text!r}:\n{out}"
    )
    # Must add value beyond the bare POSIX generic line.
    assert _GENERIC not in out, (
        f"Windows error {err_text!r} fell through to the generic bullet:\n{out}"
    )


@pytest.mark.parametrize(
    "err_text",
    [
        "[WinError 193] %1 is not a valid Win32 application",
        "is not a valid Win32 application",
    ],
)
def test_l6_windows_arch_mismatch_gives_hint(err_text):
    """A bitness/arch mismatch (error 193) must mention 32/64-bit."""
    out = kr._format_load_error("C:\\build\\kernel.dll", OSError(err_text))
    low = out.lower()
    assert any(tok in low for tok in _WIN_ARCH_TOKENS), (
        f"no bitness hint produced for {err_text!r}:\n{out}"
    )
    assert _GENERIC not in out


def test_l6_posix_paths_still_recognised():
    """Existing POSIX heuristics must keep working after broadening."""
    # cudart / cannot-open-shared-object branch.
    out = kr._format_load_error(
        "/tmp/k.so",
        OSError("libcudart.so.12: cannot open shared object file: No such file or directory"),
    )
    assert "LD_LIBRARY_PATH" in out
    # driver / forward-compatibility branch.
    out2 = kr._format_load_error(
        "/tmp/k.so",
        OSError("forward compatibility was attempted on non supported HW"),
    )
    assert "driver" in out2.lower()
