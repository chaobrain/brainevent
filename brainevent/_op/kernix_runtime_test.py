# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Tests for CompiledModule load-time error wrapping."""

import threading

import pytest

from brainevent._op import kernix_runtime as kr
from brainevent._error import KernelLoadError, KernelRegistrationError


def test_dlopen_failure_wrapped(monkeypatch, tmp_path):
    """A load failure of an *existing* .so yields the cu12/LD_LIBRARY_PATH hint.

    The file must exist so the missing-artefact branch (finding 19) does not
    fire — here the artefact is present but a dependent runtime is missing.
    """
    so = tmp_path / "present.so"
    so.write_bytes(b"\x7fELF")  # exists, but CDLL is patched to fail below

    def boom(path):
        raise OSError("libcudart.so.12: cannot open shared object file: No such file or directory")
    monkeypatch.setattr(kr.ctypes, "CDLL", boom)
    with pytest.raises(KernelLoadError) as ei:
        kr.CompiledModule(str(so), ["f"])
    msg = str(ei.value)
    assert "E-LOAD" in msg
    assert str(so) in msg
    assert "LD_LIBRARY_PATH" in msg


def test_missing_so_reports_disappeared_not_cu12(monkeypatch):
    """F19: a vanished cache artefact reports 'missing', not the cu12 hint.

    If the .so disappears between lookup() and CDLL (e.g. clear_cache() from
    another process), dlopen's 'cannot open shared object file' otherwise
    matches the cu12 'missing CUDA runtime' branch and misdirects the user.
    """
    def boom(path):
        raise OSError("cannot open shared object file: No such file or directory")
    monkeypatch.setattr(kr.ctypes, "CDLL", boom)
    with pytest.raises(KernelLoadError) as ei:
        kr.CompiledModule("/tmp/nonexistent-vanished-abc123.so", ["f"])
    msg = str(ei.value)
    assert "E-LOAD-MISSING" in msg
    assert "cleared concurrently" in msg
    # Must NOT misattribute to a missing CUDA runtime.
    assert "LD_LIBRARY_PATH" not in msg
    assert "CUDA runtime" not in msg


# ---------------------------------------------------------------------------
# Audit regression tests
# ---------------------------------------------------------------------------


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


class _FakeContentModule(_FakeModule):
    """A fake module whose *content* identity is explicit (finding 4).

    Two instances may share a ``path`` but differ in ``content_hash`` — modelling
    an edited-and-rebuilt kernel that republishes to the same cache path with
    different bytes.
    """

    def __init__(self, so_path: str, functions, content_hash: str):
        super().__init__(so_path, functions)
        self._content_hash = str(content_hash)

    @property
    def content_hash(self) -> str:
        return self._content_hash


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
    monkeypatch.setattr(kr, "_REGISTRATION_KEYS", {}, raising=False)
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
# F4 -- content-hash registration identity + replace semantics
# ---------------------------------------------------------------------------

def test_f4_content_hash_distinguishes_same_path(clean_registry):
    """Same path + different content_hash → treated as a *different* module.

    This is the core of finding 4: force_rebuild republishes to the same cache
    path, so identity must key on the ``.so`` bytes, not the path.
    """
    calls = clean_registry
    a = _FakeContentModule("/tmp/lib.so", ["k"], content_hash="sha256:aaaa")
    b = _FakeContentModule("/tmp/lib.so", ["k"], content_hash="sha256:bbbb")

    kr.register_ffi_target("t.k", a, "k", platform="cpu")
    # Different content under the same name, no replace → refuse (not a no-op).
    with pytest.raises(KernelRegistrationError):
        kr.register_ffi_target("t.k", b, "k", platform="cpu")
    assert calls == ["t.k"]


def test_f4_refuse_message_names_both_remedies(clean_registry):
    """The refusal must name BOTH remedies: replace=True and a distinct name."""
    a = _FakeContentModule("/tmp/lib.so", ["k"], content_hash="sha256:aaaa")
    b = _FakeContentModule("/tmp/lib.so", ["k"], content_hash="sha256:bbbb")
    kr.register_ffi_target("t.k", a, "k", platform="cpu")
    with pytest.raises(KernelRegistrationError) as ei:
        kr.register_ffi_target("t.k", b, "k", platform="cpu")
    msg = str(ei.value)
    assert "replace=True" in msg
    assert "target_prefix" in msg or "name=" in msg


def test_f4_replace_refused_deterministically(clean_registry):
    """replace=True with changed content is REFUSED on every platform.

    A live re-point cannot be verified on this JAX: the CPU/Host registry raises
    on a differing bundle address, and the CUDA registry accepts a duplicate but
    silently keeps the old handler.  Blindly re-registering would therefore
    report success on CUDA while the stale kernel keeps executing (audit finding
    4).  So brainevent refuses deterministically *without* calling jax, and the
    behaviour does not depend on the ``platform`` argument.
    """
    for platform in ("cpu", "CUDA"):
        calls = clean_registry  # fresh isolation is per-test; reuse container
        target = f"det.{platform}"
        a = _FakeContentModule("/tmp/lib.so", ["k"], content_hash="sha256:aaaa")
        b = _FakeContentModule("/tmp/lib.so", ["k"], content_hash="sha256:bbbb")

        kr.register_ffi_target(target, a, "k", platform=platform)
        n_after_first = len(calls)
        with pytest.raises(KernelRegistrationError) as ei:
            kr.register_ffi_target(target, b, "k", platform=platform, replace=True)
        msg = str(ei.value)
        assert "distinct name" in msg or "target_prefix" in msg
        # jax.ffi.register_ffi_target must NOT be attempted a second time.
        assert len(calls) == n_after_first
        # The original keep-alive and identity are untouched.
        assert kr._LIVE_MODULES[target] == [a]
        assert kr._REGISTRATION_KEYS[target] == kr._registration_key(a, "k", platform)


def test_f4_replace_same_content_is_noop(clean_registry):
    """replace=True with identical content must not re-register."""
    calls = clean_registry
    a = _FakeContentModule("/tmp/lib.so", ["k"], content_hash="sha256:aaaa")
    kr.register_ffi_target("t.k", a, "k", platform="cpu")
    kr.register_ffi_target("t.k", a, "k", platform="cpu", replace=True)
    assert calls == ["t.k"]


def test_f4_real_module_content_hash_reads_bytes(tmp_path):
    """CompiledModule.content_hash is a stable hash of the .so bytes."""
    so = tmp_path / "m.so"
    so.write_bytes(b"\x7fELF-some-bytes")
    # Build a CompiledModule without dlopen by bypassing __init__ machinery:
    mod = object.__new__(kr.CompiledModule)
    mod._so_path = str(so)
    mod._content_hash = None
    h1 = mod.content_hash
    assert h1.startswith("sha256:")
    assert mod.content_hash == h1  # memoised
    # A different-bytes file yields a different hash.
    so2 = tmp_path / "m2.so"
    so2.write_bytes(b"\x7fELF-other-bytes")
    mod2 = object.__new__(kr.CompiledModule)
    mod2._so_path = str(so2)
    mod2._content_hash = None
    assert mod2.content_hash != h1


# ---------------------------------------------------------------------------
# MEDIUM -- content_id overrides the .so byte hash for registration identity
# ---------------------------------------------------------------------------

def test_content_id_reregistration_with_different_bytes_is_noop(clean_registry):
    """(a) Same ``content_id``, different ``.so`` bytes -> idempotent no-op.

    Compilers are non-deterministic (they embed build paths/timestamps), so a
    ``force_rebuild`` of UNCHANGED source can produce different ``.so`` bytes
    even though nothing about the kernel actually changed. Keying the
    equivalence check on the caller-supplied deterministic cache key
    (``content_id``) instead of the ``.so`` byte hash makes that rebuild an
    idempotent no-op rather than a spurious ``KernelRegistrationError``.
    """
    calls = clean_registry
    m1 = _FakeContentModule("/tmp/lib.so", ["k"], content_hash="sha256:aaaa")
    m2 = _FakeContentModule("/tmp/lib.so", ["k"], content_hash="sha256:bbbb")  # different bytes

    kr.register_ffi_target("cid.k", m1, "k", platform="cpu", content_id="key:abc")
    # Different .so bytes, but the SAME content_id -> must be a no-op, not raise.
    kr.register_ffi_target("cid.k", m2, "k", platform="cpu", content_id="key:abc")

    assert calls == ["cid.k"], "equivalent content_id re-registration must not re-call FFI"
    assert kr._LIVE_MODULES["cid.k"] == [m1], "the second (equivalent) module must not be appended"


def test_content_id_change_raises(clean_registry):
    """(b) A genuinely different ``content_id`` under the same target name raises."""
    calls = clean_registry
    m1 = _FakeContentModule("/tmp/lib.so", ["k"], content_hash="sha256:aaaa")
    m2 = _FakeContentModule("/tmp/lib.so", ["k"], content_hash="sha256:bbbb")

    kr.register_ffi_target("cid.change", m1, "k", platform="cpu", content_id="key:abc")
    with pytest.raises(KernelRegistrationError):
        kr.register_ffi_target("cid.change", m2, "k", platform="cpu", content_id="key:def")
    assert calls == ["cid.change"]


def test_no_content_id_still_uses_byte_hash(clean_registry):
    """(c) Omitting ``content_id`` preserves the pre-existing byte-hash-keyed
    behavior: same path, different ``content_hash``, no ``content_id`` ->
    raises (not a no-op). Guards against the ``content_id`` addition changing
    default behavior for callers that do not pass it.
    """
    calls = clean_registry
    m1 = _FakeContentModule("/tmp/lib.so", ["k"], content_hash="sha256:aaaa")
    m2 = _FakeContentModule("/tmp/lib.so", ["k"], content_hash="sha256:bbbb")

    kr.register_ffi_target("cid.default", m1, "k", platform="cpu")
    with pytest.raises(KernelRegistrationError):
        kr.register_ffi_target("cid.default", m2, "k", platform="cpu")
    assert calls == ["cid.default"]


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
