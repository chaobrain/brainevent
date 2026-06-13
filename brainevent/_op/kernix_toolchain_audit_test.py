# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Audit-driven reproduction tests for ``kernix_toolchain``.

Each test pins a defect from the 2026-06-13 ``dev/`` audit (H5, H6, M8, M9,
M10, L12).  Every test is hermetic: ``subprocess.run`` and environment
variables are monkeypatched so no real ``nvcc``/host compiler/GPU is needed.

Run with ``pytest -m ""`` (the markerless invocation the working agreement
prescribes).
"""

import sys
import threading

import pytest

from brainevent._op import kernix_toolchain as kt
from brainevent._op.kernix_toolchain import CandidateProbe

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="kernix toolchain tests are not supported on Windows",
)


def _touch_exec(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("")
    p.chmod(0o755)


def _fake_proc(*, returncode=0, stdout="", stderr=""):
    """Build a stand-in for ``subprocess.CompletedProcess``."""
    return type(
        "R", (), {"returncode": returncode, "stdout": stdout, "stderr": stderr}
    )()


# --- H5: nvcc --version exit code never checked ---------------------------

def test_h5_nvcc_version_nonzero_exit_reports_status(monkeypatch, tmp_path):
    """nvcc that runs but exits non-zero must surface the status + output.

    Reproduces H5: ``proc.returncode`` was ignored, so a loader/driver
    mismatch (non-zero exit, empty stdout) was misreported as
    "version could not be determined".
    """
    from brainevent._error import NvccNotFoundError

    nvcc = tmp_path / "cu13" / "bin" / "nvcc"
    _touch_exec(nvcc)
    monkeypatch.setattr(
        kt, "_select_nvcc",
        lambda: (str(nvcc), [str(tmp_path / "cu13" / "include")], []),
    )
    # nvcc launches, but exits 1 with a loader error on stderr and no stdout.
    monkeypatch.setattr(
        kt.subprocess, "run",
        lambda *a, **k: _fake_proc(
            returncode=1, stdout="",
            stderr="nvcc: error while loading shared libraries: libcuda.so.1",
        ),
    )
    with pytest.raises(NvccNotFoundError) as ei:
        kt.detect_cuda_toolchain()
    msg = str(ei.value)
    # The message must point at the non-zero exit, not at "version parsing".
    assert "status 1" in msg
    assert "libcuda.so.1" in msg
    assert "could not be determined" not in msg


# --- H6: CUDA_PATH never consulted ----------------------------------------

def test_h6_select_nvcc_uses_cuda_path(monkeypatch, tmp_path):
    """``CUDA_PATH`` (the Windows installer var) must be probed like CUDA_HOME.

    Reproduces H6: only ``CUDA_HOME`` was read, so a standard Windows CUDA
    install (which sets ``CUDA_PATH``) was never found when nvcc is off PATH.
    """
    nvcc = tmp_path / "cudapath" / "bin" / "nvcc"
    _touch_exec(nvcc)
    monkeypatch.delenv("BRAINEVENT_NVCC_PATH", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setenv("CUDA_PATH", str(tmp_path / "cudapath"))
    # Nothing on PATH and no pip wheel, so CUDA_PATH is the only hit.
    monkeypatch.setattr(kt, "_find_pip_cuda", lambda roots=None: (None, []))
    monkeypatch.setattr(kt.shutil, "which", lambda n: None)
    path, includes, probes = kt._select_nvcc()
    assert path == str(nvcc)
    assert includes == [str(tmp_path / "cudapath" / "include")]
    # A probe must record that CUDA_PATH was the source.
    assert any("CUDA_PATH" in p.source for p in probes)


# --- M8: _cxx_version swallows all exceptions; stderr banner invisible -----

def test_m8_cxx_version_reads_stderr_banner(monkeypatch):
    """MSVC-style compilers print their banner to stderr; it must be parsed.

    Reproduces the stderr half of M8: only ``stdout`` was read, so every
    ``cl.exe`` yielded an empty version (cache-key collision).
    """
    monkeypatch.setattr(
        kt.subprocess, "run",
        lambda *a, **k: _fake_proc(
            returncode=0, stdout="",
            stderr="Microsoft (R) C/C++ Optimizing Compiler Version 19.39\n",
        ),
    )
    assert "Microsoft" in kt._cxx_version("cl.exe")


def test_m8_cxx_version_filenotfound_degrades(monkeypatch):
    """A missing compiler still degrades to ``""`` (no crash)."""
    def boom(*a, **k):
        raise FileNotFoundError("no such compiler")

    monkeypatch.setattr(kt.subprocess, "run", boom)
    assert kt._cxx_version("/nope/g++") == ""


def test_m8_cxx_version_does_not_swallow_keyboardinterrupt(monkeypatch):
    """A ``KeyboardInterrupt`` must propagate, not be flattened to ``""``.

    Reproduces the bare-``except`` half of M8: ``except Exception`` already
    spares ``KeyboardInterrupt``, but the original code's intent (narrow the
    except) is what we assert here.
    """
    def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(kt.subprocess, "run", boom)
    with pytest.raises(KeyboardInterrupt):
        kt._cxx_version("/usr/bin/g++")


# --- M9: normalize_arch accepts nonsensical caps --------------------------

@pytest.mark.parametrize("bad", ["1.0", "0.0", "sm_00", "sm_10", "10", "00"])
def test_m9_normalize_arch_rejects_major_lt_2(bad):
    """No real GPU has compute-capability major < 2; reject it loudly.

    Reproduces M9: ``"0.0"`` and ``"1.0"`` normalized to ``sm_00`` / ``sm_10``
    and only failed much later inside nvcc.
    """
    from brainevent._error import UnsupportedArchError
    with pytest.raises((UnsupportedArchError, ValueError)):
        kt.normalize_arch(bad)


@pytest.mark.parametrize("good,expected", [
    ("7.5", "sm_75"), ("8.6", "sm_86"), ("2.0", "sm_20"), ("9.0a", "sm_90a"),
])
def test_m9_normalize_arch_still_accepts_valid(good, expected):
    assert kt.normalize_arch(good) == expected


# --- M10: GPU arch detection swallows real driver errors ------------------

def test_m10_jax_driver_error_surfaced_in_message(monkeypatch):
    """A broken CUDA backend must surface its cause, not look like "no GPU".

    Reproduces M10: ``jax.devices("gpu")`` raising on a driver mismatch was
    flattened to the same ``None`` as "no GPU present", hiding the traceback.
    """
    from brainevent._error import GpuArchDetectionError

    sentinel = "CUDA driver version is insufficient for CUDA runtime version"

    class FakeJax:
        @staticmethod
        def devices(kind):
            raise RuntimeError(sentinel)

    monkeypatch.setattr(kt, "jax", FakeJax)
    monkeypatch.setattr(kt, "_arch_from_nvidia_smi", lambda: None)
    monkeypatch.delenv("BRAINEVENT_COMPUTE_CAPABILITIES", raising=False)
    monkeypatch.setattr(kt, "_COMPUTE_CAPABILITIES", None)

    # _arch_from_jax must not blow up, but it must record the cause.
    assert kt._arch_from_jax() is None
    with pytest.raises(GpuArchDetectionError) as ei:
        kt.resolve_compute_capabilities()
    assert sentinel in str(ei.value)


# --- L12: include validation + global-state lock --------------------------

def test_l12_include_validation_rejects_bogus_dir(monkeypatch, tmp_path):
    """A resolved include dir lacking ``cuda_runtime.h`` is rejected clearly.

    Reproduces the ``parent.parent`` half of L12: a distro ``/usr/bin/nvcc``
    shim yields an include dir with no CUDA headers.
    """
    from brainevent._error import HeaderNotFoundError

    # nvcc at <prefix>/bin/nvcc → include guessed at <prefix>/include, which
    # exists but contains no cuda_runtime.h (the distro-shim failure mode).
    prefix = tmp_path / "usr"
    nvcc = prefix / "bin" / "nvcc"
    _touch_exec(nvcc)
    (prefix / "include").mkdir(parents=True)

    with pytest.raises(HeaderNotFoundError) as ei:
        kt._validate_cuda_include([str(prefix / "include")], nvcc_probes=[])
    assert "cuda_runtime.h" in str(ei.value)


def test_l12_include_validation_accepts_real_dir(tmp_path):
    """A dir that does contain ``cuda_runtime.h`` validates without error."""
    inc = tmp_path / "include"
    inc.mkdir()
    (inc / "cuda_runtime.h").write_text("/* header */")
    # Must not raise.
    kt._validate_cuda_include([str(inc)], nvcc_probes=[])


def test_l12_global_state_lock_exists_and_used(monkeypatch):
    """A ``threading.Lock`` must guard the mutable module caches.

    Reproduces the unsynchronized-globals half of L12.
    """
    assert isinstance(kt._STATE_LOCK, type(threading.Lock()))

    # The lock must actually be taken while a setter mutates global state:
    # patch it with a recording proxy and confirm acquisition.
    acquired = []

    class RecordingLock:
        def __enter__(self):
            acquired.append(True)
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(kt, "_STATE_LOCK", RecordingLock())
    kt.set_nvcc_discovery("system")
    try:
        kt.set_compute_capabilities("8.6")
    finally:
        kt.set_compute_capabilities(None)
    assert acquired, "global-state setters must acquire _STATE_LOCK"
