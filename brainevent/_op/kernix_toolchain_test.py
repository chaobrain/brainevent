# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Tests for kernix_toolchain discovery and diagnostics."""

import sys
import threading

import pytest

from brainevent._op import kernix_toolchain as kt
from brainevent._op.kernix_toolchain import CandidateProbe, render_toolchain_error

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="kernix toolchain tests are not supported on Windows",
)


def _touch_exec(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("")
    p.chmod(0o755)


# --- renderer -------------------------------------------------------------

def test_render_sections_present():
    msg = render_toolchain_error(
        stage="nvcc discovery", code="E-NVCC", summary="nvcc not found.",
        probes=[
            CandidateProbe("BRAINEVENT_NVCC_PATH", "", "unset"),
            CandidateProbe("PATH:nvcc", "", "not-found"),
        ],
        remediation=["Install jax[cuda13]"],
    )
    assert "E-NVCC" in msg
    assert "Reason" in msg
    assert "Tried" in msg
    assert "BRAINEVENT_NVCC_PATH" in msg
    assert "How to fix" in msg
    assert "Install jax[cuda13]" in msg


def test_render_includes_command_for_compile():
    msg = render_toolchain_error(
        stage="compile", code="E-COMPILE", summary="failed",
        command="nvcc x.cu", compiler_output="error: boom", remediation=["fix it"],
    )
    assert "Command" in msg and "nvcc x.cu" in msg
    assert "Compiler output" in msg and "boom" in msg


# --- discovery preference -------------------------------------------------

def test_nvcc_discovery_default_and_env(monkeypatch):
    monkeypatch.setattr(kt, "_NVCC_DISCOVERY", None)
    monkeypatch.delenv("BRAINEVENT_NVCC_PREFER", raising=False)
    assert kt.get_nvcc_discovery() == "pip"
    monkeypatch.setenv("BRAINEVENT_NVCC_PREFER", "system")
    assert kt.get_nvcc_discovery() == "system"


def test_nvcc_discovery_function_overrides_env(monkeypatch):
    monkeypatch.setattr(kt, "_NVCC_DISCOVERY", None)
    monkeypatch.setenv("BRAINEVENT_NVCC_PREFER", "system")
    kt.set_nvcc_discovery("pip")
    assert kt.get_nvcc_discovery() == "pip"


def test_nvcc_discovery_invalid():
    with pytest.raises(ValueError):
        kt.set_nvcc_discovery("bogus")


def test_config_prefer_system_nvcc(monkeypatch):
    import brainevent.config as cfg
    monkeypatch.setattr(kt, "_NVCC_DISCOVERY", None)
    cfg.prefer_system_nvcc(True)
    assert kt.get_nvcc_discovery() == "system"
    cfg.prefer_system_nvcc(False)
    assert kt.get_nvcc_discovery() == "pip"


# --- _find_pip_cuda -------------------------------------------------------

def test_find_pip_cuda_consolidated(tmp_path):
    _touch_exec(tmp_path / "cu13" / "bin" / "nvcc")
    (tmp_path / "cu13" / "include").mkdir(parents=True)
    res, probes = kt._find_pip_cuda(roots=[str(tmp_path)])
    assert res is not None
    path, includes = res
    assert path == str(tmp_path / "cu13" / "bin" / "nvcc")
    assert includes == [str(tmp_path / "cu13" / "include")]


def test_find_pip_cuda_consolidated_picks_highest(tmp_path):
    _touch_exec(tmp_path / "cu13" / "bin" / "nvcc")
    (tmp_path / "cu13" / "include").mkdir(parents=True)
    _touch_exec(tmp_path / "cu14" / "bin" / "nvcc")
    (tmp_path / "cu14" / "include").mkdir(parents=True)
    res, _ = kt._find_pip_cuda(roots=[str(tmp_path)])
    assert res[0] == str(tmp_path / "cu14" / "bin" / "nvcc")


def test_find_pip_cuda_split(tmp_path):
    _touch_exec(tmp_path / "cuda_nvcc" / "bin" / "nvcc")
    (tmp_path / "cuda_nvcc" / "include").mkdir(parents=True)
    (tmp_path / "cuda_runtime" / "include").mkdir(parents=True)
    res, _ = kt._find_pip_cuda(roots=[str(tmp_path)])
    assert res is not None
    path, includes = res
    assert path == str(tmp_path / "cuda_nvcc" / "bin" / "nvcc")
    assert str(tmp_path / "cuda_nvcc" / "include") in includes
    assert str(tmp_path / "cuda_runtime" / "include") in includes


def test_find_pip_cuda_absent(tmp_path):
    res, probes = kt._find_pip_cuda(roots=[str(tmp_path)])
    assert res is None


# --- _find_host_cxx -------------------------------------------------------

def test_find_host_cxx_prefers_cxx_env(monkeypatch, tmp_path):
    fake = tmp_path / "mygcc"
    _touch_exec(fake)
    monkeypatch.setenv("CXX", str(fake))
    cxx, probes = kt._find_host_cxx()
    assert cxx == str(fake)


def test_find_host_cxx_conda_before_system(monkeypatch, tmp_path):
    monkeypatch.delenv("CXX", raising=False)
    gpp = tmp_path / "conda" / "bin" / "g++"
    _touch_exec(gpp)
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path / "conda"))
    monkeypatch.setattr(kt.shutil, "which", lambda n: "/usr/bin/" + n)
    cxx, probes = kt._find_host_cxx()
    assert cxx == str(gpp)


def test_find_host_cxx_system_fallback(monkeypatch):
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(kt.shutil, "which", lambda n: "/usr/bin/g++" if n == "g++" else None)
    cxx, probes = kt._find_host_cxx()
    assert cxx == "/usr/bin/g++"


def test_find_host_cxx_none(monkeypatch):
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(kt.shutil, "which", lambda n: None)
    cxx, probes = kt._find_host_cxx()
    assert cxx is None


# --- _select_nvcc ---------------------------------------------------------

def test_select_nvcc_env_override(monkeypatch, tmp_path):
    nvcc = tmp_path / "cudahome" / "bin" / "nvcc"
    _touch_exec(nvcc)
    monkeypatch.setenv("BRAINEVENT_NVCC_PATH", str(nvcc))
    path, includes, probes = kt._select_nvcc()
    assert path == str(nvcc)
    assert includes == [str(tmp_path / "cudahome" / "include")]


def test_select_nvcc_pip_first(monkeypatch):
    monkeypatch.delenv("BRAINEVENT_NVCC_PATH", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)  # H6: now also probed
    monkeypatch.setattr(kt, "_NVCC_DISCOVERY", "pip")
    monkeypatch.setattr(kt, "_find_pip_cuda",
                        lambda roots=None: (("/pip/nvcc", ["/pip/include"]), []))
    monkeypatch.setattr(kt.shutil, "which", lambda n: "/usr/bin/nvcc")
    path, includes, _ = kt._select_nvcc()
    assert path == "/pip/nvcc"


def test_select_nvcc_system_pref(monkeypatch, tmp_path):
    from pathlib import Path
    monkeypatch.delenv("BRAINEVENT_NVCC_PATH", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)  # H6: now also probed
    monkeypatch.setattr(kt, "_NVCC_DISCOVERY", "system")
    monkeypatch.setattr(kt, "_find_pip_cuda",
                        lambda roots=None: (("/pip/nvcc", ["/pip/include"]), []))
    sysnvcc = tmp_path / "sys" / "bin" / "nvcc"
    _touch_exec(sysnvcc)
    monkeypatch.setattr(kt.shutil, "which",
                        lambda n: str(sysnvcc) if n == "nvcc" else None)
    path, includes, _ = kt._select_nvcc()
    assert path == str(sysnvcc)
    # include is derived from the resolved nvcc path (mirrors _include_from_nvcc)
    assert includes == [str(Path(str(sysnvcc)).resolve().parent.parent / "include")]


def test_select_nvcc_not_found(monkeypatch):
    monkeypatch.delenv("BRAINEVENT_NVCC_PATH", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)  # H6: now also probed
    monkeypatch.setattr(kt, "_NVCC_DISCOVERY", "pip")
    monkeypatch.setattr(kt, "_find_pip_cuda", lambda roots=None: (None, []))
    monkeypatch.setattr(kt.shutil, "which", lambda n: None)
    path, includes, probes = kt._select_nvcc()
    assert path is None and includes == []


# --- detect_* -------------------------------------------------------------

def test_detect_cuda_toolchain_no_nvcc(monkeypatch):
    from brainevent._error import NvccNotFoundError
    monkeypatch.setattr(
        kt, "_select_nvcc",
        lambda: (None, [], [CandidateProbe("PATH:nvcc", "", "not-found")]),
    )
    with pytest.raises(NvccNotFoundError) as ei:
        kt.detect_cuda_toolchain()
    msg = str(ei.value)
    assert "E-NVCC" in msg
    assert "jax[cuda" in msg


def test_detect_cuda_toolchain_no_host_cxx(monkeypatch, tmp_path):
    from brainevent._error import HostCompilerNotFoundError
    nvcc = tmp_path / "cu13" / "bin" / "nvcc"
    _touch_exec(nvcc)
    inc = tmp_path / "cu13" / "include"
    inc.mkdir(parents=True, exist_ok=True)
    (inc / "cuda_runtime.h").write_text("/* header */")  # L12: include is validated
    monkeypatch.setattr(
        kt, "_select_nvcc",
        lambda: (str(nvcc), [str(inc)], []),
    )
    # returncode is now inspected (H5), so the stub must provide it.
    monkeypatch.setattr(
        kt.subprocess, "run",
        lambda *a, **k: type(
            "R", (), {"returncode": 0, "stdout": "Cuda release 13.0", "stderr": ""})(),
    )
    monkeypatch.setattr(kt, "_find_host_cxx", lambda: (None, []))
    with pytest.raises(HostCompilerNotFoundError) as ei:
        kt.detect_cuda_toolchain()
    assert "E-CXX" in str(ei.value)
    assert "conda install" in str(ei.value)


def test_cuda_toolchain_dataclass_fields():
    tc = kt.CudaToolchain(
        nvcc="/n", cxx="/c", cuda_home="/h",
        cuda_include_dirs=("/i1", "/i2"),
        xla_ffi_include_dir="/x", brainevent_include_dir="/b",
        nvcc_version="v", cxx_version="g",
    )
    assert tc.cuda_include_dirs == ("/i1", "/i2")
    assert tc.cxx_version == "g"


def test_detect_cpp_toolchain_no_cxx(monkeypatch):
    from brainevent._error import HostCompilerNotFoundError
    monkeypatch.setattr(kt, "_find_host_cxx", lambda: (None, []))
    with pytest.raises(HostCompilerNotFoundError) as ei:
        kt.detect_cpp_toolchain()
    assert "E-CXX" in str(ei.value)


def test_detect_cuda_arch_failure(monkeypatch):
    from brainevent._error import GpuArchDetectionError
    monkeypatch.delenv("BRAINEVENT_COMPUTE_CAPABILITIES", raising=False)
    # Neutralize JAX device detection so the nvidia-smi failure path is reached
    # (otherwise this passes on a real GPU box).
    monkeypatch.setattr(kt, "_arch_from_jax", lambda: None)

    def fake_run(*a, **k):
        return type("R", (), {"returncode": 1, "stdout": "", "stderr": "no smi"})()

    monkeypatch.setattr(kt.subprocess, "run", fake_run)
    with pytest.raises(GpuArchDetectionError) as ei:
        kt.detect_cuda_arch()
    assert "E-ARCH" in str(ei.value)


def test_detect_cuda_arch_env_override(monkeypatch):
    monkeypatch.setenv("BRAINEVENT_COMPUTE_CAPABILITIES", "8.6,8.0")
    assert kt.detect_cuda_arch() == ["sm_86", "sm_80"]


# --- compute capability helpers ------------------------------------------

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


def test_config_set_compute_capability():
    import brainevent

    brainevent.config.set_compute_capability("8.6")
    try:
        assert brainevent.config.get_compute_capability() == ["sm_86"]
        assert kt.resolve_compute_capabilities() == ["sm_86"]
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


def test_find_host_cxx_msvc_on_windows(monkeypatch):
    monkeypatch.setattr(kt.sys, "platform", "win32")
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(
        kt.shutil, "which",
        lambda n: "C:\\VC\\cl.exe" if n in ("cl", "cl.exe") else None)
    cxx, probes = kt._find_host_cxx()
    assert cxx and cxx.lower().endswith("cl.exe")


# --- diagnostics snapshot -------------------------------------------------

def test_collect_diagnostics_keys(monkeypatch):
    monkeypatch.setattr(kt, "_select_nvcc", lambda: ("/n/nvcc", ["/n/include"], []))
    monkeypatch.setattr(kt, "_find_host_cxx", lambda: ("/usr/bin/g++", []))
    monkeypatch.setattr(kt, "_cxx_version", lambda c: "g++ 12")
    snap = kt.collect_toolchain_diagnostics()
    assert snap["nvcc"] == "/n/nvcc"
    assert snap["host_cxx"] == "/usr/bin/g++"
    assert snap["discovery"] in ("pip", "system")
    assert "env:CUDA_HOME" in snap


def test_render_appends_snapshot_when_debug(monkeypatch):
    monkeypatch.setenv("BRAINEVENT_TOOLCHAIN_DEBUG", "1")
    monkeypatch.setattr(kt, "collect_toolchain_diagnostics", lambda: {"nvcc": "/n"})
    msg = kt.render_toolchain_error(stage="x", code="E-X", summary="s")
    assert "Toolchain snapshot" in msg and "/n" in msg


# ---------------------------------------------------------------------------
# Audit regression tests
# ---------------------------------------------------------------------------


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
