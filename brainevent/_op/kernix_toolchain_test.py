# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Tests for kernix_toolchain discovery and diagnostics."""

import sys

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
    monkeypatch.setattr(
        kt, "_select_nvcc",
        lambda: (str(nvcc), [str(tmp_path / "cu13" / "include")], []),
    )
    monkeypatch.setattr(
        kt.subprocess, "run",
        lambda *a, **k: type("R", (), {"stdout": "Cuda release 13.0", "stderr": ""})(),
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
