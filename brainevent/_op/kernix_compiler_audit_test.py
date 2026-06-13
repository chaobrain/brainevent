# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Reproduction tests for the kernix_compiler audit findings M11, M12, L13.

These tests are hermetic: they never invoke a real ``nvcc`` / ``cl`` / ``g++``.
``subprocess.run`` is monkeypatched to capture the argv (and kwargs) that the
backend assembled, and the built command list is asserted on directly.
"""

import subprocess
import sys

import pytest

from brainevent._op import kernix_compiler as kc
from brainevent._op.kernix_toolchain import CppToolchain, CudaToolchain


# ---------------------------------------------------------------------------
# Hermetic helpers
# ---------------------------------------------------------------------------

def _fake_cuda_toolchain(nvcc="nvcc", cxx="g++") -> CudaToolchain:
    return CudaToolchain(
        nvcc=nvcc,
        cxx=cxx,
        cuda_home="/usr/local/cuda",
        cuda_include_dirs=("/usr/local/cuda/include",),
        xla_ffi_include_dir="/xla/ffi/include",
        brainevent_include_dir="/be/include",
    )


def _fake_cpp_toolchain(cxx="g++") -> CppToolchain:
    return CppToolchain(
        cxx=cxx,
        xla_ffi_include_dir="/xla/ffi/include",
        brainevent_include_dir="/be/include",
    )


class _Recorder:
    """Capture the argv and kwargs of the (monkeypatched) ``subprocess.run``."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.cmd = None
        self.kwargs = None
        self._returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    def __call__(self, cmd, **kwargs):
        self.cmd = list(cmd)
        self.kwargs = kwargs
        return subprocess.CompletedProcess(
            args=cmd, returncode=self._returncode,
            stdout=self._stdout, stderr=self._stderr,
        )


def _patch_run(monkeypatch, recorder):
    # The backend calls ``subprocess.run`` via the module-level ``subprocess``.
    monkeypatch.setattr(kc.subprocess, "run", recorder)


def _compile_cuda(monkeypatch, tmp_path, *, extra_ldflags=None, verbose=False,
                  nvcc="nvcc", cxx="g++"):
    rec = _Recorder()
    _patch_run(monkeypatch, rec)
    out = str(tmp_path / "out.so")
    build = str(tmp_path / "cuda_build")
    CUDABackend = kc.CUDABackend
    backend = CUDABackend(_fake_cuda_toolchain(nvcc=nvcc, cxx=cxx))
    backend.compile_source(
        "int main(){}", out, build,
        extra_ldflags=extra_ldflags, verbose=verbose,
    )
    return rec


def _compile_cpp(monkeypatch, tmp_path, *, extra_ldflags=None, verbose=False,
                 build_dir=None, cxx="g++"):
    rec = _Recorder()
    _patch_run(monkeypatch, rec)
    out = str(tmp_path / "outdir" / "out.so")
    if build_dir is None:
        build_dir = str(tmp_path / "cpp_build")
    backend = kc.CPPBackend(_fake_cpp_toolchain(cxx=cxx))
    backend.compile_source(
        "int main(){}", out, build_dir,
        extra_ldflags=extra_ldflags, verbose=verbose,
    )
    return rec


# ---------------------------------------------------------------------------
# M11 — extra_ldflags splitting must be consistent across CUDA and CPP
# ---------------------------------------------------------------------------

def _linker_tokens_cpp(cmd):
    """The raw ldflag tokens as the CPP backend forwards them."""
    return [t for t in cmd if t in ("-L/x", "-lfoo")]


def _linker_tokens_cuda(cmd):
    """For each ``-Xlinker <tok>`` pair, return ``<tok>`` in order."""
    toks = []
    for i, t in enumerate(cmd):
        if t == "-Xlinker" and i + 1 < len(cmd):
            toks.append(cmd[i + 1])
    return toks


def test_m11_cpp_passes_ldflags_as_raw_tokens(monkeypatch, tmp_path):
    rec = _compile_cpp(monkeypatch, tmp_path, extra_ldflags=["-L/x", "-lfoo"])
    # CPP forwards each element verbatim, in order, exactly once.
    assert _linker_tokens_cpp(rec.cmd) == ["-L/x", "-lfoo"]


def test_m11_cuda_passes_each_ldflag_via_single_xlinker(monkeypatch, tmp_path):
    rec = _compile_cuda(monkeypatch, tmp_path, extra_ldflags=["-L/x", "-lfoo"])
    cmd = rec.cmd
    # The deprecated/inconsistent ``--linker-options`` form must be gone.
    assert "--linker-options" not in cmd
    # Each element becomes exactly one ``-Xlinker <token>`` pair, in order.
    assert _linker_tokens_cuda(cmd) == ["-L/x", "-lfoo"]
    assert cmd.count("-Xlinker") == 2


def test_m11_backends_are_equivalent(monkeypatch, tmp_path):
    """Both backends pass the *same* linker tokens, in the same order, once each."""
    cuda = _compile_cuda(monkeypatch, tmp_path, extra_ldflags=["-L/x", "-lfoo"])
    cpp = _compile_cpp(monkeypatch, tmp_path, extra_ldflags=["-L/x", "-lfoo"])
    assert _linker_tokens_cuda(cuda.cmd) == _linker_tokens_cpp(cpp.cmd) == ["-L/x", "-lfoo"]


# ---------------------------------------------------------------------------
# M12 — subprocess decoding must be utf-8 / errors="replace"
# ---------------------------------------------------------------------------

def test_m12_cuda_run_uses_utf8_replace(monkeypatch, tmp_path):
    rec = _compile_cuda(monkeypatch, tmp_path)
    assert rec.kwargs.get("errors") == "replace"
    assert rec.kwargs.get("encoding") == "utf-8"


def test_m12_cpp_run_uses_utf8_replace(monkeypatch, tmp_path):
    rec = _compile_cpp(monkeypatch, tmp_path)
    assert rec.kwargs.get("errors") == "replace"
    assert rec.kwargs.get("encoding") == "utf-8"


def test_m12_no_unicodedecodeerror_escapes(monkeypatch, tmp_path):
    """A non-UTF-8 locale must not let UnicodeDecodeError escape subprocess.run.

    We simulate ``text=True`` strict decoding: if the caller does *not* request
    ``errors="replace"``, the fake raises UnicodeDecodeError from *inside*
    ``subprocess.run`` (exactly where the real defect surfaces). The fix must
    pass ``errors="replace"`` so this never triggers.
    """
    def fake_run(cmd, **kwargs):
        if kwargs.get("errors") != "replace":
            raise UnicodeDecodeError("charmap", b"\x81", 0, 1, "undecodable byte")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(kc.subprocess, "run", fake_run)

    out = str(tmp_path / "out.so")
    build = str(tmp_path / "b")
    # Should not raise UnicodeDecodeError.
    kc.CUDABackend(_fake_cuda_toolchain()).compile_source("x", out, build)
    kc.CPPBackend(_fake_cpp_toolchain()).compile_source("x", out, str(tmp_path / "c"))


def test_m12_run_helper_forwards_errors_replace(monkeypatch):
    """The low-level ``_run`` helper itself must request errors='replace'."""
    rec = _Recorder()
    monkeypatch.setattr(kc.subprocess, "run", rec)
    kc._run(["nvcc", "--version"], timeout=5, stage="probe")
    assert rec.kwargs.get("errors") == "replace"
    assert rec.kwargs.get("encoding") == "utf-8"


# ---------------------------------------------------------------------------
# L13 — shlex.join for display; CPP honors build_dir; HIP stub message
# ---------------------------------------------------------------------------

def test_l13_cuda_verbose_uses_shlex_join(monkeypatch, tmp_path, capsys):
    # A compiler path containing a space must render shell-quoted, not split.
    _compile_cuda(monkeypatch, tmp_path, verbose=True, nvcc="/opt/cuda tools/nvcc")
    printed = capsys.readouterr().out
    # shlex.join quotes the spaced path; " ".join would leave it bare.
    assert "'/opt/cuda tools/nvcc'" in printed


def test_l13_cpp_verbose_uses_shlex_join(monkeypatch, tmp_path, capsys):
    _compile_cpp(monkeypatch, tmp_path, verbose=True, cxx="/opt/g cc/g++")
    printed = capsys.readouterr().out
    assert "'/opt/g cc/g++'" in printed


def test_l13_cpp_honors_caller_build_dir(monkeypatch, tmp_path):
    """CPP must write its intermediate source into the caller-provided build_dir."""
    rec = _Recorder()
    _patch_run(monkeypatch, rec)
    build_dir = tmp_path / "explicit_build"
    out = tmp_path / "outdir" / "out.so"
    kc.CPPBackend(_fake_cpp_toolchain()).compile_source(
        "int main(){}", str(out), str(build_dir),
    )
    # The kernel source goes into the honored build_dir, not the output dir.
    assert (build_dir / "kernel.cpp").exists()
    # And the compiled source argument points inside the build_dir.
    src_arg = rec.cmd[1]
    assert src_arg == str(build_dir / "kernel.cpp")


def test_l13_hip_stub_message_is_clear():
    backend = kc.HIPBackend()
    with pytest.raises(NotImplementedError) as ei:
        backend.compile_source("src", "out.so", "build")
    msg = str(ei.value)
    assert "HIP" in msg
    assert "not yet implemented" in msg.lower()
