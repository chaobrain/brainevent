# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Tests for kernix_compiler flag/error helpers (hermetic, no nvcc needed)."""

from brainevent._op import kernix_compiler as kc


def test_is_host_incompat_true():
    assert kc._is_host_incompat(
        "error: unsupported GNU version! gcc versions later than 13 are not supported"
    )


def test_is_host_incompat_false():
    assert not kc._is_host_incompat("kernel.cu(10): error: expected a ';'")


def test_allow_unsupported_compiler(monkeypatch):
    monkeypatch.setenv("BRAINEVENT_ALLOW_UNSUPPORTED_COMPILER", "1")
    assert kc._allow_unsupported_compiler() is True
    monkeypatch.setenv("BRAINEVENT_ALLOW_UNSUPPORTED_COMPILER", "0")
    assert kc._allow_unsupported_compiler() is False
    monkeypatch.delenv("BRAINEVENT_ALLOW_UNSUPPORTED_COMPILER", raising=False)
    assert kc._allow_unsupported_compiler() is False


def test_raise_compile_error_incompat():
    from brainevent._error import HostCompilerIncompatibleError
    import pytest
    with pytest.raises(HostCompilerIncompatibleError):
        kc._raise_compile_error("unsupported GNU version", "nvcc x.cu", stage="compile")


def test_raise_compile_error_generic():
    from brainevent._error import CompilationError, HostCompilerIncompatibleError
    import pytest
    with pytest.raises(CompilationError) as ei:
        kc._raise_compile_error("syntax error near ';'", "nvcc x.cu", stage="link")
    assert not isinstance(ei.value, HostCompilerIncompatibleError)
    assert ei.value.stage == "link"


def test_unsupported_arch_error():
    from brainevent._error import UnsupportedArchError
    import pytest
    with pytest.raises(UnsupportedArchError) as ei:
        kc._raise_compile_error(
            "nvcc fatal : Unsupported gpu architecture 'compute_120'",
            "nvcc x.cu", stage="compile")
    assert ei.value.stage == "compile"


def test_ninja_file_parses(tmp_path):
    """Generated build.ninja must not have indented top-level bindings."""
    from brainevent._op.kernix_compiler import NinjaBuild
    from brainevent._op.kernix_toolchain import CudaToolchain
    tc = CudaToolchain(
        nvcc="/usr/bin/nvcc", cxx="/usr/bin/g++", cuda_home="/usr/local/cuda",
        cuda_include_dirs=("/usr/local/cuda/include",),
        xla_ffi_include_dir="/x/inc", brainevent_include_dir="/be/inc",
    )
    nb = NinjaBuild(toolchain=tc, build_dir=str(tmp_path), output_name="m", gpu_arch="sm_86")
    nb.add_source(str(tmp_path / "m.cu"))
    nb.generate()
    text = (tmp_path / "build.ninja").read_text()
    for line in text.splitlines():
        if line.startswith(" ") and "=" in line and "command" not in line and "description" not in line:
            raise AssertionError(f"indented top-level binding: {line!r}")
    assert "code=compute_86" in text  # PTX present
    assert "code=sm_86" in text


def test_ninja_file_parses_with_ninja(tmp_path):
    """If ninja is installed, the generated manifest must parse cleanly."""
    import shutil
    import subprocess
    import pytest
    ninja = shutil.which("ninja")
    if ninja is None:
        pytest.skip("ninja not installed")
    from brainevent._op.kernix_compiler import NinjaBuild
    from brainevent._op.kernix_toolchain import CudaToolchain
    tc = CudaToolchain(
        nvcc="/usr/bin/nvcc", cxx="/usr/bin/g++", cuda_home="/usr/local/cuda",
        cuda_include_dirs=("/usr/local/cuda/include",),
        xla_ffi_include_dir="/x/inc", brainevent_include_dir="/be/inc",
    )
    (tmp_path / "m.cu").write_text("// empty\n")
    nb = NinjaBuild(toolchain=tc, build_dir=str(tmp_path), output_name="m", gpu_arch="sm_86")
    nb.add_source(str(tmp_path / "m.cu"))
    nb.generate()
    r = subprocess.run([ninja, "-C", str(tmp_path), "-n"], capture_output=True, text=True)
    assert r.returncode == 0, f"ninja failed to parse: {r.stderr}"
