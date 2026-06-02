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
