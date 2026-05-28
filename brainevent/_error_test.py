# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Tests for the layered toolchain/compilation exception hierarchy."""

from brainevent._error import (
    KernelError, KernelToolchainError, CompilationError,
    NvccNotFoundError, HostCompilerNotFoundError, HeaderNotFoundError,
    GpuArchDetectionError, HostCompilerIncompatibleError, KernelLoadError,
)


def test_toolchain_subclasses():
    assert issubclass(NvccNotFoundError, KernelToolchainError)
    assert issubclass(HostCompilerNotFoundError, KernelToolchainError)
    assert issubclass(HeaderNotFoundError, KernelToolchainError)
    assert issubclass(GpuArchDetectionError, KernelToolchainError)


def test_compilation_subclasses():
    assert issubclass(HostCompilerIncompatibleError, CompilationError)


def test_load_subclass():
    assert issubclass(KernelLoadError, KernelError)


def test_compilation_error_stage_field():
    e = CompilationError("boom", compiler_output="out", command="cmd", stage="link")
    assert e.stage == "link"
    assert "cmd" in str(e)
    assert "out" in str(e)


def test_compilation_error_default_stage():
    assert CompilationError("x").stage == "compile"
