# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Tests for the layered toolchain/compilation exception hierarchy."""

from brainevent._error import (
    BrainEventError, MathError,
    KernelError, KernelToolchainError, CompilationError,
    KernelExecutionError, KernelFallbackExhaustedError,
    KernelNotAvailableError, KernelCompilationError,
    CUDANotInstalledError, BenchmarkDataFnNotProvidedError,
    KernelRegistrationError,
    NvccNotFoundError, HostCompilerNotFoundError, HeaderNotFoundError,
    GpuArchDetectionError, HostCompilerIncompatibleError, KernelLoadError,
    UnsupportedArchError,
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


def test_compilation_error_bare_message_has_no_appended_sections():
    # With neither command nor compiler_output, the message is left verbatim.
    e = CompilationError("just the reason")
    assert str(e) == "just the reason"
    assert e.command == ""
    assert e.compiler_output == ""


def test_compilation_error_appends_only_provided_sections():
    only_cmd = CompilationError("boom", command="nvcc x.cu")
    assert "Command:" in str(only_cmd)
    assert "Compiler output:" not in str(only_cmd)

    only_out = CompilationError("boom", compiler_output="ptxas: error")
    assert "Compiler output:" in str(only_out)
    assert "Command:" not in str(only_out)


def test_unsupported_arch_error_stores_fields_and_message():
    e = UnsupportedArchError(
        "sm_120 unknown to this nvcc",
        compiler_output="nvcc fatal",
        command="nvcc -arch=sm_120",
        stage="compile",
    )
    assert isinstance(e, KernelToolchainError)
    assert e.compiler_output == "nvcc fatal"
    assert e.command == "nvcc -arch=sm_120"
    assert e.stage == "compile"
    # The base KernelToolchainError stores the plain message (no auto-appended
    # command/output sections, unlike CompilationError).
    assert str(e) == "sm_120 unknown to this nvcc"


def test_unsupported_arch_error_defaults_are_empty():
    e = UnsupportedArchError("nope")
    assert e.compiler_output == ""
    assert e.command == ""
    assert e.stage == ""


def test_full_exception_hierarchy_roots_at_brainevent_error():
    # Every public error ultimately derives from BrainEventError so callers can
    # catch the whole family with a single except clause.
    for exc in (
        MathError, KernelError, KernelNotAvailableError, KernelCompilationError,
        KernelFallbackExhaustedError, KernelExecutionError, KernelToolchainError,
        CompilationError, KernelRegistrationError, BenchmarkDataFnNotProvidedError,
        CUDANotInstalledError, NvccNotFoundError, HostCompilerNotFoundError,
        HeaderNotFoundError, GpuArchDetectionError, HostCompilerIncompatibleError,
        UnsupportedArchError, KernelLoadError,
    ):
        assert issubclass(exc, BrainEventError), exc.__name__


def test_kernel_errors_are_catchable_as_kernel_error():
    for exc in (
        KernelNotAvailableError, KernelCompilationError, KernelExecutionError,
        KernelToolchainError, CUDANotInstalledError, KernelLoadError,
        KernelRegistrationError,
    ):
        assert issubclass(exc, KernelError), exc.__name__
