# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Tests for CompiledModule load-time error wrapping."""

import pytest

from brainevent._op import kernix_runtime as kr
from brainevent._error import KernelLoadError


def test_dlopen_failure_wrapped(monkeypatch):
    def boom(path):
        raise OSError("libcudart.so.12: cannot open shared object file: No such file or directory")
    monkeypatch.setattr(kr.ctypes, "CDLL", boom)
    with pytest.raises(KernelLoadError) as ei:
        kr.CompiledModule("/tmp/does-not-exist.so", ["f"])
    msg = str(ei.value)
    assert "E-LOAD" in msg
    assert "/tmp/does-not-exist.so" in msg
    assert "LD_LIBRARY_PATH" in msg
