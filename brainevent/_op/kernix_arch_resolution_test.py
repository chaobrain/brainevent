# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Hermetic tests for compute-capability resolution (no GPU required)."""

import pytest

from brainevent._op import kernix_toolchain as kt


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


# --- resolve_compute_capabilities precedence ------------------------------

def test_resolve_explicit_wins(monkeypatch):
    monkeypatch.setattr(kt, "_arch_from_jax", lambda: ["sm_99"])
    assert kt.resolve_compute_capabilities("8.6") == ["sm_86"]
    assert kt.resolve_compute_capabilities(["8.6", "9.0"]) == ["sm_86", "sm_90"]


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


# --- gencode_flags --------------------------------------------------------

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
