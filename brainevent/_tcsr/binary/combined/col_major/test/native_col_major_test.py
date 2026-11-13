# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Build a plain C++ caller and link the self-contained column-major CUDA object."""

from pathlib import Path
import shutil
import subprocess

import jax
import pytest

import brainevent as _brainevent
from brainevent._op.kernix_toolchain import _select_nvcc

_BE_INCLUDE = str(Path(_brainevent.__file__).resolve().parent / "include")

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr


def _has_cuda_gpu() -> bool:
    try:
        return bool(jax.devices("cuda"))
    except RuntimeError:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not _has_cuda_gpu(), reason="CUDA device required")
def test_native_col_major(tmp_path: Path) -> None:
    nvcc, cxx = _select_nvcc()[0], shutil.which("g++")
    if nvcc is None or cxx is None:
        pytest.skip("nvcc and g++ required")
    cuda = Path(nvcc).resolve().parents[1]
    caller = tmp_path / "caller.o"
    run([cxx, "-std=c++17", "-I" + _BE_INCLUDE, "-O2", "-I" + str(cuda / "include"), "-c",
         str(ROOT / "test/native_col_major.cpp"), "-o", str(caller)])
    # The column-major module is self-contained: it also defines the combined:: API.
    module = tmp_path / "combined_spmm_col_major_binary.o"
    run([nvcc, "-std=c++17", "-I" + _BE_INCLUDE, "-O2", "-arch=native",
         "-c", str(ROOT / "combined_spmm_col_major_binary.cu"), "-o", str(module)])
    executable = tmp_path / "native_col_major"
    run([nvcc, str(caller), str(module), "-o", str(executable)])
    run([str(executable)])
