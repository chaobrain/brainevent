# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Compile a plain C++ caller and link binary/weighted native CUDA objects."""

from pathlib import Path
import shutil
import subprocess

import jax
import pytest

import brainevent as _brainevent

_BE_INCLUDE = str(Path(_brainevent.__file__).resolve().parent / "include")

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(jax.devices()[0].platform != 'gpu', reason='CUDA device required')
def test_native_weighted(tmp_path: Path) -> None:
    nvcc, cxx = shutil.which('nvcc'), shutil.which('g++')
    if nvcc is None or cxx is None:
        pytest.skip('nvcc and g++ required')
    cuda = Path(nvcc).resolve().parents[1]
    caller = tmp_path / 'caller.o'
    run([cxx, '-std=c++17', "-I" + _BE_INCLUDE, '-O2', '-I' + str(cuda / 'include'), '-c',
         str(ROOT / 'test/native_weighted.cpp'), '-o', str(caller)])
    objects = []
    for source in (ROOT / 'combined_spmm_binary.cu', ROOT / 'combined_spmm_weighted.cu',
                   ROOT / 'col_major/combined_spmm_col_major_binary.cu', ROOT / 'col_major/combined_spmm_col_major_weighted.cu'):
        obj = tmp_path / (source.stem + '.o')
        run([nvcc, '-std=c++17', "-I" + _BE_INCLUDE, '-O2', '-arch=native', '-DBE_SPMM_STANDALONE',
             '-c', str(source), '-o', str(obj)])
        objects.append(str(obj))
    executable = tmp_path / 'native_weighted'
    run([nvcc, str(caller), *objects, '-o', str(executable)])
    run([str(executable)])
