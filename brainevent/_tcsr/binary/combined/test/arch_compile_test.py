# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Compile every TCSR CUDA source for the oldest pre-sm_80 target of the toolkit."""

import re
import subprocess
from pathlib import Path

import jax
import pytest

import brainevent as _brainevent
from brainevent._op.kernix_toolchain import _select_nvcc

_BE_INCLUDE = Path(_brainevent.__file__).resolve().parent / 'include'
ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ROOT / 'combined_spmm_binary.cu',
    ROOT / 'combined_spmm_weighted.cu',
    ROOT / 'col_major' / 'combined_spmm_col_major_binary.cu',
    ROOT / 'col_major' / 'combined_spmm_col_major_weighted.cu',
)


def _oldest_pre_ampere_arch(nvcc: str) -> str | None:
    """Return the lowest ``sm_XY`` below 80 that this nvcc can target."""
    listed = subprocess.run([nvcc, '--list-gpu-arch'], capture_output=True, text=True, timeout=60)
    codes = sorted(int(m) for m in re.findall(r'compute_(\d+)', listed.stdout) if int(m) < 80)
    return None if not codes else f'sm_{codes[0]}'


@pytest.mark.slow
@pytest.mark.parametrize('source', SOURCES, ids=lambda path: path.name)
def test_sources_compile_without_sm80_intrinsics(tmp_path: Path, source: Path) -> None:
    """``__reduce_add_sync`` exists only on sm_80+; older targets must use the fallback."""
    nvcc = _select_nvcc()[0]
    if nvcc is None:
        pytest.skip('nvcc required')
    arch = _oldest_pre_ampere_arch(nvcc)
    if arch is None:
        pytest.skip('toolkit has no pre-sm_80 target')
    built = subprocess.run(
        [nvcc, '-std=c++17', f'-arch={arch}', '-I', str(_BE_INCLUDE), '-I', jax.ffi.include_dir(),
         '-I', str(ROOT), '-I', str(ROOT / 'col_major'), '-c', str(source),
         '-o', str(tmp_path / (source.stem + '.o'))],
        capture_output=True, text=True, timeout=600,
    )
    assert built.returncode == 0, built.stderr
