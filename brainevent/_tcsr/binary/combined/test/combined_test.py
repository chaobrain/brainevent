# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Build and run the standalone CUDA correctness test of combined::launch (combined_test.cu)."""

import shutil
import subprocess
from pathlib import Path

import pytest

TEST = Path(__file__).resolve().with_name("combined_test.cu")


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="requires nvcc")
def test_combined_matches_cpu_reference(tmp_path: Path) -> None:
    """int32/int64 offsets, tile boundaries, row chunks, long rows, many batches, silent and repeated calls."""
    executable = tmp_path / "combined_test"
    built = subprocess.run(["nvcc", "-std=c++17", "-O2", "-arch=native", "-DBE_SPMM_STANDALONE", str(TEST), str(TEST.parent.parent / "combined_spmm_binary.cu"),
                            "-o", str(executable)], capture_output=True, text=True, timeout=300)
    assert built.returncode == 0, built.stderr
    run = subprocess.run([str(executable)], capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "ALL PASSED" in run.stdout
