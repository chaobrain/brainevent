"""Link ordinary C++ callers with the standalone region-delay translation units."""

from pathlib import Path
import shutil
import subprocess

import pytest

from brainevent._csr.delay.test._helpers import GPU

ROOT = Path(__file__).resolve().parents[1]


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


@GPU
def test_native_interfaces(tmp_path):
    nvcc, cxx = shutil.which("nvcc"), shutil.which("g++")
    if nvcc is None or cxx is None:
        pytest.skip("nvcc and g++ required")
    cuda = Path(nvcc).resolve().parents[1]
    objects = []
    for stem in ("hybrid_delay_spmv_weighted", "hybrid_delay_spmv_binary"):
        obj = tmp_path / (stem + ".o")
        _run([nvcc, "-std=c++17", "-O2", "-arch=native", "-DBE_REGION_DELAY_STANDALONE",
              "-c", str(ROOT / "hybrid" / (stem + ".cu")), "-o", str(obj)])
        objects.append(str(obj))
    for weighted_only in (True, False):
        caller = tmp_path / f"caller_{weighted_only}.o"
        flags = ["-DREGION_DELAY_TEST_WEIGHTED_ONLY"] if weighted_only else []
        _run([cxx, "-std=c++17", "-O2", "-I" + str(cuda / "include"), *flags,
              "-c", str(ROOT / "test/native_interfaces.cpp"), "-o", str(caller)])
        executable = tmp_path / f"native_{weighted_only}"
        _run([nvcc, str(caller), *(objects[:1] if weighted_only else objects), "-o", str(executable)])
        _run([str(executable)])
