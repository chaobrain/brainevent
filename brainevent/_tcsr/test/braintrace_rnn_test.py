# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Test the BrainTrace benchmark against the local TCSR implementation."""

import importlib.util
import inspect
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from brainevent._tcsr.main import TCSR


_BENCHMARK = (
    Path(__file__).resolve().parents[1]
    / "dev"
    / "Benchmark_in_Training"
    / "OneLayerRNN"
    / "BrainTrace"
    / "braintrace_rnn.py"
)
_CASE_RUNNER = _BENCHMARK.with_name("case.py")


def _load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "_braintrace_rnn_local_tcsr_test",
        _BENCHMARK,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_braintrace_benchmark_defaults_to_exact_tcsr() -> None:
    """Expose only exact local TCSR and plain CSR benchmark routes."""
    benchmark = _load_benchmark()

    assert benchmark.SPARSE_TYPES == ("tcsr", "csr")
    assert inspect.signature(benchmark.run).parameters["sparse_type"].default == (
        "tcsr"
    )


def test_braintrace_benchmark_bootstraps_repository_from_external_cwd(
    tmp_path: Path,
) -> None:
    """Import the benchmark from outside the repository using local sources."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy, sys; "
                "runpy.run_path(sys.argv[1], run_name='_braintrace_import_test')"
            ),
            str(_BENCHMARK),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_case_runner_keeps_default_float32_and_allows_explicit_int64(
    tmp_path: Path,
) -> None:
    """Keep BrainTrace float32 while allowing TCSR's explicit int64 offsets."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("CUDA_VISIBLE_DEVICES", None)

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os, runpy, sys; "
                "scope = runpy.run_path(sys.argv[1], run_name='_case_config_test'); "
                "jax = scope['jax']; "
                "assert not jax.config.jax_enable_x64; "
                "assert os.environ['CUDA_VISIBLE_DEVICES'] == '2'; "
                "assert str(jax.numpy.eye(1).dtype) == 'float32'; "
                "assert str(jax.numpy.arange(1, dtype=jax.numpy.int64).dtype) == 'int64'"
            ),
            str(_CASE_RUNNER),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_build_recurrent_uses_checked_local_tcsr_factory() -> None:
    """Build sorted local TCSR storage without the removed BPSA mode."""
    benchmark = _load_benchmark()

    recurrent = benchmark.build_recurrent(
        4,
        0.5,
        "tcsr",
        "cuda_raw",
        seed=1,
    )

    assert isinstance(recurrent, TCSR)
    assert not hasattr(recurrent, "binary_grad_mode")
    indices = np.asarray(recurrent.indices)
    indptr = np.asarray(recurrent.indptr)
    for row in range(recurrent.shape[0]):
        begin, end = int(indptr[row]), int(indptr[row + 1])
        assert np.all(np.diff(indices[begin:end]) >= 0)
