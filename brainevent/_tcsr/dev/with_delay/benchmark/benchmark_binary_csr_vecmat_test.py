"""Check case generation, shared inputs, step statistics and CSV orchestration."""

import csv
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from brainevent._tcsr.dev.with_delay.benchmark import benchmark_binary_csr_vecmat as benchmark
from brainevent._tcsr.dev.with_delay.benchmark.native_runner import BenchmarkResult


def test_seeded_generation_and_independent_delay_assignment() -> None:
    kwargs: dict[str, Any] = dict(n=24, degree=20, event_density=.3, duration_steps=11,
                                  k=3, seed=18)
    first, gini = benchmark.generate_case_inputs(**kwargs)
    second, _ = benchmark.generate_case_inputs(**kwargs)
    for name in ("weights", "indices", "indptr", "delay_count", "events"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert gini == 0
    assert first.events.shape == (24, 11)
    assert first.events.flags.f_contiguous
    assert first.delay_steps == (0, 1, 2)
    assert first.buffer_slots == 3
    np.testing.assert_array_equal(first.delay_off, np.array([0, 24, 48]))
    np.testing.assert_array_equal(first.delay_count.sum(axis=1), np.full(24, 20))
    assert np.all(np.count_nonzero(first.delay_count, axis=1) > 1)
    # Changing the delay choices must not shift the separately seeded event stream.
    changed, _ = benchmark.generate_case_inputs(**dict(kwargs, k=2))
    np.testing.assert_array_equal(first.events, changed.events)
    different, _ = benchmark.generate_case_inputs(**dict(kwargs, seed=19))
    assert not np.array_equal(first.events, different.events)


def test_gini_and_zero_activity_and_empty_cases() -> None:
    inputs, actual_gini = benchmark.generate_case_inputs(n=64, degree=10, event_density=0,
        duration_steps=3, k=1, seed=1, target_gini=.2)
    assert abs(actual_gini - .2) < .002
    assert inputs.indices.size == 640
    assert not np.any(inputs.events)
    assert inputs.delay_steps == (0,) and inputs.buffer_slots == 1
    inputs, _ = benchmark.generate_case_inputs(n=0, degree=0, event_density=1,
        duration_steps=3, k=2, seed=1)
    assert inputs.events.shape == (0, 3)
    assert inputs.delay_count.shape == (0, 2)


def test_sparse_graph_retains_all_k_segments() -> None:
    inputs, _ = benchmark.generate_case_inputs(n=1, degree=1, event_density=1,
        duration_steps=4, k=8, seed=1)
    assert inputs.delay_steps == tuple(range(8))
    assert inputs.delay_count.shape == (1, 8)
    assert np.count_nonzero(inputs.delay_count) == 1
    assert inputs.buffer_slots == 8


@pytest.mark.parametrize("changes", [
    {"n": -1}, {"n": True}, {"degree": -1}, {"degree": 9},
    {"duration_steps": 0}, {"duration_steps": 2**31}, {"duration_steps": False},
    {"event_density": float("nan")}, {"event_density": 1.1}, {"event_density": -.1},
    {"k": 0}, {"k": -1}, {"k": False}, {"k": 3.0}, {"k": 2**31}, {"target_gini": -1},
])
def test_invalid_case_configuration(changes: dict[str, Any]) -> None:
    kwargs: dict[str, Any] = dict(n=8, degree=3, event_density=.5, duration_steps=4, k=2, seed=1)
    with pytest.raises(ValueError):
        benchmark.generate_case_inputs(**dict(kwargs, **changes))


@pytest.mark.parametrize("rate,dt", [(-1, .1), (float("inf"), .1), (1, 0), (1, float("nan")), (10001, .1)])
def test_invalid_rates(rate: float, dt: float) -> None:
    with pytest.raises(ValueError):
        benchmark.spike_density_from_rate_hz(rate, dt)


def test_rate_conversion_and_memory_accounting() -> None:
    assert benchmark.spike_density_from_rate_hz(10000) == 1
    assert benchmark.spike_density_from_rate_hz(0) == 0
    estimate = benchmark._estimated_device_bytes(100, 1000, 10, 3)
    assert estimate > 1000 * 8 * 2 + 100 * 10 * 4
    assert benchmark._estimated_device_bytes(100, 1000, 11, 3) == estimate + 400
    assert benchmark._estimated_device_bytes(100, 2**31, 10, 2**30) > 2**34


def _fake_case(monkeypatch: pytest.MonkeyPatch, fail_sparse: bool = False) -> list[Any]:
    seen: list[Any] = []

    class FakeCase:
        def __init__(self, inputs: Any) -> None:
            self.inputs = inputs
            seen.append(inputs)

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: Any) -> None:
            seen.append("closed")

        def run_hybrid(self, **kwargs: Any) -> BenchmarkResult:
            seen.append(("hybrid", kwargs))
            return BenchmarkResult(dict(timed_mean_ms=22., duration_steps=self.inputs.duration_steps), None, None)

        def run_cusparse(self, **kwargs: Any) -> BenchmarkResult:
            seen.append(("cusparse", kwargs))
            if fail_sparse:
                raise RuntimeError("failed segment")
            return BenchmarkResult(dict(timed_mean_ms=33., duration_steps=self.inputs.duration_steps), None, None)

    monkeypatch.setattr(benchmark, "NativeCase", FakeCase)
    return seen


def test_one_initialization_for_both_operators(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _fake_case(monkeypatch)
    rows = benchmark.benchmark_case(scale=.024, density_mat=.5, firing_rate_hz=100,
        target_gini=0, k=3, seed=18, duration_steps=11, warmup=3, runs=4)
    assert len(seen) == 4
    assert seen[1:3] == [("hybrid", {"warmup": 3, "runs": 4}), ("cusparse", {"warmup": 3, "runs": 4})]
    assert seen[-1] == "closed"
    assert [row["timed_mean_step_ms"] for row in rows] == [2., 3.]
    assert [row["backend"] for row in rows] == ["hybrid", "cusparse"]
    assert all("runner" not in row for row in rows)
    assert all(row["k"] == row["segment_count"] == row["buffer_slots"] == 3 for row in rows)
    assert all(row["duration_steps"] == 11 and row["timing_scope"] == "projection_only" for row in rows)


def test_per_operator_failure_preserves_success(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _fake_case(monkeypatch, fail_sparse=True)
    rows = benchmark.benchmark_case(scale=.004, density_mat=0, firing_rate_hz=0,
        target_gini=0, k=1, seed=1)
    assert not rows[0]["error"] and "failed segment" in rows[1]["error"]
    assert seen[-1] == "closed"


@pytest.mark.parametrize("changes", [{"scale": 0}, {"density_mat": 2}, {"k": 0}, {"firing_rate_hz": -1}])
def test_case_initialization_errors_are_reported_for_both(changes: dict[str, Any]) -> None:
    kwargs: dict[str, Any] = dict(scale=.004, density_mat=.5, firing_rate_hz=100,
                                  target_gini=0, k=1, seed=1)
    rows = benchmark.benchmark_case(**dict(kwargs, **changes))
    assert len(rows) == 2 and all(row["error"] for row in rows)
    assert {row["backend"] for row in rows} == {"hybrid", "cusparse"}


@pytest.mark.parametrize("k,limit", [(1, 1), (2**31 - 1, 96 * 1024**3)])
def test_memory_limit_before_generating_inputs(k: int, limit: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark, "DEVICE_MEMORY_LIMIT_BYTES", limit)

    def unexpected_generation(**kwargs: Any) -> None:
        pytest.fail("memory budget must be checked before generating inputs")

    monkeypatch.setattr(benchmark, "generate_case_inputs", unexpected_generation)
    rows = benchmark.benchmark_case(scale=1, density_mat=.1, firing_rate_hz=100,
        target_gini=0, k=k, seed=1)
    assert all("memory limit" in row["error"] for row in rows)


def test_csv_repeats_and_summary_use_complete_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_case(monkeypatch)
    monkeypatch.setattr(benchmark, "RNG_SEED_LIST", [101, 202])
    grid = [{"SCALE_LIST": [.024], "DENSITY_MAT_LIST": [.5], "DENSITY_VEC_LIST": [100],
             "GINI_LIST": [0], "K_LIST": [1, 3]}]
    rows = benchmark.run_all_cases(grid, run_directory=tmp_path, duration_steps=11)
    assert len(rows) == 8
    assert {row["k"] for row in rows} == {1, 3}
    with (tmp_path / "case_001" / "summary.csv").open() as file:
        summary = list(csv.DictReader(file))
    assert [(row["backend"], row["k"]) for row in summary] == [
        ("hybrid", "1"), ("cusparse", "1"), ("hybrid", "3"), ("cusparse", "3")]
    assert all("runner" not in row for row in summary)
    with (tmp_path / "case_001" / "case_1.csv").open() as file:
        written = list(csv.DictReader(file))
    assert [(row["backend"], row["k"]) for row in written] == [
        ("hybrid", "1"), ("cusparse", "1"), ("hybrid", "3"), ("cusparse", "3")]
    assert all("runner" not in row for row in written)
    assert [float(row["timed_mean_step_ms_mean"]) for row in summary] == [2., 3., 2., 3.]
    assert all(row["success_count"] == "2" and row["failure_count"] == "0" for row in summary)
    failed = dict(rows[0], error="failure")
    benchmark._write_summary([failed], tmp_path / "failed.csv")
    with (tmp_path / "failed.csv").open() as file:
        assert next(csv.DictReader(file))["success_count"] == "0"


def test_default_output_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark, "RUNS_DIR", tmp_path)
    assert benchmark.run_all_cases([]) == []
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize("smoke,fail", [(True, False), (False, True)])
def test_cli_selects_grid_and_reports_failure(smoke: bool, fail: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []

    def run(cases: Any, **kwargs: Any) -> Any:
        calls.append((cases, kwargs))
        return [{"error": "problem" if fail else ""}]

    monkeypatch.setattr(benchmark, "run_all_cases", run)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--gpu", "1", "--steps", "7", *(["--smoke"] if smoke else [])])
    if fail:
        with pytest.raises(SystemExit, match="1"):
            benchmark._main()
    else:
        benchmark._main()
    assert calls[0][1]["duration_steps"] == 7
    assert calls[0][0][0]["SCALE_LIST"] == ([0.256] if smoke else [1000])
    assert calls[0][0][0]["K_LIST"] == ([3] if smoke else benchmark.CASE_LIST[0]["K_LIST"])
    assert calls[0][0][0]["BACKEND_LIST"] == (
        ["hybrid", "cusparse"] if smoke else benchmark.CASE_LIST[0]["BACKEND_LIST"])


@pytest.mark.parametrize("gpu_args,expected", [([], "1"), (["--gpu", "3"], "3")])
def test_cli_selects_gpu_before_numpy_import(tmp_path: Path, gpu_args: list[str], expected: str) -> None:
    """Apply the default or explicit GPU choice before importing NumPy."""
    guard = tmp_path / "sitecustomize.py"
    guard.write_text(
        "import builtins, os\n"
        "original_import = builtins.__import__\n"
        "def guarded_import(name, *args, **kwargs):\n"
        "    if name == 'numpy' and os.environ.get('CUDA_VISIBLE_DEVICES') != " + repr(expected) + ":\n"
        "        raise RuntimeError('GPU choice was applied after NumPy import')\n"
        "    return original_import(name, *args, **kwargs)\n"
        "builtins.__import__ = guarded_import\n"
    )
    environment = dict(os.environ)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment["PYTHONPATH"] = os.pathsep.join((str(tmp_path), environment.get("PYTHONPATH", "")))
    result = subprocess.run(
        [sys.executable, str(Path(benchmark.__file__).resolve()), *gpu_args, "--help"],
        env=environment, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_selected_backend_is_only_executed_and_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run only selected backends and preserve their requested order."""
    called: list[str] = []

    class FakeCase:
        def __init__(self, inputs: Any) -> None:
            self.inputs = inputs

        def __enter__(self) -> "FakeCase":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def run_hybrid(self, **kwargs: Any) -> BenchmarkResult:
            called.append("hybrid")
            return BenchmarkResult(dict(timed_mean_ms=1., duration_steps=1), None, None)

        def run_cusparse(self, **kwargs: Any) -> BenchmarkResult:
            called.append("cusparse")
            return BenchmarkResult(dict(timed_mean_ms=2., duration_steps=1), None, None)

    monkeypatch.setattr(benchmark, "NativeCase", FakeCase)
    kwargs: dict[str, Any] = dict(scale=.004, density_mat=.5, firing_rate_hz=1.,
                                  target_gini=0., k=1, seed=1, duration_steps=1)
    rows = benchmark.benchmark_case(**kwargs, backend_list=("cusparse",))
    assert called == ["cusparse"]
    assert [row["backend"] for row in rows] == called
    called.clear()
    rows = benchmark.benchmark_case(**kwargs, backend_list=("cusparse", "hybrid"))
    assert called == ["cusparse", "hybrid"]
    assert [row["backend"] for row in rows] == called


def test_initialization_failure_reports_only_selected_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep an initialization error confined to the selected result rows."""
    monkeypatch.setattr(benchmark, "DEVICE_MEMORY_LIMIT_BYTES", 1)
    rows = benchmark.benchmark_case(scale=.004, density_mat=.5, firing_rate_hz=1.,
        target_gini=0., k=1, seed=1, backend_list=("cusparse",))
    assert [row["backend"] for row in rows] == ["cusparse"]
    assert "memory limit" in rows[0]["error"]


def test_hybrid_only_does_not_reserve_cusparse_split_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept a hybrid case that fits without the unused split allocation."""
    seen = _fake_case(monkeypatch)
    hybrid_bytes = benchmark._estimated_device_bytes(4, 8, 1, 1, include_split=False)
    both_bytes = benchmark._estimated_device_bytes(4, 8, 1, 1)
    assert hybrid_bytes < both_bytes
    monkeypatch.setattr(benchmark, "DEVICE_MEMORY_LIMIT_BYTES", hybrid_bytes)
    kwargs: dict[str, Any] = dict(scale=.004, density_mat=.5, firing_rate_hz=1.,
                                  target_gini=0., k=1, seed=1, duration_steps=1)
    hybrid = benchmark.benchmark_case(**kwargs, backend_list=("hybrid",))
    both = benchmark.benchmark_case(**kwargs)
    assert [row["error"] for row in hybrid] == [""]
    assert all("memory limit" in row["error"] for row in both)
    assert sum(isinstance(item, tuple) and item[0] == "hybrid" for item in seen) == 1


@pytest.mark.parametrize("backend_list", [(), ("hybrid", "hybrid"), ("unknown",)])
def test_invalid_backend_list_is_rejected(backend_list: tuple[str, ...]) -> None:
    """Reject empty, repeated, and unknown backend selections."""
    with pytest.raises(ValueError, match="BACKEND_LIST"):
        benchmark.benchmark_case(scale=.004, density_mat=.5, firing_rate_hz=1.,
            target_gini=0., k=1, seed=1, backend_list=backend_list)


def test_case_grid_passes_backend_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward the per-case backend selection to the case runner."""
    seen: list[Any] = []

    def fake_benchmark_case(**kwargs: Any) -> list[dict[str, Any]]:
        seen.append(kwargs["backend_list"])
        return []

    monkeypatch.setattr(benchmark, "benchmark_case", fake_benchmark_case)
    grid = [{"SCALE_LIST": [.004], "DENSITY_MAT_LIST": [.5], "DENSITY_VEC_LIST": [1],
             "GINI_LIST": [0.], "K_LIST": [1], "BACKEND_LIST": ["hybrid"]}]
    benchmark.run_all_cases(grid, run_directory=tmp_path)
    assert seen == [["hybrid"]]
