"""Benchmark independently delayed CSR projections with shared event columns."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from itertools import product
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Sequence

GPU_ID = "2"
if __name__ == "__main__":
    # Select the device before NumPy or project imports can initialize CUDA.
    _gpu_parser = argparse.ArgumentParser(add_help=False)
    _gpu_parser.add_argument("--gpu", default=GPU_ID)
    _gpu_args, _ = _gpu_parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_args.gpu

import numpy as np

# Support both `python -m ...` and direct execution from any directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from brainevent._tcsr.dev.with_delay.benchmark.gini_row_lengths import make_gini_row_lengths
from brainevent._tcsr.dev.with_delay.benchmark.native_runner import NativeCase
from brainevent._tcsr.dev.with_delay.operator.region_delay import (
    DelayInputs, TASK_NNZ, prepare_delay_inputs,
)

RNG_SEED_LIST = [101]
DURATION_STEPS = 100
WARMUP = 2
RUNS = 5
DT_MS = 0.1
DEVICE_MEMORY_LIMIT_BYTES = 96 * 1024**3
RUNS_DIR = Path(__file__).with_name("runs")
CASE_LIST = [{
    "BACKEND_LIST": ["hybrid"],
    "SCALE_LIST": [100,400,800,1200,1600,2000],
    "DENSITY_MAT_LIST": [1e-6,1e-5,1e-4,1e-3,1e-2],
    "DENSITY_VEC_LIST": [1,3,10,30,100,],# 3000,10000  # Firing rates in Hz, retained from the original grid.
    "GINI_LIST": [0],
    "K_LIST": [1,2,4,8,16,32],  # K delay segments: each edge independently draws an integer in [0, K).
}]

_IDENTITY = ("backend", "scale", "density_mat", "density_vec", "target_gini", "k", "duration_steps")
_FIELDS = [
    "backend", "seed", "scale", "n_neuron", "nnz", "value_dtype", "density_mat", "density_vec",
    "target_gini", "actual_gini", "k", "segment_count", "buffer_slots", "duration_steps",
    "timing_scope", "warmup_runs", "timed_runs", "warmup_total_ms", "timed_total_ms", "timed_mean_ms",
    "timed_mean_step_ms", "timed_std_ms", "timed_min_ms", "timed_max_ms", "spmv_calls_per_run", "error",
]


def spike_density_from_rate_hz(firing_rate_hz: float, dt_ms: float = DT_MS) -> float:
    """Convert a firing rate to the Bernoulli event probability.

    Parameters
    ----------
    firing_rate_hz : float
        Nonnegative firing rate in Hz.
    dt_ms : float, optional
        Positive duration of one simulation step in milliseconds.

    Returns
    -------
    float
        Probability in [0, 1].

    Raises
    ------
    ValueError
        If a parameter is nonfinite or implies an invalid probability.
    """
    if not math.isfinite(firing_rate_hz) or firing_rate_hz < 0 or not math.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError("require finite nonnegative firing rate and positive dt")
    density = firing_rate_hz * dt_ms / 1000
    if density > 1:
        raise ValueError("firing rate and dt imply probability above one")
    return density


def _validate_k(k: int) -> None:
    if type(k) is not int or not 1 <= k <= 2**31 - 1:
        raise ValueError("k must be a positive int32 delay segment count")


def generate_case_inputs(
    *, n: int, degree: int, event_density: float, duration_steps: int,
    k: int, seed: int, target_gini: float = 0.0,
) -> tuple[DelayInputs, float]:
    """Generate one CSR, independent edge delays and a full event matrix.

    Parameters
    ----------
    n, degree : int
        Square matrix size and mean row length, with 0 <= degree <= n.
    event_density : float
        Bernoulli activity probability in [0, 1].
    duration_steps : int
        Number T of event columns, independent of the number of delay values.
    k : int
        Positive int32 delay segment count. Each edge independently draws an
        integer uniformly from [0, k), equivalent to flooring a uniform draw
        from that interval. Retain all k segments, including empty ones.
    seed : int
        Seed shared by both operators; separate random streams generate columns,
        weights, delays and events so changing delays preserves the event data.
    target_gini : float, optional
        Desired row-length inequality; zero gives regular rows.

    Returns
    -------
    inputs : DelayInputs
        Prepared inputs for both native operators.
    actual_gini : float
        Realized row-length Gini coefficient.

    Raises
    ------
    ValueError
        If dimensions, probabilities, delays or the requested Gini are invalid.
    """
    if type(n) is not int or not 0 <= n <= 2**31 - 1 or type(degree) is not int or not 0 <= degree <= n:
        raise ValueError("n and degree must be nonnegative int32 with degree <= n")
    if type(duration_steps) is not int or not 1 <= duration_steps <= 2**31 - 1:
        raise ValueError("duration_steps must be positive int32")
    if not math.isfinite(event_density) or not 0 <= event_density <= 1:
        raise ValueError("event_density must be in [0, 1]")
    _validate_k(k)
    if target_gini == 0:
        lengths, actual_gini = np.full(n, degree, np.int32), 0.0
    else:
        lengths, actual_gini = make_gini_row_lengths(n, degree, target_gini, seed=seed)
    ptr64 = np.concatenate((np.zeros(1, np.int64), np.cumsum(lengths, dtype=np.int64)))
    nnz = int(ptr64[-1])
    ptr = ptr64.astype(np.int32) if nnz <= 2**31 - 1 else ptr64
    columns_rng, weights_rng, delay_rng, event_rng = (
        np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(4)
    )
    columns = columns_rng.integers(0, max(n, 1), size=nnz, dtype=np.int32)
    weights = weights_rng.random(nnz, dtype=np.float32) * np.float32(4) - np.float32(2)
    edge_delays = delay_rng.integers(0, k, size=nnz, dtype=np.int32)
    # The transpose is a view: logical [N,T], physical contiguous event columns.
    events = (event_rng.random((duration_steps, n), dtype=np.float32) < event_density).T
    return prepare_delay_inputs(weights, columns, ptr, edge_delays, events,
                                columns=n, delay_steps=tuple(range(k))), actual_gini


def _estimated_device_bytes(n: int, nnz: int, steps: int, k: int, *, include_split: bool = True) -> int:
    edge_bytes = 4 if nnz <= 2**31 - 1 else 8
    slots = k
    ring_bytes = 4 if slots * n <= 2**31 - 1 else 8
    capacity = min(nnz, nnz // TASK_NNZ + n * k)
    original = nnz * 8 + (n + 1 + n * k) * edge_bytes + k * ring_bytes + n * 4
    workspace = capacity * (edge_bytes * 2 + ring_bytes) + n * (steps + slots + 2) * 4
    split = (k * (n + 1) + k + 1 + nnz) * edge_bytes + nnz * 4
    return original + workspace + (split if include_split else 0)


def benchmark_case(
    *, scale: float, density_mat: float, firing_rate_hz: float, target_gini: float,
    k: int, seed: int, duration_steps: int = DURATION_STEPS,
    warmup: int = WARMUP, runs: int = RUNS,
    backend_list: Sequence[str] = ("hybrid", "cusparse"),
) -> list[dict[str, Any]]:
    """Initialize a case once and benchmark selected operators with its inputs.

    Parameters
    ----------
    scale : float
        Number of neurons divided by 1000.
    density_mat : float
        Mean row degree divided by the neuron count.
    firing_rate_hz, target_gini : float
        Activity and row-length distribution parameters.
    k : int
        Positive int32 number of delay segments, with edge delays in [0, k).
    seed : int
        Case random seed.
    duration_steps, warmup, runs : int, optional
        Event columns and repeat counts.
    backend_list : sequence of str, optional
        Ordered subset of ``"hybrid"`` and ``"cusparse"`` to execute.

    Returns
    -------
    list of dict
        One CSV-ready result per selected backend, including errors.

    Raises
    ------
    ValueError
        If ``backend_list`` is empty, repeated, or contains an unknown backend.
    """
    selected = tuple(backend_list)
    if not selected or len(selected) != len(set(selected)) or any(
        name not in ("hybrid", "cusparse") for name in selected
    ):
        raise ValueError("BACKEND_LIST must be a nonempty unique subset of hybrid and cusparse")
    rows: list[dict[str, Any]] = []
    base: dict[str, Any] = dict(seed=seed, scale=scale, density_mat=density_mat, target_gini=target_gini,
                                k=k, duration_steps=duration_steps,
                                value_dtype="f32", timing_scope="projection_only", warmup_runs=warmup, timed_runs=runs)
    try:
        if not math.isfinite(scale) or scale <= 0 or not math.isfinite(density_mat) or not 0 <= density_mat <= 1:
            raise ValueError("scale must be positive and density_mat in [0, 1]")
        n = int(scale * 1000)
        degree = 0 if density_mat == 0 else max(1, int(round(n * density_mat)))
        _validate_k(k)
        density = spike_density_from_rate_hz(firing_rate_hz)
        base.update(n_neuron=n, nnz=n * degree, density_vec=density,
                    segment_count=k, buffer_slots=k)
        if _estimated_device_bytes(n, n * degree, duration_steps, k,
                                   include_split="cusparse" in selected) > DEVICE_MEMORY_LIMIT_BYTES:
            raise ValueError("prepared inputs, split and workspaces exceed the configured device memory limit")
        inputs, actual_gini = generate_case_inputs(n=n, degree=degree, event_density=density,
            duration_steps=duration_steps, k=k, seed=seed, target_gini=target_gini)
        base["actual_gini"] = actual_gini
        with NativeCase(inputs) as case:
            runs_by_backend = {"hybrid": case.run_hybrid, "cusparse": case.run_cusparse}
            for name in selected:
                row = dict(base, backend=name, error="")
                try:
                    result = runs_by_backend[name](warmup=warmup, runs=runs)
                    row.update(result.timing, timed_mean_step_ms=result.mean_step_ms)
                except Exception as exception:
                    row["error"] = f"{type(exception).__name__}: {exception}"
                rows.append(row)
    except Exception as exception:
        rows = [dict(base, backend=name, error=f"{type(exception).__name__}: {exception}")
                for name in selected]
    return rows


def _write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(field, "") for field in _IDENTITY), []).append(row)
    fields = [*_IDENTITY, "timed_mean_step_ms_mean", "timed_mean_step_ms_std", "success_count", "failure_count"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for key, samples in groups.items():
            times = [float(sample["timed_mean_step_ms"]) for sample in samples if not sample["error"]]
            writer.writerow(dict(zip(_IDENTITY, key),
                timed_mean_step_ms_mean=statistics.fmean(times) if times else "",
                timed_mean_step_ms_std=statistics.stdev(times) if len(times) > 1 else "",
                success_count=len(times), failure_count=len(samples) - len(times)))


def run_all_cases(
    case_list: list[dict[str, Any]] = CASE_LIST, *, run_directory: str | Path | None = None,
    duration_steps: int = DURATION_STEPS, warmup: int = WARMUP, runs: int = RUNS,
) -> list[dict[str, Any]]:
    """Run configured case grids and write per-seed CSVs and summaries.

    Parameters
    ----------
    case_list : list of dict, optional
        Scale, matrix density, firing rate, Gini, K_LIST delay-count, and
        BACKEND_LIST selection grids.
    run_directory : str, pathlib.Path or None, optional
        Output directory; defaults to a timestamped directory under runs.
    duration_steps, warmup, runs : int, optional
        Number of event columns and warmup/measured repetitions.

    Returns
    -------
    list of dict
        All results, including failed cases.
    """
    root = Path(run_directory) if run_directory is not None else RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for index, grid in enumerate(case_list, 1):
        directory = root / f"case_{index:03d}"
        directory.mkdir(exist_ok=False)
        group_rows: list[dict[str, Any]] = []
        for repeat, seed in enumerate(RNG_SEED_LIST, 1):
            with (directory / f"case_{repeat}.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=_FIELDS)
                writer.writeheader()
                combinations = product(*(grid[key] for key in (
                    "SCALE_LIST", "DENSITY_MAT_LIST", "DENSITY_VEC_LIST", "GINI_LIST", "K_LIST")))
                for scale, density, rate, gini, k in combinations:
                    result = benchmark_case(scale=scale, density_mat=density, firing_rate_hz=rate,
                        target_gini=gini, k=k, seed=seed, duration_steps=duration_steps,
                        warmup=warmup, runs=runs,
                        backend_list=grid.get("BACKEND_LIST", ("hybrid", "cusparse")))
                    writer.writerows(result)
                    file.flush()
                    group_rows.extend(result)
                    print(result, flush=True)
        _write_summary(group_rows, directory / "summary.csv")
        all_rows.extend(group_rows)
    return all_rows


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default=GPU_ID)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=DURATION_STEPS)
    parser.add_argument("--smoke", action="store_true", help="run a small case on both operators")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cases = [{"SCALE_LIST": [0.256], "DENSITY_MAT_LIST": [0.15], "DENSITY_VEC_LIST": [1000],
              "GINI_LIST": [0.0], "K_LIST": [3],
              "BACKEND_LIST": ["hybrid", "cusparse"]}] if args.smoke else CASE_LIST
    results = run_all_cases(cases, run_directory=args.output, duration_steps=args.steps)
    if any(row["error"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
