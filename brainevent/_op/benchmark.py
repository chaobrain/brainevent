# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import jax
import numpy as np

__all__ = [
    'BenchmarkConfig',
    'BenchmarkResult',
    'BenchmarkReport',
    'benchmark_function',
]


@dataclass
class BenchmarkConfig:
    """A single benchmark configuration for a primitive.

    Returned by ``def_benchmark_data`` functions as part of a list.

    Attributes:
        name: A short descriptive name for this configuration
            (e.g., ``"T,homo,bool"``).
        args: Positional arguments to pass to the primitive's call function.
        kwargs: Keyword arguments to pass to the primitive's call function.
    """
    name: str
    args: tuple
    kwargs: dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of benchmarking a single backend.

    Attributes:
        backend: The backend name (e.g., 'numba', 'pallas', 'warp').
        platform: The platform name (e.g., 'cpu', 'gpu', 'tpu').
        mean_time: Mean execution time in seconds.
        std_time: Standard deviation of execution time in seconds.
        min_time: Minimum execution time in seconds.
        max_time: Maximum execution time in seconds.
        n_runs: Number of timed runs.
        success: Whether the benchmark completed successfully.
        error: Error message if the benchmark failed.
    """
    backend: str
    platform: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    n_runs: int
    success: bool
    error: Optional[str] = None

    def __repr__(self) -> str:
        if self.success:
            return (
                f"BenchmarkResult(backend='{self.backend}', platform='{self.platform}', "
                f"mean={self.mean_time * 1000:.3f}ms, std={self.std_time * 1000:.3f}ms)"
            )
        else:
            return (
                f"BenchmarkResult(backend='{self.backend}', platform='{self.platform}', "
                f"success=False, error='{self.error}')"
            )


@dataclass
class BenchmarkReport:
    """Report containing benchmark results for multiple backends.

    Attributes:
        primitive_name: Name of the primitive being benchmarked.
        platform: The platform benchmarked on.
        results: List of BenchmarkResult for each backend.
        mismatches: List of mismatch descriptions when compare_results=True.
    """
    primitive_name: str
    platform: str
    results: List[BenchmarkResult] = field(default_factory=list)
    mismatches: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate a human-readable summary of the benchmark results."""
        lines = [
            f"Benchmark Report: {self.primitive_name}",
            f"Platform: {self.platform}",
            "-" * 70,
        ]

        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        if successful:
            lines.append("Successful backends:")
            for r in sorted(successful, key=lambda x: x.mean_time):
                lines.append(
                    f"  {r.backend:15} | mean: {r.mean_time * 1000:8.3f}ms | "
                    f"std: {r.std_time * 1000:8.3f}ms | "
                    f"min: {r.min_time * 1000:8.3f}ms | "
                    f"max: {r.max_time * 1000:8.3f}ms"
                )

        if failed:
            lines.append("")
            lines.append("Failed backends:")
            for r in failed:
                error_msg = r.error[:50] + "..." if len(r.error) > 50 else r.error
                lines.append(f"  {r.backend:15} | error: {error_msg}")

        if self.mismatches:
            lines.append("")
            lines.append("Result mismatches:")
            for m in self.mismatches:
                lines.append(f"  {m}")
        elif len(successful) > 1:
            lines.append("")
            lines.append("All backend outputs match.")

        return "\n".join(lines)

    def fastest(self) -> Optional[BenchmarkResult]:
        """Return the fastest successful backend result."""
        successful = [r for r in self.results if r.success]
        if not successful:
            return None
        return min(successful, key=lambda r: r.mean_time)

    def to_dict(self) -> dict:
        """Convert the report to a dictionary."""
        return {
            'primitive_name': self.primitive_name,
            'platform': self.platform,
            'results': [
                {
                    'backend': r.backend,
                    'platform': r.platform,
                    'mean_time': r.mean_time,
                    'std_time': r.std_time,
                    'min_time': r.min_time,
                    'max_time': r.max_time,
                    'n_runs': r.n_runs,
                    'success': r.success,
                    'error': r.error,
                }
                for r in self.results
            ],
            'mismatches': self.mismatches,
        }

    def __repr__(self) -> str:
        n_success = sum(1 for r in self.results if r.success)
        n_total = len(self.results)
        fastest = self.fastest()
        fastest_str = f", fastest='{fastest.backend}'" if fastest else ""
        return (
            f"BenchmarkReport(primitive='{self.primitive_name}', platform='{self.platform}', "
            f"backends={n_success}/{n_total}{fastest_str})"
        )


def benchmark_function(
    fn,
    n_warmup: int,
    n_runs: int,
    batch_mode: bool = False,
) -> Tuple[float, float, float, float, Any]:
    """Benchmark a function and return timing statistics.

    Parameters
    ----------
    fn : callable
        A callable that takes no arguments and returns the result.
    n_warmup : int
        Number of warmup runs (not timed).
    n_runs : int
        Number of timed runs.
    batch_mode : bool, optional
        If ``False`` (default), block after each function call and time
        each run individually (measures per-call latency).  If ``True``,
        run all *n_runs* calls first, then block once at the end and
        measure total time (measures throughput, useful for async
        GPU/TPU execution).

    Returns
    -------
    tuple of (float, float, float, float, Any)
        ``(mean_time, std_time, min_time, max_time, output)`` where
        times are in seconds.
    """
    # Warmup runs
    output = None
    for _ in range(n_warmup):
        output = fn()
    jax.block_until_ready(output)

    if batch_mode:
        # Batch mode: run all n_runs, then block once at the end
        # This measures throughput and is useful for async GPU/TPU execution
        start = time.perf_counter()
        for _ in range(n_runs):
            output = fn()
        jax.block_until_ready(output)
        end = time.perf_counter()

        total_time = end - start
        mean_time = total_time / n_runs
        # In batch mode, we only have total time, so std/min/max are based on mean
        return mean_time, 0.0, mean_time, mean_time, output

    else:
        # Per-call mode: block after each call and time individually
        # This measures per-call latency
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            output = fn()
            jax.block_until_ready(output)
            end = time.perf_counter()
            times.append(end - start)

        times = np.array(times)
        return float(np.mean(times)), float(np.std(times)), float(np.min(times)), float(np.max(times)), output
