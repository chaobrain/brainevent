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

"""Comprehensive examples for XLACustomKernel.benchmark() using binary_csrmv_p.

This file demonstrates every feature of the redesigned benchmark API:

  1.  Basic run — call .benchmark() and print the table
  2.  Sorting and grouping
  3.  Baseline comparison with compare_by
  4.  Controlling timing precision with n_batch_per_run
  5.  Accessing raw records programmatically
  6.  Saving and loading results (JSON / CSV / pickle)
  7.  Plotting with .plot()
  8.  Registering custom BenchmarkConfig (with data_kwargs)
  9.  Raising BenchmarkDataFnNotProvidedError when data fn is missing
  10. Running the CLI equivalent in-process

Run from the project root:
    python examples/benchmark_example.py
"""

import os
import sys
import tempfile

# Allow running from the project root without installing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse as sp

import brainevent
from brainevent import binary_csrmv_p
from brainevent._error import BenchmarkDataFnNotProvidedError
from brainevent._op.benchmark import BenchmarkConfig, BenchmarkRecord, BenchmarkResult
from brainevent._op.main import XLACustomKernel

# Auto-detect the platform from the first available JAX device
PLATFORM = jax.devices()[0].platform  # 'cpu', 'gpu', or 'tpu'
N_WARMUP = 2
N_RUNS = 5


# ---------------------------------------------------------------------------
# Helper: build a random CSR matrix
# ---------------------------------------------------------------------------

def make_csr(n_pre: int, n_post: int, prob: float, seed: int = 0):
    """Return (weights_homo, weights_hetero, indices, indptr, nnz)."""
    rng = np.random.default_rng(seed)
    n_conn = max(1, int(n_post * prob))
    indptr = np.arange(n_pre + 1, dtype=np.int32) * n_conn
    indices = rng.integers(0, n_post, size=n_pre * n_conn, dtype=np.int32)
    nnz = n_pre * n_conn
    weights_homo = jnp.ones(1, dtype=jnp.float32)
    weights_hetero = jnp.ones(nnz, dtype=jnp.float32)
    return weights_homo, weights_hetero, jnp.asarray(indices), jnp.asarray(indptr), nnz


# ===========================================================================
# Example 1 — Basic benchmark run
# ===========================================================================

def example_1_basic():
    """Call .benchmark() and print the result table."""
    print("=" * 70)
    print("Example 1 — Basic benchmark run")
    print("=" * 70)

    result = binary_csrmv_p.benchmark(
        platform=PLATFORM,
        n_warmup=N_WARMUP,
        n_runs=N_RUNS,
    )

    # __repr__ already returns a formatted table — just print the object
    print(result)
    print()


# ===========================================================================
# Example 2 — Sort and group the output table
# ===========================================================================

def example_2_sort_and_group():
    """Demonstrate sort_by and group_by parameters of .print()."""
    print("=" * 70)
    print("Example 2 — Sorting and grouping")
    print("=" * 70)

    result = binary_csrmv_p.benchmark(
        platform=PLATFORM,
        n_warmup=N_WARMUP,
        n_runs=N_RUNS,
    )

    print("--- sorted by mean_ms (fastest first) ---")
    result.print(sort_by='mean_ms')
    print()

    print("--- grouped by label, best backend per label marked with * ---")
    result.print(group_by='label', highlight_best=True)
    print()

    print("--- grouped by backend ---")
    result.print(group_by='backend')
    print()


# ===========================================================================
# Example 3 — Baseline comparison with compare_by
# ===========================================================================

def example_3_compare_by():
    """Show speedup relative to a chosen baseline configuration."""
    print("=" * 70)
    print("Example 3 — Baseline comparison (compare_by)")
    print("=" * 70)

    result = binary_csrmv_p.benchmark(
        platform=PLATFORM,
        n_warmup=N_WARMUP,
        n_runs=N_RUNS,
    )

    # Select the first successful numba record as the baseline
    # compare_by is evaluated as a Python expression against each row dict
    print("--- speedup relative to numba / NT,homo,bool baseline ---")
    result.print(
        sort_by='mean_ms',
        compare_by="backend == 'numba' and label == 'NT,homo,bool'",
    )
    print()

    # Callable form of compare_by
    print("--- speedup relative to first numba record (callable) ---")
    result.print(
        sort_by='mean_ms',
        compare_by=lambda row: row.get('backend') == 'numba',
    )
    print()


# ===========================================================================
# Example 4 — Controlling timing precision with n_batch_per_run
# ===========================================================================

def example_4_n_batch_per_run():
    """Show how n_batch_per_run affects throughput measurement."""
    print("=" * 70)
    print("Example 4 — n_batch_per_run")
    print("=" * 70)

    print("--- n_batch_per_run=1 (per-call latency, default) ---")
    r1 = binary_csrmv_p.benchmark(
        platform=PLATFORM,
        n_warmup=N_WARMUP,
        n_runs=N_RUNS,
        n_batch_per_run=1,
    )
    r1.print(sort_by='label')
    print()

    print("--- n_batch_per_run=10 (amortised; reduces blocking overhead) ---")
    r10 = binary_csrmv_p.benchmark(
        platform=PLATFORM,
        n_warmup=N_WARMUP,
        n_runs=N_RUNS,
        n_batch_per_run=10,
    )
    r10.print(sort_by='label')
    print()


# ===========================================================================
# Example 5 — Accessing raw records programmatically
# ===========================================================================

def example_5_records():
    """Iterate over BenchmarkRecord objects for custom post-processing."""
    print("=" * 70)
    print("Example 5 — Accessing raw records")
    print("=" * 70)

    result = binary_csrmv_p.benchmark(
        platform=PLATFORM,
        n_warmup=N_WARMUP,
        n_runs=N_RUNS,
    )

    print(f"Total records: {len(result.records)}")
    print()

    # Find the overall fastest
    fastest = result.fastest()
    if fastest:
        print(f"Overall fastest: backend={fastest.backend!r}, "
              f"label={fastest.label!r}, mean_ms={fastest.mean_ms:.4f}")
    print()

    # Fastest per label
    labels = list(dict.fromkeys(r.label for r in result.records))
    print("Fastest backend per config:")
    for label in labels:
        rec = result.fastest(label=label)
        if rec:
            print(f"  [{label}]  {rec.backend:10s}  {rec.mean_ms:.4f} ms")
    print()

    # Custom aggregation — mean across labels per backend
    from collections import defaultdict
    backend_times = defaultdict(list)
    for rec in result.records:
        if rec.success:
            backend_times[rec.backend].append(rec.mean_ms)

    print("Average mean_ms across all configs per backend:")
    for be, times in sorted(backend_times.items()):
        avg = sum(times) / len(times)
        print(f"  {be:10s}  avg={avg:.4f} ms  over {len(times)} configs")
    print()


# ===========================================================================
# Example 6 — Save and load results
# ===========================================================================

def example_6_save_load():
    """Demonstrate persistence: save to JSON / CSV / pkl, then reload."""
    print("=" * 70)
    print("Example 6 — Save and load results")
    print("=" * 70)

    result = binary_csrmv_p.benchmark(
        platform=PLATFORM,
        n_warmup=N_WARMUP,
        n_runs=N_RUNS,
    )

    with tempfile.TemporaryDirectory() as tmpdir:

        # --- JSON (default, human-readable) ---
        json_path = os.path.join(tmpdir, 'bench.json')
        result.save(json_path, format='json')
        loaded_json = BenchmarkResult.load(json_path)
        print(f"JSON  round-trip: {len(loaded_json.records)} records, "
              f"primitive='{loaded_json.primitive_name}'")

        # --- CSV (flat table, easy to open in a spreadsheet) ---
        csv_path = os.path.join(tmpdir, 'bench.csv')
        result.save(csv_path, format='csv')
        loaded_csv = BenchmarkResult.load(csv_path)
        print(f"CSV   round-trip: {len(loaded_csv.records)} records")

        # --- Pickle (lossless, preserves kernel_kwargs / data_kwargs dicts) ---
        pkl_path = os.path.join(tmpdir, 'bench.pkl')
        result.save(pkl_path, format='pkl')
        loaded_pkl = BenchmarkResult.load(pkl_path)
        print(f"PKL   round-trip: {len(loaded_pkl.records)} records, "
              f"primitive='{loaded_pkl.primitive_name}'")

        # to_dict() for custom serialization / JSON embedding
        d = result.to_dict()
        print(f"to_dict() keys: {list(d.keys())}")
        print(f"  records[0] keys: {list(d['records'][0].keys())}")

    print()


# ===========================================================================
# Example 7 — Plotting (skipped if matplotlib is absent)
# ===========================================================================

def example_7_plot():
    """Produce a bar chart of mean_ms per label, coloured by backend."""
    print("=" * 70)
    print("Example 7 — Plotting")
    print("=" * 70)

    try:
        import matplotlib  # noqa: F401
        import pandas  # noqa: F401
    except ImportError as e:
        print(f"Skipping plot example — {e.name} not installed.")
        print()
        return

    result = binary_csrmv_p.benchmark(
        platform=PLATFORM,
        n_warmup=N_WARMUP,
        n_runs=N_RUNS,
    )

    tmpdir = os.path.abspath(os.path.dirname(__file__))
    out_path = os.path.join(tmpdir, 'bench_bar.png')

    # Bar chart: x=label, y=mean_ms, coloured by backend
    fig = result.plot(
        x='label',
        y='mean_ms',
        hue='backend',
        kind='bar',
        show=False,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    print(f"Bar chart saved to: {out_path}")

    # Line chart: x=label, y=min_ms
    out_line = os.path.join(tmpdir, 'bench_line.png')
    fig2 = result.plot(
        x='label',
        y='min_ms',
        hue='backend',
        kind='line',
        show=False,
    )
    fig2.tight_layout()
    fig2.savefig(out_line, dpi=100)
    print(f"Line chart saved to: {out_line}")

    print()


# ===========================================================================
# Example 8 — Custom BenchmarkConfig with data_kwargs
# ===========================================================================

def example_8_custom_benchmark_data():
    """Register a richer data function that populates data_kwargs."""
    print("=" * 70)
    print("Example 8 — Custom BenchmarkConfig with data_kwargs")
    print("=" * 70)

    def rich_benchmark_data(*, platform):
        """Generate configs that expose nnz and density in data_kwargs."""
        configs = []
        shapes = [(1000, 1000), (5000, 5000)]
        densities = [0.01, 0.05]

        for (n_pre, n_post) in shapes:
            for density in densities:
                w_homo, w_hetero, indices, indptr, nnz = make_csr(n_pre, n_post, density)
                vector = jnp.ones(n_post, dtype=jnp.float32)

                for homo, weights in [(True, w_homo), (False, w_hetero)]:
                    label = f"{n_pre}x{n_post},d={density},{'homo' if homo else 'hetero'}"
                    configs.append(
                        BenchmarkConfig(
                            name=label,
                            args=(weights, indices, indptr, vector),
                            kernel_kwargs={
                                'shape': (n_pre, n_post),
                                'transpose': False,
                            },
                            data_kwargs={
                                'nnz': nnz,
                                'density': density,
                                'n_pre': n_pre,
                                'n_post': n_post,
                                'homo': homo,
                            },
                        )
                    )
        return configs

    # Temporarily override the benchmark data function
    original_fn = binary_csrmv_p._benchmark_data_fn
    binary_csrmv_p.def_benchmark_data(rich_benchmark_data)

    result = binary_csrmv_p.benchmark(
        platform=PLATFORM,
        n_warmup=N_WARMUP,
        n_runs=N_RUNS,
    )

    # Restore original so other examples are not affected
    binary_csrmv_p.def_benchmark_data(original_fn)

    # data_kwargs columns (nnz, density, n_pre, n_post, homo) appear in the table
    print("Table includes data_kwargs columns (nnz, density, ...):")
    result.print(group_by='label')
    print()

    # Programmatic access to data_kwargs
    print("data_kwargs of first record:", result.records[0].data_kwargs)
    print()


# ===========================================================================
# Example 9 — BenchmarkDataFnNotProvidedError
# ===========================================================================

def example_9_missing_data_fn():
    """Show the clear error raised when no data function is registered."""
    print("=" * 70)
    print("Example 9 — BenchmarkDataFnNotProvidedError")
    print("=" * 70)

    prim = XLACustomKernel('_demo_no_data_fn')

    # Register a minimal CPU kernel so the platform check passes
    def _dummy_kg(**kw):
        def _kernel(*args):
            return (jnp.zeros(1),)
        return _kernel

    prim.def_numba_kernel(_dummy_kg)
    prim.def_call(lambda *a, **kw: jnp.zeros(1))

    # No def_benchmark_data() call — should raise immediately
    try:
        prim.benchmark(platform='cpu')
    except BenchmarkDataFnNotProvidedError as exc:
        print(f"Caught expected error: {type(exc).__name__}")
        print(f"  Message: {exc}")
    print()


# ===========================================================================
# Example 10 — BenchmarkResult built from scratch (offline analysis)
# ===========================================================================

def example_10_offline_analysis():
    """Build a BenchmarkResult from hand-crafted records for offline analysis."""
    print("=" * 70)
    print("Example 10 — Offline analysis from hand-crafted BenchmarkResult")
    print("=" * 70)

    # Simulate results from two platforms merged for cross-platform comparison
    records = [
        BenchmarkRecord(
            platform='cpu', backend='numba', label='small',
            mean_ms=3.2, std_ms=0.1, min_ms=3.0, throughput=None,
            success=True, error=None,
            kernel_kwargs={'shape': (1000, 1000)}, data_kwargs={'nnz': 100_000},
        ),
        BenchmarkRecord(
            platform='gpu', backend='pallas', label='small',
            mean_ms=0.4, std_ms=0.01, min_ms=0.38, throughput=None,
            success=True, error=None,
            kernel_kwargs={'shape': (1000, 1000)}, data_kwargs={'nnz': 100_000},
        ),
        BenchmarkRecord(
            platform='gpu', backend='warp', label='small',
            mean_ms=0.6, std_ms=0.02, min_ms=0.55, throughput=None,
            success=True, error=None,
            kernel_kwargs={'shape': (1000, 1000)}, data_kwargs={'nnz': 100_000},
        ),
        BenchmarkRecord(
            platform='cpu', backend='numba', label='large',
            mean_ms=45.0, std_ms=1.2, min_ms=43.5, throughput=None,
            success=True, error=None,
            kernel_kwargs={'shape': (10000, 10000)}, data_kwargs={'nnz': 10_000_000},
        ),
        BenchmarkRecord(
            platform='gpu', backend='pallas', label='large',
            mean_ms=1.8, std_ms=0.05, min_ms=1.75, throughput=None,
            success=True, error=None,
            kernel_kwargs={'shape': (10000, 10000)}, data_kwargs={'nnz': 10_000_000},
        ),
        BenchmarkRecord(
            platform='gpu', backend='warp', label='large',
            mean_ms=2.1, std_ms=0.06, min_ms=2.0, throughput=None,
            success=True, error=None,
            kernel_kwargs={'shape': (10000, 10000)}, data_kwargs={'nnz': 10_000_000},
        ),
    ]

    combined = BenchmarkResult(records, primitive_name='binary_csrmv (cross-platform)')

    print("--- All records ---")
    combined.print(group_by='label')
    print()

    print("--- Sorted by mean_ms, speedup vs CPU numba baseline ---")
    combined.print(
        sort_by='mean_ms',
        compare_by="backend == 'numba' and platform == 'cpu'",
    )
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        p = os.path.join(tmpdir, 'cross_platform.json')
        combined.save(p)
        reloaded = BenchmarkResult.load(p)
        print(f"Saved and reloaded {len(reloaded.records)} records from JSON.")
    print()


# ===========================================================================
# Main
# ===========================================================================

def main():
    print()
    print("brainevent benchmark() API — comprehensive examples")
    print(f"Platform: {PLATFORM} | n_warmup={N_WARMUP} | n_runs={N_RUNS}")
    print(f"JAX devices: {jax.devices()}")
    print()

    example_1_basic()
    example_2_sort_and_group()
    example_3_compare_by()
    example_4_n_batch_per_run()
    example_5_records()
    example_6_save_load()
    example_7_plot()
    example_8_custom_benchmark_data()
    example_9_missing_data_fn()
    example_10_offline_analysis()

    print("All examples completed.")


if __name__ == '__main__':
    main()
