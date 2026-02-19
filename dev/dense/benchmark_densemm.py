"""
Binary Dense Matrix-Matrix Multiplication Benchmark
=====================================================

Benchmarks all available GPU backends for ``binary_densemm`` (gather and scatter
modes) across problem sizes and spike densities.

This benchmark compares:
  - jax_raw:  standard ``jnp.matmul`` (cuBLAS GEMM, no event-driven skip)
  - pallas:   Pallas Triton kernel (event-driven, fori_loop-based)
  - tvmffi:   Custom CUDA kernel (event-driven, hand-tuned)

The CUDA kernel's advantage is largest when spike density is low (<10%)
and the matrix dimensions are moderate to large (>=1000).

Usage
-----
    python dev/dense/benchmark_densemm.py
    python dev/dense/benchmark_densemm.py --manual
    python dev/dense/benchmark_densemm.py --manual --n_warmup 20 --n_runs 200
"""

import argparse
import sys
import time
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import jax
import jax.numpy as jnp
import numpy as np

from brainevent import binary_densemm_p, BenchmarkConfig

# ---- Built-in benchmark data (uses the primitive's benchmark machinery) ----

# (m_or_k, k_or_m, n, density, transpose)
CONFIGS = [
    # Medium matrices
    (1000, 1000, 100, 0.001, False),
    (1000, 1000, 100, 0.01, False),
    (1000, 1000, 100, 0.10, False),
    (1000, 1000, 100, 0.001, True),
    (1000, 1000, 100, 0.01, True),
    (1000, 1000, 100, 0.10, True),
    # Large matrices
    (5000, 5000, 100, 0.001, False),
    (5000, 5000, 100, 0.01, False),
    (5000, 5000, 100, 0.10, False),
    (5000, 5000, 100, 0.001, True),
    (5000, 5000, 100, 0.01, True),
    (5000, 5000, 100, 0.10, True),
    # Very large
    (10000, 10000, 100, 0.001, False),
    (10000, 10000, 100, 0.01, False),
    (10000, 10000, 100, 0.10, False),
    (10000, 10000, 100, 0.001, True),
    (10000, 10000, 100, 0.01, True),
    (10000, 10000, 100, 0.10, True),
    # Wide output (many spike columns)
    (5000, 5000, 500, 0.01, False),
    (5000, 5000, 500, 0.10, False),
    (5000, 5000, 500, 0.01, True),
    (5000, 5000, 500, 0.10, True),
]


def _make_benchmark_data(*, platform):
    rng = np.random.default_rng(42)

    for dim0, dim1, n_cols, density, transpose in CONFIGS:
        dtype = jnp.float32
        if transpose:
            k, m = dim0, dim1
            weights = jnp.asarray(rng.standard_normal((k, m)), dtype=dtype)
            spikes = jnp.asarray(rng.random((k, n_cols)) < density, dtype=jnp.bool_)
            label_mode = "T"
        else:
            m, k = dim0, dim1
            weights = jnp.asarray(rng.standard_normal((m, k)), dtype=dtype)
            spikes = jnp.asarray(rng.random((k, n_cols)) < density, dtype=jnp.bool_)
            label_mode = "NT"

        nnz = int(spikes.sum())
        total = spikes.size
        name = f"{label_mode},f32,d={density:.1%},{dim0}x{dim1}x{n_cols},nnz={nnz}/{total}"

        yield BenchmarkConfig(
            name=name,
            args=(weights, spikes),
            kernel_kwargs={'transpose': transpose},
            data_kwargs={'dim0': dim0, 'dim1': dim1, 'n_cols': n_cols, 'density': density},
        )


# ---- Manual micro-benchmark ----

def _measure_dispatch_overhead(n_warmup=50, n_runs=500):
    """Measure dispatch overhead with a tiny (4x4) @ (4x2) operation."""
    gpu = jax.devices("gpu")[0]
    w_tiny = jax.device_put(jnp.ones((4, 4), jnp.float32), gpu).block_until_ready()
    s_tiny = jax.device_put(jnp.ones((4, 2), jnp.float32), gpu).block_until_ready()
    overheads = {}
    for backend in ['jax_raw', 'tvmffi']:
        fn = jax.jit(lambda w, s: binary_densemm_p.call(w, s, transpose=False, backend=backend))
        for _ in range(n_warmup):
            fn(w_tiny, s_tiny)[0].block_until_ready()
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            fn(w_tiny, s_tiny)[0].block_until_ready()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e6)
        overheads[backend] = {
            'mean': sum(times) / len(times),
            'min': min(times),
        }
    return overheads


def _manual_benchmark(n_warmup=20, n_runs=200):
    """
    Time each backend by calling the primitive inside jax.jit, which amortizes
    Python dispatch overhead across many calls.
    """
    rng = np.random.default_rng(42)
    gpu = jax.devices("gpu")[0]

    print("\n" + "=" * 120)
    print("Measuring dispatch overhead (4x4 no-op) ...")
    overheads = _measure_dispatch_overhead()
    for b, v in overheads.items():
        print(f"  {b:<10}  mean={v['mean']:.1f} us   min={v['min']:.1f} us")
    print()

    configs = [
        # (m_or_k, k_or_m, n_cols, density, transpose, dtype)
        # --- Gather (transpose=False) ---
        # Small n: few output columns, event-driven advantage regime
        (5000, 5000, 10, 0.001, False, jnp.float32),
        (5000, 5000, 10, 0.01, False, jnp.float32),
        (5000, 5000, 10, 0.10, False, jnp.float32),
        (10000, 10000, 10, 0.001, False, jnp.float32),
        (10000, 10000, 10, 0.01, False, jnp.float32),
        # Moderate n=100
        (5000, 5000, 100, 0.001, False, jnp.float32),
        (5000, 5000, 100, 0.01, False, jnp.float32),
        (5000, 5000, 100, 0.10, False, jnp.float32),
        (10000, 10000, 100, 0.001, False, jnp.float32),
        (10000, 10000, 100, 0.01, False, jnp.float32),
        (10000, 10000, 100, 0.10, False, jnp.float32),
        # Very large: compute-bound for cuBLAS
        (20000, 20000, 100, 0.001, False, jnp.float32),
        (20000, 20000, 100, 0.01, False, jnp.float32),
        # --- Scatter (transpose=True) ---
        (5000, 5000, 100, 0.01, True, jnp.float32),
        (5000, 5000, 100, 0.10, True, jnp.float32),
        (10000, 10000, 100, 0.01, True, jnp.float32),
        (10000, 10000, 100, 0.10, True, jnp.float32),
        # Small n scatter
        (5000, 5000, 10, 0.01, True, jnp.float32),
        (10000, 10000, 10, 0.01, True, jnp.float32),
        # --- Multi-dtype ---
        (5000, 5000, 100, 0.01, False, jnp.float16),
        (5000, 5000, 100, 0.01, False, jnp.bfloat16),
    ]

    backends = ['jax_raw', 'tvmffi']

    print("=" * 120)
    print("Manual micro-benchmark (jit-compiled, amortized dispatch)")
    print("=" * 120)
    print(f"{'config':<65} {'backend':<10} {'mean_us':>10} {'min_us':>10}"
          f" {'kern_us':>10} {'speedup':>8}")
    print("-" * 120)

    for dim0, dim1, n_cols, density, transpose, dtype in configs:
        dtype_label = {
            jnp.float32: 'f32', jnp.float16: 'f16',
            jnp.bfloat16: 'bf16', jnp.float64: 'f64',
        }.get(dtype, 'f32')

        if transpose:
            k, m = dim0, dim1
            weights = jax.device_put(
                jnp.asarray(rng.standard_normal((k, m)), dtype=dtype), gpu
            ).block_until_ready()
            spikes = jax.device_put(
                jnp.asarray(rng.random((k, n_cols)) < density, dtype=jnp.bool_), gpu
            ).block_until_ready()
            label_mode = "T"
        else:
            m, k = dim0, dim1
            weights = jax.device_put(
                jnp.asarray(rng.standard_normal((m, k)), dtype=dtype), gpu
            ).block_until_ready()
            spikes = jax.device_put(
                jnp.asarray(rng.random((k, n_cols)) < density, dtype=jnp.bool_), gpu
            ).block_until_ready()
            label_mode = "NT"

        nnz = int(spikes.sum())
        label = f"{label_mode},{dtype_label},d={density:.1%},{dim0}x{dim1}x{n_cols},nnz={nnz}"

        kern_times = {}
        for backend in backends:
            fn = jax.jit(
                lambda w, s: binary_densemm_p.call(
                    w, s, transpose=transpose, backend=backend
                )
            )

            # Warmup
            for _ in range(n_warmup):
                _ = fn(weights, spikes)[0].block_until_ready()

            # Timed runs
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = fn(weights, spikes)[0].block_until_ready()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1e6)

            mean_us = sum(times) / len(times)
            min_us = min(times)
            disp = overheads.get(backend, overheads.get('jax_raw', {})).get('min', 0)
            kern_us = max(0.0, min_us - disp)
            kern_times[backend] = kern_us

            if backend == 'jax_raw':
                speedup_str = ""
            else:
                jax_kern = kern_times.get('jax_raw', 0)
                if kern_us > 0:
                    speedup_str = f"{jax_kern / kern_us:.2f}x"
                else:
                    speedup_str = "inf"

            print(f"{label:<65} {backend:<10} {mean_us:>10.1f} {min_us:>10.1f}"
                  f" {kern_us:>10.1f} {speedup_str:>8}")

        print()


def main():
    parser = argparse.ArgumentParser(description="binary_densemm backend benchmark")
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_runs", type=int, default=20)
    parser.add_argument("--manual", action="store_true",
                        help="Run manual micro-benchmark for precise kernel timing")
    args = parser.parse_args()

    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        print("ERROR: No GPU device found.")
        return

    print(f"binary_densemm benchmark  —  GPU: {gpu}")

    if args.manual:
        _manual_benchmark(n_warmup=args.n_warmup, n_runs=args.n_runs)
        return

    print(f"warmup={args.n_warmup}  runs={args.n_runs}")
    print()

    binary_densemm_p.def_benchmark_data(_make_benchmark_data)

    result = binary_densemm_p.benchmark(
        platform='gpu',
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        compare_results=True,
        verbose=True,
    )
    result.print(vary_by='backend', highlight_best=True, speedup_vs='jax_raw')


if __name__ == "__main__":
    main()
