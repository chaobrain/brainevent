"""
Sparse-Float Dense Matrix-Matrix Multiplication Benchmark
==========================================================

Benchmarks all available GPU backends for ``spfloat_densemm`` (NT and T modes)
across problem sizes and spike densities.

This benchmark compares:
  - jax_raw:  standard ``jnp.matmul`` (cuBLAS GEMM, no event-driven skip)
  - pallas:   Pallas Triton kernel (event-driven, fori_loop-based)
  - cuda_raw:   Custom CUDA kernel (event-driven, hand-tuned)

Usage
-----
    python dev/dense/benchmark_spfloatmm.py
    python dev/dense/benchmark_spfloatmm.py --manual
    python dev/dense/benchmark_spfloatmm.py --n_warmup 10 --n_runs 100
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

from brainevent import spfloat_densemm_p, BenchmarkConfig

# (m, k, n_cols, density, transpose)
CONFIGS = [
    # Medium
    (1000, 1000, 10, 0.01, False),
    (1000, 1000, 10, 0.10, False),
    (1000, 1000, 10, 0.01, True),
    (1000, 1000, 10, 0.10, True),
    # Large
    (5000, 5000, 10, 0.001, False),
    (5000, 5000, 10, 0.01, False),
    (5000, 5000, 10, 0.10, False),
    (5000, 5000, 10, 0.001, True),
    (5000, 5000, 10, 0.01, True),
    (5000, 5000, 10, 0.10, True),
    # Large with more columns
    (5000, 5000, 50, 0.01, False),
    (5000, 5000, 50, 0.10, False),
    (5000, 5000, 50, 0.01, True),
    (5000, 5000, 50, 0.10, True),
    # Very large
    (10000, 10000, 10, 0.001, False),
    (10000, 10000, 10, 0.01, False),
    (10000, 10000, 10, 0.10, False),
    (10000, 10000, 10, 0.001, True),
    (10000, 10000, 10, 0.01, True),
    (10000, 10000, 10, 0.10, True),
]


def _make_benchmark_data(*, platform):
    rng = np.random.default_rng(42)

    for m, k, n_cols, density, transpose in CONFIGS:
        dtype = jnp.float32
        if transpose:
            # spikes[m, k] @ weights[k, n_cols] -> out[m, n_cols]
            weights = jnp.asarray(rng.standard_normal((k, n_cols)), dtype=dtype)
            mask = rng.random((m, k)) < density
            vals = rng.standard_normal((m, k)).astype(np.float32)
            spikes = jnp.asarray(np.where(mask, vals, 0.0), dtype=dtype)
            label_mode = "T"
            nnz = int(np.sum(mask))
            name = f"{label_mode},f32,d={density:.1%},{m}x{k}x{n_cols},nnz={nnz}"
        else:
            # weights[m, k] @ spikes[k, n_cols] -> out[m, n_cols]
            weights = jnp.asarray(rng.standard_normal((m, k)), dtype=dtype)
            mask = rng.random((k, n_cols)) < density
            vals = rng.standard_normal((k, n_cols)).astype(np.float32)
            spikes = jnp.asarray(np.where(mask, vals, 0.0), dtype=dtype)
            label_mode = "NT"
            nnz = int(np.sum(mask))
            name = f"{label_mode},f32,d={density:.1%},{m}x{k}x{n_cols},nnz={nnz}"

        yield BenchmarkConfig(
            name=name,
            args=(weights, spikes),
            kernel_kwargs={'transpose': transpose},
            data_kwargs={'m': m, 'k': k, 'n_cols': n_cols, 'density': density},
        )


def _measure_dispatch_overhead(n_warmup=50, n_runs=500):
    """Measure JAX dispatch overhead using a tiny operation."""
    gpu = jax.devices("gpu")[0]
    w_tiny = jax.device_put(jnp.ones((4, 4), jnp.float32), gpu).block_until_ready()
    s_tiny = jax.device_put(jnp.zeros((4, 2), jnp.float32), gpu).block_until_ready()
    overheads = {}
    for backend in ['jax_raw', 'cuda_raw']:
        fn = jax.jit(lambda w, s: spfloat_densemm_p.call(w, s, transpose=False, backend=backend))
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
    """Time each backend via jax.jit, amortizing Python dispatch overhead."""
    rng = np.random.default_rng(42)
    gpu = jax.devices("gpu")[0]

    print("\n" + "=" * 110)
    print("Measuring dispatch overhead (4x4x2 no-op) ...")
    overheads = _measure_dispatch_overhead()
    for b, v in overheads.items():
        print(f"  {b:<10}  mean={v['mean']:.1f} us   min={v['min']:.1f} us")
    print()

    configs = [
        # (m, k, n_cols, density, transpose, dtype)
        # --- NT (weights[m,k] @ spikes[k,n]) ---
        # NT parity regime: n small, all densities
        (5000, 5000, 10, 0.001, False, jnp.float32),
        (5000, 5000, 10, 0.01, False, jnp.float32),
        (5000, 5000, 10, 0.10, False, jnp.float32),
        (5000, 5000, 50, 0.01, False, jnp.float32),
        (5000, 5000, 50, 0.10, False, jnp.float32),
        (10000, 10000, 10, 0.001, False, jnp.float32),
        (10000, 10000, 10, 0.01, False, jnp.float32),
        (10000, 10000, 10, 0.10, False, jnp.float32),
        # --- T (spikes[m,k] @ weights[k,n]) small n (SNN scatter, typical) ---
        (5000, 5000, 10, 0.001, True, jnp.float32),
        (5000, 5000, 10, 0.01, True, jnp.float32),
        (5000, 5000, 10, 0.10, True, jnp.float32),
        (5000, 5000, 50, 0.01, True, jnp.float32),
        (5000, 5000, 50, 0.10, True, jnp.float32),
        (10000, 10000, 10, 0.001, True, jnp.float32),
        (10000, 10000, 10, 0.01, True, jnp.float32),
        (10000, 10000, 10, 0.10, True, jnp.float32),
        # --- T large-n winning regime: small m (batch), large k, large n ---
        # cuBLAS is slow for tiny m; event-driven skip saves large weight reads
        (10, 5000, 5000, 0.001, True, jnp.float32),
        (10, 5000, 5000, 0.01, True, jnp.float32),
        (10, 10000, 10000, 0.001, True, jnp.float32),
        (10, 10000, 10000, 0.01, True, jnp.float32),
        (50, 5000, 5000, 0.001, True, jnp.float32),
        (50, 5000, 5000, 0.01, True, jnp.float32),
        (50, 10000, 10000, 0.001, True, jnp.float32),
    ]

    backends = ['jax_raw', 'cuda_raw']

    print("=" * 110)
    print("Manual micro-benchmark (jit-compiled, amortized dispatch)")
    print("=" * 110)
    print(f"{'config':<60} {'backend':<10} {'mean_us':>10} {'min_us':>10}"
          f" {'kern_us':>10} {'speedup':>8}")
    print("-" * 110)

    for m_val, k_val, n_val, density, transpose, dtype in configs:
        dtype_label = {jnp.float32: 'f32', jnp.float16: 'f16', jnp.bfloat16: 'bf16'}.get(dtype, 'f32')
        if transpose:
            weights = jax.device_put(
                jnp.asarray(rng.standard_normal((k_val, n_val)), dtype=dtype), gpu
            ).block_until_ready()
            mask = rng.random((m_val, k_val)) < density
            vals = rng.standard_normal((m_val, k_val)).astype(np.float32)
            spikes = jax.device_put(
                jnp.asarray(np.where(mask, vals, 0.0), dtype=dtype), gpu
            ).block_until_ready()
            label_mode = "T"
            nnz = int(np.sum(mask))
        else:
            weights = jax.device_put(
                jnp.asarray(rng.standard_normal((m_val, k_val)), dtype=dtype), gpu
            ).block_until_ready()
            mask = rng.random((k_val, n_val)) < density
            vals = rng.standard_normal((k_val, n_val)).astype(np.float32)
            spikes = jax.device_put(
                jnp.asarray(np.where(mask, vals, 0.0), dtype=dtype), gpu
            ).block_until_ready()
            label_mode = "NT"
            nnz = int(np.sum(mask))

        label = f"{label_mode},{dtype_label},d={density:.1%},{m_val}x{k_val}x{n_val},nnz={nnz}"

        kern_times = {}
        for backend in backends:
            fn = jax.jit(lambda w, s: spfloat_densemm_p.call(w, s, transpose=transpose, backend=backend))

            for _ in range(n_warmup):
                _ = fn(weights, spikes)[0].block_until_ready()

            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = fn(weights, spikes)[0].block_until_ready()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1e6)

            mean_us = sum(times) / len(times)
            min_us = min(times)
            disp = overheads[backend]['min']
            kern_us = max(0.0, min_us - disp)
            kern_times[backend] = kern_us

            if backend == 'jax_raw':
                speedup_str = ""
            else:
                jax_kern = kern_times.get('jax_raw', 0)
                if kern_us > 0:
                    speedup_str = f"{jax_kern / kern_us:.1f}x"
                else:
                    speedup_str = "inf"

            print(f"{label:<60} {backend:<10} {mean_us:>10.1f} {min_us:>10.1f}"
                  f" {kern_us:>10.1f} {speedup_str:>8}")

        print()


def main():
    parser = argparse.ArgumentParser(description="spfloat_densemm backend benchmark")
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

    print(f"spfloat_densemm benchmark  —  GPU: {gpu}")

    if args.manual:
        _manual_benchmark()
        return

    print(f"warmup={args.n_warmup}  runs={args.n_runs}")
    print()

    spfloat_densemm_p.def_benchmark_data(_make_benchmark_data)

    result = spfloat_densemm_p.benchmark(
        platform='gpu',
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        compare_results=True,
        verbose=True,
    )
    result.print(vary_by='backend', highlight_best=True, speedup_vs='jax_raw')


if __name__ == "__main__":
    main()
