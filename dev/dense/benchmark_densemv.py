"""
Binary Dense Matrix-Vector Multiplication Benchmark
====================================================

Benchmarks all available GPU backends for ``binary_densemv`` (gather and scatter
modes) across problem sizes and spike densities.

This benchmark compares:
  - jax_raw:  standard ``jnp.matmul`` (cuBLAS GEMV, no event-driven skip)
  - pallas:   Pallas Triton kernel (event-driven, fori_loop-based)
  - tvmffi:   Custom CUDA kernel (event-driven, hand-tuned)

The CUDA kernel's advantage is largest when spike density is low (<10%)
and the matrix dimensions are moderate to large (>=1000).

Usage
-----
    python dev/dense/benchmark_densemv.py
    python dev/dense/benchmark_densemv.py --n_warmup 10 --n_runs 100
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

from brainevent import binary_densemv_p, BenchmarkConfig

# ---- Built-in benchmark data (uses the primitive's benchmark machinery) ----

# (dim0, dim1, density, transpose)
CONFIGS = [
    # Large matrices — where kernel compute becomes measurable
    (5000, 5000, 0.001, False),
    (5000, 5000, 0.01, False),
    (5000, 5000, 0.10, False),
    (5000, 5000, 0.001, True),
    (5000, 5000, 0.01, True),
    (5000, 5000, 0.10, True),
    # Very large — stress test
    (10000, 10000, 0.001, False),
    (10000, 10000, 0.01, False),
    (10000, 10000, 0.10, False),
    (10000, 10000, 0.001, True),
    (10000, 10000, 0.01, True),
    (10000, 10000, 0.10, True),
    # Huge — bandwidth-limited regime
    (20000, 20000, 0.001, False),
    (20000, 20000, 0.01, False),
    (20000, 20000, 0.10, False),
    (20000, 20000, 0.001, True),
    (20000, 20000, 0.01, True),
    (20000, 20000, 0.10, True),
]


def _make_benchmark_data(*, platform):
    rng = np.random.default_rng(42)

    # Main configs: float32 across all sizes
    for dim0, dim1, density, transpose in CONFIGS:
        dtype = jnp.float32
        if transpose:
            k, n = dim0, dim1
            weights = jnp.asarray(rng.standard_normal((k, n)), dtype=dtype)
            spikes = jnp.asarray(rng.random(k) < density, dtype=jnp.bool_)
            label_mode = "T"
        else:
            m, k = dim0, dim1
            weights = jnp.asarray(rng.standard_normal((m, k)), dtype=dtype)
            spikes = jnp.asarray(rng.random(k) < density, dtype=jnp.bool_)
            label_mode = "NT"

        nnz = int(spikes.sum())
        name = f"{label_mode},f32,d={density:.1%},{dim0}x{dim1},nnz={nnz}"

        yield BenchmarkConfig(
            name=name,
            args=(weights, spikes),
            kernel_kwargs={'transpose': transpose},
            data_kwargs={'dim0': dim0, 'dim1': dim1, 'density': density},
        )

    # Multi-dtype configs: f16, bf16 at medium size (bandwidth-sensitive)
    for dtype, dtype_label in [(jnp.float16, 'f16'), (jnp.bfloat16, 'bf16')]:
        for density in [0.01, 0.10]:
            for transpose in [False, True]:
                dim0, dim1 = 5000, 5000
                if transpose:
                    k, n = dim0, dim1
                    weights = jnp.asarray(rng.standard_normal((k, n)), dtype=dtype)
                    spikes = jnp.asarray(rng.random(k) < density, dtype=jnp.bool_)
                    label_mode = "T"
                else:
                    m, k = dim0, dim1
                    weights = jnp.asarray(rng.standard_normal((m, k)), dtype=dtype)
                    spikes = jnp.asarray(rng.random(k) < density, dtype=jnp.bool_)
                    label_mode = "NT"

                nnz = int(spikes.sum())
                name = f"{label_mode},{dtype_label},d={density:.1%},{dim0}x{dim1},nnz={nnz}"

                yield BenchmarkConfig(
                    name=name,
                    args=(weights, spikes),
                    kernel_kwargs={'transpose': transpose},
                    data_kwargs={'dim0': dim0, 'dim1': dim1, 'density': density},
                )


# ---- Manual micro-benchmark: amortizes dispatch overhead ----

def _measure_dispatch_overhead(n_warmup=50, n_runs=500):
    """
    Measure JAX dispatch + block_until_ready overhead using a tiny (4x4)
    operation.  The actual kernel time is negligible, so the measured time
    ≈ pure dispatch overhead.  Returns (mean_us, min_us) per backend.
    """
    gpu = jax.devices("gpu")[0]
    w_tiny = jax.device_put(jnp.ones((4, 4), jnp.float32), gpu).block_until_ready()
    s_tiny = jax.device_put(jnp.ones(4, jnp.bool_), gpu).block_until_ready()
    overheads = {}
    for backend in ['jax_raw', 'tvmffi']:
        fn = jax.jit(lambda w, s: binary_densemv_p.call(w, s, transpose=False, backend=backend))
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
    Python dispatch overhead across many calls.  This gives a cleaner picture
    of actual kernel throughput.
    """
    rng = np.random.default_rng(42)
    gpu = jax.devices("gpu")[0]

    # ---- Measure dispatch overhead first ----
    print("\n" + "=" * 100)
    print("Measuring dispatch overhead (4x4 no-op) ...")
    overheads = _measure_dispatch_overhead()
    for b, v in overheads.items():
        print(f"  {b:<10}  mean={v['mean']:.1f} us   min={v['min']:.1f} us")
    print()

    configs = [
        # (m_or_k, k_or_n, density, transpose, dtype)
        (5000, 5000, 0.01, False, jnp.float32),
        (5000, 5000, 0.10, False, jnp.float32),
        (10000, 10000, 0.01, False, jnp.float32),
        (10000, 10000, 0.10, False, jnp.float32),
        (20000, 20000, 0.01, False, jnp.float32),
        (20000, 20000, 0.10, False, jnp.float32),
        (5000, 5000, 0.01, True, jnp.float32),
        (5000, 5000, 0.10, True, jnp.float32),
        (10000, 10000, 0.01, True, jnp.float32),
        (10000, 10000, 0.10, True, jnp.float32),
        (20000, 20000, 0.01, True, jnp.float32),
        (20000, 20000, 0.10, True, jnp.float32),
        # Multi-dtype: f16 and bf16 at medium size
        (5000, 5000, 0.01, False, jnp.float16),
        (5000, 5000, 0.10, False, jnp.float16),
        (5000, 5000, 0.01, True, jnp.float16),
        (5000, 5000, 0.10, True, jnp.float16),
        (5000, 5000, 0.01, False, jnp.bfloat16),
        (5000, 5000, 0.10, False, jnp.bfloat16),
        (5000, 5000, 0.01, True, jnp.bfloat16),
        (5000, 5000, 0.10, True, jnp.bfloat16),
    ]

    backends = ['jax_raw', 'tvmffi']

    print("=" * 100)
    print("Manual micro-benchmark (jit-compiled, amortized dispatch)")
    print("=" * 100)
    print(f"{'config':<50} {'backend':<10} {'mean_us':>10} {'min_us':>10}"
          f" {'kern_us':>10} {'speedup':>8}")
    print("-" * 100)

    for dim0, dim1, density, transpose, dtype in configs:
        dtype_label = {jnp.float32: 'f32', jnp.float16: 'f16', jnp.bfloat16: 'bf16'}.get(dtype, 'f32')
        if transpose:
            k, n = dim0, dim1
            weights = jax.device_put(
                jnp.asarray(rng.standard_normal((k, n)), dtype=dtype), gpu
            ).block_until_ready()
            spikes = jax.device_put(
                jnp.asarray(rng.random(k) < density, dtype=jnp.bool_), gpu
            ).block_until_ready()
            label_mode = "T"
        else:
            m, k = dim0, dim1
            weights = jax.device_put(
                jnp.asarray(rng.standard_normal((m, k)), dtype=dtype), gpu
            ).block_until_ready()
            spikes = jax.device_put(
                jnp.asarray(rng.random(k) < density, dtype=jnp.bool_), gpu
            ).block_until_ready()
            label_mode = "NT"

        nnz = int(spikes.sum())
        label = f"{label_mode},{dtype_label},d={density:.1%},{dim0}x{dim1},nnz={nnz}"

        kern_times = {}  # backend -> estimated kernel min time
        for backend in backends:
            # JIT-compile with the specific backend
            fn = jax.jit(lambda w, s: binary_densemv_p.call(w, s, transpose=transpose, backend=backend))

            # Warmup (includes compilation)
            for _ in range(n_warmup):
                _ = fn(weights, spikes)[0].block_until_ready()

            # Timed runs
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = fn(weights, spikes)[0].block_until_ready()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1e6)  # microseconds

            mean_us = sum(times) / len(times)
            min_us = min(times)
            disp = overheads[backend]['min']
            kern_us = max(0.0, min_us - disp)
            kern_times[backend] = kern_us

            # Compute speedup (kernel time of jax_raw / this backend)
            if backend == 'jax_raw':
                speedup_str = ""
            else:
                jax_kern = kern_times.get('jax_raw', 0)
                if kern_us > 0:
                    speedup_str = f"{jax_kern / kern_us:.1f}x"
                else:
                    speedup_str = "inf"

            print(f"{label:<50} {backend:<10} {mean_us:>10.1f} {min_us:>10.1f}"
                  f" {kern_us:>10.1f} {speedup_str:>8}")

        print()


def main():
    parser = argparse.ArgumentParser(description="binary_densemv backend benchmark")
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

    print(f"binary_densemv benchmark  —  GPU: {gpu}")

    if args.manual:
        _manual_benchmark()
        return

    print(f"warmup={args.n_warmup}  runs={args.n_runs}")
    print()

    binary_densemv_p.def_benchmark_data(_make_benchmark_data)

    result = binary_densemv_p.benchmark(
        platform='gpu',
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        compare_results=True,
        verbose=True,
    )
    result.print(vary_by='backend', highlight_best=True, speedup_vs='jax_raw')


if __name__ == "__main__":
    main()
