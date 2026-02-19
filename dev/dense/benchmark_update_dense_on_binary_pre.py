"""
Dense Pre-Synaptic Plasticity Update Benchmark
===============================================

Benchmarks all available GPU backends for ``update_dense_on_binary_pre``
across problem sizes, spike densities, and weight dtypes.

Operation:
    For each active pre_spike[i]:  weight[i, :] += post_trace

This benchmark compares:
  - pallas:   JAX Pallas/Triton kernel (event-driven, fori_loop-based)
  - tvmffi:   Custom CUDA kernel (event-driven, row-parallel)

The CUDA kernel's advantage is largest when:
  - Spike density is low (<= 1%) — most blocks exit in ~1 warp instruction.
  - n_post is large (>= 1000) — row updates are long and benefit from
    coalesced 256-wide writes.

At high density (> 10%) or small n_post, the kernel approaches a dense
memset-like operation where bandwidth-bound runtimes converge.

Usage
-----
    python dev/dense/benchmark_update_dense_on_binary_pre.py
    python dev/dense/benchmark_update_dense_on_binary_pre.py --n_warmup 10 --n_runs 100
    python dev/dense/benchmark_update_dense_on_binary_pre.py --manual
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

from brainevent import BenchmarkConfig
from brainevent._dense.plasticity_binary import update_dense_on_binary_pre_p

# ---- Benchmark configurations ----

# (n_pre, n_post, density)
CONFIGS = [
    # Small regime — kernel launch / dispatch dominated
    (500,   500,   0.01),
    (500,   500,   0.10),
    # Medium regime — balance between compute and latency
    (1000,  1000,  0.001),
    (1000,  1000,  0.01),
    (1000,  1000,  0.10),
    # Large — memory bandwidth dominated
    (5000,  5000,  0.001),
    (5000,  5000,  0.01),
    (5000,  5000,  0.10),
    # Very large — stress test
    (10000, 10000, 0.001),
    (10000, 10000, 0.01),
    (10000, 10000, 0.10),
]


def _make_benchmark_data(*, platform):
    rng = np.random.default_rng(42)

    # Float32 main configs
    for n_pre, n_post, density in CONFIGS:
        dtype = jnp.float32
        weight = jnp.asarray(rng.standard_normal((n_pre, n_post)), dtype=dtype)
        spike = jnp.asarray(rng.random(n_pre) < density, dtype=jnp.bool_)
        trace = jnp.asarray(rng.standard_normal(n_post), dtype=dtype)
        nnz = int(spike.sum())
        name = f"f32,bool,d={density:.1%},{n_pre}x{n_post},nnz={nnz}"
        yield BenchmarkConfig(
            name=name,
            args=(weight, spike, trace),
            data_kwargs={'n_pre': n_pre, 'n_post': n_post, 'density': density},
        )

    # Float spike variant at medium size
    n_pre, n_post = 5000, 5000
    for density in [0.001, 0.01, 0.10]:
        dtype = jnp.float32
        weight = jnp.asarray(rng.standard_normal((n_pre, n_post)), dtype=dtype)
        spike = jnp.asarray(rng.random(n_pre), dtype=dtype)  # float spike
        trace = jnp.asarray(rng.standard_normal(n_post), dtype=dtype)
        nnz = int((spike > 0).sum())
        name = f"f32,float_spk,d~{density:.1%},{n_pre}x{n_post},nnz={nnz}"
        yield BenchmarkConfig(
            name=name,
            args=(weight, spike, trace),
            data_kwargs={'n_pre': n_pre, 'n_post': n_post, 'density': density},
        )

    # Multi-dtype at medium size: f16, bf16
    for dtype, dtype_label in [(jnp.float16, 'f16'), (jnp.bfloat16, 'bf16')]:
        for density in [0.01, 0.10]:
            n_pre, n_post = 5000, 5000
            weight = jnp.asarray(rng.standard_normal((n_pre, n_post)), dtype=dtype)
            spike = jnp.asarray(rng.random(n_pre) < density, dtype=jnp.bool_)
            trace = jnp.asarray(rng.standard_normal(n_post), dtype=dtype)
            nnz = int(spike.sum())
            name = f"{dtype_label},bool,d={density:.1%},{n_pre}x{n_post},nnz={nnz}"
            yield BenchmarkConfig(
                name=name,
                args=(weight, spike, trace),
                data_kwargs={'n_pre': n_pre, 'n_post': n_post, 'density': density},
            )

    # Float64 at small size (memory-heavy)
    for density in [0.01, 0.10]:
        n_pre, n_post = 2000, 2000
        weight = jnp.asarray(rng.standard_normal((n_pre, n_post)), dtype=jnp.float64)
        spike = jnp.asarray(rng.random(n_pre) < density, dtype=jnp.bool_)
        trace = jnp.asarray(rng.standard_normal(n_post), dtype=jnp.float64)
        nnz = int(spike.sum())
        name = f"f64,bool,d={density:.1%},{n_pre}x{n_post},nnz={nnz}"
        yield BenchmarkConfig(
            name=name,
            args=(weight, spike, trace),
            data_kwargs={'n_pre': n_pre, 'n_post': n_post, 'density': density},
        )


# ---- Manual micro-benchmark: amortizes dispatch overhead ----

def _measure_dispatch_overhead(backends, n_warmup=50, n_runs=500):
    """Measure JAX dispatch + block_until_ready overhead for a tiny op."""
    gpu = jax.devices("gpu")[0]
    w = jax.device_put(jnp.ones((4, 4), jnp.float32), gpu).block_until_ready()
    spk = jax.device_put(jnp.ones(4, jnp.bool_), gpu).block_until_ready()
    trc = jax.device_put(jnp.ones(4, jnp.float32), gpu).block_until_ready()
    overheads = {}
    for backend in backends:
        fn = jax.jit(lambda w, s, t: update_dense_on_binary_pre_p.call(w, s, t, backend=backend))
        for _ in range(n_warmup):
            fn(w, spk, trc)[0].block_until_ready()
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            fn(w, spk, trc)[0].block_until_ready()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e6)
        overheads[backend] = {'mean': sum(times) / len(times), 'min': min(times)}
    return overheads


def _manual_benchmark(n_warmup=20, n_runs=200):
    rng = np.random.default_rng(42)
    gpu = jax.devices("gpu")[0]

    backends = update_dense_on_binary_pre_p.available_backends('gpu')
    backends.remove('jax_raw')  # Exclude raw JAX for dispatch overhead comparison
    backends.insert(0, 'jax_raw')  # Add it back at the front for reference
    if not backends:
        print("No GPU backends registered.")
        return

    print("\n" + "=" * 110)
    print("Measuring dispatch overhead (4x4 no-op) ...")
    overheads = _measure_dispatch_overhead(backends)
    for b, v in overheads.items():
        print(f"  {b:<12}  mean={v['mean']:.1f} us   min={v['min']:.1f} us")
    print()

    manual_configs = [
        # (n_pre, n_post, density, dtype)
        (1000,  1000,  0.01,  jnp.float32),
        (1000,  1000,  0.10,  jnp.float32),
        (5000,  5000,  0.001, jnp.float32),
        (5000,  5000,  0.01,  jnp.float32),
        (5000,  5000,  0.10,  jnp.float32),
        (10000, 10000, 0.001, jnp.float32),
        (10000, 10000, 0.01,  jnp.float32),
        (10000, 10000, 0.10,  jnp.float32),
        (5000,  5000,  0.01,  jnp.float16),
        (5000,  5000,  0.10,  jnp.float16),
        (5000,  5000,  0.01,  jnp.bfloat16),
        (5000,  5000,  0.10,  jnp.bfloat16),
    ]

    print("=" * 110)
    print("Manual micro-benchmark (jit-compiled, amortized dispatch)")
    print("=" * 110)
    hdr = f"{'config':<55} {'backend':<12} {'mean_us':>10} {'min_us':>10} {'kern_us':>10} {'speedup':>8}"
    print(hdr)
    print("-" * 110)

    ref_backend = 'jax_raw'
    for n_pre, n_post, density, dtype in manual_configs:
        dtype_label = {jnp.float32: 'f32', jnp.float16: 'f16',
                       jnp.bfloat16: 'bf16', jnp.float64: 'f64'}.get(dtype, '?')
        weight = jax.device_put(
            jnp.asarray(rng.standard_normal((n_pre, n_post)), dtype=dtype), gpu
        ).block_until_ready()
        spike = jax.device_put(
            jnp.asarray(rng.random(n_pre) < density, dtype=jnp.bool_), gpu
        ).block_until_ready()
        trace = jax.device_put(
            jnp.asarray(rng.standard_normal(n_post), dtype=dtype), gpu
        ).block_until_ready()
        nnz = int(spike.sum())
        label = f"{dtype_label},bool,d={density:.1%},{n_pre}x{n_post},nnz={nnz}"

        kern_times = {}
        for backend in backends:
            fn = jax.jit(lambda w, s, t: update_dense_on_binary_pre_p.call(w, s, t, backend=backend))
            for _ in range(n_warmup):
                fn(weight, spike, trace)[0].block_until_ready()
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                fn(weight, spike, trace)[0].block_until_ready()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1e6)
            mean_us = sum(times) / len(times)
            min_us = min(times)
            disp = overheads[backend]['min']
            kern_us = max(0.0, min_us - disp)
            kern_times[backend] = kern_us
            if backend == ref_backend:
                speedup_str = ""
            else:
                ref_kern = kern_times.get(ref_backend, 0)
                speedup_str = f"{ref_kern / kern_us:.2f}x" if kern_us > 0 else "inf"
            print(f"{label:<55} {backend:<12} {mean_us:>10.1f} {min_us:>10.1f}"
                  f" {kern_us:>10.1f} {speedup_str:>8}")
        print()


def main():
    parser = argparse.ArgumentParser(description="update_dense_on_binary_pre benchmark")
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_runs", type=int, default=30)
    parser.add_argument("--manual", action="store_true", default=True,
                        help="Run manual micro-benchmark for precise kernel timing")
    args = parser.parse_args()

    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        print("ERROR: No GPU device found.")
        return

    print(f"update_dense_on_binary_pre benchmark  —  GPU: {gpu}")

    if args.manual:
        _manual_benchmark()
        return

    print(f"warmup={args.n_warmup}  runs={args.n_runs}")
    print()

    update_dense_on_binary_pre_p.def_benchmark_data(_make_benchmark_data)

    result = update_dense_on_binary_pre_p.benchmark(
        platform='gpu',
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        compare_results=True,
        verbose=True,
    )
    result.print(vary_by='backend', highlight_best=True, speedup_vs='jax_raw')


if __name__ == "__main__":
    main()
