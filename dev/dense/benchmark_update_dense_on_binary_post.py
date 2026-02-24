"""
Dense Post-Synaptic Plasticity Update Benchmark
================================================

Benchmarks all available GPU backends for ``update_dense_on_binary_post``
across problem sizes, spike densities, and weight dtypes.

Operation:
    For each active post_spike[j]:  weight[:, j] += pre_trace

This benchmark compares:
  - pallas:   JAX Pallas/Triton kernel (event-driven, fori_loop-based)
  - tvmffi:   Custom CUDA kernel (event-driven, 32-column tiled, coalesced)

The CUDA kernel's advantage is largest when:
  - Spike density is low (<= 1%) — most column tiles exit after the shared-
    memory spike check with no global writes.
  - n_pre is large (>= 1000) — the tiled 8-warp × 32-column kernel amortises
    the block launch overhead over many row iterations.

The tiled kernel achieves full coalesced writes:
  - Each warp handles 32 consecutive columns for one row → 128-byte aligned
    coalesced write in row-major layout.
  - 8 warps simultaneously update 8 rows of the same 32-column tile.

At high density (> 10%), the kernel approaches a full-matrix scatter and
is dominated by memory bandwidth.  All backends converge in this regime.

Usage
-----
    python dev/dense/benchmark_update_dense_on_binary_post.py
    python dev/dense/benchmark_update_dense_on_binary_post.py --n_warmup 10 --n_runs 100
    python dev/dense/benchmark_update_dense_on_binary_post.py --manual
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
from brainevent._dense.plasticity_binary import update_dense_on_binary_post_p

# ---- Benchmark configurations ----

# (n_pre, n_post, density)
CONFIGS = [
    # Small regime — kernel launch / dispatch dominated
    (500,   500,   0.01),
    (500,   500,   0.10),
    # Medium regime
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
        trace = jnp.asarray(rng.standard_normal(n_pre), dtype=dtype)
        spike = jnp.asarray(rng.random(n_post) < density, dtype=jnp.bool_)
        nnz = int(spike.sum())
        name = f"f32,bool,d={density:.1%},{n_pre}x{n_post},nnz={nnz}"
        yield BenchmarkConfig(
            name=name,
            args=(weight, trace, spike),
            data_kwargs={'n_pre': n_pre, 'n_post': n_post, 'density': density},
        )

    # Float spike variant at medium size
    n_pre, n_post = 5000, 5000
    for density in [0.001, 0.01, 0.10]:
        dtype = jnp.float32
        weight = jnp.asarray(rng.standard_normal((n_pre, n_post)), dtype=dtype)
        trace = jnp.asarray(rng.standard_normal(n_pre), dtype=dtype)
        spike = jnp.asarray(rng.random(n_post), dtype=dtype)  # float spike
        nnz = int((spike > 0).sum())
        name = f"f32,float_spk,d~{density:.1%},{n_pre}x{n_post},nnz={nnz}"
        yield BenchmarkConfig(
            name=name,
            args=(weight, trace, spike),
            data_kwargs={'n_pre': n_pre, 'n_post': n_post, 'density': density},
        )

    # Multi-dtype at medium size: f16, bf16
    for dtype, dtype_label in [(jnp.float16, 'f16'), (jnp.bfloat16, 'bf16')]:
        for density in [0.01, 0.10]:
            n_pre, n_post = 5000, 5000
            weight = jnp.asarray(rng.standard_normal((n_pre, n_post)), dtype=dtype)
            trace = jnp.asarray(rng.standard_normal(n_pre), dtype=dtype)
            spike = jnp.asarray(rng.random(n_post) < density, dtype=jnp.bool_)
            nnz = int(spike.sum())
            name = f"{dtype_label},bool,d={density:.1%},{n_pre}x{n_post},nnz={nnz}"
            yield BenchmarkConfig(
                name=name,
                args=(weight, trace, spike),
                data_kwargs={'n_pre': n_pre, 'n_post': n_post, 'density': density},
            )

    # Float64 at small size
    for density in [0.01, 0.10]:
        n_pre, n_post = 2000, 2000
        weight = jnp.asarray(rng.standard_normal((n_pre, n_post)), dtype=jnp.float64)
        trace = jnp.asarray(rng.standard_normal(n_pre), dtype=jnp.float64)
        spike = jnp.asarray(rng.random(n_post) < density, dtype=jnp.bool_)
        nnz = int(spike.sum())
        name = f"f64,bool,d={density:.1%},{n_pre}x{n_post},nnz={nnz}"
        yield BenchmarkConfig(
            name=name,
            args=(weight, trace, spike),
            data_kwargs={'n_pre': n_pre, 'n_post': n_post, 'density': density},
        )


# ---- Manual micro-benchmark: amortizes dispatch overhead ----

def _measure_dispatch_overhead(backends, n_warmup=50, n_runs=500):
    """Measure JAX dispatch + block_until_ready overhead for a tiny op."""
    gpu = jax.devices("gpu")[0]
    w = jax.device_put(jnp.ones((4, 4), jnp.float32), gpu).block_until_ready()
    trc = jax.device_put(jnp.ones(4, jnp.float32), gpu).block_until_ready()
    spk = jax.device_put(jnp.ones(4, jnp.bool_), gpu).block_until_ready()
    overheads = {}
    for backend in backends:
        fn = jax.jit(lambda w, t, s: update_dense_on_binary_post_p.call(w, t, s, backend=backend))
        for _ in range(n_warmup):
            fn(w, trc, spk)[0].block_until_ready()
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            fn(w, trc, spk)[0].block_until_ready()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e6)
        overheads[backend] = {'mean': sum(times) / len(times), 'min': min(times)}
    return overheads


def _manual_benchmark(n_warmup=20, n_runs=200):
    rng = np.random.default_rng(42)
    gpu = jax.devices("gpu")[0]

    backends = update_dense_on_binary_post_p.available_backends('gpu')
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
        trace = jax.device_put(
            jnp.asarray(rng.standard_normal(n_pre), dtype=dtype), gpu
        ).block_until_ready()
        spike = jax.device_put(
            jnp.asarray(rng.random(n_post) < density, dtype=jnp.bool_), gpu
        ).block_until_ready()
        nnz = int(spike.sum())
        label = f"{dtype_label},bool,d={density:.1%},{n_pre}x{n_post},nnz={nnz}"

        kern_times = {}
        for backend in backends:
            fn = jax.jit(lambda w, t, s: update_dense_on_binary_post_p.call(w, t, s, backend=backend))
            for _ in range(n_warmup):
                fn(weight, trace, spike)[0].block_until_ready()
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                fn(weight, trace, spike)[0].block_until_ready()
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
    parser = argparse.ArgumentParser(description="update_dense_on_binary_post benchmark")
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_runs", type=int, default=30)
    parser.add_argument("--manual", action="store_true", default=False,
                        help="Run manual micro-benchmark for precise kernel timing")
    args = parser.parse_args()

    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        print("ERROR: No GPU device found.")
        return

    print(f"update_dense_on_binary_post benchmark  —  GPU: {gpu}")

    if args.manual:
        _manual_benchmark()
        return

    print(f"warmup={args.n_warmup}  runs={args.n_runs}")
    print()

    update_dense_on_binary_post_p.def_benchmark_data(_make_benchmark_data)

    result = update_dense_on_binary_post_p.benchmark(
        platform='gpu',
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        compare_results=True,
        verbose=True,
    )
    result.print(vary_by='backend', highlight_best=True, speedup_vs='jax_raw')


if __name__ == "__main__":
    main()
