"""
Binary COO Matrix-Vector Multiplication Benchmark
==================================================

Benchmarks all available backends for ``binary_coomv`` (event-driven COO
SpMV, homo/hetero weights, bool/float spikes) across a range of problem
sizes, connection probabilities, and spike densities.

The ``tvmffi`` (TVM FFI CUDA) backend is included automatically once
registered via ``binary_coomv_p.def_tvmffi_kernel``.

Usage
-----
    python dev/coo/benchmark_binary_coomv.py
    python dev/coo/benchmark_binary_coomv.py --n_warmup 10 --n_runs 100
    python dev/coo/benchmark_binary_coomv.py --mode density
    python dev/coo/benchmark_binary_coomv.py --mode size

Modes
-----
default
    Mixed sweep across sizes and spike rates; reproduces the original benchmark.
density
    Fixed medium problem (5000x5000, p=0.05), sweeps spike densities
    0.1%, 1%, 10% for NT/T homo/hetero bool/float -- highlights event-driven
    sparsity exploitation per backend.
size
    Fixed 10% spike rate, sweeps small/medium/large problem sizes --
    highlights scaling behaviour and GPU parallelism win region.
"""

import argparse
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import jax
import jax.numpy as jnp
import numpy as np

import brainstate
from brainevent import BenchmarkConfig
from brainevent._coo.binary import binary_coomv_p

# ---------------------------------------------------------------------------
# Problem-size matrix:  (n_pre, n_post, conn_prob)
# ---------------------------------------------------------------------------
CONFIGS_DEFAULT = [
    (500,   1000,  0.01),
    (1000,  1000,  0.05),
    (1000,  1000,  0.10),
    (5000,  5000,  0.01),
    (5000,  5000,  0.05),
    (10000, 10000, 0.01),
    (10000, 10000, 0.05),
]

# For size sweep: fixed 10 % spike rate, varied matrix footprint
CONFIGS_SIZE = [
    (500,    500,   0.10),   # small
    (2000,   2000,  0.05),   # medium-small
    (5000,   5000,  0.05),   # medium
    (10000,  10000, 0.02),   # medium-large
    (20000,  20000, 0.01),   # large
    (50000,  50000, 0.002),  # very large
]

# For density sweep: fixed medium matrix
CONFIGS_DENSITY = [
    (5000, 5000, 0.05),
]
SPIKE_DENSITIES = [0.001, 0.01, 0.10]  # 0.1%, 1%, 10%


def _make_benchmark_data(*, platform, configs, spike_rates, transpose_list=(False, True),
                         homo_list=(True, False), bool_event_list=(True, False)):
    """Generate BenchmarkConfig instances for the given configs and spike rates."""
    brainstate.environ.set(precision=32)
    rng = np.random.default_rng(42)
    dtype = brainstate.environ.dftype()

    for n_pre, n_post, prob in configs:
        nnz = max(1, int(n_pre * n_post * prob))
        row = jnp.asarray(rng.integers(0, n_pre, nnz, dtype=np.int32))
        col = jnp.asarray(rng.integers(0, n_post, nnz, dtype=np.int32))

        for spike_rate in spike_rates:
            for transpose in transpose_list:
                for homo in homo_list:
                    for bool_event in bool_event_list:
                        if homo:
                            weights = jnp.ones(1, dtype=dtype)
                        else:
                            weights = jnp.asarray(rng.standard_normal(nnz), dtype=dtype)

                        v_size = n_post if not transpose else n_pre
                        if bool_event:
                            spikes = jnp.asarray(rng.random(v_size) < spike_rate, dtype=jnp.bool_)
                        else:
                            raw = rng.standard_normal(v_size)
                            mask = rng.random(v_size) < spike_rate
                            spikes = jnp.asarray(np.where(mask, np.abs(raw), 0.0), dtype=dtype)

                        name = (
                            f"{'T' if transpose else 'NT'},"
                            f"{'homo' if homo else 'hetero'},"
                            f"{'bool' if bool_event else 'float'},"
                            f"{n_pre}x{n_post},p={prob},spk={spike_rate:.1%}"
                        )
                        yield BenchmarkConfig(
                            name=name,
                            args=(weights, row, col, spikes),
                            kernel_kwargs={'shape': (n_pre, n_post), 'transpose': transpose},
                            data_kwargs={
                                'n_pre': n_pre, 'n_post': n_post, 'prob': prob,
                                'spike_rate': spike_rate, 'nnz': nnz,
                            },
                        )


def run_default(n_warmup, n_runs, spike_rate):
    """Original benchmark: fixed spike rate, varied sizes and prob."""
    def _data_gen(*, platform):
        yield from _make_benchmark_data(
            platform=platform,
            configs=CONFIGS_DEFAULT,
            spike_rates=[spike_rate],
        )

    binary_coomv_p.def_benchmark_data(_data_gen)
    result = binary_coomv_p.benchmark(
        platform='gpu',
        n_warmup=n_warmup,
        n_runs=n_runs,
        compare_results=True,
        verbose=True,
    )
    result.print(vary_by='backend', highlight_best=True, speedup_vs='jax')


def run_density(n_warmup, n_runs):
    """Density sweep: fixed matrix size, varied spike densities.

    This benchmark shows the event-driven sparsity exploitation per backend.
    At 0.1% spike density (very sparse), CUDA atomic skips 99.9% of atomicAdd
    calls; at 10% (moderately dense), the advantage diminishes.
    """
    def _data_gen(*, platform):
        yield from _make_benchmark_data(
            platform=platform,
            configs=CONFIGS_DENSITY,
            spike_rates=SPIKE_DENSITIES,
            # Focus on the most common case: NT, homo+hetero, bool
            transpose_list=(False,),
            homo_list=(True, False),
            bool_event_list=(True,),
        )

    binary_coomv_p.def_benchmark_data(_data_gen)
    result = binary_coomv_p.benchmark(
        platform='gpu',
        n_warmup=n_warmup,
        n_runs=n_runs,
        compare_results=True,
        verbose=True,
    )
    result.print(vary_by='backend', highlight_best=True, speedup_vs='jax',
                 group_by='spike_rate')


def run_size(n_warmup, n_runs):
    """Size sweep: fixed 10% spike rate, varied matrix dimensions.

    Highlights where GPU parallelism (tvmffi) wins over CPU (numba) and
    pure-JAX scatter: the crossover typically occurs around nnz ~ 50-100K.
    Very large matrices (nnz > 1M) show the clearest CUDA advantage.
    """
    def _data_gen(*, platform):
        yield from _make_benchmark_data(
            platform=platform,
            configs=CONFIGS_SIZE,
            spike_rates=[0.10],
            transpose_list=(False,),
            homo_list=(True, False),
            bool_event_list=(True,),
        )

    binary_coomv_p.def_benchmark_data(_data_gen)
    result = binary_coomv_p.benchmark(
        platform='gpu',
        n_warmup=n_warmup,
        n_runs=n_runs,
        compare_results=True,
        verbose=True,
    )
    result.print(vary_by='backend', highlight_best=True, speedup_vs='jax',
                 group_by='nnz')


def main():
    parser = argparse.ArgumentParser(description="binary_coomv backend benchmark")
    parser.add_argument("--n_warmup", type=int, default=10,
                        help="Number of warmup iterations (default: 10)")
    parser.add_argument("--n_runs", type=int, default=50,
                        help="Number of timed iterations (default: 50)")
    parser.add_argument("--spike_rate", type=float, default=0.1,
                        help="Spike density for default mode (default: 0.1 = 10%%)")
    parser.add_argument(
        "--mode", choices=["default", "density", "size"], default="default",
        help=(
            "Benchmark mode: "
            "'default' = mixed size+prob sweep; "
            "'density' = fixed medium matrix, sweep spike densities 0.1/1/10%%; "
            "'size' = fixed 10%% spikes, sweep small→very-large matrices."
        )
    )
    args = parser.parse_args()

    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        print("ERROR: No GPU device found.")
        return

    print(f"binary_coomv benchmark  —  GPU: {gpu}")
    print(f"mode={args.mode}  warmup={args.n_warmup}  runs={args.n_runs}")
    if args.mode == "default":
        print(f"spike_rate={args.spike_rate:.0%}")
    print()

    if args.mode == "default":
        run_default(args.n_warmup, args.n_runs, args.spike_rate)
    elif args.mode == "density":
        run_density(args.n_warmup, args.n_runs)
    elif args.mode == "size":
        run_size(args.n_warmup, args.n_runs)


if __name__ == "__main__":
    main()
