"""
Sparse-Float FCN Matrix-Vector Multiplication Benchmark
=======================================================

Benchmarks all available GPU backends for ``spfloat_fcnmv`` (gather and scatter
modes, homo and hetero weights, various spike rates) across a range of problem
sizes.  The primary interest is the sparsity-exploitation advantage of the CUDA
(tvmffi) backend over dense-style kernels at low firing rates.

Usage
-----
    python dev/fcn/benchmark_spfloat_fcnmv.py
    python dev/fcn/benchmark_spfloat_fcnmv.py --n_warmup 5 --n_runs 50
    python dev/fcn/benchmark_spfloat_fcnmv.py --spike_rates 0.01 0.05 0.1 0.5
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

from brainevent import BenchmarkConfig, spfloat_fcnmv_p

# (n_pre, n_post, n_conn) configurations
CONFIGS = [
    (1000, 1000, 50),
    (1000, 1000, 100),
    (5000, 5000, 200),
    (5000, 5000, 500),
    (10000, 10000, 1000),
]

# Spike rates to sweep.  At lower rates, the CUDA kernel's early-exit
# optimisation should show the largest speedup over dense-style kernels.
DEFAULT_SPIKE_RATES = [0.01, 0.05, 0.10, 0.50]


def _make_benchmark_data(*, platform, spike_rates=None):
    if spike_rates is None:
        spike_rates = DEFAULT_SPIKE_RATES
    rng = np.random.default_rng(42)
    dtype = brainstate.environ.dftype()
    for n_pre, n_post, n_conn in CONFIGS:
        indices = jnp.asarray(rng.integers(0, n_post, (n_pre, n_conn), dtype=np.int32))
        for transpose in (False, True):
            for homo in (True, False):
                if homo:
                    weights = jnp.ones(1, dtype=dtype)
                else:
                    weights = jnp.asarray(rng.standard_normal((n_pre, n_conn)), dtype=dtype)
                v_size = n_post if not transpose else n_pre
                for rate in spike_rates:
                    # Sparse-float vector: draw from N(0,1) but zero out (1-rate) fraction
                    v_raw = rng.standard_normal(v_size)
                    mask = rng.random(v_size) < rate
                    vector = jnp.asarray(v_raw * mask, dtype=dtype)
                    name = (
                        f"{'T' if transpose else 'NT'},"
                        f"{'homo' if homo else 'hetero'},"
                        f"{n_pre}x{n_post}x{n_conn},"
                        f"rate={rate:.0%}"
                    )
                    yield BenchmarkConfig(
                        name=name,
                        args=(weights, indices, vector),
                        kernel_kwargs={'shape': (n_pre, n_post), 'transpose': transpose},
                        data_kwargs={
                            'n_pre': n_pre,
                            'n_post': n_post,
                            'n_conn': n_conn,
                            'transpose': transpose,
                            'spike_rate': rate,
                        },
                    )


def main():
    parser = argparse.ArgumentParser(description="spfloat_fcnmv backend benchmark")
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_runs", type=int, default=20)
    parser.add_argument(
        "--spike_rates",
        type=float,
        nargs="+",
        default=DEFAULT_SPIKE_RATES,
        metavar="RATE",
        help="Firing rates to sweep (e.g. 0.01 0.05 0.10 0.50)",
    )
    args = parser.parse_args()

    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        print("ERROR: No GPU device found.")
        return

    print(f"spfloat_fcnmv benchmark  —  GPU: {gpu}")
    print(f"warmup={args.n_warmup}  runs={args.n_runs}")
    print(f"spike_rates={args.spike_rates}")

    def _data_gen(*, platform):
        yield from _make_benchmark_data(platform=platform, spike_rates=args.spike_rates)

    spfloat_fcnmv_p.def_benchmark_data(_data_gen)

    result = spfloat_fcnmv_p.benchmark(
        platform='gpu',
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        n_batch_per_run=1,
        compare_results=True,
        verbose=True,
    )
    result.print(vary_by='backend', highlight_best=True, speedup_vs='jax_raw')


if __name__ == "__main__":
    main()
