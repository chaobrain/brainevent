"""
Binary CSR Matrix-Vector Multiplication Benchmark
==================================================

Benchmarks all available backends for ``binary_csrmv`` (event-driven CSR
SpMV, homo/hetero weights, bool/float spikes) across a range of problem
sizes, connection probabilities, and spike rates.

The key metric is speedup of accelerated backends relative to the ``jax``
baseline.  For typical SNN firing rates (1–10%), event-driven kernels
skip inactive rows and should outperform dense SpMV.

Usage
-----
    python dev/csr/benchmark_binary_csrmv.py
    python dev/csr/benchmark_binary_csrmv.py --n_warmup 10 --n_runs 100
    python dev/csr/benchmark_binary_csrmv.py --spike_rate 0.05
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
from brainevent._csr.binary import binary_csrmv_p

# (n_pre, n_post, conn_prob)
CONFIGS = [
    (500,   1000,  0.01),
    (1000,  1000,  0.05),
    (1000,  1000,  0.10),
    (5000,  5000,  0.01),
    (5000,  5000,  0.05),
    (10000, 10000, 0.01),
    (10000, 10000, 0.05),
]


def _make_benchmark_data(*, platform, spike_rate=0.1):
    brainstate.environ.set(precision=32)
    rng = np.random.default_rng(42)
    dtype = brainstate.environ.dftype()
    for n_pre, n_post, prob in CONFIGS:
        n_conn = max(1, int(n_post * prob))
        nse = n_pre * n_conn
        indptr = jnp.asarray(np.arange(n_pre + 1, dtype=np.int32) * n_conn)
        indices = jnp.asarray(rng.integers(0, n_post, nse, dtype=np.int32))
        for transpose in (False, True):
            for homo in (True, False):
                for bool_event in (True, False):
                    if homo:
                        weights = jnp.ones(1, dtype=dtype)
                    else:
                        weights = jnp.asarray(rng.standard_normal(nse), dtype=dtype)
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
                        f"{n_pre}x{n_post},p={prob}"
                    )
                    yield BenchmarkConfig(
                        name=name,
                        args=(weights, indices, indptr, spikes),
                        kernel_kwargs={'shape': (n_pre, n_post), 'transpose': transpose},
                        data_kwargs={
                            'n_pre': n_pre, 'n_post': n_post, 'prob': prob,
                            'spike_rate': spike_rate,
                        },
                    )


def main():
    parser = argparse.ArgumentParser(description="binary_csrmv backend benchmark")
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_runs", type=int, default=50)
    parser.add_argument("--spike_rate", type=float, default=0.1,
                        help="Fraction of active spikes (default: 0.1 = 10%%)")
    args = parser.parse_args()

    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        print("ERROR: No GPU device found.")
        return

    print(f"binary_csrmv benchmark  —  GPU: {gpu}")
    print(f"warmup={args.n_warmup}  runs={args.n_runs}  spike_rate={args.spike_rate:.0%}")
    print()

    def _data_gen(*, platform):
        yield from _make_benchmark_data(platform=platform, spike_rate=args.spike_rate)

    binary_csrmv_p.def_benchmark_data(_data_gen)

    result = binary_csrmv_p.benchmark(
        platform='gpu',
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        compare_results=True,
        verbose=True,
    )
    result.print(vary_by='backend', highlight_best=True, speedup_vs='jax_raw')


if __name__ == "__main__":
    main()
