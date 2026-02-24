"""
CSR yw2y Benchmark
==================

Benchmarks all available backends for ``csrmv_yw2y`` (per-synapse
element-wise product ``out[j] = w[j] * y[row/col]``) across a range of
matrix sizes, connection probabilities, and transpose modes.

Usage
-----
    python dev/csr/benchmark_csrmv_yw2y.py
    python dev/csr/benchmark_csrmv_yw2y.py --n_warmup 10 --n_runs 100
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
from brainevent._csr.yw2y import csrmv_yw2y_p

# (n_pre, n_post, conn_prob)
CONFIGS = [
    (1000,   1000,  0.05),
    (1000,   1000,  0.10),
    (5000,   5000,  0.01),
    (5000,   5000,  0.05),
    (10000,  10000, 0.01),
    (10000,  10000, 0.02),
    (20000,  20000, 0.005),
]


def _make_benchmark_data(*, platform):
    brainstate.environ.set(precision=32)
    rng = np.random.default_rng(42)
    dtype = brainstate.environ.dftype()
    for n_pre, n_post, prob in CONFIGS:
        n_conn = max(1, int(n_post * prob))
        nse = n_pre * n_conn
        indptr = jnp.asarray(np.arange(n_pre + 1, dtype=np.int32) * n_conn)
        indices = jnp.asarray(rng.integers(0, n_post, nse, dtype=np.int32))
        w = jnp.asarray(rng.standard_normal(nse), dtype=dtype)
        for transpose in (False, True):
            y_size = n_post if transpose else n_pre
            y = jnp.asarray(rng.standard_normal(y_size), dtype=dtype)
            name = (
                f"{'T' if transpose else 'NT'},"
                f"{n_pre}x{n_post},p={prob}"
            )
            yield BenchmarkConfig(
                name=name,
                args=(y, w, indices, indptr),
                kernel_kwargs={'shape': (n_pre, n_post), 'transpose': transpose},
                data_kwargs={
                    'n_pre': n_pre, 'n_post': n_post, 'prob': prob,
                    'transpose': transpose,
                },
            )


def main():
    parser = argparse.ArgumentParser(description="csrmv_yw2y backend benchmark")
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_runs", type=int, default=50)
    args = parser.parse_args()

    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        print("ERROR: No GPU device found.")
        return

    print(f"csrmv_yw2y benchmark  —  GPU: {gpu}")
    print(f"warmup={args.n_warmup}  runs={args.n_runs}")
    print()

    def _data_gen(*, platform):
        yield from _make_benchmark_data(platform=platform)

    csrmv_yw2y_p.def_benchmark_data(_data_gen)

    result = csrmv_yw2y_p.benchmark(
        platform='gpu',
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        compare_results=True,
        verbose=True,
    )
    result.print(vary_by='backend', highlight_best=True, speedup_vs='pallas')


if __name__ == "__main__":
    main()
