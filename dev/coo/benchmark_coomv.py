"""
COO Matrix-Vector Multiplication Benchmark
==========================================

Benchmarks all available backends for ``coomv`` (float-weighted COO SpMV,
both transpose modes, homo and hetero weights) across a range of problem
sizes and connection probabilities.

Usage
-----
    python dev/coo/benchmark_coomv.py
    python dev/coo/benchmark_coomv.py --n_warmup 10 --n_runs 100
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
from brainevent._coo.float import coomv_p

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


def _make_benchmark_data(*, platform):
    brainstate.environ.set(precision=32)
    rng = np.random.default_rng(42)
    dtype = brainstate.environ.dftype()
    for n_pre, n_post, prob in CONFIGS:
        nnz = max(1, int(n_pre * n_post * prob))
        row = jnp.asarray(rng.integers(0, n_pre, nnz, dtype=np.int32))
        col = jnp.asarray(rng.integers(0, n_post, nnz, dtype=np.int32))
        for transpose in (False, True):
            for homo in (True, False):
                if homo:
                    weights = jnp.ones(1, dtype=dtype)
                else:
                    weights = jnp.asarray(rng.standard_normal(nnz), dtype=dtype)
                v_size = n_post if not transpose else n_pre
                vector = jnp.asarray(rng.standard_normal(v_size), dtype=dtype)
                name = (
                    f"{'T' if transpose else 'NT'},"
                    f"{'homo' if homo else 'hetero'},"
                    f"{n_pre}x{n_post},p={prob}"
                )
                yield BenchmarkConfig(
                    name=name,
                    args=(weights, row, col, vector),
                    kernel_kwargs={'shape': (n_pre, n_post), 'transpose': transpose},
                    data_kwargs={'n_pre': n_pre, 'n_post': n_post, 'prob': prob},
                )


def main():
    parser = argparse.ArgumentParser(description="coomv backend benchmark")
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_runs", type=int, default=50)
    args = parser.parse_args()

    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        print("ERROR: No GPU device found.")
        return

    print(f"coomv benchmark  —  GPU: {gpu}")
    print(f"warmup={args.n_warmup}  runs={args.n_runs}")
    print()

    coomv_p.def_benchmark_data(_make_benchmark_data)

    result = coomv_p.benchmark(
        platform='gpu',
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        compare_results=True,
        verbose=True,
    )
    result.print(vary_by='backend', highlight_best=True, speedup_vs='jax')


if __name__ == "__main__":
    main()
