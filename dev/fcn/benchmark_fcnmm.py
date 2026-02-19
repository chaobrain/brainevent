"""
FCN Matrix-Matrix Multiplication Benchmark
==========================================

Benchmarks all available GPU backends for ``fcnmm`` (gather and scatter modes,
homo and hetero weights) across a range of problem sizes and column counts.

Usage
-----
    python dev/fcn/benchmark_fcnmm.py
    python dev/fcn/benchmark_fcnmm.py --n_warmup 5 --n_runs 50
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

from brainevent import fcnmm_p, BenchmarkConfig

# (n_pre, n_post, n_conn, n_col)
CONFIGS = [
    (1000, 1000, 50, 10),
    (1000, 1000, 100, 128),
    (1000, 1000, 128, 64),
    (1000, 1000, 200, 256),
    (5000, 5000, 100, 128),
    (5000, 5000, 200, 64),
    (5000, 5000, 50, 512),
    (10000, 10000, 100, 128),
    (10000, 10000, 50, 32),
    (10000, 10000, 200, 64),
]


def _make_benchmark_data(*, platform):
    rng = np.random.default_rng(42)
    dtype = brainstate.environ.dftype()
    for n_pre, n_post, n_conn, n_col in CONFIGS:
        indices = jnp.asarray(rng.integers(0, n_post, (n_pre, n_conn), dtype=np.int32))
        for transpose in (False, True):
            for homo in (True, False):
                if homo:
                    weights = jnp.ones(1, dtype=dtype)
                else:
                    weights = jnp.asarray(rng.standard_normal((n_pre, n_conn)), dtype=dtype)
                b_rows = n_post if not transpose else n_pre
                matrix = jnp.asarray(rng.standard_normal((b_rows, n_col)), dtype=dtype)
                name = (
                    f"{'T' if transpose else 'NT'},"
                    f"{'homo' if homo else 'hetero'},"
                    f"{n_pre}x{n_post}x{n_conn},ncol={n_col}"
                )
                yield BenchmarkConfig(
                    name='',
                    args=(weights, indices, matrix),
                    kernel_kwargs={'shape': (n_pre, n_post), 'transpose': transpose},
                    data_kwargs={
                        'n_pre': n_pre, 'n_post': n_post,
                        'n_conn': n_conn, 'n_col': n_col,
                    },
                )


def main():
    parser = argparse.ArgumentParser(description="fcnmm backend benchmark")
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_runs", type=int, default=1)
    args = parser.parse_args()

    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        print("ERROR: No GPU device found.")
        return

    print(f"fcnmm benchmark  —  GPU: {gpu}")
    print(f"warmup={args.n_warmup}  runs={args.n_runs}")

    fcnmm_p.def_benchmark_data(_make_benchmark_data)

    result = fcnmm_p.benchmark(
        platform='gpu',
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        compare_results=True,
        verbose=True,
    )
    result.print(vary_by='backend', highlight_best=True, speedup_vs='jax_raw')


if __name__ == "__main__":
    main()
