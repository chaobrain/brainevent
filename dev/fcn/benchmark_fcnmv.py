"""
FCN Matrix-Vector Multiplication Benchmark
==========================================

Benchmarks all available GPU backends for ``fcnmv`` (gather and scatter modes,
homo and hetero weights) across a range of problem sizes.

Usage
-----
    python dev/fcn/benchmark_fcnmv.py
    python dev/fcn/benchmark_fcnmv.py --n_warmup 5 --n_runs 50
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

from brainevent import fcnmv_p, BenchmarkConfig

CONFIGS = [
    (500, 1000, 10),
    (1000, 1000, 50),
    (1000, 1000, 100),
    (1000, 1000, 128),
    (5000, 5000, 200),
    (5000, 5000, 500),
    (10000, 10000, 1000),
]


def _make_benchmark_data(*, platform):
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
                vector = jnp.asarray(rng.standard_normal(v_size), dtype=dtype)
                name = (
                    f"{'T' if transpose else 'NT'},"
                    f"{'homo' if homo else 'hetero'},"
                    f"{n_pre}x{n_post}x{n_conn}"
                )
                yield BenchmarkConfig(
                    name=name,
                    args=(weights, indices, vector),
                    kernel_kwargs={'shape': (n_pre, n_post), 'transpose': transpose},
                    data_kwargs={'n_pre': n_pre, 'n_post': n_post, 'n_conn': n_conn},
                )


def main():
    parser = argparse.ArgumentParser(description="fcnmv backend benchmark")
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_runs", type=int, default=1)
    args = parser.parse_args()

    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        print("ERROR: No GPU device found.")
        return

    print(f"fcnmv benchmark  —  GPU: {gpu}")
    print(f"warmup={args.n_warmup}  runs={args.n_runs}")

    fcnmv_p.def_benchmark_data(_make_benchmark_data)

    result = fcnmv_p.benchmark(
        platform='gpu',
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        compare_results=True,
        verbose=True,
    )
    # result.print(order_by=['transpose', 'shape', 'backend'], highlight_best=True, speedup_vs='jax_raw')
    result.print(vary_by='backend', highlight_best=True, speedup_vs='jax_raw')


if __name__ == "__main__":
    main()
