"""
JIT Uniform Matrix-Vector Multiplication Benchmark
===================================================

Benchmarks all available GPU backends for ``jitumv`` (gather and scatter modes)
across a range of problem sizes and connection probabilities.

Usage
-----
    python dev/jitu/benchmark_jitumv.py
    python dev/jitu/benchmark_jitumv.py --n_warmup 5 --n_runs 50
    python dev/jitu/benchmark_jitumv.py --backends tvmffi pallas
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

from brainevent import jitumv_p, BenchmarkConfig

CONFIGS = [
    (1000, 1000, 0.1),
    (5000, 5000, 0.1),
    (10000, 10000, 0.1),
    (10000, 10000, 0.01),
    (10000, 10000, 0.5),
]


def _make_benchmark_data(*, platform):
    dtype = jnp.float32
    for n_pre, n_post, prob in CONFIGS:
        for transpose in (False, True):
            for corder in (True, False):
                w_low = jnp.zeros(1, dtype=dtype)
                w_high = jnp.ones(1, dtype=dtype)
                clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
                v_size = n_post if not transpose else n_pre
                vector = jnp.asarray(np.random.randn(v_size), dtype=dtype)
                seed = jnp.asarray([42], dtype=jnp.uint32)
                name = (
                    f"{'T' if transpose else 'NT'},"
                    f"{'corder' if corder else 'rorder'},"
                    f"{n_pre}x{n_post},p={prob}"
                )
                yield BenchmarkConfig(
                    name=name,
                    args=(w_low, w_high, clen, vector, seed),
                    kernel_kwargs={'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder},
                    data_kwargs={'n_pre': n_pre, 'n_post': n_post, 'prob': prob},
                )


def main():
    parser = argparse.ArgumentParser(description="jitumv backend benchmark")
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--backends", nargs="+", default=["pallas", 'tvmffi'],
                        help="Backends to benchmark (default: tvmffi pallas)")
    args = parser.parse_args()

    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        print("ERROR: No GPU device found.")
        return

    print(f"jitumv benchmark  —  GPU: {gpu}")
    print(f"warmup={args.n_warmup}  runs={args.n_runs}  backends={args.backends}")

    jitumv_p.def_benchmark_data(_make_benchmark_data)

    result = jitumv_p.benchmark(
        platform='gpu',
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        compare_results=False,
        verbose=True,
        backends=args.backends,
    )
    result.print(vary_by='backend', highlight_best=True, speedup_vs='pallas')


if __name__ == "__main__":
    main()
