"""
JIT Scalar Binary Event-Driven Matrix-Matrix Benchmark
======================================================

Benchmarks all available GPU backends for ``binary_jitsmm`` (gather and scatter
modes, bool and float events) across a range of problem sizes.

Usage
-----
    python dev/jits/benchmark_binary_jitsmm.py
    python dev/jits/benchmark_binary_jitsmm.py --n_warmup 5 --n_runs 50
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

from brainevent import binary_jitsmm_p, BenchmarkConfig

CONFIGS = [
    (1000, 1000, 0.1, 10),
    (5000, 5000, 0.1, 10),
    (10000, 10000, 0.1, 10),
    (10000, 10000, 0.01, 10),
]


def _make_benchmark_data(*, platform):
    dtype = jnp.float32
    rng = np.random.default_rng(42)
    for n_pre, n_post, prob, n_cols in CONFIGS:
        for transpose in (False, True):
            for corder in (True, False):
                for bool_event in (True, False):
                    weight = jnp.ones(1, dtype=dtype)
                    clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
                    b_rows = n_post if not transpose else n_pre
                    if bool_event:
                        B = jnp.asarray(rng.random((b_rows, n_cols)) > 0.5, dtype=jnp.bool_)
                    else:
                        B = jnp.asarray(rng.random((b_rows, n_cols)), dtype=dtype)
                    seed = jnp.asarray([42], dtype=jnp.int32)
                    name = (
                        f"{'T' if transpose else 'NT'},"
                        f"{'corder' if corder else 'rorder'},"
                        f"{'bool' if bool_event else 'float'},"
                        f"{n_pre}x{n_post},p={prob},n={n_cols}"
                    )
                    yield BenchmarkConfig(
                        name=name,
                        args=(weight, clen, B, seed),
                        kernel_kwargs={'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder},
                        data_kwargs={'n_pre': n_pre, 'n_post': n_post, 'prob': prob},
                    )


def main():
    parser = argparse.ArgumentParser(description="binary_jitsmm backend benchmark")
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_runs", type=int, default=1)
    args = parser.parse_args()

    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        print("ERROR: No GPU device found.")
        return

    print(f"binary_jitsmm benchmark  —  GPU: {gpu}")
    print(f"warmup={args.n_warmup}  runs={args.n_runs}")

    binary_jitsmm_p.def_benchmark_data(_make_benchmark_data)

    result = binary_jitsmm_p.benchmark(
        platform='gpu',
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        compare_results=False,
        verbose=True,
        backends=['tvmffi'],
    )
    result.print(vary_by='backend', highlight_best=True)


if __name__ == "__main__":
    main()
