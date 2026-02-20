"""
JIT Uniform Binary Event-Driven Matrix-Vector Benchmark
=======================================================

Benchmarks all available GPU backends for ``binary_jitumv`` (gather and scatter
modes, bool and float events) across a range of problem sizes and probabilities.

Usage
-----
    python dev/jitu/benchmark_binary_jitumv.py
    python dev/jitu/benchmark_binary_jitumv.py --n_warmup 5 --n_runs 50
    python dev/jitu/benchmark_binary_jitumv.py --backends tvmffi pallas
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

from brainevent import binary_jitumv_p, BenchmarkConfig

CONFIGS = [
    (1000, 1000, 0.1),
    (5000, 5000, 0.1),
    (10000, 10000, 0.1),
    (10000, 10000, 0.01),
]


def _make_benchmark_data(*, platform):
    dtype = jnp.float32
    rng = np.random.default_rng(42)
    for n_pre, n_post, prob in CONFIGS:
        for transpose in (False, True):
            for corder in (True, False):
                for bool_event in (True, False):
                    w_low = jnp.zeros(1, dtype=dtype)
                    w_high = jnp.ones(1, dtype=dtype)
                    clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
                    v_size = n_post if not transpose else n_pre
                    if bool_event:
                        vector = jnp.asarray(rng.random(v_size) > 0.5, dtype=jnp.bool_)
                    else:
                        vector = jnp.asarray(rng.random(v_size), dtype=dtype)
                    seed = jnp.asarray([42], dtype=jnp.uint32)
                    name = (
                        f"{'T' if transpose else 'NT'},"
                        f"{'corder' if corder else 'rorder'},"
                        f"{'bool' if bool_event else 'float'},"
                        f"{n_pre}x{n_post},p={prob}"
                    )
                    yield BenchmarkConfig(
                        name=name,
                        args=(w_low, w_high, clen, vector, seed),
                        kernel_kwargs={'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder},
                        data_kwargs={'n_pre': n_pre, 'n_post': n_post, 'prob': prob},
                    )


def main():
    parser = argparse.ArgumentParser(description="binary_jitumv backend benchmark")
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--backends", nargs="+", default=["tvmffi", ],
                        help="Backends to benchmark (default: tvmffi pallas)")
    args = parser.parse_args()

    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        print("ERROR: No GPU device found.")
        return

    print(f"binary_jitumv benchmark  —  GPU: {gpu}")
    print(f"warmup={args.n_warmup}  runs={args.n_runs}  backends={args.backends}")

    binary_jitumv_p.def_benchmark_data(_make_benchmark_data)

    result = binary_jitumv_p.benchmark(
        platform='gpu',
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        compare_results=False,
        verbose=True,
        backends=args.backends,
    )
    result.print(vary_by='backend', highlight_best=True)


if __name__ == "__main__":
    main()
