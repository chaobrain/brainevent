"""
CSR Row Slicing Benchmark
=========================

Benchmarks all available backends for ``csr_slice_rows`` (extracting selected
rows from a CSR sparse matrix into a dense submatrix) across a range of
matrix sizes, connection probabilities, and numbers of selected rows.

Usage
-----
    python dev/csr/benchmark_csr_slice_rows.py
    python dev/csr/benchmark_csr_slice_rows.py --n_warmup 10 --n_runs 50
    python dev/csr/benchmark_csr_slice_rows.py --num_selected 64
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
from brainevent._csr.slice import csr_slice_rows_p

# (n_pre, n_post, conn_prob, num_selected)
CONFIGS = [
    (1000,   1000,  0.05,  16),
    (1000,   1000,  0.10,  64),
    (5000,   5000,  0.01,  32),
    (5000,   5000,  0.05,  128),
    (10000,  10000, 0.01,  64),
    (10000,  10000, 0.02,  256),
]


def _make_benchmark_data(*, platform, num_selected=None):
    brainstate.environ.set(precision=32)
    rng = np.random.default_rng(42)
    dtype = brainstate.environ.dftype()
    for n_pre, n_post, prob, default_sel in CONFIGS:
        n_sel = num_selected if num_selected is not None else default_sel
        n_conn = max(1, int(n_post * prob))
        nse = n_pre * n_conn
        indptr = jnp.asarray(np.arange(n_pre + 1, dtype=np.int32) * n_conn)
        indices = jnp.asarray(rng.integers(0, n_post, nse, dtype=np.int32))
        row_indices = jnp.asarray(
            rng.integers(0, n_pre, n_sel, dtype=np.int32)
        )
        for homo in (True, False):
            if homo:
                data = jnp.ones(1, dtype=dtype)
            else:
                data = jnp.asarray(rng.standard_normal(nse), dtype=dtype)
            name = (
                f"{'homo' if homo else 'hetero'},"
                f"{n_pre}x{n_post},p={prob},sel={n_sel}"
            )
            yield BenchmarkConfig(
                name=name,
                args=(data, indices, indptr, row_indices),
                kernel_kwargs={'shape': (n_pre, n_post)},
                data_kwargs={
                    'n_pre': n_pre, 'n_post': n_post, 'prob': prob,
                    'num_selected': n_sel,
                },
            )


def main():
    parser = argparse.ArgumentParser(description="csr_slice_rows backend benchmark")
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_runs", type=int, default=30)
    parser.add_argument("--num_selected", type=int, default=None,
                        help="Number of rows to extract (overrides per-config defaults)")
    args = parser.parse_args()

    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        print("ERROR: No GPU device found.")
        return

    print(f"csr_slice_rows benchmark  —  GPU: {gpu}")
    print(f"warmup={args.n_warmup}  runs={args.n_runs}"
          + (f"  num_selected={args.num_selected}" if args.num_selected else ""))
    print()

    def _data_gen(*, platform):
        yield from _make_benchmark_data(platform=platform, num_selected=args.num_selected)

    csr_slice_rows_p.def_benchmark_data(_data_gen)

    result = csr_slice_rows_p.benchmark(
        platform='gpu',
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        compare_results=True,
        verbose=True,
    )
    result.print(vary_by='backend', highlight_best=True, speedup_vs='pallas')


if __name__ == "__main__":
    main()
