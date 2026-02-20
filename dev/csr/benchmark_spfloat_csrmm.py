"""
Sparse-Float CSR Matrix-Matrix Multiplication Benchmark
========================================================

Benchmarks all available backends for ``spfloat_csrmm`` (sparse-float CSR
SpMM where the input matrix is sparse) across a range of problem sizes,
column counts, and input sparsities.

Usage
-----
    python dev/csr/benchmark_spfloat_csrmm.py
    python dev/csr/benchmark_spfloat_csrmm.py --n_warmup 10 --n_runs 50
    python dev/csr/benchmark_spfloat_csrmm.py --density 0.05
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
from brainevent._csr.sparse_float import spfloat_csrmm_p

# (n_pre, n_post, conn_prob, n_col)
CONFIGS = [
    (1000,  1000,  0.05,  32),
    (1000,  1000,  0.10,  128),
    (5000,  5000,  0.01,  64),
    (5000,  5000,  0.05,  128),
    (10000, 10000, 0.01,  64),
    (10000, 10000, 0.02,  128),
]


def _make_benchmark_data(*, platform, density=0.1):
    brainstate.environ.set(precision=32)
    rng = np.random.default_rng(42)
    dtype = brainstate.environ.dftype()
    for n_pre, n_post, prob, n_col in CONFIGS:
        n_conn = max(1, int(n_post * prob))
        nse = n_pre * n_conn
        indptr = jnp.asarray(np.arange(n_pre + 1, dtype=np.int32) * n_conn)
        indices = jnp.asarray(rng.integers(0, n_post, nse, dtype=np.int32))
        for transpose in (False, True):
            for homo in (True, False):
                if homo:
                    weights = jnp.ones(1, dtype=dtype)
                else:
                    weights = jnp.asarray(rng.standard_normal(nse), dtype=dtype)
                b_rows = n_post if not transpose else n_pre
                # Sparse float matrix: many zeros
                raw = rng.standard_normal((b_rows, n_col))
                mask = rng.random((b_rows, n_col)) < density
                matrix = jnp.asarray(np.where(mask, raw, 0.0), dtype=dtype)
                name = (
                    f"{'T' if transpose else 'NT'},"
                    f"{'homo' if homo else 'hetero'},"
                    f"{n_pre}x{n_post},p={prob},ncol={n_col},d={density:.0%}"
                )
                yield BenchmarkConfig(
                    name=name,
                    args=(weights, indices, indptr, matrix),
                    kernel_kwargs={'shape': (n_pre, n_post), 'transpose': transpose},
                    data_kwargs={
                        'n_pre': n_pre, 'n_post': n_post, 'prob': prob,
                        'n_col': n_col, 'density': density,
                    },
                )


def main():
    parser = argparse.ArgumentParser(description="spfloat_csrmm backend benchmark")
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_runs", type=int, default=30)
    parser.add_argument("--density", type=float, default=0.1,
                        help="Fraction of nonzero input matrix entries (default: 0.1 = 10%%)")
    args = parser.parse_args()

    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        print("ERROR: No GPU device found.")
        return

    print(f"spfloat_csrmm benchmark  —  GPU: {gpu}")
    print(f"warmup={args.n_warmup}  runs={args.n_runs}  density={args.density:.0%}")
    print()

    def _data_gen(*, platform):
        yield from _make_benchmark_data(platform=platform, density=args.density)

    spfloat_csrmm_p.def_benchmark_data(_data_gen)

    result = spfloat_csrmm_p.benchmark(
        platform='gpu',
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        compare_results=True,
        verbose=True,
    )
    result.print(vary_by='backend', highlight_best=True, speedup_vs='jax_raw')


if __name__ == "__main__":
    main()
