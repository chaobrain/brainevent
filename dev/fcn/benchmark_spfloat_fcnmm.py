"""
Sparse-Float FCN Matrix-Matrix Multiplication Benchmark
=======================================================

Benchmarks all available GPU backends for ``spfloat_fcnmm`` (gather and scatter
modes, homo and hetero weights, various spike rates) across a range of problem
sizes and column counts.

The primary interest is the sparsity-exploitation advantage of the CUDA (tvmffi)
backend: gather kernels use ``__ballot_sync`` to skip all-zero warp chunks;
scatter kernels use tile-level early exit via warp ballot + ``atomicOr`` on a
shared flag.

Usage
-----
    python dev/fcn/benchmark_spfloat_fcnmm.py
    python dev/fcn/benchmark_spfloat_fcnmm.py --n_warmup 5 --n_runs 50
    python dev/fcn/benchmark_spfloat_fcnmm.py --spike_rates 0.01 0.05 0.10 0.50
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
from brainevent import BenchmarkConfig, spfloat_fcnmm_p

# (n_pre, n_post, n_conn, n_col)
# n_col drives the second dimension of the dense matrix M.
# Include vec4-friendly (n_col % 4 == 0) and non-aligned sizes.
CONFIGS = [
    (1000,  1000,  50,   32),
    (1000,  1000, 100,  128),
    (2000,  2000, 100,   64),
    (5000,  5000, 200,  128),
    (5000,  5000, 500,   64),
    (5000,  5000,  50,  512),
    (10000, 10000, 100, 128),
    (10000, 10000, 200,  64),
    (10000, 10000,  50, 256),
]

DEFAULT_SPIKE_RATES = [0.01, 0.05, 0.10, 0.50]


def _make_benchmark_data(*, platform, spike_rates=None):
    brainstate.environ.set(precision=32)  # change to 16 or 64 for other precisions
    if spike_rates is None:
        spike_rates = DEFAULT_SPIKE_RATES
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
                # Number of rows in M depends on which dimension W contracts over:
                #   transpose=False: Y[n_pre, n_col] = W[n_pre,n_post] @ M[n_post, n_col]
                #   transpose=True:  Y[n_post,n_col] = W^T[n_post,n_pre] @ M[n_pre, n_col]
                m_rows = n_post if not transpose else n_pre
                for rate in spike_rates:
                    m_raw = rng.standard_normal((m_rows, n_col))
                    mask = (rng.random((m_rows, n_col)) < rate)
                    matrix = jnp.asarray(m_raw * mask, dtype=dtype)
                    name = (
                        f"{'T' if transpose else 'NT'},"
                        f"{'homo' if homo else 'hetero'},"
                        f"{n_pre}x{n_post}x{n_conn},"
                        f"ncol={n_col},"
                        f"rate={rate:.0%}"
                    )
                    yield BenchmarkConfig(
                        name=name,
                        args=(weights, indices, matrix),
                        kernel_kwargs={'shape': (n_pre, n_post), 'transpose': transpose},
                        data_kwargs={
                            'n_pre': n_pre,
                            'n_post': n_post,
                            'n_conn': n_conn,
                            'n_col': n_col,
                            'transpose': transpose,
                            'spike_rate': rate,
                        },
                    )


def main():
    parser = argparse.ArgumentParser(description="spfloat_fcnmm backend benchmark")
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_runs", type=int, default=20)
    parser.add_argument(
        "--spike_rates",
        type=float,
        nargs="+",
        default=DEFAULT_SPIKE_RATES,
        metavar="RATE",
        help="Firing rates to sweep (e.g. 0.01 0.05 0.10 0.50)",
    )
    args = parser.parse_args()

    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        print("ERROR: No GPU device found.")
        return

    print(f"spfloat_fcnmm benchmark  —  GPU: {gpu}")
    print(f"warmup={args.n_warmup}  runs={args.n_runs}")
    print(f"spike_rates={args.spike_rates}")

    def _data_gen(*, platform):
        yield from _make_benchmark_data(platform=platform, spike_rates=args.spike_rates)

    spfloat_fcnmm_p.def_benchmark_data(_data_gen)

    result = spfloat_fcnmm_p.benchmark(
        platform='gpu',
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        n_batch_per_run=1,
        compare_results=True,
        verbose=True,
    )
    # result.print(
    #     order_by=['transpose', 'spike_rate', 'shape', 'backend'],
    #     highlight_best=True,
    #     speedup_vs='jax_raw',
    # )
    result.print(vary_by='backend', highlight_best=True, speedup_vs='jax_raw')


if __name__ == "__main__":
    main()
