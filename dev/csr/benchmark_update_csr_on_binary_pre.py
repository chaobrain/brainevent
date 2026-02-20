"""
CSR Pre-Synaptic Plasticity Update Benchmark
=============================================

Benchmarks all available backends for ``update_csr_on_binary_pre``:

    For each active pre neuron i:
        weight[indptr[i]:indptr[i+1]] += post_trace[indices[indptr[i]:indptr[i+1]]]

This is the STDP pre-synaptic weight update for CSR sparse connectivity.

Usage
-----
    python dev/csr/benchmark_update_csr_on_binary_pre.py
    python dev/csr/benchmark_update_csr_on_binary_pre.py --n_warmup 10 --n_runs 100
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

from brainevent import BenchmarkConfig
from brainevent._csr.plasticity_binary import update_csr_on_binary_pre_p

# (n_pre, n_post, conn_prob, spike_density)
CONFIGS = [
    (500,   500,   0.01, 0.01),
    (500,   500,   0.10, 0.10),
    (1000,  1000,  0.001, 0.001),
    (1000,  1000,  0.01, 0.01),
    (1000,  1000,  0.10, 0.10),
    (5000,  5000,  0.001, 0.001),
    (5000,  5000,  0.01, 0.01),
    (5000,  5000,  0.10, 0.10),
    (10000, 10000, 0.001, 0.001),
    (10000, 10000, 0.01, 0.01),
]


def _make_benchmark_data(*, platform):
    rng = np.random.default_rng(42)
    dtype = jnp.float32
    for n_pre, n_post, prob, density in CONFIGS:
        n_conn = max(1, int(n_post * prob))
        nse = n_pre * n_conn
        indptr = jnp.asarray(np.arange(n_pre + 1, dtype=np.int32) * n_conn)
        indices = jnp.asarray(rng.integers(0, n_post, nse, dtype=np.int32))
        weight = jnp.asarray(rng.standard_normal(nse), dtype=dtype)
        post_trace = jnp.asarray(rng.standard_normal(n_post), dtype=dtype)
        for bool_event in (True, False):
            if bool_event:
                pre_spike = jnp.asarray(rng.random(n_pre) < density, dtype=jnp.bool_)
            else:
                raw = rng.standard_normal(n_pre)
                mask = rng.random(n_pre) < density
                pre_spike = jnp.asarray(np.where(mask, np.abs(raw), 0.0), dtype=dtype)
            nnz = int((pre_spike > 0).sum()) if not bool_event else int(pre_spike.sum())
            name = (
                f"{'bool' if bool_event else 'float'},"
                f"d={density:.1%},"
                f"{n_pre}x{n_post},nnz={nnz}"
            )
            yield BenchmarkConfig(
                name=name,
                args=(weight, indices, indptr, pre_spike, post_trace),
                kernel_kwargs={'shape': (n_pre, n_post)},
                data_kwargs={
                    'n_pre': n_pre, 'n_post': n_post, 'prob': prob, 'density': density,
                },
            )


def main():
    parser = argparse.ArgumentParser(description="update_csr_on_binary_pre benchmark")
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_runs", type=int, default=50)
    args = parser.parse_args()

    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        print("ERROR: No GPU device found.")
        return

    print(f"update_csr_on_binary_pre benchmark  —  GPU: {gpu}")
    print(f"warmup={args.n_warmup}  runs={args.n_runs}")
    print()

    update_csr_on_binary_pre_p.def_benchmark_data(_make_benchmark_data)

    result = update_csr_on_binary_pre_p.benchmark(
        platform='gpu',
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        compare_results=True,
        verbose=True,
    )
    result.print(vary_by='backend', highlight_best=True, speedup_vs='cusparse')


if __name__ == "__main__":
    main()
