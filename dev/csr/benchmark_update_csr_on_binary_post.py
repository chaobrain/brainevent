"""
CSR Post-Synaptic Plasticity Update Benchmark
=============================================

Benchmarks all available backends for ``update_csr_on_binary_post``:

    For each active post neuron j:
        weight[weight_indices[indptr[j]:indptr[j+1]]] +=
            pre_trace[indices[indptr[j]:indptr[j+1]]]

This is the STDP post-synaptic weight update stored in CSC order
(CSR over columns) with a weight_indices array mapping back to CSR storage.

Usage
-----
    python dev/csr/benchmark_update_csr_on_binary_post.py
    python dev/csr/benchmark_update_csr_on_binary_post.py --n_warmup 10 --n_runs 100
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
from brainevent._csr.plasticity_binary import update_csr_on_binary_post_p

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
        # CSC representation: n_post columns, each with n_conn row entries
        n_conn = max(1, int(n_pre * prob))
        nse = n_post * n_conn
        # indptr over columns (length n_post+1)
        indptr = jnp.asarray(np.arange(n_post + 1, dtype=np.int32) * n_conn)
        # indices = row (pre-neuron) indices in CSC order
        indices = jnp.asarray(rng.integers(0, n_pre, nse, dtype=np.int32))
        # weight array and mapping from CSC entries back to CSR weight storage
        weight = jnp.asarray(rng.standard_normal(nse), dtype=dtype)
        weight_indices = jnp.asarray(rng.integers(0, nse, nse, dtype=np.int32))
        pre_trace = jnp.asarray(rng.standard_normal(n_pre), dtype=dtype)
        for bool_event in (True, False):
            if bool_event:
                post_spike = jnp.asarray(rng.random(n_post) < density, dtype=jnp.bool_)
            else:
                raw = rng.standard_normal(n_post)
                mask = rng.random(n_post) < density
                post_spike = jnp.asarray(np.where(mask, np.abs(raw), 0.0), dtype=dtype)
            nnz = int((post_spike > 0).sum()) if not bool_event else int(post_spike.sum())
            name = (
                f"{'bool' if bool_event else 'float'},"
                f"d={density:.1%},"
                f"{n_pre}x{n_post},nnz={nnz}"
            )
            yield BenchmarkConfig(
                name=name,
                args=(weight, indices, indptr, weight_indices, pre_trace, post_spike),
                kernel_kwargs={'shape': (n_pre, n_post)},
                data_kwargs={
                    'n_pre': n_pre, 'n_post': n_post, 'prob': prob, 'density': density,
                },
            )


def main():
    parser = argparse.ArgumentParser(description="update_csr_on_binary_post benchmark")
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_runs", type=int, default=50)
    args = parser.parse_args()

    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        print("ERROR: No GPU device found.")
        return

    print(f"update_csr_on_binary_post benchmark  —  GPU: {gpu}")
    print(f"warmup={args.n_warmup}  runs={args.n_runs}")
    print()

    update_csr_on_binary_post_p.def_benchmark_data(_make_benchmark_data)

    result = update_csr_on_binary_post_p.benchmark(
        platform='gpu',
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        compare_results=True,
        verbose=True,
    )
    result.print(vary_by='backend', highlight_best=True, speedup_vs='jax_raw')


if __name__ == "__main__":
    main()
