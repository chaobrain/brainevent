"""
COO Pre-Synaptic Plasticity Update Benchmark
=============================================

Benchmarks all available GPU backends for ``update_coo_on_binary_pre``:

    For each synapse i:
        if pre_spike[pre_ids[i]]:
            weight[i] += post_trace[post_ids[i]]

This is the STDP pre-synaptic weight update for COO sparse connectivity.

Backends compared:
  pallas    -- JAX Pallas/Triton GPU kernel (block-based, default GPU backend)
  tvmffi    -- TVM FFI CUDA kernel (warp-ballot early exit; optimised for sparsity)
  jax       -- Pure-JAX reference (outer-product-like gather, no event-driven opt)

Problem configurations:
  (n_pre x n_post, conn_prob, spike_density)

Spike densities:
  0.1%  -- ultra-sparse (cortical-scale simulation)
  1%    -- sparse       (typical SNN)
  10%   -- moderate

Usage
-----
    python dev/coo/benchmark_update_coo_on_binary_pre.py
    python dev/coo/benchmark_update_coo_on_binary_pre.py --n_warmup 10 --n_runs 100
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
from brainevent._coo.plasticity_binary import update_coo_on_binary_pre_p

# ---------------------------------------------------------------------------
# Benchmark configurations
# (n_pre, n_post, conn_prob, spike_density)
# ---------------------------------------------------------------------------
CONFIGS = [
    # Small networks
    (500,    500,   0.10,  0.001),
    (500,    500,   0.10,  0.01),
    (500,    500,   0.10,  0.10),
    # Medium networks
    (1000,   1000,  0.01,  0.001),
    (1000,   1000,  0.01,  0.01),
    (1000,   1000,  0.01,  0.10),
    (1000,   1000,  0.10,  0.001),
    (1000,   1000,  0.10,  0.01),
    (1000,   1000,  0.10,  0.10),
    # Large networks
    (5000,   5000,  0.001, 0.001),
    (5000,   5000,  0.001, 0.01),
    (5000,   5000,  0.001, 0.10),
    (5000,   5000,  0.01,  0.001),
    (5000,   5000,  0.01,  0.01),
    (5000,   5000,  0.01,  0.10),
    # Extra-large (sparse connectivity, sparse spikes)
    (10000,  10000, 0.001, 0.001),
    (10000,  10000, 0.001, 0.01),
    (10000,  10000, 0.01,  0.001),
    (10000,  10000, 0.01,  0.01),
    (10000,  10000, 0.01,  0.10),
]


def _make_benchmark_data(*, platform):
    """Yield BenchmarkConfig entries for all (shape, density, dtype) combinations."""
    rng = np.random.default_rng(42)
    dtype = jnp.float32
    for n_pre, n_post, prob, density in CONFIGS:
        nnz = max(1, int(n_pre * n_post * prob))
        pre_ids = jnp.asarray(rng.integers(0, n_pre, nnz, dtype=np.int32))
        post_ids = jnp.asarray(rng.integers(0, n_post, nnz, dtype=np.int32))
        weight = jnp.asarray(rng.standard_normal(nnz), dtype=dtype)
        post_trace = jnp.asarray(rng.standard_normal(n_post), dtype=dtype)

        for bool_event in (True, False):
            if bool_event:
                pre_spike = jnp.asarray(rng.random(n_pre) < density, dtype=jnp.bool_)
            else:
                raw = rng.standard_normal(n_pre)
                mask = rng.random(n_pre) < density
                pre_spike = jnp.asarray(np.where(mask, np.abs(raw), 0.0), dtype=dtype)

            nnz_spk = int((pre_spike > 0).sum()) if not bool_event else int(pre_spike.sum())
            name = (
                f"{'bool' if bool_event else 'float'},"
                f"d={density:.1%},"
                f"{n_pre}x{n_post}@{prob:.1%},"
                f"nnz_spk={nnz_spk}"
            )
            yield BenchmarkConfig(
                name=name,
                args=(weight, pre_ids, post_ids, pre_spike, post_trace),
                data_kwargs={
                    'n_pre': n_pre, 'n_post': n_post,
                    'prob': prob, 'density': density,
                    'nnz': nnz, 'bool_event': bool_event,
                },
            )


def main():
    parser = argparse.ArgumentParser(description="update_coo_on_binary_pre GPU benchmark")
    parser.add_argument("--n_warmup", type=int, default=10,
                        help="Number of warm-up iterations per kernel (default: 10)")
    parser.add_argument("--n_runs", type=int, default=50,
                        help="Number of timed iterations per kernel (default: 50)")
    parser.add_argument("--no_compare", action="store_true",
                        help="Skip result correctness comparison")
    args = parser.parse_args()

    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        print("ERROR: No GPU device found. Run this benchmark on a machine with a CUDA GPU.")
        return

    print(f"update_coo_on_binary_pre GPU benchmark")
    print(f"  GPU     : {gpu}")
    print(f"  warmup  : {args.n_warmup}  runs: {args.n_runs}")
    print(f"  backends: numba (CPU only), pallas, tvmffi, jax")
    print()

    update_coo_on_binary_pre_p.def_benchmark_data(_make_benchmark_data)

    result = update_coo_on_binary_pre_p.benchmark(
        platform='gpu',
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        compare_results=not args.no_compare,
        verbose=True,
    )
    result.print(vary_by='backend', highlight_best=True, speedup_vs='jax_raw')


if __name__ == "__main__":
    main()
