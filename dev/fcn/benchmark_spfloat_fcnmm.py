"""
Sparse-Float FCN Matrix-Matrix Multiplication Benchmark
=======================================================

Benchmarks all available GPU backends for ``spfloat_fcnmm`` (gather and scatter
modes, homo and hetero weights, various spike rates) across a range of problem
sizes and column counts.

The primary interest is the sparsity-exploitation advantage of the CUDA (cuda_raw)
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
'''
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
'''
import json
current_name = 'spfloat_fcnmm'
benchmark_data_type = 'typeB'
DEFAULT_SPIKE_RATES = [0.01, 0.05, 0.10, 0.50]
config_type = "config_1"
# Spike rates to sweep.  At lower rates, the CUDA kernel's early-exit
# optimisation should show the largest speedup over dense-style kernels.
def load_benchmark_config(json_path: str, benchmark_data_type: str, operator_name: str, config_key: str = config_type) -> dict:
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
        
    if benchmark_data_type not in raw_data:
        raise KeyError(f"Type '{benchmark_data_type}' not found in configuration file.")
        
    if operator_name not in raw_data[benchmark_data_type]["operator"]:
        raise KeyError(f"operator '{benchmark_data_type}' not found in configuration file.")
    
    operator_data = raw_data[benchmark_data_type]

    if config_key not in operator_data:
        raise KeyError(f"Configuration block '{config_key}' not found under operator '{operator_name}'.")
        
    return operator_data[config_key]

config_file_path = 'benchmark_config.json'
parsed_config = load_benchmark_config(config_file_path, benchmark_data_type, current_name)

dist_type = parsed_config.get('dist_type', 'uniform')
transpose_list = parsed_config.get('transpose', [False, True])
homo_list = parsed_config.get('homo_weight', [True, False])
matrix_configs = parsed_config.get('configs', [])
default_spike_rates = parsed_config.get('spike_rates', DEFAULT_SPIKE_RATES)

base_len_config = len(matrix_configs) * len(transpose_list) * len(homo_list)

def _make_benchmark_data(*, platform, spike_rates=None):
    brainstate.environ.set(precision=32)  # change to 16 or 64 for other precisions
    if spike_rates is None:
        spike_rates = default_spike_rates

    rng = np.random.default_rng(42)
    dtype = brainstate.environ.dftype()

    for cfg in matrix_configs:
        
        n_pre = cfg['n_rows']
        n_post = cfg['n_cols']
        prob = cfg['density']
        n_col = cfg['n_col']

        n_conn = max(1, int(n_post * prob))
        indices = jnp.asarray(rng.integers(0, n_post, (n_pre, n_conn), dtype=np.int32))

        for transpose in transpose_list:
            for homo in homo_list:
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
                        f"{n_pre}x{n_post}x{prob},"
                        f"n_col={n_col},"
                        f"rate={rate:.0%}"
                    )
                    yield BenchmarkConfig(
                        name=name,
                        args=(weights, indices, matrix),
                        kernel_kwargs={'shape': (n_pre, n_post), 'transpose': transpose},
                        data_kwargs={
                            'n_pre': n_pre,
                            'n_post': n_post,
                            'prob': prob,
                            'n_col': n_col,
                            'transpose': transpose,
                            'spike_rate': rate,
                        },
                    )


def main():
    parser = argparse.ArgumentParser(description="spfloat_fcnmm backend benchmark")
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_runs", type=int, default=20)
    parser.add_argument("--n_batch_per_run", type=int, default=10)
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

    total_len_config = base_len_config * len(args.spike_rates)

    result = spfloat_fcnmm_p.benchmark_csv_output(
        platform='gpu',
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        n_batch_per_run=args.n_batch_per_run,
        compare_results=True,
        verbose=False,
        len_config = total_len_config
    )
    # result.print(
    #     order_by=['transpose', 'spike_rate', 'shape', 'backend'],
    #     highlight_best=True,
    #     speedup_vs='jax_raw',
    # )
    result.print(vary_by='backend', highlight_best=True, speedup_vs='jax_raw')


if __name__ == "__main__":
    main()
