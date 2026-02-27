"""
FCN Matrix-Vector Multiplication Benchmark
==========================================

Benchmarks all available GPU backends for ``fcnmv`` (gather and scatter modes,
homo and hetero weights) across a range of problem sizes.

Usage
-----
    python dev/fcn/benchmark_fcnmv.py
    python dev/fcn/benchmark_fcnmv.py --n_warmup 5 --n_runs 50
"""

import argparse
import sys
from pathlib import Path

import json

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import jax
import jax.numpy as jnp
import numpy as np

import brainstate

from brainevent import fcnmv_p, BenchmarkConfig

current_name = 'fcnmv'
benchmark_data_type = 'typeA'
config_type = "config_1"

def load_benchmark_config(json_path: str, benchmark_data_type: str, operator_name: str, config_key: str = config_type) -> dict:
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
        
    if benchmark_data_type not in raw_data:
        raise KeyError(f"Type '{benchmark_data_type}' not found in configuration file.")
        
    if operator_name not in raw_data[benchmark_data_type]["operator"]:
        raise KeyError(f"operator '{benchmark_data_type}' not found in configuration file.")
    
    operator_data = raw_data[benchmark_data_type]

    if config_key not in operator_data:
        raise KeyError(f"Configuration block '{config_key}' not found config_key '{config_key}'.")
        
    return operator_data[config_key]

config_file_path = 'benchmark_config.json'
parsed_config = load_benchmark_config(config_file_path, benchmark_data_type, current_name)

dist_type = parsed_config.get('dist_type', 'uniform')
transpose_list = parsed_config.get('transpose', [False, True])
homo_list = parsed_config.get('homo_weight', [True, False])
matrix_configs = parsed_config.get('configs', [])

len_config = len(matrix_configs) * len(transpose_list) * len(homo_list)

def _make_benchmark_data(*, platform):
    brainstate.environ.set(precision=32)  # change to 16 or 64 for other precisions
    rng = np.random.default_rng(42)
    dtype = brainstate.environ.dftype()
    
    for cfg in matrix_configs:
        n_pre = cfg['n_rows']
        n_post = cfg['n_cols']
        prob = cfg['density']

        n_conn = max(1, int(n_post * prob))
        indices = jnp.asarray(rng.integers(0, n_post, (n_pre, n_conn), dtype=np.int32))
        for transpose in transpose_list:
            for homo in homo_list:
                if homo:
                    weights = jnp.ones(1, dtype=dtype)
                else:
                    weights = jnp.asarray(rng.standard_normal((n_pre, n_conn)), dtype=dtype)
                v_size = n_post if not transpose else n_pre
                vector = jnp.asarray(rng.standard_normal(v_size), dtype=dtype)
                name = (
                    f"{'T' if transpose else 'NT'},"
                    f"{'homo' if homo else 'hetero'},"
                    f"{n_pre}x{n_post}x{prob}"
                )
                yield BenchmarkConfig(
                    name=name,
                    args=(weights, indices, vector),
                    kernel_kwargs={'shape': (n_pre, n_post), 'transpose': transpose},
                    data_kwargs={'n_pre': n_pre, 'n_post': n_post, 'prob': prob},
                )


def main():
    parser = argparse.ArgumentParser(description="fcnmv backend benchmark")
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--n_batch_per_run", type=int, default=10)
    args = parser.parse_args()

    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        print("ERROR: No GPU device found.")
        return

    print(f"fcnmv benchmark  —  GPU: {gpu}")
    print(f"warmup={args.n_warmup}  runs={args.n_runs}")

    fcnmv_p.def_benchmark_data(_make_benchmark_data)

    result = fcnmv_p.benchmark_csv_output(
        platform='gpu',
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        n_batch_per_run = args.n_batch_per_run,
        compare_results=True,
        verbose=False,
        len_config = len_config
    )
    # result.print(order_by=['transpose', 'shape', 'backend'], highlight_best=True, speedup_vs='jax_raw')
    result.print(vary_by='backend', highlight_best=True, speedup_vs='jax_raw')


if __name__ == "__main__":
    main()
