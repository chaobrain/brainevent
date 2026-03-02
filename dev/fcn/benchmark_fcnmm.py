"""
FCN Matrix-Matrix Multiplication Benchmark
==========================================

Benchmarks all available GPU backends for ``fcnmm`` (gather and scatter modes,
homo and hetero weights) across a range of problem sizes and column counts.

Usage
-----
    python dev/fcn/benchmark_fcnmm.py
    python dev/fcn/benchmark_fcnmm.py --n_warmup 5 --n_runs 50
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
import json
import brainstate

from brainevent import fcnmm_p, BenchmarkConfig

# (n_pre, n_post, n_conn, n_col)
current_name = 'fcnmm'
benchmark_data_type = 'typeB'
config_type = "config_2"
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
runs = parsed_config.get('runs', 10)
warmup = parsed_config.get('warmup', 10)
batch = parsed_config.get('batch', 10)

len_config = len(matrix_configs) * len(transpose_list) * len(homo_list)

def _make_benchmark_data(*, platform):
    brainstate.environ.set(precision=32)  # change to 16 or 64 for other precisions
    rng = np.random.default_rng(42)
    dtype = brainstate.environ.dftype()
    for config in matrix_configs:

        n_pre = config['n_rows']
        n_post = config['n_cols']
        prob = config['density']
        n_col = config['n_col']

        n_conn = max(1, int(n_post * prob))
        indices = jnp.asarray(rng.integers(0, n_post, (n_pre, n_conn), dtype=np.int32))
        for transpose in transpose_list:
            for homo in homo_list:
                if homo:
                    weights = jnp.ones(1, dtype=dtype)
                else:
                    weights = jnp.asarray(rng.standard_normal((n_pre, n_conn)), dtype=dtype)
                b_rows = n_post if not transpose else n_pre
                matrix = jnp.asarray(rng.standard_normal((b_rows, n_col)), dtype=dtype)
                name = (
                    f"{'T' if transpose else 'NT'},"
                    f"{'homo' if homo else 'hetero'},"
                    f"{n_pre}x{n_post}x{prob},ncol={n_col}"
                )
                yield BenchmarkConfig(
                    name=name,
                    args=(weights, indices, matrix),
                    kernel_kwargs={'shape': (n_pre, n_post), 'transpose': transpose},
                    data_kwargs={
                        'n_pre': n_pre, 'n_post': n_post,
                        'prob': prob, 'n_col': n_col,
                    },
                )




try:
    gpu = jax.devices("gpu")[0]
except RuntimeError:
    print("ERROR: No GPU device found.")

print(f"fcnmm benchmark  —  GPU: {gpu}")
print(f"warmup={warmup}  runs={runs}")

fcnmm_p.def_benchmark_data(_make_benchmark_data)

result = fcnmm_p.benchmark_csv_output(
    platform='gpu',
    n_warmup=warmup,
    n_runs=runs,
    n_batch_per_run = batch,
    compare_results=True,
    verbose=False,
    len_config = len_config
)
result.print(vary_by='backend', highlight_best=True, speedup_vs='jax_raw')
