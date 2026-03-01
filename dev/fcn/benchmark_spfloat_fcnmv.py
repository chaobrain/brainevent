"""
Sparse-Float FCN Matrix-Vector Multiplication Benchmark
=======================================================

Benchmarks all available GPU backends for ``spfloat_fcnmv`` (gather and scatter
modes, homo and hetero weights, various spike rates) across a range of problem
sizes.  The primary interest is the sparsity-exploitation advantage of the CUDA
(cuda_raw) backend over dense-style kernels at low firing rates.

Usage
-----
    python dev/fcn/benchmark_spfloat_fcnmv.py
    python dev/fcn/benchmark_spfloat_fcnmv.py --n_warmup 5 --n_runs 50
    python dev/fcn/benchmark_spfloat_fcnmv.py --spike_rates 0.01 0.05 0.1 0.5
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

from brainevent import BenchmarkConfig, spfloat_fcnmv_p

current_name = 'spfloat_fcnmv'
benchmark_data_type = 'typeA'
DEFAULT_SPIKE_RATES = [0.01, 0.05, 0.10, 0.50]
config_type = "config_2"
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
runs = parsed_config.get('runs', 10)
warmup = parsed_config.get('warmup', 10)
batch = parsed_config.get('batch', 10)
base_len_config = len(matrix_configs) * len(transpose_list) * len(homo_list)

def _make_benchmark_data(*, platform, spike_rates=None):
    brainstate.environ.set(precision=16) 
    if spike_rates is None:
        spike_rates = default_spike_rates
        
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
                
                for rate in spike_rates:
                    v_raw = rng.standard_normal(v_size)
                    mask = rng.random(v_size) < rate
                    vector = jnp.asarray(v_raw * mask, dtype=dtype)
                    
                    name = (
                        f"{'T' if transpose else 'NT'},"
                        f"{'homo' if homo else 'hetero'},"
                        f"{n_pre}x{n_post}x{prob},"
                        f"rate={rate:.0%}"
                    )
                    yield BenchmarkConfig(
                        name=name,
                        args=(weights, indices, vector),
                        kernel_kwargs={'shape': (n_pre, n_post), 'transpose': transpose},
                        data_kwargs={
                            'n_pre': n_pre,
                            'n_post': n_post,
                            'prob': prob,
                            'transpose': transpose,
                            'spike_rate': rate,
                        },
                    )

try:
    gpu = jax.devices("gpu")[0]
except RuntimeError:
    print("ERROR: No GPU device found.")


print(f"spfloat_fcnmv benchmark  —  GPU: {gpu}")
print(f"warmup={warmup}  runs={runs}")
print(f"spike_rates={default_spike_rates}")

def _data_gen(*, platform):
    yield from _make_benchmark_data(platform=platform, spike_rates=default_spike_rates)

spfloat_fcnmv_p.def_benchmark_data(_data_gen)

total_len_config = base_len_config * len(default_spike_rates)

result = spfloat_fcnmv_p.benchmark_csv_output(
    platform='gpu',
    n_warmup=warmup,
    n_runs=runs,
    n_batch_per_run=batch,
    compare_results=True,
    verbose=False,
    len_config = total_len_config
)

result.print(vary_by='backend', highlight_best=True, speedup_vs='jax_raw')
