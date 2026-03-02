# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#
# Implementation of the paper:
#
# - Brette, R., Rudolph, M., Carnevale, T., Hines, M., Beeman, D., Bower, J. M., et al. (2007),
#   Simulation of networks of spiking neurons: a review of tools and strategies., J. Comput. Neurosci., 23, 3, 349–98
#
# which is based on the balanced network proposed by:
#
# - Vogels, T. P. and Abbott, L. F. (2005), Signal propagation and logic gating in networks of integrate-and-fire neurons., J. Neurosci., 25, 46, 10786–95
#


import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import jax
import brainunit as u

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import brainevent
from COBA_2005_benchmark import make_simulation_run

current_name = 'COBA_binary_fcnmv'
benchmark_data_type = 'typeC'
config_type = "config_1"
config_file_path = 'benchmark_config.json'

default_scale = (1,  4,  8,  20,  60, 100)
defaule_backen = ('cuda_raw', 'pallas', 'jax_raw')
conns = ('post', 'pre')
warmup = 2
runs = 3
duration = 1e4

def load_benchmark_config(json_path: str, benchmark_data_type: str, operator_name: str, config_key: str = config_type):
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
        
    if benchmark_data_type not in raw_data:
        raise KeyError(f"Type '{benchmark_data_type}' not found in configuration file.")
        
    if operator_name not in raw_data[benchmark_data_type]["operator"]:
        raise KeyError(f"operator '{benchmark_data_type}' not found in configuration file.")
    
    operator_data = raw_data[benchmark_data_type]

    if config_key not in operator_data:
        raise KeyError(f"Configuration block '{config_key}' not found under operator '{operator_name}'.")
  
    config = operator_data[config_key]
    
    if 'scale' in operator_data[config_key]:
        default_scale = tuple(config["scale"])

    if 'backends' in operator_data[config_key]:
        DEFAULT_BACKENDS = tuple(config["backends"])

    if 'conns' in operator_data[config_key]:
        DEFAULT_CONNS = tuple(config["conns"])

    if 'warmup'in operator_data[config_key]:
        WARMUP = config["warmup"]

    if 'runs'in operator_data[config_key]:
        RUNS = config["runs"]

    if 'duration'in operator_data[config_key]:
        DURATION = config["duration"]

def _benchmark_single_backend(
    backend: str,
    scales: Iterable[int],
    conn: str,
    duration_ms: float,
    warmup: int,
    runs: int,
) -> list[dict]:
    brainevent.config.set_backend('gpu', backend)
    rows: list[dict] = []
    for scale in scales:
        run = make_simulation_run(
            scale=scale,
            data_type='binary',
            efferent_target=conn,
            duration=duration_ms * u.ms,
        )
        for _ in range(warmup):
            jax.block_until_ready(run())
        for run_id in range(1, runs + 1):
            t0 = time.perf_counter()
            out = jax.block_until_ready(run())
            elapsed_s = time.perf_counter() - t0
            if isinstance(out, tuple) and len(out) == 2:
                n, rate = out
            else:
                n = int(4000 * scale)
                rate = out
            row = {
                'backend': backend,
                'conn': conn,
                'scale': int(scale),
                'run_id': run_id,
                'size': int(n),
                'firing_rate_hz': float(rate),
                'elapsed_s': elapsed_s,
            }
            rows.append(row)
            print(
                f"backend={backend}, conn={conn}, scale={scale}, run={run_id}/{runs}, "
                f"time={elapsed_s:.6f}s, firing_rate={float(rate):.6f}Hz"
            )
    return rows


def _summarize(rows: Iterable[dict], baseline_backend: str) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row['backend'], row['conn'], row['scale'])].append(row)

    summary: list[dict] = []
    for (backend, conn, scale), records in sorted(grouped.items()):
        times = [float(r['elapsed_s']) for r in records]
        rates = [float(r['firing_rate_hz']) for r in records]
        size = int(records[0]['size'])
        entry = {
            'backend': backend,
            'conn': conn,
            'scale': int(scale),
            'size': size,
            'runs': len(records),
            'elapsed_min_s': min(times),       
            'elapsed_max_s': max(times),        
            'elapsed_mean_s': statistics.fmean(times),
            'elapsed_std_s': statistics.pstdev(times) if len(times) > 1 else 0.0,
            'firing_rate_mean_hz': statistics.fmean(rates),
        }
        summary.append(entry)

    baseline_lookup = {
        (e['conn'], e['scale']): e['elapsed_mean_s']
        for e in summary
        if e['backend'] == baseline_backend
    }
    for entry in summary:
        baseline = baseline_lookup.get((entry['conn'], entry['scale']))
        if baseline is None:
            entry['speedup_vs_baseline'] = None
        else:
            entry['speedup_vs_baseline'] = baseline / entry['elapsed_mean_s']
    return summary


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main() -> None:
    parser = argparse.ArgumentParser(description='Run and compare COBA_2005 binary_fcnmv backends.')
    parser.add_argument('--backends', nargs='+', default=list(defaule_backen))
    parser.add_argument('--scales', nargs='+', type=int, default=list(default_scale))
    parser.add_argument('--conns', nargs='+', choices=['post', 'pre'], default=list(conns))
    parser.add_argument('--warmup', type=int, default=warmup)
    parser.add_argument('--runs', type=int, default=runs)
    parser.add_argument('--duration-ms', type=float, default=duration)
    parser.add_argument('--baseline-backend', default='jax_raw')
    parser.add_argument('--output-dir', default='dev/fcn/results')
    parser.add_argument('--tag', default=None, help='Optional suffix for output files.')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = f'_{args.tag}' if args.tag else ''
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    failed_backends: list[tuple[str, str]] = []

    print(f"current config:{args.scales} {args.conns} {args.duration_ms}")
    for backend in args.backends:
        for conn in args.conns:
            print(f'Running backend={backend}, conn={conn} ...')
            try:
                rows = _benchmark_single_backend(
                    backend=backend,
                    scales=default_scale,
                    conn=conn,
                    duration_ms=duration,
                    warmup=warmup,
                    runs=runs,
                )
                all_rows.extend(rows)
            except Exception as e:  # noqa: BLE001
                failed_backends.append((f'{backend}/{conn}', repr(e)))
                print(f'Failed backend={backend}, conn={conn}: {e!r}')

    if not all_rows:
        raise RuntimeError('No benchmark results were collected.')

    summary = _summarize(all_rows, baseline_backend=args.baseline_backend)

    summary_csv_path = output_dir / f'coba_2005_binary_fcnmv_summary_{timestamp}{tag}.csv'

    _write_csv(
        summary_csv_path,
        summary,
        fieldnames=[
            'backend',
            'conn',
            'scale',
            'size',
            'runs',
            'elapsed_min_s',      
            'elapsed_max_s',      
            'elapsed_mean_s',
            'elapsed_std_s',
            'firing_rate_mean_hz',
            'speedup_vs_baseline',
        ],
    )


    print(f'Summary saved to: {summary_csv_path}')

    if failed_backends:
        print('Some backend/conn runs failed:')
        for backend_conn, err in failed_backends:
            print(f'  - {backend_conn}: {err}')


if __name__ == '__main__':
    #load_benchmark_config(config_file_path, benchmark_data_type, current_name)
    main()