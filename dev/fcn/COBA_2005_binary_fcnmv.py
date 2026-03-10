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
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

import brainunit as u
import jax
import brainunit as u

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import brainevent
from COBA_2005_benchmark import make_simulation_run

brainevent.config.set_backend('gpu', 'warp')
brainevent.config.set_backend('gpu', 'cuda_raw')

conn_num = 80


def _summarize(rows: Iterable[dict], baseline_backend: str) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row['backend'], row['conn'], row['scale'], row.get('conn_numbers'))].append(row)

    for s in [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]:
        run = make_simulation_run(
            scale=s,
            data_type='binary',
            efferent_target='post',
            duration=1e4 * u.ms,
            conn_num=conn_num
        )

    baseline_lookup = {
        (e['conn'], e['scale'], e['conn_numbers']): e['elapsed_mean_s']
        for e in summary
        if e['backend'] == baseline_backend
    }
    for entry in summary:
        baseline = baseline_lookup.get((entry['conn'], entry['scale'], entry['conn_numbers']))
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

    for s in [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]:
        run = make_simulation_run(
            scale=s,
            data_type='binary',
            efferent_target='pre',
            duration=1e2 * u.ms,
            conn_num=conn_num,
        )

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

    summary_csv_path = output_dir / f'coba_2005_binary_fcnmv_summary_{timestamp}_{data_type}_{duration}_{sf}.csv'

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
            'conn_numbers'
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