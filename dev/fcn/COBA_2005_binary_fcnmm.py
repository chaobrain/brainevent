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
from COBA_2005_benchmark import make_simulation_batch_run

DEFAULT_SCALES = (1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100)
DEFAULT_BACKENDS = ('tvmffi', 'pallas', 'jax_raw')
DEFAULT_CONNS = ('post', 'pre')


def _benchmark_single_backend(
    backend: str,
    scales: Iterable[int],
    conn: str,
    batch_size: int,
    duration_ms: float,
    warmup: int,
    runs: int,
) -> list[dict]:
    brainevent.config.set_backend('gpu', backend)
    rows: list[dict] = []
    for scale in scales:
        run = make_simulation_batch_run(
            scale,
            batch_size,
            'binary',
            conn,
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _plot(summary: list[dict], output_dir: Path, baseline_backend: str) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not installed; skipping plots.')
        return []

    paths: list[Path] = []
    conns = sorted({entry['conn'] for entry in summary})
    for conn in conns:
        conn_rows = [e for e in summary if e['conn'] == conn]
        backends = sorted({e['backend'] for e in conn_rows})

        fig1, ax1 = plt.subplots(figsize=(9, 5))
        for backend in backends:
            b_rows = sorted((e for e in conn_rows if e['backend'] == backend), key=lambda e: e['scale'])
            ax1.plot(
                [e['scale'] for e in b_rows],
                [e['elapsed_mean_s'] for e in b_rows],
                marker='o',
                label=backend,
            )
        ax1.set_title(f'COBA 2005 binary_fcnmm runtime ({conn} conn)')
        ax1.set_xlabel('Scale')
        ax1.set_ylabel('Mean runtime (s)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        runtime_path = output_dir / f'runtime_{conn}.png'
        fig1.tight_layout()
        fig1.savefig(runtime_path, dpi=150)
        plt.close(fig1)
        paths.append(runtime_path)

        speedup_rows = [
            e for e in conn_rows if e['backend'] != baseline_backend and e['speedup_vs_baseline'] is not None
        ]
        if speedup_rows:
            fig2, ax2 = plt.subplots(figsize=(9, 5))
            for backend in sorted({e['backend'] for e in speedup_rows}):
                b_rows = sorted((e for e in speedup_rows if e['backend'] == backend), key=lambda e: e['scale'])
                ax2.plot(
                    [e['scale'] for e in b_rows],
                    [e['speedup_vs_baseline'] for e in b_rows],
                    marker='o',
                    label=f'{backend} vs {baseline_backend}',
                )
            ax2.axhline(1.0, color='k', linestyle='--', linewidth=1)
            ax2.set_title(f'COBA 2005 binary_fcnmm speedup ({conn} conn)')
            ax2.set_xlabel('Scale')
            ax2.set_ylabel('Speedup (>1 is faster)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            speedup_path = output_dir / f'speedup_{conn}.png'
            fig2.tight_layout()
            fig2.savefig(speedup_path, dpi=150)
            plt.close(fig2)
            paths.append(speedup_path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description='Run and compare COBA_2005 binary_fcnmm backends.')
    parser.add_argument('--backends', nargs='+', default=list(DEFAULT_BACKENDS))
    parser.add_argument('--scales', nargs='+', type=int, default=list(DEFAULT_SCALES))
    parser.add_argument('--conns', nargs='+', choices=['post', 'pre'], default=list(DEFAULT_CONNS))
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--duration-ms', type=float, default=1e4)
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
    for backend in args.backends:
        for conn in args.conns:
            print(f'Running backend={backend}, conn={conn} ...')
            try:
                rows = _benchmark_single_backend(
                    backend=backend,
                    scales=args.scales,
                    conn=conn,
                    batch_size=args.batch_size,
                    duration_ms=args.duration_ms,
                    warmup=args.warmup,
                    runs=args.runs,
                )
                all_rows.extend(rows)
            except Exception as e:  # noqa: BLE001
                failed_backends.append((f'{backend}/{conn}', repr(e)))
                print(f'Failed backend={backend}, conn={conn}: {e!r}')

    if not all_rows:
        raise RuntimeError('No benchmark results were collected.')

    summary = _summarize(all_rows, baseline_backend=args.baseline_backend)

    raw_csv_path = output_dir / f'coba_2005_binary_fcnmm_raw_{timestamp}{tag}.csv'
    summary_csv_path = output_dir / f'coba_2005_binary_fcnmm_summary_{timestamp}{tag}.csv'
    summary_json_path = output_dir / f'coba_2005_binary_fcnmm_summary_{timestamp}{tag}.json'
    _write_csv(
        raw_csv_path,
        all_rows,
        fieldnames=['backend', 'conn', 'scale', 'run_id', 'size', 'firing_rate_hz', 'elapsed_s'],
    )
    _write_csv(
        summary_csv_path,
        summary,
        fieldnames=[
            'backend',
            'conn',
            'scale',
            'size',
            'runs',
            'elapsed_mean_s',
            'elapsed_std_s',
            'firing_rate_mean_hz',
            'speedup_vs_baseline',
        ],
    )
    _write_json(
        summary_json_path,
        {
            'created_at': timestamp,
            'backends': args.backends,
            'conns': args.conns,
            'scales': args.scales,
            'batch_size': args.batch_size,
            'warmup': args.warmup,
            'runs': args.runs,
            'duration_ms': args.duration_ms,
            'baseline_backend': args.baseline_backend,
            'failed_backends': failed_backends,
            'summary': summary,
        },
    )

    plot_paths: list[Path] = []
    if not args.no_plot:
        plot_paths = _plot(summary, output_dir=output_dir, baseline_backend=args.baseline_backend)

    print(f'Raw results saved to: {raw_csv_path}')
    print(f'Summary saved to: {summary_csv_path}')
    print(f'Summary JSON saved to: {summary_json_path}')
    if plot_paths:
        print('Plots saved:')
        for p in plot_paths:
            print(f'  - {p}')
    if failed_backends:
        print('Some backend/conn runs failed:')
        for backend_conn, err in failed_backends:
            print(f'  - {backend_conn}: {err}')


if __name__ == '__main__':
    main()
