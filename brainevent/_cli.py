# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

"""CLI entry point for brainevent.

Usage:
    brainevent benchmark-performance --platform {cpu|gpu|tpu} [--data {all|csr|coo|...}]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

__all__ = ['main']


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='brainevent',
        description='BrainEvent: Event-driven Computation in JAX for Brain Dynamics.',
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    bench = subparsers.add_parser(
        'benchmark-performance',
        help='Benchmark kernel performance across backends.',
    )
    bench.add_argument(
        '--platform',
        required=True,
        choices=['cpu', 'gpu', 'tpu'],
        help='Target platform to benchmark.',
    )
    bench.add_argument(
        '--data',
        default='all',
        help='Filter primitives by tag (e.g., csr, coo, dense, fcn, all). '
             'Comma-separated for multiple tags.',
    )
    bench.add_argument('--n-warmup', type=int, default=5, help='Number of warmup runs.')
    bench.add_argument('--n-runs', type=int, default=20, help='Number of timed runs.')
    bench.add_argument('--output', type=str, default=None, help='Output file path for JSON results.')

    return parser


def _filter_primitives(registry: dict, data_filter: str) -> dict:
    """Filter registry primitives by tag."""
    if data_filter == 'all':
        return registry

    tags = {t.strip() for t in data_filter.split(',')}
    result = {}
    for name, prim in registry.items():
        if hasattr(prim, '_tags') and tags.intersection(prim._tags):
            result[name] = prim
    return result


def _run_benchmark(args) -> int:
    """Run the benchmark-performance command."""
    # Import brainevent to trigger all primitive registrations
    import brainevent  # noqa: F401
    from brainevent._registry import get_registry
    from brainevent.config import save_user_defaults

    registry = get_registry()
    filtered = _filter_primitives(registry, args.data)

    if not filtered:
        print(f"No primitives found matching filter '{args.data}'.", file=sys.stderr)
        return 1

    print(f"BrainEvent Benchmark — platform={args.platform}, filter={args.data}")
    print(f"Parameters: n_warmup={args.n_warmup}, n_runs={args.n_runs}")
    print()

    # Table header
    header = f"{'Operator':<35} {'Backend':<15} {'Mean (ms)':>12} {'Std (ms)':>12} {'Min (ms)':>12} {'Winner':>8}"
    print(header)
    print("-" * len(header))

    all_reports = []
    optimal_defaults: Dict[str, Dict[str, str]] = {}

    for name in sorted(filtered.keys()):
        prim = filtered[name]

        # Skip primitives without a call function or benchmark data
        if prim._call_fn is None:
            continue
        if prim._benchmark_data_fn is None:
            continue

        # Check if the platform has registered backends
        backends = prim.available_backends(args.platform)
        if not backends:
            continue

        # Generate all benchmark configs from the data function
        try:
            configs = prim._benchmark_data_fn(platform=args.platform)
        except Exception as e:
            print(f"  {name:<35} {'SKIP':<15} Data generation failed: {e}")
            continue

        # Track wins per backend across all configs for voting
        backend_wins: Dict[str, int] = {}

        for config in configs:
            display_name = f"{name} [{config.name}]"

            # Run benchmark
            try:
                report = prim.benchmark(
                    *config.args,
                    platform=args.platform,
                    n_warmup=args.n_warmup,
                    n_runs=args.n_runs,
                    **config.kwargs,
                )
            except Exception as e:
                print(f"  {display_name:<35} {'ERROR':<15} Benchmark failed: {e}")
                continue

            all_reports.append(report)

            # Find winner
            fastest = report.fastest()
            for result in sorted(report.results, key=lambda r: (not r.success, r.mean_time)):
                winner = '*' if (result.success and fastest and result.backend == fastest.backend) else ''
                if result.success:
                    print(f"  {display_name:<35} {result.backend:<15} {result.mean_time * 1000:>12.3f} "
                          f"{result.std_time * 1000:>12.3f} {result.min_time * 1000:>12.3f} {winner:>8}")
                else:
                    error_short = (
                        result.error[:30] +
                        '...' if result.error and len(result.error) > 30 else (result.error or '')
                    )
                    print(
                        f"  {display_name:<35} {result.backend:<15} {'FAILED':>12} {'':>12} {'':>12} {error_short:>8}"
                    )

            # Track wins for voting
            if fastest:
                backend_wins[fastest.backend] = backend_wins.get(fastest.backend, 0) + 1

        # Always persist optimal backend via voting across all configs
        if backend_wins:
            best_backend = max(backend_wins, key=backend_wins.get)
            if name not in optimal_defaults:
                optimal_defaults[name] = {}
            optimal_defaults[name][args.platform] = best_backend

    print()

    # Always persist optimal defaults
    if optimal_defaults:
        metadata = {
            'last_run': datetime.now(timezone.utc).isoformat(),
            'platform': args.platform,
        }
        save_user_defaults(optimal_defaults, metadata=metadata)
        print(f"Optimal defaults persisted to config file.")

    # Write JSON output if requested
    if args.output:
        output_data = {
            'platform': args.platform,
            'reports': [r.to_dict() for r in all_reports],
            'optimal_defaults': optimal_defaults,
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
            f.write('\n')
        print(f"Results written to {args.output}")

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line arguments.  Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code (0 for success).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == 'benchmark-performance':
        return _run_benchmark(args)

    parser.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
