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
    from brainevent._error import BenchmarkDataFnNotProvidedError
    from brainevent._registry import get_registry
    registry = get_registry()
    filtered = _filter_primitives(registry, args.data)

    if not filtered:
        print(f"No primitives found matching filter '{args.data}'.", file=sys.stderr)
        return 1

    print(f"BrainEvent Benchmark — platform={args.platform}, filter={args.data}")
    print(f"Parameters: n_warmup={args.n_warmup}, n_runs={args.n_runs}")
    print()

    all_results = []
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

        print(f"  {name}")
        try:
            result = prim.benchmark(
                platform=args.platform,
                n_warmup=args.n_warmup,
                n_runs=args.n_runs,
            )
        except BenchmarkDataFnNotProvidedError:
            print(f"    SKIP — no benchmark data function registered")
            continue
        except Exception as e:
            print(f"    ERROR — {e}")
            continue

        result.print(group_by='label', highlight_best=True)
        print()

        all_results.append(result)

        # Vote for the best backend across all configs for this primitive
        backend_wins: Dict[str, int] = {}
        labels = list(dict.fromkeys(r.label for r in result.records))
        for label in labels:
            fastest = result.fastest(label=label)
            if fastest:
                backend_wins[fastest.backend] = backend_wins.get(fastest.backend, 0) + 1

        if backend_wins:
            best_backend = max(backend_wins, key=backend_wins.get)
            optimal_defaults.setdefault(name, {})[args.platform] = best_backend

    # Write JSON output if requested
    if args.output:
        output_data = {
            'platform': args.platform,
            'results': [r.to_dict() for r in all_results],
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
