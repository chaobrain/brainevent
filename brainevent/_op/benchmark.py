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

# -*- coding: utf-8 -*-

import csv
import json
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import jax
import numpy as np

__all__ = [
    'BenchmarkConfig',
    'BenchmarkRecord',
    'BenchmarkResult',
    'benchmark_function',
]

# Columns that are metrics — excluded from fixed/vary grouping keys
_METRIC_COLS = {'mean_ms', 'std_ms', 'min_ms', 'throughput', 'speedup', 'best'}


@dataclass
class BenchmarkConfig:
    """A single benchmark configuration for a primitive.

    Returned by ``def_benchmark_data`` functions as part of a list.

    Attributes
    ----------
    name : str
        A short descriptive label for this configuration
        (e.g., ``"T,homo,bool"``).
    args : tuple
        Positional arguments (the actual input data) to pass to the
        primitive's call function.
    kernel_kwargs : dict
        Keyword arguments passed directly into the kernel call
        (e.g., ``block_size``, ``shape``, ``transpose``). These may
        vary across benchmark runs and are forwarded to the call
        function.
    data_kwargs : dict
        Parameters that describe properties of the benchmark data but
        are **not** passed into the kernel (e.g., ``nnz``, ``sparsity``,
        ``density``, ``sequence_length``). These are recorded in the
        result table for reference only.
    """
    name: str
    args: tuple
    kernel_kwargs: dict = field(default_factory=dict)
    data_kwargs: dict = field(default_factory=dict)

    def put_args(self, device=None):
        args = []
        for arg in self.args:
            args.append(jax.device_put(arg, device=device).block_until_ready())
        return BenchmarkConfig(self.name, tuple(args), self.kernel_kwargs, self.data_kwargs)


@dataclass
class BenchmarkRecord:
    """One row in the benchmark result table.

    Each ``BenchmarkRecord`` represents a single (config, backend) run.

    Attributes
    ----------
    platform : str
        Hardware platform (e.g., ``'cpu'``, ``'gpu'``, ``'tpu'``).
    backend : str
        Backend name (e.g., ``'numba'``, ``'pallas'``, ``'warp'``).
    label : str
        Configuration label (from :attr:`BenchmarkConfig.name`).
    mean_ms : float
        Mean execution time in milliseconds.
    std_ms : float
        Standard deviation of execution time in milliseconds.
    min_ms : float
        Minimum execution time in milliseconds.
    throughput : float or None
        Optional throughput metric (e.g., GFLOPs/s). ``None`` if not
        computed.
    success : bool
        Whether the benchmark run completed without error.
    error : str or None
        Error message if the run failed; ``None`` on success.
    kernel_kwargs : dict
        Kernel parameters used for this run (forwarded to the call
        function).
    data_kwargs : dict
        Data-description metadata for this run (not forwarded to the
        kernel).
    """
    platform: str
    backend: str
    label: str
    mean_ms: float
    std_ms: float
    min_ms: float
    throughput: Optional[float]
    success: bool
    error: Optional[str]
    kernel_kwargs: Dict[str, Any]
    data_kwargs: Dict[str, Any]


class BenchmarkResult:
    """Unified container for benchmark timing records across all (config × backend) pairs.

    :class:`BenchmarkResult` is returned by
    :meth:`~brainevent._op.main.XLACustomKernel.benchmark`.  It stores
    every :class:`BenchmarkRecord` collected during a benchmarking
    session and exposes methods for display, comparison, plotting, and
    serialisation.

    Parameters
    ----------
    records : list of BenchmarkRecord
        All collected benchmark records.  Each record represents one
        (config × backend) pair.
    primitive_name : str, optional
        Name of the primitive that produced these records.  Used as the
        table heading when printing.  Defaults to an empty string.

    Attributes
    ----------
    primitive_name : str
        Name label for the benchmarked primitive.

    Methods — Accessors
    -------------------
    records
        Property — return a shallow copy of all stored records.
    fastest(label=None)
        Return the :class:`BenchmarkRecord` with the lowest
        ``mean_ms``, optionally restricted to a specific config *label*.

    Methods — Display
    -----------------
    print(sort_by, group_by, compare_by, highlight_best, order_by, speedup_vs)
        Print a formatted timing table to stdout.  Supports flat,
        sorted, grouped, and hierarchical layouts, plus relative speedup
        columns.

    Methods — Plotting
    ------------------
    plot(ax, x, y, hue, style, kind, show, **kwargs)
        Produce a matplotlib figure visualising the results as a line,
        bar, or scatter chart.

    Methods — Persistence
    ---------------------
    save(path, format)
        Write the result to disk as JSON (default), CSV, or pickle.
    load(path)
        Class method — deserialise a previously saved result.  Format is
        inferred from the file extension.
    to_dict()
        Return a JSON-serialisable dictionary representation of all
        records and metadata.

    Notes
    -----
    ``BenchmarkResult`` can also be constructed manually from a list of
    :class:`BenchmarkRecord` objects, which is useful for offline
    analysis (merging results from different machines, aggregating saved
    runs, etc.).

    ``__str__`` / ``__repr__`` delegate to :meth:`print` so a plain
    ``print(result)`` always shows a formatted table.

    Examples
    --------
    **Typical usage — run and display:**

    .. code-block:: python

        import brainevent
        result = brainevent.binary_csrmv_p.benchmark(
            platform='gpu',
            n_warmup=5,
            n_runs=20,
            verbose=True,
        )
        # __str__ / __repr__ renders a formatted table
        print(result)

    **Hierarchical display with per-group speedup:**

    .. code-block:: python

        # Rows grouped by (transpose, label); best backend per group
        # marked with *, plus a speedup column vs. the 'numba' baseline.
        result.print(
            order_by=['transpose', 'label', 'backend'],
            highlight_best=True,
            speedup_vs='numba',
        )

    **Flat table: sort, group, and baseline comparison:**

    .. code-block:: python

        # Sorted by mean execution time (fastest first)
        result.print(sort_by='mean_ms')

        # Best backend per config label marked with an asterisk
        result.print(group_by='label', highlight_best=True)

        # Speedup column relative to the numba baseline (string expression)
        result.print(compare_by="backend == 'numba'")

        # Callable baseline selector
        result.print(compare_by=lambda row: row.get('backend') == 'numba')

    **Accessing records programmatically:**

    .. code-block:: python

        # Iterate over all records
        for rec in result.records:
            status = 'OK' if rec.success else f'FAILED: {rec.error}'
            print(f"{rec.backend:10s} | {rec.label:20s} | {rec.mean_ms:.3f} ms | {status}")

        # Overall fastest successful record
        fastest = result.fastest()
        if fastest:
            print(f"Best overall: {fastest.backend} ({fastest.label}) — {fastest.mean_ms:.3f} ms")

        # Fastest backend per config label
        labels = dict.fromkeys(r.label for r in result.records)
        for label in labels:
            rec = result.fastest(label=label)
            if rec:
                print(f"[{label}] winner: {rec.backend} ({rec.mean_ms:.4f} ms)")

        # Custom aggregation: average mean_ms per backend
        from collections import defaultdict
        backend_times = defaultdict(list)
        for rec in result.records:
            if rec.success:
                backend_times[rec.backend].append(rec.mean_ms)
        for be, times in sorted(backend_times.items()):
            avg = sum(times) / len(times)
            print(f"{be:10s}: avg={avg:.4f} ms over {len(times)} configs")

    **Save and reload:**

    .. code-block:: python

        # JSON (default) — human-readable, round-trips with full fidelity
        result.save('bench.json')
        result2 = BenchmarkResult.load('bench.json')

        # CSV — flat table, easy to open in a spreadsheet
        result.save('bench.csv', format='csv')
        result3 = BenchmarkResult.load('bench.csv')

        # Pickle — lossless, preserves all dict fields
        result.save('bench.pkl', format='pkl')
        result4 = BenchmarkResult.load('bench.pkl')

    **Embedding in a larger JSON document:**

    .. code-block:: python

        import json

        d = result.to_dict()
        report = {
            'experiment': 'csrmv_gpu_sweep',
            'hardware': 'A100',
            'results': d,
        }
        with open('report.json', 'w') as f:
            json.dump(report, f, indent=2)

    **Building from scratch for offline / cross-platform analysis:**

    .. code-block:: python

        from brainevent._op.benchmark import BenchmarkRecord, BenchmarkResult

        # Combine records collected on two separate machines
        records = [
            BenchmarkRecord(
                platform='cpu', backend='numba', label='1k×1k',
                mean_ms=3.2, std_ms=0.1, min_ms=3.0,
                throughput=None, success=True, error=None,
                kernel_kwargs={'shape': (1000, 1000)},
                data_kwargs={'nnz': 100_000},
            ),
            BenchmarkRecord(
                platform='gpu', backend='pallas', label='1k×1k',
                mean_ms=0.42, std_ms=0.01, min_ms=0.40,
                throughput=None, success=True, error=None,
                kernel_kwargs={'shape': (1000, 1000)},
                data_kwargs={'nnz': 100_000},
            ),
            BenchmarkRecord(
                platform='gpu', backend='warp', label='1k×1k',
                mean_ms=0.60, std_ms=0.02, min_ms=0.58,
                throughput=None, success=True, error=None,
                kernel_kwargs={'shape': (1000, 1000)},
                data_kwargs={'nnz': 100_000},
            ),
        ]
        combined = BenchmarkResult(records, primitive_name='binary_csrmv')
        combined.print(group_by='label', highlight_best=True)
        # Speedup vs. CPU numba baseline
        combined.print(
            sort_by='mean_ms',
            compare_by="backend == 'numba' and platform == 'cpu'",
        )

    **Plotting:**

    .. code-block:: python

        # Bar chart: one bar per (label, backend) pair
        fig = result.plot(x='label', y='mean_ms', hue='backend', kind='bar')
        fig.tight_layout()
        fig.savefig('bench_bar.png', dpi=150)

        # Line chart over config labels, one line per backend
        fig2 = result.plot(x='label', y='min_ms', hue='backend', kind='line')
        fig2.savefig('bench_line.png', dpi=150)

    See Also
    --------
    BenchmarkConfig : Input specification for one benchmark configuration.
    BenchmarkRecord : Individual timing record stored in this container.
    XLACustomKernel.benchmark : Primary method that produces a
        :class:`BenchmarkResult`.
    benchmark_function : Low-level timing utility used internally.
    """

    def __init__(
        self,
        records: List[BenchmarkRecord],
        primitive_name: str = '',
    ):
        self._records: List[BenchmarkRecord] = list(records)
        self.primitive_name: str = primitive_name

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------

    @property
    def records(self) -> List[BenchmarkRecord]:
        """Return a shallow copy of all benchmark records.

        Returns
        -------
        list of BenchmarkRecord
            A new list containing every stored :class:`BenchmarkRecord`.
            Each record represents one (config × backend) timing run.
            Modifying the returned list does not affect the internal
            state.

        Examples
        --------
        .. code-block:: python

            result = binary_csrmv_p.benchmark(platform='gpu')
            print(f"Total records: {len(result.records)}")

            # Filter to successful records only
            ok = [r for r in result.records if r.success]

            # Custom aggregation: geometric mean per backend
            import math
            from collections import defaultdict
            backend_times = defaultdict(list)
            for rec in result.records:
                if rec.success:
                    backend_times[rec.backend].append(rec.mean_ms)
            for be, times in sorted(backend_times.items()):
                gm = math.exp(sum(math.log(t) for t in times) / len(times))
                print(f"{be}: geomean={gm:.4f} ms over {len(times)} configs")

        See Also
        --------
        fastest : Return the single fastest successful record.
        """
        return list(self._records)

    def fastest(self, label: Optional[str] = None) -> Optional[BenchmarkRecord]:
        """Return the fastest successful record.

        Parameters
        ----------
        label : str or None, optional
            If given, consider only records whose
            :attr:`~BenchmarkRecord.label` matches *label* exactly.
            Pass ``None`` (default) to search across all records.

        Returns
        -------
        BenchmarkRecord or None
            The :class:`BenchmarkRecord` with the smallest ``mean_ms``
            among all successful records (after optional label
            filtering), or ``None`` if no successful records exist.

        Examples
        --------
        .. code-block:: python

            result = binary_csrmv_p.benchmark(platform='gpu')

            # Overall fastest backend across all config labels
            rec = result.fastest()
            if rec:
                print(f"Best overall: {rec.backend} ({rec.label}) — {rec.mean_ms:.3f} ms")

            # Fastest for a specific config label
            rec = result.fastest(label='NT,homo,bool')
            if rec:
                print(f"Best for NT,homo,bool: {rec.backend} — {rec.mean_ms:.3f} ms")

            # Tabulate the winner for every label
            labels = dict.fromkeys(r.label for r in result.records)
            for label in labels:
                r = result.fastest(label=label)
                if r:
                    print(f"[{label}] winner: {r.backend} ({r.mean_ms:.4f} ms)")

        See Also
        --------
        records : Access all records for custom filtering and aggregation.
        print : Display a formatted table with ``highlight_best=True``
            to mark per-group winners visually.
        """
        candidates = [r for r in self._records if r.success]
        if label is not None:
            candidates = [r for r in candidates if r.label == label]
        if not candidates:
            return None
        return min(candidates, key=lambda r: r.mean_ms)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_extra_keys(self) -> Tuple[List[str], List[str]]:
        """Collect the union of kernel_kwargs and data_kwargs keys."""
        all_kw: set = set()
        all_dw: set = set()
        for r in self._records:
            all_kw.update(r.kernel_kwargs.keys())
            all_dw.update(r.data_kwargs.keys())
        return sorted(all_kw), sorted(all_dw)

    def _to_flat_rows(self) -> List[OrderedDict]:
        """Flatten records into a list of ordered dicts for display / export."""
        all_kw_keys, all_dw_keys = self._get_extra_keys()
        has_throughput = any(r.throughput is not None for r in self._records)

        rows = []
        for r in self._records:
            row: OrderedDict = OrderedDict()
            row['platform'] = r.platform
            row['backend'] = r.backend
            # kernel_kwargs columns
            for k in all_kw_keys:
                row[k] = r.kernel_kwargs.get(k, '')
            # data_kwargs columns
            for k in all_dw_keys:
                row[k] = r.data_kwargs.get(k, '')
            row['label'] = r.label
            if r.success:
                row['mean_ms'] = round(r.mean_ms, 4)
                row['std_ms'] = round(r.std_ms, 4)
                row['min_ms'] = round(r.min_ms, 4)
            else:
                row['mean_ms'] = 'FAILED'
                row['std_ms'] = ''
                row['min_ms'] = ''
            if has_throughput:
                row['throughput'] = r.throughput if r.throughput is not None else ''
            rows.append(row)
        return rows

    def _apply_sort(self, rows: List[OrderedDict], sort_by) -> List[OrderedDict]:
        if sort_by is None:
            return rows
        cols = [sort_by] if isinstance(sort_by, str) else list(sort_by)

        def _sort_key(row):
            parts = []
            for c in cols:
                v = row.get(c, '')
                # numeric values sort numerically; strings sort lexicographically
                if isinstance(v, (int, float)):
                    parts.append((0, v, ''))
                else:
                    parts.append((1, 0, str(v)))
            return parts

        return sorted(rows, key=_sort_key)

    def _apply_compare(self, rows: List[OrderedDict], compare_by) -> List[OrderedDict]:
        """Compute relative speedup against a baseline selected by *compare_by*."""
        if compare_by is None:
            return rows
        if callable(compare_by):
            baseline_rows = [r for r in rows if compare_by(r)]
        else:
            # Treat as a string expression evaluated against the row dict
            baseline_rows = []
            for row in rows:
                try:
                    if eval(compare_by, {}, dict(row)):  # noqa: S307
                        baseline_rows.append(row)
                except Exception:
                    pass
        if not baseline_rows:
            return rows
        bm = baseline_rows[0].get('mean_ms')
        if not isinstance(bm, (int, float)):
            return rows
        for row in rows:
            m = row.get('mean_ms')
            if isinstance(m, (int, float)) and m > 0:
                row['speedup'] = round(bm / m, 3)
            else:
                row['speedup'] = ''
        return rows

    def _compute_best(
        self,
        rows: List[OrderedDict],
        group_by,
    ) -> Dict[Any, str]:
        """Return a mapping from group-key → best backend within that group."""
        if group_by is None:
            return {}
        best: Dict[Any, Tuple[float, str]] = {}
        for row in rows:
            if isinstance(group_by, str):
                gk = row.get(group_by, '')
            else:
                gk = tuple(row.get(g, '') for g in group_by)
            m = row.get('mean_ms')
            be = row.get('backend', '')
            if not isinstance(m, (int, float)):
                continue
            if gk not in best or m < best[gk][0]:
                best[gk] = (m, be)
        return {gk: v[1] for gk, v in best.items()}

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _format_table(
        self,
        sort_by=None,
        group_by=None,
        compare_by=None,
        highlight_best: bool = True,
        order_by=None,
        speedup_vs=None,
        vary_by=None,
    ) -> str:
        if not self._records:
            return f"BenchmarkResult(primitive='{self.primitive_name}', 0 records)"

        rows = self._to_flat_rows()
        rows = self._apply_compare(rows, compare_by)

        title = f"BenchmarkResult: {self.primitive_name}" if self.primitive_name else "BenchmarkResult"

        if order_by is not None:
            return self._format_hierarchical(rows, title, order_by, highlight_best, speedup_vs)

        if vary_by is not None:
            fixed_keys, vary_keys = self._get_vary_by_order(vary_by)
            return self._format_vary_by(rows, title, fixed_keys, vary_keys, highlight_best, speedup_vs)

        rows = self._apply_sort(rows, sort_by)
        best_map = self._compute_best(rows, group_by) if highlight_best else {}

        # Attempt pandas + tabulate for the nicest output
        try:
            import pandas as pd

            df = pd.DataFrame(rows)
            if highlight_best and best_map and group_by is not None:
                def _mark(row_):
                    if isinstance(group_by, str):
                        gk = row_.get(group_by, '')
                    else:
                        gk = tuple(row_.get(g, '') for g in group_by)
                    return '*' if row_.get('backend') == best_map.get(gk) else ''

                df.insert(len(df.columns), 'best', [_mark(r) for r in rows])

            try:
                from tabulate import tabulate
                table_str = tabulate(df, headers='keys', tablefmt='simple', showindex=False)
            except ImportError:
                table_str = df.to_string(index=False)

            return f"{title}\n{table_str}"
        except ImportError:
            pass

        # Manual fallback (no dependencies)
        return self._format_manual(rows, title, best_map, group_by, highlight_best)

    def _format_hierarchical(
        self,
        rows: List[OrderedDict],
        title: str,
        order_by: List[str],
        highlight_best: bool,
        speedup_vs: Optional[str] = None,
    ) -> str:
        """Render a hierarchical table grouped by all but the last column in *order_by*.

        Rows are sorted by *order_by*; repeated values in the group-key
        columns are suppressed after the first row of each group, and a
        separator line is inserted between groups.  The fastest backend
        within each group is marked with ``*``.  When *speedup_vs* is
        given, a ``speedup`` column is appended showing
        ``baseline_mean / row_mean`` within each group.
        """
        if not rows:
            return title

        # Only keep order_by keys that actually exist in the rows
        all_keys = list(rows[0].keys())
        order_by = [k for k in order_by if k in all_keys]
        if not order_by:
            return self._format_manual(rows, title, {}, None, highlight_best)

        group_keys = order_by[:-1]
        leaf_key = order_by[-1]

        # Reorder columns: order_by cols first, then the rest
        remaining = [k for k in all_keys if k not in order_by]
        columns = order_by + remaining

        # Sort rows
        rows = sorted(rows, key=lambda r: tuple(str(r.get(k, '')) for k in order_by))

        # Compute best per group
        best_map = self._compute_best(rows, group_keys) if highlight_best and group_keys else {}
        # If no group keys (order_by has only one element), find global best
        if highlight_best and not group_keys:
            global_best: Optional[Tuple[float, str]] = None
            for row in rows:
                m = row.get('mean_ms')
                be = row.get(leaf_key, '')
                if isinstance(m, (int, float)):
                    if global_best is None or m < global_best[0]:
                        global_best = (m, be)
            if global_best is not None:
                best_map = {(): global_best[1]}

        # Build per-group baseline mean for speedup_vs
        # baseline_mean_map: group_key_tuple -> float
        baseline_mean_map: Dict[tuple, float] = {}
        if speedup_vs is not None:
            for row in rows:
                gk = tuple(row.get(k, '') for k in group_keys)
                if row.get(leaf_key, '') == speedup_vs:
                    m = row.get('mean_ms')
                    if isinstance(m, (int, float)):
                        baseline_mean_map[gk] = m

        # Inject speedup values into rows (as a new key)
        if speedup_vs is not None:
            speedup_col = f'vs_{speedup_vs}'
            if speedup_col not in columns:
                columns = columns + [speedup_col]
            for row in rows:
                gk = tuple(row.get(k, '') for k in group_keys)
                base = baseline_mean_map.get(gk)
                m = row.get('mean_ms')
                if base is not None and isinstance(m, (int, float)) and m > 0:
                    row[speedup_col] = f'{base / m:.2f}x'
                else:
                    row[speedup_col] = ''
        else:
            speedup_col = None

        # Compute column widths from actual values (not suppressed display)
        col_widths = {
            c: max(len(str(c)), max(len(str(r.get(c, ''))) for r in rows))
            for c in columns
        }

        sep = ' | '
        header = sep.join(str(c).ljust(col_widths[c]) for c in columns)
        rule = '-+-'.join('-' * col_widths[c] for c in columns)

        lines = [title, header, rule]

        prev_group_key: Optional[tuple] = None
        for row in rows:
            group_key = tuple(str(row.get(k, '')) for k in group_keys)
            is_new_group = (group_key != prev_group_key)

            if is_new_group and prev_group_key is not None:
                lines.append(rule)

            parts = []
            for c in columns:
                val = str(row.get(c, ''))
                if c in group_keys and not is_new_group:
                    val = ''
                parts.append(val.ljust(col_widths[c]))
            line = sep.join(parts)

            if highlight_best:
                if group_keys:
                    lookup_key = tuple(row.get(k, '') for k in group_keys)
                else:
                    lookup_key = ()
                if row.get(leaf_key, '') == best_map.get(lookup_key, ''):
                    line += '  *'

            lines.append(line)
            prev_group_key = group_key

        return '\n'.join(lines)

    def _format_manual(
        self,
        rows: List[OrderedDict],
        title: str,
        best_map: Dict,
        group_by,
        highlight_best: bool,
    ) -> str:
        if not rows:
            return title
        columns = list(rows[0].keys())
        # Compute column widths
        col_widths = {
            c: max(len(str(c)), max(len(str(r.get(c, ''))) for r in rows))
            for c in columns
        }
        sep = ' | '
        header = sep.join(str(c).ljust(col_widths[c]) for c in columns)
        rule = '-+-'.join('-' * col_widths[c] for c in columns)
        lines = [title, header, rule]
        for row in rows:
            parts = [str(row.get(c, '')).ljust(col_widths[c]) for c in columns]
            line = sep.join(parts)
            if highlight_best and best_map and group_by is not None:
                if isinstance(group_by, str):
                    gk = row.get(group_by, '')
                else:
                    gk = tuple(row.get(g, '') for g in group_by)
                if row.get('backend') == best_map.get(gk):
                    line += '  *'
            lines.append(line)
        return '\n'.join(lines)

    def _get_vary_by_order(self, vary_by) -> Tuple[List[str], List[str]]:
        """Return (fixed_group_keys, vary_keys) from a vary_by spec."""
        if isinstance(vary_by, str):
            vary_by = [vary_by]
        rows = self._to_flat_rows()
        if not rows:
            return [], list(vary_by)
        all_cols = list(rows[0].keys())
        vary_set = set(vary_by)
        fixed = [c for c in all_cols if c not in vary_set and c not in _METRIC_COLS]
        return fixed, list(vary_by)

    def _format_vary_by(
        self,
        rows: List[OrderedDict],
        title: str,
        fixed_keys: List[str],
        vary_keys: List[str],
        highlight_best: bool,
        speedup_vs: Optional[str],
    ) -> str:
        """Render a table grouped by fixed_keys, with vary_keys varying within each group.

        Separator lines appear only between changes in fixed_group_keys.
        Outer vary levels are suppressed when they repeat consecutively.
        The * marker and speedup are computed per (fixed_keys + outer_vary_keys) sub-group.
        """
        if not rows:
            return title

        leaf_key = vary_keys[-1]
        outer_vary_keys = vary_keys[:-1]

        # Column order: fixed, vary_keys, then remaining (metrics)
        all_keys = list(rows[0].keys())
        displayed = fixed_keys + vary_keys
        remaining = [k for k in all_keys if k not in displayed]
        columns = displayed + remaining

        # Sort: fixed keys, then vary_keys in order
        sort_keys = fixed_keys + vary_keys
        rows = sorted(rows, key=lambda r: tuple(str(r.get(k, '')) for k in sort_keys))

        def _subgroup_key(row):
            return tuple(row.get(k, '') for k in fixed_keys + outer_vary_keys)

        # Compute best per sub-group (lowest mean_ms)
        best_map: Dict[tuple, str] = {}
        if highlight_best:
            sub_best: Dict[tuple, Tuple[float, str]] = {}
            for row in rows:
                sk = _subgroup_key(row)
                m = row.get('mean_ms')
                lv = row.get(leaf_key, '')
                if isinstance(m, (int, float)):
                    if sk not in sub_best or m < sub_best[sk][0]:
                        sub_best[sk] = (m, lv)
            best_map = {sk: v[1] for sk, v in sub_best.items()}

        # Inject speedup column
        speedup_col: Optional[str] = None
        if speedup_vs is not None:
            speedup_col = f'vs_{speedup_vs}'
            if speedup_col not in columns:
                columns = columns + [speedup_col]
            baseline_map: Dict[tuple, float] = {}
            for row in rows:
                if row.get(leaf_key, '') == speedup_vs:
                    sk = _subgroup_key(row)
                    m = row.get('mean_ms')
                    if isinstance(m, (int, float)):
                        baseline_map[sk] = m
            for row in rows:
                sk = _subgroup_key(row)
                base = baseline_map.get(sk)
                m = row.get('mean_ms')
                if base is not None and isinstance(m, (int, float)) and m > 0:
                    row[speedup_col] = f'{base / m:.2f}x'
                else:
                    row[speedup_col] = ''

        # Compute column widths from actual data values (not suppressed display)
        col_widths = {
            c: max(len(str(c)), max((len(str(r.get(c, ''))) for r in rows), default=0))
            for c in columns
        }

        sep = ' | '
        header = sep.join(str(c).ljust(col_widths[c]) for c in columns)
        rule = '-+-'.join('-' * col_widths[c] for c in columns)
        lines = [title, header, rule]

        prev_fixed: Optional[tuple] = None
        prev_outer: Optional[tuple] = None
        for row in rows:
            cur_fixed = tuple(str(row.get(k, '')) for k in fixed_keys)
            cur_outer = tuple(str(row.get(k, '')) for k in outer_vary_keys)
            new_fixed_group = (cur_fixed != prev_fixed)
            new_outer_group = new_fixed_group or (cur_outer != prev_outer)

            if new_fixed_group and prev_fixed is not None:
                lines.append(rule)

            parts = []
            for c in columns:
                val = str(row.get(c, ''))
                if c in fixed_keys and not new_fixed_group:
                    val = ''
                elif c in outer_vary_keys and not new_outer_group:
                    val = ''
                parts.append(val.ljust(col_widths[c]))

            line = sep.join(parts)
            if highlight_best:
                sk = _subgroup_key(row)
                if row.get(leaf_key, '') == best_map.get(sk, object()):
                    line += '  *'
            lines.append(line)
            prev_fixed = cur_fixed
            prev_outer = cur_outer

        return '\n'.join(lines)

    def __repr__(self) -> str:
        return self._format_table()

    def __str__(self) -> str:
        return self._format_table()

    def print(
        self,
        sort_by=None,
        group_by=None,
        compare_by=None,
        highlight_best: bool = True,
        order_by=None,
        speedup_vs=None,
        vary_by=None,
    ) -> None:
        """Print the benchmark table with optional sorting, grouping, and comparison.

        Parameters
        ----------
        sort_by : str or list of str or None, optional
            Column name(s) to sort rows by.  Numeric columns are sorted
            numerically; string columns lexicographically.  Ignored
            when *order_by* is set.
        group_by : str or list of str or None, optional
            Column name(s) to group rows by.  Within each group the
            fastest backend is identified for highlighting and relative
            speedup computation.  Ignored when *order_by* is set.
        compare_by : str, callable, or None, optional
            Designate a baseline config for normalising performance.
            Pass a string expression (e.g., ``"label=='baseline'"``)
            evaluated against each row dict, or a callable
            ``(row_dict) -> bool``.  A ``speedup`` column is added
            showing ``baseline_mean / row_mean``.
        highlight_best : bool, optional
            If ``True`` (default), visually mark the best-performing
            config per group with an asterisk (``*``).
        order_by : list of str or None, optional
            When provided, render the table in **hierarchical mode**.
            Rows are sorted and visually grouped by all columns in
            *order_by* except the last one.  Repeated values in the
            group-key columns are suppressed after the first row of each
            group, and a separator line is drawn between groups.  The
            fastest entry within each group (determined by the last
            column in *order_by*) is marked ``*``.  Overrides
            *sort_by*, *group_by*, and *vary_by*.

            Example::

                result.print(order_by=['transpose', 'shape', 'backend'])
        speedup_vs : str or None, optional
            Active with *order_by* or *vary_by*.  Name of the leaf-column
            value (typically a backend name) to use as the per-group
            baseline.  Adds a ``vs_<name>`` column showing
            ``baseline_mean / row_mean`` for every row in that group.
            A value > 1 means the row is faster than the baseline.

            Example::

                result.print(
                    order_by=['transpose', 'shape', 'backend'],
                    speedup_vs='numba',
                )
        vary_by : str or list of str or None, optional
            **Shorthand grouping mode.**  Names the column(s) that
            *vary* within each group; everything else (excluding metrics)
            forms the fixed group boundary.

            - **Single string** — one column varies, all others are the
              group key.  A separator line is drawn between each group
              and the fastest leaf-column value is marked ``*``::

                result.print(vary_by='backend')

            - **Ordered list** — multiple columns vary; the separator
              fires only when the *fixed* columns change; earlier
              vary-columns are suppressed when they repeat consecutively;
              the last element is the finest leaf::

                result.print(vary_by=['transpose', 'backend'])

            ``*`` and *speedup_vs* are computed per
            ``(fixed_keys + outer_vary_keys)`` sub-group.
            *order_by* takes precedence if both are given.

        Examples
        --------
        .. code-block:: python

            result = binary_csrmv_p.benchmark(platform='gpu')

            # Default: plain table in insertion order
            result.print()

            # Sorted by mean execution time (fastest first)
            result.print(sort_by='mean_ms')

            # Group by config label; fastest backend per group marked *
            result.print(group_by='label', highlight_best=True)

            # Speedup column vs. the numba baseline (string expression)
            result.print(
                sort_by='mean_ms',
                compare_by="backend == 'numba'",
            )

            # Callable baseline selector
            result.print(compare_by=lambda row: row.get('backend') == 'numba')

            # Hierarchical view: group by (transpose, label), mark best backend
            result.print(
                order_by=['transpose', 'label', 'backend'],
                highlight_best=True,
            )

            # Hierarchical + per-group speedup vs. numba
            result.print(
                order_by=['transpose', 'label', 'backend'],
                highlight_best=True,
                speedup_vs='numba',
            )

            # vary_by shorthand: backend varies, everything else is the group
            result.print(vary_by='backend', speedup_vs='numba_cuda')

            # vary_by with two levels: transpose is outer, backend is leaf
            result.print(vary_by=['transpose', 'backend'], speedup_vs='numba_cuda')

        See Also
        --------
        fastest : Return the fastest record programmatically.
        plot : Produce a matplotlib visualisation.
        """
        print(
            self._format_table(
                sort_by=sort_by,
                group_by=group_by,
                compare_by=compare_by,
                highlight_best=highlight_best,
                order_by=order_by,
                speedup_vs=speedup_vs,
                vary_by=vary_by,
            )
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        ax=None,
        x: Optional[str] = None,
        y: str = 'mean_ms',
        hue: Optional[str] = None,
        style: Optional[str] = None,
        kind: str = 'line',
        show: bool = False,
        **kwargs,
    ):
        """Produce a visualization of the benchmark results.

        Parameters
        ----------
        ax : matplotlib Axes or None, optional
            Axes to draw into.  If ``None``, a new figure and axes are
            created.
        x : str or None, optional
            Column name for the x-axis (e.g., ``'label'``, ``'n_pre'``).
        y : str, optional
            Column name for the y-axis.  Defaults to ``'mean_ms'``.
        hue : str or None, optional
            Column name used to colour-code different series
            (e.g., ``'backend'``).
        style : str or None, optional
            Column name used to set line/marker style (seaborn only).
        kind : {'line', 'bar', 'scatter'}, optional
            Plot type.  Defaults to ``'line'``.
        show : bool, optional
            If ``True``, call ``plt.show()`` after drawing.  Defaults
            to ``False``.
        **kwargs
            Additional keyword arguments forwarded to the underlying
            matplotlib / seaborn plotting function.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.

        Raises
        ------
        ImportError
            If *matplotlib* or *pandas* is not installed.

        Examples
        --------
        .. code-block:: python

            result = binary_csrmv_p.benchmark(platform='gpu')

            # Bar chart: mean_ms per config, coloured by backend
            fig = result.plot(x='label', y='mean_ms', hue='backend', kind='bar')
            fig.tight_layout()
            fig.savefig('bench_bar.png', dpi=150)

            # Line chart: min_ms vs. config label
            fig2 = result.plot(x='label', y='min_ms', hue='backend', kind='line')
            fig2.savefig('bench_line.png', dpi=150)

            # Scatter: draw into an existing axes
            import matplotlib.pyplot as plt
            fig3, ax = plt.subplots()
            result.plot(ax=ax, x='label', y='mean_ms', kind='scatter')
            plt.show()

        See Also
        --------
        print : Display a formatted text table instead.
        save : Persist results to disk for later off-line plotting.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "plot() requires matplotlib. Install with: pip install matplotlib"
            ) from exc

        try:
            import pandas as pd
            rows = self._to_flat_rows()
            df = pd.DataFrame(rows)
            # Keep only successful rows for plotting
            df = df[df['mean_ms'].apply(lambda v: isinstance(v, (int, float)))]
        except ImportError as exc:
            raise ImportError(
                "plot() requires pandas. Install with: pip install pandas"
            ) from exc

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        try:
            import seaborn as sns
            _sns_plot(sns, df, ax, x, y, hue, style, kind, kwargs)
        except ImportError:
            _mpl_plot(df, ax, x, y, hue, kind, kwargs)

        ax.set_title(self.primitive_name or 'Benchmark')
        ax.set_xlabel(x or '')
        ax.set_ylabel(y or '')

        if show:
            plt.show()

        return fig

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        path: Union[str, 'Path'],
        format: Literal['json', 'csv', 'pkl'] = 'json',
    ) -> None:
        """Serialize the result to disk.

        Parameters
        ----------
        path : str or Path
            Destination file path.  Parent directories are created
            automatically if they do not exist.
        format : {'json', 'csv', 'pkl'}, optional
            Serialization format:

            - ``'json'`` (default) — human-readable JSON; round-trips
              with full fidelity for all field types supported by
              :meth:`to_dict`.
            - ``'csv'`` — flat CSV table; easily opened in spreadsheet
              tools.  ``kernel_kwargs`` and ``data_kwargs`` are not
              preserved as nested dicts (they are omitted from the flat
              rows).
            - ``'pkl'`` — binary pickle; lossless, preserves all
              ``dict`` fields but not portable across Python versions.

        Raises
        ------
        ValueError
            If *format* is not one of the supported values.

        Examples
        --------
        .. code-block:: python

            result = binary_csrmv_p.benchmark(platform='gpu')

            # Default JSON
            result.save('results/bench.json')

            # CSV for spreadsheet analysis
            result.save('results/bench.csv', format='csv')

            # Lossless pickle
            result.save('results/bench.pkl', format='pkl')

        See Also
        --------
        load : Deserialise a previously saved file.
        to_dict : Access the JSON-serialisable dict directly.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self._to_serializable_dict(), f, indent=2)
                f.write('\n')

        elif format == 'csv':
            rows = self._to_flat_rows()
            if rows:
                cols = list(rows[0].keys())
                with open(path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=cols)
                    writer.writeheader()
                    writer.writerows(rows)

        elif format == 'pkl':
            with open(path, 'wb') as f:
                pickle.dump(self, f)

        else:
            raise ValueError(
                f"Unsupported format: {format!r}. Choose from 'json', 'csv', 'pkl'."
            )

    @classmethod
    def load(cls, path: Union[str, 'Path']) -> 'BenchmarkResult':
        """Deserialize a previously saved result.

        The format is inferred from the file extension (``.json``,
        ``.csv``, ``.pkl``).  Files without one of these suffixes are
        assumed to be JSON.

        Parameters
        ----------
        path : str or Path
            Path to the file written by :meth:`save`.

        Returns
        -------
        BenchmarkResult
            A new :class:`BenchmarkResult` populated from the file.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If a ``.pkl`` file does not contain a :class:`BenchmarkResult`.

        Examples
        --------
        .. code-block:: python

            # Round-trip with JSON
            result.save('bench.json')
            reloaded = BenchmarkResult.load('bench.json')
            print(reloaded)

            # Round-trip with CSV
            result.save('bench.csv', format='csv')
            reloaded_csv = BenchmarkResult.load('bench.csv')

            # Round-trip with pickle
            result.save('bench.pkl', format='pkl')
            reloaded_pkl = BenchmarkResult.load('bench.pkl')

        See Also
        --------
        save : Serialise a result to disk.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()

        if suffix == '.pkl':
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            if not isinstance(obj, cls):
                raise ValueError(
                    f"Loaded object is not a BenchmarkResult: {type(obj)}"
                )
            return obj

        if suffix == '.csv':
            return cls._load_csv(path)

        # Default: JSON
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls._from_serializable_dict(data)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def _to_serializable_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            'primitive_name': self.primitive_name,
            'records': [
                {
                    'platform': r.platform,
                    'backend': r.backend,
                    'label': r.label,
                    'mean_ms': r.mean_ms,
                    'std_ms': r.std_ms,
                    'min_ms': r.min_ms,
                    'throughput': r.throughput,
                    'success': r.success,
                    'error': r.error,
                    'kernel_kwargs': {k: _json_safe(v) for k, v in r.kernel_kwargs.items()},
                    'data_kwargs': {k: _json_safe(v) for k, v in r.data_kwargs.items()},
                }
                for r in self._records
            ],
        }

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dictionary representation.

        The returned dictionary contains the primitive name and the full
        list of records in the same format used by :meth:`save` (JSON).
        It can be passed directly to :func:`json.dump`, embedded in a
        larger document, or used to reconstruct a :class:`BenchmarkResult`
        via :meth:`load`.

        Returns
        -------
        dict
            A dictionary with two top-level keys:

            ``'primitive_name'`` : str
                The benchmarked primitive's name.
            ``'records'`` : list of dict
                One dict per :class:`BenchmarkRecord`.  Each dict has
                keys: ``platform``, ``backend``, ``label``, ``mean_ms``,
                ``std_ms``, ``min_ms``, ``throughput``, ``success``,
                ``error``, ``kernel_kwargs``, ``data_kwargs``.

        Examples
        --------
        .. code-block:: python

            result = binary_csrmv_p.benchmark(platform='gpu')
            d = result.to_dict()

            # Pretty-print to console
            import json
            print(json.dumps(d, indent=2))

            # Embed in a larger report document
            report = {
                'experiment': 'csrmv_gpu_sweep',
                'hardware': 'A100-SXM4-80GB',
                'results': d,
            }
            with open('report.json', 'w') as f:
                json.dump(report, f, indent=2)

            # Access individual record fields
            for rec in d['records']:
                print(rec['backend'], rec['mean_ms'])

        See Also
        --------
        save : Write directly to disk (JSON / CSV / pickle).
        load : Reconstruct a :class:`BenchmarkResult` from a file.
        """
        return self._to_serializable_dict()

    @classmethod
    def _load_csv(cls, path: Path) -> 'BenchmarkResult':
        """Load from a CSV file, trying pandas first then falling back to csv module."""
        try:
            import pandas as pd
            df = pd.read_csv(path)
            row_dicts = df.to_dict(orient='records')
        except ImportError:
            with open(path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                row_dicts = list(reader)

        records = []
        for row in row_dicts:
            raw_mean = row.get('mean_ms', 0.0)
            success = str(raw_mean) != 'FAILED'
            try:
                mean_ms = float(raw_mean)
            except (ValueError, TypeError):
                mean_ms = 0.0
                success = False
            tp = row.get('throughput')
            try:
                tp = float(tp) if tp not in (None, '', 'None') else None
            except (ValueError, TypeError):
                tp = None
            records.append(
                BenchmarkRecord(
                    platform=str(row.get('platform', '')),
                    backend=str(row.get('backend', '')),
                    label=str(row.get('label', '')),
                    mean_ms=mean_ms,
                    std_ms=float(row.get('std_ms') or 0.0),
                    min_ms=float(row.get('min_ms') or 0.0),
                    throughput=tp,
                    success=success,
                    error=None,
                    kernel_kwargs={},
                    data_kwargs={},
                )
            )
        return cls(records=records)

    @classmethod
    def _from_serializable_dict(cls, data: dict) -> 'BenchmarkResult':
        records = []
        for r in data.get('records', []):
            records.append(
                BenchmarkRecord(
                    platform=r.get('platform', ''),
                    backend=r.get('backend', ''),
                    label=r.get('label', ''),
                    mean_ms=float(r.get('mean_ms', 0.0)),
                    std_ms=float(r.get('std_ms', 0.0)),
                    min_ms=float(r.get('min_ms', 0.0)),
                    throughput=r.get('throughput'),
                    success=bool(r.get('success', True)),
                    error=r.get('error'),
                    kernel_kwargs=r.get('kernel_kwargs', {}),
                    data_kwargs=r.get('data_kwargs', {}),
                )
            )
        return cls(records=records, primitive_name=data.get('primitive_name', ''))


# ---------------------------------------------------------------------------
# Internal plotting helpers
# ---------------------------------------------------------------------------

def _sns_plot(sns, df, ax, x, y, hue, style, kind, kwargs):
    plot_kw = dict(data=df, x=x, y=y, ax=ax)
    if hue:
        plot_kw['hue'] = hue
    if style:
        plot_kw['style'] = style
    plot_kw.update(kwargs)
    if kind == 'line':
        sns.lineplot(**plot_kw)
    elif kind == 'bar':
        plot_kw.pop('style', None)
        sns.barplot(**plot_kw)
    elif kind == 'scatter':
        sns.scatterplot(**plot_kw)


def _mpl_plot(df, ax, x, y, hue, kind, kwargs):
    if kind == 'bar':
        if hue and x:
            df.groupby([x, hue])[y].mean().unstack().plot(kind='bar', ax=ax, **kwargs)
        elif x:
            df.groupby(x)[y].mean().plot(kind='bar', ax=ax, **kwargs)
        return

    if kind == 'scatter':
        if hue and x:
            for k, grp in df.groupby(hue):
                ax.scatter(grp[x].tolist(), grp[y].tolist(), label=str(k), **kwargs)
            ax.legend()
        elif x:
            ax.scatter(df[x].tolist(), df[y].tolist(), **kwargs)
        return

    # Default: line
    if hue and x:
        for k, grp in df.groupby(hue):
            ax.plot(grp[x].tolist(), grp[y].tolist(), label=str(k), **kwargs)
        ax.legend()
    elif x:
        ax.plot(df[x].tolist(), df[y].tolist(), **kwargs)


def _json_safe(v: Any) -> Any:
    """Convert a value to a JSON-serializable type."""
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    return str(v)


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------

def benchmark_function(
    fn,
    n_warmup: int,
    n_runs: int,
    n_batch_per_run: int = 1,
    data: Tuple = (),
) -> Tuple[float, float, float, float, Any]:
    """Benchmark a function and return timing statistics.

    Parameters
    ----------
    fn : callable
        A callable that takes no arguments and returns the result.
    n_warmup : int
        Number of warmup runs (not timed).
    n_runs : int
        Number of timed measurement intervals.
    n_batch_per_run : int, optional
        Number of back-to-back ``fn()`` calls issued within each timed
        interval before blocking.  Default is ``1``, which blocks after
        every call and measures per-call latency.  Values greater than
        ``1`` amortise blocking overhead across multiple kernel launches,
        which is useful for measuring throughput on asynchronous
        GPU/TPU execution.  The reported times are always **per-call**
        (i.e. the interval time divided by *n_batch_per_run*).

    Returns
    -------
    tuple of (float, float, float, float, Any)
        ``(mean_time, std_time, min_time, max_time, output)`` where
        times are in seconds and represent per-call values.
    """

    # Run fn once to get the output structure needed as fori_loop carry init
    output = fn(*data)
    jax.block_until_ready(output)

    @jax.jit
    def run_fn(*args):
        if n_batch_per_run == 1:
            res = fn(*args)
        else:
            res = jax.lax.fori_loop(0, n_batch_per_run, lambda i, carry: fn(*args), output)
        return jax.tree.leaves(res)[0]  # Return a single leaf for timing

    # Warmup runs
    for _ in range(n_warmup):
        run_fn(*data).block_until_ready()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        run_fn(*data).block_until_ready()
        end = time.perf_counter()
        times.append((end - start) / n_batch_per_run)

    times = np.array(times)
    return (
        float(np.mean(times)),
        float(np.std(times)),
        float(np.min(times)),
        float(np.max(times)),
        output,
    )
