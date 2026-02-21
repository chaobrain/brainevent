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

"""Comprehensive examples for BenchmarkResult.print() using synthetic data.

Every parameter combination of ``BenchmarkResult.print()`` is demonstrated
with realistic-looking synthetic records.  No JAX, GPU, or network access is
required — all data is built from hand-crafted ``BenchmarkRecord`` objects.

Parameters covered
------------------
* (default)                 — plain table in insertion order
* sort_by (str)             — sort by a single numeric or string column
* sort_by (list)            — multi-column sort
* group_by (str)            — group rows, mark best with *
* group_by (list)           — multi-column grouping
* highlight_best=False      — suppress the best-marker
* compare_by (str)          — speedup column via string expression baseline
* compare_by (callable)     — speedup column via callable baseline selector
* order_by (2-level)        — hierarchical display, 2 sort keys
* order_by (3-level)        — hierarchical display, 3 sort keys
* order_by + speedup_vs     — per-group speedup column vs. a named backend
* order_by + highlight_best=False   — hierarchical without asterisks
* failed records            — FAILED rows displayed alongside successes
* empty result              — graceful output for zero records

Run from the project root::

    python examples/benchmark_print_examples.py
"""

import os
import sys

# Allow running from the project root without installing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brainevent._op.benchmark import BenchmarkRecord, BenchmarkResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEP = '=' * 72


def section(title: str) -> None:
    print()
    print(SEP)
    print(f'  {title}')
    print(SEP)


def subsection(title: str) -> None:
    print()
    print(f'--- {title} ---')


# ---------------------------------------------------------------------------
# Synthetic dataset 1 — single platform (GPU), three backends, three sizes
#
# Simulates a binary CSR matrix-vector multiplication sweep:
#   backends : numba_cuda (baseline), pallas, warp
#   sizes    : 1k×1k, 5k×5k, 10k×10k
#   weight   : homogeneous
#   transpose: not transposed
#
# Timing values (mean_ms) are chosen to tell a clear story:
#   pallas is fastest for small, warp catches up at large sizes,
#   numba_cuda is always the slowest on GPU.
# ---------------------------------------------------------------------------

_GPU_RECORDS_FLAT = [
    # ---- 1k×1k ----
    BenchmarkRecord(
        platform='gpu', backend='numba_cuda', label='1k×1k',
        mean_ms=1.820, std_ms=0.050, min_ms=1.770,
        throughput=None, success=True, error=None,
        kernel_kwargs={'shape': (1000, 1000), 'transpose': False},
        data_kwargs={'n_pre': 1000, 'n_post': 1000, 'nnz': 50_000},
    ),
    BenchmarkRecord(
        platform='gpu', backend='pallas', label='1k×1k',
        mean_ms=0.210, std_ms=0.008, min_ms=0.200,
        throughput=None, success=True, error=None,
        kernel_kwargs={'shape': (1000, 1000), 'transpose': False},
        data_kwargs={'n_pre': 1000, 'n_post': 1000, 'nnz': 50_000},
    ),
    BenchmarkRecord(
        platform='gpu', backend='warp', label='1k×1k',
        mean_ms=0.310, std_ms=0.012, min_ms=0.295,
        throughput=None, success=True, error=None,
        kernel_kwargs={'shape': (1000, 1000), 'transpose': False},
        data_kwargs={'n_pre': 1000, 'n_post': 1000, 'nnz': 50_000},
    ),
    # ---- 5k×5k ----
    BenchmarkRecord(
        platform='gpu', backend='numba_cuda', label='5k×5k',
        mean_ms=9.450, std_ms=0.210, min_ms=9.200,
        throughput=None, success=True, error=None,
        kernel_kwargs={'shape': (5000, 5000), 'transpose': False},
        data_kwargs={'n_pre': 5000, 'n_post': 5000, 'nnz': 1_250_000},
    ),
    BenchmarkRecord(
        platform='gpu', backend='pallas', label='5k×5k',
        mean_ms=0.870, std_ms=0.030, min_ms=0.840,
        throughput=None, success=True, error=None,
        kernel_kwargs={'shape': (5000, 5000), 'transpose': False},
        data_kwargs={'n_pre': 5000, 'n_post': 5000, 'nnz': 1_250_000},
    ),
    BenchmarkRecord(
        platform='gpu', backend='warp', label='5k×5k',
        mean_ms=0.920, std_ms=0.035, min_ms=0.882,
        throughput=None, success=True, error=None,
        kernel_kwargs={'shape': (5000, 5000), 'transpose': False},
        data_kwargs={'n_pre': 5000, 'n_post': 5000, 'nnz': 1_250_000},
    ),
    # ---- 10k×10k ----
    BenchmarkRecord(
        platform='gpu', backend='numba_cuda', label='10k×10k',
        mean_ms=38.200, std_ms=0.800, min_ms=37.400,
        throughput=None, success=True, error=None,
        kernel_kwargs={'shape': (10000, 10000), 'transpose': False},
        data_kwargs={'n_pre': 10000, 'n_post': 10000, 'nnz': 5_000_000},
    ),
    BenchmarkRecord(
        platform='gpu', backend='pallas', label='10k×10k',
        mean_ms=3.410, std_ms=0.060, min_ms=3.340,
        throughput=None, success=True, error=None,
        kernel_kwargs={'shape': (10000, 10000), 'transpose': False},
        data_kwargs={'n_pre': 10000, 'n_post': 10000, 'nnz': 5_000_000},
    ),
    BenchmarkRecord(
        platform='gpu', backend='warp', label='10k×10k',
        mean_ms=3.180, std_ms=0.055, min_ms=3.120,
        throughput=None, success=True, error=None,
        kernel_kwargs={'shape': (10000, 10000), 'transpose': False},
        data_kwargs={'n_pre': 10000, 'n_post': 10000, 'nnz': 5_000_000},
    ),
]

# ---------------------------------------------------------------------------
# Synthetic dataset 2 — transpose sweep
#
# Same three backends and three sizes, but now both transpose=False and
# transpose=True are included, giving 18 records total.  The transposed
# pass is typically slower because of non-coalesced memory access.
# ---------------------------------------------------------------------------

_TRANSPOSE_RECORDS = []
for _rec in _GPU_RECORDS_FLAT:
    # Non-transposed copy (already exists)
    _TRANSPOSE_RECORDS.append(_rec)
    # Transposed copy — 30-60 % slower
    _factor = 1.45 if _rec.label == '1k×1k' else (1.35 if _rec.label == '5k×5k' else 1.30)
    _TRANSPOSE_RECORDS.append(
        BenchmarkRecord(
            platform=_rec.platform,
            backend=_rec.backend,
            label=_rec.label,
            mean_ms=round(_rec.mean_ms * _factor, 3),
            std_ms=round(_rec.std_ms * _factor, 3),
            min_ms=round(_rec.min_ms * _factor, 3),
            throughput=None,
            success=True,
            error=None,
            kernel_kwargs={**_rec.kernel_kwargs, 'transpose': True},
            data_kwargs=_rec.data_kwargs,
        )
    )

# ---------------------------------------------------------------------------
# Synthetic dataset 3 — multi-platform
#
# CPU (numba) results added alongside the GPU records so that cross-platform
# comparisons can be demonstrated.
# ---------------------------------------------------------------------------

_CPU_RECORDS = [
    BenchmarkRecord(
        platform='cpu', backend='numba', label='1k×1k',
        mean_ms=4.200, std_ms=0.150, min_ms=4.050,
        throughput=None, success=True, error=None,
        kernel_kwargs={'shape': (1000, 1000), 'transpose': False},
        data_kwargs={'n_pre': 1000, 'n_post': 1000, 'nnz': 50_000},
    ),
    BenchmarkRecord(
        platform='cpu', backend='numba', label='5k×5k',
        mean_ms=95.000, std_ms=2.100, min_ms=92.800,
        throughput=None, success=True, error=None,
        kernel_kwargs={'shape': (5000, 5000), 'transpose': False},
        data_kwargs={'n_pre': 5000, 'n_post': 5000, 'nnz': 1_250_000},
    ),
    BenchmarkRecord(
        platform='cpu', backend='numba', label='10k×10k',
        mean_ms=385.000, std_ms=8.000, min_ms=377.000,
        throughput=None, success=True, error=None,
        kernel_kwargs={'shape': (10000, 10000), 'transpose': False},
        data_kwargs={'n_pre': 10000, 'n_post': 10000, 'nnz': 5_000_000},
    ),
]

_CROSS_PLATFORM_RECORDS = _CPU_RECORDS + _GPU_RECORDS_FLAT

# ---------------------------------------------------------------------------
# Synthetic dataset 4 — with a failed record
# ---------------------------------------------------------------------------

_RECORDS_WITH_FAILURE = list(_GPU_RECORDS_FLAT) + [
    BenchmarkRecord(
        platform='gpu', backend='triton', label='1k×1k',
        mean_ms=0.0, std_ms=0.0, min_ms=0.0,
        throughput=None, success=False,
        error='cuModuleLoadData failed: CUDA_ERROR_INVALID_PTX',
        kernel_kwargs={'shape': (1000, 1000), 'transpose': False},
        data_kwargs={'n_pre': 1000, 'n_post': 1000, 'nnz': 50_000},
    ),
    BenchmarkRecord(
        platform='gpu', backend='triton', label='5k×5k',
        mean_ms=0.750, std_ms=0.020, min_ms=0.730,
        throughput=None, success=True, error=None,
        kernel_kwargs={'shape': (5000, 5000), 'transpose': False},
        data_kwargs={'n_pre': 5000, 'n_post': 5000, 'nnz': 1_250_000},
    ),
    BenchmarkRecord(
        platform='gpu', backend='triton', label='10k×10k',
        mean_ms=2.900, std_ms=0.045, min_ms=2.855,
        throughput=None, success=True, error=None,
        kernel_kwargs={'shape': (10000, 10000), 'transpose': False},
        data_kwargs={'n_pre': 10000, 'n_post': 10000, 'nnz': 5_000_000},
    ),
]

# ---------------------------------------------------------------------------
# Convenience: build BenchmarkResult wrappers used in multiple examples
# ---------------------------------------------------------------------------

RESULT_FLAT = BenchmarkResult(_GPU_RECORDS_FLAT, primitive_name='binary_csrmv')
RESULT_TRANSPOSE = BenchmarkResult(_TRANSPOSE_RECORDS, primitive_name='binary_csrmv')
RESULT_XPLAT = BenchmarkResult(_CROSS_PLATFORM_RECORDS, primitive_name='binary_csrmv')
RESULT_FAILED = BenchmarkResult(_RECORDS_WITH_FAILURE, primitive_name='binary_csrmv')


# ===========================================================================
# Example 1 — default print (no arguments)
# ===========================================================================

def example_1_default():
    """Plain table in insertion order.  This is identical to str(result)."""
    section('Example 1 — default print() — insertion order')
    print(
        'Calling result.print() with no arguments renders a plain table\n'
        'in the order records were inserted.  __str__ / __repr__ do the same.\n'
        'Data: 3 backends × 3 sizes, GPU only.'
    )
    RESULT_FLAT.print()


# ===========================================================================
# Example 2 — sort_by (single column)
# ===========================================================================

def example_2_sort_by_single():
    """Sort by a single column — numeric or string."""
    section('Example 2 — sort_by (single column)')

    subsection("sort_by='mean_ms'  →  fastest rows first")
    print('Numeric columns sort in ascending order by default.')
    RESULT_FLAT.print(sort_by='mean_ms')

    subsection("sort_by='min_ms'  →  order by best single-run time")
    RESULT_FLAT.print(sort_by='min_ms')

    subsection("sort_by='label'  →  alphabetical by config label")
    print('String columns sort lexicographically.')
    RESULT_FLAT.print(sort_by='label')

    subsection("sort_by='backend'  →  alphabetical by backend name")
    RESULT_FLAT.print(sort_by='backend')


# ===========================================================================
# Example 3 — sort_by (multiple columns)
# ===========================================================================

def example_3_sort_by_multi():
    """Sort by multiple columns as a list — primary, then secondary."""
    section('Example 3 — sort_by (multiple columns)')

    subsection("sort_by=['label', 'mean_ms']  →  group configs, sort by time within each")
    print('The first key is the primary sort column; subsequent keys break ties.')
    RESULT_FLAT.print(sort_by=['label', 'mean_ms'])

    subsection("sort_by=['backend', 'mean_ms']  →  group backends, fastest first per backend")
    RESULT_FLAT.print(sort_by=['backend', 'mean_ms'])


# ===========================================================================
# Example 4 — group_by (single column) + highlight_best
# ===========================================================================

def example_4_group_by_single():
    """Group by a single column; fastest backend per group is marked with *."""
    section('Example 4 — group_by (single column)')

    subsection("group_by='label', highlight_best=True  (default)")
    print(
        'Within each config label the backend with the lowest mean_ms\n'
        'is marked with an asterisk (*).'
    )
    RESULT_FLAT.print(group_by='label', highlight_best=True)

    subsection("group_by='label', highlight_best=False  →  suppress the marker")
    RESULT_FLAT.print(group_by='label', highlight_best=False)

    subsection("group_by='backend'  →  identify the best config per backend")
    RESULT_FLAT.print(group_by='backend', highlight_best=True)


# ===========================================================================
# Example 5 — group_by (multiple columns)
# ===========================================================================

def example_5_group_by_multi():
    """Group by a composite key; best entry within each (label, transpose) pair."""
    section('Example 5 — group_by (multiple columns)')

    subsection("group_by=['label', 'transpose']  →  best backend per (size, mode)")
    print(
        'Grouping by a list of columns creates a composite group key.\n'
        'Data: 3 backends × 3 sizes × 2 transpose modes = 18 records.'
    )
    RESULT_TRANSPOSE.print(group_by=['label', 'transpose'], highlight_best=True)

    subsection("group_by=['backend', 'transpose']  →  best size per (backend, mode)")
    RESULT_TRANSPOSE.print(group_by=['backend', 'transpose'], highlight_best=True)


# ===========================================================================
# Example 6 — compare_by (string expression)
# ===========================================================================

def example_6_compare_by_string():
    """Speedup column via a string expression evaluated against each row dict."""
    section('Example 6 — compare_by (string expression)')

    subsection("compare_by=\"backend == 'numba_cuda'\"  →  speedup vs. numba_cuda baseline")
    print(
        'compare_by selects the baseline row(s) by evaluating a Python\n'
        'expression against each row dict.  A speedup column is added:\n'
        '  speedup = baseline_mean_ms / row_mean_ms\n'
        'Values > 1 mean the row is faster than the baseline.'
    )
    RESULT_FLAT.print(
        sort_by='mean_ms',
        compare_by="backend == 'numba_cuda'",
    )

    subsection("compare_by with label filter  →  single anchor row")
    print(
        "When the expression selects a single row (the first match is used),\n"
        "the speedup is relative only to that row's mean_ms."
    )
    RESULT_FLAT.print(
        compare_by="backend == 'numba_cuda' and label == '5k×5k'",
    )

    subsection("compare_by with group_by  →  speedup within each group")
    print(
        'Combining compare_by with group_by shows the speedup column\n'
        'alongside the group-level * marker.'
    )
    RESULT_FLAT.print(
        group_by='label',
        compare_by="backend == 'numba_cuda'",
    )


# ===========================================================================
# Example 7 — compare_by (callable)
# ===========================================================================

def example_7_compare_by_callable():
    """Speedup column via a Python callable instead of a string expression."""
    section('Example 7 — compare_by (callable)')

    subsection('Callable selecting all numba_cuda rows')
    print(
        'A callable receives each row dict and should return True for\n'
        'the row(s) to treat as the baseline.  The first matching row\n'
        'is used as the reference.'
    )
    RESULT_FLAT.print(
        sort_by='label',
        compare_by=lambda row: row.get('backend') == 'numba_cuda',
    )

    subsection('Callable with multi-condition baseline')
    RESULT_FLAT.print(
        compare_by=lambda row: (
            row.get('backend') == 'numba_cuda' and row.get('label') == '1k×1k'
        ),
    )


# ===========================================================================
# Example 8 — order_by (2-level hierarchy)
# ===========================================================================

def example_8_order_by_2level():
    """Hierarchical display with two sort keys: group column + leaf column."""
    section('Example 8 — order_by (2-level hierarchy)')

    subsection("order_by=['label', 'backend']  →  configs are groups, backends are rows")
    print(
        'order_by triggers hierarchical mode:\n'
        '  • All columns except the last are "group keys".\n'
        '  • Repeated group-key values are suppressed after the first row.\n'
        '  • A separator line is drawn between groups.\n'
        '  • The fastest backend within each group is marked *.\n'
        'With 2 keys, the first key forms the group; the second is the leaf.'
    )
    RESULT_FLAT.print(
        order_by=['label', 'backend'],
        highlight_best=True,
    )

    subsection("order_by=['backend', 'label']  →  backends are groups, configs are rows")
    RESULT_FLAT.print(
        order_by=['backend', 'label'],
        highlight_best=True,
    )


# ===========================================================================
# Example 9 — order_by (3-level hierarchy)
# ===========================================================================

def example_9_order_by_3level():
    """Hierarchical display with three sort keys."""
    section('Example 9 — order_by (3-level hierarchy)')

    subsection("order_by=['transpose', 'label', 'backend']  →  mode → size → backend")
    print(
        'With 3 keys the first two form the group key (transpose, label)\n'
        'and the third (backend) is the leaf that varies within each group.\n'
        'Data: 3 backends × 3 sizes × 2 transpose modes = 18 records.'
    )
    RESULT_TRANSPOSE.print(
        order_by=['transpose', 'label', 'backend'],
        highlight_best=True,
    )

    subsection("order_by=['label', 'transpose', 'backend']  →  size → mode → backend")
    RESULT_TRANSPOSE.print(
        order_by=['label', 'transpose', 'backend'],
        highlight_best=True,
    )


# ===========================================================================
# Example 10 — order_by + speedup_vs
# ===========================================================================

def example_10_speedup_vs():
    """Per-group speedup column relative to a named backend."""
    section('Example 10 — order_by + speedup_vs')

    subsection("speedup_vs='numba_cuda'  →  show GPU speedup over the CUDA baseline")
    print(
        'speedup_vs names a leaf-column value (typically a backend).\n'
        'A vs_<name> column is appended showing:\n'
        '  baseline_mean_ms / row_mean_ms\n'
        '>1 means the row is faster than the named baseline; <1 means slower.'
    )
    RESULT_FLAT.print(
        order_by=['label', 'backend'],
        highlight_best=True,
        speedup_vs='numba_cuda',
    )

    subsection("speedup_vs='pallas'  →  speedup or slowdown relative to pallas")
    RESULT_FLAT.print(
        order_by=['label', 'backend'],
        highlight_best=True,
        speedup_vs='pallas',
    )

    subsection("3-level hierarchy + speedup_vs='numba_cuda'")
    print('speedup_vs is computed independently within each group.')
    RESULT_TRANSPOSE.print(
        order_by=['transpose', 'label', 'backend'],
        highlight_best=True,
        speedup_vs='numba_cuda',
    )


# ===========================================================================
# Example 11 — order_by with highlight_best=False
# ===========================================================================

def example_11_order_by_no_highlight():
    """Hierarchical display without the best-marker asterisk."""
    section('Example 11 — order_by + highlight_best=False')

    subsection("Clean hierarchical table without * markers")
    print(
        'Passing highlight_best=False suppresses the asterisk column\n'
        'while keeping the hierarchical grouping and separators.'
    )
    RESULT_FLAT.print(
        order_by=['label', 'backend'],
        highlight_best=False,
    )

    subsection("With speedup_vs but no highlighting")
    RESULT_FLAT.print(
        order_by=['label', 'backend'],
        highlight_best=False,
        speedup_vs='numba_cuda',
    )


# ===========================================================================
# Example 12 — cross-platform data (CPU + GPU)
# ===========================================================================

def example_12_cross_platform():
    """Mix CPU and GPU records in a single result."""
    section('Example 12 — cross-platform records (CPU + GPU)')

    subsection('Default print — CPU numba row appears next to GPU rows')
    print(
        'Data: CPU numba + GPU numba_cuda + GPU pallas + GPU warp\n'
        'across 3 config sizes (12 records total).'
    )
    RESULT_XPLAT.print()

    subsection("group_by='label'  →  winner per config (CPU vs GPU)")
    RESULT_XPLAT.print(group_by='label', highlight_best=True)

    subsection("order_by=['label', 'backend'], speedup_vs='numba'")
    print('speedup_vs computes GPU speedup relative to the CPU numba baseline.')
    RESULT_XPLAT.print(
        order_by=['label', 'backend'],
        highlight_best=True,
        speedup_vs='numba',
    )

    subsection("compare_by with CPU numba baseline  →  global speedup column")
    RESULT_XPLAT.print(
        sort_by='mean_ms',
        compare_by="backend == 'numba' and platform == 'cpu'",
    )


# ===========================================================================
# Example 13 — failed records
# ===========================================================================

def example_13_failed_records():
    """FAILED rows are displayed alongside successes."""
    section('Example 13 — failed records')

    subsection('Default print — FAILED rows show mean_ms=FAILED')
    print(
        "Data: 9 GPU records (3 backends × 3 sizes) + 3 triton records,\n"
        "where triton failed on the 1k×1k config."
    )
    RESULT_FAILED.print()

    subsection("group_by='label'  →  failed rows do not affect the * marker")
    print('fastest() and highlight_best skip failed records (success=False).')
    RESULT_FAILED.print(group_by='label', highlight_best=True)

    subsection("sort_by='mean_ms'  →  FAILED rows sort after successful rows")
    RESULT_FAILED.print(sort_by='mean_ms')


# ===========================================================================
# Example 14 — empty result
# ===========================================================================

def example_14_empty():
    """Graceful output when there are no records."""
    section('Example 14 — empty BenchmarkResult')
    print('An empty result prints a one-line summary instead of a table.')
    empty = BenchmarkResult([], primitive_name='binary_csrmv')
    empty.print()


# ===========================================================================
# Example 15 — combining sort_by + group_by
# ===========================================================================

def example_15_sort_and_group():
    """Combining sort_by and group_by for fine-grained control."""
    section('Example 15 — sort_by + group_by combined')

    subsection("sort_by='mean_ms', group_by='label'  →  fastest-first within groups")
    print(
        'sort_by is applied before grouping, so rows within each group\n'
        'are already in ascending timing order when highlight_best marks them.'
    )
    RESULT_FLAT.print(sort_by='mean_ms', group_by='label')

    subsection("sort_by='backend', group_by='label'  →  alphabetical within groups")
    RESULT_FLAT.print(sort_by='backend', group_by='label')


# ===========================================================================
# Example 16 — pretty output when pandas + tabulate are available
# ===========================================================================

def example_16_pandas_tabulate():
    """Show that richer rendering is used when pandas / tabulate are installed."""
    section('Example 16 — pandas / tabulate rendering (if installed)')
    try:
        import pandas  # noqa: F401
        print('pandas is installed — the table uses pandas DataFrame rendering.')
        try:
            import tabulate  # noqa: F401
            print('tabulate is also installed — tabulate simple format is used.')
        except ImportError:
            print('tabulate is NOT installed — falls back to pandas .to_string().')
    except ImportError:
        print('pandas is NOT installed — falls back to the manual ASCII renderer.')
    print()
    RESULT_FLAT.print(group_by='label', highlight_best=True)


# ===========================================================================
# Example 17 — vary_by (single string)
# ===========================================================================

def example_17_vary_by_single():
    """vary_by='backend' — backend varies within each size group."""
    section("Example 17 — vary_by (single string: 'backend')")

    subsection("vary_by='backend'  →  3 groups (one per size), backend rows within each")
    print(
        "vary_by names the column that varies inside each group.\n"
        "Everything else (minus metrics) forms the fixed group boundary.\n"
        "A separator line is drawn between groups; * marks the fastest backend.\n"
        "Data: 3 backends × 3 sizes, GPU only."
    )
    RESULT_FLAT.print(vary_by='backend')

    subsection("vary_by='backend' + speedup_vs='numba_cuda'")
    print("speedup_vs adds a vs_numba_cuda column computed per group.")
    RESULT_FLAT.print(vary_by='backend', speedup_vs='numba_cuda')

    subsection("vary_by='backend' + highlight_best=False")
    RESULT_FLAT.print(vary_by='backend', highlight_best=False)


# ===========================================================================
# Example 18 — vary_by (list: two levels)
# ===========================================================================

def example_18_vary_by_list():
    """vary_by=['transpose', 'backend'] — outer vary + leaf vary within each group."""
    section("Example 18 — vary_by (list: ['transpose', 'backend'])")

    subsection("vary_by=['transpose', 'backend']  →  3 groups, 2-level within each")
    print(
        "When vary_by is a list, the separator fires only when the *fixed* columns\n"
        "change (i.e. when 'label' changes).  Within each group 'transpose' is the\n"
        "outer vary level (suppressed when repeated) and 'backend' is the leaf.\n"
        "* and speedup are computed per (label, transpose) sub-group.\n"
        "Data: 3 backends × 3 sizes × 2 transpose modes = 18 records."
    )
    RESULT_TRANSPOSE.print(vary_by=['transpose', 'backend'])

    subsection("vary_by=['transpose', 'backend'] + speedup_vs='numba_cuda'")
    print("Each (label, transpose) sub-group gets its own speedup baseline.")
    RESULT_TRANSPOSE.print(vary_by=['transpose', 'backend'], speedup_vs='numba_cuda')


# ===========================================================================
# Example 19 — vary_by vs order_by precedence
# ===========================================================================

def example_19_vary_by_vs_order_by():
    """order_by takes precedence when both vary_by and order_by are given."""
    section("Example 19 — vary_by vs order_by precedence")

    subsection("Both vary_by and order_by given  →  order_by wins")
    print(
        "If order_by is set, vary_by is ignored.\n"
        "This is the same as example 10 (order_by wins)."
    )
    RESULT_FLAT.print(
        order_by=['label', 'backend'],
        vary_by='backend',   # ignored because order_by is set
        highlight_best=True,
        speedup_vs='numba_cuda',
    )


# ===========================================================================
# Example 20 — vary_by with cross-platform data
# ===========================================================================

def example_20_vary_by_cross_platform():
    """vary_by on data that spans CPU and GPU backends."""
    section("Example 20 — vary_by with cross-platform data (CPU + GPU)")

    subsection("vary_by='backend'  →  numba/pallas/warp/numba_cuda grouped by size")
    print(
        "CPU and GPU records are merged.  The fixed group key is now\n"
        "(platform, shape, transpose, label, n_pre, n_post, nnz).\n"
        "Because platform and label differ between CPU and GPU records,\n"
        "each (platform, label) combination forms its own group."
    )
    RESULT_XPLAT.print(vary_by='backend', highlight_best=True)


# ===========================================================================
# Example 21 — vary_by with failed records
# ===========================================================================

def example_21_vary_by_failed():
    """vary_by displays FAILED rows inline with successes."""
    section("Example 21 — vary_by with failed records")

    subsection("vary_by='backend'  →  FAILED row shown inline, excluded from *")
    print(
        "Failed records (mean_ms='FAILED') are included in the table but\n"
        "excluded from the best-marker and speedup computation."
    )
    RESULT_FAILED.print(vary_by='backend', speedup_vs='numba_cuda')


# ===========================================================================
# Main
# ===========================================================================

def main():
    print()
    print('BenchmarkResult.print() — comprehensive examples with synthetic data')
    print('No JAX / GPU required: all records are hand-crafted BenchmarkRecord objects.')
    print()

    example_1_default()
    example_2_sort_by_single()
    example_3_sort_by_multi()
    example_4_group_by_single()
    example_5_group_by_multi()
    example_6_compare_by_string()
    example_7_compare_by_callable()
    example_8_order_by_2level()
    example_9_order_by_3level()
    example_10_speedup_vs()
    example_11_order_by_no_highlight()
    example_12_cross_platform()
    example_13_failed_records()
    example_14_empty()
    example_15_sort_and_group()
    example_16_pandas_tabulate()
    example_17_vary_by_single()
    example_18_vary_by_list()
    example_19_vary_by_vs_order_by()
    example_20_vary_by_cross_platform()
    example_21_vary_by_failed()

    print()
    print(SEP)
    print('  All examples completed.')
    print(SEP)
    print()


if __name__ == '__main__':
    main()
