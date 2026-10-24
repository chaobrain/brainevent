# Unify Tile and batchmajor cuSPARSE figures

## Problem

The refreshed batchmajor figures use `plot_batch_major_speedup.py`, while the
Tile figures were generated with the legacy `plot_mm_speedup.py`. They differ
in palette, typography, legends, axis layout, filenames and figure count.

## Scope

Reuse the batchmajor renderer for Tile/cuSPARSE. Add `--align-cusparse-to-tile`
to generate both cuSPARSE comparisons with the Tile parameter grid. Default
invocation must retain both existing batchmajor comparisons. Tile comes from
`mm/case_001/case_1.csv` and the dense baseline from `mm/case_002/case_1.csv`.

Each comparison exports `speedup_heatmap.svg`, `speedup_lines_grid.svg` and,
when those densities exist, `speedup_density_three_rows.svg`, with `data/speedup.csv`
and `data/summary.json`. Titles, labels, source provenance, CSV prefixes and
status names must identify the actual target. The older Tile outputs will be
retained in an archive outside the active `figures` and `data` directories.

The user explicitly requested matching Tile's parameter ranges. Both comparisons
must use the Cartesian grid of Tile scales, matrix densities, vector densities
and batches, including failures and missing combinations. This selects five
firing rates (10, 30, 100, 300, 1000 Hz) from batchmajor's eight available rates.
Retain source row counts/hashes and report the excluded batchmajor row count.
Both comparisons must share figure geometry, neuron colors and markers,
heatmap color limits and line-chart y limits. Speedup remains baseline ms/step divided
by target ms/step, normalized independently for each measurement. Preserve
failed and missing cells and all source metadata. Tile/cuSPARSE pairing also
requires matching `n_tail`; batchmajor has no such parameter.

## Validation

First reproduce the missing Tile CLI/figure support with sibling suffix-style
tests. Verify all three outputs per comparison, correct Tile labels and timing
denominators, filtering to Tile's parameter ranges, identical rendering geometry
and shared axis/color bounds, red below 1 and
white at 1, failures, missing records, and compatible `n_tail` metadata.
Run the existing regression tests, coverage above 90% for the modified plotting
module, and mypy. Regenerate real data and verify SVG syntax, per-step ratios,
source hashes and successful pair counts (Tile: 2,896). No JAX dependency changes.

## Completed validation

- 23 sibling tests pass; plotting-module line coverage is 99.03%.
- mypy passes for the plotting script and sibling tests using
  `--explicit-package-bases --follow-imports=silent`.
- Both real-data exports use 3,465 grid cells and identical SVG dimensions,
  panel identifiers, color limits and line-axis limits. All six SVGs parse and
  their per-step ratios match the audit CSVs.
- batchmajor has 2,900 valid pairs and Tile 2,896. The four additional batchmajor
  points correspond to failed Tile measurements, retained as gaps in Tile plots.
- Shared heatmap limits are 0.1–10,000; shared line limits are approximately
  0.102923–1,988.826. Each curve retains the same neuron color and marker.
- Visual inspection exposed overlapping shared and per-row y labels on narrow
  figures. A failing label-bounds test reproduced this; fixed physical margins
  resolve it for one/five columns and full/three-row figures.
- Earlier figures and audit data remain under each comparison's
  `archive/before_tile_alignment_20261018_144154/` directory.

To refresh both aligned comparisons from the plot directory:

```bash
conda run -n oper-benchmark python plot_batch_major_speedup.py --align-cusparse-to-tile
```

For future datasets, retain regression cases for failed/missing measurements,
different duration steps, speedups below one, incompatible tail metadata,
out-of-range parameters and label overlap as panel counts change.
