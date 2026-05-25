#!/usr/bin/env python3

# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Compare no-op dummy FCNMV runs against full COBA-EI runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


REAL_FILES = {
    "binary": "COBA-EI-4.27_binary_homo_cuda_raw_post-row_gather-float-input-16GB.csv",
    "bitpack": "COBA-EI-4.27_bitpack_homo_cuda_raw_post-row_gather-float-input-16GB.csv",
    "compact": "COBA-EI-4.27_compact_homo_cuda_raw_post-row_gather-float-input-16GB.csv",
}

DUMMY_FILES = {
    "binary": "COBA-EI no op -dummy_binary_dummy_binary_dummy_kernel_post-row_gather_loop-10000-float-input-16GB.csv",
    "bitpack": "COBA-EI no op -dummy_bitpack_dummy_bitpack_dummy_kernel_post-row_gather_loop-10000-float-input-16GB.csv",
    "compact": "COBA-EI no op -dummy_compact_vector_active_dummy_compact_dummy_kernel_vector_active_post-row_gather_loop-10000-float-input-16GB.csv",
}

SERIES_COLORS = {
    "binary": "#1f2a44",
    "bitpack": "#2a9d8f",
    "compact": "#d1495b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmarker-test",
        help="Directory containing COBA benchmark CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmarker-test" / "preprocess-plots",
        help="Directory to save plots and merged analysis tables.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure DPI.",
    )
    return parser.parse_args()


def load_elapsed_series(path: Path) -> tuple[dict[tuple[int, int], float], float]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    durations = {float(row["duration"]) for row in rows}
    if len(durations) != 1:
        raise ValueError(f"Expected one duration value in {path}, got {sorted(durations)}")
    duration = next(iter(durations))
    data = {
        (int(float(row["scale"])), int(float(row["conn_num"]))): float(row["elapsed_s"])
        for row in rows
    }
    return data, duration


def build_grid(values: Mapping[tuple[int, int], float]) -> tuple[list[int], list[int], np.ndarray]:
    scales = sorted({scale for scale, _ in values})
    conns = sorted({conn for _, conn in values})
    arr = np.full((len(scales), len(conns)), np.nan, dtype=float)
    scale_idx = {value: idx for idx, value in enumerate(scales)}
    conn_idx = {value: idx for idx, value in enumerate(conns)}
    for (scale, conn), value in values.items():
        arr[scale_idx[scale], conn_idx[conn]] = value
    return scales, conns, arr


def draw_percent_heatmap(
    ax,
    values: Mapping[tuple[int, int], float],
    *,
    title: str,
    color_label: str,
    vmax: float | None = None,
) -> None:
    scales, conns, arr = build_grid(values)
    masked = np.ma.masked_invalid(arr)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="#efefef")
    vmax = float(np.nanmax(arr)) if vmax is None else vmax
    image = ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xlabel("conn_num")
    ax.set_ylabel("scale")
    ax.set_xticks(range(len(conns)))
    ax.set_xticklabels(conns, rotation=45, ha="right")
    ax.set_yticks(range(len(scales)))
    ax.set_yticklabels(scales)
    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(color_label, rotation=90)


def draw_ratio_heatmap(
    ax,
    values: Mapping[tuple[int, int], float],
    *,
    title: str,
    color_label: str,
) -> None:
    scales, conns, arr = build_grid(values)
    masked = np.ma.masked_invalid(arr)
    transformed = np.ma.log2(masked)
    max_abs = float(np.nanmax(np.abs(transformed))) if transformed.count() else 1.0
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#efefef")
    image = ax.imshow(
        transformed,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs),
    )
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xlabel("conn_num")
    ax.set_ylabel("scale")
    ax.set_xticks(range(len(conns)))
    ax.set_xticklabels(conns, rotation=45, ha="right")
    ax.set_yticks(range(len(scales)))
    ax.set_yticklabels(scales)
    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(color_label, rotation=90)


def nearest_existing(values: Iterable[int], targets: Iterable[int], limit: int) -> list[int]:
    available = sorted(set(values))
    chosen: list[int] = []
    for target in targets:
        nearest = min(available, key=lambda value: abs(value - target))
        if nearest not in chosen:
            chosen.append(nearest)
    for value in available:
        if value not in chosen:
            chosen.append(value)
        if len(chosen) >= limit:
            break
    return chosen[:limit]


def pick_balanced_representative_points(
    point_map: Mapping[tuple[int, int], dict[str, float | int | None]],
) -> list[tuple[int, int]]:
    by_scale: dict[int, list[int]] = {}
    for scale, conn in sorted(point_map):
        by_scale.setdefault(scale, []).append(conn)

    chosen_scales = nearest_existing(sorted(by_scale), [20, 267, 515, 1010, 1752, 2000], limit=6)
    selected: list[tuple[int, int]] = []
    for scale in chosen_scales:
        conns = sorted(by_scale[scale])
        if len(conns) <= 3:
            picks = conns
        else:
            picks = [conns[0], conns[(len(conns) - 1) // 2], conns[-1]]
        for conn in picks:
            point = (scale, conn)
            if point not in selected:
                selected.append(point)
    return selected


def plot_share_curves_by_scale(
    path: Path,
    shares: Mapping[str, Mapping[tuple[int, int], float]],
    dpi: int,
) -> None:
    all_points = {point for mapping in shares.values() for point in mapping}
    scales = sorted({scale for scale, _ in all_points})
    chosen_scales = nearest_existing(scales, [20, 267, 1010, 2000], limit=4)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes = axes.ravel()

    for ax, scale in zip(axes, chosen_scales):
        for name in ("binary", "bitpack", "compact"):
            conn_vals = sorted(conn for (s, conn) in shares[name] if s == scale)
            y_vals = [shares[name][(scale, conn)] for conn in conn_vals]
            ax.plot(
                conn_vals,
                y_vals,
                color=SERIES_COLORS[name],
                marker="o",
                linewidth=2.0,
                label=name,
            )
        ax.set_title(f"scale={scale}", fontsize=10, weight="bold")
        ax.set_xlabel("conn_num")
        ax.set_ylabel("dummy share of full COBA (%)")
        ax.grid(True, alpha=0.22)
        ax.set_ylim(bottom=0.0)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, frameon=False)
    fig.suptitle("Scaled no-op Dummy Share by Fixed Scale", fontsize=14, weight="bold")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_representative_decomposition(
    path: Path,
    merged_rows: list[dict[str, float | int | None]],
    dpi: int,
) -> None:
    point_map = {(int(row["scale"]), int(row["conn_num"])): row for row in merged_rows}
    chosen = pick_balanced_representative_points(point_map)
    n_cols = 5
    n_rows = int(np.ceil(len(chosen) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.6 * n_rows), constrained_layout=True)
    axes = axes.ravel()

    for ax, point in zip(axes, chosen):
        row = point_map[point]
        labels = ["binary", "bitpack", "compact"]
        total = np.array([row["real_binary"], row["real_bitpack"], row["real_compact"]], dtype=float)
        generic = np.array([row["dummy_binary_scaled"]] * 3, dtype=float)
        extra = np.array(
            [
                0.0,
                row["dummy_bitpack_scaled"] - row["dummy_binary_scaled"],
                row["dummy_compact_scaled"] - row["dummy_binary_scaled"],
            ],
            dtype=float,
        )
        extra = np.maximum(extra, 0.0)
        remaining = np.maximum(total - generic - extra, 0.0)
        x = np.arange(len(labels))
        ax.bar(x, generic, color="#b0b8c5", label="generic no-op baseline")
        ax.bar(x, extra, bottom=generic, color="#ef9aa6", label="extra representation/no-op delta")
        ax.bar(x, remaining, bottom=generic + extra, color="#7cc6b8", label="residual not explained by dummy")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("elapsed_s")
        ax.set_title(f"scale={point[0]}, conn={point[1]}", fontsize=10, weight="bold")
        ax.grid(True, axis="y", alpha=0.22)

    for ax in axes[len(chosen):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, frameon=False, fontsize=8)
    fig.suptitle("Balanced Runtime Decomposition Across scale/conn Bands", fontsize=14, weight="bold")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    real = {}
    real_duration = {}
    for name, filename in REAL_FILES.items():
        real[name], real_duration[name] = load_elapsed_series(base_dir / filename)

    dummy = {}
    dummy_duration = {}
    for name, filename in DUMMY_FILES.items():
        dummy[name], dummy_duration[name] = load_elapsed_series(base_dir / "dummy" / filename)

    scaled_dummy = {}
    for name in REAL_FILES:
        scale_factor = real_duration[name] / dummy_duration[name]
        scaled_dummy[name] = {point: value * scale_factor for point, value in dummy[name].items()}

    preprocess_share_pct = {}
    for name in REAL_FILES:
        share = {}
        for point, real_elapsed in real[name].items():
            if point in scaled_dummy[name] and real_elapsed > 0:
                share[point] = 100.0 * scaled_dummy[name][point] / real_elapsed
        preprocess_share_pct[name] = share

    extra_share_pct = {"bitpack": {}, "compact": {}}
    common_with_binary_dummy = set(scaled_dummy["binary"])
    for name in ("bitpack", "compact"):
        for point, real_elapsed in real[name].items():
            if point in scaled_dummy[name] and point in common_with_binary_dummy and real_elapsed > 0:
                delta = max(0.0, scaled_dummy[name][point] - scaled_dummy["binary"][point])
                extra_share_pct[name][point] = 100.0 * delta / real_elapsed

    compact_bitpack_ratio = {}
    compact_bitpack_dummy_delta = {}
    compact_bitpack_explain_pct = {}
    common_cbp = set(real["compact"]) & set(real["bitpack"]) & set(scaled_dummy["compact"]) & set(scaled_dummy["bitpack"])
    for point in common_cbp:
        real_delta = real["compact"][point] - real["bitpack"][point]
        dummy_delta = scaled_dummy["compact"][point] - scaled_dummy["bitpack"][point]
        compact_bitpack_ratio[point] = real["compact"][point] / real["bitpack"][point]
        compact_bitpack_dummy_delta[point] = 1000.0 * dummy_delta
        if real_delta > 0.0:
            compact_bitpack_explain_pct[point] = 100.0 * dummy_delta / real_delta

    compact_binary_ratio = {}
    compact_binary_explain_pct = {}
    common_cb = set(real["compact"]) & set(real["binary"]) & set(scaled_dummy["compact"]) & set(scaled_dummy["binary"])
    for point in common_cb:
        compact_binary_ratio[point] = real["compact"][point] / real["binary"][point]
        real_delta = real["compact"][point] - real["binary"][point]
        dummy_delta = scaled_dummy["compact"][point] - scaled_dummy["binary"][point]
        if abs(real_delta) > 1e-12:
            compact_binary_explain_pct[point] = 100.0 * dummy_delta / real_delta

    vmax_share = max(max(values.values()) for values in preprocess_share_pct.values())
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)
    draw_percent_heatmap(
        axes[0],
        preprocess_share_pct["binary"],
        title="binary dummy / full COBA",
        color_label="share (%)",
        vmax=vmax_share,
    )
    draw_percent_heatmap(
        axes[1],
        preprocess_share_pct["bitpack"],
        title="bitpack dummy / full COBA",
        color_label="share (%)",
        vmax=vmax_share,
    )
    draw_percent_heatmap(
        axes[2],
        preprocess_share_pct["compact"],
        title="compact-active dummy / full COBA",
        color_label="share (%)",
        vmax=vmax_share,
    )
    fig.suptitle("Scaled no-op Dummy Share of Full COBA Runtime", fontsize=15, weight="bold")
    fig.savefig(output_dir / "01_preprocess_share_heatmaps.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    vmax_extra = max(
        max(values.values()) if values else 0.0
        for values in (extra_share_pct["bitpack"], extra_share_pct["compact"])
    )
    extra_diff = {
        point: extra_share_pct["compact"][point] - extra_share_pct["bitpack"][point]
        for point in set(extra_share_pct["compact"]) & set(extra_share_pct["bitpack"])
    }
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)
    draw_percent_heatmap(
        axes[0],
        extra_share_pct["bitpack"],
        title="bitpack extra share over binary dummy baseline",
        color_label="share (%)",
        vmax=vmax_extra,
    )
    draw_percent_heatmap(
        axes[1],
        extra_share_pct["compact"],
        title="compact extra share over binary dummy baseline",
        color_label="share (%)",
        vmax=vmax_extra,
    )
    draw_percent_heatmap(
        axes[2],
        extra_diff,
        title="compact extra share - bitpack extra share",
        color_label="delta share (%)",
        vmax=max(extra_diff.values()) if extra_diff else 1.0,
    )
    fig.suptitle("Type-specific no-op Overhead Beyond Binary Baseline", fontsize=15, weight="bold")
    fig.savefig(output_dir / "02_extra_share_heatmaps.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    plot_share_curves_by_scale(
        output_dir / "03_share_curves_by_scale.png",
        preprocess_share_pct,
        args.dpi,
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    draw_ratio_heatmap(
        axes[0, 0],
        compact_binary_ratio,
        title="real compact / binary",
        color_label="log2 ratio",
    )
    draw_percent_heatmap(
        axes[0, 1],
        compact_binary_explain_pct,
        title="compact-vs-binary slowdown explained by dummy delta",
        color_label="explained (%)",
        vmax=max(compact_binary_explain_pct.values()) if compact_binary_explain_pct else 100.0,
    )
    draw_ratio_heatmap(
        axes[1, 0],
        compact_bitpack_ratio,
        title="real compact / bitpack",
        color_label="log2 ratio",
    )
    draw_percent_heatmap(
        axes[1, 1],
        compact_bitpack_explain_pct,
        title="compact-vs-bitpack slowdown explained by dummy delta",
        color_label="explained (%)",
        vmax=max(compact_bitpack_explain_pct.values()) if compact_bitpack_explain_pct else 100.0,
    )
    fig.suptitle("How Much of compact Slowdown Is Visible in no-op Dummy Runs", fontsize=15, weight="bold")
    fig.savefig(output_dir / "04_slowdown_explanation_heatmaps.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    plot_representative_decomposition(
        output_dir / "05_representative_decomposition.png",
        [
            {
                "scale": point[0],
                "conn_num": point[1],
                "real_binary": real["binary"].get(point),
                "real_bitpack": real["bitpack"].get(point),
                "real_compact": real["compact"].get(point),
                "dummy_binary_scaled": scaled_dummy["binary"].get(point),
                "dummy_bitpack_scaled": scaled_dummy["bitpack"].get(point),
                "dummy_compact_scaled": scaled_dummy["compact"].get(point),
            }
            for point in sorted(set(real["compact"]) & set(real["bitpack"]) & set(real["binary"]) & set(scaled_dummy["binary"]) & set(scaled_dummy["bitpack"]) & set(scaled_dummy["compact"]))
        ],
        args.dpi,
    )

    merged_points = sorted(
        set(real["binary"]) | set(real["bitpack"]) | set(real["compact"]) | set(scaled_dummy["binary"]) | set(scaled_dummy["bitpack"]) | set(scaled_dummy["compact"])
    )
    merged_rows: list[dict[str, float | int | None]] = []
    for scale, conn in merged_points:
        row: dict[str, float | int | None] = {
            "scale": scale,
            "conn_num": conn,
            "real_binary": real["binary"].get((scale, conn)),
            "real_bitpack": real["bitpack"].get((scale, conn)),
            "real_compact": real["compact"].get((scale, conn)),
            "dummy_binary_scaled": scaled_dummy["binary"].get((scale, conn)),
            "dummy_bitpack_scaled": scaled_dummy["bitpack"].get((scale, conn)),
            "dummy_compact_scaled": scaled_dummy["compact"].get((scale, conn)),
            "share_binary_pct": preprocess_share_pct["binary"].get((scale, conn)),
            "share_bitpack_pct": preprocess_share_pct["bitpack"].get((scale, conn)),
            "share_compact_pct": preprocess_share_pct["compact"].get((scale, conn)),
            "extra_share_bitpack_pct": extra_share_pct["bitpack"].get((scale, conn)),
            "extra_share_compact_pct": extra_share_pct["compact"].get((scale, conn)),
            "compact_over_binary": compact_binary_ratio.get((scale, conn)),
            "compact_over_bitpack": compact_bitpack_ratio.get((scale, conn)),
            "compact_binary_explained_pct": compact_binary_explain_pct.get((scale, conn)),
            "compact_bitpack_explained_pct": compact_bitpack_explain_pct.get((scale, conn)),
            "compact_bitpack_dummy_delta_ms": compact_bitpack_dummy_delta.get((scale, conn)),
        }
        merged_rows.append(row)

    merged_csv_path = output_dir / "preprocess_share_table.csv"
    fieldnames = list(merged_rows[0].keys())
    with merged_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    summary_lines = [
        "# COBA-vs-dummy preprocessing summary",
        "",
        f"- binary scale factor: {real_duration['binary'] / dummy_duration['binary']:.6f}",
        f"- bitpack scale factor: {real_duration['bitpack'] / dummy_duration['bitpack']:.6f}",
        f"- compact scale factor: {real_duration['compact'] / dummy_duration['compact']:.6f}",
        "",
        f"- binary median share (%): {np.median(list(preprocess_share_pct['binary'].values())):.4f}",
        f"- bitpack median share (%): {np.median(list(preprocess_share_pct['bitpack'].values())):.4f}",
        f"- compact median share (%): {np.median(list(preprocess_share_pct['compact'].values())):.4f}",
        "",
        f"- compact-vs-bitpack explained median (%): {np.median(list(compact_bitpack_explain_pct.values())):.4f}",
        f"- compact-vs-bitpack explained max (%): {np.max(list(compact_bitpack_explain_pct.values())):.4f}",
        f"- compact-vs-binary explained median (%): {np.median(list(compact_binary_explain_pct.values())):.4f}",
        f"- compact-vs-binary explained min (%): {np.min(list(compact_binary_explain_pct.values())):.4f}",
        "",
        f"- merged table: {merged_csv_path}",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Saved preprocess-share plots to: {output_dir}")


if __name__ == "__main__":
    main()
