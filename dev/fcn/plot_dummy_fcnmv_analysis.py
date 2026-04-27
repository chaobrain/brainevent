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

"""Plot analysis figures for COBA-EI dummy/real FCNMV benchmark CSVs."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


REAL_FILE = "bit-compact-conly.csv"

NOOP_FILES = {
    "binary": "COBA-EI no op -dummy_binary_dummy_binary_dummy_kernel_post-row_gather_loop-10000-float-input-16GB.csv",
    "bitpack": "COBA-EI no op -dummy_bitpack_dummy_bitpack_dummy_kernel_post-row_gather_loop-10000-float-input-16GB.csv",
    "compact_packed": "COBA-EI no op -dummy_compact_packed_dummy_compact_dummy_kernel_post-row_gather_loop-10000-float-input-16GB.csv",
    "compact_full": "COBA-EI no op -dummy_compact_vector_full_dummy_compact_dummy_kernel_vector_full_post-row_gather_loop-10000-float-input-16GB.csv",
    "compact_active": "COBA-EI no op -dummy_compact_vector_active_dummy_compact_dummy_kernel_vector_active_post-row_gather_loop-10000-float-input-16GB.csv",
}

PLOT_STYLE = {
    "binary": {"color": "#1f2a44", "marker": "o", "label": "binary"},
    "bitpack": {"color": "#2a9d8f", "marker": "s", "label": "bitpack"},
    "compact": {"color": "#d1495b", "marker": "^", "label": "compact"},
    "compact_packed": {"color": "#d1495b", "marker": "^", "label": "compact packed"},
    "compact_full": {"color": "#edae49", "marker": "D", "label": "compact full-launch"},
    "compact_active": {"color": "#00798c", "marker": "P", "label": "compact active-launch"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmarker-test" / "dummy",
        help="Directory that contains benchmark CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmarker-test" / "dummy" / "plots",
        help="Directory to save plots and summary files.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output figure DPI.",
    )
    return parser.parse_args()


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_elapsed_map(path: Path, type_key: str = "data_type") -> dict[tuple[int, int], dict[str, float]]:
    rows = load_csv_rows(path)
    result: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (int(float(row["scale"])), int(float(row["conn_num"])))
        result[key][row[type_key]] = float(row["elapsed_s"])
    return dict(result)


def load_single_series(path: Path, series_name: str) -> dict[tuple[int, int], float]:
    rows = load_csv_rows(path)
    return {
        (int(float(row["scale"])), int(float(row["conn_num"]))): float(row["elapsed_s"])
        for row in rows
        if row.get("elapsed_s")
    }


def sorted_unique_pairs(mapping: Mapping[tuple[int, int], object]) -> tuple[list[int], list[int]]:
    scales = sorted({scale for scale, _ in mapping})
    conns = sorted({conn for _, conn in mapping})
    return scales, conns


def matrix_from_pair_values(
    scales: list[int],
    conns: list[int],
    pair_to_value: Mapping[tuple[int, int], float],
) -> np.ndarray:
    arr = np.full((len(scales), len(conns)), np.nan, dtype=float)
    scale_index = {value: idx for idx, value in enumerate(scales)}
    conn_index = {value: idx for idx, value in enumerate(conns)}
    for (scale, conn), value in pair_to_value.items():
        arr[scale_index[scale], conn_index[conn]] = value
    return arr


def ratio_map(
    data: Mapping[tuple[int, int], dict[str, float]],
    numerator: str,
    denominator: str,
) -> dict[tuple[int, int], float]:
    result: dict[tuple[int, int], float] = {}
    for key, values in data.items():
        if numerator in values and denominator in values and values[denominator] > 0:
            result[key] = values[numerator] / values[denominator]
    return result


def diff_map(
    minuend: Mapping[tuple[int, int], float],
    subtrahend: Mapping[tuple[int, int], float],
) -> dict[tuple[int, int], float]:
    common = set(minuend) & set(subtrahend)
    return {key: minuend[key] - subtrahend[key] for key in common}


def annotate_heatmap(ax, values: np.ndarray, fmt: str, fontsize: int = 7) -> None:
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isnan(value):
                continue
            if fmt == "ratio":
                text = f"{value:.2f}x"
            elif fmt == "delta":
                text = f"{value:.2f}"
            else:
                text = format(value, fmt)
            color = "white" if value > np.nanmean(values) else "#111111"
            ax.text(col, row, text, ha="center", va="center", fontsize=fontsize, color=color)


def draw_heatmap(
    ax,
    values: np.ndarray,
    scales: list[int],
    conns: list[int],
    *,
    title: str,
    value_kind: str,
    cmap: str,
    color_label: str,
) -> None:
    masked = np.ma.masked_invalid(values)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#ececec")

    if value_kind == "ratio":
        transformed = np.ma.log2(masked)
        max_abs = float(np.nanmax(np.abs(transformed))) if transformed.count() else 1.0
        norm = colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
        image = ax.imshow(transformed, origin="lower", aspect="auto", cmap=cmap_obj, norm=norm)
        annotate_heatmap(ax, values, fmt="ratio")
    elif value_kind == "delta":
        max_val = float(np.nanmax(masked)) if masked.count() else 1.0
        norm = colors.Normalize(vmin=0.0, vmax=max_val)
        image = ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap_obj, norm=norm)
        annotate_heatmap(ax, values, fmt="delta")
    else:
        raise ValueError(f"Unsupported value_kind: {value_kind!r}")

    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xlabel("conn_num")
    ax.set_ylabel("scale")
    ax.set_xticks(range(len(conns)))
    ax.set_xticklabels(conns, rotation=45, ha="right")
    ax.set_yticks(range(len(scales)))
    ax.set_yticklabels(scales)
    ax.grid(False)
    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(color_label, rotation=90)


def nearest_existing(values: Iterable[int], targets: Iterable[int], limit: int = 5) -> list[int]:
    available = sorted(set(values))
    selected: list[int] = []
    for target in targets:
        nearest = min(available, key=lambda val: abs(val - target))
        if nearest not in selected:
            selected.append(nearest)
    if len(selected) < limit:
        for value in available:
            if value not in selected:
                selected.append(value)
            if len(selected) >= limit:
                break
    return selected[:limit]


def plot_ratio_small_multiples_by_scale(
    path: Path,
    real_data: Mapping[tuple[int, int], dict[str, float]],
    scales: list[int],
    conns: list[int],
    dpi: int,
) -> None:
    chosen_scales = nearest_existing(scales, [20, 240, 460, 900, 2000])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.ravel()

    for ax, scale in zip(axes, chosen_scales):
        conn_vals = sorted(conn for s, conn in real_data if s == scale)
        bitpack = []
        compact = []
        for conn in conn_vals:
            values = real_data[(scale, conn)]
            bitpack.append(values["bitpack"] / values["binary"])
            compact.append(values["compact"] / values["binary"])
        ax.axhline(1.0, color="#1f2a44", linestyle="--", linewidth=1.0, label="binary baseline")
        ax.plot(conn_vals, bitpack, color=PLOT_STYLE["bitpack"]["color"], marker="s", linewidth=2.0, label="bitpack/binary")
        ax.plot(conn_vals, compact, color=PLOT_STYLE["compact"]["color"], marker="^", linewidth=2.0, label="compact/binary")
        ax.set_xscale("log")
        ax.set_title(f"scale={scale}", fontsize=10, weight="bold")
        ax.set_xlabel("conn_num")
        ax.set_ylabel("ratio to binary")
        ax.grid(True, alpha=0.22)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].axis("off")
    axes[-1].legend(handles, labels, loc="center", frameon=False)
    fig.suptitle("Real Benchmark Ratios by Fixed Scale", fontsize=14, weight="bold")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_ratio_small_multiples_by_conn(
    path: Path,
    real_data: Mapping[tuple[int, int], dict[str, float]],
    scales: list[int],
    conns: list[int],
    dpi: int,
) -> None:
    chosen_conns = nearest_existing(conns, [20, 462, 904, 1346, 4000])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.ravel()

    for ax, conn in zip(axes, chosen_conns):
        scale_vals = sorted(scale for scale, c in real_data if c == conn)
        bitpack = []
        compact = []
        for scale in scale_vals:
            values = real_data[(scale, conn)]
            bitpack.append(values["bitpack"] / values["binary"])
            compact.append(values["compact"] / values["binary"])
        ax.axhline(1.0, color="#1f2a44", linestyle="--", linewidth=1.0, label="binary baseline")
        ax.plot(scale_vals, bitpack, color=PLOT_STYLE["bitpack"]["color"], marker="s", linewidth=2.0, label="bitpack/binary")
        ax.plot(scale_vals, compact, color=PLOT_STYLE["compact"]["color"], marker="^", linewidth=2.0, label="compact/binary")
        ax.set_xscale("log")
        ax.set_title(f"conn_num={conn}", fontsize=10, weight="bold")
        ax.set_xlabel("scale")
        ax.set_ylabel("ratio to binary")
        ax.grid(True, alpha=0.22)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].axis("off")
    axes[-1].legend(handles, labels, loc="center", frameon=False)
    fig.suptitle("Real Benchmark Ratios by Fixed conn_num", fontsize=14, weight="bold")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_representative_elapsed_bars(
    path: Path,
    real_data: Mapping[tuple[int, int], dict[str, float]],
    noop: Mapping[str, Mapping[tuple[int, int], float]],
    dpi: int,
) -> None:
    points = [(20, 4000), (240, 2231), (460, 904), (1120, 904), (2000, 20)]
    available_points = [point for point in points if point in real_data]
    fig, axes = plt.subplots(len(available_points), 1, figsize=(10, 2.6 * len(available_points)), constrained_layout=True)
    if len(available_points) == 1:
        axes = [axes]

    for ax, point in zip(axes, available_points):
        labels = ["binary", "bitpack", "compact"]
        real_vals = [real_data[point][label] for label in labels]
        noop_vals = [
            noop["binary"].get(point, np.nan),
            noop["bitpack"].get(point, np.nan),
            noop["compact_active"].get(point, np.nan),
        ]
        x = np.arange(len(labels))
        width = 0.38
        ax.bar(x - width / 2, real_vals, width, color=["#1f2a44", "#2a9d8f", "#d1495b"], label="real")
        ax.bar(x + width / 2, noop_vals, width, color=["#7a8599", "#88c9c0", "#ef9aa6"], label="no-op match")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("elapsed_s")
        ax.set_title(f"Representative point scale={point[0]}, conn_num={point[1]}", fontsize=10, weight="bold")
        ax.grid(True, axis="y", alpha=0.22)

    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle("Real vs Matching no-op Elapsed Time", fontsize=14, weight="bold")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    path: Path,
    real_data: Mapping[tuple[int, int], dict[str, float]],
    noop: Mapping[str, Mapping[tuple[int, int], float]],
) -> None:
    common_points = sorted(
        key for key, values in real_data.items() if {"binary", "bitpack", "compact"} <= set(values)
    )
    compact_ratios = [(key, real_data[key]["compact"] / real_data[key]["binary"]) for key in common_points]
    bitpack_ratios = [(key, real_data[key]["bitpack"] / real_data[key]["binary"]) for key in common_points]
    compact_ratios.sort(key=lambda item: item[1])
    bitpack_ratios.sort(key=lambda item: item[1])

    def median(values: list[float]) -> float:
        ordered = sorted(values)
        return ordered[len(ordered) // 2]

    lines = [
        "# Dummy FCNMV Analysis Summary",
        "",
        f"- Common real benchmark points: {len(common_points)}",
        f"- compact/binary mean ratio: {np.mean([ratio for _, ratio in compact_ratios]):.4f}",
        f"- compact/binary median ratio: {median([ratio for _, ratio in compact_ratios]):.4f}",
        f"- bitpack/binary mean ratio: {np.mean([ratio for _, ratio in bitpack_ratios]):.4f}",
        f"- bitpack/binary median ratio: {median([ratio for _, ratio in bitpack_ratios]):.4f}",
        "",
        "## Best compact points",
    ]
    for point, ratio in compact_ratios[:5]:
        values = real_data[point]
        lines.append(
            f"- scale={point[0]}, conn_num={point[1]}: compact/binary={ratio:.4f}, "
            f"compact={values['compact']:.6f}s, binary={values['binary']:.6f}s"
        )
    lines.extend(["", "## Worst compact points"])
    for point, ratio in compact_ratios[-5:]:
        values = real_data[point]
        lines.append(
            f"- scale={point[0]}, conn_num={point[1]}: compact/binary={ratio:.4f}, "
            f"compact={values['compact']:.6f}s, binary={values['binary']:.6f}s"
        )

    compact_real_minus_active = diff_map(
        {key: value["compact"] for key, value in real_data.items()},
        noop["compact_active"],
    )
    if compact_real_minus_active:
        mean_delta = np.mean(list(compact_real_minus_active.values()))
        lines.extend(
            [
                "",
                "## Real-minus-no-op",
                f"- compact(real - active-noop) mean delta: {mean_delta:.6f}s",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    real_path = base_dir / REAL_FILE
    real_data = load_elapsed_map(real_path)
    scales, conns = sorted_unique_pairs(real_data)

    noop_series = {
        name: load_single_series(base_dir / filename, name)
        for name, filename in NOOP_FILES.items()
    }

    compact_ratio = ratio_map(real_data, "compact", "binary")
    bitpack_ratio = ratio_map(real_data, "bitpack", "binary")
    real_binary = {key: values["binary"] for key, values in real_data.items() if "binary" in values}
    real_bitpack = {key: values["bitpack"] for key, values in real_data.items() if "bitpack" in values}
    real_compact = {key: values["compact"] for key, values in real_data.items() if "compact" in values}

    ratio_scales = sorted({scale for scale, _ in compact_ratio})
    ratio_conns = sorted({conn for _, conn in compact_ratio})

    compact_ratio_mat = matrix_from_pair_values(ratio_scales, ratio_conns, compact_ratio)
    bitpack_ratio_mat = matrix_from_pair_values(ratio_scales, ratio_conns, bitpack_ratio)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    draw_heatmap(
        axes[0],
        bitpack_ratio_mat,
        ratio_scales,
        ratio_conns,
        title="Real Benchmark: bitpack / binary",
        value_kind="ratio",
        cmap="RdBu_r",
        color_label="log2 ratio",
    )
    draw_heatmap(
        axes[1],
        compact_ratio_mat,
        ratio_scales,
        ratio_conns,
        title="Real Benchmark: compact / binary",
        value_kind="ratio",
        cmap="RdBu_r",
        color_label="log2 ratio",
    )
    fig.suptitle("COBA-EI FCNMV Relative Performance Heatmaps", fontsize=15, weight="bold")
    fig.savefig(output_dir / "01_real_ratio_heatmaps.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    plot_ratio_small_multiples_by_scale(
        output_dir / "02_real_ratio_by_scale.png",
        real_data,
        scales,
        conns,
        args.dpi,
    )
    plot_ratio_small_multiples_by_conn(
        output_dir / "03_real_ratio_by_conn.png",
        real_data,
        scales,
        conns,
        args.dpi,
    )

    noop_common = sorted(set(noop_series["compact_packed"]) & set(noop_series["compact_full"]) & set(noop_series["compact_active"]))
    noop_scales = sorted({scale for scale, _ in noop_common})
    noop_conns = sorted({conn for _, conn in noop_common})
    packed_full = {key: noop_series["compact_packed"][key] / noop_series["compact_full"][key] for key in noop_common}
    active_full = {key: noop_series["compact_active"][key] / noop_series["compact_full"][key] for key in noop_common}
    packed_active = {key: noop_series["compact_packed"][key] / noop_series["compact_active"][key] for key in noop_common}

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)
    draw_heatmap(
        axes[0],
        matrix_from_pair_values(noop_scales, noop_conns, packed_full),
        noop_scales,
        noop_conns,
        title="no-op: compact packed / full-launch",
        value_kind="ratio",
        cmap="RdBu_r",
        color_label="log2 ratio",
    )
    draw_heatmap(
        axes[1],
        matrix_from_pair_values(noop_scales, noop_conns, active_full),
        noop_scales,
        noop_conns,
        title="no-op: compact active-launch / full-launch",
        value_kind="ratio",
        cmap="RdBu_r",
        color_label="log2 ratio",
    )
    draw_heatmap(
        axes[2],
        matrix_from_pair_values(noop_scales, noop_conns, packed_active),
        noop_scales,
        noop_conns,
        title="no-op: compact packed / active-launch",
        value_kind="ratio",
        cmap="RdBu_r",
        color_label="log2 ratio",
    )
    fig.suptitle("Compact no-op Variant Decomposition", fontsize=15, weight="bold")
    fig.savefig(output_dir / "04_noop_compact_variant_heatmaps.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    delta_maps = {
        "binary": diff_map(real_binary, noop_series["binary"]),
        "bitpack": diff_map(real_bitpack, noop_series["bitpack"]),
        "compact": diff_map(real_compact, noop_series["compact_active"]),
    }
    delta_scales = sorted({scale for mapping in delta_maps.values() for scale, _ in mapping})
    delta_conns = sorted({conn for mapping in delta_maps.values() for _, conn in mapping})

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)
    draw_heatmap(
        axes[0],
        matrix_from_pair_values(delta_scales, delta_conns, delta_maps["binary"]),
        delta_scales,
        delta_conns,
        title="real - no-op: binary",
        value_kind="delta",
        cmap="YlOrRd",
        color_label="delta seconds",
    )
    draw_heatmap(
        axes[1],
        matrix_from_pair_values(delta_scales, delta_conns, delta_maps["bitpack"]),
        delta_scales,
        delta_conns,
        title="real - no-op: bitpack",
        value_kind="delta",
        cmap="YlOrRd",
        color_label="delta seconds",
    )
    draw_heatmap(
        axes[2],
        matrix_from_pair_values(delta_scales, delta_conns, delta_maps["compact"]),
        delta_scales,
        delta_conns,
        title="real - no-op: compact vs active-launch dummy",
        value_kind="delta",
        cmap="YlOrRd",
        color_label="delta seconds",
    )
    fig.suptitle("Real-minus-no-op Delta Heatmaps", fontsize=15, weight="bold")
    fig.savefig(output_dir / "05_real_minus_noop_heatmaps.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    plot_representative_elapsed_bars(
        output_dir / "06_representative_elapsed_bars.png",
        real_data,
        noop_series,
        args.dpi,
    )

    write_summary(output_dir / "analysis_summary.md", real_data, noop_series)

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
