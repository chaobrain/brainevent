import math
import os
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Dict, List, Optional, Tuple, Sequence

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
#from fire import build_speedup_matrix, build_white_center_colormap
from matplotlib import colors

# ─────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────
HATCH_PATTERNS = ['', '///', '\\\\\\', 'xxx', '...', '+++', 'ooo', '***', 'OOO']
PALETTE_NAME = "tab10"
ALL_LABEL = "(All)"
NONE_LABEL = "(None)"
NUMERIC_COLS = [
    'mean_ms', 'std_ms', 'min_ms', 'max_ms',
    'elapsed_min_s', 'elapsed_max_s', 'elapsed_mean_s', 'elapsed_std_s',
    'elapsed_s', 'firing_rate', 'firing_rate_mean_hz', 'speedup_vs_baseline'
]

Y_COL_LABELS = {
    'mean_ms': 'Mean Time (ms)',
    'std_ms':  'Std Dev (ms)',
    'min_ms':  'Min Time (ms)',
    'max_ms':  'Max Time (ms)',
}

def _sort_key(val_str: str) -> float:
    s = str(val_str)
    if 'x' in s.lower():
        nums = re.findall(r'\d+\.?\d*', s)
        if nums:
            return math.prod(float(n) for n in nums)
    try:
        return float(s)
    except ValueError:
        nums = re.findall(r'\d+\.?\d*', s)
        return float(nums[0]) if nums else 0.0


def load_and_transform_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.replace(r'^\s*nan\s*$', np.nan, regex=True, inplace=False)
    if 'mean_ms' in df.columns:
        df['mean_ms'] = pd.to_numeric(df['mean_ms'], errors='coerce')
        df = df.dropna(subset=['mean_ms'])
    elif 'elapsed_mean_s' in df.columns:
        df['elapsed_mean_s'] = pd.to_numeric(df['elapsed_mean_s'], errors='coerce')
        df = df.dropna(subset=['elapsed_mean_s'])

    # Keep behavior close to the original script: parse config_name into explicit columns.
    if 'config_name' in df.columns:
        all_keys = set()
        parsed = []
        for row_idx, name in df['config_name'].items():
            d = {}
            for token in [p.strip() for p in str(name).split(',') if '=' in p]:
                k, v = token.split('=', 1)
                k = k.strip()
                v = v.strip()
                if k in d:
                    raise ValueError(
                        f"Duplicate key '{k}' in config_name at row {row_idx}: {name}"
                    )
                d[k] = v
                all_keys.add(k)
            parsed.append(d)
        cfg_df = pd.DataFrame(parsed, index=df.index)
        for k in sorted(all_keys):
            if k not in df.columns:
                df[k] = cfg_df.get(k)
    return df


def filter_base_data(
    df: pd.DataFrame,
    primitive: str,
    base_tags: Sequence[Tuple[str, str]],
    target_backend: str,
    baseline_backend: str,
) -> pd.DataFrame:
    sub = df[df['primitive_name'].astype(str) == str(primitive)].copy()
    for k, v in base_tags:
        if k in sub.columns:
            sub = sub[sub[k].astype(str) == str(v)]
    sub = sub[sub['backend'].astype(str).isin([str(target_backend), str(baseline_backend)])]
    return sub


def build_speedup_matrix(
    df: pd.DataFrame,
    target_backend: str,
    baseline_backend: str,
    scale_col: str = 'scale',
    spike_col: str = 'spike_rate',
    val_col: str = 'mean_ms',
    key_col: str = 'backend',
) -> pd.DataFrame:
    required = [key_col, val_col, scale_col, spike_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df.copy()
    data[val_col] = pd.to_numeric(data[val_col], errors='coerce')
    data = data.dropna(subset=[val_col, scale_col, spike_col])

    data[scale_col] = data[scale_col].astype(str)
    data[spike_col] = data[spike_col].astype(str)

    key_cols = [key_col, spike_col, scale_col]
    dup = data[data.duplicated(subset=key_cols, keep=False)]
    if not dup.empty:
        raise ValueError(
            "Duplicate rows detected for heatmap keys "
            f"{key_cols}. Please make each ({key_col}, {spike_col}, {scale_col}) unique."
        )

    scale_order = sorted(data[scale_col].unique().tolist(), key=_sort_key)
    spike_order = sorted(data[spike_col].unique().tolist(), key=_sort_key)

    target_df = data[data[key_col].astype(str) == str(target_backend)]
    base_df = data[data[key_col].astype(str) == str(baseline_backend)]

    target_pivot = target_df.pivot(
        index=spike_col,
        columns=scale_col,
        values=val_col,
    )
    base_pivot = base_df.pivot(
        index=spike_col,
        columns=scale_col,
        values=val_col,
    )

    target_pivot = target_pivot.reindex(index=spike_order, columns=scale_order)
    base_pivot = base_pivot.reindex(index=spike_order, columns=scale_order)
    return base_pivot / target_pivot


def build_white_center_colormap(
    yellow_ratio: float = 0.5,
    blue_ratio: float = 1.5,
    lemon_yellow: str = '#FFF44F',
    blue: str = '#4F81BD',
) -> Tuple[colors.LinearSegmentedColormap, colors.Normalize]:
    if not (yellow_ratio < 1.0 < blue_ratio):
        raise ValueError('Require yellow_ratio < 1.0 < blue_ratio.')
    t_white = (1.0 - yellow_ratio) / (blue_ratio - yellow_ratio)
    cmap = colors.LinearSegmentedColormap.from_list(
        'yellow_white_blue_centered',
        [(0.0, lemon_yellow), (t_white, '#FFFFFF'), (1.0, blue)],
    )
    norm = colors.Normalize(vmin=yellow_ratio, vmax=blue_ratio, clip=True)
    return cmap, norm


def plot_speedup_heatmap(
    speedup_matrix: pd.DataFrame,
    title: str,
    output_filename: str,
    dpi: int = 300,
    annotate: bool = True,
    yellow_ratio: float = 0.5,
    blue_ratio: float = 1.5,
) -> None:
    cmap, norm = build_white_center_colormap(
        yellow_ratio=yellow_ratio,
        blue_ratio=blue_ratio,
    )

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
    sns.heatmap(
        speedup_matrix,
        annot=annotate,
        fmt='.2f',
        cmap=cmap,
        norm=norm,
        linewidths=0.5,
        cbar_kws={
            'label': f'Speedup (baseline/target), 1.0=white, '
                     f'{yellow_ratio:.2f}=yellow, {blue_ratio:.2f}=blue'
        },
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Scale', fontsize=11)
    ax.set_ylabel('Spike Rate', fontsize=11)
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

# ─────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.replace(r'^\s*nan\s*$', np.nan, regex=True)
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'mean_ms' in df.columns:
        df = df.dropna(subset=['mean_ms'])
    elif 'elapsed_mean_s' in df.columns:
        df = df.dropna(subset=['elapsed_mean_s'])
    elif 'elapsed_s' in df.columns:
        df = df.dropna(subset=['elapsed_s'])
    return df


def parse_config_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    if 'config_name' not in df.columns:
        return df, []
    all_keys: set = set()
    parsed_rows: List[dict] = []
    for row_idx, cfg in df['config_name'].items():
        parts = [p.strip() for p in str(cfg).split(',') if '=' in p]
        d = {}
        for p in parts:
            k, v = p.split('=', 1)
            k, v = k.strip(), v.strip()
            if k in d:
                raise ValueError(
                    f"Duplicate key '{k}' in config_name at row {row_idx}: {cfg}"
                )
            d[k] = v
            all_keys.add(k)
        parsed_rows.append(d)
    config_df = pd.DataFrame(parsed_rows, index=df.index)
    keys = sorted(all_keys)
    new_cols = [c for c in config_df.columns if c not in df.columns]
    result = pd.concat([df, config_df[new_cols]], axis=1)
    return result, keys


def sort_key(val_str) -> float:
    s = str(val_str)
    if 'x' in s.lower():
        nums = re.findall(r'\d+\.?\d*', s)
        if nums:
            return math.prod(float(n) for n in nums)
    try:
        return float(s)
    except ValueError:
        return 0.0


def _cap(s: str) -> str:
    """Capitalize first letter of every word, keep rest."""
    return s.title() if s else s


# ─────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────
class CompareApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Benchmark Compare Viewer")
        self.state("zoomed")

        self.raw_df = pd.DataFrame()
        self.df = pd.DataFrame()
        self.config_keys: List[str] = []

        # ── Shared variables ──
        self.operator_var = tk.StringVar()

        # Tab 1 – Latency
        self.t1_x_var      = tk.StringVar()
        self.t1_y_var      = tk.StringVar(value='mean_ms')
        self.t1_color_var  = tk.StringVar(value='backend')
        self.t1_hatch_var  = tk.StringVar(value=NONE_LABEL)
        self.t1_scale_var  = tk.StringVar(value='linear')
        self.t1_annot_var  = tk.BooleanVar(value=True)

        # Tab 2 – Speedup
        self.t2_x_var      = tk.StringVar()
        self.t2_y_var      = tk.StringVar(value='mean_ms')
        self.t2_baseline_var = tk.StringVar()
        self.t2_color_var  = tk.StringVar(value='backend')
        self.t2_hatch_var  = tk.StringVar(value=NONE_LABEL)
        self.t2_scale_var  = tk.StringVar(value='linear')
        self.t2_annot_var  = tk.BooleanVar(value=True)

        # Tab 3 - Speedup Heatmap
        self.t3_target_var = tk.StringVar()
        self.t3_baseline_var = tk.StringVar()
        self.t3_yellow_ratio_var = tk.StringVar(value='0.5')
        self.t3_blue_ratio_var = tk.StringVar(value='1.5')
        self.t3_annot_var = tk.BooleanVar(value=True)

        # Tab 4 - COBA Absolute
        self.c1_x_var = tk.StringVar()
        self.c1_y_var = tk.StringVar()
        self.c1_color_var = tk.StringVar(value='backend')
        self.c1_hatch_var = tk.StringVar(value=NONE_LABEL)
        self.c1_scale_var = tk.StringVar(value='linear')
        self.c1_annot_var = tk.BooleanVar(value=True)

        # Tab 5 - COBA Speedup
        self.c2_x_var = tk.StringVar()
        self.c2_y_var = tk.StringVar()
        self.c2_baseline_var = tk.StringVar()
        self.c2_color_var = tk.StringVar(value='backend')
        self.c2_hatch_var = tk.StringVar(value=NONE_LABEL)
        self.c2_scale_var = tk.StringVar(value='linear')
        self.c2_annot_var = tk.BooleanVar(value=True)

        # Tab 6 - COBA Heatmap
        self.c3_target_var = tk.StringVar()
        self.c3_baseline_var = tk.StringVar()
        self.c3_yellow_ratio_var = tk.StringVar(value='0.5')
        self.c3_blue_ratio_var = tk.StringVar(value='1.5')
        self.c3_annot_var = tk.BooleanVar(value=True)

        # Filters (dynamic)
        self.filter_vars: Dict[str, tk.StringVar] = {}
        self._combo_refs: Dict[int, ttk.Combobox] = {}

        self._build_ui()

    # ═══════════════ UI build ═══════════════
    def _build_ui(self):
        # Menu
        menu_bar = tk.Menu(self)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open CSV...", command=self._open_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.destroy)
        menu_bar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menu_bar)

        # Notebook (tabs)
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill='both', expand=True)

        # --- Tab 1: Absolute Latency ---
        tab1 = ttk.Frame(self.nb)
        self.nb.add(tab1, text="Absolute Latency")
        self._build_tab_latency(tab1)

        # --- Tab 2: Speedup ---
        tab2 = ttk.Frame(self.nb)
        self.nb.add(tab2, text="Speedup")
        self._build_tab_speedup(tab2)

        # --- Tab 3: Speedup Heatmap ---
        tab3 = ttk.Frame(self.nb)
        self.nb.add(tab3, text="Speedup Heatmap")
        self._build_tab_speedup_heatmap(tab3)

        # --- Tab 4: COBA Absolute ---
        tab4 = ttk.Frame(self.nb)
        self.nb.add(tab4, text="COBA Absolute")
        self._build_tab_coba_absolute(tab4)

        # --- Tab 5: COBA Speedup ---
        tab5 = ttk.Frame(self.nb)
        self.nb.add(tab5, text="COBA Speedup")
        self._build_tab_coba_speedup(tab5)

        # --- Tab 6: COBA Heatmap ---
        tab6 = ttk.Frame(self.nb)
        self.nb.add(tab6, text="COBA Heatmap")
        self._build_tab_coba_heatmap(tab6)

    # ---------- Tab 1 ----------
    def _build_tab_latency(self, parent):
        top = ttk.Frame(parent)
        top.pack(side=tk.TOP, fill='x', padx=4, pady=2)

        # Axis & grouping
        af = ttk.LabelFrame(top, text="Axis & Grouping")
        af.pack(side=tk.LEFT, fill='y', padx=2, pady=2)
        r = 0
        self._add_combo(af, "Operator", self.operator_var, r, 0,
                        callback=self._on_operator_change)
        r += 1
        self._add_combo(af, "X-Axis", self.t1_x_var, r, 0)
        self._add_combo(af, "Y-Axis", self.t1_y_var, r, 2)
        r += 1
        self._add_combo(af, "Color Group", self.t1_color_var, r, 0)
        self._add_combo(af, "Hatch Group", self.t1_hatch_var, r, 2)
        r += 1
        self._add_combo(af, "Y Scale", self.t1_scale_var, r, 0,
                        values=["linear", "log2", "log4", "log10"])
        ttk.Checkbutton(af, text="Annotate Values", variable=self.t1_annot_var
                        ).grid(row=r, column=2, padx=6, pady=4, sticky='w')
        r += 1
        ttk.Button(af, text="▶ Plot", command=self._draw_latency
                   ).grid(row=r, column=0, columnspan=2, padx=6, pady=6, sticky='ew')
        ttk.Button(af, text="Export PNG", command=lambda: self._export_png(self.fig1)
                   ).grid(row=r, column=2, columnspan=2, padx=6, pady=6, sticky='ew')

        # Filter panel
        self.filter_outer_1 = ttk.LabelFrame(top, text="Attribute Filter ('(All)' = no filter)")
        self.filter_outer_1.pack(side=tk.LEFT, fill='both', expand=True, padx=2, pady=2)
        self.filter_inner = ttk.Frame(self.filter_outer_1)
        self.filter_inner.pack(fill='both', expand=True)

        # Plot area
        pf = ttk.Frame(parent)
        pf.pack(side=tk.TOP, fill='both', expand=True, padx=4, pady=2)
        self.fig1 = Figure(figsize=(14, 8), dpi=100)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=pf)
        NavigationToolbar2Tk(self.canvas1, pf).update()
        self.canvas1.get_tk_widget().pack(fill='both', expand=True)

    # ---------- Tab 2 ----------
    def _build_tab_speedup(self, parent):
        top = ttk.Frame(parent)
        top.pack(side=tk.TOP, fill='x', padx=4, pady=2)

        af = ttk.LabelFrame(top, text="Axis & Grouping")
        af.pack(side=tk.LEFT, fill='y', padx=2, pady=2)
        r = 0
        # Re-use operator_var (shared)
        self._add_combo(af, "Operator", self.operator_var, r, 0,
                        callback=self._on_operator_change)
        r += 1
        self._add_combo(af, "X-Axis", self.t2_x_var, r, 0)
        self._add_combo(af, "Y-Axis", self.t2_y_var, r, 2)
        r += 1
        self._add_combo(af, "Baseline", self.t2_baseline_var, r, 0)
        self._add_combo(af, "Color Group", self.t2_color_var, r, 2)
        r += 1
        self._add_combo(af, "Hatch Group", self.t2_hatch_var, r, 0)
        self._add_combo(af, "Y Scale", self.t2_scale_var, r, 2,
                        values=["linear", "log2", "log4", "log10"])
        r += 1
        ttk.Checkbutton(af, text="Annotate Values", variable=self.t2_annot_var
                        ).grid(row=r, column=2, padx=6, pady=4, sticky='w')
        r += 1
        ttk.Button(af, text="▶ Plot", command=self._draw_speedup
                   ).grid(row=r, column=0, columnspan=2, padx=6, pady=6, sticky='ew')
        ttk.Button(af, text="Export PNG", command=lambda: self._export_png(self.fig2)
                   ).grid(row=r, column=2, columnspan=2, padx=6, pady=6, sticky='ew')

        # Filter panel (mirrors tab1 – we share filter_vars)
        self.filter_outer_2 = ttk.LabelFrame(top, text="Attribute Filter ('(All)' = no filter)")
        self.filter_outer_2.pack(side=tk.LEFT, fill='both', expand=True, padx=2, pady=2)
        self.filter_inner_2 = ttk.Frame(self.filter_outer_2)
        self.filter_inner_2.pack(fill='both', expand=True)

        pf = ttk.Frame(parent)
        pf.pack(side=tk.TOP, fill='both', expand=True, padx=4, pady=2)
        self.fig2 = Figure(figsize=(14, 8), dpi=100)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=pf)
        NavigationToolbar2Tk(self.canvas2, pf).update()
        self.canvas2.get_tk_widget().pack(fill='both', expand=True)

    # ---------- Tab 3 ----------
    def _build_tab_speedup_heatmap(self, parent):
        top = ttk.Frame(parent)
        top.pack(side=tk.TOP, fill='x', padx=4, pady=2)

        af = ttk.LabelFrame(top, text="Heatmap Controls")
        af.pack(side=tk.LEFT, fill='y', padx=2, pady=2)
        r = 0
        self._add_combo(
            af,
            "Operator",
            self.operator_var,
            r,
            0,
            callback=self._on_operator_change,
        )
        r += 1
        self._add_combo(af, "Target Backend", self.t3_target_var, r, 0)
        self._add_combo(af, "Baseline", self.t3_baseline_var, r, 2)
        r += 1
        ttk.Label(af, text="Yellow Ratio").grid(row=r, column=0, padx=4, pady=3, sticky='w')
        ttk.Entry(af, textvariable=self.t3_yellow_ratio_var, width=26
                  ).grid(row=r, column=1, padx=4, pady=3, sticky='w')
        ttk.Label(af, text="Blue Ratio").grid(row=r, column=2, padx=4, pady=3, sticky='w')
        ttk.Entry(af, textvariable=self.t3_blue_ratio_var, width=26
                  ).grid(row=r, column=3, padx=4, pady=3, sticky='w')
        r += 1
        ttk.Checkbutton(af, text="Annotate Values", variable=self.t3_annot_var
                        ).grid(row=r, column=0, padx=6, pady=4, sticky='w')
        ttk.Button(af, text="▶ Plot", command=self._draw_speedup_heatmap
                   ).grid(row=r, column=1, padx=6, pady=6, sticky='ew')
        ttk.Button(af, text="Export PNG", command=lambda: self._export_png(self.fig3)
                   ).grid(row=r, column=2, padx=6, pady=6, sticky='ew')

        self.filter_outer_3 = ttk.LabelFrame(top, text="Attribute Filter ('(All)' = no filter)")
        self.filter_outer_3.pack(side=tk.LEFT, fill='both', expand=True, padx=2, pady=2)
        self.filter_inner_3 = ttk.Frame(self.filter_outer_3)
        self.filter_inner_3.pack(fill='both', expand=True)

        pf = ttk.Frame(parent)
        pf.pack(side=tk.TOP, fill='both', expand=True, padx=4, pady=2)
        self.fig3 = Figure(figsize=(14, 8), dpi=100)
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=pf)
        NavigationToolbar2Tk(self.canvas3, pf).update()
        self.canvas3.get_tk_widget().pack(fill='both', expand=True)

    # ---------- Tab 4 ----------
    def _build_tab_coba_absolute(self, parent):
        top = ttk.Frame(parent)
        top.pack(side=tk.TOP, fill='x', padx=4, pady=2)

        af = ttk.LabelFrame(top, text="COBA Controls")
        af.pack(side=tk.LEFT, fill='y', padx=2, pady=2)
        r = 0
        self._add_combo(af, "Operator", self.operator_var, r, 0,
                        callback=self._on_operator_change)
        r += 1
        self._add_combo(af, "X-Axis", self.c1_x_var, r, 0)
        self._add_combo(af, "Y-Axis", self.c1_y_var, r, 2)
        r += 1
        self._add_combo(af, "Color Group", self.c1_color_var, r, 0)
        self._add_combo(af, "Hatch Group", self.c1_hatch_var, r, 2)
        r += 1
        self._add_combo(af, "Y Scale", self.c1_scale_var, r, 0,
                        values=["linear", "log2", "log4", "log10"])
        ttk.Checkbutton(af, text="Annotate Values", variable=self.c1_annot_var
                        ).grid(row=r, column=2, padx=6, pady=4, sticky='w')
        r += 1
        ttk.Button(af, text="▶ Plot", command=self._draw_coba_latency
                   ).grid(row=r, column=0, columnspan=2, padx=6, pady=6, sticky='ew')
        ttk.Button(af, text="Export PNG", command=lambda: self._export_png(self.fig4)
                   ).grid(row=r, column=2, columnspan=2, padx=6, pady=6, sticky='ew')

        self.filter_outer_4 = ttk.LabelFrame(top, text="Attribute Filter ('(All)' = no filter)")
        self.filter_outer_4.pack(side=tk.LEFT, fill='both', expand=True, padx=2, pady=2)
        self.filter_inner_4 = ttk.Frame(self.filter_outer_4)
        self.filter_inner_4.pack(fill='both', expand=True)

        pf = ttk.Frame(parent)
        pf.pack(side=tk.TOP, fill='both', expand=True, padx=4, pady=2)
        self.fig4 = Figure(figsize=(14, 8), dpi=100)
        self.canvas4 = FigureCanvasTkAgg(self.fig4, master=pf)
        NavigationToolbar2Tk(self.canvas4, pf).update()
        self.canvas4.get_tk_widget().pack(fill='both', expand=True)

    # ---------- Tab 5 ----------
    def _build_tab_coba_speedup(self, parent):
        top = ttk.Frame(parent)
        top.pack(side=tk.TOP, fill='x', padx=4, pady=2)

        af = ttk.LabelFrame(top, text="COBA Controls")
        af.pack(side=tk.LEFT, fill='y', padx=2, pady=2)
        r = 0
        self._add_combo(af, "Operator", self.operator_var, r, 0,
                        callback=self._on_operator_change)
        r += 1
        self._add_combo(af, "X-Axis", self.c2_x_var, r, 0)
        self._add_combo(af, "Y-Axis", self.c2_y_var, r, 2)
        r += 1
        self._add_combo(af, "Baseline", self.c2_baseline_var, r, 0)
        self._add_combo(af, "Color Group", self.c2_color_var, r, 2)
        r += 1
        self._add_combo(af, "Hatch Group", self.c2_hatch_var, r, 0)
        self._add_combo(af, "Y Scale", self.c2_scale_var, r, 2,
                        values=["linear", "log2", "log4", "log10"])
        r += 1
        ttk.Checkbutton(af, text="Annotate Values", variable=self.c2_annot_var
                        ).grid(row=r, column=0, padx=6, pady=4, sticky='w')
        ttk.Button(af, text="▶ Plot", command=self._draw_coba_speedup
                   ).grid(row=r, column=1, padx=6, pady=6, sticky='ew')
        ttk.Button(af, text="Export PNG", command=lambda: self._export_png(self.fig5)
                   ).grid(row=r, column=2, padx=6, pady=6, sticky='ew')

        self.filter_outer_5 = ttk.LabelFrame(top, text="Attribute Filter ('(All)' = no filter)")
        self.filter_outer_5.pack(side=tk.LEFT, fill='both', expand=True, padx=2, pady=2)
        self.filter_inner_5 = ttk.Frame(self.filter_outer_5)
        self.filter_inner_5.pack(fill='both', expand=True)

        pf = ttk.Frame(parent)
        pf.pack(side=tk.TOP, fill='both', expand=True, padx=4, pady=2)
        self.fig5 = Figure(figsize=(14, 8), dpi=100)
        self.canvas5 = FigureCanvasTkAgg(self.fig5, master=pf)
        NavigationToolbar2Tk(self.canvas5, pf).update()
        self.canvas5.get_tk_widget().pack(fill='both', expand=True)

    # ---------- Tab 6 ----------
    def _build_tab_coba_heatmap(self, parent):
        top = ttk.Frame(parent)
        top.pack(side=tk.TOP, fill='x', padx=4, pady=2)

        af = ttk.LabelFrame(top, text="COBA Heatmap Controls")
        af.pack(side=tk.LEFT, fill='y', padx=2, pady=2)
        r = 0
        self._add_combo(af, "Operator", self.operator_var, r, 0,
                        callback=self._on_operator_change)
        r += 1
        self._add_combo(af, "Target Backend", self.c3_target_var, r, 0)
        self._add_combo(af, "Baseline", self.c3_baseline_var, r, 2)
        r += 1
        ttk.Label(af, text="Yellow Ratio").grid(row=r, column=0, padx=4, pady=3, sticky='w')
        ttk.Entry(af, textvariable=self.c3_yellow_ratio_var, width=26
                  ).grid(row=r, column=1, padx=4, pady=3, sticky='w')
        ttk.Label(af, text="Blue Ratio").grid(row=r, column=2, padx=4, pady=3, sticky='w')
        ttk.Entry(af, textvariable=self.c3_blue_ratio_var, width=26
                  ).grid(row=r, column=3, padx=4, pady=3, sticky='w')
        r += 1
        ttk.Checkbutton(af, text="Annotate Values", variable=self.c3_annot_var
                        ).grid(row=r, column=0, padx=6, pady=4, sticky='w')
        ttk.Button(af, text="▶ Plot", command=self._draw_coba_heatmap
                   ).grid(row=r, column=1, padx=6, pady=6, sticky='ew')
        ttk.Button(af, text="Export PNG", command=lambda: self._export_png(self.fig6)
                   ).grid(row=r, column=2, padx=6, pady=6, sticky='ew')

        self.filter_outer_6 = ttk.LabelFrame(top, text="Attribute Filter ('(All)' = no filter)")
        self.filter_outer_6.pack(side=tk.LEFT, fill='both', expand=True, padx=2, pady=2)
        self.filter_inner_6 = ttk.Frame(self.filter_outer_6)
        self.filter_inner_6.pack(fill='both', expand=True)

        pf = ttk.Frame(parent)
        pf.pack(side=tk.TOP, fill='both', expand=True, padx=4, pady=2)
        self.fig6 = Figure(figsize=(14, 8), dpi=100)
        self.canvas6 = FigureCanvasTkAgg(self.fig6, master=pf)
        NavigationToolbar2Tk(self.canvas6, pf).update()
        self.canvas6.get_tk_widget().pack(fill='both', expand=True)

    def _preferred_numeric_defaults(self, cols: List[str]) -> Tuple[str, str]:
        y_abs_candidates = [
            'elapsed_s', 'elapsed_mean_s', 'mean_ms', 'firing_rate_mean_hz',
            'speedup_vs_baseline', 'elapsed_min_s', 'elapsed_max_s', 'elapsed_std_s',
        ]
        y_spd_candidates = ['elapsed_s', 'elapsed_mean_s', 'mean_ms']
        y_abs = next((c for c in y_abs_candidates if c in cols), cols[0] if cols else '')
        y_spd = next((c for c in y_spd_candidates if c in cols), y_abs)
        return y_abs, y_spd

    def _preferred_x_default(self, cat_cols: List[str]) -> str:
        for c in ['scale', 'size', 'conn']:
            if c in cat_cols:
                return c
        return cat_cols[0] if cat_cols else ''

    def _base_subset(self) -> pd.DataFrame:
        if self.df.empty:
            return self.df.copy()
        if 'operator' in self.df.columns:
            op = self.operator_var.get()
            if op:
                return self.df[self.df['operator'].astype(str) == op].copy()
            return self.df.iloc[0:0].copy()
        if 'primitive_name' in self.df.columns:
            prim = self.operator_var.get()
            if prim:
                return self.df[self.df['primitive_name'].astype(str) == prim].copy()
            return self.df.iloc[0:0].copy()
        return self.df.copy()

    def _title_name(self) -> str:
        op = self.operator_var.get()
        if op:
            return _cap(op)
        return 'COBA'

    # ═══════════════ Combo helpers ═══════════════
    def _add_combo(self, parent, label, var, row, col, values=None, callback=None):
        ttk.Label(parent, text=label).grid(row=row, column=col, padx=4, pady=3, sticky='w')
        cb = ttk.Combobox(parent, textvariable=var, state='readonly',
                          width=24, values=values or [])
        cb.grid(row=row, column=col + 1, padx=4, pady=3, sticky='w')
        if callback:
            cb.bind("<<ComboboxSelected>>", lambda _e: callback())
        # store by id – note: operator_var may be added multiple times; combos share same var
        self._combo_refs.setdefault(id(var), [])
        if not isinstance(self._combo_refs.get(id(var)), list):
            self._combo_refs[id(var)] = [self._combo_refs[id(var)]]
        self._combo_refs[id(var)].append(cb)

    def _update_combo(self, var: tk.StringVar, values: list, default: Optional[str] = None):
        cbs = self._combo_refs.get(id(var))
        if cbs is None:
            return
        vals = list(values)
        if not isinstance(cbs, list):
            cbs = [cbs]
        for cb in cbs:
            cb['values'] = vals
        if var.get() not in vals:
            if default and default in vals:
                var.set(default)
            elif vals:
                var.set(vals[0])
            else:
                var.set('')

    # ═══════════════ Data loading ═══════════════
    def _open_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            self.raw_df = load_csv(path)
            self.df, self.config_keys = parse_config_columns(self.raw_df)
        except Exception as e:
            messagebox.showerror("Load Failed", str(e))
            return
        messagebox.showinfo("Success",
                            f"Loaded {len(self.df)} rows\nFrom: {os.path.basename(path)}")
        self._refresh_operators()

    def _refresh_operators(self):
        if 'operator' in self.df.columns:
            ops = sorted(self.df['operator'].dropna().unique().astype(str))
            self._update_combo(self.operator_var, ops)
            if ops:
                self.operator_var.set(ops[0])
        elif 'primitive_name' in self.df.columns:
            ops = sorted(self.df['primitive_name'].dropna().unique().astype(str))
            self._update_combo(self.operator_var, ops)
            if ops:
                self.operator_var.set(ops[0])
        else:
            self._update_combo(self.operator_var, [])
            self.operator_var.set('')
        self._on_operator_change()

    def _on_operator_change(self):
        if self.df.empty:
            return
        sub = self._base_subset()
        if sub.empty and ('operator' in self.df.columns or 'primitive_name' in self.df.columns):
            return

        cat_cols: List[str] = []
        for k in self.config_keys:
            if k in sub.columns and k not in cat_cols:
                cat_cols.append(k)

        # Include low-cardinality columns as grouping dimensions even if numeric
        # (e.g. COBA scale/size/conn), but exclude pure measurement columns.
        metric_like = {
            'elapsed_min_s', 'elapsed_max_s', 'elapsed_mean_s', 'elapsed_std_s',
            'elapsed_s', 'firing_rate', 'firing_rate_mean_hz', 'speedup_vs_baseline',
            'mean_ms', 'std_ms', 'min_ms', 'max_ms',
        }
        for col in sub.columns:
            if col in ('operator', 'primitive_name', 'config_name', 'testing_type'):
                continue
            if col in metric_like:
                continue
            nunique = int(sub[col].nunique(dropna=True))
            if 1 <= nunique <= 40 and col not in cat_cols:
                cat_cols.append(col)

        preferred_order = ['scale', 'size', 'conn', 'conn_num', 'data_type',
                           'synaptic_type', 'homo', 'backend', 'platform']
        cat_cols = [c for c in preferred_order if c in cat_cols] + [
            c for c in cat_cols if c not in preferred_order
        ]
        num_cols = [c for c in sub.columns if pd.api.types.is_numeric_dtype(sub[c])]

        # Use backend column for baseline/speedup calculations
        backends = (
            sorted(sub['backend'].dropna().unique().astype(str))
            if 'backend' in sub.columns else []
        )
        x_default = self._preferred_x_default(cat_cols)
        y_abs_default, y_speed_default = self._preferred_numeric_defaults(num_cols)
        color_default = 'backend' if 'backend' in cat_cols else (cat_cols[0] if cat_cols else '')

        # Tab 1
        self._update_combo(self.t1_x_var, cat_cols, default=x_default)
        self._update_combo(self.t1_y_var, num_cols, default=y_abs_default)
        self._update_combo(self.t1_color_var, cat_cols, default=color_default)
        self._update_combo(self.t1_hatch_var, [NONE_LABEL] + cat_cols, default=NONE_LABEL)

        # Tab 2
        self._update_combo(self.t2_x_var, cat_cols, default=x_default)
        self._update_combo(self.t2_y_var, num_cols, default=y_abs_default)
        self._update_combo(self.t2_baseline_var, backends)
        self._update_combo(self.t2_color_var, cat_cols, default=color_default)
        self._update_combo(self.t2_hatch_var, [NONE_LABEL] + cat_cols, default=NONE_LABEL)

        # Tab 3
        self._update_combo(self.t3_target_var, backends)
        self._update_combo(self.t3_baseline_var, backends)
        if self.t3_target_var.get() == self.t3_baseline_var.get() and len(backends) >= 2:
            self.t3_target_var.set(backends[0])
            self.t3_baseline_var.set(backends[1])

        # Tab 4
        self._update_combo(self.c1_x_var, cat_cols, default=x_default)
        self._update_combo(self.c1_y_var, num_cols, default=y_abs_default)
        self._update_combo(self.c1_color_var, cat_cols, default=color_default)
        self._update_combo(self.c1_hatch_var, [NONE_LABEL] + cat_cols, default=NONE_LABEL)

        # Tab 5
        self._update_combo(self.c2_x_var, cat_cols, default=x_default)
        self._update_combo(self.c2_y_var, num_cols, default=y_speed_default)
        self._update_combo(self.c2_baseline_var, backends)
        self._update_combo(self.c2_color_var, cat_cols, default=color_default)
        self._update_combo(self.c2_hatch_var, [NONE_LABEL] + cat_cols, default=NONE_LABEL)

        # Tab 6
        self._update_combo(self.c3_target_var, backends)
        self._update_combo(self.c3_baseline_var, backends)
        if self.c3_target_var.get() == self.c3_baseline_var.get() and len(backends) >= 2:
            self.c3_target_var.set(backends[0])
            self.c3_baseline_var.set(backends[1])

        self._build_filters(sub, cat_cols)

    # ═══════════════ Filters ═══════════════
    def _build_filters(self, sub: pd.DataFrame, cat_cols: List[str]):
        # Build in both tabs
        for container in (
            self.filter_inner,
            self.filter_inner_2,
            self.filter_inner_3,
            self.filter_inner_4,
            self.filter_inner_5,
            self.filter_inner_6,
        ):
            for w in container.winfo_children():
                w.destroy()
        self.filter_vars.clear()

        cols_per_row = 4
        for idx, key in enumerate(cat_cols):
            vals = sorted(sub[key].dropna().unique().astype(str), key=sort_key)
            var = tk.StringVar(value=ALL_LABEL)
            self.filter_vars[key] = var
            r = (idx // cols_per_row) * 2
            c = (idx % cols_per_row) * 2
            for container in (
                self.filter_inner,
                self.filter_inner_2,
                self.filter_inner_3,
                self.filter_inner_4,
                self.filter_inner_5,
                self.filter_inner_6,
            ):
                ttk.Label(container, text=key, font=('', 8)
                          ).grid(row=r, column=c, padx=3, pady=1, sticky='w')
                ttk.Combobox(container, textvariable=var, state='readonly',
                             width=16, values=[ALL_LABEL] + vals
                             ).grid(row=r + 1, column=c, padx=3, pady=1, sticky='w')

    def _get_heatmap_data(self) -> pd.DataFrame:
        sub = self._base_subset()
        # Keep axis columns free to form a complete matrix.
        skip_filter_cols = {'backend', 'scale', 'spike_rate', 'conn_num', 'conn_numbers'}
        for key, var in self.filter_vars.items():
            val = var.get()
            if val == ALL_LABEL or key in skip_filter_cols:
                continue
            if key in sub.columns:
                sub = sub[sub[key].astype(str) == val]
        return sub

    def _get_filtered_data(self, x_var, color_var, hatch_var
                           ) -> Tuple[pd.DataFrame, Optional[str]]:
        sub = self._base_subset()
        x_col = x_var.get()
        color_col = color_var.get()
        hatch_col = hatch_var.get()
        if hatch_col == NONE_LABEL:
            hatch_col = None
        for key, var in self.filter_vars.items():
            val = var.get()
            if val == ALL_LABEL:
                continue
            if key in sub.columns and key not in (x_col, color_col, hatch_col):
                sub = sub[sub[key].astype(str) == val]
        return sub, hatch_col

    def _filter_subtitle(self, x_col, color_col, hatch_col) -> str:
        parts = []
        for key, var in self.filter_vars.items():
            val = var.get()
            if val != ALL_LABEL and key not in (x_col, color_col, hatch_col or ''):
                parts.append(f"{_cap(key)}={val}")
        return ", ".join(parts) if parts else "No Extra Filter"

    def _show_rows_detail(self, title: str, summary: str, rows: pd.DataFrame):
        messagebox.showerror("Error", summary)
        win = tk.Toplevel(self)
        win.title(title)
        win.geometry("1300x700")

        ttk.Label(win, text=summary, justify='left').pack(anchor='w', padx=8, pady=6)

        text_frame = ttk.Frame(win)
        text_frame.pack(fill='both', expand=True, padx=8, pady=6)

        y_scroll = ttk.Scrollbar(text_frame, orient='vertical')
        y_scroll.pack(side='right', fill='y')
        x_scroll = ttk.Scrollbar(text_frame, orient='horizontal')
        x_scroll.pack(side='bottom', fill='x')

        txt = tk.Text(
            text_frame,
            wrap='none',
            xscrollcommand=x_scroll.set,
            yscrollcommand=y_scroll.set,
            font=('Consolas', 9),
        )
        txt.pack(side='left', fill='both', expand=True)
        y_scroll.config(command=txt.yview)
        x_scroll.config(command=txt.xview)

        display = rows.copy()
        display.insert(0, '_row_index', display.index)
        with pd.option_context(
            'display.max_rows', None,
            'display.max_columns', None,
            'display.width', 2000,
            'display.max_colwidth', 200,
        ):
            txt.insert('1.0', display.to_string(index=False))
        txt.config(state='disabled')

    def _ensure_unique_match(self, df: pd.DataFrame, key_cols: List[str], context: str) -> bool:
        cols = [c for c in key_cols if c and c in df.columns]
        if not cols:
            return True
        dup_mask = df.duplicated(subset=cols, keep=False)
        if not dup_mask.any():
            return True

        dup_rows = df.loc[dup_mask].copy()
        dup_rows = dup_rows.sort_values(by=cols, key=lambda s: s.astype(str))
        summary = (
            f"{context}: Multiple rows matched the same constraints.\n"
            f"Duplicate keys: {', '.join(cols)}\n"
            "Please refine filters so each plotted item maps to exactly one row."
        )
        self._show_rows_detail(f"{context} - Duplicate Matches", summary, dup_rows)
        return False

    def _ensure_baseline_coverage(
        self,
        merged: pd.DataFrame,
        color_col: str,
        baseline: str,
        context: str,
    ) -> bool:
        missing = merged[(merged[color_col].astype(str) != baseline) & merged['_base_val'].isna()]
        if missing.empty:
            return True
        summary = (
            f"{context}: Missing baseline rows for part of the selected data.\n"
            f"Baseline={baseline}. The following rows cannot compute speedup."
        )
        self._show_rows_detail(f"{context} - Missing Baseline", summary, missing)
        return False

    # ═══════════════ Drawing: Tab 1 (Latency) ═══════════════
    def _draw_latency(self):
        if self.df.empty:
            messagebox.showerror("Error", "Please open a CSV file first.")
            return
        x_col = self.t1_x_var.get()
        y_col = self.t1_y_var.get()
        color_col = self.t1_color_var.get()
        if not x_col or not y_col or not color_col:
            messagebox.showerror("Error", "Please select X-Axis, Y-Axis, and Color Group.")
            return

        sub, hatch_col = self._get_filtered_data(self.t1_x_var, self.t1_color_var, self.t1_hatch_var)
        if sub.empty:
            messagebox.showerror("Error", "No data after filtering.")
            return
        sub[y_col] = pd.to_numeric(sub[y_col], errors='coerce')
        sub = sub.dropna(subset=[y_col])
        if sub.empty:
            messagebox.showerror("Error", "No valid numeric data for selected Y-Axis.")
            return
        if not self._ensure_unique_match(sub, [x_col, color_col, hatch_col], "Absolute Latency"):
            return

        self.fig1.clf()
        ax = self.fig1.add_subplot(111)

        groups, color_map, hatch_map = self._plot_grouped_bar(
            ax, sub, x_col, y_col, color_col, hatch_col,
            annotate=self.t1_annot_var.get())

        # Title / labels
        prim = self._title_name()
        y_label = Y_COL_LABELS.get(y_col, _cap(y_col))
        subtitle = self._filter_subtitle(x_col, color_col, hatch_col)
        hatch_info = f", Hatch={_cap(hatch_col)}" if hatch_col else ""
        ax.set_title(
            f"Absolute Latency — {prim}\n"
            f"Color={_cap(color_col)}{hatch_info}  |  {subtitle}",
            fontsize=13, fontweight='bold')
        ax.set_xlabel(_cap(x_col), fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)

        self._apply_scale(ax, self.t1_scale_var.get())
        self._add_legend(ax, groups, color_map, hatch_map, hatch_col)
        ax.grid(axis='y', alpha=0.3)

        self.fig1.tight_layout()
        self.canvas1.draw()

    # ═══════════════ Drawing: Tab 2 (Speedup) ═══════════════
    def _draw_speedup(self):
        if self.df.empty:
            messagebox.showerror("Error", "Please open a CSV file first.")
            return
        x_col = self.t2_x_var.get()
        baseline = self.t2_baseline_var.get()
        color_col = self.t2_color_var.get()
        if not x_col or not baseline or not color_col:
            messagebox.showerror("Error", "Please select X-Axis, Baseline, and Color Group.")
            return
        if x_col == color_col:
            messagebox.showerror(
                "Error",
                "For Speedup, X-Axis and Color Group cannot be the same column.",
            )
            return

        sub, hatch_col = self._get_filtered_data(self.t2_x_var, self.t2_color_var, self.t2_hatch_var)
        if sub.empty:
            messagebox.showerror("Error", "No data after filtering.")
            return
        y_col = self.t2_y_var.get()
        if not y_col:
            messagebox.showerror("Error", "Please select a Y-Axis column.")
            return
        sub[y_col] = pd.to_numeric(sub[y_col], errors='coerce')
        sub = sub.dropna(subset=[y_col])
        if sub.empty:
            messagebox.showerror("Error", f"No valid numeric data for {y_col}.")
            return
        if not self._ensure_unique_match(sub, [x_col, color_col, hatch_col], "Relative Speedup"):
            return

        # Compute speedup: baseline / this, per (x, hatch) group.
        group_keys = [x_col]
        if hatch_col:
            group_keys.append(hatch_col)

        base_df = sub[sub[color_col].astype(str) == baseline]
        if base_df.empty:
            messagebox.showerror("Error", f"No baseline rows found for {baseline}.")
            return
        base_agg = base_df[group_keys + [y_col]].rename(columns={y_col: '_base_val'})

        merged = sub.merge(base_agg, on=group_keys, how='left', validate='many_to_one')
        if not self._ensure_baseline_coverage(
            merged=merged,
            color_col=color_col,
            baseline=baseline,
            context="Relative Speedup",
        ):
            return
        merged['speedup'] = np.where(merged[y_col] > 0,
                                     merged['_base_val'] / merged[y_col], np.nan)

        # Remove baseline from bars
        plot_df = merged[merged[color_col].astype(str) != baseline].copy()
        if plot_df.empty:
            messagebox.showerror("Error", "No non-baseline data to plot.")
            return
        plot_df = plot_df[np.isfinite(plot_df['speedup'])]
        if plot_df.empty:
            messagebox.showerror(
                "Error",
                "Speedup values are all NaN/Inf. Choose an X-Axis different from Color Group and check baseline coverage.",
            )
            return

        self.fig2.clf()
        ax = self.fig2.add_subplot(111)

        groups, color_map, hatch_map = self._plot_grouped_bar(
            ax, plot_df, x_col, 'speedup', color_col, hatch_col,
            annotate=self.t2_annot_var.get())

        # Baseline reference line at y = 1
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, zorder=1,
                    label=f"Baseline ({baseline})")

        # Title / labels
        prim = self._title_name()
        subtitle = self._filter_subtitle(x_col, color_col, hatch_col)
        hatch_info = f", Hatch={_cap(hatch_col)}" if hatch_col else ""
        ax.set_title(
            f"Relative Speedup — {prim}  (Baseline: {baseline})\n"
            f"Color={_cap(color_col)}{hatch_info}  |  {subtitle}",
            fontsize=13, fontweight='bold')
        ax.set_xlabel(_cap(x_col), fontsize=12)
        ax.set_ylabel("Speedup (×)", fontsize=12)

        self._apply_scale(ax, self.t2_scale_var.get())
        self._add_legend(ax, groups, color_map, hatch_map, hatch_col,
                         extra_handles=[
                             mpatches.Patch(facecolor='none', edgecolor='gray',
                                            linestyle='--', linewidth=2,
                                            label=f"Baseline ({baseline})")
                         ])
        ax.grid(axis='y', alpha=0.3)

        self.fig2.tight_layout()
        self.canvas2.draw()

    # ═══════════════ Drawing: Tab 3 (Speedup Heatmap) ═══════════════
    def _draw_speedup_heatmap(self):
        if self.df.empty:
            messagebox.showerror("Error", "Please open a CSV file first.")
            return

        target_backend = self.t3_target_var.get()
        baseline_backend = self.t3_baseline_var.get()
        if not target_backend or not baseline_backend:
            messagebox.showerror("Error", "Please select Target Backend and Baseline.")
            return
        if target_backend == baseline_backend:
            messagebox.showerror("Error", "Target Backend and Baseline must be different.")
            return

        try:
            yellow_ratio = float(self.t3_yellow_ratio_var.get())
            blue_ratio = float(self.t3_blue_ratio_var.get())
        except ValueError:
            messagebox.showerror("Error", "Yellow Ratio and Blue Ratio must be numeric.")
            return
        if not (yellow_ratio < 1.0 < blue_ratio):
            messagebox.showerror(
                "Error",
                "Require: Yellow Ratio < 1.0 < Blue Ratio (so speedup=1 stays pure white).",
            )
            return

        sub = self._get_heatmap_data()
        dt_col = 'backend'
        val_col_hm = 'mean_ms' if 'mean_ms' in sub.columns else 'elapsed_s'
        required = [dt_col, val_col_hm, 'scale', 'spike_rate']
        missing = [c for c in required if c not in sub.columns]
        if missing:
            messagebox.showerror(
                "Error",
                f"Missing columns for heatmap: {missing}. Need backend, {val_col_hm}, scale, spike_rate.",
            )
            return

        sub = sub[sub[dt_col].astype(str).isin([target_backend, baseline_backend])].copy()
        if sub.empty:
            messagebox.showerror("Error", "No data after filtering and backend selection.")
            return
        if not self._ensure_unique_match(
            sub,
            [dt_col, 'scale', 'spike_rate'],
            "Speedup Heatmap",
        ):
            return

        try:
            speedup_matrix = build_speedup_matrix(
                sub,
                target_backend=target_backend,
                baseline_backend=baseline_backend,
                scale_col='scale',
                spike_col='spike_rate',
                val_col=val_col_hm,
                key_col=dt_col,
            )
            cmap, norm = build_white_center_colormap(
                yellow_ratio=yellow_ratio,
                blue_ratio=blue_ratio,
                lemon_yellow='#FFF44F',
                blue='#4F81BD',
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.fig3.clf()
        ax = self.fig3.add_subplot(111)
        prim = self._title_name()
        subtitle = self._filter_subtitle('scale', dt_col, None)
        sns.heatmap(
            speedup_matrix,
            annot=self.t3_annot_var.get(),
            fmt='.2f',
            cmap=cmap,
            norm=norm,
            linewidths=0.5,
            cbar_kws={
                'label': (
                    f'Speedup (baseline/target), 1.0=white, '
                    f'{yellow_ratio:.2f}=yellow, {blue_ratio:.2f}=blue'
                )
            },
            ax=ax,
        )
        ax.set_title(
            f"Speedup Heatmap - {prim}\n"
            f"Baseline={baseline_backend}, Target={target_backend}  |  {subtitle}",
            fontsize=13,
            fontweight='bold',
        )
        ax.set_xlabel('Scale', fontsize=12)
        ax.set_ylabel('Spike Rate', fontsize=12)
        ax.tick_params(axis='x', rotation=0)
        for label in ax.get_xticklabels():
            label.set_ha('center')

        self.fig3.tight_layout()
        self.canvas3.draw()

    # ═══════════════ Drawing: Tab 4 (COBA Absolute) ═══════════════
    def _draw_coba_latency(self):
        if self.df.empty:
            messagebox.showerror("Error", "Please open a CSV file first.")
            return

        x_col = self.c1_x_var.get()
        y_col = self.c1_y_var.get()
        color_col = self.c1_color_var.get()
        if not x_col or not y_col or not color_col:
            messagebox.showerror("Error", "Please select X-Axis, Y-Axis, and Color Group.")
            return

        sub, hatch_col = self._get_filtered_data(self.c1_x_var, self.c1_color_var, self.c1_hatch_var)
        if sub.empty:
            messagebox.showerror("Error", "No data after filtering.")
            return
        sub[y_col] = pd.to_numeric(sub[y_col], errors='coerce')
        sub = sub.dropna(subset=[y_col])
        if sub.empty:
            messagebox.showerror("Error", "No valid numeric data for selected Y-Axis.")
            return
        if not self._ensure_unique_match(sub, [x_col, color_col, hatch_col], "COBA Absolute Latency"):
            return

        self.fig4.clf()
        ax = self.fig4.add_subplot(111)
        groups, color_map, hatch_map = self._plot_grouped_bar(
            ax, sub, x_col, y_col, color_col, hatch_col,
            annotate=self.c1_annot_var.get())

        title_name = self._title_name()
        subtitle = self._filter_subtitle(x_col, color_col, hatch_col)
        hatch_info = f", Hatch={_cap(hatch_col)}" if hatch_col else ""
        ax.set_title(
            f"COBA Absolute Latency - {title_name}\n"
            f"Color={_cap(color_col)}{hatch_info}  |  {subtitle}",
            fontsize=13,
            fontweight='bold',
        )
        ax.set_xlabel(_cap(x_col), fontsize=12)
        ax.set_ylabel(_cap(y_col), fontsize=12)
        self._apply_scale(ax, self.c1_scale_var.get())
        self._add_legend(ax, groups, color_map, hatch_map, hatch_col)
        ax.grid(axis='y', alpha=0.3)
        self.fig4.tight_layout()
        self.canvas4.draw()

    # ═══════════════ Drawing: Tab 5 (COBA Speedup) ═══════════════
    def _draw_coba_speedup(self):
        if self.df.empty:
            messagebox.showerror("Error", "Please open a CSV file first.")
            return

        x_col = self.c2_x_var.get()
        y_col = self.c2_y_var.get()
        baseline = self.c2_baseline_var.get()
        color_col = self.c2_color_var.get()
        if not x_col or not y_col or not baseline or not color_col:
            messagebox.showerror("Error", "Please select X-Axis, Y-Axis, Baseline, and Color Group.")
            return
        if x_col == color_col:
            messagebox.showerror(
                "Error",
                "For COBA Speedup, X-Axis and Color Group cannot be the same column.",
            )
            return

        sub, hatch_col = self._get_filtered_data(self.c2_x_var, self.c2_color_var, self.c2_hatch_var)
        if sub.empty:
            messagebox.showerror("Error", "No data after filtering.")
            return
        sub[y_col] = pd.to_numeric(sub[y_col], errors='coerce')
        sub = sub.dropna(subset=[y_col])
        if sub.empty:
            messagebox.showerror("Error", "No valid numeric data for selected Y-Axis.")
            return
        if not self._ensure_unique_match(sub, [x_col, color_col, hatch_col], "COBA Relative Speedup"):
            return

        group_keys = [x_col]
        if hatch_col:
            group_keys.append(hatch_col)

        base_df = sub[sub[color_col].astype(str) == baseline]
        if base_df.empty:
            messagebox.showerror("Error", f"No baseline rows found for {baseline}.")
            return
        base_agg = base_df[group_keys + [y_col]].rename(columns={y_col: '_base_val'})

        merged = sub.merge(base_agg, on=group_keys, how='left', validate='many_to_one')
        if not self._ensure_baseline_coverage(
            merged=merged,
            color_col=color_col,
            baseline=baseline,
            context="COBA Relative Speedup",
        ):
            return
        merged['speedup'] = np.where(merged[y_col] > 0,
                                     merged['_base_val'] / merged[y_col], np.nan)
        plot_df = merged[merged[color_col].astype(str) != baseline].copy()
        if plot_df.empty:
            messagebox.showerror("Error", "No non-baseline data to plot.")
            return
        plot_df = plot_df[np.isfinite(plot_df['speedup'])]
        if plot_df.empty:
            messagebox.showerror(
                "Error",
                "Speedup values are all NaN/Inf. Choose an X-Axis different from Color Group and check baseline coverage.",
            )
            return

        self.fig5.clf()
        ax = self.fig5.add_subplot(111)
        groups, color_map, hatch_map = self._plot_grouped_bar(
            ax, plot_df, x_col, 'speedup', color_col, hatch_col,
            annotate=self.c2_annot_var.get())
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, zorder=1,
                   label=f"Baseline ({baseline})")

        title_name = self._title_name()
        subtitle = self._filter_subtitle(x_col, color_col, hatch_col)
        hatch_info = f", Hatch={_cap(hatch_col)}" if hatch_col else ""
        ax.set_title(
            f"COBA Relative Speedup - {title_name}  (Baseline: {baseline})\n"
            f"Y={_cap(y_col)}, Color={_cap(color_col)}{hatch_info}  |  {subtitle}",
            fontsize=13,
            fontweight='bold',
        )
        ax.set_xlabel(_cap(x_col), fontsize=12)
        ax.set_ylabel("Speedup (x)", fontsize=12)
        self._apply_scale(ax, self.c2_scale_var.get())
        self._add_legend(
            ax,
            groups,
            color_map,
            hatch_map,
            hatch_col,
            extra_handles=[
                mpatches.Patch(facecolor='none', edgecolor='gray',
                               linestyle='--', linewidth=2,
                               label=f"Baseline ({baseline})")
            ],
        )
        ax.grid(axis='y', alpha=0.3)
        self.fig5.tight_layout()
        self.canvas5.draw()

    # ═══════════════ Drawing: Tab 6 (COBA Heatmap) ═══════════════
    def _draw_coba_heatmap(self):
        if self.df.empty:
            messagebox.showerror("Error", "Please open a CSV file first.")
            return

        target_backend = self.c3_target_var.get()
        baseline_backend = self.c3_baseline_var.get()
        if not target_backend or not baseline_backend:
            messagebox.showerror("Error", "Please select Target Backend and Baseline.")
            return
        if target_backend == baseline_backend:
            messagebox.showerror("Error", "Target Backend and Baseline must be different.")
            return

        try:
            yellow_ratio = float(self.c3_yellow_ratio_var.get())
            blue_ratio = float(self.c3_blue_ratio_var.get())
        except ValueError:
            messagebox.showerror("Error", "Yellow Ratio and Blue Ratio must be numeric.")
            return
        if not (yellow_ratio < 1.0 < blue_ratio):
            messagebox.showerror(
                "Error",
                "Require: Yellow Ratio < 1.0 < Blue Ratio (so speedup=1 stays pure white).",
            )
            return

        dt_col_c = 'backend'
        val_col_c = (
            'elapsed_s' if 'elapsed_s' in self.df.columns else
            ('elapsed_mean_s' if 'elapsed_mean_s' in self.df.columns else 'mean_ms')
        )
        conn_col = 'conn_num' if 'conn_num' in self.df.columns else 'conn_numbers'

        sub = self._get_heatmap_data()
        required = [dt_col_c, val_col_c, 'scale', conn_col]
        missing = [c for c in required if c not in sub.columns]
        if missing:
            messagebox.showerror(
                "Error",
                f"Missing columns for COBA heatmap: {missing}. Need backend, {val_col_c}, scale, {conn_col}.",
            )
            return

        sub = sub[sub[dt_col_c].astype(str).isin([target_backend, baseline_backend])].copy()
        if sub.empty:
            messagebox.showerror("Error", "No data after filtering and backend selection.")
            return
        if not self._ensure_unique_match(
            sub,
            [dt_col_c, 'scale', conn_col],
            "COBA Heatmap",
        ):
            return

        try:
            speedup_matrix = build_speedup_matrix(
                sub,
                target_backend=target_backend,
                baseline_backend=baseline_backend,
                scale_col='scale',
                spike_col=conn_col,
                val_col=val_col_c,
                key_col=dt_col_c,
            )
            cmap, norm = build_white_center_colormap(
                yellow_ratio=yellow_ratio,
                blue_ratio=blue_ratio,
                lemon_yellow='#FFF44F',
                blue='#4F81BD',
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.fig6.clf()
        ax = self.fig6.add_subplot(111)
        prim = self._title_name()
        subtitle = self._filter_subtitle('scale', dt_col_c, None)
        sns.heatmap(
            speedup_matrix,
            annot=self.c3_annot_var.get(),
            fmt='.2f',
            cmap=cmap,
            norm=norm,
            linewidths=0.5,
            cbar_kws={
                'label': (
                    f'Speedup (baseline/target), 1.0=white, '
                    f'{yellow_ratio:.2f}=yellow, {blue_ratio:.2f}=blue'
                )
            },
            ax=ax,
        )
        ax.set_title(
            f"COBA Heatmap - {prim}\n"
            f"Baseline={baseline_backend}, Target={target_backend}  |  {subtitle}",
            fontsize=13,
            fontweight='bold',
        )
        ax.set_xlabel('Scale', fontsize=12)
        ax.set_ylabel(_cap(conn_col), fontsize=12)
        ax.tick_params(axis='x', rotation=0)
        for label in ax.get_xticklabels():
            label.set_ha('center')

        self.fig6.tight_layout()
        self.canvas6.draw()

    # ═══════════════ Core bar-plot engine ═══════════════
    def _plot_grouped_bar(self, ax, df, x_col, y_col, color_col, hatch_col,
                          annotate=True):
        x_vals = sorted(df[x_col].dropna().unique().astype(str), key=sort_key)
        n_x = len(x_vals)
        if n_x == 0:
            return [], {}, {}

        color_vals = sorted(df[color_col].dropna().unique().astype(str))
        palette = sns.color_palette(PALETTE_NAME, n_colors=max(1, len(color_vals)))
        color_map = {v: palette[i % len(palette)] for i, v in enumerate(color_vals)}

        if hatch_col and hatch_col in df.columns:
            hatch_vals = sorted(df[hatch_col].dropna().unique().astype(str), key=sort_key)
        else:
            hatch_vals = [None]
        hatch_map = {v: HATCH_PATTERNS[i % len(HATCH_PATTERNS)]
                     for i, v in enumerate(hatch_vals)}

        groups: List[Tuple[str, Optional[str]]] = []
        for cv in color_vals:
            for hv in hatch_vals:
                groups.append((cv, hv))
        n_groups = len(groups)
        if n_groups == 0:
            return groups, color_map, hatch_map

        bar_width = 0.8 / max(1, n_groups)
        x_pos = np.arange(n_x)

        for gi, (cv, hv) in enumerate(groups):
            mask = df[color_col].astype(str) == cv
            if hv is not None and hatch_col:
                mask &= df[hatch_col].astype(str) == hv
            sub = df[mask]

            vals = []
            for xv in x_vals:
                rows = sub[sub[x_col].astype(str) == xv]
                vals.append(rows[y_col].mean() if len(rows) > 0 else 0)

            offset = (gi - n_groups / 2 + 0.5) * bar_width
            hatch = hatch_map.get(hv, '')
            bars = ax.bar(
                x_pos + offset, vals,
                width=bar_width * 0.9,
                color=color_map[cv],
                hatch=hatch,
                edgecolor='white' if hatch else color_map[cv],
                linewidth=0.8, zorder=2,
            )
            if annotate:
                for bar_rect, v in zip(bars, vals):
                    if v > 0:
                        ax.annotate(
                            f"{v:.2f}",
                            (bar_rect.get_x() + bar_rect.get_width() / 2,
                             bar_rect.get_height()),
                            ha='center', va='bottom', fontsize=7,
                            rotation=45, xytext=(0, 2),
                            textcoords='offset points',
                        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_vals,
                           rotation=0 if n_x <= 12 else 30,
                           ha='right' if n_x > 12 else 'center')
        return groups, color_map, hatch_map

    # ═══════════════ Helpers ═══════════════
    @staticmethod
    def _apply_scale(ax, mode: str):
        if mode == 'log2':
            ax.set_yscale('log', base=2)
        elif mode == 'log4':
            ax.set_yscale('log', base=4)
        elif mode == 'log10':
            ax.set_yscale('log', base=10)
        else:
            ax.set_yscale('linear')

    @staticmethod
    def _add_legend(ax, groups, color_map, hatch_map, hatch_col,
                    extra_handles=None):
        handles = []
        for cv, hv in groups:
            hatch = hatch_map.get(hv, '')
            lbl = f"{cv}" if hv is None else f"{cv} | {_cap(hatch_col)}={hv}"
            handles.append(mpatches.Patch(
                facecolor=color_map[cv], hatch=hatch,
                edgecolor='white' if hatch else color_map[cv],
                label=lbl, linewidth=0.8))
        if extra_handles:
            handles.extend(extra_handles)
        ncol = min(5, len(handles))
        ax.legend(handles=handles, loc='best', ncol=ncol,
                  fontsize=8, title_fontsize=9)

    def _export_png(self, fig):
        if not fig.axes:
            messagebox.showerror("Error", "Please plot a chart first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png")])
        if not path:
            return
        fig.savefig(path, bbox_inches='tight', dpi=200)
        messagebox.showinfo("Saved", f"Exported: {path}")


# ─────────────────────────────────────────────────
def main():
    app = CompareApp()
    app.mainloop()


if __name__ == "__main__":
    main()