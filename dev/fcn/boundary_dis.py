import pandas as pd
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from typing import Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib import ticker as mticker
from scipy.interpolate import griddata


class PerformanceBoundaryApp:
    _AXIS_COLS = frozenset({'scale', 'conn_num', 'elapsed_s'})

    def __init__(self, root):
        self.root = root
        self.root.title("Sparse Matrix Performance Boundary & Speedup Analyzer")
        self.root.geometry("1440x960")
        self.df: Optional[pd.DataFrame] = None
        self.comboboxes: dict = {}   # dynamic filter comboboxes
        self.extra_curves: list = []  # extra boundary curves list
        self.sparsity_curves: list = []  # manual sparsity curve list
        self.mode = tk.StringVar(value='mv')
        self.var_auto_sparsity = tk.BooleanVar(value=False)
        self.var_manual_sparsity = tk.BooleanVar(value=False)
        self._setup_ui()

    # ─────────────────────────────────────────────────────────────────────────
    def _setup_ui(self):
        # ── Menu Bar ──────────────────────────────────────────────────────────
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load CSV", command=self.load_data)
        file_menu.add_command(label="Export Image", command=self.export_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        mode_menu = tk.Menu(menubar, tearoff=0)
        mode_menu.add_radiobutton(label="MV Mode (matrix-vector)", variable=self.mode, value='mv')
        mode_menu.add_radiobutton(label="MM Mode (matrix-matrix)", variable=self.mode, value='mm')
        menubar.add_cascade(label="Mode", menu=mode_menu)

        # ── Global Controls ──────────────────────────────────────────────────
        ctrl = ttk.LabelFrame(self.root, text="Global Settings", padding=8)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=5, pady=4)

        ttk.Label(ctrl, text="_N (elem/scale):").pack(side=tk.LEFT, padx=(10, 2))
        self.entry_n = ttk.Entry(ctrl, width=9)
        self.entry_n.insert(0, "4000")
        self.entry_n.pack(side=tk.LEFT)

        ttk.Label(ctrl, text="VRAM (GB):").pack(side=tk.LEFT, padx=(10, 2))
        self.entry_limit = ttk.Entry(ctrl, width=7)
        self.entry_limit.insert(0, "16")
        self.entry_limit.pack(side=tk.LEFT)

        self.var_auto_vram = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Auto VRAM", variable=self.var_auto_vram,
                        command=self._on_auto_vram_changed).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(ctrl, text="Data Type:").pack(side=tk.LEFT, padx=(10, 2))
        self.combo_dtype = ttk.Combobox(ctrl, state="readonly", width=8,
                                        values=['float32', 'int32', 'int8'])
        self.combo_dtype.current(0)
        self.combo_dtype.pack(side=tk.LEFT)

        ttk.Label(ctrl, text="Topology:").pack(side=tk.LEFT, padx=(8, 2))
        self.combo_topology = ttk.Combobox(ctrl, state="readonly", width=8,
                                           values=['homo', 'hetero'])
        self.combo_topology.current(0)
        self.combo_topology.pack(side=tk.LEFT)

        ttk.Label(ctrl, text="Batch Size:").pack(side=tk.LEFT, padx=(8, 2))
        self.entry_batch_size = ttk.Entry(ctrl, width=7)
        self.entry_batch_size.insert(0, "1000")
        self.entry_batch_size.pack(side=tk.LEFT)

        ttk.Label(ctrl, text="Contour Lines:").pack(side=tk.LEFT, padx=(10, 2))
        self.entry_contours = ttk.Entry(ctrl, width=5)
        self.entry_contours.insert(0, "5")
        self.entry_contours.pack(side=tk.LEFT)

        ttk.Label(ctrl, text="Custom Lines:").pack(side=tk.LEFT, padx=(10, 2))
        self.entry_custom_lines = ttk.Entry(ctrl, width=16)
        self.entry_custom_lines.insert(0, "")
        self.entry_custom_lines.pack(side=tk.LEFT)
        ttk.Label(ctrl, text="(Comma-separated values, only for interpolation/speedup plots)",
              foreground='gray').pack(side=tk.LEFT, padx=(2, 0))

        ttk.Button(ctrl, text="Update Plots", command=self.update_plots).pack(side=tk.LEFT, padx=12)

        ttk.Separator(ctrl, orient='vertical').pack(side=tk.LEFT, fill='y', padx=8, pady=2)

        ttk.Button(ctrl, text="Export Image", command=self.export_image).pack(side=tk.LEFT, padx=4)
        ttk.Label(ctrl, text="DPI:").pack(side=tk.LEFT, padx=(6, 2))
        self.entry_dpi = ttk.Entry(ctrl, width=5)
        self.entry_dpi.insert(0, "150")
        self.entry_dpi.pack(side=tk.LEFT)

        # ── Comparison Configuration (Compare Field → Target / Baseline) ──────
        cmp_bar = ttk.LabelFrame(self.root, text="Comparison Configuration", padding=8)
        cmp_bar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=4)

        ttk.Label(cmp_bar, text="Compare Field:").pack(side=tk.LEFT, padx=(0, 2))
        self.combo_compare_field = ttk.Combobox(cmp_bar, state="readonly", width=18)
        self.combo_compare_field.pack(side=tk.LEFT)
        self.combo_compare_field.bind("<<ComboboxSelected>>", self._on_compare_field_changed)

        ttk.Separator(cmp_bar, orient='vertical').pack(side=tk.LEFT, fill='y', padx=10, pady=2)

        ttk.Label(cmp_bar, text="Target:").pack(side=tk.LEFT, padx=(0, 2))
        self.combo_target = ttk.Combobox(cmp_bar, state="readonly", width=18)
        self.combo_target.pack(side=tk.LEFT)

        ttk.Label(cmp_bar, text="Baseline:").pack(side=tk.LEFT, padx=(8, 2))
        self.combo_baseline = ttk.Combobox(cmp_bar, state="readonly", width=18)
        self.combo_baseline.pack(side=tk.LEFT)

        # ── Speedup Color Settings ─────────────────────────────────────────────
        sp_bar = ttk.LabelFrame(self.root, text="Speedup Color Settings", padding=8)
        sp_bar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=4)

        self.var_auto_speedup = tk.BooleanVar(value=True)
        ttk.Checkbutton(sp_bar, text="Auto (compute extrema from data)",
                        variable=self.var_auto_speedup,
                        command=self._on_auto_speedup_changed).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(sp_bar, text="Yellow Deepest (speedup ≤):").pack(side=tk.LEFT, padx=(0, 2))
        self.entry_yellow_depth = ttk.Entry(sp_bar, width=7)
        self.entry_yellow_depth.insert(0, "0.5")
        self.entry_yellow_depth.pack(side=tk.LEFT)

        ttk.Label(sp_bar, text="Blue Deepest (speedup ≥):").pack(side=tk.LEFT, padx=(16, 2))
        self.entry_blue_depth = ttk.Entry(sp_bar, width=7)
        self.entry_blue_depth.insert(0, "2.0")
        self.entry_blue_depth.pack(side=tk.LEFT)

        ttk.Label(sp_bar,
              text="  If Auto checked: compute extrema from data (yellow ≤ 0.5 / blue ≥ 1.5); otherwise use manual inputs on the left",
              foreground='gray').pack(side=tk.LEFT, padx=(12, 0))

        self._on_auto_speedup_changed()

        # ── Extra Boundary Curves (multi-curve list) ──────────────────────────
        ext_bar = ttk.LabelFrame(self.root, text="Extra Boundary Curves", padding=8)
        ext_bar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=4)

        curve_area = ttk.LabelFrame(ext_bar, text="VRAM Boundary Curves", padding=6)
        curve_area.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        ttk.Separator(ext_bar, orient='vertical').pack(side=tk.LEFT, fill='y', padx=4, pady=2)

        sparsity_area = ttk.LabelFrame(ext_bar, text="Sparsity Lines", padding=6)
        sparsity_area.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

        # — Listbox (left) —
        list_frame = ttk.Frame(curve_area)
        list_frame.pack(side=tk.LEFT, padx=(0, 8))
        self.extra_curves_listbox = tk.Listbox(list_frame, height=4, width=40,
                                               selectmode=tk.SINGLE, exportselection=False)
        self.extra_curves_listbox.pack(side=tk.LEFT, fill=tk.Y)
        _sb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL,
                            command=self.extra_curves_listbox.yview)
        _sb.pack(side=tk.LEFT, fill=tk.Y)
        self.extra_curves_listbox.config(yscrollcommand=_sb.set)
        self.extra_curves_listbox.bind("<<ListboxSelect>>", self._on_extra_curve_selected)
        ttk.Label(list_frame, text="Click item to edit; deselect to add new",
                  foreground='gray').pack(anchor='w', pady=(2, 0))

        # — Editor (center) —
        editor = ttk.Frame(curve_area)
        editor.pack(side=tk.LEFT, padx=4)

        ttk.Label(editor, text="VRAM (GB):").grid(row=0, column=0, sticky='e', padx=(0, 2), pady=2)
        self.entry_extra_limit = ttk.Entry(editor, width=7)
        self.entry_extra_limit.insert(0, "16")
        self.entry_extra_limit.grid(row=0, column=1, padx=2, pady=2)

        ttk.Label(editor, text="Data Type:").grid(row=0, column=2, sticky='e', padx=(8, 2), pady=2)
        self.combo_extra_dtype = ttk.Combobox(editor, state="readonly", width=8,
                                              values=['float32', 'int32', 'int8'])
        self.combo_extra_dtype.current(0)
        self.combo_extra_dtype.grid(row=0, column=3, padx=2, pady=2)

        ttk.Label(editor, text="Topology:").grid(row=0, column=4, sticky='e', padx=(8, 2), pady=2)
        self.combo_extra_topology = ttk.Combobox(editor, state="readonly", width=8,
                                                 values=['homo', 'hetero'])
        self.combo_extra_topology.current(0)
        self.combo_extra_topology.grid(row=0, column=5, padx=2, pady=2)

        ttk.Label(editor, text="Color:").grid(row=1, column=0, sticky='e', padx=(0, 2), pady=2)
        self.entry_extra_color = ttk.Entry(editor, width=10)
        self.entry_extra_color.insert(0, "#E53935")
        self.entry_extra_color.grid(row=1, column=1, padx=2, pady=2)
        ttk.Button(editor, text="Pick…", command=self._pick_extra_color,
                   width=5).grid(row=1, column=2, padx=2, pady=2)

        ttk.Label(editor, text="Label:").grid(row=1, column=3, sticky='e', padx=(4, 2), pady=2)
        self.entry_extra_label = ttk.Entry(editor, width=24)
        self.entry_extra_label.insert(0, "")
        self.entry_extra_label.grid(row=1, column=4, columnspan=2, padx=2, pady=2, sticky='ew')
        editor.columnconfigure(5, weight=1)

        sparsity_list_frame = ttk.Frame(sparsity_area)
        sparsity_list_frame.pack(side=tk.LEFT, padx=(0, 8))
        self.sparsity_listbox = tk.Listbox(sparsity_list_frame, height=3, width=28,
                                           selectmode=tk.SINGLE, exportselection=False)
        self.sparsity_listbox.pack(side=tk.LEFT, fill=tk.Y)
        _sp_sb = ttk.Scrollbar(sparsity_list_frame, orient=tk.VERTICAL,
                               command=self.sparsity_listbox.yview)
        _sp_sb.pack(side=tk.LEFT, fill=tk.Y)
        self.sparsity_listbox.config(yscrollcommand=_sp_sb.set)
        self.sparsity_listbox.bind("<<ListboxSelect>>", self._on_sparsity_curve_selected)
        ttk.Label(sparsity_list_frame, text="Click item to edit",
                  foreground='gray').pack(anchor='w', pady=(2, 0))

        sparsity_editor = ttk.Frame(sparsity_area)
        sparsity_editor.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        ttk.Checkbutton(sparsity_editor, text="Auto Sparsity",
                        variable=self.var_auto_sparsity,
                        command=self._on_sparsity_option_changed).grid(
                            row=0, column=0, columnspan=5, sticky='w', padx=(0, 2), pady=(0, 2))
        ttk.Label(sparsity_editor, text="Count:").grid(
            row=1, column=0, sticky='e', padx=(0, 2), pady=2)
        self.entry_auto_sparsity_count = ttk.Entry(sparsity_editor, width=8)
        self.entry_auto_sparsity_count.insert(0, "3")
        self.entry_auto_sparsity_count.grid(row=1, column=1, padx=2, pady=2, sticky='w')
        ttk.Label(sparsity_editor, text="Angles evenly spaced from -90 to 0 degrees",
                  foreground='gray').grid(row=2, column=0, columnspan=5,
                                          sticky='w', padx=(0, 2), pady=(0, 8))

        ttk.Checkbutton(sparsity_editor, text="Manual Sparsity",
                        variable=self.var_manual_sparsity,
                        command=self._on_sparsity_option_changed).grid(
                            row=3, column=0, columnspan=5, sticky='w', padx=(0, 2), pady=2)
        ttk.Label(sparsity_editor, text="Value:").grid(
            row=4, column=0, sticky='e', padx=(0, 2), pady=2)
        self.entry_manual_sparsity = ttk.Entry(sparsity_editor, width=16)
        self.entry_manual_sparsity.insert(0, "")
        self.entry_manual_sparsity.grid(row=4, column=1, padx=2, pady=2, sticky='ew')
        ttk.Label(sparsity_editor, text="Color:").grid(
            row=4, column=2, sticky='e', padx=(8, 2), pady=2)
        self.entry_sparsity_color = ttk.Entry(sparsity_editor, width=10)
        self.entry_sparsity_color.insert(0, "#000000")
        self.entry_sparsity_color.grid(row=4, column=3, padx=2, pady=2, sticky='w')
        self.btn_pick_sparsity_color = ttk.Button(
            sparsity_editor, text="Pick", command=self._pick_sparsity_color, width=5)
        self.btn_pick_sparsity_color.grid(row=4, column=4, padx=2, pady=2)

        ttk.Label(sparsity_editor, text="Label:").grid(
            row=5, column=0, sticky='e', padx=(0, 2), pady=2)
        self.entry_sparsity_label = ttk.Entry(sparsity_editor, width=24)
        self.entry_sparsity_label.insert(0, "")
        self.entry_sparsity_label.grid(row=5, column=1, columnspan=4,
                                       padx=2, pady=2, sticky='ew')
        ttk.Label(sparsity_editor,
                  text="s = conn / (_N × scale), e.g. 0.01, 0.05 or 1%, 5%",
                  foreground='gray').grid(row=6, column=0, columnspan=5,
                                          sticky='w', padx=(0, 2), pady=(2, 0))
        sparsity_editor.columnconfigure(1, weight=1)

        sparsity_btn_f = ttk.Frame(sparsity_area)
        sparsity_btn_f.pack(side=tk.LEFT, padx=8)
        self.sparsity_buttons = []
        for text, command in [
            ("Add / Save", self._sparsity_curve_add_save),
            ("Delete", self._sparsity_curve_delete),
            ("Clear All", self._sparsity_curve_clear_all),
        ]:
            btn = ttk.Button(sparsity_btn_f, text=text, command=command, width=12)
            btn.pack(fill=tk.X, pady=2)
            self.sparsity_buttons.append(btn)

        # — Buttons (right) —
        btn_f = ttk.Frame(curve_area)
        btn_f.pack(side=tk.LEFT, padx=8)
        ttk.Button(btn_f, text="Add / Save", command=self._extra_curve_add_save,
                   width=12).pack(fill=tk.X, pady=2)
        ttk.Button(btn_f, text="Delete", command=self._extra_curve_delete,
                   width=12).pack(fill=tk.X, pady=2)
        ttk.Button(btn_f, text="Clear All", command=self._extra_curve_clear_all,
                   width=12).pack(fill=tk.X, pady=2)
        self._on_sparsity_option_changed()

        # ── Dynamic Data Slicing Filters ────────────────────────────────────────
        self.flt_outer = ttk.LabelFrame(self.root, text="Data Slicing Filters", padding=8)
        self.flt_outer.pack(side=tk.TOP, fill=tk.X, padx=5, pady=4)
        self.filter_row = ttk.Frame(self.flt_outer)
        self.filter_row.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(self.filter_row, text="(Filter options populated after loading CSV)",
              foreground='gray').pack(side=tk.LEFT)

        # ── Plot container ────────────────────────────────────────────────────
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=4)

        self.tabs: dict = {}
        tab_configs = [
            ("Raw Scatter",    "Target Elapsed Time  –  Scatter"),
            ("Raw Interp",     "Target Elapsed Time  –  Interpolated"),
            ("Speedup Interp", "Speedup  (Baseline / Target)  –  Interpolated Grid"),
        ]
        for name, title in tab_configs:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=name)
            fig = Figure(figsize=(13, 5.2), dpi=100)
            ax = fig.add_subplot(111)
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            NavigationToolbar2Tk(canvas, frame)
            self.tabs[name] = {
                'fig': fig, 'ax': ax, 'canvas': canvas, 'title': title,
                'cbar': None, 'hover_cid': None, 'scatter': None,
            }

    def _on_compare_field_changed(self, event=None):
        if self.df is None:
            return
        field = self.combo_compare_field.get()
        if not field:
            return
        vals = sorted(self.df[field].dropna().unique().tolist(), key=str)
        self.combo_target['values'] = vals
        self.combo_baseline['values'] = vals
        if vals:
            self.combo_target.current(0)
            self.combo_baseline.current(min(1, len(vals) - 1))
        self._rebuild_filter_row()

    def _rebuild_filter_row(self):
        for widget in self.filter_row.winfo_children():
            widget.destroy()
        self.comboboxes = {}
        if self.df is None:
            ttk.Label(self.filter_row, text="(Filter options populated after loading CSV)",
                      foreground='gray').pack(side=tk.LEFT)
            return
        compare_field = self.combo_compare_field.get()
        exclude = self._AXIS_COLS | ({compare_field} if compare_field else set())
        filter_cols = [c for c in self.df.columns if c not in exclude]
        if not filter_cols:
            ttk.Label(self.filter_row, text="(No additional filter columns)",
                      foreground='gray').pack(side=tk.LEFT)
            return
        for col in filter_cols:
            f = ttk.Frame(self.filter_row)
            f.pack(side=tk.LEFT, padx=5)
            ttk.Label(f, text=f"{col}:").pack(anchor='w')
            cb = ttk.Combobox(f, state="readonly", width=13)
            vals_list = [''] + sorted([str(v) for v in self.df[col].dropna().unique()])
            cb['values'] = vals_list
            cb.current(0)
            cb.pack()
            self.comboboxes[col] = cb

    def _on_auto_speedup_changed(self):
        state = 'disabled' if self.var_auto_speedup.get() else 'normal'
        self.entry_yellow_depth.config(state=state)
        self.entry_blue_depth.config(state=state)

    def _on_auto_vram_changed(self):
        state = 'disabled' if self.var_auto_vram.get() else 'normal'
        self.entry_limit.config(state=state)

    def _on_sparsity_option_changed(self):
        auto_state = 'normal' if self.var_auto_sparsity.get() else 'disabled'
        manual_state = 'normal' if self.var_manual_sparsity.get() else 'disabled'
        self.entry_auto_sparsity_count.config(state=auto_state)
        self.entry_manual_sparsity.config(state=manual_state)
        self.entry_sparsity_color.config(state=manual_state)
        self.entry_sparsity_label.config(state=manual_state)
        self.btn_pick_sparsity_color.config(state=manual_state)
        for btn in getattr(self, 'sparsity_buttons', []):
            btn.config(state=manual_state)

    @staticmethod
    def _parse_sparsity_value(raw_value: str) -> float:
        token = raw_value.strip()
        if not token:
            raise ValueError("Sparsity value cannot be empty.")
        as_percent = token.endswith('%')
        if as_percent:
            token = token[:-1].strip()
        value = float(token)
        if as_percent or value > 1:
            value /= 100.0
        if not (0 < value <= 1):
            raise ValueError("Sparsity must be in (0, 1], or use percentages like 5%.")
        return value

    @staticmethod
    def _format_sparsity_label(value: float) -> str:
        return f"{value * 100:.2f}%"

    @staticmethod
    def _sparsity_curve_display(c: dict) -> str:
        return f"{PerformanceBoundaryApp._format_sparsity_label(c['value'])} | {c['color']}  -> {c['label']}"

    def _refresh_sparsity_listbox(self):
        self.sparsity_listbox.delete(0, tk.END)
        for c in self.sparsity_curves:
            self.sparsity_listbox.insert(tk.END, self._sparsity_curve_display(c))

    def _on_sparsity_curve_selected(self, event=None):
        sel = self.sparsity_listbox.curselection()
        if not sel:
            return
        c = self.sparsity_curves[sel[0]]
        self.entry_manual_sparsity.delete(0, tk.END)
        self.entry_manual_sparsity.insert(0, self._format_sparsity_label(c['value']))
        self.entry_sparsity_color.delete(0, tk.END)
        self.entry_sparsity_color.insert(0, c['color'])
        self.entry_sparsity_label.delete(0, tk.END)
        self.entry_sparsity_label.insert(0, c['label'])

    def _pick_sparsity_color(self):
        current = self.entry_sparsity_color.get().strip() or '#000000'
        result = colorchooser.askcolor(color=current, title="Pick Sparsity Line Color",
                                       parent=self.root)
        if result and result[1]:
            self.entry_sparsity_color.delete(0, tk.END)
            self.entry_sparsity_color.insert(0, result[1])

    def _sparsity_curve_add_save(self):
        try:
            value = self._parse_sparsity_value(self.entry_manual_sparsity.get())
        except ValueError as exc:
            messagebox.showerror("Input Error", str(exc))
            return
        color = self.entry_sparsity_color.get().strip() or '#000000'
        label = self.entry_sparsity_label.get().strip() or self._format_sparsity_label(value)
        c = dict(value=value, color=color, label=label)
        sel = self.sparsity_listbox.curselection()
        if sel:
            self.sparsity_curves[sel[0]] = c
        else:
            self.sparsity_curves.append(c)
        self._refresh_sparsity_listbox()

    def _sparsity_curve_delete(self):
        sel = self.sparsity_listbox.curselection()
        if not sel:
            messagebox.showinfo("Delete", "Select a sparsity line from the list first.")
            return
        self.sparsity_curves.pop(sel[0])
        self._refresh_sparsity_listbox()

    def _sparsity_curve_clear_all(self):
        if not self.sparsity_curves:
            return
        if messagebox.askyesno("Clear All", "Remove all manual sparsity lines?"):
            self.sparsity_curves.clear()
            self._refresh_sparsity_listbox()

    def _collect_sparsity_lines(self, _N, x_min, x_max, y_max):
        if not self.var_auto_sparsity.get() and not self.var_manual_sparsity.get():
            return []

        lines = []
        seen = set()

        if self.var_auto_sparsity.get():
            try:
                auto_count = int(self.entry_auto_sparsity_count.get())
            except ValueError as exc:
                raise ValueError("Auto sparsity count must be a positive integer.") from exc
            if auto_count <= 0:
                raise ValueError("Auto sparsity count must be a positive integer.")

            angles = np.linspace(-90.0, 0.0, auto_count + 2, dtype=float)[1:-1]
            label_fracs = np.linspace(0.22, 0.82, len(angles), dtype=float)
            auto_legend_added = False
            for angle, label_frac in zip(angles, label_fracs):
                lines.append({
                    'target_angle': float(angle),
                    'inline_text': None,
                    'color': 'black',
                    'linestyle': '--',
                    'lw': 1.4,
                    'label_frac': float(label_frac),
                    'legend_label': "Auto sparsity lines" if not auto_legend_added else None,
                    'fontweight': 'normal',
                })
                auto_legend_added = True

        if self.var_manual_sparsity.get():
            manual_specs = list(self.sparsity_curves)
            draft = self.entry_manual_sparsity.get().strip()
            if draft:
                for part in draft.split(','):
                    token = part.strip()
                    if token:
                        value = self._parse_sparsity_value(token)
                        manual_specs.append({
                            'value': value,
                            'color': self.entry_sparsity_color.get().strip() or '#000000',
                            'label': self.entry_sparsity_label.get().strip() or self._format_sparsity_label(value),
                        })

            if not manual_specs:
                raise ValueError("Manual sparsity is checked, but no sparsity line was added.")

            for c in manual_specs:
                value = float(c['value'])
                key = round(value, 12)
                if key in seen:
                    continue
                seen.add(key)
                label = c.get('label') or self._format_sparsity_label(value)
                lines.append({
                    'value': value,
                    'inline_text': label,
                    'color': c.get('color', '#000000'),
                    'linestyle': '--',
                    'lw': 1.8,
                    'legend_label': label,
                    'fontweight': 'normal',
                })

        return lines

    def _compute_auto_vram(self, df_subset) -> float:
        """Return the minimum VRAM (GB) for the VRAM boundary to encompass
        all (scale, conn_num) data points in *df_subset* (with a 5 % margin)."""
        if df_subset.empty:
            return 1.0
        try:
            _N = int(self.entry_n.get())
        except ValueError:
            _N = 4000
        data_size, times = self._get_boundary_params()
        if self.mode.get() == 'mm' and 'batch_size' in df_subset.columns:
            bs_vals = df_subset['batch_size'].values.astype(float)
            size    = df_subset['scale'].values.astype(float) * _N
            conn    = df_subset['conn_num'].values.astype(float)
            required = (conn * size * times + 2 * bs_vals * size) * data_size
        else:
            mv_layout_factors = self._mv_layout_factors_for_rows(df_subset)
            required = self._vram_bytes(df_subset['conn_num'], df_subset['scale'],
                                        data_size, times, _N,
                                        mv_layout_factor=mv_layout_factors)
        return float(required.max()) / (1024 ** 3) * 1.05

    @staticmethod
    def _dtype_size(dtype_str: str) -> int:
        """Return bytes-per-element for the given dtype label."""
        return {'float32': 4, 'int32': 4, 'int8': 1}.get(dtype_str, 4)

    @staticmethod
    def _topology_times(topo_str: str) -> int:
        """Return memory-multiplier for homo (1) or hetero (2) topology."""
        return 2 if topo_str == 'hetero' else 1

    @staticmethod
    def _single_value_or_default(df_subset, column: str, default: str) -> str:
        if df_subset is None or df_subset.empty or column not in df_subset.columns:
            return default
        values = df_subset[column].dropna().astype(str).unique()
        return str(values[0]) if len(values) == 1 else default

    @staticmethod
    def _mv_layout_factor(mv_layout: str = 'row_gather', data_type: str = 'binary') -> int:
        """Return MV storage factor: row_gather is 1x, col_scatter dual-view is 2x."""
        if mv_layout == 'col_scatter' and data_type in ('binary', 'compact'):
            return 2
        return 1

    def _mv_layout_factors_for_rows(self, df_subset):
        if df_subset is None or df_subset.empty:
            return np.asarray([], dtype=float)
        if 'mv_layout' in df_subset.columns:
            layouts = df_subset['mv_layout'].fillna('row_gather').astype(str).to_numpy()
        else:
            layouts = np.full(len(df_subset), 'row_gather', dtype=object)
        if 'data_type' in df_subset.columns:
            data_types = df_subset['data_type'].fillna('binary').astype(str).to_numpy()
        else:
            data_types = np.full(len(df_subset), 'binary', dtype=object)
        return np.asarray([
            self._mv_layout_factor(layout, data_type)
            for layout, data_type in zip(layouts, data_types)
        ], dtype=float)

    def _mv_boundary_context(self, df_subset) -> tuple[str, int]:
        mv_layout = self._single_value_or_default(df_subset, 'mv_layout', 'mixed')
        factors = self._mv_layout_factors_for_rows(df_subset)
        factor = int(factors.max()) if len(factors) else 1
        if mv_layout == 'mixed' and factor == 1:
            mv_layout = 'row_gather'
        return mv_layout, factor

    def _get_boundary_params(self):
        """Read data_size and times from the main boundary UI controls."""
        return (self._dtype_size(self.combo_dtype.get()),
                self._topology_times(self.combo_topology.get()))

    def _get_batch_size(self):
        try:
            return int(self.entry_batch_size.get())
        except ValueError:
            return 1000

    def _vram_bytes(self, conn, scale, data_size, times, _N, bs=None, mv_layout_factor=1):
        """Compute VRAM usage in bytes based on current mode (mv / mm)."""
        if self.mode.get() == 'mm':
            if bs is None:
                bs = self._get_batch_size()
            size = scale * _N
            return (conn * size * times + 2 * bs * size) * data_size
        return conn * scale * data_size * times * _N * mv_layout_factor

    def _boundary_conn(self, scale, limit_bytes, data_size, times, _N, bs=None, mv_layout_factor=1):
        """Compute max conn_num for given scale values and VRAM limit."""
        if self.mode.get() == 'mm':
            if bs is None:
                bs = self._get_batch_size()
            size = scale * _N
            return (limit_bytes / data_size - 2 * bs * size) / (size * times)
        return limit_bytes / (data_size * times * scale * _N * mv_layout_factor)

    # ── Extra boundary curve list helpers ─────────────────────────────────────
    @staticmethod
    def _extra_curve_display(c: dict) -> str:
        return f"{c['limit_gb']} GB | {c['dtype']} | {c['topology']}  →  {c['label']}"

    def _refresh_extra_listbox(self):
        self.extra_curves_listbox.delete(0, tk.END)
        for c in self.extra_curves:
            self.extra_curves_listbox.insert(tk.END, self._extra_curve_display(c))

    def _on_extra_curve_selected(self, event=None):
        sel = self.extra_curves_listbox.curselection()
        if not sel:
            return
        c = self.extra_curves[sel[0]]
        self.entry_extra_limit.delete(0, tk.END)
        self.entry_extra_limit.insert(0, str(c['limit_gb']))
        self.combo_extra_dtype.set(c['dtype'])
        self.combo_extra_topology.set(c['topology'])
        self.entry_extra_color.delete(0, tk.END)
        self.entry_extra_color.insert(0, c['color'])
        self.entry_extra_label.delete(0, tk.END)
        self.entry_extra_label.insert(0, c['label'])

    def _pick_extra_color(self):
        current = self.entry_extra_color.get().strip() or '#E53935'
        result = colorchooser.askcolor(color=current, title="Pick Curve Color",
                                       parent=self.root)
        if result and result[1]:
            self.entry_extra_color.delete(0, tk.END)
            self.entry_extra_color.insert(0, result[1])

    def _extra_curve_add_save(self):
        try:
            limit_gb = float(self.entry_extra_limit.get())
        except ValueError:
            messagebox.showerror("Input Error", "VRAM must be a valid number.")
            return
        dtype    = self.combo_extra_dtype.get()
        topology = self.combo_extra_topology.get()
        color    = self.entry_extra_color.get().strip() or '#E53935'
        label    = self.entry_extra_label.get().strip()
        if not label:
            label = f"{limit_gb} GB / {dtype} / {topology}"
        c = dict(limit_gb=limit_gb, dtype=dtype, topology=topology,
                 color=color, label=label)
        sel = self.extra_curves_listbox.curselection()
        if sel:
            self.extra_curves[sel[0]] = c
        else:
            self.extra_curves.append(c)
        self._refresh_extra_listbox()

    def _extra_curve_delete(self):
        sel = self.extra_curves_listbox.curselection()
        if not sel:
            messagebox.showinfo("Delete", "Select a curve from the list first.")
            return
        self.extra_curves.pop(sel[0])
        self._refresh_extra_listbox()

    def _extra_curve_clear_all(self):
        if not self.extra_curves:
            return
        if messagebox.askyesno("Clear All", "Remove all extra boundary curves?"):
            self.extra_curves.clear()
            self._refresh_extra_listbox()

    def export_image(self):
        idx = self.notebook.index(self.notebook.select())
        tab = list(self.tabs.values())[idx]
        try:
            dpi = int(self.entry_dpi.get())
            if dpi <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "DPI must be a positive integer.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"), ("All", "*.*")],
        )
        if path:
            try:
                # Export the current figure frame only; do not expand the
                # output to include artists extending beyond the figure bounds.
                tab['canvas'].draw()
                tab['fig'].savefig(
                    path,
                    dpi=dpi,
                    bbox_inches=None,
                    pad_inches=0,
                    facecolor=tab['fig'].get_facecolor(),
                    edgecolor='none',
                )
                messagebox.showinfo("Exported", f"Saved: {path}  ({dpi} DPI)")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to save image:\n{e}")

    def load_data(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if not path:
            return
        try:
            self.df = pd.read_csv(path)
            required = {'scale', 'conn_num', 'elapsed_s'}
            missing = required - set(self.df.columns)
            if missing:
                raise ValueError(f"CSV missing columns: {missing}")
            # Comparison fields = all non-axis columns
            cmp_cols = [c for c in self.df.columns if c not in self._AXIS_COLS]
            self.combo_compare_field['values'] = cmp_cols
            if cmp_cols:
                self.combo_compare_field.current(0)
            self._on_compare_field_changed()
            messagebox.showinfo("Success", f"Loaded {len(self.df)} rows.")
            self.update_plots()
        except Exception as e:
            messagebox.showerror("Data Load Error", str(e))

    def _subtitle(self, include_baseline: bool = False) -> str:
        parts = [f"{c} = {cb.get()}"
                 for c, cb in self.comboboxes.items() if cb.get()]
        cmp_field = self.combo_compare_field.get()
        parts.append(f"Target [{cmp_field}]: {self.combo_target.get()}")
        if include_baseline:
            parts.append(f"Baseline [{cmp_field}]: {self.combo_baseline.get()}")
        return "  │  ".join(parts)

    def update_plots(self):
        if self.df is None or self.df.empty:
            return
        try:
            _N = int(self.entry_n.get())
        except ValueError:
            messagebox.showerror("Input Error", "N 输入无效。")
            return
        if _N <= 0:
            messagebox.showerror("Input Error", "N 必须为正整数。")
            return

        # ── Remove old colorbars and clear axes (avoid overplotting) ──────────
        for tab in self.tabs.values():
            if tab['cbar'] is not None:
                try:
                    tab['cbar'].remove()
                except Exception:
                    pass
                tab['cbar'] = None
            if tab['hover_cid'] is not None:
                try:
                    tab['canvas'].mpl_disconnect(tab['hover_cid'])
                except Exception:
                    pass
                tab['hover_cid'] = None
            tab['ax'].clear()

        # ── Common filtering ──────────────────────────────────────────────────
        mask = pd.Series(True, index=self.df.index)
        for col, cb in self.comboboxes.items():
            val = cb.get()
            if val:
                mask &= (self.df[col].astype(str) == str(val))
        df_f = self.df[mask]

        cmp_field  = self.combo_compare_field.get()
        target_val = self.combo_target.get()
        base_val   = self.combo_baseline.get()

        if not cmp_field:
            messagebox.showwarning("No Compare Field", "Please select a compare field first.")
            return

        df_t = df_f[df_f[cmp_field].astype(str) == str(target_val)]
        df_b = df_f[df_f[cmp_field].astype(str) == str(base_val)]

        if df_t.empty:
            messagebox.showwarning("No Data", "Target data subset is empty.")
            for t in self.tabs.values():
                t['canvas'].draw()
            return

        mv_layout, mv_layout_factor = self._mv_boundary_context(df_t)
        # ── MM 模式：从过滤后的数据中提取实际 batch_size 用于显存边界计算 ──────────────────
        eff_bs = None
        if self.mode.get() == 'mm':
            if 'batch_size' in df_t.columns and not df_t.empty:
                try:
                    eff_bs = int(df_t['batch_size'].iloc[0])
                except (ValueError, TypeError):
                    eff_bs = self._get_batch_size()
            else:
                eff_bs = self._get_batch_size()
            self.entry_batch_size.delete(0, tk.END)
            self.entry_batch_size.insert(0, str(eff_bs))
        # ── Compute VRAM limit (auto or manual) ───────────────────────────────
        if self.var_auto_vram.get():
            df_union = pd.concat([df_t, df_b]).drop_duplicates() if not df_b.empty else df_t
            auto_gb  = self._compute_auto_vram(df_union)
            auto_gb  = max(auto_gb, 0.001)
            self.entry_limit.config(state='normal')
            self.entry_limit.delete(0, tk.END)
            self.entry_limit.insert(0, f"{auto_gb:.3g}")
            self.entry_limit.config(state='disabled')
            limit_gb = auto_gb
        else:
            try:
                limit_gb = float(self.entry_limit.get())
            except ValueError:
                messagebox.showerror("Input Error", "VRAM 限制输入无效。")
                return

        x_max = max(df_t['scale'].max(),
                    df_b['scale'].max() if not df_b.empty else 1, 1)
        x_min = max(min(df_t['scale'].min(),
                        df_b['scale'].min() if not df_b.empty else 1), 0.1)
        y_max = max(df_t['conn_num'].max(),
                    df_b['conn_num'].max() if not df_b.empty else 1, 1)
        try:
            sparsity_lines = self._collect_sparsity_lines(_N, x_min, x_max, y_max)
        except ValueError as exc:
            messagebox.showerror("Input Error", str(exc))
            return

        xt, yt, zt = df_t['scale'].values, df_t['conn_num'].values, df_t['elapsed_s'].values
        sub_raw   = self._subtitle(False)
        kw_common = dict(
            _N=_N,
            limit_gb=limit_gb,
            x_min=x_min,
            x_max=x_max,
            y_max=y_max,
            bs=eff_bs,
            mv_layout=mv_layout,
            mv_layout_factor=mv_layout_factor,
            sparsity_lines=sparsity_lines,
        )

        self._render_scatter(self.tabs["Raw Scatter"], xt, yt, zt,
                             z_label="Elapsed Time (s)", cmap_name='Blues',
                             subtitle=sub_raw, **kw_common)
        self._render_interpolation(self.tabs["Raw Interp"], xt, yt, zt,
                                   z_label="Elapsed Time (s)", cmap_name='Blues',
                                   subtitle=sub_raw, **kw_common)

        ax_sp = self.tabs["Speedup Interp"]['ax']
        if df_b.empty:
            ax_sp.text(0.5, 0.5, "Baseline data subset is empty.",
                       ha='center', va='center', transform=ax_sp.transAxes, fontsize=11)
            self.tabs["Speedup Interp"]['canvas'].draw()
        elif len(xt) < 4 or len(df_b) < 4:
            ax_sp.text(0.5, 0.5, "Not enough data points (<4) for interpolation.",
                       ha='center', va='center', transform=ax_sp.transAxes, fontsize=11)
            self.tabs["Speedup Interp"]['canvas'].draw()
        else:
            xb = df_b['scale'].values
            yb = df_b['conn_num'].values
            zb = df_b['elapsed_s'].values

            grid_x, grid_y = np.mgrid[0 : x_max * 1.1 : 200j, 0 : y_max * 1.2 : 200j]
            grid_zt = griddata((xt, yt), zt, (grid_x, grid_y), method='linear')
            grid_zb = griddata((xb, yb), zb, (grid_x, grid_y), method='linear')

            limit_bytes = limit_gb * (1024 ** 3)
            data_size, times = self._get_boundary_params()
            valid = (
                (grid_y <= grid_x * _N)
                & (
                    self._vram_bytes(
                        grid_y, grid_x, data_size, times, _N,
                        bs=eff_bs, mv_layout_factor=mv_layout_factor,
                    ) <= limit_bytes
                )
            )

            with np.errstate(divide='ignore', invalid='ignore'):
                grid_sp = np.where(
                    np.isfinite(grid_zt) & np.isfinite(grid_zb) & (grid_zt > 0),
                    grid_zb / grid_zt,
                    np.nan,
                )
            grid_sp_masked = np.where(valid, grid_sp, np.nan)

            sub_sp = self._subtitle(True)
            self._render_speedup_interp(
                self.tabs["Speedup Interp"],
                grid_x, grid_y, grid_sp_masked,
                subtitle=sub_sp, **kw_common,
            )

    @staticmethod
    def _plot_limits(x_max, y_max):
        return 0, x_max * 1.1, max(y_max * 1.2, 100), 0

    @staticmethod
    def _axes_data_ratio(ax, x0, x1, y0, y1):
        bbox = ax.get_window_extent()
        width = max(float(bbox.width), 1.0)
        height = max(float(bbox.height), 1.0)
        x_span = max(abs(x1 - x0), 1e-12)
        y_span = max(abs(y0 - y1), 1e-12)
        return (height / y_span) / (width / x_span)

    def _sparsity_from_display_angle(self, angle_deg, _N, ax, x0, x1, y0, y1):
        ratio = self._axes_data_ratio(ax, x0, x1, y0, y1)
        visible_angle = float(np.clip(angle_deg, -89.0, -1.0))
        data_slope = abs(np.tan(np.radians(visible_angle))) / max(ratio, 1e-12)
        return float(np.clip(data_slope / _N, 1e-12, 1.0))

    def _sparsity_label_x(self, value, _N, x0, x1, y0, label_frac):
        slope = max(value * _N, 1e-12)
        x_at_y = y0 / slope
        x_visible_max = min(x1 * 0.96, x_at_y * 0.92)
        x_visible_min = max(x0 + (x1 - x0) * 0.06, 1e-12)
        if x_visible_max <= x_visible_min:
            lower = max(x0 + (x1 - x0) * 0.002, 1e-12)
            return float(np.clip(x_at_y * 0.7, lower, x1 * 0.96))
        return float(x_visible_min + (x_visible_max - x_visible_min) * label_frac)

    def _draw_boundaries(
        self,
        ax,
        _N,
        limit_gb,
        x_min,
        x_max,
        y_max,
        bs=None,
        mv_layout='row_gather',
        mv_layout_factor=1,
        sparsity_lines=None,
    ):
        x0, x1, y0, y1 = self._plot_limits(x_max, y_max)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.figure.canvas.draw()
        xv = np.linspace(max(0.01, x0 + (x1 - x0) * 0.001), x_max * 1.25, 600)
        ax.plot(xv, xv * _N,
                color='gold', lw=2, label=f'conn = scale × N  (N={_N:,} elem/scale)')
        data_size, times = self._get_boundary_params()
        limit_bytes = limit_gb * (1024 ** 3)
        dtype_label = self.combo_dtype.get()
        topo_label  = self.combo_topology.get()
        yv_main = self._boundary_conn(
            xv, limit_bytes, data_size, times, _N,
            bs=bs, mv_layout_factor=mv_layout_factor,
        )
        mid = len(xv) // 3

        def _draw_with_inline_label(x_coords, y_coords, x_pos, label_text, color, lw,
                                    linestyle, legend_label, text_color=None,
                                    fontsize=7, fontweight='bold',
                                    label_padding_px=2.0):
            """Draw curve as two segments, using rendered text width as the gap."""
            idx = int(np.argmin(np.abs(x_coords - x_pos)))
            i0 = max(idx - 2, 0)
            i1 = min(idx + 2, len(x_coords) - 1)
            p0 = ax.transData.transform((x_coords[i0], y_coords[i0]))
            p1 = ax.transData.transform((x_coords[i1], y_coords[i1]))
            angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))

            text_artist = ax.text(
                x_coords[idx],
                y_coords[idx],
                label_text,
                fontsize=fontsize,
                ha='center',
                va='center',
                color=text_color or color,
                fontweight=fontweight,
                rotation=angle,
                zorder=7,
            )
            try:
                renderer = ax.figure.canvas.get_renderer()
            except Exception:
                renderer = None
            if renderer is None:
                ax.figure.canvas.draw()
                renderer = ax.figure.canvas.get_renderer()

            original_rotation = text_artist.get_rotation()
            text_artist.set_rotation(0)
            bbox = text_artist.get_window_extent(renderer=renderer)
            text_artist.set_rotation(original_rotation)

            center = ax.transData.transform((x_coords[idx], y_coords[idx]))
            direction = p1 - p0
            direction_len = float(np.hypot(direction[0], direction[1]))
            if direction_len <= 1e-9:
                direction = np.array([1.0, 0.0])
            else:
                direction = direction / direction_len

            half_gap_px = max(float(bbox.width) / 2.0 + float(label_padding_px), 1.0)
            inv = ax.transData.inverted()
            left_data = inv.transform(center - direction * half_gap_px)
            right_data = inv.transform(center + direction * half_gap_px)
            x_lo, x_hi = sorted((float(left_data[0]), float(right_data[0])))

            x_min = float(np.nanmin(x_coords))
            x_max = float(np.nanmax(x_coords))
            x_lo = float(np.clip(x_lo, x_min, x_max))
            x_hi = float(np.clip(x_hi, x_min, x_max))
            if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_hi <= x_lo:
                x_range = x_coords[-1] - x_coords[0] if x_coords[-1] > x_coords[0] else 1.0
                half_w = x_range * 0.005
                x_lo = x_coords[idx] - half_w
                x_hi = x_coords[idx] + half_w
            legend_token = legend_label if legend_label else '_nolegend_'

            # Left segment — carries the legend entry
            mask_l = x_coords < x_lo
            if mask_l.any():
                y_at_lo = np.interp(x_lo, x_coords, y_coords)
                xl = np.append(x_coords[mask_l], x_lo)
                yl = np.append(y_coords[mask_l], y_at_lo)
                ax.plot(xl, yl, color=color, lw=lw, linestyle=linestyle, label=legend_token)
            else:
                ax.plot([], [], color=color, lw=lw, linestyle=linestyle, label=legend_token)

            # Right segment — no legend label (same visual line)
            mask_r = x_coords > x_hi
            if mask_r.any():
                y_at_hi = np.interp(x_hi, x_coords, y_coords)
                xr = np.insert(x_coords[mask_r], 0, x_hi)
                yr = np.insert(y_coords[mask_r], 0, y_at_hi)
                ax.plot(xr, yr, color=color, lw=lw, linestyle=linestyle)

        mode_tag = self.mode.get().upper()
        if self.mode.get() == 'mm':
            _bs = bs if bs is not None else self._get_batch_size()
            main_legend = f'VRAM = {limit_gb:.2f} GB  ({dtype_label}, {topo_label}, bs={_bs}) [{mode_tag}]'
        else:
            main_legend = (
                f'VRAM = {limit_gb:.2f} GB  ({dtype_label}, {topo_label}, '
                f'{mv_layout}, layout×{mv_layout_factor}) [{mode_tag}]'
            )
        _draw_with_inline_label(xv, yv_main, xv[mid],
                                f'{limit_gb:.2f} GB', 'red', 2, 'solid', main_legend)
        for c in self.extra_curves:
            e_data_size   = self._dtype_size(c['dtype'])
            e_times       = self._topology_times(c['topology'])
            e_limit_bytes = c['limit_gb'] * (1024 ** 3)
            yv_extra      = self._boundary_conn(
                xv, e_limit_bytes, e_data_size, e_times, _N,
                bs=bs, mv_layout_factor=mv_layout_factor,
            )
            _draw_with_inline_label(xv, yv_extra, xv[mid],
                                    f"{c['limit_gb']:.2f} GB", c['color'], 2, 'dashed',
                                    c['label'])
        for line in sparsity_lines or []:
            value = line.get('value')
            if value is None:
                value = self._sparsity_from_display_angle(
                    line.get('target_angle', -45.0), _N, ax, x0, x1, y0, y1)
            yv_sparse = value * xv * _N
            label_x = line.get('label_x')
            if label_x is None:
                label_x = self._sparsity_label_x(
                    value, _N, x0, x1, y0, line.get('label_frac', 0.5))
            inline_text = line.get('inline_text') or self._format_sparsity_label(value)
            _draw_with_inline_label(
                xv,
                yv_sparse,
                label_x,
                inline_text,
                line.get('color', '#546E7A'),
                line.get('lw', 1.5),
                line.get('linestyle', ':'),
                line.get('legend_label'),
                text_color='black',
                fontsize=7,
                fontweight=line.get('fontweight', 'normal'),
                label_padding_px=line.get('label_padding_px', 1.0),
            )
        ax.legend(loc='upper right', fontsize=8)

    def _draw_contours_and_labels(self, ax, grid_x, grid_y, grid_z_masked, z_pts):
        """Draw contour lines and per-band μ / deviation annotations.

        z_pts : 1-D array used for band statistics.
                Pass original scatter values for raw plots,
                or finite grid values for speedup plot.
        """
        try:
            n_contours = max(0, int(self.entry_contours.get()))
        except ValueError:
            n_contours = 0
        if n_contours == 0:
            return

        z_finite = grid_z_masked[np.isfinite(grid_z_masked)]
        if len(z_finite) <= 1:
            return

        z_lo, z_hi = float(z_finite.min()), float(z_finite.max())
        if z_lo >= z_hi:
            return

        levels = np.unique(np.linspace(z_lo, z_hi, n_contours + 2)[1:-1])
        if len(levels) == 0:
            return

        cs = ax.contour(grid_x, grid_y, grid_z_masked,
                        levels=levels, colors='black', linewidths=1.0, alpha=0.75)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.3g")

    def _draw_custom_contours(self, ax, grid_x, grid_y, grid_z_masked):
        raw = self.entry_custom_lines.get().strip()
        if not raw:
            return
        try:
            levels = sorted({float(v.strip()) for v in raw.split(',') if v.strip()})
        except ValueError:
            return
        if not levels:
            return
        z_finite = grid_z_masked[np.isfinite(grid_z_masked)]
        if len(z_finite) == 0:
            return
        z_lo, z_hi = float(z_finite.min()), float(z_finite.max())
        levels = [lv for lv in levels if z_lo <= lv <= z_hi]
        if not levels:
            return
        cs = ax.contour(grid_x, grid_y, grid_z_masked,
                        levels=levels, colors='black',
                        linewidths=1.4, linestyles='dashed', alpha=0.9, zorder=5)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.3g")

    def _setup_scatter_hover(self, tab, x, y, z, z_label):
        canvas, ax = tab['canvas'], tab['ax']
        sc = tab['scatter']
        annot = ax.annotate(
            "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.92),
            arrowprops=dict(arrowstyle="->", color='gray'),
            fontsize=8, visible=False, zorder=10,
        )
        xc, yc, zc = x.copy(), y.copy(), z.copy()

        def on_hover(event):
            if event.inaxes != ax:
                if annot.get_visible():
                    annot.set_visible(False)
                    canvas.draw_idle()
                return
            cont, ind = sc.contains(event)
            if cont:
                i = ind["ind"][0]
                annot.xy = (xc[i], yc[i])
                annot.set_text(
                    f"Scale:  {xc[i]:.4g}\n"
                    f"Conn:   {yc[i]:.0f}\n"
                    f"{z_label}:  {zc[i]:.4g}"
                )
                annot.set_visible(True)
            else:
                if annot.get_visible():
                    annot.set_visible(False)
            canvas.draw_idle()

        tab['hover_cid'] = canvas.mpl_connect("motion_notify_event", on_hover)

    def _setup_interp_hover(self, tab, grid_x, grid_y, grid_z, z_label):
        canvas, ax = tab['canvas'], tab['ax']
        annot = ax.annotate(
            "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.92),
            fontsize=8, visible=False, zorder=10,
        )
        gx_col = grid_x[:, 0].copy()
        gy_row = grid_y[0, :].copy()
        gz     = grid_z.copy()

        def on_hover(event):
            if event.inaxes != ax or event.xdata is None:
                if annot.get_visible():
                    annot.set_visible(False)
                    canvas.draw_idle()
                return
            ix  = int(np.argmin(np.abs(gx_col - event.xdata)))
            iy  = int(np.argmin(np.abs(gy_row - event.ydata)))
            val = gz[ix, iy]
            annot.xy = (event.xdata, event.ydata)
            if np.isnan(val):
                annot.set_text(
                    f"Scale:  {event.xdata:.4g}\n"
                    f"Conn:   {event.ydata:.0f}\n"
                    "[outside valid domain]"
                )
            else:
                annot.set_text(
                    f"Scale:  {event.xdata:.4g}\n"
                    f"Conn:   {event.ydata:.0f}\n"
                    f"{z_label}:  {val:.4g}"
                )
            annot.set_visible(True)
            canvas.draw_idle()

        tab['hover_cid'] = canvas.mpl_connect("motion_notify_event", on_hover)

    def _render_scatter(
        self,
        tab,
        x,
        y,
        z,
        _N,
        limit_gb,
        x_min,
        x_max,
        y_max,
        sparsity_lines,
        z_label,
        cmap_name,
        subtitle,
        bs=None,
        mv_layout='row_gather',
        mv_layout_factor=1,
    ):
        ax, canvas, title = tab['ax'], tab['canvas'], tab['title']

        sc = ax.scatter(x, y, c=z, cmap=cmap_name, s=100, edgecolors='black', zorder=3)
        tab['cbar']    = tab['fig'].colorbar(sc, ax=ax, label=z_label)
        tab['scatter'] = sc

        for i in range(len(x)):
            ax.text(x[i], y[i], f"{z[i]:.3g}", fontsize=7,
                    ha='center', va='bottom', zorder=4)

        self._draw_boundaries(
            ax, _N, limit_gb, x_min, x_max, y_max,
            bs=bs,
            mv_layout=mv_layout,
            mv_layout_factor=mv_layout_factor,
            sparsity_lines=sparsity_lines,
        )
        ax.set_title(f"{title}\n{subtitle}", fontsize=10, loc='left', pad=8)
        ax.set_xlabel(f"Scale  (N = {_N:,} elements per scale unit)")
        ax.set_ylabel("Connection Number  (synapses / neuron)")
        ax.grid(True, linestyle='--', alpha=0.5)

        self._setup_scatter_hover(tab, x, y, z, z_label)
        canvas.draw()

    def _render_interpolation(
        self,
        tab,
        x,
        y,
        z,
        _N,
        limit_gb,
        x_min,
        x_max,
        y_max,
        sparsity_lines,
        z_label,
        cmap_name,
        subtitle,
        bs=None,
        mv_layout='row_gather',
        mv_layout_factor=1,
    ):
        ax, canvas, title = tab['ax'], tab['canvas'], tab['title']

        if len(x) < 4:
            ax.text(0.5, 0.5, "less than 4",
                    ha='center', va='center', transform=ax.transAxes, fontsize=11)
            canvas.draw()
            return

        grid_x, grid_y = np.mgrid[0 : x_max * 1.1 : 200j, 0 : y_max * 1.2 : 200j]
        grid_z  = griddata((x, y), z, (grid_x, grid_y), method='linear')

        limit_bytes = limit_gb * (1024 ** 3)
        data_size, times = self._get_boundary_params()
        valid = (
            (grid_y <= grid_x * _N)
            & (
                self._vram_bytes(
                    grid_y, grid_x, data_size, times, _N,
                    bs=bs, mv_layout_factor=mv_layout_factor,
                ) <= limit_bytes
            )
        )
        grid_z_masked = np.where(valid, grid_z, np.nan)

        im          = ax.pcolormesh(grid_x, grid_y, grid_z_masked, shading='auto', cmap=cmap_name)
        tab['cbar'] = tab['fig'].colorbar(im, ax=ax, label=z_label)

        self._draw_contours_and_labels(ax, grid_x, grid_y, grid_z_masked, z_pts=z)
        self._draw_custom_contours(ax, grid_x, grid_y, grid_z_masked)
        self._draw_boundaries(
            ax, _N, limit_gb, x_min, x_max, y_max,
            bs=bs,
            mv_layout=mv_layout,
            mv_layout_factor=mv_layout_factor,
            sparsity_lines=sparsity_lines,
        )
        ax.set_title(f"{title}\n{subtitle}", fontsize=10, loc='left', pad=8)
        ax.set_xlabel(f"Scale  (N = {_N:,} elements per scale unit)")
        ax.set_ylabel("Connection Number  (synapses / neuron)")
        ax.grid(True, linestyle='--', alpha=0.3)

        self._setup_interp_hover(tab, grid_x, grid_y, grid_z_masked, z_label)
        canvas.draw()

    def _render_speedup_interp(
        self,
        tab,
        grid_x,
        grid_y,
        grid_z_masked,
        _N,
        limit_gb,
        x_min,
        x_max,
        y_max,
        sparsity_lines,
        subtitle,
        bs=None,
        mv_layout='row_gather',
        mv_layout_factor=1,
    ):
        ax, canvas, title = tab['ax'], tab['canvas'], tab['title']
        z_label = "Speedup  (Baseline / Target)"

        cmap_sp = LinearSegmentedColormap.from_list(
            'speedup_bwy', ['#FFAA00', 'white', '#1565C0']
        )
        z_finite = grid_z_masked[np.isfinite(grid_z_masked)]
        if self.var_auto_speedup.get():
            if len(z_finite) > 0:
                yellow_min = min(float(z_finite.min()), 0.5)
                blue_max   = max(float(z_finite.max()), 1.5)
            else:
                yellow_min, blue_max = 0.5, 1.5
        else:
            try:
                yellow_min = float(self.entry_yellow_depth.get())
                blue_max   = float(self.entry_blue_depth.get())
            except ValueError:
                yellow_min, blue_max = 0.5, 2.0
        yellow_min = min(yellow_min, 0.9999)   # 必须 < 1.0
        blue_max   = max(blue_max,   1.0001)   # 必须 > 1.0
        if yellow_min >= blue_max:
            yellow_min, blue_max = 0.5, 2.0

        grid_sp_clipped = np.where(
            np.isfinite(grid_z_masked),
            np.clip(grid_z_masked, yellow_min, blue_max),
            np.nan,
        )
        norm = TwoSlopeNorm(vmin=yellow_min, vcenter=1.0, vmax=blue_max)
        im   = ax.pcolormesh(grid_x, grid_y, grid_sp_clipped,
                             shading='auto', cmap=cmap_sp, norm=norm)
        n_side   = 4
        y_ticks  = np.linspace(yellow_min, 1.0, n_side + 1)[:-1]
        b_ticks  = np.linspace(1.0, blue_max, n_side + 1)
        cb_ticks = np.unique(np.round(np.concatenate([y_ticks, b_ticks]), 6))
        cbar     = tab['fig'].colorbar(im, ax=ax, label=z_label, ticks=cb_ticks)
        cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3g'))
        tab['cbar'] = cbar

        z_finite = grid_z_masked[np.isfinite(grid_z_masked)]
        self._draw_contours_and_labels(ax, grid_x, grid_y, grid_z_masked, z_pts=z_finite)
        self._draw_custom_contours(ax, grid_x, grid_y, grid_z_masked)

        self._draw_boundaries(
            ax, _N, limit_gb, x_min, x_max, y_max,
            bs=bs,
            mv_layout=mv_layout,
            mv_layout_factor=mv_layout_factor,
            sparsity_lines=sparsity_lines,
        )
        ax.set_title(f"{title}\n{subtitle}", fontsize=10, loc='left', pad=8)
        ax.set_xlabel(f"Scale  (N = {_N:,} elements per scale unit)")
        ax.set_ylabel("Connection Number  (synapses / neuron)")
        ax.grid(True, linestyle='--', alpha=0.3)

        self._setup_interp_hover(tab, grid_x, grid_y, grid_z_masked, z_label)
        canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = PerformanceBoundaryApp(root)
    root.mainloop()
