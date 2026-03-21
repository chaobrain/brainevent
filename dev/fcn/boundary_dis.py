import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.interpolate import griddata


class PerformanceBoundaryApp:
    _AXIS_COLS = frozenset({'scale', 'conn_num', 'elapsed_s'})

    def __init__(self, root):
        self.root = root
        self.root.title("Sparse Matrix Performance Boundary & Speedup Analyzer")
        self.root.geometry("1440x960")
        self.df: Optional[pd.DataFrame] = None
        self.comboboxes: dict = {}   # dynamic filter comboboxes
        self._setup_ui()

    # ─────────────────────────────────────────────────────────────────────────
    def _setup_ui(self):
        # ── 全局控制栏 ────────────────────────────────────────────────────────
        ctrl = ttk.LabelFrame(self.root, text="Global Settings", padding=8)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=5, pady=4)

        ttk.Button(ctrl, text="Load CSV", command=self.load_data).pack(side=tk.LEFT, padx=4)

        ttk.Label(ctrl, text="_N (elem/scale):").pack(side=tk.LEFT, padx=(10, 2))
        self.entry_n = ttk.Entry(ctrl, width=9)
        self.entry_n.insert(0, "4000")
        self.entry_n.pack(side=tk.LEFT)

        ttk.Label(ctrl, text="VRAM (GB):").pack(side=tk.LEFT, padx=(10, 2))
        self.entry_limit = ttk.Entry(ctrl, width=7)
        self.entry_limit.insert(0, "16")
        self.entry_limit.pack(side=tk.LEFT)

        ttk.Label(ctrl, text="Contour Lines:").pack(side=tk.LEFT, padx=(10, 2))
        self.entry_contours = ttk.Entry(ctrl, width=5)
        self.entry_contours.insert(0, "5")
        self.entry_contours.pack(side=tk.LEFT)

        ttk.Label(ctrl, text="Custom Lines:").pack(side=tk.LEFT, padx=(10, 2))
        self.entry_custom_lines = ttk.Entry(ctrl, width=16)
        self.entry_custom_lines.insert(0, "")
        self.entry_custom_lines.pack(side=tk.LEFT)
        ttk.Label(ctrl, text="(逗号分隔值，仅插值/加速比图)",
                  foreground='gray').pack(side=tk.LEFT, padx=(2, 0))

        ttk.Button(ctrl, text="Update Plots", command=self.update_plots).pack(side=tk.LEFT, padx=12)

        ttk.Separator(ctrl, orient='vertical').pack(side=tk.LEFT, fill='y', padx=8, pady=2)

        ttk.Button(ctrl, text="Export Image", command=self.export_image).pack(side=tk.LEFT, padx=4)
        ttk.Label(ctrl, text="DPI:").pack(side=tk.LEFT, padx=(6, 2))
        self.entry_dpi = ttk.Entry(ctrl, width=5)
        self.entry_dpi.insert(0, "150")
        self.entry_dpi.pack(side=tk.LEFT)

        # ── 对比配置区（Compare Field → Target / Baseline）────────────────────
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

        # ── 加速比配色设置栏 ────────────────────────────────────────────────────
        sp_bar = ttk.LabelFrame(self.root, text="Speedup Color Settings", padding=8)
        sp_bar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=4)

        ttk.Label(sp_bar, text="Yellow Deepest (speedup ≤):").pack(side=tk.LEFT, padx=(0, 2))
        self.entry_yellow_depth = ttk.Entry(sp_bar, width=7)
        self.entry_yellow_depth.insert(0, "0.5")
        self.entry_yellow_depth.pack(side=tk.LEFT)

        ttk.Label(sp_bar, text="Blue Deepest (speedup ≥):").pack(side=tk.LEFT, padx=(16, 2))
        self.entry_blue_depth = ttk.Entry(sp_bar, width=7)
        self.entry_blue_depth.insert(0, "2.0")
        self.entry_blue_depth.pack(side=tk.LEFT)

        ttk.Label(sp_bar,
                  text="  加速比 = 1 → 白色；< Yellow Deepest 值 → 最深黄色；> Blue Deepest 值 → 最深蓝色",
                  foreground='gray').pack(side=tk.LEFT, padx=(12, 0))

        # ── 动态数据过滤栏 ─────────────────────────────────────────────────────
        self.flt_outer = ttk.LabelFrame(self.root, text="Data Slicing Filters", padding=8)
        self.flt_outer.pack(side=tk.TOP, fill=tk.X, padx=5, pady=4)
        self.filter_row = ttk.Frame(self.flt_outer)
        self.filter_row.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(self.filter_row, text="(加载 CSV 后自动填充过滤选项)",
                  foreground='gray').pack(side=tk.LEFT)

        # ── 图表容器 ──────────────────────────────────────────────────────────
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
            fig = Figure(figsize=(10, 7), dpi=100)
            ax = fig.add_subplot(111)
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            NavigationToolbar2Tk(canvas, frame)
            self.tabs[name] = {
                'fig': fig, 'ax': ax, 'canvas': canvas, 'title': title,
                'cbar': None, 'hover_cid': None, 'scatter': None,
            }

    # ── 对比字段变更响应 ──────────────────────────────────────────────────────
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

    # ── 重建动态过滤栏 ────────────────────────────────────────────────────────
    def _rebuild_filter_row(self):
        for widget in self.filter_row.winfo_children():
            widget.destroy()
        self.comboboxes = {}
        if self.df is None:
            ttk.Label(self.filter_row, text="(加载 CSV 后自动填充过滤选项)",
                      foreground='gray').pack(side=tk.LEFT)
            return
        compare_field = self.combo_compare_field.get()
        exclude = self._AXIS_COLS | ({compare_field} if compare_field else set())
        filter_cols = [c for c in self.df.columns if c not in exclude]
        if not filter_cols:
            ttk.Label(self.filter_row, text="(无额外过滤列)",
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

    # ── 导出图片 ──────────────────────────────────────────────────────────────
    def export_image(self):
        idx = self.notebook.index(self.notebook.select())
        tab = list(self.tabs.values())[idx]
        try:
            dpi = int(self.entry_dpi.get())
            if dpi <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "DPI 必须为正整数。")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"), ("All", "*.*")],
        )
        if path:
            tab['fig'].savefig(path, dpi=dpi, bbox_inches='tight')
            messagebox.showinfo("Exported", f"已保存: {path}  ({dpi} DPI)")

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    def load_data(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if not path:
            return
        try:
            self.df = pd.read_csv(path)
            required = {'scale', 'conn_num', 'elapsed_s'}
            missing = required - set(self.df.columns)
            if missing:
                raise ValueError(f"CSV 缺少列: {missing}")
            # 可比较字段 = 所有非轴列
            cmp_cols = [c for c in self.df.columns if c not in self._AXIS_COLS]
            self.combo_compare_field['values'] = cmp_cols
            if cmp_cols:
                self.combo_compare_field.current(0)
            self._on_compare_field_changed()
            messagebox.showinfo("Success", f"已加载 {len(self.df)} 行。")
            self.update_plots()
        except Exception as e:
            messagebox.showerror("Data Load Error", str(e))

    # ── 副标题生成 ────────────────────────────────────────────────────────────
    def _subtitle(self, include_baseline: bool = False) -> str:
        parts = [f"{c} = {cb.get()}"
                 for c, cb in self.comboboxes.items() if cb.get()]
        cmp_field = self.combo_compare_field.get()
        parts.append(f"Target [{cmp_field}]: {self.combo_target.get()}")
        if include_baseline:
            parts.append(f"Baseline [{cmp_field}]: {self.combo_baseline.get()}")
        return "  │  ".join(parts)

    # ── 刷新所有图表 ──────────────────────────────────────────────────────────
    def update_plots(self):
        if self.df is None or self.df.empty:
            return
        try:
            _N       = int(self.entry_n.get())
            limit_gb = float(self.entry_limit.get())
        except ValueError:
            messagebox.showerror("Input Error", "N 或 VRAM 限制输入无效。")
            return

        # ── 先移除旧 colorbar，再清空坐标轴（避免叠加）────────────────────────
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

        # ── 公共过滤 ──────────────────────────────────────────────────────────
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
            messagebox.showwarning("No Compare Field", "请先选择对比字段。")
            return

        df_t = df_f[df_f[cmp_field].astype(str) == str(target_val)]
        df_b = df_f[df_f[cmp_field].astype(str) == str(base_val)]

        if df_t.empty:
            messagebox.showwarning("No Data", "目标数据子集为空。")
            for t in self.tabs.values():
                t['canvas'].draw()
            return

        x_max = max(df_t['scale'].max(),
                    df_b['scale'].max() if not df_b.empty else 1, 1)
        x_min = max(min(df_t['scale'].min(),
                        df_b['scale'].min() if not df_b.empty else 1), 0.1)
        y_max = max(df_t['conn_num'].max(),
                    df_b['conn_num'].max() if not df_b.empty else 1, 1)

        xt, yt, zt = df_t['scale'].values, df_t['conn_num'].values, df_t['elapsed_s'].values
        sub_raw   = self._subtitle(False)
        kw_common = dict(_N=_N, limit_gb=limit_gb, x_min=x_min, x_max=x_max, y_max=y_max)

        self._render_scatter(self.tabs["Raw Scatter"], xt, yt, zt,
                             z_label="Elapsed Time (s)", cmap_name='Blues',
                             subtitle=sub_raw, **kw_common)
        self._render_interpolation(self.tabs["Raw Interp"], xt, yt, zt,
                                   z_label="Elapsed Time (s)", cmap_name='Blues',
                                   subtitle=sub_raw, **kw_common)

        # ── 基于插值网格计算加速比 ─────────────────────────────────────────────
        ax_sp = self.tabs["Speedup Interp"]['ax']
        if df_b.empty:
            ax_sp.text(0.5, 0.5, "Baseline 数据子集为空。",
                       ha='center', va='center', transform=ax_sp.transAxes, fontsize=11)
            self.tabs["Speedup Interp"]['canvas'].draw()
        elif len(xt) < 4 or len(df_b) < 4:
            ax_sp.text(0.5, 0.5, "数据点不足 4 个，无法插值。",
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
            valid = (grid_y <= grid_x * _N) & (grid_y * grid_x * 8 * _N <= limit_bytes)

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

    # ── 边界线 ────────────────────────────────────────────────────────────────
    def _draw_boundaries(self, ax, _N, limit_gb, x_min, x_max):
        xv = np.linspace(max(0.01, x_min * 0.9), x_max * 1.25, 600)
        ax.plot(xv, xv * _N,
                color='gold', lw=2, label=f'conn = scale × N  (N={_N:,} elem/scale)')
        limit_bytes = limit_gb * (1024 ** 3)
        ax.plot(xv, limit_bytes / (8 * xv * _N),
                color='red', lw=2, label=f'VRAM = {limit_gb} GB  (float32)')
        ax.legend(loc='upper right', fontsize=8)

    # ── 等高线 + 区间统计标注（通用）────────────────────────────────────────
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

        bounds = np.concatenate([[-np.inf], levels, [np.inf]])
        for i in range(len(bounds) - 1):
            lo, hi   = bounds[i], bounds[i + 1]
            last_seg = (i == len(bounds) - 2)
            pt_mask  = (z_pts >= lo) if last_seg else ((z_pts >= lo) & (z_pts < hi))
            pts      = z_pts[pt_mask]
            if len(pts) == 0:
                continue

            mu      = float(np.mean(pts))
            dev_pos = float(np.max(pts)) - mu
            dev_neg = mu - float(np.min(pts))

            g_lo   = lo if not np.isinf(lo) else z_lo - 1e-9
            g_hi   = hi if not np.isinf(hi) else z_hi + 1e-9
            region = (np.isfinite(grid_z_masked) &
                      (grid_z_masked >= g_lo) & (grid_z_masked < g_hi))
            if not region.any():
                continue

            cx = float(np.mean(grid_x[region]))
            cy = float(np.mean(grid_y[region]))
            ax.text(
                cx, cy,
                f"μ = {mu:.3g}\n+{dev_pos:.2g} / −{dev_neg:.2g}",
                fontsize=7, ha='center', va='center', color='white', zorder=6,
                bbox=dict(boxstyle='round,pad=0.25', fc='black', alpha=0.55),
            )

    # ── 自定义虚线等值线（仅插值图 / 加速比图）────────────────────────────────
    def _draw_custom_contours(self, ax, grid_x, grid_y, grid_z_masked):
        """按用户输入的逗号分隔数值绘制黑色虚线等值线。"""
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

    # ── 鼠标悬停：散点图 ──────────────────────────────────────────────────────
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

    # ── 鼠标悬停：插值图 ──────────────────────────────────────────────────────
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

    # ── 渲染：散点图 ──────────────────────────────────────────────────────────
    def _render_scatter(self, tab, x, y, z, _N, limit_gb, x_min, x_max, y_max,
                        z_label, cmap_name, subtitle):
        ax, canvas, title = tab['ax'], tab['canvas'], tab['title']

        sc = ax.scatter(x, y, c=z, cmap=cmap_name, s=100, edgecolors='black', zorder=3)
        tab['cbar']    = tab['fig'].colorbar(sc, ax=ax, label=z_label)
        tab['scatter'] = sc

        for i in range(len(x)):
            ax.text(x[i], y[i], f"{z[i]:.3g}", fontsize=7,
                    ha='center', va='bottom', zorder=4)

        self._draw_boundaries(ax, _N, limit_gb, x_min, x_max)
        ax.set_title(f"{title}\n{subtitle}", fontsize=10, loc='left', pad=8)
        ax.set_xlabel(f"Scale  (N = {_N:,} elements per scale unit)")
        ax.set_ylabel("Connection Number  (synapses / neuron)")
        ax.set_ylim(max(y_max * 1.2, 100), 0)
        ax.set_xlim(0, x_max * 1.1)
        ax.grid(True, linestyle='--', alpha=0.5)

        self._setup_scatter_hover(tab, x, y, z, z_label)
        canvas.draw()

    # ── 渲染：插值图（原始时间）──────────────────────────────────────────────
    def _render_interpolation(self, tab, x, y, z, _N, limit_gb, x_min, x_max, y_max,
                              z_label, cmap_name, subtitle):
        ax, canvas, title = tab['ax'], tab['canvas'], tab['title']

        if len(x) < 4:
            ax.text(0.5, 0.5, "数据点不足 4 个，无法插值。",
                    ha='center', va='center', transform=ax.transAxes, fontsize=11)
            canvas.draw()
            return

        grid_x, grid_y = np.mgrid[0 : x_max * 1.1 : 200j, 0 : y_max * 1.2 : 200j]
        grid_z  = griddata((x, y), z, (grid_x, grid_y), method='linear')

        limit_bytes = limit_gb * (1024 ** 3)
        valid = (grid_y <= grid_x * _N) & (grid_y * grid_x * 8 * _N <= limit_bytes)
        grid_z_masked = np.where(valid, grid_z, np.nan)

        im          = ax.pcolormesh(grid_x, grid_y, grid_z_masked, shading='auto', cmap=cmap_name)
        tab['cbar'] = tab['fig'].colorbar(im, ax=ax, label=z_label)

        self._draw_contours_and_labels(ax, grid_x, grid_y, grid_z_masked, z_pts=z)
        self._draw_custom_contours(ax, grid_x, grid_y, grid_z_masked)
        self._draw_boundaries(ax, _N, limit_gb, x_min, x_max)
        ax.set_title(f"{title}\n{subtitle}", fontsize=10, loc='left', pad=8)
        ax.set_xlabel(f"Scale  (N = {_N:,} elements per scale unit)")
        ax.set_ylabel("Connection Number  (synapses / neuron)")
        ax.set_ylim(max(y_max * 1.2, 100), 0)
        ax.set_xlim(0, x_max * 1.1)
        ax.grid(True, linestyle='--', alpha=0.3)

        self._setup_interp_hover(tab, grid_x, grid_y, grid_z_masked, z_label)
        canvas.draw()

    # ── 渲染：加速比插值图（基于预计算网格）─────────────────────────────────
    def _render_speedup_interp(self, tab, grid_x, grid_y, grid_z_masked,
                               _N, limit_gb, x_min, x_max, y_max, subtitle):
        ax, canvas, title = tab['ax'], tab['canvas'], tab['title']
        z_label = "Speedup  (Baseline / Target)"

        # ── 自定义配色：深黄 ← 白（=1） → 深蓝 ──────────────────────────────
        cmap_sp = LinearSegmentedColormap.from_list(
            'speedup_bwy', ['#FFAA00', 'white', '#1565C0']
        )
        try:
            yellow_min = float(self.entry_yellow_depth.get())
            blue_max   = float(self.entry_blue_depth.get())
        except ValueError:
            yellow_min, blue_max = 0.5, 2.0
        yellow_min = min(yellow_min, 0.9999)   # 必须 < 1.0
        blue_max   = max(blue_max,   1.0001)   # 必须 > 1.0
        if yellow_min >= blue_max:
            yellow_min, blue_max = 0.5, 2.0

        # 超出范围截断 → 显示最深颜色
        grid_sp_clipped = np.where(
            np.isfinite(grid_z_masked),
            np.clip(grid_z_masked, yellow_min, blue_max),
            np.nan,
        )
        norm        = TwoSlopeNorm(vmin=yellow_min, vcenter=1.0, vmax=blue_max)
        im          = ax.pcolormesh(grid_x, grid_y, grid_sp_clipped,
                                    shading='auto', cmap=cmap_sp, norm=norm)
        tab['cbar'] = tab['fig'].colorbar(im, ax=ax, label=z_label)

        # 使用网格内有限值作为等高线统计参考
        z_finite = grid_z_masked[np.isfinite(grid_z_masked)]
        self._draw_contours_and_labels(ax, grid_x, grid_y, grid_z_masked, z_pts=z_finite)
        self._draw_custom_contours(ax, grid_x, grid_y, grid_z_masked)

        self._draw_boundaries(ax, _N, limit_gb, x_min, x_max)
        ax.set_title(f"{title}\n{subtitle}", fontsize=10, loc='left', pad=8)
        ax.set_xlabel(f"Scale  (N = {_N:,} elements per scale unit)")
        ax.set_ylabel("Connection Number  (synapses / neuron)")
        ax.set_ylim(max(y_max * 1.2, 100), 0)
        ax.set_xlim(0, x_max * 1.1)
        ax.grid(True, linestyle='--', alpha=0.3)

        self._setup_interp_hover(tab, grid_x, grid_y, grid_z_masked, z_label)
        canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = PerformanceBoundaryApp(root)
    root.mainloop()