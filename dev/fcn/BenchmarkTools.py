
import os
import re
import jax
from pathlib import Path

MIN_GENERATED_SCALE = 20
MIN_GENERATED_CONN = 20
FIXED_GENERATED_BATCHES = (16, 32, 64, 128, 256)


def _clamped_sampling_min(max_val: int, preferred_min: int) -> int:
    if max_val < 1:
        raise ValueError(f"max_val must be >= 1, got {max_val}")
    return min(max_val, preferred_min)


def _fixed_batch_candidates(batch_max: int) -> list[int]:
    if batch_max < 1:
        raise ValueError(f"batch_max must be >= 1, got {batch_max}")
    batches = [b for b in FIXED_GENERATED_BATCHES if b <= batch_max]
    return batches if batches else [batch_max]


class CSV_record():


    @staticmethod
    def _extract_value(param):
 
 
        if hasattr(param, '__class__') and param.__class__.__name__ == 'Quantity':
            return param.magnitude
        return param
    
    def __init__(
        self,
        CSV_name: str,
        operator: str,
        testing_type: str,
        duration: float,
        suffix: str = '',
        output_dir: str | None = None,
        append: bool = False,
        conn: int | None = None,
        conn: int | None = None,
    ) -> None:
        self.name = CSV_name
        self.suffix = suffix
        self.conn = conn
        self.operator_name = operator
        self.operator_name = operator
        # Handle brainunit Quantity objects
        self.duration = self._extract_value(duration)

        # default common fields
        if 'mv' in self.operator_name:   
            self.fieldnames: list[str] = [
                'operator', 'data_type', 'backend', 'synaptic_type', 'scale', 'conn_num',
                'elapsed_s', 'firing_rate', 'duration', 'homo'
            ]
            self.operator_type = 'mv'
        elif 'mm' in self.operator_name:
            self.fieldnames: list[str] = [
                'operator', 'data_type', 'backend', 'synaptic_type', 'scale', 'conn_num',
                'batch_size', 'elapsed_s', 'firing_rate', 'duration', 'homo'
            ]
            self.operator_type = 'mm'
        else:
            self.fieldnames: list[str] = [
                'operator', 'data_type', 'backend', 'synaptic_type', 'scale', 'conn_num',
                'elapsed_s', 'firing_rate', 'duration', 'homo'
            ]
            self.operator_type = 'unknown'
        
        if 'mv' in self.operator_name:   
            self.fieldnames: list[str] = [
                'operator', 'data_type', 'backend', 'synaptic_type', 'scale', 'conn_num',
                'elapsed_s', 'firing_rate', 'duration', 'homo'
            ]
            self.operator_type = 'mv'
        elif 'mm' in self.operator_name:
            self.fieldnames: list[str] = [
                'operator', 'data_type', 'backend', 'synaptic_type', 'scale', 'conn_num',
                'batch_size', 'elapsed_s', 'firing_rate', 'duration', 'homo'
            ]
            self.operator_type = 'mm'
        else:
            self.fieldnames: list[str] = [
                'operator', 'data_type', 'backend', 'synaptic_type', 'scale', 'conn_num',
                'elapsed_s', 'firing_rate', 'duration', 'homo'
            ]
            self.operator_type = 'unknown'
        
        self.rows: list[dict] = []


        self.testing_type = testing_type  
        if testing_type == 'COBA': self.testing_type = 'coba' # coba or benchmark

        # output dir default: same folder as this file /results
        try:
            self.base = Path(__file__).resolve().parent
        except NameError:
            self.base = Path.cwd()

        self.output_dir = Path(output_dir) if output_dir else self.base / 'results'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.append = append
        self._tags: dict = {}

        self.width = 70

    def _write_csv(self, file_name: str, rows: list[dict], fieldnames: list[str], mode: str = 'w', silent: bool = False) -> None:
    def _write_csv(self, file_name: str, rows: list[dict], fieldnames: list[str], mode: str = 'w', silent: bool = False) -> None:
        import csv
        from pathlib import Path

        file_path = Path(self.output_dir) / f'{file_name}.csv'
        write_header = True
        effective_fieldnames = list(fieldnames)

        if mode == 'a' and file_path.exists():
            write_header = False
            # Read existing fieldnames to maintain schema consistency
            with file_path.open('r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                existing_fields = next(reader, [])
            if existing_fields:
                # Union: existing fields first, then any new fields not yet present
                merged = list(existing_fields)
                for fn in fieldnames:
                    if fn not in merged:
                        merged.append(fn)
                effective_fieldnames = merged

        with file_path.open(mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=effective_fieldnames, restval='default')
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

        if not silent:
            print(f"result has been saved: {file_path}")
        if not silent:
            print(f"result has been saved: {file_path}")

    def add_tag(self, tag_name: str, tag_value) -> None:
        """Set a persistent tag that will be automatically included in all subsequent rows.

        Call this before ``single_COBA_data_add`` (or ``add_row``) to attach
        an extra labeled field to every row recorded afterward.
        ``fieldnames`` is updated immediately so the column appears in the CSV.
        """
        self._tags[tag_name] = tag_value
        if tag_name not in self.fieldnames:
            self.fieldnames.append(tag_name)

    def add_row(self, row: dict) -> None:
        """Add a generic row (dict). New keys will be added to fieldnames.

        Active tags (set via ``add_tag``) are merged into the row automatically;
        explicit values in *row* take precedence over tag values.
        """
        # Merge active tags first, then let row values override
        merged_row = dict(self._tags)
        merged_row.update(row)
        for key in merged_row.keys():
            if key not in self.fieldnames:
                self.fieldnames.append(key)
        self.rows.append(merged_row)

    def single_COBA_data_add(self, operator: str,
                             data_type: str,
                             backend: str,
                             synaptic_type: str,
                             conn_num: int,
                             scale: int,
                             elapsed_s: float,
                             firing_rate: float,
                             duration: float,
                             homo: str = 'default',
                             batch_size: int = 1,
                             batch_size: int = 1,
                             **kwargs):
        """Backwards-compatible helper used by existing benchmarks.
        
        Automatically extracts numeric values from brainunit Quantity objects.
        """
        if self.operator_type == 'mv':   
            row = {
            'operator': self.operator_name,
            'data_type': data_type,
            'backend': backend,
            'testing_type': self.testing_type,
            'synaptic_type': synaptic_type,
            'scale': scale,
            'conn_num': conn_num,
            'elapsed_s': self._extract_value(elapsed_s),
            'firing_rate': self._extract_value(firing_rate),
            'duration': self._extract_value(duration),
            'homo': homo,
        }
        elif self.operator_type == 'mm':
            row = {
                'operator': self.operator_name,
                'data_type': data_type,
                'backend': backend,
                'testing_type': self.testing_type,
                'synaptic_type': synaptic_type,
                'scale': scale,
                'conn_num': conn_num,
                'batch_size': batch_size,
                'elapsed_s': self._extract_value(elapsed_s),
                'firing_rate': self._extract_value(firing_rate),
                'duration': self._extract_value(duration),
                'homo': homo,
            }
        else:
            row = {
                'operator': self.operator_name,
                'data_type': data_type,
                'backend': backend,
                'testing_type': self.testing_type,
                'synaptic_type': synaptic_type,
                'scale': scale,
                'conn_num': conn_num,
                'elapsed_s': self._extract_value(elapsed_s),
                'firing_rate': self._extract_value(firing_rate),
                'duration': self._extract_value(duration),
                'homo': homo,
            }
        if self.operator_type == 'mv':   
            row = {
            'operator': self.operator_name,
            'data_type': data_type,
            'backend': backend,
            'testing_type': self.testing_type,
            'synaptic_type': synaptic_type,
            'scale': scale,
            'conn_num': conn_num,
            'elapsed_s': self._extract_value(elapsed_s),
            'firing_rate': self._extract_value(firing_rate),
            'duration': self._extract_value(duration),
            'homo': homo,
        }
        elif self.operator_type == 'mm':
            row = {
                'operator': self.operator_name,
                'data_type': data_type,
                'backend': backend,
                'testing_type': self.testing_type,
                'synaptic_type': synaptic_type,
                'scale': scale,
                'conn_num': conn_num,
                'batch_size': batch_size,
                'elapsed_s': self._extract_value(elapsed_s),
                'firing_rate': self._extract_value(firing_rate),
                'duration': self._extract_value(duration),
                'homo': homo,
            }
        else:
            row = {
                'operator': self.operator_name,
                'data_type': data_type,
                'backend': backend,
                'testing_type': self.testing_type,
                'synaptic_type': synaptic_type,
                'scale': scale,
                'conn_num': conn_num,
                'elapsed_s': self._extract_value(elapsed_s),
                'firing_rate': self._extract_value(firing_rate),
                'duration': self._extract_value(duration),
                'homo': homo,
            }

        # merge extras, also extract values from them
        for key, value in kwargs.items():
            row[key] = self._extract_value(value)

        self.add_row(row)



    def record_finish(self, dir:str = '', suffix: str = '', file_name: str | None = None) -> None:
        """Write accumulated rows to disk.

        - `suffix` will be used in auto-generated filename if provided.
        - `file_name` overrides auto filename.
        - When suffix is 'default' (or empty), results are **always appended** to
          ``{testing_type}_default.csv`` so that multiple operators / data-types
          are aggregated into one file for unified comparison.
        - For other files, automatically detects if the target file exists and
          appends if it does.
        """
        if not self.rows:
            return
        if dir != '':
        if not self.rows:
            return
        if dir != '':
            self.output_dir = self.base / dir
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        suf = suffix if suffix else self.suffix

        is_default = (suf == 'default' or not suf)

        if file_name:
            out_name = file_name
            # Check if file exists and determine mode
            file_path = self.output_dir / f'{file_name}.csv'
            mode = 'a' if file_path.exists() else 'w'
        elif is_default:
            # Aggregate everything into <testing_type>_default.csv (always append)
            out_name = f'{self.testing_type}_default'
            mode = 'a'
        else:
            out_name = f'{self.testing_type}_{self.operator_name}_{self.name}_{suf}'
            # Check if file exists and determine mode
            file_path = self.output_dir / f'{out_name}.csv'
            mode = 'a' if file_path.exists() else 'w'

        self._write_csv(out_name, self.rows, self.fieldnames, mode=mode)

    def flush_and_clear(self, file_name: str, dir: str = '') -> 'Path | None':
        """Immediately write buffered rows to disk and clear the buffer.

        Use this for incremental flushing after each completed test round
        instead of waiting for ``record_finish`` at the end.
        If there are no buffered rows this is a no-op.

        Parameters
        ----------
        file_name : str
            Output filename (without .csv extension).
        dir : str, optional
            Sub-directory under ``self.base`` to write into.
            If empty, uses the current ``self.output_dir``.
        """
        if not self.rows:
            return
        if dir:
            target_dir = self.base / dir
            target_dir.mkdir(parents=True, exist_ok=True)
        else:
            target_dir = self.output_dir
        file_path = target_dir / f'{file_name}.csv'
        mode = 'a' if file_path.exists() else 'w'
        _saved = self.output_dir
        self.output_dir = target_dir
        self._write_csv(file_name, self.rows, self.fieldnames, mode=mode, silent=True)
        self.output_dir = _saved
        self.rows = []
        return file_path

    def flush_and_clear(self, file_name: str, dir: str = '') -> 'Path | None':
        """Immediately write buffered rows to disk and clear the buffer.

        Use this for incremental flushing after each completed test round
        instead of waiting for ``record_finish`` at the end.
        If there are no buffered rows this is a no-op.

        Parameters
        ----------
        file_name : str
            Output filename (without .csv extension).
        dir : str, optional
            Sub-directory under ``self.base`` to write into.
            If empty, uses the current ``self.output_dir``.
        """
        if not self.rows:
            return
        if dir:
            target_dir = self.base / dir
            target_dir.mkdir(parents=True, exist_ok=True)
        else:
            target_dir = self.output_dir
        file_path = target_dir / f'{file_name}.csv'
        mode = 'a' if file_path.exists() else 'w'
        _saved = self.output_dir
        self.output_dir = target_dir
        self._write_csv(file_name, self.rows, self.fieldnames, mode=mode, silent=True)
        self.output_dir = _saved
        self.rows = []
        return file_path


    def print_header(self, *, operator: str, data_type: str, backend: str,
                     mode: str, conn_num: int | None = None, duration=None, duration_ms: float | None = None,
                     batch_size: int | None = None, **extra) -> None:
        """Print parameter condition header block.

        Accepts either ``duration`` (a numeric value or brainunit Quantity) or
        the legacy ``duration_ms`` float.  When both are supplied, ``duration``
        takes precedence.
        """
        if duration is not None:
            dur_ms_val = float(self._extract_value(duration))
        elif duration_ms is not None:
            dur_ms_val = float(duration_ms)
        else:
            dur_ms_val = float(self._extract_value(self.duration))

        print(f'\n{"=" * self.width}')
        print(f'  operator={self.operator_name} | data_type={data_type} | backend={backend}')
        parts: list[str] = [f'mode={mode:<8s}']
        if batch_size is not None:
            parts.append(f'batch_size={batch_size}')
        if conn_num is not None:
            parts.append(f'conn_num={conn_num}')
        parts.append(f'duration={dur_ms_val:.1f} ms')
        for k, v in extra.items():
            parts.append(f'{k}={v}')
        print('  ' + ' | '.join(parts))
        print(f'{"=" * self.width}')

    def print_table_header(self, show_conn: bool = False, show_batch: bool = False) -> None:
    def print_table_header(self, show_conn: bool = False, show_batch: bool = False) -> None:
        """Print column header for the standard result table."""
        if show_batch and show_conn:
            print(f'  {"Scale":>5s} | {"BatchSz":>7s} | {"ConnNum":>8s} | {"Neurons":>7s} | {"Elapsed (s)":>11s} | {"Rate (Hz)":>9s}')
            print(f'  {"-----":>5s}-+-{"-------":>7s}-+-{"--------":>8s}-+-{"-------":>7s}-+-{"----------":>11s}-+-{"------":>9s}')
        elif show_batch:
            print(f'  {"Scale":>5s} | {"BatchSz":>7s} | {"Neurons":>7s} | {"Elapsed (s)":>11s} | {"Rate (Hz)":>9s}')
            print(f'  {"-----":>5s}-+-{"-------":>7s}-+-{"-------":>7s}-+-{"----------":>11s}-+-{"------":>9s}')
        elif show_conn:
        if show_batch and show_conn:
            print(f'  {"Scale":>5s} | {"BatchSz":>7s} | {"ConnNum":>8s} | {"Neurons":>7s} | {"Elapsed (s)":>11s} | {"Rate (Hz)":>9s}')
            print(f'  {"-----":>5s}-+-{"-------":>7s}-+-{"--------":>8s}-+-{"-------":>7s}-+-{"----------":>11s}-+-{"------":>9s}')
        elif show_batch:
            print(f'  {"Scale":>5s} | {"BatchSz":>7s} | {"Neurons":>7s} | {"Elapsed (s)":>11s} | {"Rate (Hz)":>9s}')
            print(f'  {"-----":>5s}-+-{"-------":>7s}-+-{"-------":>7s}-+-{"----------":>11s}-+-{"------":>9s}')
        elif show_conn:
            print(f'  {"Scale":>5s} | {"ConnNum":>8s} | {"Neurons":>7s} | {"Elapsed (s)":>11s} | {"Rate (Hz)":>9s}')
            print(f'  {"-----":>5s}-+-{"--------":>8s}-+-{"-------":>7s}-+-{"----------":>11s}-+-{"------":>9s}')
        else:
            print(f'  {"Scale":>5s} | {"Neurons":>7s} | {"Elapsed (s)":>11s} | {"Rate (Hz)":>9s}')
            print(f'  {"-----":>5s}-+-{"-------":>7s}-+-{"----------":>11s}-+-{"------":>9s}')

    @staticmethod
    def print_row(scale: int, neurons: int, elapsed: float, rate: float, conn_num=None, batch_size=None) -> None:
    def print_row(scale: int, neurons: int, elapsed: float, rate: float, conn_num=None, batch_size=None) -> None:
        """Print one benchmark result row."""
        if batch_size is not None and conn_num is not None:
            print(f'  {scale:>5d} | {batch_size:>7d} | {conn_num:>8g} | {neurons:>7d} | {elapsed:>11.3f} | {float(rate):>9.2f}')
        elif batch_size is not None:
            print(f'  {scale:>5d} | {batch_size:>7d} | {neurons:>7d} | {elapsed:>11.3f} | {float(rate):>9.2f}')
        elif conn_num is not None:
        if batch_size is not None and conn_num is not None:
            print(f'  {scale:>5d} | {batch_size:>7d} | {conn_num:>8g} | {neurons:>7d} | {elapsed:>11.3f} | {float(rate):>9.2f}')
        elif batch_size is not None:
            print(f'  {scale:>5d} | {batch_size:>7d} | {neurons:>7d} | {elapsed:>11.3f} | {float(rate):>9.2f}')
        elif conn_num is not None:
            print(f'  {scale:>5d} | {conn_num:>8g} | {neurons:>7d} | {elapsed:>11.3f} | {float(rate):>9.2f}')
        else:
            print(f'  {scale:>5d} | {neurons:>7d} | {elapsed:>11.3f} | {float(rate):>9.2f}')

def dump_jax_ir(func, args=(), kwargs=None, prefix="dump"):

    if kwargs is None:
        kwargs = {}
        
    ansi_escape_pattern = re.compile(r'\x1b\[[0-9;]*m')
    
    jaxpr_path = os.path.abspath(f"{prefix}_jaxpr.txt")
    hlo_path = os.path.abspath(f"{prefix}_hlo.txt")

    print("Tracing JAXPR (Frontend Logic State)...")
    jaxpr_ir = jax.make_jaxpr(func)(*args, **kwargs)
    clean_jaxpr_str = ansi_escape_pattern.sub('', str(jaxpr_ir))
    
    with open(jaxpr_path, "w") as f:
        f.write(clean_jaxpr_str)
    print(f"[*] Clean JAXPR saved to: {jaxpr_path}")

    print("Lowering to HLO (XLA Physical Fusion State)...")
    lowered_executable = func.lower(*args, **kwargs)
    hlo_text = lowered_executable.as_text()
    
    with open(hlo_path, "w") as f:
        f.write(hlo_text)
    print(f"[*] HLO saved to: {hlo_path}")
    
    return jaxpr_path, hlo_path


class TestingParamsGenerator_mv():

    def __init__(self,
                 limit_GB: int,
                 _N: int,
                 scale_max: int = 2000,
                 conn_max: int = 4000,
                 sample_points:int = 50):
        
        self._limit_GB = limit_GB
        self._N = _N
        self._limit_bytes = self._limit_GB * (1024)**3
        self.scale_max = scale_max
        self.conn_max = conn_max
        self.sample_points = sample_points
    
    def is_valid_mv(self, scale, conn, homo, data_size):
        if homo:
            times = 1
        else:
            times = 2
        matrix_memory_bytes = conn * scale * self._N * data_size * times #1 if homo else 2
        return matrix_memory_bytes <= self._limit_bytes and conn <= self._N * scale

    def generate_params(
        self,
        dis_type: str = 'log',
        target_samples: int = 500,
        data_size: int = 4,
        homo: bool = True
    ) -> list:
        """
        Generates a list of valid (scale, conn_num) parameter states within VRAM limits.

class TestingParamsGenerator_mv():

    def __init__(self,
                 limit_GB: int,
                 _N: int,
                 scale_max: int = 2000,
                 conn_max: int = 4000,
                 sample_points:int = 50):
        
        self._limit_GB = limit_GB
        self._N = _N
        self._limit_bytes = self._limit_GB * (1024)**3
        self.scale_max = scale_max
        self.conn_max = conn_max
        self.sample_points = sample_points
    
    def is_valid_mv(self, scale, conn, homo, data_size):
        if homo:
            times = 1
        else:
            times = 2
        matrix_memory_bytes = conn * scale * self._N * data_size * times #1 if homo else 2
        return matrix_memory_bytes <= self._limit_bytes and conn <= self._N * scale

    def generate_params(
        self,
        dis_type: str = 'log',
        target_samples: int = 500,
        data_size: int = 4,
        homo: bool = True
    ) -> list:
        """
        Generates a list of valid (scale, conn_num) parameter states within VRAM limits.

        Boundary conditions
        -------------------
        - matrix_memory_bytes = conn_num * scale * _N * data_size * 2 <= limit_bytes
        - conn_num <= _N * scale
        Boundary conditions
        -------------------
        - matrix_memory_bytes = conn_num * scale * _N * data_size * 2 <= limit_bytes
        - conn_num <= _N * scale

        Parameters
        ----------
        dis_type : str
            Sampling strategy: 'uniform' (linear grid), 'log' (geometric grid),
            or 'monte_carlo' (random sampling).
        _N : int
            Number of neurons per scale unit.
        limit_gb : float
            VRAM limit in gigabytes.
        target_samples : int
            Approximate number of valid states to generate (actual count ≈ ±50).
        data_size : int
            Bytes per element (4 for float32/int32, 1 for bool/int8).
        scale_max : int
            Upper bound of scale search range.
        conn_max : int
            Upper bound of conn_num search range.
        Parameters
        ----------
        dis_type : str
            Sampling strategy: 'uniform' (linear grid), 'log' (geometric grid),
            or 'monte_carlo' (random sampling).
        _N : int
            Number of neurons per scale unit.
        limit_gb : float
            VRAM limit in gigabytes.
        target_samples : int
            Approximate number of valid states to generate (actual count ≈ ±50).
        data_size : int
            Bytes per element (4 for float32/int32, 1 for bool/int8).
        scale_max : int
            Upper bound of scale search range.
        conn_max : int
            Upper bound of conn_num search range.

        Returns
        -------
        list of (scale, conn_num) tuples, sorted by memory footprint.
        """
        import numpy as np
        Returns
        -------
        list of (scale, conn_num) tuples, sorted by memory footprint.
        """
        import numpy as np

        min_scale = _clamped_sampling_min(self.scale_max, MIN_GENERATED_SCALE)
        min_conn = _clamped_sampling_min(self.conn_max, MIN_GENERATED_CONN)

        if dis_type == 'monte_carlo':
            valid_states = set()
            while len(valid_states) < target_samples:
                s = int(np.random.uniform(min_scale, self.scale_max + 1))
                c = int(np.random.uniform(min_conn, self.conn_max + 1))
                if self.is_valid_mv(s, c, homo, data_size):
                    valid_states.add((s, None, c))
            sorted_states = sorted(list(valid_states), key=lambda state: state[0] * state[2])
            print(f"Generated {len(sorted_states)} valid parameter states under {self._limit_GB}GB boundary.")
            return sorted_states

        # For grid-based methods: the valid region is a curved hyperbolic area;
        # ~3x oversampling of grid points relative to target gives ~±50 accuracy after filtering.
        grid_res = int(np.sqrt(target_samples * 3))
        # For grid-based methods: the valid region is a curved hyperbolic area;
        # ~3x oversampling of grid points relative to target gives ~±50 accuracy after filtering.
        grid_res = int(np.sqrt(target_samples * 3))

        if dis_type == 'uniform':
            scales_raw = np.unique(np.linspace(min_scale, self.scale_max, num=grid_res, dtype=int))
            conn_nums_raw = np.unique(np.linspace(min_conn, self.conn_max, num=grid_res, dtype=int))
        elif dis_type == 'log':
            scales_raw = np.unique(np.geomspace(min_scale, self.scale_max, num=grid_res, dtype=int))
            conn_nums_raw = np.unique(np.geomspace(min_conn, self.conn_max, num=grid_res, dtype=int))
        else:
            raise ValueError(f"Unknown dis_type: '{dis_type}'. Choose from 'uniform', 'log', 'monte_carlo'.")

        valid_states = [
            (int(s), None, int(c))
            for s in scales_raw
            for c in conn_nums_raw
            if self.is_valid_mv(s, c, homo, data_size)
        ]
        valid_states.sort(key=lambda state: state[0] * state[2])
        print(f"Generated {len(valid_states)} valid parameter states under {self._limit_GB}GB boundary.")
        return valid_states
        valid_states = [
            (int(s), None, int(c))
            for s in scales_raw
            for c in conn_nums_raw
            if self.is_valid_mv(s, c, homo, data_size)
        ]
        valid_states.sort(key=lambda state: state[0] * state[2])
        print(f"Generated {len(valid_states)} valid parameter states under {self._limit_GB}GB boundary.")
        return valid_states

    def generate_boundary_params(
        self,
        homo_or_not: bool = True,
        _N: int = 4000,
        data_size: int = 4,
        sample_points: int = 10,
        include_dense_ref: bool = False,
    ):
        """Generate ``(conn, m)`` pairs near the GPU memory boundary.

        Returns a list of ``(conn, m)`` tuples where ``m = scale * _N``, such
        that the estimated memory usage stays just below *memory_limit* GiB.
        Integer truncation (``int()``) is used so that values always lean
        toward the **smaller** side to avoid overflow.

        Memory model
        -------------
        * **Sparse arrays only** (``include_dense_ref=False``):
        ``m * conn * data_size * times``
        where *times* = 1 (homo, indices only) or 2 (hetero, indices + weights).

        * **With dense reference** (``include_dense_ref=True``):
        ``m * conn * data_size * times  +  m² * data_size``
        Use this when the test constructs a full ``(m, m)`` dense matrix for
        correctness comparison.

        Parameters
        ----------
        memory_limit : float
            GPU memory budget in GiB.
        homo_or_not : bool
            True  → only indices are stored (1× sparse budget).
            False → indices + weights are stored (2× sparse budget).
        scale_max : int
            Upper bound for the scale factor *s* (``m = s * _N``).
        conn_max : int
            Maximum number of connections per row.
        _N : int
            Base neuron count per scale unit.
        data_size : int
            Bytes per element (4 for float32 / int32).
        sample_points : int
            Number of sample points to generate.
        include_dense_ref : bool
            If True, the budget also accounts for a dense ``(m, m)`` reference
            matrix of size ``m² * data_size``.
        """
        
        import math
        import numpy as np

        times = 1 if homo_or_not else 2
        min_scale = _clamped_sampling_min(self.scale_max, MIN_GENERATED_SCALE)
        min_conn = _clamped_sampling_min(self.conn_max, MIN_GENERATED_CONN)

        # K = budget expressed in units of (_N * data_size) bytes
        K = self._limit_bytes / (_N * data_size)

        # --- valid scale range ---------------------------------------------------
        if include_dense_ref:
            # For conn > 0 we need  K - s² * _N > 0  →  s < sqrt(K / _N)
            s_max = min(self.scale_max, int(math.sqrt(K / _N)))
        else:
            # conn = K / (s * times) >= 1  →  s <= K / times
            s_max = min(self.scale_max, int(K / times))

        # s_min: when *not* including the dense ref, large s → small conn,
        # which is fine; but very small s → conn > conn_max, wasting budget.
        if not include_dense_ref:
            s_min = max(min_scale, math.ceil(K / (times * self.conn_max)))
        else:
            s_min = min_scale

        if s_min > s_max:
            raise ValueError(
                f"No valid (conn, m) pairs: s_min={s_min} > s_max={s_max}. "
                f"Try increasing memory_limit or decreasing scale_max/conn_max."
            )

        s_samples = np.geomspace(s_min, s_max, sample_points)

        valid_pairs = []
        seen = set()

        for s_val in s_samples:
            s_int = max(min_scale, int(round(s_val)))
            s_int = min(s_int, s_max)  # clamp

            # Compute max conn (floor to stay below the boundary)
            if include_dense_ref:
                budget_for_sparse = K - s_int ** 2 * _N
                if budget_for_sparse <= 0:
                    continue
                c_int = int(budget_for_sparse / (s_int * times))
            else:
                c_int = int(K / (s_int * times))

            c_int = min(c_int, self.conn_max)

            m = s_int * _N

            if c_int >= min_conn and c_int <= m and s_int <= self.scale_max:
                pair = (s_int, c_int)
                if pair not in seen:
                    seen.add(pair)
                    valid_pairs.append(pair)

        return valid_pairs

    def make_simulation_params_probs(self, 
                                     conn_prob : list, 
                                     scale: list, 
                                     limit_memory:int = 24,
                                     data_size:int = 4,
                                     homo:bool = True):
        """Generate valid (scale, conn_num) pairs from probability & scale lists.

        Parameters
        ----------
        conn_prob : list of float
            Connection probabilities.
        scale : list of int
            Scale values to test.
        limit_memory : int, optional
            Override VRAM limit in GB (uses constructor default if None).
        data_size : int
            Bytes per element.
        homo : bool
            Homogeneous (1×) or heterogeneous (2×) memory multiplier.

        Returns
        -------
        list of (scale, conn_num) tuples, filtered by VRAM limit.
        """
        valid_pairs = []
        for s in scale:
            for prob in conn_prob:
                conn_number = int(prob * s * self._N)
                if conn_number < 1:
                    conn_number = 1
                if self.is_valid_mv(s, conn_number, homo, data_size):
                    single_point = (int(s), None, int(conn_number))
                    valid_pairs.append(single_point)
                else:
                    continue
        return valid_pairs

    def generate_coba_vram_sequence(self,
                                    vram_steps: list | None = None,
                                    sample_points: int = 5,
                                    homo: bool = True,
                                    data_size: int = 4):
        """Generate test parameters for progressive VRAM limit testing.

        For each VRAM level, uses :meth:`generate_boundary_params` to produce
        ``(scale, conn)`` boundary candidate pairs near that VRAM level.

        Parameters
        ----------
        vram_steps : list of int, optional
            VRAM levels in GB to test (default: 1..24).
        sample_points : int
            Number of sample points per VRAM level.
        homo : bool
            Homogeneous or heterogeneous memory model.
        data_size : int
            Bytes per element.

        Returns
        -------
        OrderedDict mapping vram_gb -> list of (scale, conn) tuples.
        """
        from collections import OrderedDict

        if vram_steps is None:
            vram_steps = list(range(1, 25))

        result = OrderedDict()
        for vram_gb in vram_steps:
            gen = TestingParamsGenerator_mv(
                limit_GB=vram_gb,
                _N=self._N,
                scale_max=self.scale_max,
                conn_max=self.conn_max,
            )
            try:
                pairs = gen.generate_boundary_params(
                    homo_or_not=homo,
                    _N=self._N,
                    data_size=data_size,
                    sample_points=sample_points,
                )
                if pairs:
                    result[vram_gb] = pairs
            except ValueError:
                continue

        print(f"Generated VRAM sequence: {len(result)} levels, "
              f"total {sum(len(v) for v in result.values())} parameter pairs.")
        return result


# Backward-compatible alias: external code uses ``BT.TestingParamsGenerator``
TestingParamsGenerator = TestingParamsGenerator_mv


class TestingParamsGenerator_mm:
    """Generate valid (scale, batch_size, conn_num) triples for mm benchmarks.

    Memory model
    -------------
    Total VRAM ≈ conn_num * m * data_size * times  +  batch_size * m * data_size
    where ``m = scale * _N`` and ``times = 1`` (homo) or ``2`` (hetero).

    The first term is the sparse connection index (and weight) storage;
    the second is the batched event / result matrices.
    """

    def __init__(
        self,
        limit_GB: int = 24,
        _N: int = 4000,
        scale_max: int = 200,
        conn_max: int = 4000,
        batch_max: int = 128,
    ):
        self._limit_GB = limit_GB
        self._N = _N
        self._limit_bytes = self._limit_GB * (1024) ** 3
        self.scale_max = scale_max
        self.conn_max = conn_max
        self.batch_max = batch_max

    def is_valid_mm(self, scale: int, batch_size: int, conn: int,
                 homo: bool, data_size: int = 4) -> bool:
        times = 1 if homo else 2
        size = scale * self._N
        mem = conn * size * times + batch_size * size + size * batch_size
        mem = mem * data_size
        return mem <= self._limit_bytes and conn <= size

    def generate_params(
        self,
        dis_type: str = 'uniform',
        target_samples: int = 500,
        data_size: int = 4,
        homo: bool = True,
    ) -> list:
        """Generate valid ``(scale, batch_size, conn_num)`` triples within VRAM limits.

        Parameters
        ----------
        dis_type : str
            ``'uniform'`` (linear grid), ``'log'`` (geometric grid),
            or ``'monte_carlo'`` (random sampling).
        target_samples : int
            Approximate number of valid triples to generate.
        data_size : int
            Bytes per element (4 for float32/int32).
        homo : bool
            Homogeneous (1×) or heterogeneous (2×) memory multiplier.

        Returns
        -------
        list of ``(scale, batch_size, conn_num)`` tuples, sorted by estimated
        memory footprint.
        """
        import numpy as np

        min_scale = _clamped_sampling_min(self.scale_max, MIN_GENERATED_SCALE)
        min_conn = _clamped_sampling_min(self.conn_max, MIN_GENERATED_CONN)
        batch_choices = np.asarray(_fixed_batch_candidates(self.batch_max), dtype=int)

        if dis_type == 'monte_carlo':
            valid = set()
            while len(valid) < target_samples:
                s = int(np.random.uniform(min_scale, self.scale_max + 1))
                b = int(np.random.choice(batch_choices))
                c = int(np.random.uniform(min_conn, self.conn_max + 1))
                if self.is_valid_mm(s, b, c, homo, data_size):
                    valid.add((s, b, c))
            result = sorted(valid, key=lambda t: t[0] * t[1] * t[2])
            print(f"Generated {len(result)} valid mm parameter states under {self._limit_GB}GB boundary.")
            return result

        # Batch uses a fixed candidate set, so only scale/conn need continuous sampling.
        target_plane_samples = max(1, int(round(target_samples / len(batch_choices))))
        grid_res = max(3, int(round((target_plane_samples * 3) ** 0.5)))

        if dis_type == 'uniform':
            scales_raw = np.unique(np.linspace(min_scale, self.scale_max, num=grid_res, dtype=int))
            batches_raw = batch_choices
            conns_raw = np.unique(np.linspace(min_conn, self.conn_max, num=grid_res, dtype=int))
        elif dis_type == 'log':
            scales_raw = np.unique(np.geomspace(min_scale, self.scale_max, num=grid_res, dtype=int))
            batches_raw = batch_choices
            conns_raw = np.unique(np.geomspace(min_conn, self.conn_max, num=grid_res, dtype=int))
        else:
            raise ValueError(f"Unknown dis_type: '{dis_type}'. Choose from 'uniform', 'log', 'monte_carlo'.")

        valid = [
            (int(s), int(b), int(c))
            for s in scales_raw
            for b in batches_raw
            for c in conns_raw
            if self.is_valid_mm(s, b, c, homo, data_size)
        ]
        valid.sort(key=lambda t: t[0] * t[1] * t[2])
        print(f"Generated {len(valid)} valid mm parameter states under {self._limit_GB}GB boundary.")
        return valid

    def make_simulation_params_probs(self, 
                                     conn_prob : list, 
                                     scale: list, 
                                     batch_size: list,
                                     limit_memory:int = 24,
                                     data_size:int = 4,
                                     
                                     homo:bool = True):

        valid_pairs = []
        for s in scale:
            for batch in batch_size:
                for prob in conn_prob:
                    conn_number = int(prob * s * self._N)
                    if conn_number < 1:
                        conn_number = 1
                    if self.is_valid_mm(s, batch, conn_number, homo, data_size):
                        single_point = (int(s), prob, int(conn_number), batch)
                        valid_pairs.append(single_point)
                    else:
                        continue
        return valid_pairs
    
    def generate_params_steps(
        self,
        target_points: int = 100,
        step_ratio: tuple[float, float, float] = (1.0, 1.0, 1.0),
        data_size: int = 4,
        homo: bool = True,
    ) -> list[tuple[int, int, int]]:
        import math
        import numpy as np

        if target_points < 1:
            raise ValueError("target_points must be >= 1")

        if len(step_ratio) != 3:
            raise ValueError("step_ratio must be a tuple of length 3: (scale_ratio, conn_ratio, batch_ratio)")

        rs, rc, rb = step_ratio
        if rs <= 0 or rc <= 0 or rb <= 0:
            raise ValueError("All step_ratio values must be > 0")

        min_scale = _clamped_sampling_min(self.scale_max, MIN_GENERATED_SCALE)
        min_conn = _clamped_sampling_min(self.conn_max, MIN_GENERATED_CONN)
        batches_raw = np.asarray(_fixed_batch_candidates(self.batch_max), dtype=int)
        n_fixed_batches = len(batches_raw)

        def make_axis_points(min_val: int, max_val: int, n_points: int) -> np.ndarray:
            """在 [min_val, max_val] 上均匀取 n_points 个整数点，并保证包含端点。"""
            if max_val < 1:
                raise ValueError(f"max_val must be >= 1, got {max_val}")
            if min_val > max_val:
                raise ValueError(f"min_val must be <= max_val, got min_val={min_val}, max_val={max_val}")
            if n_points <= 1:
                return np.array([min_val, max_val], dtype=int) if max_val > min_val else np.array([min_val], dtype=int)

            pts = np.linspace(min_val, max_val, num=n_points)
            pts = np.rint(pts).astype(int)
            pts[0] = min_val
            pts[-1] = max_val
            return np.unique(pts)

        # 按比例分配三个维度的“采样密度”
        # batch 轴固定为预定义候选值，因此这里仅为 scale / conn 分配采样密度。
        # 希望 ns * nc * len(batches_raw) ≈ target_points
        #
        # 令两个维度采样点数与 1/ratio 成正比：
        # ratio 越大，步长越粗，点数越少
        inv_rs, inv_rc = 1.0 / rs, 1.0 / rc
        target_plane_points = max(1, int(round(target_points / n_fixed_batches)))
        base = (target_plane_points / (inv_rs * inv_rc)) ** 0.5

        n_scale = max(2, int(round(base * inv_rs)))
        n_conn = max(2, int(round(base * inv_rc)))

        # 微调，尽量让乘积接近 target_points
        def prod(a, b):
            return a * b * n_fixed_batches

        counts = [n_scale, n_conn]
        invs = [inv_rs, inv_rc]

        # 若点数太少，不断给“最该加密”的维度加 1
        while prod(*counts) < target_points:
            scores = [invs[i] / counts[i] for i in range(2)]
            idx = int(np.argmax(scores))
            counts[idx] += 1

        # 若点数明显过多，尝试减小
        while min(counts) > 2:
            current = prod(*counts)
            best_idx = None
            best_over = current - target_points
            for i in range(2):
                if counts[i] <= 2:
                    continue
                trial = counts.copy()
                trial[i] -= 1
                trial_prod = prod(*trial)
                if trial_prod >= target_points and (trial_prod - target_points) < best_over:
                    best_idx = i
                    best_over = trial_prod - target_points
            if best_idx is None:
                break
            counts[best_idx] -= 1

        n_scale, n_conn = counts

        scales_raw = make_axis_points(min_scale, self.scale_max, n_scale)
        conns_raw = make_axis_points(min_conn, self.conn_max, n_conn)

        valid = [
            (int(s), int(b), int(c))
            for s in scales_raw
            for c in conns_raw
            for b in batches_raw
            if self.is_valid_mm(s, b, c, homo, data_size)
        ]

        valid.sort(key=lambda t: (t[0], t[2], t[1]))

        print(
            f"Generated {len(valid)} valid mm parameter states "
            f"under {self._limit_GB}GB boundary. "
            f"(grid: {len(scales_raw)} x {len(conns_raw)} x {len(batches_raw)} = "
            f"{len(scales_raw) * len(conns_raw) * len(batches_raw)})"
        )
        return valid
