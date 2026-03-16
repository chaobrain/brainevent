
import os
import re
import jax
from pathlib import Path

class CSV_record():
    """Generic CSV recorder used by benchmarks.

    Features:
    - accepts arbitrary extra fields per row and tracks fieldnames
    - configurable output directory
    - append or overwrite mode
    - convenience helper `single_COBA_data_add` kept for compatibility
    - automatic handling of brainunit Quantity objects (extracts numeric values)
    - automatic append mode detection for existing files
    """
    
    @staticmethod
    def _extract_value(param):
        """Extract numeric value from parameter, handling brainunit.Quantity objects.
        
        If the parameter is a brainunit Quantity object, extracts the magnitude.
        Otherwise returns the parameter as-is.
        """
        if hasattr(param, '__class__') and param.__class__.__name__ == 'Quantity':
            return param.magnitude
        return param
    
    def __init__(
        self,
        CSV_name: str,
        operator: str,
        testing_type: str,
        duration: float,
        conn: int,
        suffix: str = '',
        output_dir: str | None = None,
        append: bool = False,
        
    ) -> None:
        self.name = CSV_name
        self.suffix = suffix
        self.conn = conn
        # Handle brainunit Quantity objects
        self.duration = self._extract_value(duration)

        # default common fields
        self.fieldnames: list[str] = [
            'operator', 'data_type', 'backend', 'synaptic_type', 'scale', 'conn_num',
            'elapsed_s', 'firing_rate', 'duration', 'homo'
        ]
        self.rows: list[dict] = []
        self.testing_type = testing_type  
        if testing_type == 'COBA': self.testing_type = 'coba' # coba or benchmark
        self.operator_name = operator

        # output dir default: same folder as this file /results
        try:
            base = Path(__file__).resolve().parent
        except NameError:
            base = Path.cwd()

        self.output_dir = Path(output_dir) if output_dir else base / 'results'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.append = append

        self.width = 70

    def _write_csv(self, file_name: str, rows: list[dict], fieldnames: list[str], mode: str = 'w') -> None:
        import csv
        from pathlib import Path

        file_path = Path(self.output_dir) / f'{file_name}.csv'
        write_header = True
        if mode == 'a' and file_path.exists():
            # if appending and file exists, do not write header again
            write_header = False

        with file_path.open(mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval=None)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

        print(f"result has been saved: {file_path}")

    def add_row(self, row: dict) -> None:
        """Add a generic row (dict). New keys will be added to fieldnames."""
        for key in row.keys():
            if key not in self.fieldnames:
                self.fieldnames.append(key)
        self.rows.append(row)

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
                             **kwargs):
        """Backwards-compatible helper used by existing benchmarks.
        
        Automatically extracts numeric values from brainunit Quantity objects.
        """
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

    def record_finish(self, suffix: str = '', file_name: str | None = None) -> None:
        """Write accumulated rows to disk.

        - `suffix` will be used in auto-generated filename if provided.
        - `file_name` overrides auto filename.
        - When suffix is 'default' (or empty), results are **always appended** to
          ``{testing_type}_default.csv`` so that multiple operators / data-types
          are aggregated into one file for unified comparison.
        - For other files, automatically detects if the target file exists and
          appends if it does.
        """
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


    def print_header(self, *, operator: str, data_type: str, backend: str,
                     mode: str, conn_num: int, duration_ms: float,
                     batch_size: int | None = None, **extra) -> None:
        """Print parameter condition header block."""
        print(f'\n{"=" * self.width}')
        print(f'  operator={self.operator_name} | data_type={data_type} | backend={backend}')
        parts: list[str] = [f'mode={mode:<8s}']
        if batch_size is not None:
            parts.append(f'batch_size={batch_size}')
        parts.append(f'conn_num={conn_num}')
        parts.append(f'duration={duration_ms:.1f} ms')
        for k, v in extra.items():
            parts.append(f'{k}={v}')
        print('  ' + ' | '.join(parts))
        print(f'{"=" * self.width}')

    def print_table_header(self) -> None:
        """Print column header for the standard result table."""
        print(f'  {"Scale":>5s} | {"Neurons":>7s} | {"Elapsed (s)":>11s} | {"Rate (Hz)":>9s}')
        print(f'  {"-----":>5s}-+-{"-------":>7s}-+-{"----------":>11s}-+-{"------":>9s}')

    @staticmethod
    def print_row(scale: int, neurons: int, elapsed: float, rate: float) -> None:
        """Print one benchmark result row."""
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
