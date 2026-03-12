
class CSV_record():
    """Generic CSV recorder used by benchmarks.

    Features:
    - accepts arbitrary extra fields per row and tracks fieldnames
    - configurable output directory
    - append or overwrite mode
    - convenience helper `single_COBA_data_add` kept for compatibility
    """
    def __init__(
        self,
        CSV_name: str,
        operator: str,
        testing_type: str,
        suffix: str = '',
        output_dir: str | None = None,
        append: bool = False,
    ) -> None:
        from pathlib import Path

        self.name = CSV_name
        self.suffix = suffix
        # default common fields
        self.fieldnames: list[str] = [
            'operator', 'data_type', 'synaptic_type', 'scale', 'conn_num',
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
                             synaptic_type: str,
                             conn_num: int,
                             scale: int,
                             elapsed_s: float,
                             firing_rate: float,
                             duration: float,
                             homo: str = 'default',
                             **kwargs):
        """Backwards-compatible helper used by existing benchmarks."""
        row = {
            'operator': operator,
            'data_type': data_type,
            'testing_type': self.testing_type,
            'synaptic_type': synaptic_type,
            'scale': scale,
            'conn_num': conn_num,
            'elapsed_s': elapsed_s,
            'firing_rate': firing_rate,
            'duration': duration,
            'homo': homo,
        }

        # merge extras
        for key, value in kwargs.items():
            row[key] = value

        self.add_row(row)

    def record_finish(self, suffix: str = '', file_name: str | None = None) -> None:
        """Write accumulated rows to disk.

        - `suffix` will be used in auto-generated filename if provided.
        - `file_name` overrides auto filename.
        - When suffix is 'default' (or empty), results are **always appended** to
          ``{testing_type}_default.csv`` so that multiple operators / data-types
          are aggregated into one file for unified comparison.
        """
        suf = suffix if suffix else self.suffix

        is_default = (suf == 'default' or not suf)

        if file_name:
            out_name = file_name
            mode = 'a' if self.append else 'w'
        elif is_default:
            # Aggregate everything into <testing_type>_default.csv (always append)
            out_name = f'{self.testing_type}_default'
            mode = 'a'
        else:
            out_name = f'{self.testing_type}_{self.operator_name}_{self.name}_{suf}'
            mode = 'a' if self.append else 'w'

        self._write_csv(out_name, self.rows, self.fieldnames, mode=mode)
