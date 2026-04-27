import os
import re
from pathlib import Path

import jax

MIN_GENERATED_SCALE = 20
MIN_GENERATED_CONN = 20
FIXED_GENERATED_BATCHES = (16, 32, 64, 128, 256)

_VALID_MV_MODES = frozenset({'pre', 'post'})
_VALID_MV_LAYOUTS = frozenset({'row_gather', 'col_scatter', 'auto'})
_DUAL_LAYOUT_MV_TYPES = frozenset({'binary', 'compact'})


def _clamped_sampling_min(max_val: int, preferred_min: int) -> int:
    if max_val < 1:
        raise ValueError(f"max_val must be >= 1, got {max_val}")
    return min(max_val, preferred_min)


def _fixed_batch_candidates(batch_max: int) -> list[int]:
    if batch_max < 1:
        raise ValueError(f"batch_max must be >= 1, got {batch_max}")
    batches = [b for b in FIXED_GENERATED_BATCHES if b <= batch_max]
    return batches if batches else [batch_max]


def _validate_mv_mode(mode: str) -> str:
    if mode not in _VALID_MV_MODES:
        raise ValueError(f"mode must be one of {_VALID_MV_MODES}, got {mode!r}.")
    return mode


def _validate_mv_layout(mv_layout: str) -> str:
    if mv_layout not in _VALID_MV_LAYOUTS:
        raise ValueError(f"mv_layout must be one of {_VALID_MV_LAYOUTS}, got {mv_layout!r}.")
    return mv_layout


def _resolve_mv_context(
    *,
    mode: str | None,
    data_type: str | None,
    mv_layout: str | None,
    default_mode: str,
    default_data_type: str,
    default_mv_layout: str,
) -> tuple[str, str, str]:
    resolved_mode = _validate_mv_mode(default_mode if mode is None else mode)
    resolved_layout = _validate_mv_layout(default_mv_layout if mv_layout is None else mv_layout)
    resolved_type = default_data_type if data_type is None else data_type
    return resolved_mode, resolved_type, resolved_layout


def _mv_mode_storage_factor(mode: str) -> int:
    # Post keeps one outgoing FCN view across E/I groups; pre keeps one
    # incoming FCN view per target group, doubling the stored nnz.
    return 1 if mode == 'post' else 2


def _mv_dual_layout_factor(mode: str, data_type: str, mv_layout: str) -> int:
    if mode == 'pre' and data_type in _DUAL_LAYOUT_MV_TYPES and mv_layout == 'col_scatter':
        return 2
    return 1


def _estimate_mv_sparse_bytes(
    *,
    scale: int,
    conn: int,
    _N: int,
    homo: bool,
    data_size: int,
    mode: str,
    data_type: str,
    mv_layout: str,
) -> int:
    times = 1 if homo else 2
    mode_factor = _mv_mode_storage_factor(mode)
    layout_factor = _mv_dual_layout_factor(mode, data_type, mv_layout)
    return conn * scale * _N * data_size * times * mode_factor * layout_factor


def _resolve_resume_csv_path(base_dir: Path, flush_dir: str | None, flush_file_name: str) -> Path:
    candidate = Path(flush_file_name)
    if candidate.suffix == '.csv' or candidate.parent != Path('.'):
        if not candidate.suffix:
            candidate = candidate.with_suffix('.csv')
        if candidate.is_absolute():
            return candidate
        if candidate.exists():
            return candidate.resolve()
        return (base_dir / candidate).resolve()

    target_dir = base_dir / flush_dir if flush_dir else base_dir
    return target_dir / f'{flush_file_name}.csv'


def _parse_resume_int_value(value, *, column: str, file_path: Path, row_index: int) -> int:
    if value is None or str(value).strip() == '':
        raise ValueError(
            f"Resume CSV has empty value for column {column!r} at row {row_index}: {file_path}"
        )
    try:
        return int(float(str(value).strip()))
    except ValueError as exc:
        raise ValueError(
            f"Resume CSV has non-integer value {value!r} for column {column!r} "
            f"at row {row_index}: {file_path}"
        ) from exc


def _load_resume_points(
    *,
    file_path: Path,
    required_columns: tuple[str, ...],
    point_from_row,
    label: str,
) -> tuple[set[tuple[int, ...]], int, int]:
    import csv

    if not file_path.exists():
        return set(), 0, 0

    with file_path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing_columns = [column for column in required_columns if column not in fieldnames]
        if missing_columns:
            raise ValueError(
                f"{label} resume CSV missing required columns {missing_columns}: {file_path}"
            )

        tested_points: set[tuple[int, ...]] = set()
        row_count = 0
        for row_index, row in enumerate(reader, start=2):
            row_count += 1
            tested_points.add(point_from_row(row=row, file_path=file_path, row_index=row_index))

    return tested_points, row_count, len(tested_points)


def _filter_states_by_existing_points(
    states,
    *,
    point_from_state,
    tested_points: set[tuple[int, ...]],
    non_repeat: bool,
    label: str,
):
    states_list = list(states)
    seen_points: set[tuple[int, ...]] = set()
    filtered_states = []
    skipped_existing = 0
    skipped_duplicates = 0

    for state in states_list:
        point = point_from_state(state)
        if non_repeat and point in tested_points:
            skipped_existing += 1
            continue
        if point in seen_points:
            skipped_duplicates += 1
            continue
        seen_points.add(point)
        filtered_states.append(state)

    removed_count = skipped_existing + skipped_duplicates
    if non_repeat or removed_count:
        print(
            f"{label} sample filter: removed={removed_count} "
            f"(existing={skipped_existing}, duplicate={skipped_duplicates}), "
            f"remaining={len(filtered_states)}"
        )
    return filtered_states


def _mv_point_from_row(*, row, file_path: Path, row_index: int) -> tuple[int, int]:
    return (
        _parse_resume_int_value(row.get('scale'), column='scale', file_path=file_path, row_index=row_index),
        _parse_resume_int_value(row.get('conn_num'), column='conn_num', file_path=file_path, row_index=row_index),
    )


def _mv_point_from_state(state) -> tuple[int, int]:
    if len(state) != 3:
        raise ValueError(f"MV state must be (scale, prob, conn_num), got {state!r}")
    scale, _, conn_num = state
    return int(scale), int(conn_num)


def _mm_point_from_row(*, row, file_path: Path, row_index: int) -> tuple[int, int, int]:
    return (
        _parse_resume_int_value(row.get('scale'), column='scale', file_path=file_path, row_index=row_index),
        _parse_resume_int_value(row.get('batch_size'), column='batch_size', file_path=file_path, row_index=row_index),
        _parse_resume_int_value(row.get('conn_num'), column='conn_num', file_path=file_path, row_index=row_index),
    )


def _mm_point_from_state(state) -> tuple[int, int, int]:
    if len(state) == 3:
        scale, batch_size, conn_num = state
    elif len(state) == 4:
        scale, _, conn_num, batch_size = state
    else:
        raise ValueError(
            "MM state must be (scale, batch_size, conn_num) or "
            f"(scale, prob, conn_num, batch_size), got {state!r}"
        )
    return int(scale), int(batch_size), int(conn_num)


class CSV_record:
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
    ) -> None:
        self.name = CSV_name
        self.suffix = suffix
        self.conn = conn
        self.operator_name = operator
        self.duration = self._extract_value(duration)

        if 'mv' in self.operator_name:
            self.fieldnames: list[str] = [
                'operator', 'data_type', 'backend', 'synaptic_type', 'scale', 'conn_num',
                'elapsed_s', 'firing_rate', 'duration', 'homo',
            ]
            self.operator_type = 'mv'
        elif 'mm' in self.operator_name:
            self.fieldnames = [
                'operator', 'data_type', 'backend', 'synaptic_type', 'scale', 'conn_num',
                'batch_size', 'elapsed_s', 'firing_rate', 'duration', 'homo',
            ]
            self.operator_type = 'mm'
        else:
            self.fieldnames = [
                'operator', 'data_type', 'backend', 'synaptic_type', 'scale', 'conn_num',
                'elapsed_s', 'firing_rate', 'duration', 'homo',
            ]
            self.operator_type = 'unknown'

        self.rows: list[dict] = []
        self.testing_type = 'coba' if testing_type == 'COBA' else testing_type

        try:
            self.base = Path(__file__).resolve().parent
        except NameError:
            self.base = Path.cwd()

        self.output_dir = Path(output_dir) if output_dir else self.base / 'results'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.append = append
        self._tags: dict = {}
        self.width = 70

    def _write_csv(
        self,
        file_name: str,
        rows: list[dict],
        fieldnames: list[str],
        mode: str = 'w',
        silent: bool = False,
    ) -> None:
        import csv

        file_path = Path(self.output_dir) / f'{file_name}.csv'
        write_header = True
        effective_fieldnames = list(fieldnames)

        if mode == 'a' and file_path.exists():
            write_header = False
            with file_path.open('r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                existing_fields = next(reader, [])
            if existing_fields:
                merged = list(existing_fields)
                for field in fieldnames:
                    if field not in merged:
                        merged.append(field)
                effective_fieldnames = merged

        with file_path.open(mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=effective_fieldnames, restval='default')
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

        if not silent:
            print(f"result has been saved: {file_path}")

    def add_tag(self, tag_name: str, tag_value) -> None:
        self._tags[tag_name] = tag_value
        if tag_name not in self.fieldnames:
            self.fieldnames.append(tag_name)

    def add_row(self, row: dict) -> None:
        merged_row = dict(self._tags)
        merged_row.update(row)
        for key in merged_row.keys():
            if key not in self.fieldnames:
                self.fieldnames.append(key)
        self.rows.append(merged_row)

    def single_COBA_data_add(
        self,
        operator: str,
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
        **kwargs,
    ):
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

        for key, value in kwargs.items():
            row[key] = self._extract_value(value)
        self.add_row(row)

    def record_finish(self, dir: str = '', suffix: str = '', file_name: str | None = None) -> None:
        if not self.rows:
            return
        if dir:
            self.output_dir = self.base / dir
            self.output_dir.mkdir(parents=True, exist_ok=True)

        suf = suffix if suffix else self.suffix
        is_default = (suf == 'default' or not suf)

        if file_name:
            out_name = file_name
            file_path = self.output_dir / f'{file_name}.csv'
            mode = 'a' if file_path.exists() else 'w'
        elif is_default:
            out_name = f'{self.testing_type}_default'
            mode = 'a'
        else:
            out_name = f'{self.testing_type}_{self.operator_name}_{self.name}_{suf}'
            file_path = self.output_dir / f'{out_name}.csv'
            mode = 'a' if file_path.exists() else 'w'

        self._write_csv(out_name, self.rows, self.fieldnames, mode=mode)

    def flush_and_clear(self, file_name: str, dir: str = '') -> Path | None:
        if not self.rows:
            return None
        if dir:
            target_dir = self.base / dir
            target_dir.mkdir(parents=True, exist_ok=True)
        else:
            target_dir = self.output_dir
        file_path = target_dir / f'{file_name}.csv'
        mode = 'a' if file_path.exists() else 'w'
        saved_output_dir = self.output_dir
        self.output_dir = target_dir
        self._write_csv(file_name, self.rows, self.fieldnames, mode=mode, silent=True)
        self.output_dir = saved_output_dir
        self.rows = []
        return file_path

    def print_header(
        self,
        *,
        operator: str,
        data_type: str,
        backend: str,
        mode: str,
        conn_num: int | None = None,
        duration=None,
        duration_ms: float | None = None,
        batch_size: int | None = None,
        **extra,
    ) -> None:
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
        for key, value in extra.items():
            parts.append(f'{key}={value}')
        print('  ' + ' | '.join(parts))
        print(f'{"=" * self.width}')

    def print_table_header(self, show_conn: bool = False, show_batch: bool = False) -> None:
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

    with open(jaxpr_path, "w", encoding='utf-8') as f:
        f.write(clean_jaxpr_str)
    print(f"[*] Clean JAXPR saved to: {jaxpr_path}")

    print("Lowering to HLO (XLA Physical Fusion State)...")
    lowered_executable = func.lower(*args, **kwargs)
    hlo_text = lowered_executable.as_text()

    with open(hlo_path, "w", encoding='utf-8') as f:
        f.write(hlo_text)
    print(f"[*] HLO saved to: {hlo_path}")

    return jaxpr_path, hlo_path


class TestingParamsGenerator_mv:
    def __init__(
        self,
        limit_GB: int,
        _N: int,
        scale_max: int = 2000,
        conn_max: int = 4000,
        sample_points: int = 50,
        *,
        mode: str = 'post',
        data_type: str = 'binary',
        mv_layout: str = 'row_gather',
        non_repeat: bool = False,
        flush_file_name: str | None = None,
        flush_dir: str | None = None,
    ):
        self._limit_GB = limit_GB
        self._N = _N
        self._limit_bytes = self._limit_GB * (1024) ** 3
        self.scale_max = scale_max
        self.conn_max = conn_max
        self.sample_points = sample_points
        self.mode = _validate_mv_mode(mode)
        self.data_type = data_type
        self.mv_layout = _validate_mv_layout(mv_layout)
        self.non_repeat = non_repeat
        self.flush_file_name = flush_file_name
        self.flush_dir = flush_dir
        self._tested_points: set[tuple[int, int]] = set()
        self._loaded_row_count = 0
        self._loaded_unique_count = 0
        self._resume_file_path: Path | None = None

        if self.non_repeat:
            if not self.flush_file_name:
                raise ValueError("flush_file_name must be provided when non_repeat=True for mv benchmarks")
            base_dir = Path(__file__).resolve().parent
            self._resume_file_path = _resolve_resume_csv_path(base_dir, self.flush_dir, self.flush_file_name)
            (
                self._tested_points,
                self._loaded_row_count,
                self._loaded_unique_count,
            ) = _load_resume_points(
                file_path=self._resume_file_path,
                required_columns=('scale', 'conn_num'),
                point_from_row=_mv_point_from_row,
                label='MV',
            )
            exists_label = 'exists' if self._resume_file_path.exists() else 'missing'
            print(
                f"MV resume CSV: {self._resume_file_path} ({exists_label}), "
                f"rows={self._loaded_row_count}, unique_points={self._loaded_unique_count}"
            )

    def _context(
        self,
        *,
        mode: str | None = None,
        data_type: str | None = None,
        mv_layout: str | None = None,
    ) -> tuple[str, str, str]:
        return _resolve_mv_context(
            mode=mode,
            data_type=data_type,
            mv_layout=mv_layout,
            default_mode=self.mode,
            default_data_type=self.data_type,
            default_mv_layout=self.mv_layout,
        )

    def estimate_memory_bytes(
        self,
        scale: int,
        conn: int,
        homo: bool,
        data_size: int = 4,
        *,
        mode: str | None = None,
        data_type: str | None = None,
        mv_layout: str | None = None,
    ) -> int:
        resolved_mode, resolved_type, resolved_layout = self._context(
            mode=mode,
            data_type=data_type,
            mv_layout=mv_layout,
        )
        return _estimate_mv_sparse_bytes(
            scale=scale,
            conn=conn,
            _N=self._N,
            homo=homo,
            data_size=data_size,
            mode=resolved_mode,
            data_type=resolved_type,
            mv_layout=resolved_layout,
        )

    def is_valid_mv(
        self,
        scale: int,
        conn: int,
        homo: bool,
        data_size: int,
        *,
        mode: str | None = None,
        data_type: str | None = None,
        mv_layout: str | None = None,
    ) -> bool:
        matrix_memory_bytes = self.estimate_memory_bytes(
            scale,
            conn,
            homo,
            data_size,
            mode=mode,
            data_type=data_type,
            mv_layout=mv_layout,
        )
        return matrix_memory_bytes <= self._limit_bytes and conn <= self._N * scale

    def _prob_to_conn_number(self, prob: float, scale: int, *, mode: str) -> int:
        base_conn = prob * scale * self._N
        if mode == 'post':
            return int(base_conn)
        # The benchmark uses one shared integer conn_num for both E and I
        # incoming matrices in pre mode, so use the average source-group size
        # to keep total synapse count close to the post-mode interpretation.
        return int(base_conn / 2.0)

    def filter_existing_states(self, states) -> list[tuple[int, None, int]]:
        return _filter_states_by_existing_points(
            states,
            point_from_state=_mv_point_from_state,
            tested_points=self._tested_points,
            non_repeat=self.non_repeat,
            label='MV',
        )

    def generate_params(
        self,
        dis_type: str = 'log',
        target_samples: int = 500,
        data_size: int = 4,
        homo: bool = True,
        *,
        mode: str | None = None,
        data_type: str | None = None,
        mv_layout: str | None = None,
    ) -> list[tuple[int, None, int]]:
        import numpy as np

        min_scale = _clamped_sampling_min(self.scale_max, MIN_GENERATED_SCALE)
        min_conn = _clamped_sampling_min(self.conn_max, MIN_GENERATED_CONN)

        if dis_type == 'monte_carlo':
            valid_states = set()
            while len(valid_states) < target_samples:
                s = int(np.random.uniform(min_scale, self.scale_max + 1))
                c = int(np.random.uniform(min_conn, self.conn_max + 1))
                if self.is_valid_mv(
                    s,
                    c,
                    homo,
                    data_size,
                    mode=mode,
                    data_type=data_type,
                    mv_layout=mv_layout,
                ):
                    valid_states.add((s, None, c))
            sorted_states = sorted(
                valid_states,
                key=lambda state: self.estimate_memory_bytes(
                    state[0],
                    state[2],
                    homo,
                    data_size,
                    mode=mode,
                    data_type=data_type,
                    mv_layout=mv_layout,
                ),
            )
            print(f"Generated {len(sorted_states)} valid parameter states under {self._limit_GB}GB boundary.")
            return self.filter_existing_states(sorted_states)

        grid_res = int(np.sqrt(target_samples * 3))
        if dis_type == 'uniform':
            scales_raw = np.unique(np.linspace(min_scale, self.scale_max, num=grid_res, dtype=int))
            conn_nums_raw = np.unique(np.linspace(min_conn, self.conn_max, num=grid_res, dtype=int))
        elif dis_type == 'log':
            scales_raw = np.unique(np.geomspace(min_scale, self.scale_max, num=grid_res, dtype=int))
            conn_nums_raw = np.unique(np.geomspace(min_conn, self.conn_max, num=grid_res, dtype=int))
        else:
            raise ValueError(f"Unknown dis_type: {dis_type!r}.")

        valid_states = [
            (int(s), None, int(c))
            for s in scales_raw
            for c in conn_nums_raw
            if self.is_valid_mv(
                int(s),
                int(c),
                homo,
                data_size,
                mode=mode,
                data_type=data_type,
                mv_layout=mv_layout,
            )
        ]
        valid_states.sort(
            key=lambda state: self.estimate_memory_bytes(
                state[0],
                state[2],
                homo,
                data_size,
                mode=mode,
                data_type=data_type,
                mv_layout=mv_layout,
            )
        )
        print(f"Generated {len(valid_states)} valid parameter states under {self._limit_GB}GB boundary.")
        return self.filter_existing_states(valid_states)

    def generate_boundary_params(
        self,
        homo_or_not: bool = True,
        _N: int = 4000,
        data_size: int = 4,
        sample_points: int = 10,
        include_dense_ref: bool = False,
        *,
        mode: str | None = None,
        data_type: str | None = None,
        mv_layout: str | None = None,
    ) -> list[tuple[int, int]]:
        import math
        import numpy as np

        resolved_mode, resolved_type, resolved_layout = self._context(
            mode=mode,
            data_type=data_type,
            mv_layout=mv_layout,
        )
        times = 1 if homo_or_not else 2
        mode_factor = _mv_mode_storage_factor(resolved_mode)
        layout_factor = _mv_dual_layout_factor(resolved_mode, resolved_type, resolved_layout)
        sparse_factor = times * mode_factor * layout_factor

        min_scale = _clamped_sampling_min(self.scale_max, MIN_GENERATED_SCALE)
        min_conn = _clamped_sampling_min(self.conn_max, MIN_GENERATED_CONN)

        K = self._limit_bytes / (_N * data_size)

        if include_dense_ref:
            s_max = min(self.scale_max, int(math.sqrt(K / _N)))
        else:
            s_max = min(self.scale_max, int(K / sparse_factor))

        if not include_dense_ref:
            s_min = max(min_scale, math.ceil(K / (sparse_factor * self.conn_max)))
        else:
            s_min = min_scale

        if s_min > s_max:
            raise ValueError(
                f"No valid (conn, m) pairs: s_min={s_min} > s_max={s_max}. "
                f"Try increasing memory_limit or decreasing scale_max/conn_max."
            )

        s_samples = np.geomspace(s_min, s_max, sample_points)
        valid_pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()

        for s_val in s_samples:
            s_int = max(min_scale, int(round(s_val)))
            s_int = min(s_int, s_max)

            if include_dense_ref:
                budget_for_sparse = K - s_int ** 2 * _N
                if budget_for_sparse <= 0:
                    continue
                c_int = int(budget_for_sparse / (s_int * sparse_factor))
            else:
                c_int = int(K / (s_int * sparse_factor))

            c_int = min(c_int, self.conn_max)
            m = s_int * _N

            if c_int >= min_conn and c_int <= m and s_int <= self.scale_max:
                pair = (s_int, c_int)
                if pair not in seen:
                    seen.add(pair)
                    valid_pairs.append(pair)

        return valid_pairs

    def make_simulation_params_probs(
        self,
        conn_prob: list,
        scale: list,
        limit_memory: int | None = None,
        data_size: int = 4,
        homo: bool = True,
        *,
        mode: str | None = None,
        data_type: str | None = None,
        mv_layout: str | None = None,
    ) -> list[tuple[int, None, int]]:
        resolved_mode, _, _ = self._context(mode=mode, data_type=data_type, mv_layout=mv_layout)
        if limit_memory is not None and limit_memory != self._limit_GB:
            generator = TestingParamsGenerator_mv(
                limit_GB=limit_memory,
                _N=self._N,
                scale_max=self.scale_max,
                conn_max=self.conn_max,
                sample_points=self.sample_points,
                mode=resolved_mode,
                data_type=self.data_type if data_type is None else data_type,
                mv_layout=self.mv_layout if mv_layout is None else mv_layout,
                non_repeat=False,
            )
        else:
            generator = self

        valid_pairs: list[tuple[int, None, int]] = []
        for s in scale:
            for prob in conn_prob:
                conn_number = generator._prob_to_conn_number(prob, s, mode=resolved_mode)
                if conn_number < 1:
                    conn_number = 1
                if generator.is_valid_mv(
                    s,
                    conn_number,
                    homo,
                    data_size,
                    mode=resolved_mode,
                    data_type=data_type,
                    mv_layout=mv_layout,
                ):
                    valid_pairs.append((int(s), None, int(conn_number)))
        return self.filter_existing_states(valid_pairs)

    def generate_coba_vram_sequence(
        self,
        vram_steps: list | None = None,
        sample_points: int = 5,
        homo: bool = True,
        data_size: int = 4,
        *,
        mode: str | None = None,
        data_type: str | None = None,
        mv_layout: str | None = None,
    ):
        from collections import OrderedDict

        resolved_mode, resolved_type, resolved_layout = self._context(
            mode=mode,
            data_type=data_type,
            mv_layout=mv_layout,
        )
        if vram_steps is None:
            vram_steps = list(range(1, 25))

        result = OrderedDict()
        for vram_gb in vram_steps:
            gen = TestingParamsGenerator_mv(
                limit_GB=vram_gb,
                _N=self._N,
                scale_max=self.scale_max,
                conn_max=self.conn_max,
                sample_points=self.sample_points,
                mode=resolved_mode,
                data_type=resolved_type,
                mv_layout=resolved_layout,
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

        print(f"Generated VRAM sequence: {len(result)} levels, total {sum(len(v) for v in result.values())} parameter pairs.")
        return result


TestingParamsGenerator = TestingParamsGenerator_mv


def memory_limit(
    conn: int,
    *,
    scale: int,
    limit_GB: int = 16,
    _N: int = 4000,
    homo: bool = True,
    data_size: int = 4,
    mode: str = 'post',
    data_type: str = 'binary',
    mv_layout: str = 'row_gather',
) -> bool:
    generator = TestingParamsGenerator_mv(
        limit_GB=limit_GB,
        _N=_N,
        mode=mode,
        data_type=data_type,
        mv_layout=mv_layout,
    )
    return not generator.is_valid_mv(scale, conn, homo, data_size)


class TestingParamsGenerator_mm:
    """Generate valid (scale, batch_size, conn_num) triples for mm benchmarks."""

    def __init__(
        self,
        limit_GB: int = 24,
        _N: int = 4000,
        scale_max: int = 200,
        conn_max: int = 4000,
        batch_max: int = 128,
        *,
        non_repeat: bool = False,
        flush_file_name: str | None = None,
        flush_dir: str | None = None,
    ):
        self._limit_GB = limit_GB
        self._N = _N
        self._limit_bytes = self._limit_GB * (1024) ** 3
        self.scale_max = scale_max
        self.conn_max = conn_max
        self.batch_max = batch_max
        self.non_repeat = non_repeat
        self.flush_file_name = flush_file_name
        self.flush_dir = flush_dir
        self._tested_points: set[tuple[int, int, int]] = set()
        self._loaded_row_count = 0
        self._loaded_unique_count = 0
        self._resume_file_path: Path | None = None

        if self.non_repeat:
            if not self.flush_file_name:
                raise ValueError("flush_file_name must be provided when non_repeat=True for mm benchmarks")
            base_dir = Path(__file__).resolve().parent
            self._resume_file_path = _resolve_resume_csv_path(base_dir, self.flush_dir, self.flush_file_name)
            (
                self._tested_points,
                self._loaded_row_count,
                self._loaded_unique_count,
            ) = _load_resume_points(
                file_path=self._resume_file_path,
                required_columns=('scale', 'batch_size', 'conn_num'),
                point_from_row=_mm_point_from_row,
                label='MM',
            )
            exists_label = 'exists' if self._resume_file_path.exists() else 'missing'
            print(
                f"MM resume CSV: {self._resume_file_path} ({exists_label}), "
                f"rows={self._loaded_row_count}, unique_points={self._loaded_unique_count}"
            )

    def is_valid_mm(self, scale: int, batch_size: int, conn: int,
                 homo: bool, data_size: int = 4) -> bool:
        times = 1 if homo else 2
        size = scale * self._N
        mem = conn * size * times + batch_size * size + size * batch_size
        mem = mem * data_size
        return mem <= self._limit_bytes and conn <= size

    def filter_existing_states(self, states):
        return _filter_states_by_existing_points(
            states,
            point_from_state=_mm_point_from_state,
            tested_points=self._tested_points,
            non_repeat=self.non_repeat,
            label='MM',
        )

    def generate_params(
        self,
        dis_type: str = 'uniform',
        target_samples: int = 500,
        data_size: int = 4,
        homo: bool = True,
    ) -> list:
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
            return self.filter_existing_states(result)

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
            raise ValueError(f"Unknown dis_type: {dis_type!r}.")

        valid = [
            (int(s), int(b), int(c))
            for s in scales_raw
            for b in batches_raw
            for c in conns_raw
            if self.is_valid_mm(s, b, c, homo, data_size)
        ]
        valid.sort(key=lambda t: t[0] * t[1] * t[2])
        print(f"Generated {len(valid)} valid mm parameter states under {self._limit_GB}GB boundary.")
        return self.filter_existing_states(valid)

    def make_simulation_params_probs(self,
                                     conn_prob: list,
                                     scale: list,
                                     batch_size: list,
                                     limit_memory: int = 24,
                                     data_size: int = 4,
                                     homo: bool = True):
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
        return self.filter_existing_states(valid_pairs)

    def generate_params_steps(
        self,
        target_points: int = 100,
        step_ratio: tuple[float, float, float] = (1.0, 1.0, 1.0),
        data_size: int = 4,
        homo: bool = True,
    ) -> list[tuple[int, int, int]]:
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

        inv_rs, inv_rc = 1.0 / rs, 1.0 / rc
        target_plane_points = max(1, int(round(target_points / n_fixed_batches)))
        base = (target_plane_points / (inv_rs * inv_rc)) ** 0.5

        n_scale = max(2, int(round(base * inv_rs)))
        n_conn = max(2, int(round(base * inv_rc)))

        def prod(a, b):
            return a * b * n_fixed_batches

        counts = [n_scale, n_conn]
        invs = [inv_rs, inv_rc]

        while prod(*counts) < target_points:
            scores = [invs[i] / counts[i] for i in range(2)]
            idx = int(np.argmax(scores))
            counts[idx] += 1

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
        return self.filter_existing_states(valid)
