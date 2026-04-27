from pathlib import Path
import sys

import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import dev.fcn.BenchmarkTools as BT


def _write_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    lines = [','.join(header)]
    for row in rows:
        lines.append(','.join(str(value) for value in row))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def test_mv_memory_model_is_mode_and_layout_aware():
    scale = 2
    conn = 10
    data_size = 4

    post_gen = BT.TestingParamsGenerator_mv(
        limit_GB=8,
        _N=4000,
        mode='post',
        data_type='binary',
        mv_layout='row_gather',
    )
    pre_gen = BT.TestingParamsGenerator_mv(
        limit_GB=8,
        _N=4000,
        mode='pre',
        data_type='binary',
        mv_layout='row_gather',
    )
    pre_col_gen = BT.TestingParamsGenerator_mv(
        limit_GB=8,
        _N=4000,
        mode='pre',
        data_type='binary',
        mv_layout='col_scatter',
    )

    post_bytes = post_gen.estimate_memory_bytes(scale, conn, homo=True, data_size=data_size)
    pre_bytes = pre_gen.estimate_memory_bytes(scale, conn, homo=True, data_size=data_size)
    pre_col_bytes = pre_col_gen.estimate_memory_bytes(scale, conn, homo=True, data_size=data_size)

    assert post_bytes == conn * scale * 4000 * data_size
    assert pre_bytes == 2 * post_bytes
    assert pre_col_bytes == 2 * pre_bytes


def test_mv_prob_conversion_reduces_shared_pre_conn_num():
    post_gen = BT.TestingParamsGenerator_mv(limit_GB=8, _N=4000, mode='post')
    pre_gen = BT.TestingParamsGenerator_mv(limit_GB=8, _N=4000, mode='pre')

    post_pairs = post_gen.make_simulation_params_probs([0.1], [1], homo=True)
    pre_pairs = pre_gen.make_simulation_params_probs([0.1], [1], homo=True)

    assert post_pairs == [(1, None, 400)]
    assert pre_pairs == [(1, None, 200)]


def test_mv_non_repeat_filters_generated_prob_and_manual_states(tmp_path):
    csv_path = tmp_path / 'mv_resume.csv'
    _write_csv(
        csv_path,
        ['scale', 'conn_num'],
        [(20, 20), (1, 400)],
    )

    generator = BT.TestingParamsGenerator_mv(
        limit_GB=8,
        _N=4000,
        scale_max=20,
        conn_max=20,
        mode='post',
        non_repeat=True,
        flush_file_name=str(csv_path),
    )

    assert generator._loaded_row_count == 2
    assert generator._loaded_unique_count == 2
    assert generator.filter_existing_states([(20, None, 20), (20, None, 20), (21, None, 21)]) == [(21, None, 21)]
    assert generator.generate_params(dis_type='uniform', target_samples=3, homo=True) == []
    assert generator.make_simulation_params_probs([0.1], [1], homo=True) == []


def test_mm_non_repeat_filters_generated_prob_step_and_manual_states(tmp_path):
    csv_path = tmp_path / 'mm_resume.csv'
    _write_csv(
        csv_path,
        ['scale', 'batch_size', 'conn_num'],
        [(20, 16, 20), (1, 16, 40)],
    )

    generator = BT.TestingParamsGenerator_mm(
        limit_GB=8,
        _N=4000,
        scale_max=20,
        conn_max=20,
        batch_max=16,
        non_repeat=True,
        flush_file_name=str(csv_path),
    )

    filtered_states = generator.filter_existing_states(
        [(1, 0.01, 40, 16), (1, 0.02, 80, 16), (1, 0.02, 80, 16)]
    )
    assert filtered_states == [(1, 0.02, 80, 16)]
    assert all(len(state) == 4 for state in filtered_states)
    assert generator.generate_params(dis_type='uniform', target_samples=3, homo=True) == []
    assert generator.make_simulation_params_probs([0.01], [1], [16], homo=True) == []
    assert generator.generate_params_steps(target_points=2, homo=True) == []


def test_non_repeat_missing_csv_does_not_remove_states(tmp_path):
    generator = BT.TestingParamsGenerator_mv(
        limit_GB=8,
        _N=4000,
        mode='post',
        non_repeat=True,
        flush_file_name=str(tmp_path / 'missing_resume.csv'),
    )

    assert generator._loaded_row_count == 0
    assert generator._loaded_unique_count == 0
    assert generator.filter_existing_states([(20, None, 20), (20, None, 20)]) == [(20, None, 20)]


@pytest.mark.parametrize(
    ('generator_ctor', 'header'),
    [
        (
            lambda tmp_path: BT.TestingParamsGenerator_mv(
                limit_GB=8,
                _N=4000,
                non_repeat=True,
                flush_file_name='bad_mv_resume',
                flush_dir=str(tmp_path),
            ),
            ['scale'],
        ),
        (
            lambda tmp_path: BT.TestingParamsGenerator_mm(
                limit_GB=8,
                _N=4000,
                non_repeat=True,
                flush_file_name='bad_mm_resume',
                flush_dir=str(tmp_path),
            ),
            ['scale', 'conn_num'],
        ),
    ],
)
def test_non_repeat_raises_on_missing_required_resume_columns(tmp_path, generator_ctor, header):
    file_name = 'bad_mv_resume.csv' if 'conn_num' not in header else 'bad_mm_resume.csv'
    _write_csv(tmp_path / file_name, header, [])

    with pytest.raises(ValueError, match='missing required columns'):
        generator_ctor(tmp_path)
