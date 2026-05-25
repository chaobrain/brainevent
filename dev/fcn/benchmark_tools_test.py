from pathlib import Path
import sys

import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import dev.fcn.BenchmarkTools as BT
import dev.fcn.COBA_EI_binary_fcnmm_VRAM_limit_CsvOutput as EI_VRAM_MM


def _write_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    lines = [','.join(header)]
    for row in rows:
        lines.append(','.join(str(value) for value in row))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _assert_mv_states_valid(generator, states, *, homo=True, data_size=4):
    assert len({(scale, conn_num) for scale, _, conn_num in states}) == len(states)
    for scale, _, conn_num in states:
        assert scale >= BT.MIN_GENERATED_SCALE
        assert scale <= generator.scale_max
        assert conn_num >= BT.MIN_GENERATED_CONN
        assert conn_num <= generator.conn_max
        assert conn_num <= scale * generator._N
        assert generator.is_valid_mv(scale, conn_num, homo=homo, data_size=data_size)


def _assert_mm_states_valid(generator, states, *, homo=True, data_size=4):
    assert len(set(states)) == len(states)
    for scale, batch_size, conn_num in states:
        assert scale >= BT.MIN_GENERATED_SCALE
        assert scale <= generator.scale_max
        assert conn_num >= BT.MIN_GENERATED_CONN
        assert conn_num <= generator.conn_max
        assert conn_num <= scale * generator._N
        assert generator.is_valid_mm(scale, batch_size, conn_num, homo=homo, data_size=data_size)


def _assert_aligned_mv_grid_has_no_holes(generator, states, *, homo=True, data_size=4):
    points = {(scale, conn_num) for scale, _, conn_num in states}
    scales = sorted({scale for scale, _ in points})
    conns = sorted({conn_num for _, conn_num in points})
    assert scales
    assert conns
    for scale in scales:
        for conn in conns:
            if generator.is_valid_mv(scale, conn, homo=homo, data_size=data_size):
                assert (scale, conn) in points


def _assert_aligned_mm_grid_has_no_holes(generator, states, *, homo=True, data_size=4):
    for batch_size in {batch for _, batch, _ in states}:
        points = {(scale, conn_num) for scale, batch, conn_num in states if batch == batch_size}
        scales = sorted({scale for scale, _ in points})
        conns = sorted({conn_num for _, conn_num in points})
        assert scales
        assert conns
        for scale in scales:
            for conn in conns:
                if generator.is_valid_mm(scale, batch_size, conn, homo=homo, data_size=data_size):
                    assert (scale, conn) in points


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
    post_col_gen = BT.TestingParamsGenerator_mv(
        limit_GB=8,
        _N=4000,
        mode='post',
        data_type='binary',
        mv_layout='col_scatter',
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
    post_col_bytes = post_col_gen.estimate_memory_bytes(scale, conn, homo=True, data_size=data_size)
    pre_bytes = pre_gen.estimate_memory_bytes(scale, conn, homo=True, data_size=data_size)
    pre_col_bytes = pre_col_gen.estimate_memory_bytes(scale, conn, homo=True, data_size=data_size)

    assert post_bytes == conn * scale * 4000 * data_size
    assert pre_bytes == post_bytes
    assert post_col_bytes == 2 * post_bytes
    assert pre_col_bytes == 2 * post_bytes


def test_mv_prob_conversion_keeps_shared_pre_conn_num():
    post_gen = BT.TestingParamsGenerator_mv(limit_GB=8, _N=4000, mode='post')
    pre_gen = BT.TestingParamsGenerator_mv(limit_GB=8, _N=4000, mode='pre')

    post_pairs = post_gen.make_simulation_params_probs([0.1], [1], homo=True)
    pre_pairs = pre_gen.make_simulation_params_probs([0.1], [1], homo=True)

    assert post_pairs == [(1, None, 400)]
    assert pre_pairs == [(1, None, 400)]


def test_mv_generate_params_respects_sample_points_default_limit():
    generator = BT.TestingParamsGenerator_mv(
        limit_GB=16,
        _N=4000,
        scale_max=1250,
        conn_max=2000,
        sample_points=25,
        mode='post',
    )

    valid_states = generator.generate_params(dis_type='uniform', homo=True)
    explicit_states = generator.generate_params(dis_type='uniform', target_samples=25, homo=True)

    assert 25 * 0.9 <= len(valid_states) <= 25 * 1.1
    assert valid_states == explicit_states
    _assert_mv_states_valid(generator, valid_states)
    _assert_aligned_mv_grid_has_no_holes(generator, valid_states)


def test_mv_grid_uniform_is_aligned_and_complete():
    generator = BT.TestingParamsGenerator_mv(
        limit_GB=16,
        _N=4000,
        scale_max=100,
        conn_max=100,
        sample_points=5,
        mode='post',
    )

    states = generator.generate_params(dis_type='grid_uniform', target_samples=5, homo=True)

    assert states
    _assert_mv_states_valid(generator, states)
    _assert_aligned_mv_grid_has_no_holes(generator, states)


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


def test_mv_uniform_non_repeat_skips_existing_grid_point(tmp_path):
    csv_path = tmp_path / 'mv_resume.csv'
    _write_csv(csv_path, ['scale', 'conn_num'], [(20, 20)])

    generator = BT.TestingParamsGenerator_mv(
        limit_GB=8,
        _N=4000,
        scale_max=20,
        conn_max=25,
        mode='post',
        non_repeat=True,
        flush_file_name=str(csv_path),
    )

    states = generator.generate_params(dis_type='uniform', target_samples=3, homo=True)

    assert (20, None, 20) not in states
    _assert_mv_states_valid(generator, states)


def test_mv_uniform_warns_when_available_states_below_tolerance(capsys):
    generator = BT.TestingParamsGenerator_mv(
        limit_GB=8,
        _N=4000,
        scale_max=20,
        conn_max=21,
        mode='post',
    )

    states = generator.generate_params(dis_type='uniform', target_samples=10, homo=True)

    assert states == [(20, None, 20), (20, None, 21)]
    assert 'outside 10% of requested target=10' in capsys.readouterr().out
    _assert_mv_states_valid(generator, states)


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


def test_mm_uniform_non_repeat_skips_existing_grid_point(tmp_path):
    csv_path = tmp_path / 'mm_resume.csv'
    _write_csv(csv_path, ['scale', 'batch_size', 'conn_num'], [(20, 16, 20)])

    generator = BT.TestingParamsGenerator_mm(
        limit_GB=8,
        _N=4000,
        scale_max=20,
        conn_max=25,
        batch_max=16,
        non_repeat=True,
        flush_file_name=str(csv_path),
    )

    states = generator.generate_params(
        dis_type='uniform',
        target_samples=3,
        homo=True,
        batch_sizes=[16],
        target_samples_per_batch=3,
    )

    assert (20, 16, 20) not in states
    _assert_mm_states_valid(generator, states)


def test_mm_generate_params_uses_given_batch_sizes_and_target_samples_per_batch():
    generator = BT.TestingParamsGenerator_mm(
        limit_GB=5,
        _N=4000,
        scale_max=2000,
        conn_max=4000,
        batch_max=256,
    )

    valid_states = generator.generate_params(
        dis_type='uniform',
        target_samples=4,
        data_size=4,
        homo=True,
        batch_sizes=[32, 128],
        target_samples_per_batch=4,
    )

    batches = {batch for _, batch, _ in valid_states}
    assert batches <= {32, 128}
    assert 32 in batches
    assert 128 in batches

    per_batch = {batch: 0 for batch in batches}
    for _, batch, _ in valid_states:
        per_batch[batch] += 1
    assert all(4 * 0.9 <= count <= 4 * 1.1 for count in per_batch.values())
    _assert_mm_states_valid(generator, valid_states)
    _assert_aligned_mm_grid_has_no_holes(generator, valid_states)


def test_mm_generate_params_splits_target_across_batches_when_per_batch_missing():
    generator = BT.TestingParamsGenerator_mm(
        limit_GB=5,
        _N=4000,
        scale_max=2000,
        conn_max=4000,
        batch_max=256,
    )

    valid_states = generator.generate_params(
        dis_type='uniform',
        target_samples=7,
        data_size=4,
        homo=True,
        batch_sizes=[32, 128],
    )

    per_batch = {32: 0, 128: 0}
    for _, batch, _ in valid_states:
        per_batch[batch] += 1
    assert per_batch[32] >= 1
    assert per_batch[128] >= 1
    _assert_mm_states_valid(generator, valid_states)
    _assert_aligned_mm_grid_has_no_holes(generator, valid_states)


def test_mm_grid_uniform_is_aligned_and_complete():
    generator = BT.TestingParamsGenerator_mm(
        limit_GB=8,
        _N=4000,
        scale_max=100,
        conn_max=100,
        batch_max=16,
    )

    states = generator.generate_params(
        dis_type='grid_uniform',
        target_samples=5,
        homo=True,
        batch_sizes=[16],
        target_samples_per_batch=5,
    )

    assert states
    _assert_mm_states_valid(generator, states)
    _assert_aligned_mm_grid_has_no_holes(generator, states)


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


def test_ei_mm_estimate_matches_current_formula():
    bytes_used = EI_VRAM_MM._estimate_mm_bytes(
        scale=2,
        batch_size=32,
        conn=10,
        _N=4000,
        homo=True,
        data_size=4,
    )

    expected = (10 * 2 * 4000 + 2 * 32 * 2 * 4000) * 4
    assert bytes_used == expected


def test_ei_mm_boundary_pairs_respect_batch_and_point_limits():
    pairs = EI_VRAM_MM._generate_mm_boundary_pairs(
        limit_gb=1,
        batch_size=32,
        sample_points_per_batch=2,
        _N=4000,
        scale_max=2000,
        conn_max=4000,
        homo=True,
        data_size=4,
    )

    assert 1 <= len(pairs) <= 2
    assert len(set(pairs)) == len(pairs)
    for scale, conn in pairs:
        assert scale >= 20
        assert conn >= 20
        assert conn <= 4000
        assert conn <= scale * 4000
        assert EI_VRAM_MM._estimate_mm_bytes(scale, 32, conn, 4000, True, 4) <= 1 * (1024 ** 3)


def test_ei_mm_vram_sequence_uses_only_fixed_batches():
    plan = EI_VRAM_MM._generate_mm_vram_sequence(
        vram_steps=[1, 2],
        fixed_batch_sizes=(32, 64, 128),
        sample_points_per_batch=2,
        _N=4000,
        scale_max=2000,
        conn_max=4000,
        homo=True,
        data_size=4,
    )

    assert list(plan.keys()) == [1, 2]
    for batch_plan in plan.values():
        assert list(batch_plan.keys()) == [32, 64, 128]
        for pairs in batch_plan.values():
            assert len(pairs) <= 2


def test_ei_mm_txt_report_contains_summary_and_details(tmp_path):
    vram_plan = EI_VRAM_MM._generate_mm_vram_sequence(
        vram_steps=[1],
        fixed_batch_sizes=(32, 64, 128),
        sample_points_per_batch=1,
        _N=4000,
        scale_max=2000,
        conn_max=4000,
        homo=True,
        data_size=4,
    )
    case_records = [{
        'benchmark_name': 'coba_ei',
        'backend': 'cuda_raw',
        'data_type': 'binary',
        'synaptic_type': 'post',
        'scale': 20,
        'conn_num': 20,
        'batch_size': 32,
        'limit_GB': 2,
        'current_VRAM': 1,
        'current_VRAM_bytes': 123456,
        'current_VRAM_GiB': 0.000115,
        'mv_layout': 'row_gather',
        'point_index': '1/3',
        'homo': 'homo',
        'neurons': 4000,
        'elapsed_s': 0.1,
        'firing_rate': 12.3,
        'message': 'OK',
        'exception': '',
    }]
    summary_counts, level_summaries = EI_VRAM_MM._summarize_case_records(vram_plan, case_records)
    report = EI_VRAM_MM._build_txt_report(
        created_at='2026-05-03T12:00:00',
        runtime_platform='gpu',
        backend='cuda_raw',
        data_type='binary',
        homo=True,
        efferent_target='post',
        mv_layout='row_gather',
        fixed_batch_sizes=(32, 64, 128),
        sample_points_per_batch=1,
        _N=4000,
        vram_steps=[1],
        scale_max=2000,
        conn_max=4000,
        total_candidate_points=3,
        summary_counts=summary_counts,
        level_summaries=level_summaries,
        case_records=case_records,
        csv_output_path=tmp_path / 'result.csv',
    )

    assert '# COBA EI FCNMM VRAM-limit Report' in report
    assert '[Summary]' in report
    assert '[VRAM-level summary]' in report
    assert '[Detailed cases]' in report
    assert 'point=1/3' in report
    assert 'batch=32' in report
    assert 'message=OK' in report
