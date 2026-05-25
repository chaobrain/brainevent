from pathlib import Path
import importlib.util

import numpy as np


def _load_benchmark_module():
    path = Path(__file__).with_name("benchmark_fcn_operators.py")
    spec = importlib.util.spec_from_file_location("benchmark_fcn_operators", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shrink_benchmark_shape(bench):
    bench.SCALE = 0.016
    bench.CONN = 8
    bench.N_B = 8
    bench.SPIKE_STEPS = 4


def test_seeded_spike_series_is_reproducible_and_varies_by_step():
    bench = _load_benchmark_module()
    _shrink_benchmark_shape(bench)

    first = bench.make_fixed_data()
    second = bench.make_fixed_data()

    first_vectors = np.asarray(first.spike_vectors)
    second_vectors = np.asarray(second.spike_vectors)
    first_matrices = np.asarray(first.spike_matrices)
    second_matrices = np.asarray(second.spike_matrices)

    assert np.array_equal(first_vectors, second_vectors)
    assert np.array_equal(first_matrices, second_matrices)
    assert len({row.tobytes() for row in first_vectors}) == bench.SPIKE_STEPS
    assert len({row.tobytes() for row in first_matrices}) == bench.SPIKE_STEPS


def test_fixed_data_shape_uses_coba_ei_scale_conn_and_batch_size():
    bench = _load_benchmark_module()
    _shrink_benchmark_shape(bench)

    data = bench.make_fixed_data()

    expected_size = int(bench.COBA_EI_N * bench.SCALE)
    assert data.shape == (expected_size, expected_size)
    assert data.indices.shape == (expected_size, bench.CONN)
    assert data.weights_hetero.shape == (expected_size, bench.CONN)
    assert data.spike_vectors.shape == (bench.SPIKE_STEPS, expected_size)
    assert data.spike_matrices.shape == (bench.SPIKE_STEPS, expected_size, bench.N_B)


def test_hardcoded_conn_interface_builds_scale_conn_batch_backend_cases(monkeypatch):
    bench = _load_benchmark_module()
    monkeypatch.setattr(bench, "scales", [0.01, 0.02])
    monkeypatch.setattr(bench, "conn_nums", [3])
    monkeypatch.setattr(bench, "default_batch_sizes", [5, 7])
    monkeypatch.setattr(bench, "backends", ["jax_raw"])

    cases = bench.build_benchmark_cases()

    assert cases == (
        bench.BenchmarkCase(0.01, 3, 5, "jax_raw"),
        bench.BenchmarkCase(0.01, 3, 7, "jax_raw"),
        bench.BenchmarkCase(0.02, 3, 5, "jax_raw"),
        bench.BenchmarkCase(0.02, 3, 7, "jax_raw"),
    )


def test_benchmark_conn_applies_hardcoded_case_controls(monkeypatch):
    bench = _load_benchmark_module()
    monkeypatch.setattr(bench, "SPIKE_STEPS", 4)
    monkeypatch.setattr(bench, "N_WARMUP", 0)
    monkeypatch.setattr(bench, "N_RUNS", 1)
    monkeypatch.setattr(bench, "PREPROCESS_MODES", ())

    observed = []

    def fake_benchmark_spec(spec):
        observed.append((bench.SCALE, bench.CONN, bench.N_B, bench.BACKENDS, spec.operator))
        return bench.BenchmarkRow(
            spec.operator,
            spec.kind,
            spec.backend,
            spec.preprocess,
            spec.transpose,
            spec.homo,
            0.0,
            0.0,
            0.0,
            True,
            "",
        )

    monkeypatch.setattr(bench, "benchmark_spec", fake_benchmark_spec)
    monkeypatch.setattr(bench, "print_rows", lambda rows: None)

    rows = bench.benchmark_conn(
        scale=0.016,
        conn_num=3,
        batch_size=5,
        backend="jax_raw",
        platform="cpu",
    )

    assert rows
    assert observed
    assert all(item[:4] == (0.016, 3, 5, ("jax_raw",)) for item in observed)


def test_benchmark_conn_accepts_coba_ei_bitpack_a1_post_cuda_case(monkeypatch):
    bench = _load_benchmark_module()
    observed = []

    def fake_run_case(case, *, platform):
        observed.append(
            {
                "case": case,
                "platform": platform,
                "operator_filter": bench.OPERATOR_FILTER,
                "transpose_filter": bench.TRANSPOSE_FILTER,
                "homo_filter": bench.HOMO_FILTER,
            }
        )
        return [
            bench.BenchmarkRow(
                "bitpack_binary_fcnmm",
                "mm",
                case.backend,
                "included",
                False,
                True,
                0.0,
                0.0,
                0.0,
                True,
                "",
            )
        ]

    monkeypatch.setattr(bench, "_runtime_device", lambda platform: "CUDA_TEST_DEVICE")
    monkeypatch.setattr(bench, "_run_benchmark_case", fake_run_case)

    rows = bench.benchmark_conn(
        data_type="bitpack_a1",
        mode="post",
        scale=106,
        conn_num=2515,
        homo=True,
        backend="cuda_raw",
        mv_layout="row_gather",
    )

    assert rows
    assert observed == [
        {
            "case": bench.BenchmarkCase(106, 2515, bench.N_B, "cuda_raw"),
            "platform": bench.PLATFORM,
            "operator_filter": ("bitpack_binary_fcnmm", "bitpack_binary_fcnmm_p"),
            "transpose_filter": (True,),
            "homo_filter": (True,),
        }
    ]


def test_operator_filter_selects_mv_or_mm_for_data_type():
    bench = _load_benchmark_module()

    assert bench._operator_filter_for_data_type("float", kind="mv") == ("fcnmv",)
    assert bench._operator_filter_for_data_type("float", kind="mm") == ("fcnmm",)
    assert bench._operator_filter_for_data_type("binary", kind="mv") == ("binary_fcnmv",)
    assert bench._operator_filter_for_data_type("binary", kind="mm") == ("binary_fcnmm",)
    assert bench._operator_filter_for_data_type("bitpack_a1", kind="mv") == ("bitpack_binary_fcnmv",)
    assert bench._operator_filter_for_data_type("bitpack_a1", kind="mm") == (
        "bitpack_binary_fcnmm",
        "bitpack_binary_fcnmm_p",
    )
    assert bench._operator_filter_for_data_type("compact", kind="mv") == ("compact_binary_fcnmv",)
    assert bench._operator_filter_for_data_type("compact", kind="mm") == ("compact_binary_fcnmm",)


def test_benchmark_conn_accepts_mv_kind_filter(monkeypatch):
    bench = _load_benchmark_module()
    observed = []

    def fake_run_case(case, *, platform):
        observed.append(
            {
                "case": case,
                "platform": platform,
                "operator_filter": bench.OPERATOR_FILTER,
                "transpose_filter": bench.TRANSPOSE_FILTER,
                "homo_filter": bench.HOMO_FILTER,
            }
        )
        return [
            bench.BenchmarkRow(
                "bitpack_binary_fcnmv",
                "mv",
                case.backend,
                "included",
                True,
                True,
                0.0,
                0.0,
                0.0,
                True,
                "",
            )
        ]

    monkeypatch.setattr(bench, "_runtime_device", lambda platform: "CUDA_TEST_DEVICE")
    monkeypatch.setattr(bench, "_run_benchmark_case", fake_run_case)

    rows = bench.benchmark_conn(
        data_type="bitpack_a1",
        kind="mv",
        mode="post",
        scale=106,
        conn_num=2515,
        homo=True,
        backend="cuda_raw",
        mv_layout="row_gather",
    )

    assert rows
    assert observed == [
        {
            "case": bench.BenchmarkCase(106, 2515, bench.N_B, "cuda_raw"),
            "platform": bench.PLATFORM,
            "operator_filter": ("bitpack_binary_fcnmv",),
            "transpose_filter": (True,),
            "homo_filter": (True,),
        }
    ]


def test_default_ncu_case_matches_bitpack_a1_basic_csv_row():
    bench = _load_benchmark_module()

    assert bench.MAIN_DATA_TYPE == "binary"
    assert bench.MAIN_KIND == "mm"
    assert bench.MAIN_MODE == "post"
    assert bench.SCALE == 106
    assert bench.CONN == 2515
    assert bench.N_B == 32
    assert bench.BENCHMARK_STEPS == 3
    assert bench._timed_step_count() == 3
    assert bench.MAIN_HOMO is True
    assert bench.MAIN_BACKEND == "cuda_raw"
    assert bench.MAIN_MV_LAYOUT == "row_gather"
    assert bench.PREPROCESS_MODES == ("prepacked",)
    assert bench.SPIKE_DT_MS == 0.1
    assert np.isclose(bench.SPIKE_RATE, 60.23233413696289 * 0.1 / 1000.0)


def test_mode_filter_matches_coba_ei_primitive_transpose_direction():
    bench = _load_benchmark_module()

    assert bench._transpose_filter_for_mode("post") == (True,)
    assert bench._transpose_filter_for_mode("pre") == (False,)


def test_filtered_bitpack_a1_post_homo_data_omits_unused_large_inputs():
    bench = _load_benchmark_module()
    _shrink_benchmark_shape(bench)
    bench.OPERATOR_FILTER = ("bitpack_binary_fcnmm",)
    bench.TRANSPOSE_FILTER = (False,)
    bench.HOMO_FILTER = (True,)

    data = bench.make_fixed_data()

    assert data.weights_homo is not None
    assert data.weights_hetero is None
    assert data.spike_vectors is None
    assert data.float_vectors is None
    assert data.float_matrices is None
    assert data.spike_matrices.shape == (bench.SPIKE_STEPS, int(bench.COBA_EI_N * bench.SCALE), bench.N_B)


def test_prepacked_bitpack_fcnmm_spec_builds_primitive_operator():
    bench = _load_benchmark_module()
    _shrink_benchmark_shape(bench)
    bench.PREPROCESS_MODES = ("prepacked",)
    bench.OPERATOR_FILTER = ("bitpack_binary_fcnmm_p",)
    bench.TRANSPOSE_FILTER = (True,)
    bench.HOMO_FILTER = (True,)

    data = bench.make_fixed_data()
    specs = bench.build_benchmark_specs(data, platform="cpu")

    assert len(specs) == 1
    spec = specs[0]
    packed, matrix = spec.step_args[0]
    expected_size = int(bench.COBA_EI_N * bench.SCALE)

    assert spec.operator == "bitpack_binary_fcnmm_p"
    assert spec.kind == "mm"
    assert spec.preprocess == "prepacked"
    assert spec.transpose is True
    assert packed.shape == (expected_size, (bench.N_B + 31) // 32)
    assert matrix.shape == (expected_size, bench.N_B)


def test_prepacked_bitpack_fcnmm_spec_calls_primitive_p_call(monkeypatch):
    bench = _load_benchmark_module()
    _shrink_benchmark_shape(bench)
    bench.PREPROCESS_MODES = ("prepacked",)
    bench.OPERATOR_FILTER = ("bitpack_binary_fcnmm_p",)
    bench.TRANSPOSE_FILTER = (True,)
    bench.HOMO_FILTER = (True,)
    calls = []

    def fake_p_call(weights, indices, packed, matrix, **kwargs):
        calls.append(
            {
                "weights": weights,
                "indices": indices,
                "packed": packed,
                "matrix": matrix,
                "kwargs": kwargs,
            }
        )
        out_shape = (kwargs["shape"][1], matrix.shape[1])
        return (bench.jnp.zeros(out_shape, dtype=weights.dtype),)

    monkeypatch.setattr(bench, "bitpack_binary_fcnmm_p_call", fake_p_call)

    data = bench.make_fixed_data()
    spec = bench.build_benchmark_specs(data, platform="cpu")[0]
    result = spec.fn(*spec.step_args[0])

    assert result.shape == (data.shape[1], bench.N_B)
    assert len(calls) == 1
    assert calls[0]["kwargs"]["shape"] == data.shape
    assert calls[0]["kwargs"]["transpose"] is True
    assert calls[0]["kwargs"]["pack_axis"] == bench.PACK_AXIS
    assert calls[0]["kwargs"]["backend"] == spec.backend


def test_print_rows_uses_coba_ei_elapsed_seconds_columns(capsys):
    bench = _load_benchmark_module()

    bench.print_rows(
        [
            bench.BenchmarkRow(
                "bitpack_binary_fcnmm",
                "mm",
                "cuda_raw",
                "included",
                True,
                True,
                0.001,
                0.0002,
                0.0008,
                True,
                "",
            )
        ]
    )

    output = capsys.readouterr().out
    assert "elapsed_s" in output
    assert "single_step_s" in output
    assert "timed_steps" in output
    assert "std_s" in output
    assert "min_s" in output
    assert "mean_ms" not in output
    assert "std_ms" not in output
    assert "min_ms" not in output


def test_print_rows_handles_empty_filtered_spec_list(capsys):
    bench = _load_benchmark_module()

    bench.print_rows([])

    assert "No benchmark rows to display." in capsys.readouterr().out


def test_elapsed_s_scales_single_step_to_coba_batch_duration():
    bench = _load_benchmark_module()
    bench.BENCHMARK_STEPS = None
    row = bench.BenchmarkRow(
        "bitpack_binary_fcnmm",
        "mm",
        "cuda_raw",
        "included",
        True,
        True,
        0.01,
        0.002,
        0.008,
        True,
        "",
    )

    formatted = bench._format_row(row)

    assert formatted["single_step_s"] == "0.010000"
    assert formatted["elapsed_s"] == "5.000000"
    assert formatted["timed_steps"] == "500"
    assert formatted["coba_steps"] == "500"
    assert formatted["conn_updates"] == "1"


def test_elapsed_s_uses_manual_benchmark_steps_when_set():
    bench = _load_benchmark_module()
    bench.BENCHMARK_STEPS = 32
    row = bench.BenchmarkRow(
        "bitpack_binary_fcnmm_p",
        "mm",
        "cuda_raw",
        "prepacked",
        True,
        True,
        0.01,
        0.002,
        0.008,
        True,
        "",
    )

    formatted = bench._format_row(row)

    assert bench._timed_step_count() == 32
    assert formatted["single_step_s"] == "0.010000"
    assert formatted["elapsed_s"] == "0.320000"
    assert formatted["timed_steps"] == "32"
    assert formatted["coba_steps"] == "500"


def test_benchmark_spec_uses_timed_steps_and_cycles_spike_inputs(monkeypatch):
    bench = _load_benchmark_module()
    bench.SPIKE_STEPS = 2
    bench.N_WARMUP = 0
    bench.N_RUNS = 99
    bench.BENCHMARK_STEPS = 5
    observed = []

    def fake_fn(value):
        observed.append(value)
        return value

    spec = bench.BenchmarkSpec(
        "fake_op",
        "mm",
        "fake_backend",
        "prepacked",
        True,
        True,
        fake_fn,
        ((0,), (1,)),
    )
    monkeypatch.setattr(bench.jax, "jit", lambda fn: fn)
    monkeypatch.setattr(bench, "_block_until_ready", lambda output: output)

    row = bench.benchmark_spec(spec)

    assert row.success is True
    assert observed == [0, 0, 1, 0, 1, 0]


def test_benchmark_specs_cover_fcn_family_and_preprocess_modes():
    bench = _load_benchmark_module()
    _shrink_benchmark_shape(bench)
    bench.PREPROCESS_MODES = ("excluded", "included")
    data = bench.make_fixed_data()

    specs = bench.build_benchmark_specs(data, platform="cpu")
    keys = {(spec.operator, spec.preprocess) for spec in specs}

    assert ("fcnmv", "none") in keys
    assert ("fcnmm", "none") in keys
    assert ("binary_fcnmv", "none") in keys
    assert ("binary_fcnmm", "none") in keys
    assert ("bitpack_binary_fcnmv", "excluded") in keys
    assert ("bitpack_binary_fcnmv", "included") in keys
    assert ("bitpack_binary_fcnmm", "excluded") in keys
    assert ("bitpack_binary_fcnmm", "included") in keys
    assert ("compact_binary_fcnmv", "excluded") not in keys
    assert ("compact_binary_fcnmv", "included") not in keys
    assert ("compact_binary_fcnmm", "excluded") not in keys
    assert ("compact_binary_fcnmm", "included") not in keys
