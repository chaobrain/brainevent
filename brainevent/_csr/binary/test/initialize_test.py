# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Tests for ``brainevent/_csr/binary/initialize.py``, grouped by behavior."""

import json
import sys
import types
from itertools import product

import numpy as np
import pytest

import brainevent
from brainevent._csr.binary import initialize
from brainevent._csr.binary.initialize import load_csr_config, save_csr_config
from brainevent._csr.config import CSRConfig, CSRDeviceInfo, _BenchmarkRecord


# Configuration persistence, initialization and benchmarking
# ----------------------------------------------------------

def _has_gpu() -> bool:
    try:
        import jax

        return bool(jax.devices("gpu"))
    except (ImportError, RuntimeError):
        return False


@pytest.fixture(autouse=False)
def _restore_cache_dir():
    cache_dir = brainevent.get_cache_dir()
    try:
        yield
    finally:
        brainevent.set_cache_dir(cache_dir)


@pytest.mark.usefixtures('_restore_cache_dir')
def test_default_config_matches_cu_defaults():
    default_spec = CSRConfig().kernel_args()[0]
    assert default_spec == CSRConfig(512, 1024, 128, 512).kernel_args()[0]
    assert any(candidate.kernel_args()[0] == default_spec
               for candidate in initialize.DEFAULT_CANDIDATES)


@pytest.mark.usefixtures('_restore_cache_dir')
def test_hybrid_task_capacity_matches_cuda_chunking(monkeypatch):
    # threshold 128, nnz 4096 -> ceil(len/4096) for rows longer than 128, else 0.
    monkeypatch.setenv(
        initialize._ENV_OVERRIDE,
        '{"block_size":256,"fixed_scatter_blocks":2048,"tpr_threshold":128,"task_nnz":4096}',
    )
    # cumulative indptr for row lengths [0, 128, 129, 4096, 4097, 8192]
    lengths = [0, 128, 129, 4096, 4097, 8192]
    indptr = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    cfg = initialize.load_csr_config()
    assert cfg.mv_capacity(int(indptr[-1]), len(lengths), max(lengths)) >= 0 + 0 + 1 + 1 + 2 + 2


@pytest.mark.usefixtures('_restore_cache_dir')
def test_env_override_and_task_constants_stay_in_lockstep(monkeypatch):
    monkeypatch.setenv(
        initialize._ENV_OVERRIDE,
        '{"block_size":256,"fixed_scatter_blocks":2048,"tpr_threshold":256,"task_nnz":512}',
    )
    cfg = initialize.load_csr_config()
    assert (cfg.tpr_threshold, cfg.task_nnz) == (256, 512)
    # row lengths [256, 257, 1024, 1025] with threshold 256, nnz 512
    lengths = [256, 257, 1024, 1025]
    indptr = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    assert cfg.mv_capacity(int(indptr[-1]), len(lengths), max(lengths)) >= 0 + 1 + 2 + 3
    assert "-DBE_HYBRID_TASK_NNZ=512" in cfg.compile_flags()


@pytest.mark.usefixtures('_restore_cache_dir')
def test_save_and_reload_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("BRAINEVENT_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv(initialize._ENV_OVERRIDE, raising=False)
    # Point the shared cache instance at tmp_path so get_cache_dir() follows.
    brainevent.set_cache_dir(str(tmp_path))

    cfg = CSRConfig(512, 4096, 256, 8192)
    initialize.save_csr_config(cfg, device_kind="TEST_GPU")

    monkeypatch.setattr(initialize, "_current_device_kind", lambda: "TEST_GPU")
    assert initialize.load_csr_config().kernel_args()[0] == cfg.kernel_args()[0]

    # A different device falls back to defaults.
    monkeypatch.setattr(initialize, "_current_device_kind", lambda: "OTHER_GPU")
    assert initialize.load_csr_config().kernel_args()[0] == CSRConfig().kernel_args()[0]


@pytest.mark.usefixtures('_restore_cache_dir')
def test_init_csr_config_is_exported():
    assert brainevent.init_csr_config is initialize.init_csr_config


@pytest.mark.usefixtures('_restore_cache_dir')
def test_default_candidates_are_the_complete_ordered_cartesian_grid():
    actual = [
        (candidate.block_size, candidate.fixed_scatter_blocks,
         candidate.tpr_threshold, candidate.task_nnz)
        for candidate in initialize.DEFAULT_CANDIDATES
    ]
    expected = list(product((512, 256), (512, 1024, 2048),
                            (128, 256), (512, 1024, 2048, 4096)))
    assert actual == expected
    assert len(actual) == len(set(actual)) == 48
    assert (512, 1024, 128, 512) in actual


@pytest.mark.usefixtures('_restore_cache_dir')
def test_run_benchmark_shows_progress_without_candidate_details(monkeypatch, capsys):
    fake_jax = types.ModuleType("jax")
    fake_jnp = types.ModuleType("jax.numpy")
    fake_jnp.asarray = np.asarray
    fake_jnp.float32 = np.float32
    fake_jnp.int32 = np.int32
    fake_jnp.bool_ = np.bool_
    fake_jax.block_until_ready = lambda value: value
    fake_jax.numpy = fake_jnp
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", fake_jnp)
    monkeypatch.setattr(
        initialize,
        "_make_uniform_csr",
        lambda **kwargs: (
            np.array([0], dtype=np.int32),
            np.array([0, 1], dtype=np.int32),
        ),
    )
    monkeypatch.setattr(
        initialize,
        "_make_spike_batch",
        lambda **kwargs: np.array([[True]], dtype=np.bool_),
    )

    candidates = (
        CSRConfig(128, 1024, 128, 2048),
        CSRConfig(256, 2048, 512, 2048),
    )
    timings = {
        candidates[0].module_suffix(): (12.345, 123.450),
        candidates[1].module_suffix(): (67.890, 678.900),
    }

    def fake_benchmark_config(config, **kwargs):
        elapsed_ms, per_call_us = timings[config.module_suffix()]
        return {
            "config": config,
            "elapsed_ms": elapsed_ms,
            "per_call_us": per_call_us,
            "task_capacity": 0,
        }

    monkeypatch.setattr(initialize, "_benchmark_config", fake_benchmark_config)

    initialize.run_benchmark(
        n_pre=1,
        n_post=1,
        conn=1,
        batch_size=1,
        candidates=candidates,
    )

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert captured.out == ""
    assert "CSR hybrid tuning" in captured.err
    assert "2/2" in captured.err
    assert "100.0%" in captured.err
    for candidate, (elapsed_ms, per_call_us) in timings.items():
        assert str(candidate) not in combined
        assert f"{elapsed_ms:.3f} ms" not in combined
        assert f"{per_call_us:.3f} us/call" not in combined


@pytest.mark.usefixtures('_restore_cache_dir')
def test_run_benchmark_defaults_to_200_steps_and_50_warmup_while_reusing_spike_batch(monkeypatch):
    fake_jax = types.ModuleType("jax")
    fake_jnp = types.ModuleType("jax.numpy")
    fake_jnp.asarray = np.asarray
    fake_jnp.float32 = np.float32
    fake_jnp.int32 = np.int32
    fake_jnp.bool_ = np.bool_
    fake_jax.block_until_ready = lambda value: value
    fake_jax.numpy = fake_jnp
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", fake_jnp)
    monkeypatch.setattr(
        initialize,
        "_make_uniform_csr",
        lambda **kwargs: (
            np.array([0], dtype=np.int32),
            np.array([0, 1], dtype=np.int32),
        ),
    )

    spike_batch_sizes = []

    def fake_make_spike_batch(**kwargs):
        spike_batch_sizes.append(kwargs["batch_size"])
        return np.ones((kwargs["batch_size"], kwargs["n_pre"]), dtype=np.bool_)

    captured_kwargs = []

    def fake_benchmark_config(config, **kwargs):
        captured_kwargs.append(kwargs)
        return {
            "config": config,
            "elapsed_ms": 12.345,
            "per_call_us": 12.345,
            "task_capacity": 0,
        }

    monkeypatch.setattr(initialize, "_make_spike_batch", fake_make_spike_batch)
    monkeypatch.setattr(initialize, "_benchmark_config", fake_benchmark_config)

    initialize.run_benchmark(
        n_pre=1,
        n_post=1,
        conn=1,
        candidates=(CSRConfig(256, 2048, 512, 2048),),
        show_progress=False,
    )

    assert spike_batch_sizes == [100]
    assert captured_kwargs[0]["batch_size"] == 100
    assert captured_kwargs[0]["steps"] == 200
    assert captured_kwargs[0]["warmup"] == 50


@pytest.mark.usefixtures('_restore_cache_dir')
def test_benchmark_config_cycles_spike_batch_for_timed_steps(monkeypatch):
    import brainevent._op as op

    fake_jax = types.ModuleType("jax")
    fake_jnp = types.ModuleType("jax.numpy")
    fake_jnp.dtype = np.dtype
    fake_jnp.empty = np.empty
    fake_jnp.int32 = np.int32
    fake_jax.ShapeDtypeStruct = lambda shape, dtype: types.SimpleNamespace(shape=shape, dtype=dtype)
    fake_jax.block_until_ready = lambda value: value
    fake_jax.device_get = lambda value: value
    fake_jax.jit = lambda fn: fn

    vectors = []

    def fake_ffi_call(target_name, outs, input_output_aliases=None):
        def run(weights, indices, indptr, vector, task_begin, task_end, status, *, task_capacity):
            vectors.append(tuple(np.asarray(vector).tolist()))
            return (
                np.empty((2,), dtype=weights.dtype),
                task_begin,
                task_end,
                np.array([0, 0], dtype=np.int32),
            )

        return run

    fake_jax.ffi = types.SimpleNamespace(ffi_call=fake_ffi_call)
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", fake_jnp)
    monkeypatch.setattr(
        op,
        "load_cuda_file",
        lambda *args, **kwargs: types.SimpleNamespace(
            function_names={initialize._TARGET_FUNCTION},
            path="fake.so",
        ),
    )
    counter = iter((1.0, 1.5))
    monkeypatch.setattr(initialize.time, "perf_counter", lambda: next(counter))

    result = initialize._benchmark_config(
        CSRConfig(256, 2048, 512, 2048),
        weights=np.array([1.0], dtype=np.float32),
        indices=np.array([0], dtype=np.int32),
        indptr=np.array([0, 1], dtype=np.int32),
        spikes=np.array(
            [
                [True, False],
                [False, True],
                [True, True],
            ],
            dtype=np.bool_,
        ),
        n_pre=2,
        n_post=2,
        conn=1,
        batch_size=3,
        warmup=0,
        steps=5,
        force_rebuild=False,
        verbose_compile=False,
    )

    assert vectors == [
        (True, False),  # Untimed compilation even with warmup=0.
        (True, False),
        (False, True),
        (True, True),
        (True, False),
        (False, True),
    ]
    assert result["elapsed_ms"] == 500.0
    assert result["per_call_us"] == 100000.0


@pytest.mark.usefixtures('_restore_cache_dir')
def test_init_csr_config_does_not_print_benchmark_details(monkeypatch, capsys):
    cfg = CSRConfig(256, 2048, 512, 2048)
    monkeypatch.setattr(
        initialize,
        "run_benchmark",
        lambda **kwargs: [
            {
                "config": cfg,
                "elapsed_ms": 12.345,
                "per_call_us": 123.450,
                "task_capacity": 0,
            }
        ],
    )

    results = initialize.init_csr_config(save=False)
    assert len(results) == 2
    assert [item["metadata"]["requested_density"] for item in results] == [0.03, 0.003]
    assert all(item["best_config"].mv == cfg.mv for item in results)

    captured = capsys.readouterr()
    assert captured.out == ""


@pytest.mark.usefixtures('_restore_cache_dir')
def test_init_csr_config_saves_records_for_load_csr_config_print(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("BRAINEVENT_CACHE_DIR", str(tmp_path))
    brainevent.set_cache_dir(str(tmp_path))
    monkeypatch.setattr(initialize, "_current_device_kind", lambda: "TEST_GPU")

    best = CSRConfig(256, 2048, 512, 2048)
    other = CSRConfig(128, 1024, 128, 2048)
    monkeypatch.setattr(
        initialize,
        "run_benchmark",
        lambda **kwargs: [
            {
                "config": best,
                "elapsed_ms": 12.345,
                "per_call_us": 123.450,
                "task_capacity": 0,
            },
            {
                "config": other,
                "elapsed_ms": 67.890,
                "per_call_us": 678.900,
                "task_capacity": 1,
            },
        ],
    )

    results = initialize.init_csr_config(n_pre=100, n_post=1000)
    assert all(item["best_config"].mv == best.mv for item in results)
    assert capsys.readouterr().out == ""

    output = str(initialize.load_csr_config(matrix_metadata=results[0]["metadata"]))
    assert output.startswith(str(best))
    assert str(best) in output
    assert "12.345 ms" in output
    assert "123.450 us/call" in output
    assert str(other) in output
    assert "67.890 ms" in output
    assert "678.900 us/call" in output


@pytest.mark.usefixtures('_restore_cache_dir')
@pytest.mark.slow
@pytest.mark.skipif(not _has_gpu(), reason="CSR tuning smoke requires a GPU")
def test_gpu_smoke_runs_two_small_configs_when_gpu_is_available():
    candidates = (
        CSRConfig(128, 1024, 128, 2048),
        CSRConfig(256, 2048, 128, 4096),
    )
    try:
        results = initialize.run_benchmark(
            n_pre=64,
            n_post=128,
            conn=32,
            batch_size=4,
            spike_sparsity=0.05,
            seed=123,
            warmup=1,
            candidates=candidates,
        )
    except Exception as exc:  # noqa: BLE001 - backend-specific exception types.
        message = str(exc)
        if "RESOURCE_EXHAUSTED" in message or "CUDA_ERROR_OUT_OF_MEMORY" in message:
            pytest.skip(f"GPU memory unavailable for CUDA FFI smoke: {message}")
        raise

    assert len(results) == 2
    assert results[0]["elapsed_ms"] <= results[1]["elapsed_ms"]
    assert {r["config"].block_size for r in results} == {128, 256}


# Persisted schema, environment overrides and recovery
# ----------------------------------------------------

_persistence_DEVICE = CSRDeviceInfo('gpu', 'TEST_GPU', l2_bytes=4096, sms=16)


def _persistence_prepared(lengths=(16, 17, 0, 513, 1025), cols=128, **kwargs):
    indptr = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    indices = (np.arange(indptr[-1]) % cols).astype(np.int32)
    config = CSRConfig(device_info=_persistence_DEVICE, **kwargs).prepare_structure(
        indices, indptr, shape=(len(lengths), cols))
    return config, indices, indptr


@pytest.fixture
def _persistence_isolated_store(monkeypatch, tmp_path):
    old = brainevent.get_cache_dir()
    monkeypatch.delenv('BRAINEVENT_CSR_HYBRID_CONFIG', raising=False)
    brainevent.set_cache_dir(str(tmp_path))
    try:
        yield tmp_path
    finally:
        brainevent.set_cache_dir(old)


def test_persistence_keeps_old_schema_and_excludes_runtime_plans(_persistence_isolated_store):
    config, _, _ = _persistence_prepared(block_size=128)
    record = _BenchmarkRecord(config, 4.0, 20.0, 10)
    path = save_csr_config(config, 'TEST_GPU', [record])
    save_csr_config(CSRConfig(512), 'OTHER_GPU')
    stored = json.loads(path.read_text())
    assert set(stored) == {'TEST_GPU', 'OTHER_GPU'}
    assert set(stored['TEST_GPU']) == {'block_size', 'fixed_scatter_blocks', 'tpr_threshold',
                                       'task_nnz', 'benchmark_records'}
    assert path.name == 'csr_hybrid_config.json'
    loaded = load_csr_config('TEST_GPU')
    assert loaded.block_size == 128
    assert loaded.primary is loaded.mirror is loaded.device_info is None
    assert loaded.benchmark_records[0].elapsed_ms == 4.0


def test_load_returns_fresh_snapshots_and_respects_environment(_persistence_isolated_store, monkeypatch):
    first = CSRConfig(128)
    path = save_csr_config(first, 'TEST_GPU')
    snapshot = load_csr_config('TEST_GPU')
    save_csr_config(CSRConfig(512), 'TEST_GPU')
    assert snapshot.block_size == 128 and load_csr_config('TEST_GPU').block_size == 512
    monkeypatch.setenv('BRAINEVENT_CSR_HYBRID_CONFIG', json.dumps(
        dict(block_size=256, fixed_scatter_blocks=1024, tpr_threshold=256, task_nnz=512)))
    assert load_csr_config('TEST_GPU').task_nnz == 512
    monkeypatch.delenv('BRAINEVENT_CSR_HYBRID_CONFIG')
    path.write_text('corrupt')
    assert load_csr_config('TEST_GPU').kernel_args()[0] == CSRConfig().kernel_args()[0]
    assert load_csr_config('MISSING').kernel_args()[0] == CSRConfig().kernel_args()[0]


def test_persistence_recovers_invalid_store_but_rejects_invalid_override(_persistence_isolated_store, monkeypatch):
    path = _persistence_isolated_store / 'csr_hybrid_config.json'
    path.write_text('[]')
    assert load_csr_config('TEST_GPU').kernel_args()[0] == CSRConfig().kernel_args()[0]
    with pytest.warns(UserWarning, match='corrupt'):
        save_csr_config(CSRConfig(), 'TEST_GPU')
    path.write_text('broken')
    with pytest.warns(UserWarning, match='corrupt'):
        save_csr_config(CSRConfig(), 'TEST_GPU')
    monkeypatch.setenv('BRAINEVENT_CSR_HYBRID_CONFIG', '{')
    with pytest.raises(ValueError):
        load_csr_config('TEST_GPU')
    with pytest.raises(RuntimeError, match='device_kind'):
        save_csr_config(CSRConfig(), '')


def _metadata(rows=20, cols=1000, conn=30):
    return {'shape': (rows, cols), 'nnz': rows * conn}


def _timing(config, elapsed_ms=1.0):
    return {'config': config, 'elapsed_ms': elapsed_ms,
            'per_call_us': elapsed_ms * 5, 'task_capacity': 0,
            'status': 'ok', 'checked': True}


def test_workload_roundtrips_keep_both_winners_legacy_fallback_and_other_devices(
    _persistence_isolated_store,
):
    legacy = CSRConfig(128)
    dense, sparse, another_size = CSRConfig(256), CSRConfig(512, 2048), CSRConfig(256, 512)
    path = save_csr_config(legacy, 'TEST_GPU')
    save_csr_config(CSRConfig(128, 4096), 'OTHER_GPU')
    for config, metadata in ((dense, _metadata()), (sparse, _metadata(conn=3)),
                             (another_size, _metadata(rows=40))):
        save_csr_config(config, 'TEST_GPU', [_timing(config)], matrix_metadata=metadata)

    assert load_csr_config('TEST_GPU').mv == legacy.mv
    assert load_csr_config('TEST_GPU', matrix_metadata=_metadata()).mv == dense.mv
    assert load_csr_config('TEST_GPU', matrix_metadata=_metadata(conn=3)).mv == sparse.mv
    assert load_csr_config('TEST_GPU', matrix_metadata=_metadata(rows=40)).mv == another_size.mv
    assert load_csr_config('OTHER_GPU').mv == CSRConfig(128, 4096).mv

    replacement = CSRConfig(256, 1024, 256, 4096)
    save_csr_config(replacement, 'TEST_GPU', [_timing(replacement, 2.0)],
                    matrix_metadata=_metadata())
    stored = json.loads(path.read_text())
    assert len(stored['TEST_GPU']['workloads']) == 3
    assert stored['TEST_GPU']['block_size'] == 128
    assert load_csr_config('TEST_GPU', matrix_metadata=_metadata()).mv == replacement.mv
    assert load_csr_config('TEST_GPU', matrix_metadata=_metadata(conn=3)).mv == sparse.mv
    assert load_csr_config('OTHER_GPU').mv == CSRConfig(128, 4096).mv
    record = load_csr_config('TEST_GPU', matrix_metadata=_metadata()).benchmark_records[0]
    assert record.elapsed_ms == 2.0


def test_loader_uses_environment_exact_legacy_other_then_builtin(
    _persistence_isolated_store, monkeypatch,
):
    legacy, matched, other, explicit = (CSRConfig(128), CSRConfig(256),
                                        CSRConfig(512, 512), CSRConfig(256, 2048))
    path = save_csr_config(legacy, 'TEST_GPU')
    save_csr_config(matched, 'TEST_GPU', matrix_metadata=_metadata())
    save_csr_config(other, 'TEST_GPU', matrix_metadata=_metadata(conn=3))
    monkeypatch.setenv(initialize._ENV_OVERRIDE, json.dumps({
        name: getattr(explicit, name) for name in initialize._CONFIG_FIELDS
    }))
    assert load_csr_config('TEST_GPU', matrix_metadata=_metadata()).mv == explicit.mv
    monkeypatch.delenv(initialize._ENV_OVERRIDE)
    assert load_csr_config('TEST_GPU', matrix_metadata=_metadata()).mv == matched.mv
    assert load_csr_config('TEST_GPU', matrix_metadata=_metadata(rows=99)).mv == legacy.mv

    stored = json.loads(path.read_text())
    for workload in stored['TEST_GPU']['workloads'].values():
        if workload['metadata']['nnz'] == _metadata()['nnz']:
            workload['best_config']['block_size'] = 0
    path.write_text(json.dumps(stored))
    assert load_csr_config('TEST_GPU', matrix_metadata=_metadata()).mv == legacy.mv

    for name in initialize._CONFIG_FIELDS:
        stored['TEST_GPU'].pop(name)
    path.write_text(json.dumps(stored))
    assert load_csr_config('TEST_GPU', matrix_metadata=_metadata()).mv == other.mv
    stored['TEST_GPU']['workloads'].clear()
    path.write_text(json.dumps(stored))
    assert load_csr_config('TEST_GPU', matrix_metadata=_metadata()).mv == CSRConfig().mv
    assert load_csr_config('MISSING_GPU', matrix_metadata=_metadata()).mv == CSRConfig().mv


@pytest.mark.parametrize('metadata', [None, {}, {'shape': (20, 1000)},
                                      {'shape': (0, 1000), 'nnz': 0}])
def test_nonmatching_fallback_is_deterministic_without_complete_metadata(
    _persistence_isolated_store, metadata,
):
    dense, sparse = CSRConfig(256), CSRConfig(512, 512)
    path = save_csr_config(dense, 'TEST_GPU', matrix_metadata=_metadata())
    save_csr_config(sparse, 'TEST_GPU', matrix_metadata=_metadata(conn=3))
    assert load_csr_config('TEST_GPU', matrix_metadata=metadata).mv == sparse.mv
    stored = json.loads(path.read_text())
    workloads = stored['TEST_GPU']['workloads']
    stored['TEST_GPU']['workloads'] = dict(reversed(list(workloads.items())))
    path.write_text(json.dumps(stored))
    assert load_csr_config('TEST_GPU', matrix_metadata=metadata).mv == sparse.mv


def test_invalid_diagnostic_records_do_not_discard_valid_workload_parameters(
    _persistence_isolated_store,
):
    config = CSRConfig(256, 512)
    path = save_csr_config(config, 'TEST_GPU', matrix_metadata=_metadata())
    stored = json.loads(path.read_text())
    workload = next(iter(stored['TEST_GPU']['workloads'].values()))
    workload['benchmark_records'] = [{'config': {'block_size': 'broken'}}]
    path.write_text(json.dumps(stored))
    loaded = load_csr_config('TEST_GPU', matrix_metadata=_metadata())
    assert loaded.mv == config.mv
    assert loaded.benchmark_records == ()


def test_init_measures_all_candidates_separately_and_can_select_different_winners(monkeypatch):
    calls = []

    def measure(**kwargs):
        calls.append(kwargs)
        candidates = kwargs['candidates']
        winner = candidates[0] if kwargs['density'] == .03 else candidates[-1]
        return sorted([_timing(cfg, 1.0 if cfg == winner else 2.0) for cfg in candidates],
                      key=lambda record: record['elapsed_ms'])

    monkeypatch.setattr(initialize, 'run_benchmark', measure)
    results = initialize.init_csr_config(save=False, n_pre=100, n_post=1000)
    assert [call['density'] for call in calls] == [.03, .003]
    assert all((call['n_pre'], call['n_post']) == (100, 1000) for call in calls)
    assert all(tuple(call['candidates']) == initialize.DEFAULT_CANDIDATES for call in calls)
    assert [item['metadata']['conn'] for item in results] == [30, 3]
    assert [item['metadata']['nnz'] for item in results] == [3000, 300]
    assert [item['best_config'].mv for item in results] == [
        initialize.DEFAULT_CANDIDATES[0].mv, initialize.DEFAULT_CANDIDATES[-1].mv]
    assert all(len(item['records']) == 48 and item['status'] == 'ok' for item in results)


def test_init_failed_density_keeps_previous_workload_and_saves_successful_one(
    _persistence_isolated_store, monkeypatch,
):
    old_dense, new_sparse = CSRConfig(128), CSRConfig(256)
    save_csr_config(old_dense, 'TEST_GPU', matrix_metadata=_metadata(rows=100))

    def measure(**kwargs):
        if kwargs['density'] == .03:
            raise MemoryError('not enough memory for the requested matrix')
        return [_timing(new_sparse)]

    monkeypatch.setattr(initialize, 'run_benchmark', measure)
    results = initialize.init_csr_config(device_kind='TEST_GPU', n_pre=100, n_post=1000)
    assert results[0]['status'] == 'error'
    assert results[0]['best_config'] is None
    assert 'memory' in results[0]['error']
    assert results[1]['best_config'].mv == new_sparse.mv
    assert load_csr_config('TEST_GPU', matrix_metadata=_metadata(rows=100)).mv == old_dense.mv
    assert load_csr_config('TEST_GPU', matrix_metadata=_metadata(rows=100, conn=3)).mv == new_sparse.mv


def test_rounded_densities_keep_independent_records_even_with_the_same_winner(
    _persistence_isolated_store, monkeypatch,
):
    winner = CSRConfig(256)
    monkeypatch.setattr(initialize, 'run_benchmark', lambda **kwargs: [_timing(winner)])
    results = initialize.init_csr_config(device_kind='TEST_GPU', n_pre=2, n_post=10)
    assert [item['metadata']['nnz'] for item in results] == [2, 2]
    path = _persistence_isolated_store / initialize.CONFIG_FILENAME
    stored = json.loads(path.read_text())['TEST_GPU']['workloads']
    assert len(stored) == 2
    assert {item['metadata']['requested_density'] for item in stored.values()} == {.03, .003}
    initialize.init_csr_config(device_kind='TEST_GPU', n_pre=2, n_post=10, densities=(.03,))
    updated = json.loads(path.read_text())['TEST_GPU']['workloads']
    assert len(updated) == 2
    assert {item['metadata']['requested_density'] for item in updated.values()} == {.03, .003}


def test_all_failed_candidates_leave_the_previous_workload_intact(
    _persistence_isolated_store, monkeypatch,
):
    previous, candidate = CSRConfig(128), CSRConfig(256)
    metadata = dict(_metadata(rows=100), requested_density=.03)
    save_csr_config(previous, 'TEST_GPU', matrix_metadata=metadata)

    def measure(**kwargs):
        assert tuple(kwargs['candidates']) == (candidate,)
        return [{'config': candidate, 'status': 'error', 'error': 'task queue overflowed'}]

    monkeypatch.setattr(initialize, 'run_benchmark', measure)
    results = initialize.init_csr_config(device_kind='TEST_GPU', n_pre=100, n_post=1000,
                                         densities=(.03,), candidates=(candidate,))
    assert results[0]['status'] == 'error' and results[0]['best_config'] is None
    assert results[0]['records'][0]['error'] == 'task queue overflowed'
    assert load_csr_config('TEST_GPU', matrix_metadata=metadata).mv == previous.mv


@pytest.mark.parametrize('kwargs', [
    {'conn': 1, 'density': .03}, {'density': 0}, {'density': -0.1}, {'density': 1.1},
    {'n_pre': 0}, {'n_post': 0}, {'steps': 0}, {'batch_size': 0}, {'candidates': ()},
])
def test_invalid_benchmark_requests_fail_before_matrix_allocation(monkeypatch, kwargs):
    def allocate(**unused):
        pytest.fail('invalid workload reached matrix allocation')

    monkeypatch.setattr(initialize, '_make_uniform_csr', allocate)
    with pytest.raises(ValueError):
        initialize.run_benchmark(show_progress=False, **kwargs)


def test_large_density_requires_int64_support_before_allocation(monkeypatch):
    import jax

    def allocate(**unused):
        pytest.fail('large workload allocated while x64 offsets were disabled')

    monkeypatch.setattr(initialize, '_make_uniform_csr', allocate)
    was_enabled = jax.config.jax_enable_x64
    try:
        jax.config.update('jax_enable_x64', False)
        with pytest.raises(RuntimeError, match='64|x64'):
            initialize.run_benchmark(n_pre=500_000, n_post=500_000, density=.03,
                                     show_progress=False)
    finally:
        jax.config.update('jax_enable_x64', was_enabled)


def test_missing_x64_marks_only_large_density_failed_and_continues(monkeypatch):
    import jax

    run_benchmark = initialize.run_benchmark
    winner = CSRConfig(256)
    calls = []

    def allocate(**unused):
        pytest.fail('x64-incompatible workload reached allocation')

    def measure(**kwargs):
        calls.append(kwargs['density'])
        if kwargs['density'] == .03:
            return run_benchmark(**kwargs)
        return [_timing(winner)]

    monkeypatch.setattr(initialize, '_make_uniform_csr', allocate)
    monkeypatch.setattr(initialize, 'run_benchmark', measure)
    was_enabled = jax.config.jax_enable_x64
    try:
        jax.config.update('jax_enable_x64', False)
        # 500k x 500k at 3% exceeds int32 offsets; 0.3% does not.
        results = initialize.init_csr_config(save=False, show_progress=False, scale=500.0)
    finally:
        jax.config.update('jax_enable_x64', was_enabled)
    assert calls == [.03, .003]
    assert results[0]['status'] == 'error' and results[0]['best_config'] is None
    assert 'x64' in results[0]['error']
    assert results[1]['status'] == 'ok' and results[1]['best_config'].mv == winner.mv


def test_failed_candidate_is_recorded_without_discarding_successful_candidate(monkeypatch):
    bad, good = CSRConfig(512), CSRConfig(256)

    def measure(config, **kwargs):
        if config == bad:
            raise RuntimeError('CUDA hybrid task queue overflowed')
        return _timing(config)

    monkeypatch.setattr(initialize, '_benchmark_config', measure)
    records = initialize.run_benchmark(n_pre=4, n_post=1000, density=.03,
                                       candidates=(bad, good), batch_size=2, steps=3,
                                       warmup=1, show_progress=False)
    assert len(records) == 2
    assert records[0]['config'] == good and records[0]['status'] == 'ok'
    assert records[1]['config'] == bad and records[1]['status'] == 'error'
    assert 'overflow' in records[1]['error']
    assert 'elapsed_ms' not in records[1]


def test_large_workspace_candidate_does_not_prevent_smaller_candidate_from_running(monkeypatch):
    fake_jax, fake_jnp = types.ModuleType('jax'), types.ModuleType('jax.numpy')
    fake_jnp.asarray, fake_jnp.float32 = np.asarray, np.float32
    fake_jnp.int32, fake_jnp.bool_ = np.int32, np.bool_
    fake_jax.block_until_ready = lambda value: value
    # Input/output buffers need 20040 bytes; only the smaller task queue fits.
    fake_jax.devices = lambda: [types.SimpleNamespace(
        memory_stats=lambda: {'bytes_limit': 20080, 'bytes_in_use': 0})]
    monkeypatch.setitem(sys.modules, 'jax', fake_jax)
    monkeypatch.setitem(sys.modules, 'jax.numpy', fake_jnp)
    large, small = CSRConfig(task_nnz=512), CSRConfig(task_nnz=4096)
    measured = []

    def measure(config, **kwargs):
        measured.append(config.mv)
        if config is large:
            raise MemoryError('candidate task workspace exceeds remaining memory')
        return _timing(config)

    monkeypatch.setattr(initialize, '_benchmark_config', measure)
    records = initialize.run_benchmark(n_pre=4, n_post=1000, conn=1000,
                                       batch_size=2, candidates=(large, small),
                                       steps=1, warmup=0, show_progress=False)
    assert measured == [large.mv, small.mv]
    assert records[0]['config'] is small and records[0]['status'] == 'ok'
    assert records[1]['config'] is large and records[1]['status'] == 'error'


@pytest.mark.parametrize('steps, expected', [(3, [2., 0., 1., 0.]),
                                             (4, [0., 1., 2., 0.])])
def test_reference_counts_duplicate_edges_for_the_last_timed_vector(monkeypatch, steps, expected):
    monkeypatch.setattr(initialize, '_make_uniform_csr', lambda **kwargs: (
        np.array([0, 0, 2, 1, 2, 2], np.int32), np.array([0, 3, 6], np.int32)))
    monkeypatch.setattr(initialize, '_make_spike_batch', lambda **kwargs: np.array(
        [[True, False], [False, True]], np.bool_))
    references = []

    def measure(config, **kwargs):
        references.append(kwargs['expected_output'])
        return _timing(config)

    monkeypatch.setattr(initialize, '_benchmark_config', measure)
    initialize.run_benchmark(n_pre=2, n_post=4, conn=3, batch_size=2, steps=steps,
                             candidates=(CSRConfig(),), warmup=0, show_progress=False)
    np.testing.assert_array_equal(references, [expected])


def test_paired_initialization_rejects_conflicting_conn_without_running_benchmark(monkeypatch):
    def measure(**unused):
        pytest.fail('conflicting conn reached paired benchmark')

    monkeypatch.setattr(initialize, 'run_benchmark', measure)
    with pytest.raises((TypeError, ValueError), match='conn'):
        initialize.init_csr_config(conn=2000, save=False)


@pytest.mark.parametrize('output, overflow, expected_error', [
    ([1., 0.], 0, None),
    ([0., 0.], 0, 'output|correct|numer|mismatch'),
    ([1., 0.], 1, 'overflow'),
])
def test_benchmark_validates_output_and_queue_after_timing(
    monkeypatch, output, overflow, expected_error,
):
    import brainevent._op as op

    fake_jax = types.ModuleType('jax')
    fake_jnp = types.ModuleType('jax.numpy')
    fake_jnp.dtype, fake_jnp.empty, fake_jnp.int32 = np.dtype, np.empty, np.int32
    fake_jax.ShapeDtypeStruct = lambda shape, dtype: types.SimpleNamespace(shape=shape, dtype=dtype)
    fake_jax.block_until_ready = fake_jax.device_get = lambda value: value
    fake_jax.jit = lambda function: function

    def fake_ffi(*unused, **ignored):
        def call(weights, indices, indptr, vector, task_begin, task_end, status, *, task_capacity):
            return (np.asarray(output, np.float32), task_begin, task_end,
                    np.asarray([0, overflow], np.int32))
        return call

    fake_jax.ffi = types.SimpleNamespace(ffi_call=fake_ffi)
    monkeypatch.setitem(sys.modules, 'jax', fake_jax)
    monkeypatch.setitem(sys.modules, 'jax.numpy', fake_jnp)
    monkeypatch.setattr(op, 'load_cuda_file', lambda *args, **kwargs: types.SimpleNamespace(
        function_names={initialize._TARGET_FUNCTION}, path='fake.so'))
    clock = iter((1.0, 1.5))
    monkeypatch.setattr(initialize.time, 'perf_counter', lambda: next(clock))
    kwargs = dict(weights=np.array([1.0], np.float32), indices=np.array([0, 1], np.int32),
                  indptr=np.array([0, 1, 2], np.int32),
                  spikes=np.array([[True, False]], np.bool_), n_pre=2, n_post=2, conn=1,
                  batch_size=1, warmup=0, steps=3, force_rebuild=False, verbose_compile=False,
                  expected_output=np.array([1., 0.], np.float32))
    if expected_error:
        with pytest.raises(RuntimeError, match=expected_error):
            initialize._benchmark_config(CSRConfig(), **kwargs)
    else:
        record = initialize._benchmark_config(CSRConfig(), **kwargs)
        assert record['checked'] is True
        assert record['elapsed_ms'] == 500.


# Workload matching, sweep robustness and store integrity
# --------------------------------------------------------

def test_default_tuning_workload_fits_int32_offsets_and_modest_memory():
    import inspect

    for function in (initialize.run_benchmark, initialize.init_csr_config):
        scale = inspect.signature(function).parameters['scale'].default
        rows, cols = initialize._workload_shape(scale, None, None)
        for density in initialize.DEFAULT_DENSITIES:
            nnz = rows * initialize._density_conn(cols, density)
            assert nnz <= initialize._INT32_MAX
            assert 4 * nnz <= 1 << 30  # int32 indices of the largest default matrix


def test_nonexact_workload_uses_nearest_mean_row_length(_persistence_isolated_store):
    dense, sparse = CSRConfig(256), CSRConfig(512, 512)
    save_csr_config(dense, 'TEST_GPU', matrix_metadata=_metadata(conn=30))
    save_csr_config(sparse, 'TEST_GPU', matrix_metadata=_metadata(conn=3))
    # 25 edges per row is nearest to 30 (log scale); 4 edges per row to 3.
    assert load_csr_config('TEST_GPU', matrix_metadata=_metadata(rows=40, cols=2000, conn=25)).mv == dense.mv
    assert load_csr_config('TEST_GPU', matrix_metadata=_metadata(rows=10, conn=4)).mv == sparse.mv
    assert load_csr_config('TEST_GPU', matrix_metadata={'shape': (20, 1000), 'nnz': 999}).mv == dense.mv
    # Equal row lengths (30 edges): the nearer density decides.
    wide = CSRConfig(512, 1024)
    save_csr_config(wide, 'TEST_GPU', matrix_metadata=_metadata(cols=10000, conn=30))
    assert load_csr_config('TEST_GPU', matrix_metadata=_metadata(rows=40, cols=2000, conn=30)).mv == dense.mv
    assert load_csr_config('TEST_GPU', matrix_metadata=_metadata(rows=40, cols=20000, conn=30)).mv == wide.mv
    # An explicit device fallback still takes precedence over a nearest workload.
    legacy = CSRConfig(128)
    save_csr_config(legacy, 'TEST_GPU')
    assert load_csr_config('TEST_GPU', matrix_metadata=_metadata(rows=10, conn=4)).mv == legacy.mv
    assert load_csr_config('OTHER_GPU', matrix_metadata=_metadata(conn=30)).mv == CSRConfig().mv


def test_compilation_failure_marks_only_that_candidate_failed(monkeypatch):
    from brainevent._error import KernelCompilationError

    bad, good = CSRConfig(512), CSRConfig(256)

    def measure(config, **kwargs):
        if config == bad:
            raise KernelCompilationError('nvcc failed for this candidate')
        return _timing(config)

    monkeypatch.setattr(initialize, '_benchmark_config', measure)
    records = initialize.run_benchmark(n_pre=4, n_post=1000, density=.03, candidates=(bad, good),
                                       batch_size=2, steps=3, warmup=1, show_progress=False)
    assert [record['status'] for record in records] == ['ok', 'error']
    assert 'nvcc failed' in records[1]['error']


def test_load_parses_the_store_once_until_it_changes(_persistence_isolated_store, monkeypatch):
    save_csr_config(CSRConfig(128), 'TEST_GPU')
    parsed = []
    loads = initialize.json.loads
    monkeypatch.setattr(initialize.json, 'loads', lambda text: parsed.append(1) or loads(text))
    for _ in range(3):
        assert load_csr_config('TEST_GPU').block_size == 128
    assert len(parsed) == 1
    save_csr_config(CSRConfig(256), 'TEST_GPU')
    parsed.clear()
    assert load_csr_config('TEST_GPU').block_size == 256
    assert load_csr_config('TEST_GPU').block_size == 256
    assert len(parsed) == 1


def test_corrupt_store_is_backed_up_before_it_is_replaced(_persistence_isolated_store):
    path = _persistence_isolated_store / initialize.CONFIG_FILENAME
    path.write_text('{"TEST_GPU": broken')
    with pytest.warns(UserWarning, match='corrupt'):
        save_csr_config(CSRConfig(256), 'TEST_GPU')
    backups = list(_persistence_isolated_store.glob(initialize.CONFIG_FILENAME + '.corrupt-*'))
    assert len(backups) == 1 and backups[0].read_text() == '{"TEST_GPU": broken'
    assert load_csr_config('TEST_GPU').block_size == 256
    assert not list(_persistence_isolated_store.glob('*.tmp'))


def test_concurrent_saves_keep_every_device_entry(_persistence_isolated_store):
    import threading

    devices = [f'GPU_{i}' for i in range(16)]
    barrier = threading.Barrier(len(devices))
    errors = []

    def save(device_kind):
        try:
            barrier.wait()
            save_csr_config(CSRConfig(256), device_kind)
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=save, args=(device,)) for device in devices]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    stored = json.loads((_persistence_isolated_store / initialize.CONFIG_FILENAME).read_text())
    assert set(stored) == set(devices)
    assert not list(_persistence_isolated_store.glob('*.tmp'))
