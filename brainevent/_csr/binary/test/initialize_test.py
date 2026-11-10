# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Tests for ``brainevent/_csr/binary/initialize.py``, grouped by behavior."""

import json
import sys
import types

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
def test_default_candidates_cover_more_tpr512_task_nnz2048_cases():
    tpr512_nnz2048_cases = {
        (candidate.block_size, candidate.fixed_scatter_blocks)
        for candidate in initialize.DEFAULT_CANDIDATES
        if candidate.tpr_threshold == 512 and candidate.task_nnz == 2048
    }
    assert tpr512_nnz2048_cases >= {
        (128, 2048),
        (256, 1024),
        (256, 2048),
        (512, 2048),
    }


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
        return np.array([[True]], dtype=np.bool_)

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

    assert initialize.init_csr_config(save=False) == cfg

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

    assert initialize.init_csr_config() == best
    assert capsys.readouterr().out == ""

    output = str(initialize.load_csr_config())
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
    save_csr_config(CSRConfig(), 'TEST_GPU')
    path.write_text('broken')
    save_csr_config(CSRConfig(), 'TEST_GPU')
    monkeypatch.setenv('BRAINEVENT_CSR_HYBRID_CONFIG', '{')
    with pytest.raises(ValueError):
        load_csr_config('TEST_GPU')
    with pytest.raises(RuntimeError, match='device_kind'):
        save_csr_config(CSRConfig(), '')
