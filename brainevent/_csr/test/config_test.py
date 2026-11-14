# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Tests for ``brainevent/_csr/config.py``, grouped by behavior.

The shape-only hybrid MM plan (``_mm_plan``) is tested in ``mm_plan_test.py``.
"""

import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
import brainevent._csr.config as config_mod
from brainevent._csr.binary.initialize import load_csr_config, save_csr_config
from brainevent._csr.config import (
    CSRConfig,
    MVSchedule,
    _BenchmarkRecord,
    _ConfigAux,
    _config_from_spec,
    _select_config,
)


# MV scheduler parameters and accessors
# -------------------------------------

def test_nested_mv_schedule_preserves_flat_accessors():
    mv = MVSchedule(512, 1024, 256, 8192)
    config = CSRConfig(mv=mv)
    assert brainevent.MVSchedule is MVSchedule
    assert config.mv is mv
    assert (config.block_size, config.fixed_scatter_blocks,
            config.tpr_threshold, config.task_nnz) == (512, 1024, 256, 8192)
    assert config.compile_flags() == mv.compile_flags()
    assert config.module_suffix() == mv.module_suffix()
    assert config.mv_capacity(8193, 2) == mv.mv_capacity(8193, 2)


def test_flat_parameters_default_individually_and_cannot_mix_with_mv():
    assert CSRConfig().mv == MVSchedule() == MVSchedule(512, 1024, 128, 512)
    assert CSRConfig(block_size=256).mv == MVSchedule(256, 1024, 128, 512)
    assert CSRConfig(task_nnz=4096).mv == MVSchedule(512, 1024, 128, 4096)
    with pytest.raises(ValueError, match='cannot be combined'):
        CSRConfig(256, mv=MVSchedule())
    with pytest.raises(TypeError, match='MVSchedule'):
        CSRConfig(mv=(512, 1024, 128, 512))


def test_config_from_spec_restores_a_primitive_parameter():
    assert _config_from_spec(None).mv == CSRConfig().mv
    spec = MVSchedule(256, 2048, 256, 1024)
    assert _config_from_spec(spec).mv is spec


def test_validate_config_rejects_invalid_scheduler_values():
    for values in ((250, 2048, 128, 4096), (1056, 2048, 128, 4096),
                   (256, 0, 128, 4096), (256, 2048, -1, 4096), (256, 2048, 128, 0)):
        with pytest.raises(ValueError):
            CSRConfig(*values)
        with pytest.raises(ValueError):
            MVSchedule(*values)
    for bad in (256.0, True, '256'):
        with pytest.raises(ValueError, match='integer'):
            CSRConfig(block_size=bad)


def test_explicit_validation_returns_the_config_and_rechecks_the_schedule():
    config = CSRConfig(256)
    assert config.validate() is config
    # MVSchedule validates on construction; ``validate`` catches a bypassed one.
    schedule = MVSchedule()
    object.__setattr__(schedule, 'block_size', 100)
    with pytest.raises(ValueError, match='block_size'):
        CSRConfig(mv=schedule).validate()


def test_compile_flags_map_config_to_cuda_defines():
    flags = CSRConfig(512, 4096, 256, 8192).compile_flags()
    assert flags == [
        "-DBE_HYBRID_BLOCK_SIZE=512",
        "-DBE_HYBRID_FIXED_SCATTER_BLOCKS=4096",
        "-DBE_HYBRID_TPR_THRESHOLD=256",
        "-DBE_HYBRID_TASK_NNZ=8192",
    ]


def test_module_suffix_is_config_specific():
    a = CSRConfig(256, 2048, 128, 4096).module_suffix()
    b = CSRConfig(512, 2048, 128, 4096).module_suffix()
    assert a == "_b256_s2048_t128_n4096"
    assert a != b
    # Every field takes part in the module identity.
    suffixes = {CSRConfig(*values).module_suffix() for values in (
        (256, 2048, 128, 4096), (256, 1024, 128, 4096), (256, 2048, 256, 4096), (256, 2048, 128, 512))}
    assert len(suffixes) == 4


# Pytree identity and diagnostics
# -------------------------------

def test_config_pytree_has_no_leaves_and_its_schedule_is_the_treedef():
    config = CSRConfig(256, 2048, 256, 1024)
    leaves, tree = jax.tree_util.tree_flatten(config)
    assert leaves == []
    children, aux = config.tree_flatten()
    assert children == () and aux == _ConfigAux(config.mv, ())
    restored = jax.tree_util.tree_unflatten(tree, leaves)
    assert isinstance(restored, CSRConfig) and restored is not config
    assert restored.mv == config.mv
    # Equal schedules give equal treedefs; any scheduler field changes it.
    assert tree == jax.tree_util.tree_structure(CSRConfig(mv=MVSchedule(256, 2048, 256, 1024)))
    for other in (CSRConfig(), CSRConfig(256, 2048, 256, 512)):
        assert tree != jax.tree_util.tree_structure(other)
    # Configs are unhashable; identity is carried by the hashable aux data.
    with pytest.raises(TypeError):
        hash(config)
    assert hash(aux) == hash(_ConfigAux(MVSchedule(256, 2048, 256, 1024), ()))


def test_benchmark_records_do_not_change_pytree_identity():
    config = CSRConfig(256)
    record = _BenchmarkRecord(CSRConfig(128, 1024, 256, 512), 2.0, 10.0, 4)
    with_records = replace(config, benchmark_records=(record,))
    assert with_records.mv == config.mv
    assert jax.tree_util.tree_structure(config) == jax.tree_util.tree_structure(with_records)
    aux, aux_records = config.tree_flatten()[1], with_records.tree_flatten()[1]
    assert aux == aux_records and hash(aux) == hash(aux_records)
    # The records still travel through a flatten / unflatten round trip.
    leaves, tree = jax.tree_util.tree_flatten(with_records)
    assert jax.tree_util.tree_unflatten(tree, leaves).benchmark_records == (record,)


def test_benchmark_records_render_in_str():
    config = CSRConfig(256)
    assert str(config) == ('CSRConfig(block_size=256, fixed_scatter_blocks=1024, '
                           'tpr_threshold=128, task_nnz=512)')
    records = (_BenchmarkRecord(CSRConfig(128, 1024, 256, 512), 2.0, 10.0, 4),
               _BenchmarkRecord(CSRConfig(512), 3.5, 17.25))
    lines = str(replace(config, benchmark_records=records)).splitlines()
    assert lines[0] == str(config) and lines[1] == 'benchmark results:'
    assert lines[2].startswith('  1. CSRConfig(block_size=128,')
    assert '2.000 ms (10.000 us/call)' in lines[2] and lines[2].endswith('task_capacity=4')
    assert '3.500 ms (17.250 us/call)' in lines[3] and 'task_capacity' not in lines[3]


def test_config_is_a_jit_argument_and_scan_carry_keyed_by_its_schedule():
    traces = []

    @jax.jit
    def run(cfg, x):
        traces.append(cfg.mv)

        def body(carry, unused):
            return carry, carry.block_size

        final, sizes = jax.lax.scan(body, cfg, None, length=3)
        return final, x + sizes

    final, values = run(CSRConfig(256), jnp.zeros(3, jnp.int32))
    np.testing.assert_array_equal(values, [256, 256, 256])
    assert isinstance(final, CSRConfig) and final.mv == CSRConfig(256).mv
    # An equal schedule (even with diagnostics) reuses the executable; another one retraces.
    run(CSRConfig(256, benchmark_records=(_BenchmarkRecord(CSRConfig(), 1.0, 1.0),)), jnp.zeros(3, jnp.int32))
    assert traces == [CSRConfig(256).mv]
    run(CSRConfig(512), jnp.zeros(3, jnp.int32))
    assert traces == [CSRConfig(256).mv, CSRConfig(512).mv]


def test_matrices_of_any_structure_share_the_config_treedef():
    # Same shape and nnz, different rows: no structure state reaches the treedef.
    first = brainevent.CSR((np.ones(4, np.float32), np.array([0, 1, 2, 3], np.int32),
                            np.array([0, 4, 4], np.int32)), shape=(2, 4))
    second = brainevent.CSR((np.ones(4, np.float32), np.array([0, 1, 2, 3], np.int32),
                             np.array([0, 1, 4], np.int32)), shape=(2, 4))
    assert jax.tree_util.tree_structure(first.config) == jax.tree_util.tree_structure(second.config)
    assert jax.tree_util.tree_leaves(first.config) == []


def test_removed_mm_planning_surface_stays_removed():
    for name in ('HybridConfig', 'get_hybrid_config', 'MMStructureSpec', 'MMStructurePlan',
                 'MMCallPlan', 'CSRKernelSpec', 'CSRDeviceInfo', 'get_device_info'):
        assert not hasattr(brainevent, name), name
    for name in ('MMStructureSpec', 'MMStructurePlan', 'MMCallPlan', 'CSRKernelSpec',
                 'CSRDeviceInfo', 'get_device_info', '_prepare_csr_config', '_sync_mirror_config',
                 '_config_for_new_structure', '_structure_plan', '_call_plan', '_bucket'):
        assert not hasattr(config_mod, name), name
    for name in ('prepare_structure', 'mm_plan', 'kernel_args', 'from_kernel_args', 'swap_layouts',
                 'parameter_snapshot', 'primary', 'mirror', 'device_info', 'shape'):
        assert not hasattr(CSRConfig(), name), name
    for cls in (brainevent.CSR, brainevent.CSC, brainevent.FixedNumPerPre, brainevent.FixedNumPerPost):
        assert not hasattr(cls, 'with_rebuilt_config'), cls
    assert brainevent.CSRConfig is CSRConfig
    assert brainevent.load_csr_config is load_csr_config
    assert brainevent.save_csr_config is save_csr_config


# MV task-capacity bounds and row-length hints
# --------------------------------------------

def _exact_mv_capacity(indptr, config):
    lengths = np.diff(indptr)
    heavy = lengths > config.tpr_threshold
    return int(((lengths[heavy] + config.task_nnz - 1) // config.task_nnz).sum())


@pytest.mark.parametrize('config', [CSRConfig(), CSRConfig(tpr_threshold=0, task_nnz=1),
                                    CSRConfig(tpr_threshold=3, task_nnz=5)])
def test_capacity_bound_is_never_below_the_exact_capacity(config):
    # The bound sizes the hybrid task queue from static shapes; it must cover
    # every structure with those shapes or the CUDA kernel drops work.
    rng = np.random.default_rng(0)
    for _ in range(300):
        n_rows = int(rng.integers(1, 40))
        lengths = rng.integers(0, int(rng.choice([4, 50, 9000])), n_rows)
        if rng.random() < 0.3:
            lengths[rng.integers(n_rows)] = int(rng.integers(0, 20000))
        indptr = np.concatenate([[0], np.cumsum(lengths)])
        nse = int(indptr[-1])
        exact = _exact_mv_capacity(indptr, config)
        assert config.mv_capacity(nse, n_rows) >= exact
        assert config.mv_capacity(nse, n_rows, int(np.diff(indptr).max(initial=0))) >= exact


def test_capacity_bound_uses_the_row_length_hint():
    config = CSRConfig()
    # No heavy row: the hint proves the queue is never used.
    assert config.mv_capacity(10_000, 100, max_row_nnz=config.tpr_threshold) == 0
    assert config.mv_capacity(10_000, 100) > 0
    assert config.mv_capacity(0, 100) == 0
    assert config.mv_capacity(10, 0) == 0
    # A single 9000-long row needs ceil(9000 / task_nnz) chunks.
    assert config.mv_capacity(9000, 1, max_row_nnz=9000) == (9000 + config.task_nnz - 1) // config.task_nnz
    # Exact hint for rows 129, 4096 and 4097 long.
    indptr = np.array([0, 129, 129 + 4096, 129 + 4096 + 4097], dtype=np.int64)
    assert config.mv_capacity(int(indptr[-1]), 3, 4097) >= _exact_mv_capacity(indptr, config)


def test_capacity_rejects_invalid_counts_and_int32_overflow():
    with pytest.raises(ValueError, match='int32'):
        CSRConfig(tpr_threshold=0, task_nnz=1).mv_capacity(2**31, 1)
    with pytest.raises(ValueError):
        CSRConfig().mv_capacity(-1, 1)
    with pytest.raises(ValueError, match='integer'):
        CSRConfig().mv_capacity(1.5, 1)
    with pytest.raises(ValueError, match='max_row_nnz'):
        CSRConfig().mv_capacity(10, 1, max_row_nnz=-1)


# Native kernel adapters that do not need a GPU
# ---------------------------------------------

@pytest.mark.parametrize('config', [None, CSRConfig()])
def test_nonhybrid_mm_keeps_jax_supported_float8_dtype(config):
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 1, 2], dtype=jnp.int32)
    values = jnp.array([1, 2], dtype=jnp.float8_e4m3fn)
    events = jnp.ones((2, 2), dtype=jnp.bool_)
    result = brainevent.binary_csrmm(values, indices, indptr, events, shape=(2, 2),
                                     backend='jax_raw', config=config)
    assert result.dtype == values.dtype
    np.testing.assert_array_equal(result.astype(jnp.float32), [[1, 1], [2, 2]])


def test_hybrid_mm_rejects_unsupported_dtype_before_planning(monkeypatch):
    import brainevent._csr.binary.binary as binary_mod

    monkeypatch.setattr(binary_mod, '_mm_plan', lambda *args: pytest.fail('planned an unsupported dtype'))
    with pytest.raises(ValueError, match='supports only float16'):
        binary_mod._make_hybrid_mm_kernel(
            config_spec=None,
            weight_info=jax.ShapeDtypeStruct((2,), jnp.float8_e4m3fn),
            vector_info=jax.ShapeDtypeStruct((2, 2), jnp.bool_),
            indices_info=jax.ShapeDtypeStruct((2,), jnp.int32),
            indptr_info=jax.ShapeDtypeStruct((3,), jnp.int32),
            out_info=jax.ShapeDtypeStruct((2, 2), jnp.float8_e4m3fn))


@pytest.mark.parametrize('idx', [[1], [-1]])
def test_explicit_config_does_not_skip_index_validation(idx):
    indices, indptr = np.asarray(idx, np.int32), np.asarray([0, 1], np.int32)
    with pytest.raises(ValueError, match='index|indices'):
        brainevent.CSR(np.ones(1, np.float32), indices, indptr, shape=(1, 1), config=CSRConfig())


# Placement metadata read during matrix construction
# --------------------------------------------------

class _FakeDevice:
    def __init__(self, kind):
        self.device_kind = kind


class _PlacedArray:
    def __init__(self, *devices):
        self._devices = set(devices)

    def devices(self):
        return self._devices


def test_structure_device_and_kind_of_concrete_arrays():
    cpu = jax.devices('cpu')[0]
    placed = jax.device_put(jnp.arange(3), cpu)
    assert config_mod._structure_device(placed) == cpu
    assert config_mod._device_kind(placed) == cpu.device_kind
    # Host arrays carry no placement.
    assert config_mod._structure_device(np.arange(3)) is None
    assert config_mod._device_kind(np.arange(3)) is None
    a, b = _FakeDevice('GPU_A'), _FakeDevice('GPU_A')
    # Replicated over same-kind devices: no single device, but one kind.
    assert config_mod._structure_device(_PlacedArray(a, b)) is None
    assert config_mod._device_kind(_PlacedArray(a, b)) == 'GPU_A'
    assert config_mod._device_kind(_PlacedArray(a, _FakeDevice('GPU_B'))) is None


def test_structure_device_and_kind_of_tracers_are_unknown():
    seen = []

    @jax.jit
    def probe(x):
        seen.append((config_mod._structure_device(x), config_mod._device_kind(x)))
        return x

    probe(jnp.arange(3))
    assert seen == [(None, None)]


def test_host_structure_array_reuses_host_input_and_skips_tracers(monkeypatch):
    host = np.arange(4, dtype=np.int32)
    assert config_mod._host_structure_array(host, jnp.asarray(host)) is host
    listed = config_mod._host_structure_array([0, 1, 2], jnp.arange(3))
    np.testing.assert_array_equal(listed, [0, 1, 2])
    device = jnp.arange(5)
    np.testing.assert_array_equal(config_mod._host_structure_array(device, device), np.arange(5))

    monkeypatch.setattr(jax, 'device_get', lambda *args: pytest.fail('a tracer must not be read'))
    seen = []
    jax.jit(lambda x: seen.append(config_mod._host_structure_array(x, x)) or x)(jnp.arange(3))
    assert seen == [None]


# Persisted tuning selection during matrix construction
# -----------------------------------------------------

@pytest.fixture
def _selection_store(monkeypatch, tmp_path):
    previous = brainevent.get_cache_dir()
    monkeypatch.delenv('BRAINEVENT_CSR_HYBRID_CONFIG', raising=False)
    # Exercise persisted GPU selection with CPU arrays.
    monkeypatch.setattr(config_mod, '_device_kind', lambda indices: 'TEST_GPU')
    brainevent.set_cache_dir(str(tmp_path))
    try:
        yield tmp_path
    finally:
        brainevent.set_cache_dir(previous)


@pytest.mark.parametrize('cls', [brainevent.CSR, brainevent.CSC])
def test_construction_selects_both_densities_in_storage_direction(_selection_store, cls):
    shape = (2, 1000)
    choices = ((60, CSRConfig(256, 512, 128, 1024)),
               (6, CSRConfig(512, 2048, 256, 4096)))
    for nnz, config in choices:
        save_csr_config(config, 'TEST_GPU', matrix_metadata={'shape': shape, 'nnz': nnz})

    for nnz, expected in choices:
        pointers = np.array([0, nnz // 2, nnz], np.int32)
        indices = np.arange(nnz, dtype=np.int32)
        public_shape = shape if cls is brainevent.CSR else shape[::-1]
        matrix = cls((np.ones(nnz, np.float32), indices, pointers),
                     shape=public_shape, backend='jax_raw')
        assert matrix.config.mv == expected.mv


@pytest.mark.parametrize('cls', [brainevent.FixedNumPerPre, brainevent.FixedNumPerPost])
def test_fixed_connections_select_using_flattened_edge_count(_selection_store, cls):
    shape = (2, 1000)
    wrong_count = CSRConfig(512, 512, 128, 512)
    expected = CSRConfig(256, 2048, 256, 4096)
    save_csr_config(wrong_count, 'TEST_GPU', matrix_metadata={'shape': shape, 'nnz': 2})
    save_csr_config(expected, 'TEST_GPU', matrix_metadata={'shape': shape, 'nnz': 6})
    public_shape = shape if cls is brainevent.FixedNumPerPre else shape[::-1]
    matrix = cls(np.ones((2, 3), np.float32),
                 np.array([[0, 1, 2], [3, 4, 5]], np.int32),
                 shape=public_shape, backend='jax_raw')
    assert matrix.config.mv == expected.mv


def test_selection_is_keyed_by_the_structure_device_kind(monkeypatch, tmp_path):
    previous = brainevent.get_cache_dir()
    monkeypatch.delenv('BRAINEVENT_CSR_HYBRID_CONFIG', raising=False)
    brainevent.set_cache_dir(str(tmp_path))
    try:
        cpu = jax.devices('cpu')[0]
        metadata = {'shape': (2, 4), 'nnz': 3}
        save_csr_config(CSRConfig(256), cpu.device_kind, matrix_metadata=metadata)
        save_csr_config(CSRConfig(128), 'OTHER_GPU', matrix_metadata=metadata)
        indices = jax.device_put(jnp.array([0, 1, 2], jnp.int32), cpu)
        assert _select_config(indices, shape=(2, 4)).mv == CSRConfig(256).mv
    finally:
        brainevent.set_cache_dir(previous)


def test_explicit_config_beats_environment_and_local_store(_selection_store, monkeypatch):
    from brainevent._csr.binary import initialize

    save_csr_config(CSRConfig(512), 'TEST_GPU',
                    matrix_metadata={'shape': (2, 4), 'nnz': 3})
    monkeypatch.setenv('BRAINEVENT_CSR_HYBRID_CONFIG', json.dumps({
        'block_size': 128, 'fixed_scatter_blocks': 512,
        'tpr_threshold': 128, 'task_nnz': 4096,
    }))
    monkeypatch.setattr(initialize, 'load_csr_config',
                        lambda **kwargs: pytest.fail('explicit config must not load persisted tuning'))
    explicit = CSRConfig(256, 1024, 256, 1024)
    matrix = brainevent.CSR((np.ones(3, np.float32), np.array([0, 1, 2], np.int32),
                            np.array([0, 1, 3], np.int32)),
                            shape=(2, 4), backend='jax_raw', config=explicit)
    assert matrix.config is explicit
    assert _select_config(matrix.indices, shape=(2, 4), config=explicit) is explicit


def test_select_config_rejects_a_non_config():
    with pytest.raises(TypeError, match='CSRConfig'):
        _select_config(np.arange(3), shape=(2, 4), config=MVSchedule())


def test_traced_selection_forwards_static_metadata_without_reading_arrays(monkeypatch):
    from brainevent._csr.binary import initialize

    calls = []
    expected = CSRConfig(256, 512, 256, 1024)

    def load(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(initialize, 'load_csr_config', load)
    monkeypatch.setattr(jax, 'device_get',
                        lambda *args, **kwargs: pytest.fail('metadata must not read tracer arrays'))

    @jax.jit
    def build(data, indices, indptr):
        matrix = brainevent.CSR((data, indices, indptr), shape=(2, 5), backend='jax_raw')
        assert matrix.config is expected
        return indices + matrix.config.block_size

    actual = build(jnp.ones(3), jnp.array([0, 1, 2], jnp.int32), jnp.array([0, 1, 3], jnp.int32))
    np.testing.assert_array_equal(actual, np.array([256, 257, 258]))
    assert calls == [{'device_kind': None, 'matrix_metadata': {'shape': (2, 5), 'nnz': 3}}]


@pytest.mark.parametrize('shape', [(0, 5), (2, 0)])
def test_empty_selection_uses_static_metadata(monkeypatch, shape):
    from brainevent._csr.binary import initialize

    expected = CSRConfig(256, 512, 128, 1024)
    calls = []

    def load(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(initialize, 'load_csr_config', load)
    assert _select_config(np.empty(0, np.int32), shape=shape) is expected
    assert calls == [{'device_kind': None, 'matrix_metadata': {'shape': shape, 'nnz': 0}}]
    matrix = brainevent.CSR((np.ones(0, np.float32), np.empty(0, np.int32),
                             np.zeros(shape[0] + 1, np.int32)), shape=shape, backend='jax_raw')
    assert matrix.config is expected


def test_existing_matrix_keeps_its_snapshot_after_persisted_winner_changes(_selection_store):
    metadata = {'shape': (2, 4), 'nnz': 3}
    original = CSRConfig(256, 512, 128, 1024)
    replacement = CSRConfig(512, 2048, 256, 4096)
    save_csr_config(original, 'TEST_GPU', matrix_metadata=metadata)
    parts = (np.ones(3, np.float32), np.array([0, 1, 2], np.int32),
             np.array([0, 1, 3], np.int32))
    matrix = brainevent.CSR(parts, shape=(2, 4), backend='jax_raw')
    save_csr_config(replacement, 'TEST_GPU', matrix_metadata=metadata)

    fresh = brainevent.CSR(parts, shape=(2, 4), backend='jax_raw')
    assert matrix.config.mv == original.mv
    assert fresh.config.mv == replacement.mv
    # Structure-preserving derivations and pytree round trips keep the snapshot.
    assert matrix.with_data(np.zeros(3, np.float32)).config is matrix.config
    assert jax.tree_util.tree_map(lambda x: x, matrix).config.mv == original.mv
