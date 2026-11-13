# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Tests for ``brainevent/_csr/config.py``, grouped by behavior."""

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
import brainevent._csr.binary.binary as binary_mod
import brainevent._csr.config as config_mod
from brainevent._csr.binary.initialize import load_csr_config, save_csr_config
from brainevent._csr.config import (
    CSRConfig,
    CSRDeviceInfo,
    MMStructurePlan,
    MMStructureSpec,
    MVSchedule,
    _BenchmarkRecord,
    _call_plan,
    get_device_info,
)
from brainevent._structure import StructureCache


# Structure plans, scheduling snapshots and validation
# ----------------------------------------------------

DEVICE = CSRDeviceInfo('gpu', 'TEST_GPU', l2_bytes=4096, sms=16)


def _prepared(lengths=(16, 17, 0, 513, 1025), cols=128, **kwargs):
    indptr = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    indices = (np.arange(indptr[-1]) % cols).astype(np.int32)
    config = CSRConfig(device_info=DEVICE, **kwargs).prepare_structure(
        indices, indptr, shape=(len(lengths), cols))
    return config, indices, indptr


def test_structure_plan_uses_integer_group_rule_and_exact_long_mask():
    config, indices, indptr = _prepared()
    primary = config.primary
    # Row statistics are power-of-two bounds clamped to nnz: 1025 -> 1571, and
    # 5 long-row tasks take the smallest nonzero task bucket, 64.
    assert (config.shape, config.nnz, config.primary_max_row_nnz) == ((5, 128), 1571, 1571)
    assert (primary.group, primary.long_threshold, primary.long_tasks) == (4, 64, 64)
    np.testing.assert_array_equal(primary.long_mask, np.array([0b11000], dtype=np.uint32))
    assert config.mirror is None and config.mirror_max_row_nnz is None
    np.testing.assert_array_equal(indptr, [0, 16, 33, 33, 546, 1571])


class _DeviceOnlyIndices:
    """Index metadata whose values must never be copied to the host."""

    ndim = 1
    dtype = np.dtype(np.int32)

    def __init__(self, size):
        self.shape = (size,)

    def __array__(self, *args, **kwargs):
        raise AssertionError('preparation must not copy index values to the host')


def test_prepare_structure_reads_indptr_but_not_index_values():
    expected, indices, indptr = _prepared()
    config = CSRConfig(device_info=DEVICE).prepare_structure(
        _DeviceOnlyIndices(indices.size), indptr, shape=expected.shape)
    assert config.kernel_args()[0] == expected.kernel_args()[0]
    np.testing.assert_array_equal(config.primary.long_mask, expected.primary.long_mask)
    with pytest.raises(ValueError, match='end at nnz'):
        CSRConfig().prepare_structure(_DeviceOnlyIndices(indices.size - 1), indptr, shape=expected.shape)


def _mirror_cache(indices, indptr, shape):
    from brainevent._misc import csr_to_csc_index

    return StructureCache(
        mirror=csr_to_csc_index(indptr, indices, shape=shape))


def test_primary_preparation_does_not_count_columns(monkeypatch):
    def no_count(*args, **kwargs):
        raise AssertionError(
            'primary preparation must not compute mirror statistics')

    monkeypatch.setattr(config_mod.np, 'bincount', no_count)
    config, _, _ = _prepared()
    assert config.primary is not None
    assert config.mirror is config.mirror_max_row_nnz is None


def test_mirror_statistics_are_prepared_once_and_shared_without_recounting(
    monkeypatch,
):
    config, indices, indptr = _prepared()
    cache = _mirror_cache(indices, indptr, config.shape)
    lengths = np.diff(np.asarray(cache.mirror[0]))
    prepared = config_mod._sync_mirror_config(
        config, cache, shape=config.shape[::-1])
    assert prepared.primary is config.primary
    assert prepared.mirror_max_row_nnz == config_mod._bucket(int(lengths.max()), config.nnz)
    assert prepared.mirror_max_row_nnz >= int(lengths.max())
    assert cache.mirror_preparation == (
        prepared.mirror, prepared.mirror_max_row_nnz)

    def no_plan(*args, **kwargs):
        raise AssertionError('mirror statistics must be reused')

    monkeypatch.setattr(config_mod, '_structure_plan', no_plan)
    sibling = config_mod._sync_mirror_config(
        config, cache, shape=config.shape[::-1], prepare=False)
    assert sibling.mirror is prepared.mirror
    assert config_mod._sync_mirror_config(
        prepared, cache, shape=config.shape[::-1]) is prepared
    assert config.mirror is None


def test_mirror_sync_only_reads_pointers_and_preserves_existing_plan(
    monkeypatch,
):
    config, indices, indptr = _prepared()
    cache = _mirror_cache(indices, indptr, config.shape)

    class NoHostRead:
        shape = indices.shape
        dtype = np.dtype(np.int32)

        def __array__(self, *args, **kwargs):
            raise AssertionError(
                'mirror statistics need only the compressed pointers')

    cache.mirror = (cache.mirror[0], NoHostRead(), NoHostRead())
    prepared = config_mod._sync_mirror_config(
        config, cache, shape=config.shape[::-1])
    assert prepared.mirror is not None
    alternate = replace(
        prepared.mirror, long_mask=jnp.array(prepared.mirror.long_mask))
    independent = replace(prepared, mirror=alternate)
    assert config_mod._sync_mirror_config(
        independent, cache, shape=config.shape[::-1]) is independent


def test_mirror_sync_without_preparation_preserves_an_incomplete_snapshot(
    monkeypatch,
):
    config, indices, indptr = _prepared()
    cache = _mirror_cache(indices, indptr, config.shape)

    def no_plan(*args, **kwargs):
        raise AssertionError('packing must not create a plan')

    monkeypatch.setattr(config_mod, '_structure_plan', no_plan)
    assert config_mod._sync_mirror_config(
        config, cache, shape=config.shape[::-1], prepare=False) is config
    assert cache.mirror_preparation is None


def test_mirror_sync_recovers_structure_identity_from_cached_preparation():
    config, indices, indptr = _prepared()
    cache = _mirror_cache(indices, indptr, config.shape)
    prepared = config_mod._sync_mirror_config(
        config, cache, shape=config.shape[::-1])
    restored = config_mod._sync_mirror_config(
        config.parameter_snapshot(), cache, shape=config.shape[::-1],
        prepare=False)
    assert restored.shape == config.shape and restored.nnz == config.nnz
    assert restored.mirror is prepared.mirror
    assert restored.validate() is restored


def test_traced_mirror_without_metadata_is_not_scanned():
    config, indices, indptr = _prepared()
    mirror = _mirror_cache(indices, indptr, config.shape).mirror

    @jax.jit
    def run(ptr, idx, perm):
        cache = StructureCache(mirror=(ptr, idx, perm))
        result = config_mod._sync_mirror_config(
            config, cache, shape=config.shape[::-1])
        assert result is config and cache.mirror_preparation is None
        return ptr

    np.testing.assert_array_equal(run(*mirror), mirror[0])


@pytest.mark.parametrize('pointer', [
    None, jax.ShapeDtypeStruct((129,), np.int32),
    np.zeros(129, dtype=jax.dtypes.float0),
])
def test_mirror_sync_does_not_scan_placeholder_pointers(pointer):
    config, indices, _ = _prepared()
    cache = StructureCache(mirror=(pointer, indices, None))
    assert config_mod._sync_mirror_config(
        config, cache, shape=config.shape[::-1]) is config
    assert cache.mirror_preparation is None


def test_known_mirror_plan_seeds_cache_without_reading_structure(monkeypatch):
    config, indices, indptr = _prepared()
    cache = _mirror_cache(indices, indptr, config.shape)
    prepared = config_mod._sync_mirror_config(
        config, cache, shape=config.shape[::-1])
    cache.mirror_preparation = None

    def no_plan(*args, **kwargs):
        raise AssertionError('a known reverse-layout plan must be reused')

    monkeypatch.setattr(config_mod, '_structure_plan', no_plan)
    assert config_mod._sync_mirror_config(
        prepared, cache, shape=config.shape[::-1], prepare=False) is prepared
    assert cache.mirror_preparation[0] is prepared.mirror


@pytest.mark.parametrize('pointers', [
    np.zeros((2, 2), np.int32), np.zeros(3, np.int32),
    np.array([1, 1, 2], np.int32), np.array([0, 0, 1], np.int32),
    np.array([0, 3, 2], np.int32),
])
def test_invalid_mirror_pointers_do_not_publish_preparation(pointers):
    config = CSRConfig().prepare_structure(
        np.array([0, 1], np.int32), np.array([0, 2], np.int32), shape=(1, 2))
    cache = StructureCache(mirror=(pointers, np.array([0, 0], np.int32), None))
    with pytest.raises(ValueError, match='mirror indptr'):
        config_mod._sync_mirror_config(config, cache, shape=(2, 1))
    assert cache.mirror_preparation is None
    assert config.mirror is None


def test_failed_mirror_plan_preparation_does_not_publish_metadata(monkeypatch):
    config, indices, indptr = _prepared()
    cache = _mirror_cache(indices, indptr, config.shape)

    def fail(*args, **kwargs):
        raise RuntimeError('injected preparation failure')

    monkeypatch.setattr(config_mod, '_structure_plan', fail)
    with pytest.raises(RuntimeError, match='injected'):
        config_mod._sync_mirror_config(config, cache, shape=config.shape[::-1])
    assert cache.mirror_preparation is None and config.mirror is None


def test_mirror_sync_without_cached_mirror_does_not_prepare():
    config, _, _ = _prepared()
    assert config_mod._sync_mirror_config(
        config, StructureCache(), shape=config.shape[::-1]) is config


def test_concrete_mirror_preparation_under_trace_does_not_cache_tracers():
    config, indices, indptr = _prepared()
    cache = _mirror_cache(indices, indptr, config.shape)

    @jax.jit
    def run(snapshot):
        return config_mod._sync_mirror_config(
            snapshot, cache, shape=config.shape[::-1]).mirror.long_mask

    result = run(config)
    assert not isinstance(
        cache.mirror_preparation[0].long_mask, jax.core.Tracer)
    np.testing.assert_array_equal(
        result, cache.mirror_preparation[0].long_mask)


def test_existing_traced_mirror_mask_does_not_escape_into_shared_cache():
    config, indices, indptr = _prepared()
    cache = _mirror_cache(indices, indptr, config.shape)
    prepared = config_mod._sync_mirror_config(
        config, cache, shape=config.shape[::-1])
    cache.mirror_preparation = None

    @jax.jit
    def run(snapshot):
        return config_mod._sync_mirror_config(
            snapshot, cache, shape=config.shape[::-1],
            prepare=False).mirror.long_mask

    np.testing.assert_array_equal(run(prepared), prepared.mirror.long_mask)
    assert cache.mirror_preparation is None


def test_same_shape_and_nnz_different_rows_produce_different_plans():
    first, _, _ = _prepared((513, 512), cols=2)
    second, _, _ = _prepared((1025, 0), cols=2)
    assert first.nnz == second.nnz
    # Two and three tasks share the smallest task bucket; the row bounds differ.
    assert first.primary.long_tasks == second.primary.long_tasks == 64
    assert first.primary_max_row_nnz != second.primary_max_row_nnz
    assert first.kernel_args()[0] != second.kernel_args()[0]


def test_mm_call_plan_reacts_to_batch_dtype_device_and_weighted_queue():
    config, _, _ = _prepared()
    plan = config.mm_plan(10, np.float32)
    tasks = config.primary.long_tasks
    assert tasks == config_mod._long_task_bucket(5, config.nnz) == 64
    # The short-row lists hold one batch group of chunk * rows entries.
    assert (plan.chunk, plan.task_capacity, plan.short_rows_size) == (2, 2 * tasks, 10)
    assert plan.workspace_words == (80 + 5 * 8 + 2 * tasks * 12) // 4
    assert config.mm_plan(10, np.float32) is plan
    wide = config.mm_plan(10, np.float64)
    assert (wide.chunk, wide.task_capacity) == (1, tasks)
    # Half values accumulate in a float32 output, so they group like float32.
    for half in (np.float16, jnp.bfloat16):
        assert config.mm_plan(10, half).chunk == plan.chunk
    one = config.mm_plan(1, np.float32)
    assert (one.chunk, one.task_capacity) == (1, tasks)
    weighted = config.mm_plan(10, np.float32, weighted=True)
    assert weighted.workspace_words == plan.workspace_words + plan.task_capacity
    smaller = replace(config, device_info=replace(DEVICE, l2_bytes=2048)).mm_plan(10, np.float32)
    assert smaller.chunk == 1


def test_half_weighted_scratch_rounds_odd_queue_to_uint32_alignment():
    config, _, _ = _prepared((513,), cols=1)
    # No eligible short row: G=32, and a 513-edge row takes two tasks (bucket 64).
    assert config.primary.long_tasks == 64
    config, _, _ = _prepared((1, 1, 1, 1, 65), cols=1)
    # Prepared buckets are even; an odd task count must still pad the half queue.
    config = replace(config, primary=replace(config.primary, long_tasks=1))
    binary = config.mm_plan(1, np.float16)
    weighted = config.mm_plan(1, np.float16, weighted=True)
    assert weighted.workspace_words == binary.workspace_words + 1


def test_mm_geometry_is_independent_of_persisted_mv_tuning():
    default, indices, indptr = _prepared()
    alternate = CSRConfig(128, 1024, 0, 1, device_info=DEVICE).prepare_structure(
        indices, indptr, shape=(5, 128))
    assert default.kernel_args()[0].structure == alternate.kernel_args()[0].structure
    assert default.mm_plan(10, np.float32) == alternate.mm_plan(10, np.float32)
    assert default.compile_flags() != alternate.compile_flags()
    assert (default.mm_iterations, default.mm_task_edges, default.geometry_version) == (16, 512, 1)
    for name in ('mm_iterations', 'mm_task_edges', 'geometry_version'):
        with pytest.raises(AttributeError):
            setattr(default, name, 123)


def test_nested_mv_schedule_preserves_legacy_accessors():
    mv = MVSchedule(512, 1024, 256, 8192)
    config = CSRConfig(mv=mv)
    assert brainevent.MVSchedule is MVSchedule
    assert config.mv is mv
    assert (config.block_size, config.fixed_scatter_blocks,
            config.tpr_threshold, config.task_nnz) == (512, 1024, 256, 8192)
    assert config.compile_flags() == mv.compile_flags()
    assert config.mv_capacity(8193, 2) == mv.mv_capacity(8193, 2)


def test_call_plan_keeps_weighted_identity_even_without_long_tasks():
    config, _, _ = _prepared((1,), cols=1)
    for batch in (0, 2):
        binary = config.mm_plan(batch, np.float32, weighted=False)
        weighted = config.mm_plan(batch, np.float32, weighted=True)
        assert binary.long_tasks == weighted.long_tasks == 0
        assert binary.workspace_words == weighted.workspace_words
        assert not binary.weighted and weighted.weighted
        assert binary != weighted

    from brainevent._csr.binary.binary import _make_hybrid_mm_kernel as make_kernel
    with pytest.raises(ValueError, match='call plan does not match'):
        make_kernel(config_spec=config.kernel_args()[0], call_plan=binary, weighted=True,
                    weight_info=jax.ShapeDtypeStruct((1,), jnp.float32),
                    vector_info=jax.ShapeDtypeStruct((1, 2), jnp.bool_),
                    indices_info=jax.ShapeDtypeStruct((1,), jnp.int32),
                    indptr_info=jax.ShapeDtypeStruct((2,), jnp.int32),
                    out_info=jax.ShapeDtypeStruct((1, 2), jnp.float32))


def test_selected_kernel_roundtrip_and_layout_swap_share_masks():
    config, indices, indptr = _prepared()
    config = config_mod._sync_mirror_config(
        config, _mirror_cache(indices, indptr, config.shape),
        shape=config.shape[::-1])
    spec, mask = config.kernel_args('mirror')
    restored = CSRConfig.from_kernel_args(spec, mask)
    assert restored.kernel_args()[0].structure == spec.structure
    assert restored.primary.long_mask is config.mirror.long_mask
    assert restored.device_info is config.device_info
    assert restored.mirror is None
    swapped = config.swap_layouts()
    assert swapped.primary is config.mirror and swapped.mirror is config.primary
    assert swapped.swap_layouts().primary is config.primary
    bare = config.parameter_snapshot()
    assert bare.primary is None and bare.mirror is None and bare.device_info is DEVICE
    assert hash(spec) == hash(config.kernel_args('mirror')[0])
    assert restored is not config and restored != config
    with pytest.raises(TypeError):
        hash(config)
    with pytest.raises(ValueError, match='layout'):
        config.kernel_args('invalid')


@pytest.mark.parametrize('with_mirror', [False, True])
def test_config_pytree_uses_mask_leaves_and_survives_jit_scan(with_mirror):
    config, indices, indptr = _prepared()
    if with_mirror:
        config = config_mod._sync_mirror_config(
            config, _mirror_cache(indices, indptr, config.shape),
            shape=config.shape[::-1])
    leaves, tree = jax.tree_util.tree_flatten(config)
    assert len(leaves) == 1 + int(with_mirror)
    restored = jax.tree_util.tree_unflatten(tree, leaves)
    assert restored.kernel_args()[0] == config.kernel_args()[0]
    np.testing.assert_array_equal(restored.primary.long_mask, config.primary.long_mask)

    @jax.jit
    def run(cfg):
        def body(carry, unused):
            return carry, jnp.sum(carry.primary.long_mask)
        return jax.lax.scan(body, cfg, None, length=3)

    result, values = run(config)
    np.testing.assert_array_equal(values, [24, 24, 24])
    assert result.kernel_args()[0].structure == config.kernel_args()[0].structure


def test_traced_structure_is_not_scanned_or_implicitly_prepared():
    config, indices, indptr = _prepared()

    @jax.jit
    def check(idx, ptr):
        empty = config.prepare_structure(idx, ptr, shape=(5, 128))
        assert empty.primary is None and empty.mirror is None
        assert empty.device_info is DEVICE
        return empty.kernel_args()[1]

    assert check(indices, indptr).shape == (0,)


def test_benchmark_diagnostics_do_not_change_kernel_or_pytree_identity():
    config, _, _ = _prepared()
    record = _BenchmarkRecord(CSRConfig(128, 1024, 256, 512), 2.0, 10.0, 4)
    with_records = replace(config, benchmark_records=(record,))
    assert with_records.kernel_args()[0] == config.kernel_args()[0]
    assert jax.tree_util.tree_structure(config) == jax.tree_util.tree_structure(with_records)
    assert '2.000 ms (10.000 us/call)' in str(with_records)
    assert 'task_capacity=4' in str(with_records)


@pytest.mark.parametrize('lengths,cols', [((), 0), ((0, 0), 0), ((0, 0), 3)])
def test_empty_structure_has_valid_empty_or_zero_masks(lengths, cols):
    ptr = np.zeros(len(lengths) + 1, dtype=np.int32)
    config = CSRConfig(device_info=DEVICE).prepare_structure(
        np.empty(0, dtype=np.int32), ptr, shape=(len(lengths), cols))
    assert config.nnz == config.primary.long_tasks == 0
    assert config.mirror is None
    cache = _mirror_cache(np.empty(0, np.int32), ptr, config.shape)
    prepared = config_mod._sync_mirror_config(
        config, cache, shape=config.shape[::-1])
    assert prepared.mirror.long_tasks == prepared.mirror_max_row_nnz == 0
    assert config.mm_plan(0, np.float32).task_capacity == 0


@pytest.mark.parametrize('idx,ptr,shape', [
    ([0], [1, 1], (1, 1)), ([0], [0, 0], (1, 1)),
    ([0], [0, 2, 1], (2, 1)), ([0], [0, 1], (2, 1)),
])
def test_invalid_structure_rejected_before_preparation(idx, ptr, shape):
    with pytest.raises(ValueError):
        CSRConfig().prepare_structure(np.asarray(idx, np.int32), np.asarray(ptr, np.int32), shape=shape)


@pytest.mark.parametrize('idx', [[1], [-1]])
def test_index_values_are_validated_by_the_matrix_constructor(idx):
    # Planning reads indptr only; out-of-range index values are the
    # constructor's job (``check_structure=True``), not the planner's.
    indices, indptr = np.asarray(idx, np.int32), np.asarray([0, 1], np.int32)
    CSRConfig().prepare_structure(indices, indptr, shape=(1, 1))
    with pytest.raises(ValueError, match='index|indices'):
        brainevent.CSR(np.ones(1, np.float32), indices, indptr, shape=(1, 1), config=CSRConfig())


def test_invalid_types_dimensions_and_capacity_overflow():
    with pytest.raises(ValueError, match='dtype'):
        CSRConfig().prepare_structure(np.array([], float), np.array([0]), shape=(0, 0))
    with pytest.raises(ValueError, match='one-dimensional'):
        CSRConfig().prepare_structure(np.zeros((1, 1), np.int32), np.array([0, 1]), shape=(1, 1))
    with pytest.raises(ValueError, match='two dimensions'):
        CSRConfig().prepare_structure(np.array([], np.int32), np.array([0]), shape=(0,))
    with pytest.raises(ValueError, match='int32'):
        CSRConfig().prepare_structure(np.array([], np.int32), np.array([0]), shape=(0, 2**31))
    with pytest.raises(ValueError, match='integer'):
        CSRConfig(block_size=256.0)
    with pytest.raises(ValueError, match='int32'):
        CSRConfig(tpr_threshold=0, task_nnz=1).mv_capacity(2**31, 1)
    with pytest.raises(ValueError):
        CSRConfig().mv_capacity(-1, 1)


def test_missing_preparation_and_mismatching_plan_rejected():
    assert CSRConfig().mm_plan(8, np.float32) is None
    config, _, _ = _prepared()
    assert replace(config, device_info=CSRDeviceInfo('cpu', 'cpu')).mm_plan(8, np.float32) is None
    with pytest.raises(ValueError, match='prepared'):
        CSRConfig().require_prepared()
    with pytest.raises(ValueError, match='rows'):
        config.require_prepared(rows=6)
    with pytest.raises(ValueError, match='mask'):
        replace(config, primary=replace(config.primary, long_mask=jnp.empty(2, jnp.uint32))).require_prepared()
    with pytest.raises(ValueError, match='uint32'):
        replace(config, primary=replace(config.primary, long_mask=jnp.empty(1, jnp.int32))).require_prepared()
    assert config.require_prepared(rows=5, cols=128, nnz=1571) is config.primary
    with pytest.raises(ValueError, match='integer'):
        config.mm_plan(1.5, np.float32)
    with pytest.raises(ValueError, match='int32'):
        config.mm_plan(2**31, np.float32)
    assert config.mm_plan(1, np.int32) is None


def test_scalar_plan_cache_is_bounded():
    config, _, _ = _prepared()
    for batch in range(1, 280):
        config.mm_plan(batch, np.float32)
    assert _call_plan.cache_info().currsize <= 256


def test_incomplete_snapshots_roundtrip_without_creating_plan_arrays():
    config, indices, indptr = _prepared()
    config = config_mod._sync_mirror_config(
        config, _mirror_cache(indices, indptr, config.shape),
        shape=config.shape[::-1])
    for snapshot in (CSRConfig(), replace(config, primary=None), replace(config, mirror=None)):
        leaves, tree = jax.tree_util.tree_flatten(snapshot)
        restored = jax.tree_util.tree_unflatten(tree, leaves)
        assert (restored.primary is None) == (snapshot.primary is None)
        assert (restored.mirror is None) == (snapshot.mirror is None)
        assert restored.device_info == snapshot.device_info
    spec, empty = CSRConfig().kernel_args()
    assert empty.shape == (0,) and empty.dtype == np.uint32
    assert CSRConfig.from_kernel_args(spec, empty).primary is None


def test_invalid_static_metadata_and_oversized_queue_rejected():
    with pytest.raises(ValueError, match='integer'):
        CSRConfig(block_size=True)
    with pytest.raises(ValueError, match='l2_bytes'):
        CSRDeviceInfo('gpu', 'bad', l2_bytes=-1)
    assert CSRDeviceInfo('gpu', 'test', 4096, 16, 1) == CSRDeviceInfo('gpu', 'test', 4096, 16, 2)
    with pytest.raises(ValueError, match='geometry'):
        MMStructureSpec(1, 1, 1, 1, 3, 48, 0)
    with pytest.raises(ValueError, match='geometry'):
        MMStructureSpec(1, 1, 1, 1, 4, 64, 0, geometry_version=2)
    with pytest.raises(ValueError, match='nnz'):
        MMStructureSpec(1, 1, 1, 2, 4, 64, 0)
    with pytest.raises(ValueError, match='nonzero'):
        MMStructureSpec(0, 1, 1, 1, 4, 64, 0)
    with pytest.raises(ValueError, match='int32'):
        MMStructureSpec(2**31, 1, 0, 0, 4, 64, 0)
    nnz = 2**63 - 1
    plan = MMStructurePlan(32, 512, (nnz + 511) // 512,
                           jnp.array([1], dtype=jnp.uint32))
    cfg = CSRConfig(device_info=replace(DEVICE, l2_bytes=2**63), shape=(1, 1),
                    nnz=nnz, primary_max_row_nnz=nnz, primary=plan)
    with pytest.raises(ValueError, match='int64'):
        cfg.mm_plan(65535, np.float32)


def test_explicit_validation_checks_mirror_stats_without_reading_masks():
    config, _, _ = _prepared()
    assert config.validate() is config
    assert CSRConfig().validate().primary is None
    assert config.primary.geometry_version == 1
    with pytest.raises(ValueError, match='nnz'):
        replace(config, mirror_max_row_nnz=config.nnz + 1).validate()

    @jax.jit
    def validate_mask(mask):
        selected = CSRConfig.from_kernel_args(config.kernel_args()[0], mask)
        return selected.validate().primary.long_mask

    np.testing.assert_array_equal(validate_mask(config.primary.long_mask), config.primary.long_mask)


def test_old_configuration_exports_are_removed():
    for name in ('HybridConfig', 'get_hybrid_config'):
        assert not hasattr(brainevent, name)
    assert brainevent.CSRConfig is CSRConfig
    assert brainevent.load_csr_config is load_csr_config
    assert brainevent.save_csr_config is save_csr_config


@pytest.mark.parametrize('prepared', [False, True])
def test_nonhybrid_mm_keeps_jax_supported_float8_dtype(prepared):
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 1, 2], dtype=jnp.int32)
    config = (CSRConfig(device_info=DEVICE).prepare_structure(indices, indptr, shape=(2, 2))
              if prepared else None)
    values = jnp.array([1, 2], dtype=jnp.float8_e4m3fn)
    events = jnp.ones((2, 2), dtype=jnp.bool_)
    result = brainevent.binary_csrmm(values, indices, indptr, events, shape=(2, 2),
                                     backend='jax_raw', config=config)
    assert result.dtype == values.dtype
    np.testing.assert_array_equal(result.astype(jnp.float32), [[1, 1], [2, 2]])
    assert CSRConfig().mm_plan(2, values.dtype) is None


def test_hybrid_mm_unsupported_dtype_error_precedes_missing_preparation():
    from brainevent._csr.binary.binary import _make_hybrid_mm_kernel as make_kernel

    with pytest.raises(ValueError, match='supports only float16'):
        make_kernel(config_spec=None, call_plan=None,
                    weight_info=jax.ShapeDtypeStruct((2,), jnp.float8_e4m3fn),
                    vector_info=jax.ShapeDtypeStruct((2, 2), jnp.bool_),
                    indices_info=jax.ShapeDtypeStruct((2,), jnp.int32),
                    indptr_info=jax.ShapeDtypeStruct((3,), jnp.int32),
                    out_info=jax.ShapeDtypeStruct((2, 2), jnp.float8_e4m3fn))


def test_mm_plan_matches_native_host_planner_and_workspace(tmp_path):
    """Compare against the packaged C++ formulas, including odd half queues."""
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable for native planner parity')
    root = Path(__file__).parents[1] / 'binary' / 'hybrid_mm'
    declarations = (root / 'hybrid_batchedv3_spmm.cuh').read_text().split('namespace hybrid', 1)[1]
    declarations = declarations.split('// device_info,', 1)[0]
    source = r'''
#include <iostream>
#include <cstdint>
#include <vector>
#define __host__
#define __device__
using cudaError_t = int;
constexpr int cudaSuccess = 0, cudaErrorInvalidValue = 1;
constexpr int cudaDevAttrMultiProcessorCount = 0, cudaDevAttrL2CacheSize = 1;
int cudaGetDevice(int *);
int cudaDeviceGetAttribute(int *, int, int);
namespace hybrid
''' + declarations + '\n#include ' + json.dumps(str(root / 'hybrid_batchedv3_spmm_plan.cuh')) + r'''
int main() {
    int rows, cols, batch, l2, out_bytes, value_bytes, weighted;
    int64_t bucketed_tasks;
    while (std::cin >> rows >> cols >> batch >> l2 >> out_bytes >> value_bytes >> weighted >> bucketed_tasks) {
        std::vector<int64_t> ptr(rows + 1);
        for (auto &value : ptr) std::cin >> value;
        hybrid::Plan plan;
        // Batch groups are sized by the output (accumulator) element, the queue by the value.
        if (hybrid::make_plan(ptr.data(), rows, cols, batch, l2, out_bytes, &plan) != cudaSuccess) return 2;
        auto mask = hybrid::long_row_mask(ptr.data(), rows, plan);
        std::cout << plan.group << ' ' << plan.long_threshold << ' ' << plan.long_tasks << ' ' << plan.chunk;
        // Python prepares a power-of-two bound of the exact task count; size scratch from it.
        plan.long_tasks = bucketed_tasks;
        plan.task_capacity = plan.chunk * bucketed_tasks;
        std::cout << ' ' << plan.task_capacity << ' '
                  << hybrid::workspace_bytes(plan, batch, weighted ? value_bytes : 0) / 4;
        for (auto word : mask) std::cout << ' ' << word;
        std::cout << '\n';
    }
}
'''
    cpp, binary = tmp_path / 'plan.cpp', tmp_path / 'plan'
    cpp.write_text(source)
    subprocess.run([compiler, '-std=c++17', '-O2', str(cpp), '-o', str(binary)], check=True,
                   capture_output=True, text=True)
    rng = np.random.default_rng(4)
    cases = []
    golden_lengths = [
        [1, 1, 1, 1, 65], [16, 17, 513], [17, 18, 513], [32, 33, 513],
        [33, 34, 513], [64, 65, 513], [65, 66, 513], [0, 513, 1024, 1025],
        [512, 513], [1] * 32 + [65], [1] * 64 + [65], [1025],
    ]
    for i in range(36 + len(golden_lengths)):
        lengths = rng.integers(0, 1400, size=int(rng.integers(1, 80)))
        if i < len(golden_lengths):
            lengths = np.array(golden_lengths[i])
        cols, batch = int(rng.integers(1, 256)), int(rng.integers(1, 128))
        dtype = (np.float16, np.float32, np.float64)[i % 3]
        weighted = bool(i % 2)
        if i == 0:
            cols, batch, weighted = 128, 1, True
        config, _, ptr = _prepared(lengths, cols=cols)
        if i == 11:
            batch = 70000
            config = replace(config, device_info=replace(DEVICE, l2_bytes=2**30))
        plan = config.mm_plan(batch, dtype, weighted)
        itemsize = np.dtype(dtype).itemsize
        line = [len(lengths), cols, batch, config.device_info.l2_bytes, max(4, itemsize), itemsize,
                int(weighted), config.primary.long_tasks, *ptr]
        cases.append((' '.join(map(str, line)), config, plan))
    native = subprocess.run([str(binary)], input='\n'.join(case[0] for case in cases) + '\n',
                            capture_output=True, check=True, text=True)
    for (_, config, plan), line in zip(cases, native.stdout.splitlines(), strict=True):
        numbers = list(map(int, line.split()))
        assert config_mod._long_task_bucket(numbers[2], config.nnz) == plan.long_tasks >= numbers[2]
        assert numbers[:6] == [plan.group, plan.long_threshold, numbers[2],
                               plan.chunk, plan.task_capacity, plan.workspace_words]
        np.testing.assert_array_equal(numbers[6:], config.primary.long_mask)


# Device information and shard validation
# ---------------------------------------

def test_gpu_info_probe_accepts_cpu_shards_without_explicit_device():
    class CPUDevice:
        platform = 'cpu'

    cpu0 = CPUDevice()
    cpu1 = CPUDevice()

    class PlacedArray:
        def __init__(self, devices):
            self._devices = devices

        def devices(self):
            return self._devices

    assert get_device_info(PlacedArray((cpu0, cpu1))) is None


class _FakeCudaDevice:
    platform = 'gpu'
    client = type('Client', (), {'platform_version': 'cuda 13.0'})()

    def __init__(self, id, device_kind='FAKE_GPU', core_count=None):
        self.id = self.local_hardware_id = id
        self.device_kind = device_kind
        self.core_count = core_count


class _PlacedArray:
    def __init__(self, *devices):
        self._devices = set(devices)

    def devices(self):
        return self._devices


def test_replicated_same_kind_gpus_prepare_on_the_lowest_device(monkeypatch):
    queried = []
    monkeypatch.setattr(config_mod, '_query_cuda_device_info', lambda device: queried.append(device) or 'info')
    first, second = _FakeCudaDevice(3), _FakeCudaDevice(1)
    assert get_device_info(_PlacedArray(first, second), _PlacedArray(second)) == 'info'
    assert queried == [second]


@pytest.mark.parametrize('devices', [
    (_FakeCudaDevice(0, 'GPU_A'), _FakeCudaDevice(1, 'GPU_B')),
    (_FakeCudaDevice(0), type('CPUDevice', (), {'platform': 'cpu', 'id': 1})()),
], ids=['mixed_gpu_kinds', 'gpu_and_cpu'])
def test_ambiguous_placements_leave_the_config_unprepared(monkeypatch, devices):
    monkeypatch.setattr(config_mod, '_query_cuda_device_info', lambda device: pytest.fail('ambiguous placement queried'))
    assert get_device_info(_PlacedArray(*devices)) is None


def _no_toolchain(*args, **kwargs):
    from brainevent._error import NvccNotFoundError
    raise NvccNotFoundError('nvcc is not installed')


def test_device_query_without_driver_or_toolchain_leaves_config_unprepared(monkeypatch, caplog):
    import brainevent._op as op_mod

    def no_driver(device):
        raise OSError('libcuda.so.1: cannot open shared object file')

    monkeypatch.setattr(config_mod, '_driver_device_attributes', no_driver)
    monkeypatch.setattr(op_mod, 'load_cuda_file', _no_toolchain)
    config_mod._query_cuda_device_info.cache_clear()
    try:
        with caplog.at_level('DEBUG', logger=config_mod.__name__):
            assert config_mod._query_cuda_device_info(_FakeCudaDevice(0)) is None
    finally:
        config_mod._query_cuda_device_info.cache_clear()
    assert 'nvcc is not installed' in caplog.text


def test_driver_query_mismatching_jax_core_count_uses_native_probe(monkeypatch):
    monkeypatch.setattr(config_mod, '_driver_device_attributes', lambda device: (80, 4096))
    monkeypatch.setattr(config_mod, '_probe_device_attributes', lambda device: (82, 8192))
    config_mod._query_cuda_device_info.cache_clear()
    try:
        info = config_mod._query_cuda_device_info(_FakeCudaDevice(0, core_count=82))
        assert (info.sms, info.l2_bytes, info.device_kind) == (82, 8192, 'FAKE_GPU')
        info = config_mod._query_cuda_device_info(_FakeCudaDevice(1, core_count=80))
        assert (info.sms, info.l2_bytes) == (80, 4096)
    finally:
        config_mod._query_cuda_device_info.cache_clear()


def test_device_query_needs_no_cuda_toolchain(monkeypatch):
    if jax.default_backend() != 'gpu':
        pytest.skip('CUDA device query requires a GPU')
    import brainevent._op as op_mod

    gpu = jax.devices('gpu')[0]
    monkeypatch.setattr(op_mod, 'load_cuda_file', _no_toolchain)
    config_mod._query_cuda_device_info.cache_clear()
    try:
        info = config_mod.get_device_info(device=gpu)
    finally:
        config_mod._query_cuda_device_info.cache_clear()
    assert info is not None and info.l2_bytes > 0
    assert info.sms == getattr(gpu, 'core_count', info.sms) > 0


# Scheduler validation, compile flags and module identity
# -------------------------------------------------------

@pytest.fixture(autouse=False)
def _scheduler_restore_cache_dir():
    cache_dir = brainevent.get_cache_dir()
    try:
        yield
    finally:
        brainevent.set_cache_dir(cache_dir)


@pytest.mark.usefixtures('_scheduler_restore_cache_dir')
def test_validate_config_rejects_invalid_scheduler_values():
    for values in ((250, 2048, 128, 4096), (1056, 2048, 128, 4096),
                   (256, 0, 128, 4096), (256, 2048, -1, 4096), (256, 2048, 128, 0)):
        with pytest.raises(ValueError):
            CSRConfig(*values)


@pytest.mark.usefixtures('_scheduler_restore_cache_dir')
def test_compile_flags_map_config_to_cuda_defines():
    flags = CSRConfig(512, 4096, 256, 8192).compile_flags()
    assert flags == [
        "-DBE_HYBRID_BLOCK_SIZE=512",
        "-DBE_HYBRID_FIXED_SCATTER_BLOCKS=4096",
        "-DBE_HYBRID_TPR_THRESHOLD=256",
        "-DBE_HYBRID_TASK_NNZ=8192",
    ]


@pytest.mark.usefixtures('_scheduler_restore_cache_dir')
def test_module_suffix_is_config_specific():
    a = CSRConfig(256, 2048, 128, 4096).module_suffix()
    b = CSRConfig(512, 2048, 128, 4096).module_suffix()
    assert a == "_b256_s2048_t128_n4096"
    assert a != b


# Task-capacity bounds and row-length hints
# -----------------------------------------

def _capacity_capacity(indptr, config):
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
        exact = _capacity_capacity(indptr, config)
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


def test_prepared_row_hint_matches_concrete_structure():
    indptr = np.array([0, 129, 129 + 4096, 129 + 4096 + 4097], dtype=np.int64)
    indices = np.zeros(int(indptr[-1]), dtype=np.int32)
    config = CSRConfig().prepare_structure(indices, indptr, shape=(3, 1))
    assert config.primary_max_row_nnz == 8192  # power-of-two bound of 4097, below nnz
    assert config.mv_capacity(len(indices), 3, config.primary_max_row_nnz) >= _capacity_capacity(indptr, config)


# Device preparation and native query isolation
# ---------------------------------------------

def test_cpu_preparation_does_not_load_cuda(monkeypatch):
    monkeypatch.setattr(config_mod, '_query_cuda_device_info', lambda *args: pytest.fail('CPU preparation must not query CUDA'))
    cpu = jax.devices('cpu')[0]
    assert config_mod.get_device_info(device=cpu) is None
    assert config_mod.get_device_info(np.arange(3)) is None
    assert config_mod.get_device_info() is None


def test_device_query_does_not_load_mm_module(monkeypatch):
    if jax.default_backend() != 'gpu':
        pytest.skip('CUDA device query requires a GPU')
    gpu = jax.devices('gpu')[0]
    config_mod._query_cuda_device_info.cache_clear()
    monkeypatch.setattr(
        binary_mod, 'load_cuda_file',
        lambda *args, **kwargs: pytest.fail('device query must not load the MM module'),
    )
    info = config_mod.get_device_info(device=gpu)
    assert info is not None
    assert info.device_kind == str(gpu.device_kind)
    assert info.device_id == gpu.id
    assert info.sms > 0 and info.l2_bytes > 0


@pytest.mark.parametrize('platform_version', ['ROCm 7.0', 'Metal', ''])
def test_non_cuda_gpu_preparation_does_not_query_cuda(monkeypatch, platform_version):
    from types import SimpleNamespace
    device = SimpleNamespace(platform='gpu', client=SimpleNamespace(platform_version=platform_version))
    monkeypatch.setattr(config_mod, '_query_cuda_device_info', lambda *args: pytest.fail('non-CUDA device query'))
    assert config_mod.get_device_info(device=device) is None


# Workload-aware selection during matrix preparation
# --------------------------------------------------

@pytest.fixture
def _preparation_store(monkeypatch, tmp_path):
    from brainevent._csr.binary import initialize

    previous = brainevent.get_cache_dir()
    monkeypatch.delenv('BRAINEVENT_CSR_HYBRID_CONFIG', raising=False)
    monkeypatch.setattr(initialize, '_current_device_kind', lambda: 'TEST_GPU')
    # Exercise persisted selection with CPU arrays and no device-property query.
    monkeypatch.setattr(config_mod, '_structure_device', lambda *args, **kwargs: None)
    brainevent.set_cache_dir(str(tmp_path))
    try:
        yield tmp_path
    finally:
        brainevent.set_cache_dir(previous)


@pytest.mark.parametrize('cls', [brainevent.CSR, brainevent.CSC])
def test_construction_selects_both_densities_in_storage_direction(_preparation_store, cls):
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
        assert matrix.config.shape == shape
        assert matrix.config.nnz == nnz
        assert matrix.config.primary is not None


@pytest.mark.parametrize('cls', [brainevent.FixedNumPerPre, brainevent.FixedNumPerPost])
def test_fixed_connections_select_using_flattened_edge_count(_preparation_store, cls):
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
    assert matrix.config.shape == shape
    assert matrix.config.nnz == 6


def test_explicit_preparation_config_beats_environment_and_local_store(
    _preparation_store, monkeypatch,
):
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
    assert matrix.config.mv == explicit.mv
    assert matrix.config.primary is not None


def test_traced_preparation_forwards_static_metadata_without_reading_arrays(monkeypatch):
    from brainevent._csr.binary import initialize

    calls = []
    expected = CSRConfig(256, 512, 256, 1024)

    def load(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(initialize, 'load_csr_config', load)
    monkeypatch.setattr(jax, 'device_get',
                        lambda *args, **kwargs: pytest.fail('metadata must not read tracer arrays'))
    monkeypatch.setattr(CSRConfig, 'prepare_structure',
                        lambda *args, **kwargs: pytest.fail('traced structure cannot be prepared'))

    @jax.jit
    def prepare(indices, indptr):
        result = config_mod._prepare_csr_config(
            indices, indptr, shape=(2, 5), backend='jax_raw',
            matrix_metadata={'shape': (2, 5), 'nnz': indices.shape[0]},
        )
        assert result.primary is result.mirror is None
        return indices + result.block_size

    actual = prepare(jnp.array([0, 1, 2], jnp.int32), jnp.array([0, 1, 3], jnp.int32))
    np.testing.assert_array_equal(actual, np.array([256, 257, 258]))
    assert len(calls) == 1
    assert calls[0]['matrix_metadata']['shape'] == (2, 5)
    assert calls[0]['matrix_metadata']['nnz'] == 3


@pytest.mark.parametrize('metadata', [
    {'shape': (3, 4), 'nnz': 3},
    {'shape': (2, 4), 'nnz': 2},
])
def test_preparation_rejects_metadata_that_disagrees_with_actual_structure(metadata):
    with pytest.raises(ValueError):
        config_mod._prepare_csr_config(
            np.array([0, 1, 2], np.int32), np.array([0, 1, 3], np.int32),
            shape=(2, 4), backend='jax_raw', config=CSRConfig(), matrix_metadata=metadata,
        )


@pytest.mark.parametrize('shape', [(0, 5), (2, 0)])
def test_empty_preparation_infers_static_metadata_and_preserves_selected_parameters(
    monkeypatch, shape,
):
    from brainevent._csr.binary import initialize

    expected = CSRConfig(256, 512, 128, 1024)
    calls = []

    def load(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(initialize, 'load_csr_config', load)
    result = config_mod._prepare_csr_config(
        np.empty(0, np.int32), np.zeros(shape[0] + 1, np.int32),
        shape=shape, backend='jax_raw',
    )
    assert result.mv == expected.mv
    assert result.shape == shape and result.nnz == 0
    assert calls[0]['matrix_metadata']['shape'] == shape
    assert calls[0]['matrix_metadata']['nnz'] == 0


def test_rebuild_keeps_matched_snapshot_after_persisted_winner_changes(_preparation_store):
    metadata = {'shape': (2, 4), 'nnz': 3}
    original = CSRConfig(256, 512, 128, 1024)
    replacement = CSRConfig(512, 2048, 256, 4096)
    save_csr_config(original, 'TEST_GPU', matrix_metadata=metadata)
    parts = (np.ones(3, np.float32), np.array([0, 1, 2], np.int32),
             np.array([0, 1, 3], np.int32))
    matrix = brainevent.CSR(parts, shape=(2, 4), backend='jax_raw')
    save_csr_config(replacement, 'TEST_GPU', matrix_metadata=metadata)

    rebuilt = matrix.with_rebuilt_config()
    fresh = brainevent.CSR(parts, shape=(2, 4), backend='jax_raw')
    assert matrix.config.mv == rebuilt.config.mv == original.mv
    assert fresh.config.mv == replacement.mv
    assert rebuilt.config.primary is not matrix.config.primary
    assert rebuilt.indices is matrix.indices and rebuilt.indptr is matrix.indptr
