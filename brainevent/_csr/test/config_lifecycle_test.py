"""Scheduler snapshots follow CSR/CSC structure through JAX reconstruction."""

import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent import CSR, CSC, FixedNumPerPre, FixedNumPerPost, BinaryArray
from brainevent._csr.config import CSRConfig, get_device_info


def _matrix(cls=CSR, **kwargs):
    shape = (2, 3) if cls is CSR else (3, 2)
    return cls(
        (jnp.array([1., 2., 3.]), jnp.array([0, 2, 1], jnp.int32),
         jnp.array([0, 2, 3], jnp.int32)),
        shape=shape, backend='jax_raw', **kwargs,
    )


@pytest.mark.parametrize('cls', [CSR, CSC])
def test_config_snapshot_is_loaded_once_and_survives_reconstruction(monkeypatch, cls):
    from brainevent._csr.binary import initialize

    calls = []
    snapshot = CSRConfig(block_size=128, tpr_threshold=7)
    monkeypatch.setattr(initialize, 'load_csr_config', lambda **kw: calls.append(kw) or snapshot)
    matrix = _matrix(cls)
    assert len(calls) == 1
    assert matrix.config.shape == (2, 3)
    assert matrix.config.nnz == 3
    assert matrix.config.tpr_threshold == 7

    def no_load(**kwargs):
        raise AssertionError('reconstruction must not load persisted configuration')

    monkeypatch.setattr(initialize, 'load_csr_config', no_load)
    assert matrix.with_data(matrix.data * 2).config is matrix.config
    assert matrix.T.config is matrix.config
    transformed = jax.jit(lambda value: value.with_data(value.data * 2))(matrix)
    np.testing.assert_array_equal(transformed.data, matrix.data * 2)
    assert transformed.config.tpr_threshold == 7
    assert transformed.config.shape == (2, 3)
    leaves = jax.tree.leaves(matrix)
    assert any(value is matrix.config.primary.long_mask for value in leaves)
    assert any(value is matrix.config.mirror.long_mask for value in leaves)


@pytest.mark.parametrize('cls', [CSR, CSC])
def test_from_parts_and_unflatten_do_not_prepare(monkeypatch, cls):
    matrix = _matrix(cls, config=CSRConfig())

    def no_prepare(*args, **kwargs):
        raise AssertionError('trusted reconstruction must not scan structure')

    monkeypatch.setattr(CSRConfig, 'prepare_structure', no_prepare)
    reconstructed = cls._from_parts(
        None, None, None, shape=matrix.shape, config=matrix.config,
    )
    assert reconstructed.config is matrix.config
    assert cls._from_parts(None, None, None, shape=matrix.shape).config.primary is None
    mapped = jax.tree.map(lambda _: None, matrix)
    assert mapped.indices is None
    assert mapped.config.primary.long_mask is None


@pytest.mark.parametrize('cls', [CSR, CSC])
def test_format_conversion_swaps_prepared_layouts_without_reloading(monkeypatch, cls):
    matrix = _matrix(cls, config=CSRConfig(tpr_threshold=5))
    matrix.build_weight_indices()
    converted = matrix.tocsc() if cls is CSR else matrix.tocsr()
    assert converted.config.primary is matrix.config.mirror
    assert converted.config.mirror is matrix.config.primary
    assert converted.config.shape == matrix.config.shape[::-1]
    assert converted.config.nnz == matrix.config.nnz
    assert converted.config.primary_max_row_nnz == matrix.config.mirror_max_row_nnz
    assert converted.config.mirror_max_row_nnz == matrix.config.primary_max_row_nnz
    np.testing.assert_allclose(converted.todense(), matrix.todense())


def test_structure_changes_rebind_eagerly_and_invalidate_traced_plans():
    matrix = _matrix(config=CSRConfig(tpr_threshold=5))
    sliced = matrix.slice_rows([0])
    assert sliced.config.shape == (1, 3)
    assert sliced.config.nnz == 2
    assert sliced.config.tpr_threshold == 5

    dense = jnp.array([[0., 1.], [2., 0.]])
    matrix = CSR.fromdense(dense, backend='jax_raw', config=CSRConfig(tpr_threshold=5))
    eager = matrix.diag_add(jnp.ones(2))
    assert eager.config.nnz == 4
    traced = jax.jit(lambda value: value.diag_add(jnp.ones(2)))(matrix.prepare_diag_add())
    assert traced.config.primary is None
    assert traced.config.mirror is None
    assert traced.config.tpr_threshold == 5
    prepared = traced.prepare_config()
    assert prepared.config.nnz == 4
    assert prepared.config.tpr_threshold == 5


def test_replacing_a_mirror_detaches_shared_cache_and_rebuilds_its_plan():
    matrix = _matrix(config=CSRConfig()).build_weight_indices()
    sibling = matrix.with_data(matrix.data * 2)
    original_cache = sibling._cache
    matrix.set_buffer('csc', tuple(jnp.array(value) for value in matrix.buffers['csc']))
    assert matrix._cache is not original_cache
    assert sibling._cache is original_cache
    assert matrix.config.primary is sibling.config.primary
    assert matrix.config.mirror is not None
    assert sibling.config.mirror is not None


@pytest.mark.parametrize('cls', [CSR, CSC, FixedNumPerPre, FixedNumPerPost])
@pytest.mark.parametrize('setter', ['register_buffer', 'set_buffer'])
@pytest.mark.parametrize('reordered', [False, True], ids=['same', 'reordered'])
def test_setting_valid_mirror_keeps_indexed_mm_plan(cls, setter, reordered):
    if cls in (CSR, CSC):
        matrix = cls(
            jnp.array([1., 2., 3., 4.]), jnp.array([0, 2, 1, 2], jnp.int32),
            jnp.array([0, 2, 4], jnp.int32),
            shape=(2, 3) if cls is CSR else (3, 2),
            backend='jax_raw', config=CSRConfig(),
        ).build_weight_indices()
        key = 'csc' if cls is CSR else 'csr'
    else:
        matrix = cls(
            jnp.array([[1., 2.], [3., 4.]]), jnp.array([[0, 2], [1, 2]], jnp.int32),
            shape=(2, 3) if cls is FixedNumPerPre else (3, 2),
            backend='jax_raw', config=CSRConfig(),
        ).build_weight_indices()
        key = 'csc'
    mirror = matrix.buffers[key]
    if reordered:
        order = jnp.array([0, 1, 3, 2])
        mirror = (mirror[0], mirror[1][order], mirror[2][order])
    sibling = matrix.with_data(matrix.data * 2)
    original_primary = matrix.config.primary
    getattr(matrix, setter)(key, mirror)
    assert sibling._cache is not matrix._cache
    assert sibling.config.mirror is not None
    assert matrix.config.primary is original_primary
    assert matrix.config.mirror is not None
    assert matrix.config.mirror_max_row_nnz == 2
    if cls in (CSR, FixedNumPerPre):
        events = BinaryArray(jnp.array([[True, False], [False, True], [True, True]]))
        actual = matrix @ events
        expected = matrix.todense() @ events.value
    else:
        events = BinaryArray(jnp.array([[True, False, True], [False, True, True]]))
        actual = events @ matrix
        expected = events.value @ matrix.todense()
    np.testing.assert_allclose(actual, expected)


@pytest.mark.skipif(jax.default_backend() != 'gpu', reason='CUDA hybrid MM requires a GPU')
@pytest.mark.parametrize('setter', ['register_buffer', 'set_buffer'])
@pytest.mark.parametrize('reordered', [False, True], ids=['same', 'reordered'])
def test_fixednum_mirror_update_keeps_cuda_mm_usable(setter, reordered):
    matrix = FixedNumPerPre(
        jnp.array([[1., 2.], [3., 4.]], jnp.float32),
        jnp.array([[0, 2], [1, 2]], jnp.int32),
        shape=(2, 3), backend='cuda_raw', config=CSRConfig(),
    ).build_weight_indices()
    events = BinaryArray(jnp.array([[True, False], [False, True], [True, True]]))
    expected = jnp.array([[3., 2.], [4., 7.]])
    np.testing.assert_allclose(matrix @ events, expected)
    mirror = matrix.buffers['csc']
    if reordered:
        order = jnp.array([0, 1, 3, 2])
        mirror = (mirror[0], mirror[1][order], mirror[2][order])
    getattr(matrix, setter)('csc', mirror)
    np.testing.assert_allclose(matrix @ events, expected)


@pytest.mark.parametrize('cls', [CSR, FixedNumPerPre])
def test_invalid_mirror_update_keeps_previous_cache_and_plan(cls):
    if cls is CSR:
        matrix = cls(
            jnp.ones(2), jnp.array([0, 1], jnp.int32),
            jnp.array([0, 1, 2], jnp.int32), shape=(2, 2),
            backend='jax_raw', config=CSRConfig(),
        ).build_weight_indices()
    else:
        matrix = cls(
            jnp.ones((2, 1)), jnp.array([[0], [1]], jnp.int32),
            shape=(2, 2), backend='jax_raw', config=CSRConfig(),
        ).build_weight_indices()
    cache, config = matrix._cache, matrix.config
    _, indices, perm = matrix.buffers['csc']
    bad_mirror = (jnp.zeros((3,), jnp.int32), indices, perm)
    with pytest.raises(ValueError, match='indptr'):
        matrix.set_buffer('csc', bad_mirror)
    assert matrix._cache is cache
    assert matrix.config is config


@pytest.mark.parametrize('cls', [CSR, CSC, FixedNumPerPre, FixedNumPerPost])
def test_constructor_prepares_supplied_mirror_without_reloading(monkeypatch, cls):
    from brainevent._csr.binary import initialize

    if cls in (CSR, CSC):
        source = _matrix(cls, config=CSRConfig()).build_weight_indices()
        args = (source.data, source.indices, source.indptr)
    else:
        source = cls(
            jnp.ones((2, 2)), jnp.array([[0, 2], [1, 2]], jnp.int32),
            shape=(2, 3) if cls is FixedNumPerPre else (3, 2),
            backend='jax_raw', config=CSRConfig(),
        ).build_weight_indices()
        args = (source.data, source.indices)
    loads = []
    monkeypatch.setattr(initialize, 'load_csr_config', lambda **kw: loads.append(kw) or CSRConfig())
    matrix = cls(
        *args, shape=source.shape, buffers=source.buffers,
        backend='cuda_raw' if jax.default_backend() == 'gpu' else 'jax_raw',
    )
    assert len(loads) == 1
    assert matrix.config.mirror is not None
    assert matrix.config.mirror.long_mask.devices() == matrix.config.primary.long_mask.devices()
    dense = matrix.todense()
    right = jnp.ones((matrix.shape[1], 4), bool)
    left = jnp.ones((4, matrix.shape[0]), bool)
    np.testing.assert_allclose(matrix @ BinaryArray(right), dense @ right)
    np.testing.assert_allclose(BinaryArray(left) @ matrix, left @ dense)


def test_prepare_config_rejects_traced_structure():
    matrix = _matrix(config=CSRConfig())
    with pytest.raises(ValueError, match='concrete structure'):
        jax.jit(lambda value: value.prepare_config())(matrix)


def test_constructor_reuses_validation_host_copies_for_plan_preparation(monkeypatch):
    from brainevent._csr import main

    reads = []
    validated = {}
    device_get = jax.device_get
    as_indices = main._as_int32_indices
    check_structure = main._check_compressed_structure
    prepare = CSRConfig.prepare_structure

    def record_get(value):
        reads.append(value)
        return device_get(value)

    def record_indices(*args, **kwargs):
        validated['indices'] = kwargs['host_indices']
        return as_indices(*args, **kwargs)

    def record_indptr(*args, **kwargs):
        validated['indptr'] = kwargs['host_indptr']
        return check_structure(*args, **kwargs)

    def record_prepare(config, indices, indptr, **kwargs):
        assert indices is validated['indices']
        assert indptr is validated['indptr']
        return prepare(config, indices, indptr, **kwargs)

    monkeypatch.setattr(jax, 'device_get', record_get)
    monkeypatch.setattr(main, '_as_int32_indices', record_indices)
    monkeypatch.setattr(main, '_check_compressed_structure', record_indptr)
    monkeypatch.setattr(CSRConfig, 'prepare_structure', record_prepare)
    matrix = _matrix(config=CSRConfig())
    assert sum(value is matrix.indices for value in reads) == 1
    assert sum(value is matrix.indptr for value in reads) == 1


def test_numpy_structure_is_prepared_without_device_readback(monkeypatch):
    def no_readback(*args, **kwargs):
        raise AssertionError('host structure is already available')

    monkeypatch.setattr(jax, 'device_get', no_readback)
    matrix = CSR(
        np.ones(3), np.array([0, 2, 1], np.int32), np.array([0, 2, 3], np.int32),
        shape=(2, 3), backend='jax_raw', config=CSRConfig(),
    )
    assert matrix.config.nnz == 3


@pytest.mark.parametrize('cls', [CSR, CSC, FixedNumPerPre, FixedNumPerPost])
@pytest.mark.parametrize('backend,global_backend,expected_queries', [
    (None, 'jax_raw', 0),
    (None, 'cuda_raw', 1),
    (None, 'cusparse', 1),
    (None, 'numba', 1),
    ('jax_raw', 'cuda_raw', 0),
    ('cuda_raw', 'jax_raw', 1),
    ('cusparse', 'cuda_raw', 0),
])
def test_auto_preparation_uses_object_then_global_backend(
    monkeypatch, cls, backend, global_backend, expected_queries,
):
    import brainevent.config as runtime_config
    import brainevent._csr.config as csr_config

    monkeypatch.setitem(runtime_config._global_backends, 'gpu', global_backend)
    queries = []
    monkeypatch.setattr(csr_config, 'get_device_info',
                        lambda *args, **kwargs: queries.append((args, kwargs)) or None)
    if cls in (CSR, CSC):
        matrix = cls(
            np.ones(2), np.array([0, 1], np.int32), np.array([0, 1, 2], np.int32),
            shape=(2, 2), backend=backend, config=CSRConfig(),
        )
    else:
        matrix = cls(
            np.ones((2, 1)), np.array([[0], [1]], np.int32),
            shape=(2, 2), backend=backend, config=CSRConfig(),
        )
    assert matrix.backend == backend
    assert len(queries) == expected_queries
    prepared = matrix.prepare_config()
    assert prepared.backend == backend
    assert len(queries) == 2 * expected_queries


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

    class GPUDevice:
        platform = 'gpu'

    with pytest.raises(ValueError, match='single device'):
        get_device_info(PlacedArray((GPUDevice(), GPUDevice())))


def test_cpu_sharded_structure_replicates_prepared_masks():
    # JAX must see the extra CPU devices before this process imports it.
    script = '''
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from brainevent import CSR, CSC, FixedNumPerPre, FixedNumPerPost, BinaryArray, config
from brainevent._csr.config import CSRConfig

devices = jax.devices('cpu')
assert len(devices) == 2
mesh = Mesh(np.asarray(devices), ('cpu_shard',))
config.set_backend('cpu', 'jax_raw')
def placed(value, spec):
    return jax.device_put(jnp.asarray(value), NamedSharding(mesh, spec))
def verify(config):
    for plan in (config.primary, config.mirror):
        mask = plan.long_mask
        assert isinstance(mask.sharding, NamedSharding), mask.sharding
        assert mask.sharding.spec == P(), mask.sharding
        assert mask.devices() == set(devices), mask.devices()

for cls in (CSR, CSC, FixedNumPerPre, FixedNumPerPost):
    if cls in (CSR, CSC):
        args = (
            placed(np.ones(4, np.float32), P('cpu_shard')),
            placed(np.array([0, 1, 0, 1], np.int32), P('cpu_shard')),
            placed(np.array([0, 2, 4], np.int32), P()),
        )
    else:
        args = (
            placed(np.ones((2, 2), np.float32), P('cpu_shard')),
            placed(np.array([[0, 1], [0, 1]], np.int32), P('cpu_shard')),
        )
    direct_indices = args[1].reshape(-1) if cls in (FixedNumPerPre, FixedNumPerPost) else args[1]
    direct_indptr = (np.array([0, 2, 4], np.int32)
                     if cls in (FixedNumPerPre, FixedNumPerPost) else args[2])
    verify(CSRConfig().prepare_structure(direct_indices, direct_indptr, shape=(2, 2)))
    matrix = cls(*args, shape=(2, 2), config=CSRConfig())
    verify(matrix.config)
    verify(matrix.prepare_config().config)
    explicit = matrix.prepare_config(device=devices[0])
    assert explicit.config.primary.long_mask.devices() == {devices[0]}
    assert explicit.config.mirror.long_mask.devices() == {devices[0]}
    result = jax.jit(lambda m, x: m @ BinaryArray(x))(
        matrix, placed(np.ones(2, np.float32), P()))
    np.testing.assert_array_equal(np.asarray(result), [2, 2])
    mirror = matrix.build_weight_indices()
    rebuilt = cls(*args, shape=(2, 2),
                  buffers=mirror.buffers, config=CSRConfig())
    verify(rebuilt.config)
'''
    env = dict(os.environ, JAX_PLATFORM_NAME='cpu',
               XLA_FLAGS='--xla_force_host_platform_device_count=2')
    result = subprocess.run(
        [sys.executable, '-c', script], cwd=Path(__file__).parents[3],
        env=env, text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(jax.default_backend() != 'gpu', reason='GPU placement requires CUDA')
def test_global_jax_backend_skips_native_probe_on_gpu(monkeypatch):
    import brainevent.config as runtime_config
    import brainevent._csr.config as csr_config

    monkeypatch.setitem(runtime_config._global_backends, 'gpu', 'jax_raw')

    def no_native_probe(*args, **kwargs):
        raise AssertionError('global jax_raw must not query CUDA device information')

    monkeypatch.setattr(csr_config, '_query_cuda_device_info', no_native_probe)
    with jax.default_device(jax.devices('gpu')[0]):
        matrix = CSR(
            jnp.ones(2), jnp.array([0, 1], jnp.int32),
            jnp.array([0, 1, 2], jnp.int32),
            shape=(2, 2), config=CSRConfig(),
        )
    assert matrix.config.device_info is None
    assert matrix.config.primary is not None


@pytest.mark.skipif(jax.default_backend() != 'gpu', reason='GPU placement requires CUDA')
def test_explicit_gpu_preparation_moves_both_masks_and_queries_on_jax_backend():
    cpu, gpu = jax.devices('cpu')[0], jax.devices('gpu')[0]
    with jax.default_device(cpu):
        matrix = _matrix(config=CSRConfig()).build_weight_indices()
    assert matrix.config.device_info is None
    prepared = matrix.prepare_config(device=gpu)
    assert prepared.backend == 'jax_raw'
    assert prepared.config.device_info.device_kind == gpu.device_kind
    assert prepared.config.primary.long_mask.devices() == {gpu}
    assert prepared.config.mirror.long_mask.devices() == {gpu}


@pytest.mark.skipif(jax.default_backend() != 'gpu', reason='GPU device query requires CUDA')
def test_closed_concrete_structure_can_prepare_with_cold_device_cache_under_trace():
    from brainevent._csr.config import _query_cuda_device_info

    indices = jnp.array([0, 1], jnp.int32)
    indptr = jnp.array([0, 1, 2], jnp.int32)
    _query_cuda_device_info.cache_clear()
    make = lambda data: CSR(
        data, indices, indptr, shape=(2, 2), backend='cuda_raw', config=CSRConfig(),
    )
    closed = jax.make_jaxpr(lambda data: make(data).data)(jnp.ones(2))
    assert not closed.jaxpr.eqns
    matrix = jax.jit(make)(jnp.ones(2))
    assert matrix.config.device_info.platform == 'gpu'
    assert matrix.config.primary is not None
