import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax
import jax.numpy as jnp
import pytest

import brainunit as u
import brainevent._fcn.binary as binary_mod

from dev.fcn.coba_ei_benchmark_test_helpers import (
    _HAS_BINARY_JAX_RAW,
    apply_conn_like_coba_ei,
    assert_quantity_allclose,
    assert_spike_history_matches,
    assert_step_trace_matches,
    binary_jax_route_reference,
    build_shared_batch_mm_conn_pair,
    build_shared_mm_large_scale_point,
    build_conn_like_coba_ei,
    coba_ei_module,
    install_fake_brainpy_state,
    matrix_from_seed,
    numpy_seed,
    preferred_real_mm_backend,
    run_full_batch_e2e_spike_history_once,
    run_batch_e2e_spike_history_once,
    run_batch_e2e_step_trace_once,
)


SUCCESS_MM_ROUTE_CASES = (
    ('binary', 'post', 'row_gather'),
    ('compact', 'post', 'row_gather'),
    ('compact', 'pre', 'row_gather'),
    ('bitpack_a0', 'pre', 'row_gather'),
    ('bitpack_a0', 'post', 'row_gather'),
    ('bitpack_a1', 'pre', 'row_gather'),
    ('bitpack_a1', 'post', 'row_gather'),
)

SKIPPED_MM_ROUTE_CASES = (
    pytest.param(
        'binary',
        'pre',
        'col_scatter',
        marks=pytest.mark.skip(
            reason='binary/pre/col_scatter MM col-scatter operator is not registered; do not route through ordinary binary_fcnmm'
        ),
        id='binary-pre-col_scatter-unregistered-mm-col-scatter',
    ),
)

MM_ROUTE_CASES = (*SUCCESS_MM_ROUTE_CASES, *SKIPPED_MM_ROUTE_CASES)

COBA_BATCH_E2E_MM_ROUTE_CASES = (
    ('binary', 'post', 'row_gather'),
    ('compact', 'post', 'row_gather'),
    ('compact', 'pre', 'row_gather'),
    ('bitpack_a0', 'pre', 'row_gather'),
    ('bitpack_a1', 'pre', 'row_gather'),
    *SKIPPED_MM_ROUTE_CASES,
)

BINARY_FCNMM_CUDA_GENERATORS = {
    '_binary_fcnmm_cuda_kernel',
    '_binary_fcnmm_test_colmajor_fullwarp_nocap_kernel',
}


def _spy_binary_fcnmm_kernel_generators(monkeypatch):
    calls = []
    for platform, entries in binary_mod.binary_fcnmm_p._kernels.items():
        for backend, entry in entries.items():
            original = entry.kernel_generator
            generator_name = getattr(original, '__name__', repr(original))

            def _wrapped_generator(*args, _original=original, _platform=platform,
                                   _backend=backend, _generator_name=generator_name, **kwargs):
                matrix_info = kwargs.get('matrix_info')
                calls.append(
                    {
                        'platform': _platform,
                        'backend': _backend,
                        'generator': _generator_name,
                        'shape': kwargs.get('shape'),
                        'transpose': kwargs.get('transpose'),
                        'matrix_shape': getattr(matrix_info, 'shape', None),
                        'matrix_dtype': getattr(matrix_info, 'dtype', None),
                    }
                )
                return _original(*args, **kwargs)

            monkeypatch.setattr(entry, 'kernel_generator', _wrapped_generator)
    jax.clear_caches()
    return calls


def _assert_binary_fcnmm_kernel_generators_called(calls, *, actual_backend):
    actual_generators = {
        call['generator']
        for call in calls
        if call['platform'] == jax.default_backend() and call['backend'] == actual_backend
    }
    jax_generators = {
        call['generator']
        for call in calls
        if call['platform'] == jax.default_backend() and call['backend'] == 'jax_raw'
    }
    missing = []
    if not actual_generators.intersection(BINARY_FCNMM_CUDA_GENERATORS):
        missing.append(
            f'{actual_backend} did not call one of {sorted(BINARY_FCNMM_CUDA_GENERATORS)}'
        )
    if '_binary_fcnmm_jax_kernel' not in jax_generators:
        missing.append('jax_raw did not call _binary_fcnmm_jax_kernel')
    assert not missing, f'Missing binary_fcnmm kernel-generator calls: {missing}; calls={calls}'


def test_shared_mm_large_scale_point_uses_scale100_conn607():
    assert build_shared_mm_large_scale_point() == (100, 607)


def test_make_simulation_batch_run_accepts_backend_keyword(monkeypatch):
    mod = coba_ei_module()
    install_fake_brainpy_state(mod, monkeypatch)
    run = mod.make_simulation_batch_run(
        scale=0.01,
        batch_size=2,
        data_type='binary',
        efferent_target='post',
        duration=1.0 * u.ms,
        conn_num=3,
        homo=True,
        mv_layout='row_gather',
        backend='jax_raw',
    )
    assert callable(run)


def test_build_shared_batch_mm_conn_pair_keeps_actual_backend_global_only():
    actual_conn, reference_conn = build_shared_batch_mm_conn_pair(
        32,
        40,
        3,
        data_type='binary',
        efferent_target='post',
        homo=True,
        conn_weight_base=0.6 * u.mS,
        mv_layout='row_gather',
        reference_backend='jax_raw',
    )

    assert actual_conn.backend is None
    assert reference_conn.backend == 'jax_raw'


def test_batch_step_trace_contains_full_packed_spike_bits(monkeypatch):
    actual_backend = preferred_real_mm_backend('binary')
    if actual_backend is None:
        pytest.skip('No supported real backend for binary e2e')

    mod = coba_ei_module()
    install_fake_brainpy_state(mod, monkeypatch)
    actual, expected = run_batch_e2e_step_trace_once(
        mod,
        scale=0.01,
        batch_size=2,
        data_type='binary',
        efferent_target='post',
        conn_num=3,
        homo=True,
        mv_layout='row_gather',
        actual_backend=actual_backend,
        steps=2,
    )

    assert 'spike_bits' in actual
    assert 'spike_shape' in actual
    assert 'spike_summary' not in actual
    assert actual['spike_bits'].dtype == jnp.uint8
    assert jnp.array_equal(actual['spike_bits'], expected['spike_bits'])
    assert jnp.array_equal(actual['spike_shape'], expected['spike_shape'])


@pytest.mark.parametrize(('data_type', 'efferent_target', 'mv_layout'), MM_ROUTE_CASES)
@pytest.mark.parametrize('homo', [True, False])
def test_each_route_mm_matches_binary_jax_reference_on_small_shapes(
    data_type,
    efferent_target,
    mv_layout,
    homo,
):
    source_size = 17
    target_size = 23
    seed = 3
    batch_size = 5
    with numpy_seed(seed):
        conn = build_conn_like_coba_ei(
            source_size,
            target_size,
            5,
            data_type=data_type,
            efferent_target=efferent_target,
            homo=homo,
            conn_weight_base=0.6 * u.mS,
            mv_layout=mv_layout,
            backend=preferred_real_mm_backend(data_type),
        )

    spikes = matrix_from_seed(source_size, batch_size, 11, efferent_target=efferent_target)
    actual = apply_conn_like_coba_ei(spikes, conn, data_type=data_type, efferent_target=efferent_target)
    expected = binary_jax_route_reference(
        source_size,
        target_size,
        5,
        spikes,
        seed=seed,
        efferent_target=efferent_target,
        homo=homo,
        mv_layout=mv_layout,
        conn_weight_base=0.6 * u.mS,
    )

    jax.block_until_ready(u.get_mantissa(actual))
    jax.block_until_ready(u.get_mantissa(expected))
    assert_quantity_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_direct_binary_mm_route_calls_fcnmm_kernel_generators(monkeypatch):
    actual_backend = preferred_real_mm_backend('binary')
    if actual_backend is None:
        pytest.skip('No supported real binary fcnmm backend.')

    calls = _spy_binary_fcnmm_kernel_generators(monkeypatch)
    source_size = 17
    target_size = 23
    seed = 3
    batch_size = 5
    with numpy_seed(seed):
        conn = build_conn_like_coba_ei(
            source_size,
            target_size,
            5,
            data_type='binary',
            efferent_target='post',
            homo=True,
            conn_weight_base=0.6 * u.mS,
            mv_layout='row_gather',
            backend=actual_backend,
        )

    spikes = matrix_from_seed(source_size, batch_size, 11, efferent_target='post')
    actual = apply_conn_like_coba_ei(spikes, conn, data_type='binary', efferent_target='post')
    expected = binary_jax_route_reference(
        source_size,
        target_size,
        5,
        spikes,
        seed=seed,
        efferent_target='post',
        homo=True,
        mv_layout='row_gather',
        conn_weight_base=0.6 * u.mS,
    )

    jax.block_until_ready(u.get_mantissa(actual))
    jax.block_until_ready(u.get_mantissa(expected))
    assert_quantity_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    _assert_binary_fcnmm_kernel_generators_called(calls, actual_backend=actual_backend)


@pytest.mark.skipif(
    not _HAS_BINARY_JAX_RAW,
    reason='binary jax_raw backend is unavailable',
)
@pytest.mark.parametrize(('data_type', 'efferent_target', 'mv_layout'), COBA_BATCH_E2E_MM_ROUTE_CASES)
def test_small_scale_mm_e2e_spikes_match_binary_jax_reference(
    data_type,
    efferent_target,
    mv_layout,
    monkeypatch,
):
    actual_backend = preferred_real_mm_backend(data_type)
    if actual_backend is None:
        pytest.skip(f'No supported real backend for e2e data_type={data_type!r}')

    mod = coba_ei_module()
    install_fake_brainpy_state(mod, monkeypatch)
    actual, expected = run_batch_e2e_spike_history_once(
        mod,
        scale=1,
        batch_size=2,
        data_type=data_type,
        efferent_target=efferent_target,
        conn_num=80,
        homo=True,
        mv_layout=mv_layout,
        actual_backend=actual_backend,
        steps=10,
    )
    assert_spike_history_matches(actual, expected, atol=1e-3)


@pytest.mark.skipif(
    not _HAS_BINARY_JAX_RAW,
    reason='binary jax_raw backend is unavailable',
)
def test_small_scale_binary_mm_e2e_spikes_match_binary_jax_reference_after_mm_routes(monkeypatch):
    actual_backend = preferred_real_mm_backend('binary')
    if actual_backend is None:
        pytest.skip('No supported real binary fcnmm backend.')

    mod = coba_ei_module()
    install_fake_brainpy_state(mod, monkeypatch)
    actual, expected = run_batch_e2e_spike_history_once(
        mod,
        scale=1,
        batch_size=2,
        data_type='binary',
        efferent_target='post',
        conn_num=80,
        homo=True,
        mv_layout='row_gather',
        actual_backend=actual_backend,
        steps=2,
    )
    assert_spike_history_matches(actual, expected, atol=1e-3)


@pytest.mark.skipif(
    not _HAS_BINARY_JAX_RAW,
    reason='binary jax_raw backend is unavailable',
)
def test_full_coba_ei_batch_spike_history_uses_real_benchmark_construction(monkeypatch):
    actual_backend = preferred_real_mm_backend('binary')
    if actual_backend is None:
        pytest.skip('No supported real backend for binary e2e')

    mod = coba_ei_module()
    install_fake_brainpy_state(mod, monkeypatch)
    actual, expected = run_full_batch_e2e_spike_history_once(
        mod,
        scale=0.01,
        batch_size=2,
        data_type='binary',
        efferent_target='post',
        conn_num=5,
        homo=True,
        mv_layout='row_gather',
        actual_backend=actual_backend,
        reference_backend='jax_raw',
        steps=3,
    )
    assert_spike_history_matches(actual, expected, atol=1e-3)


@pytest.mark.skipif(
    not _HAS_BINARY_JAX_RAW,
    reason='binary jax_raw backend is unavailable',
)
@pytest.mark.parametrize(('data_type', 'efferent_target', 'mv_layout'), COBA_BATCH_E2E_MM_ROUTE_CASES)
def test_large_scale_mm_e2e_spikes_match_binary_jax_reference(
    data_type,
    efferent_target,
    mv_layout,
    monkeypatch,
):
    actual_backend = preferred_real_mm_backend(data_type)
    if actual_backend is None:
        pytest.skip(f'No supported real backend for e2e data_type={data_type!r}')

    mod = coba_ei_module()
    install_fake_brainpy_state(mod, monkeypatch)
    batch_size = 32
    scale, conn_num = build_shared_mm_large_scale_point()
    actual, expected = run_batch_e2e_step_trace_once(
        mod,
        scale=scale,
        batch_size=batch_size,
        data_type=data_type,
        efferent_target=efferent_target,
        conn_num=conn_num,
        homo=True,
        mv_layout=mv_layout,
        actual_backend=actual_backend,
        steps=10,
    )
    assert_step_trace_matches(actual, expected, atol=5e-2)
