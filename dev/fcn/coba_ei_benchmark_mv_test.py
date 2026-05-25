import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax
import jax.numpy as jnp
import pytest
import warnings

import brainunit as u

from dev.fcn.coba_ei_benchmark_test_helpers import (
    _HAS_BINARY_JAX_RAW,
    COMPACT_ROUTE_CASES,
    ROUTE_CASES,
    apply_conn_like_coba_ei,
    assert_final_run_matches_binary_jax,
    assert_quantity_allclose,
    assert_spike_history_matches,
    assert_step_trace_matches,
    binary_array_cls,
    binary_jax_route_reference,
    bitpacked_binary_cls,
    build_shared_mv_large_scale_point,
    build_conn_like_coba_ei,
    coba_ei_module,
    install_fake_brainpy_state,
    numpy_seed,
    preferred_real_backend,
    run_full_e2e_spike_history_once,
    run_e2e_spike_history_once,
    run_e2e_step_trace_once,
    run_simulation_once,
    spikes_from_seed,
)
from brainevent._fcn.binary import binary_fcnmv
from brainevent._misc import fixed_conn_num_to_csc
from brainevent._test_util import generate_fixed_conn_num_indices


REMOVED_ROUTE_ERROR_CASES = (
    (
        'binary',
        'pre',
        'row_gather',
        'Binary_fcnmv no longer supports this path',
    ),
    (
        'bitpack',
        'post',
        'row_gather',
        'Bitpack_binary_fcnmv no longer supports this path',
    ),
    (
        'bitpack_a0',
        'post',
        'row_gather',
        'Bitpack_binary_fcnmv no longer supports this path',
    ),
    (
        'bitpack_a1',
        'post',
        'row_gather',
        'Bitpack_binary_fcnmv no longer supports this path',
    ),
)

SUCCESS_ROUTE_CASES = tuple(case for case in ROUTE_CASES if case not in tuple(case[:3] for case in REMOVED_ROUTE_ERROR_CASES))


@pytest.mark.parametrize(
    ('data_type', 'efferent_target', 'expected_kind'),
    [
        ('binary', 'post', 'binary'),
        ('binary', 'pre', 'binary'),
        ('bitpack', 'post', 'bitpack'),
        ('bitpack_a0', 'pre', 'bitpack'),
        ('bitpack_a1', 'post', 'bitpack'),
    ],
)
def test_prepare_operand_selects_expected_representation(data_type, efferent_target, expected_kind):
    spikes = jnp.asarray([1, 0, 1, 0, 1], dtype=jnp.bool_)
    from dev.fcn.coba_ei_benchmark_test_helpers import prepare_operand_like_coba_ei

    operand = prepare_operand_like_coba_ei(spikes, data_type=data_type)

    if expected_kind == 'binary':
        assert isinstance(operand, binary_array_cls())
    else:
        assert isinstance(operand, bitpacked_binary_cls())


@pytest.mark.parametrize(('data_type', 'efferent_target', 'mv_layout'), COMPACT_ROUTE_CASES)
def test_compact_routes_raise_clear_not_implemented_error(data_type, efferent_target, mv_layout):
    mod = coba_ei_module()
    with numpy_seed(0):
        conn = mod._make_post_conn(
            7,
            11,
            3,
            data_type=data_type,
            efferent_target=efferent_target,
            homo=True,
            conn_weight_base=0.6 * u.mS,
            mv_layout=mv_layout,
            backend='cuda_raw',
        )

    spikes = jnp.asarray([1, 0, 1, 0, 1, 0, 1], dtype=jnp.bool_)
    with pytest.raises(NotImplementedError, match='Compact binary_fcnmv / compact_binary_fcnmm operators are no longer supported'):
        mod._apply_conn(spikes, conn, data_type=data_type, efferent_target=efferent_target)


@pytest.mark.parametrize(('data_type', 'efferent_target', 'mv_layout', 'error_match'), REMOVED_ROUTE_ERROR_CASES)
@pytest.mark.parametrize('homo', [True, False])
def test_removed_mv_routes_raise_documented_errors(data_type, efferent_target, mv_layout, error_match, homo):
    # These MV routes are intentionally kept in test coverage even though the
    # implementation removed them, so full route coverage also verifies the
    # documented failure mode and message.
    source_size = 17
    target_size = 23
    seed = 3
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
            backend=preferred_real_backend(data_type),
        )

    spikes = spikes_from_seed(source_size, 5, p_active=0.35)
    with pytest.raises(ValueError, match=error_match):
        apply_conn_like_coba_ei(spikes, conn, data_type=data_type, efferent_target=efferent_target)


def test_apply_conn_uses_expected_matmul_side(monkeypatch):
    mod = coba_ei_module()
    calls = []

    class Operand:
        def __matmul__(self, other):
            calls.append(('operand', other))
            return 'post-route'

    class Conn:
        def __matmul__(self, other):
            calls.append(('conn', other))
            return 'pre-route'

    monkeypatch.setattr(mod, '_prepare_operand', lambda *args, **kwargs: Operand())
    conn = Conn()

    assert mod._apply_conn(jnp.array([True]), conn, data_type='binary', efferent_target='post') == 'post-route'
    assert mod._apply_conn(jnp.array([True]), conn, data_type='binary', efferent_target='pre') == 'pre-route'
    assert calls[0][0] == 'operand'
    assert calls[1][0] == 'conn'


@pytest.mark.parametrize(
    ('data_type', 'efferent_target', 'mv_layout', 'expect_dual_layout'),
    [
        ('binary', 'pre', 'col_scatter', True),
        ('binary', 'pre', 'row_gather', False),
        ('binary', 'post', 'row_gather', False),
        ('bitpack', 'pre', 'row_gather', False),
        ('bitpack_a0', 'pre', 'row_gather', False),
        ('bitpack_a1', 'post', 'row_gather', False),
    ],
)
def test_make_post_conn_enables_dual_layout_only_when_needed(
    data_type,
    efferent_target,
    mv_layout,
    expect_dual_layout,
):
    with numpy_seed(0):
        conn = build_conn_like_coba_ei(
            7,
            11,
            3,
            data_type=data_type,
            efferent_target=efferent_target,
            homo=True,
            conn_weight_base=0.6 * u.mS,
            mv_layout=mv_layout,
            backend='jax_raw',
        )

    assert conn.maintain_dual_layout is expect_dual_layout
    if expect_dual_layout:
        assert conn.col_weights is not None
        assert conn.col_indices is not None
        assert conn.col_indptr is not None
    else:
        assert conn.col_weights is None
        assert conn.col_indices is None
        assert conn.col_indptr is None


@pytest.mark.parametrize(
    ('data_type', 'expected_pack_axis'),
    [
        ('binary', 0),
        ('bitpack', 0),
        ('bitpack_a0', 0),
        ('bitpack_a1', 1),
    ],
)
def test_make_post_conn_assigns_expected_bitpack_mm_pack_axis(data_type, expected_pack_axis):
    with numpy_seed(0):
        conn = build_conn_like_coba_ei(
            7,
            11,
            3,
            data_type=data_type,
            efferent_target='post',
            homo=True,
            conn_weight_base=0.6 * u.mS,
            mv_layout='row_gather',
            backend='jax_raw',
        )

    assert conn.bitpack_mm_pack_axis == expected_pack_axis


@pytest.mark.parametrize(('data_type', 'efferent_target', 'mv_layout'), SUCCESS_ROUTE_CASES)
@pytest.mark.parametrize('homo', [True, False])
def test_each_route_mv_matches_binary_jax_reference_on_small_shapes(
    data_type,
    efferent_target,
    mv_layout,
    homo,
):
    source_size = 17
    target_size = 23
    seed = 3
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
            backend=preferred_real_backend(data_type),
        )

    spikes = spikes_from_seed(source_size, 5, p_active=0.35)
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


@pytest.mark.parametrize(('data_type', 'efferent_target', 'mv_layout'), SUCCESS_ROUTE_CASES)
@pytest.mark.parametrize('homo', [True, False])
def test_scale10_conn800_mv_routes_match_binary_jax_reference(
    data_type,
    efferent_target,
    mv_layout,
    homo,
):
    seed = 42
    scale = 10
    conn_num = 800
    source_size = int(3200 * scale)
    target_size = int(4000 * scale)
    with numpy_seed(seed):
        conn = build_conn_like_coba_ei(
            source_size,
            target_size,
            conn_num,
            data_type=data_type,
            efferent_target=efferent_target,
            homo=homo,
            conn_weight_base=0.6 * u.mS,
            mv_layout=mv_layout,
            backend=preferred_real_backend(data_type),
        )
    spikes = spikes_from_seed(source_size, seed, p_active=0.02)
    actual = apply_conn_like_coba_ei(spikes, conn, data_type=data_type, efferent_target=efferent_target)
    expected = binary_jax_route_reference(
        source_size,
        target_size,
        conn_num,
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


@pytest.mark.skipif(
    not _HAS_BINARY_JAX_RAW,
    reason='binary jax_raw backend is unavailable',
)
@pytest.mark.parametrize(('data_type', 'efferent_target', 'mv_layout'), SUCCESS_ROUTE_CASES)
def test_small_scale_mv_e2e_conductance_trace_matches_binary_jax_reference(
    data_type,
    efferent_target,
    mv_layout,
    monkeypatch,
):
    actual_backend = preferred_real_backend(data_type)
    if actual_backend is None:
        pytest.skip(f'No supported real backend for e2e data_type={data_type!r}')

    mod = coba_ei_module()
    install_fake_brainpy_state(mod, monkeypatch)
    actual, expected = run_e2e_step_trace_once(
        mod,
        scale=1,
        data_type=data_type,
        efferent_target=efferent_target,
        conn_num=80,
        homo=True,
        mv_layout=mv_layout,
        actual_backend=actual_backend,
        steps=10,
        summarize=False,
    )
    assert_step_trace_matches(actual, expected, atol=1e-3)


@pytest.mark.skipif(
    not _HAS_BINARY_JAX_RAW,
    reason='binary jax_raw backend is unavailable',
)
@pytest.mark.parametrize(('data_type', 'efferent_target', 'mv_layout'), SUCCESS_ROUTE_CASES)
def test_small_scale_mv_e2e_spikes_match_binary_jax_reference(
    data_type,
    efferent_target,
    mv_layout,
    monkeypatch,
):
    actual_backend = preferred_real_backend(data_type)
    if actual_backend is None:
        pytest.skip(f'No supported real backend for e2e data_type={data_type!r}')

    mod = coba_ei_module()
    install_fake_brainpy_state(mod, monkeypatch)
    actual, expected = run_e2e_spike_history_once(
        mod,
        scale=1,
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
@pytest.mark.parametrize(
    ('scale', 'conn_num'),
    [
        (1, 80),
        (50, 2000),
    ],
)
def test_full_coba_ei_mv_spike_history_uses_real_benchmark_construction(scale, conn_num, monkeypatch):
    actual_backend = preferred_real_backend('binary')
    if actual_backend is None:
        pytest.skip('No supported real backend for binary e2e')

    mod = coba_ei_module()
    install_fake_brainpy_state(mod, monkeypatch)
    actual, expected = run_full_e2e_spike_history_once(
        mod,
        scale=scale,
        data_type='binary',
        efferent_target='post',
        conn_num=conn_num,
        homo=True,
        mv_layout='row_gather',
        actual_backend=actual_backend,
        reference_backend='jax_raw',
        steps=20,
    )
    assert_spike_history_matches(actual, expected, atol=1e-3)


@pytest.mark.skipif(
    not _HAS_BINARY_JAX_RAW,
    reason='binary jax_raw backend is unavailable',
)
@pytest.mark.parametrize(('data_type', 'efferent_target', 'mv_layout'), SUCCESS_ROUTE_CASES)
@pytest.mark.parametrize('homo', [True, False])
def test_large_scale_mv_e2e_step_trace_matches_binary_jax_reference(
    data_type,
    efferent_target,
    mv_layout,
    homo,
    monkeypatch,
):
    actual_backend = preferred_real_backend(data_type)
    if actual_backend is None:
        pytest.skip(f'No supported real backend for e2e data_type={data_type!r}')

    mod = coba_ei_module()
    install_fake_brainpy_state(mod, monkeypatch)
    scale, conn_num = build_shared_mv_large_scale_point()
    actual, expected = run_e2e_step_trace_once(
        mod,
        scale=scale,
        data_type=data_type,
        efferent_target=efferent_target,
        conn_num=conn_num,
        homo=homo,
        mv_layout=mv_layout,
        actual_backend=actual_backend,
        steps=20,
    )
    assert_step_trace_matches(actual, expected, atol=1e-3)


@pytest.mark.skipif(
    not _HAS_BINARY_JAX_RAW,
    reason='binary jax_raw backend is unavailable',
)
@pytest.mark.parametrize('data_type', ['binary'])
@pytest.mark.parametrize('homo', [True, False])
def test_pre_col_scatter_mv_small_scale_step_trace_matches_binary_jax_reference(
    data_type,
    homo,
    monkeypatch,
):
    actual_backend = preferred_real_backend(data_type)
    if actual_backend is None:
        pytest.skip(f'No supported real backend for e2e data_type={data_type!r}')

    mod = coba_ei_module()
    install_fake_brainpy_state(mod, monkeypatch)
    actual, expected = run_e2e_step_trace_once(
        mod,
        scale=1,
        data_type=data_type,
        efferent_target='pre',
        conn_num=80,
        homo=homo,
        mv_layout='col_scatter',
        actual_backend=actual_backend,
        steps=10,
    )
    assert_step_trace_matches(actual, expected, atol=1e-3)


@pytest.mark.skipif(
    not _HAS_BINARY_JAX_RAW,
    reason='binary jax_raw backend is unavailable',
)
@pytest.mark.parametrize('data_type', ['binary'])
@pytest.mark.parametrize('homo', [True, False])
def test_pre_col_scatter_mv_final_run_matches_binary_jax_reference(
    data_type,
    homo,
    monkeypatch,
):
    actual_backend = preferred_real_backend(data_type)
    if actual_backend is None:
        pytest.skip(f'No supported real backend for final-run data_type={data_type!r}')

    mod = coba_ei_module()
    install_fake_brainpy_state(mod, monkeypatch)
    actual = run_simulation_once(
        mod,
        seed=0,
        scale=0.01,
        data_type=data_type,
        efferent_target='pre',
        duration=1.0 * u.ms,
        conn_num=5,
        homo=homo,
        mv_layout='col_scatter',
        backend=actual_backend,
    )
    expected = run_simulation_once(
        mod,
        seed=0,
        scale=0.01,
        data_type='binary',
        efferent_target='pre',
        duration=1.0 * u.ms,
        conn_num=5,
        homo=homo,
        mv_layout='col_scatter',
        backend='jax_raw',
    )
    assert_final_run_matches_binary_jax(actual, expected, rtol=1e-5, atol=1e-5)


def test_pre_col_scatter_cuda_route_avoids_non_cuda_col_scatter_fallback_warning():
    actual_backend = preferred_real_backend('binary')
    if actual_backend != 'cuda_raw':
        pytest.skip(f'Expected cuda_raw backend for binary pre col_scatter route, got {actual_backend!r}')

    source_size = 17
    target_size = 23
    seed = 3
    with numpy_seed(seed):
        conn = build_conn_like_coba_ei(
            source_size,
            target_size,
            5,
            data_type='binary',
            efferent_target='pre',
            homo=True,
            conn_weight_base=0.6 * u.mS,
            mv_layout='col_scatter',
            backend=actual_backend,
        )

    spikes = spikes_from_seed(source_size, 5, p_active=0.35)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        actual = apply_conn_like_coba_ei(
            spikes,
            conn,
            data_type='binary',
            efferent_target='pre',
        )

    jax.block_until_ready(u.get_mantissa(actual))
    messages = [str(w.message) for w in caught]
    assert not any('fall back to the default gather/scatter path' in msg for msg in messages)


def test_pre_col_scatter_jax_raw_reference_emits_expected_fallback_warning():
    shape = (20, 40)
    indices = generate_fixed_conn_num_indices(shape[0], shape[1], 4)
    weights = jnp.asarray([1.5], dtype=jnp.float32)
    col_weights, col_indices, col_indptr = fixed_conn_num_to_csc(weights, indices, shape=shape)
    spikes = spikes_from_seed(shape[1], 13, p_active=0.35)
    with pytest.warns(
        UserWarning,
        match='Binary_fcnmv does not support col-scatter options on this backend.*fall back to the default gather/scatter path.*performance may degrade',
    ):
        expected = binary_fcnmv(
            weights,
            indices,
            spikes,
            shape=shape,
            transpose=False,
            backend='jax_raw',
            col_weights=col_weights,
            col_indices=col_indices,
            col_indptr=col_indptr,
        )

    jax.block_until_ready(u.get_mantissa(expected))


def test_training_entrypoints_remain_unimplemented():
    mod = coba_ei_module()
    with pytest.raises(NotImplementedError, match='not implemented'):
        mod.make_training_run(scale=0.01)()
